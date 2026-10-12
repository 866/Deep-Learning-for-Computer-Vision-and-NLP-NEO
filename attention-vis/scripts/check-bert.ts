(globalThis as any).window ??= globalThis;
// Dev-time checks for the BERT model family (weights, tokenizer, forward pass, layout).
// Follows the same esbuild-bundle-then-node pattern as scripts/validate-model.ts.
// Run: npm run check:bert
import fs from 'fs';
import { TensorF32 } from '../src/utils/tensor';
import {
    bertSentiment, bertTopK, bertVocabFromConfig, createBertModel, IBertModelLink, setBertSentence,
    uploadAllBert, wordpieceEncode,
} from '../src/llm/BertModel';
import { embedTable, genGptModelLayout, inputEmbedParts } from '../src/llm/GptModelLayout';
import { genBertModelLayout } from '../src/llm/BertModelLayout';
import { bertArrowPlan, drawBertArrows } from '../src/llm/components/Arrow';
import { drawBertBlockLabels } from '../src/llm/components/SectionLabels';
import { bertMlmChartRegion, drawBertModelCard } from '../src/llm/components/ModelCard';
import { familyOfPhase, initWalkthrough, isBertPhase, isVitPhase, Phase, PhaseGroup } from '../src/llm/walkthrough/Walkthrough';
import { configToShape } from '../src/llm/TransformerModel';

// Paths are relative to the package root, matching scripts/validate-model.ts: npm runs
// scripts with cwd = package root, while __dirname here is node_modules/.cache (the
// esbuild output directory), which is why resolving from __dirname looks in the wrong tree.
const FIGS_DIR = 'public/bert-figs';

let checks = 0;
let failures: string[] = [];

function check(name: string, cond: boolean, detail = '') {
    checks++;
    if (cond) {
        console.log(`  ok   ${name}`);
    } else {
        failures.push(name);
        console.log(`  FAIL ${name}${detail ? ` — ${detail}` : ''}`);
    }
}

function summary() {
    if (failures.length) {
        console.log(`\n${failures.length}/${checks} checks FAILED`);
        process.exit(1);
    }
    console.log(`\nall ${checks} checks passed`);
}

// ---- Task 1: the BERT paper figures are present as attributed assets (spec §6) ----
const figureStems = ['fig1-overview', 'fig2-input', 'fig3-comparison', 'fig4-fine-tune'];
for (let stem of figureStems) {
    let svgFile = `${FIGS_DIR}/${stem}.svg`;
    let pngFile = `${FIGS_DIR}/${stem}.png`;
    if (fs.existsSync(svgFile)) {
        let bytes = fs.readFileSync(svgFile);
        let text = bytes.toString('utf8');
        // pdftocairo's default SVG draws text as SVG-1.1 glyph references with no embedded
        // font, which Firefox renders as nothing: the figure shows up with its labels missing.
        // get-bert-figures.sh outlines the text; this is the check that it actually did.
        let glyphRefs = (text.match(/href="#glyph/g) ?? []).length;
        check(`figure ${stem}.svg`, bytes.length > 2048 && text.includes('<svg') && glyphRefs === 0,
            `${bytes.length} bytes, ${glyphRefs} glyph references (want 0 — text must be outlined)`);
    } else if (fs.existsSync(pngFile)) {
        let bytes = fs.readFileSync(pngFile);
        check(`figure ${stem}.png`, bytes.length > 2048, `${bytes.length} bytes`);
        console.log(`     note: ${stem} is the PNG fallback — it renders everywhere but will not scale as sharply`);
    } else {
        check(`figure ${stem}`, false, 'missing — run scripts/get-bert-figures.sh');
    }
}

// ---- Task 3: the browser's WordPiece must agree with Python's on every fixture (spec §3.3) ----
function loadJson(file: string) {
    return JSON.parse(fs.readFileSync(file, 'utf8'));
}

const model = loadJson('public/bert-sst2-model.json');
const partials = loadJson('public/bert-sst2-partials.json');
const vocab = bertVocabFromConfig(model.config);
const T = model.config.block_size as number;
const fixtures = partials.sentences as { text: string; ids: number[]; toks: string[]; truncated: boolean; rawIds: number[]; rawToks: string[] }[];

function eq(a: unknown[], b: unknown[]) {
    return a.length === b.length && a.every((x, i) => x === b[i]);
}

let mismatches: string[] = [];
for (let f of fixtures) {
    let got = wordpieceEncode(vocab, f.text, T);
    let ok = eq(got.ids, f.ids) && eq(got.toks, f.toks) && got.truncated === f.truncated;
    // The fixtures also carry the *untruncated* encoding (rawIds). Comparing that too covers the
    // pieces that fall outside the 32-cube window — including [UNK], which on a long sentence is
    // truncated away from `ids` and would otherwise leave that branch untested.
    if (ok) {
        let unc = wordpieceEncode(vocab, f.text, f.rawIds.length + 3);
        let want = [vocab.cls, ...f.rawIds, vocab.sep];
        if (unc.ids.slice(0, want.length).join() !== want.join() || unc.truncated) {
            ok = false;
            mismatches.push(`${JSON.stringify(f.text.slice(0, 40))}: uncapped pieces differ from Python's rawIds`);
        }
    }
    if (!ok) {
        let firstDiff = got.ids.findIndex((x, i) => x !== f.ids[i]);
        mismatches.push(`${JSON.stringify(f.text.slice(0, 40))}: first diff at ${firstDiff} `
            + `(js ${got.ids[firstDiff]} vs py ${f.ids[firstDiff]}), toks ${JSON.stringify(got.toks.slice(Math.max(0, firstDiff - 1), firstDiff + 3))} `
            + `vs ${JSON.stringify(f.toks.slice(Math.max(0, firstDiff - 1), firstDiff + 3))}, truncated js=${got.truncated} py=${f.truncated}`);
    }
}
check(`wordpiece agrees with Python on all ${fixtures.length} fixtures`, mismatches.length === 0,
    mismatches.slice(0, 3).join('\n         '));
console.log(`     compared ${fixtures.length} fixtures, capped and uncapped (${fixtures.filter(f => f.truncated).length} `
    + `truncated by the ${T}-cube window, ${fixtures.filter(f => f.rawToks.includes('[UNK]')).length} with [UNK])`);

check('some fixture exercises subword splitting', fixtures.some(f => f.toks.some(t => t.startsWith('##') && t !== '##')));

let unkFixture = fixtures.find(f => f.rawToks.includes('[UNK]'));
check('a fixture exercises the [UNK] branch', !!unkFixture, 'the shipped fixtures contain no [UNK] case');
if (unkFixture) {
    // [UNK] means "a character this vocabulary cannot spell at all", which replaces the whole word:
    // maelström is one [UNK], not a run of pieces ending in one.
    let got = wordpieceEncode(vocab, unkFixture.text, unkFixture.rawIds.length + 3);
    check('[UNK] replaces the unspellable word, and only there',
        got.ids.filter(x => x === vocab.unk).length === unkFixture.rawToks.filter(t => t === '[UNK]').length
        && eq(got.ids.slice(0, unkFixture.rawIds.length + 2), [vocab.cls, ...unkFixture.rawIds, vocab.sep]),
        `js ${got.ids.filter(x => x === vocab.unk).length} unk vs py ${unkFixture.rawToks.filter(t => t === '[UNK]').length}`);
}

let truncFixture = fixtures.find(f => f.truncated);
check('a fixture exercises truncation', !!truncFixture);
if (truncFixture) {
    let got = wordpieceEncode(vocab, truncFixture.text, T);
    let lastReal = got.ids.length - 1;
    while (got.ids[lastReal] === vocab.pad) lastReal--;
    // truncated at T with padding, so [SEP] is the last *non-pad* id — a pair's [SEP] is what T-3
    // reserves for, and a single sentence reserves the same room so its ids match training exactly
    check('truncated encoding keeps [CLS] first and [SEP] last, padded to T',
        got.ids[0] === vocab.cls && got.ids[lastReal] === vocab.sep && got.ids.length === T,
        `len=${got.ids.length} first=${got.ids[0]} lastReal=${got.ids[lastReal]} (T=${T})`);
    check('truncated encoding fills the whole window', got.ids.filter(x => x !== vocab.pad).length === T - 1,
        `${got.ids.filter(x => x !== vocab.pad).length} real ids`);
}

let empty = wordpieceEncode(vocab, '', T);
check('empty input encodes to [CLS] [SEP] padded to T',
    empty.ids.length === T && empty.ids[0] === vocab.cls && empty.ids[1] === vocab.sep
    && empty.ids.slice(2).every(x => x === vocab.pad) && !empty.truncated, JSON.stringify(empty.ids.slice(0, 4)));

let throwCases = ['   ', ' . , ; ', '\n\t', 'a'.repeat(200), 'maelström', '😀 emoji outside the vocab'];
let thrown: string[] = [];
let finite = true;
for (let text of throwCases) {
    try {
        let r = wordpieceEncode(vocab, text, T);
        if (r.ids.length !== T || r.ids.some(x => !Number.isInteger(x) || x < 0 || x >= vocab.vocab.length)) {
            finite = false;
            mismatches.push(`${JSON.stringify(text)} produced out-of-range ids`);
        }
    } catch (e) {
        thrown.push(`${JSON.stringify(text)}: ${e}`);
    }
}
check('whitespace/punctuation/oversized/emoji inputs do not throw', thrown.length === 0, thrown.join('; '));
check('every id they produce is a real vocab row', finite, mismatches.slice(-1).join(''));

// ---- Task 4: the forward pass, its masks, and its cost (spec §8.2) ----
// Same trick validate-model.ts uses: enough of a fake WebGL2 context that createBufferTex and
// writeToBufferTex work, so the buffers under test are the real ones. writeToBufferTex throws
// "Data length does not match" on any size slip, which is what the over-long-input test leans on.
let gl: any = new Proxy({}, { get: (_t, p) => p === 'createTexture' ? () => ({}) : typeof p === 'string' && p.toUpperCase() === p ? 0 : () => {} });

function loadTensors(file: string) {
    let data = JSON.parse(fs.readFileSync(file, 'utf8'));
    for (let k in data) {
        if (data[k].shape) data[k] = TensorF32.fromJson(data[k]);
    }
    return data;
}

let bertWeights = loadTensors('public/bert-sst2-model.json') as any;
let bert: IBertModelLink = createBertModel(gl, bertWeights, 0);

let short = 'great film';
let shortReal = bert.inputBuf.findIndex(x => x === 0);   // first [PAD]
setBertSentence(bert, short, [1]);
check('a 2-word sentence leaves room for padding', shortReal > 4, `first pad at ${shortReal}`);
check('short sentence is not flagged truncated', !bert.truncated);

// every MLM row is a distribution over the vocabulary, including the rows the user will never look at
{
    let probs = bert.mlmProbs.localBuffer!;
    let T = bert.shape.T, V = bert.shape.vocabSize;
    let worstRow = 0;
    for (let t = 0; t < T; t++) {
        let sum = 0;
        for (let i = 0; i < V; i++) sum += probs[t * V + i];
        worstRow = Math.max(worstRow, Math.abs(sum - 1));
    }
    check('every mlmProbs row sums to 1', worstRow <= 1e-4, `worst |sum-1| = ${worstRow.toExponential(2)}`);
}

// the padding mask is the one thing here that can be subtly wrong and still produce plausible
// numbers: [PAD] columns must carry exactly zero attention weight for a real query row, or every
// cube's attention map is quietly diluted by the 20-odd padding positions
{
    let sm = bert.blocks[0].selfAttn.attnMatrixSoftmax.localBuffer!;
    let { T, nHeads } = bert.shape;
    let padCols: number[] = [];
    for (let tk = 0; tk < T; tk++) if (bert.padBuf[tk]) padCols.push(tk);
    let bad = 0;
    let maxPad = 0;
    for (let h = 0; h < nHeads; h++) {
        for (let tq = 0; tq <= shortReal - 1; tq++) {          // real query rows only
            for (let tk of padCols) {
                let v = sm[(h * T + tq) * T + tk];
                if (v !== 0) { bad++; maxPad = Math.max(maxPad, v); }
            }
        }
    }
    check('pad columns get exactly 0 attention', bad === 0, `${bad} non-zero pad entries, max ${maxPad}`);
    check('pad columns carry the -1e9 sentinel pre-softmax',
        bert.blocks[0].selfAttn.attnMatrix.localBuffer![(shortReal) * T + (T - 1)] === -1e9,
        `saw ${bert.blocks[0].selfAttn.attnMatrix.localBuffer![(shortReal) * T + (T - 1)]}`);
    // and a real row still attends to something: an all-masked softmax would be NaN or uniform
    let rowSum = 0;
    for (let tk = 0; tk < T; tk++) rowSum += sm[tk];
    check('a real query row distributes its full weight over real keys', Math.abs(rowSum - 1) < 1e-5, `sum ${rowSum}`);
}

// spec §9 Review Focus #4: a sentence far longer than the window must truncate, not corrupt
{
    let long = Array.from({ length: 120 }, (_, i) => 'word' + (i % 7)).join(' ');
    let threw = '';
    try {
        setBertSentence(bert, long, [2]);
        uploadAllBert(bert);
    } catch (e) {
        threw = `${e}`;
    }
    check('a 120-word sentence does not throw', threw === '', threw);
    check('a 120-word sentence is reported as truncated', bert.truncated === true);
    let consistent = true;
    for (let t = 0; t < bert.shape.T; t++) {
        if ((bert.inputBuf[t] === bertVocabFromConfig(bert.config).pad) !== (bert.padBuf[t] === 1)) consistent = false;
    }
    check('padBuf agrees with inputBuf position by position', consistent);
}

// spec §9: degenerate inputs must leave the model showing numbers, not NaN
for (let text of ['', '   ', ' . , ; ']) {
    let threw = '';
    try {
        setBertSentence(bert, text, [0]);
    } catch (e) {
        threw = `${e}`;
    }
    let badBuffers = 0;
    for (let buf of [bert.embed.localBuffer, bert.mlmProbs.localBuffer, bert.sentimentProbs.localBuffer,
                     bert.blocks[0].selfAttn.attnMatrixSoftmax.localBuffer, bert.blocks[1].ln2.output.localBuffer]) {
        for (let i = 0; i < buf!.length; i++) if (!isFinite(buf![i])) badBuffers++;
    }
    check(`input ${JSON.stringify(text)}: no throw, all buffers finite`, threw === '' && badBuffers === 0,
        `${threw} ${badBuffers} non-finite values`);
}

{
    setBertSentence(bert, partials.text, [partials.mlmTarget]);
    let top = bertTopK(bert, partials.mlmTarget, 4);
    let ordered = top.every((o, i) => i === 0 || o.p <= top[i - 1].p + 1e-12);
    let known = top.every(o => bert.config.vocab.includes(o.tok));
    check('bertTopK returns 4 known tokens in order', top.length === 4 && known && ordered,
        `${top.length} entries, known=${known}, ordered=${ordered}`);

    // deliberately NOT an assertion that the model is right here: MLM top-1 on frequent whole words
    // measures config.metrics.mlm_top1_frequent (28.9%), so a chapter that always showed the gold
    // token on top would be theatre and the first reader who types their own sentence would see
    // through it. What the helper owes the chapter is that its numbers are the buffers' numbers.
    let probs = bert.mlmProbs.localBuffer!;
    let V = bert.shape.vocabSize;
    let pos = partials.mlmTarget;
    let agreesWithBuffer = top.every(o => {
        let id = bert.config.vocab.indexOf(o.tok);
        return Math.abs(probs[pos * V + id] - o.p) < 1e-9;
    });
    check('bertTopK reports the same probabilities the buffers hold', agreesWithBuffer);

    let goldId = bert.config.vocab.indexOf(partials.mlmTargetToken);
    let goldP = probs[pos * V + goldId];
    let goldRank = 1;
    for (let i = 0; i < V; i++) if (probs[pos * V + i] > goldP) goldRank++;
    console.log(`     for "${partials.text}" masked at ${pos}: gold "${partials.mlmTargetToken}" ranks #${goldRank} of ${V} `
        + `(top-1 here: ${top[0].tok} at ${(top[0].p * 100).toFixed(1)}%) — MLM accuracy is reported as `
        + `${(bert.config.metrics.mlm_top1_frequent * 100).toFixed(1)}% on frequent whole words, so this is the honest case, not a bug`);

    let sp = bert.sentimentProbs.localBuffer!;
    let s = bertSentiment(bert);
    check('bertSentiment agrees with sentimentProbs', s.prob === Math.max(sp[0], sp[1]) && s.label === (sp[1] >= sp[0] ? 1 : 0),
        `got label=${s.label} prob=${s.prob}`);
}

// spec §8.2 measured 10.6 ms per pass with these helpers; the budget claim depends on it, so a
// surprise here should stop the work rather than ship quietly
{
    setBertSentence(bert, partials.text);            // warm up any lazy state
    let t0 = Date.now();
    for (let i = 0; i < 20; i++) {
        setBertSentence(bert, i % 2 ? partials.text : partials.sentences[3].text, [partials.mlmTarget]);
    }
    let ms = (Date.now() - t0) / 20;
    console.log(`     forward pass: ${ms.toFixed(2)} ms per re-run of the whole model (spec §8.2 measured 10.6 ms)`);
    check('a full re-run stays inside the §8.2 budget', ms < 21.2, `${ms.toFixed(2)} ms > 2x the measured 10.6 ms`);
}

// ---- Task 5: inputEmbedParts speaks N tables, not the Transformer's two (spec §4.1) ----
// BERT adds a third embedding table (segment) between token and position, so the helper that names
// tokEmbed/posEmbed/idxObj outright cannot describe it. The refactor must leave the Transformer's
// own layout looking exactly like it did, which is what these assertions pin: same blocks, same
// names, only a more general way of handing them back.
{
    let trShape = configToShape(loadJson('public/transformer-reverse-model.json').config);
    let trLayout = genGptModelLayout(trShape as any);
    let parts = inputEmbedParts(trLayout.enc.residual0);
    check('transformer input-embed block still resolves', !!parts);
    if (parts) {
        check('transformer exposes 2 embedding tables (no segment)', parts.tables.length === 2,
            `got ${parts.tables.length}: ${parts.tables.map(t => t.what).join(', ')}`);
        check('the token table is the block the Transformer layout calls Token Embed',
            embedTable(parts, 'token')?.name === 'Token Embed', `got ${embedTable(parts, 'token')?.name}`);
        check('the position table is the block it calls Pos Encoding',
            embedTable(parts, 'position')?.name === 'Pos Encoding', `got ${embedTable(parts, 'position')?.name}`);
        check('there is no segment table to draw for the Transformer', embedTable(parts, 'segment') === null);
        check('the index row is still found', parts.idx.name === 'Tokens', `got ${parts.idx.name}`);
        check('tables come back in layout order', parts.tables.map(t => t.what).join() === 'token,position',
            parts.tables.map(t => t.what).join());
        // The decoder side embeds the same way and is the one that actually renders the lookup
        // arrows on hover, so it has to resolve under the same call.
        let decParts = inputEmbedParts(trLayout.dec.residual0);
        check('the decoder input-embed block resolves too', !!decParts && decParts.tables.length === 2);
    }
}

// ---- Task 6: the 3D layout, and the laptop budget that constrains it (spec §8.1) ----
{
    // 20 generations, timed: this runs on every hover-driven layout rebuild, so a slow layout is a
    // slow viewer. The spec's guardrail is 2 ms mean.
    let t0 = performance.now();
    let layout = genBertModelLayout(bert.shape);
    for (let i = 0; i < 19; i++) {
        genBertModelLayout(bert.shape);
    }
    let genMs = (performance.now() - t0) / 20;

    check('layout identifies its family', layout.kind === 'bert', layout.kind);
    check('one layout block per transformer layer', layout.blocks.length === bert.shape.nBlocks,
        `${layout.blocks.length} vs ${bert.shape.nBlocks}`);
    check(`layout fits the laptop cube budget (<= 160)`, layout.cubes.length <= 160,
        `${layout.cubes.length} cubes`);
    check(`layout generation is cheap (mean ${genMs.toFixed(2)} ms <= 2 ms)`, genMs <= 2, `${genMs.toFixed(3)} ms`);
    console.log(`       layout: ${layout.cubes.length} cubes, ${genMs.toFixed(2)} ms/gen, height ${layout.height.toFixed(1)}`);

    let badDim = layout.cubes.filter(b => ![b.x, b.y, b.z, b.dx, b.dy, b.dz].every(Number.isFinite) || b.dx <= 0 || b.dy <= 0 || b.dz <= 0);
    check('every cube has finite, positive extent', badDim.length === 0,
        badDim.map(b => `${b.name}(${b.dx},${b.dy},${b.dz})`).join(', '));
    check('every cube records its own index', layout.cubes.every((b, i) => b.idx === i));
    check('layout has labels', layout.labels.length > 0, `${layout.labels.length}`);
    let cubeSet = new Set(layout.cubes);
    let orphans = layout.labels.flatMap(l => l.cubes).filter(c => !cubeSet.has(c));
    check('every labelled cube is in the layout', orphans.length === 0, `${orphans.length} orphans: ${orphans.slice(0, 3).map(o => o.name).join(', ')}`);
    check('layout has positive height', layout.height > 0, `${layout.height}`);
    check('weight count covers the architecture', layout.weightCount > 0 && Number.isFinite(layout.weightCount), `${layout.weightCount}`);
}

// ---- Task 6: the layout must point at the real buffers, not placeholders ----
{
    let layout = genBertModelLayout(bert.shape, bert);
    check('the embedding block reads model.embed', layout.residual0.access?.src === bert.embed);
    check('the MLM probabilities block reads model.mlmProbs', layout.mlmProbs.access?.src === bert.mlmProbs);
    check('the sentiment block reads model.sentimentProbs', layout.sentimentProbs.access?.src === bert.sentimentProbs);
    check('the token row reads model.tokens', layout.tokens.access?.src === bert.tokens);
    check('the embed LN block reads the embedding norm output', layout.embedLN.lnResid.access?.src === bert.embedLN.output);
    check('block 1 reads its own attention output', layout.blocks[1].attn.attnOut.access?.src === bert.blocks[1].selfAttn?.proj.output);
}

// ---- Task 7: the BERT-specific arrows are a plan, so they can be checked without a GPU ----
// The renderers (triRender / lineRender / fontAtlas) need a real WebGL2 context, so "call it and see
// if it throws" would only be provable in the browser, where Task 10 checks it anyway. What is
// checkable here is the thing that actually breaks: an arrow wired to the wrong block, or one whose
// coordinates came from a block that isn't part of the layout. Both are invisible until you look at
// the render, and both are caught by asserting on the plan the drawer iterates.
{
    let layout = genBertModelLayout(bert.shape, bert);
    let plan = bertArrowPlan(layout);
    let cubeSet = new Set(layout.cubes);
    check('the arrow plan is not empty', plan.length >= 8, `${plan.length} arrows`);
    let unknown = plan.filter(a => !cubeSet.has(a.from) || !cubeSet.has(a.to));
    check('every arrow connects two blocks that are in the layout', unknown.length === 0,
        unknown.map(a => `${a.kind}:${a.from.name}->${a.to.name}`).join(', '));
    let nan = plan.filter(a => ![a.from.x, a.from.y, a.from.z, a.to.x, a.to.y, a.to.z].every(Number.isFinite));
    check('no arrow has non-finite endpoints', nan.length === 0, nan.map(a => a.kind).join(', '));

    // identity, not names: two blocks can share a name ('Layer Norm') and a wire to the wrong one of
    // them is exactly the bug worth catching
    let last = layout.blocks[layout.blocks.length - 1].output;
    let expect: [string, any, any, string][] = [
        ['embed', layout.tokens, layout.residual0, 'token ids select the rows of the embedding sum'],
        ['embed', layout.tokEmbed, layout.residual0, 'token table feeds z0'],
        ['embed', layout.segEmbed, layout.residual0, 'segment table feeds z0'],
        ['embed', layout.posEmbed, layout.residual0, 'position table feeds z0'],
        ['mlm', layout.mlmDense, layout.mlmAct, 'dense -> GELU'],
        ['mlm', layout.mlmOutWeight, layout.mlmLogits, 'tied E^T -> logits'],
        ['cls', last, layout.poolOut, 'the pooler reads [CLS] from the last block'],
        ['cls', layout.poolOut, layout.nspLogits, 'the pooler feeds NSP'],
        ['cls', last, layout.sentimentLogits, 'the classifier reads raw [CLS], not the pooled vector'],
    ];
    let missing = expect.filter(([kind, from, to]) => !plan.some(a => a.kind === kind && a.from === from && a.to === to));
    check('the plan contains the Figure 2 / Figure 1 wires', missing.length === 0,
        `missing: ${missing.map(m => m[3]).join(' | ')}`);

    // MLM chart region: it is what the camera frames on, so it has to be a real rectangle
    let region = bertMlmChartRegion(layout);
    check('the MLM chart region is a sane rectangle',
        Number.isFinite(region.x + region.y + region.dx + region.dy) && region.dx > 0 && region.dy > 0,
        JSON.stringify(region));

    // the draw functions themselves are only exercised for real in the browser (Task 10); here we
    // assert they exist and are the functions the wiring will call
    check('the three BERT draw entry points are functions',
        typeof drawBertArrows === 'function' && typeof drawBertBlockLabels === 'function'
        && typeof drawBertModelCard === 'function');
}

// ---- Task 8: which model a phase shows is set membership, not an ordinal range ----
// The old rule was `phase >= Phase.Vit_Intro`. It silently claimed every later phase for the ViT,
// so the first BERT phase added would have drawn the ViT's layout while its chapter text talked about
// BERT \u2014 a bug that only shows up as "the picture doesn't match the words".
{
    let bertPhases = [Phase.Bert_Intro, Phase.Bert_Input, Phase.Bert_Attn, Phase.Bert_Mlm, Phase.Bert_PretrainFT, Phase.Bert_Sentiment];
    let vitPhases = [Phase.Vit_Intro, Phase.Vit_Embedding, Phase.Vit_Encoder, Phase.Vit_Head, Phase.Vit_Attention];
    let trPhases = [Phase.None, Phase.Intro_Intro, Phase.Intro_Prelim, Phase.Enc_Embedding, Phase.Enc_Stack,
                    Phase.Dec_MaskedAttention, Phase.Dec_CrossAttention, Phase.Dec_Output];

    check('every BERT phase belongs to the BERT family', bertPhases.every(p => familyOfPhase(p) === 'bert'),
        bertPhases.filter(p => familyOfPhase(p) !== 'bert').map(p => Phase[p]).join(', '));
    check('no BERT phase is mistaken for the ViT', bertPhases.every(p => !isVitPhase(p)),
        bertPhases.filter(isVitPhase).map(p => Phase[p]).join(', '));
    check('every BERT phase reports itself as BERT', bertPhases.every(isBertPhase));
    check('every ViT phase still belongs to the ViT', vitPhases.every(p => isVitPhase(p) && familyOfPhase(p) === 'vit'),
        vitPhases.filter(p => !isVitPhase(p)).map(p => Phase[p]).join(', '));
    check('no ViT phase is mistaken for BERT', vitPhases.every(p => !isBertPhase(p)));
    check('the Transformer chapters are still the Transformer family', trPhases.every(p => familyOfPhase(p) === 'transformer'),
        trPhases.filter(p => familyOfPhase(p) !== 'transformer').map(p => Phase[p]).join(', '));

    // the phase ids must be appended, never inserted: SavedState stores the number in localStorage, and
    // a renumbered enum sends every returning reader to the wrong chapter
    check('BERT phase ids come after the ViT ones',
        Math.min(...bertPhases) > Math.max(...vitPhases),
        `min bert ${Math.min(...bertPhases)} vs max vit ${Math.max(...vitPhases)}`);

    let wt = initWalkthrough();
    let ids = wt.phaseList.flatMap(g => g.phases.map(p => p.id));
    let dupes = ids.filter((id, i) => ids.indexOf(id) !== i);
    check('no phase id is listed twice', dupes.length === 0, dupes.map(p => Phase[p]).join(', '));
    check('every BERT phase is reachable from the table of contents',
        bertPhases.every(p => ids.includes(p)),
        bertPhases.filter(p => !ids.includes(p)).map(p => Phase[p]).join(', '));
    let bertGroup = wt.phaseList.find(g => g.groupId === PhaseGroup.Bert);
    check('the BERT chapter group is listed under its own group', !!bertGroup && bertGroup.phases.length === bertPhases.length,
        bertGroup ? `${bertGroup.phases.length} phases` : 'no group');
    // a phase is only valid if some group claims it; phaseToGroup() assumes that with a !
    let unclaimed = ids.filter(id => !wt.phaseList.some(g => g.phases.some(p => p.id === id)));
    check('every listed phase is claimed by a group', unclaimed.length === 0);
    check('every phase in every group has a group', wt.phaseList.every(g => g.phases.every(p => typeof p.id === 'number')));
}

summary();
