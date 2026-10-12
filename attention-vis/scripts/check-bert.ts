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

summary();
