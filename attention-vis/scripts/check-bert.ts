(globalThis as any).window ??= globalThis;
// Dev-time checks for the BERT model family (weights, tokenizer, forward pass, layout).
// Follows the same esbuild-bundle-then-node pattern as scripts/validate-model.ts.
// Run: npm run check:bert
import fs from 'fs';
import { bertVocabFromConfig, wordpieceEncode } from '../src/llm/BertModel';

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

summary();
