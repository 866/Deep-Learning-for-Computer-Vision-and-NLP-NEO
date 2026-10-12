(globalThis as any).window ??= globalThis;
// Dev-time checks for the BERT model family (weights, tokenizer, forward pass, layout).
// Follows the same esbuild-bundle-then-node pattern as scripts/validate-model.ts.
// Run: npm run check:bert
import fs from 'fs';

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

summary();
