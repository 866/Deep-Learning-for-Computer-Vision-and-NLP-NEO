(globalThis as any).window ??= globalThis;
// Checks the JS forward passes against the PyTorch activations dumped by training/train.py and train_vit.py.
// Run: npm run validate
import fs from 'fs';
import { TensorF32 } from '../src/utils/tensor';
import { createTransformerModel, ITransformerModelLink } from '../src/llm/TransformerModel';
import { createVitModel, setVitImage } from '../src/llm/VitModel';
import { createBertModel, setBertSentence } from '../src/llm/BertModel';

function load(path: string) {
    let data = JSON.parse(fs.readFileSync(path, 'utf8'));
    for (let k in data) {
        if (data[k].shape) data[k] = TensorF32.fromJson(data[k]);
    }
    return data;
}

let weights = load('public/transformer-reverse-model.json');
let partials = load('public/transformer-reverse-partials.json');

// fake just enough of WebGL2 for createBufferTex/writeToBufferTex
let gl: any = new Proxy({}, { get: (_t, p) => p === 'createTexture' ? () => ({}) : typeof p === 'string' && p.toUpperCase() === p ? 0 : () => {} });

let src = [...partials.src.toFloat32Array()];
let model: ITransformerModelLink = createTransformerModel(gl, weights, src);

let worst = 0;
function check(name: string, actual: Float32Array | undefined, permute?: (t: TensorF32) => Float32Array) {
    let t: TensorF32 = partials[name];
    if (!t) {
        console.log(`${name}: missing from the partials`);
        worst = Infinity;
        return;
    }
    let expected = permute ? permute(t) : t.toFloat32Array();
    if (!actual || actual.length !== expected.length) {
        console.log(`${name}: size mismatch ${actual?.length} vs ${expected.length}`);
        worst = Infinity;
        return;
    }
    let maxDiff = 0;
    for (let i = 0; i < expected.length; i++) {
        let e = expected[i], a = actual[i];
        if (!isFinite(e)) e = 0; // masked (-inf) attention entries are stored as 0
        maxDiff = Math.max(maxDiff, Math.abs(e - a));
    }
    worst = Math.max(worst, maxDiff);
    console.log(`${name.padEnd(28)} max |diff| = ${maxDiff.toExponential(2)}`);
}

// (h, T, A) -> (T, h*A)
let headsToCols = (t: TensorF32) => {
    let [h, T, A] = t.shape;
    let src = t.toFloat32Array();
    let out = new Float32Array(T * h * A);
    for (let i = 0; i < h; i++) for (let j = 0; j < T; j++) for (let a = 0; a < A; a++) out[j * h * A + i * A + a] = src[(i * T + j) * A + a];
    return out;
};

console.log('decoder input:', [...model.dec.inputBuf].map(i => model.vocab[i]).join(' '));
check('enc.embed', model.enc.embed.localBuffer);
check('dec.embed', model.dec.embed.localBuffer);
model.enc.blocks.forEach((b, i) => {
    check(`enc.${i}.selfAttn.q`, b.selfAttn.q.localBuffer, headsToCols);
    check(`enc.${i}.selfAttn.attn`, b.selfAttn.attnMatrix.localBuffer);
    check(`enc.${i}.selfAttn.attnSm`, b.selfAttn.attnMatrixSoftmax.localBuffer);
    check(`enc.${i}.selfAttn.vOut`, b.selfAttn.scaledVectors.localBuffer);
    check(`enc.${i}.attnResid`, b.selfAttn.residual.localBuffer);
    check(`enc.${i}.ln1`, b.ln1.output.localBuffer);
    check(`enc.${i}.ffnAct`, b.ffn.act.localBuffer);
    check(`enc.${i}.ln2`, b.ln2.output.localBuffer);
});
model.dec.blocks.forEach((b, i) => {
    check(`dec.${i}.selfAttn.attn`, b.selfAttn.attnMatrix.localBuffer);
    check(`dec.${i}.selfAttn.attnSm`, b.selfAttn.attnMatrixSoftmax.localBuffer);
    check(`dec.${i}.ln1`, b.ln1.output.localBuffer);
    // cross-attention: Q from the decoder, K and V from the encoder output (the memory)
    check(`dec.${i}.crossAttn.q`, b.crossAttn.q.localBuffer, headsToCols);
    check(`dec.${i}.crossAttn.k`, b.crossAttn.k.localBuffer, headsToCols);
    check(`dec.${i}.crossAttn.v`, b.crossAttn.v.localBuffer, headsToCols);
    check(`dec.${i}.crossAttn.vOut`, b.crossAttn.scaledVectors.localBuffer);
    check(`dec.${i}.crossAttn.attnSm`, b.crossAttn.attnMatrixSoftmax.localBuffer);
    check(`dec.${i}.ln2`, b.ln2.output.localBuffer);
    check(`dec.${i}.ln3`, b.ln3.output.localBuffer);
});
check('logits', model.logits.localBuffer);
check('probs', model.probs.localBuffer);

// ---- Vision Transformer ----
console.log('\nViT (digits):');
let vitWeights = load('public/vit-digits-model.json');
partials = load('public/vit-digits-partials.json');
let vit = createVitModel(gl, vitWeights, 0);
setVitImage(vit, partials.image.toFloat32Array(), null);

// (T-1, P^2) patches, with an empty row for the [class] token in JS
let withClsRow = (t: TensorF32) => {
    let src = t.toFloat32Array();
    let out = new Float32Array(src.length + t.shape[1]);
    out.set(src, t.shape[1]);
    return out;
};
check('patches', vit.patches.localBuffer, withClsRow);
check('tokens', vit.patchEmbed.output.localBuffer);
check('embed', vit.embed.localBuffer);
vit.blocks.forEach((b, i) => {
    check(`${i}.ln1`, b.ln1.output.localBuffer);
    check(`${i}.attn.q`, b.attn.q.localBuffer, headsToCols);
    check(`${i}.attn.attn`, b.attn.attnMatrix.localBuffer);
    check(`${i}.attn.attnSm`, b.attn.attnMatrixSoftmax.localBuffer);
    check(`${i}.attnResid`, b.attn.residual.localBuffer);
    check(`${i}.ln2`, b.ln2.output.localBuffer);
    check(`${i}.mlpAct`, b.mlp.act.localBuffer);
    check(`${i}.mlpResid`, b.mlp.residual.localBuffer);
});
check('lnf', vit.lnF.output.localBuffer);
check('logits', vit.logits.localBuffer);
check('probs', vit.probs.localBuffer);

// ---- BERT (SST-2) ----
console.log('\nBERT (SST-2):');
let bertWeights = load('public/bert-sst2-model.json');
partials = load('public/bert-sst2-partials.json');
let bert = createBertModel(gl, bertWeights, 0);

// The token ids come first and are not a tolerance check: if the browser's WordPiece produces even
// one different id, every cube after it is labelled with the wrong word and no amount of numerical
// agreement downstream is worth anything.
// the unmasked sentence: [] asks for no [MASK], which is what these partials were exported from
// (omitting the argument instead would apply the demo's own mask rule and compare two different inputs)
setBertSentence(bert, partials.text as unknown as string, []);
{
    let wantIds = partials.ids.toFloat32Array();
    let wantSeg = partials.seg.toFloat32Array();
    let bad = -1;
    for (let i = 0; i < wantIds.length; i++) {
        if (bert.inputBuf[i] !== wantIds[i] || bert.segBuf[i] !== wantSeg[i]) { bad = i; break; }
    }
    if (bad >= 0) {
        console.log(`token ids: MISMATCH at position ${bad}: js ${bert.inputBuf[bad]}/${bert.segBuf[bad]} ` +
            `vs py ${wantIds[bad]}/${wantSeg[bad]} (${bert.vocab[bert.inputBuf[bad]]} vs ${bertWeights.config.vocab[wantIds[bad]]})`);
        worst = Infinity;
    } else {
        console.log(`token ids                      exact match over ${wantIds.length} positions`);
    }
}

check('embedLN', bert.embedLN.output.localBuffer);
bert.blocks.forEach((b, i) => {
    check(`blocks.${i}.q`, b.selfAttn.q.localBuffer);
    check(`blocks.${i}.attn`, b.selfAttn.attnMatrix.localBuffer);
    check(`blocks.${i}.attnSm`, b.selfAttn.attnMatrixSoftmax.localBuffer);
    check(`blocks.${i}.resid`, b.selfAttn.residual.localBuffer);
    check(`blocks.${i}.ln1`, b.ln1.output.localBuffer);
    check(`blocks.${i}.ffnAct`, b.ffn.act.localBuffer);
    check(`blocks.${i}.ln2`, b.ln2.output.localBuffer);
});
// the partials' "embed" is the encoder's final output (PyTorch called the embedding sum z0 and the
// result x); the link's own `embed` buffer holds z0, which the layout draws as the residual base
check('embed', bert.blocks[bert.blocks.length - 1].output.localBuffer);
check('sentimentProbs', bert.sentimentProbs.localBuffer);

// the MLM head reads a [MASK]ed copy of the same sentence, so it gets its own pass
setBertSentence(bert, partials.text as unknown as string, [partials.mlmTarget as unknown as number]);
{
    let wantMlm = partials.mlmIds.toFloat32Array();
    let bad = [...wantMlm].findIndex((v, i) => v !== bert.inputBuf[i]);
    if (bad >= 0) {
        console.log(`masked ids: MISMATCH at ${bad}: js ${bert.inputBuf[bad]} vs py ${wantMlm[bad]}`);
        worst = Infinity;
    } else {
        console.log(`masked ids                     exact match (target ${partials.mlmTargetToken})`);
    }
}
check('mlmLogits', bert.mlmLogits.localBuffer);
check('mlmProbs', bert.mlmProbs.localBuffer);

if (worst > 1e-3) {
    console.error(`FAILED (worst diff ${worst})`);
    process.exit(1);
}
console.log('OK');
