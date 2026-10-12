import { createBufferTex, IBufferTex, writeToBufferTex } from "@/src/utils/renderPhases";
import { TensorF32 } from "@/src/utils/tensor";
import { add, attentionForward, IAttentionLayerLink, IFfnLayerLink, ILayerNormLayerLink, ILinearLayerLink, layerNormForward, linearForward } from "./TransformerModel";

/*
The Vision Transformer (ViT) from "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020), run on the CPU in JS.

Same buffer conventions as TransformerModel.ts (rows x cols): activations are (B * T) x C, linear weights nOut x nIn.
Token 0 is the [class] token; tokens 1..N are the image patches in row-major order.
*/

export interface IVitShape {
    B: number;
    T: number;          // N patches + 1 [class] token
    C: number;          // D, the hidden size
    nHeads: number;
    A: number;          // D / heads
    nBlocks: number;    // L, the number of layers
    dFF: number;        // MLP size
    nClasses: number;
    imageSize: number;
    patchSize: number;
    channels: number;
    patchDim: number;   // P^2 * channels
}

export interface IVitConfig {
    image_size: number;
    patch_size: number;
    n_patches: number;
    block_size: number;
    n_embd: number;
    n_head: number;
    n_layer: number;
    d_mlp: number;
    n_classes: number;
    classes: string[];
    n_params: number;
    test_accuracy: number;
    n_test: number;
}

export type IVitWeights = Record<string, TensorF32> & { config: IVitConfig };

export interface IVitBlockLink {
    ln1: ILayerNormLayerLink;
    attn: IAttentionLayerLink;
    ln2: ILayerNormLayerLink;
    mlp: IFfnLayerLink;
    output: IBufferTex;
}

export interface IVitModelLink {
    gl: WebGL2RenderingContext;
    shape: IVitShape;
    config: IVitConfig;
    image: Float32Array;          // imageSize x imageSize, values in [0, 1]
    imageLabel: number | null;    // the true class, if known
    patches: IBufferTex;          // T x patchDim (row 0, the [class] position, is empty)
    patchEmbed: ILinearLayerLink; // E: patchDim -> C; output is T x C with row 0 = the [class] token
    clsToken: IBufferTex;         // C x 1
    posEmbed: IBufferTex;         // T x C
    embed: IBufferTex;            // z_0 = [x_class; x_p E] + E_pos
    blocks: IVitBlockLink[];
    lnF: ILayerNormLayerLink;
    head: ILinearLayerLink;       // C -> nClasses, applied to the [class] token only
    logits: IBufferTex;           // 1 x nClasses
    softmaxAgg: IBufferTex;       // 1 x 1 x [1/sum, max]
    probs: IBufferTex;            // 1 x nClasses
    examples: { image: Float32Array, label: number }[];
}

export function vitConfigToShape(config: IVitConfig): IVitShape {
    return {
        B: 1,
        T: config.block_size,
        C: config.n_embd,
        nHeads: config.n_head,
        A: config.n_embd / config.n_head,
        nBlocks: config.n_layer,
        dFF: config.d_mlp,
        nClasses: config.n_classes,
        imageSize: config.image_size,
        patchSize: config.patch_size,
        channels: 1,
        patchDim: config.patch_size * config.patch_size,
    };
}

/** Image patches, in row-major patch order, each flattened row-major: (nPatches x P^2). */
export function patchify(image: Float32Array, imageSize: number, patchSize: number) {
    let g = imageSize / patchSize;
    let P2 = patchSize * patchSize;
    let out = new Float32Array(g * g * P2);
    for (let py = 0; py < g; py++) {
        for (let px = 0; px < g; px++) {
            let n = py * g + px;
            for (let iy = 0; iy < patchSize; iy++) {
                for (let ix = 0; ix < patchSize; ix++) {
                    out[n * P2 + iy * patchSize + ix] = image[(py * patchSize + iy) * imageSize + px * patchSize + ix];
                }
            }
        }
    }
    return out;
}

export function createVitModel(gl: WebGL2RenderingContext, weights: IVitWeights, exampleIdx: number = 0): IVitModelLink {
    let config = weights.config;
    let shape = vitConfigToShape(config);
    let { T, C, nBlocks, dFF, nClasses, patchDim, imageSize } = shape;

    let tex = (rows: number, cols: number, channels: number = 1) => createBufferTex(gl, cols, rows, channels);

    function weightTex(name: string, rows: number, cols: number) {
        let t = weights[name];
        if (!t) {
            throw new Error(`missing tensor ${name}`);
        }
        let buf = tex(rows, cols);
        buf.localBuffer = t.toFloat32Array();
        return buf;
    }

    function linear(prefix: string, nIn: number, nOut: number, nRows: number = T): ILinearLayerLink {
        return {
            weight: weightTex(prefix + '.weight', nOut, nIn),
            bias: weightTex(prefix + '.bias', nOut, 1),
            output: tex(nRows, nOut),
        };
    }

    function layerNorm(prefix: string): ILayerNormLayerLink {
        return {
            normWeight: weightTex(prefix + '.weight', C, 1),
            normBias: weightTex(prefix + '.bias', C, 1),
            normAgg: tex(T, 1, 2),
            output: tex(T, C),
        };
    }

    function attention(prefix: string): IAttentionLayerLink {
        let { nHeads } = shape;
        return {
            qWeight: weightTex(prefix + '.q.weight', C, C),
            kWeight: weightTex(prefix + '.k.weight', C, C),
            vWeight: weightTex(prefix + '.v.weight', C, C),
            qBias: weightTex(prefix + '.q.bias', C, 1),
            kBias: weightTex(prefix + '.k.bias', C, 1),
            vBias: weightTex(prefix + '.v.bias', C, 1),
            q: tex(T, C),
            k: tex(T, C),
            v: tex(T, C),
            attnMatrix: tex(nHeads * T, T),
            attnMatrixAgg: tex(nHeads * T, 1, 2),
            attnMatrixSoftmax: tex(nHeads * T, T),
            scaledVectors: tex(T, C),
            proj: linear(prefix + '.o', C, C),
            residual: tex(T, C),
        };
    }

    function mlp(prefix: string): IFfnLayerLink {
        return {
            fc: linear(prefix + '.fc1', C, dFF),
            act: tex(T, dFF),
            proj: linear(prefix + '.fc2', dFF, C),
            residual: tex(T, C),
        };
    }

    let blocks: IVitBlockLink[] = [];
    for (let i = 0; i < nBlocks; i++) {
        let p = `blocks.${i}`;
        let m = mlp(p + '.mlp');
        blocks.push({ ln1: layerNorm(p + '.ln1'), attn: attention(p + '.attn'), ln2: layerNorm(p + '.ln2'), mlp: m, output: m.residual });
    }

    let exImages = weights['examples.images'].toFloat32Array();
    let exLabels = weights['examples.labels'].toFloat32Array();
    let nPix = imageSize * imageSize;
    let examples = Array.from(exLabels).map((label, i) => ({ image: exImages.slice(i * nPix, (i + 1) * nPix), label }));

    let model: IVitModelLink = {
        gl,
        shape,
        config,
        image: new Float32Array(nPix),
        imageLabel: null,
        patches: tex(T, patchDim),
        patchEmbed: linear('patch_embed', patchDim, C),
        clsToken: weightTex('cls_token', C, 1),
        posEmbed: weightTex('pos_embed', T, C),
        embed: tex(T, C),
        blocks,
        lnF: layerNorm('ln_f'),
        head: linear('head', C, nClasses, 1),
        logits: tex(1, nClasses),
        softmaxAgg: tex(1, 1, 2),
        probs: tex(1, nClasses),
        examples,
    };

    let ex = examples[exampleIdx] ?? examples[0];
    setVitImage(model, ex.image, ex.label);
    return model;
}

/** Sets the input image (imageSize^2 values in [0, 1]), runs the model, and uploads all values to the GPU. */
export function setVitImage(model: IVitModelLink, image: Float32Array, label: number | null) {
    model.image = new Float32Array(image);
    model.imageLabel = label;
    runVitModel(model);
    uploadAllVit(model);
}

export function runVitModel(model: IVitModelLink) {
    let { T, C, nClasses, patchDim, imageSize, patchSize } = model.shape;

    // patches: row 0 is the [class] position, which has no patch
    let p = patchify(model.image, imageSize, patchSize);
    let patches = new Float32Array(T * patchDim);
    patches.set(p, patchDim);
    model.patches.localBuffer = patches;

    // x_p E + b for every patch; row 0 holds the learnable [class] embedding instead
    let tokens = linearForward(model, model.patchEmbed, patches);
    tokens.set(model.clsToken.localBuffer!.subarray(0, C), 0);
    model.patchEmbed.output.localBuffer = tokens;

    let x = add(tokens, model.posEmbed.localBuffer!);
    model.embed.localBuffer = x;

    for (let blk of model.blocks) {
        // pre-LN: z' = MSA(LN(z)) + z ;  z = MLP(LN(z')) + z'
        let l1 = layerNormForward(model, blk.ln1, x);
        x = attentionForward(model, blk.attn, l1, l1, false, x);
        let l2 = layerNormForward(model, blk.ln2, x);
        x = mlpForward(model, blk.mlp, l2, x);
    }

    let xf = layerNormForward(model, model.lnF, x);

    // the head only sees the [class] token: y = LN(z_L^0)
    let W = model.head.weight.localBuffer!;
    let b = model.head.bias!.localBuffer!;
    let logits = new Float32Array(nClasses);
    for (let o = 0; o < nClasses; o++) {
        let a = b[o];
        for (let i = 0; i < C; i++) {
            a += xf[i] * W[o * C + i];
        }
        logits[o] = a;
    }
    model.head.output.localBuffer = logits;
    model.logits.localBuffer = logits;

    let m = Math.max(...logits);
    let sum = 0;
    for (let o = 0; o < nClasses; o++) sum += Math.exp(logits[o] - m);
    let probs = new Float32Array(nClasses);
    for (let o = 0; o < nClasses; o++) probs[o] = Math.exp(logits[o] - m) / sum;
    model.softmaxAgg.localBuffer = new Float32Array([1 / sum, m]);
    model.probs.localBuffer = probs;
}

function mlpForward(model: IVitModelLink, layer: IFfnLayerLink, x: Float32Array, residSrc: Float32Array) {
    let h = linearForward(model, layer.fc, x);
    let a = new Float32Array(h.length);
    for (let i = 0; i < h.length; i++) a[i] = gelu(h[i]);
    layer.act.localBuffer = a;
    let y = linearForward(model, layer.proj, a);
    let resid = add(residSrc, y);
    layer.residual.localBuffer = resid;
    return resid;
}

/** The exact GELU, x * Phi(x), as used by the paper's implementation (and PyTorch's default). */
export function gelu(x: number) {
    return 0.5 * x * (1 + erf(x / Math.SQRT2));
}

// erf via erfc's Chebyshev fit (Numerical Recipes, fractional error < 1.2e-7)
function erf(x: number) {
    let z = Math.abs(x);
    let t = 1 / (1 + 0.5 * z);
    let r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
        t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
        t * (-0.82215223 + t * 0.17087277)))))))));
    return x >= 0 ? 1 - r : r - 1;
}

/** The predicted class and its probability. */
export function vitPrediction(model: IVitModelLink) {
    let probs = model.probs.localBuffer!;
    let best = 0;
    for (let i = 1; i < probs.length; i++) {
        if (probs[i] > probs[best]) best = i;
    }
    return { cls: best, prob: probs[best] };
}

function allVitBuffers(model: IVitModelLink): IBufferTex[] {
    let lin = (l: ILinearLayerLink) => [l.weight, l.bias, l.output].filter(a => a) as IBufferTex[];
    let ln = (l: ILayerNormLayerLink) => [l.normAgg, l.normWeight, l.normBias, l.output];
    let attn = (l: IAttentionLayerLink) => [l.qWeight, l.kWeight, l.vWeight, l.qBias!, l.kBias!, l.vBias!, l.q, l.k, l.v,
        l.attnMatrix, l.attnMatrixAgg, l.attnMatrixSoftmax, l.scaledVectors, ...lin(l.proj), l.residual];
    let mlp = (l: IFfnLayerLink) => [...lin(l.fc), l.act, ...lin(l.proj), l.residual];
    let bufs: IBufferTex[] = [model.patches, ...lin(model.patchEmbed), model.clsToken, model.posEmbed, model.embed,
        ...ln(model.lnF), ...lin(model.head), model.logits, model.softmaxAgg, model.probs];
    for (let b of model.blocks) {
        bufs.push(...ln(b.ln1), ...attn(b.attn), ...ln(b.ln2), ...mlp(b.mlp));
    }
    return bufs;
}

export function uploadAllVit(model: IVitModelLink) {
    for (let buf of allVitBuffers(model)) {
        if (buf.localBuffer) {
            buf.rms = undefined;
            writeToBufferTex(model.gl, buf, buf.localBuffer);
        }
    }
}
