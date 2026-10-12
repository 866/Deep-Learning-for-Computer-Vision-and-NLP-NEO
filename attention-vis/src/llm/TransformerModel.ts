import { createBufferTex, IBufferTex, writeToBufferTex } from "@/src/utils/renderPhases";
import { TensorF32 } from "@/src/utils/tensor";

/*
The encoder-decoder Transformer from "Attention Is All You Need", run on the CPU in JS.

The model is tiny, so the whole forward pass takes well under a millisecond. Every intermediate value is
written into a buffer texture (laid out the same way as llm-viz's GPT buffers) so the 3D blocks can sample it.

Conventions (rows x cols, i.e. texture height x width):
  - activations:   (B * T) x C
  - linear weight: nOut x nIn (PyTorch layout)
  - attention:     (B * nHeads * T) x T, softmax aggregates (B * nHeads * T) x 1 x [1/sum, max]
  - layer norm:    aggregates (B * T) x 1 x [mean, 1/std]
*/

export interface IModelShape {
    B: number;
    T: number; // sequence length (same for source & target)
    C: number; // d_model
    nHeads: number;
    A: number; // d_k = d_v = C / nHeads
    nBlocks: number; // N (layers per stack)
    vocabSize: number;
    dFF: number;
}

export interface ITransformerConfig {
    vocab: string[];
    vocab_size: number;
    bos: number;
    block_size: number;
    n_embd: number;
    n_head: number;
    n_layer: number;
    d_ff: number;
    n_params: number;
}

export type ITransformerWeights = Record<string, TensorF32> & { config: ITransformerConfig };

export interface ILinearLayerLink {
    weight: IBufferTex;
    bias: IBufferTex | null;
    output: IBufferTex;
}

export interface ILayerNormLayerLink {
    normAgg: IBufferTex;
    normWeight: IBufferTex;
    normBias: IBufferTex;
    output: IBufferTex;
}

export interface IAttentionLayerLink {
    qWeight: IBufferTex;
    kWeight: IBufferTex;
    vWeight: IBufferTex;
    qBias?: IBufferTex | null; // the Transformer has no Q/K/V biases; ViT does
    kBias?: IBufferTex | null;
    vBias?: IBufferTex | null;
    q: IBufferTex;
    k: IBufferTex;
    v: IBufferTex;
    attnMatrix: IBufferTex;
    attnMatrixAgg: IBufferTex;
    attnMatrixSoftmax: IBufferTex;
    scaledVectors: IBufferTex;
    proj: ILinearLayerLink;
    residual: IBufferTex;
}

export interface IFfnLayerLink {
    fc: ILinearLayerLink;
    act: IBufferTex;
    proj: ILinearLayerLink;
    residual: IBufferTex;
}

export interface IEncoderBlockLink {
    selfAttn: IAttentionLayerLink;
    ln1: ILayerNormLayerLink;
    ffn: IFfnLayerLink;
    ln2: ILayerNormLayerLink;
    output: IBufferTex;
}

export interface IDecoderBlockLink {
    selfAttn: IAttentionLayerLink;
    ln1: ILayerNormLayerLink;
    crossAttn: IAttentionLayerLink;
    ln2: ILayerNormLayerLink;
    ffn: IFfnLayerLink;
    ln3: ILayerNormLayerLink;
    output: IBufferTex;
}

export interface IStackInputLink {
    inputBuf: Float32Array; // token indices (length T)
    tokens: IBufferTex;
    embed: IBufferTex; // tokEmbed * sqrt(C) + posEnc
    tokEmbedOut: IBufferTex; // tokEmbed * sqrt(C)
}

export interface ITransformerModelLink {
    gl: WebGL2RenderingContext;
    shape: IModelShape;
    config: ITransformerConfig;
    vocab: string[];
    embedWeight: IBufferTex; // shared between encoder, decoder & the pre-softmax linear layer
    posEnc: IBufferTex;
    enc: IStackInputLink & { blocks: IEncoderBlockLink[], output: IBufferTex };
    dec: IStackInputLink & { blocks: IDecoderBlockLink[], output: IBufferTex };
    logits: IBufferTex;
    softmaxAgg: IBufferTex;
    probs: IBufferTex;
    inputLen: number; // number of valid decoder input tokens
    sortedBuf: Float32Array | null; // per position: [tokIdx, prob] * vocabSize, sorted by prob
}

/**
 * Loads a model JSON written by training/*.py and turns every tensor-shaped field into a TensorF32.
 * Shared by all three families: the files differ in their `config` block, not in their structure.
 */
export async function fetchModelWeights<T>(url: string): Promise<T> {
    let resp = await fetch(url);
    let data = await resp.json();
    for (let k in data) {
        if (data[k].shape) {
            data[k] = TensorF32.fromJson(data[k]);
        }
    }
    return data;
}

export async function fetchTransformerWeights(url: string): Promise<ITransformerWeights> {
    return fetchModelWeights<ITransformerWeights>(url);
}

export function configToShape(config: ITransformerConfig): IModelShape {
    return {
        B: 1,
        T: config.block_size,
        C: config.n_embd,
        nHeads: config.n_head,
        A: config.n_embd / config.n_head,
        nBlocks: config.n_layer,
        vocabSize: config.vocab_size,
        dFF: config.d_ff,
    };
}

export function sinusoidalPositionalEncoding(T: number, C: number) {
    let pe = new Float32Array(T * C);
    for (let t = 0; t < T; t++) {
        for (let i = 0; i < C; i += 2) {
            let angle = t / Math.pow(10000, i / C);
            pe[t * C + i] = Math.sin(angle);
            if (i + 1 < C) {
                pe[t * C + i + 1] = Math.cos(angle);
            }
        }
    }
    return pe;
}

/** Creates the model, runs it greedily (autoregressively) on `src`, and uploads all values to the GPU. */
export function createTransformerModel(gl: WebGL2RenderingContext, weights: ITransformerWeights, src: number[]): ITransformerModelLink {
    let config = weights.config;
    let shape = configToShape(config);
    let { T, C, nHeads, nBlocks, vocabSize, dFF } = shape;

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

    function linear(prefix: string, nIn: number, nOut: number, bias: boolean): ILinearLayerLink {
        return {
            weight: weightTex(prefix + '.weight', nOut, nIn),
            bias: bias ? weightTex(prefix + '.bias', nOut, 1) : null,
            output: tex(T, nOut),
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
        return {
            qWeight: weightTex(prefix + '.q.weight', C, C),
            kWeight: weightTex(prefix + '.k.weight', C, C),
            vWeight: weightTex(prefix + '.v.weight', C, C),
            q: tex(T, C),
            k: tex(T, C),
            v: tex(T, C),
            attnMatrix: tex(nHeads * T, T),
            attnMatrixAgg: tex(nHeads * T, 1, 2),
            attnMatrixSoftmax: tex(nHeads * T, T),
            scaledVectors: tex(T, C),
            proj: linear(prefix + '.o', C, C, false),
            residual: tex(T, C),
        };
    }

    function ffn(prefix: string): IFfnLayerLink {
        return {
            fc: linear(prefix + '.fc1', C, dFF, true),
            act: tex(T, dFF),
            proj: linear(prefix + '.fc2', dFF, C, true),
            residual: tex(T, C),
        };
    }

    function stackInput(): IStackInputLink {
        return {
            inputBuf: new Float32Array(T),
            tokens: tex(T, 1),
            embed: tex(T, C),
            tokEmbedOut: tex(T, C),
        };
    }

    let encBlocks: IEncoderBlockLink[] = [];
    for (let i = 0; i < nBlocks; i++) {
        let p = `encoder.${i}`;
        let ln2 = layerNorm(p + '.ln2');
        encBlocks.push({ selfAttn: attention(p + '.self_attn'), ln1: layerNorm(p + '.ln1'), ffn: ffn(p + '.ffn'), ln2, output: ln2.output });
    }

    let decBlocks: IDecoderBlockLink[] = [];
    for (let i = 0; i < nBlocks; i++) {
        let p = `decoder.${i}`;
        let ln3 = layerNorm(p + '.ln3');
        decBlocks.push({
            selfAttn: attention(p + '.self_attn'), ln1: layerNorm(p + '.ln1'),
            crossAttn: attention(p + '.cross_attn'), ln2: layerNorm(p + '.ln2'),
            ffn: ffn(p + '.ffn'), ln3, output: ln3.output,
        });
    }

    let posEnc = tex(T, C);
    posEnc.localBuffer = sinusoidalPositionalEncoding(T, C);

    let model: ITransformerModelLink = {
        gl,
        shape,
        config,
        vocab: config.vocab,
        embedWeight: weightTex('embed.weight', vocabSize, C),
        posEnc,
        enc: { ...stackInput(), blocks: encBlocks, output: encBlocks[nBlocks - 1].output },
        dec: { ...stackInput(), blocks: decBlocks, output: decBlocks[nBlocks - 1].output },
        logits: tex(T, vocabSize),
        softmaxAgg: tex(T, 1, 2),
        probs: tex(T, vocabSize),
        inputLen: 1,
        sortedBuf: null,
    };

    setSourceSequence(model, src);
    return model;
}

/** Sets the encoder input, then decodes greedily until the decoder input is full. */
export function setSourceSequence(model: ITransformerModelLink, src: number[]) {
    let { T } = model.shape;
    model.enc.inputBuf.set(src.slice(0, T));
    model.dec.inputBuf.fill(0);
    model.dec.inputBuf[0] = model.config.bos;
    model.inputLen = 1;

    runTransformerModel(model);
    while (model.inputLen < T) {
        stepDecoder(model);
    }
    uploadAll(model);
}

/**
 * Greedily decodes until the decoder input has `n` valid tokens (1 = just the start token), as happens part-way through
 * generation. Positions after that hold placeholder tokens.
 */
export function setDecodedLength(model: ITransformerModelLink, n: number) {
    let { T } = model.shape;
    n = Math.max(1, Math.min(T, n));
    if (model.inputLen === n) {
        return;
    }
    model.dec.inputBuf.fill(0);
    model.dec.inputBuf[0] = model.config.bos;
    model.inputLen = 1;
    runTransformerModel(model);
    while (model.inputLen < n) {
        stepDecoder(model);
    }
    uploadAll(model);
}

/** Appends the most likely token from the last valid position to the decoder input, then re-runs the model. */
export function stepDecoder(model: ITransformerModelLink) {
    let { T, vocabSize } = model.shape;
    if (model.inputLen >= T || !model.sortedBuf) {
        return;
    }
    let tIdx = model.inputLen - 1;
    model.dec.inputBuf[tIdx + 1] = model.sortedBuf[(tIdx * vocabSize) * 2 + 0];
    model.inputLen += 1;
    runTransformerModel(model);
}

export function runTransformerModel(model: ITransformerModelLink) {
    let { T, C, vocabSize } = model.shape;
    let E = model.embedWeight.localBuffer!;

    let encX = embedInput(model, model.enc);
    for (let blk of model.enc.blocks) {
        encX = attentionForward(model, blk.selfAttn, encX, encX, false);
        encX = layerNormForward(model, blk.ln1, encX);
        encX = ffnForward(model, blk.ffn, encX);
        encX = layerNormForward(model, blk.ln2, encX);
    }
    let memory = encX;

    let decX = embedInput(model, model.dec);
    for (let blk of model.dec.blocks) {
        decX = attentionForward(model, blk.selfAttn, decX, decX, true);
        decX = layerNormForward(model, blk.ln1, decX);
        decX = attentionForward(model, blk.crossAttn, decX, memory, false);
        decX = layerNormForward(model, blk.ln2, decX);
        decX = ffnForward(model, blk.ffn, decX);
        decX = layerNormForward(model, blk.ln3, decX);
    }

    // pre-softmax linear layer, sharing the embedding matrix: logits = x @ E^T
    let logits = matmulWT(decX, E, T, C, vocabSize, null);
    model.logits.localBuffer = logits;

    let agg = new Float32Array(T * 2);
    let probs = new Float32Array(T * vocabSize);
    for (let t = 0; t < T; t++) {
        let m = -Infinity;
        for (let i = 0; i < vocabSize; i++) m = Math.max(m, logits[t * vocabSize + i]);
        let sum = 0;
        for (let i = 0; i < vocabSize; i++) sum += Math.exp(logits[t * vocabSize + i] - m);
        agg[t * 2 + 0] = 1 / sum;
        agg[t * 2 + 1] = m;
        for (let i = 0; i < vocabSize; i++) probs[t * vocabSize + i] = Math.exp(logits[t * vocabSize + i] - m) / sum;
    }
    model.softmaxAgg.localBuffer = agg;
    model.probs.localBuffer = probs;

    let sortedBuf = new Float32Array(T * vocabSize * 2);
    for (let t = 0; t < T; t++) {
        let options = [...probs.slice(t * vocabSize, (t + 1) * vocabSize)].map((v, i) => ({ v, i }));
        options.sort((a, b) => b.v - a.v);
        for (let i = 0; i < options.length; i++) {
            sortedBuf[(t * vocabSize + i) * 2 + 0] = options[i].i;
            sortedBuf[(t * vocabSize + i) * 2 + 1] = options[i].v;
        }
    }
    model.sortedBuf = sortedBuf;
}

function embedInput(model: ITransformerModelLink, stack: IStackInputLink) {
    let { T, C } = model.shape;
    let E = model.embedWeight.localBuffer!;
    let pe = model.posEnc.localBuffer!;
    let scale = Math.sqrt(C);

    let tokEmbedOut = new Float32Array(T * C);
    let out = new Float32Array(T * C);
    for (let t = 0; t < T; t++) {
        let tok = stack.inputBuf[t];
        for (let c = 0; c < C; c++) {
            tokEmbedOut[t * C + c] = E[tok * C + c] * scale;
            out[t * C + c] = tokEmbedOut[t * C + c] + pe[t * C + c];
        }
    }
    stack.tokens.localBuffer = stack.inputBuf;
    stack.tokEmbedOut.localBuffer = tokEmbedOut;
    stack.embed.localBuffer = out;
    return out;
}

/** What the forward-pass helpers need from a model (shared with the ViT). */
export interface IForwardModel {
    shape: { T: number, C: number, nHeads: number, A: number };
}

// y[n, o] = sum_i x[n, i] * W[o, i] + b[o]
export function matmulWT(x: Float32Array, W: Float32Array, N: number, nIn: number, nOut: number, b: Float32Array | null) {
    let y = new Float32Array(N * nOut);
    for (let n = 0; n < N; n++) {
        for (let o = 0; o < nOut; o++) {
            let a = b ? b[o] : 0;
            for (let i = 0; i < nIn; i++) {
                a += x[n * nIn + i] * W[o * nIn + i];
            }
            y[n * nOut + o] = a;
        }
    }
    return y;
}

export function linearForward(model: IForwardModel, layer: ILinearLayerLink, x: Float32Array) {
    let nOut = layer.weight.height;
    let nIn = layer.weight.width;
    let y = matmulWT(x, layer.weight.localBuffer!, model.shape.T, nIn, nOut, layer.bias?.localBuffer ?? null);
    layer.output.localBuffer = y;
    return y;
}

export function add(a: Float32Array, b: Float32Array) {
    let y = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) y[i] = a[i] + b[i];
    return y;
}

/**
 * Multi-head attention; returns residSrc + attention output (residSrc defaults to the query input, as in post-LN).
 *
 * `padMask` marks key positions that are padding (1 = padding) and must be kept out of every softmax —
 * BERT runs a whole batch-width of [PAD] rows that would otherwise dilute each row's distribution.
 * Leaving it out (null/undefined) leaves this function exactly as it was, which is what the Transformer
 * and ViT rely on; npm run validate is the regression proof for that claim.
 */
export function attentionForward(model: IForwardModel, layer: IAttentionLayerLink, xq: Float32Array, xkv: Float32Array, causal: boolean, residSrc: Float32Array = xq, padMask?: Uint8Array | null) {
    let { T, C, nHeads, A } = model.shape;

    let q = matmulWT(xq, layer.qWeight.localBuffer!, T, C, C, layer.qBias?.localBuffer ?? null);
    let k = matmulWT(xkv, layer.kWeight.localBuffer!, T, C, C, layer.kBias?.localBuffer ?? null);
    let v = matmulWT(xkv, layer.vWeight.localBuffer!, T, C, C, layer.vBias?.localBuffer ?? null);

    let attn = new Float32Array(nHeads * T * T);
    let agg = new Float32Array(nHeads * T * 2);
    let sm = new Float32Array(nHeads * T * T);
    let out = new Float32Array(T * C);
    let scale = 1 / Math.sqrt(A);

    for (let h = 0; h < nHeads; h++) {
        for (let tq = 0; tq < T; tq++) {
            let row = (h * T + tq) * T;
            let maxK = causal ? tq : T - 1;
            let isKey = (tk: number) => !(padMask && padMask[tk]);
            let m = -Infinity;
            for (let tk = 0; tk <= maxK; tk++) {
                if (!isKey(tk)) {
                    // masked_fill replaces the score rather than ignoring it, so the stored value is
                    // the sentinel itself; matching it exactly is what lets validate diff this buffer
                    attn[row + tk] = -1e9;
                    continue;
                }
                let a = 0;
                for (let i = 0; i < A; i++) {
                    a += q[tq * C + h * A + i] * k[tk * C + h * A + i];
                }
                attn[row + tk] = a * scale;
                m = Math.max(m, a * scale);
            }
            let sum = 0;
            for (let tk = 0; tk <= maxK; tk++) {
                if (isKey(tk)) sum += Math.exp(attn[row + tk] - m);
            }
            agg[(h * T + tq) * 2 + 0] = 1 / sum;
            agg[(h * T + tq) * 2 + 1] = m;
            for (let tk = 0; tk <= maxK; tk++) {
                sm[row + tk] = isKey(tk) ? Math.exp(attn[row + tk] - m) / sum : 0;
            }
            for (let i = 0; i < A; i++) {
                let a = 0;
                for (let tk = 0; tk <= maxK; tk++) {
                    if (isKey(tk)) a += sm[row + tk] * v[tk * C + h * A + i];
                }
                out[tq * C + h * A + i] = a;
            }
        }
    }

    layer.q.localBuffer = q;
    layer.k.localBuffer = k;
    layer.v.localBuffer = v;
    layer.attnMatrix.localBuffer = attn;
    layer.attnMatrixAgg.localBuffer = agg;
    layer.attnMatrixSoftmax.localBuffer = sm;
    layer.scaledVectors.localBuffer = out;

    let proj = linearForward(model, layer.proj, out);
    let resid = add(residSrc, proj);
    layer.residual.localBuffer = resid;
    return resid;
}

export function layerNormForward(model: IForwardModel, layer: ILayerNormLayerLink, x: Float32Array) {
    let { T, C } = model.shape;
    let w = layer.normWeight.localBuffer!;
    let b = layer.normBias.localBuffer!;
    let agg = new Float32Array(T * 2);
    let y = new Float32Array(T * C);
    let eps = 1e-5;
    for (let t = 0; t < T; t++) {
        let mean = 0;
        for (let c = 0; c < C; c++) mean += x[t * C + c];
        mean /= C;
        let v = 0;
        for (let c = 0; c < C; c++) v += (x[t * C + c] - mean) ** 2;
        let stdInv = 1 / Math.sqrt(v / C + eps);
        agg[t * 2 + 0] = mean;
        agg[t * 2 + 1] = stdInv;
        for (let c = 0; c < C; c++) {
            y[t * C + c] = (x[t * C + c] - mean) * stdInv * w[c] + b[c];
        }
    }
    layer.normAgg.localBuffer = agg;
    layer.output.localBuffer = y;
    return y;
}

function ffnForward(model: ITransformerModelLink, layer: IFfnLayerLink, x: Float32Array) {
    let h = linearForward(model, layer.fc, x);
    let a = new Float32Array(h.length);
    for (let i = 0; i < h.length; i++) a[i] = Math.max(0, h[i]);
    layer.act.localBuffer = a;
    let y = linearForward(model, layer.proj, a);
    let resid = add(x, y);
    layer.residual.localBuffer = resid;
    return resid;
}

function allBuffers(model: ITransformerModelLink): IBufferTex[] {
    let bufs: IBufferTex[] = [model.embedWeight, model.posEnc, model.logits, model.softmaxAgg, model.probs];
    let lin = (l: ILinearLayerLink) => [l.weight, l.bias, l.output].filter(a => a) as IBufferTex[];
    let ln = (l: ILayerNormLayerLink) => [l.normAgg, l.normWeight, l.normBias, l.output];
    let attn = (l: IAttentionLayerLink) => [l.qWeight, l.kWeight, l.vWeight, l.q, l.k, l.v, l.attnMatrix, l.attnMatrixAgg,
        l.attnMatrixSoftmax, l.scaledVectors, ...lin(l.proj), l.residual];
    let ffn = (l: IFfnLayerLink) => [...lin(l.fc), l.act, ...lin(l.proj), l.residual];
    for (let s of [model.enc, model.dec]) {
        bufs.push(s.tokens, s.embed, s.tokEmbedOut);
    }
    for (let b of model.enc.blocks) {
        bufs.push(...attn(b.selfAttn), ...ln(b.ln1), ...ffn(b.ffn), ...ln(b.ln2));
    }
    for (let b of model.dec.blocks) {
        bufs.push(...attn(b.selfAttn), ...ln(b.ln1), ...attn(b.crossAttn), ...ln(b.ln2), ...ffn(b.ffn), ...ln(b.ln3));
    }
    return bufs;
}

export function uploadAll(model: ITransformerModelLink) {
    for (let buf of allBuffers(model)) {
        if (buf.localBuffer) {
            buf.rms = undefined;
            writeToBufferTex(model.gl, buf, buf.localBuffer);
        }
    }
}

/** Root-mean-square of a buffer's values (cached), used to pick a display scale for each block. */
export function bufferRms(buf: IBufferTex | null | undefined, channel: number = 0): number | null {
    if (!buf?.localBuffer) {
        return null;
    }
    if (buf.rms === undefined) {
        let arr = buf.localBuffer;
        let sum = 0;
        let n = 0;
        for (let i = channel; i < arr.length; i += buf.channels) {
            sum += arr[i] * arr[i];
            n++;
        }
        buf.rms = Math.sqrt(sum / Math.max(1, n)) || 1;
    }
    return buf.rms;
}
