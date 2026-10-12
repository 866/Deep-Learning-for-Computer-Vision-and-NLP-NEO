// BERT in the browser: tokenizer first, then the model. The tokenizer has to agree with Python's
// `tokenizers` crate bit for bit: the visualization prints token ids next to the cubes they index
// into, and a drift of one id would silently mislabel the whole sentence. npm run check:bert
// compares every fixture here against the ids training/train_bert.py exported, so "agrees" is a
// tested claim rather than an intention.
import { createBufferTex, IBufferTex, writeToBufferTex } from "@/src/utils/renderPhases";
import { TensorF32 } from "@/src/utils/tensor";
import {
    add, attentionForward, IAttentionLayerLink, IFfnLayerLink, IForwardModel, ILayerNormLayerLink,
    ILinearLayerLink, layerNormForward, linearForward, matmulWT,
} from "./TransformerModel";
import { gelu } from "./VitModel";

export interface IBertVocab {
    vocab: string[];       // id -> token, exactly as config.vocab
    pad: number;
    unk: number;
    cls: number;
    sep: number;
    mask: number;
}

// The vocab array and the five special ids travel together: a tokenizer handed one without the
// other would print ids against the wrong rows, and every caller needs both anyway. This is
// spec §3.2's wordpieceEncode with its two loose arguments fused into one object.
export function bertVocabFromConfig(config: { vocab: string[]; special: Record<string, number> }): IBertVocab {
    let cached = vocabCache.get(config);
    if (cached) return cached;
    let v: IBertVocab = {
        vocab: config.vocab,
        pad: config.special.pad,
        unk: config.special.unk,
        cls: config.special.cls,
        sep: config.special.sep,
        mask: config.special.mask,
    };
    vocabCache.set(config, v);
    return v;
}

// token -> id, plus the set of characters the trained pieces can actually spell. Both are built
// once per vocab object: a WeakMap rather than a field keeps the interface free of cache noise and
// lets a replaced vocab be collected with its index.
interface IVocabIndex {
    ids: Map<string, number>;
    chars: Set<string>;
}

const indexCache = new WeakMap<IBertVocab, IVocabIndex>();

// The derived index is keyed on the IBertVocab object, so the same config must keep handing back
// the same one — this runs on every keystroke in the sentence picker.
const vocabCache = new WeakMap<object, IBertVocab>();

function vocabIndex(v: IBertVocab): IVocabIndex {
    let cached = indexCache.get(v);
    if (!cached) {
        const ids = new Map<string, number>();
        const chars = new Set<string>();
        for (let i = 0; i < v.vocab.length; i++) {
            const tok = v.vocab[i];
            ids.set(tok, i);
            // Special tokens ([PAD] etc.) don't contribute spellable characters: a WordPiece model
            // knows the letters of its own pieces, and the specials were added around it. (In this
            // vocabulary both choices encode all 68 221 SST-2 sentences identically, so nothing
            // observable depends on the call — it's made for the case where something does.)
            if (tok === '[PAD]' || tok === '[UNK]' || tok === '[CLS]' || tok === '[SEP]' || tok === '[MASK]') continue;
            for (const c of tok.startsWith('##') ? tok.slice(2) : tok) chars.add(c);
        }
        cached = { ids, chars };
        indexCache.set(v, cached);
    }
    return cached;
}

// The crate's Whitespace pre-tokenizer splits on Unicode White_Space, not on ASCII only.
const WHITESPACE = /[\p{White_Space}]+/u;

// WordPiece's guard: a single word longer than this is unrecoverable, so the crate replaces the
// whole word with one [UNK] rather than attempting it.
const MAX_CHARS_PER_WORD = 100;

// The "##" prefix is suppressed when a piece crosses a punctuation boundary in *either* direction,
// and required when it stays on the same side of one — verified against the crate on all four
// shapes:  n! -> n + !      but  x!!! -> x + ! + ##! + ##!
//                                        !!n -> ! + ##! + n
//                              it's -> it + ' + s   (bare `s`, not ##s, which is also in the vocab)
// A plain OR gets 49 SST-2 sentences wrong (the censored-word "***" runs); XOR gets 0 wrong.
const FILTER_CHAR = /[\p{White_Space}\p{P}\p{S}]/u;
function isFilterChar(c: string): boolean {
    return FILTER_CHAR.test(c);
}
function isAllFilter(s: string[]): boolean {
    return s.length > 0 && s.every(isFilterChar);
}

/**
 * Encode one sentence as [CLS] pieces… [SEP], padded to `maxLen`.
 *
 * `maxLen` is the sequence length T. Content is capped at T-3 pieces, not T-2: a sentence *pair*
 * needs a second [SEP] at the end, and a single sentence reserves the same room so that its ids are
 * byte-identical to what training produced (69% of SST-2 dev sentences are affected by this cap —
 * see the truncation metric). Callers pass config.block_size.
 */
export function wordpieceEncode(v: IBertVocab, text: string, maxLen: number):
    { ids: number[]; toks: string[]; truncated: boolean } {
    const index = vocabIndex(v);
    const maxPieces = Math.max(0, maxLen - 3);    const pieces: number[] = [];
    let truncated = false;

    const push = (id: number) => {
        if (pieces.length < maxPieces) {
            pieces.push(id);
        } else {
            truncated = true;
        }
    };

    for (const word of text.split(WHITESPACE)) {
        if (word === '') continue;
        const chars = Array.from(word);        // code points, so a non-BMP glyph is one char
        // A word containing a character no piece can spell (an accented vowel outside this 600-token
        // vocab) or exceeding the length guard becomes ONE [UNK] and is skipped whole — maelström is
        // [UNK], not m + ##a + ##el + … + [UNK]. Getting this wrong shifts every id after it.
        if (chars.length > MAX_CHARS_PER_WORD || chars.some(c => !index.chars.has(c))) {
            push(v.unk);
            continue;
        }
        let start = 0;
        let prevAllFilter = false;
        while (start < chars.length) {
            const isFirst = start === 0 || isFilterChar(chars[start]) !== prevAllFilter;
            let found = -1;
            let foundEnd = start;
            for (let end = chars.length; end > start; end--) {       // greedy: longest piece first
                const piece = (isFirst ? '' : '##') + chars.slice(start, end).join('');
                const id = index.ids.get(piece);
                if (id !== undefined) {
                    found = id;
                    foundEnd = end;
                    break;
                }
            }
            if (found < 0) {
                // No piece covers what is left of this word. The crate emits a single [UNK] and
                // abandons the remainder of the word rather than retrying shorter prefixes.
                push(v.unk);
                break;
            }
            push(found);
            prevAllFilter = isAllFilter(chars.slice(start, foundEnd));
            start = foundEnd;
        }
    }

    const ids = [v.cls, ...pieces, v.sep];
    while (ids.length < maxLen) ids.push(v.pad);
    return { ids, toks: ids.map(id => v.vocab[id]), truncated };
}

// ---------------------------------------------------------------------------------------------
// The model. BERT-TINY on SST-2, run on the CPU in JS and written into buffer textures with the
// same conventions the Transformer and ViT use, so the layout builder, the shaders and the arrow
// drawer treat all three families identically:
//
//   activations:   T x C          linear weight: nOut x nIn (PyTorch layout)
//   attention:     (nHeads*T) x T with aggregates (nHeads*T) x 1 x [1/sum, max]
//   layer norm:    aggregates T x 1 x [mean, 1/std]
//
// Differences from the tiny Transformer worth knowing about, all of them BERT's rather than ours:
// the attention is bidirectional (no causal mask) but carries a *padding* mask; the encoder blocks
// are post-LN with two norms; the FFN is GELU, not ReLU; and the MLM output layer is tied to the
// token-embedding matrix, so it needs no weight of its own beyond its bias.
// ---------------------------------------------------------------------------------------------

export interface IBertMetrics {
    sentiment_dev_acc: number;
    sentiment_dev_acc_untruncated: number;
    n_dev_untruncated: number;
    majority_baseline: number;
    mlm_top1_all: number;
    mlm_top1_frequent: number;
    nsp_dev_acc: number;
    truncation_rate: number;
    n_dev: number;
}

export interface IBertShape {
    B: number;
    T: number;        // sequence length (32 — see the truncation metric)
    C: number;        // hidden size
    nHeads: number;
    A: number;        // C / nHeads
    nBlocks: number;
    dFF: number;
    vocabSize: number;
    nSegments: number;
}

export interface IBertConfig {
    vocab: string[];
    vocab_size: number;
    block_size: number;
    n_embd: number;
    n_head: number;
    n_layer: number;
    d_ff: number;
    n_params: number;
    special: Record<string, number>;
    classes: string[];
    metrics: IBertMetrics;
}

export type IBertWeights = Record<string, TensorF32> & {
    config: IBertConfig;
    'examples.text': string[];
    'examples.labels': TensorF32;
};

export interface IBertBlockLink {
    selfAttn: IAttentionLayerLink;
    ln1: ILayerNormLayerLink;
    ffn: IFfnLayerLink;
    ln2: ILayerNormLayerLink;
    output: IBufferTex;   // == ln2.output; the last block's is the encoder's output
}

export interface IBertModelLink {
    gl: WebGL2RenderingContext;
    shape: IBertShape;
    config: IBertConfig;
    vocab: string[];

    embedWeight: IBufferTex;   // V x C, shared with the MLM output layer (paper §3: "tied")
    segEmbed: IBufferTex;      // 2 x C
    posEmbed: IBufferTex;      // T x C, learned (paper §3.1: sinusoidal "we leave for future work")

    embedLN: ILayerNormLayerLink;   // the norm applied to token+segment+position before block 0

    inputBuf: Float32Array;    // token ids, length T
    segBuf: Float32Array;      // segment ids, length T
    padBuf: Uint8Array;        // 1 where the token is [PAD]; the attention mask is derived from this

    tokens: IBufferTex;        // T x 1
    segIds: IBufferTex;        // T x 1
    embed: IBufferTex;         // T x C — z0, the embedding sum before embedLN

    blocks: IBertBlockLink[];

    mlm: { fc: ILinearLayerLink; act: IBufferTex; ln: ILayerNormLayerLink };
    mlmBias: IBufferTex;       // V, the output bias that stands in for an untied embedding
    mlmLogits: IBufferTex;     // T x V
    mlmAgg: IBufferTex;        // T x 1 x [1/sum, max], the layout's softmax aggregates
    mlmProbs: IBufferTex;      // T x V

    pool: ILinearLayerLink;    // C x C, tanh-applied — used by NSP only, never by the classifier
    nsp: ILinearLayerLink;     // 2 x C
    nspLogits: IBufferTex;     // 1 x 2 (the same texture as nsp.output)
    nspProbs: IBufferTex;      // 1 x 2

    clsWeight: IBufferTex;     // 2 x C, applied to the raw [CLS] vector (paper §4.1: no pooler)
    clsBias: IBufferTex;
    sentimentLogits: IBufferTex;   // 1 x 2
    sentimentProbs: IBufferTex;    // 1 x 2

    examples: { text: string; label: number; ids: number[] }[];
    masked: number[];          // positions forced to [MASK] by the last setBertSentence call
    text: string;
    truncated: boolean;        // the sentence had more pieces than T-3, so the tail is not there
}

export function bertConfigToShape(config: IBertConfig): IBertShape {
    return {
        B: 1,
        T: config.block_size,
        C: config.n_embd,
        nHeads: config.n_head,
        A: config.n_embd / config.n_head,
        nBlocks: config.n_layer,
        dFF: config.d_ff,
        vocabSize: config.vocab_size,
        nSegments: 2,
    };
}

export function createBertModel(gl: WebGL2RenderingContext, weights: IBertWeights, exampleIdx: number = 0): IBertModelLink {
    let config = weights.config;
    let shape = bertConfigToShape(config);
    let { T, C, nHeads, A, nBlocks, dFF, vocabSize } = shape;

    let tex = (rows: number, cols: number, channels: number = 1) => createBufferTex(gl, cols, rows, channels);

    function weightTex(name: string, rows: number, cols: number): IBufferTex {
        let t = weights[name];
        if (!t) {
            throw new Error(`missing tensor ${name}`);
        }
        let buf = tex(rows, cols);
        buf.localBuffer = t.toFloat32Array();
        return buf;
    }

    // `rows` is T for anything the blocks touch (so the layout's textures are uniform) and 1 for the
    // [CLS]-only heads, whose textures are one row wide; writeToBufferTex rejects a length mismatch,
    // which is the guarantee the over-long-input test relies on.
    function linear(prefix: string, nIn: number, nOut: number, bias: boolean, rows: number = T): ILinearLayerLink {
        return {
            weight: weightTex(prefix + '.weight', nOut, nIn),
            bias: bias ? weightTex(prefix + '.bias', nOut, 1) : null,
            output: tex(rows, nOut),
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
            proj: linear(prefix + '.o', C, C, true),
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

    let blocks: IBertBlockLink[] = [];
    for (let i = 0; i < nBlocks; i++) {
        let p = `blocks.${i}`;
        let ln2 = layerNorm(p + '.ln2');
        blocks.push({ selfAttn: attention(p + '.attn'), ln1: layerNorm(p + '.ln1'), ffn: ffn(p + '.ffn'), ln2, output: ln2.output });
    }

    let nsp = linear('nsp', C, 2, true, 1);
    let model: IBertModelLink = {
        gl,
        shape,
        config,
        vocab: config.vocab,
        embedWeight: weightTex('embed.weight', vocabSize, C),
        segEmbed: weightTex('seg_embed.weight', 2, C),
        posEmbed: weightTex('pos_embed.weight', T, C),
        embedLN: layerNorm('embed_ln'),
        inputBuf: new Float32Array(T),
        segBuf: new Float32Array(T),
        padBuf: new Uint8Array(T),
        tokens: tex(T, 1),
        segIds: tex(T, 1),
        embed: tex(T, C),
        blocks,
        mlm: { fc: linear('mlm.fc', C, C, true), act: tex(T, C), ln: layerNorm('mlm.ln') },
        mlmBias: weightTex('mlm.bias', vocabSize, 1),
        mlmLogits: tex(T, vocabSize),
        mlmAgg: tex(T, 1, 2),
        mlmProbs: tex(T, vocabSize),
        pool: linear('pool', C, C, true, 1),
        nsp,
        nspLogits: nsp.output,
        nspProbs: tex(1, 2),
        clsWeight: weightTex('cls.weight', 2, C),
        clsBias: weightTex('cls.bias', 2, 1),
        sentimentLogits: tex(1, 2),
        sentimentProbs: tex(1, 2),
        examples: [],
        masked: [],
        text: '',
        truncated: false,
    };

    // the picker's example sentences, tokenised once so the first frame has something to show
    let vocab = bertVocabFromConfig(config);
    let texts = weights['examples.text'] ?? [];
    let labels = weights['examples.labels'] ? weights['examples.labels'].toFloat32Array() : new Float32Array(0);
    model.examples = texts.map((text, i) => ({
        text,
        label: labels[i] ?? 0,
        ids: wordpieceEncode(vocab, text, T).ids,
    }));

    setBertSentence(model, texts.length ? texts[Math.min(exampleIdx, texts.length - 1)] : '');
    return model;
}

/**
 * Tokenises `text`, forces `maskPositions` to [MASK] (that is how a chapter asks "what goes
 * here?"), and runs the whole model. This is the only edit path: typing a sentence re-runs every
 * block, which is the point — the numbers on screen are that sentence's.
 */
export function setBertSentence(model: IBertModelLink, text: string, maskPositions?: number[]) {
    let { T } = model.shape;
    let v = bertVocabFromConfig(model.config);
    let enc = wordpieceEncode(v, text, T);
    let ids = enc.ids.slice();

    model.text = text;
    model.truncated = enc.truncated;
    model.masked = [];
    for (let p of maskPositions ?? []) {
        if (p >= 0 && p < T && ids[p] !== v.pad) {
            ids[p] = v.mask;
            model.masked.push(p);
        }
    }

    for (let t = 0; t < T; t++) {
        model.inputBuf[t] = ids[t];
        model.segBuf[t] = 0;                                   // single sentences: segment A throughout
        model.padBuf[t] = ids[t] === v.pad ? 1 : 0;
    }

    runBertModel(model);
}

/** The GELU feed-forward with a residual add, mirroring the Transformer's ReLU one. */
function ffnBert(model: IForwardModel, layer: IFfnLayerLink, x: Float32Array) {
    let h = linearForward(model, layer.fc, x);
    let a = new Float32Array(h.length);
    for (let i = 0; i < h.length; i++) {
        a[i] = gelu(h[i]);
    }
    layer.act.localBuffer = a;
    let y = linearForward(model, layer.proj, a);
    let resid = add(x, y);
    layer.residual.localBuffer = resid;
    return resid;
}

function softmaxRows(logits: Float32Array, rows: number, n: number) {
    let agg = new Float32Array(rows * 2);
    let probs = new Float32Array(rows * n);
    for (let r = 0; r < rows; r++) {
        let m = -Infinity;
        let off = r * n;
        for (let i = 0; i < n; i++) m = Math.max(m, logits[off + i]);
        // one exp per element, normalised in place afterwards: this is over the whole vocabulary at
        // every position and it runs again on every keystroke, so the second exp the obvious
        // two-pass version does is the largest single waste in the file
        let sum = 0;
        for (let i = 0; i < n; i++) {
            let e = Math.exp(logits[off + i] - m);
            probs[off + i] = e;
            sum += e;
        }
        let inv = 1 / sum;
        agg[r * 2 + 0] = inv;
        agg[r * 2 + 1] = m;
        for (let i = 0; i < n; i++) probs[off + i] *= inv;
    }
    return { agg, probs };
}

export function runBertModel(model: IBertModelLink) {
    let { T, C, vocabSize } = model.shape;
    let E = model.embedWeight.localBuffer!;

    // z0 = token embedding + segment embedding + position embedding (paper Figure 2), then one norm
    let z0 = new Float32Array(T * C);
    for (let t = 0; t < T; t++) {
        let eOff = model.inputBuf[t] * C;
        let sOff = model.segBuf[t] * C;
        for (let c = 0; c < C; c++) {
            z0[t * C + c] = E[eOff + c] + model.segEmbed.localBuffer![sOff + c] + model.posEmbed.localBuffer![t * C + c];
        }
    }
    model.tokens.localBuffer = model.inputBuf;
    model.segIds.localBuffer = model.segBuf;
    model.embed.localBuffer = z0;

    let x = layerNormForward(model, model.embedLN, z0);

    for (let blk of model.blocks) {
        // bidirectional: no causal mask, but [PAD] keys are excluded from every softmax
        x = attentionForward(model, blk.selfAttn, x, x, false, x, model.padBuf);
        x = layerNormForward(model, blk.ln1, x);
        x = ffnBert(model, blk.ffn, x);
        x = layerNormForward(model, blk.ln2, x);
    }

    // ---- masked language model head, over the whole sequence at once ----
    let mlmH = linearForward(model, model.mlm.fc, x);
    let mlmA = new Float32Array(mlmH.length);
    for (let i = 0; i < mlmH.length; i++) {
        mlmA[i] = gelu(mlmH[i]);
    }
    model.mlm.act.localBuffer = mlmA;
    let mlmX = layerNormForward(model, model.mlm.ln, mlmA);
    // tied output layer: logits = h @ E^T + mlm.bias, with E the token-embedding matrix itself
    let mlmLogits = matmulWT(mlmX, E, T, C, vocabSize, model.mlmBias.localBuffer!);
    model.mlmLogits.localBuffer = mlmLogits;
    let mlmSm = softmaxRows(mlmLogits, T, vocabSize);
    model.mlmAgg.localBuffer = mlmSm.agg;
    model.mlmProbs.localBuffer = mlmSm.probs;

    // ---- the two [CLS]-only heads. BERT's own pooler feeds NSP and nothing else; the task
    //      classifier reads the raw [CLS] vector (paper §4.1), which is why both exist here.
    let cls = x.slice(0, C);
    let pooled = new Float32Array(C);
    let poolRaw = matmulWT(cls, model.pool.weight.localBuffer!, 1, C, C, model.pool.bias!.localBuffer!);
    for (let c = 0; c < C; c++) {
        pooled[c] = Math.tanh(poolRaw[c]);
    }
    model.pool.output.localBuffer = pooled;
    let nspLogits = matmulWT(pooled, model.nsp.weight.localBuffer!, 1, C, 2, model.nsp.bias!.localBuffer!);
    model.nsp.output.localBuffer = nspLogits;
    model.nspProbs.localBuffer = softmaxRows(nspLogits, 1, 2).probs;

    let sentLogits = matmulWT(cls, model.clsWeight.localBuffer!, 1, C, 2, model.clsBias.localBuffer!);
    model.sentimentLogits.localBuffer = sentLogits;
    model.sentimentProbs.localBuffer = softmaxRows(sentLogits, 1, 2).probs;
}

/** Top-k most likely tokens for one position, straight out of the MLM distribution. */
export function bertTopK(model: IBertModelLink, pos: number, k: number): { tok: string; p: number }[] {
    let { T, vocabSize } = model.shape;
    let probs = model.mlmProbs.localBuffer;
    if (!probs || pos < 0 || pos >= T) {
        return [];
    }
    let options: { tok: string; p: number }[] = [];
    for (let i = 0; i < vocabSize; i++) {
        options.push({ tok: model.vocab[i], p: probs[pos * vocabSize + i] });
    }
    options.sort((a, b) => b.p - a.p);
    return options.slice(0, k);
}

export function bertSentiment(model: IBertModelLink): { label: number; prob: number } {
    let probs = model.sentimentProbs.localBuffer!;
    return probs[1] >= probs[0] ? { label: 1, prob: probs[1] } : { label: 0, prob: probs[0] };
}

function allBertBuffers(model: IBertModelLink): IBufferTex[] {
    let bufs: IBufferTex[] = [
        model.embedWeight, model.segEmbed, model.posEmbed, model.embed, model.tokens, model.segIds,
        model.mlmBias, model.mlmLogits, model.mlmAgg, model.mlmProbs,
        model.nspLogits, model.nspProbs, model.clsWeight, model.clsBias,
        model.sentimentLogits, model.sentimentProbs,
    ];
    let lin = (l: ILinearLayerLink) => [l.weight, l.bias, l.output].filter(a => a) as IBufferTex[];
    let ln = (l: ILayerNormLayerLink) => [l.normAgg, l.normWeight, l.normBias, l.output];
    let attn = (l: IAttentionLayerLink): IBufferTex[] => ([
        l.qWeight, l.kWeight, l.vWeight, l.qBias, l.kBias, l.vBias, l.q, l.k, l.v, l.attnMatrix,
        l.attnMatrixAgg, l.attnMatrixSoftmax, l.scaledVectors, ...lin(l.proj), l.residual]
        .filter(a => a) as IBufferTex[]);
    bufs.push(...ln(model.embedLN), ...lin(model.mlm.fc), model.mlm.act, ...ln(model.mlm.ln), ...lin(model.pool));
    for (let b of model.blocks) {
        bufs.push(...attn(b.selfAttn), ...ln(b.ln1), ...lin(b.ffn.fc), b.ffn.act, ...lin(b.ffn.proj), b.ffn.residual, ...ln(b.ln2));
    }
    return bufs;
}

export function uploadAllBert(model: IBertModelLink) {
    for (let buf of allBertBuffers(model)) {
        if (buf.localBuffer) {
            buf.rms = undefined;
            writeToBufferTex(model.gl, buf, buf.localBuffer);
        }
    }
}
