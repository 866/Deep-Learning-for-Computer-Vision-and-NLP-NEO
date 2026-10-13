import { AttnKind, BlKDepSpecial, createLayoutBuilder, IBlkDef } from "./GptModelLayout";
import { IBufferTex } from "@/src/utils/renderPhases";
import { IBertModelLink, IBertShape } from "./BertModel";
import { DimStyle } from "./walkthrough/WalkthroughTools";

export type IBertModelLayout = ReturnType<typeof genBertModelLayout>;
export type IBertBlockLayout = IBertModelLayout['blocks'][0];

/*
Layout of BERT (encoder only, post-LN), built on the same builder the Transformer and ViT layouts use.

BERT keeps the original Transformer's "Add & Norm" (post-LN) encoder, so this lays its stack out exactly
like the tiny Transformer's encoder: the residual pathway runs down the column at x = 0, each layer
norm sits *in* that column and its output becomes the new residual, and the side computations go to
the left.
  - the token-id row and, below it, the segment-id row (one cell per token)
  - the three embedding tables of Figure 2: token (V×C) on the left, segment (2×C) and position
    (T×C) on the right, all summed into z₀, then the embedding LayerNorm
  - L layers: bidirectional self-attention (with biases) -> Add & Norm -> FFN (GELU) -> Add & Norm
  - the MLM head under the stack, over every position: dense C->C, GELU, LayerNorm, then Eᵀ (C->V,
    tied to the token table), then a softmax per position
  - the two [CLS] heads to the right of the stack output (they only read column 0): the tanh pooler
    feeding the NSP head, and the K×H sentiment classifier on the raw [CLS] vector (§4.1)

Bidirectional attention is the same QKᵀ -> softmax -> AV maths as the encoder's, with the one difference
that matters visually: the score matrix has no lower-triangle mask, so every cell is live. The causal
mask is a flag on the dependency (IBlkDeps.lowerTri) that this layout simply never sets.
*/
export function genBertModelLayout(shape: IBertShape, model: IBertModelLink | null = null) {
    let { B, T, C, nBlocks, dFF, vocabSize, nSegments } = shape;

    let L = createLayoutBuilder(shape);
    let { s, cubes, cell, margin, mk, mkLabel, sc, leftX, rightX, createLn, createAttention, createFfn } = L;

    s.X0 = 0;
    s.dir = -1;
    s.y = 0;

    // ------ Token & segment id rows ------
    let tokens = mk({
        t: 'i', cx: T, cz: B, cy: 1, y: s.y,
        xM: 0, zM: 0,
        access: { src: model?.tokens, x: [0, 1, 0], y: [1, 0, T], scale: 1 / vocabSize },
        dimX: DimStyle.T, dimY: DimStyle.None,
        name: 'Tokens',
    });

    s.y += cell + margin;

    let segIds = mk({
        t: 'i', cx: T, cz: B, cy: 1, y: s.y,
        xM: 0, zM: 0,
        access: { src: model?.segIds, x: [0, 1, 0], y: [1, 0, T], scale: 0.5 },
        dimX: DimStyle.T, dimY: DimStyle.None,
        name: 'Segment Ids',
    });

    s.y += cell + margin;

    // ------ The three embedding tables of Figure 2 ------
    // The token table goes where the Transformer's does (left); the segment and position tables go on
    // the right, the small segment table next to the column so its arrow is the short one.
    let tokEmbed = mk({
        t: 'w', cx: vocabSize, cz: 1, cy: C, y: s.y,
        xR: leftX, zM: 0,
        access: { src: model?.embedWeight, x: [0, 1, 0], y: [1, 0, 0], scale: sc(model?.embedWeight) },
        dimX: DimStyle.n_vocab, dimY: DimStyle.C,
        name: 'Token Embed',
        embedPart: 'token',
    });

    let segEmbed = mk({
        t: 'w', cx: nSegments, cz: 1, cy: C, y: s.y,
        xL: rightX, zM: 0,
        access: { src: model?.segEmbed, x: [0, 1, 0], y: [1, 0, 0], scale: sc(model?.segEmbed) },
        dimX: DimStyle.n_segments, dimY: DimStyle.C,
        name: 'Segment Embed',
        embedPart: 'segment',
    });

    let posEmbed = mk({
        t: 'w', cx: T, cz: 1, cy: C, y: s.y,
        xL: rightX + nSegments * cell + 2 * margin, zM: 0,
        access: { src: model?.posEmbed, x: [0, 1, 0], y: [1, 0, 0], scale: sc(model?.posEmbed) },
        dimX: DimStyle.T, dimY: DimStyle.C,
        name: 'Position Embed',
        embedPart: 'position',
    });

    let residual0 = mk({
        t: 'i', cx: T, cz: B, cy: C, y: s.y,
        xM: 0, zM: 0,
        access: { src: model?.embed, x: [0, 1, 0], y: [1, 0, T], scale: sc(model?.embed) },
        // 'iy' = take the row from the looked-up index (tokens / segment ids), the channel from y;
        // 'xy' = the position table is indexed by the column itself; 'x0' is the index row itself
        deps: { add: [[tokEmbed, 'iy'], [segEmbed, 'iy'], [posEmbed, 'xy'], [tokens, 'x0']], special: BlKDepSpecial.InputEmbed },
        dimX: DimStyle.T, dimY: DimStyle.C,
        name: 'Embedding (z₀)',
    });

    let embedCubes = [tokens, segIds, tokEmbed, segEmbed, posEmbed, residual0];
    cubes.push(...embedCubes);
    let embedLabel = mkLabel(1.0, embedCubes);

    s.y += C * cell + margin;

    // the norm BERT applies to the embedding sum before layer 1: in the column, like every post-LN norm
    let embedLN = createLn(residual0, model?.embedLN);

    // ------ The encoder stack (post-LN, exactly as the Transformer's encoder) ------
    let blockHalfMargin = 2 * margin;
    s.y += blockHalfMargin;

    let blocks = [];
    let x = embedLN.lnResid;
    for (let i = 0; i < nBlocks; i++) {
        let target = model?.blocks[i];
        s.y += blockHalfMargin;

        let attn = createAttention(x, x, target?.selfAttn, AttnKind.Self, { bias: true });
        let ln1 = createLn(attn.attnResidual, target?.ln1);
        let ffn = createFfn(ln1.lnResid, target?.ffn, { gelu: true, prefix: 'FFN' });
        let ln2 = createLn(ffn.mlpResidual, target?.ln2);

        let blockCubes = [...attn.cubes, ...ln1.cubes, ...ffn.cubes, ...ln2.cubes];
        let blockLabel = mkLabel(1.0, blockCubes);
        blocks.push({
            idx: i, attn, ln1, ffn, ln2, input: x, output: ln2.lnResid, x0: s.X0,
            cubes: blockCubes, blockLabel,
            labels: [blockLabel, ln1.label, ln2.label, ffn.ffnLabel, ...attn.labels],
        });
        x = ln2.lnResid;
        s.y += blockHalfMargin;
    }

    // ------ The MLM head, under the stack, over every position ------
    s.y += blockHalfMargin;

    let mlmFcWeight = mk({
        t: 'w', cx: C, cz: 1, cy: C, y: s.y,
        xR: leftX, zM: 0,
        access: { src: model?.mlm.fc.weight, x: [1, 0, 0], y: [0, 1, 0], scale: sc(model?.mlm.fc.weight) },
        dimX: DimStyle.C, dimY: DimStyle.C,
        name: 'MLM Dense Weights',
    });

    let mlmFcBias = mk({
        t: 'w', cx: 1, cz: 1, cy: C, y: s.y,
        xR: leftX - C * cell - margin, zM: 0,
        access: { src: model?.mlm.fc.bias ?? undefined, x: [0, 0, 0], y: [0, 1, 0], scale: sc(model?.mlm.fc.bias) },
        dimX: DimStyle.None, dimY: DimStyle.C, small: true,
        name: 'MLM Dense Bias',
    });

    let mlmDense = mk({
        t: 'i', cx: T, cz: B, cy: C, y: s.y,
        xM: 0, zM: 0,
        access: { src: model?.mlm.fc.output, x: [0, 1, 0], y: [1, 0, T], scale: sc(model?.mlm.fc.output) },
        deps: { dot: [[mlmFcWeight, 'iy'], [x, 'xi']], dotLen: C, add: [[mlmFcBias, '0y']] },
        dimX: DimStyle.T, dimY: DimStyle.C,
        name: 'MLM Dense',
    });

    s.y += C * cell + margin;

    let mlmAct = mk({
        t: 'i', cx: T, cz: B, cy: C, y: s.y,
        xM: 0, zM: 0,
        access: { src: model?.mlm.act, x: [0, 1, 0], y: [1, 0, T], scale: sc(model?.mlm.act) },
        deps: { add: [[mlmDense, 'xy']], special: BlKDepSpecial.Gelu },
        dimX: DimStyle.T, dimY: DimStyle.C,
        name: 'MLM GELU',
    });

    s.y += C * cell + margin;

    // the transform is Linear -> GELU -> LayerNorm; the norm is the last thing before the vocabulary
    // projection, and it is a real LayerNorm with its own learned scale and shift
    // cube order is computation order (the walkthrough's processUpTo animates blocks in that order), and
    // createLn pushes its cubes straight away, so the dense + GELU blocks go in first
    let mlmTransformCubes = [mlmFcWeight, mlmFcBias, mlmDense, mlmAct];
    cubes.push(...mlmTransformCubes);
    let mlmLn = createLn(mlmAct, model?.mlm.ln);
    let mlmNormed = mlmLn.lnResid;

    // the output layer is the token embedding table, transposed \u2014 the weights the rows were looked
    // up from are the same numbers scored against, which is the whole point of the tied weights
    let mlmOutWeight = mk({
        t: 'w', cx: C, cz: 1, cy: vocabSize, y: s.y,
        xR: leftX, zM: 0,
        access: { src: model?.embedWeight, x: [1, 0, 0], y: [0, 1, 0], scale: sc(model?.embedWeight) },
        dimX: DimStyle.C, dimY: DimStyle.n_vocab,
        name: 'Token Embed\u1D40 (tied)',
    });

    let mlmOutBias = mk({
        t: 'w', cx: 1, cz: 1, cy: vocabSize, y: s.y,
        xR: leftX - C * cell - margin, zM: 0,
        access: { src: model?.mlmBias, x: [0, 0, 0], y: [0, 1, 0], scale: sc(model?.mlmBias) },
        dimX: DimStyle.None, dimY: DimStyle.n_vocab, small: true,
        name: 'MLM Output Bias',
    });

    let mlmLogits = mk({
        t: 'i', cx: T, cz: B, cy: vocabSize, y: s.y,
        xM: 0, zM: 0,
        access: { src: model?.mlmLogits, x: [0, 1, 0], y: [1, 0, T], scale: sc(model?.mlmLogits, 0.4) },
        deps: { dot: [[mlmOutWeight, 'iy'], [mlmNormed, 'xi']], dotLen: C, add: [[mlmOutBias, '0y']] },
        dimX: DimStyle.T, dimY: DimStyle.n_vocab,
        name: 'MLM Logits',
    });

    s.y += vocabSize * cell + margin;

    let mlmAgg2 = mk({
        t: 'a', cx: T, cy: 1, cz: B, y: s.y,
        xM: 0, zM: 0,
        access: { src: model?.mlmAgg, x: [0, 1, 0], y: [1, 0, T], channel: 'g', scale: sc(model?.mlmLogits, 0.4) },
        deps: { add: [[mlmLogits, 'xi']], special: BlKDepSpecial.SoftmaxAggMax },
        dimX: DimStyle.T, dimY: DimStyle.None, small: true,
        name: 'SM Agg',
    });

    let mlmAgg1 = mk({
        t: 'a', cx: T, cy: 1, cz: B, y: s.y + cell,
        xM: 0, zM: 0,
        access: { src: model?.mlmAgg, x: [0, 1, 0], y: [1, 0, T], channel: 'r' },
        deps: { add: [[mlmLogits, 'xi'], [mlmAgg2, 'x0']], special: BlKDepSpecial.SoftmaxAggExp },
        dimX: DimStyle.T, dimY: DimStyle.None, small: true,
        name: '',
    });

    s.y += 2 * cell + margin;

    let mlmProbs = mk({
        t: 'i', cx: T, cy: vocabSize, cz: B, y: s.y,
        xM: 0, zM: 0,
        access: { src: model?.mlmProbs, x: [0, 1, 0], y: [1, 0, T] },
        deps: { add: [[mlmLogits, 'xy'], [mlmAgg1, 'xi'], [mlmAgg2, 'xi']], special: BlKDepSpecial.Softmax },
        dimX: DimStyle.T, dimY: DimStyle.n_vocab,
        name: 'MLM Probabilities',
    });

    s.y += vocabSize * cell + margin;

    let mlmOutCubes = [mlmOutWeight, mlmOutBias, mlmLogits, mlmAgg1, mlmAgg2, mlmProbs];
    cubes.push(...mlmOutCubes);
    let mlmCubes = [...mlmTransformCubes, ...mlmOutCubes];
    let mlmLabel = mkLabel(1.0, [...mlmCubes, ...mlmLn.cubes]);

    // ------ The two [CLS] heads, to the right of the stack output ------
    // Both read only column 0 of the last layer's output, so they sit beside it rather than under the
    // MLM head (which is n_vocab tall), on the side of the column that is otherwise empty. The builder
    // mirrors anything placed while s.dir = +1, so these use the same "data next to the column, weights
    // further out" positions as every other side computation, just on the right.
    s.dir = 1;
    let headY = x.y;
    let dataR = leftX - margin;                // inner edge of the head's data blocks: leaves room for a bus arrow
    let weightR = dataR - cell - margin;       // inner edge of the head weights
    let biasR = weightR - C * cell - margin;   // inner edge of the head biases

    // one Linear(C -> nOut) on a single [CLS] column, plus (optionally) its softmax
    function clsLinear(y: number, nOut: number, dimY: DimStyle, prefix: string,
            weight: IBufferTex | undefined, bias: IBufferTex | undefined, out: IBufferTex | undefined, src: IBlkDef, srcDep: string, name: string) {
        let w = mk({
            t: 'w', cx: C, cz: 1, cy: nOut, y,
            xR: weightR, zM: 0,
            access: { src: weight, x: [1, 0, 0], y: [0, 1, 0], scale: sc(weight) },
            dimX: DimStyle.C, dimY,
            name: `${prefix} Weights`,
        });
        let b = mk({
            t: 'w', cx: 1, cz: 1, cy: nOut, y,
            xR: biasR, zM: 0,
            access: { src: bias, x: [0, 0, 0], y: [0, 1, 0], scale: sc(bias) },
            dimX: DimStyle.None, dimY, small: true,
            name: `${prefix} Bias`,
        });
        let o = mk({
            t: 'i', cx: 1, cy: nOut, cz: B, y,
            xR: dataR, zM: 0,
            access: { src: out, x: [0, 0, 0], y: [1, 0, 0], scale: sc(out, nOut > 2 ? 0.5 : 0.4) },
            deps: { dot: [[w, 'iy'], [src, srcDep]], dotLen: C, add: [[b, '0y']] },
            dimX: DimStyle.None, dimY,
            name,
        });
        return { w, b, o };
    }

    function clsSoftmax(y: number, nOut: number, dimY: DimStyle, logits: IBlkDef, agg: IBufferTex | undefined, probs: IBufferTex | undefined, name: string) {
        let agg2 = mk({
            t: 'a', cx: 1, cy: 1, cz: B, y,
            xR: dataR, zM: 0,
            access: { src: agg, x: [0, 0, 0], y: [0, 0, 0], channel: 'g', scale: logits.access?.scale },
            deps: { add: [[logits, 'xi']], special: BlKDepSpecial.SoftmaxAggMax },
            dimX: DimStyle.None, dimY: DimStyle.None, small: true,
            name: 'SM Agg',
        });
        let agg1 = mk({
            t: 'a', cx: 1, cy: 1, cz: B, y: y + cell,
            xR: dataR, zM: 0,
            access: { src: agg, x: [0, 0, 0], y: [0, 0, 0], channel: 'r' },
            deps: { add: [[logits, 'xi'], [agg2, 'x0']], special: BlKDepSpecial.SoftmaxAggExp },
            dimX: DimStyle.None, dimY: DimStyle.None, small: true,
            name: '',
        });
        let p = mk({
            t: 'i', cx: 1, cy: nOut, cz: B, y: y + 2 * cell + margin,
            xR: dataR, zM: 0,
            access: { src: probs, x: [0, 0, 0], y: [1, 0, 0] },
            deps: { add: [[logits, 'xy'], [agg1, 'xi'], [agg2, 'xi']], special: BlKDepSpecial.Softmax },
            dimX: DimStyle.None, dimY,
            name,
        });
        return { agg1, agg2, p };
    }

    // the pooler's tanh is applied in the same step as its matmul, so the block shows one step,
    // which is what actually happens; it is level with the stack output, so its arrow is a straight split
    let pool = clsLinear(headY, C, DimStyle.C, 'Pooler', model?.pool.weight, model?.pool.bias ?? undefined, model?.pool.output, x, 'xi', 'Pooled [CLS] (tanh)');
    let poolWeight = pool.w, poolBias = pool.b, poolOut = pool.o;

    s.y = headY + C * cell + margin;
    let nsp = clsLinear(s.y, 2, DimStyle.n_classes, 'NSP', model?.nsp.weight, model?.nsp.bias ?? undefined, model?.nspLogits, poolOut, 'xi', 'NSP Logits');
    let nspWeight = nsp.w, nspBias = nsp.b, nspLogits = nsp.o;
    s.y += 2 * cell + margin;
    let nspSm = clsSoftmax(s.y, 2, DimStyle.n_classes, nspLogits, model?.nspAgg, model?.nspProbs, 'NSP Probabilities');
    let nspAgg1 = nspSm.agg1, nspAgg2 = nspSm.agg2, nspProbs = nspSm.p;
    s.y = nspProbs.y + 2 * cell + 4 * margin;

    // the classifier hangs off the *raw* [CLS], not the pooled vector (paper \u00a74.1: "we use BERT_1
    // with a classifier on top"); the pooler feeds NSP and nothing else
    let sent = clsLinear(s.y, 2, DimStyle.n_classes, 'Sentiment', model?.clsWeight, model?.clsBias ?? undefined, model?.sentimentLogits, x, 'xi', 'Sentiment Logits');
    let sentimentWeight = sent.w, sentimentBias = sent.b, sentimentLogits = sent.o;
    s.y += 2 * cell + margin;
    let sentSm = clsSoftmax(s.y, 2, DimStyle.n_classes, sentimentLogits, model?.sentimentAgg, model?.sentimentProbs, 'Sentiment Probabilities');
    let sentimentAgg1 = sentSm.agg1, sentimentAgg2 = sentSm.agg2, sentimentProbs = sentSm.p;

    s.dir = -1;

    let outputCubes = [
        poolWeight, poolBias, poolOut,
        nspWeight, nspBias, nspLogits, nspAgg1, nspAgg2, nspProbs,
        sentimentWeight, sentimentBias, sentimentLogits, sentimentAgg1, sentimentAgg2, sentimentProbs,
    ];
    cubes.push(...outputCubes);
    let outputLabel = mkLabel(1.0, outputCubes);

    for (let i = 0; i < cubes.length; i++) {
        cubes[i].idx = i;
    }

    // params: the three tables and the embedding norm, the stack, the MLM head (whose output weights are
    // the token table again, so only its bias is new), the pooler + NSP head, and the classifier
    let lnParams = 2 * C;
    let tableParams = vocabSize * C + nSegments * C + T * C + lnParams;
    let blockParams = (4 * C * C + 4 * C) + lnParams + (C * dFF + dFF + dFF * C + C) + lnParams;
    let mlmParams = (C * C + C) + lnParams + vocabSize;
    let clsParams = (C * C + C) + (2 * C + 2) + (2 * C + 2);
    let weightCount = tableParams + nBlocks * blockParams + mlmParams + clsParams;

    let height = Math.max(...cubes.map(c => c.y + c.dy));

    return {
        kind: 'bert' as const,
        cubes,
        cell,
        margin,
        model,
        shape,
        height,
        midX: 0,
        dir: -1,
        weightCount,
        tokens, segIds, tokEmbed, segEmbed, posEmbed, residual0, embedLabel,
        embedLN,
        blocks,
        mlmFcWeight, mlmFcBias, mlmDense, mlmAct, mlmLn, mlmOutWeight, mlmOutBias, mlmLogits, mlmAgg1, mlmAgg2, mlmProbs, mlmLabel,
        poolWeight, poolBias, poolOut,
        nspWeight, nspBias, nspLogits, nspAgg1, nspAgg2, nspProbs,
        sentimentWeight, sentimentBias, sentimentLogits, sentimentAgg1, sentimentAgg2, sentimentProbs,
        clsWeight: model?.clsWeight, clsBias: model?.clsBias,
        outputLabel,
        output: x,
        labels: [embedLabel, mlmLabel, outputLabel, embedLN.label, mlmLn.label, ...blocks.flatMap(b => b.labels)],
    };
}
