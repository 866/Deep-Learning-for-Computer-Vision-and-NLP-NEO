import { AttnKind, BlKDepSpecial, createLayoutBuilder } from "./GptModelLayout";
import { IBertModelLink, IBertShape } from "./BertModel";
import { DimStyle } from "./walkthrough/WalkthroughTools";

export type IBertModelLayout = ReturnType<typeof genBertModelLayout>;
export type IBertBlockLayout = IBertModelLayout['blocks'][0];

/*
Layout of BERT (encoder only, pre-LN), built on the same builder the GPT and ViT layouts use.

The residual pathway runs down the column at x = 0 (s.X0), with side computations to its left:
  - the token-id row and, below it, the segment-id row (one cell per token)
  - the three embedding tables of Figure 2 — token (V×C) and segment (2×C) to the left, position
    (T×C) to the right of the sum they build
  - L pre-LN layers: LN -> bidirectional self-attention (with biases) -> + ; LN -> FFN (GELU) -> +
  - the MLM head, which runs over every position: transform C->C, GELU, then Eᵀ (C->V, tied to the
    token table), then a softmax per position
  - the sentence-pair head on column 0 ([CLS]): pool C->C + tanh, then the 2×C NSP head and the 2×C
    sentiment classifier

Bidirectional attention is the same QKᵀ -> softmax -> AV maths as the ViT's, with one difference
that matters visually: the score matrix has no lower-triangle mask, so every cell is live. The
causal mask is a flag on the dependency (IBlkDeps.lowerTri) that this layout simply never sets.
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
        dimX: DimStyle.T, dimY: DimStyle.n_segments,
        name: 'Segment Ids',
    });

    s.y += cell + margin;

    // ------ The three embedding tables of Figure 2 ------
    let tokEmbed = mk({
        t: 'w', cx: vocabSize, cz: 1, cy: C, y: s.y,
        xR: leftX, zM: 0,
        access: { src: model?.embedWeight, x: [0, 1, 0], y: [1, 0, 0], scale: sc(model?.embedWeight) },
        dimX: DimStyle.n_vocab, dimY: DimStyle.C,
        name: 'Token Embed',
        embedPart: 'token',
    });

    // the segment table is two rows; it sits left of the token table rather than below it so the
    // three of them read as one row of lookups into one sum
    let segEmbed = mk({
        t: 'w', cx: nSegments, cz: 1, cy: C, y: s.y,
        xR: leftX - vocabSize * cell - 2 * margin, zM: 0,
        access: { src: model?.segEmbed, x: [0, 1, 0], y: [1, 0, 0], scale: sc(model?.segEmbed) },
        dimX: DimStyle.n_segments, dimY: DimStyle.C,
        name: 'Segment Embed',
        embedPart: 'segment',
    });

    let posEmbed = mk({
        t: 'w', cx: T, cz: 1, cy: C, y: s.y,
        xL: rightX, zM: 0,
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

    // ------ The encoder stack (pre-LN, exactly as the ViT lays its layers out) ------
    let blockHalfMargin = 2 * margin;
    s.y += blockHalfMargin;

    let blocks = [];
    let x = residual0;
    for (let i = 0; i < nBlocks; i++) {
        let target = model?.blocks[i];
        s.y += blockHalfMargin;

        let ln1 = createLn(x, target?.ln1, true, i === 0 ? s.y : undefined);
        let attn = createAttention(ln1.lnResid, ln1.lnResid, target?.selfAttn, AttnKind.Self, { residSrc: x, bias: true });
        let ln2 = createLn(attn.attnResidual, target?.ln2, true);
        let ffn = createFfn(ln2.lnResid, target?.ffn, { residSrc: attn.attnResidual, gelu: true, prefix: 'FFN' });

        let blockCubes = [...ln1.cubes, ...attn.cubes, ...ln2.cubes, ...ffn.cubes];
        let blockLabel = mkLabel(1.0, blockCubes);
        blocks.push({
            idx: i, ln1, attn, ln2, ffn, input: x, output: ffn.mlpResidual, x0: s.X0,
            cubes: blockCubes, blockLabel,
            labels: [blockLabel, ln1.label, ln2.label, ffn.ffnLabel, ...attn.labels],
        });
        x = ffn.mlpResidual;
        s.y += blockHalfMargin;
    }

    // ------ The encoder's own final norm, then the two heads ------
    s.y += blockHalfMargin;
    let embedLN = createLn(residual0, model?.embedLN, false);

    // The MLM head reads the *normed* stack output. Note this is not a separate "final LN": BERT
    // keeps the encoder's per-layer norms only, and z_L itself is what the head consumes.
    s.y = x.y + x.dy + blockHalfMargin;

    let mlmFcWeight = mk({
        t: 'w', cx: C, cz: 1, cy: C, y: s.y,
        xR: leftX, zM: 0,
        access: { src: model?.mlm.fc.weight, x: [1, 0, 0], y: [0, 1, 0], scale: sc(model?.mlm.fc.weight) },
        dimX: DimStyle.C, dimY: DimStyle.C,
        name: 'MLM W\u2081',
    });

    let mlmFcBias = mk({
        t: 'w', cx: 1, cz: 1, cy: C, y: s.y,
        xR: leftX - C * cell - 2 * margin, zM: 0,
        access: { src: model?.mlm.fc.bias ?? undefined, x: [0, 0, 0], y: [0, 1, 0], scale: sc(model?.mlm.fc.bias) },
        dimX: DimStyle.None, dimY: DimStyle.C, small: true,
        name: 'MLM b\u2081',
    });

    let mlmDense = mk({
        t: 'i', cx: T, cz: B, cy: C, y: s.y,
        xM: 0, zM: 0,
        access: { src: model?.mlm.fc.output, x: [0, 1, 0], y: [1, 0, T], scale: sc(model?.mlm.fc.output) },
        deps: { dot: [[mlmFcWeight, 'iy'], [x, 'yi']], dotLen: C, add: [[mlmFcBias, '0y']] },
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
    let mlmLn = createLn(mlmAct, model?.mlm.ln, false);
    let mlmNormed = mlmLn.lnResid;
    s.y = mlmNormed.y + C * cell + margin;

    // the output layer is the token embedding table, transposed \u2014 the weights the rows were looked
    // up from are the same numbers scored against, which is the whole point of the tied weights
    let mlmOutWeight = mk({
        t: 'w', cx: C, cz: 1, cy: vocabSize, y: s.y,
        xR: leftX, zM: 0,
        access: { src: model?.embedWeight, x: [1, 0, 0], y: [0, 1, 0], scale: sc(model?.embedWeight) },
        dimX: DimStyle.C, dimY: DimStyle.n_vocab,
        name: 'E\u1D40 (tied)',
    });

    let mlmOutBias = mk({
        t: 'w', cx: 1, cz: 1, cy: vocabSize, y: s.y,
        xR: leftX - C * cell - 2 * margin, zM: 0,
        access: { src: model?.mlmBias, x: [0, 0, 0], y: [0, 1, 0], scale: sc(model?.mlmBias) },
        dimX: DimStyle.None, dimY: DimStyle.n_vocab, small: true,
        name: 'MLM bias (untied)',
    });

    let mlmLogits = mk({
        t: 'i', cx: T, cz: B, cy: vocabSize, y: s.y,
        xM: 0, zM: 0,
        access: { src: model?.mlmLogits, x: [0, 1, 0], y: [1, 0, T], scale: sc(model?.mlmLogits, 0.4) },
        deps: { dot: [[mlmOutWeight, 'iy'], [mlmNormed, 'yi']], dotLen: C, add: [[mlmOutBias, '0y']] },
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
        access: { src: model?.mlmAgg, x: [0, 1, 0], y: [1, 0, T], channel: 'r', scale: sc(model?.mlmLogits, 0.4) },
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

    let mlmCubes = [mlmFcWeight, mlmFcBias, mlmDense, mlmAct, mlmOutWeight, mlmOutBias, mlmLogits, mlmAgg1, mlmAgg2, mlmProbs];
    cubes.push(...mlmCubes);
    let mlmLabel = mkLabel(1.0, mlmCubes);

    // ------ Sentence-pair head on column 0: pool -> NSP, and the classifier on the raw [CLS] ------
    // Both read the [CLS] column of the stack output, which is the left edge of the last residual
    // block; the builder's frame is un-mirrored, hence the -X0.
    let headLeftX = x.x - s.X0;

    let poolWeight = mk({
        t: 'w', cx: C, cz: 1, cy: C, y: s.y,
        xR: headLeftX - margin, zM: 0,
        access: { src: model?.pool.weight, x: [1, 0, 0], y: [0, 1, 0], scale: sc(model?.pool.weight) },
        dimX: DimStyle.C, dimY: DimStyle.C,
        name: 'Pool W',
    });

    let poolBias = mk({
        t: 'w', cx: 1, cz: 1, cy: C, y: s.y,
        xR: headLeftX - C * cell - 2 * margin, zM: 0,
        access: { src: model?.pool.bias ?? undefined, x: [0, 0, 0], y: [0, 1, 0], scale: sc(model?.pool.bias) },
        dimX: DimStyle.None, dimY: DimStyle.C, small: true,
        name: 'Pool b',
    });

    // the pooler's tanh is applied in the same step as its matmul (the `tanhFragSrc` the ViT head
    // also uses), so the block shows one step, which is what actually happens
    let poolOut = mk({
        t: 'i', cx: 1, cy: C, cz: B, y: s.y,
        xL: headLeftX, zM: 0,
        access: { src: model?.pool.output, x: [0, 0, 0], y: [1, 0, 0], scale: sc(model?.pool.output) },
        deps: { dot: [[poolWeight, 'iy'], [x, 'xi']], dotLen: C, add: [[poolBias, '0y']] },
        dimX: DimStyle.None, dimY: DimStyle.C,
        name: 'Pooled [CLS] (tanh)',
    });

    s.y += C * cell + margin;

    let nspWeight = mk({
        t: 'w', cx: C, cz: 1, cy: 2, y: s.y,
        xR: headLeftX - margin, zM: 0,
        access: { src: model?.nsp.weight, x: [1, 0, 0], y: [0, 1, 0], scale: sc(model?.nsp.weight) },
        dimX: DimStyle.C, dimY: DimStyle.n_segments,
        name: 'NSP W',
    });

    let nspBias = mk({
        t: 'w', cx: 1, cz: 1, cy: 2, y: s.y,
        xR: headLeftX - C * cell - 2 * margin, zM: 0,
        access: { src: model?.nsp.bias ?? undefined, x: [0, 0, 0], y: [0, 1, 0], scale: sc(model?.nsp.bias) },
        dimX: DimStyle.None, dimY: DimStyle.n_segments, small: true,
        name: 'NSP b',
    });

    let nspLogits = mk({
        t: 'i', cx: 1, cy: 2, cz: B, y: s.y,
        xL: headLeftX, zM: 0,
        access: { src: model?.nspLogits, x: [0, 0, 0], y: [1, 0, 0], scale: sc(model?.nspLogits, 0.4) },
        deps: { dot: [[nspWeight, 'iy'], [poolOut, 'yi']], dotLen: C, add: [[nspBias, '0y']] },
        dimX: DimStyle.None, dimY: DimStyle.n_segments,
        name: 'NSP Logits',
    });

    s.y += 2 * cell + margin;

    let nspAgg2 = mk({
        t: 'a', cx: 1, cy: 1, cz: B, y: s.y,
        xL: headLeftX, zM: 0,
        access: { src: model?.nspAgg, x: [0, 0, 0], y: [0, 0, 0], channel: 'g', scale: sc(model?.nspLogits, 0.4) },
        deps: { add: [[nspLogits, 'xi']], special: BlKDepSpecial.SoftmaxAggMax },
        dimX: DimStyle.None, dimY: DimStyle.None, small: true,
        name: 'SM Agg',
    });

    let nspAgg1 = mk({
        t: 'a', cx: 1, cy: 1, cz: B, y: s.y + cell,
        xL: headLeftX, zM: 0,
        access: { src: model?.nspAgg, x: [0, 0, 0], y: [0, 0, 0], channel: 'r', scale: sc(model?.nspLogits, 0.4) },
        deps: { add: [[nspLogits, 'xi'], [nspAgg2, 'x0']], special: BlKDepSpecial.SoftmaxAggExp },
        dimX: DimStyle.None, dimY: DimStyle.None, small: true,
        name: '',
    });

    s.y += 2 * cell + margin;

    let nspProbs = mk({
        t: 'i', cx: 1, cy: 2, cz: B, y: s.y,
        xL: headLeftX, zM: 0,
        access: { src: model?.nspProbs, x: [0, 0, 0], y: [1, 0, 0] },
        deps: { add: [[nspLogits, 'xy'], [nspAgg1, 'xi'], [nspAgg2, 'xi']], special: BlKDepSpecial.Softmax },
        dimX: DimStyle.None, dimY: DimStyle.n_segments,
        name: 'NSP Probabilities',
    });

    s.y += 2 * cell + 3 * margin;

    // the classifier hangs off the *raw* [CLS], not the pooled vector (paper \u00a74.1: "we use BERT_1
    // with a classifier on top"); the pooler feeds NSP and nothing else
    let sentimentWeight = mk({
        t: 'w', cx: C, cz: 1, cy: 2, y: s.y,
        xR: headLeftX - margin, zM: 0,
        access: { src: model?.clsWeight, x: [1, 0, 0], y: [0, 1, 0], scale: sc(model?.clsWeight) },
        dimX: DimStyle.C, dimY: DimStyle.n_classes,
        name: 'Sentiment W',
    });

    let sentimentBias = mk({
        t: 'w', cx: 1, cz: 1, cy: 2, y: s.y,
        xR: headLeftX - C * cell - 2 * margin, zM: 0,
        access: { src: model?.clsBias ?? undefined, x: [0, 0, 0], y: [0, 1, 0], scale: sc(model?.clsBias) },
        dimX: DimStyle.None, dimY: DimStyle.n_classes, small: true,
        name: 'Sentiment b',
    });

    let sentimentLogits = mk({
        t: 'i', cx: 1, cy: 2, cz: B, y: s.y,
        xL: headLeftX, zM: 0,
        access: { src: model?.sentimentLogits, x: [0, 0, 0], y: [1, 0, 0], scale: sc(model?.sentimentLogits, 0.4) },
        deps: { dot: [[sentimentWeight, 'iy'], [x, 'xi']], dotLen: C, add: [[sentimentBias, '0y']] },
        dimX: DimStyle.None, dimY: DimStyle.n_classes,
        name: 'Sentiment Logits',
    });

    s.y += 2 * cell + margin;

    let sentimentAgg2 = mk({
        t: 'a', cx: 1, cy: 1, cz: B, y: s.y,
        xL: headLeftX, zM: 0,
        access: { src: model?.sentimentAgg, x: [0, 0, 0], y: [0, 0, 0], channel: 'g', scale: sc(model?.sentimentLogits, 0.4) },
        deps: { add: [[sentimentLogits, 'xi']], special: BlKDepSpecial.SoftmaxAggMax },
        dimX: DimStyle.None, dimY: DimStyle.None, small: true,
        name: 'SM Agg',
    });

    let sentimentAgg1 = mk({
        t: 'a', cx: 1, cy: 1, cz: B, y: s.y + cell,
        xL: headLeftX, zM: 0,
        access: { src: model?.sentimentAgg, x: [0, 0, 0], y: [0, 0, 0], channel: 'r', scale: sc(model?.sentimentAgg) },
        deps: { add: [[sentimentLogits, 'xi'], [sentimentAgg2, 'x0']], special: BlKDepSpecial.SoftmaxAggExp },
        dimX: DimStyle.None, dimY: DimStyle.None, small: true,
        name: '',
    });

    s.y += 2 * cell + margin;

    let sentimentProbs = mk({
        t: 'i', cx: 1, cy: 2, cz: B, y: s.y,
        xL: headLeftX, zM: 0,
        access: { src: model?.sentimentProbs, x: [0, 0, 0], y: [1, 0, 0] },
        deps: { add: [[sentimentLogits, 'xy'], [sentimentAgg1, 'xi'], [sentimentAgg2, 'xi']], special: BlKDepSpecial.Softmax },
        dimX: DimStyle.None, dimY: DimStyle.n_classes,
        name: 'Sentiment',
    });

    s.y += 2 * cell + margin;

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

    // params: the three tables, the stack, the encoder's two LNs, and the three heads
    let tableParams = vocabSize * C + nSegments * C + T * C;
    let lnParams = 2 * C + 2 * C;
    let blockParams = 4 * C * C + 4 * C + 4 * C + 4 * C + (C * dFF + dFF + dFF * C + C) + lnParams;
    let headParams = (C * C + C) + (C * vocabSize + vocabSize) + (C * C + C) + (2 * C + 2) + (2 * C + 2);
    let weightCount = tableParams + nBlocks * blockParams + headParams;

    let height = Math.max(s.y, ...blocks.map(b => b.output.y + b.output.dy));

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
