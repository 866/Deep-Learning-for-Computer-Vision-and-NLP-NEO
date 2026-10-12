import { AttnKind, BlKDepSpecial, createLayoutBuilder, IBlkDef } from "./GptModelLayout";
import { IVitModelLink, IVitShape } from "./VitModel";
import { DimStyle } from "./walkthrough/WalkthroughTools";

export type IVitModelLayout = ReturnType<typeof genVitModelLayout>;
export type IVitBlockLayout = IVitModelLayout['blocks'][0];

/*
Layout of the Vision Transformer (encoder only, pre-LN), adapted from llm-viz's GPT layout.

The residual pathway runs down the column at x = 0 (well, s.X0), with the side computations to its left:
  - Patches (P^2 x T, one column per token; column 0 is the [class] position and has no patch)
  - the patch projection E (+ bias) beside the patch embeddings; the [class] token to the right of them
  - the position embeddings E_pos to the right of z_0
  - L pre-LN layers: LN -> multi-head self-attention (with biases) -> + ; LN -> MLP (GELU) -> +
  - the final LN and the classification head, which only reads the [class] token (column 0)

Models with more than 12 layers wrap them into several columns, as llm-viz does for GPT-3.
*/
export function genVitModelLayout(shape: IVitShape, model: IVitModelLink | null = null) {
    let { B, T, C, nBlocks, dFF, nClasses, patchDim } = shape;

    let L = createLayoutBuilder(shape);
    let { s, cubes, cell, margin, mk, mkLabel, sc, leftX, rightX, createLn, createAttention, createFfn } = L;

    s.X0 = 0;
    s.dir = -1;
    s.y = 0;

    // ------ Patch + position embedding ------
    let patches = mk({
        t: 'i', cx: T, cz: B, cy: patchDim, y: s.y,
        xM: 0, zM: 0,
        access: { src: model?.patches, x: [0, 1, 0], y: [1, 0, T], scale: 1.0 },
        dimX: DimStyle.T, dimY: DimStyle.P,
        name: 'Patches',
    });

    s.y += patchDim * cell + margin;

    let patchWeight = mk({
        t: 'w', cx: patchDim, cz: 1, cy: C, y: s.y,
        xR: leftX, zM: 0,
        access: { src: model?.patchEmbed.weight, x: [1, 0, 0], y: [0, 1, 0], scale: sc(model?.patchEmbed.weight) },
        dimX: DimStyle.P, dimY: DimStyle.C,
        name: 'E',
    });

    let patchBias = mk({
        t: 'w', cx: 1, cz: 1, cy: C, y: s.y,
        xR: leftX - patchDim * cell - 2 * margin, zM: 0,
        access: { src: model?.patchEmbed.bias ?? undefined, x: [0, 0, 0], y: [0, 1, 0], scale: sc(model?.patchEmbed.bias) },
        dimX: DimStyle.None, dimY: DimStyle.C, small: true,
        name: 'Bias',
    });

    let clsToken = mk({
        t: 'w', cx: 1, cz: 1, cy: C, y: s.y,
        xL: rightX + margin, zM: 0,
        access: { src: model?.clsToken, x: [0, 0, 0], y: [0, 1, 0], scale: sc(model?.patchEmbed.output) },
        dimX: DimStyle.None, dimY: DimStyle.C,
        name: '[class]',
    });

    let tokens = mk({
        t: 'i', cx: T, cz: B, cy: C, y: s.y,
        xM: 0, zM: 0,
        access: { src: model?.patchEmbed.output, x: [0, 1, 0], y: [1, 0, T], scale: sc(model?.patchEmbed.output) },
        // column 0 is the [class] token rather than a projected patch
        deps: { dot: [[patchWeight, 'iy'], [patches, 'xi']], dotLen: patchDim, add: [[patchBias, '0y'], [clsToken, '0y']], special: BlKDepSpecial.PatchEmbed },
        dimX: DimStyle.T, dimY: DimStyle.C,
        name: 'Patch Embeddings',
    });

    s.y += C * cell + margin;

    let posEmbed = mk({
        t: 'w', cx: T, cz: 1, cy: C, y: s.y,
        xL: rightX, zM: 0,
        access: { src: model?.posEmbed, x: [0, 1, 0], y: [1, 0, 0], scale: sc(model?.posEmbed) },
        dimX: DimStyle.T, dimY: DimStyle.C,
        name: 'Position Embed',
    });

    let residual0 = mk({
        t: 'i', cx: T, cz: B, cy: C, y: s.y,
        xM: 0, zM: 0,
        access: { src: model?.embed, x: [0, 1, 0], y: [1, 0, T], scale: sc(model?.embed) },
        deps: { add: [[tokens, 'xy'], [posEmbed, 'xy']] },
        dimX: DimStyle.T, dimY: DimStyle.C,
        name: 'Embedding (z₀)',
    });

    let embedCubes = [patches, patchWeight, patchBias, clsToken, tokens, posEmbed, residual0];
    cubes.push(...embedCubes);
    let embedLabel = mkLabel(1.0, embedCubes);

    s.y += C * cell + margin;

    // ------ Encoder layers ------
    let blockHalfMargin = 2 * margin;
    s.y += blockHalfMargin;

    let blocksPerColumn = 12;
    let columnWidth = Math.max(C * 14, dFF + 3 * T + 2 * C) * cell + 2 * margin;
    let blockYTop = s.y;

    let blocks = [];
    let x = residual0;
    for (let i = 0; i < nBlocks; i++) {
        let newColumn = i > 0 && i % blocksPerColumn === 0;
        if (newColumn) {
            s.y = blockYTop;
            s.X0 += columnWidth;
        }
        let target = model?.blocks[i];
        s.y += blockHalfMargin;
        let ln1 = createLn(x, target?.ln1, true, newColumn ? s.y : undefined);
        let attn = createAttention(ln1.lnResid, ln1.lnResid, target?.attn, AttnKind.Self, { residSrc: x, bias: true });
        let ln2 = createLn(attn.attnResidual, target?.ln2, true);
        let ffn = createFfn(ln2.lnResid, target?.mlp, { residSrc: attn.attnResidual, gelu: true, prefix: 'MLP' });
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

    // ------ Final layer norm + classification head on the [class] token ------
    s.y += blockHalfMargin;
    let lnF = createLn(x, model?.lnF, true);
    s.y = lnF.lnResid.y + C * cell + margin;

    // the [class] column of the final layer norm sits at its left edge
    let clsX = lnF.lnResid.x;
    let headLeftX = clsX - s.X0; // in the builder's (unmirrored) frame

    let headWeight = mk({
        t: 'w', cx: C, cz: 1, cy: nClasses, y: s.y,
        xR: headLeftX - margin, zM: 0,
        access: { src: model?.head.weight, x: [1, 0, 0], y: [0, 1, 0], scale: sc(model?.head.weight) },
        dimX: DimStyle.C, dimY: DimStyle.n_classes,
        name: 'Head Weights',
    });

    let headBias = mk({
        t: 'w', cx: 1, cz: 1, cy: nClasses, y: s.y,
        xR: headLeftX - C * cell - 2 * margin, zM: 0,
        access: { src: model?.head.bias ?? undefined, x: [0, 0, 0], y: [0, 1, 0], scale: sc(model?.head.bias) },
        dimX: DimStyle.None, dimY: DimStyle.n_classes, small: true,
        name: 'Head Bias',
    });

    let logits = mk({
        t: 'i', cx: 1, cz: B, cy: nClasses, y: s.y,
        xL: headLeftX, zM: 0,
        access: { src: model?.logits, x: [0, 1, 0], y: [1, 0, 0], scale: sc(model?.logits, 0.4) },
        deps: { dot: [[headWeight, 'iy'], [lnF.lnResid, 'xi']], dotLen: C, add: [[headBias, '0y']] },
        dimX: DimStyle.None, dimY: DimStyle.n_classes,
        name: 'Logits',
    });

    s.y += nClasses * cell + margin;

    let logitsAgg2 = mk({
        t: 'a', cx: 1, cy: 1, cz: B, y: s.y,
        xL: headLeftX, zM: 0,
        access: { src: model?.softmaxAgg, x: [0, 0, 0], y: [0, 0, 0], channel: 'g', scale: sc(model?.logits, 0.4) },
        deps: { add: [[logits, 'xi']], special: BlKDepSpecial.SoftmaxAggMax },
        dimX: DimStyle.None, dimY: DimStyle.None,
        name: 'SM Agg',
    });

    let logitsAgg1 = mk({
        t: 'a', cx: 1, cy: 1, cz: B, y: s.y + cell,
        xL: headLeftX, zM: 0,
        access: { src: model?.softmaxAgg, x: [0, 0, 0], y: [0, 0, 0], channel: 'r' },
        deps: { add: [[logits, 'xi'], [logitsAgg2, 'x0']], special: BlKDepSpecial.SoftmaxAggExp },
        dimX: DimStyle.None, dimY: DimStyle.None,
        name: '',
    });

    s.y += 2 * cell + margin;

    let logitsSoftmax = mk({
        t: 'i', cx: 1, cy: nClasses, cz: B, y: s.y,
        xL: headLeftX, zM: 0,
        access: { src: model?.probs, x: [0, 1, 0], y: [1, 0, 0] },
        deps: { add: [[logits, 'xy'], [logitsAgg1, 'xi'], [logitsAgg2, 'xi']], special: BlKDepSpecial.Softmax },
        dimX: DimStyle.None, dimY: DimStyle.n_classes,
        name: 'Class Probabilities',
    });

    s.y += nClasses * cell + margin;

    let outputCubes = [headWeight, headBias, logits, logitsAgg1, logitsAgg2, logitsSoftmax];
    cubes.push(...outputCubes);
    let outputLabel = mkLabel(1.0, [...lnF.cubes, ...outputCubes]);

    for (let i = 0; i < cubes.length; i++) {
        cubes[i].idx = i;
    }

    let layerParams = 4 * C * C + 4 * C + 2 * 2 * C + (C * dFF + dFF + dFF * C + C);
    let weightCount = patchDim * C + C + C + T * C + nBlocks * layerParams + 2 * C + C * nClasses + nClasses;

    let height = Math.max(s.y, ...blocks.map(b => b.output.y + b.output.dy));

    return {
        kind: 'vit' as const,
        cubes,
        cell,
        margin,
        model,
        shape,
        height,
        midX: 0,
        dir: -1,
        weightCount,
        patches, patchWeight, patchBias, clsToken, tokens, posEmbed, residual0, embedLabel,
        blocks,
        lnF,
        headWeight, headBias, logits, logitsAgg1, logitsAgg2, logitsSoftmax, outputLabel,
        output: x,
        labels: [embedLabel, outputLabel, lnF.label, ...blocks.flatMap(b => b.labels)],
    };
}

/** For the patch-embedding block: the cell at token 0 is the [class] token; the others are projected patches. */
export function isPatchEmbedBlock(blk: IBlkDef) {
    return blk.deps?.special === BlKDepSpecial.PatchEmbed;
}
