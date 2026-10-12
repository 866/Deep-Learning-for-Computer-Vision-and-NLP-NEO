import { TextAlignHoriz, TextAlignVert } from "../Annotations";
import { IAttentionLayout, IBlkDef, IFfnLayout, IGptLayerNormLayout, IGptModelLayout, IStackLayout } from "../GptModelLayout";
import { measureTextWidth, writeTextToBuffer } from "../render/fontRender";
import { addLine } from "../render/lineRender";
import { IRenderState } from "../render/modelRender";
import { lerp } from "@/src/utils/math";
import { Mat4f } from "@/src/utils/matrix";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { IBertModelLayout } from "../BertModelLayout";
import { IVitModelLayout } from "../VitModelLayout";

// probably should limit this to specific blocks (defined by walkthrough)
// can configure this on the blocks themselves, or the layout object
// either way, this runs after the walkthrough, for positioning, and the walkthrough
// needs to configure this is some way

/** Label helpers; `stack.dir` says which side of the residual column the labels go (-1: left, +1: right). */
function labelDrawer(state: IRenderState, margin: number) {
    let baseColor = new Vec4(0.4, 0.4, 0.4, 1.0);

    function outerEdge(stack: { dir: number }, cubes: IBlkDef[]) {
        let visible = cubes.filter(c => c.opacity > 0);
        if (visible.length === 0) {
            visible = cubes;
        }
        return stack.dir < 0 ? Math.min(...visible.map(c => c.x)) : Math.max(...visible.map(c => c.x + c.dx));
    }

    function yRange(cubes: IBlkDef[]) {
        return [Math.min(...cubes.map(c => c.y)), Math.max(...cubes.map(c => c.y + c.dy))];
    }

    function vertLabel(stack: { dir: number }, text: string, x: number, cubes: IBlkDef[], fontSize: number, color: Vec4, pad?: number) {
        let [top, bot] = yRange(cubes);
        drawSectionLabel(state, text, new Vec3(x, top - margin / 2, 0), new Vec3(x, bot + margin / 2, 0), { color, fontSize, pad, dir: stack.dir });
    }

    function attnLabels(stack: { dir: number }, attn: IAttentionLayout, title: string, x: number) {
        let headCubes = attn.heads.flatMap(h => h.cubes);
        let color = baseColor.mul(attn.attnResidual.opacity * attn.attnLabel.visible);
        vertLabel(stack, title, x, [...headCubes, attn.attnResidual], 12, color);

        let projColor = baseColor.mul(attn.attnOut.opacity * attn.projLabel.visible);
        let projX = outerEdge(stack, [attn.projWeight]) + stack.dir * margin;
        vertLabel(stack, 'Projection', projX, [attn.projWeight], 10, projColor);

        let headX = outerEdge(stack, attn.heads.map(h => h.qWeightBlock)) + stack.dir * margin;
        let headLabelX = headX + stack.dir * margin * 3;
        attn.heads.forEach((head, headIdx) => {
            {
                let color = baseColor.mul(head.attnMtx.opacity * head.headLabel.visible);
                let tl = new Vec3(headLabelX, head.vBlock.y, head.vBlock.z + head.vBlock.dz / 2);
                let br = new Vec3(headLabelX, head.qBlock.y + head.qBlock.dy, head.qBlock.z + head.qBlock.dz / 2);
                if (head.qBlock.y !== head.vBlock.y) {
                    tl = new Vec3(headLabelX, head.vBlock.y, head.vOutBlock.z + head.vOutBlock.dz / 2);
                    br = new Vec3(headLabelX, head.vOutBlock.y + head.vOutBlock.dy, head.vOutBlock.z + head.vOutBlock.dz / 2);
                }
                drawSectionLabel(state, `Head ${headIdx + 1}`, tl, br, { color, fontSize: 10, dir: stack.dir });
            }

            for (let [lbl, blk, label] of [['Q', head.qBlock, head.qLabel], ['K', head.kBlock, head.kLabel], ['V', head.vBlock, head.vLabel]] as const) {
                let color = baseColor.mul(blk.opacity * label.visible);
                let tl = new Vec3(headX, blk.y, blk.z + blk.dz / 2);
                let br = new Vec3(headX, blk.y + blk.dy, blk.z + blk.dz / 2);
                drawSectionLabel(state, lbl, tl, br, { color, fontSize: 6, pad: 4, dir: stack.dir });
            }
        });
    }

    function lnLabel(stack: { dir: number }, ln: IGptLayerNormLayout, text: string = 'Add & Norm') {
        let color = baseColor.mul(ln.lnResid.opacity * ln.label.visible);
        let x = outerEdge(stack, [ln.lnMu]) + stack.dir * margin * 0.5;
        vertLabel(stack, text, x, [ln.lnMu], 5, color, 4);
    }

    function ffnLabel(stack: { dir: number }, ffn: IFfnLayout, x: number, text: string = 'Feed Forward') {
        let color = baseColor.mul(ffn.mlpAct.opacity * ffn.ffnLabel.visible);
        vertLabel(stack, text, x, ffn.cubes, 12, color);
    }

    return { baseColor, outerEdge, yRange, vertLabel, attnLabels, lnLabel, ffnLabel };
}

export function drawBlockLabels(state: IRenderState, layout: IGptModelLayout) {
    let margin = layout.margin;
    let { baseColor, outerEdge, vertLabel, attnLabels, lnLabel, ffnLabel } = labelDrawer(state, margin);

    for (let stack of [layout.enc, layout.dec] as IStackLayout[]) {
        let isEnc = stack === layout.enc;
        let allBlockCubes = stack.blocks.flatMap(b => b.cubes);
        let outer = outerEdge(stack, allBlockCubes);
        let sublayerX = outer + stack.dir * margin * 3;
        let layerX = outer + stack.dir * margin * 10;
        let stackX = outer + stack.dir * margin * 17;

        {
            let color = baseColor.mul(stack.embedLabel.visible);
            let x = outerEdge(stack, [stack.tokEmbedObj]) + stack.dir * margin * 2;
            vertLabel(stack, isEnc ? 'Input Embedding' : 'Output Embedding', x, [stack.tokEmbedObj], 6, color, 4);
        }

        {
            let color = baseColor.mul(0.7);
            vertLabel(stack, isEnc ? 'Encoder' : 'Decoder', stackX, allBlockCubes, 40, color, 20);
        }

        for (let block of stack.blocks) {
            let color = baseColor.mul(block.output.opacity * block.blockLabel.visible);
            vertLabel(stack, `${isEnc ? 'Encoder' : 'Decoder'} Layer ${block.idx + 1}`, layerX, block.cubes, 22, color);

            if ('attn' in block) {
                attnLabels(stack, block.attn, 'Self-Attention', sublayerX);
                lnLabel(stack, block.ln1);
                ffnLabel(stack, block.ffn, sublayerX);
                lnLabel(stack, block.ln2);
            } else {
                attnLabels(stack, block.selfAttn, 'Masked Self-Attention', sublayerX);
                lnLabel(stack, block.ln1);
                attnLabels(stack, block.crossAttn, 'Cross-Attention', sublayerX);
                lnLabel(stack, block.ln2);
                ffnLabel(stack, block.ffn, sublayerX);
                lnLabel(stack, block.ln3);
            }
        }
    }

    {
        let color = baseColor.mul(layout.outputLabel.visible);
        let x = outerEdge(layout.dec, [layout.lmHeadWeight]) + margin;
        vertLabel(layout.dec, 'Linear + Softmax', x, [layout.lmHeadWeight, layout.logitsSoftmax], 8, color, 4);
    }
}

export interface ILabelOpts {
    color: Vec4;
    fontSize: number;
    textAlign?: TextAlignHoriz;
    textAlignV?: TextAlignVert;
    pad?: number;
    dir?: number; // -1: label to the left of the line (default), +1: to the right
}

function drawSectionLabel(state: IRenderState, text: string, tl: Vec3, br: Vec3, opts: ILabelOpts) {
    let mtx = new Mat4f();
    mtx[14] = (tl.z + br.z) / 2;

    let color = opts.color;
    let fontScale = opts.fontSize;
    let pad = opts.pad ?? 10;

    let textColor = color;
    let lineColor = color.mul(0.4);

    let tw = measureTextWidth(state.modelFontBuf, text, fontScale);

    let dir = opts.dir ?? -1;
    let textX = dir < 0 ? tl.x - tw - 2 * pad : tl.x + 2 * pad;
    writeTextToBuffer(state.modelFontBuf, text, textColor, textX, (tl.y + br.y) / 2 - fontScale / 2, fontScale, mtx);

    let p0 = new Vec3(tl.x, tl.y, (tl.z + br.z) / 2);
    let p1 = new Vec3(br.x, br.y, (tl.z + br.z) / 2);

    if (tl.z != br.z) {
        p0 = new Vec3(tl.x, (tl.y + br.y) / 2, tl.z);
        p1 = new Vec3(tl.x, (tl.y + br.y) / 2, br.z);
    }

    let inward = new Vec3(-dir, 0, 0);

    addLine(state.lineRender, 1.0, lineColor, p0.mulAdd(inward, -pad), p1.mulAdd(inward, -pad), undefined);

    addLine(state.lineRender, 1.0, lineColor, p0.mulAdd(inward, -pad), p0, undefined);
    addLine(state.lineRender, 1.0, lineColor, p1.mulAdd(inward, -pad), p1, undefined);

}

export function drawVitBlockLabels(state: IRenderState, layout: IVitModelLayout) {
    let margin = layout.margin;
    let { baseColor, outerEdge, vertLabel, attnLabels, lnLabel, ffnLabel } = labelDrawer(state, margin);
    let stack = { dir: layout.dir };

    {
        let color = baseColor.mul(layout.embedLabel.visible);
        let x = outerEdge(stack, [layout.patchBias]) - margin * 2;
        vertLabel(stack, 'Patch + Position Embedding', x, [layout.patchWeight], 8, color, 4);
    }

    // one "Transformer Encoder" label per column of layers
    let columns = new Map<number, IBlkDef[]>();
    for (let block of layout.blocks) {
        columns.set(block.x0, [...(columns.get(block.x0) ?? []), ...block.cubes]);
    }
    let first = true;
    for (let cubes of columns.values()) {
        let x = outerEdge(stack, cubes) - margin * 17;
        vertLabel(stack, first ? 'Transformer Encoder' : 'Transformer Encoder (cont.)', x, cubes, 40, baseColor.mul(0.7), 20);
        first = false;
    }

    for (let block of layout.blocks) {
        let outer = outerEdge(stack, block.cubes);
        let color = baseColor.mul(block.output.opacity * block.blockLabel.visible);
        vertLabel(stack, `Layer ${block.idx + 1}`, outer - margin * 10, block.cubes, 22, color);
        lnLabel(stack, block.ln1, 'Norm');
        attnLabels(stack, block.attn, 'Multi-Head Attention', outer - margin * 3);
        lnLabel(stack, block.ln2, 'Norm');
        ffnLabel(stack, block.ffn, outer - margin * 3, 'MLP');
    }

    lnLabel(stack, layout.lnF, 'Norm');
    {
        let color = baseColor.mul(layout.outputLabel.visible);
        let x = outerEdge(stack, [layout.headBias]) - margin;
        vertLabel(stack, 'MLP Head', x, [layout.headWeight, layout.logitsSoftmax], 8, color, 4);
    }
}

export function drawBertBlockLabels(state: IRenderState, layout: IBertModelLayout) {
    let margin = layout.margin;
    let { baseColor, outerEdge, vertLabel, attnLabels, lnLabel, ffnLabel } = labelDrawer(state, margin);
    let stack = { dir: layout.dir };

    {
        let color = baseColor.mul(layout.embedLabel.visible);
        let x = outerEdge(stack, [layout.segEmbed]) - margin * 2;
        vertLabel(stack, 'Token + Segment + Position', x, [layout.tokEmbed], 8, color, 4);
    }

    lnLabel(stack, layout.embedLN, 'Norm');

    // one "BERT Encoder" label per column of layers (the tiny model fits in a single column)
    let columns = new Map<number, IBlkDef[]>();
    for (let block of layout.blocks) {
        columns.set(block.x0, [...(columns.get(block.x0) ?? []), ...block.cubes]);
    }
    let first = true;
    for (let cubes of columns.values()) {
        let x = outerEdge(stack, cubes) - margin * 17;
        vertLabel(stack, first ? 'BERT Encoder' : 'BERT Encoder (cont.)', x, cubes, 40, baseColor.mul(0.7), 20);
        first = false;
    }

    for (let block of layout.blocks) {
        let outer = outerEdge(stack, block.cubes);
        let color = baseColor.mul(block.output.opacity * block.blockLabel.visible);
        vertLabel(stack, `Layer ${block.idx + 1}`, outer - margin * 10, block.cubes, 22, color);
        lnLabel(stack, block.ln1, 'Norm');
        attnLabels(stack, block.attn, 'Bidirectional Self-Attention', outer - margin * 3);
        lnLabel(stack, block.ln2, 'Norm');
        ffnLabel(stack, block.ffn, outer - margin * 3, 'FFN');
    }

    {
        let color = baseColor.mul(layout.mlmLabel.visible);
        let x = outerEdge(stack, [layout.mlmOutBias]) - margin;
        vertLabel(stack, 'MLM Head', x, [layout.mlmFcWeight, layout.mlmProbs], 14, color, 8);
    }
    {
        let color = baseColor.mul(layout.outputLabel.visible);
        let x = outerEdge(stack, [layout.poolBias]) - margin;
        vertLabel(stack, '[CLS] Heads', x, [layout.poolWeight, layout.sentimentProbs], 14, color, 8);
    }
}
