
/* Arrow Design:

This is a flat, rectangular, ribbon-like pathway with lines down the edges.

It has a defined width. The ribbon might have some motion animation when I write the shader for it.

The sides will have lines to border the transparent ribbon.

We define a start, end, width, and "normal". The normal is perpendicular to the direction of the arrow,
and the normal lies within all parts of the ribbon.

The ribbon can bend, subject to the normal. The ribbon always starts in-plane. We'll have to do some
curve generation to make the ribbon bend. (this probably requires a 3rd degree bezier curve)

The arrow head is fairly flat, and is a triangle. We define the depth of the arrow head, and the
width. Not sure if the width should be a constant, or proportional to the width/length of the arrow.

Arrows will also have padding between blocks.

Some special arrows bend at the start in the plane of the ribbon, where they diverge from a vertical arrow.

This should be able to be implemented just with lines & tris. Depth buf will be interesting.
*/

import { IAttentionLayout, IBlkDef, IFfnLayout, IGptLayerNormLayout, IGptModelLayout, IModelLayout, IStackLayout } from "../GptModelLayout";
import { IBertModelLayout } from "../BertModelLayout";
import { IVitModelLayout } from "../VitModelLayout";
import { addLine, drawLineSegs, ILineOpts, makeLineOpts } from "../render/lineRender";
import { IRenderState } from "../render/modelRender";
import { addPrimitiveRestart, addQuad, addVert } from "../render/triRender";
import { bezierCurveBuild } from "@/src/utils/bezier";
import { Mat4f } from "@/src/utils/matrix";
import { Vec3, Vec3Buf, Vec4 } from "@/src/utils/vector";

/** The arrow helpers for a layout; setDir() picks which side of the residual column the side computations are on. */
function arrowDrawer(state: IRenderState, layout: IModelLayout) {

    let pad = 2.0;
    let residWidth = 6;
    let weightColor = Vec4.fromHexColor('#3333aa');
    let dataColor = Vec4.fromHexColor('#33aa33');

    // which side of a block faces the residual column (inSide), and which faces away (outSide)
    let inSide = BlockPos.Right;
    let outSide = BlockPos.Left;
    let dirX = -1;

    function setDir(dir: number) {
        dirX = dir;
        inSide = dirX < 0 ? BlockPos.Right : BlockPos.Left;
        outSide = dirX < 0 ? BlockPos.Left : BlockPos.Right;
    }

    function drawEmbedding(stack: IStackLayout) {
        drawVertArrow(stack.idxObj, stack.residual0);
        drawInward(stack.tokEmbedObj, stack.residual0);
        drawArrowBetween(stack.posEmbedObj, outSide, stack.residual0, inSide);
    }

    function drawAttention(attn: IAttentionLayout) {
        drawVertArrow(attn.residSrc, attn.attnResidual);

        for (let head of attn.heads) {
            drawSplitTo(attn.xq, head.qBlock);
            drawSplitTo(attn.xkv, head.kBlock);
            drawSplitTo(attn.xkv, head.vBlock);

            drawInward(head.qWeightBlock, head.qBlock);
            drawInward(head.kWeightBlock, head.kBlock);
            drawInward(head.vWeightBlock, head.vBlock);

            head.qBiasBlock && drawInward(head.qBiasBlock, head.qWeightBlock);
            head.kBiasBlock && drawInward(head.kBiasBlock, head.kWeightBlock);
            head.vBiasBlock && drawInward(head.vBiasBlock, head.vWeightBlock);

            drawArrowBotToSide(head.qBlock, head.attnMtx, 0, undefined, head.qBlock.y !== head.kBlock.y);
            drawArrowBotToSide(head.kBlock, head.attnMtx, 0, undefined, head.kBlock.y !== head.qBlock.y);
            drawArrowBotToSide(head.vBlock, head.vOutBlock, 0, undefined, head.vBlock.y !== head.kBlock.y);

            drawArrowBetween(head.attnMtx, outSide, head.attnMtxAgg2, inSide);
            drawArrowBetween(head.attnMtxAgg1, outSide, head.attnMtxSm, inSide);
            drawArrowBetween(head.attnMtxSm, BlockPos.Bot, head.vOutBlock, outSide);

            drawArrowBetween(head.vOutBlock, BlockPos.Bot, attn.attnOut, BlockPos.Top);
        }

        attn.projBias && drawInward(attn.projBias, attn.projWeight);
        drawInward(attn.projWeight, attn.attnOut);
        drawInward(attn.attnOut, attn.attnResidual);
    }

    function drawLn(ln: IGptLayerNormLayout) {
        if (ln.preNorm) {
            // the layer norm is beside the residual column: branch off to it
            drawArrowBetween(ln.residSrc, BlockPos.Bot, ln.lnAgg1, inSide);
        } else {
            drawVertArrow(ln.residSrc, ln.lnAgg1);
        }
        drawVertArrow(ln.lnAgg2, ln.lnResid);
        drawInward(ln.lnMu, ln.lnSigma);
        drawInward(ln.lnSigma, ln.lnResid);
    }

    function drawFfn(ffn: IFfnLayout) {
        drawVertArrow(ffn.residSrc, ffn.mlpResidual);
        drawArrowBetween(ffn.src, BlockPos.Bot, ffn.mlpFc, inSide);

        drawVertArrow(ffn.mlpFcBias, ffn.mlpFcWeight);
        drawVertArrow(ffn.mlpFcWeight, ffn.mlpFc, 12);
        drawVertArrow(ffn.mlpFc, ffn.mlpAct, 12);
        drawInward(ffn.mlpProjBias, ffn.mlpProjWeight);
        drawInward(ffn.mlpProjWeight, ffn.mlpResult);
        drawInward(ffn.mlpResult, ffn.mlpResidual);
        drawArrowBetween(ffn.mlpAct, inSide, ffn.mlpResult, BlockPos.Top);
    }

    function blkColor(src: IBlkDef) {
        return src.t === 'w' ? weightColor : dataColor;
    }

    function drawVertArrow(src: IBlkDef, dest: IBlkDef, width: number = 6) {
        drawArrowBetween(src, BlockPos.Bot, dest, BlockPos.Top, width);
    }

    // arrow from a residual-column block out to a side block: horizontal if they're level, otherwise a dogleg
    function drawSplitTo(src: IBlkDef, dest: IBlkDef, width: number = 6) {
        let destMidY = dest.y + dest.dy / 2;
        if (destMidY >= src.y && destMidY <= src.y + src.dy) {
            drawArrowBetween(src, outSide, dest, inSide, width, true);
        } else {
            drawArrowBetween(src, BlockPos.Bot, dest, inSide, width);
        }
    }

    // horizontal arrow, flowing towards the residual column
    function drawInward(src: IBlkDef, dest: IBlkDef, width: number = 6) {
        drawArrowBetween(src, inSide, dest, outSide, width);
    }

    function blockPos(block: IBlkDef, pos: BlockPos) {
        let z = block.z + block.dz / 2;
        switch (pos) {
            case BlockPos.Left: return new Vec3(block.x - pad, block.y + block.dy / 2, z);
            case BlockPos.Right: return new Vec3(block.x + block.dx + pad, block.y + block.dy / 2, z);
            case BlockPos.Top: return new Vec3(block.x + block.dx / 2, block.y - pad, z);
            case BlockPos.Bot: return new Vec3(block.x + block.dx / 2, block.y + block.dy + pad, z);
        }
    }

    function drawArrowBotToSide(src: IBlkDef, dest: IBlkDef, offset: number, width: number = 6, forceOffset: boolean = false) {
        let start = blockPos(src, BlockPos.Bot);
        let left = start.z > dest.z + dest.dz / 2;
        let end = new Vec3(dest.x + dest.dx / 2, dest.y + layout.cell * (offset + 0.5), left ? dest.z + dest.dz / 2 + pad : dest.z - pad);

        let opacity = Math.min(src.opacity, dest.opacity);
        if (opacity === 0) {
            return;
        }

        let normal = new Vec3(0, 0, 1);
        let color = blkColor(src).mul(opacity);
        let endDir = new Vec3(0, 0, left ? -1 : 1);

        let areClose = Math.abs(start.z - (dest.z + dest.dz / 2)) < 1.0;
        if (areClose && !forceOffset) {
            endDir = undefined!;
            end = blockPos(dest, BlockPos.Top);
        }

        drawArrow(state, start, end, width, normal, color, true, CornerMode.None, endDir);
    }

    function drawArrowBetween(src: IBlkDef, srcPos: BlockPos, dest: IBlkDef, destPos: BlockPos, width: number = 6, alignY: boolean = false) {
        let start = blockPos(src, srcPos);
        let end = blockPos(dest, destPos);

        let opacity = Math.min(src.opacity, dest.opacity);
        if (opacity === 0) {
            return;
        }

        let normal = new Vec3(0, 0, 1);
        let color = blkColor(src).mul(opacity);

        if (alignY) {
            start.y = end.y;
        }

        let isSide = (p: BlockPos) => p === BlockPos.Left || p === BlockPos.Right;

        if (isSide(srcPos) && destPos === BlockPos.Top) {
            // dogleg sideways => down
            let sx = Math.sign(end.x - start.x);
            let mid0 = new Vec3(end.x - sx * width / 2, start.y, start.z);
            let mid1 = new Vec3(end.x, start.y + width / 2, end.z);

            drawArrow(state, start, mid0, width, normal, color, false);
            drawArrow(state, mid1, end, width, normal, color, true, sx > 0 ? CornerMode.Left : CornerMode.Right);

        } else if (srcPos === BlockPos.Bot && isSide(destPos)) {
            // dogleg down => sideways
            let sx = Math.sign(end.x - start.x);
            let mid0 = new Vec3(start.x, end.y - width / 2, end.z);
            let mid1 = new Vec3(start.x + sx * width / 2, end.y, end.z);

            drawArrow(state, start, mid0, width, normal, color, false, CornerMode.None, new Vec3(0, 1, 0));
            drawArrow(state, mid1, end, width, normal, color, true, sx > 0 ? CornerMode.Right : CornerMode.Left);

        } else {

            drawArrow(state, start, end, width, normal, color, true);
        }
    }

    return {
        pad, residWidth, dataColor, weightColor,
        setDir, drawEmbedding, drawAttention, drawLn, drawFfn,
        drawVertArrow, drawSplitTo, drawInward, drawArrowBetween, blkColor,
    };
}

export function drawAllArrows(state: IRenderState, layout: IGptModelLayout) {
    let { pad, residWidth, dataColor, setDir, drawEmbedding, drawAttention, drawLn, drawFfn, drawVertArrow, drawInward } = arrowDrawer(state, layout);

    setDir(layout.enc.dir);
    drawEmbedding(layout.enc);
    for (let block of layout.enc.blocks) {
        drawAttention(block.attn);
        drawLn(block.ln1);
        drawFfn(block.ffn);
        drawLn(block.ln2);
    }

    setDir(layout.dec.dir);
    drawEmbedding(layout.dec);
    for (let block of layout.dec.blocks) {
        drawAttention(block.selfAttn);
        drawLn(block.ln1);
        drawAttention(block.crossAttn);
        drawLn(block.ln2);
        drawFfn(block.ffn);
        drawLn(block.ln3);
    }

    drawVertArrow(layout.dec.output, layout.logits);
    drawInward(layout.lmHeadWeight, layout.logits);
    drawVertArrow(layout.logits, layout.logitsAgg2);
    drawVertArrow(layout.logitsAgg1, layout.logitsSoftmax);

    // The encoder output ("memory") is routed across the gap between the stacks to each decoder layer's
    // cross-attention. It enters the memory block from below, crossing the decoder's residual column where it
    // only contains the residual arrow.
    let encOut = layout.enc.output;
    let busX = layout.midX;
    for (let block of layout.dec.blocks) {
        let mem = block.crossAttn.memCopy!;
        let opacity = Math.min(encOut.opacity, mem.opacity);
        if (opacity === 0) {
            continue;
        }
        let crossY = mem.y + mem.dy + layout.margin * 0.7;
        let memMidX = mem.x + mem.dx / 2;
        let start = new Vec3(encOut.x + encOut.dx + pad, encOut.y + encOut.dy / 2, encOut.z + encOut.dz / 2);
        let pts = [
            start,
            new Vec3(busX, start.y, start.z),
            new Vec3(busX, crossY, start.z),
            new Vec3(memMidX, crossY, start.z),
            new Vec3(memMidX, mem.y + mem.dy + pad, start.z),
        ];
        drawPolyArrow(state, pts, residWidth, dataColor.mul(opacity));
    }

}

/**
 * One BERT-specific arrow, named by the two blocks it joins.
 *
 * The per-layer arrows come from the shared drawer and are the Transformer's / ViT's own code, already
 * exercised; what is new here is the wiring around it \u2014 three embedding tables feeding one sum, the
 * MLM chain, and the two [CLS] heads. Keeping those in a plain list means the wiring is checkable in
 * node (scripts/check-bert.ts) instead of only by looking at a render and hoping.
 */
export interface IBertArrow {
    kind: 'embed' | 'mlm' | 'cls';
    from: IBlkDef;
    to: IBlkDef;
    mode: 'vert' | 'inward' | 'left' | 'split';
}

export function bertArrowPlan(layout: IBertModelLayout): IBertArrow[] {
    let last = layout.blocks[layout.blocks.length - 1]?.output ?? layout.residual0;
    let plan: IBertArrow[] = [
        // Figure 2: the token id picks a row of the token table, the segment id a row of the segment
        // table, the position picks its own row, and all three land in one sum.
        { kind: 'embed', from: layout.tokens, to: layout.residual0, mode: 'vert' },
        { kind: 'embed', from: layout.segIds, to: layout.segEmbed, mode: 'left' },
        { kind: 'embed', from: layout.tokEmbed, to: layout.residual0, mode: 'inward' },
        { kind: 'embed', from: layout.segEmbed, to: layout.residual0, mode: 'inward' },
        { kind: 'embed', from: layout.posEmbed, to: layout.residual0, mode: 'left' },
        { kind: 'embed', from: layout.residual0, to: layout.embedLN.lnResid, mode: 'split' },

        // the MLM head runs over every position: dense \u2192 GELU \u2192 LayerNorm \u2192 E\u1D40 \u2192 softmax
        { kind: 'mlm', from: layout.mlmFcBias, to: layout.mlmFcWeight, mode: 'vert' },
        { kind: 'mlm', from: layout.mlmFcWeight, to: layout.mlmDense, mode: 'vert' },
        { kind: 'mlm', from: layout.mlmDense, to: layout.mlmAct, mode: 'vert' },
        { kind: 'mlm', from: layout.mlmOutBias, to: layout.mlmOutWeight, mode: 'vert' },
        { kind: 'mlm', from: layout.mlmOutWeight, to: layout.mlmLogits, mode: 'inward' },
        { kind: 'mlm', from: layout.mlmLogits, to: layout.mlmAgg2, mode: 'vert' },
        { kind: 'mlm', from: layout.mlmAgg1, to: layout.mlmProbs, mode: 'vert' },

        // BERT's own pooler feeds NSP; the task classifier reads the raw [CLS] (\u00a74.1), so both
        // arrows start at the same column of the last block
        { kind: 'cls', from: last, to: layout.poolOut, mode: 'vert' },
        { kind: 'cls', from: layout.poolBias, to: layout.poolWeight, mode: 'vert' },
        { kind: 'cls', from: layout.poolWeight, to: layout.poolOut, mode: 'inward' },
        { kind: 'cls', from: layout.poolOut, to: layout.nspLogits, mode: 'vert' },
        { kind: 'cls', from: layout.nspBias, to: layout.nspWeight, mode: 'vert' },
        { kind: 'cls', from: layout.nspWeight, to: layout.nspLogits, mode: 'inward' },
        { kind: 'cls', from: layout.nspLogits, to: layout.nspAgg2, mode: 'vert' },
        { kind: 'cls', from: layout.nspAgg1, to: layout.nspProbs, mode: 'vert' },
        { kind: 'cls', from: last, to: layout.sentimentLogits, mode: 'vert' },
        { kind: 'cls', from: layout.sentimentBias, to: layout.sentimentWeight, mode: 'vert' },
        { kind: 'cls', from: layout.sentimentWeight, to: layout.sentimentLogits, mode: 'inward' },
        { kind: 'cls', from: layout.sentimentLogits, to: layout.sentimentAgg2, mode: 'vert' },
        { kind: 'cls', from: layout.sentimentAgg1, to: layout.sentimentProbs, mode: 'vert' },
    ];
    return plan;
}

export function drawBertArrows(state: IRenderState, layout: IBertModelLayout) {
    let { setDir, drawAttention, drawLn, drawFfn, drawVertArrow, drawInward, drawSplitTo, drawArrowBetween } = arrowDrawer(state, layout);
    setDir(layout.dir);

    drawLn(layout.embedLN);

    for (let block of layout.blocks) {
        drawLn(block.ln1);
        drawAttention(block.attn);
        drawLn(block.ln2);
        drawFfn(block.ffn);
    }

    drawLn(layout.mlmLn);

    for (let arrow of bertArrowPlan(layout)) {
        switch (arrow.mode) {
            case 'vert': drawVertArrow(arrow.from, arrow.to); break;
            case 'inward': drawInward(arrow.from, arrow.to); break;
            case 'left': drawArrowBetween(arrow.from, BlockPos.Left, arrow.to, BlockPos.Right); break;
            case 'split': drawSplitTo(arrow.from, arrow.to); break;
        }
    }
}

export function drawVitArrows(state: IRenderState, layout: IVitModelLayout) {
    let { residWidth, dataColor, setDir, drawAttention, drawLn, drawFfn, drawVertArrow, drawInward, drawArrowBetween } = arrowDrawer(state, layout);
    setDir(layout.dir);

    drawVertArrow(layout.patches, layout.tokens);
    drawInward(layout.patchBias, layout.patchWeight);
    drawInward(layout.patchWeight, layout.tokens);
    drawArrowBetween(layout.clsToken, BlockPos.Left, layout.tokens, BlockPos.Right);
    drawVertArrow(layout.tokens, layout.residual0);
    drawArrowBetween(layout.posEmbed, BlockPos.Left, layout.residual0, BlockPos.Right);

    let prevX0 = layout.blocks[0]?.x0;
    for (let block of layout.blocks) {
        // the residual pathway jumps to the top of the next column (for > 12 layers); we don't draw that jump
        if (block.x0 === prevX0) {
            drawLn(block.ln1);
        } else {
            drawVertArrow(block.ln1.lnAgg2, block.ln1.lnResid);
            drawInward(block.ln1.lnMu, block.ln1.lnSigma);
            drawInward(block.ln1.lnSigma, block.ln1.lnResid);
        }
        drawAttention(block.attn);
        drawLn(block.ln2);
        drawFfn(block.ffn);
        prevX0 = block.x0;
    }

    drawLn(layout.lnF);

    // the head only reads the [class] token (column 0 of the final layer norm)
    let ln = layout.lnF.lnResid;
    let logits = layout.logits;
    let opacity = Math.min(ln.opacity, logits.opacity);
    if (opacity > 0) {
        let x = logits.x + logits.dx / 2;
        let z = ln.z + ln.dz / 2;
        drawPolyArrow(state, [new Vec3(x, ln.y + ln.dy + 2, z), new Vec3(x, logits.y - 2, z)], residWidth, dataColor.mul(opacity));
    }
    drawInward(layout.headBias, layout.headWeight);
    drawInward(layout.headWeight, layout.logits);
    drawVertArrow(layout.logits, layout.logitsAgg2);
    drawVertArrow(layout.logitsAgg1, layout.logitsSoftmax);
}

/** A ribbon arrow along an axis-aligned polyline (in the x-y plane), with corners at each turn. */
export function drawPolyArrow(state: IRenderState, pts: Vec3[], width: number, color: Vec4) {
    let normal = new Vec3(0, 0, 1);
    let dirs = [];
    for (let i = 0; i < pts.length - 1; i++) {
        dirs.push(pts[i + 1].sub(pts[i]).normalize());
    }
    for (let i = 0; i < pts.length - 1; i++) {
        let d = dirs[i];
        let isLast = i === pts.length - 2;
        let start = i > 0 ? pts[i].mulAdd(d, width / 2) : pts[i];
        let end = !isLast ? pts[i + 1].mulAdd(d, -width / 2) : pts[i + 1];
        let corner = CornerMode.None;
        if (i > 0) {
            let prev = dirs[i - 1];
            let cross = prev.x * d.y - prev.y * d.x;
            corner = cross > 0 ? CornerMode.Left : CornerMode.Right;
        }
        if (end.sub(start).len() < 0.01) {
            continue;
        }
        drawArrow(state, start, end, width, normal, color, isLast, corner);
    }
}

export enum BlockPos {
    Left,
    Right,
    Top,
    Bot,
}

export enum CornerMode {
    None,
    Left,
    Right,
}

export interface IEndDir {
    dir: Vec3;
}

let _bezierLineBuf = new Float32Array(4 * 3);
export function drawArrow(state: IRenderState, start: Vec3, end: Vec3, width: number, normal: Vec3, color: Vec4, drawHead: boolean = true, drawCorner: CornerMode = CornerMode.None, endDir?: Vec3) {

    let dir = end.sub(start);
    dir.z = 0.0;
    dir = dir.normalize();

    let len = end.sub(start).len();
    let headExtra = 3.0;
    let headDepth = drawHead ? Math.min(len * 0.7, headExtra * 1.0) : 0;
    let mtx = new Mat4f();

    let side = Vec3.cross(dir, normal).mul(-1).normalize();
    normal = Vec3.cross(side, dir).normalize();

    mtx[0] = side.x;
    mtx[1] = side.y;
    mtx[2] = side.z;
    mtx[4] = dir.x;
    mtx[5] = dir.y;
    mtx[6] = dir.z;
    mtx[8] = normal.x;
    mtx[9] = normal.y;
    mtx[10] = normal.z;
    // mtx[14] = start.z;

    start = mtx.mulVec3Proj(start);
    end = mtx.mulVec3Proj(end);

    let borderColor = color.mul(0.8);
    let ribbonColor = color.mul(0.3);

    let opts: IArrowOpts = {
        width,
        borderColor,
        ribbonColor,
        headDepth,
        headExtra,
        lineThick: 1.2,
        mtx,
    };

    endDir = endDir ? mtx.mulVec3ProjVec(endDir) : undefined;

    // matrix is constructed such that the ribbons are always in the x-y plane, and going from -y to +y
    // The ribbon will be curved if it is not in the x-y plane, but the start and end remain tangent to the plane

    function drawBezierRibbon() {
        let dist = Math.max(headDepth, Math.abs(start.y - end.y - headDepth) / 2);
        let p0 = new Vec3(start.x, start.y               , start.z);
        let p1 = new Vec3(start.x, start.y + dist        , start.z);
        let p2 = new Vec3(end.x, end.y - headDepth - dist, end.z);
        let p3 = new Vec3(end.x, end.y - headDepth       , end.z);
        if (endDir) {
            p2 = end.mulAdd(endDir, -headDepth - dist);
            p3 = end.mulAdd(endDir, -headDepth);
        }
        let steps = bezierCurveBuild(p0, p1, p2, p3, 0.1);
        let nPts = steps.length / 3;

        let headNPts = drawHead ? 3 : 0;
        let nFloats = (nPts * 2 + headNPts) * 3;
        if (_bezierLineBuf.length < nFloats) {
            _bezierLineBuf = new Float32Array(nFloats);
        }
        let arrowLine = _bezierLineBuf.subarray(0, nFloats);

        for (let i = 0; i < nPts - 1; i++) {
            let p0 = new Vec3(steps[i*3+0], steps[i*3+1], steps[i*3+2]);
            let p1 = new Vec3(steps[i*3+3], steps[i*3+4], steps[i*3+5]);
            drawArrowSeg(state, p0, p1, opts);
        }

        let arrowStartIdx = nPts;
        let arrowEndIdx = arrowStartIdx + headNPts;
        for (let i = 0; i < nPts; i++) {
            let j = arrowEndIdx + i;
            arrowLine[j*3+0] = steps[i*3+0] + opts.width / 2;
            arrowLine[j*3+1] = steps[i*3+1];
            arrowLine[j*3+2] = steps[i*3+2];
            let k = nPts - i - 1;
            arrowLine[k*3+0] = steps[i*3+0] - opts.width / 2;
            arrowLine[k*3+1] = steps[i*3+1];
            arrowLine[k*3+2] = steps[i*3+2];
        }

        if (drawHead) {
            let headExtra = 3.0;
            endDir = endDir ?? new Vec3(0, 1, 0);
            let endA = end.mulAdd(endDir, -headDepth);
            let i = arrowStartIdx;
            arrowLine[i*3+0] = endA.x - opts.width / 2 - headExtra;
            arrowLine[i*3+1] = endA.y;
            arrowLine[i*3+2] = endA.z;
            i += 1;
            arrowLine[i*3+0] = end.x;
            arrowLine[i*3+1] = end.y;
            arrowLine[i*3+2] = end.z;
            i += 1;
            arrowLine[i*3+0] = endA.x + opts.width / 2 + headExtra;
            arrowLine[i*3+1] = endA.y;
            arrowLine[i*3+2] = endA.z;
        }

        let lineOpts = makeLineOpts({ thick: opts.lineThick, mtx: opts.mtx, color: opts.borderColor });
        drawLineSegs(state.lineRender, arrowLine, lineOpts);
    }

    if (Math.abs(start.z - end.z) > 0.01 || endDir) {
        drawBezierRibbon();

    } else {
        drawArrowSeg(state, start, end.sub(new Vec3(0, headDepth)), opts);
    }

    if (drawCorner !== CornerMode.None) {
        drawArrowCorner(state, start.sub(new Vec3(0, width/2)), drawCorner, opts);
    }
    if (drawHead) {
        endDir = endDir ?? new Vec3(0, 1, 0);
        drawArrowHead(state, end.mulAdd(endDir, -headDepth), end, opts);
    }
}

interface IArrowOpts {
    width: number;
    headExtra: number;
    headDepth: number;
    borderColor: Vec4;
    ribbonColor: Vec4;
    lineThick: number;
    mtx: Mat4f;
}

// Inner loop of bezier curve, so make this fast
let _segTl = new Vec3();
let _segBr = new Vec3();
let _segBl = new Vec3();
let _segTr = new Vec3();
export function drawArrowSeg(state: IRenderState, start: Vec3, end: Vec3, opts: IArrowOpts) {
    _segTl.x = start.x - opts.width/2;
    _segTl.y = start.y;
    _segTl.z = start.z;

    _segBr.x = end.x + opts.width/2;
    _segBr.y = end.y;
    _segBr.z = end.z;

    addQuad(state.triRender, _segTl, _segBr, opts.ribbonColor, opts.mtx);

    // _segBl.x = _segTl.x;
    // _segBl.y = _segBr.y;
    // _segBl.z = _segBr.z;

    // _segTr.x = _segBr.x;
    // _segTr.y = _segTl.y;
    // _segTr.z = _segTl.z;

    // let n = undefined;
    // let thick = opts.lineThick;
    // addLine(state.lineRender, thick, opts.borderColor, _segTl, _segBl, n, opts.mtx);
    // addLine(state.lineRender, thick, opts.borderColor, _segTr, _segBr, n, opts.mtx);
}

let _headTl = new Vec3();
let _headTr = new Vec3();
let _headBr = new Vec3();
let _headLeft = new Vec3();
let _headRight = new Vec3();
let _headTip = new Vec3();
let _headN = new Vec3(0, 0, 1);
export function drawArrowHead(state: IRenderState, a: Vec3, b: Vec3, opts: IArrowOpts) {
    let headExtra = 3.0;

    _headTl.copy_(a);
    _headTl.x -= opts.width/2;
    _headTr.copy_(a);
    _headTr.x += opts.width/2;
    _headBr.copy_(b);
    _headBr.x += opts.width/2;
    _headLeft.copy_(a);
    _headLeft.x = _headTl.x - headExtra;
    _headRight.copy_(a);
    _headRight.x = _headBr.x + headExtra;
    _headTip.copy_(b);
    _headTip.x = _headTl.x + opts.width / 2;

    addVert(state.triRender, _headLeft, opts.ribbonColor, _headN, opts.mtx);
    addVert(state.triRender, _headTip, opts.ribbonColor, _headN, opts.mtx);
    addVert(state.triRender, _headRight, opts.ribbonColor, _headN, opts.mtx);
    addPrimitiveRestart(state.triRender);

    // let thick = opts.lineThick;
    // addLine(state.lineRender, thick, opts.borderColor, _headTl, _headLeft, undefined, opts.mtx);
    // addLine(state.lineRender, thick, opts.borderColor, _headTip, _headLeft, undefined, opts.mtx);
    // addLine(state.lineRender, thick, opts.borderColor, _headTip, _headRight, undefined, opts.mtx);
    // addLine(state.lineRender, thick, opts.borderColor, _headTr, _headRight, undefined, opts.mtx);
}

let _cornerPivot = new Vec3();
let _cornerN = new Vec3(0, 0, 1);
let _cornerCurrP = new Vec3();
let _cornerPrevP = new Vec3();
export function drawArrowCorner(state: IRenderState, center: Vec3, mode: CornerMode, opts: IArrowOpts) {

    // coming from the left (-x), and going downwards (+y)
    let mul = mode === CornerMode.Left ? 1 : -1;

    _cornerPivot.x = center.x + opts.width / 2 * mul;
    _cornerPivot.y = center.y + opts.width / 2;
    _cornerPivot.z = center.z;

    _cornerCurrP.z = center.z;
    _cornerPrevP.z = center.z;
    let count = 8;

    for (let i = 0; i < count; i++) {
        let theta = i / (count - 1) * Math.PI / 2;
        let c = opts.width * Math.cos(theta) * mul;
        let s = opts.width * Math.sin(theta);
        _cornerCurrP.x = _cornerPivot.x - c;
        _cornerCurrP.y = _cornerPivot.y - s;

        addVert(state.triRender, _cornerCurrP, opts.ribbonColor, _cornerN, opts.mtx);
        addVert(state.triRender, _cornerPivot, opts.ribbonColor, _cornerN, opts.mtx);

        if (i > 0) {
            // addLine(state.lineRender, opts.lineThick, opts.borderColor, _cornerPrevP, _cornerCurrP, undefined, opts.mtx);
        }
        let tmp = _cornerPrevP;
        _cornerPrevP = _cornerCurrP;
        _cornerCurrP = tmp;
    }

    addPrimitiveRestart(state.triRender);
}
