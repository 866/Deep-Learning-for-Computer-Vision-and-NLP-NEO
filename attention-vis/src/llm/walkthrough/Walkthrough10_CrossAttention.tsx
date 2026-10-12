import React from "react";
import { Dim, Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { processUpTo, startProcessBefore } from "./Walkthrough00_Intro";
import { processDim } from "./Walkthrough03_SelfAttention";
import { cameraFor, commentary, DimStyle, IWalkthroughArgs, moveCameraTo, setInitialCamera } from "./WalkthroughTools";
import { embedInline } from "./Walkthrough01_Prelim";
import { getBlockValueAtIdx } from "../components/DataFlow";
import { lerp } from "@/src/utils/math";
import { BlockText } from "../components/CommentaryHelpers";

export function walkthrough10_CrossAttention(args: IWalkthroughArgs) {
    let { walkthrough: wt, state, layout, tools: { afterTime, breakAfter, c_blockRef, c_dimRef } } = args;

    if (wt.phase !== Phase.Dec_CrossAttention) {
        return;
    }

    let { T, nBlocks } = layout.shape;
    let dec = layout.dec;
    let cross0 = dec.blocks[0].crossAttn;
    let crossN = dec.blocks[nBlocks - 1].crossAttn;
    let head0 = cross0.heads[0];

    // pick the head of the last cross-attention whose pattern looks most like reversal
    let bestHead = crossN.heads[0];
    let bestScore = -1;
    for (let h of crossN.heads) {
        let score = 0;
        for (let t = 0; t < T; t++) {
            score += getBlockValueAtIdx(h.attnMtxSm, new Vec3(T - 1 - t, t, 0)) ?? 0;
        }
        if (score > bestScore) {
            bestScore = score;
            bestHead = h;
        }
    }
    let bestHeadIdx = crossN.heads.indexOf(bestHead);

    let camOverview = cameraFor([cross0.memCopy!, ...cross0.cubes, cross0.xq], new Vec3(262, 14, 0), 1.0);
    let camHeads = cameraFor([cross0.xq, cross0.memCopy!, ...cross0.heads.flatMap(h => [h.qBlock, h.kBlock, h.vBlock, h.attnMtx])], new Vec3(250, 25, 0), 0.9);
    let camMtx0 = cameraFor([head0.attnMtx, head0.attnMtxSm, head0.vOutBlock], new Vec3(268, -25, 0), 0.8);
    let camMtxN = cameraFor(crossN.heads.flatMap(h => [h.attnMtxSm, h.attnMtx, h.vOutBlock]), new Vec3(305, 25, 0), 0.8);

    setInitialCamera(state, camOverview.center, camOverview.angle);
    wt.dimHighlightBlocks = [cross0.xq, cross0.memCopy!, ...cross0.heads.flatMap(h => h.cubes)];

    commentary(wt, null, 0)`
After masked self-attention (and its Add & Norm), each decoder layer has a second attention sublayer: _cross-attention_, which the paper calls
_encoder-decoder attention_. This is where the decoder reads the source sequence.

It's the same multi-head attention as before, with one twist: the queries and the keys/values come from different places.

${embedInline(<ul>
    <li>The Q vectors come from the decoder: the <BlockText blk={cross0.xq}>normalized output of the masked self-attention</BlockText>.</li>
    <li>The K and V vectors come from the <BlockText blk={cross0.memCopy!}>encoder output</BlockText>, the same memory for every decoder layer.</li>
</ul>)}
So each output position asks a question (its query), and looks for the answer among the source positions (their keys).`;
    breakAfter();

    let t_highlightInputs = afterTime(null, 1.5, 0.3);
    let t_camHeads = afterTime(null, 1.0, 0.3);
    let t_processQkv = afterTime(null, 3.0, 0.3);

    breakAfter();

    commentary(wt)`
The ${c_blockRef('K', head0.kBlock)} and ${c_blockRef('V vectors', head0.vBlock)} are computed from the ${c_blockRef('encoder output', cross0.memCopy!)}, and the ${c_blockRef('Q vectors', head0.qBlock)} from
the ${c_blockRef('decoder', cross0.xq)}. The ${c_blockRef('attention matrix', head0.attnMtx)} now has one row per _output_ position and one column per _source_ position.
There's no mask: every output position can look at the whole source sequence (it was fully known from the start).

Let's compute a row of it:`;
    breakAfter();

    let t_camMtx = afterTime(null, 1.0, 0.3);
    let t_row = afterTime(null, 2.0, 0.3);
    let t_rowSm = afterTime(null, 1.0, 0.3);
    let t_processAll = afterTime(null, 5.0, 0.3);

    breakAfter();

    commentary(wt)`
Now let's look at the cross-attention of the last decoder layer, after the softmax. Hover over the matrices to see the values.

Look at head ${c_dimRef(`${bestHeadIdx + 1}`, DimStyle.n_heads)}: output position 0 puts almost all of its weight on source position ${c_dimRef(`${T - 1}`, DimStyle.T)}, position 1
on source position ${c_dimRef(`${T - 2}`, DimStyle.T)}, and so on, forming a rough _anti-diagonal_. That's exactly what reversing needs: the letter to write at output
position ${c_dimRef('t', DimStyle.T)} is the source letter at position ${c_dimRef(`${T - 1} − t`, DimStyle.T)}, and attention fetches its value. Nobody told the model to do this;
it learned it from examples.

In translation, cross-attention similarly learns to _align_ each output word with the relevant source words, even when the word order differs
between the two languages.`;
    breakAfter();

    let t_camN = afterTime(null, 1.0, 0.3);
    let t_highlightN = afterTime(null, 2.0, 0.3);

    if (t_highlightInputs.active && t_camHeads.t < 1) {
        let v = Math.sin(t_highlightInputs.t * Math.PI) * 0.5;
        cross0.memCopy!.highlight = v;
        layout.enc.output.highlight = v;
        cross0.xq.highlight = v;
    }

    moveCameraTo(state, t_camHeads, camHeads.center, camHeads.angle);
    moveCameraTo(state, t_camMtx, camMtx0.center, camMtx0.angle);
    moveCameraTo(state, t_camN, camMtxN.center, camMtxN.angle);

    let attnInfo = startProcessBefore(state, cross0.heads[0].qBlock);
    let headBlocks = cross0.heads.flatMap(h => h.cubes).filter(b => b.t !== 'w');
    if (t_camHeads.active && t_processAll.t < 1) {
        for (let blk of [...headBlocks, cross0.attnOut, cross0.attnResidual]) {
            blk.access!.disable = true;
        }
    }
    if (t_processQkv.active && t_processAll.t === 0) {
        for (let h of cross0.heads) {
            let info = startProcessBefore(state, h.qBlock);
            processUpTo(state, t_processQkv, h.vBlock, info);
        }
    }
    let rowIdx = 1;
    if (t_row.active && t_processAll.t === 0) {
        processDim(state, head0.attnMtx, Dim.Y, rowIdx, t_row.t, { pinIdx: new Vec3(rowIdx, 0, 0), clamp: true, hidePopup: t_rowSm.t > 0 });
    }
    if (t_rowSm.active && t_processAll.t === 0) {
        processDim(state, head0.attnMtxAgg2, Dim.Y, rowIdx, 1, { clamp: true, hidePopup: true });
        processDim(state, head0.attnMtxAgg1, Dim.Y, rowIdx, 1, { clamp: true, hidePopup: true });
        processDim(state, head0.attnMtxSm, Dim.Y, rowIdx, t_rowSm.t, { pinIdx: new Vec3(rowIdx, 0, 0), clamp: true, hidePopup: t_rowSm.t >= 1 });
    }
    if (t_processAll.active) {
        processUpTo(state, t_processAll, cross0.attnResidual, attnInfo);
    }

    if (t_highlightN.active) {
        for (let h of crossN.heads) {
            h.attnMtxSm.highlight = h === bestHead ? lerp(0, 0.35, t_highlightN.t) : 0;
            h.headLabel.visible = 1.0;
        }
    }
}
