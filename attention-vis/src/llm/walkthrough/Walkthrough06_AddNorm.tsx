import React from "react";
import { findSubBlocks, splitGrid } from "../Annotations";
import { drawDataFlow } from "../components/DataFlow";
import { drawDependences } from "../Interaction";
import { clamp } from "@/src/utils/data";
import { lerp } from "@/src/utils/math";
import { Dim, Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { processUpTo, startProcessBefore } from "./Walkthrough00_Intro";
import { cameraFor, commentary, DimStyle, IWalkthroughArgs, moveCameraTo, setInitialCamera } from "./WalkthroughTools";

export function walkthrough06_AddNorm(args: IWalkthroughArgs) {
    let { walkthrough: wt, layout, state, tools: { afterTime, c_str, c_blockRef, c_dimRef, breakAfter, cleanup } } = args;
    let { C } = layout.shape;

    if (wt.phase !== Phase.Enc_AddNorm) {
        return;
    }

    let ln = layout.enc.blocks[0].ln1;
    let inputBlock = ln.residSrc;
    let attn = layout.enc.blocks[0].attn;

    let camStart = cameraFor([attn.xq, attn.attnResidual, ...ln.cubes], new Vec3(290, 8, 0), 1.0);
    let camLn = cameraFor([inputBlock, ...ln.cubes], new Vec3(282, 8, 0), 0.9);
    let camClose = cameraFor([inputBlock, ln.lnResid], new Vec3(272, 4, 0), 0.75);
    setInitialCamera(state, camStart.center, camStart.angle);
    wt.dimHighlightBlocks = [inputBlock, ...ln.cubes];

    commentary(wt, null, 0)`

After the residual connection, the ${c_blockRef('_sum_', inputBlock)} of the self-attention output and its input goes through _layer normalization_.
Together these two steps are the "Add & Norm" boxes in the paper's diagram, and they follow every sublayer of the Transformer.

Layer normalization is an operation that normalizes the values in each column of the matrix separately.`;
    breakAfter();

    let t_moveCamera = afterTime(null, 1.0);
    let t_hideExtra = afterTime(null, 1.0, 1.0);
    let t_moveCameraClose = afterTime(null, 0.5);

    breakAfter();
    commentary(wt)`
Normalization is an important step in the training of deep neural networks, and it helps improve the
stability of the model during training.

Without it, the values in the residual pathway could keep growing (or shrinking) as each sublayer adds its result. We can regard each
column separately, so let's focus on the 4th column (${c_dimRef('t = 3', DimStyle.T)}) for now.`;

    breakAfter();
    let t_focusColumn = afterTime(null, 0.5);

    // mu ascii: \u03bc
    // sigma ascii: \u03c3
    breakAfter();
    commentary(wt)`
The goal is to make the average value in the column equal to 0 and the standard deviation equal to 1. To do this,
we find both of these quantities (${c_blockRef('mean (\u03bc)', ln.lnAgg1)} & ${c_blockRef('std dev (\u03c3)', ln.lnAgg2)}) for the column and then subtract the average and divide by the standard deviation.`;

    breakAfter();

    let t_calcMuAgg = afterTime(null, 0.5);
    let t_calcVarAgg = afterTime(null, 0.5);

    // 1e-5 as 1x10^-5, but with superscript: 1x10<sup>-5</sup>

    breakAfter();
    commentary(wt)`
The notation we use here is E[x] for the average and Var[x] for the variance (of the column of length ${c_dimRef('C', DimStyle.C)}). The
variance is simply the standard deviation squared. The epsilon term (ε = ${<>1&times;10<sup>-5</sup></>}) is there to prevent division by zero.

We compute and store these values in our aggregation layer since we're applying them to all values in the column.

Finally, once we have the normalized values, we multiply each element in the column by a learned
${c_blockRef('weight (\u03b3)', ln.lnSigma)} and then add a ${c_blockRef('bias (β)', ln.lnMu)} value, resulting in our ${c_blockRef('normalized values', ln.lnResid)}.`;

    breakAfter();

    let t_clean_aggs = afterTime(null, 0.2);
    cleanup(t_clean_aggs, [t_calcMuAgg, t_calcVarAgg]);
    let t_colSequence = afterTime(null, 2.0);

    breakAfter();
    commentary(wt)`
We run this normalization operation on each column of the ${c_blockRef('residual sum', inputBlock)}, and the result is
the ${c_blockRef('normalized matrix', ln.lnResid)}, which becomes the new residual pathway, ready to be passed into the feed-forward network.

A note on the placement: the original paper normalizes _after_ the residual addition ("post-norm"), as shown here. Most later models, including
GPT-2, moved the layer norm to the _start_ of each sublayer ("pre-norm"), which makes deep models easier to train.
`;

    breakAfter();
    let t_cleanupSplits = afterTime(null, 0.5);
    cleanup(t_cleanupSplits, [t_focusColumn]);
    if (t_cleanupSplits.t > 0) {
        t_colSequence.t = 0;
    }
    let t_runAggFull = afterTime(null, 2.0);
    let t_runNormFull = afterTime(null, 6.0);

    moveCameraTo(state, t_moveCamera, camLn.center, camLn.angle);

    let exampleIdx = 3;
    let ln1 = ln;

    inputBlock.highlight = lerp(0, 0.3, t_hideExtra.t) * (1 - t_moveCameraClose.t);

    let relevantBlocks = new Set([
        inputBlock,
        ...ln1.cubes,
    ]);

    for (let blk of layout.cubes) {
            if (!relevantBlocks.has(blk)) {
            blk.opacity = lerp(1.0, 0.0, t_hideExtra.t);
        }
    }

    for (let blk of relevantBlocks) {
        if (blk != inputBlock && blk.t !== 'w' && t_hideExtra.t > 0) {
            blk.access!.disable = true;
        }
    }

    moveCameraTo(state, t_moveCameraClose, camClose.center, camClose.angle);

    let splitAmt = lerp(0.0, 2.0, t_focusColumn.t);
    let splitPos = exampleIdx + 0.5;

    let otherColOpacity = lerp(1.0, 0.3, t_focusColumn.t);
    ln1.lnAgg1.opacity = otherColOpacity;
    ln1.lnAgg2.opacity = otherColOpacity;
    ln1.lnResid.opacity = otherColOpacity;
    inputBlock.opacity = otherColOpacity;

    if (t_focusColumn.t > 0) {
        let aggMuCol = splitGrid(layout, ln1.lnAgg1, Dim.X, splitPos, splitAmt)!;
        let aggVarCol = splitGrid(layout, ln1.lnAgg2, Dim.X, splitPos, splitAmt)!;
        let residCol = splitGrid(layout, ln1.lnResid, Dim.X, splitPos, splitAmt)!;
        let inputCol = splitGrid(layout, inputBlock, Dim.X, splitPos, splitAmt)!;
        aggMuCol.opacity = 1.0;
        aggVarCol.opacity = 1.0;
        residCol.opacity = 1.0;
        inputCol.opacity = 1.0;

        let aggDestIdx = new Vec3(exampleIdx, 0, 0);
        if (t_calcMuAgg.t > 0.0) {
            let pinIdx = new Vec3(0, 10, 0);
            drawDependences(state, ln1.lnAgg1, aggDestIdx);
            drawDataFlow(state, ln1.lnAgg1, aggDestIdx, pinIdx);
            aggMuCol.access!.disable = false;
            inputCol.highlight = 0.3;
        }

        if (t_calcVarAgg.t > 0.0) {
            let pinIdx = new Vec3(9, 9, 0);
            drawDependences(state, ln1.lnAgg2, aggDestIdx);
            drawDataFlow(state, ln1.lnAgg2, aggDestIdx, pinIdx);
            aggVarCol.access!.disable = false;
        }

        if (t_colSequence.t > 0.0) {
            aggMuCol.access!.disable = false;
            aggVarCol.access!.disable = false;

            let pinIdx = new Vec3(-10, 0, 0);

            let cPos = t_colSequence.t * C;
            let cIdx = clamp(Math.floor(cPos), 0, C - 1);
            let destIdx = new Vec3(exampleIdx, cIdx, 0);
            drawDependences(state, ln1.lnResid, destIdx);
            drawDataFlow(state, ln1.lnResid, destIdx, pinIdx);

            let targetCell = splitGrid(layout, residCol, Dim.Y, cIdx + 0.5, 0.0)!;
            targetCell.highlight = 0.3;

            findSubBlocks(residCol, Dim.Y, 0, cIdx).forEach((blk) => {
                blk.access!.disable = false;
            });
        }
    }

    if (t_runAggFull.t > 0.0) {
        try {
            let processInfo = startProcessBefore(state, ln1.lnAgg1);
            processUpTo(state, t_runAggFull, ln1.lnAgg2, processInfo);
            processUpTo(state, t_runNormFull, ln1.lnResid, processInfo);
        } catch (e) {
            console.log(e);
        }
    }
}
