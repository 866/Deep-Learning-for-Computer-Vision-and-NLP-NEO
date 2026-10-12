import React from "react";
import { Dim, Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { processUpTo, revealBlockAlongX, revealSequence, startProcessBefore } from "./Walkthrough00_Intro";
import { processDim } from "./Walkthrough03_SelfAttention";
import { cameraFor, commentary, DimStyle, dimStyleColor, IWalkthroughArgs, moveCameraTo, setInitialCamera } from "./WalkthroughTools";
import { embedInline } from "./Walkthrough01_Prelim";
import { clamp } from "@/src/utils/data";
import { splitGrid, findSubBlocks } from "../Annotations";

export function walkthrough09_MaskedAttention(args: IWalkthroughArgs) {
    let { walkthrough: wt, state, layout, tools: { afterTime, breakAfter, c_blockRef, c_dimRef, c_str } } = args;

    if (wt.phase !== Phase.Dec_MaskedAttention) {
        return;
    }

    let { T } = layout.shape;
    let dec = layout.dec;
    let attn = dec.blocks[0].selfAttn;
    let head = attn.heads[0];
    let rowIdx = 2;

    let camEmbed = cameraFor([dec.idxObj, dec.tokEmbedObj, dec.posEmbedObj, dec.residual0], new Vec3(260, 12, 0), 0.9);
    let camAttn = cameraFor([attn.xq, ...attn.cubes], new Vec3(250, 22, 0), 0.9);
    let camMtx = cameraFor([head.attnMtx, head.attnMtxSm, head.qBlock], new Vec3(268, -25, 0), 0.8);

    setInitialCamera(state, camEmbed.center, camEmbed.angle);
    wt.dimHighlightBlocks = [dec.residual0, ...attn.heads.flatMap(h => h.cubes)];

    commentary(wt, null, 0)`
Now let's turn to the right half of the model: the _decoder_. Its job is to produce the output sequence, one token at a time.

Its input is the output sequence _shifted right_: the start token ${c_str('<s>', 0, DimStyle.Token)}, followed by the output tokens produced so far.
At position ${c_dimRef('t', DimStyle.T)}, the decoder's job is to predict output token ${c_dimRef('t', DimStyle.T)}, having seen the tokens before it.

During training, we know the whole correct answer, so we feed it all in at once (this is called _teacher forcing_). Here we show the
completed answer, which is also what the decoder sees at the final step of generation. The ${c_blockRef('target tokens', dec.idxObj, DimStyle.TokenIdx)} are embedded exactly like
the source tokens: the same ${c_blockRef('token embedding matrix', dec.tokEmbedObj)} (the paper shares it between the encoder, the decoder and the output layer),
scaled by √C, plus the same ${c_blockRef('positional encoding', dec.posEmbedObj)}.`;
    breakAfter();

    let t_tokens = afterTime(null, 1.0, 0.2);
    let t_embed = afterTime(null, 2.0, 0.3);

    breakAfter();

    commentary(wt)`
The first sublayer of each decoder layer is _masked_ self-attention. It works exactly like the encoder's self-attention, with one difference.

When we generate the output, we produce it one token at a time, so at position ${c_dimRef('t', DimStyle.T)} the later tokens don't exist yet. To make
training match this, each position is only allowed to attend to itself and the positions _before_ it. The decoder can't "see into the future".

This is done by _masking_ the attention matrix: before the softmax, every score above the diagonal is set to −∞, so after the softmax its weight is exactly 0.`;
    breakAfter();

    let t_camAttn = afterTime(null, 1.0, 0.3);
    let t_processQkv = afterTime(null, 3.0, 0.3);
    let t_camMtx = afterTime(null, 1.0, 0.3);
    let t_row = afterTime(null, 2.0, 0.3);
    let t_rowSm = afterTime(null, 1.5, 0.3);

    breakAfter();

    commentary(wt)`
For example, for row ${c_dimRef(`t = ${rowIdx}`, DimStyle.T)} of the ${c_blockRef('attention matrix', head.attnMtx)}, the query is only compared with the keys at positions 0 to ${rowIdx}.
The ${c_blockRef('softmax', head.attnMtxSm)} then only spreads its weight over those positions.

So the attention matrices in the decoder's self-attention are _lower triangular_: the empty cells above the diagonal are the masked ones. In GPT, which is just a decoder
without the cross-attention, every attention layer works this way.`;
    breakAfter();

    let t_processAll = afterTime(null, 5.0, 0.3);

    if (t_tokens.active) {
        state.display.decTokenIdxColors = { mixes: revealSequence(T, t_tokens.t), color2: dimStyleColor(DimStyle.TokenIdx) };
    }
    let embedInfo = startProcessBefore(state, dec.residual0);
    if (t_embed.active) {
        for (let blk of [dec.residual0]) {
            blk.access!.disable = true;
        }
        processUpTo(state, t_embed, dec.residual0, embedInfo);
    }

    moveCameraTo(state, t_camAttn, camAttn.center, camAttn.angle);
    moveCameraTo(state, t_camMtx, camMtx.center, camMtx.angle);

    let attnInfo = startProcessBefore(state, attn.heads[0].qBlock);
    let headBlocks = attn.heads.flatMap(h => h.cubes).filter(b => b.t !== 'w');
    if (t_camAttn.active && t_processAll.t < 1) {
        for (let blk of [...headBlocks, attn.attnOut, attn.attnResidual]) {
            blk.access!.disable = true;
        }
    }
    if (t_processQkv.active && t_processAll.t === 0) {
        for (let h of attn.heads) {
            let info = startProcessBefore(state, h.qBlock);
            processUpTo(state, t_processQkv, h.vBlock, info);
        }
    }
    if (t_camMtx.active && t_processAll.t === 0) {
        // show the earlier rows of the matrices as already computed, and our row being computed
        for (let blk of [head.attnMtx, head.attnMtxSm]) {
            let rows = splitGrid(layout, blk, Dim.Y, rowIdx - 0.5, 0);
            for (let sub of findSubBlocks(blk, Dim.Y, null, rowIdx - 1)) {
                sub.access!.disable = false;
            }
        }
    }
    if (t_row.active && t_processAll.t === 0) {
        processDim(state, head.attnMtx, Dim.Y, rowIdx, t_row.t, { pinIdx: new Vec3(rowIdx, 0, 0), clamp: true, maxIdx: rowIdx + 1, hidePopup: t_rowSm.t > 0 });
    }
    if (t_rowSm.active && t_processAll.t === 0) {
        processDim(state, head.attnMtxAgg2, Dim.Y, rowIdx, clamp(t_rowSm.t * 3, 0, 1), { clamp: true, hidePopup: true });
        processDim(state, head.attnMtxAgg1, Dim.Y, rowIdx, clamp(t_rowSm.t * 3 - 1, 0, 1), { clamp: true, hidePopup: true });
        processDim(state, head.attnMtxSm, Dim.Y, rowIdx, clamp(t_rowSm.t * 3 - 2, 0, 1), { pinIdx: new Vec3(rowIdx, 0, 0), clamp: true, maxIdx: rowIdx + 1, hidePopup: t_rowSm.t >= 1 });
    }
    if (t_processAll.active) {
        moveCameraTo(state, t_processAll, camAttn.center, camAttn.angle);
        processUpTo(state, t_processAll, attn.attnResidual, attnInfo);
    }
}
