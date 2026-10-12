import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { processDim } from "./Walkthrough03_SelfAttention";
import { cameraFor, commentary, DimStyle, IWalkthroughArgs, moveCameraTo, setInitialCamera } from "./WalkthroughTools";
import { Dim } from "@/src/utils/vector";
import { processUpTo, startProcessBefore } from "./Walkthrough00_Intro";

export function walkthrough04_Softmax(args: IWalkthroughArgs) {
    let { walkthrough: wt, state, layout, tools: { afterTime, breakAfter, c_blockRef, c_dimRef } } = args;

    if (wt.phase !== Phase.Enc_Softmax) {
        return;
    }

    let head = layout.enc.blocks[0].attn.heads[0];
    let rowIdx = 1;

    let camAttn = cameraFor([head.attnMtx, head.attnMtxAgg1, head.attnMtxSm], new Vec3(270, 6, 0), 0.5);
    let camOut = cameraFor([layout.logits, layout.logitsSoftmax], new Vec3(270, 6, 0), 0.6);
    setInitialCamera(state, camAttn.center, camAttn.angle);
    wt.dimHighlightBlocks = [head.attnMtx, head.attnMtxAgg1, head.attnMtxAgg2, head.attnMtxSm];

    commentary(wt, null, 0)`

The softmax operation is used as part of self-attention, as seen in the previous section, and it
will also appear at the very end of the model.

Its goal is to take a vector and normalize its values so that they sum to 1.0. However, it's not as
simple as dividing by the sum. Instead, each input value is first exponentiated.

  a = exp(x_1)

This has the effect of making all values positive. Once we have a vector of our exponentiated
values, we can then divide each value by the sum of all the values. This will ensure that the sum
of the values is 1.0. Since all the exponentiated values are positive, we know that the resulting
values will be between 0.0 and 1.0, which provides a probability distribution over the original values.

That's it for softmax: simply exponentiate the values and then divide by the sum.

However, there's a slight complication. If any of the input values are quite large, then the
exponentiated values will be very large. We'll end up dividing a large number by a very large number,
and this can cause issues with floating-point arithmetic.

One useful property of the softmax operation is that if we add a constant to all the input values,
the result will be the same. So we can find the largest value in the input vector and subtract it
from all the values. This ensures that the largest value is 0.0, and the softmax remains numerically
stable.

Let's take a look at the softmax operation in the context of the self-attention layer. Our input
vector for each softmax operation is a row of the ${c_blockRef('attention matrix', head.attnMtx)} (in the encoder, the whole row).

Like with layer normalization, we have an intermediate step where we store some aggregation values
to keep the process efficient.

For each row, we store the ${c_blockRef('max value', head.attnMtxAgg2)} in the row and the ${c_blockRef('sum of the shifted & exponentiated values', head.attnMtxAgg1)}.
Then, to produce the corresponding ${c_blockRef('output row', head.attnMtxSm)}, we can perform a small set of operations: subtract the
max, exponentiate, and divide by the sum.`;
    breakAfter();

    let t_max = afterTime(null, 1.0, 0.3);
    let t_sum = afterTime(null, 1.0, 0.3);
    let t_row = afterTime(null, 2.0, 0.5);

    breakAfter();

    commentary(wt)`
What's with the name "softmax"? The "hard" version of this operation, called argmax, simply finds
the maximum value, sets it to 1.0, and assigns 0.0 to all other values. In contrast, the softmax
operation serves as a "softer" version of that. Due to the exponentiation involved in softmax, the
largest value is emphasized and pushed towards 1.0, while still maintaining a probability distribution
over all input values. This allows for a more nuanced representation that captures not only the most
likely option but also the relative likelihood of other options.

At the end of the model, a softmax turns the decoder's scores for each token of the vocabulary (${c_dimRef('n_vocab', DimStyle.n_vocab)} of them) into the
${c_blockRef('output probabilities', layout.logitsSoftmax)}.`;
    breakAfter();

    let t_camOut = afterTime(null, 1.0, 0.3);
    let t_processOut = afterTime(null, 3.0, 0.3);

    for (let blk of [head.attnMtxAgg1, head.attnMtxAgg2, head.attnMtxSm]) {
        if (t_max.t > 0 && t_camOut.t === 0 && blk.access) {
            blk.access.disable = true;
        }
    }

    if (t_max.t > 0 && t_camOut.t === 0) {
        let hidePopup = t_sum.t > 0;
        processDim(state, head.attnMtxAgg2, Dim.Y, rowIdx, t_max.t, { pinIdx: new Vec3(5, 0, 0), clamp: true, hidePopup });
    }
    if (t_sum.t > 0 && t_camOut.t === 0) {
        let hidePopup = t_row.t > 0;
        processDim(state, head.attnMtxAgg1, Dim.Y, rowIdx, t_sum.t, { pinIdx: new Vec3(-12, 0, 0), clamp: true, hidePopup });
    }
    if (t_row.t > 0 && t_camOut.t === 0) {
        processDim(state, head.attnMtxSm, Dim.Y, rowIdx, t_row.t, { pinIdx: new Vec3(5, 0, 0), clamp: true, hidePopup: t_row.t >= 1 });
    }

    moveCameraTo(state, t_camOut, camOut.center, camOut.angle);
    if (t_processOut.active) {
        let info = startProcessBefore(state, layout.logitsAgg2);
        processUpTo(state, t_processOut, layout.logitsSoftmax, info);
    }
}
