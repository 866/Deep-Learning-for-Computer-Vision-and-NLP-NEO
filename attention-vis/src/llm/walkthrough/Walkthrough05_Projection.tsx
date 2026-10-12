import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { cameraFor, commentary, DimStyle, IWalkthroughArgs, moveCameraTo, setInitialCamera } from "./WalkthroughTools";
import { embedInline } from "./Walkthrough01_Prelim";
import React from "react";
import { lerp, lerpSmoothstep } from "@/src/utils/math";
import { processUpTo, startProcessBefore } from "./Walkthrough00_Intro";

export function walkthrough05_Projection(args: IWalkthroughArgs) {
    let { walkthrough: wt, state, layout, tools: { breakAfter, afterTime, c_blockRef, c_dimRef, cleanup } } = args;

    if (wt.phase !== Phase.Enc_Projection) {
        return;
    }

    let block = layout.enc.blocks[0].attn;
    let { C, A, nHeads } = layout.shape;
    let exIdx = 3;

    let camStart = cameraFor([...block.heads.map(h => h.vOutBlock), block.projWeight, block.attnOut], new Vec3(293, 4, 0), 1.0);
    let camResid = cameraFor([block.xq, block.attnOut, block.attnResidual, block.projWeight], new Vec3(290, 4, 0), 0.8);
    setInitialCamera(state, camStart.center, camStart.angle);
    wt.dimHighlightBlocks = [...block.heads.map(h => h.vOutBlock), block.projWeight, block.attnOut];

    let outBlocks = block.heads.map(h => h.vOutBlock);

    commentary(wt, null, 0)`

After the self-attention process, we have outputs from each of the heads. These outputs are the
appropriately mixed V vectors, influenced by the Q and K vectors.

Why several heads? Each head has its own Q, K and V weights, so each can attend to different things: one might look for
copies of the same letter, another for the neighbouring position. The paper found that several smaller heads work better than a single
big one.

To combine the ${c_blockRef('output vectors', outBlocks)} from each head, we simply stack them on top of each other (the paper's _Concat_). So, for time
${c_dimRef(`t = ${exIdx}`, DimStyle.T)}, we go from ${c_dimRef(`${nHeads}`, DimStyle.n_heads)} vectors of length ${c_dimRef(`A = ${A}`, DimStyle.A)} to 1 vector of length ${c_dimRef(`C = ${C}`, DimStyle.C)}.`;

    breakAfter();

    let t_fadeOut = afterTime(null, 1.0, 0.5);
    // let t_zoomToStack = afterTime(null, 1.0);
    let t_stack = afterTime(null, 1.0);

    breakAfter();

    commentary(wt)`

It's worth noting that the length of the vectors within a head (${c_dimRef(`A = ${A}`, DimStyle.A)}) is equal to ${c_dimRef('C', DimStyle.C)} / h.
This ensures that when we stack them back together, we get the original length, ${c_dimRef('C', DimStyle.C)}, and that multi-head attention
costs about the same as a single head of full size.

From here, we perform the projection to get the output of the layer. This is a simple matrix-vector
multiplication on a per-column basis with the ${c_blockRef('output weights', block.projWeight)} (W^O in the paper).

${embedInline(<div className="text-center my-2 italic">MultiHead(Q, K, V) = Concat(head<sub>1</sub>, ..., head<sub>h</sub>) W<sup>O</sup></div>)}`;

    breakAfter();

    let t_process = afterTime(null, 3.0);

    breakAfter();

    commentary(wt)`

Now we have the ${c_blockRef('output of the self-attention layer', block.attnOut)}. Instead of passing this output directly to the
next phase, we add it element-wise to the ${c_blockRef('input', block.xq)} of the layer. This process, denoted by the green
vertical arrow, is called the _residual connection_ or _residual pathway_.
`;

    breakAfter();

    let t_zoomOut = afterTime(null, 1.0, 0.5);
    let t_processResid = afterTime(null, 3.0);

    cleanup(t_zoomOut, [t_fadeOut, t_stack]);

    breakAfter();

    commentary(wt)`

The residual pathway is important for enabling effective learning in deep neural networks: each sublayer only
has to learn a _change_ to its input, and gradients can flow straight through the additions during training.

The ${c_blockRef('result', block.attnResidual)} then goes through a _layer normalization_, which we'll look at next. The paper describes each
sublayer's output as LayerNorm(x + Sublayer(x)), and the diagram labels this step "Add & Norm".
`;

    breakAfter();

    if (t_fadeOut.active) {
        for (let head of block.heads) {
            for (let blk of head.cubes) {
                if (blk !== head.vOutBlock) {
                    blk.opacity = lerpSmoothstep(1, 0, t_fadeOut.t);
                }
            }
        }
    }

    if (t_stack.active) {
        let targetZ = block.attnOut.z;
        for (let headIdx = 0; headIdx < block.heads.length; headIdx++) {
            let head = block.heads[headIdx];
            let targetY = head.vOutBlock.y + head.vOutBlock.dy * (headIdx - block.heads.length + 1);
            head.vOutBlock.y = lerp(head.vOutBlock.y, targetY, t_stack.t);
            head.vOutBlock.z = lerp(head.vOutBlock.z, targetZ, t_stack.t);
        }
    }

    let processInfo = startProcessBefore(state, block.attnOut);

    if (t_process.active) {
        processUpTo(state, t_process, block.attnOut, processInfo);
    }

    moveCameraTo(state, t_zoomOut, camResid.center, camResid.angle);

    if (t_processResid.active) {
        processUpTo(state, t_processResid, block.attnResidual, processInfo);
    }
}
