import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { processUpTo, startProcessBefore } from "./Walkthrough00_Intro";
import { cameraFor, commentary, DimStyle, IWalkthroughArgs, moveCameraTo, setInitialCamera } from "./WalkthroughTools";

export function walkthrough11_DecoderStack(args: IWalkthroughArgs) {
    let { walkthrough: wt, state, layout, tools: { afterTime, breakAfter, c_blockRef, c_dimRef } } = args;

    if (wt.phase !== Phase.Dec_Stack) {
        return;
    }

    let dec = layout.dec;
    let { nBlocks } = layout.shape;
    let allDecCubes = dec.blocks.flatMap(b => b.cubes);
    let camStack = cameraFor([dec.residual0, ...allDecCubes], new Vec3(252, 14, 0), 1.0);
    let camLayer0 = cameraFor(dec.blocks[0].cubes, new Vec3(252, 14, 0), 1.0);
    let camLayerN = cameraFor(dec.blocks[nBlocks - 1].cubes, new Vec3(252, 14, 0), 1.0);

    setInitialCamera(state, camStack.center, camStack.angle);

    for (let b of dec.blocks) {
        b.blockLabel.visible = 1.0;
    }

    let b0 = dec.blocks[0];

    commentary(wt, null, 0)`
We've now seen all the pieces of a ${c_blockRef('decoder layer', b0.cubes)}. It has three sublayers, each followed by Add & Norm:

1. ${c_blockRef('Masked self-attention', b0.selfAttn.cubes)}, where each output position looks at the earlier output positions.

2. ${c_blockRef('Cross-attention', b0.crossAttn.cubes)}, where each output position looks at the encoder output.

3. A ${c_blockRef('feed-forward network', b0.ffn.cubes)}, identical in form to the encoder's.

Like the encoder, the decoder is a stack of ${c_dimRef(`N = ${nBlocks}`, DimStyle.n_layers)} such layers, each with its own weights, feeding into each other.
Every one of them gets the same encoder output for its cross-attention. Let's run the whole decoder:`;
    breakAfter();

    let t_cam0 = afterTime(null, 1.0, 0.3);
    let t_layer0 = afterTime(null, 6.0, 0.3);
    let t_camN = afterTime(null, 1.0, 0.3);
    let t_layerN = afterTime(null, 6.0, 0.3);

    breakAfter();

    commentary(wt)`
The ${c_blockRef('output of the last decoder layer', dec.output)} has one column per output position, each a vector of length ${c_dimRef('C', DimStyle.C)}. All that's
left is to turn each of these vectors into a prediction of the next token.`;

    moveCameraTo(state, t_cam0, camLayer0.center, camLayer0.angle);
    moveCameraTo(state, t_camN, camLayerN.center, camLayerN.angle);

    let info = startProcessBefore(state, b0.selfAttn.heads[0].qBlock);
    for (let blk of allDecCubes) {
        if (t_cam0.t > 0 && t_layerN.t < 1 && blk.access && blk.t !== 'w' && !dec.blocks.some(b => b.crossAttn.memCopy === blk)) {
            blk.access.disable = true;
        }
    }
    if (t_layer0.active) {
        processUpTo(state, t_layer0, dec.blocks[0].output, info);
    }
    if (t_layerN.active) {
        processUpTo(state, t_layerN, dec.output, info);
    }
}
