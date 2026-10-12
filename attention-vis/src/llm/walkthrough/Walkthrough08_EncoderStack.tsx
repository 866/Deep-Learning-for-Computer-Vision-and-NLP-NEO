import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { processUpTo, startProcessBefore } from "./Walkthrough00_Intro";
import { cameraFor, commentary, DimStyle, IWalkthroughArgs, moveCameraTo, setInitialCamera } from "./WalkthroughTools";

export function walkthrough08_EncoderStack(args: IWalkthroughArgs) {
    let { walkthrough: wt, state, layout, tools: { afterTime, breakAfter, c_blockRef, c_dimRef } } = args;

    if (wt.phase !== Phase.Enc_Stack) {
        return;
    }

    let enc = layout.enc;
    let { nBlocks } = layout.shape;
    let allEncCubes = enc.blocks.flatMap(b => b.cubes);
    let camStack = cameraFor([enc.residual0, ...allEncCubes], new Vec3(291, 14, 0), 1.0);
    let camLayer0 = cameraFor(enc.blocks[0].cubes, new Vec3(291, 14, 0), 1.0);
    let camLayerN = cameraFor(enc.blocks[nBlocks - 1].cubes, new Vec3(291, 14, 0), 1.0);
    let camMemory = cameraFor([enc.output, ...layout.dec.blocks.map(b => b.crossAttn.memCopy!)], new Vec3(270, 10, 0), 1.0);

    setInitialCamera(state, camStack.center, camStack.angle);

    for (let b of enc.blocks) {
        b.blockLabel.visible = 1.0;
    }

    commentary(wt, null, 0)`
And that's a complete ${c_blockRef('encoder layer', enc.blocks[0].cubes)}: self-attention followed by a feed-forward network, each wrapped in a
residual connection and layer normalization.

The encoder is a stack of ${c_dimRef(`N = ${nBlocks}`, DimStyle.n_layers)} of these layers (6 in the paper). They all have the same structure, but each has its own weights.
The output of one layer, a ${c_dimRef('T', DimStyle.T)} × ${c_dimRef('C', DimStyle.C)} matrix just like its input, is the input to the next, continuing the residual pathway.
Let's run the whole encoder:`;
    breakAfter();

    let t_cam0 = afterTime(null, 1.0, 0.3);
    let t_layer0 = afterTime(null, 5.0, 0.3);
    let t_camN = afterTime(null, 1.0, 0.3);
    let t_layerN = afterTime(null, 5.0, 0.3);

    breakAfter();

    commentary(wt)`
As is common in deep learning, it's hard to say exactly what each of these layers is doing. Here we can make a reasonable guess:
to reverse the sequence, the decoder will need to know, for each source letter, both _what_ it is and _where_ it is. The encoder layers
mix the token and position information together (via attention), so that later stages can look letters up by their position.

The ${c_blockRef('output of the last encoder layer', enc.output)} is the encoder's final result, sometimes called the _memory_. It's computed once,
and then shared with _every_ layer of the decoder, via the arrows leading across to the right.`;
    breakAfter();

    let t_camMem = afterTime(null, 1.0, 0.3);

    moveCameraTo(state, t_cam0, camLayer0.center, camLayer0.angle);
    moveCameraTo(state, t_camN, camLayerN.center, camLayerN.angle);
    moveCameraTo(state, t_camMem, camMemory.center, camMemory.angle);

    let info = startProcessBefore(state, enc.blocks[0].attn.heads[0].qBlock);
    for (let blk of allEncCubes) {
        if (t_cam0.t > 0 && t_layerN.t < 1 && blk.access && blk.t !== 'w') {
            blk.access.disable = true;
        }
    }
    if (t_layer0.active) {
        processUpTo(state, t_layer0, enc.blocks[0].output, info);
    }
    if (t_layerN.active) {
        processUpTo(state, t_layerN, enc.output, info);
    }

    if (t_camMem.active) {
        enc.output.highlight = 0.4 * t_camMem.t;
        for (let b of layout.dec.blocks) {
            b.crossAttn.memCopy!.highlight = 0.4 * t_camMem.t;
        }
    }
}
