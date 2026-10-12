import React, { useState } from "react";
import { Dim, Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { cameraFor, commentary, DimStyle, dimStyleColor, embed, hideFromBlock, IVitWalkthroughArgs, moveCameraToFor, setInitialCamera, setInitialCameraFor } from "./WalkthroughTools";
import { numberToCommaSep, processUpTo, revealBlockAlongX, revealSequence, startProcessBefore } from "./Walkthrough00_Intro";
import { processDim } from "./Walkthrough03_SelfAttention";
import { embedInline, embedLink } from "./Walkthrough01_Prelim";
import { useProgramState } from "../Sidebar";
import { splitGrid } from "../Annotations";
import { IVitModelLink, vitPrediction } from "../VitModel";
import { IBlkDef } from "../GptModelLayout";
import { makeArrayRange } from "@/src/utils/data";
import clsx from "clsx";
import { vitProbsChartRegion } from "../components/ModelCard";

/*
The Vision Transformer chapters, after "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
(Dosovitskiy et al., 2020). They run on the tiny ViT (state.vit), which replaces the tiny Transformer in the 3D view.
*/

const vitPaperUrl = 'https://arxiv.org/abs/2010.11929';

/** The input image drawn above the Patches block (see ModelCard.ts), as a pseudo-block for framing the camera. */
function imageRegion(layout: IVitWalkthroughArgs['layout']): IBlkDef {
    let p = layout.patches;
    let h = layout.margin * 2.5 + 34;
    return { x: p.x - 8, y: p.y - h, z: 0, dx: p.dx + 16, dy: h, dz: 0 } as IBlkDef;
}

/** Just the head and the class-probability chart. */
function headCloseBlocks(layout: IVitWalkthroughArgs['layout']) {
    let chart = { ...vitProbsChartRegion(layout), z: 0, dz: 0 } as unknown as IBlkDef;
    return [layout.headWeight, layout.headBias, layout.logits, layout.logitsSoftmax, chart];
}

function headBlocks(layout: IVitWalkthroughArgs['layout']) {
    // plus the probability chart drawn beside the class probabilities (see ModelCard.ts)
    let chart = { ...vitProbsChartRegion(layout), z: 0, dz: 0 } as unknown as IBlkDef;
    return [...layout.lnF.cubes, layout.headWeight, layout.headBias, layout.logits, layout.logitsSoftmax, chart];
}

function hideIntermediates(blocks: IBlkDef[]) {
    for (let cube of blocks) {
        if (cube.t === 'i' && cube.access) {
            cube.access.disable = true;
        }
    }
}

export function walkthroughVit01_Intro(args: IVitWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools: { breakAfter, afterTime, c_str, c_blockRef, c_dimRef } } = args;
    if (wt.phase !== Phase.Vit_Intro) {
        return;
    }
    let model = layout.model;
    let { T, C, nBlocks, nClasses } = layout.shape;
    let acc = model ? (model.config.test_accuracy * 100).toFixed(1) : '98.6';
    let nTest = model?.config.n_test ?? 360;

    let whole = cameraFor(layout.cubes, new Vec3(270, 14, 0), 1.1);
    setInitialCamera(state, whole.center, whole.angle);

    let c0 = commentary(wt, null, 0)`The Transformer was designed for language. In 2020, the paper
${embedLink('"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"', vitPaperUrl)} (Dosovitskiy et al.)
showed that the same architecture works for images, almost unchanged. It introduced the _Vision Transformer_, or _ViT_.

The idea is simple: cut the image into fixed-size _patches_, and treat each patch like a token (a "word") of a sentence.
The sequence of patches then goes through a standard Transformer _encoder_, the left half of the model we've been looking at.
There's no decoder: we only want a single answer per image, its class.

Here's a tiny ViT with ${embedInline(numberToCommaSep(layout.weightCount))} parameters. It classifies 8x8 grayscale images of handwritten digits,
like this one: ${embed(VitInputImage)}
into one of ${nClasses} classes, the digits 0 to 9. It gets ${acc}% of the ${nTest} test images (which it never saw during training) right.`;

    if (c0.t > 0) {
        hideIntermediates(layout.cubes);
    }

    let embedBlocks = [imageRegion(layout), layout.patches, layout.patchWeight, layout.patchBias, layout.tokens, layout.clsToken, layout.posEmbed, layout.residual0];

    let t_cam = afterTime(null, 1.5, 0.4);
    moveCameraToFor(state, t_cam, embedBlocks, new Vec3(278, 12, 0), 0.75);
    let t_reveal = afterTime(null, 1.5, 0.3);
    if (t_reveal.active) {
        state.display.tokenIdxColors = { mixes: revealSequence(T, t_reveal.t), color2: dimStyleColor(DimStyle.TokenIdx) };
        revealBlockAlongX(layout, layout.patches, t_reveal.t);
    }

    breakAfter();

    commentary(wt)`The image is cut into 2x2 patches, giving N = ${T - 1} patches. Each patch is flattened into a list of its
${c_dimRef('P² = 4', DimStyle.P)} pixel values, which makes one column of the ${c_blockRef('Patches', layout.patches)} block. (In the paper,
224x224 color images are cut into 16x16 patches: 196 patches of 16 x 16 x 3 = 768 values each. Hence the title.)

The first column is left empty. That position belongs to an extra, learnable ${c_blockRef('[class] token', layout.clsToken)}, so the sequence
has T = ${T} tokens.`;
    breakAfter();

    let t_embed = afterTime(null, 3.0, 0.5);
    let info = processUpTo(state, t_embed, layout.residual0);

    breakAfter();

    commentary(wt)`Each patch is projected to a vector of length ${c_dimRef('D = ' + C, DimStyle.C)} (this model's size), and _position embeddings_ are added:
the ${c_blockRef('embeddings', layout.residual0)} are ready. From here on, it's the Transformer encoder that we know: the ${T} vectors flow down
through ${nBlocks} layers of self-attention and MLPs.`;
    breakAfter();

    let blockCubes = layout.blocks.flatMap(b => b.cubes);
    let t_cam2 = afterTime(null, 1.0, 0.3);
    moveCameraToFor(state, t_cam2, [layout.residual0, ...blockCubes], new Vec3(285, 14, 0), 0.9);
    let t_layers = afterTime(null, 6.0, 0.5);
    processUpTo(state, t_layers, layout.output, info);

    breakAfter();

    commentary(wt)`At the end, only the vector at the [class] token's position is used. Through self-attention, it has gathered what's needed from all the patches.
A final layer norm and a linear layer (the ${c_blockRef('_head_', layout.headWeight)}) turn it into a score for each of the ${nClasses} digits,
and a softmax turns the scores into ${c_blockRef('probabilities', layout.logitsSoftmax)}.`;
    breakAfter();

    let t_cam3 = afterTime(null, 1.0, 0.3);
    moveCameraToFor(state, t_cam3, headCloseBlocks(layout), new Vec3(274, 6, 0), 1.3);
    let t_head = afterTime(null, 3.0, 0.5);
    processUpTo(state, t_head, layout.logitsSoftmax, info);

    breakAfter();

    commentary(wt)`Compared with the Transformer's encoder, there are only a few changes, and we'll go through each of them.

_The input_: image patches, multiplied by a matrix, rather than tokens looked up in an embedding table. The position embeddings are learned, not sinusoidal.

_The [class] token_: an extra token, whose output is the only one the classifier reads.

_Pre-norm_: the layer norms come _before_ each sublayer, not after the residual addition.

_Details_: the MLPs use the GELU activation, and the attention projections have biases.`;
    breakAfter();
}

export function walkthroughVit02_Embedding(args: IVitWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools: { breakAfter, afterTime, c_str, c_blockRef, c_dimRef } } = args;
    if (wt.phase !== Phase.Vit_Embedding) {
        return;
    }
    let { T, C, patchDim } = layout.shape;
    let { patches, patchWeight, patchBias, tokens, clsToken, posEmbed, residual0 } = layout;

    setInitialCameraFor(state, [imageRegion(layout), patches, patchWeight, patchBias, tokens, clsToken, posEmbed, residual0], new Vec3(280, 12, 0), 0.8);
    wt.dimHighlightBlocks = [patches, patchWeight, tokens, posEmbed, residual0];

    let c0 = commentary(wt, null, 0)`The paper's equation (1) describes how the input sequence z₀ is formed:
${embed(VitEq1)}
Let's go through it piece by piece. Each patch x_p is a column of the ${c_blockRef('Patches', patches)} block, of length ${c_dimRef('P² · C = ' + patchDim, DimStyle.P)}
(P = 2 is the patch size, and here there's C = 1 color channel; this C isn't the hidden size, which the paper calls D).`;

    if (c0.t > 0) {
        hideFromBlock(state.render, layout, tokens);
    }

    breakAfter();

    let targetIdx = 6;
    commentary(wt)`Each patch is multiplied by the ${c_blockRef('patch projection E', patchWeight)}, a ${c_dimRef(String(patchDim), DimStyle.P)} x ${c_dimRef(String(C), DimStyle.C)} matrix
(plus a ${c_blockRef('bias', patchBias)}), giving a vector of length ${c_dimRef('D = ' + C, DimStyle.C)}: the _patch embedding_.

This plays the part of the Transformer's token embedding. A word comes from a fixed vocabulary, so its vector can be looked up in a table, but a patch can be any
combination of pixel values, so ViT uses a linear layer instead. Here's patch ${targetIdx}:`;
    breakAfter();

    let t_proc = afterTime(null, 3.0, 0.3);
    let t_all = afterTime(null, 1.5, 0.3);

    if (t_all.t === 0 && t_proc.active) {
        processDim(state, tokens, Dim.X, targetIdx, t_proc.t, { pinIdx: new Vec3(targetIdx, 0, 0) });
    }
    if (t_all.active) {
        revealBlockAlongX(layout, tokens, t_all.t);
    }

    breakAfter();

    commentary(wt)`The first column of the ${c_blockRef('patch embeddings', tokens)} is different: it's a copy of the ${c_blockRef('[class] token', clsToken)}, x_class.
This is a learned vector, the same for every image, borrowed from BERT's [CLS] token. It doesn't correspond to any part of the image. Since attention lets every token
read from every other token, x_class gathers information from all the patches as it goes through the layers. Its final value z_L⁰ is the image representation the classifier uses.`;
    breakAfter();

    let t_cls = afterTime(null, 1.5, 0.5);
    if (t_cls.active && t_cls.t < 1) {
        let col = splitGrid(layout, tokens, Dim.X, 0.5, 0);
        if (col) {
            col.highlight = 0.6 * Math.sin(t_cls.t * Math.PI);
        }
        clsToken.highlight = 0.6 * Math.sin(t_cls.t * Math.PI);
    }

    breakAfter();

    commentary(wt)`Finally, the ${c_blockRef('position embeddings', posEmbed)} E_pos are added. Without them, the model couldn't tell where each patch came from:
self-attention treats its input as a set. Unlike the Transformer's fixed sinusoids, these are _learned_ parameters, one vector of length D for each of the ${T} positions.

They're plain 1D positions (0, 1, ..., ${T - 1}): the model isn't told that the patches form a 4x4 grid. The paper tried 2D-aware variants and found they made no
real difference, because the model learns the 2D structure by itself. Our tiny model did too. Here's how similar (cosine similarity) the position embedding of each patch
is to those of all the other patches, laid out like the image:
${embed(PosEmbedSimilarity)}
Each patch's embedding is most similar to its neighbours', especially along its row. (This is the paper's Figure 7, center, for our model.)`;
    breakAfter();

    let t_resid = afterTime(null, 3.0, 0.5);
    if (t_resid.active) {
        revealBlockAlongX(layout, tokens, 1.0);
        processUpTo(state, t_resid, residual0, startProcessBefore(state, residual0));
    }

    breakAfter();

    commentary(wt)`The result, ${c_blockRef('z₀', residual0)}, is the input to the encoder: ${T} vectors of length ${C}, exactly like the Transformer's input embeddings.`;
    breakAfter();
}

export function walkthroughVit03_Encoder(args: IVitWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools: { breakAfter, afterTime, c_str, c_blockRef, c_dimRef } } = args;
    if (wt.phase !== Phase.Vit_Encoder) {
        return;
    }
    let { nBlocks, dFF, C, nHeads } = layout.shape;
    let blk = layout.blocks[0];

    setInitialCameraFor(state, [layout.residual0, ...blk.ln1.cubes, ...blk.attn.cubes], new Vec3(285, 14, 0), 0.9);
    wt.dimHighlightBlocks = [blk.ln1.lnResid, blk.attn.attnResidual, blk.ffn.mlpFc, blk.ffn.mlpResidual];

    let c0 = commentary(wt, null, 0)`Each of the ${nBlocks} encoder layers works just like a layer of the Transformer's encoder: ${nHeads}-head self-attention, then an MLP
(the feed-forward network), each wrapped in a residual connection. In the paper's equations (2) and (3):
${embed(VitEq23)}
The difference is _where_ the layer norm (LN) goes. In the Transformer it comes after the residual addition, as "Add & Norm":
LayerNorm(x + Sublayer(x)). In ViT it's applied to the _input_ of each sublayer: x + Sublayer(LayerNorm(x)). This is called _pre-norm_.`;

    if (c0.t > 0) {
        hideFromBlock(state.render, layout, blk.ln1.lnAgg1);
    }
    breakAfter();

    commentary(wt)`So the ${c_blockRef('layer norm', blk.ln1.lnResid)} sits _beside_ the residual pathway, and only feeds the attention.
The residual pathway itself is never normalized: each sublayer's output is simply added to it. This makes deep stacks easier to train,
and has become the standard choice. ViT-Huge has 32 layers.`;
    breakAfter();

    let t_ln = afterTime(null, 2.5, 0.3);
    let info = processUpTo(state, t_ln, blk.ln1.lnResid, startProcessBefore(state, blk.ln1.lnAgg1));

    breakAfter();

    commentary(wt)`Then comes multi-head self-attention, as before. With images there's no masking: every patch can look at every other patch and at the [class]
token, however far apart they are in the image. The Q, K and V projections (and the output projection) also have a ${c_blockRef('bias', blk.attn.heads.map(h => h.qBiasBlock!).filter(b => b))}
here, which the Transformer paper's didn't.`;
    breakAfter();

    let t_attnCam = afterTime(null, 1.0, 0.3);
    moveCameraToFor(state, t_attnCam, blk.attn.cubes, new Vec3(290, 16, 0), 0.95);
    let t_attn = afterTime(null, 6.0, 0.5);
    processUpTo(state, t_attn, blk.attn.attnResidual, info);

    breakAfter();

    commentary(wt)`The second half is the MLP. A ${c_blockRef('layer norm', blk.ln2.lnResid)}, then a linear layer up to ${c_dimRef(String(dFF), DimStyle.C4)} (4 x D),
an activation, and a linear layer back down to ${c_dimRef(String(C), DimStyle.C)}, added to the residual pathway.

The activation is the GELU rather than the ReLU. It's like the ReLU, but with a smooth curve instead of a sharp corner (and a small dip below zero):
${embed(GeluGraph)}`;
    breakAfter();

    let t_mlpCam = afterTime(null, 1.0, 0.3);
    moveCameraToFor(state, t_mlpCam, [blk.attn.attnResidual, ...blk.ln2.cubes, ...blk.ffn.cubes], new Vec3(285, 14, 0), 0.9);
    let t_mlp = afterTime(null, 5.0, 0.5);
    processUpTo(state, t_mlp, blk.ffn.mlpResidual, info);

    breakAfter();

    commentary(wt)`That's one layer. The output has the same shape as the input, so the next layer is identical, with its own weights.
Our model has ${nBlocks} of them, the paper's models 12 to 32.`;
    breakAfter();

    let t_allCam = afterTime(null, 1.0, 0.3);
    moveCameraToFor(state, t_allCam, layout.blocks.flatMap(b => b.cubes), new Vec3(285, 14, 0), 0.9);
    let t_all = afterTime(null, 4.0, 0.5);
    processUpTo(state, t_all, layout.output, info);
    breakAfter();
}

export function walkthroughVit04_Head(args: IVitWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools: { breakAfter, afterTime, c_str, c_blockRef, c_dimRef } } = args;
    if (wt.phase !== Phase.Vit_Head) {
        return;
    }
    let { nClasses, C } = layout.shape;
    let lnF = layout.lnF;

    setInitialCameraFor(state, headBlocks(layout), new Vec3(276, 8, 0), 1.2);
    wt.dimHighlightBlocks = [lnF.lnResid, layout.headWeight, layout.logits, layout.logitsSoftmax];

    let c0 = commentary(wt, null, 0)`After the last layer, there's a ${c_blockRef('final layer norm', lnF.lnResid)} (needed with pre-norm, since the residual pathway
itself was never normalized). Then we take just the [class] token's vector, the first column. This is the paper's equation (4):
${embed(VitEq4)}`;
    if (c0.t > 0) {
        hideFromBlock(state.render, layout, lnF.lnAgg1);
    }
    breakAfter();

    let t_ln = afterTime(null, 2.0, 0.3);
    let info = processUpTo(state, t_ln, lnF.lnResid, startProcessBefore(state, lnF.lnAgg1));
    let t_col = afterTime(null, 1.0, 0.3);
    if (t_col.active) {
        let col = splitGrid(layout, lnF.lnResid, Dim.X, 0.5, 0);
        if (col) {
            col.highlight = 0.5;
        }
    }

    breakAfter();

    commentary(wt)`The classification ${c_blockRef('head', layout.headWeight)} is a single linear layer here: ${c_dimRef(String(C), DimStyle.C)} inputs, one output
(a ${c_blockRef('logit', layout.logits)}) per class. The paper uses an MLP with one hidden layer while pre-training, and a single linear layer when fine-tuning.
A softmax turns the logits into ${c_blockRef('probabilities', layout.logitsSoftmax)}, and the most likely class is the prediction.`;
    breakAfter();

    let t_headCam = afterTime(null, 1.0, 0.3);
    moveCameraToFor(state, t_headCam, headCloseBlocks(layout), new Vec3(274, 6, 0), 1.3);
    let t_head = afterTime(null, 3.0, 0.5);
    processUpTo(state, t_head, layout.logitsSoftmax, info);

    breakAfter();

    commentary(wt)`Try it: pick another test image, or draw your own digit (click and drag; right-click or shift to erase). The whole model runs again, so every value in the 3D view updates.
${embed(VitImagePicker)}`;
    breakAfter();
}

export function walkthroughVit05_Attention(args: IVitWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools: { breakAfter, afterTime, c_str, c_blockRef, c_dimRef } } = args;
    if (wt.phase !== Phase.Vit_Attention) {
        return;
    }

    let whole = cameraFor(layout.cubes, new Vec3(270, 14, 0), 1.1);
    setInitialCamera(state, whole.center, whole.angle);

    commentary(wt, null, 0)`What does the model look at? Each attention head computes, for each token, how much weight to give every token. The row of the
${c_blockRef('attention matrix', layout.blocks.flatMap(b => b.attn.heads.map(h => h.attnMtxSm)))} for the [class] token tells us how much each patch
contributes to it. Laid over the image, it becomes an _attention map_ (more orange = more attention):
${embed(AttentionMaps)}
The last map is the _attention rollout_ (Abnar & Zuidema, 2020), which the paper uses in its Figure 6. It follows the attention through all the layers:
the attention matrices are averaged over heads, the residual connection is accounted for by adding the identity, and the results are multiplied together.
The paper also measures how far each head looks across the image (its "attention distance"). Even in the first layer, some heads look across the whole image.
A CNN can only do that after many layers.`;
    breakAfter();

    commentary(wt)`So, does ViT beat convolutional networks (CNNs)? It depends on the amount of data. A CNN has strong assumptions about images built in
(_inductive biases_): nearby pixels matter together, and the same pattern means the same thing anywhere in the image. ViT has very little of this: only the patch
cutting, and whatever it learns about positions. With "only" ImageNet's 1.3M images, ViTs do worse than comparable ResNets.

But with enough data, learning those patterns beats building them in. Pre-trained on the 303M images of JFT-300M, ViT-H/14 reached 88.55% top-1 accuracy on ImageNet,
matching or beating the best CNNs while taking substantially less compute to pre-train (the paper reports 2.5k TPUv3-core-days).`;
    breakAfter();

    commentary(wt)`The paper's models (Table 1) reuse BERT's sizes:
${embed(VitSizesTable)}
You can see them at full size, as shapes without weights, with the buttons at the top left of the 3D view. Note how many more tokens they have: a 224x224 image in 16x16 patches is
196 patches, plus the [class] token. Since every token attends to every other, the cost of attention grows with the square of the number of patches,
so smaller patches get expensive quickly: ViT-H/14's 14x14 patches give 256 patches instead of 196.

That's the Vision Transformer: images go in as sequences of patches, and the Transformer encoder does the rest.`;
    breakAfter();
}

// ---------------------------------------------------------------------------------------------------------------
// Widgets

const VitEq1: React.FC = () => {
    return <div className="my-2 text-center font-serif text-[15px]">
        z<sub>0</sub> = [ x<sub>class</sub> ; x<sub>p</sub><sup>1</sup>E ; x<sub>p</sub><sup>2</sup>E ; … ; x<sub>p</sub><sup>N</sup>E ] + E<sub>pos</sub>,
        <div className="text-sm text-gray-600">E ∈ ℝ<sup>(P²·C)×D</sup>, &nbsp; E<sub>pos</sub> ∈ ℝ<sup>(N+1)×D</sup></div>
    </div>;
};

const VitEq23: React.FC = () => {
    return <div className="my-2 text-center font-serif text-[15px]">
        <div>z′<sub>ℓ</sub> = MSA(LN(z<sub>ℓ−1</sub>)) + z<sub>ℓ−1</sub></div>
        <div>z<sub>ℓ</sub> = MLP(LN(z′<sub>ℓ</sub>)) + z′<sub>ℓ</sub></div>
    </div>;
};

const VitEq4: React.FC = () => {
    return <div className="my-2 text-center font-serif text-[15px]">
        y = LN(z<sub>L</sub><sup>0</sup>)
    </div>;
};

function grayColor(v: number) {
    let c = Math.round(255 * (1 - Math.max(0, Math.min(1, v)) * 0.92));
    return `rgb(${c},${c},${c})`;
}

/** An image, with optional patch grid lines and an optional overlay (one value per patch, 0..1). */
const ImageSvg: React.FC<{ image: Float32Array, size: number, imageSize: number, patchSize: number, grid?: boolean, numbers?: boolean, overlay?: Float32Array | null, onClick?: () => void, className?: string }> = ({ image, size, imageSize, patchSize, grid, numbers, overlay, onClick, className }) => {
    let px = size / imageSize;
    let g = imageSize / patchSize;
    let cells: React.ReactNode[] = [];
    for (let y = 0; y < imageSize; y++) {
        for (let x = 0; x < imageSize; x++) {
            cells.push(<rect key={`${x},${y}`} x={x * px} y={y * px} width={px + 0.3} height={px + 0.3} fill={grayColor(image[y * imageSize + x])} />);
        }
    }
    let overlays: React.ReactNode[] = [];
    if (overlay) {
        let max = Math.max(...overlay, 1e-6);
        for (let n = 0; n < g * g; n++) {
            let a = overlay[n] / max;
            overlays.push(<rect key={'o' + n} x={(n % g) * patchSize * px} y={Math.floor(n / g) * patchSize * px} width={patchSize * px} height={patchSize * px}
                fill={`rgba(255, 140, 0, ${(0.85 * a).toFixed(3)})`} />);
        }
    }
    let lines: React.ReactNode[] = [];
    if (grid) {
        for (let i = 1; i < g; i++) {
            lines.push(<line key={'v' + i} x1={i * patchSize * px} x2={i * patchSize * px} y1={0} y2={size} stroke="#d33" strokeWidth={1} />);
            lines.push(<line key={'h' + i} y1={i * patchSize * px} y2={i * patchSize * px} x1={0} x2={size} stroke="#d33" strokeWidth={1} />);
        }
    }
    let nums: React.ReactNode[] = [];
    if (numbers) {
        for (let n = 0; n < g * g; n++) {
            nums.push(<text key={'n' + n} x={(n % g) * patchSize * px + 2} y={Math.floor(n / g) * patchSize * px + 9} fontSize={8} fill="#1b495d">{n + 1}</text>);
        }
    }
    return <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className={clsx("border border-gray-400", onClick && "cursor-pointer hover:outline hover:outline-2 hover:outline-blue-400", className)} onClick={onClick}>
        {cells}
        {overlays}
        {lines}
        {nums}
    </svg>;
};

const VitInputImage: React.FC = () => {
    let state = useProgramState();
    let model = state.vit.model;
    if (!model) {
        return null;
    }
    let { imageSize, patchSize } = model.shape;
    return <div className="flex justify-center items-center my-2 gap-4">
        <ImageSvg image={model.image} size={96} imageSize={imageSize} patchSize={patchSize} />
        <div className="text-gray-500">→</div>
        <ImageSvg image={model.image} size={96} imageSize={imageSize} patchSize={patchSize} grid numbers />
    </div>;
};

function cosineSimilarities(model: IVitModelLink) {
    let { T, C } = model.shape;
    let pe = model.posEmbed.localBuffer!;
    let n = T - 1;
    let norms = makeArrayRange(n, 1, n).map((_, i) => Math.sqrt(pe.subarray((i + 1) * C, (i + 2) * C).reduce((a, b) => a + b * b, 0)));
    let sim = new Float32Array(n * n);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            let d = 0;
            for (let c = 0; c < C; c++) {
                d += pe[(i + 1) * C + c] * pe[(j + 1) * C + c];
            }
            sim[i * n + j] = d / (norms[i] * norms[j]);
        }
    }
    return sim;
}

const PosEmbedSimilarity: React.FC = () => {
    let state = useProgramState();
    let model = state.vit.model;
    if (!model) {
        return null;
    }
    let { imageSize, patchSize } = model.shape;
    let g = imageSize / patchSize;
    let n = g * g;
    let sim = cosineSimilarities(model);
    let cellPx = 8;
    let tile = g * cellPx;
    let gap = 4;
    let total = g * tile + (g - 1) * gap;

    // stretch the color scale over the range of similarities between different patches (self-similarity is always 1)
    let offDiag = Array.from(sim).filter((_, k) => Math.floor(k / n) !== k % n);
    let lo = Math.min(...offDiag), hi = Math.max(...offDiag);

    function color(v: number) {
        // lo => white, hi (and above) => dark teal
        let t = Math.max(0, Math.min(1, (v - lo) / (hi - lo)));
        let r = Math.round(255 - t * (255 - 20));
        let gC = Math.round(255 - t * (255 - 110));
        let b = Math.round(255 - t * (255 - 140));
        return `rgb(${r},${gC},${b})`;
    }

    let rects: React.ReactNode[] = [];
    for (let i = 0; i < n; i++) {
        let ox = (i % g) * (tile + gap);
        let oy = Math.floor(i / g) * (tile + gap);
        for (let j = 0; j < n; j++) {
            rects.push(<rect key={`${i},${j}`} x={ox + (j % g) * cellPx} y={oy + Math.floor(j / g) * cellPx} width={cellPx} height={cellPx} fill={color(sim[i * n + j])} />);
        }
        rects.push(<rect key={`b${i}`} x={ox} y={oy} width={tile} height={tile} fill="none" stroke="#999" strokeWidth={0.5} />);
    }
    return <div className="flex flex-col items-center my-2">
        <svg width={total} height={total} viewBox={`0 0 ${total} ${total}`}>{rects}</svg>
        <div className="text-xs text-gray-500 mt-1 text-center">Tile (row r, column c): similarity of patch (r, c)'s position embedding<br />to every other patch's (dark = similar, light = dissimilar)</div>
    </div>;
};

/** The [class] token's attention to each patch, per layer & head; plus the attention rollout through all layers. */
function attentionMaps(model: IVitModelLink) {
    let { T, nHeads } = model.shape;
    let maps: { layer: number, head: number, map: Float32Array }[] = [];
    let rollout: Float32Array | null = null; // T x T

    for (let l = 0; l < model.blocks.length; l++) {
        let sm = model.blocks[l].attn.attnMatrixSoftmax.localBuffer;
        if (!sm) {
            return null;
        }
        let avg = new Float32Array(T * T);
        for (let h = 0; h < nHeads; h++) {
            let map = new Float32Array(T - 1);
            for (let k = 1; k < T; k++) {
                map[k - 1] = sm[(h * T + 0) * T + k];
            }
            maps.push({ layer: l, head: h, map });
            for (let i = 0; i < T * T; i++) {
                avg[i] += sm[h * T * T + i] / nHeads;
            }
        }
        // A' = 0.5 A + 0.5 I (residual), rows renormalized; rollout = A'_l ... A'_1
        for (let i = 0; i < T; i++) {
            avg[i * T + i] += 1;
            for (let j = 0; j < T; j++) avg[i * T + j] *= 0.5;
        }
        if (!rollout) {
            rollout = avg;
        } else {
            let next = new Float32Array(T * T);
            for (let i = 0; i < T; i++) for (let k = 0; k < T; k++) {
                let a = avg[i * T + k];
                for (let j = 0; j < T; j++) next[i * T + j] += a * rollout[k * T + j];
            }
            rollout = next;
        }
    }
    let rolloutMap = new Float32Array(T - 1);
    for (let k = 1; k < T; k++) rolloutMap[k - 1] = rollout![0 * T + k];
    return { maps, rolloutMap };
}

const AttentionMaps: React.FC = () => {
    let state = useProgramState();
    let model = state.vit.model;
    if (!model) {
        return null;
    }
    let res = attentionMaps(model);
    if (!res) {
        return null;
    }
    let { imageSize, patchSize, nHeads } = model.shape;
    let size = 64;
    return <div className="my-2 flex flex-col items-center text-xs text-gray-600">
        {model.blocks.map((_, l) => <div key={l} className="flex items-center gap-2 my-1">
            <div className="w-14 text-right">Layer {l + 1}</div>
            {res!.maps.filter(m => m.layer === l).map(m => <div key={m.head} className="flex flex-col items-center">
                <ImageSvg image={model!.image} size={size} imageSize={imageSize} patchSize={patchSize} overlay={m.map} />
                <div>head {m.head + 1}</div>
            </div>)}
        </div>)}
        <div className="flex items-center gap-2 my-1">
            <div className="w-14 text-right">Rollout</div>
            <ImageSvg image={model.image} size={size} imageSize={imageSize} patchSize={patchSize} overlay={res.rolloutMap} />
            <div className="w-[136px] text-left">all {nHeads * model.blocks.length} heads, through all layers</div>
        </div>
    </div>;
};

const GeluGraph: React.FC = () => {
    let fnRelu = (x: number) => Math.max(0, x);
    let fnGelu = (x: number) => x * 0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
    let w = 200;
    let h = 140;
    let halfW = 3.2;
    let halfH = halfW * h / w;
    let hOffset = 1.1;
    let xScale = (x: number) => (x + halfW) / (2 * halfW) * w;
    let yScale = (y: number) => h - (y - (-halfH + hOffset)) / (2 * halfH) * h;
    let xPts = makeArrayRange(100, -halfW, halfW);
    let makePath = (fn: (x: number) => number) => xPts.map((x, i) => `${i ? 'L' : 'M'}${xScale(x)},${yScale(fn(x))}`).join(' ');

    return <div className="flex justify-center my-2">
        <svg viewBox={`0 0 ${w} ${h}`} width={w} height={h} className="bg-slate-200 rounded">
            <line x1={0} x2={w} y1={yScale(0)} y2={yScale(0)} stroke="gray" strokeWidth={1} />
            <line x1={xScale(0)} x2={xScale(0)} y1={0} y2={h} stroke="gray" strokeWidth={1} />
            <path d={makePath(fnRelu)} stroke="#999" fill="none" strokeWidth={1.5} strokeDasharray="4,3" />
            <path d={makePath(fnGelu)} stroke={dimStyleColor(DimStyle.Intermediates).toHexColor()} fill="none" strokeWidth={3} />
            <text x={8} y={14} fontSize={10} fill="#333">GELU (solid), ReLU (dashed)</text>
        </svg>
    </div>;
};

const VitSizesTable: React.FC = () => {
    let rows = [
        ['ViT-Base', 12, 768, 3072, 12, '86M'],
        ['ViT-Large', 24, 1024, 4096, 16, '307M'],
        ['ViT-Huge', 32, 1280, 5120, 16, '632M'],
    ];
    let state = useProgramState();
    let tiny = state.vit.layout;
    return <table className="my-2 mx-auto text-sm border-collapse">
        <thead>
            <tr className="border-b border-gray-400">{['Model', 'Layers', 'Hidden size D', 'MLP size', 'Heads', 'Params'].map(h => <th key={h} className="px-2 font-normal text-gray-600">{h}</th>)}</tr>
        </thead>
        <tbody>
            {rows.map(r => <tr key={r[0] as string}>{r.map((c, i) => <td key={i} className="px-2 text-center">{c}</td>)}</tr>)}
            <tr className="border-t border-gray-300 text-gray-600">
                <td className="px-2 text-center">ours</td>
                <td className="px-2 text-center">{tiny.shape.nBlocks}</td>
                <td className="px-2 text-center">{tiny.shape.C}</td>
                <td className="px-2 text-center">{tiny.shape.dFF}</td>
                <td className="px-2 text-center">{tiny.shape.nHeads}</td>
                <td className="px-2 text-center">{(tiny.weightCount / 1000).toFixed(0)}k</td>
            </tr>
        </tbody>
    </table>;
};

/** Pick a test image, or draw one, and re-run the model on it. */
const VitImagePicker: React.FC = () => {
    let state = useProgramState();
    let model = state.vit.model;
    let [draft, setDraft] = useState<Float32Array | null>(null);
    let [painting, setPainting] = useState<number | null>(null);
    if (!model) {
        return null;
    }
    let { imageSize, patchSize } = model.shape;
    let pred = vitPrediction(model);

    function run(image: Float32Array, label: number | null) {
        state.vit.pendingImage = { image, label };
        state.markDirty();
    }

    let pad = draft ?? model.image;

    function paint(idx: number, value: number) {
        let next = new Float32Array(pad);
        next[idx] = value;
        setDraft(next);
        run(next, null);
    }

    let padPx = 18;
    return <div className="my-2 flex flex-col items-center">
        <div className="flex flex-wrap justify-center gap-[3px] max-w-[460px]">
            {model.examples.map((ex, i) => <ImageSvg key={i} image={ex.image} size={30} imageSize={imageSize} patchSize={patchSize}
                onClick={() => { setDraft(null); run(ex.image, ex.label); }} />)}
        </div>
        <div className="flex items-center gap-4 mt-3">
            <div className="select-none" onMouseLeave={() => setPainting(null)} onMouseUp={() => setPainting(null)} onContextMenu={ev => ev.preventDefault()}>
                <svg width={imageSize * padPx} height={imageSize * padPx} className="border border-gray-500 cursor-crosshair">
                    {Array.from(pad).map((v, i) => {
                        let x = i % imageSize, y = Math.floor(i / imageSize);
                        return <rect key={i} x={x * padPx} y={y * padPx} width={padPx} height={padPx} fill={grayColor(v)} stroke="#ddd" strokeWidth={0.5}
                            onMouseDown={ev => { let val = ev.button === 2 || ev.shiftKey ? 0 : 1; setPainting(val); paint(i, val); }}
                            onMouseEnter={() => { if (painting !== null) paint(i, painting); }} />;
                    })}
                    {makeArrayRange(imageSize / patchSize - 1, 1, imageSize / patchSize - 1).map(k => <React.Fragment key={k}>
                        <line x1={k * patchSize * padPx} x2={k * patchSize * padPx} y1={0} y2={imageSize * padPx} stroke="#d33" strokeWidth={0.7} />
                        <line y1={k * patchSize * padPx} y2={k * patchSize * padPx} x1={0} x2={imageSize * padPx} stroke="#d33" strokeWidth={0.7} />
                    </React.Fragment>)}
                </svg>
            </div>
            <div className="flex flex-col items-start gap-1 text-sm">
                <div>Prediction: <span className="text-xl font-bold">{model.config.classes[pred.cls]}</span> ({(pred.prob * 100).toFixed(1)}%)</div>
                {model.imageLabel !== null && <div className="text-gray-600">Label: {model.config.classes[model.imageLabel]}</div>}
                <button className="px-3 py-1 rounded shadow bg-gray-200 hover:bg-blue-200" onClick={() => { let z = new Float32Array(imageSize * imageSize); setDraft(z); run(z, null); }}>Clear</button>
                <button className="px-3 py-1 rounded shadow bg-gray-200 hover:bg-blue-200" onClick={() => {
                    let ex = model!.examples[Math.floor(Math.random() * model!.examples.length)];
                    setDraft(null); run(ex.image, ex.label);
                }}>Random</button>
            </div>
        </div>
    </div>;
};
