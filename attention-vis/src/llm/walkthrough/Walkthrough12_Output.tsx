import React from "react";
import { Dim, Vec3, Vec4 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { processUpTo, revealSequence, startProcessBefore } from "./Walkthrough00_Intro";
import { cameraFor, commentary, DimStyle, embed, IWalkthroughArgs, moveCameraTo, setInitialCamera } from "./WalkthroughTools";
import { findSubBlocks, splitGrid } from "../Annotations";
import { IGptModelLayout } from "../GptModelLayout";
import { setDecodedLength } from "../TransformerModel";
import { SourcePicker } from "../components/SourcePicker";
import { embedInline, embedLink } from "./Walkthrough01_Prelim";
import { clamp } from "@/src/utils/data";

export function walkthrough12_Output(args: IWalkthroughArgs) {
    let { walkthrough: wt, state, layout, tools: { afterTime, breakAfter, c_blockRef, c_dimRef, c_str } } = args;

    if (wt.phase !== Phase.Dec_Output) {
        return;
    }

    let { T } = layout.shape;
    let dec = layout.dec;

    let camOut = cameraFor([dec.output, layout.lmHeadWeight, layout.logits, layout.logitsSoftmax], new Vec3(266, 6, 0), 1.4);
    let camDecoder = cameraFor([dec.idxObj, ...dec.blocks.flatMap(b => b.cubes), layout.logitsSoftmax], new Vec3(262, 10, 0), 1.0);
    let camAll = cameraFor(layout.cubes, new Vec3(270, 14, 0), 1.05);

    setInitialCamera(state, camOut.center, camOut.angle);
    wt.dimHighlightBlocks = [dec.output, layout.lmHeadWeight, layout.logits, layout.logitsSoftmax];

    commentary(wt, null, 0)`
Finally, we come to the end of the model. We apply a linear transformation (a matrix multiplication, without a bias) to the
${c_blockRef('output of the decoder', dec.output)}.

This transformation takes each of our column vectors from length ${c_dimRef('C', DimStyle.C)} to length ${c_dimRef('n_vocab', DimStyle.n_vocab)}. Hence,
it's effectively producing a score for each token in the vocabulary for each of our columns. These
scores have a special name: ${c_blockRef('logits', layout.logits)}.

Notice that the ${c_blockRef('weight matrix', layout.lmHeadWeight)} is the same ${c_blockRef('token embedding matrix', dec.tokEmbedObj)} we started with (here drawn transposed).
The paper shares one matrix between the encoder's input, the decoder's input, and this output layer, which saves parameters: the score for a
token is simply the dot product of the decoder's output with that token's embedding.`;
    breakAfter();

    let t_logits = afterTime(null, 2.0, 0.3);

    breakAfter();

    commentary(wt)`
To convert these scores into nice probabilities, we pass them through a softmax operation. Now, for each column, we have a ${c_blockRef('probability', layout.logitsSoftmax)}
the model assigns to each token in the vocabulary.`;
    breakAfter();

    let t_softmax = afterTime(null, 2.0, 0.3);
    let t_outputToks = afterTime(null, 1.0, 0.3);

    breakAfter();

    commentary(wt)`
Column ${c_dimRef('t', DimStyle.T)} holds the prediction for output token ${c_dimRef('t', DimStyle.T)}. This model has learned its task well, so the probabilities are
heavily weighted toward the correct answer, which you can read off below the model.

But wait: we fed the decoder the complete answer, so isn't that cheating? During training, that's what we do (every position is trained
at once, and the mask stops it from peeking ahead). But to actually _generate_ an answer, we have to run the model one token at a time.

We start with the decoder input containing only ${c_str('<s>', 0, DimStyle.Token)}. We run the model, and take the most likely token at the last position.
This becomes the next token of the decoder input, and we run the decoder again, and again, until the output is complete:`;
    breakAfter();

    let t_camDecoder = afterTime(null, 1.0, 0.3);
    let t_decode = afterTime(null, 9.0, 0.5);

    breakAfter();

    commentary(wt)`
This is called _autoregressive_ decoding. The encoder only runs once: its output stays the same at every step. And thanks to the mask,
the values computed for the earlier positions don't change when a token is added, which is why real implementations cache them.

Here we always pick the single most likely token ("greedy" decoding). The paper instead used _beam search_, which keeps the
4 most likely partial outputs at each step and picks the best complete one at the end.

Now try it yourself! Click the letters to change the source sequence, then press Run. The whole model, both in the 3D view and in the
diagrams, will update. ${embed(SourcePicker)}`;
    breakAfter();

    let t_camAll = afterTime(null, 1.0, 0.3);

    commentary(wt)`
And that's the Transformer! The real models in the paper are the same, only bigger: use the buttons at the top of the 3D view to see the
base and big configurations. Trained on millions of sentence pairs, the big model reached a BLEU score of 28.4 on WMT 2014 English-to-German
translation, a new state of the art, after 3.5 days of training on 8 GPUs.

Its success, and the parallelism that comes from replacing recurrence with attention, made the Transformer the foundation of
practically all of today's large language models.

It's not just for language either. Continue to the next part to see the _Vision Transformer_, which uses the Transformer's encoder to classify images.`;

    moveCameraTo(state, t_camDecoder, camDecoder.center, camDecoder.angle);
    moveCameraTo(state, t_camAll, camAll.center, camAll.angle);

    let outBlocks = [layout.logits, layout.logitsAgg2, layout.logitsAgg1, layout.logitsSoftmax];
    if (t_logits.t === 0 || (t_logits.active && t_camDecoder.t === 0)) {
        for (let blk of outBlocks) {
            if (t_logits.active || t_logits.t === 0) {
                blk.access!.disable = true;
            }
        }
    }
    let info = startProcessBefore(state, layout.logits);
    if (t_logits.active && t_camDecoder.t === 0) {
        processUpTo(state, t_logits, layout.logits, info);
    }
    if (t_softmax.active && t_camDecoder.t === 0) {
        processUpTo(state, t_softmax, layout.logitsSoftmax, info);
    }
    if (t_softmax.t === 0 || (t_logits.active && t_outputToks.t === 0)) {
        state.display.tokenOutputColors = { color1: new Vec4(0,0,0,0), color2: Vec4.fromHexColor('#000', 1), mixes: revealSequence(T, t_outputToks.t) };
    }

    // step-by-step generation
    let model = state.model;
    if (model) {
        let n = T;
        if (t_camDecoder.active && t_decode.t < 1) {
            n = 1 + clamp(Math.floor(t_decode.t * T), 0, T - 1);
        }
        setDecodedLength(model, n);
        if (n < T) {
            hideFuturePositions(layout, n);
        }
    }
}

/** Disables (greys out) the decoder's values at positions >= n, which haven't been generated yet. */
function hideFuturePositions(layout: IGptModelLayout, n: number) {
    let memCopies = new Set(layout.dec.blocks.map(b => b.crossAttn.memCopy));
    let startIdx = layout.cubes.indexOf(layout.dec.idxObj);
    for (let i = startIdx; i < layout.cubes.length; i++) {
        let blk = layout.cubes[i];
        if (blk.t === 'w' || !blk.access || memCopies.has(blk)) {
            continue;
        }
        // the position (query) axis is y for the attention matrices & transposed blocks, otherwise x
        let dim = blk.dimY === DimStyle.T ? Dim.Y : Dim.X;
        splitGrid(layout, blk, dim, n, 0);
        for (let sub of findSubBlocks(blk, dim, n, null)) {
            if (sub.access) {
                sub.access = { ...sub.access, disable: true };
            }
        }
    }
}
