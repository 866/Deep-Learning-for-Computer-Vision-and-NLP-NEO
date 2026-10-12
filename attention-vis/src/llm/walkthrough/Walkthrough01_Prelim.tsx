import React from 'react';
import { Phase } from "./Walkthrough";
import { cameraFor, commentary, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";
import s from './Walkthrough.module.scss';
import { Vec3 } from '@/src/utils/vector';

let paperLink = 'https://arxiv.org/abs/1706.03762';
let llmVizLink = 'https://bbycroft.net/llm';
let llmVizRepoLink = 'https://github.com/bbycroft/llm-viz';
let annotatedLink = 'https://nlp.seas.harvard.edu/annotated-transformer/';
let pytorchLink = 'https://pytorch.org/';

export function walkthrough01_Prelim(args: IWalkthroughArgs) {
    let { state, walkthrough: wt, layout } = args;

    if (wt.phase !== Phase.Intro_Prelim) {
        return;
    }

    let cam = cameraFor(layout.cubes, new Vec3(282, 18, 0), 1.1);
    setInitialCamera(state, cam.center, cam.angle);

    let { C, nHeads, nBlocks, dFF } = layout.shape;

    let c0 = commentary(wt, null, 0)`
Before we delve into the details, let's take a brief step back.

This guide focuses on _inference_, not training. The model's weights have already been trained (in ${embedLink('PyTorch', pytorchLink)},
in a few seconds), and we use them to compute the output. It all runs directly in your browser, so every number you see is real.

The Transformer was introduced in 2017 by Vaswani et al. in ${embedLink('"Attention Is All You Need"', paperLink)}, as a model for machine
translation. Until then, translation models were built on recurrent neural networks, which read a sentence one word at a time.
The Transformer instead relies entirely on _attention_: every position can look at every other position directly, all in parallel.
Its encoder-decoder design is the ancestor of today's language models: GPT keeps only the decoder half, and BERT only the encoder half.

The paper's "base" model has ${embedInline(<b>N = 6</b>)} layers in each stack, vectors of length ${embedInline(<b>d_model = 512</b>)}, ${embedInline(<b>h = 8</b>)} attention heads and
about 65 million parameters. Ours is identical in structure, just smaller: ${embedInline(<b>N = {nBlocks}</b>)}, ${embedInline(<b>d_model = {C}</b>)}, ${embedInline(<b>h = {nHeads}</b>)},
and a feed-forward size of ${embedInline(<b>d_ff = {dFF}</b>)}. You can see the real sizes using the buttons at the top of the 3D view.

We follow the paper closely: sinusoidal positional encodings, embeddings scaled by √d_model and shared with the output layer,
_post-norm_ residual connections ("Add & Norm"), and ReLU feed-forward networks. In the 3D view we call d_model ${embedInline(<b>C</b>)}, and the
size of each head's vectors, d_k = d_model / h, ${embedInline(<b>A</b>)}.

This guide is an adaptation of Brendan Bycroft's ${embedLink('LLM Visualization', llmVizLink)} (${embedLink('source', llmVizRepoLink)}), which walks through a GPT model in the same way;
its 3D renderer and much of its text are reused here. ${embedLink('The Annotated Transformer', annotatedLink)} is another great companion
for reading the paper alongside code.

Alright, let's get started!
`;

}

export function embedLink(a: React.ReactNode, href: string) {
    return embedInline(<a className={s.externalLink} href={href} target="_blank" rel="noopener noreferrer">{a}</a>);
}

export function embedInline(a: React.ReactNode) {
    return { insertInline: a };
}
