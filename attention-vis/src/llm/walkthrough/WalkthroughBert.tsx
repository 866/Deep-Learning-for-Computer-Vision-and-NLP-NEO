import React, { useState } from "react";
import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, DimStyle, dimStyleColor, embed, IBertWalkthroughArgs, moveCameraToFor, setInitialCamera, setInitialCameraFor } from "./WalkthroughTools";
import { numberToCommaSep, processUpTo, revealBlockAlongX, revealSequence, startProcessBefore } from "./Walkthrough00_Intro";
import { embedInline, embedLink } from "./Walkthrough01_Prelim";
import { useProgramState } from "../Sidebar";
import { bertSentiment, bertTopK } from "../BertModel";
import { IBlkDef } from "../GptModelLayout";
import { IProgramState } from "../Program";
import { addPrimitiveRestart, addVert } from "../render/triRender";
import { Vec4 } from "@/src/utils/vector";
import { bertMlmChartRegion, bertSentimentChartRegion } from "../components/ModelCard";

/*
The BERT chapters, after "BERT: Pre-training of Deep Bidirectional Transformers for Language
Understanding" (Devlin, Chang, Lee & Toutanova, 2018, arXiv:1810.04805). They run on the tiny BERT
trained by training/train_bert.py (state.bert), which replaces the tiny Transformer in the 3D view.

Every accuracy figure spoken here is read out of the checkpoint's own metrics block (config.metrics)
rather than typed into this file, so the prose cannot drift from the weights being shipped. The
figures are the paper's originals, served from public/bert-figs/ \u2014 see LICENSE-BERT-figures.md.
*/

const bertPaperUrl = 'https://arxiv.org/abs/1810.04805';

function hideIntermediates(blocks: IBlkDef[]) {
    for (let cube of blocks) {
        if (cube.t === 'i' && cube.access) {
            cube.access.disable = true;
        }
    }
}

/** Everything but the n_vocab-sized blocks (the token table and the MLM output, and what sits under it), which would dwarf the rest. */
function overviewCubes(layout: IBertWalkthroughArgs['layout']) {
    let big = layout.shape.vocabSize * layout.cell * 0.5;
    return layout.cubes.filter(c => Math.max(c.dx, c.dy) < big && c.y < layout.mlmLogits.y);
}

/**
 * Start a processing walk at `blk`: everything before it (in cube order, which is computation order)
 * is shown as already computed, so the walk spends its time on what the chapter is about rather than
 * re-running the embeddings off screen.
 */
function startWalkAt(state: IProgramState, blk: IBlkDef) {
    for (let c of state.mainLayout.cubes) {
        if (c === blk) {
            break;
        }
        if (c.t !== 'w' && c.access) {
            c.access.disable = false;
        }
    }
    return startProcessBefore(state, blk);
}

/** The input pieces drawn above the token row (see ModelCard.ts), as a framing region. */
function inputStripBlk(layout: IBertWalkthroughArgs['layout']) {
    let tok = layout.tokens;
    let w = 7 * layout.shape.T;
    return regionBlk({ x: tok.x + tok.dx / 2 - w / 2, y: tok.y - layout.margin - 14, dx: w, dy: layout.margin + 14 });
}

/**
 * Shade the strictly-upper triangle of an attention matrix: the cells (key after query) that a
 * left-to-right model like GPT would mask. Drawn on the block's front face, in model space.
 */
function shadeUpperTriangle(state: IProgramState, blk: IBlkDef, amount: number) {
    if (amount <= 0 || blk.opacity === 0) {
        return;
    }
    let render = state.render.triRender;
    let c = blk.dx / blk.cx; // cell size; cell x = key, cell y = query
    let z = blk.z + blk.dz + 0.05;
    let color = Vec4.fromHexColor('#c92a2a').mul(0.55 * amount);
    let x0 = blk.x, y0 = blk.y, x1 = blk.x + blk.dx, y1 = blk.y + blk.dy;
    addVert(render, new Vec3(x0 + c, y0, z), color);
    addVert(render, new Vec3(x1, y0, z), color);
    addVert(render, new Vec3(x1, y1 - c, z), color);
    addPrimitiveRestart(render);
}

/** A camera-framing stand-in for a 2D region drawn in the z = 0 plane (a chart beside a block). */
function regionBlk(r: { x: number, y: number, dx: number, dy: number }) {
    return { x: r.x, y: r.y, z: 0, dx: r.dx, dy: r.dy, dz: 0 } as unknown as IBlkDef;
}

/** The sentiment head and the chart under it. */
function sentimentCubes(layout: IBertWalkthroughArgs['layout']) {
    return [layout.sentimentWeight, layout.sentimentBias, layout.sentimentLogits, layout.sentimentProbs, regionBlk(bertSentimentChartRegion(layout))];
}

/** The MLM transform and the two [CLS] heads: the head section without its n_vocab-tall output blocks. */
function headOverviewCubes(layout: IBertWalkthroughArgs['layout']) {
    let top = layout.mlmLogits;
    return [layout.output, layout.mlmFcWeight, layout.mlmDense, layout.mlmAct, ...layout.mlmLn.cubes,
        regionBlk({ x: top.x, y: top.y, dx: top.dx, dy: 40 }),
        layout.poolWeight, layout.poolOut, layout.nspProbs, ...sentimentCubes(layout)];
}

// ---------------------------------------------------------------- 1. overview

export function walkthroughBert01_Intro(args: IBertWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools: { breakAfter, afterTime, c_blockRef, c_dimRef } } = args;
    if (wt.phase !== Phase.Bert_Intro) {
        return;
    }
    let model = layout.model;
    let { T, C, nBlocks, nHeads, vocabSize } = layout.shape;
    let m = model?.config.metrics;

    setInitialCameraFor(state, overviewCubes(layout), new Vec3(270, 14, 0), 1.0);

    let c0 = commentary(wt, null, 0)`BERT stands for _Bidirectional Encoder Representations from Transformers_. It was
${embedLink('the paper', bertPaperUrl)} (Devlin, Chang, Lee & Toutanova, 2018) that made pre-training the standard way to build
a language model, and it is the ancestor of most of what came after it. Its claim, in the abstract's words, is that it is
"conceptually simple and empirically powerful".

The trick is not a new architecture \u2014 it is the encoder we have already looked at, many times over. The trick is how it is
_trained first_. Here is a tiny one: ${embedInline(numberToCommaSep(layout.weightCount))} parameters, ${nBlocks} layers,
${c_dimRef(`H = ${C}`, DimStyle.C)} hidden size, ${nHeads} attention heads, a vocabulary of ${vocabSize} word pieces.

${embed(Fig1Overview)}

What the top half of that figure shows is a _model_ being trained on tasks that need no labels, and what the bottom half shows is
that same model being finished off on a task that does. In the paper's own words, after pre-training, downstream models are built
"apart from output layers, [with] the same architectures \u2026 used in both pre-training and fine-tuning".`;

    if (c0.t > 0) {
        hideIntermediates(layout.cubes);
    }

    breakAfter();

    // the input strip and the id rows; the 600-wide token table would push them off the edge of the view
    let inputCubes = [inputStripBlk(layout), layout.tokens, layout.segIds, layout.segEmbed, layout.posEmbed, layout.residual0];
    let t_cam = afterTime(null, 1.5, 0.4);
    moveCameraToFor(state, t_cam, inputCubes, new Vec3(276, 8, 0), 1.0);
    let t_reveal = afterTime(null, 1.5, 0.3);
    if (t_reveal.active) {
        state.display.tokenIdxColors = { mixes: revealSequence(T, t_reveal.t), color2: dimStyleColor(DimStyle.TokenIdx) };
        revealBlockAlongX(layout, layout.tokens, t_reveal.t);
        revealBlockAlongX(layout, layout.segIds, t_reveal.t);
    }

    breakAfter();

    commentary(wt)`The sentence comes in as word pieces \u2014 the same idea as the BPE tokens you saw in the Transformer chapters \u2014 looked up
in the ${c_blockRef('token embedding table', layout.tokEmbed)}, added to a ${c_blockRef('segment', layout.segEmbed)} and a
${c_blockRef('position', layout.posEmbed)} embedding. Chapter 2 stays on this.`;
    breakAfter();

    let blockCubes = layout.blocks.flatMap(b => b.cubes);
    let t_cam2 = afterTime(null, 1.2, 0.3);
    moveCameraToFor(state, t_cam2, [layout.residual0, ...blockCubes], new Vec3(285, 14, 0), 0.9);
    let t_layers = afterTime(null, 6.0, 0.5);
    let walk = processUpTo(state, t_layers, layout.output);

    breakAfter();

    commentary(wt)`${nBlocks} identical layers of _bidirectional_ self-attention and feed-forward \u2014 the encoder stack, unmodified. Everything
BERT does, every task it wins, comes out of this one stack.`;
    breakAfter();

    // the two heads side by side: the MLM transform under the stack, the [CLS] heads beside it
    let t_cam3 = afterTime(null, 1.2, 0.3);
    moveCameraToFor(state, t_cam3, headOverviewCubes(layout), new Vec3(272, 8, 0), 1.0);
    let t_mlm = afterTime(null, 3.0, 0.3);
    processUpTo(state, t_mlm, layout.mlmLn.lnResid, walk);
    // the n_vocab-tall logits run off the bottom of the view, so they are shown computed rather than walked
    let t_cls = afterTime(null, 3.0, 0.5);
    if (t_cls.active) {
        processUpTo(state, t_cls, layout.sentimentProbs, startWalkAt(state, layout.poolOut));
    }

    breakAfter();

    commentary(wt)`And then the part that changes per task: a ${c_blockRef('masked-language-model head', layout.mlmProbs)} here, a
${c_blockRef('classification head', layout.sentimentProbs)} there. ${m
        ? `This one reads ${(m.sentiment_dev_acc * 100).toFixed(1)}% on ${m.n_dev} sentences it was never trained on, where always guessing
the more common class gets ${(m.majority_baseline * 100).toFixed(1)}%. The full picture is on the card above the model.`
        : 'The numbers are on the card above the model.'}`;
    breakAfter();
}

// ---------------------------------------------------------------- 2. input representation

export function walkthroughBert02_Input(args: IBertWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools: { breakAfter, afterTime, c_blockRef, c_dimRef, c_str } } = args;
    if (wt.phase !== Phase.Bert_Input) {
        return;
    }
    let model = layout.model;
    let { T, C, vocabSize } = layout.shape;

    let inputCubes = [layout.tokens, layout.segIds, layout.tokEmbed, layout.segEmbed, layout.posEmbed, layout.residual0, layout.embedLN.lnResid];
    setInitialCameraFor(state, inputCubes, new Vec3(276, 8, 0), 0.85);

    let c0 = commentary(wt)`Figure 2 of the paper is the whole input story in one picture, and it is worth reading slowly.

${embed(Fig2Input)}

Three embeddings are _summed_, not concatenated. The token embedding says _which_ piece it is; the segment embedding says
_which sentence_ it belongs to (0 or 1 \u2014 that is all ${c_blockRef('this table', layout.segEmbed)} holds); the position embedding says
_where_ it is, because nothing in attention carries an order.`;

    if (c0.t > 0) {
        hideIntermediates(layout.cubes);
    }
    breakAfter();

    commentary(wt)`Here is what the model is actually holding for the sentence on screen, produced by the same WordPiece routine the
training script used. A piece that continues a word carries the ${c_str('##')} prefix, so one written word can occupy several columns \u2014
and each column below is exactly one ${c_blockRef('column of the embedding block', layout.residual0)}.

${embed(InputStrip)}`;
    breakAfter();

    let t_lookup = afterTime(null, 4.0, 0.5);
    let walk = processUpTo(state, t_lookup, layout.residual0);
    breakAfter();

    let t_ln = afterTime(null, 2.0, 0.4);
    processUpTo(state, t_ln, layout.embedLN.lnResid, walk);
    breakAfter();

    commentary(wt)`Three special tokens give the architecture handles. ${c_str('[CLS]')} sits at position 0 and is what the classification
heads read. ${c_str('[SEP]')} ends each sentence, and its id is what a segment id counts. ${c_str('[MASK]')} appears only in the second
chapter of this model's life \u2014 pre-training \u2014 and is the subject of chapter 4.

Note what is _not_ here: ${c_dimRef(`T = ${T}`, DimStyle.T)} is a hard ceiling. A longer sentence is cut, and the model never learns that
anything was removed \u2014 the truncation figure on the card is what that costs.`;
    breakAfter();
}

// ---------------------------------------------------------------- 3. bidirectional attention

export function walkthroughBert03_Attn(args: IBertWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools: { breakAfter, afterTime, c_blockRef } } = args;
    if (wt.phase !== Phase.Bert_Attn) {
        return;
    }
    let { T } = layout.shape;
    let block = layout.blocks[0];
    let head = block.attn.heads[0];

    // the same three-quarter view of the heads the Transformer's self-attention chapter uses: the heads
    // are spread along z, so a straight-on view stacks them on top of each other
    setInitialCameraFor(state, block.attn.heads.flatMap(h => h.cubes), new Vec3(290, 24, 0), 1.0);

    let c0 = commentary(wt)`This is the one architectural thing BERT changed, and \u00a71 explains why it had to: "standard language models are unidirectional, and this limits the choice of architectures that can be used during pre-training". A model that reads only left-to-right can be trained by predicting its own next token — which is the cheapest source of training data there is, namely text itself.

${embed(Fig3Comparison)}

Those arrows are attention arrows. "BERT uses a bidirectional Transformer. OpenAI GPT uses a left-to-right Transformer. ELMo uses the concatenation of independently trained left-to-right and right-to-left LSTMs" \u2014 that is Figure 1's own caption. And ELMo, the paper notes, is "not deeply bidirectional": its two directions are trained separately and only glued together afterwards. Below is what BERT's choice looks like from the inside, one head of one layer.`;

    if (c0.t > 0) {
        hideIntermediates(layout.cubes);
    }
    // the embeddings are already computed; the walk is about this layer's attention
    let walkStart = startWalkAt(state, head.qBlock);
    breakAfter();

    // Q, K, V, the scores and their softmax, for the first head
    let t_attn = afterTime(null, 6.0, 0.5);
    let walk = processUpTo(state, t_attn, head.attnMtxSm, walkStart);
    breakAfter();

    commentary(wt)`${c_blockRef('The attention matrix', head.attnMtx)} for one head, one layer. Row by row, each position asks every
position \u2014 including the ones that come _after_ it \u2014 and after the ${c_blockRef('softmax', head.attnMtxSm)} every row sums to 1.
Nothing above the diagonal has been thrown away. (The [PAD] columns are: they are padding, not words, so every row gives them a weight of 0.)`;
    breakAfter();

    // Shade the strictly-upper triangle of every head's matrices in and out: those are the cells a
    // left-to-right model would be forced to ignore. Underneath, they hold this model's real values.
    // look at the score and softmax matrices from low down, under the Q/K/V weights that sit above them
    let mtxCubes = block.attn.heads.flatMap(h => [h.attnMtx, h.attnMtxAgg1, h.attnMtxAgg2, h.attnMtxSm]);
    let t_maskCam = afterTime(null, 1.2, 0.2);
    moveCameraToFor(state, t_maskCam, mtxCubes, new Vec3(286, 8, 0), 0.7);
    let t_mask = afterTime(null, 3.0, 0.4);
    if (t_mask.active && t_mask.t < 1) {
        let pulse = Math.sin(t_mask.t * Math.PI);
        for (let h of block.attn.heads) {
            shadeUpperTriangle(state, h.attnMtx, pulse);
            shadeUpperTriangle(state, h.attnMtxSm, pulse);
        }
    }

    commentary(wt)`If this were GPT, everything strictly _above_ the diagonal would be forced to zero \u2014 a token could not see the future.
That is not a detail: it is the reason GPT can generate and the reason BERT, as shipped, cannot. A model that may look both ways has no
next token to predict, which is why BERT had to invent the task in the next chapter.`;
    breakAfter();

    // the other head, the projection and the residual add
    let t_outCam = afterTime(null, 1.2, 0.2);
    moveCameraToFor(state, t_outCam, [...block.attn.cubes, block.input], new Vec3(288, 18, 0), 0.95);
    let t_out = afterTime(null, 4.0, 0.4);
    processUpTo(state, t_out, block.attn.attnResidual, walk);
    breakAfter();

    commentary(wt)`The rest is the machinery you already know: ${c_blockRef('the projection', block.attn.projWeight)}, the residual add and
${c_blockRef('norm', block.ln1.lnResid)} (BERT keeps the original Transformer's post-norm "Add & Norm"), the feed-forward block. Bidirectionality cost about two characters of code \u2014 not setting the
causal flag \u2014 and changed what the whole field does next.`;
    breakAfter();
}

// ---------------------------------------------------------------- 4. masked language model

export function walkthroughBert04_Mlm(args: IBertWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools: { breakAfter, afterTime, c_blockRef, c_dimRef, c_str } } = args;
    if (wt.phase !== Phase.Bert_Mlm) {
        return;
    }
    let model = layout.model;
    let m = model?.config.metrics;
    let { T, vocabSize } = layout.shape;

    // the transform (dense -> GELU -> norm) first; the n_vocab-tall output and its chart come later
    let transformCubes = [layout.output, layout.mlmFcWeight, layout.mlmFcBias, layout.mlmDense, layout.mlmAct, ...layout.mlmLn.cubes];
    setInitialCameraFor(state, transformCubes, new Vec3(272, 6, 0), 1.0);

    let c0 = commentary(wt)`Bidirectional attention makes the usual task impossible, so BERT invented one: the _masked language model_. Take a
sentence, mask some tokens, and ask the model to guess what was there, using only context.

The procedure, from \u00a73.1, verbatim in its essentials: the generator chooses "15% of the token positions at random for prediction"
and replaces the chosen token with (1) the ${c_str('[MASK]')} token "80% of the time", (2) a random token "10% of the time",
(3) the unchanged token "10% of the time".

That odd 80/10/10 is not decoration. If the input always showed ${c_str('[MASK]')}, the model would learn to predict only at masked positions,
and ${c_str('[MASK]')} never appears at fine-tuning time. The random and unchanged cases keep the rest of the sentence honest.`;

    if (c0.t > 0) {
        hideIntermediates(layout.cubes);
    }
    // the stack has already run; the walk is the head's own
    let walkStart = startWalkAt(state, layout.mlmDense);
    breakAfter();

    // dense -> GELU -> norm, then down to the tied output layer and the logits it scores
    let t_head = afterTime(null, 4.0, 0.5);
    let walk = processUpTo(state, t_head, layout.mlmLn.lnResid, walkStart);
    let t_logitsCam = afterTime(null, 1.2, 0.2);
    moveCameraToFor(state, t_logitsCam, [layout.mlmLn.lnResid, layout.mlmOutWeight, layout.mlmOutBias, layout.mlmLogits], new Vec3(272, 6, 0), 1.0);
    let t_logits = afterTime(null, 4.0, 0.5);
    walk = processUpTo(state, t_logits, layout.mlmLogits, walk);
    breakAfter();

    commentary(wt)`The head is small: a square ${c_blockRef('projection', layout.mlmFcWeight)}, a GELU, a ${c_blockRef('norm', layout.mlmLn.lnResid)},
and then \u2014 the elegant part \u2014 a projection onto ${c_blockRef('the token embedding table itself', layout.mlmOutWeight)}, transposed. The rows the
input looked up _are_ the columns the output scores against, so the table is learned once and used twice. It costs
${c_dimRef(`V = ${vocabSize}`, DimStyle.n_vocab)} logits per position.`;
    breakAfter();

    // the softmax over the vocabulary, with the whole column in view, then down to the predictions under it
    let probs = layout.mlmProbs;
    let t_smCam = afterTime(null, 1.2, 0.2);
    moveCameraToFor(state, t_smCam, [layout.mlmLogits, probs], new Vec3(272, 6, 0), 1.0);
    let t_sm = afterTime(null, 3.0, 0.4);
    processUpTo(state, t_sm, probs, walk);
    let t_chartCam = afterTime(null, 1.2, 0.3);
    let probsBottom = regionBlk({ x: probs.x, y: probs.y + probs.dy - 60, dx: probs.dx, dy: 60 });
    moveCameraToFor(state, t_chartCam, [probsBottom, regionBlk(bertMlmChartRegion(layout))], new Vec3(270, 4, 0), 1.0);
    breakAfter();

    let mlmAll = m ? (m.mlm_top1_all * 100).toFixed(1) : 'n/a';
    let mlmFreq = m ? (m.mlm_top1_frequent * 100).toFixed(1) : 'n/a';
    commentary(wt)`Which is why this model's masked-token accuracy has to be read with care. Fine-tuned on SST-2 through a
${c_dimRef(`V = ${vocabSize}`, DimStyle.n_vocab)}-piece vocabulary, it gets ${mlmAll}% of masked positions right overall, and
${mlmFreq}% when the masked piece is a frequent whole word.

The split is ours, not the paper's, and it is the expected shape of the result: the masks this vocabulary makes easy are function words
\u2014 ${c_str('the')}, ${c_str('and')}, ${c_str('a')} \u2014 recoverable from grammar alone, while content words need a world model that two layers
and one sentiment dataset cannot supply.

The ${c_str('[MASK]')} below sits at the middle piece of the current sentence \u2014 a fixed rule, chosen so you can predict
where it lands, not tuned to wherever this model does best. Its candidates, straight out of ${c_blockRef('mlmProbs', layout.mlmProbs)}:

${embed(MaskPredictionsAtMask)}

Try a sentence where the missing word is grammatically pinned, and then one where it is a noun of your choosing. The gap you see is
${mlmAll}% versus ${mlmFreq}% made visible. If the top candidate is not what you expected, that is the honest result, not a bug: the metric
beside it is the same one this table reports.`;
    breakAfter();
}

// ---------------------------------------------------------------- 5. pre-training & fine-tuning

export function walkthroughBert05_PretrainFT(args: IBertWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools: { breakAfter, afterTime, c_blockRef } } = args;
    if (wt.phase !== Phase.Bert_PretrainFT) {
        return;
    }
    let model = layout.model;
    let m = model?.config.metrics;

    let nspCubes = [layout.poolWeight, layout.poolBias, layout.poolOut, layout.nspWeight, layout.nspLogits, layout.nspProbs];
    setInitialCameraFor(state, nspCubes, new Vec3(270, 2, 0), 1.1);

    let c0 = commentary(wt)`Pre-training had a second task, and it is the one BERT later dropped. The _next sentence prediction_ head is given a
pair (A, B) and must say whether B actually follows A.

To read a sentence-level relation, BERT uses the ${c_blockRef('[CLS]', layout.poolOut)} vector \u2014 but pushed through a square ${c_blockRef('pooler', layout.poolWeight)}
plus a tanh first. It is the only place in the whole architecture where a pooler appears.`;

    if (c0.t > 0) {
        hideIntermediates(layout.cubes);
    }
    let walkStart = startWalkAt(state, layout.poolOut);
    breakAfter();

    let t_nsp = afterTime(null, 4.0, 0.5);
    processUpTo(state, t_nsp, layout.nspProbs, walkStart);
    breakAfter();

    commentary(wt)`And here is what this checkpoint says: ${(m ? (m.nsp_dev_acc * 100).toFixed(1) : '50.0')}%.
${m && Math.abs(m.nsp_dev_acc - 0.5) < 0.06
        ? 'Which is coin-flipping. This demo feeds it a sentence and nothing, on a single-sentence task, with two layers and a 600-piece vocabulary \u2014 so the head learned nothing, sits at chance, and the honest thing is to show you that rather than hide it.'
        : 'Reported from the checkpoint rather than retyped.'}

That is a preview of history rather than a bug in this demo. The paper's own ablation (Table 3) measured it: dropping NSP cost BERT-BASE
0.1 of a point on SST-2, 92.7% to 92.6%. RoBERTa (Liu et al., 2019) later removed the task outright and did better. The head is drawn here
because Figure 1 has it, and because running the paper's recipe honestly \u2014 including the part that barely helped \u2014 teaches more than a curated
model would.`;
    breakAfter();

    let t_fig = afterTime(null, 0.5, 0.2);
    commentary(wt)`Fine-tuning is the other half of the recipe, and it is short: take the pre-trained weights, add a head, keep training
_everything_ (not just the head) on the labelled task. The paper's Figure 4 shows the range \u2014 one architecture, eleven tasks, only the
shaded output layers differing.

${embed(Fig4FineTune)}

Two of those shapes matter here. Row (a) is a pair classifier \u2014 what NSP was. Row (b) is a single-sentence classifier, which is what SST-2
is, and what the rest of this walkthrough uses: our "sentence pair" is really the degenerate case, sentence A and nothing.`;
    breakAfter();

    let t_cls = afterTime(null, 3.0, 0.4);
    moveCameraToFor(state, t_cls, sentimentCubes(layout), new Vec3(270, 2, 0), 1.0);
    if (t_cls.active) {
        processUpTo(state, t_cls, layout.sentimentProbs, startWalkAt(state, layout.sentimentLogits));
    }
    breakAfter();

    commentary(wt)`The classifier itself is one matrix, and \u00a74.1 writes it down exactly:

${embedInline(<div className="text-center my-2 italic">log(softmax(C W<sup>T</sup>)), &nbsp; W \u2208 \u211D<sup>K\u00d7H</sup></div>)}

where C is the final [CLS] vector. Here K = 2 (negative, positive) and H is this model's hidden size, so ${c_blockRef('W', layout.sentimentWeight)} is
${layout.shape.C} \u00d7 2 and that is the entire set of new parameters \u2014 apart from it, nothing about the architecture changed between pre-training and
this task.`;
    breakAfter();
}

// ---------------------------------------------------------------- 6. sentiment classifier, live

export function walkthroughBert06_Sentiment(args: IBertWalkthroughArgs) {
    let { state, layout, walkthrough: wt, tools: { breakAfter, afterTime, c_blockRef, c_dimRef } } = args;
    if (wt.phase !== Phase.Bert_Sentiment) {
        return;
    }
    let model = layout.model;
    let m = model?.config.metrics;
    let { C, nBlocks, nHeads } = layout.shape;

    setInitialCameraFor(state, sentimentCubes(layout), new Vec3(270, 4, 0), 1.0);

    let c0 = commentary(wt)`Last chapter: the model doing the thing it was trained to do, on a sentence you choose.

${embed(SentencePicker)}

Every edit re-runs the whole network \u2014 forward pass through both layers, the MLM head and both ${c_blockRef('[CLS]', layout.poolOut)} heads \u2014 so the box commits on Enter or on a pause in typing, not on every keystroke.`;

    if (c0.t > 0) {
        hideIntermediates(layout.cubes);
    }
    // everything up to the classifier has run (the MLM and NSP heads included); the walk is the classifier
    let walkStart = startWalkAt(state, layout.sentimentLogits);
    breakAfter();

    let t_run = afterTime(null, 4.0, 0.5);
    processUpTo(state, t_run, layout.sentimentProbs, walkStart);
    breakAfter();

    commentary(wt)`The vector it reads is position 0 of the last layer's output: ${c_dimRef(`C = ${C}`, DimStyle.C)} numbers that attention spent
two layers pulling together from the whole sentence. The ${c_blockRef('two rows of W', layout.sentimentWeight)} turn those into two scores; a
softmax turns the scores into the ${c_blockRef('two probabilities', layout.sentimentProbs)} you see.`;
    breakAfter();

    commentary(wt)`How good is it, really? From the checkpoint's metrics, measured on ${m ? m.n_dev : 872} SST-2 development sentences it never saw:

${embed(BertResultsTable)}

The honest summary: ${((m ? m.sentiment_dev_acc : 0) * 100).toFixed(1)}% against a
${((m ? m.majority_baseline : 0) * 100).toFixed(1)}% baseline is a real signal \u2014 and it is far short of the paper's own BERT-BASE, which scored
93.5% on the SST-2 test set (Table 1) with 12 layers, a hidden size of 768 and 110 M parameters. Two layers, ${c_dimRef(`C = ${C}`, DimStyle.C)} and a
600-piece vocabulary were never going to match that, and the number is printed on the card rather than a nicer one.`;
    breakAfter();

    let t_cam = afterTime(null, 2.0, 0.4);
    moveCameraToFor(state, t_cam, overviewCubes(layout), new Vec3(270, 10, 0), 1.0);
    breakAfter();

    commentary(wt)`For scale, what the paper actually shipped, next to the two shapes in the toolbar:

${embed(BertSizesTable)}

Three ideas carried all of it: bidirectional attention, a masked token to make that trainable, and fine-tuning everything afterwards. That is the
whole of BERT.`;
    breakAfter();
}

// ---------------------------------------------------------------- helpers & widgets

/** Framing helper: the camera llm-viz's other chapters build, as the pair setInitialCamera wants. */
/** One of the paper's original figures, with attribution. The caption text is load-bearing: the check
 * harness asserts each of these names Devlin and the arXiv id, because shipping someone's figure without
 * saying whose it is is not an option we get to forget about later. */
export const PaperFigure: React.FC<{ name: string }> = ({ name }) => {
    let captions: Record<string, string> = {
        'fig1-overview': 'Figure 1: overall pre-training and fine-tuning procedures for BERT. "Apart from output layers, the same architectures are used in both pre-training and fine-tuning." \u2014 Devlin, Chang, Lee & Toutanova, BERT (arXiv:1810.04805), Figure 1.',
        'fig2-input': 'Figure 2: "The input embeddings are the sum of the token embeddings, the segmentation embeddings and the position embeddings." \u2014 Devlin et al., BERT (arXiv:1810.04805), Figure 2.',
        'fig3-comparison': 'Figure 1b: "BERT uses a bidirectional Transformer. OpenAI GPT uses a left-to-right Transformer. ELMo uses the concatenation of independently trained left-to-right and right-to-left LSTMs." \u2014 Devlin et al., BERT (arXiv:1810.04805).',
        'fig4-fine-tune': 'Figure 4: "Illustrations of Fine-tuning BERT on Different Tasks", with the learned (shaded) and pre-trained (unshaded) output layers marked. \u2014 Devlin et al., BERT (arXiv:1810.04805), Figure 4.',
    };
    return <figure className="my-3 mx-auto max-w-[620px] text-center">
        <img src={`bert-figs/${name}.svg`} alt={`BERT paper figure: ${name}`} className="w-full bg-white rounded border border-gray-300" />
        <figcaption className="mt-1 text-xs leading-snug text-gray-600 text-left">{captions[name]}</figcaption>
    </figure>;
};

export const Fig1Overview: React.FC = () => <PaperFigure name="fig1-overview" />;
export const Fig2Input: React.FC = () => <PaperFigure name="fig2-input" />;
export const Fig3Comparison: React.FC = () => <PaperFigure name="fig3-comparison" />;
export const Fig4FineTune: React.FC = () => <PaperFigure name="fig4-fine-tune" />;

/** The live WordPiece encoding of the current sentence: pieces, ids, segment ids. */
export const InputStrip: React.FC = () => {
    let state = useProgramState();
    let model = state.bert.model;
    if (!model) {
        return null;
    }
    // what the model actually holds, [MASK] included, rather than a re-tokenisation of the text
    let toks = Array.from(model.inputBuf, id => model!.vocab[id]);
    let tokColor = dimStyleColor(DimStyle.Token).toHexColor();
    let idxColor = dimStyleColor(DimStyle.TokenIdx).toHexColor();
    let segColor = dimStyleColor(DimStyle.n_segments).toHexColor();
    let maskColor = dimStyleColor(DimStyle.n_vocab).toHexColor();
    return <div className="my-2 mx-auto max-w-[640px] text-sm">
        <div className="flex flex-wrap gap-[3px] items-end">
            {toks.map((p, i) => <div key={i} className={'flex flex-col items-center rounded border border-gray-300 bg-white px-1 py-0.5 min-w-[2.2rem]' + (p === '[PAD]' ? ' opacity-40' : '')}>
                <div style={{ color: p === '[MASK]' ? maskColor : tokColor }}>{p}</div>
                <div className="text-[10px]" style={{ color: idxColor }}>{model!.inputBuf[i]}</div>
                <div className="text-[10px]" style={{ color: segColor }}>seg {model!.segBuf[i]}</div>
            </div>)}
        </div>
        {model.truncated && <div className="mt-2 text-xs text-red-700">
            This sentence has more pieces than the model can take ({model.shape.T - 3}); the rest was cut off before the forward pass.
        </div>}
    </div>;
};

/** Top-k candidates for a masked position, straight out of mlmProbs. */
export const MaskPredictionsAtMask: React.FC = () => {
    let state = useProgramState();
    let model = state.bert.model;
    if (!model) {
        return null;
    }
    let pos = model.masked[0];
    let gold = model.maskedGold[0];
    let top = bertTopK(model, pos, 5);
    if (top.length === 0 || pos === undefined) {
        return <div className="text-sm text-gray-500 my-2">Nothing masked in the current sentence.</div>;
    }
    let max = top[0].p || 1;
    let hit = gold !== undefined && top[0].tok === model.vocab[gold];
    return <table className="my-2 mx-auto text-sm border-collapse">
        <tbody>
            {gold !== undefined && <tr>
                <td className="pr-2 text-right text-gray-500 text-xs">the piece that was hidden</td>
                <td colSpan={2} className="pl-1 text-xs">
                    <span>{model.vocab[gold]}</span>{' '}
                    <span className={hit ? 'text-green-700' : 'text-red-700'}>{hit ? '\u2014 this model got it right' : '\u2014 this model did not'}</span>
                </td>
            </tr>}
            {top.map((c, i) => <tr key={i}>
                <td className="pr-2 text-right w-24">{c.tok}</td>
                <td className="w-48"><div className="h-3 rounded-sm" style={{ width: `${Math.max(2, c.p / max * 100)}%`, background: i === 0 ? '#2f9e44' : '#7c3c8d99' }} /></td>
                <td className="pl-2 text-gray-600">{(c.p * 100).toFixed(c.p > 0.1 ? 1 : 2)}%</td>
            </tr>)}
        </tbody>
    </table>;
};

const BertResultsTable: React.FC = () => {
    let state = useProgramState();
    let m = state.bert.model?.config.metrics;
    if (!m) {
        return null;
    }
    let rows: [string, string][] = [
        ['sentiment accuracy (dev)', `${(m.sentiment_dev_acc * 100).toFixed(1)}%`],
        [`\u2026 on the ${m.n_dev_untruncated} sentences that fit whole`, `${(m.sentiment_dev_acc_untruncated * 100).toFixed(1)}%`],
        ['majority baseline', `${(m.majority_baseline * 100).toFixed(1)}%`],
        ['masked-token top-1, all masks', `${(m.mlm_top1_all * 100).toFixed(1)}%`],
        ['\u2026 on frequent whole words', `${(m.mlm_top1_frequent * 100).toFixed(1)}%`],
        ['next-sentence prediction', `${(m.nsp_dev_acc * 100).toFixed(1)}%`],
        ['dev sentences truncated at T=32', `${(m.truncation_rate * 100).toFixed(1)}%`],
    ];
    return <table className="my-2 mx-auto text-sm border-collapse">
        <tbody>
            {rows.map(([k, v]) => <tr key={k}><td className="pr-3 text-gray-600">{k}</td><td className="text-right tabular-nums">{v}</td></tr>)}
        </tbody>
    </table>;
};

const BertSizesTable: React.FC = () => {
    let state = useProgramState();
    let tiny = state.bert.layout;
    let rows = [
        ['BERT-BASE', 12, 768, 3072, 12, '110M'],
        ['BERT-LARGE', 24, 1024, 4096, 16, '340M'],
    ];
    return <table className="my-2 mx-auto text-sm border-collapse">
        <thead>
            <tr className="border-b border-gray-400">{['Model', 'Layers L', 'Hidden H', 'FFN', 'Heads A', 'Params'].map(h => <th key={h} className="px-2 font-normal text-gray-600">{h}</th>)}</tr>
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

/** Type a sentence, re-run BERT on it. Commits on Enter or after a pause \u2014 never per keystroke (\u00a78.2). */
export const SentencePicker: React.FC = () => {
    let state = useProgramState();
    let model = state.bert.model;
    let [draft, setDraft] = useState<string | null>(null);
    const [timer, setTimer] = useState<ReturnType<typeof setTimeout> | null>(null);
    if (!model) {
        return null;
    }
    let text = draft ?? model.text;
    let pred = bertSentiment(model);
    let probs = model.sentimentProbs.localBuffer;
    let classes = model.config.classes ?? ['negative', 'positive'];

    function run(text: string) {
        state.bert.pendingSentence = text;
        state.markDirty();
    }

    function onChange(next: string) {
        setDraft(next);
        if (timer) {
            clearTimeout(timer);
        }
        setTimer(setTimeout(() => { setDraft(null); run(next); }, 500));
    }

    return <div className="my-2 mx-auto max-w-[620px]">
        <div className="flex gap-2 items-center">
            <input className="flex-1 px-2 py-1 rounded border border-gray-400 bg-white text-gray-900" value={text} spellCheck={false}
                onChange={ev => onChange(ev.target.value)}
                onKeyDown={ev => { if (ev.key === 'Enter') { if (timer) clearTimeout(timer); setDraft(null); run((ev.target as HTMLInputElement).value); } }} />
        </div>
        <div className="mt-2 flex flex-wrap gap-2 items-center text-sm">
            {model.examples.slice(0, 4).map((ex, i) => <button key={i}
                className="px-3 py-1 rounded shadow bg-gray-200 hover:bg-blue-200 text-xs text-left" onClick={() => { setDraft(null); run(ex.text); }}>{ex.text}</button>)}
        </div>
        {probs && <div className="mt-2 flex items-center gap-3 text-sm">
            <div>Prediction: <span className="text-xl font-bold">{classes[pred.label]}</span> ({(pred.prob * 100).toFixed(1)}%)</div>
            {model.truncated && <span className="text-xs text-red-700">truncated \u2014 only the first {model.shape.T - 3} pieces reached the model</span>}
        </div>}
    </div>;
};
