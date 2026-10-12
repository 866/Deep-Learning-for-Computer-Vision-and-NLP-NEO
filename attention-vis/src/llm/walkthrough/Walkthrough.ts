import { addSourceDestCurveLine, blockDimension, blockIndex, drawTextOnModel, findSubBlocks, indexMappingLines, renderIndexes, splitGrid, TextAlignHoriz, TextAlignVert } from "../Annotations";
import { ICamera, ICameraPos } from "../Camera";
import { IBlkDef } from "../GptModelLayout";
import { IProgramState } from "../Program";
import { IRenderView } from "../render/modelRender";
import { drawThread } from "../render/threadRender";
import { SavedState } from "../SavedState";
import { oneHotArray } from "@/src/utils/data";
import { lerp, lerpSmoothstep } from "@/src/utils/math";
import { Dim, Vec3, Vec4 } from "@/src/utils/vector";
import { DimStyle, dimStyleColor, hideFromBlock, ICommentary, ICommentaryRes, IPhaseGroup, ITimeInfo, IVitWalkthroughArgs, IWalkthroughArgs, moveCameraTo, phaseTools } from "./WalkthroughTools";
import { walkthroughIntro } from "./Walkthrough00_Intro";
import { walkthrough01_Prelim } from "./Walkthrough01_Prelim";
import { walkthrough02_Embedding } from "./Walkthrough02_Embedding";
import { walkthrough03_SelfAttention } from "./Walkthrough03_SelfAttention";
import { walkthrough04_Softmax } from "./Walkthrough04_Softmax";
import { walkthrough05_Projection } from "./Walkthrough05_Projection";
import { walkthrough06_AddNorm } from "./Walkthrough06_AddNorm";
import { walkthrough07_FeedForward } from "./Walkthrough07_FeedForward";
import { walkthrough08_EncoderStack } from "./Walkthrough08_EncoderStack";
import { walkthrough09_MaskedAttention } from "./Walkthrough09_MaskedAttention";
import { walkthrough10_CrossAttention } from "./Walkthrough10_CrossAttention";
import { walkthrough11_DecoderStack } from "./Walkthrough11_DecoderStack";
import { walkthrough12_Output } from "./Walkthrough12_Output";
import { walkthroughVit01_Intro, walkthroughVit02_Embedding, walkthroughVit03_Encoder, walkthroughVit04_Head, walkthroughVit05_Attention } from "./WalkthroughVit";

/**

Thoughts about the walkthrough:

- Linking text events to the visual actions doesn't work well.
- Attention needs to flip between them directly. Much better to do a chunk of text, then [Spacebar], then action.
  Can still do aligned highlights though, but focused on the model, and linking back to the text.
  This also works well with hover on either. The color-coding helps a lot anyway.

- Better to do all this text in html, it adds searchability, and supports more things. _However_, probably
  difficult to do TeX fonts. Although maybe that's a bit excessive anyway.

- Can show a reasonable amount of text: Limit to half a screen on mobile, say.

- Fast but smooth transitions, with a pause in between, is much better.
- Slow, linked transitions is kinda nauseating.

- Need more controls! Mainly jumping between phases, and selecting times on phases.
- Can compute the latest time on each phase by keeping track of the events. Can keep this in an array for
  display as well.
- Need to push back into react-land at the end of each frame. (commentary; stats; etc)
- Probably need to query all the phases up front somehow. Could have a pass where we don't actually
  update anything, but read off info for each one. A pain for the in-place updates though. Likely to break
  things. Easier to just have a list of them and info like titles.
- Hmm, do want to expand to total time, hmm.

- Need a bunch more annotations/effects yet
  - Get the trails working well
  - Want a descriptor of how a cell is computed (x, y from these; mul by a, b from these, plus this; all added, [t])
  - It applies to everything, and could allow for 'draw for this 1 cell', 'draw for this col/row of cells'.
  - Curved arrows and flows with thickness
    - Arrows for showing embedding mapping, and for linking src & sink
    - Flows for the residual pathway: wide transparent no-width path, with glowing more-solid sides
      - Basically wide arrows
      - Also need side bit coming off of it for when it's copied
      - Not as wide as the actual block; keep it maybe half-T thick
      - Some indication of flow
  - Highlight block or row/column

 */


export type IWalkthrough = ReturnType<typeof initWalkthrough>;

export function initWalkthrough() {
    return {
        phase: SavedState.state?.phase ?? Phase.Intro_Intro,
        time: SavedState.state?.phaseTime ?? 0,
        viewDt: 0,
        dt: 0,
        prevTime: 0,
        prevPhase: Phase.None,
        running: false,
        speed: 1,
        cameraInitial: null as ICameraPos | null,
        commentary: null as ICommentaryRes | null,
        times: [] as (ITimeInfo | ICommentary)[],
        phaseLength: 0,
        dimHighlightBlocks: null as IBlkDef[] | null,
        markDirty: () => { }, // bit of a hack to get it to WalkthroughSidebar
        phaseData: new Map<Phase, any>(),
        phaseTransitiveData: null as any,
        phaseList: [{
            groupId: PhaseGroup.Intro,
            title: 'Introduction',
            phases: [
                { id: Phase.Intro_Intro, title: 'Overview' },
                { id: Phase.Intro_Prelim, title: 'Preliminaries' },
            ],
        }, {
            groupId: PhaseGroup.Encoder,
            title: 'Encoder',
            phases: [
                { id: Phase.Enc_Embedding, title: 'Embedding & Positional Encoding' },
                { id: Phase.Enc_SelfAttention, title: 'Self-Attention' },
                { id: Phase.Enc_Softmax, title: 'Softmax' },
                { id: Phase.Enc_Projection, title: 'Multi-Head Attention' },
                { id: Phase.Enc_AddNorm, title: 'Add & Norm' },
                { id: Phase.Enc_FeedForward, title: 'Feed-Forward Network' },
                { id: Phase.Enc_Stack, title: 'The Encoder Stack' },
            ],
        }, {
            groupId: PhaseGroup.Decoder,
            title: 'Decoder',
            phases: [
                { id: Phase.Dec_MaskedAttention, title: 'Masked Self-Attention' },
                { id: Phase.Dec_CrossAttention, title: 'Cross-Attention' },
                { id: Phase.Dec_Stack, title: 'The Decoder Stack' },
                { id: Phase.Dec_Output, title: 'Output & Decoding' },
            ],
        }, {
            groupId: PhaseGroup.Vit,
            title: 'Vision Transformer',
            phases: [
                { id: Phase.Vit_Intro, title: 'ViT Overview' },
                { id: Phase.Vit_Embedding, title: 'Patch & Position Embeddings' },
                { id: Phase.Vit_Encoder, title: 'The Encoder (Pre-Norm)' },
                { id: Phase.Vit_Head, title: 'Classification Head' },
                { id: Phase.Vit_Attention, title: 'Attention Maps & Scale' },
            ],
        }, {
            groupId: PhaseGroup.Bert,
            title: 'BERT',
            phases: [
                { id: Phase.Bert_Intro, title: 'BERT Overview' },
                { id: Phase.Bert_Input, title: 'Tokens, Segments & Positions' },
                { id: Phase.Bert_Attn, title: 'Bidirectional Attention' },
                { id: Phase.Bert_Mlm, title: 'Masked Language Model' },
                { id: Phase.Bert_PretrainFT, title: 'Pre-training & Fine-tuning' },
                { id: Phase.Bert_Sentiment, title: 'Sentiment Classifier' },
            ],
        }] as IPhaseGroup[],
    };
}

interface ICameraData {
    initialCaptured?: ICameraPos;
    target: ICameraPos;
}

export enum PhaseGroup {
    Intro,
    Encoder,
    Decoder,
    Vit,
    Bert,
}

export enum Phase {
    None,

    Intro_Intro,
    Intro_Prelim,

    Enc_Embedding,
    Enc_SelfAttention,
    Enc_Softmax,
    Enc_Projection,
    Enc_AddNorm,
    Enc_FeedForward,
    Enc_Stack,

    Dec_MaskedAttention,
    Dec_CrossAttention,
    Dec_Stack,
    Dec_Output,

    // the Vision Transformer chapters (these show the tiny ViT instead of the tiny Transformer)
    Vit_Intro,
    Vit_Embedding,
    Vit_Encoder,
    Vit_Head,
    Vit_Attention,

    // The BERT chapters. Appended, never inserted: SavedState writes the numeric phase id to
    // localStorage, so renumbering existing ids sends every returning reader to a different chapter.
    Bert_Intro,
    Bert_Input,
    Bert_Attn,
    Bert_Mlm,
    Bert_PretrainFT,
    Bert_Sentiment,
}

/** Which tiny model a chapter shows at the origin. */
export type Family = 'transformer' | 'vit' | 'bert';

// Set membership rather than `phase >= Phase.Vit_Intro`: an ordinal range claims every phase added
// after it, so the first BERT phase would have drawn the ViT while its text described BERT.
const VIT_PHASES = new Set<Phase>([
    Phase.Vit_Intro, Phase.Vit_Embedding, Phase.Vit_Encoder, Phase.Vit_Head, Phase.Vit_Attention,
]);
const BERT_PHASES = new Set<Phase>([
    Phase.Bert_Intro, Phase.Bert_Input, Phase.Bert_Attn, Phase.Bert_Mlm, Phase.Bert_PretrainFT, Phase.Bert_Sentiment,
]);

export function familyOfPhase(phase: Phase): Family {
    if (BERT_PHASES.has(phase)) {
        return 'bert';
    }
    if (VIT_PHASES.has(phase)) {
        return 'vit';
    }
    return 'transformer';
}

export function isVitPhase(phase: Phase) {
    return VIT_PHASES.has(phase);
}

export function isBertPhase(phase: Phase) {
    return BERT_PHASES.has(phase);
}

export function phaseToGroup(wt: IWalkthrough) {
    return wt.phaseList.find(g => g.phases.find(p => p.id === wt.phase))!;
}


export function runWalkthrough(state: IProgramState, view: IRenderView) {
    let wt = state.walkthrough;
    wt.viewDt = view.dt;

    if (wt.running) {
        let dtSeconds = view.dt / 1000 * wt.speed;
        wt.time += dtSeconds;
        wt.dt = dtSeconds;

        if (wt.time > wt.phaseLength) {
            wt.running = false;
            wt.time = wt.phaseLength;
        }

        view.markDirty();
    }

    SavedState.state = { phase: wt.phase, phaseTime: wt.time, camera: state.camera };

    if (wt.prevPhase !== wt.phase) {
        wt.phaseTransitiveData = null;
    }

    wt.cameraInitial = null;
    wt.times = [];
    wt.phaseLength = 0;
    wt.dimHighlightBlocks = null;

    let wtArgs: IWalkthroughArgs = { state, layout: state.layout, tools: phaseTools(state), walkthrough: wt };

    walkthroughIntro(wtArgs);
    walkthrough01_Prelim(wtArgs);
    walkthrough02_Embedding(wtArgs);
    walkthrough03_SelfAttention(wtArgs);
    walkthrough04_Softmax(wtArgs);
    walkthrough05_Projection(wtArgs);
    walkthrough06_AddNorm(wtArgs);
    walkthrough07_FeedForward(wtArgs);
    walkthrough08_EncoderStack(wtArgs);
    walkthrough09_MaskedAttention(wtArgs);
    walkthrough10_CrossAttention(wtArgs);
    walkthrough11_DecoderStack(wtArgs);
    walkthrough12_Output(wtArgs);

    let vitArgs: IVitWalkthroughArgs = { ...wtArgs, layout: state.vit.layout };
    walkthroughVit01_Intro(vitArgs);
    walkthroughVit02_Embedding(vitArgs);
    walkthroughVit03_Encoder(vitArgs);
    walkthroughVit04_Head(vitArgs);
    walkthroughVit05_Attention(vitArgs);

    wt.prevPhase = wt.phase;
    wt.prevTime = wt.time;
}
