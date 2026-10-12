import { cameraForBounds, genModelViewMatrices, ICamera, ICameraPos, layoutBounds, stackBounds, updateCamera } from "./Camera";
import { drawAllArrows, drawBertArrows, drawVitArrows } from "./components/Arrow";
import { drawBertBlockLabels, drawBlockLabels, drawVitBlockLabels } from "./components/SectionLabels";
import { drawBertModelCard, drawModelCard, drawVitModelCard } from "./components/ModelCard";
import { IModelShape, ITransformerModelLink, setDecodedLength, setSourceSequence } from "./TransformerModel";
import { genGptModelLayout, IBlkDef, IGptModelLayout } from "./GptModelLayout";
import { drawText, IFontAtlasData, IFontOpts, measureText } from "./render/fontRender";
import { initRender, IRenderState, IRenderView, renderModel, resetRenderBuffers } from "./render/modelRender";
import { beginQueryAndGetPrevMs, endQuery } from "./render/queryManager";
import { SavedState } from "./SavedState";
import { isNotNil } from "@/src/utils/data";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { familyOfPhase, Family, initWalkthrough, Phase, runWalkthrough } from "./walkthrough/Walkthrough";
import { createBertModel, IBertModelLink, IBertShape, runBertModel, setBertSentence } from "./BertModel";
import { genBertModelLayout, IBertModelLayout } from "./BertModelLayout";
import { IVitModelLink, IVitShape, setVitImage } from "./VitModel";
import { genVitModelLayout, IVitModelLayout } from "./VitModelLayout";
import { IColorMix } from "./Annotations";
import { Mat4f } from "@/src/utils/matrix";
import { runMouseHitTesting } from "./Interaction";
import { RenderPhase } from "./render/sharedRender";
import { drawBlockInfo } from "./components/BlockInfo";
import { IMovementInfo } from "./components/MovementControls";
import { disposeBlockRender, IBlockRender, initBlockRender } from "./render/blockRender";
import { ILayout } from "../utils/layout";
import { DimStyle } from "./walkthrough/WalkthroughTools";
import { Subscriptions } from "../utils/hooks";
import { ensureShadersReady } from "../utils/shader";

/** Which tiny model is shown at the origin: it follows the walkthrough chapter. */
export type MainKind = Family;
export type { Family };
export type IAnyModelLayout = IGptModelLayout | IVitModelLayout | IBertModelLayout;

export interface IVitState {
    shape: IVitShape;
    model: IVitModelLink | null;
    layout: IVitModelLayout;
    pendingImage: { image: Float32Array, label: number | null } | null; // a new input image to run on the next frame
}

export interface IBertState {
    shape: IBertShape;
    model: IBertModelLink | null;
    layout: IBertModelLayout;
    pendingSentence: string | null; // a new sentence to tokenise & run on the next frame
}

export interface IProgramState {
    pendingSource: number[] | null; // a new source sequence to encode & decode on the next frame
    mouse: IMouseState;
    render: IRenderState;
    inWalkthrough: boolean;
    walkthrough: ReturnType<typeof initWalkthrough>;
    camera: ICamera;
    htmlSubs: Subscriptions;
    layout: IGptModelLayout;       // the tiny encoder-decoder Transformer (only regenerated while it's the main model)
    vit: IVitState;                // the tiny ViT
    bert: IBertState;              // the tiny BERT
    mainKind: MainKind;
    mainLayout: IAnyModelLayout;   // the layout that's rendered at the origin, hovered & animated
    mainExample: IModelExample;    // the tiny Transformer's card & camera
    vitMainExample: IModelExample; // the tiny ViT's card & camera
    bertMainExample: IModelExample; // the tiny BERT's card & camera
    examples: IModelExample[];
    currExampleId: number;
    shape: IModelShape;
    model: ITransformerModelLink | null;
    movement: IMovementInfo;
    display: IDisplayState;
    pageLayout: ILayout;
    markDirty: () => void;
}

export interface IModelExample {
    name: string;
    family: MainKind;
    shape: IModelShape | IVitShape;
    enabled: boolean;
    // only allocated while the example is enabled (see setExampleEnabled)
    layout?: IAnyModelLayout;
    blockRender: IBlockRender | null;
    offset: Vec3;
    modelCardOffset: Vec3;
    camera?: ICameraPos;
}

export interface IMouseState {
    mousePos: Vec3;
}

export interface IDisplayState {
    tokenColors: IColorMix | null;
    tokenIdxColors: IColorMix | null;
    tokenOutputColors: IColorMix | null;
    decTokenColors?: IColorMix | null;
    decTokenIdxColors?: IColorMix | null;
    tokenIdxModelOpacity?: number[];
    topOutputOpacity?: number;
    lines: string[];
    hoverTarget: IHoverTarget | null;
    blkIdxHover: number[] | null;
    dimHover: DimStyle | null;
}

export interface IHoverTarget {
    subCube: IBlkDef;
    mainCube: IBlkDef;
    mainIdx: Vec3;
}

export function initProgramState(canvasEl: HTMLCanvasElement, fontAtlasData: IFontAtlasData): IProgramState {

    let render = initRender(canvasEl, fontAtlasData);
    let walkthrough = initWalkthrough();

    let prevState = SavedState.state;
    let camera: ICamera = {
        angle: prevState?.camera.angle ?? new Vec3(296, 16, 13.5),
        center: prevState?.camera.center ?? new Vec3(-8.4, 0, -481.5),
        transition: {},
        modelMtx: new Mat4f(),
        viewMtx: new Mat4f(),
        lookAtMtx: new Mat4f(),
        camPos: new Vec3(),
        camPosModel: new Vec3(),
    }

    // the tiny model trained by training/train.py (gets replaced by the actual config once loaded)
    let shape: IModelShape = {
        B: 1,
        T: 6,
        C: 48,
        nHeads: 3,
        A: 48 / 3,
        nBlocks: 2,
        vocabSize: 4,
        dFF: 4 * 48,
    };

    // the two configurations from the paper (Table 3); vocab is the shared 37k BPE vocabulary for EN-DE,
    // and we pick a sequence length of 64 tokens for display
    let transformerBase: IModelShape = {
        B: 1,
        T: 64,
        C: 512,
        nHeads: 8,
        A: 512 / 8,
        nBlocks: 6,
        vocabSize: 37000,
        dFF: 2048,
    };

    let transformerBig: IModelShape = {
        B: 1,
        T: 64,
        C: 1024,
        nHeads: 16,
        A: 1024 / 16,
        nBlocks: 6,
        vocabSize: 37000,
        dFF: 4096,
    };

    // the tiny ViT trained by training/train_vit.py (replaced by the actual config once loaded)
    let vitShape: IVitShape = {
        B: 1, T: 17, C: 48, nHeads: 3, A: 16, nBlocks: 2, dFF: 192, nClasses: 10,
        imageSize: 8, patchSize: 2, channels: 1, patchDim: 4,
    };

    // the tiny BERT trained by training/train_bert.py (replaced by the actual config once loaded)
    let bertShape: IBertShape = {
        B: 1, T: 32, C: 128, nHeads: 2, A: 64, nBlocks: 2, dFF: 512, vocabSize: 600, nSegments: 2,
    };

    // the ViT paper's models (Table 1), at 224 x 224 resolution, fine-tuned on ImageNet's 1000 classes
    let vitPaper = (C: number, nBlocks: number, nHeads: number, dFF: number, patchSize: number): IVitShape => {
        let n = (224 / patchSize) ** 2;
        return {
            B: 1, T: n + 1, C, nHeads, A: C / nHeads, nBlocks, dFF, nClasses: 1000,
            imageSize: 224, patchSize, channels: 3, patchDim: patchSize * patchSize * 3,
        };
    };

    let mainLayout = genGptModelLayout(shape);
    let mainBB = layoutBounds(mainLayout);
    let vitLayout = genVitModelLayout(vitShape);
    let vitBB = layoutBounds(vitLayout);
    let bertLayout = genBertModelLayout(bertShape);
    let bertBB = layoutBounds(bertLayout);
    let startFamily = familyOfPhase(walkthrough.phase);
    if (!prevState) {
        let initCam = cameraForBounds(startFamily === 'vit' ? vitBB : startFamily === 'bert' ? bertBB : mainBB);
        camera.angle = initCam.angle;
        camera.center = initCam.center;
    }

    let examples: IModelExample[] = [];
    let leftEdge = Math.min(mainBB.min.x, vitBB.min.x, bertBB.min.x);
    let rightEdge = Math.max(mainBB.max.x, vitBB.max.x, bertBB.max.x);
    let bertPaper = (C: number, nBlocks: number, nHeads: number, dFF: number, vocabSize: number): IBertShape => ({
        B: 1, T: 128, C, nHeads, A: C / nHeads, nBlocks, dFF, vocabSize, nSegments: 2,
    });

    let exampleDefs: [string, MainKind, IModelShape | IVitShape | IBertShape, number][] = [
        ['Transformer (base)', 'transformer', transformerBase, -1],
        ['Transformer (big)', 'transformer', transformerBig, 1],
        ['ViT-Base/16', 'vit', vitPaper(768, 12, 12, 3072, 16), 1],
        ['ViT-Large/16', 'vit', vitPaper(1024, 24, 16, 4096, 16), 1],
        ['ViT-Huge/14', 'vit', vitPaper(1280, 32, 16, 5120, 14), 1],
        // the paper's own two sizes (§3): the shapes are real, the weights are not — these are
        // structure-only comparisons, same as the Transformer (base) / ViT-Base examples beside them
        ['BERT-Base', 'bert', bertPaper(768, 12, 12, 3072, 30522), -1],
        ['BERT-Large', 'bert', bertPaper(1024, 24, 16, 4096, 30522), -1],
    ];
    for (let [name, family, exShape, side] of exampleDefs) {
        let layout = genLayoutFor(family, exShape);
        let bb = layoutBounds(layout);
        let gap = 2000;
        let offsetX = side < 0 ? leftEdge - gap - bb.max.x : rightEdge + gap - bb.min.x;
        if (side < 0) {
            leftEdge = offsetX + bb.min.x;
        } else {
            rightEdge = offsetX + bb.max.x;
        }
        let offset = new Vec3(offsetX, 0, 0);
        // the layout is only needed here for placement; it (and the GL resources) get created when the example is enabled
        examples.push({
            name,
            family,
            enabled: false,
            shape: exShape,
            offset,
            modelCardOffset: new Vec3(layout.midX, 0, 0),
            blockRender: null,
            camera: cameraForBounds(stackBounds(layout), offset),
        });
    }

    return {
        pendingSource: null,
        render: render!,
        inWalkthrough: true,
        walkthrough,
        camera,
        shape: shape,
        layout: mainLayout,
        vit: { shape: vitShape, model: null, layout: vitLayout, pendingImage: null },
        bert: { shape: bertShape, model: null, layout: bertLayout, pendingSentence: null },
        mainKind: startFamily,
        mainLayout: startFamily === 'vit' ? vitLayout : startFamily === 'bert' ? bertLayout : mainLayout,
        currExampleId: -1,
        mainExample: {
            name: 'Tiny Transformer',
            family: 'transformer',
            enabled: true,
            shape: shape,
            offset: new Vec3(),
            modelCardOffset: new Vec3(mainLayout.midX, 0, 0),
            blockRender: null!,
            camera: cameraForBounds(mainBB, new Vec3()),
        },
        vitMainExample: {
            name: 'Tiny ViT',
            family: 'vit',
            enabled: true,
            shape: vitShape,
            offset: new Vec3(),
            modelCardOffset: new Vec3(),
            blockRender: null!,
            camera: cameraForBounds(vitBB, new Vec3()),
        },
        bertMainExample: {
            name: 'Tiny BERT',
            family: 'bert',
            enabled: true,
            shape: bertShape,
            offset: new Vec3(),
            modelCardOffset: new Vec3(),
            blockRender: null!,
            camera: cameraForBounds(bertBB, new Vec3()),
        },
        examples,
        model: null,
        markDirty: () => { },
        htmlSubs: new Subscriptions(),
        mouse: {
            mousePos: new Vec3(),
        },
        movement: {
            action: null,
            actionHover: null,
            target: [0, 0],
            depth: 1,
            cameraLerp: null,
         },
        display: {
            tokenColors: null,
            tokenIdxColors: null,
            tokenOutputColors: null,
            lines: [],
            hoverTarget: null,
            dimHover: null,
            blkIdxHover: null,
        },
        pageLayout: {
            height: 0,
            width: 0,
            isDesktop: true,
            isPhone: true,
        }
    };
}

export function runProgram(view: IRenderView, state: IProgramState) {
    let timer0 = performance.now();

    if (!state.render) {
        return;
    }

    resetRenderBuffers(state.render);
    state.render.sharedRender.activePhase = RenderPhase.Opaque;
    state.display.lines = [];
    state.display.hoverTarget = null;
    state.display.tokenColors = null;
    state.display.tokenIdxColors = null;
    state.display.decTokenColors = null;
    state.display.decTokenIdxColors = null;

    if (state.pendingSource && state.model) {
        setSourceSequence(state.model, state.pendingSource);
        state.pendingSource = null;
    }

    if (state.vit.pendingImage && state.vit.model) {
        setVitImage(state.vit.model, state.vit.pendingImage.image, state.vit.pendingImage.label);
        state.vit.pendingImage = null;
    }

    if (state.bert.pendingSentence && state.bert.model) {
        // one forward pass per committed edit, never per frame (spec §8.2); it costs ~15 ms because it
        // runs the MLM head over the whole vocabulary at every position
        setBertSentence(state.bert.model, state.bert.pendingSentence);
        runBertModel(state.bert.model);
        state.bert.pendingSentence = null;
    }

    // only the decoding chapter shows partially-decoded outputs
    if (state.model && state.walkthrough.phase !== Phase.Dec_Output && state.model.inputLen < state.shape.T) {
        setDecodedLength(state.model, state.shape.T);
    }

    // generate the main model's layout (the walkthrough modifies it each frame), incorporating its data if available
    state.mainKind = familyOfPhase(state.walkthrough.phase);
    if (state.mainKind === 'vit') {
        state.vit.layout = genVitModelLayout(state.vit.shape, state.vit.model);
        state.mainLayout = state.vit.layout;
    } else if (state.mainKind === 'bert') {
        state.bert.layout = genBertModelLayout(state.bert.shape, state.bert.model);
        state.mainLayout = state.bert.layout;
    } else {
        state.layout = genGptModelLayout(state.shape, state.model);
        state.mainLayout = state.layout;
    }

    genModelViewMatrices(state, state.mainLayout);

    let queryRes = beginQueryAndGetPrevMs(state.render.queryManager, 'render');
    if (isNotNil(queryRes)) {
        state.render.lastGpuMs = queryRes;
    }

    state.render.renderTiming = false; // state.pageLayout.isDesktop;

    // will modify layout; view; render a few things.
    if (state.inWalkthrough) {
        runWalkthrough(state, view);
    }

    updateCamera(state, view);

    drawBlockInfo(state);
    // these will get modified by the walkthrough (stored where?)
    drawLayoutArrows(state.render, state.mainLayout);

    for (let example of state.examples) {
        if (example.enabled && example.layout) {
            drawArrowsWithOffset(state.render, example.layout, example.offset);
        }
    }

    let mainEx = currentMainExample(state);
    FAMILIES[state.mainLayout.kind].drawCard(state, state.mainLayout, mainEx.name, mainEx.modelCardOffset, true);

    for (let example of state.examples) {
        if (example.enabled && example.layout) {
            let offset = example.offset.add(example.modelCardOffset);
            FAMILIES[example.layout.kind].drawCard(state, example.layout, example.name, offset, false);
        }
    }

    // manageMovement(state, view);
    runMouseHitTesting(state);
    state.render.sharedRender.activePhase = RenderPhase.Opaque;
    FAMILIES[state.mainLayout.kind].drawLabels(state.render, state.mainLayout);

    let lineNo = 1;
    let tw = state.render.size.x;
    state.render.sharedRender.activePhase = RenderPhase.Overlay2D;
    for (let line of state.display.lines) {
        let opts: IFontOpts = { color: new Vec4(), size: 14 };
        let w = measureText(state.render.modelFontBuf, line, opts);
        drawText(state.render.modelFontBuf, line, tw - w - 4, lineNo * opts.size * 1.3 + 4, opts)
        lineNo++;
    }

    // render everything; i.e. here's where we actually do gl draw calls
    // up until now, we've just been putting data in cpu-side buffers
    renderModel(state);

    endQuery(state.render.queryManager, 'render');
    state.render.gl.flush();

    state.render.lastJsMs = performance.now() - timer0;
}

/** The arrow helpers emit world-space vertices, so draw the example's arrows in its own layout space and then
 * shift the newly added vertices (position is the first 3 floats of each vertex in both buffers) by its offset. */
function drawArrowsWithOffset(render: IRenderState, layout: IAnyModelLayout, offset: Vec3) {
    let bufs = [render.triRender.vbo.localBufs[0], render.lineRender.floatBuf.localBufs[0]];
    let starts = bufs.map(b => b.usedEls);
    drawLayoutArrows(render, layout);
    bufs.forEach((b, i) => {
        for (let el = starts[i]; el < b.usedEls; el++) {
            let off = el * b.strideFloats;
            b.buf[off + 0] += offset.x;
            b.buf[off + 1] += offset.y;
            b.buf[off + 2] += offset.z;
        }
    });
}

/**
 * Everything the renderer needs to know about a model family, in one place. Adding BERT used to mean
 * finding each `layout.kind === 'vit' ? … : …` ladder and inserting a rungs — and a missed one silently
 * draws the Transformer's arrows under a BERT layout.
 */
const FAMILIES: Record<Family, {
    genLayout: (shape: any) => IAnyModelLayout;
    drawArrows: (render: IRenderState, layout: any) => void;
    drawCard: (state: IProgramState, layout: any, name: string, offset: Vec3, isMain: boolean) => void;
    drawLabels: (render: IRenderState, layout: any) => void;
}> = {
    transformer: {
        genLayout: shape => genGptModelLayout(shape as IModelShape),
        drawArrows: (render, layout) => drawAllArrows(render, layout),
        drawCard: (state, layout, name, offset) => drawModelCard(state, layout, name, offset),
        drawLabels: (render, layout) => drawBlockLabels(render, layout),
    },
    vit: {
        genLayout: shape => genVitModelLayout(shape as IVitShape),
        drawArrows: (render, layout) => drawVitArrows(render, layout),
        drawCard: (state, layout, name, offset, isMain) => drawVitModelCard(state, layout, name, offset, isMain),
        drawLabels: (render, layout) => drawVitBlockLabels(render, layout),
    },
    bert: {
        genLayout: shape => genBertModelLayout(shape as IBertShape),
        drawArrows: (render, layout) => drawBertArrows(render, layout),
        drawCard: (state, layout, name, offset, isMain) => drawBertModelCard(state, layout, name, offset, isMain),
        drawLabels: (render, layout) => drawBertBlockLabels(render, layout),
    },
};

function drawLayoutArrows(render: IRenderState, layout: IAnyModelLayout) {
    FAMILIES[layout.kind].drawArrows(render, layout);
}

function genLayoutFor(family: MainKind, shape: IModelShape | IVitShape | IBertShape): IAnyModelLayout {
    return FAMILIES[family].genLayout(shape);
}

/** The card & camera of the tiny model currently at the origin. */
export function currentMainExample(state: IProgramState) {
    return state.mainKind === 'vit' ? state.vitMainExample
        : state.mainKind === 'bert' ? state.bertMainExample
        : state.mainExample;
}

/** Turns one of the paper-sized example models on or off. Off frees its layout and GL resources, so a hidden model
 * costs nothing per frame. */
export function setExampleEnabled(state: IProgramState, example: IModelExample, enabled: boolean) {
    if (example.enabled === enabled) {
        return;
    }
    example.enabled = enabled;
    let ctx = state.render?.ctx;

    if (enabled) {
        example.layout = genLayoutFor(example.family, example.shape);
        if (ctx) {
            example.blockRender = initBlockRender(ctx);
            // these shaders are created after initRender linked everything else
            ensureShadersReady(ctx.shaderManager);
        }
    } else {
        if (ctx && example.blockRender) {
            disposeBlockRender(ctx, example.blockRender);
        }
        example.blockRender = null;
        example.layout = undefined;

        if (state.examples[state.currExampleId] === example) {
            state.currExampleId = -1;
            state.camera.desiredCamera = currentMainExample(state).camera;
        }
    }
    state.markDirty();
}
