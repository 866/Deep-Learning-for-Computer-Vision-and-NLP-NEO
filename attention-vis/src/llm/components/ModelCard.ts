import { cameraToMatrixView } from "../Camera";
import { cellPosition, IGptModelLayout, IModelLayout, IStackLayout } from "../GptModelLayout";
import { IBertModelLayout } from "../BertModelLayout";
import { bertTopK } from "../BertModel";
import { IVitModelLayout } from "../VitModelLayout";
import { IProgramState } from "../Program";
import { drawText, IFontOpts, measureText, measureTextWidth, writeTextToBuffer } from "../render/fontRender";
import { addLine, addLine2 as drawLine, drawLineSegs, ILineOpts, makeLineOpts } from "../render/lineRender";
import { IRenderState } from "../render/modelRender";
import { addQuad } from "../render/triRender";
import { lerp } from "@/src/utils/math";
import { Mat4f } from "@/src/utils/matrix";
import { Dim, Vec3, Vec4 } from "@/src/utils/vector";
import { DimStyle, dimStyleColor } from "../walkthrough/WalkthroughTools";
import { lineHeight } from "./TextLayout";
import { IColorMix } from "../Annotations";
import { clamp } from "@/src/utils/data";

export function drawModelCard(state: IProgramState, layout: IGptModelLayout, title: string, offset: Vec3) {
    let { nBlocks, C, nHeads, vocabSize } = layout.shape;
    let shapeText = `N = ${nBlocks},  d_model = ${C},  h = ${nHeads},  d_ff = ${layout.shape.dFF},  n_vocab = ${vocabSize}`;
    drawCardFrame(state, title, layout.weightCount, shapeText, offset, layout);

    if (layout === state.mainLayout) {
        renderOutputAtBottom(state);
        renderInputAtTop(state);
    }
}

/** The ViT's card; the main (tiny) one also shows its input image and the class probabilities. */
export function drawVitModelCard(state: IProgramState, layout: IVitModelLayout, title: string, offset: Vec3, isMain: boolean) {
    let { nBlocks, C, nHeads, dFF, patchSize, imageSize, nClasses, T } = layout.shape;
    let shapeText = `L = ${nBlocks},  D = ${C},  heads = ${nHeads},  MLP = ${dFF},  ${imageSize}x${imageSize} image,  ${patchSize}x${patchSize} patches (N = ${T - 1}),  ${nClasses} classes`;
    drawCardFrame(state, title, layout.weightCount, shapeText, offset, layout);

    if (isMain && layout.model) {
        renderVitImageAtTop(state, layout);
        renderVitClassProbs(state, layout);
    }
}

function drawCardFrame(state: IProgramState, title: string, weightCount: number, shapeText: string, offset: Vec3, layout: IModelLayout, extra?: string[]) {
    let { render } = state;
    let { camPos } = cameraToMatrixView(state.camera);
    // the card's own position in world space (model (x, y, z) => world (x, -z, -y))
    let dist = camPos.dist(new Vec3(offset.x, 0, -30));

    // grows as the camera moves away, but no wider than its model, so neighbouring cards don't overlap
    let width = layoutWidth(layout);
    let scale = clamp(dist / 500.0, 1.0, Math.max(1.0, width / 110));

    let pinY = -60;
    let mtx = Mat4f.fromScaleTranslation(new Vec3(scale, scale, scale), new Vec3(0, pinY, 0).add(offset))
        .mul(Mat4f.fromTranslation(new Vec3(0, -pinY, 0)));

    let thick = 1.0 / 10.0 * scale;
    let borderColor = Vec4.fromHexColor("#555599", 0.8);
    let backgroundColor = Vec4.fromHexColor("#93c5fd", 0.3);
    let titleColor = Vec4.fromHexColor("#000000", 1.0);
    let n = new Vec3(0, 0, 1);

    let lineOpts: ILineOpts = { color: borderColor, mtx, thick, n };

    let tl = new Vec3(-52, -97, 0);
    // the frame grows downward to fit any extra lines, so the Transformer / ViT cards keep their exact geometry
    let br = new Vec3( 52, -66 - (extra?.length ?? 0) * 5, 0);
    drawLineRect(render, tl, br, lineOpts);

    addQuad(render.triRender, new Vec3(tl.x, tl.y, -0.1), new Vec3(br.x, br.y, -0.1), backgroundColor, mtx);

    // let w = measureTextWidth(state.modelFontBuf, title, .0);
    let midX = (tl.x + br.x) / 2;
    let paramLeft = br.x - 50;
    let paramOff = tl.y + 2;

    let paramLineHeight = 1.3;
    let paramFontScale = 4;
    let numWidth = paramFontScale * 0.6;
    let paramHeight = 2 + paramLineHeight * paramFontScale * 3 + 1;

    let titleFontScale = 13;
    let titleW = measureTextWidth(render.modelFontBuf, title, titleFontScale);
    let titleHeight = titleFontScale * paramLineHeight;
    writeTextToBuffer(render.modelFontBuf, title, titleColor, midX - titleW / 2, tl.y + 2, titleFontScale, mtx);

    let nParamsText = `n_params = `;
    let weightCountText = numberToCommaSep(weightCount);

    let weightSize = 8;
    let weightTitleW = measureTextWidth(render.modelFontBuf, nParamsText, paramFontScale);
    let weightCountW = measureTextWidth(render.modelFontBuf, weightCountText, weightSize);
    // let infoText = "goal: sort 6 letters from { A, B, C } into ascending order";
    // writeTextToBuffer(render.modelFontBuf, infoText, titleColor, tl.x + 2, tl.y + paramHeight + 2, 4, mtx);

    paramOff = tl.y + titleHeight + 4;
    let weightX = midX - (weightCountW + weightTitleW) / 2;

    writeTextToBuffer(render.modelFontBuf, nParamsText, titleColor, weightX, paramOff - paramFontScale / 2, paramFontScale, mtx);
    writeTextToBuffer(render.modelFontBuf, weightCountText, titleColor, weightX + weightTitleW, paramOff - weightSize / 2, weightSize, mtx);

    let shapeW = measureTextWidth(render.modelFontBuf, shapeText, paramFontScale * 0.8);
    writeTextToBuffer(render.modelFontBuf, shapeText, titleColor.mul(0.7), midX - shapeW / 2, paramOff + weightSize * 0.75, paramFontScale * 0.8, mtx);

    // measured numbers from the checkpoint's own metrics block \u2014 the card is where a reader meets
    // "74.9% against a 50.9% majority baseline", so it has to be the shipped figures, not rounded claims
    let extraY = paramOff + weightSize * 0.75 + 3;
    for (let line of extra ?? []) {
        let w = measureTextWidth(render.modelFontBuf, line, 3.2);
        writeTextToBuffer(render.modelFontBuf, line, titleColor.mul(0.85), midX - w / 2, extraY, 3.2, mtx);
        extraY += 5;
    }
    // addParam("C (channels) = ", C.toString(), dimStyleColor(DimStyle.C));
    // addParam("T (time) = ", T.toString(), dimStyleColor(DimStyle.T));
    // addParam("B (batches) = ", B.toString(), dimStyleColor(DimStyle.B));
    // paramOff = tl.y + 2;
    // paramLeft += 35;
    // addParam("n_vocab = ", vocabSize.toString(), dimStyleColor(DimStyle.n_vocab));
    // addParam("n_layers = ", nBlocks.toString(), dimStyleColor(DimStyle.n_layers));
    // addParam("n_heads = ", nHeads.toString(), dimStyleColor(DimStyle.n_heads));

    // function addParam(name: string, value: string, color: Vec4 = borderColor) {
    //     let y = paramOff;
    //     let w = measureTextWidth(render.modelFontBuf, name, paramFontScale);
    //     let numW = measureTextWidth(render.modelFontBuf, value, paramFontScale);
    //     let left = paramLeft;
    //     writeTextToBuffer(render.modelFontBuf, name, color,  left - w        , y, paramFontScale, mtx);
    //     writeTextToBuffer(render.modelFontBuf, value, color, left + maxLen * numWidth - numW, y, paramFontScale, mtx);
    //     paramOff += paramFontScale * paramLineHeight;
    // }

    // addLine(render.lineRender, thick, borderColor, new Vec3(tl.x, tl.y + paramHeight), new Vec3(br.x, tl.y + paramHeight), n, mtx);
}

let _widthCache = new WeakMap<IModelLayout, number>();
function layoutWidth(layout: IModelLayout) {
    let w = _widthCache.get(layout);
    if (w === undefined) {
        let lo = Infinity, hi = -Infinity;
        for (let c of layout.cubes) {
            lo = Math.min(lo, c.x);
            hi = Math.max(hi, c.x + c.dx);
        }
        w = hi - lo;
        _widthCache.set(layout, w);
    }
    return w;
}

/** The input image, split into patches, above the Patches block; each patch is wired to its column (token). */
function renderVitImageAtTop(state: IProgramState, layout: IVitModelLayout) {
    let render = state.render;
    let model = layout.model!;
    let { imageSize, patchSize, T } = layout.shape;
    let patchesBlk = layout.patches;
    let g = imageSize / patchSize;

    let pixW = 3;
    let gap = 1; // between patches
    let imgW = imageSize * pixW + (g - 1) * gap;
    let topMid = new Vec3(patchesBlk.x + patchesBlk.dx / 2, patchesBlk.y - layout.margin * 2.5);
    let tl = new Vec3(topMid.x - imgW / 2, topMid.y - imgW);
    let mtx = new Mat4f();

    let lineOpts = makeLineOpts({ color: Vec4.fromHexColor("#000000", 0.25), mtx, thick: 1.5 });
    let titleTextOpts: IFontOpts = { color: Vec4.fromHexColor("#666666", 1.0), mtx, size: 1.9 };
    let numTextOpts: IFontOpts = { color: Vec4.fromHexColor("#e8590c", 1.0), mtx, size: 1.4 };

    let patchTl = (n: number) => {
        let py = Math.floor(n / g), px = n % g;
        return new Vec3(tl.x + px * (patchSize * pixW + gap), tl.y + py * (patchSize * pixW + gap));
    };

    for (let n = 0; n < g * g; n++) {
        let ptl = patchTl(n);
        let mix = state.display.tokenIdxColors ? mixColorValues(state.display.tokenIdxColors, Vec4.fromHexColor("#000000", 0.25), n + 1) : null;
        for (let iy = 0; iy < patchSize; iy++) {
            for (let ix = 0; ix < patchSize; ix++) {
                let v = model.image[(Math.floor(n / g) * patchSize + iy) * imageSize + (n % g) * patchSize + ix];
                let c = 1 - clamp(v, 0, 1) * 0.92;
                let p0 = ptl.add(new Vec3(ix * pixW, iy * pixW, 0));
                addQuad(render.triRender, p0, p0.add(new Vec3(pixW, pixW, 0)), new Vec4(c, c, c, 1));
            }
        }
        let pbr = ptl.add(new Vec3(patchSize * pixW, patchSize * pixW, 0));
        drawLineRect(render, ptl, pbr, mix ? { ...lineOpts, color: mix } : lineOpts);
        let numStr = (n + 1).toString();
        let w = measureText(render.modelFontBuf, numStr, numTextOpts);
        drawText(render.modelFontBuf, numStr, ptl.x + 0.3, ptl.y + 0.2, { ...numTextOpts, size: numTextOpts.size * (w > 2 ? 0.8 : 1) });
    }

    let label = model.imageLabel !== null ? ` (label: ${model.config.classes[model.imageLabel]})` : '';
    drawText(render.modelFontBuf, `Input image, ${imageSize}x${imageSize}, split into ${g * g} patches of ${patchSize}x${patchSize}${label}`, tl.x - 10, tl.y - lineHeight(titleTextOpts) - 1, titleTextOpts);

    // patch n -> column n + 1 of the Patches block (column 0 is the [class] token's position)
    let by = patchesBlk.y - 0.5 * layout.cell;
    let ty = tl.y + imgW + layout.cell;
    for (let n = 0; n < g * g; n++) {
        let ptl = patchTl(n);
        let tx = ptl.x + patchSize * pixW / 2;
        let pty = ptl.y + patchSize * pixW;
        let bx = cellPosition(layout, patchesBlk, Dim.X, n + 1) + 0.5 * layout.cell;
        let lo = lineOpts;
        if (state.display.tokenIdxColors) {
            lo = { ...lineOpts, color: mixColorValues(state.display.tokenIdxColors, lineOpts.color, n + 1) };
        }
        let midY = lerp(ty, by, 0.6);
        drawLine(render.lineRender, new Vec3(tx, pty), new Vec3(tx, ty), lo);
        drawLine(render.lineRender, new Vec3(tx, ty), new Vec3(bx, midY), lo);
        drawLine(render.lineRender, new Vec3(bx, midY), new Vec3(bx, by), lo);
        let arrLen = 0.6;
        drawLine(render.lineRender, new Vec3(bx - arrLen, by - arrLen), new Vec3(bx, by), lo);
        drawLine(render.lineRender, new Vec3(bx + arrLen, by - arrLen), new Vec3(bx, by), lo);
    }

    // the [class] position
    let cx = cellPosition(layout, patchesBlk, Dim.X, 0) + 0.5 * layout.cell;
    let clsText = '[class]';
    let clsOpts = { ...numTextOpts, size: 1.6 };
    let cw = measureText(render.modelFontBuf, clsText, clsOpts);
    drawText(render.modelFontBuf, clsText, cx - cw / 2, by - 3.5, clsOpts);
}

/** The size of the class-probability chart drawn beside the class-probabilities block (for framing the camera). */
export function vitProbsChartRegion(layout: IVitModelLayout) {
    let sm = layout.logitsSoftmax;
    let rowH = 4.5;
    let h = layout.shape.nClasses * rowH;
    return { x: sm.x + sm.dx + layout.margin * 1.6, y: sm.y + sm.dy / 2 - h / 2 - 2, dx: 80, dy: h + 12, rowH };
}

/** A bar chart of the class probabilities, beside the class-probabilities block. */
function renderVitClassProbs(state: IProgramState, layout: IVitModelLayout) {
    let render = state.render;
    let model = layout.model!;
    let probs = model.probs.localBuffer;
    if (!probs) {
        return;
    }
    let sm = layout.logitsSoftmax;
    let mtx = new Mat4f();
    let { nClasses } = layout.shape;
    let opacity = sm.opacity;
    let region = vitProbsChartRegion(layout);
    let { rowH } = region;
    let left = region.x;
    let top = region.y;
    let barMax = 50;
    let best = 0;
    for (let i = 1; i < nClasses; i++) if (probs[i] > probs[best]) best = i;

    // a bracket from the block to the chart
    let lineOpts = makeLineOpts({ color: Vec4.fromHexColor("#000000", 0.25 * opacity), mtx, thick: 1.5 });
    let midY = sm.y + sm.dy / 2;
    drawLine(render.lineRender, new Vec3(sm.x + sm.dx + 1, midY), new Vec3(left - 2, midY), lineOpts);

    let textOpts: IFontOpts = { color: Vec4.fromHexColor("#000000", opacity), mtx, size: 3.4 };
    for (let i = 0; i < nClasses; i++) {
        let y = top + i * rowH;
        let label = model.config.classes[i];
        drawText(render.modelFontBuf, label, left, y + 0.4, textOpts);
        let color = i === best ? Vec4.fromHexColor('#2f9e44', opacity) : Vec4.fromHexColor('#7c3c8d', 0.6 * opacity);
        let x0 = left + 4;
        addQuad(render.triRender, new Vec3(x0, y + 0.5, 0), new Vec3(x0 + Math.max(0.2, probs[i] * barMax), y + rowH - 0.5, 0), color);
        if (probs[i] > 0.005) {
            drawText(render.modelFontBuf, `${(probs[i] * 100).toFixed(probs[i] > 0.1 ? 0 : 1)}%`, x0 + probs[i] * barMax + 1, y + 0.8, { ...textOpts, size: 2.6 });
        }
    }

    let correct = model.imageLabel === null ? '' : model.imageLabel === best ? '  (correct)' : `  (label: ${model.config.classes[model.imageLabel]})`;
    let summary = `Prediction: ${model.config.classes[best]}, ${(probs[best] * 100).toFixed(1)}%${correct}`;
    drawText(render.modelFontBuf, summary, left, top + nClasses * rowH + 1.5, { ...textOpts, size: 4.2 });
}

export function tokenToString(layout: IGptModelLayout, a: number) {
    return layout.model?.vocab[a] ?? String.fromCharCode('A'.charCodeAt(0) + a);
}

export interface IInputBoxOpts {
    tokMixes?: IColorMix | null;
    idxMixes?: IColorMix | null;
}

export function renderInputBoxes(state: IProgramState, layout: IGptModelLayout, tokens: Float32Array | undefined, count: number, tl: Vec3, br: Vec3, cellW: number, fontSize: number, lineOpts: ILineOpts, opts?: IInputBoxOpts) {
    let render = state.render;
    let { T } = layout.shape;
    let inCellH = br.y - tl.y;

    let tokTextOpts: IFontOpts = { color: Vec4.fromHexColor("#000000", 1.0), mtx: lineOpts.mtx, size: fontSize };
    let idxTextOpts: IFontOpts = { color: Vec4.fromHexColor("#666666", 1.0), mtx: lineOpts.mtx, size: fontSize * 0.6 };

    let dimmedTokTextOpts: IFontOpts = { ...tokTextOpts, color: tokTextOpts.color.mul(0.3) };
    let dimmedIdxTextOpts: IFontOpts = { ...idxTextOpts, color: idxTextOpts.color.mul(0.3) };

    drawLineRect(render, tl, br, lineOpts);

    for (let i = 0; i < T; i++) {

        if (i > 0) {
            let lineX = tl.x + i * cellW;
            drawLine(render.lineRender, new Vec3(lineX, tl.y, 0), new Vec3(lineX, br.y, 0), lineOpts);
        }

        if (tokens && i < count) {
            let cx = tl.x + (i + 0.5) * cellW;

            let tokOpts = { ...tokTextOpts, color: mixColorValues(opts?.tokMixes ?? null, tokTextOpts.color, i) };
            let tokIdxOpts = { ...idxTextOpts, color: mixColorValues(opts?.idxMixes ?? null, idxTextOpts.color, i) };
            let tokStr = tokenToString(layout, tokens[i]);
            if (tokStr.length > 1) {
                tokOpts.size = tokTextOpts.size * 0.55;
            }
            let tokW = measureText(render.modelFontBuf, tokStr, tokOpts);
            let idxW = measureText(render.modelFontBuf, tokens[i].toString(), idxTextOpts);
            let totalH = tokOpts.size + idxTextOpts.size;
            let top = tl.y + (inCellH - totalH) / 2;

            drawText(render.modelFontBuf, tokStr, cx - tokW / 2, top, tokOpts);
            drawText(render.modelFontBuf, tokens[i].toString(),  cx - idxW / 2, top + tokOpts.size, tokIdxOpts);
        }

    }
}

export interface IOutputBoxOpts {
    opacity?: number;
    boldLast?: boolean;
    tokMixes?: IColorMix | null;
}

export function renderOutputBoxes(state: IProgramState, layout: IGptModelLayout, tl: Vec3, br: Vec3, cellW: number, fontSize: number, lineOpts: ILineOpts, opts?: IOutputBoxOpts) {
    let render = state.render;
    let { T, vocabSize } = layout.shape;
    let outCellH = br.y - tl.y;

    let opacity = opts?.opacity ?? 1.0;
    let boldLast = opts?.boldLast ?? true;

    lineOpts = { ...lineOpts, color: lineOpts.color.mul(opacity ?? 1.0) };
    let tokTextOpts: IFontOpts = { color: Vec4.fromHexColor("#000000", opacity), mtx: lineOpts.mtx, size: fontSize };
    let idxTextOpts: IFontOpts = { color: Vec4.fromHexColor("#666666", opacity), mtx: lineOpts.mtx, size: fontSize * 0.6 };

    let dimmedTokTextOpts: IFontOpts = { ...tokTextOpts, color: tokTextOpts.color.mul(0.3) };
    let dimmedIdxTextOpts: IFontOpts = { ...idxTextOpts, color: idxTextOpts.color.mul(0.3) };

    drawLineRect(render, tl, br, lineOpts);

    let sortedOutput = layout.model?.sortedBuf;

    for (let i = 0; i < T; i++) {
        if (i > 0) {
            let lineX = tl.x + i * cellW;
            drawLine(render.lineRender, new Vec3(lineX, tl.y, 0), new Vec3(lineX, br.y, 0), lineOpts);
        }

        if (sortedOutput && i < layout.model!.inputLen) {
            let usedSoFar = 0.0;
            let cx = tl.x + (i + 0.5) * cellW;

            for (let j = 0; j < vocabSize; j++) {
                let tokIdx = sortedOutput[(i * vocabSize + j) * 2 + 0];
                let tokProb = sortedOutput[(i * vocabSize + j) * 2 + 1];

                let partTop = tl.y + usedSoFar * outCellH;
                let partH = tokProb * outCellH;

                let dimmed = i < layout.model!.inputLen - 1 || !boldLast;

                let color = mixColorValues(opts?.tokMixes ?? null, tokTextOpts.color, i);
                if (dimmed) {
                    color = color.mul(0.3);
                }

                let tokOpts = { ...tokTextOpts, color };
                let idxOpts = { ...idxTextOpts, color: color.mul(0.6) };

                let tokStr = tokenToString(layout, tokIdx);
                let tokW = measureText(render.modelFontBuf, tokStr, tokOpts);
                let idxW = measureText(render.modelFontBuf, tokIdx.toString(), idxOpts);
                let textH = tokOpts.size + idxOpts.size;
                let top = partTop + (partH - textH) / 2;

                if (partH > textH) {
                    drawText(render.modelFontBuf, tokStr, cx - tokW / 2, top, tokOpts);
                    drawText(render.modelFontBuf, tokIdx.toString(),  cx - idxW / 2, top + tokOpts.size, idxOpts);
                }

                usedSoFar += tokProb;

                drawLine(render.lineRender, new Vec3(cx - cellW/2, partTop + partH, 0), new Vec3(cx + cellW/2, partTop + partH, 0), lineOpts);
                if (usedSoFar >= 1.0 - 1e-4) {
                    break;
                }
            }
        }
    }
}

export function mixColorValues(mixes: IColorMix | null, baseColor: Vec4, idx: number) {
    if (!mixes) {
        return baseColor;
    }
    let mix = mixes.mixes[idx] ?? 0.0;
    return Vec4.lerp(mixes.color1 ?? baseColor, mixes.color2, mix);
}

let _lineRectArr = new Float32Array(3 * 4);
export function drawLineRect(render: IRenderState, tl: Vec3, br: Vec3, opts: ILineOpts) {

    _lineRectArr[0] = tl.x;
    _lineRectArr[1] = tl.y;
    _lineRectArr[2] = 0;
    _lineRectArr[3] = br.x;
    _lineRectArr[4] = tl.y;
    _lineRectArr[5] = 0;
    _lineRectArr[6] = br.x;
    _lineRectArr[7] = br.y;
    _lineRectArr[8] = 0;
    _lineRectArr[9] = tl.x;
    _lineRectArr[10] = br.y;
    _lineRectArr[11] = 0;

    drawLineSegs(render.lineRender, _lineRectArr, makeLineOpts({ ...opts, closed: true }));
}

function numberToCommaSep(a: number) {
    let s = a.toString();
    let out = "";
    for (let i = 0; i < s.length; i++) {
        if (i > 0 && (s.length - i) % 3 == 0) {
            out += ",";
        }
        out += s[i];
    }
    return out;
}

function renderInputAtTop(state: IProgramState) {
    let layout = state.layout;
    let model = layout.model;

    renderInputAboveStack(state, layout.enc, 'Source (encoder input)', model?.enc.inputBuf, layout.shape.T, state.display.tokenColors, state.display.tokenIdxColors);
    renderInputAboveStack(state, layout.dec, 'Target, shifted right (decoder input)', model?.dec.inputBuf, model?.inputLen ?? 0, state.display.decTokenColors ?? null, state.display.decTokenIdxColors ?? null);
}

function renderInputAboveStack(state: IProgramState, stack: IStackLayout, title: string, tokens: Float32Array | undefined, count: number, tokMixes: IColorMix | null, idxMixes: IColorMix | null) {
    let layout = state.layout;
    let render = state.render;

    let inputTokBlk = stack.idxObj;

    let topMid = new Vec3(inputTokBlk.x + inputTokBlk.dx/2, inputTokBlk.y - layout.margin);

    let inCellH = 10;
    let inCellW = 6;

    let nCells = layout.shape.T;
    let tl = new Vec3(topMid.x - inCellW * nCells / 2, topMid.y - inCellH);
    let br = new Vec3(topMid.x + inCellW * nCells / 2, topMid.y);

    let lineOpts = makeLineOpts({ color: Vec4.fromHexColor("#000000", 0.2), mtx: new Mat4f(), thick: 1.5 });
    let titleTextOpts: IFontOpts = { color: Vec4.fromHexColor("#666666", 1.0), mtx: lineOpts.mtx, size: 1.9 };

    renderInputBoxes(state, layout, tokens, count, tl, br, inCellW, 4, lineOpts, { tokMixes, idxMixes });

    drawText(render.modelFontBuf, title, tl.x, tl.y - lineHeight(titleTextOpts), titleTextOpts);

    for (let i = 0; i < nCells; i++) {
        let lineOptsLocal = { ...lineOpts, color: mixColorValues(idxMixes, lineOpts.color, i) };

        let tx = tl.x + (i + 0.5) * inCellW;
        let ty = tl.y + layout.cell + inCellH;
        let bx = cellPosition(layout, inputTokBlk, Dim.X, i) + 0.5 * layout.cell;
        let by = inputTokBlk.y - 0.5 * layout.cell;

        let midY1 = lerp(by, ty, 1/6);
        let midY2 = lerp(by, ty, 3/4);

        drawLine(state.render.lineRender, new Vec3(bx, by), new Vec3(bx, midY1), lineOptsLocal);
        drawLine(state.render.lineRender, new Vec3(bx, midY1), new Vec3(tx, midY2), lineOptsLocal);
        drawLine(state.render.lineRender, new Vec3(tx, midY2), new Vec3(tx, ty), lineOptsLocal);

        let arrLen = 0.6;
        let arrowLeft = new Vec3(bx - arrLen, by - arrLen);
        let arrowRight = new Vec3(bx + arrLen, by - arrLen);
        drawLine(state.render.lineRender, arrowLeft, new Vec3(bx, by), lineOptsLocal);
        drawLine(state.render.lineRender, arrowRight, new Vec3(bx, by), lineOptsLocal);
    }
}

function renderOutputAtBottom(state: IProgramState) {
    let layout = state.layout;

    let softmax = layout.logitsSoftmax;


    let topMid = new Vec3(softmax.x + softmax.dx/2, softmax.y + softmax.dy + layout.margin);

    let outCellH = 10;
    let outCellW = 6;

    let nCells = layout.shape.T;
    let tl = new Vec3(topMid.x - outCellW * nCells / 2, topMid.y);
    let br = new Vec3(topMid.x + outCellW * nCells / 2, topMid.y + outCellH);

    let lineOpts = makeLineOpts({ color: Vec4.fromHexColor("#000000", 0.2), mtx: new Mat4f(), thick: 1.5 });

    renderOutputBoxes(state, layout, tl, br, outCellW, 4, lineOpts, { boldLast: true, tokMixes: state.display.tokenOutputColors });

    let titleTextOpts: IFontOpts = { color: Vec4.fromHexColor("#666666", 1.0), mtx: lineOpts.mtx, size: 1.9 };
    drawText(state.render.modelFontBuf, 'Output probabilities', tl.x, br.y + 1, titleTextOpts);

    for (let i = 0; i < nCells; i++) {
        let tx = cellPosition(layout, softmax, Dim.X, i) + 0.5 * layout.cell;
        let ty = softmax.y + softmax.dy + 0.5 * layout.cell;
        let bx = tl.x + (i + 0.5) * outCellW;
        let by = tl.y - layout.cell;

        let midY1 = lerp(ty, by, 1/6);
        let midY2 = lerp(ty, by, 3/4);

        drawLine(state.render.lineRender, new Vec3(tx, ty), new Vec3(tx, midY1), lineOpts);
        drawLine(state.render.lineRender, new Vec3(tx, midY1), new Vec3(bx, midY2), lineOpts);
        drawLine(state.render.lineRender, new Vec3(bx, midY2), new Vec3(bx, by), lineOpts);

        let arrLen = 0.6;
        let arrowLeft = new Vec3(bx - arrLen, by - arrLen);
        let arrowRight = new Vec3(bx + arrLen, by - arrLen);
        drawLine(state.render.lineRender, arrowLeft, new Vec3(bx, by), lineOpts);
        drawLine(state.render.lineRender, arrowRight, new Vec3(bx, by), lineOpts);
    }

}

/**
 * The BERT card: same frame as the Transformer / ViT cards, plus the measured accuracy figures
 * (they belong where the reader first sees the model) and, for the main layout, the token strip,
 * the [MASK] candidates and the sentiment verdict.
 */
export function drawBertModelCard(state: IProgramState, layout: IBertModelLayout, title: string, offset: Vec3, isMain: boolean) {
    let { nBlocks, C, nHeads, dFF, vocabSize, T } = layout.shape;
    let model = layout.model;
    let shapeText = `L = ${nBlocks},  H = ${C},  A = ${nHeads},  dFF = ${dFF},  V = ${vocabSize},  T = ${T}`;

    let extra: string[] = [];
    let m = model?.config.metrics;
    if (m) {
        extra.push(`SST-2 dev accuracy ${(m.sentiment_dev_acc * 100).toFixed(1)}%  \u00b7  majority baseline ${(m.majority_baseline * 100).toFixed(1)}%`);
        extra.push(`masked-token top-1 ${(m.mlm_top1_all * 100).toFixed(1)}%  \u00b7  NSP ${(m.nsp_dev_acc * 100).toFixed(1)}% (${(m.nsp_dev_acc * 100).toFixed(0)}% = chance)`);
        if (model?.truncated) {
            extra.push(`this sentence is cut off: only its first ${T - 3} pieces are in the input`);
        }
    }

    drawCardFrame(state, title, layout.weightCount, shapeText, offset, layout, extra);

    if (isMain && model) {
        renderBertTokenStrip(state, layout);
        renderBertMlmCandidates(state, layout);
        renderBertSentiment(state, layout);
    }
}

/** The WordPiece strings above the token-id row, so `##ing` reads as the literal it is (Figure 2). */
function renderBertTokenStrip(state: IProgramState, layout: IBertModelLayout) {
    let model = layout.model!;
    let blk = layout.tokens;
    let cellW = layout.cell;
    let rowH = 8;
    let mtx = new Mat4f();
    let lineOpts: ILineOpts = makeLineOpts({ color: Vec4.fromHexColor("#000000", 0.4), mtx, thick: 1.2 });

    let tl = new Vec3(blk.x, blk.y - layout.margin * 1.5 - rowH, 0);
    let br = new Vec3(blk.x + layout.shape.T * layout.cell, tl.y + rowH, 0);
    // renderInputBoxes is the Transformer strip\u2019s renderer; it only needs a vocab to look ids up in
    renderInputBoxes(state, { shape: { T: layout.shape.T }, model: { vocab: model.vocab } } as unknown as IGptModelLayout,
        model.inputBuf, layout.shape.T, tl, br, cellW, 2.6, lineOpts);
}

/** Where the [MASK] candidate chart sits, so the walkthrough can frame the camera on it. */
export function bertMlmChartRegion(layout: IBertModelLayout) {
    let blk = layout.mlmProbs;
    let rowH = 4.5;
    let rows = 5;
    return { x: blk.x + blk.dx + layout.margin * 1.6, y: blk.y + blk.dy / 2 - (rows * rowH) / 2 - 6, dx: 90, dy: rows * rowH + 12, rowH };
}

/** Top candidates for the first masked position, straight out of mlmProbs. */
function renderBertMlmCandidates(state: IProgramState, layout: IBertModelLayout) {
    let model = layout.model!;
    let render = state.render;
    let pos = model.masked[0];
    if (pos === undefined) {
        return;
    }
    let top = bertTopK(model, pos, 5);
    if (top.length === 0) {
        return;
    }
    let mtx = new Mat4f();
    let opacity = layout.mlmProbs.opacity;
    let region = bertMlmChartRegion(layout);
    let textOpts: IFontOpts = { color: Vec4.fromHexColor("#000000", opacity), mtx, size: 3.2 };

    drawText(render.modelFontBuf, `top candidates for position ${pos} (a [MASK])`, region.x, region.y, { ...textOpts, size: 2.8 });
    let barMax = 40;
    top.forEach((cand, i) => {
        let y = region.y + 5 + i * region.rowH;
        drawText(render.modelFontBuf, cand.tok, region.x, y + 0.4, textOpts);
        let w = Math.max(0.2, cand.p * barMax);
        let color = i === 0 ? Vec4.fromHexColor("#2f9e44", opacity) : Vec4.fromHexColor("#7c3c8d", 0.55 * opacity);
        addQuad(render.triRender, new Vec3(region.x + 22, y + 0.5, 0), new Vec3(region.x + 22 + w, y + region.rowH - 0.8, 0), color);
        drawText(render.modelFontBuf, `${(cand.p * 100).toFixed(cand.p > 0.1 ? 0 : 1)}%`, region.x + 23 + w, y + 0.8, { ...textOpts, size: 2.5 });
    });
}

/** The sentiment verdict beside the two-cell output block. */
function renderBertSentiment(state: IProgramState, layout: IBertModelLayout) {
    let model = layout.model!;
    let probs = model.sentimentProbs.localBuffer;
    if (!probs) {
        return;
    }
    let render = state.render;
    let blk = layout.sentimentProbs;
    let mtx = new Mat4f();
    let opacity = blk.opacity;
    let classes = model.config.classes ?? ['negative', 'positive'];
    let left = blk.x + blk.dx + layout.margin * 1.6;
    let top = blk.y;
    let rowH = 4.5;
    let barMax = 45;
    let textOpts: IFontOpts = { color: Vec4.fromHexColor("#000000", opacity), mtx, size: 3.2 };

    let best = probs[1] >= probs[0] ? 1 : 0;
    for (let i = 0; i < 2; i++) {
        let y = top + i * rowH;
        drawText(render.modelFontBuf, classes[i] ?? String(i), left, y + 0.4, textOpts);
        let w = Math.max(0.2, probs[i] * barMax);
        let color = i === best ? Vec4.fromHexColor("#2f9e44", opacity) : Vec4.fromHexColor("#7c3c8d", 0.55 * opacity);
        addQuad(render.triRender, new Vec3(left + 16, y + 0.5, 0), new Vec3(left + 16 + w, y + rowH - 0.8, 0), color);
        drawText(render.modelFontBuf, `${(probs[i] * 100).toFixed(1)}%`, left + 17 + w, y + 0.8, { ...textOpts, size: 2.5 });
    }
    drawText(render.modelFontBuf, `prediction: ${classes[best] ?? best}`, left, top + 2 * rowH + 1, { ...textOpts, size: 3.6 });
}
