import { duplicateGrid, splitGrid } from "../Annotations";
import { getBlockValueAtIdx } from "../components/DataFlow";
import { IBlkDef } from "../GptModelLayout";
import { drawText, IFontOpts, measureText } from "../render/fontRender";
import { lerp } from "@/src/utils/math";
import { Mat4f } from "@/src/utils/matrix";
import { clamp } from "@/src/utils/data";
import { Dim, Vec3, Vec4 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { embedInline } from "./Walkthrough01_Prelim";
import { commentary, DimStyle, embed, IWalkthroughArgs, moveCameraToFor, setInitialCameraFor } from "./WalkthroughTools";
import React from "react";
import { useProgramState } from "../Sidebar";
import { sinusoidalPositionalEncoding } from "../TransformerModel";
import { processUpTo, startProcessBefore } from "./Walkthrough00_Intro";

export function walkthrough02_Embedding(args: IWalkthroughArgs) {
    let { walkthrough: wt, state, tools: { c_str, c_blockRef, c_dimRef, afterTime, cleanup, breakAfter }, layout } = args;
    let render = state.render;

    if (wt.phase !== Phase.Enc_Embedding) {
        return;
    }

    let enc = layout.enc;
    let { C } = layout.shape;
    let embedBlocks = [enc.idxObj, enc.tokEmbedObj, enc.posEmbedObj, enc.residual0];
    setInitialCameraFor(state, embedBlocks, new Vec3(285, 14, 0), 0.9);
    wt.dimHighlightBlocks = embedBlocks;

    let tokName = (i: number) => layout.model?.vocab[i] ?? 'B';
    let exampleIdx = 3;
    let tokValue = getBlockValueAtIdx(enc.idxObj, new Vec3(exampleIdx, 0, 0)) ?? 1;

    commentary(wt)`
We saw previously how the tokens are mapped to a sequence of integers using a simple lookup table.
These integers, the ${c_blockRef('_token indices_', enc.idxObj, DimStyle.TokenIdx)}, are the first and only time we see integers in the model.
From here on out, we're using floats (decimal numbers).

We'll follow the source sequence into the encoder (the decoder's input is embedded in exactly the same way).
Let's take a look at how the 4th token (index 3) is used to generate the 4th column vector of our ${c_blockRef('_input embedding_', enc.residual0)}.`;
    breakAfter();

    let t_moveCamera = afterTime(null, 1.0);
    let t0_splitEmbedAnim = afterTime(null, 0.3);

    breakAfter();

    commentary(wt)`
We use the token index (in this case ${c_str(tokName(tokValue), DimStyle.Token)} = ${c_dimRef(String(tokValue), DimStyle.TokenIdx)}) to select column ${c_dimRef(String(tokValue), DimStyle.TokenIdx)} of the ${c_blockRef('_token embedding matrix_', enc.tokEmbedObj)} on the left.
Note we're using 0-based indexing here, so the first column is at index 0.

This produces a column vector of size ${c_dimRef(`_C_ = ${C}`, DimStyle.C)}, which we describe as the token embedding. Following the paper,
we multiply it by √${c_dimRef('C', DimStyle.C)} (≈ ${c_str((Math.sqrt(C)).toFixed(2), 0)}), so that it's on a similar scale to the positional encoding.
    `;
    breakAfter();

    let t1_fadeEmbedAnim = afterTime(null, 0.3);
    let t2_highlightTokenEmbed = afterTime(null, 0.8);

    breakAfter();

    commentary(wt)`
Attention by itself has no idea of the order of the tokens: it treats the input as a _set_. So we need to add some information about
each token's _position_. Since we're looking at our token ${c_str(tokName(tokValue), DimStyle.Token)} in the 4th position (t = ${c_dimRef('3', DimStyle.T)}), we'll take the 4th column of the
${c_blockRef('_positional encoding matrix_', enc.posEmbedObj)} on the right. This also produces a column vector of size ${c_dimRef(`_C_ = ${C}`, DimStyle.C)}.
    `;
    breakAfter();

    let t4_highlightPosEmbed = afterTime(null, 0.8);

    breakAfter();

    commentary(wt)`
The token embeddings are learned during training (indicated by their blue color). The positional encodings, however, aren't learned
at all: they're fixed waves of different frequencies,

${embedInline(<div className="text-center my-1 italic">PE(t, 2i) = sin(t / 10000<sup>2i/C</sup>), &nbsp; PE(t, 2i+1) = cos(t / 10000<sup>2i/C</sup>)</div>)}

so the first rows oscillate quickly as t increases, and later rows very slowly: ${embed(PosEncodingPlot)}
Every position gets a unique pattern, and the paper argues that the regular structure makes it easy for the model to work with _relative_ positions.

Now that we have these two column vectors, we simply add them together to produce another column vector of size ${c_dimRef(`_C_ = ${C}`, DimStyle.C)}.
`;

    breakAfter();

    let t3_moveTokenEmbed = afterTime(null, 0.8);
    let t5_movePosEmbed = afterTime(null, 0.8);
    let t6_plusSymAnim = afterTime(null, 0.8);
    let t7_addAnim = afterTime(null, 0.8);
    let t8_placeAnim = afterTime(null, 0.8);
    let t9_cleanupInstant = afterTime(null, 0.0);
    let t10_fadeAnim = afterTime(null, 0.8);

    breakAfter();

    commentary(wt)`
We now run this same process for all of the tokens in the input sequence, creating a set of vectors which incorporate both the token values and their positions.

`;

    breakAfter();

    let t11_fillRest = afterTime(null, 5.0);

    breakAfter();

    commentary(wt)`
Feel free to hover over individual cells on the ${c_blockRef('_input embedding_', enc.residual0)} matrix to see the computations and their sources.

We see that running this process for all the tokens in the input sequence produces a matrix of size ${c_dimRef('_T_', DimStyle.T)} x ${c_dimRef('_C_', DimStyle.C)}.
The ${c_dimRef('_T_', DimStyle.T)} stands for ${c_dimRef('_time_', DimStyle.T)}, i.e., you can think of tokens later in the sequence as later in time.
The ${c_dimRef('_C_', DimStyle.C)} stands for ${c_dimRef('_channel_', DimStyle.C)}, and is called d_model in the paper. This length, ${c_dimRef('_C_', DimStyle.C)},
is one of the several "hyperparameters" of the model, and is chosen by the designer as a tradeoff between model size and performance.

This matrix, which we'll refer to as the ${c_blockRef('_input embedding_', enc.residual0)}, is now ready to be passed down through the encoder.
This collection of ${c_dimRef('T', DimStyle.T)} columns each of length ${c_dimRef('C', DimStyle.C)} will become a familiar sight throughout this guide.
`;

    cleanup(t9_cleanupInstant, [t3_moveTokenEmbed, t5_movePosEmbed, t6_plusSymAnim, t7_addAnim, t8_placeAnim]);
    cleanup(t10_fadeAnim, [t0_splitEmbedAnim, t1_fadeEmbedAnim, t2_highlightTokenEmbed, t4_highlightPosEmbed]);

    moveCameraToFor(state, t_moveCamera, [enc.idxObj, enc.tokEmbedObj, enc.posEmbedObj, enc.residual0], new Vec3(278, 10, 0), 0.85);

    let residCol: IBlkDef = null!;
    if ((t0_splitEmbedAnim.t > 0.0 || t10_fadeAnim.t > 0.0) && t11_fillRest.t === 0) {
        splitGrid(layout, enc.idxObj, Dim.X, exampleIdx + 0.5, t0_splitEmbedAnim.t * 4.0);

        enc.residual0.access!.disable = true;
        enc.residual0.opacity = lerp(1.0, 0.1, t1_fadeEmbedAnim.t);

        residCol = splitGrid(layout, enc.residual0, Dim.X, exampleIdx + 0.5, t0_splitEmbedAnim.t * 4.0)!;
        residCol.highlight = 0.3;

        residCol.opacity = lerp(1.0, 0.0, t1_fadeEmbedAnim.t);

    }



    let tokColDupe: IBlkDef | null = null;
    let posColDupe: IBlkDef | null = null;

    if (t2_highlightTokenEmbed.t > 0.0) {
        let tokEmbedCol = splitGrid(layout, enc.tokEmbedObj, Dim.X, tokValue + 0.5, t2_highlightTokenEmbed.t * 4.0)!;

        tokColDupe = duplicateGrid(layout, tokEmbedCol);
        tokColDupe.t = 'i';
        tokEmbedCol.highlight = 0.3;

        let startPos = new Vec3(tokEmbedCol.x, tokEmbedCol.y, tokEmbedCol.z);
        let targetPos = new Vec3(residCol.x, residCol.y, residCol.z).add(new Vec3(-2.0, 0, 3.0));

        let pos = startPos.lerp(targetPos, t3_moveTokenEmbed.t);

        tokColDupe.x = pos.x;
        tokColDupe.y = pos.y;
        tokColDupe.z = pos.z;
    }


    if (t4_highlightPosEmbed.t > 0.0) {
        let posEmbedCol = splitGrid(layout, enc.posEmbedObj, Dim.X, exampleIdx + 0.5, t4_highlightPosEmbed.t * 4.0)!;

        posColDupe = duplicateGrid(layout, posEmbedCol);
        posColDupe.t = 'i';
        posEmbedCol.highlight = 0.3;

        let startPos = new Vec3(posEmbedCol.x, posEmbedCol.y, posEmbedCol.z);
        let targetPos = new Vec3(residCol.x, residCol.y, residCol.z).add(new Vec3(2.0, 0, 3.0));

        let pos = startPos.lerp(targetPos, t5_movePosEmbed.t);

        posColDupe.x = pos.x;
        posColDupe.y = pos.y;
        posColDupe.z = pos.z;
    }

    if (t6_plusSymAnim.t > 0.0 && tokColDupe && posColDupe && t7_addAnim.t < 1.0) {
        for (let c = 0; c < C; c++) {
            let plusCenter = new Vec3(
                (tokColDupe.x + tokColDupe.dx + posColDupe.x) / 2,
                tokColDupe.y + layout.cell * (c + 0.5),
                tokColDupe.z + tokColDupe.dz / 2);

            let isActive = t6_plusSymAnim.t > (c + 1) / C;
            let opacity = lerp(0.0, 1.0, isActive ? 1 : 0);

            let fontOpts: IFontOpts = { color: new Vec4(0, 0, 0, 1).mul(opacity), size: 1.5, mtx: Mat4f.fromTranslation(plusCenter) };
            let w = measureText(render.modelFontBuf, '+', fontOpts);

            drawText(render.modelFontBuf, '+', -w/2, -fontOpts.size/2, fontOpts);
        }
    }

    let origResidPos = residCol ? new Vec3(residCol.x, residCol.y, residCol.z) : new Vec3();
    let offsetResidPos = origResidPos.add(new Vec3(0.0, 0, 3.0));

    if (t7_addAnim.t > 0.0 && tokColDupe && posColDupe) {
        let targetPos = offsetResidPos;
        let tokStartPos = new Vec3(tokColDupe.x, tokColDupe.y, tokColDupe.z);
        let posStartPos = new Vec3(posColDupe.x, posColDupe.y, posColDupe.z);

        let tokPos = tokStartPos.lerp(targetPos, t7_addAnim.t);
        let posPos = posStartPos.lerp(targetPos, t7_addAnim.t);

        tokColDupe.x = tokPos.x;
        tokColDupe.y = tokPos.y;
        tokColDupe.z = tokPos.z;
        posColDupe.x = posPos.x;
        posColDupe.y = posPos.y;
        posColDupe.z = posPos.z;

        if (t7_addAnim.t > 0.95) {
            tokColDupe.opacity = 0.0;
            posColDupe.opacity = 0.0;
            residCol.opacity = 1.0;
            residCol.highlight = 0.0;
            residCol.access!.disable = false;
            residCol.x = targetPos.x;
            residCol.y = targetPos.y;
            residCol.z = targetPos.z;
        }
    }

    if (t8_placeAnim.t > 0.0) {
        let startPos = offsetResidPos;
        let targetPos = origResidPos;
        let pos = startPos.lerp(targetPos, t8_placeAnim.t);
        residCol.x = pos.x;
        residCol.y = pos.y;
        residCol.z = pos.z;
    }

    if (t9_cleanupInstant.t > 0.0 && residCol) {
        residCol.opacity = 1.0;
        residCol.highlight = 0.0;
        residCol.access!.disable = false;
    }

    if (t11_fillRest.t > 0.0) {
        enc.residual0.access!.disable = true;

        let prevInfo = startProcessBefore(state, enc.residual0);
        processUpTo(state, t11_fillRest, enc.residual0, prevInfo);
    }
}

const PosEncodingPlot: React.FC = () => {
    let state = useProgramState();
    let { T, C } = state.layout.shape;
    let pe = sinusoidalPositionalEncoding(T, C);
    let cellW = 22;
    let cellH = 4;
    let w = T * cellW;
    let h = C * cellH;
    let cells: React.ReactNode[] = [];
    for (let t = 0; t < T; t++) {
        for (let c = 0; c < C; c++) {
            let v = clamp(pe[t * C + c], -1, 1);
            let color = v >= 0 ? `rgba(76, 76, 255, ${v})` : `rgba(0, 0, 0, ${-v})`;
            cells.push(<rect key={t * C + c} x={t * cellW} y={c * cellH} width={cellW - 1} height={cellH} fill={color} />);
        }
    }
    return <div className="flex justify-center my-2">
        <svg width={w + 60} height={h + 22} viewBox={`-50 -4 ${w + 60} ${h + 22}`}>
            <rect x={0} y={0} width={w} height={h} fill="#888" />
            {cells}
            {Array.from({ length: T }, (_, t) => <text key={t} x={t * cellW + cellW / 2} y={h + 14} fontSize={11} textAnchor="middle" fill="#359da8">{t}</text>)}
            <text x={-6} y={8} fontSize={11} textAnchor="end" fill="#ce2983">c = 0</text>
            <text x={-6} y={h} fontSize={11} textAnchor="end" fill="#ce2983">c = {C - 1}</text>
        </svg>
    </div>;
};
