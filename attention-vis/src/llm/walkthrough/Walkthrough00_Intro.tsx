import { IWalkthrough, Phase } from "./Walkthrough";
import { cameraFor, Colors, commentary, DimStyle, dimStyleColor, embed, ITimeInfo, IWalkthroughArgs, moveCameraTo, moveCameraToFor, phaseTools, setInitialCamera } from "./WalkthroughTools";
import s from './Walkthrough.module.scss';
import { Dim, Vec3, Vec4 } from "@/src/utils/vector";
import { clamp, makeArray } from "@/src/utils/data";
import React, { useState } from "react";
import { useProgramState } from "../Sidebar";
import { dimProps, findSubBlocks, splitGrid } from "../Annotations";
import { useGlobalDrag } from "@/src/utils/pointer";
import { BlkSpecial, IBlkDef, IGptModelLayout, IModelLayout } from "../GptModelLayout";
import { IColorMix } from "../Annotations";
import { IProgramState } from "../Program";
import { lerp } from "@/src/utils/math";
import { drawDependences } from "../Interaction";
import { drawDataFlow } from "../components/DataFlow";
import { embedInline } from "./Walkthrough01_Prelim";

/*
We're mostly on the right track here I think.

Main things that could do with improvement:

 - Make the camera movement more robust. It should basically be on rails when moving the slider.
 - We store the camera pos at the start of a given movement, and use that as a lerp, and also
   have a default so we can do a reversal

 - Have a highlight-region animation in both the text and the model, to bring attention to a specific
   point. Probably a rectangle with a rotating border. Could also include a shaded background on the text

 - Scroll the text to the next region to read

 - Have more pronounced (& consistent) delays between: camera movement/scroll -> higlight -> action
   - Can probably combine the camera movement and the highlight into a single action, but still have
     a delay between the highlight and the action

 - Generally lean on the new display features more. Might need to add a few more (e.g. animate the dot product)
 - Also probably need to do TeX style layout for maths in the html view :( Could probably reuse the code from the
   3d layout? Just need to get the glyphs in via css/ttf, and then maybe use canvas for positioning? Hmm should
   be able to use divs with abs positioning.

 - For stopping/starting, use t + dt to figure out when it was crossed, rather than storing the lastPauseTime etc

*/

interface IIntroState {

}

function getIntroState(walkthrough: IWalkthrough): IIntroState {
    return walkthrough.phaseData.get(Phase.Intro_Intro) as IIntroState;
}

export function walkthroughIntro(args: IWalkthroughArgs) {
    let { breakAfter, afterTime, c_str, c_blockRef } = phaseTools(args.state);
    let { state, layout, walkthrough: wt } = args;
    let { T } = layout.shape;

    if (wt.phase !== Phase.Intro_Intro) {
        return;
    }

    let { enc, dec } = layout;
    let wholeModelCam = cameraFor(layout.cubes, new Vec3(270, 14, 0), 1.1);

    setInitialCamera(state, wholeModelCam.center, wholeModelCam.angle);

    let c0 = commentary(wt, null, 0)`Welcome to the walkthrough of the _Transformer_, the neural network architecture introduced in the 2017 paper
"Attention Is All You Need". Here we'll explore a tiny Transformer with a mere ${embedInline(numberToCommaSep(layout.weightCount))} parameters.

Like the model in the paper (which translates English to German), it maps one sequence to another. Our task is a much simpler one: take a sequence of six letters: ${embed(ExampleInput)}
and _reverse_ it, i.e. produce ${embed(ExampleOutput)}`;

    if (c0.t > 0) {
        for (let cube of layout.cubes) {
            if (cube.t === 'i' && cube.access) {
                cube.access.disable = true;
            }
        }
    }

    let t4 = afterTime(null, 1.5, 0.4);
    moveCameraToFor(state, t4, [enc.idxObj, enc.tokEmbedObj, enc.posEmbedObj, enc.residual0, dec.idxObj, dec.residual0], new Vec3(275, 10, 0), 0.9);
    let t6 = afterTime(null, 1.0, 0.2);

    if (t6.active && t6.t < 1.0) {
        state.display.tokenColors = { mixes: flashSequence(T, t6.t), color2: dimStyleColor(DimStyle.Token) };
    }

    breakAfter();

    let tokenStr = c_str('_token_', 0, DimStyle.Token);
    let tokenIdxStr = c_str('_token index_', 0, DimStyle.TokenIdx);

    commentary(wt, t6)`We call each of these letters a ${tokenStr}, and the set of the model's different tokens make up its _vocabulary_:${embed(TokenVocab)}

    From this table, each token is assigned a number, its ${tokenIdxStr}. The extra token ${c_str('<s>', 0, DimStyle.Token)} marks the _start_ of a sequence; we'll need it for the output.
    Now we can enter the sequence of numbers into the model:${embed(ExampleTokenValues)}\n`;
    breakAfter();

    let t7 = afterTime(null, 1.5, 0.5);

    if (t7.active) {
        state.display.tokenIdxColors = { mixes: revealSequence(T, t7.t), color2: dimStyleColor(DimStyle.TokenIdx) };
        revealBlockAlongX(layout, enc.idxObj, t7.t);
    }

    breakAfter();

    let c5 = commentary(wt)`In the 3D view, each green cell represents a number being processed, and each blue cell is a weight. ${embed(GreenBlueCells)}
    Each number in the sequence first gets turned into a ${layout.shape.C} element vector (a size chosen for this particular model). This is called an _embedding_.`;
    breakAfter(c5);

    {
        let t_makeVecs = afterTime(null, 2.0, 0.5);

        if (t_makeVecs.active) {
            revealBlockAlongX(layout, enc.idxObj, 1.0);
            revealBlockAlongX(layout, enc.residual0, t_makeVecs.t);
        }
    }

    breakAfter();
    commentary(wt)`A Transformer has two halves. On the left is the ${c_blockRef('_encoder_', enc.blocks.flatMap(b => b.cubes))}, which reads the whole input sequence.
    On the right is the ${c_blockRef('_decoder_', dec.blocks.flatMap(b => b.cubes))}, which writes the output sequence, one token at a time.

    The input embeddings flow down through the encoder: a stack of ${layout.shape.nBlocks} identical _layers_, each made of a self-attention step and a feed-forward step.`;
    breakAfter();

    let processState: IProcessInfo;
    {
        let t_encCam = afterTime(null, 1.0, 0.5);
        moveCameraToFor(state, t_encCam, [enc.residual0, ...enc.blocks[0].cubes], new Vec3(285, 16, 0), 0.9);
        let t_encWalk = afterTime(null, 5.0, 0.5);
        processState = processUpTo(state, t_encWalk, enc.blocks[0].output);

        let t_encCam2 = afterTime(null, 1.0, 0.5);
        moveCameraToFor(state, t_encCam2, enc.blocks.flatMap(b => b.cubes), new Vec3(285, 16, 0), 0.9);
        let t_encWalk2 = afterTime(null, 3.0, 0.5);
        processUpTo(state, t_encWalk2, enc.output, processState);

        if (t_encCam.active) {
            enc.blocks[0].blockLabel.visible = Math.max(enc.blocks[0].blockLabel.visible, t_encCam.t);
        }
        if (t_encCam2.active) {
            enc.blocks[1 % enc.blocks.length].blockLabel.visible = Math.max(enc.blocks[1 % enc.blocks.length].blockLabel.visible, t_encCam2.t);
        }
    }

    breakAfter();
    commentary(wt)`The result is the ${c_blockRef('_encoder output_', enc.output)}: one vector per input letter, but now each one has gathered information from the whole sequence.
    This is all the decoder will ever see of the input.

    The decoder gets a sequence of its own: the output we've produced so far, _shifted right_ by one position, starting with ${c_str('<s>', 0, DimStyle.Token)}.
    Here, it's the complete answer: ${embed(ExampleDecoderInput)}

    It flows through its own stack of layers. In each one, a _cross-attention_ step looks up the information it needs from the encoder output.`;
    breakAfter();

    {
        let t_decCam = afterTime(null, 1.0, 0.5);
        moveCameraToFor(state, t_decCam, [enc.output, dec.residual0, ...dec.blocks[0].cubes], new Vec3(268, 14, 0), 0.9);

        let t_decInput = afterTime(null, 1.5, 0.3);
        if (t_decInput.active) {
            state.display.decTokenIdxColors = { mixes: revealSequence(T, t_decInput.t), color2: dimStyleColor(DimStyle.TokenIdx) };
            revealBlockAlongX(layout, dec.idxObj, t_decInput.t);
        }

        let t_decWalk = afterTime(null, 6.0, 0.5);
        processUpTo(state, t_decWalk, dec.blocks[0].output, processState!);

        if (t_decCam.active) {
            dec.blocks[0].blockLabel.visible = Math.max(dec.blocks[0].blockLabel.visible, t_decCam.t);
        }

        let t_decCam2 = afterTime(null, 1.0, 0.5);
        moveCameraToFor(state, t_decCam2, [...dec.blocks.flatMap(b => b.cubes), layout.logitsSoftmax], new Vec3(268, 14, 0), 0.9);
        let t_decWalk2 = afterTime(null, 4.0, 0.5);
        processUpTo(state, t_decWalk2, layout.logitsSoftmax, processState!);

        let t_outputCam = afterTime(null, 1.0, 0.2);
        moveCameraToFor(state, t_outputCam, [dec.output, layout.logits, layout.logitsSoftmax], new Vec3(268, 8, 0), 0.45);
        let t_outputToks = afterTime(null, 1.0, 0.5);

        if (t_decCam.active) {
            let arr = makeArray(T, 0);
            if (t_outputToks.active) {
                arr = revealSequence(T, t_outputToks.t);
            }
            state.display.tokenOutputColors = { color1: new Vec4(0,0,0,0), color2: Vec4.fromHexColor('#000', 1), mixes: arr };
        }
    }

    commentary(wt)`So what's the output? At each position, the model predicts the _next_ token of the output: a probability for each token in the vocabulary.
        At the first position (where the decoder input is ${c_str('<s>', 0, DimStyle.Token)}), it predicts the first output letter; at the second position, the second letter, and so on.`;

    commentary(wt)`When generating an answer, we don't know the output in advance. So we start with just ${c_str('<s>', 0, DimStyle.Token)}, take the most likely
    prediction, append it to the decoder input, and run the decoder again, repeating until we're done. The encoder only needs to run once.`;

    breakAfter();
}

export function numberToCommaSep(a: number) {
    return a.toLocaleString('en-US');
}

/** Flash each of the n items in turn, as t goes 0 -> 1 */
export function flashSequence(n: number, t: number) {
    let mixes = makeArray(n, 0);
    for (let i = 0; i < n; i++) {
        let highT = (i + 1.5) / (n + 2);
        mixes[i] = 1.0 - clamp(Math.abs(t - highT) * 4, 0, 1);
    }
    return mixes;
}

/** Turn on each of the n items in turn, as t goes 0 -> 1 */
export function revealSequence(n: number, t: number) {
    let mixes = makeArray(n, 0);
    for (let i = 0; i < n; i++) {
        let highT = (i + 1.5) / (n + 2);
        mixes[i] = clamp((t - highT) * 4, 0, 1);
    }
    return mixes;
}

/** Enable the values of a block, column by column (along x), as t goes 0 -> 1 */
export function revealBlockAlongX(layout: IModelLayout, blk: IBlkDef, t: number) {
    let n = blk.cx;
    let idxPos = t * n;
    if (t < 1.0) {
        splitGrid(layout, blk, Dim.X, idxPos, clamp(n - idxPos, 0, 1));
        for (let sub of findSubBlocks(blk, Dim.X, null, Math.min(n - 1, Math.floor(idxPos)))) {
            if (sub.access) {
                sub.access.disable = false;
            }
        }
    } else if (blk.access) {
        blk.access.disable = false;
    }
}

interface IProcessInfo {
    lastBlockIdx: number;
}

export function startProcessBefore(state: IProgramState, block: IBlkDef): IProcessInfo {
    let activeBlocks = state.mainLayout.cubes.filter(a => a.t !== 'w');

    return {
        lastBlockIdx: activeBlocks.indexOf(block) - 1,
    };
}

export function processUpTo(state: IProgramState, timer: ITimeInfo, block: IBlkDef, prevInfo?: IProcessInfo): IProcessInfo {

    let activeBlocks = state.mainLayout.cubes.filter(a => a.t !== 'w');

    let firstIdx = prevInfo ? prevInfo.lastBlockIdx + 1 : 0;
    let lastIdx = activeBlocks.indexOf(block);

    // we weight the time on each block by the number of cells in the block, times by how many dependent cells it has
    // although to make this less extreme, we take a fractional power of this
    let cellCounts = activeBlocks
        .filter((_, i) => i >= firstIdx && i <= lastIdx)
        .map(a => (a.cx * a.cy) * Math.pow(a.deps?.dotLen ?? 1, 0.25));
    let totalCells = cellCounts.reduce((a, b) => a + b, 0);

    let accCell = 0;
    let currIdx = firstIdx;
    let subPos = 0; // 0 -> 1
    for (let i = firstIdx; i <= lastIdx; i++) {
        let blockFract = cellCounts[i - firstIdx] / totalCells;
        accCell += blockFract;
        if (timer.t < accCell) {
            currIdx = i;
            subPos = (timer.t - (accCell - blockFract)) / blockFract;
            break;
        }
    }

    let blk = activeBlocks[currIdx];

    // default, but switched for attention matrix
    let dim0 = Dim.X;
    let dim1 = Dim.Y;
    if (blk.transpose) {
        dim0 = Dim.Y;
        dim1 = Dim.X;
    }

    let { cx } = dimProps(blk, dim0);
    let { cx: cy } = dimProps(blk, dim1);

    let horizPos = lerp(0, cx, subPos);
    let horizIdx = Math.floor(horizPos);

    let vertPos = lerp(0, cy, horizPos - horizIdx);
    let vertIdx = Math.floor(vertPos);

    let blockPos = new Vec3().withSetAt(dim0, horizIdx).withSetAt(dim1, vertIdx); // new Vec3(horizIdx, vertIdx, 0);
    let pinPos = new Vec3(Math.floor(cx / 2), 0, 0);

    if (blk === state.layout.enc.residual0 || blk === state.layout.dec.residual0) {
        pinPos = new Vec3(cx * 2, -2, 0);
    }

    if (timer.t >= 1.0) {
        currIdx = lastIdx;
    }

    for (let i = firstIdx; i < currIdx; i++) {
        let blk = activeBlocks[i];
        if (blk.access) {
            blk.access.disable = false;
        }
    }

    if (timer.active && timer.t < 1.0) {
        drawDependences(state, blk, blockPos);
        drawDataFlow(state, blk, blockPos, pinPos);

        for (let label of state.mainLayout.labels) {
            for (let c of label.cubes) {
                if (c === blk) {
                    label.visible = 1.0;
                }
            }
        }

        blk.highlight = 0.3;

        let column = splitGrid(state.mainLayout, blk, dim0, horizPos, 0);
        if (column) {
            for (let col of findSubBlocks(blk, dim0, null, horizIdx)) {
                if (col.access) {
                    col.access.disable = false;
                    col.highlight = 0.1;
                }
            }
            column.highlight = 0.4;

            let curr = splitGrid(state.mainLayout, column, dim1, vertPos, 0);
            for (let blk of findSubBlocks(column, dim1, null, vertIdx)) {
                if (blk.access) {
                    blk.access.disable = false;
                }
            }
            if (curr) {
                curr.highlight = 0.7;
            }
        }


    } else if (timer.active) {
        let blk = activeBlocks[lastIdx];
        if (blk.access) {
            blk.access.disable = false;
        }
    }

    let info = prevInfo ?? { lastBlockIdx: currIdx };
    info.lastBlockIdx = lastIdx;
    return info;
}

function tokensToStrings(state: IProgramState, buf: Float32Array | undefined, fallback: string[]) {
    let model = state.model;
    if (!model || !buf) {
        return fallback;
    }
    return [...buf].map(i => model!.vocab[i]);
}

const TokenRow: React.FC<{ toks: string[], color: Vec4, mixes?: IColorMix | null, startIdx?: number }> = ({ toks, color, mixes, startIdx }) => {
    return <div className={s.tableWrap}>
        <div>{toks.map((c, i) => {
            let baseColor = color;
            if (mixes) {
                baseColor = Vec4.lerp(baseColor, mixes.color2, mixes.mixes[i + (startIdx ?? 0)] ?? 0);
            }
            return <span key={i} style={{ color: baseColor.toHexColor() }}>{c} </span>;
        })}</div>
    </div>;
};

const ExampleInput: React.FC = () => {
    let state = useProgramState();
    let toks = tokensToStrings(state, state.model?.enc.inputBuf, 'CABBCA'.split(''));
    return <TokenRow toks={toks} color={dimStyleColor(DimStyle.Token)} mixes={state.display.tokenColors} />;
};

const ExampleOutput: React.FC = () => {
    let state = useProgramState();
    let toks = tokensToStrings(state, state.model?.enc.inputBuf, 'CABBCA'.split('')).reverse();
    return <TokenRow toks={toks} color={dimStyleColor(DimStyle.Token)} />;
};

const ExampleDecoderInput: React.FC = () => {
    let state = useProgramState();
    let toks = tokensToStrings(state, state.model?.dec.inputBuf, ['<s>', ...'ACBBA'.split('')]);
    return <TokenRow toks={toks} color={dimStyleColor(DimStyle.Token)} />;
};

const ExampleTokenValues: React.FC = () => {
    let state = useProgramState();
    let buf = state.model?.enc.inputBuf;
    let toks = buf ? [...buf].map(a => a.toString()) : '201120'.split('');
    return <TokenRow toks={toks} color={dimStyleColor(DimStyle.TokenIdx)} mixes={state.display.tokenIdxColors} />;
};

const TokenVocab: React.FC = () => {
    let state = useProgramState();
    let vocab = state.model?.vocab ?? ['A', 'B', 'C', '<s>'];

    return <div className={s.tableWrap}>
        <table className={s.table}>
            <tbody>
                <tr className={s.tokString} style={{ color: dimStyleColor(DimStyle.Token).toHexColor() }}>
                    <th>token</th>{vocab.map((v, i) => <td key={i}>{v}</td>)}
                </tr>
                <tr className={s.tokIndex} style={{ color: dimStyleColor(DimStyle.TokenIdx).toHexColor() }}>
                    <th>index</th>{vocab.map((v, i) => <td key={i}>{i}</td>)}
                </tr>
            </tbody>
        </table>
    </div>
};

const GreenBlueCells: React.FC = () => {

    let [blueNums, setBlueNums] = useState([-0.7, 0.7, -0.1]);
    let [greenNums, setGreenNums] = useState([-0.7, 0.4, 0.8]);

    let blueColor = new Vec4(0.3, 0.3, 1.0);
    let greenColor = new Vec4(0.3, 0.9, 0.3);

    return <div className={s.tableWrap}>
        <div className={s.cellInfoCols}>
            <div className={s.cellInfoCol}>
                <Cell nums={greenNums} color={greenColor} mul={0.5} />
                <Graph nums={greenNums} color={greenColor} setNums={setGreenNums} />
                <div className={s.cellInfoText}>being processed</div>
            </div>
            <div className={s.cellInfoCol}>
                <Cell nums={blueNums} color={blueColor} mul={1} />
                <Graph nums={blueNums} color={blueColor} setNums={setBlueNums} />
                <div className={s.cellInfoText}>weights</div>
            </div>
        </div>
    </div>
};

const Cell: React.FC<{ nums: number[], color: Vec4, mul?: number }> = ({ color, nums, mul }) => {

    let grey = new Vec4(0.5, 0.5, 0.5, 1.0);
    let cellLight = Vec4.lerp(color, grey, 0.9);
    let cellDark = cellLight.mul(0.98);
    cellDark.w = 1.0;

    let cellColor = (n: number) => {
        let weight = clamp(Math.abs(n), 0.0, 1.0);

        let negColor = new Vec4(0.0, 0.0, 0.0);
        let posColor = color;
        let zeroColor = new Vec4(0.5, 0.5, 0.5);
        if (n < 0.0) {
            return Vec4.lerp(zeroColor, negColor, weight).toHexColor();
        } else {
            return Vec4.lerp(zeroColor, posColor, weight).toHexColor();
        }
    };

    return <div className={s.cellArrayHoriz}>
        {nums.map((n, i) => {
            return <div className={s.cellRect} key={i} style={{ backgroundColor: (i % 2 === 0 ? cellLight : cellDark).toHexColor() }}>
                <div className={s.cellCircle} style={{ backgroundColor: cellColor(n * (mul ?? 1.0)) }} />
            </div>;
        })}
    </div>
};

const Graph: React.FC<{
    nums: number[],
    color: Vec4,
    max?: number,
    setNums?: (nums: number[]) => void,
}> = ({ color, nums, max, setNums }) => {
    let [graphEl, setGraphEl] = useState<HTMLDivElement | null>(null);

    let ticks = [-1, 0, 1];
    let cellW = 30;
    let dispColor = color.mul(1.0);
    dispColor.w = 0.5;

    interface IDragInitial {
        index: number;
        nums: number[];
    }

    let [, setDragStart] = useGlobalDrag<IDragInitial>(function handleMove(ev, ds) {
        let dy = ev.clientY - ds.clientY;
        let h = graphEl!.clientHeight * 0.5;
        let nums = [...ds.data.nums];
        nums[ds.data.index] = clamp(nums[ds.data.index] - dy / h, -1.0, 1.0);
        setNums?.(nums);
        ev.preventDefault();
        ev.stopImmediatePropagation();
    })

    return <div className={s.graph} style={{ width: cellW * nums.length }} ref={setGraphEl}>

        <div className={s.axisLeft} />

        <div className={s.axisZero} />

        {nums.map((n, i) => {
            let nScaled = n / (max ?? 1.0);

            return <div className={s.graphCol} key={i}>
                <div className={s.graphBar} style={{
                    backgroundColor: dispColor.toHexColor(),
                    top: nScaled < 0 ? '50%' : `${(0.5 - nScaled/2) * 100}%`,
                    height: `${(Math.abs(nScaled)/2) * 100}%`,
                }} />
                <div
                    className={s.graphBarHit}
                    onMouseDown={ev => {
                        setDragStart(ev, { index: i, nums });
                        ev.stopPropagation();
                        ev.preventDefault();
                    }}
                    style={{
                        top: `${(0.5 - nScaled/2) * 100}%`
                    }} />
                <div className={s.graphBarLabel} style={{
                     bottom: nScaled < 0 ? '50%' : undefined,
                        top: nScaled > 0 ? '50%' : undefined,
                }}>
                    {n.toFixed(1)}
                </div>
            </div>;
        })}

    </div>;

};
