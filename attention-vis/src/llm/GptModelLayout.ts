import { IAttentionLayerLink, IFfnLayerLink, ILayerNormLayerLink, IModelShape, ITransformerModelLink, IStackInputLink, bufferRms } from "./TransformerModel";
import { isNil } from "@/src/utils/data";
import { Mat4f } from "@/src/utils/matrix";
import { Dim, Vec3 } from "@/src/utils/vector";
import { IBufferTex } from "@/src/utils/renderPhases";
import { dimProps } from "./Annotations";
import { DimStyle } from "./walkthrough/WalkthroughTools";

export interface IBlkDef {
    idx: number; // index in the layout.cubes array
    t: 'w' | 'i' | 'a', // weights; intermediate value; aggregate (just LN & softmax)
    x: number;
    y: number;
    z: number;
    dx: number; // units: model-space
    dy: number;
    dz: number;
    cx: number; // units: number of cells
    cy: number;
    cz: number;
    access?: IBlkAccess;
    deps?: IBlkDeps;
    dimX: DimStyle;
    dimY: DimStyle;
    name: string;
    // For a table that one input-embedding sum reads a row out of: which of the three BERT-style
    // embeddings it is. The layout renderer uses it to draw the right lookup arrow, and the ViT has
    // none at all, so it is optional everywhere.
    embedPart?: EmbedPartKind;
    small: boolean; // small enough to not be worth rendering in large models
    // implicit dimZ = DimStyle.Batch for t === 'i'

    // fields that are post-added by the walk-through for various rendering configurations

    localMtx?: Mat4f; // for creating blocks that are sub-parts of a block
    // what to do for different axes?
    rangeOffsetsX?: [number, number][]; // if this block has been split, map from [[s0, xOff], [s1, xOff], ...] to the original block
    rangeOffsetsY?: [number, number][];
    rangeOffsetsZ?: [number, number][];
    highlight: number; // 0 - 1 (0 = no highlight, 1 = full highlight)
    opacity: number; // 0 - 1 (0 = transparent, 1 = opaque)
    special: BlkSpecial;
    transpose?: boolean; // transpose the process direction
    subs?: IBlkDef[]; // substitutes for this block (i.e. render these instead)
    offX?: number; // offset from the original block
    offY?: number;
    offZ?: number;
    sizeX?: number; // size of the sub block
    sizeY?: number;
    sizeZ?: number;
}

export enum BlkSpecial {
    None,
    Attention,
}

// define how a cell is computed from other blocks
// matrix-mulplication: cell(x, y, b) = sum_i(A[i, y] * B[x, i, b]) + C[0, y]
export interface IBlkDeps {
    dot?: [IBlkCellDep, IBlkCellDep];
    dotLen?: number;
    add?: IBlkCellDep[];
    special: BlKDepSpecial;
    lowerTri?: boolean;
}

export interface IBlkCellDep {
    src: IBlkDef;
    srcIdxMtx: Mat4f; // inputs: [x, y, b, [i]], outputs: [x, y, b]
}

interface IBlkDepArgs {
    dot?: [[IBlkDef, string], [IBlkDef, string]];
    dotLen?: number;
    add?: [IBlkDef, string][];
    lowerTri?: boolean; // only use the lower triangle of the matrix (causal attention matrices)
    special?: BlKDepSpecial;
}

export enum BlKDepSpecial {
    None,
    Softmax,
    Gelu,
    Relu,
    LayerNorm,
    InputEmbed,
    LayerNormMu,
    LayerNormSigma,
    SoftmaxAggMax,
    SoftmaxAggExp,
    Attention,
    PatchEmbed, // ViT: x_p E + b, except column 0, which is the [class] token
}

let depIdxVars = '0xybi';
function parseDepIdxStr(str: string): Mat4f {
    let mtx = Mat4f.zeros();
    for (let destI = 0; destI < str.length; destI++) {
        let srcIdx = depIdxVars.indexOf(str[destI]);
        if (srcIdx > 0) {
            mtx.s(destI, srcIdx - 1, 1.0);
        }
    }
    return mtx;
}

function depArgsToDeps(args: IBlkDepArgs): IBlkDeps {
    let makeBlkDeps = (src: IBlkDef, depStr: string) => ({ src, srcIdxMtx: parseDepIdxStr(depStr) });
    return {
        dot: args.dot && args.dot.map(([src, depStr]) => makeBlkDeps(src, depStr)) as [IBlkCellDep, IBlkCellDep],
        dotLen: args.dotLen,
        add: args.add && args.add.map(([src, depStr]) => makeBlkDeps(src, depStr)),
        special: args.special ?? BlKDepSpecial.None,
        lowerTri: args.lowerTri,
    };
}

/** The dependencies of one cell of a block (they only differ by cell for the ViT patch embeddings). */
export function depsAt(blk: IBlkDef, idx: Vec3): IBlkDeps | undefined {
    let deps = blk.deps;
    if (deps?.special !== BlKDepSpecial.PatchEmbed || !deps.add) {
        return deps;
    }
    let [bias, cls] = deps.add;
    if (Math.round(idx.x) === 0) {
        return { add: [cls], special: BlKDepSpecial.PatchEmbed };
    }
    return { dot: deps.dot, dotLen: deps.dotLen, add: [bias], special: BlKDepSpecial.None };
}

/**
 * Which of the input-embedding tables a block is: BERT sums a token, a segment and a position
 * embedding (paper Figure 2); the Transformer from the original paper has no segment table, and the
 * ViT replaces the token table with a patch projection.
 */
export type EmbedPartKind = 'token' | 'segment' | 'position';

export interface IInputEmbedParts {
    embed: IBlkDef;                                                    // the sum itself
    idx: IBlkDef;                                                      // the row-index block it looks up by
    tables: { src: IBlkDef; what: EmbedPartKind }[];                    // in layout order
}

/**
 * For an input-embedding block, the index row and the embedding tables it is summed from.
 *
 * Written against a list rather than three fixed fields because BERT's z0 = token + segment +
 * position has a table the Transformer does not have; naming them tokEmbed/posEmbed/idxObj meant a
 * third table had to be bolted on and every consumer had to know it was there.
 *
 * Tables are found by their `embedPart` marker. A layout that sets no markers falls back to the
 * order the Transformer has always used (token, position, index), so this cannot silently change
 * what the existing model draws.
 */
export function inputEmbedParts(blk: IBlkDef): IInputEmbedParts | null {
    let add = blk.deps?.add;
    if (blk.deps?.special !== BlKDepSpecial.InputEmbed || !add || add.length < 3) {
        return null;
    }

    let tables: { src: IBlkDef; what: EmbedPartKind }[] = [];
    let rest: IBlkDef[] = [];
    for (let dep of add) {
        if (dep.src.embedPart) {
            tables.push({ src: dep.src, what: dep.src.embedPart });
        } else {
            rest.push(dep.src);
        }
    }
    if (tables.length === 0) {
        let [tok, pos, idx] = add.map(d => d.src);
        return { embed: blk, idx, tables: [{ src: tok, what: 'token' }, { src: pos, what: 'position' }] };
    }
    // anything left over that isn't a table is looked up *by*, which is what the index row is
    let idx = rest[0] ?? add[add.length - 1].src;
    return { embed: blk, idx, tables };
}

/** The one table a caller cares about, or null when this model has no such thing (no segment for the Transformer). */
export function embedTable(parts: IInputEmbedParts, what: EmbedPartKind): IBlkDef | null {
    return parts.tables.find(t => t.what === what)?.src ?? null;
}

export function getBlkDimensions(blk: IBlkDef) {
    let { x, y, z, dx, dy, dz } = blk;
    return {
        tl: new Vec3(x, y, z),
        br: new Vec3(x + dx, y + dy, z + dz),
    };
}

export function setBlkPosition(blk: IBlkDef, pos: Vec3) {
    blk.x = pos.x;
    blk.y = pos.y;
    blk.z = pos.z;
}

export interface IBlkAccess {
    src: IBufferTex;
    channel: 'r' | 'g' | 'b';
    scale: number;
    // Packed as GPU row-major mat4x2: first 4 floats = texX row (coeffs of cell x,y,z,1),
    // next 4 floats = texY row. See getBlockValueAtIdx / blockRender u_accessMtx.
    mat: Mat4f;
    disable?: boolean;
}

interface IBlkAccessDefArgs {
    src?: IBufferTex;
    channel?: 'r' | 'g' | 'b';
    scale?: number;
    // texX = x[0]*cellX + x[1]*cellY + x[2]*cellZ + x[3]
    // texY = y[0]*cellX + y[1]*cellY + y[2]*cellZ + y[3]
    // (missing trailing coeffs are treated as 0)
    x: number[];
    y: number[];
}

interface IBlkDefArgs {
    t: 'w' | 'i' | 'a', // weights; intermediate value
    xL?: number; // pos of Left edge
    xR?: number; // Right
    xM?: number; // Middle
    zF?: number; // Front
    zB?: number; // Back
    zM?: number; // Middle
    name?: string;
    embedPart?: EmbedPartKind;
    y: number;
    cx: number; // units: number of cells
    cz: number;
    cy: number;
    dimX: DimStyle;
    dimY: DimStyle;
    special?: BlkSpecial;
    access?: IBlkAccessDefArgs;
    deps?: IBlkDepArgs;
    small?: boolean;
    hidden?: boolean;
    transpose?: boolean;
}

export interface IBlkLabel {
    visible: number;
    cubes: IBlkDef[];
}

export interface IModelLayout {
    cell: number;
    height: number;
    margin: number;
    cubes: IBlkDef[];
}

export function cellPosition(layout: IModelLayout, blk: IBlkDef, dim: Dim, index: number) {
    let { x, rangeOffsets } = dimProps(blk, dim);
    let base = x + layout.cell * index;
    if (!rangeOffsets) {
        return base;
    }
    for (let [s, xOff] of rangeOffsets!) {
        if (index < s) {
            return base + xOff;
        }
    }
    return base;
}

export type IGptModelLayout = ReturnType<typeof genGptModelLayout>;
export type ILayoutBuilder = ReturnType<typeof createLayoutBuilder>;
export type IGptLayerNormLayout = ReturnType<ILayoutBuilder['createLn']>;
export type IAttentionLayout = ReturnType<ILayoutBuilder['createAttention']>;
export type IFfnLayout = ReturnType<ILayoutBuilder['createFfn']>;
export type IStackLayout = IGptModelLayout['enc'] | IGptModelLayout['dec'];

/** The dimensions the shared layout builder needs (both the encoder-decoder Transformer and the ViT have these). */
export interface ILayoutShape {
    B: number;
    T: number;
    C: number;
    nHeads: number;
    A: number;
    nBlocks: number;
    dFF: number;
}

export enum StackKind {
    Encoder,
    Decoder,
}

export enum AttnKind {
    Self,
    Masked,
    Cross,
}

/**
The building blocks shared by the model layouts: layer norms, attention and feed-forward (MLP) sublayers.

`s` is the builder's cursor. Block x-positions are written as if the side computations go to the left of the
residual column (centered at x = s.X0); a stack with s.dir = +1 mirrors them to the right. s.y is the next free row.
*/
export function createLayoutBuilder(shape: ILayoutShape) {
    let { B, T, C, nHeads, A, nBlocks, dFF } = shape;

    let isLargeModel = nBlocks > 12;

    let cell = 1.5;
    let margin = Math.max(12, C / 10);

    let s = { X0: 0, dir: -1, y: 0 };

    function mk(args: IBlkDefArgs): IBlkDef {
        let xDef = [args.xL, args.xR, args.xM].map(a => +!isNil(a)).reduce((a, b) => a + b, 0);
        let yDef = [args.zF, args.zB, args.zM].map(a => +!isNil(a)).reduce((a, b) => a + b, 0);
        if (xDef !== 1 || yDef !== 1) {
            throw new Error(`Must supply exactly 1 x arg & 1 y arg: ${JSON.stringify(args)}`);
        }
        let dx = args.cx * cell;
        let dy = args.cz * cell;

        let xL = args.xL, xR = args.xR, xM = args.xM;
        if (s.dir > 0) {
            [xL, xR] = [isNil(xR) ? undefined : -xR, isNil(xL) ? undefined : -xL];
            xM = isNil(xM) ? undefined : -xM;
        }
        let x = !isNil(xL) ? xL : !isNil(xR) ? xR - dx : xM! - dx / 2;
        x += s.X0;
        let z = !isNil(args.zB) ? args.zB : !isNil(args.zF) ? args.zF - dy : args.zM! - dy / 2;

        function ensure4(a: number[]) {
            return a.length === 4 ? a : [...a, 0];
        }

        return {
            dx: args.cx * cell,
            dy: args.cy * cell,
            dz: args.cz * cell,
            t: args.t,
            x: x,
            y: args.y,
            z: z,
            cx: args.cx,
            cy: args.cy,
            cz: args.cz,
            dimX: args.dimX,
            dimY: args.dimY,
            name: args.name ?? "<unknown>",
            embedPart: args.embedPart,
            access: args.access?.src ? {
                channel: args.access.channel ?? 'r',
                src: args.access.src,
                scale: args.access.scale ?? 1.0,
                mat: Mat4f.fromColMajor([...ensure4(args.access.x), ...ensure4(args.access.y), 0, 0, 0, 0, 0, 0, 0, 0]),
            } : undefined,
            deps: args.deps ? depArgsToDeps(args.deps) : undefined,
            opacity: args.hidden ? 0.0 : 1.0,
            highlight: 0.0,
            small: args.small ?? false,
            special: args.special ?? BlkSpecial.None,
            transpose: args.transpose,
            idx: -1,
        };
    }

    function mkLabel(init: number, cubes?: IBlkDef[]): IBlkLabel {
        return { visible: 0, cubes: cubes ?? [] };
    }

    // display scale: values at ~2x the buffer's rms are fully saturated
    function sc(buf: IBufferTex | null | undefined, k = 0.5) {
        let rms = bufferRms(buf);
        return rms ? k / rms : 1.0;
    }

    let cubes: IBlkDef[] = [];

    // x positions (local frame), working outwards from the residual column
    let leftX = -T * cell / 2 - margin;                // right edge of the slot next to the residual
    let rightX = T * cell / 2 + margin;                // left edge of the slot on the other side
    let attnLeftX = leftX - (T + 2) * cell - 3 * margin; // right edge of the attention-head slot
    let qkvValLeftX = attnLeftX - T * cell - margin;   // right edge of the Q/K/V weights
    let memLeftX = leftX - 2 * cell - 2 * margin;      // right edge of the encoder-output copy (cross-attention)

    // Post-LN "Add & Norm" (Transformer): the layer norm is applied to the residual sum, and its output becomes the
    // new residual pathway. The per-column aggregates (μ, σ) sit in the residual column, between the two.
    // Pre-LN (ViT): the layer norm only feeds the next sublayer, so it sits beside the residual column, and the
    // residual pathway skips past it. The sublayer then starts level with the normalized values.
    // y0: where to start (default: just below src); used when the layers wrap into a new column
    function createLn(src: IBlkDef, target?: ILayerNormLayerLink, preNorm: boolean = false, y0?: number) {
        s.y = y0 ?? src.y + src.dy + margin;
        let y = s.y;
        let place = preNorm ? { xR: leftX } : { xM: 0 };
        let gammaRightX = preNorm ? leftX - T * cell - margin : leftX;

        let lnAgg1 = mk({
            t: 'a', cx: T, cz: B, cy: 1, y: y,
            ...place, zM: 0,
            access: { src: target?.normAgg, x: [0, 1, 0], y: [1, 0, T], scale: 1.0, channel: 'r' },
            deps: { add: [[src, 'xi']], special: BlKDepSpecial.LayerNormMu },
            dimX: DimStyle.T, dimY: DimStyle.None, small: true,
            name: 'LN Agg: μ, σ',
        });
        let lnAgg2 = mk({
            t: 'a', cx: T, cz: B, cy: 1, y: y + cell,
            ...place, zM: 0,
            access: { src: target?.normAgg, x: [0, 1, 0], y: [1, 0, T], scale: 0.5, channel: 'g' },
            deps: { add: [[src, 'xi']], special: BlKDepSpecial.LayerNormSigma },
            dimX: DimStyle.T, dimY: DimStyle.None, small: true,
            name: '',
        });

        y += 2 * cell + margin;

        let lnSigma = mk({
            t: 'w', cx: 1, cz: 1, cy: C, y: y,
            xR: gammaRightX, zM: 0,
            access: { src: target?.normWeight, x: [1, 0, 0], y: [0, 1, 0], scale: 0.5 }, // mostly around 1.0
            dimX: DimStyle.None, dimY: DimStyle.C,
            name: 'γ', small: true,
        });
        let lnMu = mk({
            t: 'w', cx: 1, cz: 1, cy: C, y: y,
            xR: gammaRightX - cell - margin, zM: 0,
            access: { src: target?.normBias, x: [1, 0, 0], y: [0, 1, 0], scale: sc(target?.normBias) },
            dimX: DimStyle.None, dimY: DimStyle.C,
            name: 'β', small: true,
        });
        let lnResid = mk({
            t: 'i', cx: T, cz: B, cy: C, y: y,
            ...place, zM: 0,
            access: { src: target?.output, x: [0, 1, 0], y: [1, 0, T], scale: 0.5 },
            deps: { add: [[src, 'xy'], [lnAgg1, 'xi'], [lnAgg2, 'xi'], [lnSigma, '0y'], [lnMu, '0y']], special: BlKDepSpecial.LayerNorm }, // lnSigma is really mul rather than add
            dimX: DimStyle.T, dimY: DimStyle.C,
            name: 'Layer Norm',
        });

        s.y = preNorm ? y : y + C * cell + margin;

        let lnCubes = [lnAgg1, lnAgg2, lnSigma, lnMu, lnResid];
        cubes.push(...lnCubes);
        let label = mkLabel(1.0, lnCubes);
        return { lnAgg1, lnAgg2, lnResid, lnSigma, lnMu, cubes: lnCubes, label, residSrc: src, preNorm };
    }

    // residSrc: what the sublayer output is added to (the query input for post-LN; the residual stream for pre-LN).
    // bias: whether the Q/K/V/O projections have biases (they don't in the Transformer paper; they do in ViT).
    function createAttention(xq: IBlkDef, xkvSrc: IBlkDef, target: IAttentionLayerLink | undefined, kind: AttnKind, opts: { residSrc?: IBlkDef, bias?: boolean } = {}) {
        let causal = kind === AttnKind.Masked;
        let residSrc = opts.residSrc ?? xq;
        let hasBias = opts.bias ?? false;
        // start the heads level with the query input (beside it), unless the layout has already moved past it
        let y = s.y = Math.max(s.y, xq.y);

        let memCopy: IBlkDef | null = null;
        if (kind === AttnKind.Cross) {
            memCopy = mk({
                t: 'i', cx: T, cz: B, cy: C, y: y,
                xR: memLeftX, zM: 0,
                access: xkvSrc.access ? { src: xkvSrc.access.src, x: [0, 1, 0], y: [1, 0, T], scale: xkvSrc.access.scale } : undefined,
                deps: { add: [[xkvSrc, 'xy']] },
                dimX: DimStyle.T, dimY: DimStyle.C,
                name: 'Encoder Output',
            });
        }
        let xkv = memCopy ?? xkvSrc;

        let interHeadMargin = 3 * margin + (C * cell) / 16;
        let qkvMargin = 1 * margin + (C * cell) / 16;

        let headWidth = 3 * B * cell + qkvMargin * 2 + (isLargeModel ? 0 : interHeadMargin);

        let attn1Y = y + A * cell + margin + (isLargeModel ? 2 * A * cell : 0);
        let vOutY = attn1Y + T * cell + margin;
        let attn2LeftX = attnLeftX - (T + 2) * cell - 2 * margin;
        let qkvBiasLeftX = qkvValLeftX - C * cell - margin;

        let qScale = sc(target?.q, 0.5), kScale = sc(target?.k, 0.5), vScale = sc(target?.v, 0.5);
        let wScale = sc(target?.qWeight, 0.5);
        let attnScale = sc(target?.attnMatrix, 0.5);

        let heads = [];
        for (let i = 0; i < nHeads; i++) {
            let headZMid = headWidth * i - (nHeads - 1) * headWidth / 2;
            let qMid = headZMid + B * cell + qkvMargin;
            let kMid = headZMid;
            let vMid = headZMid - B * cell - qkvMargin;

            // x: C, y: A
            let qWeightBlock = mk({
                t: 'w', cx: C, cz: 1, cy: A, y: y,
                xR: qkvValLeftX, zM: qMid,
                access: { src: target?.qWeight, x: [1, 0, 0], y: [0, 1, 0, A * i], scale: wScale },
                dimX: DimStyle.C, dimY: DimStyle.A,
                name: 'Q Weights (W^Q)',
            });

            let kWeightBlock = mk({
                t: 'w', cx: C, cz: 1, cy: A, y: y,
                xR: qkvValLeftX, zM: kMid,
                access: { src: target?.kWeight, x: [1, 0, 0], y: [0, 1, 0, A * i], scale: wScale },
                dimX: DimStyle.C, dimY: DimStyle.A,
                name: 'K Weights (W^K)',
            });

            let vWeightBlock = mk({
                t: 'w', cx: C, cz: 1, cy: A, y: y,
                xR: qkvValLeftX, zM: vMid,
                access: { src: target?.vWeight, x: [1, 0, 0], y: [0, 1, 0, A * i], scale: wScale },
                dimX: DimStyle.C, dimY: DimStyle.A,
                name: 'V Weights (W^V)',
            });

            let mkBias = (src: IBufferTex | null | undefined, zMid: number, name: string) => hasBias ? mk({
                t: 'w', cx: 1, cz: 1, cy: A, y: y,
                xR: qkvBiasLeftX, zM: zMid,
                access: { src: src ?? undefined, x: [0, 0, 0], y: [0, 1, 0, A * i], scale: sc(src) },
                dimX: DimStyle.None, dimY: DimStyle.A, small: true,
                name,
            }) : null;
            let qBiasBlock = mkBias(target?.qBias, qMid, 'Q Bias');
            let kBiasBlock = mkBias(target?.kBias, kMid, 'K Bias');
            let vBiasBlock = mkBias(target?.vBias, vMid, 'V Bias');
            let biasAdd = (b: IBlkDef | null) => b ? [[b, '0y'] as [IBlkDef, string]] : undefined;

            let qBlock = mk({
                t: 'i', cx: T, cz: B, cy: A, y: y,
                xR: attnLeftX, zM: qMid,
                access: { src: target?.q, x: [0, 1, 0, A * i], y: [1, 0, T], scale: qScale },
                deps: { dot: [[qWeightBlock, 'iy'], [xq, 'xi']], dotLen: C, add: biasAdd(qBiasBlock) },
                dimX: DimStyle.T, dimY: DimStyle.A,
                name: 'Q vectors',
            });

            let kBlock = mk({
                t: 'i', cx: T, cz: B, cy: A, y: y,
                xR: attnLeftX, zM: kMid,
                access: { src: target?.k, x: [0, 1, 0, A * i], y: [1, 0, T], scale: kScale },
                deps: { dot: [[kWeightBlock, 'iy'], [xkv, 'xi']], dotLen: C, add: biasAdd(kBiasBlock) },
                dimX: DimStyle.T, dimY: DimStyle.A,
                name: 'K vectors',
            });

            let vBlock = mk({
                t: 'i', cx: T, cz: B, cy: A, y: y,
                xR: attnLeftX, zM: vMid,
                access: { src: target?.v, x: [0, 1, 0, A * i], y: [1, 0, T], scale: vScale },
                deps: { dot: [[vWeightBlock, 'iy'], [xkv, 'xi']], dotLen: C, add: biasAdd(vBiasBlock) },
                dimX: DimStyle.T, dimY: DimStyle.A,
                name: 'V vectors',
            });

            let attnMtx = mk({
                t: 'i', cx: T, cz: B, cy: T, y: attn1Y,
                xR: attnLeftX, zM: headZMid,
                access: { src: target?.attnMatrix, x: [1, 0, 0], y: [0, 1, nHeads * T, T * i], scale: attnScale },
                deps: { dot: [[qBlock, 'yi'], [kBlock, 'xi']], lowerTri: causal, dotLen: A, special: BlKDepSpecial.Attention },
                dimX: DimStyle.T, dimY: DimStyle.T,
                special: BlkSpecial.Attention,
                transpose: true,
                name: 'Attention Matrix',
            });

            let attnMtxAgg1 = mk({
                t: 'a', cx: 1, cz: B, cy: T, y: attn1Y,
                xR: attnLeftX - T * cell - margin - cell, zM: headZMid,
                access: { src: target?.attnMatrixAgg, x: [0, 0, 0, 0], y: [0, 1, nHeads * T, T * i], channel: 'r' },
                deps: { add: [[attnMtx, 'iy']], special: BlKDepSpecial.SoftmaxAggExp },
                dimX: DimStyle.None, dimY: DimStyle.T, small: true,
                name: '',
            });

            let attnMtxAgg2 = mk({
                t: 'a', cx: 1, cz: B, cy: T, y: attn1Y,
                xR: attnLeftX - T * cell - margin, zM: headZMid,
                access: { src: target?.attnMatrixAgg, x: [0, 0, 0, 0], y: [0, 1, nHeads * T, T * i], channel: 'g', scale: attnScale },
                deps: { add: [[attnMtx, 'iy']], special: BlKDepSpecial.SoftmaxAggMax },
                dimX: DimStyle.None, dimY: DimStyle.T, small: true,
                name: '',
            });

            let attnMtxSm = mk({
                t: 'i', cx: T, cz: B, cy: T, y: attn1Y,
                xR: attn2LeftX, zM: headZMid,
                access: { src: target?.attnMatrixSoftmax, x: [1, 0, 0], y: [0, 1, nHeads * T, T * i], scale: 1.0 },
                deps: { add: [[attnMtx, 'xy'], [attnMtxAgg1, 'iy'], [attnMtxAgg2, 'iy']], lowerTri: causal, special: BlKDepSpecial.Softmax },
                dimX: DimStyle.T, dimY: DimStyle.T,
                special: BlkSpecial.Attention,
                transpose: true,
                name: 'Attn Matrix Softmax',
            });

            let vOutBlock = mk({
                t: 'i', cx: T, cz: B, cy: A, y: vOutY,
                xR: attnLeftX, zM: headZMid,
                access: { src: target?.scaledVectors, x: [0, 1, 0, i * A], y: [1, 0, T], scale: sc(target?.scaledVectors) },
                deps: { dot: [[vBlock, 'iy'], [attnMtxSm, 'ix']], dotLen: T },
                dimX: DimStyle.T, dimY: DimStyle.A,
                name: 'V Output',
            });

            let biasCubes = [qBiasBlock, kBiasBlock, vBiasBlock].filter(b => !!b) as IBlkDef[];
            let headCubes = [qWeightBlock, kWeightBlock, vWeightBlock, ...biasCubes, qBlock, kBlock, vBlock,
                attnMtx, attnMtxAgg1, attnMtxAgg2, attnMtxSm, vOutBlock];

            let headLabel = mkLabel(1.0, headCubes);
            let qLabel = mkLabel(1.0, [qWeightBlock, ...(qBiasBlock ? [qBiasBlock] : []), qBlock]);
            let kLabel = mkLabel(1.0, [kWeightBlock, ...(kBiasBlock ? [kBiasBlock] : []), kBlock]);
            let vLabel = mkLabel(1.0, [vWeightBlock, ...(vBiasBlock ? [vBiasBlock] : []), vBlock]);
            let mtxLabel = mkLabel(1.0, [attnMtx, attnMtxAgg1, attnMtxAgg2, attnMtxSm]);
            let vectorLabel = mkLabel(1.0, [vOutBlock]);

            heads.push({
                qWeightBlock, kWeightBlock, vWeightBlock,
                qBiasBlock, kBiasBlock, vBiasBlock,
                qBlock, kBlock, vBlock,
                attnMtx, attnMtxAgg1, attnMtxAgg2, attnMtxSm, vOutBlock,
                qLabel, kLabel, vLabel, mtxLabel, vectorLabel, headLabel,
                cubes: headCubes,
                labels: [qLabel, kLabel, vLabel, mtxLabel, vectorLabel, headLabel],
            });
        }

        let vOutCombined = mk({
            t: 'i', cx: T, cz: B, cy: C, y: vOutY,
            xR: attnLeftX, zF: - headWidth * nHeads / 2,
            dimX: DimStyle.T, dimY: DimStyle.C,
            hidden: true,
            name: 'V Output Combined',
        });

        let vFinalZ = Math.max(
            vOutY + A * cell + 2 * margin,
            y + C * cell + margin, // the input block is taller than the head blocks
        );

        let projWeight = mk({
            t: 'w', cx: C, cz: 1, cy: C, y: vFinalZ,
            xR: qkvValLeftX, zM: 0,
            access: { src: target?.proj.weight, x: [1, 0, 0], y: [0, 1, 0], scale: sc(target?.proj.weight) },
            dimX: DimStyle.C, dimY: DimStyle.C,
            name: 'Output Weights (W^O)',
        });

        let projBias = hasBias ? mk({
            t: 'w', cx: 1, cz: 1, cy: C, y: vFinalZ,
            xR: qkvValLeftX - C * cell - margin, zM: 0,
            access: { src: target?.proj.bias ?? undefined, x: [0, 0, 0], y: [0, 1, 0], scale: sc(target?.proj.bias) },
            dimX: DimStyle.None, dimY: DimStyle.C, small: true,
            name: 'Output Bias',
        }) : null;

        let attnOut = mk({
            t: 'i', cx: T, cz: B, cy: C, y: vFinalZ,
            xR: attnLeftX, zM: 0,
            access: { src: target?.proj.output, x: [0, 1, 0], y: [1, 0, T], scale: sc(target?.proj.output) },
            // vOutCombined isn't displayed, so also add from the heads (for highlighting)
            deps: {
                dot: [[projWeight, 'iy'], [vOutCombined, 'xi']], dotLen: C,
                add: [...(projBias ? [[projBias, '0y'] as [IBlkDef, string]] : []), ...heads.map(h => [h.vOutBlock, 'xi'] as [IBlkDef, string])],
            },
            dimX: DimStyle.T, dimY: DimStyle.C,
            name: 'Attention Output',
        });

        let attnResidual = mk({
            t: 'i', cx: T, cz: B, cy: C, y: vFinalZ,
            xM: 0, zM: 0,
            access: { src: target?.residual, x: [0, 1, 0], y: [1, 0, T], scale: sc(target?.residual) },
            deps: { add: [[attnOut, 'xy'], [residSrc, 'xy']] },
            dimX: DimStyle.T, dimY: DimStyle.C,
            name: kind === AttnKind.Cross ? 'Cross-Attention Residual' : 'Attention Residual',
        });

        s.y = vFinalZ + C * cell + margin;

        let headsCubes = heads.flatMap(h => h.cubes);
        let projCubes = [projWeight, ...(projBias ? [projBias] : []), attnOut, attnResidual];
        let attnCubes = [...(memCopy ? [memCopy] : []), ...headsCubes, ...projCubes];
        cubes.push(...attnCubes);

        let projLabel = mkLabel(1.0, projCubes);
        let attnLabel = mkLabel(1.0, attnCubes);

        return {
            kind,
            causal,
            xq,
            xkv,
            residSrc,
            memCopy,
            heads,
            projWeight,
            projBias,
            attnOut,
            attnResidual,
            cubes: attnCubes,
            attnLabel,
            projLabel,
            labels: [attnLabel, projLabel, ...heads.flatMap(h => h.labels)],
        };
    }

    // residSrc: what the output is added to (the input for post-LN; the residual stream for pre-LN).
    // gelu: GELU (ViT) rather than ReLU (Transformer) activation; prefix: 'FFN' (Transformer) or 'MLP' (ViT).
    function createFfn(src: IBlkDef, target: IFfnLayerLink | undefined, opts: { residSrc?: IBlkDef, gelu?: boolean, prefix?: string } = {}) {
        let residSrc = opts.residSrc ?? src;
        let pre = opts.prefix ?? 'FFN';
        let fcY = src.y;
        let y = fcY;

        let mlpFcBias = mk({
            t: 'w', cx: dFF, cz: 1, cy: 1, y: fcY - 1 * cell - margin,
            xR: attnLeftX, zM: 0,
            access: { src: target?.fc.bias!, x: [0, 1, 0], y: [1, 0, 0], scale: sc(target?.fc.bias) },
            dimX: DimStyle.C4, dimY: DimStyle.None,
            name: `${pre} Bias (b₁)`, small: true,
        });

        let mlpFcWeight = mk({
            t: 'w', cx: dFF, cz: 1, cy: C, y: fcY,
            xR: attnLeftX, zM: 0,
            access: { src: target?.fc.weight, x: [0, 1, 0], y: [1, 0, 0], scale: sc(target?.fc.weight) },
            dimX: DimStyle.C4, dimY: DimStyle.C,
            name: `${pre} Weights (W₁)`,
        });

        y = fcY + C * cell + margin;

        let mlpFc = mk({
            t: 'i', cx: dFF, cz: B, cy: T, y: y,
            xR: attnLeftX, zM: 0,
            access: { src: target?.fc.output, x: [1, 0, 0], y: [0, 1, T], scale: sc(target?.fc.output) },
            deps: { dot: [[mlpFcWeight, 'xi'], [src, 'yi']], dotLen: C, add: [[mlpFcBias, 'x']] },
            dimX: DimStyle.C4, dimY: DimStyle.T,
            name: `${pre} Hidden`,
            transpose: true,
        });

        y += T * cell + margin;

        let mlpAct = mk({
            t: 'i', cx: dFF, cz: B, cy: T, y: y,
            xR: attnLeftX, zM: 0,
            access: { src: target?.act, x: [1, 0, 0], y: [0, 1, T], scale: sc(target?.fc.output) },
            deps: { add: [[mlpFc, 'xy']], special: opts.gelu ? BlKDepSpecial.Gelu : BlKDepSpecial.Relu },
            dimX: DimStyle.C4, dimY: DimStyle.T,
            name: `${pre} ${opts.gelu ? 'GELU' : 'ReLU'}`,
            transpose: true,
        });

        y += T * cell + margin;

        let mlpProjWeight = mk({
            t: 'w', cx: dFF, cz: 1, cy: C, y: y,
            xR: attnLeftX, zM: 0,
            access: { src: target?.proj.weight, x: [1, 0, 0], y: [0, 1, 0], scale: sc(target?.proj.weight) },
            dimX: DimStyle.C4, dimY: DimStyle.C,
            name: `${pre} Weights (W₂)`,
        });

        let mlpProjBias = mk({
            t: 'w', cx: 1, cz: 1, cy: C, y: y,
            xR: attnLeftX - dFF * cell - margin, zM: 0,
            access: { src: target?.proj.bias!, x: [1, 0, 0], y: [0, 1, 0], scale: sc(target?.proj.bias) },
            dimX: DimStyle.None, dimY: DimStyle.C, small: true,
            name: `${pre} Bias (b₂)`,
        });

        let mlpResult = mk({
            t: 'i', cx: T, cz: B, cy: C, y: y,
            xL: attnLeftX + margin, zM: 0,
            access: { src: target?.proj.output, x: [0, 1, 0], y: [1, 0, T], scale: sc(target?.proj.output) },
            deps: { dot: [[mlpProjWeight, 'iy'], [mlpAct, 'ix']], dotLen: dFF, add: [[mlpProjBias, '0y']] },
            dimX: DimStyle.T, dimY: DimStyle.C,
            name: `${pre} Result`,
        });

        let mlpResidual = mk({
            t: 'i', cx: T, cz: B, cy: C, y: y,
            xM: 0, zM: 0,
            access: { src: target?.residual, x: [0, 1, 0], y: [1, 0, T], scale: sc(target?.residual) },
            deps: { add: [[mlpResult, 'xy'], [residSrc, 'xy']] },
            dimX: DimStyle.T, dimY: DimStyle.C,
            name: `${pre} Residual`,
        });

        y += C * cell + margin;
        s.y = y;

        let ffnCubes = [mlpFcBias, mlpFcWeight, mlpFc, mlpAct, mlpProjWeight, mlpProjBias, mlpResult, mlpResidual];
        cubes.push(...ffnCubes);

        return {
            src,
            residSrc,
            mlpFcBias, mlpFcWeight, mlpFc, mlpAct, mlpProjWeight, mlpProjBias, mlpResult, mlpResidual,
            cubes: ffnCubes,
            ffnLabel: mkLabel(1.0, ffnCubes),
        };
    }

    return {
        s, cubes, cell, margin, isLargeModel,
        mk, mkLabel, sc,
        leftX, rightX, attnLeftX, qkvValLeftX,
        createLn, createAttention, createFfn,
    };
}

/*
Layout of the encoder-decoder Transformer.

The encoder stack is on the left and the decoder stack on the right, like Figure 1 of the paper. Each stack has a
"residual pathway" column (the B x T x C activations), flowing downwards. The encoder puts all of its side
computations (attention heads, feed-forward weights, layer-norm aggregates) to the left of its column; the decoder
mirrors this and puts them to the right. The space between the two columns carries the encoder output ("memory")
to each decoder layer's cross-attention.

x is to the left and right, y is positive going down, z is coming out of the page (attention heads are spread out
along z).
*/
export function genGptModelLayout(shape: IModelShape, model: ITransformerModelLink | null = null) {
    let { B, T, C, vocabSize, nBlocks, dFF } = shape;

    let L = createLayoutBuilder(shape);
    let { s, cubes, cell, margin, mk, mkLabel, sc, leftX, rightX, createLn, createAttention, createFfn } = L;

    function createEmbedding(stack: IStackInputLink | undefined, kind: StackKind) {
        let idxObj = mk({
            t: 'i', cx: T, cz: B, cy: 1, y: s.y,
            xM: 0, zM: 0,
            access: { src: stack?.tokens, x: [0, 1, 0], y: [1, 0, T], scale: 1 / vocabSize },
            dimX: DimStyle.T, dimY: DimStyle.None,
            name: 'Tokens',
        });

        s.y += cell + margin;

        let tokEmbedObj = mk({
            t: 'w',
            xR: leftX, zM: 0, y: s.y,
            cx: vocabSize, cz: 1, cy: C,
            access: { src: model?.embedWeight, x: [0, 1, 0], y: [1, 0, 0], scale: sc(model?.embedWeight) },
            dimX: DimStyle.n_vocab, dimY: DimStyle.C,
            name: 'Token Embed',
            embedPart: 'token',
        });

        let posEmbedObj = mk({
            t: 'w',
            xL: rightX, zM: 0, y: s.y,
            cx: T, cz: 1, cy: C,
            access: { src: model?.posEnc, x: [0, 1, 0], y: [1, 0, 0], scale: 0.8 },
            dimX: DimStyle.T, dimY: DimStyle.C,
            name: 'Pos Encoding',
            embedPart: 'position',
        });

        let residual0 = mk({
            t: 'i',
            xM: 0, zM: 0, y: s.y,
            cx: T, cz: B, cy: C,
            access: { src: stack?.embed, x: [0, 1, 0], y: [1, 0, T], scale: sc(stack?.embed) },
            deps: { add: [[tokEmbedObj, 'iy'], [posEmbedObj, 'xy'], [idxObj, 'x0']], special: BlKDepSpecial.InputEmbed }, // the i comes from the idxObj lookup
            dimX: DimStyle.T, dimY: DimStyle.C,
            name: 'Input Embed',
        });
        cubes.push(idxObj, tokEmbedObj, posEmbedObj, residual0);

        let embedLabel = mkLabel(s.y, [idxObj, tokEmbedObj, posEmbedObj, residual0]);

        s.y += C * cell + margin;

        return { idxObj, tokEmbedObj, posEmbedObj, residual0, embedLabel, cubes: [idxObj, tokEmbedObj, posEmbedObj, residual0] };
    }

    let blockHalfMargin = 2 * margin;

    // ------ Encoder ------
    s.X0 = 0;
    s.dir = -1;
    s.y = 0;

    let encEmbed = createEmbedding(model?.enc, StackKind.Encoder);
    s.y += blockHalfMargin;

    let encBlocks = [];
    let encX = encEmbed.residual0;
    for (let i = 0; i < nBlocks; i++) {
        let target = model?.enc.blocks[i];
        s.y += blockHalfMargin;
        let attn = createAttention(encX, encX, target?.selfAttn, AttnKind.Self);
        let ln1 = createLn(attn.attnResidual, target?.ln1);
        let ffn = createFfn(ln1.lnResid, target?.ffn);
        let ln2 = createLn(ffn.mlpResidual, target?.ln2);
        let blockCubes = [...attn.cubes, ...ln1.cubes, ...ffn.cubes, ...ln2.cubes];
        let blockLabel = mkLabel(1.0, blockCubes);
        encBlocks.push({
            idx: i, attn, ln1, ffn, ln2, output: ln2.lnResid, input: encX,
            cubes: blockCubes, blockLabel,
            labels: [blockLabel, ln1.label, ln2.label, ffn.ffnLabel, ...attn.labels],
        });
        encX = ln2.lnResid;
        s.y += blockHalfMargin;
    }
    let encOutput = encX;
    let encHeight = s.y;

    // ------ Decoder ------
    s.X0 = 3 * T * cell + 6 * margin;
    s.dir = 1;
    s.y = 0;

    let decEmbed = createEmbedding(model?.dec, StackKind.Decoder);
    s.y += blockHalfMargin;

    let decBlocks = [];
    let decX = decEmbed.residual0;
    for (let i = 0; i < nBlocks; i++) {
        let target = model?.dec.blocks[i];
        s.y += blockHalfMargin;
        let selfAttn = createAttention(decX, decX, target?.selfAttn, AttnKind.Masked);
        let ln1 = createLn(selfAttn.attnResidual, target?.ln1);
        let crossAttn = createAttention(ln1.lnResid, encOutput, target?.crossAttn, AttnKind.Cross);
        let ln2 = createLn(crossAttn.attnResidual, target?.ln2);
        let ffn = createFfn(ln2.lnResid, target?.ffn);
        let ln3 = createLn(ffn.mlpResidual, target?.ln3);
        let blockCubes = [...selfAttn.cubes, ...ln1.cubes, ...crossAttn.cubes, ...ln2.cubes, ...ffn.cubes, ...ln3.cubes];
        let blockLabel = mkLabel(1.0, blockCubes);
        decBlocks.push({
            idx: i, selfAttn, ln1, crossAttn, ln2, ffn, ln3, output: ln3.lnResid, input: decX,
            cubes: blockCubes, blockLabel,
            labels: [blockLabel, ln1.label, ln2.label, ln3.label, ffn.ffnLabel, ...selfAttn.labels, ...crossAttn.labels],
        });
        decX = ln3.lnResid;
        s.y += blockHalfMargin;
    }

    // ------ Output: linear (sharing the embedding matrix) + softmax ------
    s.y += blockHalfMargin;

    let lmHeadWeight = mk({
        t: 'w', cx: C, cy: vocabSize, cz: 1, y: s.y,
        xR: leftX, zM: 0,
        access: { src: model?.embedWeight, x: [1, 0, 0], y: [0, 1, 0], scale: sc(model?.embedWeight) },
        dimX: DimStyle.C, dimY: DimStyle.n_vocab,
        name: 'Linear (shared embed)',
    });

    let logits = mk({
        t: 'i', cx: T, cy: vocabSize, cz: B, y: s.y,
        xM: 0, zM: 0,
        access: { src: model?.logits, x: [0, 1, 0], y: [1, 0, T], scale: sc(model?.logits, 0.4) },
        deps: { dot: [[lmHeadWeight, 'iy'], [decX, 'xi']], dotLen: C },
        dimX: DimStyle.T, dimY: DimStyle.n_vocab,
        name: 'Logits',
    });

    s.y += vocabSize * cell + margin;

    let logitsAgg2 = mk({
        t: 'a', cx: T, cy: 1, cz: B, y: s.y,
        xM: 0, zM: 0,
        access: { src: model?.softmaxAgg, x: [0, 1, 0], y: [1, 0, T], channel: 'g', scale: sc(model?.logits, 0.4) },
        deps: { add: [[logits, 'xi']], special: BlKDepSpecial.SoftmaxAggMax },
        dimX: DimStyle.T, dimY: DimStyle.None,
        name: 'SM Agg',
    });

    let logitsAgg1 = mk({
        t: 'a', cx: T, cy: 1, cz: B, y: s.y + cell,
        xM: 0, zM: 0,
        access: { src: model?.softmaxAgg, x: [0, 1, 0], y: [1, 0, T], channel: 'r' },
        deps: { add: [[logits, 'xi'], [logitsAgg2, 'x0']], special: BlKDepSpecial.SoftmaxAggExp },
        dimX: DimStyle.T, dimY: DimStyle.None,
        name: '',
    });

    s.y += 2 * cell + margin;

    let logitsSoftmax = mk({
        t: 'i', cx: T, cy: vocabSize, cz: B, y: s.y,
        xM: 0, zM: 0,
        access: { src: model?.probs, x: [0, 1, 0], y: [1, 0, T] },
        deps: { add: [[logits, 'xy'], [logitsAgg1, 'xi'], [logitsAgg2, 'xi']], special: BlKDepSpecial.Softmax },
        dimX: DimStyle.T, dimY: DimStyle.n_vocab,
        name: 'Output Probabilities',
    });

    s.y += vocabSize * cell + margin;

    let outputCubes = [lmHeadWeight, logits, logitsAgg1, logitsAgg2, logitsSoftmax];
    cubes.push(...outputCubes);
    let outputLabel = mkLabel(1.0, outputCubes);

    let decX0 = s.X0;
    let decHeight = s.y;

    for (let i = 0; i < cubes.length; i++) {
        cubes[i].idx = i;
    }

    let encLayerParams = 4 * C * C + 2 * 2 * C + (C * dFF + dFF + dFF * C + C);
    let decLayerParams = 8 * C * C + 3 * 2 * C + (C * dFF + dFF + dFF * C + C);
    let weightCount = vocabSize * C + nBlocks * (encLayerParams + decLayerParams);

    let enc = {
        kind: StackKind.Encoder,
        x0: 0,
        dir: -1,
        ...encEmbed,
        blocks: encBlocks,
        output: encOutput,
        height: encHeight,
    };

    let dec = {
        kind: StackKind.Decoder,
        x0: decX0,
        dir: 1,
        ...decEmbed,
        blocks: decBlocks,
        output: decX,
        height: decHeight,
    };

    return {
        kind: 'transformer' as const,
        cubes,
        cell,
        margin,
        enc,
        dec,
        lmHeadWeight,
        logits,
        logitsAgg1,
        logitsAgg2,
        logitsSoftmax,
        outputLabel,
        height: Math.max(encHeight, decHeight),
        midX: (enc.x0 + dec.x0) / 2,
        model,
        labels: [
            enc.embedLabel, dec.embedLabel, outputLabel,
            ...encBlocks.flatMap(b => b.labels),
            ...decBlocks.flatMap(b => b.labels),
        ],
        weightCount,
        shape,
    };
}
