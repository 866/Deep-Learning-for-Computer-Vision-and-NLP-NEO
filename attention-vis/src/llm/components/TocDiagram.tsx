import React, { memo, useCallback, useState } from 'react';
import { BoundingBox3d, Vec3 } from '@/src/utils/vector';
import s from './TocDiagram.module.scss';
import * as d3Color from 'd3-color';
import { isNotNil } from '@/src/utils/data';
import { familyOfPhase, Phase } from '../walkthrough/Walkthrough';
import clsx from 'clsx';
import { jumpToPhase } from '../Commentary';
import { useProgramState } from '../Sidebar';
import { Subscriptions } from '@/src/utils/hooks';

interface IEntryGroup {
    groupName: string;
    entries: IEntryInfo[];
}

interface IEntryInfo {
    id: Phase;
    title: string;
    ids: string[];
    groupIds: boolean;
}

interface ICell {
    id: string;
    label: string[];
    x: number; // center
    y: number; // top
    w: number;
    h: number;
    color: string;
}

/*
A diagram of the encoder-decoder Transformer, after Figure 1 of the paper (but flowing downwards, like the 3D view),
which doubles as the table of contents. For the Vision Transformer chapters, it shows the ViT paper's Figure 1 instead.
*/
export const TocDiagram: React.FC<{
    activePhase: Phase | null;
    onEnterPhase?: (phase: Phase) => void;
}> = ({ activePhase, onEnterPhase }) => {
    let [entryManager] = useState(() => new EntryManager());
    let [diagramEl, setDiagramEl] = useState<SVGElement | null>(null);
    let progState = useProgramState();

    let [hoverId, setHoverId] = useState<Phase | null>(null);

    let setHover = useCallback((ev: React.MouseEvent, id: Phase, hover: boolean) => {
         setHoverId(hover ? id : null);
    }, []);

    let [activeId, setActiveId] = useState<Phase | null>(null);

    let setActive = useCallback((ev: React.MouseEvent, id: Phase, active: boolean) => {
        jumpToPhase(progState.walkthrough, id);
        onEnterPhase?.(id);
        setActiveId(active ? id : null);
    }, [progState.walkthrough, onEnterPhase]);

    activeId = activePhase ?? activeId;

    let colors = {
        ln: '#e9f29e',
        multihead: '#f2d59e',
        feedForward: '#9ef2f2',
        tokEmbed: '#f0a8fc',
        linear: '#a8c3fc',
        softmax: '#a8fcaf',
        block: '#e4e4e4',
        line: '#000',
        focus: '#338',
    };

    let fontSize = 11;
    let encX = 88;
    let decX = 226;
    let cellW = 104;
    let blockPad = 8;

    let family = familyOfPhase((hoverId ?? activeId ?? Phase.Intro_Intro) as Phase);
    let vitMode = family === 'vit';
    let bertMode = family === 'bert';

    let cells: ICell[] = [];
    function cell(id: string, label: string | string[], x: number, y: number, h: number, color: string, w: number = cellW) {
        let c: ICell = { id, label: Array.isArray(label) ? label : [label], x, y, w, h, color };
        cells.push(c);
        return c;
    }

    let embedY = 44;
    let peY = 82;
    let blockTop = 100;

    let encEmbed = cell('encEmbed', 'Input Embedding', encX, embedY, 20, colors.tokEmbed);
    let encAttn = cell('encAttn', ['Multi-Head', 'Attention'], encX, blockTop + 22, 34, colors.multihead);
    let encLn1 = cell('encLn1', 'Add & Norm', encX, blockTop + 64, 20, colors.ln);
    let encFfn = cell('encFfn', ['Feed', 'Forward'], encX, blockTop + 102, 34, colors.feedForward);
    let encLn2 = cell('encLn2', 'Add & Norm', encX, blockTop + 144, 20, colors.ln);
    let encBlockBot = encLn2.y + encLn2.h + blockPad;

    let decEmbed = cell('decEmbed', 'Output Embedding', decX, embedY, 20, colors.tokEmbed);
    let decSelf = cell('decSelfAttn', ['Masked Multi-Head', 'Attention'], decX, blockTop + 22, 34, colors.multihead);
    let decLn1 = cell('decLn1', 'Add & Norm', decX, blockTop + 64, 20, colors.ln);
    let decCross = cell('decCrossAttn', ['Multi-Head', 'Attention'], decX, blockTop + 102, 34, colors.multihead);
    let decLn2 = cell('decLn2', 'Add & Norm', decX, blockTop + 144, 20, colors.ln);
    let decFfn = cell('decFfn', ['Feed', 'Forward'], decX, blockTop + 182, 34, colors.feedForward);
    let decLn3 = cell('decLn3', 'Add & Norm', decX, blockTop + 224, 20, colors.ln);
    let decBlockBot = decLn3.y + decLn3.h + blockPad;

    let linear = cell('linear', 'Linear', decX, decBlockBot + 18, 20, colors.linear);
    let softmax = cell('softmaxOut', 'Softmax', decX, linear.y + 38, 20, colors.softmax);

    let blocks: Record<string, BoundingBox3d> = {
        encBlock: new BoundingBox3d(new Vec3(encX - cellW / 2 - blockPad, blockTop), new Vec3(encX + cellW / 2 + blockPad, encBlockBot)),
        decBlock: new BoundingBox3d(new Vec3(decX - cellW / 2 - blockPad, blockTop), new Vec3(decX + cellW / 2 + blockPad, decBlockBot)),
        encPE: new BoundingBox3d(new Vec3(encX - 46, peY - 11), new Vec3(encX + 10, peY + 11)),
        decPE: new BoundingBox3d(new Vec3(decX - 10, peY - 11), new Vec3(decX + 46, peY + 11)),
    };

    let height = softmax.y + softmax.h + 34;
    let width = 310;

    let entries: IEntryInfo[] = [];
    let entryGroups: IEntryGroup[] = [];

    function makeEntry(id: Phase, title: string, ids: string[] = [], groupIds: boolean = false) {
        let entry = { id, title, ids, groupIds };
        entries.push(entry);
        entryGroups[entryGroups.length - 1].entries.push(entry);
    }

    entryGroups.push({ groupName: 'Intro', entries: [] });
    makeEntry(Phase.Intro_Intro, 'Introduction', []);
    makeEntry(Phase.Intro_Prelim, 'Preliminaries', []);

    entryGroups.push({ groupName: 'Encoder', entries: [] });
    makeEntry(Phase.Enc_Embedding, 'Embedding', ['encEmbed', 'encPE'], true);
    makeEntry(Phase.Enc_SelfAttention, 'Self-Attention', ['encAttn']);
    makeEntry(Phase.Enc_Softmax, 'Softmax', ['encAttn', 'softmaxOut']);
    makeEntry(Phase.Enc_Projection, 'Multi-Head Attention', ['encAttn']);
    makeEntry(Phase.Enc_AddNorm, 'Add & Norm', ['encLn1', 'encLn2', 'decLn1', 'decLn2', 'decLn3']);
    makeEntry(Phase.Enc_FeedForward, 'Feed Forward', ['encFfn', 'decFfn']);
    makeEntry(Phase.Enc_Stack, 'Encoder Stack', ['encBlock']);

    entryGroups.push({ groupName: 'Decoder', entries: [] });
    makeEntry(Phase.Dec_MaskedAttention, 'Masked Self-Attention', ['decEmbed', 'decSelfAttn']);
    makeEntry(Phase.Dec_CrossAttention, 'Cross-Attention', ['decCrossAttn']);
    makeEntry(Phase.Dec_Stack, 'Decoder Stack', ['decBlock']);
    makeEntry(Phase.Dec_Output, 'Output & Decoding', ['linear', 'softmaxOut'], true);

    entryGroups.push({ groupName: 'BERT', entries: [] });
    makeEntry(Phase.Bert_Intro, 'BERT Overview', []);
    makeEntry(Phase.Bert_Input, 'Tokens, Segments & Positions', ['bertEmb', 'bertSum'], true);
    makeEntry(Phase.Bert_Attn, 'Bidirectional Attention', ['bertBlock', 'bertMha']);
    makeEntry(Phase.Bert_Mlm, 'Masked Language Model', ['bertMlm']);
    makeEntry(Phase.Bert_PretrainFT, 'Pre-training & Fine-tuning', ['bertMlm', 'bertCls']);
    makeEntry(Phase.Bert_Sentiment, 'Sentiment Classifier', ['bertCls']);

    entryGroups.push({ groupName: 'Vision Transformer', entries: [] });
    makeEntry(Phase.Vit_Intro, 'ViT Overview', []);
    makeEntry(Phase.Vit_Embedding, 'Patch & Position Embeddings', ['vitProj', 'vitTokens'], true);
    makeEntry(Phase.Vit_Encoder, 'Encoder (Pre-Norm)', ['vitBlock']);
    makeEntry(Phase.Vit_Head, 'Classification Head', ['vitHead']);
    makeEntry(Phase.Vit_Attention, 'Attention Maps & Scale', ['vitMha']);

    function buildBertDiagram() {
        let x = 150;
        let emb = cell('bertEmb', ['Token + Segment', '+ Position Embedding'], x, 20, 26, colors.tokEmbed, 200);
        let sum = cell('bertSum', 'Embedding (z\u2080)', x, emb.y + emb.h + 12, 18, colors.tokEmbed, 140);
        let blockTop = sum.y + sum.h + 18;
        let ln1 = cell('bertLn1', 'Norm', x, blockTop + 18, 18, colors.ln);
        let mha = cell('bertMha', ['Bidirectional', 'Self-Attention'], x, ln1.y + ln1.h + 14, 32, colors.multihead);
        let plus1Y = mha.y + mha.h + 14;
        let ln2 = cell('bertLn2', 'Norm', x, plus1Y + 16, 18, colors.ln);
        let ffn = cell('bertFfn', 'FFN', x, ln2.y + ln2.h + 12, 20, colors.feedForward);
        let plus2Y = ffn.y + ffn.h + 12;
        let blockBot = plus2Y + 6 + blockPad;
        blocks.bertBlock = new BoundingBox3d(new Vec3(x - cellW / 2 - blockPad, blockTop), new Vec3(x + cellW / 2 + blockPad, blockBot));

        let mlm = cell('bertMlm', ['MLM Head', '(E\u1D40 tied to the token table)'], x, blockBot + 22, 28, colors.softmax, 210);
        let cls = cell('bertCls', ['[CLS] \u2192 pool \u2192 NSP', 'and sentiment classifier'], x, mlm.y + mlm.h + 16, 28, colors.linear, 210);
        let bertHeight = cls.y + cls.h + 34;

        vert(emb, sum, x);
        line([new Vec3(x, sum.y + sum.h), new Vec3(x, blockTop + 6)]);
        // pre-norm, drawn the way the ViT's is: the residual leaves before the norm and rejoins after the sublayer
        let resX = x + cellW / 2 + 4;
        line([new Vec3(x, blockTop + 10), new Vec3(resX, blockTop + 10), new Vec3(resX, plus1Y), new Vec3(x + 7, plus1Y)]);
        qkvInto(mha, ln1.y + ln1.h, x);
        line([new Vec3(x, mha.y + mha.h), new Vec3(x, plus1Y - 7)]);
        line([new Vec3(x, plus1Y + 6), new Vec3(x, ln2.y - 1)]);
        line([new Vec3(x, plus1Y + 10), new Vec3(resX, plus1Y + 10), new Vec3(resX, plus2Y), new Vec3(x + 7, plus2Y)]);
        vert(ln2, ffn, x);
        line([new Vec3(x, ffn.y + ffn.h), new Vec3(x, plus2Y - 7)]);
        line([new Vec3(x, plus2Y + 6), new Vec3(x, mlm.y - 1)]);
        vert(mlm, cls, x);

        return {
            height: bertHeight,
            content: <>
                {renderBlock('bertBlock', -1, 'L\u00d7')}
            </>,
        };
    }

    function findByIdBounds(id: string): BoundingBox3d | null {
        let c = cells.find(c => c.id === id);
        if (c) {
            return new BoundingBox3d(new Vec3(c.x - c.w / 2, c.y), new Vec3(c.x + c.w / 2, c.y + c.h));
        }
        return (blocks as Record<string, BoundingBox3d>)[id] ?? null;
    }

    function idIsDimmed(id: string): boolean {
        let targetId = isNotNil(hoverId) ? hoverId : activeId;
        let entry = entries.find(e => e.id === targetId);
        if (!entry || entry.ids.length === 0) {
            return false;
        }
        return !entry.ids.includes(id);
    }

    function renderPlus(pos: Vec3, key: string) {
        let plusLen = 4;
        return <g key={key}>
            <circle cx={pos.x} cy={pos.y} r={6} stroke='#000000' fill='white' />
            <line x1={pos.x - plusLen} y1={pos.y} x2={pos.x + plusLen} y2={pos.y} stroke='#000000' strokeWidth={1} />
            <line x1={pos.x} y1={pos.y - plusLen} x2={pos.x} y2={pos.y + plusLen} stroke='#000000' strokeWidth={1} />
        </g>;
    }

    function renderArrow(to: Vec3, from: Vec3, key: string) {
        let dir = to.sub(from).normalize();
        let dirProp = new Vec3(dir.y, -dir.x);
        let arrowLen = 5;
        let arrowWidth = 4;
        let left = to.mulAdd(dir, -arrowLen).mulAdd(dirProp, -arrowWidth / 2);
        let right = to.mulAdd(dir, -arrowLen).mulAdd(dirProp, arrowWidth / 2);
        let path = `M${left.x},${left.y} L${to.x},${to.y} L${right.x},${right.y}Z`;
        return <path key={key} d={path} fill={colors.line} className={s.dataPath} />;
    }

    let paths: React.ReactNode[] = [];
    let pathIdx = 0;
    function line(pts: Vec3[], arrow: boolean = true) {
        let d = pts.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ');
        paths.push(<path key={pathIdx++} d={d} fill='none' strokeLinejoin='round' className={s.dataPath} />);
        if (arrow) {
            paths.push(renderArrow(pts[pts.length - 1], pts[pts.length - 2], `arr${pathIdx++}`));
        }
    }

    function vert(from: ICell | number, to: ICell, x: number) {
        let y0 = typeof from === 'number' ? from : from.y + from.h;
        line([new Vec3(x, y0), new Vec3(x, to.y - 1)]);
    }

    // three-pronged arrow into an attention cell (Q, K, V)
    function qkvInto(target: ICell, fromY: number, srcX: number) {
        let spread = 26;
        let midY = target.y - 7;
        for (let dx of [-spread, 0, spread]) {
            line([new Vec3(srcX, fromY), new Vec3(srcX, midY), new Vec3(target.x + dx, midY), new Vec3(target.x + dx, target.y - 1)]);
        }
    }

    // residual connection around a sublayer, on the outer side
    function residual(from: number, target: ICell, side: number) {
        let x = target.x + side * (cellW / 2 + 4);
        let cx = target.x;
        let targetY = target.y + target.h / 2;
        line([new Vec3(cx, from), new Vec3(x, from), new Vec3(x, targetY), new Vec3(target.x + side * cellW / 2 + 1, targetY)]);
    }

    function renderStackPaths(embed: ICell, peX: number, side: number, sublayers: [ICell, ICell][], blockBot: number) {
        // embedding + positional encoding
        let x = embed.x;
        line([new Vec3(x, embed.y + embed.h), new Vec3(x, peY - 7)], false);
        line([new Vec3(x, peY + 6), new Vec3(x, blockTop + 4)], false);
        let prevY = blockTop + 4;
        for (let [sub, ln] of sublayers) {
            let isAttn = sub.label.join(' ').includes('Attention');
            if (isAttn && sub.id !== 'decCrossAttn') {
                qkvInto(sub, prevY, x);
            } else if (sub.id === 'decCrossAttn') {
                // Q from below (decoder), K & V from the encoder
                line([new Vec3(x, prevY), new Vec3(x, sub.y - 7), new Vec3(sub.x + 26, sub.y - 7), new Vec3(sub.x + 26, sub.y - 1)]);
            } else {
                vert(prevY, sub, x);
            }
            vert(sub, ln, x);
            residual(prevY + 4, ln, side);
            prevY = ln.y + ln.h;
        }
        line([new Vec3(x, prevY), new Vec3(x, blockBot + (side > 0 ? 16 : 6))], side > 0);
    }

    renderStackPaths(encEmbed, encX - 30, -1, [[encAttn, encLn1], [encFfn, encLn2]], encBlockBot);
    renderStackPaths(decEmbed, decX + 30, 1, [[decSelf, decLn1], [decCross, decLn2], [decFfn, decLn3]], decBlockBot);
    vert(linear, softmax, decX);

    // encoder output => cross-attention K & V
    {
        let gapX = (encX + cellW / 2 + blockPad + decX - cellW / 2 - blockPad) / 2;
        let start = new Vec3(encX, encBlockBot + 6);
        let kvY = decCross.y - 13;
        line([start, new Vec3(encX, encBlockBot + 16), new Vec3(gapX, encBlockBot + 16), new Vec3(gapX, kvY),
            new Vec3(decCross.x - 26, kvY), new Vec3(decCross.x - 26, decCross.y - 1)]);
        line([new Vec3(decCross.x - 26, kvY), new Vec3(decCross.x, kvY), new Vec3(decCross.x, decCross.y - 1)]);
    }

    function renderCell(c: ICell) {
        let stroke = d3Color.color(c.color)?.darker(0.5).toString();
        return <g key={c.id} transform={`translate(${c.x - c.w / 2}, ${c.y})`} opacity={idIsDimmed(c.id) ? 0.5 : 1}>
            <rect className={s.cell} width={c.w} height={c.h} fill={c.color} rx={3} ry={3} stroke={stroke} strokeWidth={1} />
            {c.label.map((l, i) => <text key={i} x={c.w / 2} y={c.h / 2 + (i - (c.label.length - 1) / 2) * 13 + 4} fontSize={fontSize} textAnchor='middle'>{l}</text>)}
        </g>;
    }

    function renderPosEnc(x: number, side: number, key: string) {
        let embedX = x + side * 26;
        let textX = embedX + side * 12;
        return <g key={key} opacity={idIsDimmed(key) ? 0.5 : 1}>
            <circle cx={embedX} cy={peY} r={9} stroke={colors.line} fill='none' />
            <text textAnchor='middle' x={embedX} y={peY + 5} fontSize={16}>~</text>
            <line x1={embedX - side * 9} x2={x + side * 8} y1={peY} y2={peY} stroke={colors.line} strokeWidth={1} />
            {renderArrow(new Vec3(x + side * 7, peY), new Vec3(x + side * 12, peY), key + 'arrow')}
            <text fontSize={8} textAnchor={side < 0 ? 'end' : 'start'} x={textX} y={peY - 1}>Positional</text>
            <text fontSize={8} textAnchor={side < 0 ? 'end' : 'start'} x={textX} y={peY + 8}>Encoding</text>
            {renderPlus(new Vec3(x, peY), key + 'plus')}
        </g>;
    }

    function renderBlock(id: string, side: number, label: string = 'N×') {
        let bb = blocks[id];
        let labelX = side < 0 ? bb.min.x - 3 : bb.max.x + 3;
        return <g key={id}>
            <rect className={s.block} x={bb.min.x} y={bb.min.y} width={bb.size().x} height={bb.size().y} fill={colors.block} rx={4} ry={4} stroke={'#bbb'} />
            <text fontSize={12} textAnchor={side < 0 ? 'end' : 'start'} x={labelX} y={(bb.min.y + bb.max.y) / 2 + 4} fill={'#555'}>{label}</text>
        </g>;
    }

    // The ViT paper's Figure 1 (flowing downwards): image -> patches -> linear projection -> tokens (+ position)
    // -> pre-norm Transformer encoder (L x) -> MLP head on the [class] token.
    function buildVitDiagram() {
        let x = 150;
        let model = progState.vit.model;
        let imgPx = 44;
        let imgTop = 14;
        let projY = imgTop + imgPx + 16;
        let proj = cell('vitProj', 'Linear Projection of Flattened Patches', x, projY, 20, colors.tokEmbed, 200);
        let tokY = projY + 20 + 16;
        let nTok = model ? model.shape.T : 17;
        let tokW = 9, tokH = 12, tokGap = 1;
        let rowW = nTok * (tokW + tokGap) - tokGap;
        let tokX0 = x - rowW / 2;
        blocks.vitTokens = new BoundingBox3d(new Vec3(tokX0, tokY), new Vec3(tokX0 + rowW, tokY + tokH * 2 + 2));

        let blockTop = tokY + tokH * 2 + 20;
        let ln1 = cell('vitLn1', 'Norm', x, blockTop + 18, 18, colors.ln);
        let mha = cell('vitMha', ['Multi-Head', 'Attention'], x, ln1.y + 18 + 16, 32, colors.multihead);
        let plus1Y = mha.y + 32 + 12;
        let ln2 = cell('vitLn2', 'Norm', x, plus1Y + 16, 18, colors.ln);
        let mlp = cell('vitMlp', 'MLP', x, ln2.y + 18 + 12, 20, colors.feedForward);
        let plus2Y = mlp.y + 20 + 12;
        let blockBot = plus2Y + 6 + blockPad;
        blocks.vitBlock = new BoundingBox3d(new Vec3(x - cellW / 2 - blockPad, blockTop), new Vec3(x + cellW / 2 + blockPad, blockBot));
        let head = cell('vitHead', 'MLP Head', x, blockBot + 22, 20, colors.linear);
        let vitHeight = head.y + head.h + 34;

        line([new Vec3(x, imgTop + imgPx + 2), new Vec3(x, proj.y - 1)]);
        line([new Vec3(x, proj.y + proj.h), new Vec3(x, tokY - 1)]);
        line([new Vec3(x, tokY + tokH * 2 + 2), new Vec3(x, blockTop + 6)], false);
        // pre-norm: the residual branches off before the norm, and is added after the sublayer
        let resX = x + cellW / 2 + 4;
        line([new Vec3(x, blockTop + 6), new Vec3(x, ln1.y - 1)]);
        line([new Vec3(x, blockTop + 10), new Vec3(resX, blockTop + 10), new Vec3(resX, plus1Y), new Vec3(x + 7, plus1Y)]);
        qkvInto(mha, ln1.y + ln1.h, x);
        line([new Vec3(x, mha.y + mha.h), new Vec3(x, plus1Y - 7)]);
        line([new Vec3(x, plus1Y + 6), new Vec3(x, ln2.y - 1)]);
        line([new Vec3(x, plus1Y + 10), new Vec3(resX, plus1Y + 10), new Vec3(resX, plus2Y), new Vec3(x + 7, plus2Y)]);
        vert(ln2, mlp, x);
        line([new Vec3(x, mlp.y + mlp.h), new Vec3(x, plus2Y - 7)]);
        line([new Vec3(x, plus2Y + 6), new Vec3(x, head.y - 1)]);
        line([new Vec3(x, head.y + head.h), new Vec3(x, vitHeight - 26)]);

        let g = model ? model.shape.imageSize / model.shape.patchSize : 4;
        let imgEls: React.ReactNode[] = [];
        if (model) {
            let n = model.shape.imageSize;
            let px = imgPx / n;
            for (let i = 0; i < n * n; i++) {
                let v = Math.round(255 * (1 - Math.min(1, model.image[i]) * 0.92));
                imgEls.push(<rect key={i} x={x - imgPx / 2 + (i % n) * px} y={imgTop + Math.floor(i / n) * px} width={px + 0.2} height={px + 0.2} fill={`rgb(${v},${v},${v})`} />);
            }
        }
        for (let k = 1; k < g; k++) {
            let o = k * imgPx / g;
            imgEls.push(<line key={'v' + k} x1={x - imgPx / 2 + o} x2={x - imgPx / 2 + o} y1={imgTop} y2={imgTop + imgPx} stroke='#d33' strokeWidth={0.8} />);
            imgEls.push(<line key={'h' + k} x1={x - imgPx / 2} x2={x + imgPx / 2} y1={imgTop + o} y2={imgTop + o} stroke='#d33' strokeWidth={0.8} />);
        }
        imgEls.push(<rect key='border' x={x - imgPx / 2} y={imgTop} width={imgPx} height={imgPx} fill='none' stroke='#888' />);

        let toks: React.ReactNode[] = [];
        for (let i = 0; i < nTok; i++) {
            let tx = tokX0 + i * (tokW + tokGap);
            // position embedding (top) + patch embedding (bottom), like the paper's purple & pink boxes
            toks.push(<rect key={'p' + i} x={tx} y={tokY} width={tokW} height={tokH} fill='#d9c8f5' stroke='#9a86c4' strokeWidth={0.6} />);
            toks.push(<text key={'pt' + i} x={tx + tokW / 2} y={tokY + 9} fontSize={6.5} textAnchor='middle' fill='#444'>{i}</text>);
            toks.push(<rect key={'e' + i} x={tx} y={tokY + tokH + 2} width={tokW} height={tokH} fill={i === 0 ? '#f7c873' : colors.tokEmbed} stroke='#b38bbd' strokeWidth={0.6} />);
            if (i === 0) {
                toks.push(<text key='cls' x={tx + tokW / 2} y={tokY + tokH + 11} fontSize={8} textAnchor='middle' fill='#444'>*</text>);
            }
        }

        let pred = '';
        if (model?.probs.localBuffer) {
            let p = model.probs.localBuffer;
            let best = 0;
            for (let i = 1; i < p.length; i++) if (p[i] > p[best]) best = i;
            pred = model.config.classes[best];
        }

        let content = <>
            <text fontSize={10} textAnchor='middle' x={x} y={10} fill='#666'>Input image, cut into patches</text>
            {imgEls}
            {toks}
            <text fontSize={8} textAnchor='start' x={tokX0 + rowW + 4} y={tokY + 9} fill='#555'>Position</text>
            <text fontSize={8} textAnchor='start' x={tokX0 + rowW + 4} y={tokY + tokH + 11} fill='#555'>Patch</text>
            <text fontSize={8} textAnchor='end' x={tokX0 - 4} y={tokY + tokH + 11} fill='#555'>[class] *</text>
            {renderPlus(new Vec3(x, plus1Y), 'vitPlus1')}
            {renderPlus(new Vec3(x, plus2Y), 'vitPlus2')}
            <text fontSize={10} textAnchor='middle' x={x} y={vitHeight - 16} fill='#666'>Class</text>
            <text fontSize={13} textAnchor='middle' x={x} y={vitHeight - 2} fill='#333'>{pred || '7'}</text>
        </>;
        return { height: vitHeight, content };
    }

    function renderIoText() {
        let model = progState.model;
        let src = model ? [...model.enc.inputBuf].map(i => model!.vocab[i]).join(' ') : 'C A B B C A';
        let tgt = model ? [...model.dec.inputBuf].map(i => model!.vocab[i]).join(' ') : '<s> A C B B A';
        let out = model ? [...model.enc.inputBuf].reverse().map(i => model!.vocab[i]).join(' ') : 'A C B B A C';
        line([new Vec3(encX, 34), new Vec3(encX, embedY - 1)]);
        line([new Vec3(decX, 34), new Vec3(decX, embedY - 1)]);
        line([new Vec3(decX, softmax.y + softmax.h), new Vec3(decX, height - 26)]);
        return <>
            <text fontSize={10} textAnchor='middle' x={encX} y={12} fill='#666'>Inputs</text>
            <text fontSize={13} textAnchor='middle' x={encX} y={30} fill='#333'>{src}</text>
            <text fontSize={10} textAnchor='middle' x={decX} y={12} fill='#666'>Outputs (shifted right)</text>
            <text fontSize={13} textAnchor='middle' x={decX} y={30} fill='#333'>{tgt}</text>
            <text fontSize={10} textAnchor='middle' x={decX} y={height - 16} fill='#666'>Output Probabilities</text>
            <text fontSize={13} textAnchor='middle' x={decX} y={height - 2} fill='#333'>{out}</text>
        </>;
    }

    function getFocusBounds(entry: IEntryInfo): BoundingBox3d[] {
        let idBounds = entry.ids.map(findByIdBounds).filter(isNotNil);

        if (entry.groupIds) {
            let box = new BoundingBox3d();
            idBounds.forEach(b => box.combineInPlace(b));
            return [box];
        } else {
            return idBounds;
        }
    }

    function renderFocusBoxes() {
        let focusId = hoverId ?? activeId;
        let hoverEntry = entries.find(e => e.id === focusId);
        return hoverEntry ? getFocusBounds(hoverEntry).map((bb, i) => renderFocusBox(bb, i)) : [];
    }

    function renderFocusBox(bb: BoundingBox3d, i: number) {
        let innerPad = new Vec3(4, 4);
        let tl = bb.min.sub(innerPad);
        let br = bb.max.add(innerPad);
        let pad = 30;

        let color = colors.focus;

        return <React.Fragment key={i}>
            <defs>
                <mask id={"hole" + i}>
                    <rect x={tl.x - pad} y={tl.y - pad} width={br.x - tl.x + 2 * pad} height={br.y - tl.y + 2 * pad} fill={'white'} />
                    <rect x={tl.x} y={tl.y} width={br.x - tl.x} height={br.y - tl.y} fill="black"/>
                </mask>
            </defs>
            <rect x={tl.x} y={tl.y} width={br.x - tl.x} height={br.y - tl.y} fill={color}
                mask={`url(#hole${i})`}
                style={{ filter: `drop-shadow(0px 0px 5px ${color})`}}
                />
            <rect x={tl.x} y={tl.y} width={br.x - tl.x} height={br.y - tl.y} fill={'none'} stroke={'#338a'} strokeWidth={2} strokeDasharray={'8,4'} />
        </React.Fragment>;
    }

    function renderTocToDigramLines() {
        if (!diagramEl) {
            return null;
        }

        let svgRect = diagramEl.getBoundingClientRect();

        let offsetLeft = width - 6;
        let i = 0;
        let result: React.ReactNode[] = [];

        let focusId = hoverId ?? activeId;
        let entry = entries.find(e => e.id === focusId);
        if (!entry) {
            return null;
        }
        let tocEl = entryManager.entries.get(entry.id);
        if (!tocEl) {
            return null;
        }
        let tocRect = tocEl.getBoundingClientRect();
        let endPt = new Vec3(tocRect.left - svgRect.left - 4, tocRect.top - svgRect.top + tocRect.height / 2);
        let allBounds = getFocusBounds(entry);
        if (allBounds.length === 0) {
            return null;
        }

        let yPts = [...allBounds.map(b => b.center().y), endPt.y];
        let minY = Math.min(...yPts);
        let maxY = Math.max(...yPts);
        let midX = offsetLeft;
        let pathOpts = { stroke: colors.focus, strokeWidth: 1, fill: 'none' };
        result.push(<path key={i++} {...pathOpts} d={`M${midX},${minY} L${midX},${maxY} M${midX},${endPt.y} L${endPt.x},${endPt.y}`} />);

        for (let bound of allBounds) {
            let startPt = new Vec3(bound.max.x + 8, bound.center().y);
            result.push(<path key={i++} {...pathOpts} d={`M${startPt.x},${startPt.y} L${midX},${startPt.y}`} />);
        }

        return result;
    }

    let ioText = renderIoText();

    // build the ViT diagram into its own cells & paths, then pick which one to show
    let transformerCells = cells, transformerPaths = paths;
    cells = [];
    paths = [];
    let vitDiagram = buildVitDiagram();
    cells = [];
    paths = [];
    let bertDiagram = buildBertDiagram();
    if (bertMode) {
        height = bertDiagram.height;
    } else if (vitMode) {
        height = vitDiagram.height;
    } else {
        cells = transformerCells;
        paths = transformerPaths;
    }

    return <div>
        <div className={s.tocDiagram}>
            <svg viewBox={`0 0 ${width} ${height}`} width={`${width}px`} height={height} ref={setDiagramEl}>
                {bertMode ? <>
                    {bertDiagram.content}
                    {paths}
                    {cells.map(renderCell)}
                </> : vitMode ? <>
                    {renderBlock('vitBlock', -1, 'L×')}
                    {paths}
                    {vitDiagram.content}
                    {cells.map(renderCell)}
                </> : <>
                    {renderBlock('encBlock', -1)}
                    {renderBlock('decBlock', 1)}
                    {paths}
                    {ioText}
                    {cells.map(renderCell)}
                    {renderPosEnc(encX, -1, 'encPE')}
                    {renderPosEnc(decX, 1, 'decPE')}
                </>}
                {renderFocusBoxes()}
                {renderTocToDigramLines()}
            </svg>
            <div className={s.toc}>
                <div className={s.tocTitle}>Table of Contents</div>
                {entryGroups.map((group, i) => {

                    return <React.Fragment key={i}>
                        <div className={s.tocGroupTitle}>{group.groupName}</div>
                        {group.entries.map((entry, j) => {
                            return <MenuEntry
                                key={j}
                                entryManager={entryManager}
                                title={entry.title}
                                id={entry.id}
                                active={entry.id === activeId}
                                hover={entry.id === hoverId}
                                setHover={setHover}
                                setActive={setActive}
                            />;
                        })}
                </React.Fragment>;
                })}
            </div>
        </div>
    </div>;
};

class EntryManager {
    subscriptions = new Subscriptions();
    entries = new Map<Phase, HTMLDivElement | null>();
    setEl(id: Phase, ref: HTMLDivElement | null) {
        if (ref) {
            this.entries.set(id, ref);
            return () => {
                this.entries.delete(id);
            }
        } else {
            this.entries.delete(id);
        }
    }
}

export const MenuEntry: React.FC<{
    entryManager: EntryManager,
    title: string,
    id: Phase,
    active: boolean,
    hover: boolean,
    setHover: (ev: React.MouseEvent, id: Phase, hover: boolean) => void,
    setActive: (ev: React.MouseEvent, id: Phase, active: boolean) => void,
}> = memo(function MenuEntry({ entryManager, id, title, active, hover, setHover, setActive }) {

    let setDivRef = useCallback((div: HTMLDivElement | null) => {
        entryManager.setEl(id, div);
    }, [entryManager, id]);


    function handleClick(ev: React.MouseEvent) {
        setActive(ev, id, !active);
    }

    function handleHover(ev: React.MouseEvent, hover: boolean) {
        setHover(ev, id, hover);
    }

    return <div
        ref={setDivRef}
        className={clsx("cursor-pointer p-1 pl-3 " + s.menuEntry, active && "bg-blue-200", hover && "bg-blue-800 text-white")}
        onClick={handleClick}
        onMouseEnter={ev => handleHover(ev, true)}
        onMouseLeave={ev => handleHover(ev, false)}
    >
        {title}
    </div>;
});
