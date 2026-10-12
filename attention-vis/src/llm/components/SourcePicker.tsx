import React, { useState } from 'react';
import clsx from 'clsx';
import { useProgramState } from '../Sidebar';
import { DimStyle, dimStyleColor } from '../walkthrough/WalkthroughTools';

/** Lets the reader type in their own source sequence; the whole model re-runs on it. */
export const SourcePicker: React.FC = () => {
    let state = useProgramState();
    let model = state.model;
    let T = state.shape.T;
    let letters = model ? model.vocab.slice(0, model.config.bos) : ['A', 'B', 'C'];
    let current = model ? [...model.enc.inputBuf] : [2, 0, 1, 1, 2, 0];
    let [draft, setDraft] = useState<number[] | null>(null);
    let seq = draft ?? current;

    function cycle(i: number) {
        let next = [...seq];
        next[i] = (next[i] + 1) % letters.length;
        setDraft(next);
    }

    function run(src: number[]) {
        state.pendingSource = src;
        setDraft(null);
        state.markDirty();
    }

    function randomize() {
        run(Array.from({ length: T }, () => Math.floor(Math.random() * letters.length)));
    }

    let dirty = draft && draft.some((v, i) => v !== current[i]);
    let output = model ? [...model.dec.inputBuf.slice(1), model.sortedBuf![(T - 1) * model.shape.vocabSize * 2]] : [];

    return <div className='my-2 flex flex-col items-center'>
        <div className='flex items-center'>
            {seq.map((tok, i) => <button
                key={i}
                className='w-8 h-9 m-[2px] rounded border border-gray-400 bg-white hover:bg-blue-100 text-xl'
                style={{ color: dimStyleColor(DimStyle.Token).toHexColor() }}
                onClick={() => cycle(i)}
                title='Click to change'
            >{letters[tok]}</button>)}
        </div>
        <div className='flex mt-1 text-sm'>
            <button className={clsx('px-3 py-1 m-1 rounded shadow', dirty ? 'bg-blue-300 hover:bg-blue-400' : 'bg-gray-200 text-gray-500')} onClick={() => dirty && run(draft!)}>Run</button>
            <button className='px-3 py-1 m-1 rounded shadow bg-gray-200 hover:bg-blue-200' onClick={randomize}>Random</button>
        </div>
        {model && <div className='text-sm text-gray-600 mt-1'>
            model output: <span style={{ color: dimStyleColor(DimStyle.Token).toHexColor() }}>{output.map(i => model!.vocab[i]).join(' ')}</span>
        </div>}
    </div>;
};
