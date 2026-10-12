import React from 'react';
import { LayerView } from '@/src/llm/LayerView';
import { InfoButton } from '@/src/llm/WelcomePopup';
import s from './app.module.scss';

export const App: React.FC = () => {
    return <div className={s.body}>
        <div className="flex justify-between items-center px-2 py-1 bg-blue-950 text-white h-[2.0rem] flex-shrink-0">
            <div className="flex items-center"><InfoButton /></div>
            <div className="text-xl">Transformer Visualization</div>
            <div className="flex gap-4 text-sm">
                <a className="hover:underline" href="https://arxiv.org/abs/1706.03762" target="_blank" rel="noreferrer">Attention Is All You Need</a>
                <a className="hover:underline" href="https://arxiv.org/abs/2010.11929" target="_blank" rel="noreferrer">An Image is Worth 16x16 Words</a>
                <a className="hover:underline" href="https://arxiv.org/abs/1810.04805" target="_blank" rel="noreferrer">BERT</a>
            </div>
        </div>
        <LayerView />
    </div>;
};
