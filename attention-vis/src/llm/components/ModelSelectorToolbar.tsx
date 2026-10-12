import React from 'react';
import { useProgramState } from '../Sidebar';
import clsx from 'clsx';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faExpand, faEye, faEyeSlash, faMagnifyingGlass } from '@fortawesome/free-solid-svg-icons';
import { Vec3 } from '@/src/utils/vector';
import { Mat4f } from '@/src/utils/matrix';
import { currentMainExample, IModelExample, MainKind, setExampleEnabled } from '../Program';
import { Family, familyOfPhase, Phase } from '../walkthrough/Walkthrough';
import { jumpToPhase } from '../Commentary';

export const ModelSelectorToolbar: React.FC<{
}> = () => {
    let progState = useProgramState();

    // the tiny models: they live at the origin, and which one is shown follows the walkthrough chapter
    const MAIN_PHASE: Record<Family, Phase> = {
        transformer: Phase.Intro_Intro, vit: Phase.Vit_Intro, bert: Phase.Bert_Intro,
    };
    const MAIN_TITLE: Record<Family, string> = {
        transformer: 'The tiny encoder-decoder Transformer (opens its chapters)',
        vit: 'The tiny Vision Transformer (opens its chapters)',
        bert: 'The tiny BERT, trained on SST-2 (opens its chapters)',
    };

    function makeMainButton(kind: MainKind) {
        let example = kind === 'vit' ? progState.vitMainExample : kind === 'bert' ? progState.bertMainExample : progState.mainExample;
        let isActive = progState.currExampleId === -1 && progState.mainKind === kind;

        function handleClick() {
            let wt = progState.walkthrough;
            if (familyOfPhase(wt.phase) !== kind) {
                jumpToPhase(wt, MAIN_PHASE[kind]);
            }
            progState.currExampleId = -1;
            progState.camera.desiredCamera = example.camera;
            progState.markDirty();
        }

        return <div className={clsx('m-1 flex items-stretch rounded shadow overflow-hidden', isActive ? 'bg-blue-200' : 'bg-white')}>
            <div className='px-2 py-1 cursor-pointer hover:bg-blue-300' onClick={handleClick} title={MAIN_TITLE[kind]}>
                {example.name}
            </div>
        </div>;
    }

    // the paper-sized models: shapes only, each can be turned on/off
    function makeButton(example: IModelExample) {
        let egIndex = progState.examples.indexOf(example);
        let isEnabled = example.enabled;
        let isActive = progState.currExampleId === egIndex;

        function handleClick() {
            setExampleEnabled(progState, example, true);
            progState.currExampleId = egIndex;
            progState.camera.desiredCamera = example.camera;
            progState.markDirty();
        }

        function handleToggle(ev: React.MouseEvent) {
            ev.stopPropagation();
            setExampleEnabled(progState, example, !isEnabled);
        }

        return <div key={example.name} className={clsx('m-1 flex items-stretch rounded shadow overflow-hidden', isActive ? 'bg-blue-200' : 'bg-white', !isEnabled && 'text-gray-400')}>
            <div className='px-2 py-1 cursor-pointer hover:bg-blue-300' onClick={handleClick} title={isEnabled ? undefined : 'Turn on and go to this model'}>
                {example.name}
            </div>
            <div
                className='px-2 flex items-center cursor-pointer border-l border-gray-300 hover:bg-blue-300'
                onClick={handleToggle}
                title={isEnabled ? 'Turn off (frees its memory and rendering)' : 'Turn on'}
            >
                <FontAwesomeIcon icon={isEnabled ? faEye : faEyeSlash} />
            </div>
        </div>;
    }

    function currentExample() {
        return progState.examples[progState.currExampleId] ?? currentMainExample(progState);
    }

    function onExpandClick() {
        progState.camera.desiredCamera = currentExample().camera;
        progState.markDirty();
    }

    function onMagnifyClick() {
        let example = currentExample();
        let layout = example.layout ?? progState.mainLayout;

        let obj = layout.kind === 'transformer' ? layout.enc.residual0 : layout.residual0;
        let modelTarget = new Vec3(obj.x, obj.y, obj.z);
        let modelMtx = progState.camera.modelMtx.mul(Mat4f.fromTranslation(example.offset))

        let center = modelMtx.mulVec3Proj(modelTarget);
        let zoom = progState.currExampleId === -1 ? 0.7 : 4;
        progState.camera.desiredCamera = {
            center, angle: new Vec3(270, 4.5, zoom),
        };
        progState.markDirty();
    }

    let transformerExamples = progState.examples.filter(e => e.family === 'transformer');
    let vitExamples = progState.examples.filter(e => e.family === 'vit');
    let bertExamples = progState.examples.filter(e => e.family === 'bert');

    // One row per model family, keyed by the Family union rather than hand-written JSX per family.
    // The hand-written version rendered two rows and compiled: BERT had a main card, two paper
    // shapes and a `berts` filter, and nothing on screen - its chapters were unreachable from the
    // canvas. Keyed by Family, a family without a row is a type error. Insertion order is the
    // on-screen order; `leading` puts a paper shape to the left of its tiny model (Transformer's
    // base sits left of the tiny one, which is how the row was laid out before BERT existed).
    let rows: Record<Family, { leading?: IModelExample; rest: IModelExample[] }> = {
        transformer: { leading: transformerExamples[0], rest: transformerExamples.slice(1) },
        vit: { rest: vitExamples },
        bert: { rest: bertExamples },
    };

    return <div className='absolute top-0 left-0 flex flex-col'>
        {(Object.keys(rows) as Family[]).map(kind => {
            let { leading, rest } = rows[kind];
            return <div key={kind} className={clsx('flex flex-row flex-wrap items-center', kind === 'transformer' ? 'mt-2 ml-2' : 'ml-2')}>
                {leading && makeButton(leading)}
                {makeMainButton(kind)}
                {rest.map(makeButton)}
            </div>;
        })}
        <div className='ml-2 flex flex-row'>
            <div className={clsx('m-2 p-2 bg-white min-w-[2rem] flex justify-center rounded shadow cursor-pointer hover:bg-blue-300')} onClick={onExpandClick}>
                <FontAwesomeIcon icon={faExpand} />
            </div>
            <div className={clsx('m-2 p-2 bg-white min-w-[2rem] flex justify-center rounded shadow cursor-pointer hover:bg-blue-300')} onClick={onMagnifyClick}>
                <FontAwesomeIcon icon={faMagnifyingGlass} />
            </div>
        </div>
    </div>;

};
