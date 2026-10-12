import { ICameraPos } from "../Camera";

// (keyboard/button navigation through the model is disabled, as in llm-viz; only the types remain)

export interface ICameraLerp {
    camInitial: ICameraPos;
    camFinal: ICameraPos;
    duration: number;
    t: number;
}

export interface IMovementInfo {
    action: MovementAction | null;
    actionHover: MovementAction | null;

    depth: number;
    target: number[]; // index maps to depth, which maps to the INavLevel tree

    cameraLerp: ICameraLerp | null;
}

export enum MovementAction {
    Up,
    Down,
    Left,
    Right,
    Focus,
    In,
    Out,
    Expand,
}
