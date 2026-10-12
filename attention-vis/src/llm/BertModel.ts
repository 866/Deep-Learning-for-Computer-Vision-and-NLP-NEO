// BERT in the browser: tokenizer first (this file), forward pass in the task that follows.
// The tokenizer has to agree with Python's `tokenizers` crate bit for bit: the visualization
// prints token ids next to the cubes they index into, and a drift of one id would silently
// mislabel the whole sentence. npm run check:bert compares every fixture here against the ids
// training/train_bert.py exported, so "agrees" is a tested claim rather than an intention.

export interface IBertVocab {
    vocab: string[];       // id -> token, exactly as config.vocab
    pad: number;
    unk: number;
    cls: number;
    sep: number;
    mask: number;
}

// The vocab array and the five special ids travel together: a tokenizer handed one without the
// other would print ids against the wrong rows, and every caller needs both anyway. This is
// spec §3.2's wordpieceEncode with its two loose arguments fused into one object.
export function bertVocabFromConfig(config: { vocab: string[]; special: Record<string, number> }): IBertVocab {
    return {
        vocab: config.vocab,
        pad: config.special.pad,
        unk: config.special.unk,
        cls: config.special.cls,
        sep: config.special.sep,
        mask: config.special.mask,
    };
}

// token -> id, plus the set of characters the trained pieces can actually spell. Both are built
// once per vocab object: a WeakMap rather than a field keeps the interface free of cache noise and
// lets a replaced vocab be collected with its index.
interface IVocabIndex {
    ids: Map<string, number>;
    chars: Set<string>;
}

const indexCache = new WeakMap<IBertVocab, IVocabIndex>();

function vocabIndex(v: IBertVocab): IVocabIndex {
    let cached = indexCache.get(v);
    if (!cached) {
        const ids = new Map<string, number>();
        const chars = new Set<string>();
        for (let i = 0; i < v.vocab.length; i++) {
            const tok = v.vocab[i];
            ids.set(tok, i);
            // Special tokens ([PAD] etc.) don't contribute spellable characters: a WordPiece model
            // knows the letters of its own pieces, and the specials were added around it. (In this
            // vocabulary both choices encode all 68 221 SST-2 sentences identically, so nothing
            // observable depends on the call — it's made for the case where something does.)
            if (tok === '[PAD]' || tok === '[UNK]' || tok === '[CLS]' || tok === '[SEP]' || tok === '[MASK]') continue;
            for (const c of tok.startsWith('##') ? tok.slice(2) : tok) chars.add(c);
        }
        cached = { ids, chars };
        indexCache.set(v, cached);
    }
    return cached;
}

// The crate's Whitespace pre-tokenizer splits on Unicode White_Space, not on ASCII only.
const WHITESPACE = /[\p{White_Space}]+/u;

// WordPiece's guard: a single word longer than this is unrecoverable, so the crate replaces the
// whole word with one [UNK] rather than attempting it.
const MAX_CHARS_PER_WORD = 100;

// The "##" prefix is suppressed when a piece crosses a punctuation boundary in *either* direction,
// and required when it stays on the same side of one — verified against the crate on all four
// shapes:  n! -> n + !      but  x!!! -> x + ! + ##! + ##!
//                                        !!n -> ! + ##! + n
//                              it's -> it + ' + s   (bare `s`, not ##s, which is also in the vocab)
// A plain OR gets 49 SST-2 sentences wrong (the censored-word "***" runs); XOR gets 0 wrong.
const FILTER_CHAR = /[\p{White_Space}\p{P}\p{S}]/u;
function isFilterChar(c: string): boolean {
    return FILTER_CHAR.test(c);
}
function isAllFilter(s: string[]): boolean {
    return s.length > 0 && s.every(isFilterChar);
}

/**
 * Encode one sentence as [CLS] pieces… [SEP], padded to `maxLen`.
 *
 * `maxLen` is the sequence length T. Content is capped at T-3 pieces, not T-2: a sentence *pair*
 * needs a second [SEP] at the end, and a single sentence reserves the same room so that its ids are
 * byte-identical to what training produced (69% of SST-2 dev sentences are affected by this cap —
 * see the truncation metric). Callers pass config.block_size.
 */
export function wordpieceEncode(v: IBertVocab, text: string, maxLen: number):
    { ids: number[]; toks: string[]; truncated: boolean } {
    const index = vocabIndex(v);
    const maxPieces = Math.max(0, maxLen - 3);    const pieces: number[] = [];
    let truncated = false;

    const push = (id: number) => {
        if (pieces.length < maxPieces) {
            pieces.push(id);
        } else {
            truncated = true;
        }
    };

    for (const word of text.split(WHITESPACE)) {
        if (word === '') continue;
        const chars = Array.from(word);        // code points, so a non-BMP glyph is one char
        // A word containing a character no piece can spell (an accented vowel outside this 600-token
        // vocab) or exceeding the length guard becomes ONE [UNK] and is skipped whole — maelström is
        // [UNK], not m + ##a + ##el + … + [UNK]. Getting this wrong shifts every id after it.
        if (chars.length > MAX_CHARS_PER_WORD || chars.some(c => !index.chars.has(c))) {
            push(v.unk);
            continue;
        }
        let start = 0;
        let prevAllFilter = false;
        while (start < chars.length) {
            const isFirst = start === 0 || isFilterChar(chars[start]) !== prevAllFilter;
            let found = -1;
            let foundEnd = start;
            for (let end = chars.length; end > start; end--) {       // greedy: longest piece first
                const piece = (isFirst ? '' : '##') + chars.slice(start, end).join('');
                const id = index.ids.get(piece);
                if (id !== undefined) {
                    found = id;
                    foundEnd = end;
                    break;
                }
            }
            if (found < 0) {
                // No piece covers what is left of this word. The crate emits a single [UNK] and
                // abandons the remainder of the word rather than retrying shorter prefixes.
                push(v.unk);
                break;
            }
            push(found);
            prevAllFilter = isAllFilter(chars.slice(start, foundEnd));
            start = foundEnd;
        }
    }

    const ids = [v.cls, ...pieces, v.sep];
    while (ids.length < maxLen) ids.push(v.pad);
    return { ids, toks: ids.map(id => v.vocab[id]), truncated };
}
