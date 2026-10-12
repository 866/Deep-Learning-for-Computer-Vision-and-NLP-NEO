# Design: BERT visualization in `attention-vis`

Date: 2026-10-12
Status: approved (spec reviewed 2026-10-12); implemented by docs/superpowers/plans/2026-10-12-bert-visualization.md
Scope: add BERT (Devlin et al., 2018, arXiv:1810.04805) as the third model family in the
interactive 3D walkthrough, alongside the encoder-decoder Transformer and the ViT.

---

## 1. Purpose and success criteria

The repo already visualizes two papers, on the rule stated in its README: *"the forward pass runs in
the browser, so every value shown is real."* A third family (BERT) should be added on the same rule,
reusing the existing renderer, layout builder, forward-pass helpers and walkthrough engine rather
than duplicating them.

Success means:

1. A tiny BERT that is **actually trained** on real data, whose forward pass runs in JS and produces
   the same numbers as PyTorch (`npm run validate`).
2. Chapters that follow the paper's own argument and figures: Figure 2 (input representations),
   Figure 3 (bidirectional vs left-to-right vs ELMo), §3.1 (MLM with the 15% / 80-10-10 procedure,
   NSP), §3.2 + Figure 4 (fine-tuning only swaps the output layer).
3. Every number shown to the reader is measured, including the ones that are unflattering.
4. No regression in the two existing families; `npm run typecheck` and `npm run validate` stay green.

### Non-goals (explicitly excluded)

- No 110M/340M weights in the browser. BERT-BASE and BERT-LARGE appear as **shapes without weights**
  in the toolbar, exactly like Transformer base/big and ViT-Base/Large/Huge.
- No fine-tuning in the browser, no SQuAD span head, no NER head, no GLUE leaderboard tables.
- No sentence-pair *input* UI (the interactive demo is single-sentence, §4b's degenerate text-∅ pair).
- No changes to the WebGL renderer, shaders, or the Transformer/ViT chapters.

---

## 2. Measured evidence (this is why the design is shaped this way)

All numbers below were measured on this machine before the design was approved
(probe first, then reproduced by `training/train_bert.py` — `torch 2.14.1+cu130`, CUDA, 142 s end to
end. The values below are the **shipped** run's, read from `config.metrics` in the exported file):

| quantity | measured | how |
| --- | --- | --- |
| parameters | **512,348** | L=2, H=128, A=2, dFF=512, T=32, V=600 |
| sentiment, SST-2 dev | **74.3%** (probe 74.4%) | 4 fine-tune epochs after pre-training, all 872 rows |
| sentiment, sentences that fit whole | **83.2%** | the 267/872 rows under MAXP; the rest are seen truncated |
| majority-class baseline | **50.9%** | label balance of the 872 dev rows |
| MLM masked top-1 (all masks) | **37.8%** (probe 38.2%) | 20 000 pre-train steps, dev, 3 mask seeds |
| MLM masked top-1 (frequent whole words) | **28.9%** (probe 30.9%) | n=5 216 masks on non-`##` tokens with corpus freq ≥ 100 |
| NSP held-out accuracy | **50.0% — at chance** (loss 0.693 = ln 2) | 436 adjacent vs 436 random-partner pairs |
| dev sentences truncated | **69.4%** (605/872) | median dev review is 39 WordPiece pieces, MAXP = T−3 = 29 |
| WordPiece V=600, dev sentences with zero `[UNK]` | **99.7%** | 869/872 |
| word-level (non-subword) vocab instead | 8–10 usable dev rows | rejected: no honest accuracy possible |

Two consequences that the design must respect:

- **MLM is weak on content words and good on function words** (`of` → 75%, `to` → 62%, `sp` → 96%,
  but `##c` → `##t`). The chapter teaches bidirectionality *through* that pattern rather than
  claiming a strong MLM score.
- **NSP learns nothing here.** This is reported as-is, with the note that RoBERTa (Liu et al., 2019)
  later found NSP contributes little. Keeping it is a fidelity choice (Figure 1 shows it), not a claim
  that it works.

Data-source availability, verified by request: `https://dl.fbaipublicfiles.com/glue/data/SST-2.zip`
→ 200, 7 439 277 bytes, 67 349 train / 872 dev / 1 821 unlabeled test rows.

---

## 3. Model definition (BERT-TINY dims, measured configuration)

"Smallest BERT version" = the smallest named compact BERT, BERT-TINY (Turbo et al. 2019,
"Well-Read Students Learn Better"; HF `prajjwal1/bert-tiny`, `google/bert_uncased_L-2_H-128_A-2`):

| | value | note |
| --- | --- | --- |
| L (layers) | 2 | BERT-TINY |
| H (hidden size) | 128 | BERT-TINY |
| A (heads) | 2 | BERT-TINY (H/64) |
| dFF | 512 | = 4H, as the paper's footnote 3 |
| T (sequence) | 32 | **30.6%** of dev sentences fit whole (measured; the 77% an early estimate inferred from *word* counts was wrong — a median review is 39 *pieces*). Truncation is measured, printed, and the picker/examples are chosen from sentences that fit |
| vocab | WordPiece, V=600 | paper §"Input/Output": WordPiece, 30 000 tokens → ours is 600 |
| specials | `[PAD] [UNK] [CLS] [SEP] [MASK]` (ids 0–4) | |
| norm placement | **post-LN** `Add & Norm` | as the paper's encoder (tensor2tensor), unlike ViT's pre-LN |
| attention | **bidirectional**, Q/K/V/O **with** biases | paper: BERT uses bidirectional self-attention |
| activation | ReLU | the original encoder's FFN |
| embedding LayerNorm | yes, after the token+segment+position sum | as in the original implementation |

Heads (fixed to match the measured probe exactly):

- **MLM head**: `Linear(H,H) + bias` → GELU → `LayerNorm` → decoded with the **transposed shared token
  embedding matrix** + a `V`-wide bias. The shared-matrix trick already exists in
  `genGptModelLayout` (`lmHeadWeight` reads `model.embedWeight`).
- **NSP head**: pooler `Linear(H,H)` + `tanh` on the `[CLS]` vector → `Linear(H,2)`.
- **Sentiment head**: `Linear(H,2)` applied **directly** to the final `[CLS]` vector `C`, i.e.
  `log(softmax(C Wᵀ))` — literally paper §4.1, which introduces only `W ∈ ℝ^{K×H}` and no pooler.

### 3.1 Buffer conventions (unchanged from `TransformerModel.ts`)

```
activations        (B*T) x C          T=32, C=128, B=1
linear weight      nOut x nIn         PyTorch layout
attention matrix   (B*nHeads*T) x T   2 heads -> 64 x 32
softmax agg        (rows) x 1 x 2     [1/sum, max]
layer-norm agg     (B*T) x 1 x 2      [mean, 1/std]
token ids          (T) x 1            also segment ids (T) x 1
MLM logits/probs   (T) x V            32 x 600
```

### 3.2 Public interface (`src/llm/BertModel.ts`)

```ts
export interface IBertShape  { B, T, C, nHeads, A, nBlocks, dFF, vocabSize, nSegments }
export interface IBertConfig { vocab: string[]; vocab_size; block_size; n_embd; n_head;
                               n_layer; d_ff; n_params; special: {pad,unk,cls,sep,mask};
                               metrics: IBertMetrics }
export interface IBertMetrics { sentiment_dev_acc; majority_baseline; mlm_top1_all;
                                mlm_top1_frequent; nsp_val_acc; truncation_rate; n_dev }
export interface IBertModelLink {
    gl; shape; config; vocab: string[];
    embedWeight;                     // V x C, shared with the MLM decoder
    segEmbed;                        // 2 x C
    posEmbed;                        // T x C
    inputBuf: Float32Array;          // token ids of the current sentence
    segBuf: Float32Array;            // segment ids
    tokens; segIds;                  // display buffers (T x 1)
    embedLN: ILayerNormLayerLink;    // LayerNorm after the embedding sum
    embed;                           // T x C  = z0
    blocks: IBertBlockLink[];        // { selfAttn, ln1, ffn, ln2, output } — same shape as IEncoderBlockLink
    mlm: { fc: ILinearLayerLink; act; ln: ILayerNormLayerLink };
    mlmLogits; mlmAgg; mlmProbs;     // (T x V), (T x 1 x 2), (T x V)
    pool: ILinearLayerLink;          // H -> H, tanh
    nspLogits;                       // 1 x 2
    clsWeight; clsBias;              // H -> 2
    sentimentLogits; sentimentProbs; // 1 x 2
    examples: { text: string; label: number; ids: number[] }[];
    masked: number[];                // positions currently masked in inputBuf
}
export function bertConfigToShape(config: IBertConfig): IBertShape
export async function fetchBertWeights(url: string): Promise<IBertWeights>   // reuses fetchTransformerWeights' body
export function createBertModel(gl, weights, exampleIdx = 0): IBertModelLink
export function setBertSentence(model: IBertModelLink, text: string, maskPositions?: number[]): void
export function runBertModel(model: IBertModelLink): void
export function uploadAllBert(model: IBertModelLink): void
export function bertTopK(model: IBertModelLink, pos: number, k: number): { tok: string; p: number }[]
export function wordpieceEncode(vocab: string[], text: string, opts): { ids: number[]; toks: string[]; truncated: boolean }
```

`runBertModel` composes the **existing** helpers — `matmulWT`, `linearForward`, `layerNormForward`,
`attentionForward(model, layer, x, x, /*causal=*/false)`, `add` — exactly as `runTransformerModel`'s
encoder loop does, minus the decoder. Only `wordpieceEncode` and the two heads are new math.

`fetchBertWeights` does not duplicate `fetchTransformerWeights`: that function's body moves to a shared
`fetchModelWeights<T>(url): Promise<T>` (it is already generic apart from its name and return type) and
the Transformer/ViT/BERT loaders each become a one-line call to it.

### 3.3 JS WordPiece tokenizer

~30 lines, greedy longest match, the standard WordPiece decode strategy: split on whitespace; for
each token take the longest vocab prefix, then longest continuations prefixed `##`; if no piece
matches, emit `[UNK]`. Truncation keeps `[CLS]`, as many pieces as fit, and the trailing `[SEP]`.

This must agree with Python `tokenizers` WordPiece on the exported example ids — asserted in
`npm run validate` (a token-id mismatch fails the run, not just a float diff).

### 3.4 Files

New: `src/llm/BertModel.ts`, `src/llm/BertModelLayout.ts`, `src/llm/walkthrough/WalkthroughBert.tsx`,
`training/train_bert.py`, `public/bert-sst2-model.json`, `public/bert-sst2-partials.json`,
`public/bert-figs/*`, `LICENSE-BERT-figures.md`.

Touched: `GptModelLayout.ts` (§4.1 generalization + `fetchModelWeights`), `Program.ts` (§4.2),
`walkthrough/Walkthrough.ts` (phases, `PhaseGroup.Bert`, `familyOfPhase`),
`walkthrough/WalkthroughTools.ts` (`IBertWalkthroughArgs`; `DimStyle.n_segments` + its
text/colour entries), `LayerView.tsx` (third weights fetch, `createBertModel`),
`components/ModelCard.ts` (`drawBertModelCard`), `components/Arrow.ts`
(`drawBertArrows`, new export only — §4.3), `components/SectionLabels.ts` (`drawBertBlockLabels`),
`components/TocDiagram.tsx`, `components/ModelSelectorToolbar.tsx`, `Interaction.ts`,
`scripts/validate-model.ts`, `App.tsx` (header link to arXiv:1810.04805), `README.md`.

---

## 4. Layout (`src/llm/BertModelLayout.ts`)

Single stack at the origin (no decoder half), `s.dir = -1`, side computations to the left — the same
frame the encoder already uses.

```
[CLS]  my  dog  is  cute  …                      <- token row (T x 1) + segment row (T x 1)
  E_[CLS] E_my …            E_A E_B …           E_0 E_1 …    <- three embedding tables
                       z0 = token + segment + position
                       LayerNorm(z0)
  L x {  self-attn (full, biased)  ->  Add & Norm  ->  FFN  ->  Add & Norm  }
  MLM head on the masked columns -> logits over V -> softmax aggregates -> probabilities
  [CLS] column -> pooler -> NSP ;  [CLS] column -> Linear(H,2) -> sentiment
```

Reuse, unchanged: `createLayoutBuilder(shape)` with `createLn(x, target, /*preNorm=*/false)`,
`createAttention(xq, xkv, target, AttnKind.Self, { bias: true })`, `createFfn(src, target,
{ prefix: 'FFN' })`, plus `mk`, `mkLabel`, `sc`, `leftX`/`rightX`/`attnLeftX`/`qkvValLeftX`, and the
`BlKDepSpecial.LayerNorm / SoftmaxAggMax / SoftmaxAggExp / Softmax / Attention` cell-dependency
descriptions. The encoder block is therefore the *same code* as `genGptModelLayout`'s encoder blocks.

New in this file: the three-table embedding row, the segment-id block, the MLM head, and the two
`[CLS]` heads — all built with the same `mk({...})` idiom, and the returned object carries
`kind: 'bert' as const` so the dispatch table below can pick it up.

### 4.1 One generalization in `GptModelLayout.ts`

`inputEmbedParts()` currently assumes exactly `add[0]=tokEmbed, add[1]=posEmbed, add[2]=idxObj` (it guards on
`add.length < 3`). BERT adds a fourth part (segment embeddings). Change it to return a generic list:

```ts
export function inputEmbedParts(blk: IBlkDef):
    { embed: IBlkDef; idx: IBlkDef; tables: { src: IBlkDef; what: 'token' | 'segment' | 'position' }[] } | null
```

Verified call sites (found by grep — all four must be updated):

| site | uses |
| --- | --- |
| `src/llm/GptModelLayout.ts:129` | the definition itself |
| `src/llm/components/DataFlow.ts:226` (`drawOLIndexLookup`) | `parts.idxObj`, `parts.tokEmbed` |
| `src/llm/components/DataFlow.ts:270` (`drawOLPosEmbedLookup`) | `parts.posEmbed` |
| `src/llm/components/DataFlow.ts:815` (`drawDepArrows`) | embed parts for dependency arrows |
| `src/llm/Interaction.ts:204` | hover hit-testing on the lookup column |

Behavior for the Transformer stays identical; the mapping is a pure regrouping of the same three
sources (`token` replaces `tokEmbed`, `position` replaces `posEmbed`, `idx` is unchanged).

`BlKDepSpecial` gains `MlmDecode` (shared-matrix decode) and `SegmentEmbed` is *not* needed — the sum
is a plain `add` of four parts. `DimStyle` gains `n_segments` (green, echoing the paper's Figure 2
color) with `dimStyleText`/`dimStyleTextShort` entries.

### 4.2 Family dispatch (targeted improvement)

Verified current state of the per-family branching, by grep:

| file | today |
| --- | --- |
| `Program.ts` | `if (layout.kind === 'vit')` in `drawLayoutArrows`, the block-label call, and the model-card call (twice: main + examples), plus `genLayoutFor` and `currentMainExample` |
| `Walkthrough.ts:170` | `isVitPhase = phase >= Phase.Vit_Intro` — an **ordinal range test** |
| `components/TocDiagram.tsx:80` | `vitMode = isVitPhase(...)` picks which TOC band to draw |
| `components/ModelSelectorToolbar.tsx:23` | `if (isVitPhase(wt.phase) !== (kind === 'vit'))` — "fly to the family currently at the origin" |

Replace the `kind === 'vit'` chain with one table (types elided for brevity):

```ts
type Family = 'transformer' | 'vit' | 'bert';
const FAMILIES: Record<Family, {
    genLayout(shape: any, model?: any): IAnyModelLayout;
    drawArrows(render: IRenderState, layout: any): void;
    drawLabels(render: IRenderState, layout: any): void;
    drawCard(state: IProgramState, layout: any, title: string, off: Vec3, isMain: boolean): void;
}>;
```

`MainKind` becomes `Family`, and the four predicate sites above collapse into one `familyOfPhase(phase)`.
`IProgramState` gains `bert: { shape; model; layout; pendingSentence }` mirroring the existing
`vit: IVitState`, plus `bertMainExample`.

**Bug that must not survive:** with `Bert_*` phases appended after `Vit_Attention`, `isVitPhase()`
returns `true` for every BERT phase, so the ViT layout would be drawn under the BERT chapters.
Deriving `Family` from explicit phase sets fixes this by construction; covered by the regression check
in §10. New enum values are appended last so previously saved `SavedState.phase` values stay valid.

### 4.3 Arrow drawing, and one deliberate cut

`Arrow.ts` already factors the shared part of arrow drawing into `arrowDrawer(state, layout)` (yielding
`drawAttention`, `drawLn`, `drawFfn`, `drawVertArrow`, `drawInward`, `drawArrowBetween`), used by both
`drawAllArrows` and `drawVitArrows`. BERT needs a `drawBertArrows` written against that same helper
(~40 lines, mirroring `drawVitArrows`: embedding row → z₀ → per-block LN/attn/FFN → MLM head → the two
`[CLS]` heads). No edit to `Arrow.ts` itself.

**Deliberate cut:** `DataFlow.ts` (the per-cell "how is this value computed" overlay) branches on
`blk.deps.special`, with family-specific branches such as `InputEmbed` → `drawOLInputEmbed` and ViT's
`PatchEmbed` → `drawClassToken`. Anything unrecognised falls through to the generic `drawDepArrows`
path, which already renders correct add/dot dependency arrows from `blk.deps`. BERT's four-way embedding
sum and its MLM decode therefore render correctly through the **existing generic path**, with no new
overlay code. A dedicated `drawOLMlmDecode` overlay (a `[MASK] → vocab` animation to equal
`drawOLInputEmbed`) is the natural follow-up polish and is out of scope here.

---

## 5. Chapters (`src/llm/walkthrough/WalkthroughBert.tsx`)

New `PhaseGroup.Bert` ("BERT") with 6 phases, written with the same `commentary` / `breakAfter` /
`afterTime` / `processUpTo` / `moveCameraToFor` idiom as `WalkthroughVit.tsx`. `WalkthroughTools.ts`
gains `IBertWalkthroughArgs`, mirroring the existing `IVitWalkthroughArgs` (which is
`Omit<IWalkthroughArgs, 'layout'>` with `layout` narrowed) — a third narrow interface in the established
style, not a generic refactor of the existing two.

| # | phase id | content, and the paper material it uses |
| --- | --- | --- |
| 1 | `Bert_Intro` | Figure 1 (shipped image). Pre-train → fine-tune; "only the output layer changes". Model card: 512,348 params, 74.4% dev vs 50.9% baseline. Quotes the abstract's "conceptually simple and empirically powerful". |
| 2 | `Bert_Input` | Figure 2 (shipped image) **plus a live strip** built from the real `wordpieceEncode` of the current sentence, so the reader sees `char ##m ##ing` the way the figure shows `play ##ing`. Then the hoverable token/segment/position sum → z₀ → LayerNorm. `[CLS]`, `[SEP]`, `[MASK]` roles. |
| 3 | `Bert_Attn` | Figure 3 (shipped image). BERT's full attention matrix vs the *same* Q/K values with a causal mask applied (drawn from the running model's real `attnMatrixSoftmax`, captioned "if it were masked, as GPT does") vs a note on ELMo's shallow bidirectionality. Reuses `lowerTri` for the masked rendering. |
| 4 | `Bert_Mlm` | §3.1 Task #1 verbatim logic: 15% of positions, then 80/10/10 (mask/random/unchanged). Live top-k for a chosen `[MASK]` via `bertTopK`. Honest numbers: 38.2% overall, 30.9% on frequent whole words; the function-word/content-word split explained. |
| 5 | `Bert_PretrainFT` | §3.1 Task #2 (NSP) with the pooler head, **reporting the chance result** and the RoBERTa follow-up. Then §3.2: fine-tuning swaps only the head; Figure 4 (shipped image) for the task zoo; the degenerate text-∅ pair that our single-sentence demo is. |
| 6 | `Bert_Sentiment` | §4.1: `log(softmax(C Wᵀ))`, K=2, `W ∈ ℝ^{2×128}` shown as real blocks. Interactive `SentencePicker` (pattern of `SourcePicker`/`VitImagePicker`) re-running the whole model per edit; 74.4% on 872 real dev rows; truncation rate; BERT-BASE/LARGE toolbar note; L/H/A scaling table like `VitSizesTable`. |

`TocDiagram.tsx` gains a BERT band so the table of contents shows three families; `App.tsx` gains a
header link to `https://arxiv.org/abs/1810.04805`.

---

## 6. Figures and attribution

- Source: the four vector figures from the arXiv e-print (`BERT_Overall.pdf` = Fig 1,
  `Input_Emebeddings.pdf` = Fig 2, `BERT_comparisons.pdf` = Fig 3, `BERT_fine_tune.pdf` = Fig 4),
  already extracted and visually verified.
- Conversion: ghostscript `-dNoOutputFonts` then `pdftocairo -svg` into `public/bert-figs/`. The extra
  ghostscript pass is not optional decoration: `pdftocairo -svg` alone writes text as SVG-1.1 glyph
  references with no embedded font, which Firefox drops entirely (measured on Fig 2: 195 glyph
  references, 158 paths — after outlining, 0 and 295). Outlining keeps the figures vector-crisp at
  any camera zoom with no font dependency. `pdftoppm -png -r 300` remains as a per-file fallback, and
  `npm run check:bert` asserts the shipped SVGs contain no glyph references, so a regression fails
  loudly rather than shipping figures with missing labels.
- Display: `<img>` inside the React commentary widgets (the `embed()` mechanism), captioned
  "Figure N from Devlin et al., 2018, arXiv:1810.04805" with an `embedLink` to the paper.
- `LICENSE-BERT-figures.md` next to `LICENSE-llm-viz`, recording the source, the four files, and that
  they are used with attribution for commentary/education. Checked, not assumed: the published NAACL
  2019 PDF footers say "© 2019 Association for Computational Linguistics" with no open-content notice,
  and the arXiv e-print we convert from is under arXiv's non-exclusive *distribution* licence rather
  than a CC option. There is therefore no open-content grant to claim, so the file states the use and
  the evidence instead of asserting a licence — an attribution note, deliberately not a licence grant.

---

## 7. Training (`training/train_bert.py`)

Mirrors `train_vit.py`'s shape (stdlib + numpy + torch + tokenizers; no HuggingFace `transformers`).

1. **Fetch**: download SST-2 to `<repo>/data/SST-2/` and reuse the cache if present. Fail with a clear
   message if offline.
2. **Tokenizer**: train WordPiece (V=600, min_freq 2, specials as above) on the train split only.
3. **Pre-train**: MLM (15% + 80/10/10) + NSP from adjacent corpus sentences; 20 000 steps, batch 128,
   AdamW lr 3e-4 — the measured configuration. Fixed seed (`torch.manual_seed`, `numpy`, `random`).
4. **Fine-tune**: sentiment head, 4 epochs, batch 128, lr 3e-4.
5. **Evaluate**: sentiment dev accuracy + majority baseline; MLM top-1 overall and on frequent whole
   words; NSP dev accuracy; dev truncation rate.
6. **Export** `public/bert-sst2-model.json`: all weight tensors via the existing `tensor_to_json`
   convention, `config` (vocab, dims, special ids, `n_params`, all `metrics`), and 8 example sentences
   with labels/ids for the picker. Export `public/bert-sst2-partials.json`: one fixed example's
   `ids`, `seg`, `embedLN`, `embed`, per-layer `q`/`attn`/`attnSm`/`resid`/`ln1`/`ffnAct`/`ln2`,
   `mlmLogits`, `mlmProbs`, `sentimentProbs`.

Cost: measured **~2 min on CUDA** (249 steps/s, consistent with the 145 s end-to-end run) and
**~44 min CPU-only** (10.8 steps/s at 16 threads: ~31 min pre-train + ~13 min fine-tune).
Guardrails: the trained JSON is committed, so *viewing the page never requires training*; `--steps` /
`--epochs` flags exist; and a documented `--fast` profile (4 000 steps + 2 epochs, ~9 min on CPU) is
for iteration only — the shipped metrics must come from the full profile, or §2's table must be
rewritten from whatever actually ran. Device is chosen automatically (CUDA if present).

`scripts/validate-model.ts` gains a BERT section using the existing `check()`/`headsToCols()` harness
(no GL context needed; the fake-`gl` Proxy trick already works): assert JS token ids == Python ids,
then per-tensor `max |diff| < 1e-3`, keeping the single `worst` gate that already exits non-zero.

---

## 8. Resource budget (measured on this laptop, 2026-10-12)

The requirement from the human partner: **do not overload the laptop**. So this section is measured,
not estimated. Probe method is given with each table.

### 8.1 Per-frame cost (this is what decides whether the tab is smooth)

`runProgram` regenerates the **main** layout every frame, and `blockRender` refills its UBOs from
`layout.cubes`, so cube count *is* the per-frame cost. Optional example layouts are built once in
`setExampleEnabled`; only their arrows/labels/cards are drawn per frame.

| layout | cubes | layout gen | status |
| --- | ---: | ---: | --- |
| tiny Transformer | 313 | 4.4 ms | always on today |
| tiny ViT | 146 | 2.5 ms | always on today |
| **tiny BERT (proposed)** | **129** | **0.5 ms** | always on |
| Transformer (base) | 1 903 | 7.7 ms | optional (today) |
| BERT-BASE | 2 309 | 7.3 ms | optional |
| ViT-Huge/14 | 7 890 | 23.7 ms | optional (today) |
| BERT-LARGE | 5 933 | 45.4 ms | optional |

Measured by driving the repo's own `createLayoutBuilder` / `genGptModelLayout` / `genVitModelLayout` in
node (BERT counted through the real builder, mirroring the block sequence §4 describes).
**Conclusion: BERT's always-on layout is the cheapest of the three** — fewer blocks than the tiny ViT,
~41% of the tiny Transformer's — so steady-state frame cost *drops* rather than rises. BERT-LARGE sits
between two shapes that already ship; its 45 ms is one-off on toggle-on, and toggling off frees its
layout and block-render buffers through the existing path.

### 8.2 Per user action (the one place BERT costs more than what exists)

| action | measured |
| --- | ---: |
| tiny ViT forward (runs on *every pixel* you draw today) | 1.23 ms |
| tiny Transformer full re-decode (6 sequential passes) | 4.15 ms |
| **tiny BERT forward (one pass)** | **10.6 ms** |

plus a 2.73 MiB texture re-upload. The 10.6 ms was measured by driving the repo's real `matmulWT` over
BERT-TINY's exact shapes: Q/K/V/O projections, bidirectional scores + softmax + weighted V, both FFN
layers, the MLM decode through the shared embedding matrix, and the softmax over V=600 for all 32
positions.

Guardrails baked into the design:

- `setBertSentence` runs only from `state.bert.pendingSentence` inside `runProgram` — the ViT's existing
  pattern — and the picker sets that on **commit/debounce, not per keystroke**. Worst case is one
  ~10–30 ms hitch per edit (browsers trail node on low-power cores), never per frame.
- The forward pass and GPU upload are never inside the render loop.
- No new timers, autoplay or continuous animation: the walkthrough advances only on spacebar, and
  `LayerView.loop` already returns without rendering when nothing is dirty, so an idle tab stays ~0.
- BERT-BASE / BERT-LARGE are **off by default**, matching the existing paper-sized shapes; enabling one
  is an explicit click and the toolbar's eye icon frees it again.

### 8.3 Memory

- Tiny BERT texture data: **2.73 MiB**; largest single buffer is `embedWeight` (600×128 float32 ≈ 300 KB).
  Optional shapes allocate **no** weight textures (`model = null` ⇒ no `access.src`).
- `blockRender` starts at `numBlocks = 1024` and grows through `ensureFloatBufferSize(…, cubes.length)`;
  only the optional paper-sized layouts exceed that, as they already do today.
- No new persistent JS structures beyond one `IBertModelLink` and the 600-entry vocab array.

### 8.4 Page payload — the one honest regression

Weights serialize as base64 float32 (`training/train_vit.py:tensor_to_json` → `TensorF32.fromJson`), so
512 348 params ≈ **2.6 MiB of JSON**, and float32 gzips poorly (~2 MiB on the wire). Today:
`vit-digits-model.json` 328 KB, `transformer-reverse-model.json` 692 KB, whole `dist/` 2.0 MB. So first
load transfers roughly 2 MiB more.

Invisible on `npm run dev` (localhost); it only matters if the site is deployed. Two levers, decided at
implementation time — default (a), fidelity:

- **(a)** ship it: ~2.6 MiB JSON, keeps BERT-TINY's published dims and the measured 74.4% accuracy;
- **(b)** shrink to H=64 (~150 k params, ~0.8 MB) — no longer BERT-TINY's configuration, and its
  accuracy would be **unmeasured**, so choosing (b) requires re-running the training probe and
  rewriting §2 before any chapter text quotes a number.

---

## 9. Error handling

- Missing weight tensor → the existing `throw new Error('missing tensor ${name}')` in the `weightTex`
  idiom; unchanged.
- Sentence longer than `T-3` subwords → truncate, keep `[CLS]`/`[SEP]`, show a truncation note in the
  picker and the aggregate rate in the model card (never silent).
- Unrecognized characters → `[UNK]`, rendered grey with the token's surface form in the live strip.
- Empty sentence → the model still runs on `[CLS] [SEP]`; picker disables "Run" rather than special-casing the model.
- Weights JSON absent (e.g. `npm run dev` before training) → BERT phases show the layout without data,
  like the existing "no model yet" path; `LayerView` already tolerates a null model.
- Saved state pointing at a BERT phase from a build without weights → same degraded-but-usable path,
  no crash.

---

## 10. Verification plan (evidence before claims)

1. `python training/train_bert.py` → prints and embeds the measured metrics; compare the printed
   sentiment dev accuracy against the design's 74.4% claim and update this spec if the real run differs.
2. `npm run validate` → JS vs PyTorch, all three families, `OK` with worst diff < 1e-3.
3. `npm run typecheck` → clean.
4. `npm run build` → succeeds; record the added JSON payload size.
5. Manual scrub of all 6 new phases (space/Delete/PageUp-Down), checking: figures load, live strip
   matches the model's token ids, `[MASK]` top-k changes when the sentence changes, camera framing
   never clips the MLM head (widest new block).
6. **Resource re-check against §8 in the real app**, not just the probe — DevTools open, on the laptop:
   (i) an idle tab burns ~0 CPU (dirty-flag loop); (ii) frame time in the BERT chapters is no worse
   than in the Transformer chapters; (iii) one sentence edit costs a single visible hitch (<50 ms);
   (iv) the JS heap stays flat across 50 edits (no per-edit leak);
   (v) `du -h public/bert-sst2-model.json` matches the §8.4 estimate.
7. Regression check on the existing families: Transformer ch. 1 + 8 + 12, ViT ch. 1 + 5, and the
   toolbar toggles for the four paper-sized shapes (Transformer base/big, ViT-Base/Large/Huge) plus
   the new BERT-BASE/LARGE — including a toggle-on → off → on cycle to confirm buffers are freed.
8. Visual smoke test via the existing `window.__canvasRender` DEV hook.

---

## 11. Risks and mitigations

| risk | mitigation |
| --- | --- |
| **viewer's laptop overloaded** | measured (§8): BERT's always-on layout is the *cheapest* of the three (129 cubes / 0.5 ms vs 313 / 4.4 ms); 10.6 ms forward only on an explicit edit; 2.73 MiB VRAM; large shapes off by default; no new per-frame or timed work |
| MLM accuracy too low to demo | measured (38.2%) and designed *around*: chapter teaches the function-word/content-word split; numbers printed in the model card |
| NSP reads as a broken feature | stated as at-chance with the RoBERTa follow-up; it is Figure 1 fidelity, not a capability claim |
| JS/Python tokenizer drift | token ids asserted in `npm run validate` |
| `isVitPhase` ordinal bug breaking the ViT chapters | converted to phase sets in this change; covered by the ViT regression check |
| T=32 truncation distorting claims | truncation rate measured and shown; sentences in the picker chosen to fit |
| 69.4% of dev sentences truncated at T=32 (measured; an early estimate said 23%) | accuracy reported on the full 872 with truncation **and** on the 267 that fit whole (74.3% / 83.2%), stated as such in the chapter |
| payload size (600-token vocab + 512k params) | measured: `bert-sst2-model.json` 2.62 MiB + `bert-sst2-partials.json` 0.63 MiB = **3.25 MiB** of JSON, disclosed in §8.4 with the H=64 fallback recorded |
| figure licensing | attribution file + caption on every use; user-approved decision |
| training needs CUDA to be quick | measured 249 steps/s GPU (~2 min) vs 10.8 steps/s CPU (~44 min); not on the viewing path (JSON committed); `--steps`/`--fast` documented |

---

## 12. Out of scope for the implementation plan

Nothing in §1's non-goals. If, during implementation, the `inputEmbedParts` generalization turns out
to touch more call sites than expected, or the dispatch table requires renderer changes, that is a
scope increase: stop and re-open this spec rather than absorbing it silently.
