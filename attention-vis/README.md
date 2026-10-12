# Transformer Visualization

An interactive 3D walkthrough of the encoder-decoder Transformer from
["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017), in the style of
Brendan Bycroft's [LLM Visualization](https://bbycroft.net/llm). Its 3D renderer, interaction, walkthrough
engine and much of its text are reused from [bbycroft/llm-viz](https://github.com/bbycroft/llm-viz) (MIT, see
`LICENSE-llm-viz`).

The model is a tiny Transformer (N = 2, d_model = 48, h = 3, d_ff = 192, 131,136 parameters) that reverses a
sequence of six letters from {A, B, C}. It follows the paper: sinusoidal positional encodings, embeddings scaled by
√d_model and shared with the output layer, post-norm "Add & Norm", ReLU feed-forward networks, masked decoder
self-attention and encoder-decoder (cross) attention. The forward pass runs in the browser, so every value shown is real.
The paper's base and big configurations can be shown at full size (without weights) next to it. They are off by default; click a model's name in the top-left toolbar to turn it on and fly to it, and use its eye icon to turn it off again (this frees its layout and GPU buffers).

## Vision Transformer

A second part covers the Vision Transformer (ViT) from
["An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"](https://arxiv.org/abs/2010.11929)
(Dosovitskiy et al., 2020). Its chapters show a tiny ViT (L = 2, D = 48, 3 heads, MLP size 192, 58,234 parameters) that
classifies the 8x8 handwritten digits of scikit-learn's digits dataset, cut into 16 patches of 2x2 pixels plus the [class]
token (98.6% accuracy on the 360 held-out test images). Like the paper's model it has a linear patch projection, a learnable
[class] token, learned 1D position embeddings, pre-norm encoder layers, GELU MLPs and a classification head on the [class]
token. The chapters include the learned position-embedding similarities (the paper's Figure 7), per-head attention maps and
attention rollout (Figure 6), and an image picker where you can also draw your own digit. ViT-Base/16, ViT-Large/16 and
ViT-Huge/14 (Table 1) can be turned on in the toolbar, like the Transformer's base and big models.

## BERT

A third part covers BERT from
["BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"](https://arxiv.org/abs/1810.04805)
(Devlin, Chang, Lee & Toutanova, 2018). Its chapters show a tiny BERT (L = 2, H = 128, A = 2 heads, FFN size 512,
600-piece WordPiece vocabulary, 512,348 parameters, sequence length T = 32) that was pre-trained with the paper's own
two objectives \u2014 a masked language model and next-sentence prediction \u2014 and then fine-tuned on SST-2 movie reviews,
the paper's Figure 4b setup. It reaches **74.9% on the 872 dev sentences it never saw**, against a 50.9% majority
baseline (80.9% on the 267 dev sentences that fit inside T = 32 without truncation).

Like the paper's model it has summed token + segment + position embeddings, learned position embeddings, pre-norm
encoder layers with biased, GELU feed-forward blocks, a masked-LM head whose output layer is the token embedding table
transposed, a tanh pooler feeding the NSP head, and a K $\u00d7$ H classifier reading the raw [CLS] vector (\u00a74.1). Bidirectional
attention is the real thing: no causal mask is ever set.

The numbers are reported as measured, including the ones that are not flattering. The masked-token objective gets 37.1%
of masks right overall and 28.9% when the masked piece is a frequent whole word; next-sentence prediction sits at 50.6%,
i.e. chance, because a single-sentence task gives the head nothing to learn from (the paper's own Table 3 found NSP worth
0.1 of a point, and RoBERTa later dropped it); and 69.4% of SST-2 dev sentences are longer than T = 32, so they reach the
model truncated \u2014 which is why the untruncated accuracy is quoted next to the headline number. Every figure on screen, and
every number the chapters speak, is read out of the checkpoint's own `config.metrics` rather than typed into the prose;
`npm run check:bert` fails if a chapter hard-codes an accuracy that is not in that file or not a cited table from the paper.

BERT-Base and BERT-Large (\u00a73) can be turned on in the toolbar like the other paper-sized models, for structure only \u2014
they carry no weights, and the tiny BERT is the one that runs.

## Running

Requires Node 18+.

    npm install
    npm run dev        # http://localhost:5173
    npm run build      # static site in dist/

## Retraining the model

    python training/train.py       # needs PyTorch; writes public/transformer-reverse-*.json
    python training/train_vit.py   # needs PyTorch & scikit-learn; writes public/vit-digits-*.json
    ../.venv/bin/python training/train_bert.py   # needs PyTorch & the SST-2 data in data/SST-2/; writes public/bert-sst2-*.json
    npm run validate               # checks both JS forward passes against PyTorch's activations
    npm run check:bert             # checks the BERT assets, weights, tokenizer, forward pass and chapters

`train_bert.py` pre-trains (MLM + NSP) and then fine-tunes the sentiment classifier in one run. It is measured at 249
steps/s on a GPU (~2 min) and 10.8 steps/s on CPU (~44 min); it prefers CUDA when there is one, and falls back to CPU
without argument. Retraining moves the reported accuracies \u2014 seeds fix initialisation and data order, not GPU reduction
order \u2014 so the committed `public/bert-sst2-model.json` is what the viewer and the chapters report, and `npm run validate`
re-checks the JS forward pass against whatever weights are present.

Training is not on the viewing path: the viewer loads the committed JSON.

## BERT paper figures

    bash scripts/get-bert-figures.sh   # fetches the Devlin et al. (2018) e-print and vectorises its four figures

The four figures shown in the BERT chapters are reproductions of the published ones, converted by
that script from the authors' arXiv source. They carry attribution and no licence grant from this
project; see `LICENSE-BERT-figures.md` (an attribution note, unlike `LICENSE-llm-viz`, which is a
licence).

## Layout

- `src/llm/TransformerModel.ts`: JS forward pass, greedy decoding, GPU buffers
- `src/llm/GptModelLayout.ts`: the shared layout builder (layer norm, attention, FFN), and the 3D layout of the encoder (left) and decoder (right)
- `src/llm/VitModel.ts`, `src/llm/VitModelLayout.ts`: the ViT's JS forward pass and 3D layout
- `src/llm/BertModel.ts`: BERT's JS forward pass, its WordPiece tokenizer, and the MLM / NSP / sentiment heads
- `src/llm/BertModelLayout.ts`: BERT's 3D layout (141 cubes \u2014 the cheapest of the three always-on layouts)
- `src/llm/walkthrough/`: the chapters of the guide (`WalkthroughVit.tsx` for the ViT, `WalkthroughBert.tsx` for BERT)
- `src/llm/components/TocDiagram.tsx`: the Figure 1-style table of contents
