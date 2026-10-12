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

## Running

Requires Node 18+.

    npm install
    npm run dev        # http://localhost:5173
    npm run build      # static site in dist/

## Retraining the model

    python training/train.py       # needs PyTorch; writes public/transformer-reverse-*.json
    python training/train_vit.py   # needs PyTorch & scikit-learn; writes public/vit-digits-*.json
    npm run validate               # checks both JS forward passes against PyTorch's activations
    npm run check:bert             # checks the BERT assets, weights, tokenizer and forward pass

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
- `src/llm/walkthrough/`: the chapters of the guide (`WalkthroughVit.tsx` for the ViT)
- `src/llm/components/TocDiagram.tsx`: the Figure 1-style table of contents
