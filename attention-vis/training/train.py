"""
Train the tiny encoder-decoder Transformer that is shown in the visualization.

The architecture follows "Attention Is All You Need" (Vaswani et al., 2017) as closely as possible:
  - token embeddings scaled by sqrt(d_model), shared between encoder, decoder and the pre-softmax linear layer
  - fixed sinusoidal positional encodings
  - post-LN sublayers: x = LayerNorm(x + Sublayer(x))
  - multi-head attention without biases in the Q/K/V/O projections
  - position-wise feed-forward network with ReLU, d_ff = 4 * d_model
  - decoder: masked self-attention, encoder-decoder (cross) attention, feed-forward

Task (the "smallest example"): reverse a sequence of 6 letters drawn from {A, B, C}.
  source:         C A B B C A
  decoder input:  <s> A C B B A
  target:         A C B B A C

Run from the repo root:  python training/train.py
Outputs public/transformer-reverse-model.json (weights) and public/transformer-reverse-partials.json
(intermediate activations for one example, used to validate the JS forward pass).
"""

import base64
import json
import math
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = ['A', 'B', 'C', '<s>']
BOS = 3
T = 6
D_MODEL = 48
N_HEADS = 3
N_LAYERS = 2
D_FF = 4 * D_MODEL

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'public')


def sinusoidal_pe(T, C):
    pos = torch.arange(T, dtype=torch.float32)[:, None]
    i = torch.arange(0, C, 2, dtype=torch.float32)[None, :]
    angle = pos / torch.pow(10000.0, i / C)
    pe = torch.zeros(T, C)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle)
    return pe


class MultiHeadAttention(nn.Module):
    def __init__(self, C, nHeads):
        super().__init__()
        self.nHeads = nHeads
        self.A = C // nHeads
        self.q = nn.Linear(C, C, bias=False)
        self.k = nn.Linear(C, C, bias=False)
        self.v = nn.Linear(C, C, bias=False)
        self.o = nn.Linear(C, C, bias=False)

    def forward(self, xq, xkv, causal, cap=None):
        B, Tq, C = xq.shape
        Tk = xkv.shape[1]
        h, A = self.nHeads, self.A
        q = self.q(xq).view(B, Tq, h, A).transpose(1, 2)
        k = self.k(xkv).view(B, Tk, h, A).transpose(1, 2)
        v = self.v(xkv).view(B, Tk, h, A).transpose(1, 2)
        att = q @ k.transpose(-2, -1) / math.sqrt(A)  # (B, h, Tq, Tk)
        if causal:
            mask = torch.tril(torch.ones(Tq, Tk, dtype=torch.bool, device=xq.device))
            att = att.masked_fill(~mask, float('-inf'))
        sm = F.softmax(att, dim=-1)
        y = (sm @ v).transpose(1, 2).contiguous().view(B, Tq, C)
        out = self.o(y)
        if cap is not None:
            cap.update(q=q, k=k, v=v, attn=att, attnSm=sm, vOut=y, proj=out)
        return out


class FeedForward(nn.Module):
    def __init__(self, C, dff):
        super().__init__()
        self.fc1 = nn.Linear(C, dff)
        self.fc2 = nn.Linear(dff, C)

    def forward(self, x, cap=None):
        h = self.fc1(x)
        a = F.relu(h)
        y = self.fc2(a)
        if cap is not None:
            cap.update(ffnFc=h, ffnAct=a, ffnProj=y)
        return y


class EncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MultiHeadAttention(D_MODEL, N_HEADS)
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.ffn = FeedForward(D_MODEL, D_FF)
        self.ln2 = nn.LayerNorm(D_MODEL)

    def forward(self, x, cap=None):
        c1 = {} if cap is not None else None
        r1 = x + self.self_attn(x, x, False, c1)
        x = self.ln1(r1)
        c2 = {} if cap is not None else None
        r2 = x + self.ffn(x, c2)
        x2 = self.ln2(r2)
        if cap is not None:
            cap.update({f'selfAttn.{k}': v for k, v in c1.items()})
            cap.update(c2)
            cap.update(attnResid=r1, ln1=x, ffnResid=r2, ln2=x2)
        return x2


class DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = MultiHeadAttention(D_MODEL, N_HEADS)
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.cross_attn = MultiHeadAttention(D_MODEL, N_HEADS)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ffn = FeedForward(D_MODEL, D_FF)
        self.ln3 = nn.LayerNorm(D_MODEL)

    def forward(self, x, mem, cap=None):
        c1 = {} if cap is not None else None
        r1 = x + self.self_attn(x, x, True, c1)
        x1 = self.ln1(r1)
        c2 = {} if cap is not None else None
        r2 = x1 + self.cross_attn(x1, mem, False, c2)
        x2 = self.ln2(r2)
        c3 = {} if cap is not None else None
        r3 = x2 + self.ffn(x2, c3)
        x3 = self.ln3(r3)
        if cap is not None:
            cap.update({f'selfAttn.{k}': v for k, v in c1.items()})
            cap.update({f'crossAttn.{k}': v for k, v in c2.items()})
            cap.update(c3)
            cap.update(attnResid=r1, ln1=x1, crossResid=r2, ln2=x2, ffnResid=r3, ln3=x3)
        return x3


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(len(VOCAB), D_MODEL)
        nn.init.normal_(self.embed.weight, std=D_MODEL ** -0.5)
        self.register_buffer('pe', sinusoidal_pe(T, D_MODEL), persistent=False)
        self.encoder = nn.ModuleList([EncoderLayer() for _ in range(N_LAYERS)])
        self.decoder = nn.ModuleList([DecoderLayer() for _ in range(N_LAYERS)])

    def embed_tokens(self, idx):
        return self.embed(idx) * math.sqrt(D_MODEL) + self.pe[:idx.shape[1]]

    def forward(self, src, tgt_in, cap=None):
        x = self.embed_tokens(src)
        if cap is not None:
            cap['enc.embed'] = x
        for i, layer in enumerate(self.encoder):
            c = {} if cap is not None else None
            x = layer(x, c)
            if cap is not None:
                cap.update({f'enc.{i}.{k}': v for k, v in c.items()})
        mem = x
        y = self.embed_tokens(tgt_in)
        if cap is not None:
            cap['dec.embed'] = y
        for i, layer in enumerate(self.decoder):
            c = {} if cap is not None else None
            y = layer(y, mem, c)
            if cap is not None:
                cap.update({f'dec.{i}.{k}': v for k, v in c.items()})
        logits = y @ self.embed.weight.T  # pre-softmax linear shares the embedding matrix
        if cap is not None:
            cap['logits'] = logits
            cap['probs'] = F.softmax(logits, dim=-1)
        return logits


def make_batch(seqs, device):
    src = torch.tensor(seqs, dtype=torch.long, device=device)
    tgt = src.flip(1)
    bos = torch.full((src.shape[0], 1), BOS, dtype=torch.long, device=device)
    tgt_in = torch.cat([bos, tgt[:, :-1]], dim=1)
    return src, tgt_in, tgt


def tensor_to_json(t):
    t = t.detach().cpu().float().contiguous()
    return {'shape': list(t.shape), 'dtype': 'torch.float32', 'data': base64.b64encode(t.numpy().tobytes()).decode()}


def main():
    torch.manual_seed(1337)
    random.seed(1337)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    all_seqs = [[(n // 3 ** j) % 3 for j in range(T)] for n in range(3 ** T)]
    model = Transformer().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'params: {n_params}')

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)
    warmup = 200
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))

    for step in range(3000):
        batch = random.choices(all_seqs, k=128)
        src, tgt_in, tgt = make_batch(batch, device)
        logits = model(src, tgt_in)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(f'step {step} loss {loss.item():.4f}')

    model.eval()
    with torch.no_grad():
        # greedy autoregressive decoding over every possible input
        src, _, tgt = make_batch(all_seqs, device)
        dec = torch.full((src.shape[0], 1), BOS, dtype=torch.long, device=device)
        for _ in range(T):
            logits = model(src, dec)
            dec = torch.cat([dec, logits[:, -1].argmax(-1, keepdim=True)], dim=1)
        acc = (dec[:, 1:] == tgt).all(dim=1).float().mean().item()
        print(f'greedy sequence accuracy over all {len(all_seqs)} inputs: {acc * 100:.1f}%')

    model.cpu()
    sd = model.state_dict()
    config = {
        'vocab': VOCAB, 'vocab_size': len(VOCAB), 'bos': BOS, 'block_size': T,
        'n_embd': D_MODEL, 'n_head': N_HEADS, 'n_layer': N_LAYERS, 'd_ff': D_FF,
        'n_params': n_params,
    }
    with open(os.path.join(OUT_DIR, 'transformer-reverse-model.json'), 'w') as f:
        json.dump({'config': config, **{k: tensor_to_json(v) for k, v in sd.items()}}, f)

    # capture all intermediate values for one fully-decoded example, for JS validation
    example = [2, 0, 1, 1, 2, 0]  # C A B B C A
    src, tgt_in, tgt = make_batch([example], 'cpu')
    cap = {}
    with torch.no_grad():
        model(src, tgt_in, cap)
    out = {'src': tensor_to_json(src.float()), 'tgt_in': tensor_to_json(tgt_in.float()),
           **{k: tensor_to_json(v[0]) for k, v in cap.items()}}
    with open(os.path.join(OUT_DIR, 'transformer-reverse-partials.json'), 'w') as f:
        json.dump(out, f)
    print('pred:', ' '.join(VOCAB[i] for i in cap['probs'][0].argmax(-1).tolist()))


if __name__ == '__main__':
    main()
