"""
Train the tiny Vision Transformer (ViT) that is shown in the visualization.

The architecture follows "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
(Dosovitskiy et al., 2020), Section 3.1 / Eq. 1-4:
  - the image is split into fixed-size patches, each patch is flattened and linearly projected (E, with bias)
  - a learnable [class] embedding is prepended to the sequence of patch embeddings
  - learnable 1D position embeddings are added
  - pre-LN encoder layers: z' = MSA(LN(z)) + z ;  z = MLP(LN(z')) + z'
  - multi-head attention with biases, MLP with two layers and a GELU non-linearity, d_mlp = 4 * D
  - y = LN(z_L^0): the final layer norm is applied, and the [class] token's output goes through the head
  - classification head: a single linear layer (as used for fine-tuning in the paper)

Task (the "smallest example"): classify the 8x8 handwritten digits of scikit-learn's digits dataset (UCI).
With 2x2 patches, an 8x8 image is worth 16 "words", plus the [class] token: 17 tokens.

Run from the repo root (needs PyTorch and scikit-learn):  python training/train_vit.py
Outputs public/vit-digits-model.json (weights + some test images) and public/vit-digits-partials.json
(intermediate activations for one example, used to validate the JS forward pass).
"""

import base64
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

IMG = 8
PATCH = 2
N_PATCHES = (IMG // PATCH) ** 2  # 16
T = N_PATCHES + 1                # + [class] token
PATCH_DIM = PATCH * PATCH        # grayscale: P^2 * 1 channel
D_MODEL = 48
N_HEADS = 3
N_LAYERS = 2
D_MLP = 4 * D_MODEL
N_CLASSES = 10

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'public')


def patchify(img):
    """(B, 8, 8) -> (B, 16, 4): row-major patches, each flattened row-major."""
    B = img.shape[0]
    g = IMG // PATCH
    x = img.reshape(B, g, PATCH, g, PATCH).permute(0, 1, 3, 2, 4)
    return x.reshape(B, g * g, PATCH * PATCH)


class Attention(nn.Module):
    def __init__(self, C, nHeads):
        super().__init__()
        self.nHeads = nHeads
        self.A = C // nHeads
        self.q = nn.Linear(C, C)
        self.k = nn.Linear(C, C)
        self.v = nn.Linear(C, C)
        self.o = nn.Linear(C, C)

    def forward(self, x, cap=None):
        B, T_, C = x.shape
        h, A = self.nHeads, self.A
        q = self.q(x).view(B, T_, h, A).transpose(1, 2)
        k = self.k(x).view(B, T_, h, A).transpose(1, 2)
        v = self.v(x).view(B, T_, h, A).transpose(1, 2)
        att = q @ k.transpose(-2, -1) / A ** 0.5
        sm = F.softmax(att, dim=-1)
        y = (sm @ v).transpose(1, 2).contiguous().view(B, T_, C)
        out = self.o(y)
        if cap is not None:
            cap.update(q=q, k=k, v=v, attn=att, attnSm=sm, vOut=y, proj=out)
        return out


class Mlp(nn.Module):
    def __init__(self, C, d):
        super().__init__()
        self.fc1 = nn.Linear(C, d)
        self.fc2 = nn.Linear(d, C)

    def forward(self, x, cap=None):
        h = self.fc1(x)
        a = F.gelu(h)  # exact (erf) GELU, as in the paper's implementation
        y = self.fc2(a)
        if cap is not None:
            cap.update(mlpFc=h, mlpAct=a, mlpProj=y)
        return y


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.attn = Attention(D_MODEL, N_HEADS)
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.mlp = Mlp(D_MODEL, D_MLP)

    def forward(self, x, cap=None):
        c1 = {} if cap is not None else None
        l1 = self.ln1(x)
        x1 = x + self.attn(l1, c1)
        l2 = self.ln2(x1)
        c2 = {} if cap is not None else None
        x2 = x1 + self.mlp(l2, c2)
        if cap is not None:
            cap.update({f'attn.{k}': v for k, v in c1.items()})
            cap.update(c2)
            cap.update(ln1=l1, attnResid=x1, ln2=l2, mlpResid=x2)
        return x2


class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(PATCH_DIM, D_MODEL)
        self.cls_token = nn.Parameter(torch.zeros(D_MODEL))
        self.pos_embed = nn.Parameter(torch.randn(T, D_MODEL) * 0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList([Block() for _ in range(N_LAYERS)])
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, N_CLASSES)

    def forward(self, img, cap=None):
        B = img.shape[0]
        patches = patchify(img)
        pe = self.patch_embed(patches)
        tokens = torch.cat([self.cls_token.expand(B, 1, D_MODEL), pe], dim=1)  # (B, T, C)
        x = tokens + self.pos_embed
        if cap is not None:
            cap.update(patches=patches, tokens=tokens, embed=x)
        for i, blk in enumerate(self.blocks):
            c = {} if cap is not None else None
            x = blk(x, c)
            if cap is not None:
                cap.update({f'{i}.{k}': v for k, v in c.items()})
        xf = self.ln_f(x)
        logits = self.head(xf[:, 0])
        if cap is not None:
            cap.update(lnf=xf, logits=logits, probs=F.softmax(logits, dim=-1))
        return logits


def augment(img):
    """Random shifts by up to 1 pixel (the digits are tiny and the dataset is small)."""
    B = img.shape[0]
    pad = F.pad(img[:, None], (1, 1, 1, 1))[:, 0]
    dx = torch.randint(0, 3, (B,))
    dy = torch.randint(0, 3, (B,))
    out = torch.stack([pad[b, dy[b]:dy[b] + IMG, dx[b]:dx[b] + IMG] for b in range(B)])
    return out


def tensor_to_json(t):
    t = t.detach().cpu().float().contiguous()
    return {'shape': list(t.shape), 'dtype': 'torch.float32', 'data': base64.b64encode(t.numpy().tobytes()).decode()}


def main():
    torch.manual_seed(1337)
    random.seed(1337)
    np.random.seed(1337)

    digits = load_digits()
    X = torch.tensor(digits.images / 16.0, dtype=torch.float32)  # (1797, 8, 8) in [0, 1]
    y = torch.tensor(digits.target, dtype=torch.long)
    idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=0.2, random_state=0, stratify=digits.target)
    Xtr, ytr, Xte, yte = X[idx_train], y[idx_train], X[idx_test], y[idx_test]

    model = ViT()
    n_params = sum(p.numel() for p in model.parameters())
    print(f'params: {n_params}')

    steps = 6000
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.05)
    warmup = 300
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / steps))))

    for step in range(steps):
        bi = torch.randint(0, len(ytr), (128,))
        logits = model(augment(Xtr[bi]))
        loss = F.cross_entropy(logits, ytr[bi], label_smoothing=0.05)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0:
            with torch.no_grad():
                acc = (model(Xte).argmax(-1) == yte).float().mean().item()
            print(f'step {step} loss {loss.item():.4f} test acc {acc * 100:.1f}%')

    model.eval()
    with torch.no_grad():
        test_pred = model(Xte).argmax(-1)
        acc = (test_pred == yte).float().mean().item()
        train_acc = (model(Xtr).argmax(-1) == ytr).float().mean().item()
    print(f'train accuracy {train_acc * 100:.1f}%, test accuracy {acc * 100:.1f}% ({len(yte)} test images)')

    # a few test images for the in-browser picker: 6 of each digit, in test-set order
    picks = []
    for d in range(N_CLASSES):
        picks += [i for i in range(len(yte)) if yte[i] == d][:6]
    picks.sort()

    sd = model.state_dict()
    config = {
        'image_size': IMG, 'patch_size': PATCH, 'n_patches': N_PATCHES, 'block_size': T,
        'n_embd': D_MODEL, 'n_head': N_HEADS, 'n_layer': N_LAYERS, 'd_mlp': D_MLP,
        'n_classes': N_CLASSES, 'classes': [str(d) for d in range(N_CLASSES)],
        'n_params': n_params, 'test_accuracy': acc, 'n_test': len(yte),
    }
    out = {
        'config': config,
        **{k: tensor_to_json(v) for k, v in sd.items()},
        'examples.images': tensor_to_json(Xte[picks]),
        'examples.labels': tensor_to_json(yte[picks].float()),
    }
    with open(os.path.join(OUT_DIR, 'vit-digits-model.json'), 'w') as f:
        json.dump(out, f)

    # all intermediate values for the first picked example, for JS validation
    ex = Xte[picks][0:1]
    cap = {}
    with torch.no_grad():
        model(ex, cap)
    partials = {'image': tensor_to_json(ex[0]), **{k: tensor_to_json(v[0]) for k, v in cap.items()}}
    with open(os.path.join(OUT_DIR, 'vit-digits-partials.json'), 'w') as f:
        json.dump(partials, f)
    print('example label', int(yte[picks][0]), 'pred', int(cap['probs'][0].argmax()))


if __name__ == '__main__':
    main()
