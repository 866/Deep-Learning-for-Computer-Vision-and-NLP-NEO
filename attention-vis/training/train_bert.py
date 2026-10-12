"""Train a BERT-TINY-sized model (L=2, H=128, A=2, dFF=4H, V=600 WordPiece) on real SST-2:
pre-train with Masked Language Modeling + Next Sentence Prediction, then fine-tune a
[CLS] sentiment classifier — the full Figure 1 story of Devlin et al. (2018), arXiv:1810.04805.

    python training/train_bert.py            # full profile: 20000 pre-train steps, 4 fine-tune epochs
    python training/train_bert.py --fast     # 4000 steps / 2 epochs, for iteration only

Writes public/bert-sst2-model.json (weights + config + metrics + picker examples) and
public/bert-sst2-partials.json (activations for one fixed sentence, plus 20 tokenizer
fixtures). npm run check:bert and npm run validate compare the browser's forward pass against
these files, so every number the visualization shows is measured rather than invented.

Module attributes are named to match the exported tensor names (blocks.i.ln1, blocks.i.ffn.fc1,
...) so the export needs no per-block renaming and a missing tensor is obvious.

Two deliberate departures from the throwaway probes this design was measured with: a masked
position keeps its *original* token for the last 10% of the 15% (the paper's 80/10/10 is
mask / random token / unchanged; a probe had used [UNK] there), and a Next-Sentence positive
really is the adjacent sentence in corpus order rather than another random one.
"""
import argparse
import base64
import collections
import csv
import json
import os
import random
import time
import urllib.request
import zipfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)             # attention-vis/
REPO_ROOT = os.path.dirname(PKG_ROOT)        # the course repo, which holds data/
DATA_DIR = os.path.join(REPO_ROOT, 'data', 'SST-2')
OUT_DIR = os.path.join(PKG_ROOT, 'public')
SST2_URL = 'https://dl.fbaipublicfiles.com/glue/data/SST-2.zip'

# BERT-TINY (Turbo et al. 2019): the smallest named BERT configuration. T = 32 keeps a whole
# dev sentence inside the layout, which is what makes every cube on screen mean something.
T = 32
H = 128
A = 2
NL = 2
DFF = 4 * H
V_TARGET = 600
MAXP = T - 3          # room for [CLS], this sentence's [SEP], and a pair's second [SEP]


# PAD/UNK/CLS/SEP/MASK are filled in from the trained vocabulary before any encoding happens.
PAD = UNK = CLS = SEP = MASK = None


def fetch_sst2():
    """Download GLUE's SST-2 into <repo root>/data/SST-2/ unless it is already cached there."""
    needed = [os.path.join(DATA_DIR, n) for n in ('train.tsv', 'dev.tsv')]
    if all(os.path.exists(p) for p in needed):
        print(f'using cached SST-2 in {DATA_DIR}')
        return needed
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, 'SST-2.zip')
    print(f'downloading {SST2_URL}')
    with urllib.request.urlopen(SST2_URL, timeout=180) as r, open(zip_path, 'wb') as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    with zipfile.ZipFile(zip_path) as z:
        # the archive has its own top-level "SST-2/" directory, so it unpacks into data/ and lands
        # exactly on DATA_DIR rather than into it (which would give data/SST-2/SST-2/train.tsv)
        z.extractall(os.path.dirname(DATA_DIR))
    for p in needed:
        assert os.path.exists(p), f'{p} missing after unzip — unexpected archive layout'
    print(f'cached in {DATA_DIR} ({os.path.getsize(zip_path):,} bytes)')
    return needed


def load_tsv(path):
    with open(path, encoding='utf-8') as f:
        return [(row['sentence'].strip(), int(row['label'])) for row in csv.DictReader(f, delimiter='\t')]


class Attn(nn.Module):
    def __init__(s):
        super().__init__()
        s.qkv = nn.Linear(H, 3 * H)
        s.o = nn.Linear(H, H)
        s.d = H // A

    def forward(s, x, m, cap=None):
        B, n, _ = x.shape
        q, k, v = s.qkv(x).chunk(3, -1)
        sh = lambda t: t.view(B, n, A, s.d).transpose(1, 2)
        qh, kh, vh = sh(q), sh(k), sh(v)
        # bidirectional: no causal mask, only the padding mask
        attn = (qh @ kh.transpose(-1, -2)) / np.sqrt(s.d)
        attn = attn.masked_fill(~m[:, None, None, :], -1e9)
        p = attn.softmax(-1)
        if cap is not None:
            cap['q'] = q
            cap['attn'] = attn
            cap['attnSm'] = p
        return s.o((p @ vh).transpose(1, 2).reshape(B, n, H))


class Ffn(nn.Module):
    def __init__(s):
        super().__init__()
        s.fc1 = nn.Linear(H, DFF)
        s.fc2 = nn.Linear(DFF, H)

    def forward(s, h, cap=None):
        act = F.gelu(s.fc1(h))
        if cap is not None:
            cap['ffnAct'] = act
        return s.fc2(act)


class Blk(nn.Module):
    """Post-LN encoder block, as in the original BERT encoder: one norm after each sublayer,
    and no norm inside attention itself."""

    def __init__(s):
        super().__init__()
        s.attn = Attn()
        s.ln1 = nn.LayerNorm(H)
        s.ffn = Ffn()
        s.ln2 = nn.LayerNorm(H)

    def forward(s, x, m, cap=None):
        resid = x + s.attn(x, m, cap)
        h = s.ln1(resid)
        out = s.ln2(h + s.ffn(h, cap))
        if cap is not None:
            cap['resid'] = resid
            cap['ln1'] = h
            cap['ln2'] = out
        return out


class Bert(nn.Module):
    def __init__(s, vocab_size):
        super().__init__()
        V = vocab_size
        s.tok = nn.Embedding(V, H)
        s.seg = nn.Embedding(2, H)
        s.pos = nn.Embedding(T, H)
        s.pre = nn.LayerNorm(H)
        s.blocks = nn.ModuleList([Blk() for _ in range(NL)])
        s.mlm = nn.Linear(H, H)      # MLM head: transform + GELU + LN, then decode with the shared embedding matrix
        s.mlmln = nn.LayerNorm(H)
        s.mbias = nn.Parameter(torch.zeros(V))
        s.pool = nn.Linear(H, H)     # used by the NSP head only (tanh(pool([CLS])))
        s.nsp = nn.Linear(H, 2)
        s.cls = nn.Linear(H, 2)      # applied to the raw final [CLS] vector (paper §4.1: no pooler in between)

    def enc(s, ids, seg, cap=None):
        x = s.tok(ids) + s.seg(seg) + s.pos(torch.arange(ids.shape[1], device=ids.device)[None])
        x = s.pre(x)
        m = ids != PAD
        if cap is not None:
            cap['embedLN'] = x
        for i, b in enumerate(s.blocks):
            bcap = {} if cap is not None else None
            x = b(x, m, bcap)
            if cap is not None:
                for key, val in bcap.items():
                    cap[f'blocks.{i}.{key}'] = val
        if cap is not None:
            cap['embed'] = x
        return x, m

    def mlm_head(s, x):
        return s.mlmln(F.gelu(s.mlm(x))) @ s.tok.weight.T + s.mbias


def batch(idss, segs, dev):
    return (torch.tensor(np.array(idss), dtype=torch.long, device=dev),
            torch.tensor(np.array(segs), dtype=torch.long, device=dev))


def encode_pair(tk, a, b=None):
    """[CLS] A [SEP] (B [SEP]) with segment ids, truncated so the special tokens survive.
    Returns (ids, seg, n_real, truncated)."""
    ia = tk.encode(a).ids
    trunc = len(ia) > MAXP
    ia = ia[:MAXP]
    ids = [CLS] + ia + [SEP]
    seg = [0] * len(ids)
    if b is not None:
        ib_raw = tk.encode(b).ids
        keep = max(0, MAXP - len(ia))
        ib = ib_raw[:keep]
        trunc = trunc or len(ib_raw) > keep
        ids = ids + ib + [SEP]
        seg = [0] * (len(ia) + 2) + [1] * (len(ib) + 1)
    assert len(ids) <= T, (len(ids), a, b)
    n = len(ids)
    return ids + [PAD] * (T - n), seg + [0] * (T - n), n, trunc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=20000, help='pre-training steps')
    ap.add_argument('--epochs', type=int, default=4, help='fine-tuning epochs')
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--fast', action='store_true',
                    help='4000 steps / 2 epochs — iteration only, do not ship its metrics')
    args = ap.parse_args()
    if args.fast:
        args.steps, args.epochs = 4000, 2
        print('!! --fast: 4000 pre-train steps / 2 fine-tune epochs. These metrics are NOT what the '
              'visualization ships; run the full profile before regenerating public/bert-sst2-*.json')

    torch.manual_seed(1337)
    np.random.seed(1337)
    random.seed(1337)
    dev_t = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.time()

    # ---- data ----
    fetch_sst2()
    train = [(s, l) for s, l in load_tsv(os.path.join(DATA_DIR, 'train.tsv')) if s]
    dev = [(s, l) for s, l in load_tsv(os.path.join(DATA_DIR, 'dev.tsv')) if s]
    print(f'SST-2: {len(train)} train / {len(dev)} dev rows')

    # ---- WordPiece vocabulary from the real corpus (paper §3.1: WordPiece, 30k for BERT-Base) ----
    tk = Tokenizer(models.WordPiece(vocab={}, unk_token='[UNK]'))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.train_from_iterator([s for s, _ in train], trainers.WordPieceTrainer(
        vocab_size=V_TARGET, min_frequency=2, special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']))
    global PAD, UNK, CLS, SEP, MASK
    PAD, UNK, CLS, SEP, MASK = (tk.token_to_id(t) for t in ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'])
    V = len(tk.get_vocab())
    vocab = [tk.id_to_token(i) for i in range(V)]
    special = {PAD, UNK, CLS, SEP, MASK}
    print(f'WordPiece vocab: {V} tokens (PAD={PAD} UNK={UNK} CLS={CLS} SEP={SEP} MASK={MASK})')

    TR = train
    dev_enc = [encode_pair(tk, s) for s, _ in dev]
    dev_ids = torch.tensor(np.array([e[0] for e in dev_enc]), device=dev_t)
    dev_seg = torch.tensor(np.array([e[1] for e in dev_enc]), device=dev_t)
    dev_y = torch.tensor([l for _, l in dev], device=dev_t)
    # truncation_rate: share of the 872 dev sentences whose WordPiece encoding does not fit MAXP
    trunc = float(np.mean([e[3] for e in dev_enc]))

    model = Bert(V).to(dev_t)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'params: {n_params:,} (V={V}, T={T}, H={H}, L={NL}, A={A}, dFF={DFF}) device={dev_t}')

    rng = np.random.default_rng(0)

    # ---- stage 1: pre-training, MLM + NSP jointly (paper §3.1) ----
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    special_ids = torch.tensor(sorted(special), device=dev_t)
    for step in range(args.steps):
        idx = rng.integers(0, len(TR), args.batch)
        isnext = rng.random(args.batch) < 0.5
        pair = []
        for i, nx in zip(idx, isnext):
            if nx and i + 1 < len(TR):
                pair.append((TR[i][0], TR[i + 1][0]))          # adjacent in corpus order
            else:
                pair.append((TR[i][0], TR[rng.integers(0, len(TR))][0]))
        ids, seg, _, _ = zip(*[encode_pair(tk, a, b) for a, b in pair])
        ids, seg = batch(ids, seg, dev_t)
        lab = torch.tensor(isnext.astype(np.int64), device=dev_t)

        real = ~torch.isin(ids, special_ids)                    # never mask a special token
        keep = (torch.tensor(rng.random(ids.shape) < 0.15, device=dev_t)) & real
        lbl = ids.clone()
        lbl[~keep] = -100
        inp = ids.clone()
        r = torch.rand(ids.shape, device=dev_t)
        inp[keep & (r < 0.8)] = MASK                            # 80% -> [MASK]
        mid = keep & (r >= 0.8) & (r < 0.9)
        inp[mid] = torch.randint(len(special), V, (int(mid.sum()),), device=dev_t)   # 10% -> random token
        # the last 10% is left unchanged, exactly as the paper describes it

        x, _ = model.enc(inp, seg)
        logits = model.mlm_head(x)
        mlm_loss = F.cross_entropy(logits[keep], lbl[keep]) if bool(keep.any()) else 0.0 * logits.sum()
        nsp_loss = F.cross_entropy(model.nsp(model.pool(x[:, 0]).tanh()), lab)
        loss = mlm_loss + nsp_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 1000 == 0 or step == args.steps - 1:
            with torch.no_grad():
                acc = (logits[keep].argmax(-1) == lbl[keep]).float().mean().item() if bool(keep.any()) else 0.0
            print(f'  pre-train {step:5d}  mlm loss {mlm_loss.item():.3f} acc {acc:.3f}  '
                  f'nsp loss {nsp_loss.item():.3f}  ({time.time()-t0:.0f}s)', flush=True)

    # ---- NSP on held-out pairs: an adjacent dev pair vs. a random partner ----
    with torch.no_grad():
        model.eval()
        n_ok = n_tot = 0
        for i in range(0, len(dev) - 1, 2):
            pos = encode_pair(tk, dev[i][0], dev[i + 1][0])
            neg = encode_pair(tk, dev[i][0], dev[int(rng.integers(0, len(dev)))][0])
            ids, seg = batch([pos[0], neg[0]], [pos[1], neg[1]], dev_t)
            x, _ = model.enc(ids, seg)
            pred = model.nsp(model.pool(x[:, 0]).tanh()).argmax(-1)
            n_ok += int(pred[0] == 1) + int(pred[1] == 0)
            n_tot += 2
    nsp_acc = n_ok / max(1, n_tot)
    print(f'  NSP held-out acc {nsp_acc*100:.2f}% (n={n_tot}) — at chance is the expected result at this scale')
    model.train()

    # ---- MLM accuracy: overall, and on frequent whole words only (what a reader can judge) ----
    freq = collections.Counter()
    for st, _ in train:
        for t in tk.encode(st).ids:
            freq[t] += 1
    good = {t for t, c in freq.items() if c >= 100 and not tk.id_to_token(t).startswith('##')}

    def mlm_eval():
        model.eval()
        n_all = c_all = n_good = c_good = 0
        rngm = np.random.default_rng(7)
        with torch.no_grad():
            for st, _ in dev:
                ids, seg, _, _ = encode_pair(tk, st)
                for _ in range(3):
                    inp = list(ids)
                    maskidx = []
                    for i, t in enumerate(ids):
                        if t in special:
                            continue
                        if rngm.random() < 0.15:
                            inp[i] = MASK
                            maskidx.append(i)
                    if not maskidx:
                        continue
                    xt, _ = model.enc(torch.tensor([inp], device=dev_t), torch.tensor([seg], device=dev_t))
                    p = model.mlm_head(xt)[0]
                    for i in maskidx:
                        ok = int(p[i].argmax()) == ids[i]
                        n_all += 1
                        c_all += ok
                        if ids[i] in good:
                            n_good += 1
                            c_good += ok
        model.train()
        return c_all / max(1, n_all), c_good / max(1, n_good), n_good

    mlm_all, mlm_good, mlm_good_n = mlm_eval()
    print(f'  MLM masked top-1: all {mlm_all*100:.1f}%  |  frequent whole words {mlm_good*100:.1f}% (n={mlm_good_n})')

    # ---- stage 2: fine-tune the [CLS] sentiment head (paper §4.1, Figure 4b) ----
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    sent_acc = 0.0
    for ep in range(args.epochs):
        perm = rng.permutation(len(TR))
        for i in range(0, len(TR) - args.batch + 1, args.batch):
            sl = perm[i:i + args.batch]
            ids, seg, _, _ = zip(*[encode_pair(tk, TR[j][0]) for j in sl])
            ids, seg = batch(ids, seg, dev_t)
            y = torch.tensor([TR[j][1] for j in sl], device=dev_t)
            x, _ = model.enc(ids, seg)
            loss = F.cross_entropy(model.cls(x[:, 0]), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            model.eval()
            outs = []
            for i in range(0, len(dev), 256):
                x, _ = model.enc(dev_ids[i:i + 256], dev_seg[i:i + 256])
                outs.append(model.cls(x[:, 0]))
            sent_acc = (torch.cat(outs).argmax(-1) == dev_y).float().mean().item()
            model.train()
        print(f'  fine-tune epoch {ep+1}/{args.epochs}: SST-2 dev acc {sent_acc*100:.2f}%   ({time.time()-t0:.0f}s)', flush=True)
    model.eval()

    maj = float((dev_y == 1).float().mean().item())
    # 69% of the dev sentences are longer than MAXP pieces, so the headline accuracy is measured on
    # inputs the model can only see the beginning of. Reporting it alone would flatter neither the
    # model nor the layout, so the accuracy on the sentences that fit whole is reported beside it.
    whole = torch.tensor([not e[3] for e in dev_enc], device=dev_t)
    with torch.no_grad():
        outs = []
        for i in range(0, len(dev), 256):
            x, _ = model.enc(dev_ids[i:i + 256], dev_seg[i:i + 256])
            outs.append(model.cls(x[:, 0]))
        pred = torch.cat(outs).argmax(-1) == dev_y
        acc_whole = float(pred[whole].float().mean().item())
    n_whole = int(whole.sum().item())

    metrics = {
        'sentiment_dev_acc': sent_acc,
        'sentiment_dev_acc_untruncated': acc_whole,
        'n_dev_untruncated': n_whole,
        'majority_baseline': max(maj, 1 - maj),
        'mlm_top1_all': mlm_all,
        'mlm_top1_frequent': mlm_good,
        'nsp_dev_acc': nsp_acc,
        'truncation_rate': trunc,
        'n_dev': len(dev),
    }
    print('metrics:', json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}))

    # ---- export weights ----
    def tensor_to_json(t):
        t = t.detach().cpu().float().contiguous()
        return {'shape': list(t.shape), 'dtype': 'torch.float32',
                'data': base64.b64encode(t.numpy().tobytes()).decode()}

    RENAMES = {
        'tok.weight': 'embed.weight',
        'seg.weight': 'seg_embed.weight',
        'pos.weight': 'pos_embed.weight',
        'pre.weight': 'embed_ln.weight', 'pre.bias': 'embed_ln.bias',
        'mlm.weight': 'mlm.fc.weight', 'mlm.bias': 'mlm.fc.bias',
        'mlmln.weight': 'mlm.ln.weight', 'mlmln.bias': 'mlm.ln.bias',
        'mbias': 'mlm.bias',
    }
    out = {'config': {
        'model': 'bert-tiny-sst2',
        'vocab': vocab,
        'vocab_size': V,
        'block_size': T,
        'n_embd': H,
        'n_head': A,
        'n_layer': NL,
        'd_ff': DFF,
        'n_params': n_params,
        'special': {'pad': PAD, 'unk': UNK, 'cls': CLS, 'sep': SEP, 'mask': MASK},
        'classes': ['negative', 'positive'],
        'task': 'SST-2 sentiment classification, fine-tuned from MLM + NSP pre-training',
        'metrics': metrics,
        'mlm_frequent_word_count': mlm_good_n,
        'data': 'GLUE SST-2 (https://dl.fbaipublicfiles.com/glue/data/SST-2.zip)',
    }}
    for k, v in model.state_dict().items():
        if k.endswith('qkv.weight'):
            # split the fused QKV linear into q/k/v in that row order — numerically identical to
            # the fused form training ran, and the layout draws three separate matrices
            pre = k[:-len('qkv.weight')]
            for j, name in enumerate('qkv'):
                out[f'{pre}{name}.weight'] = tensor_to_json(v[j * H:(j + 1) * H])
        elif k.endswith('qkv.bias'):
            pre = k[:-len('qkv.bias')]
            for j, name in enumerate('qkv'):
                out[f'{pre}{name}.bias'] = tensor_to_json(v[j * H:(j + 1) * H])
        else:
            out[RENAMES.get(k, k)] = tensor_to_json(v)

    want_names = ['embed.weight', 'seg_embed.weight', 'pos_embed.weight', 'embed_ln.weight', 'embed_ln.bias',
                  'mlm.fc.weight', 'mlm.fc.bias', 'mlm.ln.weight', 'mlm.ln.bias', 'mlm.bias',
                  'pool.weight', 'pool.bias', 'nsp.weight', 'nsp.bias', 'cls.weight', 'cls.bias']
    for i in range(NL):
        want_names += [f'blocks.{i}.attn.{p}.{s}' for p in 'qkvo' for s in ('weight', 'bias')]
        want_names += [f'blocks.{i}.ln{s_}.{s}' for s_ in (1, 2) for s in ('weight', 'bias')]
        want_names += [f'blocks.{i}.ffn.{f}.{s}' for f in ('fc1', 'fc2') for s in ('weight', 'bias')]
    missing = [n for n in want_names if n not in out]
    assert not missing, f'export is missing tensors the browser expects: {missing}'

    # 8 held-out dev sentences for the in-browser picker: label-balanced, untruncated, not one-word
    picks = []
    for want in (1, 0):
        for i in range(len(dev)):
            if len([p for p in picks if dev[p][1] == want]) >= 4:
                break
            if dev[i][1] == want and len(dev[i][0]) > 25 and not dev_enc[i][3]:
                picks.append(i)
    assert len(picks) == 8, f'only found {len(picks)} picker sentences'
    out['examples.text'] = [dev[i][0] for i in picks]
    out['examples.labels'] = tensor_to_json(torch.tensor([float(dev[i][1]) for i in picks]))

    # ---- partials: one fixed sentence for the JS forward pass to reproduce, plus fixtures ----
    fixed_i = next(i for i in range(len(dev))
                   if not dev_enc[i][3] and sum(1 for t in dev_enc[i][0] if t in good) >= 2)
    f_ids, f_seg, _, _ = dev_enc[fixed_i]
    cap = {}
    with torch.no_grad():
        ids_t = torch.tensor([f_ids], device=dev_t)
        seg_t = torch.tensor([f_seg], device=dev_t)
        x, _ = model.enc(ids_t, seg_t, cap=cap)
        sent_probs = model.cls(x[:, 0]).softmax(-1)
        # The MLM display needs a [MASK] to mean anything, so it gets its own pass on a masked
        # input; the attention/residual walk-through and the sentiment bar are from the plain
        # sentence above, which is what the chapters step through.
        cand = [pos for pos, t in enumerate(f_ids) if t in good]
        mlm_pos = cand[len(cand) // 2]
        m_ids = list(f_ids)
        m_ids[mlm_pos] = MASK
        xm, _ = model.enc(torch.tensor([m_ids], device=dev_t), seg_t)
        mlm_logits = model.mlm_head(xm)

    partials = {
        'text': dev[fixed_i][0],
        'ids': tensor_to_json(ids_t[0]),
        'seg': tensor_to_json(seg_t[0]),
        'mlmIds': tensor_to_json(torch.tensor(m_ids, device=dev_t)),
        'mlmTarget': int(mlm_pos),
        'mlmTargetToken': vocab[f_ids[mlm_pos]],
        'sentimentLabel': int(dev[fixed_i][1]),
        'sentimentProbs': tensor_to_json(sent_probs[0]),
        'mlmLogits': tensor_to_json(mlm_logits[0]),
        'mlmProbs': tensor_to_json(mlm_logits[0].softmax(-1)),
    }
    for k, v in cap.items():
        partials[k] = tensor_to_json(v[0])

    def fixture(text):
        ids, _, _, tr = encode_pair(tk, text)
        raw = tk.encode(text).ids
        return {'text': text, 'ids': ids, 'toks': [vocab[t] for t in ids], 'truncated': bool(tr),
                'rawIds': raw, 'rawToks': [vocab[t] for t in raw]}

    # 20 tokenizer fixtures for the browser-side WordPiece check: real dev sentences, then the
    # edge cases a hand-written tokenizer gets wrong (UNK, truncation, punctuation-only,
    # apostrophes, punctuation glued to words, empty input)
    fixtures = [fixture(dev[i][0]) for i in range(14)]
    fixtures += [fixture(t) for t in [
        'the zzyzx quixotry of unobtainium',
        'this movie was so good that i would gladly watch it again next year and the year after '
        'that and probably a third time with friends too',
        '!!! ... ??? --',
        'it was n\'t great , but not awful either',
        'widely-praised , yet boring',
        '',
    ]]
    assert len(fixtures) == 20, f'{len(fixtures)} fixtures, expected 20'
    partials['sentences'] = fixtures

    os.makedirs(OUT_DIR, exist_ok=True)
    model_path = os.path.join(OUT_DIR, 'bert-sst2-model.json')
    partials_path = os.path.join(OUT_DIR, 'bert-sst2-partials.json')
    with open(model_path, 'w') as f:
        json.dump(out, f)
    with open(partials_path, 'w') as f:
        json.dump(partials, f)
    print(f'wrote {os.path.basename(model_path)} ({os.path.getsize(model_path)/1e6:.1f} MB) and '
          f'{os.path.basename(partials_path)} ({os.path.getsize(partials_path)/1e6:.1f} MB)')
    print(f'fixed partials sentence: {dev[fixed_i][0]!r} label={dev[fixed_i][1]} '
          f'mlm target={partials["mlmTargetToken"]!r}')
    print(f'total {time.time()-t0:.0f}s on {dev_t}')

    verify(out, partials)
    print('verify: OK')


def verify(out_model, out_partials):
    """Hard floors on what gets shipped. A green run here is what licenses the walkthrough to
    print real numbers, so these are assertions rather than a printed summary."""
    cfg = out_model['config']
    m = cfg['metrics']

    assert cfg['n_params'] == 512348, f'n_params moved: {cfg["n_params"]} != 512348'
    assert len(cfg['vocab']) == 600, f'vocab is {len(cfg["vocab"])} entries, expected 600'
    assert cfg['vocab_size'] == 600, f'config.vocab_size is {cfg["vocab_size"]}'
    assert 0.0 <= m['mlm_top1_all'] <= 1.0, f'mlm_top1_all out of range: {m["mlm_top1_all"]}'
    assert m['sentiment_dev_acc'] >= 0.70, (
        f'sentiment dev accuracy {m["sentiment_dev_acc"]:.4f} is below the 0.70 floor (majority '
        f'baseline is {m["majority_baseline"]:.4f}) — the visualization would be showing a model that does not work')
    assert m['mlm_top1_all'] >= 0.30, f'mlm_top1_all {m["mlm_top1_all"]:.4f} is below the 0.30 floor'
    assert m['n_dev'] == 872, f'dev set is {m["n_dev"]} rows, expected 872'

    for k, v in out_partials.items():
        if isinstance(v, dict) and 'data' in v:
            arr = np.frombuffer(base64.b64decode(v['data']), dtype=np.float32)
            assert bool(np.isfinite(arr).all()), f'partials[{k}] contains NaN/Inf'
    sp = np.frombuffer(base64.b64decode(out_partials['sentimentProbs']['data']), dtype=np.float32)
    assert abs(float(sp.sum()) - 1.0) < 1e-4, f'sentimentProbs sum to {float(sp.sum())}, not 1'


if __name__ == '__main__':
    main()
