"""
qatt_agnews.py — Quantum Attention experiment on AG News, for a Slurm cluster.

One seed per process (pass --seed), so it slots into a Slurm job array where each
array task runs one seed in parallel. Writes a per-seed JSON to avoid write
collisions between array tasks. Merge afterwards with merge_agnews.py.

Backend note: for this batch-heavy workload (batch x L^2 circuit evals per step),
default.qubit vectorizes the whole batch through torch and threads across CPU cores,
and is usually FASTER here than lightning.qubit. Benchmark both with --backend.

Typical use on the cluster (see README):
    python qatt_agnews.py --seed 0 --data_dir ./agnews --out_dir ./agnews_results
    # or inside a Slurm array: --seed $SLURM_ARRAY_TASK_ID
"""

import os, sys, csv, json, time, math, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# DEFAULTS  (override via CLI)
# --------------------------------------------------------------------------- #
N_CLASSES   = 4                 # AG News: World, Sports, Business, Sci/Tech
MAX_LEN     = 30                # AG News text is longer than TREC; cost ~ L^2
DEFAULT_N   = 2000              # train-fit size (matched to TREC for fair compare)
EPOCHS      = 20
VAL_SIZE    = 500
MIN_FREQ    = 2

PAD, UNK = 0, 1

CONDITIONS = {
    "classical": dict(use_quantum=False, entangle="full"),
    "qnone":     dict(use_quantum=True,  entangle="none"),
    "qintra":    dict(use_quantum=True,  entangle="intra"),
    "qcross":    dict(use_quantum=True,  entangle="cross"),
    "qfull":     dict(use_quantum=True,  entangle="full"),
}

# --------------------------------------------------------------------------- #
# PENNYLANE  — backend chosen at runtime (default.qubit recommended here)
# --------------------------------------------------------------------------- #
import pennylane as qml
_BACKEND  = "default.qubit"
_DIFFMETH = "backprop"

def build_quantum_score_qnode(n_qubits=4, entangle="full"):
    assert n_qubits % 2 == 0 and n_qubits >= 2
    h = n_qubits // 2
    q_wires = list(range(h)); k_wires = list(range(h, n_qubits))
    cross = [(q_wires[i], k_wires[i]) for i in range(h)]
    intra = ([(q_wires[i], q_wires[i+1]) for i in range(h-1)]
           + [(k_wires[i], k_wires[i+1]) for i in range(h-1)])
    cz_pairs = {"none": [], "intra": intra, "cross": cross,
                "full": cross + intra}[entangle]
    dev = qml.device(_BACKEND, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=_DIFFMETH)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        for w in range(n_qubits):
            qml.RX(weights[0, w, 0], wires=w); qml.RY(weights[0, w, 1], wires=w); qml.RZ(weights[0, w, 2], wires=w)
        for a, b in cz_pairs:
            qml.CZ(wires=[a, b])
        for w in range(n_qubits):
            qml.RX(weights[1, w, 0], wires=w); qml.RY(weights[1, w, 1], wires=w); qml.RZ(weights[1, w, 2], wires=w)
        return qml.expval(qml.PauliZ(q_wires[0]) @ qml.PauliZ(k_wires[0]))
    return circuit

# --------------------------------------------------------------------------- #
# MODEL  (verbatim from the notebook core)
# --------------------------------------------------------------------------- #
class QuantumAttentionHead(nn.Module):
    def __init__(self, head_dim, n_qubits=4, entangle="full", chunk=8192):
        super().__init__()
        assert n_qubits % 2 == 0
        self.n_qubits = n_qubits; self.h = n_qubits // 2
        self.entangle = entangle; self.chunk = chunk
        self.q_proj = nn.Linear(head_dim, self.h)
        self.k_proj = nn.Linear(head_dim, self.h)
        self.qweights = nn.Parameter(0.1 * torch.randn(2, n_qubits, 3))
        self.scale = nn.Parameter(torch.tensor(4.0))
        self.circuit = build_quantum_score_qnode(n_qubits, entangle)

    def _run_circuit(self, flat, qw):
        n = flat.shape[0]
        if n <= self.chunk:
            return self.circuit(flat, qw)
        outs = [self.circuit(flat[i:i+self.chunk], qw) for i in range(0, n, self.chunk)]
        return torch.cat(outs, dim=0)

    def forward(self, q, k, v, key_mask=None, debug=False):
        B, L, D = q.shape
        qa = torch.tanh(self.q_proj(q)) * math.pi
        ka = torch.tanh(self.k_proj(k)) * math.pi
        qa_exp = qa.unsqueeze(2).expand(B, L, L, self.h)
        ka_exp = ka.unsqueeze(1).expand(B, L, L, self.h)
        pairs = torch.cat([qa_exp, ka_exp], dim=-1)
        flat = pairs.reshape(-1, self.n_qubits).float()
        scores = self._run_circuit(flat.cpu(), self.qweights.cpu())
        scores = scores.reshape(B, L, L).to(device=v.device, dtype=v.dtype)
        scores = scores * self.scale
        if key_mask is not None:
            scores = scores.masked_fill(key_mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v), attn


class MixedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, use_quantum=True, entangle="full", n_qubits=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.head_dim = d_model // n_heads
        self.use_quantum = use_quantum
        self.q_lin = nn.Linear(d_model, d_model); self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model); self.out_lin = nn.Linear(d_model, d_model)
        if use_quantum:
            self.qhead = QuantumAttentionHead(self.head_dim, n_qubits, entangle)

    def forward(self, x, key_mask=None, return_attn=False, debug=False):
        B, L, _ = x.shape; H, Dh = self.n_heads, self.head_dim
        q = self.q_lin(x).view(B, L, H, Dh).transpose(1, 2)
        k = self.k_lin(x).view(B, L, H, Dh).transpose(1, 2)
        v = self.v_lin(x).view(B, L, H, Dh).transpose(1, 2)
        outs, attns = [], {}
        for hh in range(H):
            if self.use_quantum and hh == 0:
                o, a = self.qhead(q[:,hh], k[:,hh], v[:,hh], key_mask=key_mask, debug=debug)
                attns["quantum"] = a
            else:
                sc = torch.matmul(q[:,hh], k[:,hh].transpose(-2,-1)) / math.sqrt(Dh)
                if key_mask is not None:
                    sc = sc.masked_fill(key_mask == 0, float("-inf"))
                a = F.softmax(sc, dim=-1); o = torch.matmul(a, v[:,hh])
                if hh == 1 or (not self.use_quantum and hh == 0):
                    attns.setdefault("classical", a)
            outs.append(o)
        out = torch.stack(outs, dim=1).transpose(1,2).reshape(B, L, H*Dh)
        return (self.out_lin(out), attns) if return_attn else self.out_lin(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, use_quantum=False, entangle="full", n_qubits=4, dropout=0.1):
        super().__init__()
        self.attn = MixedMultiHeadAttention(d_model, n_heads, use_quantum, entangle, n_qubits)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(), nn.Linear(ff_dim, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_mask=None, return_attn=False, debug=False):
        if return_attn:
            a_out, attns = self.attn(x, key_mask, return_attn=True, debug=debug)
        else:
            a_out, attns = self.attn(x, key_mask, debug=debug), None
        x = self.norm1(x + self.drop(a_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        return (x, attns) if return_attn else x


class TextTransformer(nn.Module):
    def __init__(self, vocab_size, n_classes, d_model=64, n_heads=4, ff_dim=128,
                 max_len=40, use_quantum=True, entangle="full", n_qubits=4,
                 quantum_in_block=0, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.max_len = max_len
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim,
                             use_quantum=(use_quantum and b == quantum_in_block),
                             entangle=entangle, n_qubits=n_qubits, dropout=dropout)
            for b in range(2)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, input_ids, attn_mask, return_attn=False, debug=False):
        B, L = input_ids.shape; L = min(L, self.max_len)
        input_ids = input_ids[:, :L]; attn_mask = attn_mask[:, :L]
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        km = attn_mask.unsqueeze(1); collected = {}
        for bi, blk in enumerate(self.blocks):
            if return_attn:
                x, attns = blk(x, km, return_attn=True, debug=(debug and bi == 0))
                if attns: collected[f"block{bi}"] = attns
            else:
                x = blk(x, km, debug=(debug and bi == 0))
        x = self.norm(x)
        m = attn_mask.unsqueeze(-1).float()
        pooled = (x*m).sum(1) / m.sum(1).clamp(min=1.0)
        return (self.head(pooled), collected) if return_attn else self.head(pooled)


def attention_entropy(attn, key_mask=None, eps=1e-9):
    p = attn.clamp(min=eps); ent = -(p * p.log()).sum(-1)
    if key_mask is not None:
        valid = key_mask.squeeze(1).float()
        return ((ent*valid).sum() / valid.sum().clamp(min=1.0)).item()
    return ent.mean().item()

# --------------------------------------------------------------------------- #
# DATA  — AG News local CSV  [class(1-4), title, description]
# --------------------------------------------------------------------------- #
def tokenize(s): return s.lower().split()

def build_vocab(examples, min_freq=2):
    from collections import Counter
    c = Counter()
    for e in examples: c.update(tokenize(e["text"]))
    v = {"<pad>": PAD, "<unk>": UNK}
    for w, f in c.most_common():
        if f >= min_freq: v[w] = len(v)
    return v

def encode(s, v, ml): return [v.get(t, UNK) for t in tokenize(s)][:ml]

def batchify(examples, v, ml, bs, shuffle, seed, device):
    idx = list(range(len(examples)))
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(examples), generator=g).tolist()
    batches = []
    for i in range(0, len(examples), bs):
        chunk = [examples[j] for j in idx[i:i+bs]]
        seqs = [encode(e["text"], v, ml) for e in chunk]
        L = max(max((len(s) for s in seqs), default=1), 1)
        ids = torch.full((len(seqs), L), PAD, dtype=torch.long)
        m = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, s in enumerate(seqs):
            if s:
                ids[r, :len(s)] = torch.tensor(s); m[r, :len(s)] = 1
        y = torch.tensor([e["label"] for e in chunk])
        batches.append((ids.to(device), m.to(device), y.to(device)))
    return batches

def load_agnews(data_dir):
    def rd(path):
        ex = []
        with open(path, encoding="utf-8") as f:
            for row in csv.reader(f):
                if len(row) < 3: continue
                label = int(row[0]) - 1                       # 1..4 -> 0..3
                text = (row[1] + " " + row[2]).replace("\\", " ").strip()
                if text:
                    ex.append({"text": text, "label": label})
        return ex
    train = rd(os.path.join(data_dir, "train.csv"))
    test  = rd(os.path.join(data_dir, "test.csv"))
    return train, test

@torch.no_grad()
def accuracy(model, batches):
    model.eval(); c = t = 0
    for ids, m, y in batches:
        c += (model(ids, m).argmax(-1) == y).sum().item(); t += y.numel()
    return c / max(t, 1)

@torch.no_grad()
def mean_entropy(model, batches, max_batches=6):
    model.eval(); q, cl = [], []
    for bi, (ids, m, y) in enumerate(batches):
        if bi >= max_batches: break
        _, attns = model(ids, m, return_attn=True)
        km = m[:, :ids.shape[1]].unsqueeze(1)
        qblk = next((d for d in attns.values() if "quantum" in d), {})
        if "quantum" in qblk: q.append(attention_entropy(qblk["quantum"], km))
        cblk = qblk if "classical" in qblk else next((d for d in attns.values() if "classical" in d), {})
        if "classical" in cblk: cl.append(attention_entropy(cblk["classical"], km))
    return (sum(q)/len(q) if q else float("nan"),
            sum(cl)/len(cl) if cl else float("nan"))

# --------------------------------------------------------------------------- #
# TRAIN ONE (condition, seed)
# --------------------------------------------------------------------------- #
def run_one(cond, seed, data, vocab, n_classes, cfg):
    train_fit, val, test = data
    cc = CONDITIONS[cond]; device = cfg["device"]
    torch.manual_seed(seed)
    model = TextTransformer(
        vocab_size=len(vocab), n_classes=n_classes,
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], max_len=cfg["max_len"],
        use_quantum=cc["use_quantum"], entangle=cc["entangle"],
        n_qubits=cfg["n_qubits"], quantum_in_block=cfg.get("quantum_in_block", 0)).to(device)
    qp = [p for n, p in model.named_parameters() if "qweights" in n]
    cp = [p for n, p in model.named_parameters() if "qweights" not in n]
    groups = [{"params": cp, "lr": cfg["lr"]}]
    if qp: groups.append({"params": qp, "lr": cfg["qlr"]})
    opt = torch.optim.Adam(groups)
    val_b  = batchify(val,  vocab, cfg["max_len"], 128, False, 0, device)
    test_b = batchify(test, vocab, cfg["max_len"], 128, False, 0, device)
    best_val, test_at_best, best_ep = -1.0, 0.0, 0
    t0 = time.time()
    for ep in range(1, cfg["epochs"]+1):
        model.train()
        for ids, m, y in batchify(train_fit, vocab, cfg["max_len"], cfg["batch_size"], True, 1000*seed+ep, device):
            opt.zero_grad(); F.cross_entropy(model(ids, m), y).backward(); opt.step()
        va = accuracy(model, val_b)
        if va > best_val:
            best_val, test_at_best, best_ep = va, accuracy(model, test_b), ep
        print(f"      [{cond} seed{seed}] epoch {ep:2d}  val={va:.4f}  best={best_val:.4f}", flush=True)
    h_q, h_c = mean_entropy(model, test_b)
    return dict(test_acc=test_at_best, val_acc=best_val, best_epoch=best_ep,
                H_quantum=h_q, H_classical=h_c,
                params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                n_qubits=cfg["n_qubits"], n_heads=cfg["n_heads"],
                quantum_in_block=cfg.get("quantum_in_block", 0),
                seconds=round(time.time()-t0, 1))

# --------------------------------------------------------------------------- #
# MAIN — one seed, all five conditions, write per-seed JSON
# --------------------------------------------------------------------------- #
def _save(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f: json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def _load(path):
    return json.load(open(path)) if os.path.exists(path) else {}

def main():
    global _BACKEND, _DIFFMETH, MAX_LEN
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data_dir", type=str, default="./agnews")
    ap.add_argument("--out_dir",  type=str, default="./agnews_results")
    ap.add_argument("--limit_train", type=int, default=DEFAULT_N)
    ap.add_argument("--max_len", type=int, default=MAX_LEN)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--quantum_in_block", type=int, default=0, choices=[0, 1])
    ap.add_argument("--n_qubits", type=int, default=4,
                    help="circuit width; encoding bottleneck is h=n_qubits//2 dims per q,k")
    ap.add_argument("--conditions", type=str, default="all",
                    help="comma-separated subset of conditions, or 'all'")
    ap.add_argument("--tag", type=str, default="",
                    help="filename tag -> agnews_<tag>_seed<N>.json (for sweeps)")
    ap.add_argument("--backend", type=str, default="default.qubit",
                    choices=["default.qubit", "lightning.qubit"])
    ap.add_argument("--threads", type=int, default=0,
                    help="torch intra-op threads; 0 = use all (from SLURM_CPUS_PER_TASK if set)")
    args = ap.parse_args()

    # backend
    _BACKEND = args.backend
    _DIFFMETH = "adjoint" if args.backend == "lightning.qubit" else "backprop"
    MAX_LEN = args.max_len

    # threads
    nthreads = args.threads or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or os.cpu_count()
    torch.set_num_threads(nthreads)
    os.environ.setdefault("OMP_NUM_THREADS", str(nthreads))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[cfg] seed={args.seed} backend={_BACKEND} threads={nthreads} device={device} "
          f"max_len={args.max_len} limit_train={args.limit_train} "
          f"quantum_in_block={args.quantum_in_block}", flush=True)

    # data — fixed split (seed 12345) so every array task sees the same split
    train_all, test_all = load_agnews(args.data_dir)
    print(f"[data] AG News train={len(train_all)} test={len(test_all)} classes={N_CLASSES}", flush=True)
    g = torch.Generator().manual_seed(12345)
    perm = torch.randperm(len(train_all), generator=g).tolist()
    tr = [train_all[i] for i in perm][:args.limit_train + VAL_SIZE]
    val = tr[:VAL_SIZE]; train_fit = tr[VAL_SIZE:]
    vocab = build_vocab(train_fit, MIN_FREQ)
    data = (train_fit, val, test_all)
    print(f"[data] train_fit={len(train_fit)} val={len(val)} vocab={len(vocab)}", flush=True)

    cfg = dict(epochs=args.epochs, max_len=args.max_len, batch_size=64, d_model=64,
               n_heads=4, n_qubits=args.n_qubits, lr=2e-3, qlr=1e-2, device=device,
               quantum_in_block=args.quantum_in_block)

    if args.tag:
        base = f"agnews_{args.tag}"
    elif args.quantum_in_block == 1:
        base = "agnews_block1"
    else:
        base = "agnews"
    out_path = os.path.join(args.out_dir, f"{base}_seed{args.seed}.json")
    results = _load(out_path)   # resume within this seed if re-run

    if args.conditions == "all":
        conds_to_run = list(CONDITIONS)
    else:
        conds_to_run = [c.strip() for c in args.conditions.split(",") if c.strip()]
        bad = [c for c in conds_to_run if c not in CONDITIONS]
        if bad:
            raise SystemExit(f"unknown conditions: {bad}; valid: {list(CONDITIONS)}")
    print(f"[cfg] n_qubits={args.n_qubits} conditions={conds_to_run} out={out_path}", flush=True)

    for cond in conds_to_run:
        if cond in results:
            print(f"  skip {cond} (already done: test={results[cond]['test_acc']:.4f})", flush=True)
            continue
        print(f"  RUN  {cond} seed{args.seed} ...", flush=True)
        r = run_one(cond, args.seed, data, vocab, N_CLASSES, cfg)
        results[cond] = r
        _save(out_path, results)
        print(f"       test={r['test_acc']:.4f} val={r['val_acc']:.4f} "
              f"H_q={r['H_quantum']:.3f} H_c={r['H_classical']:.3f} ({r['seconds']}s)", flush=True)

    print(f"[done] seed {args.seed} -> {out_path}", flush=True)

if __name__ == "__main__":
    main()
