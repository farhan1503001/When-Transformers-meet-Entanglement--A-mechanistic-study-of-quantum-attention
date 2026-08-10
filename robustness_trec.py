"""
robustness_trec.py  —  Adversarial robustness experiment for the Quantum Attention paper.

Runs on a local Windows PC (no Colab, no Google Drive).
Trains all five conditions across SEEDS, saves best model weights, then evaluates
three token-level attacks at five intensities and records accuracy-vs-intensity curves.

Usage (PowerShell):
    python robustness_trec.py
    python robustness_trec.py --seeds 0 1 2          # quick 3-seed run
    python robustness_trec.py --data_dir C:/data/trec # custom TREC path

Outputs (all under RESULTS_DIR = quantum_attention/trec_robustness/):
    train_results.json          — accuracy / entropy per (condition, seed)
    robustness_results.json     — per-attack, per-intensity accuracy
    models/<cond>_seed<s>.pt    — best-epoch weights for each (condition, seed)
    summary printed to stdout   — AUC table + permutation test classical vs qcross
"""

import os, sys, json, time, math, itertools, random, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# CONFIG  (edit these if needed)
# --------------------------------------------------------------------------- #
DATA_DIR    = "trec"                          # folder with train.label, test.label
RESULTS_DIR = "quantum_attention/trec_robustness"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS      = 20
MAX_LEN     = 20
DEFAULT_N   = 2000
VAL_SIZE    = 500
MIN_FREQ    = 2
N_CLASSES   = 6
SEEDS       = [0, 1, 2, 3, 4]

# robustness sweep
ATK_INTENSITIES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
# conditions to run robustness eval on (classical + the three entanglement variants)
ROB_CONDS = ["classical", "qnone", "qcross", "qfull"]

BASE_CFG = dict(epochs=EPOCHS, max_len=MAX_LEN, batch_size=64, d_model=64,
                n_heads=4, n_qubits=4, lr=2e-3, qlr=1e-2, device=DEVICE,
                quantum_in_block=0)

# --------------------------------------------------------------------------- #
# PENNYLANE DEVICE  — lightning.qubit with adjoint diff (fast); fallback to
# default.qubit on backprop.  Quantum circuits always compute on CPU regardless
# of DEVICE above; the classical layers benefit from CUDA.
# --------------------------------------------------------------------------- #
import pennylane as qml
try:
    import pennylane_lightning   # noqa: F401
    _QML_BACKEND  = "lightning.qubit"
    _QML_DIFFMETH = "adjoint"
    print(f"[pennylane] using lightning.qubit (adjoint diff) — fast path")
except ImportError:
    _QML_BACKEND  = "default.qubit"
    _QML_DIFFMETH = "backprop"
    print(f"[pennylane] lightning not found, using default.qubit (backprop)")

PAD, UNK = 0, 1

# --------------------------------------------------------------------------- #
# QUANTUM CIRCUIT
# --------------------------------------------------------------------------- #
def build_quantum_score_qnode(n_qubits=4, entangle="full"):
    assert n_qubits % 2 == 0 and n_qubits >= 2
    h = n_qubits // 2
    q_wires = list(range(h))
    k_wires  = list(range(h, n_qubits))
    cross = [(q_wires[i], k_wires[i]) for i in range(h)]
    intra = ([(q_wires[i], q_wires[i+1]) for i in range(h-1)]
           + [(k_wires[i], k_wires[i+1]) for i in range(h-1)])
    cz_pairs = {"none": [], "intra": intra, "cross": cross,
                "full": cross + intra}[entangle]
    dev = qml.device(_QML_BACKEND, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method=_QML_DIFFMETH)
    def circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        for w in range(n_qubits):
            qml.RX(weights[0, w, 0], wires=w)
            qml.RY(weights[0, w, 1], wires=w)
            qml.RZ(weights[0, w, 2], wires=w)
        for a, b in cz_pairs:
            qml.CZ(wires=[a, b])
        for w in range(n_qubits):
            qml.RX(weights[1, w, 0], wires=w)
            qml.RY(weights[1, w, 1], wires=w)
            qml.RZ(weights[1, w, 2], wires=w)
        return qml.expval(qml.PauliZ(q_wires[0]) @ qml.PauliZ(k_wires[0]))

    return circuit

# --------------------------------------------------------------------------- #
# MODEL  (verbatim from notebook cell 7)
# --------------------------------------------------------------------------- #
class QuantumAttentionHead(nn.Module):
    def __init__(self, head_dim, n_qubits=4, entangle="full", chunk=8192):
        super().__init__()
        assert n_qubits % 2 == 0
        self.n_qubits = n_qubits
        self.h        = n_qubits // 2
        self.entangle = entangle
        self.chunk    = chunk
        self.q_proj   = nn.Linear(head_dim, self.h)
        self.k_proj   = nn.Linear(head_dim, self.h)
        self.qweights = nn.Parameter(0.1 * torch.randn(2, n_qubits, 3))
        self.scale    = nn.Parameter(torch.tensor(4.0))
        self.circuit  = build_quantum_score_qnode(n_qubits, entangle)

    def _run_circuit(self, flat, qw):
        n = flat.shape[0]
        if n <= self.chunk:
            return self.circuit(flat, qw)
        outs = [self.circuit(flat[i:i+self.chunk], qw)
                for i in range(0, n, self.chunk)]
        return torch.cat(outs, dim=0)

    def forward(self, q, k, v, key_mask=None, debug=False):
        B, L, D = q.shape
        qa = torch.tanh(self.q_proj(q)) * math.pi
        ka = torch.tanh(self.k_proj(k)) * math.pi
        qa_exp = qa.unsqueeze(2).expand(B, L, L, self.h)
        ka_exp = ka.unsqueeze(1).expand(B, L, L, self.h)
        pairs  = torch.cat([qa_exp, ka_exp], dim=-1)
        flat   = pairs.reshape(-1, self.n_qubits).float()
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
        self.head_dim   = d_model // n_heads
        self.use_quantum = use_quantum
        self.q_lin  = nn.Linear(d_model, d_model)
        self.k_lin  = nn.Linear(d_model, d_model)
        self.v_lin  = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model)
        if use_quantum:
            self.qhead = QuantumAttentionHead(self.head_dim, n_qubits, entangle)

    def forward(self, x, key_mask=None, return_attn=False, debug=False):
        B, L, _ = x.shape
        H, Dh   = self.n_heads, self.head_dim
        q = self.q_lin(x).view(B, L, H, Dh).transpose(1, 2)
        k = self.k_lin(x).view(B, L, H, Dh).transpose(1, 2)
        v = self.v_lin(x).view(B, L, H, Dh).transpose(1, 2)
        outs, attns = [], {}
        for hh in range(H):
            if self.use_quantum and hh == 0:
                o, a = self.qhead(q[:,hh], k[:,hh], v[:,hh],
                                  key_mask=key_mask, debug=debug)
                attns["quantum"] = a
            else:
                sc = torch.matmul(q[:,hh], k[:,hh].transpose(-2,-1)) / math.sqrt(Dh)
                if key_mask is not None:
                    sc = sc.masked_fill(key_mask == 0, float("-inf"))
                a = F.softmax(sc, dim=-1)
                o = torch.matmul(a, v[:,hh])
                if hh == 1 or (not self.use_quantum and hh == 0):
                    attns.setdefault("classical", a)
            outs.append(o)
        out = torch.stack(outs, dim=1).transpose(1,2).reshape(B, L, H*Dh)
        out = self.out_lin(out)
        return (out, attns) if return_attn else out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, use_quantum=False,
                 entangle="full", n_qubits=4, dropout=0.1):
        super().__init__()
        self.attn  = MixedMultiHeadAttention(d_model, n_heads, use_quantum,
                                             entangle, n_qubits)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(),
                                   nn.Linear(ff_dim, d_model))
        self.drop  = nn.Dropout(dropout)

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
        self.blocks  = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim,
                             use_quantum=(use_quantum and b == quantum_in_block),
                             entangle=entangle, n_qubits=n_qubits, dropout=dropout)
            for b in range(2)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, input_ids, attn_mask, return_attn=False, debug=False):
        B, L = input_ids.shape
        L    = min(L, self.max_len)
        input_ids = input_ids[:, :L]
        attn_mask = attn_mask[:, :L]
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x   = self.tok_emb(input_ids) + self.pos_emb(pos)
        km  = attn_mask.unsqueeze(1)
        collected = {}
        for bi, blk in enumerate(self.blocks):
            if return_attn:
                x, attns = blk(x, km, return_attn=True, debug=(debug and bi==0))
                if attns:
                    collected[f"block{bi}"] = attns
            else:
                x = blk(x, km, debug=(debug and bi==0))
        x = self.norm(x)
        m = attn_mask.unsqueeze(-1).float()
        pooled = (x*m).sum(1) / m.sum(1).clamp(min=1.0)
        logits = self.head(pooled)
        return (logits, collected) if return_attn else logits


# --------------------------------------------------------------------------- #
# DATA UTILITIES  (verbatim from notebook)
# --------------------------------------------------------------------------- #
def tokenize(s):
    return s.lower().split()

def build_vocab(examples, min_freq=2):
    from collections import Counter
    c = Counter()
    for e in examples:
        c.update(tokenize(e["text"]))
    v = {"<pad>": PAD, "<unk>": UNK}
    for w, f in c.most_common():
        if f >= min_freq:
            v[w] = len(v)
    return v

def encode(s, v, ml):
    return [v.get(t, UNK) for t in tokenize(s)][:ml]

def batchify(examples, v, ml, bs, shuffle, seed, device):
    idx = list(range(len(examples)))
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(examples), generator=g).tolist()
    batches = []
    for i in range(0, len(examples), bs):
        chunk = [examples[j] for j in idx[i:i+bs]]
        seqs  = [encode(e["text"], v, ml) for e in chunk]
        L     = max(max((len(s) for s in seqs), default=1), 1)
        ids   = torch.full((len(seqs), L), PAD, dtype=torch.long)
        m     = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, s in enumerate(seqs):
            if s:
                ids[r, :len(s)] = torch.tensor(s)
                m[r,   :len(s)] = 1
        y = torch.tensor([e["label"] for e in chunk])
        batches.append((ids.to(device), m.to(device), y.to(device)))
    return batches

@torch.no_grad()
def accuracy(model, batches):
    model.eval()
    c = t = 0
    for ids, m, y in batches:
        c += (model(ids, m).argmax(-1) == y).sum().item()
        t += y.numel()
    return c / max(t, 1)

@torch.no_grad()
def mean_entropy(model, batches, max_batches=6):
    model.eval()
    q, cl = [], []
    for bi, (ids, m, y) in enumerate(batches):
        if bi >= max_batches: break
        _, attns = model(ids, m, return_attn=True)
        km = m[:, :ids.shape[1]].unsqueeze(1)
        qblk = next((d for d in attns.values() if "quantum" in d), {})
        if "quantum" in qblk:
            q.append(_attn_entropy(qblk["quantum"], km))
        cblk = qblk if "classical" in qblk else \
               next((d for d in attns.values() if "classical" in d), {})
        if "classical" in cblk:
            cl.append(_attn_entropy(cblk["classical"], km))
    return (sum(q)/len(q) if q else float("nan"),
            sum(cl)/len(cl) if cl else float("nan"))

def _attn_entropy(attn, key_mask=None, eps=1e-9):
    p = attn.clamp(min=eps)
    ent = -(p * p.log()).sum(-1)
    if key_mask is not None:
        valid = key_mask.squeeze(1).float()
        return ((ent*valid).sum() / valid.sum().clamp(min=1.0)).item()
    return ent.mean().item()

# --------------------------------------------------------------------------- #
# DATASET LOADING  (local paths, no Colab)
# --------------------------------------------------------------------------- #
COARSE = ["ABBR","ENTY","DESC","HUM","LOC","NUM"]
C2I    = {c:i for i,c in enumerate(COARSE)}

def load_trec(data_dir):
    def rd(p):
        ex = []
        for line in open(p, "rb"):
            line = line.replace(b"\xf0", b" ").strip().decode("latin-1")
            if not line: continue
            lab, _, t = line.partition(" ")
            c = lab.split(":")[0]
            if c in C2I and t:
                ex.append({"text": t, "label": C2I[c]})
        return ex
    train = rd(os.path.join(data_dir, "train.label"))
    test  = rd(os.path.join(data_dir, "test.label"))
    print(f"TREC-6  train={len(train)}  test={len(test)}")
    return train, test

# --------------------------------------------------------------------------- #
# SAVE / LOAD UTILITIES
# --------------------------------------------------------------------------- #
def _load(path):
    return json.load(open(path)) if os.path.exists(path) else {}

def _save(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def _model_path(cond, seed):
    return os.path.join(RESULTS_DIR, "models", f"{cond}_seed{seed}.pt")

def _rebuild_model(cond, vocab, cfg, ff_dim=None):
    """Re-instantiate the model from the checkpoint's own cfg."""
    cc = CONDITIONS[cond]
    ffd = ff_dim if ff_dim is not None else cfg["d_model"] * 2
    return TextTransformer(
        vocab_size=len(vocab), n_classes=N_CLASSES,
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], max_len=cfg["max_len"],
        ff_dim=ffd,
        use_quantum=cc["use_quantum"], entangle=cc["entangle"],
        n_qubits=cfg["n_qubits"],
        quantum_in_block=cfg.get("quantum_in_block", 0))

def _load_checkpoint(mp):
    """Load checkpoint; supports both new {state_dict,cfg} and legacy bare state_dict."""
    ckpt = torch.load(mp, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt["state_dict"], ckpt.get("cfg", BASE_CFG), ckpt.get("ff_dim", None)
    return ckpt, BASE_CFG, None   # legacy: bare state_dict

# --------------------------------------------------------------------------- #
# CONDITIONS
# --------------------------------------------------------------------------- #
CONDITIONS = {
    "classical": dict(use_quantum=False, entangle="full"),
    "qnone":     dict(use_quantum=True,  entangle="none"),
    "qintra":    dict(use_quantum=True,  entangle="intra"),
    "qcross":    dict(use_quantum=True,  entangle="cross"),
    "qfull":     dict(use_quantum=True,  entangle="full"),
}

# --------------------------------------------------------------------------- #
# TRAINING  (run_one + model saving)
# --------------------------------------------------------------------------- #
def run_one_and_save(cond, seed, data, vocab, cfg):
    train_fit, val, test = data
    cc     = CONDITIONS[cond]
    device = cfg["device"]
    torch.manual_seed(seed)
    model = TextTransformer(
        vocab_size=len(vocab), n_classes=N_CLASSES,
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], max_len=cfg["max_len"],
        use_quantum=cc["use_quantum"], entangle=cc["entangle"],
        n_qubits=cfg["n_qubits"],
        quantum_in_block=cfg.get("quantum_in_block", 0)).to(device)
    qp = [p for n, p in model.named_parameters() if "qweights" in n]
    cp = [p for n, p in model.named_parameters() if "qweights" not in n]
    groups = [{"params": cp, "lr": cfg["lr"]}]
    if qp:
        groups.append({"params": qp, "lr": cfg["qlr"]})
    opt = torch.optim.Adam(groups)
    val_b  = batchify(val,  vocab, cfg["max_len"], 128, False, 0, device)
    test_b = batchify(test, vocab, cfg["max_len"], 128, False, 0, device)
    best_val, test_at_best, best_ep = -1.0, 0.0, 0
    best_state = None
    t0 = time.time()
    for ep in range(1, cfg["epochs"]+1):
        model.train()
        for ids, m, y in batchify(train_fit, vocab, cfg["max_len"],
                                  cfg["batch_size"], True, 1000*seed+ep, device):
            opt.zero_grad()
            F.cross_entropy(model(ids, m), y).backward()
            opt.step()
        va = accuracy(model, val_b)
        if va > best_val:
            best_val, test_at_best, best_ep = va, accuracy(model, test_b), ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    # save best-epoch weights + cfg so _rebuild_model always uses the right dims
    mp = _model_path(cond, seed)
    os.makedirs(os.path.dirname(os.path.abspath(mp)), exist_ok=True)
    torch.save({"state_dict": best_state, "cfg": cfg, "cond": cond,
                "vocab_size": len(vocab), "ff_dim": cfg["d_model"]*2}, mp)
    # restore best weights for entropy measurement
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    h_q, h_c = mean_entropy(model, test_b)
    return dict(test_acc=test_at_best, val_acc=best_val, best_epoch=best_ep,
                H_quantum=h_q, H_classical=h_c,
                params=sum(p.numel() for p in model.parameters() if p.requires_grad),
                n_qubits=cfg["n_qubits"], n_heads=cfg["n_heads"],
                quantum_in_block=cfg.get("quantum_in_block", 0),
                seconds=round(time.time()-t0, 1))

def run_training(train_all, test_all, seeds):
    path   = os.path.join(RESULTS_DIR, "train_results.json")
    results = _load(path)
    g = torch.Generator().manual_seed(12345)
    tr   = list(train_all)
    perm = torch.randperm(len(tr), generator=g).tolist()
    tr   = [tr[i] for i in perm]
    train_fit = tr[VAL_SIZE:VAL_SIZE+DEFAULT_N]
    val       = tr[:VAL_SIZE]
    vocab     = build_vocab(train_fit, MIN_FREQ)
    data      = (train_fit, val, test_all)
    for cond in CONDITIONS:
        results.setdefault(cond, {})
        for s in seeds:
            if str(s) in results[cond]:
                print(f"  skip {cond} seed{s}  "
                      f"(test={results[cond][str(s)]['test_acc']:.4f})")
                continue
            mp = _model_path(cond, s)
            if os.path.exists(mp):
                print(f"  model exists for {cond} seed{s}, skipping retrain — "
                      f"add to results dict only")
                # load for entropy measurement only
                sd, ckpt_cfg, ff_dim = _load_checkpoint(mp)
                model = _rebuild_model(cond, vocab, ckpt_cfg, ff_dim).to(DEVICE)
                model.load_state_dict({k: v.to(DEVICE) for k, v in sd.items()})
                test_b = batchify(test_all, vocab, MAX_LEN, 128, False, 0, DEVICE)
                h_q, h_c = mean_entropy(model, test_b)
                r = dict(test_acc=accuracy(model, test_b),
                         val_acc=float("nan"), best_epoch=-1,
                         H_quantum=h_q, H_classical=h_c,
                         params=sum(p.numel() for p in model.parameters()),
                         n_qubits=BASE_CFG["n_qubits"], n_heads=BASE_CFG["n_heads"],
                         quantum_in_block=0, seconds=0.0)
            else:
                print(f"  train {cond} seed{s} ...", flush=True)
                r = run_one_and_save(cond, s, data, vocab, BASE_CFG)
            results[cond][str(s)] = r
            _save(path, results)
            print(f"         test={r['test_acc']:.4f}  H_q={r['H_quantum']:.3f}  "
                  f"H_c={r['H_classical']:.3f}  ({r['seconds']}s)", flush=True)
    return results, vocab, data

# --------------------------------------------------------------------------- #
# ATTACK FUNCTIONS  — operate on list-of-examples, return new list
# --------------------------------------------------------------------------- #
def _rng(seed): return random.Random(seed)

def attack_word_drop(examples, p, seed=42):
    """Drop each token independently with probability p."""
    rng = _rng(seed)
    out = []
    for e in examples:
        toks = tokenize(e["text"])
        kept = [t for t in toks if rng.random() > p] or toks[:1]
        out.append({"text": " ".join(kept), "label": e["label"]})
    return out

def attack_word_replace(examples, p, vocab_words, seed=42):
    """Replace each token with a random vocab word with probability p."""
    rng  = _rng(seed)
    pool = vocab_words
    out  = []
    for e in examples:
        toks = tokenize(e["text"])
        toks = [rng.choice(pool) if rng.random() < p else t for t in toks]
        out.append({"text": " ".join(toks), "label": e["label"]})
    return out

def attack_oov_inject(examples, p, seed=42):
    """Replace each token with an OOV token (not in vocab) with probability p."""
    rng = _rng(seed)
    OOV = "xqz_oov_xqz"   # guaranteed absent from any real vocabulary
    out = []
    for e in examples:
        toks = tokenize(e["text"])
        toks = [OOV if rng.random() < p else t for t in toks]
        out.append({"text": " ".join(toks), "label": e["label"]})
    return out

# --------------------------------------------------------------------------- #
# ROBUSTNESS EVALUATION
# --------------------------------------------------------------------------- #
def robustness_eval(test_all, vocab, seeds, intensities=ATK_INTENSITIES):
    """
    For each condition in ROB_CONDS, loads best-epoch weights for each seed,
    evaluates three attacks at each intensity, and averages across seeds.
    Returns a nested dict: results[cond][attack][str(p)] = mean_acc
    """
    path = os.path.join(RESULTS_DIR, "robustness_results.json")
    res  = _load(path)
    vocab_words = [w for w in vocab if w not in ("<pad>", "<unk>")]

    for cond in ROB_CONDS:
        res.setdefault(cond, {})
        for atk_name in ("word_drop", "word_replace", "oov_inject"):
            res[cond].setdefault(atk_name, {})
            for p in intensities:
                pk = f"{p:.2f}"
                if pk in res[cond][atk_name]:
                    print(f"  skip {cond} {atk_name} p={p:.1f}")
                    continue
                seed_accs = []
                for s in seeds:
                    mp = _model_path(cond, s)
                    if not os.path.exists(mp):
                        print(f"  WARNING: no model at {mp}, skipping seed {s}")
                        continue
                    sd, ckpt_cfg, ff_dim = _load_checkpoint(mp)
                    model = _rebuild_model(cond, vocab, ckpt_cfg, ff_dim).to(DEVICE)
                    model.load_state_dict({k: v.to(DEVICE) for k, v in sd.items()})
                    model.eval()
                    if atk_name == "word_drop":
                        atk_test = attack_word_drop(test_all, p, seed=s)
                    elif atk_name == "word_replace":
                        atk_test = attack_word_replace(test_all, p, vocab_words, seed=s)
                    else:
                        atk_test = attack_oov_inject(test_all, p, seed=s)
                    batches = batchify(atk_test, vocab, MAX_LEN, 128, False, 0, DEVICE)
                    seed_accs.append(accuracy(model, batches))
                if not seed_accs: continue
                mean_acc = sum(seed_accs) / len(seed_accs)
                res[cond][atk_name][pk] = round(mean_acc, 4)
                _save(path, res)
                print(f"  {cond:10} {atk_name:14} p={p:.1f}  acc={mean_acc:.4f}",
                      flush=True)
    return res

# --------------------------------------------------------------------------- #
# AUC HELPER
# --------------------------------------------------------------------------- #
def auc(acc_dict, intensities=ATK_INTENSITIES):
    """Trapezoidal AUC over intensity curve (higher = more robust)."""
    ys = [acc_dict.get(f"{p:.2f}", float("nan")) for p in intensities]
    pairs = [(intensities[i], ys[i], intensities[i+1], ys[i+1])
             for i in range(len(intensities)-1)
             if not (math.isnan(ys[i]) or math.isnan(ys[i+1]))]
    if not pairs: return float("nan")
    return sum(0.5*(y0+y1)*(x1-x0) for x0,y0,x1,y1 in pairs)

# --------------------------------------------------------------------------- #
# SUMMARY PRINTER
# --------------------------------------------------------------------------- #
def print_summary(train_res, rob_res):
    print("\n" + "="*72)
    print("CLEAN ACCURACY  (mean ± std across seeds)")
    print("="*72)
    print(f"{'Condition':12} {'Test acc':>10} {'H_quantum':>10} {'H_classical':>12}")
    for cond in CONDITIONS:
        accs = [train_res[cond][str(s)]["test_acc"]
                for s in SEEDS if str(s) in train_res.get(cond,{})]
        hqs  = [train_res[cond][str(s)]["H_quantum"]
                for s in SEEDS if str(s) in train_res.get(cond,{})
                and not math.isnan(train_res[cond][str(s)]["H_quantum"])]
        hcs  = [train_res[cond][str(s)]["H_classical"]
                for s in SEEDS if str(s) in train_res.get(cond,{})
                and not math.isnan(train_res[cond][str(s)]["H_classical"])]
        mu_a = sum(accs)/len(accs) if accs else float("nan")
        sd_a = (sum((x-mu_a)**2 for x in accs)/(len(accs)-1))**0.5 if len(accs)>1 else 0.0
        mu_q = sum(hqs)/len(hqs) if hqs else float("nan")
        mu_c = sum(hcs)/len(hcs) if hcs else float("nan")
        print(f"{cond:12} {mu_a:.4f}±{sd_a:.4f}  {mu_q:>9.3f}  {mu_c:>11.3f}")

    if not rob_res:
        return

    print("\n" + "="*72)
    print("ROBUSTNESS  — AUC under accuracy-vs-intensity curve (higher = more robust)")
    print("="*72)
    print(f"{'Condition':12} {'word_drop':>10} {'word_replace':>13} {'oov_inject':>11}")
    for cond in ROB_CONDS:
        if cond not in rob_res: continue
        a_wd = auc(rob_res[cond].get("word_drop",{}))
        a_wr = auc(rob_res[cond].get("word_replace",{}))
        a_ov = auc(rob_res[cond].get("oov_inject",{}))
        print(f"{cond:12} {a_wd:>10.4f} {a_wr:>13.4f} {a_ov:>11.4f}")

    print("\n" + "="*72)
    print("ACCURACY at p=0.3  (standard robustness operating point)")
    print("="*72)
    print(f"{'Condition':12} {'word_drop':>10} {'word_replace':>13} {'oov_inject':>11}")
    for cond in ROB_CONDS:
        if cond not in rob_res: continue
        wd = rob_res[cond].get("word_drop",{}).get("0.30", float("nan"))
        wr = rob_res[cond].get("word_replace",{}).get("0.30", float("nan"))
        ov = rob_res[cond].get("oov_inject",{}).get("0.30", float("nan"))
        print(f"{cond:12} {wd:>10.4f} {wr:>13.4f} {ov:>11.4f}")

# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--skip_train", action="store_true",
                        help="skip training, only run robustness on saved models")
    parser.add_argument("--skip_robustness", action="store_true",
                        help="only run training, skip robustness eval")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"device      : {DEVICE}")
    print(f"results dir : {os.path.abspath(RESULTS_DIR)}")
    print(f"seeds       : {args.seeds}")

    train_all, test_all = load_trec(args.data_dir)

    # ---- TRAINING ----
    if not args.skip_train:
        print("\n[PHASE 1] Training all conditions ...")
        train_res, vocab, data = run_training(train_all, test_all, args.seeds)
    else:
        print("\n[PHASE 1] Skipped (--skip_train).  Loading existing results ...")
        train_res = _load(os.path.join(RESULTS_DIR, "train_results.json"))
        # rebuild vocab from the same split
        g = torch.Generator().manual_seed(12345)
        tr   = list(train_all)
        perm = torch.randperm(len(tr), generator=g).tolist()
        tr   = [tr[i] for i in perm]
        vocab = build_vocab(tr[VAL_SIZE:VAL_SIZE+DEFAULT_N], MIN_FREQ)

    # ---- ROBUSTNESS ----
    rob_res = {}
    if not args.skip_robustness:
        print("\n[PHASE 2] Robustness evaluation ...")
        rob_res = robustness_eval(test_all, vocab, args.seeds)
    else:
        print("\n[PHASE 2] Skipped (--skip_robustness).")
        rob_res = _load(os.path.join(RESULTS_DIR, "robustness_results.json"))

    print_summary(train_res, rob_res)
    print(f"\nAll results saved under: {os.path.abspath(RESULTS_DIR)}")

if __name__ == "__main__":
    main()
