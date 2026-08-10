"""
experiments.py — Rigorous multi-seed ablation for Quantum Attention.
Supports multiple text datasets via --dataset (trec | sst2).

Conditions
----------
  classical : baseline, head 0 is ordinary dot-product attention
  qnone     : quantum head, NO entangling gates        (score provably separable)
  qintra    : quantum head, within-q / within-k CZ only (score still separable)
  qcross    : quantum head, query<->key CZ only         (genuine q-k entanglement)
  qfull     : quantum head, all CZ gates

Datasets
--------
  trec : TREC-6 question classification (6 classes). Local files:
         <data_dir>/train.label, <data_dir>/test.label   (default data_dir=trec)
  sst2 : SST-2 binary sentiment (2 classes). Local files:
         <data_dir>/train.tsv, <data_dir>/dev.tsv         (default data_dir=sst2)
         dev.tsv is used as the test set (GLUE's real test labels are hidden).

Rigor
-----
  * Vocab built ONLY from the train-fit split (no val/test leakage).
  * Fixed validation split used for MODEL SELECTION: we report TEST accuracy at
    the epoch with the best VALIDATION accuracy (never peek at test).
  * Same train/val/test partition across all conditions & seeds.
  * Resumable: results checkpoint to --out after every run; re-running skips
    completed (condition, seed) pairs.

Examples
--------
  python experiments.py --dataset trec --epochs 30 --seeds 5 --out results_trec.json
  python experiments.py --dataset sst2 --epochs 25 --seeds 5 --limit-train 2000 \
                        --max-len 30 --device cpu --out results_sst2.json
"""

import argparse, json, os, time, math, itertools, csv
from collections import Counter
import torch
import torch.nn.functional as F

from model import TextTransformer, attention_entropy

PAD, UNK = 0, 1
CONDITIONS = {
    "classical": dict(use_quantum=False, entangle="full"),
    "qnone":     dict(use_quantum=True,  entangle="none"),
    "qintra":    dict(use_quantum=True,  entangle="intra"),
    "qcross":    dict(use_quantum=True,  entangle="cross"),
    "qfull":     dict(use_quantum=True,  entangle="full"),
}


def log(m):
    print(m, flush=True)


# --------------------------------------------------------------------------- #
# Dataset loaders -> (train_examples, test_examples, label_names)
# each example is {"text": str, "label": int}
# --------------------------------------------------------------------------- #
TREC_COARSE = ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"]
_TREC_C2I = {c: i for i, c in enumerate(TREC_COARSE)}


def _load_trec(data_dir):
    def read(path):
        ex = []
        with open(path, "rb") as f:
            for line in f:
                line = line.replace(b"\xf0", b" ").strip().decode("latin-1")
                if not line:
                    continue
                lab, _, text = line.partition(" ")
                coarse = lab.split(":")[0]
                if coarse in _TREC_C2I and text:
                    ex.append({"text": text, "label": _TREC_C2I[coarse]})
        return ex
    return (read(os.path.join(data_dir, "train.label")),
            read(os.path.join(data_dir, "test.label")),
            TREC_COARSE)


def _load_sst2(data_dir):
    def read(path):
        ex = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                lab, text = parts[0].strip(), parts[1].strip()
                if lab in ("0", "1") and text:        # also skips the header row
                    ex.append({"text": text, "label": int(lab)})
        return ex
    # GLUE test labels are hidden -> use dev.tsv as the held-out test set.
    return (read(os.path.join(data_dir, "train.tsv")),
            read(os.path.join(data_dir, "dev.tsv")),
            ["negative", "positive"])


def _load_agnews(data_dir):
    # AG News CSV: "label","title","description"; label is 1..4. Combine title+desc.
    def read(path):
        ex = []
        with open(path, encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                if len(row) < 3 or not row[0].strip().isdigit():
                    continue
                text = (row[1] + " " + row[2]).replace("\\", " ").strip()
                if text:
                    ex.append({"text": text, "label": int(row[0]) - 1})
        return ex
    return (read(os.path.join(data_dir, "train.csv")),
            read(os.path.join(data_dir, "test.csv")),
            ["World", "Sports", "Business", "Sci/Tech"])


DATASETS = {
    "trec": dict(loader=_load_trec, default_dir="trec"),
    "sst2": dict(loader=_load_sst2, default_dir="sst2"),
    "agnews": dict(loader=_load_agnews, default_dir="agnews"),
}


# --------------------------------------------------------------------------- #
# Tokenize / vocab / batch
# --------------------------------------------------------------------------- #
def tok(s):
    return s.lower().split()


def build_vocab(examples, min_freq=2):
    c = Counter()
    for e in examples:
        c.update(tok(e["text"]))
    v = {"<pad>": PAD, "<unk>": UNK}
    for w, f in c.most_common():
        if f >= min_freq:
            v[w] = len(v)
    return v


def encode(s, v, ml):
    return [v.get(t, UNK) for t in tok(s)][:ml]


def batchify(examples, v, ml, bs, shuffle, seed, device):
    idx = list(range(len(examples)))
    if shuffle:
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(examples), generator=g).tolist()
    batches = []
    for i in range(0, len(examples), bs):
        chunk = [examples[j] for j in idx[i:i + bs]]
        seqs = [encode(e["text"], v, ml) for e in chunk]
        L = max(max((len(s) for s in seqs), default=1), 1)
        ids = torch.full((len(seqs), L), PAD, dtype=torch.long)
        m = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, s in enumerate(seqs):
            if s:
                ids[r, :len(s)] = torch.tensor(s)
                m[r, :len(s)] = 1
        y = torch.tensor([e["label"] for e in chunk])
        batches.append((ids.to(device), m.to(device), y.to(device)))
    return batches


# --------------------------------------------------------------------------- #
# Train / eval one (condition, seed)
# --------------------------------------------------------------------------- #
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
        if bi >= max_batches:
            break
        _, attns = model(ids, m, return_attn=True)
        b0 = attns.get("block0", {})
        km = m[:, :ids.shape[1]].unsqueeze(1)
        if "quantum" in b0:
            q.append(attention_entropy(b0["quantum"], km))
        if "classical" in b0:
            cl.append(attention_entropy(b0["classical"], km))
    return (sum(q) / len(q) if q else float("nan"),
            sum(cl) / len(cl) if cl else float("nan"))


def run_one(cond, seed, data, vocab, n_classes, args, device):
    train_fit, val, test = data
    cfg = CONDITIONS[cond]
    torch.manual_seed(seed)
    model = TextTransformer(
        vocab_size=len(vocab), n_classes=n_classes,
        d_model=args.d_model, n_heads=args.n_heads, max_len=args.max_len,
        use_quantum=cfg["use_quantum"], entangle=cfg["entangle"],
    ).to(device)

    qp = [p for n, p in model.named_parameters() if "qweights" in n]
    cp = [p for n, p in model.named_parameters() if "qweights" not in n]
    groups = [{"params": cp, "lr": args.lr}]
    if qp:
        groups.append({"params": qp, "lr": args.qlr})
    opt = torch.optim.Adam(groups)

    val_b = batchify(val, vocab, args.max_len, 128, False, 0, device)
    test_b = batchify(test, vocab, args.max_len, 128, False, 0, device)

    best_val, test_at_best, best_ep = -1.0, 0.0, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        for ids, m, y in batchify(train_fit, vocab, args.max_len,
                                  args.batch_size, True, 1000 * seed + ep, device):
            opt.zero_grad()
            F.cross_entropy(model(ids, m), y).backward()
            opt.step()
        va = accuracy(model, val_b)
        if va > best_val:
            best_val, test_at_best, best_ep = va, accuracy(model, test_b), ep
    h_q, h_c = mean_entropy(model, test_b)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(test_acc=test_at_best, val_acc=best_val, best_epoch=best_ep,
                H_quantum=h_q, H_classical=h_c, params=n_params,
                seconds=round(time.time() - t0, 1))


# --------------------------------------------------------------------------- #
# Exact (or sampled) two-sided permutation test on the mean difference
# --------------------------------------------------------------------------- #
def permutation_test(a, b, max_exact=200000, n_random=100000, seed=0):
    a, b = list(a), list(b)
    na, nb = len(a), len(b)
    pooled = a + b
    obs = abs(sum(a) / na - sum(b) / nb)
    total = math.comb(na + nb, na)
    count = 0
    if total <= max_exact:
        for combo in itertools.combinations(range(na + nb), na):
            s = set(combo)
            ga = [pooled[i] for i in combo]
            gb = [pooled[i] for i in range(na + nb) if i not in s]
            if abs(sum(ga) / na - sum(gb) / nb) >= obs - 1e-12:
                count += 1
        return obs, count / total, f"exact ({total} perms)"
    rng = torch.Generator().manual_seed(seed)
    for _ in range(n_random):
        perm = torch.randperm(na + nb, generator=rng).tolist()
        ga = [pooled[i] for i in perm[:na]]
        gb = [pooled[i] for i in perm[na:]]
        if abs(sum(ga) / na - sum(gb) / nb) >= obs - 1e-12:
            count += 1
    return obs, count / n_random, f"sampled ({n_random} perms)"


def mean_std(xs):
    n = len(xs)
    mu = sum(xs) / n
    sd = (sum((x - mu) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return mu, sd


def summary_mean(summary, k):
    return sum(summary[k]) / len(summary[k])


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="trec")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=5, help="uses seeds 0..N-1")
    ap.add_argument("--conditions", default="classical,qnone,qintra,qcross,qfull")
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--val-size", type=int, default=500)
    ap.add_argument("--max-len", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--qlr", type=float, default=1e-2)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--data-dir", default=None,
                    help="defaults to the dataset's standard folder")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    data_dir = args.data_dir or DATASETS[args.dataset]["default_dir"]
    out = args.out or f"results_{args.dataset}.json"
    conditions = [c.strip() for c in args.conditions.split(",")]
    seeds = list(range(args.seeds))
    log(f"[cfg] dataset={args.dataset} device={device} epochs={args.epochs} "
        f"seeds={seeds} conditions={conditions} limit_train={args.limit_train} "
        f"max_len={args.max_len}")

    # ---- load + fixed partition: train_fit / val / test (seed-independent) ----
    train_all, test, label_names = DATASETS[args.dataset]["loader"](data_dir)
    n_classes = len(label_names)
    log(f"[data] {args.dataset}: train={len(train_all)} test={len(test)} "
        f"classes={n_classes} {label_names}")
    dist = Counter(e["label"] for e in train_all)
    log(f"[data] train label counts: "
        + ", ".join(f"{label_names[i]}={dist[i]}" for i in sorted(dist)))

    g = torch.Generator().manual_seed(12345)
    perm = torch.randperm(len(train_all), generator=g).tolist()
    train_all = [train_all[i] for i in perm]
    if args.limit_train is not None:
        train_all = train_all[:args.limit_train + args.val_size]
    val = train_all[:args.val_size]
    train_fit = train_all[args.val_size:]
    vocab = build_vocab(train_fit, min_freq=args.min_freq)

    lens = sorted(len(tok(e["text"])) for e in train_fit)
    p95 = lens[int(0.95 * len(lens))] if lens else 0
    trunc = 100.0 * sum(l > args.max_len for l in lens) / max(len(lens), 1)
    log(f"[split] train_fit={len(train_fit)} val={len(val)} test={len(test)} "
        f"vocab={len(vocab)} | token len p95={p95} max_len={args.max_len} "
        f"({trunc:.1f}% truncated)")
    data = (train_fit, val, test)

    # ---- resumable results store ----
    results = {}
    if os.path.exists(out):
        results = json.load(open(out))
        log(f"[resume] loaded prior runs from {out}")

    for cond in conditions:
        results.setdefault(cond, {})
        for seed in seeds:
            key = str(seed)
            if key in results[cond]:
                log(f"[skip] {cond} seed={seed} "
                    f"(test={results[cond][key]['test_acc']:.4f})")
                continue
            log(f"[run ] {cond} seed={seed} ...")
            r = run_one(cond, seed, data, vocab, n_classes, args, device)
            results[cond][key] = r
            json.dump(results, open(out, "w"), indent=2)
            log(f"       test={r['test_acc']:.4f} val={r['val_acc']:.4f} "
                f"ep*={r['best_epoch']} H_q={r['H_quantum']:.3f} "
                f"H_c={r['H_classical']:.3f} ({r['seconds']}s)")

    # ----------------------------- report -----------------------------------
    log("\n" + "=" * 64)
    log(f"{'condition':12} {'mean_test':>10} {'std':>8} {'n':>3}   params  H_q     H_c")
    summary = {}
    for cond in conditions:
        accs = [results[cond][str(s)]["test_acc"] for s in seeds
                if str(s) in results[cond]]
        if not accs:
            continue
        mu, sd = mean_std(accs)
        summary[cond] = accs
        any_run = results[cond][str(seeds[0])]
        hq = sum(results[cond][str(s)]["H_quantum"] for s in seeds
                 if str(s) in results[cond]) / len(accs)
        hc = sum(results[cond][str(s)]["H_classical"] for s in seeds
                 if str(s) in results[cond]) / len(accs)
        log(f"{cond:12} {mu:10.4f} {sd:8.4f} {len(accs):3d}   "
            f"{any_run['params']:5d}  {hq:6.3f}  {hc:6.3f}")
    log(f"{'random':12} {1.0/n_classes:10.4f}")

    log("\nPermutation tests (two-sided, H0: same mean test acc):")
    pairs = [("qfull", "classical"), ("qfull", "qnone"),
             ("qcross", "qnone"), ("qintra", "qnone"),
             ("qcross", "qintra"), ("qfull", "qcross")]
    for x, y in pairs:
        if x in summary and y in summary and len(summary[x]) > 1 and len(summary[y]) > 1:
            _, p, mode = permutation_test(summary[x], summary[y])
            sig = "*" if p < 0.05 else " "
            log(f"  {x:8} vs {y:8}: Δmean="
                f"{summary_mean(summary, x) - summary_mean(summary, y):+.4f} "
                f"p={p:.4f}{sig}  [{mode}]")
    log("\n(* p<0.05. With few seeds, treat p-values as directional, not final.)")

    csv_path = out.replace(".json", ".csv")
    with open(csv_path, "w") as f:
        f.write("condition,seed,test_acc,val_acc,best_epoch,"
                "H_quantum,H_classical,params\n")
        for cond in conditions:
            for s in seeds:
                if str(s) in results.get(cond, {}):
                    r = results[cond][str(s)]
                    f.write(f"{cond},{s},{r['test_acc']},{r['val_acc']},"
                            f"{r['best_epoch']},{r['H_quantum']},"
                            f"{r['H_classical']},{r['params']}\n")
    log(f"\nWrote {out} and {csv_path}")


if __name__ == "__main__":
    main()
