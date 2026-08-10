"""
merge_agnews.py — combine the per-seed AG News result files into one summary
with mean±std accuracy, the entropy gap, and permutation tests (classical vs each
quantum condition, and qcross/qfull vs qnone for the entanglement-specific test).

Usage:
    python merge_agnews.py --in_dir ./agnews_results
    python merge_agnews.py --in_dir ./agnews_results --block1   # for the depth run
"""
import os, json, glob, math, itertools, argparse

CONDS = ["classical", "qnone", "qintra", "qcross", "qfull"]

def mean_std(xs):
    n = len(xs)
    if n == 0: return float("nan"), float("nan")
    mu = sum(xs)/n
    sd = (sum((x-mu)**2 for x in xs)/(n-1))**0.5 if n > 1 else 0.0
    return mu, sd

def permutation_test(a, b, max_exact=200000, n_random=100000, seed=0):
    import random
    a, b = list(a), list(b); na, nb = len(a), len(b)
    if na == 0 or nb == 0: return float("nan"), float("nan"), "empty"
    pooled = a + b
    obs = abs(sum(a)/na - sum(b)/nb)
    total = math.comb(na+nb, na); cnt = 0
    if total <= max_exact:
        for combo in itertools.combinations(range(na+nb), na):
            s = set(combo)
            ga = [pooled[i] for i in combo]
            gb = [pooled[i] for i in range(na+nb) if i not in s]
            if abs(sum(ga)/na - sum(gb)/nb) >= obs - 1e-12: cnt += 1
        return obs, cnt/total, f"exact({total})"
    rng = random.Random(seed)
    for _ in range(n_random):
        perm = rng.sample(range(na+nb), na+nb)
        ga = [pooled[i] for i in perm[:na]]; gb = [pooled[i] for i in perm[na:]]
        if abs(sum(ga)/na - sum(gb)/nb) >= obs - 1e-12: cnt += 1
    return obs, cnt/n_random, f"sampled({n_random})"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="./agnews_results")
    ap.add_argument("--block1", action="store_true", help="merge the block-1 depth files")
    args = ap.parse_args()
    tag = "_block1" if args.block1 else ""
    pattern = os.path.join(args.in_dir, f"agnews{tag}_seed*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No files match {pattern}"); return
    print(f"Merging {len(files)} seed file(s):")
    for f in files: print("  ", os.path.basename(f))

    # collect per-condition lists across seeds
    acc = {c: [] for c in CONDS}
    hq  = {c: [] for c in CONDS}
    hc  = {c: [] for c in CONDS}
    for f in files:
        d = json.load(open(f))
        for c in CONDS:
            if c in d:
                acc[c].append(d[c]["test_acc"])
                if not math.isnan(d[c]["H_quantum"]):   hq[c].append(d[c]["H_quantum"])
                if not math.isnan(d[c]["H_classical"]): hc[c].append(d[c]["H_classical"])

    print("\n" + "="*70)
    print(f"AG News {'(block 1)' if args.block1 else '(block 0)'}  — {len(files)} seeds")
    print("="*70)
    print(f"{'Condition':12} {'Test acc':>16} {'H_quantum':>11} {'H_classical':>12}")
    for c in CONDS:
        ma, sa = mean_std(acc[c]); mq, _ = mean_std(hq[c]); mc, _ = mean_std(hc[c])
        hq_s = f"{mq:.3f}" if not math.isnan(mq) else "  -  "
        print(f"{c:12} {ma:.4f} ± {sa:.4f}   {hq_s:>9}   {mc:>11.3f}")

    print("\n" + "-"*70)
    print("Permutation tests on accuracy (p high = tied, the expected result)")
    print("-"*70)
    base = acc["classical"]
    for c in ["qnone", "qintra", "qcross", "qfull"]:
        obs, p, mode = permutation_test(base, acc[c])
        print(f"  classical vs {c:8}  Δacc={obs:.4f}  p={p:.4f}  {mode}")
    print("  -- entanglement-specific (the contrast that matters most) --")
    for c in ["qcross", "qfull"]:
        obs, p, mode = permutation_test(acc["qnone"], acc[c])
        print(f"  qnone     vs {c:8}  Δacc={obs:.4f}  p={p:.4f}  {mode}")

    summary = {"n_seeds": len(files), "block": 1 if args.block1 else 0,
               "accuracy": {c: acc[c] for c in CONDS},
               "H_quantum": {c: hq[c] for c in CONDS},
               "H_classical": {c: hc[c] for c in CONDS}}
    out = os.path.join(args.in_dir, f"agnews{tag}_summary.json")
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\nSummary written -> {out}")

if __name__ == "__main__":
    main()
