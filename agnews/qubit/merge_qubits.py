"""Merge the AG News qubit-width sweep (Exp D), including qintra.
Reads agnews_D_*_seed*.json, aggregates accuracy by (condition, n_qubits)."""
import os, json, glob, math, argparse
from collections import defaultdict

CONDS = ["qnone", "qintra", "qcross", "qfull"]   # qintra included

def mean_std(xs):
    n=len(xs)
    if n==0: return float("nan"), float("nan")
    mu=sum(xs)/n
    sd=(sum((x-mu)**2 for x in xs)/(n-1))**0.5 if n>1 else 0.0
    return mu, sd

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, default="./agnews_results")
    args=ap.parse_args()
    files=sorted(glob.glob(os.path.join(args.in_dir, "agnews_D_*_seed*.json")))
    if not files:
        print("No sweep files (agnews_D_*_seed*.json) found."); return
    print(f"Merging {len(files)} sweep file(s)")

    acc=defaultdict(list); hq=defaultdict(list); hc=defaultdict(list)
    for f in files:
        d=json.load(open(f))
        for cond, v in d.items():
            nq = None if cond=="classical" else v["n_qubits"]
            acc[(cond,nq)].append(v["test_acc"])
            if not math.isnan(v["H_quantum"]):   hq[(cond,nq)].append(v["H_quantum"])
            if not math.isnan(v["H_classical"]): hc[(cond,nq)].append(v["H_classical"])

    print("\n"+"="*64)
    print("AG News — qubit-width (encoding-bottleneck) sweep, incl. qintra")
    print("="*64)
    if ("classical",None) in acc:
        m,s=mean_std(acc[("classical",None)])
        print(f"classical (reference)      acc={m:.4f} ± {s:.4f}")

    widths=sorted({nq for (c,nq) in acc if nq is not None})
    print(f"\n{'cond':8} " + " ".join(f"nq={w}(h={w//2})".rjust(15) for w in widths))
    for cond in CONDS:
        row=f"{cond:8} "
        for w in widths:
            xs=acc.get((cond,w),[])
            row += (f"{mean_std(xs)[0]:.4f}±{mean_std(xs)[1]:.4f}".rjust(16)) if xs else " "*16
        print(row)

    print("\nEntropy gap (H_quantum vs H_classical) by width:")
    for cond in CONDS:
        for w in widths:
            q=hq.get((cond,w),[]); c=hc.get((cond,w),[])
            if q:
                print(f"  {cond:7} nq={w}: H_q={mean_std(q)[0]:.3f}  H_c={mean_std(c)[0]:.3f}")

    out=os.path.join(args.in_dir, "agnews_qubitsweep_summary.json")
    json.dump({f"{c}|nq{nq}":acc[(c,nq)] for (c,nq) in acc}, open(out,"w"), indent=2)
    print(f"\nSummary -> {out}")

if __name__=="__main__":
    main()
