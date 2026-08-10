"""Merge the AG News encoding-bottleneck (qubit-width) sweep.
Reads agnews_D_*_seed*.json, aggregates accuracy by (condition, n_qubits) across
seeds, and reports whether widening the bottleneck changes the null."""
import os, json, glob, math, argparse
from collections import defaultdict

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

    # acc[(cond, nq)] = [accs across seeds];  classical stored under nq=None
    acc=defaultdict(list); hq=defaultdict(list); hc=defaultdict(list)
    for f in files:
        d=json.load(open(f))
        for cond, v in d.items():
            nq = None if cond=="classical" else v["n_qubits"]
            acc[(cond,nq)].append(v["test_acc"])
            if not math.isnan(v["H_quantum"]):   hq[(cond,nq)].append(v["H_quantum"])
            if not math.isnan(v["H_classical"]): hc[(cond,nq)].append(v["H_classical"])

    # classical reference
    print("\n"+"="*64)
    print("AG News — encoding-bottleneck (qubit-width) sweep")
    print("="*64)
    if ("classical",None) in acc:
        m,s=mean_std(acc[("classical",None)])
        print(f"classical (reference)      acc={m:.4f} ± {s:.4f}")

    widths=sorted({nq for (c,nq) in acc if nq is not None})
    print(f"\n{'cond':8} " + " ".join(f"nq={w}(h={w//2})".rjust(14) for w in widths))
    for cond in ["qnone","qcross","qfull"]:
        row=f"{cond:8} "
        for w in widths:
            xs=acc.get((cond,w),[])
            if xs:
                m,s=mean_std(xs); row+=f"{m:.4f}±{s:.4f}".rjust(15)
            else:
                row+=" "*15
        print(row)

    print("\nEntropy gap (H_quantum vs H_classical) by width:")
    for cond in ["qnone","qcross","qfull"]:
        for w in widths:
            q=hq.get((cond,w),[]); c=hc.get((cond,w),[])
            if q:
                mq,_=mean_std(q); mc,_=mean_std(c)
                print(f"  {cond:7} nq={w}: H_q={mq:.3f}  H_c={mc:.3f}")

    print("\nReading guide:")
    print("  - Within a row (qcross or qfull), does accuracy rise as nq grows?")
    print("    Flat across widths = the null is robust to bottleneck width.")
    print("  - At each width, does qcross/qfull beat qnone? If not, entanglement")
    print("    stays invisible even when the encoding bottleneck is widened.")

    out=os.path.join(args.in_dir, "agnews_qubitsweep_summary.json")
    json.dump({f"{c}|nq{nq}":acc[(c,nq)] for (c,nq) in acc}, open(out,"w"), indent=2)
    print(f"\nSummary -> {out}")

if __name__=="__main__":
    main()
