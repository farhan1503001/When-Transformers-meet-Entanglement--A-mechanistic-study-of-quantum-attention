"""
merge_qintra_dataeff.py  (AG News)

Assembles the per-seed qintra files written by run_qintra_dataeff.slurm
  agnews_results/agnews_Cqintra_N<N>_seed<s>.json   (each = {"qintra": {...}})
into the paper's Exp C key convention:
  { "C_qintra_N<N>": { "<seed>": {result}, ... }, ... }
matching how classical/qnone/qcross/qfull were saved in Colab (expC_dataeff.json).

If expC_dataeff.json (the Colab file with the 4 conditions) is present in --in_dir,
also writes expC_dataeff_with_qintra.json = the 4 conditions + qintra combined.

Usage:
    python merge_qintra_dataeff.py --in_dir ./agnews_results
"""
import os, json, glob, math, argparse

N_EFF = [250, 500, 1000, 2000, 5000]

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

    merged={}
    print("Assembling qintra data-efficiency points:")
    for N in N_EFF:
        key=f"C_qintra_N{N}"
        seed_files=sorted(glob.glob(os.path.join(args.in_dir, f"agnews_Cqintra_N{N}_seed*.json")))
        if not seed_files:
            print(f"  [missing] N={N}: no seed files yet")
            continue
        merged[key]={}
        for f in seed_files:
            s=f.split("_seed")[-1].split(".json")[0]
            d=json.load(open(f))
            if "qintra" not in d:
                print(f"  [warn] {os.path.basename(f)} has no 'qintra' key"); continue
            merged[key][s]=d["qintra"]
        accs=[merged[key][s]["test_acc"] for s in merged[key]]
        m,sd=mean_std(accs)
        print(f"  N={N:5}: {len(merged[key])} seeds  acc={m:.4f} ± {sd:.4f}")

    out=os.path.join(args.in_dir, "qintra_dataeff_agnews_merged.json")
    json.dump(merged, open(out,"w"), indent=2)
    print(f"\nWrote {out}")

    # optional: fold qintra into a copy of the Colab Exp C file if it's here
    expc=os.path.join(args.in_dir, "expC_dataeff.json")
    if os.path.exists(expc):
        base=json.load(open(expc))
        base.update(merged)
        comb=os.path.join(args.in_dir, "expC_dataeff_with_qintra.json")
        json.dump(base, open(comb,"w"), indent=2)
        print(f"Found Colab expC_dataeff.json -> wrote combined {comb}")
    else:
        print("(No expC_dataeff.json in this dir; copy it here to auto-combine,")
        print(" or just paste qintra_dataeff_agnews_merged.json into the paper chat.)")

if __name__=="__main__":
    main()
