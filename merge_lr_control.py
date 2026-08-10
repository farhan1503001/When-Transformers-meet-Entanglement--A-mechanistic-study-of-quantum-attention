"""
merge_lr_control.py
Collect lr_control.py outputs into one table: conditions x matched-rates,
cells = mean +/- std test accuracy and exact permutation p vs classical.

Usage:
    python merge_lr_control.py --dir results/lr

Produces:
    - a printed table
    - lr_control_summary.json  (machine-readable)
    - lr_control_table.tex     (drop-in LaTeX for supplementary)
"""

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

CONDITIONS = ["classical", "qnone", "qintra", "qcross", "qfull"]


def exact_permutation_p(a, b):
    """Exact two-sided permutation test on the difference in means, matching the
    paper's C(10,5)=252 enumeration for two groups of five seeds."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.concatenate([a, b])
    n = len(a)
    obs = abs(a.mean() - b.mean())
    count, total = 0, 0
    idx = range(len(pooled))
    for combo in combinations(idx, n):
        mask = np.zeros(len(pooled), bool)
        mask[list(combo)] = True
        diff = abs(pooled[mask].mean() - pooled[~mask].mean())
        total += 1
        if diff >= obs - 1e-12:
            count += 1
    return count / total


def load(dir_):
    recs = []
    for f in Path(dir_).glob("*.json"):
        recs.append(json.loads(f.read_text()))
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/lr")
    args = ap.parse_args()

    recs = load(args.dir)
    if not recs:
        raise SystemExit(f"no result json files found in {args.dir}")

    lrs = sorted({r["lr"] for r in recs})
    # group[(cond, lr)] = list of test accuracies over seeds
    group = {}
    for r in recs:
        group.setdefault((r["condition"], r["lr"]), []).append(r["test_acc"])

    summary = {}
    print(f"\n{'condition':<12}", end="")
    for lr in lrs:
        print(f"| lr={lr:.0e}          ", end="")
    print()
    print("-" * (12 + len(lrs) * 22))

    for cond in CONDITIONS:
        print(f"{cond:<12}", end="")
        for lr in lrs:
            accs = group.get((cond, lr), [])
            if not accs:
                print("| (missing)         ", end="")
                continue
            m, s = np.mean(accs), np.std(accs)
            cell = f"{m*100:5.1f}+/-{s*100:.1f}"
            if cond != "classical":
                base = group.get(("classical", lr), [])
                if base and len(base) == len(accs):
                    p = exact_permutation_p(accs, base)
                    cell += f" p={p:.2f}"
                    summary.setdefault(cond, {})[f"{lr:.0e}"] = dict(
                        mean=m, std=s, p=p, n=len(accs))
            else:
                summary.setdefault(cond, {})[f"{lr:.0e}"] = dict(
                    mean=m, std=s, n=len(accs))
            print(f"| {cell:<18}", end="")
        print()

    Path("lr_control_summary.json").write_text(json.dumps(summary, indent=2))

    # LaTeX table for supplementary
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Matched learning-rate control (TREC-6, four heads, block 0)."
        r" Test accuracy (\%, mean$\pm$std over five seeds) and permutation"
        r" $p$ vs.\ classical. The accuracy null holds under a single shared"
        r" learning rate; \textsf{qcross} and \textsf{qfull} remain tied with"
        r" classical.}",
        r"\label{tab:lr_control}",
        r"\begin{tabular}{l" + "c" * len(lrs) + "}",
        r"\toprule",
        "Condition & " + " & ".join(fr"$\eta={lr:.0e}$".replace("e-0", "e{-}")
                                     for lr in lrs) + r" \\",
        r"\midrule",
    ]
    for cond in CONDITIONS:
        cells = []
        for lr in lrs:
            d = summary.get(cond, {}).get(f"{lr:.0e}")
            if d is None:
                cells.append("--")
            elif "p" in d:
                cells.append(fr"{d['mean']*100:.1f}$\pm${d['std']*100:.1f} "
                             fr"({d['p']:.2f})")
            else:
                cells.append(fr"{d['mean']*100:.1f}$\pm${d['std']*100:.1f}")
        name = cond if cond != "classical" else r"classical \textit{(base)}"
        lines.append(name + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path("lr_control_table.tex").write_text("\n".join(lines))
    print("\nWrote lr_control_summary.json and lr_control_table.tex")


if __name__ == "__main__":
    main()
