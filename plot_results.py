"""
plot_results.py — Turn results.csv (from experiments.py) into two paper figures.

  Figure 1  ablation_accuracy.{png,pdf}
      Mean test accuracy per condition with error bars. Bars are colour-coded by
      the hypothesis: classical (grey), separable-quantum none/intra (light),
      entangled-quantum cross/full (dark). A significance bracket annotates the
      qfull-vs-classical permutation-test p-value.

  Figure 2  attention_entropy.{png,pdf}
      Per quantum condition, mean attention entropy of the quantum head (H_q)
      vs a classical head in the SAME model (H_c). Shows whether the quantum
      head attends more sharply / diffusely than classical attention.

Both are saved as PNG (300 dpi, for slides) and PDF (vector, for LaTeX).

Usage
-----
  python plot_results.py --csv results.csv --out-dir figures
  python plot_results.py --csv results.csv --err sem    # SEM instead of std
"""

import argparse, csv, math, os, itertools
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["classical", "qnone", "qintra", "qcross", "qfull"]
PRETTY = {"classical": "classical", "qnone": "quantum\nnone",
          "qintra": "quantum\nintra", "qcross": "quantum\ncross",
          "qfull": "quantum\nfull"}
# colour by hypothesis group
COLOR = {"classical": "#9aa0a6",
         "qnone": "#a8d0e6", "qintra": "#a8d0e6",     # separable -> light
         "qcross": "#1b6ca8", "qfull": "#0a3d62"}     # entangled -> dark
RANDOM = 1.0 / 6.0


def load(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    by_cond = defaultdict(lambda: defaultdict(list))
    for r in rows:
        c = r["condition"]
        by_cond[c]["test"].append(float(r["test_acc"]))
        for k in ("H_quantum", "H_classical"):
            try:
                v = float(r[k])
            except (ValueError, KeyError):
                v = float("nan")
            by_cond[c][k].append(v)
    return by_cond


def mean_std(xs):
    xs = [x for x in xs if not math.isnan(x)]
    n = len(xs)
    if n == 0:
        return float("nan"), 0.0, 0
    mu = sum(xs) / n
    sd = (sum((x - mu) ** 2 for x in xs) / (n - 1)) ** 0.5 if n > 1 else 0.0
    return mu, sd, n


def err_of(sd, n, kind):
    return sd / math.sqrt(n) if (kind == "sem" and n > 0) else sd


def permutation_p(a, b, max_exact=200000):
    a = [x for x in a if not math.isnan(x)]
    b = [x for x in b if not math.isnan(x)]
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    pooled = a + b
    obs = abs(sum(a) / na - sum(b) / nb)
    total = math.comb(na + nb, na)
    if total > max_exact:
        return None
    cnt = 0
    for combo in itertools.combinations(range(na + nb), na):
        s = set(combo)
        ga = [pooled[i] for i in combo]
        gb = [pooled[i] for i in range(na + nb) if i not in s]
        if abs(sum(ga) / na - sum(gb) / nb) >= obs - 1e-12:
            cnt += 1
    return cnt / total


def style():
    plt.rcParams.update({
        "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
        "figure.dpi": 110,
    })


def fig_accuracy(by_cond, err_kind, out_dir, dset="TREC-6"):
    conds = [c for c in ORDER if c in by_cond]
    means, errs, colors, labels = [], [], [], []
    for c in conds:
        mu, sd, n = mean_std(by_cond[c]["test"])
        means.append(mu)
        errs.append(err_of(sd, n, err_kind))
        colors.append(COLOR.get(c, "#888"))
        labels.append(PRETTY.get(c, c))

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = range(len(conds))
    bars = ax.bar(x, means, yerr=errs, capsize=5, color=colors,
                  edgecolor="black", linewidth=0.6, error_kw={"elinewidth": 1.1})
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("test accuracy")
    n_seeds = mean_std(by_cond[conds[0]]["test"])[2]
    ax.set_title(f"{dset} ablation (mean \u00B1 {err_kind.upper()}, n={n_seeds} seeds)")

    lo = min(m - e for m, e in zip(means, errs))
    hi = max(m + e for m, e in zip(means, errs))
    pad = (hi - lo) * 0.5 + 0.005
    ax.set_ylim(lo - pad, hi + pad * 1.6)

    for xi, m, e in zip(x, means, errs):
        ax.text(xi, m + e + (hi - lo) * 0.06, f"{m:.3f}",
                ha="center", va="bottom", fontsize=9)

    ax.text(0.99, 0.02, f"random baseline = {RANDOM:.3f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, style="italic", color="#666")

    # significance bracket: qfull vs classical
    if "qfull" in by_cond and "classical" in by_cond:
        p = permutation_p(by_cond["qfull"]["test"], by_cond["classical"]["test"])
        if p is not None:
            i, j = conds.index("classical"), conds.index("qfull")
            ytop = hi + pad * 0.9
            ax.plot([i, i, j, j],
                    [ytop, ytop + pad * 0.15, ytop + pad * 0.15, ytop],
                    color="black", linewidth=1.0)
            tag = f"p = {p:.3f}" + (" *" if p < 0.05 else " (n.s.)")
            ax.text((i + j) / 2, ytop + pad * 0.18, tag,
                    ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"ablation_accuracy.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)
    return p if ("qfull" in by_cond and "classical" in by_cond) else None


def fig_entropy(by_cond, err_kind, out_dir):
    qconds = [c for c in ["qnone", "qintra", "qcross", "qfull"] if c in by_cond]
    if not qconds:
        return
    hq_m, hq_e, hc_m, hc_e = [], [], [], []
    for c in qconds:
        mu, sd, n = mean_std(by_cond[c]["H_quantum"])
        hq_m.append(mu); hq_e.append(err_of(sd, n, err_kind))
        mu, sd, n = mean_std(by_cond[c]["H_classical"])
        hc_m.append(mu); hc_e.append(err_of(sd, n, err_kind))

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    x = list(range(len(qconds)))
    w = 0.38
    ax.bar([xi - w / 2 for xi in x], hq_m, w, yerr=hq_e, capsize=4,
           label="quantum head (H$_q$)", color="#0a3d62",
           edgecolor="black", linewidth=0.6)
    ax.bar([xi + w / 2 for xi in x], hc_m, w, yerr=hc_e, capsize=4,
           label="classical head (H$_c$)", color="#9aa0a6",
           edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[c].replace("quantum\n", "") for c in qconds])
    ax.set_xlabel("entanglement mode")
    ax.set_ylabel("mean attention entropy (nats)")
    ax.set_title("Attention entropy: quantum vs classical head (same model)")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"attention_entropy.{ext}"),
                    dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results.csv")
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--err", choices=["std", "sem"], default="std")
    ap.add_argument("--title", default="TREC-6",
                    help="dataset name shown in the figure title, e.g. SST-2")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    style()
    by_cond = load(args.csv)

    print("condition    mean_test    +/-    n")
    for c in ORDER:
        if c in by_cond:
            mu, sd, n = mean_std(by_cond[c]["test"])
            print(f"  {c:10} {mu:.4f}   {err_of(sd, n, args.err):.4f}  {n}")

    p = fig_accuracy(by_cond, args.err, args.out_dir, args.title)
    fig_entropy(by_cond, args.err, args.out_dir)
    if p is not None:
        print(f"\nqfull vs classical permutation p = {p:.4f}"
              f"{'  (significant)' if p < 0.05 else '  (not significant)'}")
    print(f"\nSaved figures to {args.out_dir}/ "
          f"(ablation_accuracy.png/pdf, attention_entropy.png/pdf)")


if __name__ == "__main__":
    main()
