"""
lr_control.py
Matched learning-rate robustness control for the quantum-scored attention head.

Design
------
qatt_agnews.py's primary ablation trains classical weights at cfg["lr"]=2e-3
and circuit angles at cfg["qlr"]=1e-2 -- two Adam parameter groups. This
control re-runs the same primary configuration (AG News, 4 heads, block 0)
with BOTH groups sharing a single learning rate, swept over
{5e-4, 1e-3, 2e-3}, and checks that qcross/qfull stay statistically tied
with classical -- i.e. the accuracy null is not an artifact of the
asymmetric rate.

This file imports qatt_agnews.py directly and reuses its TextTransformer,
CONDITIONS, batchify, load_agnews, build_vocab, and accuracy functions
verbatim, plus its module-level constants (DEFAULT_N, VAL_SIZE, MIN_FREQ,
N_CLASSES, MAX_LEN, EPOCHS) so nothing is hardcoded that could drift out of
sync with the primary script. The data split replicates main()'s fixed
permutation seed (12345) exactly, so every condition/seed/lr run here sees
the identical train/val/test split the primary ablation used.

The ONLY functional difference from qatt_agnews.run_one is the optimizer:
one Adam group at a shared `lr` instead of two groups at (lr, qlr).
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import qatt_agnews as qa  # must run from the same directory as qatt_agnews.py


CONDITIONS = ["classical", "qnone", "qintra", "qcross", "qfull"]
SEEDS = [0, 1, 2, 3, 4]
MATCHED_LRS = [5e-4, 1e-3, 2e-3]

# Pulled from qatt_agnews.py's module-level defaults rather than hardcoded,
# so this control can't silently drift from the primary ablation's config.
PRIMARY_CFG = dict(
    device="cuda" if torch.cuda.is_available() else "cpu",
    d_model=64,
    n_heads=4,
    max_len=qa.MAX_LEN,
    n_qubits=4,
    quantum_in_block=0,   # primary configuration: quantum head in block 0
    epochs=qa.EPOCHS,
    batch_size=64,
)


def build_data_and_vocab(data_dir, limit_train=None):
    """
    Exact replication of qatt_agnews.py main()'s data/vocab construction:
    fixed permutation seed 12345, a DEFAULT_N-example (or overridden) train
    subset, VAL_SIZE held out, vocab built from train_fit only at MIN_FREQ,
    N_CLASSES from the module. Guarantees this control sees byte-identical
    data to the primary ablation.
    """
    train_all, test_all = qa.load_agnews(data_dir)
    g = torch.Generator().manual_seed(12345)  # matches qatt_agnews.py main()
    perm = torch.randperm(len(train_all), generator=g).tolist()
    n_train = limit_train if limit_train is not None else qa.DEFAULT_N
    tr = [train_all[i] for i in perm][: n_train + qa.VAL_SIZE]
    val = tr[: qa.VAL_SIZE]
    train_fit = tr[qa.VAL_SIZE:]
    vocab = qa.build_vocab(train_fit, qa.MIN_FREQ)
    data = (train_fit, val, test_all)
    return data, vocab, qa.N_CLASSES


def run_one_matched_lr(cond, seed, data, vocab, n_classes, cfg, lr):
    """
    Copy of qatt_agnews.run_one with a single change: one Adam parameter
    group at a shared `lr` instead of two groups at (cfg['lr'], cfg['qlr']).
    Model construction, batching, training loop, epoch count, and
    validation-based checkpoint selection are all identical to run_one.
    """
    train_fit, val, test = data
    cc = qa.CONDITIONS[cond]
    device = cfg["device"]
    torch.manual_seed(seed)

    model = qa.TextTransformer(
        vocab_size=len(vocab), n_classes=n_classes,
        d_model=cfg["d_model"], n_heads=cfg["n_heads"], max_len=cfg["max_len"],
        use_quantum=cc["use_quantum"], entangle=cc["entangle"],
        n_qubits=cfg["n_qubits"], quantum_in_block=cfg.get("quantum_in_block", 0),
    ).to(device)

    qp = [p for n, p in model.named_parameters() if "qweights" in n]
    cp = [p for n, p in model.named_parameters() if "qweights" not in n]
    if cond != "classical" and len(qp) == 0:
        raise RuntimeError(
            f"expected circuit params ('qweights') for condition={cond}, found "
            f"none. Model construction may not match qatt_agnews.py's run_one."
        )

    # The only real change vs qatt_agnews.run_one: one shared rate.
    opt = torch.optim.Adam(cp + qp, lr=lr)

    val_b = qa.batchify(val, vocab, cfg["max_len"], 128, False, 0, device)
    test_b = qa.batchify(test, vocab, cfg["max_len"], 128, False, 0, device)

    best_val, test_at_best, best_ep = -1.0, 0.0, 0
    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        for ids, m, y in qa.batchify(
            train_fit, vocab, cfg["max_len"], cfg["batch_size"],
            True, 1000 * seed + ep, device
        ):
            opt.zero_grad()
            F.cross_entropy(model(ids, m), y).backward()
            opt.step()
        va = qa.accuracy(model, val_b)
        if va > best_val:
            best_val, test_at_best, best_ep = va, qa.accuracy(model, test_b), ep
        print(f"      [{cond} seed{seed} lr{lr:.0e}] epoch {ep:2d}  "
              f"val={va:.4f}  best={best_val:.4f}", flush=True)

    return dict(test_acc=test_at_best, val_acc=best_val, best_epoch=best_ep,
                n_circuit_params=len(qp))


def run_single(condition, seed, lr, cfg, out_dir, data, vocab, n_classes):
    rec = run_one_matched_lr(condition, seed, data, vocab, n_classes, cfg, lr)
    rec.update(condition=condition, seed=seed, lr=lr,
               n_heads=cfg["n_heads"], n_qubits=cfg["n_qubits"],
               quantum_in_block=cfg["quantum_in_block"], epochs=cfg["epochs"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"{condition}_lr{lr:.0e}_seed{seed}.json"
    fname.write_text(json.dumps(rec))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=CONDITIONS, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--out", default="results/lr")
    ap.add_argument("--data_dir", type=str, default="./agnews",
                     help="same default as qatt_agnews.py")
    ap.add_argument("--limit_train", type=int, default=None,
                     help="override training subset size; default matches "
                          "qatt_agnews.py's DEFAULT_N")
    args = ap.parse_args()

    # Diagnostic: confirms the imported module constants match what the
    # primary ablation actually used, visible in every job's stdout.
    print(f"[lr_control cfg-check] DEFAULT_N={qa.DEFAULT_N} VAL_SIZE={qa.VAL_SIZE} "
          f"MIN_FREQ={qa.MIN_FREQ} N_CLASSES={qa.N_CLASSES} MAX_LEN={qa.MAX_LEN} "
          f"EPOCHS={qa.EPOCHS}", flush=True)

    data, vocab, n_classes = build_data_and_vocab(args.data_dir, args.limit_train)
    print(f"[lr_control data] train_fit={len(data[0])} val={len(data[1])} "
          f"test={len(data[2])} vocab={len(vocab)}", flush=True)

    run_single(args.condition, args.seed, args.lr, PRIMARY_CFG, args.out,
               data=data, vocab=vocab, n_classes=n_classes)


if __name__ == "__main__":
    main()
