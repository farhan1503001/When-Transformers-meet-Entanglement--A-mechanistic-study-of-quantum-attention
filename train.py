"""
train.py — Train the quantum (or classical-baseline) text transformer on TREC-6.

Key diagnostics printed every epoch:
  * train loss / train acc / test acc
  * gradient norm of QUANTUM weights vs CLASSICAL weights, separately
    (this is the early-warning system for barren plateaus / dead gradients)
  * mean attention entropy of the quantum head vs a classical head
    (this is also one of your paper's headline measurements)

CLI examples
------------
  # Quantum model, full data:
  python train.py --model quantum --epochs 30

  # Classical baseline (same architecture, head 0 is dot-product instead):
  python train.py --model classical --epochs 30

  # Entanglement ablation (drop the cross q<->k CZ gates):
  python train.py --model quantum --entangle cross --epochs 30  # or none|intra|full

  # Data-efficiency point:
  python train.py --model quantum --limit-train 500 --epochs 40
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

from data import make_loaders
from model import TextTransformer, count_parameters, attention_entropy


def _check(msg):
    print(f"[check][train] {msg}", flush=True)


def grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().norm().item() ** 2
    return total ** 0.5


def split_param_groups(model, base_lr, q_lr):
    """Quantum circuit angles get their own (usually larger) LR, because
    variational gradients are tiny — a lesson learned the hard way."""
    quantum_params, classical_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "qhead.qweights" in name:
            quantum_params.append(p)
        else:
            classical_params.append(p)
    groups = [{"params": classical_params, "lr": base_lr}]
    if quantum_params:
        groups.append({"params": quantum_params, "lr": q_lr})
    _check(f"optimizer groups: classical={sum(p.numel() for p in classical_params)} "
           f"params @ lr={base_lr} | quantum={sum(p.numel() for p in quantum_params)} "
           f"params @ lr={q_lr}")
    return groups, quantum_params, classical_params


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for ids, mask, y in loader:
        ids, mask, y = ids.to(device), mask.to(device), y.to(device)
        logits = model(ids, mask)
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def measure_entropy(model, loader, device, max_batches=4):
    """Average attention entropy for quantum + classical heads over a few batches."""
    model.eval()
    q_ents, c_ents = [], []
    for bi, (ids, mask, y) in enumerate(loader):
        if bi >= max_batches:
            break
        ids, mask = ids.to(device), mask.to(device)
        _, attns = model(ids, mask, return_attn=True)
        b0 = attns.get("block0", {})
        km = mask[:, : ids.shape[1]].unsqueeze(1)
        if "quantum" in b0:
            q_ents.append(attention_entropy(b0["quantum"], km))
        if "classical" in b0:
            c_ents.append(attention_entropy(b0["classical"], km))
    q = sum(q_ents) / len(q_ents) if q_ents else float("nan")
    c = sum(c_ents) / len(c_ents) if c_ents else float("nan")
    return q, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["quantum", "classical"], default="quantum")
    ap.add_argument("--entangle", choices=["none", "intra", "cross", "full"],
                    default="full",
                    help="quantum two-qubit gate set: none | intra | cross | full")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--qlr", type=float, default=1e-2)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=40)
    ap.add_argument("--limit-train", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto",
                    help="auto | cpu | cuda. CPU is recommended: the 4-qubit "
                         "sim is CPU-bound, so GPU adds transfer overhead.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = ("cuda" if torch.cuda.is_available() else "cpu") \
        if args.device == "auto" else args.device
    _check(f"device = {device}  | config = {vars(args)}")

    train_loader, test_loader, vocab, label_names = make_loaders(
        batch_size=args.batch_size, max_len=args.max_len,
        limit_train=args.limit_train, seed=args.seed)

    use_quantum = args.model == "quantum"
    model = TextTransformer(
        vocab_size=len(vocab), n_classes=len(label_names),
        d_model=args.d_model, n_heads=args.n_heads, max_len=args.max_len,
        use_quantum=use_quantum, entangle=args.entangle,
    ).to(device)

    _check(f"built model: type={args.model}, entangle={args.entangle}")
    count_parameters(model)

    # Verify gradients flow BEFORE committing to a long run (catches dead-circuit).
    if use_quantum:
        _check("running pre-training gradient sanity check on quantum head...")
        model.blocks[0].attn.qhead.self_test(device=device)

    groups, q_params, c_params = split_param_groups(model, args.lr, args.qlr)
    opt = torch.optim.Adam(groups)

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        last_qg, last_cg = 0.0, 0.0
        for ids, mask, y in train_loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            opt.zero_grad()
            logits = model(ids, mask)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            # Capture gradient norms right after backward, before step.
            last_qg = grad_norm(q_params) if q_params else 0.0
            last_cg = grad_norm(c_params)
            opt.step()
            running += loss.item() * y.numel()
            n += y.numel()

        train_loss = running / max(n, 1)
        train_acc = evaluate(model, train_loader, device)
        test_acc = evaluate(model, test_loader, device)
        q_ent, c_ent = measure_entropy(model, test_loader, device)
        best = max(best, test_acc)

        # The quantum grad norm is the canary. If it sits near 0, the circuit
        # is not learning — stop and rethink rather than waiting out the run.
        qflag = ""
        if use_quantum and last_qg < 1e-6:
            qflag = "  <-- WARNING: quantum grad ~0 (possible plateau/dead head)"

        print(
            f"epoch {epoch:3d} | loss {train_loss:.4f} | "
            f"train_acc {train_acc:.4f} | test_acc {test_acc:.4f} | "
            f"grad[q]={last_qg:.2e} grad[c]={last_cg:.2e} | "
            f"H_q={q_ent:.3f} H_c={c_ent:.3f}{qflag}",
            flush=True,
        )

    _check(f"DONE. best test_acc = {best:.4f}  "
           f"(uniform-random baseline = {1.0/len(label_names):.4f})")


if __name__ == "__main__":
    main()
