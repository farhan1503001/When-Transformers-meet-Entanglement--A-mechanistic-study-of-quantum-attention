"""
model.py — Small Transformer with ONE quantum-scored attention head.

The whole scientific point lives in `QuantumAttentionHead`:
  classical attention score:  s(i,j) = (q_i . k_j) / sqrt(d)
  quantum   attention score:  s(i,j) = <psi(q_i, k_j)| Z_0 Z_2 |psi(q_i,k_j)>

Everything else (embeddings, FFN, other 3 heads, the second block) is standard
and IDENTICAL between the classical baseline and the quantum model, so any
measured difference is attributable to that one head.

Diagnostics ([check] prints) are gated by debug=True on forward(), plus helper
functions you can call any time:
  * count_parameters()        -> breakdown incl. the quantum weights specifically
  * attention_entropy()       -> mean entropy of an attention map (uniform vs peaked)
  * QuantumAttentionHead.self_test() -> verifies the circuit runs & gradients flow
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml


def _check(msg):
    print(f"[check][model] {msg}", flush=True)


# ===========================================================================
# Quantum scoring circuit
# ===========================================================================
def build_quantum_score_qnode(n_qubits=4, entangle="full"):
    """4 qubits: wires 0,1 carry the query angles; wires 2,3 carry the key angles.
    Returns a torch-differentiable qnode mapping (inputs[...,4], weights[2,nq,3]) -> scalar.

    `entangle` selects the two-qubit gate set (the ablation knob):
        "none"  -> no CZ gates             (score provably separable: <Z0Z2>=<Z0><Z2>)
        "intra" -> CZ(0,1), CZ(2,3)        (entangle WITHIN query and WITHIN key only)
        "cross" -> CZ(0,2), CZ(1,3)        (entangle query<->key: the genuine interaction)
        "full"  -> all four CZ gates

    IMPORTANT design point (learned via a bug): the readout <Z0 (x) Z2> is DIAGONAL
    in the Z basis, and CZ gates are also diagonal there -- so CZ gates placed right
    before a Z-measurement are INVISIBLE. We therefore apply a SECOND rotation layer
    AFTER the entangler, which rotates the entangled phases into measurable
    populations. weights has shape (2, n_qubits, 3): layer 0 before CZ, layer 1 after."""
    assert n_qubits == 4, "QuantumAttentionHead is designed for 4 qubits (2 query, 2 key)"
    assert entangle in ("none", "intra", "cross", "full"), f"bad entangle mode: {entangle}"
    cross = [(0, 2), (1, 3)]   # query<->key links: the genuine q-k interaction
    intra = [(0, 1), (2, 3)]   # within-query / within-key links
    cz_pairs = {"none": [], "intra": intra,
                "cross": cross, "full": cross + intra}[entangle]
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(inputs, weights):
        # Encode 4 angles (2 from q, 2 from k) as RY rotations.
        qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        # Rotation layer 0 (before entanglement).
        for w in range(n_qubits):
            qml.RX(weights[0, w, 0], wires=w)
            qml.RY(weights[0, w, 1], wires=w)
            qml.RZ(weights[0, w, 2], wires=w)
        for (a, b) in cz_pairs:
            qml.CZ(wires=[a, b])
        # Rotation layer 1 (AFTER entanglement) -- makes entanglement visible to
        # the Z-basis readout. Without this, the CZ gates have ZERO effect.
        for w in range(n_qubits):
            qml.RX(weights[1, w, 0], wires=w)
            qml.RY(weights[1, w, 1], wires=w)
            qml.RZ(weights[1, w, 2], wires=w)
        # Readout: correlation between one query qubit and one key qubit.
        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    return circuit


class QuantumAttentionHead(nn.Module):
    def __init__(self, head_dim, n_qubits=4, entangle="full"):
        super().__init__()
        self.head_dim = head_dim
        self.n_qubits = n_qubits
        self.entangle = entangle
        # Project each head's q,k vector (head_dim) down to 2 angles.
        self.q_proj = nn.Linear(head_dim, 2)
        self.k_proj = nn.Linear(head_dim, 2)
        # Variational weights as a plain nn.Parameter (more robust than TorchLayer).
        # Shape (2, n_qubits, 3): two rotation layers (before + after entangler).
        self.qweights = nn.Parameter(0.1 * torch.randn(2, n_qubits, 3))
        # Learnable temperature: ZZ expval is bounded in [-1,1], too flat for a
        # sharp softmax. This scale lets the head learn how peaked to be.
        self.scale = nn.Parameter(torch.tensor(4.0))
        self.circuit = build_quantum_score_qnode(n_qubits, entangle)

    def forward(self, q, k, v, key_mask=None, debug=False):
        # q, k, v: (B, L, head_dim).  key_mask: (B, 1, L) with 1=keep, 0=pad.
        B, L, D = q.shape

        # Angles in [-pi, pi] via tanh; .float() guards the float64/32 trap.
        qa = torch.tanh(self.q_proj(q)) * math.pi          # (B, L, 2)
        ka = torch.tanh(self.k_proj(k)) * math.pi          # (B, L, 2)

        # Build all (i,j) query-key pairs: shape (B, L, L, 4).
        qa_exp = qa.unsqueeze(2).expand(B, L, L, 2)        # query index i
        ka_exp = ka.unsqueeze(1).expand(B, L, L, 2)        # key   index j
        pairs = torch.cat([qa_exp, ka_exp], dim=-1)        # (B, L, L, 4)
        flat = pairs.reshape(-1, 4).float()                # (B*L*L, 4)

        # ONE batched circuit call for the whole batch of pairs.
        # default.qubit simulates its statevector on CPU, so if the model is on
        # CUDA we hand the circuit CPU tensors and move the scalar scores back.
        # .cpu()/.to(device) stay in the autograd graph, so gradients still
        # reach the (possibly GPU-resident) projections and qweights. The dtype
        # cast also dodges the float64/float32 "expected Double" trap.
        scores = self.circuit(flat.cpu(), self.qweights.cpu())   # CPU, float64
        scores = scores.reshape(B, L, L).to(device=v.device, dtype=v.dtype)

        if debug:
            with torch.no_grad():
                _check(f"[qhead] raw ZZ score range: "
                       f"[{scores.min():.3f}, {scores.max():.3f}] "
                       f"mean={scores.mean():.3f} std={scores.std():.3f} "
                       f"(should NOT be all ~constant)")

        scores = scores * self.scale
        if key_mask is not None:
            scores = scores.masked_fill(key_mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)                   # (B, L, L)
        out = torch.matmul(attn, v)                        # (B, L, head_dim)
        return out, attn

    @torch.no_grad()
    def _frozen_score_stats(self, head_dim=None):
        pass

    def self_test(self, head_dim=None, batch=2, length=3, device="cpu"):
        """Verify the circuit runs AND gradients reach the quantum weights.
        This is the single most important check given past dead-gradient bugs."""
        head_dim = head_dim or self.head_dim
        x = torch.randn(batch, length, head_dim, device=device, requires_grad=False)
        self.to(device)
        out, attn = self.forward(x, x, x, key_mask=None, debug=True)
        loss = out.sum()
        loss.backward()
        gw = self.qweights.grad
        gnorm = None if gw is None else gw.norm().item()
        _check(f"[self_test] out shape={tuple(out.shape)}, attn shape={tuple(attn.shape)}")
        _check(f"[self_test] attn rows sum to 1? "
               f"{torch.allclose(attn.sum(-1), torch.ones_like(attn.sum(-1)), atol=1e-4)}")
        _check(f"[self_test] quantum-weight grad norm = {gnorm} "
               f"({'OK, gradients flow' if gnorm and gnorm > 1e-8 else 'DEAD GRADIENT!'})")
        self.zero_grad()
        return gnorm


# ===========================================================================
# Mixed multi-head attention: head 0 quantum (optional), heads 1..H-1 classical
# ===========================================================================
class MixedMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, use_quantum=True, entangle="full", n_qubits=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model, self.n_heads = d_model, n_heads
        self.head_dim = d_model // n_heads
        self.use_quantum = use_quantum
        self.q_lin = nn.Linear(d_model, d_model)
        self.k_lin = nn.Linear(d_model, d_model)
        self.v_lin = nn.Linear(d_model, d_model)
        self.out_lin = nn.Linear(d_model, d_model)
        if use_quantum:
            self.qhead = QuantumAttentionHead(self.head_dim, n_qubits, entangle)

    def forward(self, x, key_mask=None, return_attn=False, debug=False):
        B, L, _ = x.shape
        H, Dh = self.n_heads, self.head_dim
        q = self.q_lin(x).view(B, L, H, Dh).transpose(1, 2)   # (B,H,L,Dh)
        k = self.k_lin(x).view(B, L, H, Dh).transpose(1, 2)
        v = self.v_lin(x).view(B, L, H, Dh).transpose(1, 2)

        outs, attns = [], {}
        for h in range(H):
            if self.use_quantum and h == 0:
                o, a = self.qhead(q[:, h], k[:, h], v[:, h],
                                  key_mask=key_mask, debug=debug)
                attns["quantum"] = a
            else:
                scores = torch.matmul(q[:, h], k[:, h].transpose(-2, -1)) / math.sqrt(Dh)
                if key_mask is not None:
                    scores = scores.masked_fill(key_mask == 0, float("-inf"))
                a = F.softmax(scores, dim=-1)
                o = torch.matmul(a, v[:, h])
                if h == 1:
                    attns["classical"] = a   # keep one classical head for comparison
            outs.append(o)

        out = torch.stack(outs, dim=1).transpose(1, 2).reshape(B, L, H * Dh)
        out = self.out_lin(out)
        if return_attn:
            return out, attns
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, use_quantum=False,
                 entangle="full", n_qubits=4, dropout=0.1):
        super().__init__()
        self.attn = MixedMultiHeadAttention(d_model, n_heads, use_quantum,
                                            entangle, n_qubits)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.GELU(),
            nn.Linear(ff_dim, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x, key_mask=None, return_attn=False, debug=False):
        if return_attn:
            a_out, attns = self.attn(x, key_mask, return_attn=True, debug=debug)
        else:
            a_out, attns = self.attn(x, key_mask, debug=debug), None
        x = self.norm1(x + self.drop(a_out))
        x = self.norm2(x + self.drop(self.ff(x)))
        if return_attn:
            return x, attns
        return x


class TextTransformer(nn.Module):
    def __init__(self, vocab_size, n_classes, d_model=64, n_heads=4, ff_dim=128,
                 max_len=40, use_quantum=True, entangle="full", n_qubits=4,
                 quantum_in_block=0, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.max_len = max_len
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, n_heads, ff_dim,
                use_quantum=(use_quantum and b == quantum_in_block),
                entangle=entangle, n_qubits=n_qubits, dropout=dropout)
            for b in range(2)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, input_ids, attn_mask, return_attn=False, debug=False):
        B, L = input_ids.shape
        L = min(L, self.max_len)
        input_ids = input_ids[:, :L]
        attn_mask = attn_mask[:, :L]
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        key_mask = attn_mask.unsqueeze(1)  # (B,1,L) for masking padded keys

        collected = {}
        for bi, blk in enumerate(self.blocks):
            if return_attn:
                x, attns = blk(x, key_mask, return_attn=True,
                               debug=(debug and bi == 0))
                if attns:
                    collected[f"block{bi}"] = attns
            else:
                x = blk(x, key_mask, debug=(debug and bi == 0))

        x = self.norm(x)
        # Masked mean pooling over valid tokens (ignore padding).
        m = attn_mask.unsqueeze(-1).float()
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        logits = self.head(pooled)
        if return_attn:
            return logits, collected
        return logits


# ===========================================================================
# Diagnostics helpers
# ===========================================================================
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    quantum = 0
    for name, p in model.named_parameters():
        if "qhead.qweights" in name and p.requires_grad:
            quantum += p.numel()
    qproj = sum(p.numel() for n, p in model.named_parameters()
                if "qhead." in n and "qweights" not in n and p.requires_grad)
    _check(f"total trainable params = {total}")
    _check(f"  of which quantum circuit weights (qweights) = {quantum}")
    _check(f"  of which quantum head projections+scale     = {qproj}")
    return {"total": total, "quantum_weights": quantum, "quantum_proj": qproj}


def attention_entropy(attn, key_mask=None, eps=1e-9):
    """Mean Shannon entropy (nats) of attention rows. Higher = more uniform/diffuse,
    lower = more peaked/concentrated. attn: (B, L, L)."""
    p = attn.clamp(min=eps)
    ent = -(p * p.log()).sum(-1)        # (B, L) entropy per query row
    if key_mask is not None:
        valid = key_mask.squeeze(1).float()   # (B, L) over query positions
        ent = (ent * valid).sum() / valid.sum().clamp(min=1.0)
    else:
        ent = ent.mean()
    return ent.item()


if __name__ == "__main__":
    # Standalone smoke test of the model + the all-important gradient check.
    torch.manual_seed(0)
    print("=== QuantumAttentionHead self-test ===")
    head = QuantumAttentionHead(head_dim=16, n_qubits=4, entangle="full")
    head.self_test()

    print("\n=== Full model forward/backward ===")
    model = TextTransformer(vocab_size=500, n_classes=6, max_len=40)
    count_parameters(model)
    ids = torch.randint(0, 500, (4, 12))
    mask = torch.ones(4, 12, dtype=torch.long)
    mask[0, 8:] = 0  # simulate padding
    logits, attns = model(ids, mask, return_attn=True, debug=True)
    _check(f"logits shape = {tuple(logits.shape)} (expect (4,6))")
    qa = attns["block0"]["quantum"]
    ca = attns["block0"]["classical"]
    _check(f"quantum-head attention entropy   = {attention_entropy(qa, mask.unsqueeze(1)):.4f} nats")
    _check(f"classical-head attention entropy = {attention_entropy(ca, mask.unsqueeze(1)):.4f} nats")
    loss = F.cross_entropy(logits, torch.randint(0, 6, (4,)))
    loss.backward()
    g = model.blocks[0].attn.qhead.qweights.grad
    _check(f"end-to-end quantum grad norm = {g.norm().item():.3e} "
           f"({'OK' if g.norm() > 1e-8 else 'DEAD'})")
