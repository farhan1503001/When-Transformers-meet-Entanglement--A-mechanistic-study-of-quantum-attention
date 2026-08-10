"""
data.py — TREC-6 loader for the Quantum Attention project.

Loading priority:
  1. LOCAL .label files at <data_dir>/train.label and <data_dir>/test.label
     (the robust path: download once from GitHub, no HuggingFace version issues).
  2. HuggingFace `datasets` ("CogComp/trec") as a fallback if local files absent.

To get the local files (PowerShell), run once:
  mkdir trec
  curl.exe -L "https://raw.githubusercontent.com/honnibal/dsr16_nlp/master/data/question_classification/train_5500.label" -o trec\train.label
  curl.exe -L "https://raw.githubusercontent.com/justindomingue/nlp/master/question_classification/data/TREC_10.label" -o trec\test.label

TREC-6 coarse classes (canonical order):
    0 ABBR  1 ENTY  2 DESC  3 HUM  4 LOC  5 NUM
"""

import os
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader

PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1
COARSE = ["ABBR", "ENTY", "DESC", "HUM", "LOC", "NUM"]
C2I = {c: i for i, c in enumerate(COARSE)}
DATA_DIR = "trec"


def _check(msg):
    print(f"[check][data] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _read_label_file(path):
    """Parse a TREC .label file: 'COARSE:fine question text' per line.
    Files are Latin-1 encoded; one stray byte (0xf0) appears in the corpus."""
    ex = []
    with open(path, "rb") as f:
        for line in f:
            line = line.replace(b"\xf0", b" ").strip().decode("latin-1")
            if not line:
                continue
            lab, _, text = line.partition(" ")
            coarse = lab.split(":")[0]
            if coarse in C2I and text:
                ex.append({"text": text, "label": C2I[coarse]})
    return ex


def _load_local(data_dir):
    tr = os.path.join(data_dir, "train.label")
    te = os.path.join(data_dir, "test.label")
    if os.path.exists(tr) and os.path.exists(te):
        train, test = _read_label_file(tr), _read_label_file(te)
        _check(f"loaded LOCAL .label files from '{data_dir}/': "
               f"train={len(train)} test={len(test)}")
        return train, test
    return None


def _load_hf():
    from datasets import load_dataset
    ds = load_dataset("CogComp/trec")
    cols = ds["train"].column_names
    lab = "coarse_label" if "coarse_label" in cols else "label-coarse"
    txt = "text" if "text" in cols else cols[0]
    train = [{"text": r[txt], "label": int(r[lab])} for r in ds["train"]]
    test = [{"text": r[txt], "label": int(r[lab])} for r in ds["test"]]
    _check(f"loaded TREC from HuggingFace (CogComp/trec): "
           f"train={len(train)} test={len(test)}")
    return train, test


def load_trec(data_dir=DATA_DIR):
    """Returns (train_examples, test_examples, label_names)."""
    local = _load_local(data_dir)
    if local is not None:
        train, test = local
    else:
        _check(f"no local files in '{data_dir}/'; trying HuggingFace...")
        try:
            train, test = _load_hf()
        except Exception as e:
            raise RuntimeError(
                f"Could not load TREC. No local files at '{data_dir}/train.label' "
                f"and HuggingFace failed ({type(e).__name__}: {e}). "
                f"Download the .label files (see header of data.py)."
            )
    tr_dist = Counter(e["label"] for e in train)
    _check("train label distribution: "
           + ", ".join(f"{COARSE[i]}={tr_dist[i]}" for i in sorted(tr_dist)))
    return train, test, COARSE


# --------------------------------------------------------------------------- #
# Tokenizer / vocab
# --------------------------------------------------------------------------- #
def tokenize(text):
    return text.lower().strip().split()


def build_vocab(train_examples, min_freq=2):
    counter = Counter()
    for e in train_examples:
        counter.update(tokenize(e["text"]))
    vocab = {PAD: PAD_ID, UNK: UNK_ID}
    for tok, freq in counter.most_common():
        if freq >= min_freq:
            vocab[tok] = len(vocab)
    _check(f"vocab size = {len(vocab)} (min_freq={min_freq}, "
           f"unique tokens seen = {len(counter)})")
    return vocab


def encode(text, vocab, max_len):
    return [vocab.get(t, UNK_ID) for t in tokenize(text)][:max_len]


class TrecDataset(Dataset):
    def __init__(self, examples, vocab, max_len):
        self.examples = examples
        self.vocab = vocab
        self.max_len = max_len
        oov, total = 0, 0
        for e in examples:
            for t in tokenize(e["text"]):
                total += 1
                if t not in vocab:
                    oov += 1
        rate = 100.0 * oov / max(total, 1)
        _check(f"{len(examples)} examples | OOV rate = {rate:.2f}% "
               f"({oov}/{total} tokens map to <unk>)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        e = self.examples[idx]
        return encode(e["text"], self.vocab, self.max_len), e["label"]


def collate(batch):
    seqs, labels = zip(*batch)
    max_len = max(max(len(s) for s in seqs), 1)
    input_ids = torch.full((len(seqs), max_len), PAD_ID, dtype=torch.long)
    attn_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        if len(s) > 0:
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attn_mask[i, : len(s)] = 1
    return input_ids, attn_mask, torch.tensor(labels, dtype=torch.long)


def make_loaders(batch_size=32, max_len=40, min_freq=2, limit_train=None,
                 seed=0, data_dir=DATA_DIR):
    train, test, label_names = load_trec(data_dir)
    if limit_train is not None:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(train), generator=g)[:limit_train].tolist()
        train = [train[i] for i in perm]
        _check(f"limited train set to {len(train)} examples (seed={seed})")

    vocab = build_vocab(train, min_freq=min_freq)

    lens = sorted(len(tokenize(e["text"])) for e in train)
    p50 = lens[len(lens) // 2]
    p95 = lens[int(0.95 * len(lens))]
    _check(f"token lengths: median={p50}, p95={p95}, max={lens[-1]}; "
           f"max_len cap={max_len} "
           f"({100.0*sum(l>max_len for l in lens)/len(lens):.1f}% truncated)")

    inv = {v: k for k, v in vocab.items()}
    sample_ids = encode(train[0]["text"], vocab, max_len)
    _check(f"sample raw   : {train[0]['text']!r}")
    _check(f"sample decode: {' '.join(inv.get(i, '?') for i in sample_ids)!r} "
           f"-> label {label_names[train[0]['label']]}")

    train_ds = TrecDataset(train, vocab, max_len)
    test_ds = TrecDataset(test, vocab, max_len)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate, num_workers=0)

    xb, mb, yb = next(iter(train_loader))
    _check(f"first batch: input_ids={tuple(xb.shape)}, mask={tuple(mb.shape)}, "
           f"labels={tuple(yb.shape)}, label range=[{yb.min().item()},{yb.max().item()}]")
    return train_loader, test_loader, vocab, label_names


if __name__ == "__main__":
    make_loaders(batch_size=8, limit_train=200)
