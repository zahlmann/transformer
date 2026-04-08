"""Load streaming dataset (trained BPE, vocab 32k, multi-source mix)."""

import json
import os
import pickle
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def load_data(context_len, data_dir=None):
    """Load tokenized dataset. Returns train memmap + val splits."""
    from tokenizers import Tokenizer

    token_dir = data_dir or os.path.join(DATA_DIR, "tokens_v2")
    train_bin = os.path.join(token_dir, "train.bin")
    val_npy = os.path.join(token_dir, "val.npy")
    meta_path = os.path.join(token_dir, "metadata.json")

    assert os.path.exists(train_bin), f"Missing {train_bin}. Run: uv run prepare_data_v3.py"
    assert os.path.exists(meta_path), f"Missing {meta_path}"

    with open(meta_path) as f:
        meta = json.load(f)

    vocab_size = meta["vocab_size"]
    train_mmap = np.memmap(train_bin, dtype=np.int32, mode="r")
    val_data = np.load(val_npy)

    n_val = (len(val_data) - 1) // context_len
    val_data = val_data[: n_val * context_len + 1]
    val_x = val_data[:-1].reshape(n_val, context_len)
    val_y = val_data[1:].reshape(n_val, context_len)

    n_train = (len(train_mmap) - 1) // context_len
    print(f"Dataset: {token_dir} ({meta['total_train_tokens']/1e9:.2f}B tokens, vocab={vocab_size})")
    print(f"  {n_train:,} train seqs (streamed), {n_val:,} val seqs")

    # save tokenizer ref for inference (generate.py, serve.py)
    tok_path = meta["tokenizer_path"]
    if not os.path.isabs(tok_path):
        tok_path = os.path.join(os.path.dirname(__file__), tok_path)
    assert os.path.exists(tok_path), f"Missing tokenizer at {tok_path}"
    with open(os.path.join(DATA_DIR, "bpe_vocab.pkl"), "wb") as f:
        pickle.dump({"tokenizer_path": tok_path, "vocab_size": vocab_size}, f)

    return {
        "train_tokens": train_mmap,
        "val_x": val_x,
        "val_y": val_y,
        "vocab_size": vocab_size,
    }


def load_bpe_vocab():
    """Load saved BPE vocab for inference."""
    from tokenizers import Tokenizer

    with open(os.path.join(DATA_DIR, "bpe_vocab.pkl"), "rb") as f:
        saved = pickle.load(f)
    tok_path = saved["tokenizer_path"]
    if not os.path.isabs(tok_path):
        tok_path = os.path.join(os.path.dirname(__file__), tok_path)
    tok = Tokenizer.from_file(tok_path)
    vocab_size = saved.get("vocab_size", saved.get("bpe_vocab_size"))
    assert vocab_size is not None, "bpe_vocab.pkl missing vocab_size"
    return {
        "tokenizer_path": tok_path,
        "vocab_size": vocab_size,
        "decode_fn": lambda ids: tok.decode(list(int(i) for i in ids)),
    }
