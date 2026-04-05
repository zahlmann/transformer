"""Load v2 streaming dataset (trained BPE, vocab 32k, multi-source mix)."""

import json
import os
import pickle
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _make_sequences(arr, context_len):
    n_seq = (len(arr) - 1) // context_len
    arr = arr[: n_seq * context_len + 1]
    x = arr[:-1].reshape(n_seq, context_len)
    y = arr[1:].reshape(n_seq, context_len)
    return x, y


def load_data(context_len):
    """Load v2 tokenized dataset. Returns memmap train stream + materialized val splits."""
    from tokenizers import Tokenizer

    token_dir = os.path.join(DATA_DIR, "tokens_v2")
    train_bin = os.path.join(token_dir, "train.bin")
    val_npy = os.path.join(token_dir, "val.npy")
    meta_path = os.path.join(token_dir, "metadata.json")

    assert os.path.exists(train_bin), f"V2 dataset not found at {train_bin}. Run: uv run prepare_data_v2.py"
    assert os.path.exists(meta_path), f"Metadata not found at {meta_path}"

    with open(meta_path) as f:
        meta = json.load(f)

    print(f"Loading v2 dataset ({meta['total_train_tokens']/1e9:.2f}B train tokens)...")
    print(f"  Sources: {', '.join(f'{k} ({v/1e9:.1f}B)' for k, v in meta['sources'].items())}")

    train_mmap = np.memmap(train_bin, dtype=np.int32, mode="r")
    val_data = np.load(val_npy)
    vocab_size = meta["vocab_size"]
    val_x, val_y = _make_sequences(val_data, context_len)

    n_train_seqs = (len(train_mmap) - 1) // context_len
    print(f"  {n_train_seqs:,} train seqs (streamed), {len(val_x):,} val seqs, vocab={vocab_size}")

    tok = Tokenizer.from_file(meta["tokenizer_path"])
    decode_tokens = lambda ids: tok.decode(list(int(i) for i in ids))

    # save vocab mapping for inference (generate.py, serve.py)
    vocab_path = os.path.join(DATA_DIR, "bpe_vocab.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump({
            "tokenizer_path": meta["tokenizer_path"],
            "bpe_vocab_size": vocab_size,
            "tokenizer_type": "trained_bpe",
        }, f)

    return {
        "train_tokens": train_mmap,
        "train_x": None,
        "train_y": None,
        "val_x": val_x,
        "val_y": val_y,
        "vocab_size": vocab_size,
        "decode_fn": decode_tokens,
        "tokenizer": "trained_bpe",
        "streaming": True,
    }


def load_bpe_vocab():
    """Load saved BPE vocab mapping for inference."""
    vocab_path = os.path.join(DATA_DIR, "bpe_vocab.pkl")
    with open(vocab_path, "rb") as f:
        saved = pickle.load(f)
    if saved.get("tokenizer_type") == "trained_bpe":
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(saved["tokenizer_path"])
        saved["decode_fn"] = lambda ids: tok.decode(list(int(i) for i in ids))
    return saved
