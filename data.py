"""Download and prepare Shakespeare dataset (char-level or BPE)."""

import os
import pickle
import urllib.request
from collections import Counter
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_shakespeare():
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, "input.txt")
    if not os.path.exists(path):
        print("Downloading tiny Shakespeare...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    with open(path, "r") as f:
        text = f.read()
    return text


def prepare_data(context_len=128, val_fraction=0.1, tokenizer="char", bpe_vocab_size=512):
    """Prepare Shakespeare dataset.

    Args:
        tokenizer: "char" for character-level (vocab=65),
                   "bpe" for GPT-2 BPE with restricted vocab,
                   "trained_bpe" for BPE trained on Shakespeare (0% UNK).
        bpe_vocab_size: vocabulary size (only used if tokenizer="bpe" or "trained_bpe").
    """
    text = download_shakespeare()

    if tokenizer == "char":
        return _prepare_char(text, context_len, val_fraction)
    elif tokenizer == "bpe":
        return _prepare_bpe(text, context_len, val_fraction, bpe_vocab_size)
    elif tokenizer == "trained_bpe":
        return _prepare_trained_bpe(text, context_len, val_fraction, bpe_vocab_size)
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer}")


def _prepare_char(text, context_len, val_fraction):
    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_idx = {c: i for i, c in enumerate(chars)}

    data = np.array([char_to_idx[c] for c in text], dtype=np.int32)

    n = len(data)
    split = int(n * (1 - val_fraction))
    train_data = data[:split]
    val_data = data[split:]

    def make_sequences(arr):
        n_seq = (len(arr) - 1) // context_len
        arr = arr[: n_seq * context_len + 1]
        x = arr[:-1].reshape(n_seq, context_len)
        y = arr[1:].reshape(n_seq, context_len)
        return x, y

    train_x, train_y = make_sequences(train_data)
    val_x, val_y = make_sequences(val_data)

    return {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "vocab_size": vocab_size,
        "chars": chars,
        "char_to_idx": char_to_idx,
        "tokenizer": "char",
    }


def _prepare_bpe(text, context_len, val_fraction, bpe_vocab_size):
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    all_tokens = enc.encode(text)

    # Find top-K most frequent tokens, reserve 0 for <unk>
    counts = Counter(all_tokens)
    top_tokens = [tok for tok, _ in counts.most_common(bpe_vocab_size - 1)]  # -1 for <unk>
    gpt2_to_compact = {tok: i + 1 for i, tok in enumerate(top_tokens)}  # 1..bpe_vocab_size-1
    UNK = 0

    # Build compact token array
    data = np.array([gpt2_to_compact.get(t, UNK) for t in all_tokens], dtype=np.int32)
    unk_rate = np.mean(data == UNK)
    print(f"BPE: {len(all_tokens)} tokens, vocab={bpe_vocab_size}, unk_rate={unk_rate:.3f}")

    # Decode mapping: compact_id -> bytes
    compact_to_bytes = {UNK: b"?"}
    for gpt2_tok, compact_id in gpt2_to_compact.items():
        compact_to_bytes[compact_id] = enc.decode_single_token_bytes(gpt2_tok)

    # Save vocab mapping for inference
    vocab_path = os.path.join(DATA_DIR, "bpe_vocab.pkl")
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(vocab_path, "wb") as f:
        pickle.dump({
            "compact_to_bytes": compact_to_bytes,
            "gpt2_to_compact": gpt2_to_compact,
            "bpe_vocab_size": bpe_vocab_size,
        }, f)
    print(f"Saved BPE vocab mapping to {vocab_path}")

    # Split and make sequences
    n = len(data)
    split = int(n * (1 - val_fraction))
    train_data = data[:split]
    val_data = data[split:]

    def make_sequences(arr):
        n_seq = (len(arr) - 1) // context_len
        arr = arr[: n_seq * context_len + 1]
        x = arr[:-1].reshape(n_seq, context_len)
        y = arr[1:].reshape(n_seq, context_len)
        return x, y

    train_x, train_y = make_sequences(train_data)
    val_x, val_y = make_sequences(val_data)

    def decode_tokens(ids):
        return b"".join(compact_to_bytes.get(int(i), b"?") for i in ids).decode("utf-8", errors="replace")

    return {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "vocab_size": bpe_vocab_size,
        "decode_fn": decode_tokens,
        "compact_to_bytes": compact_to_bytes,
        "tokenizer": "bpe",
    }


def _prepare_trained_bpe(text, context_len, val_fraction, vocab_size):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    tokenizer_path = os.path.join(DATA_DIR, f"trained_bpe_{vocab_size}.json")

    if os.path.exists(tokenizer_path):
        tok = Tokenizer.from_file(tokenizer_path)
    else:
        os.makedirs(DATA_DIR, exist_ok=True)
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>"])
        corpus_path = os.path.join(DATA_DIR, "input.txt")
        tok.train([corpus_path], trainer)
        tok.save(tokenizer_path)

    all_tokens = tok.encode(text).ids
    actual_vocab = tok.get_vocab_size()
    print(f"Trained BPE: {len(all_tokens)} tokens, vocab={actual_vocab}")

    data = np.array(all_tokens, dtype=np.int32)

    # Save vocab mapping for inference
    vocab_path = os.path.join(DATA_DIR, "bpe_vocab.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump({
            "tokenizer_path": tokenizer_path,
            "bpe_vocab_size": actual_vocab,
            "tokenizer_type": "trained_bpe",
        }, f)
    print(f"Saved BPE vocab mapping to {vocab_path}")

    n = len(data)
    split = int(n * (1 - val_fraction))
    train_data = data[:split]
    val_data = data[split:]

    def make_sequences(arr):
        n_seq = (len(arr) - 1) // context_len
        arr = arr[: n_seq * context_len + 1]
        x = arr[:-1].reshape(n_seq, context_len)
        y = arr[1:].reshape(n_seq, context_len)
        return x, y

    train_x, train_y = make_sequences(train_data)
    val_x, val_y = make_sequences(val_data)

    def decode_tokens(ids):
        return tok.decode(list(int(i) for i in ids))

    return {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "vocab_size": actual_vocab,
        "decode_fn": decode_tokens,
        "tokenizer": "trained_bpe",
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


if __name__ == "__main__":
    print("=== Char-level ===")
    d = prepare_data(tokenizer="char")
    print(f"Vocab size: {d['vocab_size']}")
    print(f"Train sequences: {d['train_x'].shape}")
    print(f"Val sequences: {d['val_x'].shape}")

    print("\n=== BPE ===")
    d = prepare_data(tokenizer="bpe", bpe_vocab_size=512)
    print(f"Vocab size: {d['vocab_size']}")
    print(f"Train sequences: {d['train_x'].shape}")
    print(f"Val sequences: {d['val_x'].shape}")
    sample = d["train_x"][0][:20]
    print(f"Sample tokens: {sample}")
    print(f"Decoded: {d['decode_fn'](sample)}")
