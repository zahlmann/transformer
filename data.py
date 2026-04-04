"""Download and prepare datasets (Shakespeare or TinyStories) with char-level or BPE tokenization."""

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


def load_tinystories():
    """Load TinyStories dataset from pre-downloaded text files.

    Returns (train_text, val_text). Download with:
        datasets.load_dataset('roneneldan/TinyStories')
    """
    train_path = os.path.join(DATA_DIR, "tinystories_train.txt")
    val_path = os.path.join(DATA_DIR, "tinystories_val.txt")
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"TinyStories not found at {train_path}. "
            "Download with: datasets.load_dataset('roneneldan/TinyStories')"
        )
    with open(train_path, "r") as f:
        train_text = f.read()
    with open(val_path, "r") as f:
        val_text = f.read()
    return train_text, val_text


def prepare_data(context_len=128, val_fraction=0.1, tokenizer="char",
                 bpe_vocab_size=512, dataset="shakespeare"):
    """Prepare dataset for training.

    Args:
        dataset: "shakespeare" or "tinystories".
        tokenizer: "char" for character-level,
                   "bpe" for GPT-2 BPE with restricted vocab,
                   "trained_bpe" for BPE trained on corpus (0% UNK).
        bpe_vocab_size: vocabulary size (only used if tokenizer="bpe" or "trained_bpe").
    """
    if dataset == "combined":
        return _prepare_combined(context_len)
    if dataset == "combined_epoch2":
        return _prepare_combined(context_len, epoch=2)
    if dataset == "combined_v2":
        return _prepare_combined_v2(context_len)

    if dataset == "tinystories":
        train_text, val_text = load_tinystories()
    else:
        text = download_shakespeare()
        split = int(len(text) * (1 - val_fraction))
        train_text, val_text = text[:split], text[split:]

    if tokenizer == "char":
        return _prepare_char_from_texts(train_text, val_text, context_len)
    elif tokenizer == "bpe":
        combined = train_text + val_text
        return _prepare_bpe(combined, context_len, val_fraction, bpe_vocab_size)
    elif tokenizer == "trained_bpe":
        return _prepare_trained_bpe_from_texts(
            train_text, val_text, context_len, bpe_vocab_size, dataset)
    else:
        raise ValueError(f"Unknown tokenizer: {tokenizer}")


def _make_sequences(arr, context_len):
    n_seq = (len(arr) - 1) // context_len
    arr = arr[: n_seq * context_len + 1]
    x = arr[:-1].reshape(n_seq, context_len)
    y = arr[1:].reshape(n_seq, context_len)
    return x, y


def _prepare_char_from_texts(train_text, val_text, context_len):
    all_text = train_text + val_text
    chars = sorted(set(all_text))
    vocab_size = len(chars)
    char_to_idx = {c: i for i, c in enumerate(chars)}

    train_data = np.array([char_to_idx[c] for c in train_text], dtype=np.int32)
    val_data = np.array([char_to_idx[c] for c in val_text], dtype=np.int32)

    train_x, train_y = _make_sequences(train_data, context_len)
    val_x, val_y = _make_sequences(val_data, context_len)

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

    train_x, train_y = _make_sequences(train_data, context_len)
    val_x, val_y = _make_sequences(val_data, context_len)

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


def _prepare_trained_bpe_from_texts(train_text, val_text, context_len, vocab_size,
                                     dataset="shakespeare"):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    tokenizer_path = os.path.join(DATA_DIR, f"trained_bpe_{dataset}_{vocab_size}.json")

    if os.path.exists(tokenizer_path):
        tok = Tokenizer.from_file(tokenizer_path)
    else:
        os.makedirs(DATA_DIR, exist_ok=True)
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<pad>"])
        # Write train text to temp file for tokenizer training
        corpus_path = os.path.join(DATA_DIR, f"_train_corpus_{dataset}.txt")
        with open(corpus_path, "w") as f:
            f.write(train_text)
        tok.train([corpus_path], trainer)
        tok.save(tokenizer_path)
        os.remove(corpus_path)

    # Tokenize in chunks to avoid OOM on large texts (1.9GB TinyStories uses 46GB+ RAM
    # if encoded at once due to tokenizer intermediate allocations)
    CHUNK_CHARS = 50_000_000  # 50MB chunks
    actual_vocab = tok.get_vocab_size()

    def encode_chunked(text):
        if len(text) <= CHUNK_CHARS:
            return tok.encode(text).ids
        all_ids = []
        for i in range(0, len(text), CHUNK_CHARS):
            chunk = text[i:i + CHUNK_CHARS]
            all_ids.extend(tok.encode(chunk).ids)
            if i > 0 and i % (CHUNK_CHARS * 10) == 0:
                print(f"  tokenized {i / 1e6:.0f}M / {len(text) / 1e6:.0f}M chars ({len(all_ids) / 1e6:.1f}M tokens)")
        return all_ids

    # Cache tokenized data to skip re-tokenization on subsequent runs
    cache_path = os.path.join(DATA_DIR, f"tokenized_{dataset}_{vocab_size}.npz")
    if os.path.exists(cache_path):
        print(f"Loading cached tokens from {cache_path}...")
        cached = np.load(cache_path)
        train_data = cached["train"]
        val_data = cached["val"]
        print(f"Trained BPE ({dataset}): {len(train_data)} train tokens, "
              f"{len(val_data)} val tokens, vocab={actual_vocab}")
    else:
        print(f"Tokenizing train ({len(train_text)/1e6:.0f}M chars)...")
        train_tokens = encode_chunked(train_text)
        del train_text
        print(f"Tokenizing val ({len(val_text)/1e6:.0f}M chars)...")
        val_tokens = encode_chunked(val_text)
        del val_text
        print(f"Trained BPE ({dataset}): {len(train_tokens)} train tokens, "
              f"{len(val_tokens)} val tokens, vocab={actual_vocab}")

        train_data = np.array(train_tokens, dtype=np.int32)
        del train_tokens
        val_data = np.array(val_tokens, dtype=np.int32)
        del val_tokens

        print(f"Caching tokens to {cache_path}...")
        np.savez(cache_path, train=train_data, val=val_data)

    # Save vocab mapping for inference
    vocab_path = os.path.join(DATA_DIR, "bpe_vocab.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump({
            "tokenizer_path": tokenizer_path,
            "bpe_vocab_size": actual_vocab,
            "tokenizer_type": "trained_bpe",
        }, f)
    print(f"Saved BPE vocab mapping to {vocab_path}")

    train_x, train_y = _make_sequences(train_data, context_len)
    val_x, val_y = _make_sequences(val_data, context_len)

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


def _prepare_combined(context_len, epoch=1):
    """Load the pre-tokenized multi-source dataset (from prepare_data.py)."""
    from tokenizers import Tokenizer

    token_dir = os.path.join(DATA_DIR, "tokens")
    suffix = "_epoch2" if epoch == 2 else ""
    combined_path = os.path.join(token_dir, f"combined{suffix}.npz")
    meta_path = os.path.join(token_dir, f"metadata{suffix}.json")

    if not os.path.exists(combined_path):
        raise FileNotFoundError(
            f"Combined dataset not found at {combined_path}. "
            "Run: uv run prepare_data.py"
        )

    import json
    with open(meta_path) as f:
        meta = json.load(f)

    print(f"Loading combined dataset ({meta['total_train_tokens']/1e9:.2f}B train tokens)...")
    data = np.load(combined_path)
    train_data = data["train"]
    val_data = data["val"]

    vocab_size = meta["vocab_size"]
    tok = Tokenizer.from_file(meta["tokenizer_path"])

    train_x, train_y = _make_sequences(train_data, context_len)
    val_x, val_y = _make_sequences(val_data, context_len)

    print(f"Combined: {len(train_x):,} train seqs, {len(val_x):,} val seqs, vocab={vocab_size}")

    def decode_tokens(ids):
        return tok.decode(list(int(i) for i in ids))

    # Save vocab mapping for inference
    vocab_path = os.path.join(DATA_DIR, "bpe_vocab.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump({
            "tokenizer_path": meta["tokenizer_path"],
            "bpe_vocab_size": vocab_size,
            "tokenizer_type": "trained_bpe",
        }, f)

    return {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "vocab_size": vocab_size,
        "decode_fn": decode_tokens,
        "tokenizer": "trained_bpe",
    }


def _prepare_combined_v2(context_len):
    """Load v2 tokenized data (with EOS tokens, better mix)."""
    from tokenizers import Tokenizer

    token_dir = os.path.join(DATA_DIR, "tokens_v2")
    combined_path = os.path.join(token_dir, "combined.npz")
    meta_path = os.path.join(token_dir, "metadata.json")

    if not os.path.exists(combined_path):
        raise FileNotFoundError(
            f"V2 dataset not found at {combined_path}. "
            "Run: uv run prepare_data_v2.py"
        )

    import json as _json
    with open(meta_path) as f:
        meta = _json.load(f)

    print(f"Loading v2 dataset ({meta['total_train_tokens']/1e9:.2f}B train tokens)...")
    print(f"  Sources: {', '.join(f'{k} ({v/1e9:.1f}B)' for k, v in meta['sources'].items())}")
    print(f"  EOS between docs: {meta.get('has_eos_between_docs', False)}")

    data = np.load(combined_path)
    train_data = data["train"]
    val_data = data["val"]

    vocab_size = meta["vocab_size"]
    tok = Tokenizer.from_file(meta["tokenizer_path"])

    train_x, train_y = _make_sequences(train_data, context_len)
    val_x, val_y = _make_sequences(val_data, context_len)

    print(f"  {len(train_x):,} train seqs, {len(val_x):,} val seqs, vocab={vocab_size}")

    def decode_tokens(ids):
        return tok.decode(list(int(i) for i in ids))

    vocab_path = os.path.join(DATA_DIR, "bpe_vocab.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump({
            "tokenizer_path": meta["tokenizer_path"],
            "bpe_vocab_size": vocab_size,
            "tokenizer_type": "trained_bpe",
        }, f)

    return {
        "train_x": train_x,
        "train_y": train_y,
        "val_x": val_x,
        "val_y": val_y,
        "vocab_size": vocab_size,
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
    import sys
    ds = sys.argv[1] if len(sys.argv) > 1 else "shakespeare"

    print(f"=== Char-level ({ds}) ===")
    d = prepare_data(tokenizer="char", dataset=ds)
    print(f"Vocab size: {d['vocab_size']}")
    print(f"Train sequences: {d['train_x'].shape}")
    print(f"Val sequences: {d['val_x'].shape}")

    print(f"\n=== Trained BPE ({ds}, vocab=4096) ===")
    d = prepare_data(tokenizer="trained_bpe", bpe_vocab_size=4096, dataset=ds)
    print(f"Vocab size: {d['vocab_size']}")
    print(f"Train sequences: {d['train_x'].shape}")
    print(f"Val sequences: {d['val_x'].shape}")
    sample = d["train_x"][0][:20]
    print(f"Sample tokens: {sample}")
    print(f"Decoded: {d['decode_fn'](sample)}")
