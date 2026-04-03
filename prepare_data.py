"""Download and prepare a high-quality multi-source training dataset.

Epoch 1 data mix (targeting ~2B tokens):
  60% FineWeb-Edu (educational web text, quality score >= 3)
  20% Cosmopedia v2 (synthetic textbook-style, from smollm-corpus)
  15% Wikipedia (encyclopedic knowledge)
   5% Code (code_search_net, multi-language)

Epoch 2 data mix (targeting ~2B fresh tokens, NO repetition):
  50% FineWeb-Edu (1B tokens) — new docs from sample-10BT, skip epoch 1 docs
  20% StarCoder (400M tokens) — bigcode/starcoderdata, high-quality code
  15% Wikipedia (300M tokens) — fresh articles not used in epoch 1
  15% Cosmopedia v2 (300M tokens) — fresh docs

Usage:
  uv run prepare_data.py                    # download + tokenize epoch 1
  uv run prepare_data.py --epoch 2          # download + tokenize epoch 2 (fresh data)
  uv run prepare_data.py --download-only    # just download raw text
  uv run prepare_data.py --tokenize-only    # just tokenize (assumes already downloaded)
  uv run prepare_data.py --stats            # show dataset statistics
"""

import os
import json
import argparse
import time
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
TOKEN_DIR = DATA_DIR / "tokens"

# Target token counts per source (approximate)
TARGETS = {
    "fineweb_edu": 1_200_000_000,   # 1.2B tokens (60%)
    "cosmopedia":    400_000_000,   # 400M tokens (20%)
    "wikipedia":     300_000_000,   # 300M tokens (15%)
    "code":          100_000_000,   # 100M tokens (5%)
}
TARGETS_E2 = {
    "fineweb_edu": 1_000_000_000,   # 1B tokens (50%)
    "code":          400_000_000,   # 400M tokens (20%) — StarCoder
    "wikipedia":     300_000_000,   # 300M tokens (15%)
    "cosmopedia":    300_000_000,   # 300M tokens (15%)
}
TOTAL_TARGET = sum(TARGETS.values())  # 2B tokens
VOCAB_SIZE = 32000
VAL_FRACTION = 0.005  # 0.5% for validation (10M tokens)


def download_fineweb_edu(max_tokens):
    """Download FineWeb-Edu sample, filtering for score >= 3."""
    from datasets import load_dataset

    out_path = RAW_DIR / "fineweb_edu.jsonl"
    if out_path.exists():
        n_lines = sum(1 for _ in open(out_path))
        print(f"FineWeb-Edu: already have {n_lines:,} documents at {out_path}")
        return out_path

    print(f"Downloading FineWeb-Edu (target: {max_tokens/1e9:.1f}B tokens, score >= 3)...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

    n_docs = 0
    n_chars = 0
    # Rough estimate: 1 token ≈ 4 chars for English text
    max_chars = max_tokens * 4
    t0 = time.time()

    with open(out_path, "w") as f:
        for doc in ds:
            score = doc.get("score", 0)
            if score < 3.0:
                continue
            text = doc.get("text", "")
            if len(text) < 100:  # skip very short docs
                continue
            f.write(json.dumps({"text": text, "source": "fineweb_edu"}) + "\n")
            n_docs += 1
            n_chars += len(text)
            if n_docs % 10000 == 0:
                elapsed = time.time() - t0
                est_tokens = n_chars / 4
                print(f"  {n_docs:,} docs, ~{est_tokens/1e6:.0f}M tokens, "
                      f"{elapsed:.0f}s ({n_docs/elapsed:.0f} docs/s)")
            if n_chars >= max_chars:
                break

    print(f"FineWeb-Edu: {n_docs:,} docs, ~{n_chars/4/1e6:.0f}M tokens")
    return out_path


def download_wikipedia(max_tokens):
    """Download Wikipedia from HuggingFace."""
    from datasets import load_dataset

    out_path = RAW_DIR / "wikipedia.jsonl"
    if out_path.exists():
        n_lines = sum(1 for _ in open(out_path))
        print(f"Wikipedia: already have {n_lines:,} documents at {out_path}")
        return out_path

    print(f"Downloading Wikipedia (target: {max_tokens/1e9:.1f}B tokens)...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    n_docs = 0
    n_chars = 0
    max_chars = max_tokens * 4
    t0 = time.time()

    with open(out_path, "w") as f:
        for doc in ds:
            text = doc.get("text", "")
            if len(text) < 200:  # skip stubs
                continue
            f.write(json.dumps({"text": text, "source": "wikipedia"}) + "\n")
            n_docs += 1
            n_chars += len(text)
            if n_docs % 10000 == 0:
                elapsed = time.time() - t0
                est_tokens = n_chars / 4
                print(f"  {n_docs:,} docs, ~{est_tokens/1e6:.0f}M tokens, "
                      f"{elapsed:.0f}s ({n_docs/elapsed:.0f} docs/s)")
            if n_chars >= max_chars:
                break

    print(f"Wikipedia: {n_docs:,} docs, ~{n_chars/4/1e6:.0f}M tokens")
    return out_path


def download_code(max_tokens):
    """Download code from code_search_net (multi-language, ungated)."""
    from datasets import load_dataset

    out_path = RAW_DIR / "code.jsonl"
    if out_path.exists():
        n_lines = sum(1 for _ in open(out_path))
        print(f"Code: already have {n_lines:,} documents at {out_path}")
        return out_path

    print(f"Downloading code from code_search_net (target: {max_tokens/1e9:.1f}B tokens)...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    languages = ["python", "javascript", "java", "go", "ruby", "php"]
    tokens_per_lang = max_tokens // len(languages)

    n_docs = 0
    n_chars = 0
    t0 = time.time()

    with open(out_path, "w") as f:
        for lang in languages:
            lang_chars = 0
            max_lang_chars = tokens_per_lang * 4
            print(f"  Downloading {lang}...")
            try:
                ds = load_dataset(
                    "code_search_net", lang,
                    split="train", streaming=True
                )
                for doc in ds:
                    text = doc.get("whole_func_string", "")
                    if len(text) < 50 or len(text) > 50000:
                        continue
                    f.write(json.dumps({"text": text, "source": f"code_{lang}"}) + "\n")
                    n_docs += 1
                    n_chars += len(text)
                    lang_chars += len(text)
                    if n_docs % 10000 == 0:
                        elapsed = time.time() - t0
                        print(f"    {n_docs:,} functions, ~{n_chars/4/1e6:.0f}M tokens total")
                    if lang_chars >= max_lang_chars:
                        break
            except Exception as e:
                print(f"    Warning: failed to download {lang}: {e}")
                continue

    print(f"Code: {n_docs:,} functions, ~{n_chars/4/1e6:.0f}M tokens")
    return out_path


def download_cosmopedia(max_tokens):
    """Download Cosmopedia v2 from smollm-corpus (synthetic textbook data)."""
    from datasets import load_dataset

    out_path = RAW_DIR / "cosmopedia.jsonl"
    if out_path.exists():
        n_lines = sum(1 for _ in open(out_path))
        print(f"Cosmopedia: already have {n_lines:,} documents at {out_path}")
        return out_path

    print(f"Downloading Cosmopedia v2 from smollm-corpus (target: {max_tokens/1e9:.1f}B tokens)...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2",
                       split="train", streaming=True)

    n_docs = 0
    n_chars = 0
    max_chars = max_tokens * 4
    t0 = time.time()

    with open(out_path, "w") as f:
        for doc in ds:
            text = doc.get("text", "")
            if len(text) < 100:
                continue
            f.write(json.dumps({"text": text, "source": "cosmopedia"}) + "\n")
            n_docs += 1
            n_chars += len(text)
            if n_docs % 10000 == 0:
                elapsed = time.time() - t0
                est_tokens = n_chars / 4
                print(f"  {n_docs:,} docs, ~{est_tokens/1e6:.0f}M tokens, "
                      f"{elapsed:.0f}s ({n_docs/elapsed:.0f} docs/s)")
            if n_chars >= max_chars:
                break

    print(f"Cosmopedia: {n_docs:,} docs, ~{n_chars/4/1e6:.0f}M tokens")
    return out_path


def _count_epoch1_docs(source_file):
    """Count how many docs epoch 1 used from a source (lines in raw .jsonl)."""
    path = RAW_DIR / source_file
    if not path.exists():
        return 0
    return sum(1 for _ in open(path))


def download_fineweb_edu_e2(max_tokens):
    """Download fresh FineWeb-Edu docs, skipping epoch 1 docs."""
    from datasets import load_dataset

    out_path = RAW_DIR / "fineweb_edu_e2.jsonl"
    if out_path.exists():
        n_lines = sum(1 for _ in open(out_path))
        print(f"FineWeb-Edu (e2): already have {n_lines:,} documents at {out_path}")
        return out_path

    skip_docs = _count_epoch1_docs("fineweb_edu.jsonl")
    print(f"Downloading FineWeb-Edu epoch 2 (target: {max_tokens/1e9:.1f}B tokens, "
          f"skipping {skip_docs:,} epoch 1 docs)...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)

    n_docs = 0
    n_chars = 0
    n_skipped = 0
    max_chars = max_tokens * 4
    t0 = time.time()

    with open(out_path, "w") as f:
        for doc in ds:
            score = doc.get("score", 0)
            if score < 3.0:
                continue
            text = doc.get("text", "")
            if len(text) < 100:
                continue
            # skip epoch 1 docs (same streaming order)
            if n_skipped < skip_docs:
                n_skipped += 1
                if n_skipped % 100000 == 0:
                    print(f"  Skipping epoch 1 docs: {n_skipped:,}/{skip_docs:,}")
                continue
            f.write(json.dumps({"text": text, "source": "fineweb_edu"}) + "\n")
            n_docs += 1
            n_chars += len(text)
            if n_docs % 10000 == 0:
                elapsed = time.time() - t0
                est_tokens = n_chars / 4
                print(f"  {n_docs:,} docs, ~{est_tokens/1e6:.0f}M tokens, "
                      f"{elapsed:.0f}s ({n_docs/elapsed:.0f} docs/s)")
            if n_chars >= max_chars:
                break

    print(f"FineWeb-Edu (e2): {n_docs:,} docs, ~{n_chars/4/1e6:.0f}M tokens")
    return out_path


def download_wikipedia_e2(max_tokens):
    """Download fresh Wikipedia articles, skipping epoch 1 docs."""
    from datasets import load_dataset

    out_path = RAW_DIR / "wikipedia_e2.jsonl"
    if out_path.exists():
        n_lines = sum(1 for _ in open(out_path))
        print(f"Wikipedia (e2): already have {n_lines:,} documents at {out_path}")
        return out_path

    skip_docs = _count_epoch1_docs("wikipedia.jsonl")
    print(f"Downloading Wikipedia epoch 2 (target: {max_tokens/1e9:.1f}B tokens, "
          f"skipping {skip_docs:,} epoch 1 docs)...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)

    n_docs = 0
    n_chars = 0
    n_skipped = 0
    max_chars = max_tokens * 4
    t0 = time.time()

    with open(out_path, "w") as f:
        for doc in ds:
            text = doc.get("text", "")
            if len(text) < 200:
                continue
            if n_skipped < skip_docs:
                n_skipped += 1
                if n_skipped % 100000 == 0:
                    print(f"  Skipping epoch 1 docs: {n_skipped:,}/{skip_docs:,}")
                continue
            f.write(json.dumps({"text": text, "source": "wikipedia"}) + "\n")
            n_docs += 1
            n_chars += len(text)
            if n_docs % 10000 == 0:
                elapsed = time.time() - t0
                est_tokens = n_chars / 4
                print(f"  {n_docs:,} docs, ~{est_tokens/1e6:.0f}M tokens, "
                      f"{elapsed:.0f}s ({n_docs/elapsed:.0f} docs/s)")
            if n_chars >= max_chars:
                break

    print(f"Wikipedia (e2): {n_docs:,} docs, ~{n_chars/4/1e6:.0f}M tokens")
    return out_path


def download_cosmopedia_e2(max_tokens):
    """Download fresh Cosmopedia v2 docs, skipping epoch 1 docs."""
    from datasets import load_dataset

    out_path = RAW_DIR / "cosmopedia_e2.jsonl"
    if out_path.exists():
        n_lines = sum(1 for _ in open(out_path))
        print(f"Cosmopedia (e2): already have {n_lines:,} documents at {out_path}")
        return out_path

    skip_docs = _count_epoch1_docs("cosmopedia.jsonl")
    print(f"Downloading Cosmopedia v2 epoch 2 (target: {max_tokens/1e9:.1f}B tokens, "
          f"skipping {skip_docs:,} epoch 1 docs)...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2",
                       split="train", streaming=True)

    n_docs = 0
    n_chars = 0
    n_skipped = 0
    max_chars = max_tokens * 4
    t0 = time.time()

    with open(out_path, "w") as f:
        for doc in ds:
            text = doc.get("text", "")
            if len(text) < 100:
                continue
            if n_skipped < skip_docs:
                n_skipped += 1
                if n_skipped % 100000 == 0:
                    print(f"  Skipping epoch 1 docs: {n_skipped:,}/{skip_docs:,}")
                continue
            f.write(json.dumps({"text": text, "source": "cosmopedia"}) + "\n")
            n_docs += 1
            n_chars += len(text)
            if n_docs % 10000 == 0:
                elapsed = time.time() - t0
                est_tokens = n_chars / 4
                print(f"  {n_docs:,} docs, ~{est_tokens/1e6:.0f}M tokens, "
                      f"{elapsed:.0f}s ({n_docs/elapsed:.0f} docs/s)")
            if n_chars >= max_chars:
                break

    print(f"Cosmopedia (e2): {n_docs:,} docs, ~{n_chars/4/1e6:.0f}M tokens")
    return out_path


def download_code_starcoderdata(max_tokens):
    """Download code from bigcode/starcoderdata (high-quality, deduplicated)."""
    from datasets import load_dataset

    out_path = RAW_DIR / "code_e2.jsonl"
    if out_path.exists():
        n_lines = sum(1 for _ in open(out_path))
        print(f"StarCoder code (e2): already have {n_lines:,} documents at {out_path}")
        return out_path

    print(f"Downloading StarCoder code (target: {max_tokens/1e9:.1f}B tokens)...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    languages = ["python", "javascript", "typescript", "java", "c", "c++", "rust", "go"]
    tokens_per_lang = max_tokens // len(languages)

    n_docs = 0
    n_chars = 0
    t0 = time.time()

    with open(out_path, "w") as f:
        for lang in languages:
            lang_chars = 0
            max_lang_chars = tokens_per_lang * 4
            print(f"  Downloading {lang}...")
            try:
                ds = load_dataset(
                    "bigcode/starcoderdata", data_dir=lang,
                    split="train", streaming=True
                )
                for doc in ds:
                    text = doc.get("content", "")
                    if len(text) < 100 or len(text) > 50000:
                        continue
                    f.write(json.dumps({"text": text, "source": f"code_{lang}"}) + "\n")
                    n_docs += 1
                    n_chars += len(text)
                    lang_chars += len(text)
                    if n_docs % 10000 == 0:
                        elapsed = time.time() - t0
                        print(f"    {n_docs:,} files, ~{n_chars/4/1e6:.0f}M tokens total")
                    if lang_chars >= max_lang_chars:
                        break
            except Exception as e:
                print(f"    Warning: failed to download {lang}: {e}")
                continue

    print(f"StarCoder code (e2): {n_docs:,} files, ~{n_chars/4/1e6:.0f}M tokens")
    return out_path


def download_all(epoch=1):
    """Download all data sources."""
    t0 = time.time()
    if epoch == 1:
        download_fineweb_edu(TARGETS["fineweb_edu"])
        download_wikipedia(TARGETS["wikipedia"])
        download_code(TARGETS["code"])
        download_cosmopedia(TARGETS["cosmopedia"])
    else:
        download_fineweb_edu_e2(TARGETS_E2["fineweb_edu"])
        download_code_starcoderdata(TARGETS_E2["code"])
        download_wikipedia_e2(TARGETS_E2["wikipedia"])
        download_cosmopedia_e2(TARGETS_E2["cosmopedia"])
    print(f"\nAll downloads complete in {time.time()-t0:.0f}s")


def train_tokenizer():
    """Train a BPE tokenizer on the combined corpus."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    tok_path = DATA_DIR / f"tokenizer_{VOCAB_SIZE}.json"
    if tok_path.exists():
        print(f"Tokenizer already exists at {tok_path}")
        return tok_path

    print(f"Training BPE tokenizer (vocab={VOCAB_SIZE}) on combined corpus...")

    # Collect sample text from each source for tokenizer training
    # Use ~100M chars from each source (enough for good vocab coverage)
    sample_path = RAW_DIR / "_tokenizer_sample.txt"
    max_chars_per_source = 100_000_000  # 100M chars

    with open(sample_path, "w") as out:
        for source_file in ["fineweb_edu.jsonl", "wikipedia.jsonl", "code.jsonl", "cosmopedia.jsonl"]:
            path = RAW_DIR / source_file
            if not path.exists():
                print(f"  Skipping {source_file} (not downloaded)")
                continue
            n_chars = 0
            print(f"  Sampling from {source_file}...")
            with open(path) as f:
                for line in f:
                    doc = json.loads(line)
                    text = doc["text"]
                    out.write(text + "\n")
                    n_chars += len(text)
                    if n_chars >= max_chars_per_source:
                        break
            print(f"    {n_chars/1e6:.0f}M chars sampled")

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<pad>", "<eos>"],
        min_frequency=2,
        show_progress=True,
    )
    tok.train([str(sample_path)], trainer)
    tok.save(str(tok_path))
    sample_path.unlink()  # clean up

    print(f"Tokenizer saved to {tok_path} (vocab={tok.get_vocab_size()})")
    return tok_path


def _tokenize_source(args):
    """Tokenize a single source (for use with multiprocessing)."""
    source_name, source_file, tok_path_str, target, chunk_chars = args
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path_str)
    path = RAW_DIR / source_file
    cache_path = TOKEN_DIR / f"{source_name}.npz"

    if cache_path.exists():
        cached = np.load(cache_path)
        n = len(cached["train"]) + len(cached["val"])
        print(f"  {source_name}: cached ({n/1e6:.1f}M tokens)", flush=True)
        return source_name, n

    print(f"  Tokenizing {source_name}...", flush=True)
    t0 = time.time()

    token_chunks = []
    total_tokens = 0
    n_chars = 0
    batch_texts = []
    batch_chars = 0

    with open(path) as f:
        for line in f:
            doc = json.loads(line)
            text = doc["text"]
            batch_texts.append(text)
            batch_chars += len(text)
            n_chars += len(text)

            if batch_chars >= chunk_chars:
                encodings = tok.encode_batch(batch_texts)
                for enc in encodings:
                    chunk_arr = np.array(enc.ids, dtype=np.int32)
                    token_chunks.append(chunk_arr)
                    total_tokens += len(chunk_arr)
                batch_texts = []
                batch_chars = 0
                elapsed = time.time() - t0
                print(f"  {source_name}: {total_tokens/1e6:.0f}M tokens, "
                      f"{n_chars/1e6:.0f}M chars, {elapsed:.0f}s", flush=True)
                if total_tokens >= target:
                    break

    if batch_texts:
        encodings = tok.encode_batch(batch_texts)
        for enc in encodings:
            chunk_arr = np.array(enc.ids, dtype=np.int32)
            token_chunks.append(chunk_arr)
            total_tokens += len(chunk_arr)

    tokens = np.concatenate(token_chunks)
    del token_chunks

    n_val = max(int(len(tokens) * VAL_FRACTION), 10000)
    val_tok = tokens[:n_val]
    train_tok = tokens[n_val:]

    np.savez(cache_path, train=train_tok, val=val_tok)
    print(f"  {source_name}: {len(train_tok)/1e6:.1f}M train + {len(val_tok)/1e6:.1f}M val "
          f"({time.time()-t0:.0f}s)", flush=True)
    return source_name, len(tokens)


def tokenize_all(epoch=1):
    """Tokenize all sources in parallel and combine."""
    from concurrent.futures import ProcessPoolExecutor
    from tokenizers import Tokenizer

    tok_path = train_tokenizer()
    tok = Tokenizer.from_file(str(tok_path))
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)

    CHUNK_CHARS = 50_000_000  # 50M chars per tokenization chunk

    if epoch == 1:
        sources = [
            ("fineweb_edu", "fineweb_edu.jsonl"),
            ("wikipedia", "wikipedia.jsonl"),
            ("code", "code.jsonl"),
            ("cosmopedia", "cosmopedia.jsonl"),
        ]
        targets = TARGETS
        suffix = ""
    else:
        sources = [
            ("fineweb_edu_e2", "fineweb_edu_e2.jsonl"),
            ("code_e2", "code_e2.jsonl"),
            ("wikipedia_e2", "wikipedia_e2.jsonl"),
            ("cosmopedia_e2", "cosmopedia_e2.jsonl"),
        ]
        # Map e2 source names to target keys
        target_map = {
            "fineweb_edu_e2": "fineweb_edu",
            "code_e2": "code",
            "wikipedia_e2": "wikipedia",
            "cosmopedia_e2": "cosmopedia",
        }
        targets = {name: TARGETS_E2[target_map[name]] for name in target_map}
        suffix = "_epoch2"

    # Filter to sources that exist
    sources = [(name, f) for name, f in sources if (RAW_DIR / f).exists()]
    source_stats = {}

    # Tokenize sources in parallel (each uses encode_batch internally)
    print(f"Tokenizing {len(sources)} sources in parallel (epoch {epoch})...", flush=True)
    args_list = [
        (name, f, str(tok_path), targets.get(name, 200_000_000), CHUNK_CHARS)
        for name, f in sources
    ]

    with ProcessPoolExecutor(max_workers=len(sources)) as pool:
        for source_name, n_tokens in pool.map(_tokenize_source, args_list):
            source_stats[source_name] = n_tokens

    # Combine — load from cache files to avoid keeping everything in RAM
    all_train_tokens = []
    all_val_tokens = []
    for source_name, _ in sources:
        cache_path = TOKEN_DIR / f"{source_name}.npz"
        if cache_path.exists():
            cached = np.load(cache_path)
            all_train_tokens.append(cached["train"])
            all_val_tokens.append(cached["val"])

    print("\nCombining all sources...", flush=True)
    train_combined = np.concatenate(all_train_tokens)
    val_combined = np.concatenate(all_val_tokens)
    del all_train_tokens, all_val_tokens

    # Shuffle training data at document boundary (already mixed by source order,
    # but let's do a token-level shuffle in chunks to mix sources better)
    print("Shuffling training data...")
    rng = np.random.default_rng(42 if epoch == 1 else 137)
    # Shuffle in 512-token chunks (preserves local coherence while mixing sources)
    chunk_size = 512
    n_chunks = len(train_combined) // chunk_size
    train_chunked = train_combined[:n_chunks * chunk_size].reshape(n_chunks, chunk_size)
    perm = rng.permutation(n_chunks)
    train_combined = train_chunked[perm].reshape(-1)

    # Save combined dataset
    combined_path = TOKEN_DIR / f"combined{suffix}.npz"
    np.savez(combined_path, train=train_combined, val=val_combined)

    print(f"\nFinal dataset (epoch {epoch}):")
    print(f"  Train: {len(train_combined):,} tokens ({len(train_combined)/1e9:.2f}B)")
    print(f"  Val:   {len(val_combined):,} tokens ({len(val_combined)/1e6:.1f}M)")
    print(f"  Sources: {source_stats}")
    print(f"  Saved to {combined_path}")

    # Save metadata
    meta = {
        "epoch": epoch,
        "vocab_size": VOCAB_SIZE,
        "tokenizer_path": str(tok_path),
        "sources": source_stats,
        "total_train_tokens": int(len(train_combined)),
        "total_val_tokens": int(len(val_combined)),
        "val_fraction": VAL_FRACTION,
    }
    meta_path = TOKEN_DIR / f"metadata{suffix}.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Metadata saved to {meta_path}")


def show_stats():
    """Show statistics for downloaded/tokenized data."""
    print("=== Raw data ===")
    for f in sorted(RAW_DIR.glob("*.jsonl")) if RAW_DIR.exists() else []:
        n_lines = sum(1 for _ in open(f))
        size_mb = f.stat().st_size / 1e6
        print(f"  {f.name}: {n_lines:,} docs, {size_mb:.0f} MB")

    print("\n=== Tokenized data ===")
    if TOKEN_DIR.exists():
        for f in sorted(TOKEN_DIR.glob("*.npz")):
            data = np.load(f)
            if "train" in data:
                n_train = len(data["train"])
                n_val = len(data["val"])
                print(f"  {f.name}: {n_train/1e6:.1f}M train + {n_val/1e6:.1f}M val tokens")
            else:
                for key in data:
                    print(f"  {f.name}/{key}: {len(data[key])/1e6:.1f}M tokens")

    meta_path = TOKEN_DIR / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"\n=== Combined dataset ===")
        print(f"  Vocab size: {meta['vocab_size']}")
        print(f"  Train: {meta['total_train_tokens']/1e9:.2f}B tokens")
        print(f"  Val: {meta['total_val_tokens']/1e6:.1f}M tokens")
        print(f"  Sources: {meta['sources']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare multi-source training data")
    parser.add_argument("--epoch", type=int, default=1, choices=[1, 2],
                        help="Epoch 1: original mix. Epoch 2: fresh data with StarCoder code.")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--tokenize-only", action="store_true")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    elif args.download_only:
        download_all(epoch=args.epoch)
    elif args.tokenize_only:
        tokenize_all(epoch=args.epoch)
    else:
        download_all(epoch=args.epoch)
        tokenize_all(epoch=args.epoch)
