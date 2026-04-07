"""Prepare high-quality training data v2.

Data mix (based on SmolLM2/Llama3 best practices, April 2026):
  55% Web text    — FineWeb-Edu (score>=3) + Wikipedia + Cosmopedia
  25% Code        — StarCoder (deduplicated, multi-lang)
  15% Math        — OpenWebMath (faithful notation, LaTeX)
   5% Reserved    — for instruction data in later training stage

Key improvements over v1:
  - EOS tokens between all documents
  - Maximize unique data from each source (combine all epochs)
  - Math data included
  - Larger code fraction (25% vs 6%)
  - Single combined dataset with proper shuffling

Usage:
  uv run prepare_data_v2.py                    # download + tokenize everything
  uv run prepare_data_v2.py --tokenize-only    # just retokenize existing raw data
  uv run prepare_data_v2.py --stats            # show data statistics
"""

import json
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
TOKEN_DIR = DATA_DIR / "tokens_v2"
VOCAB_SIZE = 32000
VAL_FRACTION = 0.005
EOS_TOKEN_ID = 1  # <eos> in our tokenizer

# Target tokens per source (aim for ~10B unique tokens total)
TARGETS = {
    "web": {
        "fineweb_edu":  3_500_000_000,   # 3.5B tokens — pull as much as possible
        "wikipedia":      800_000_000,   # 0.8B tokens
        "cosmopedia":     900_000_000,   # 0.9B tokens
    },  # total web: ~5.2B (55%)
    "code": {
        "starcoderdata": 2_500_000_000,  # 2.5B tokens (25%)
    },
    "math": {
        "openwebmath":   1_500_000_000,  # 1.5B tokens (15%)
    },
}


def _download_fineweb_edu(max_tokens):
    """Download FineWeb-Edu, combining existing epoch1+epoch2 files first."""
    from datasets import load_dataset

    # combine existing raw files
    existing = []
    for fname in ["fineweb_edu.jsonl", "fineweb_edu_e2.jsonl"]:
        p = RAW_DIR / fname
        if p.exists():
            existing.append(p)

    out_path = RAW_DIR / "fineweb_edu_all.jsonl"
    if out_path.exists():
        n = sum(1 for _ in open(out_path))
        est = n * 5000 / 3.5  # rough: 5000 chars/doc avg
        print(f"FineWeb-Edu: already have {n:,} docs (~{est/1e9:.1f}B tokens)")
        if est >= max_tokens * 0.9:
            return out_path

    print(f"Combining + downloading FineWeb-Edu (target: {max_tokens/1e9:.1f}B tokens)...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # first, deduplicate and combine existing files
    seen_texts = set()
    n_docs = 0
    n_chars = 0
    max_chars = max_tokens * 3.5  # approximate chars needed

    with open(out_path, "w") as fout:
        # existing data first
        for p in existing:
            print(f"  Reading {p.name}...")
            with open(p) as fin:
                for line in fin:
                    doc = json.loads(line)
                    text = doc.get("text", "")
                    if len(text) < 100:
                        continue
                    # simple dedup: first 200 chars as key
                    key = text[:200]
                    if key in seen_texts:
                        continue
                    seen_texts.add(key)
                    fout.write(line)
                    n_docs += 1
                    n_chars += len(text)

        print(f"  Existing: {n_docs:,} unique docs, ~{n_chars/3.5/1e9:.2f}B tokens")

        # download more if needed
        if n_chars < max_chars:
            print(f"  Downloading more from HuggingFace (need {(max_chars-n_chars)/1e9:.1f}B more chars)...")
            ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                              split="train", streaming=True)
            t0 = time.time()
            skip_count = 0
            for doc in ds:
                score = doc.get("score", 0)
                if score < 3.0:
                    continue
                text = doc.get("text", "")
                if len(text) < 100:
                    continue
                key = text[:200]
                if key in seen_texts:
                    skip_count += 1
                    continue
                seen_texts.add(key)
                fout.write(json.dumps({"text": text, "source": "fineweb_edu"}) + "\n")
                n_docs += 1
                n_chars += len(text)
                if n_docs % 50000 == 0:
                    print(f"    {n_docs:,} docs, ~{n_chars/3.5/1e9:.2f}B tokens, "
                          f"skipped {skip_count} dupes, {time.time()-t0:.0f}s")
                if n_chars >= max_chars:
                    break

    del seen_texts
    print(f"  FineWeb-Edu total: {n_docs:,} docs, ~{n_chars/3.5/1e9:.2f}B tokens")
    return out_path


def _download_wikipedia(max_tokens):
    """Combine existing Wikipedia files."""
    out_path = RAW_DIR / "wikipedia_all.jsonl"
    if out_path.exists():
        n = sum(1 for _ in open(out_path))
        print(f"Wikipedia: already have {n:,} docs")
        return out_path

    from datasets import load_dataset

    existing = [RAW_DIR / f for f in ["wikipedia.jsonl", "wikipedia_e2.jsonl"]
                if (RAW_DIR / f).exists()]

    seen = set()
    n_docs = 0
    n_chars = 0
    max_chars = max_tokens * 3.5

    with open(out_path, "w") as fout:
        for p in existing:
            print(f"  Reading {p.name}...")
            with open(p) as fin:
                for line in fin:
                    doc = json.loads(line)
                    text = doc.get("text", "")
                    title = doc.get("title", "")
                    key = title or text[:200]
                    if key in seen:
                        continue
                    seen.add(key)
                    fout.write(line)
                    n_docs += 1
                    n_chars += len(text)

        if n_chars < max_chars:
            print(f"  Downloading more Wikipedia...")
            ds = load_dataset("wikimedia/wikipedia", "20231101.en",
                              split="train", streaming=True)
            for doc in ds:
                text = doc.get("text", "")
                title = doc.get("title", "")
                if len(text) < 200:
                    continue
                key = title or text[:200]
                if key in seen:
                    continue
                seen.add(key)
                fout.write(json.dumps({"text": text, "title": title, "source": "wikipedia"}) + "\n")
                n_docs += 1
                n_chars += len(text)
                if n_docs % 50000 == 0:
                    print(f"    {n_docs:,} docs, ~{n_chars/3.5/1e9:.2f}B tokens")
                if n_chars >= max_chars:
                    break

    print(f"  Wikipedia total: {n_docs:,} docs, ~{n_chars/3.5/1e9:.2f}B tokens")
    return out_path


def _download_cosmopedia(max_tokens):
    """Combine existing Cosmopedia files."""
    out_path = RAW_DIR / "cosmopedia_all.jsonl"
    if out_path.exists():
        n = sum(1 for _ in open(out_path))
        print(f"Cosmopedia: already have {n:,} docs")
        return out_path

    from datasets import load_dataset

    existing = [RAW_DIR / f for f in ["cosmopedia.jsonl", "cosmopedia_e2.jsonl"]
                if (RAW_DIR / f).exists()]

    seen = set()
    n_docs = 0
    n_chars = 0
    max_chars = max_tokens * 3.5

    with open(out_path, "w") as fout:
        for p in existing:
            print(f"  Reading {p.name}...")
            with open(p) as fin:
                for line in fin:
                    doc = json.loads(line)
                    text = doc.get("text", "")
                    key = text[:200]
                    if key in seen:
                        continue
                    seen.add(key)
                    fout.write(line)
                    n_docs += 1
                    n_chars += len(text)

        if n_chars < max_chars:
            print(f"  Downloading more Cosmopedia...")
            ds = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2",
                              split="train", streaming=True)
            for doc in ds:
                text = doc.get("text", "")
                if len(text) < 100:
                    continue
                key = text[:200]
                if key in seen:
                    continue
                seen.add(key)
                fout.write(json.dumps({"text": text, "source": "cosmopedia"}) + "\n")
                n_docs += 1
                n_chars += len(text)
                if n_docs % 50000 == 0:
                    print(f"    {n_docs:,} docs, ~{n_chars/3.5/1e9:.2f}B tokens")
                if n_chars >= max_chars:
                    break

    print(f"  Cosmopedia total: {n_docs:,} docs, ~{n_chars/3.5/1e9:.2f}B tokens")
    return out_path


def _download_starcoderdata(max_tokens):
    """Download code from StarCoder (deduplicated, high quality)."""
    from datasets import load_dataset

    out_path = RAW_DIR / "starcoderdata_all.jsonl"
    if out_path.exists():
        n = sum(1 for _ in open(out_path))
        print(f"StarCoder: already have {n:,} docs")
        return out_path

    # combine existing code files first
    existing = [RAW_DIR / f for f in ["code.jsonl", "code_e2.jsonl"]
                if (RAW_DIR / f).exists()]

    seen = set()
    n_docs = 0
    n_chars = 0
    max_chars = max_tokens * 3.0  # code compresses less

    # target languages (high quality, broad coverage)
    langs = ["python", "javascript", "typescript", "java", "c", "cpp",
             "rust", "go", "shell", "sql", "html", "css", "markdown"]

    with open(out_path, "w") as fout:
        for p in existing:
            print(f"  Reading {p.name}...")
            with open(p) as fin:
                for line in fin:
                    doc = json.loads(line)
                    text = doc.get("text", doc.get("content", ""))
                    if len(text) < 50:
                        continue
                    key = text[:200]
                    if key in seen:
                        continue
                    seen.add(key)
                    fout.write(json.dumps({"text": text, "source": "starcoderdata"}) + "\n")
                    n_docs += 1
                    n_chars += len(text)

        print(f"  Existing code: {n_docs:,} docs, ~{n_chars/3.0/1e9:.2f}B tokens")

        if n_chars < max_chars:
            print(f"  Downloading StarCoder ({len(langs)} languages)...")
            t0 = time.time()
            for lang in langs:
                if n_chars >= max_chars:
                    break
                try:
                    ds = load_dataset("bigcode/starcoderdata", data_dir=lang,
                                      split="train", streaming=True)
                    lang_docs = 0
                    for doc in ds:
                        text = doc.get("content", "")
                        if len(text) < 50 or len(text) > 100000:
                            continue
                        key = text[:200]
                        if key in seen:
                            continue
                        seen.add(key)
                        fout.write(json.dumps({"text": text, "source": "starcoderdata",
                                               "lang": lang}) + "\n")
                        n_docs += 1
                        n_chars += len(text)
                        lang_docs += 1
                        if n_docs % 100000 == 0:
                            print(f"    {n_docs:,} docs ({lang}: {lang_docs:,}), "
                                  f"~{n_chars/3.0/1e9:.2f}B tokens, {time.time()-t0:.0f}s")
                        if n_chars >= max_chars:
                            break
                    print(f"    {lang}: {lang_docs:,} docs")
                except Exception as e:
                    print(f"    {lang}: FAILED ({e})")

    print(f"  StarCoder total: {n_docs:,} docs, ~{n_chars/3.0/1e9:.2f}B tokens")
    return out_path


def _download_openwebmath(max_tokens):
    """Download OpenWebMath for math pretraining data."""
    from datasets import load_dataset

    out_path = RAW_DIR / "openwebmath.jsonl"
    if out_path.exists():
        n = sum(1 for _ in open(out_path))
        est = n * 2000 / 3.5
        print(f"OpenWebMath: already have {n:,} docs (~{est/1e9:.1f}B tokens)")
        if est >= max_tokens * 0.9:
            return out_path

    print(f"Downloading OpenWebMath (target: {max_tokens/1e9:.1f}B tokens)...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)

    n_docs = 0
    n_chars = 0
    max_chars = max_tokens * 3.5
    t0 = time.time()

    with open(out_path, "w") as f:
        for doc in ds:
            text = doc.get("text", "")
            if len(text) < 100:
                continue
            f.write(json.dumps({"text": text, "source": "openwebmath"}) + "\n")
            n_docs += 1
            n_chars += len(text)
            if n_docs % 100000 == 0:
                print(f"  {n_docs:,} docs, ~{n_chars/3.5/1e9:.2f}B tokens, "
                      f"{time.time()-t0:.0f}s")
            if n_chars >= max_chars:
                break

    print(f"  OpenWebMath: {n_docs:,} docs, ~{n_chars/3.5/1e9:.2f}B tokens")
    return out_path


def download_all():
    """Download all data sources."""
    print("=" * 60)
    print("Downloading all data sources")
    print("=" * 60)

    paths = {}

    print("\n--- Web text ---")
    paths["fineweb_edu"] = _download_fineweb_edu(TARGETS["web"]["fineweb_edu"])
    paths["wikipedia"] = _download_wikipedia(TARGETS["web"]["wikipedia"])
    paths["cosmopedia"] = _download_cosmopedia(TARGETS["web"]["cosmopedia"])

    print("\n--- Code ---")
    paths["starcoderdata"] = _download_starcoderdata(TARGETS["code"]["starcoderdata"])

    print("\n--- Math ---")
    paths["openwebmath"] = _download_openwebmath(TARGETS["math"]["openwebmath"])

    return paths


def train_tokenizer():
    """Train BPE tokenizer on combined corpus sample."""
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

    tok_path = DATA_DIR / f"tokenizer_{VOCAB_SIZE}.json"
    if tok_path.exists():
        print(f"Tokenizer already exists: {tok_path}")
        return tok_path

    print(f"Training BPE tokenizer (vocab={VOCAB_SIZE})...")

    # sample ~100M chars from each source
    sample_chars = 100_000_000
    texts = []
    for f in RAW_DIR.glob("*_all.jsonl"):
        chars = 0
        with open(f) as fh:
            for line in fh:
                doc = json.loads(line)
                texts.append(doc.get("text", ""))
                chars += len(texts[-1])
                if chars >= sample_chars:
                    break
        print(f"  Sampled {len(texts)} docs from {f.name} ({chars/1e6:.0f}M chars)")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<pad>", "<eos>"],
        min_frequency=2,
        show_progress=True,
    )
    tokenizer.train_from_iterator(texts, trainer=trainer)
    tokenizer.save(str(tok_path))
    print(f"Tokenizer saved: {tok_path} (vocab={tokenizer.get_vocab_size()})")
    return tok_path


def _tokenize_source(source_name, raw_path, tok_path, target_tokens):
    """Tokenize a source with EOS tokens between documents."""
    from tokenizers import Tokenizer

    cache_path = TOKEN_DIR / f"{source_name}.npz"
    if cache_path.exists():
        data = np.load(cache_path)
        n = len(data["train"]) + len(data["val"])
        print(f"  {source_name}: cached, {n/1e9:.2f}B tokens")
        return source_name, n

    tok = Tokenizer.from_file(str(tok_path))
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)

    print(f"  Tokenizing {source_name} (target: {target_tokens/1e9:.1f}B tokens)...")
    t0 = time.time()

    token_chunks = []
    total_tokens = 0
    batch_texts = []
    batch_chars = 0
    chunk_chars = 50_000_000  # 50M char batches

    with open(raw_path) as f:
        for line in f:
            doc = json.loads(line)
            text = doc.get("text", doc.get("content", ""))
            if len(text) < 50:
                continue
            batch_texts.append(text)
            batch_chars += len(text)

            if batch_chars >= chunk_chars:
                encodings = tok.encode_batch(batch_texts)
                for enc in encodings:
                    ids = enc.ids
                    # append EOS after each document
                    chunk_arr = np.array(ids + [EOS_TOKEN_ID], dtype=np.int32)
                    token_chunks.append(chunk_arr)
                    total_tokens += len(chunk_arr)
                batch_texts = []
                batch_chars = 0
                elapsed = time.time() - t0
                print(f"    {source_name}: {total_tokens/1e6:.0f}M tokens, {elapsed:.0f}s")
                if total_tokens >= target_tokens:
                    break

    # flush remaining
    if batch_texts:
        encodings = tok.encode_batch(batch_texts)
        for enc in encodings:
            ids = enc.ids
            chunk_arr = np.array(ids + [EOS_TOKEN_ID], dtype=np.int32)
            token_chunks.append(chunk_arr)
            total_tokens += len(chunk_arr)

    tokens = np.concatenate(token_chunks)
    del token_chunks

    # split val
    n_val = max(int(len(tokens) * VAL_FRACTION), 10000)
    val_tok = tokens[:n_val]
    train_tok = tokens[n_val:]

    np.savez(cache_path, train=train_tok, val=val_tok)
    print(f"    {source_name}: {len(train_tok)/1e6:.1f}M train + {len(val_tok)/1e6:.1f}M val "
          f"({time.time()-t0:.0f}s)")
    return source_name, len(tokens)


def tokenize_all():
    """Tokenize all sources and combine."""
    tok_path = train_tokenizer()

    # map source names to raw files and targets
    source_configs = {
        "fineweb_edu": (RAW_DIR / "fineweb_edu_all.jsonl", TARGETS["web"]["fineweb_edu"]),
        "wikipedia": (RAW_DIR / "wikipedia_all.jsonl", TARGETS["web"]["wikipedia"]),
        "cosmopedia": (RAW_DIR / "cosmopedia_all.jsonl", TARGETS["web"]["cosmopedia"]),
        "starcoderdata": (RAW_DIR / "starcoderdata_all.jsonl", TARGETS["code"]["starcoderdata"]),
        "openwebmath": (RAW_DIR / "openwebmath.jsonl", TARGETS["math"]["openwebmath"]),
    }

    print("\n--- Tokenizing ---")
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    source_stats = {}

    for name, (raw_path, target) in source_configs.items():
        if not raw_path.exists():
            print(f"  {name}: SKIPPED (no raw data at {raw_path})")
            continue
        sname, n_tokens = _tokenize_source(name, raw_path, str(tok_path), target)
        source_stats[sname] = n_tokens

    # combine: memory-efficient pipeline using flat binary + memmap
    print("\n--- Combining ---")
    train_sizes = {}
    val_sizes = {}
    for name in source_configs:
        cache = TOKEN_DIR / f"{name}.npz"
        if cache.exists():
            data = np.load(cache)
            train_sizes[name] = len(data["train"])
            val_sizes[name] = len(data["val"])
            print(f"  {name}: {train_sizes[name]/1e9:.2f}B train, {val_sizes[name]/1e6:.1f}M val")
            del data

    total_train = sum(train_sizes.values())
    total_val = sum(val_sizes.values())
    print(f"  Total: {total_train/1e9:.2f}B train, {total_val/1e6:.1f}M val")

    # write val (small, fits in memory)
    val_parts = []
    for name in source_configs:
        cache = TOKEN_DIR / f"{name}.npz"
        if cache.exists():
            val_parts.append(np.load(cache)["val"])
    val_combined = np.concatenate(val_parts)
    del val_parts
    np.save(TOKEN_DIR / "val.npy", val_combined)

    # write train to flat binary, one source at a time
    train_bin = TOKEN_DIR / "train.bin"
    print(f"Writing train data to flat binary...")
    with open(train_bin, "wb") as f:
        for name in source_configs:
            cache = TOKEN_DIR / f"{name}.npz"
            if cache.exists():
                arr = np.load(cache)["train"]
                arr.tofile(f)
                print(f"  wrote {name}: {len(arr)/1e9:.2f}B tokens")
                del arr

    # shuffle via memmap: permute 512-token chunk indices, write shuffled output
    print("Shuffling (memory-mapped, chunk-by-chunk)...")
    src = np.memmap(train_bin, dtype=np.int32, mode="r")
    chunk_size = 512
    n_chunks = len(src) // chunk_size
    usable = n_chunks * chunk_size

    rng = np.random.default_rng(42)
    perm = rng.permutation(n_chunks)

    # write shuffled data in batches to avoid loading all into RAM
    shuffled_bin = TOKEN_DIR / "train_shuffled.bin"
    dst = np.memmap(shuffled_bin, dtype=np.int32, mode="w+", shape=(usable,))
    batch = 100000  # process 100K chunks at a time (~200MB)
    for i in range(0, n_chunks, batch):
        end = min(i + batch, n_chunks)
        idx = perm[i:end]
        # gather chunks from source
        for j, ci in enumerate(idx):
            dst[(i+j)*chunk_size:(i+j+1)*chunk_size] = src[ci*chunk_size:(ci+1)*chunk_size]
        if (i // batch) % 10 == 0:
            print(f"  shuffled {end}/{n_chunks} chunks ({end*chunk_size/1e9:.2f}B tokens)")
    dst.flush()
    del src, dst

    # replace original with shuffled
    train_bin.unlink()
    shuffled_bin.rename(train_bin)

    # keep flat binary for memory-mapped loading (no npz — too large for RAM)
    print(f"Final train data: {train_bin} ({usable} tokens, {usable*4/1e9:.1f}GB)")

    total_train_tok = usable
    total_val_tok = len(np.load(TOKEN_DIR / "val.npy"))

    meta = {
        "vocab_size": VOCAB_SIZE,
        "tokenizer_path": str(tok_path.relative_to(Path(__file__).parent)),
        "sources": source_stats,
        "total_train_tokens": total_train_tok,
        "total_val_tokens": total_val_tok,
        "val_fraction": VAL_FRACTION,
        "eos_token_id": EOS_TOKEN_ID,
        "has_eos_between_docs": True,
        "format": "flat_binary",  # train.bin = flat int32 array, val.npy = numpy
    }
    meta_path = TOKEN_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"FINAL DATASET:")
    print(f"  Train: {total_train_tok:,} tokens ({total_train_tok/1e9:.2f}B)")
    print(f"  Val:   {total_val_tok:,} tokens ({total_val_tok/1e6:.1f}M)")
    for name, n in source_stats.items():
        pct = n / sum(source_stats.values()) * 100
        print(f"  {name}: {n/1e9:.2f}B ({pct:.0f}%)")
    print(f"  EOS tokens between all documents")
    print(f"  Format: train.bin (flat int32) + val.npy")
    print(f"{'='*60}")


def show_stats():
    """Show current data statistics."""
    print("=== Raw data ===")
    for f in sorted(RAW_DIR.glob("*.jsonl")):
        n = sum(1 for _ in open(f))
        size = f.stat().st_size / 1e9
        print(f"  {f.name}: {n:,} docs, {size:.2f} GB")

    print("\n=== Tokenized (v2) ===")
    meta_path = TOKEN_DIR / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  Total train: {meta['total_train_tokens']/1e9:.2f}B tokens")
        print(f"  Total val: {meta['total_val_tokens']/1e6:.1f}M tokens")
        print(f"  Vocab: {meta['vocab_size']}")
        print(f"  EOS between docs: {meta.get('has_eos_between_docs', False)}")
        for name, n in meta["sources"].items():
            pct = n / sum(meta["sources"].values()) * 100
            print(f"    {name}: {n/1e9:.2f}B ({pct:.0f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenize-only", action="store_true")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    elif args.tokenize_only:
        tokenize_all()
    else:
        download_all()
        tokenize_all()
