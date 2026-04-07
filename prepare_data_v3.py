"""Prepare training data v3 — expanded dataset for continued training.

Based on research from data_research.md (April 2026).
Targeting a coding/reasoning agent model — keeps original v2 distribution.

Main dataset (~50B tokens, same ratios as v2):
  34% FineWeb-Edu (score >= 3)    17.0B — quality-filtered web
  30% StarCoder code              15.0B — deduplicated, 13 languages
  19% OpenWebMath                  9.5B — math with LaTeX
   9% Wikipedia                    4.5B — full English wiki
   8% Cosmopedia v2                4.0B — synthetic textbooks

Annealing dataset (~3B tokens, highest quality only):
  50% FineWeb-Edu score >= 4       1.5B
  27% FineMath 4+                  0.8B
  23% Stack-Edu (educational code) 0.7B

Changes from v2:
  - 6.4x more total tokens (50B vs 7.85B)
  - Same distribution: heavy code (30%) + math (19%) for reasoning
  - Separate annealing dataset for cooldown phase
  - Reuses existing raw downloads where possible

Usage:
  uv run prepare_data_v3.py                    # download + tokenize everything
  uv run prepare_data_v3.py --tokenize-only    # just retokenize existing raw data
  uv run prepare_data_v3.py --anneal-only      # prepare annealing dataset only
  uv run prepare_data_v3.py --stats            # show data statistics
"""

import json
import os
import time
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"
TOKEN_DIR = DATA_DIR / "tokens_v3"
ANNEAL_DIR = DATA_DIR / "tokens_v3_anneal"
VOCAB_SIZE = 32000
VAL_FRACTION = 0.002  # smaller val fraction since dataset is larger
EOS_TOKEN_ID = 1

# --- Main dataset targets (~50B tokens, same ratios as v2) ---
TARGETS = {
    "web": {
        "fineweb_edu":  17_000_000_000,  # 17B tokens (34%)
        "wikipedia":     4_500_000_000,  # 4.5B tokens (9%)
        "cosmopedia":    4_000_000_000,  # 4B tokens (8%)
    },  # total web: ~25.5B (51%)
    "code": {
        "starcoderdata": 15_000_000_000,  # 15B tokens (30%)
    },
    "math": {
        "openwebmath":   9_500_000_000,  # 9.5B tokens (19%)
    },
}

# --- Annealing dataset targets (~3B tokens) ---
ANNEAL_TARGETS = {
    "fineweb_edu_hq":  1_500_000_000,  # 1.5B — FineWeb-Edu score >= 4
    "finemath":          800_000_000,  # 0.8B — FineMath 4+
    "stack_edu":         700_000_000,  # 0.7B — Stack-Edu (educational code)
}


# ============================================================
# Download functions — main dataset
# ============================================================

def _download_fineweb_edu(max_tokens):
    """Download FineWeb-Edu (score >= 3), reusing existing raw files."""
    from datasets import load_dataset

    out_path = RAW_DIR / "fineweb_edu_v3.jsonl"
    if out_path.exists():
        n = sum(1 for _ in open(out_path))
        est = n * 5000 / 3.5
        print(f"FineWeb-Edu: already have {n:,} docs (~{est/1e9:.1f}B tokens)")
        if est >= max_tokens * 0.9:
            return out_path

    print(f"Downloading FineWeb-Edu (target: {max_tokens/1e9:.1f}B tokens)...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # collect existing raw files for dedup
    existing = [RAW_DIR / f for f in [
        "fineweb_edu.jsonl", "fineweb_edu_e2.jsonl", "fineweb_edu_all.jsonl"
    ] if (RAW_DIR / f).exists()]

    seen = set()
    n_docs = 0
    n_chars = 0
    max_chars = max_tokens * 3.5

    with open(out_path, "w") as fout:
        # reuse existing data
        for p in existing:
            print(f"  Reading {p.name}...")
            with open(p) as fin:
                for line in fin:
                    doc = json.loads(line)
                    text = doc.get("text", "")
                    if len(text) < 100:
                        continue
                    key = text[:200]
                    if key in seen:
                        continue
                    seen.add(key)
                    fout.write(json.dumps({"text": text, "source": "fineweb_edu"}) + "\n")
                    n_docs += 1
                    n_chars += len(text)

        print(f"  Existing: {n_docs:,} unique docs, ~{n_chars/3.5/1e9:.2f}B tokens")

        # download more from HF if needed
        if n_chars < max_chars:
            remaining = (max_chars - n_chars) / 1e9
            print(f"  Downloading more from HuggingFace (need {remaining:.1f}B more chars)...")
            ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-350BT",
                              split="train", streaming=True)
            t0 = time.time()
            for doc in ds:
                score = doc.get("score", 0)
                if score < 3.0:
                    continue
                text = doc.get("text", "")
                if len(text) < 100:
                    continue
                key = text[:200]
                if key in seen:
                    continue
                seen.add(key)
                fout.write(json.dumps({"text": text, "source": "fineweb_edu"}) + "\n")
                n_docs += 1
                n_chars += len(text)
                if n_docs % 100000 == 0:
                    elapsed = time.time() - t0
                    print(f"    {n_docs:,} docs, ~{n_chars/3.5/1e9:.2f}B tokens, {elapsed:.0f}s")
                if n_chars >= max_chars:
                    break

    del seen
    print(f"  FineWeb-Edu total: {n_docs:,} docs, ~{n_chars/3.5/1e9:.2f}B tokens")
    return out_path


def _download_wikipedia(max_tokens):
    """Download Wikipedia, reusing existing raw files."""
    from datasets import load_dataset

    out_path = RAW_DIR / "wikipedia_v3.jsonl"
    if out_path.exists():
        n = sum(1 for _ in open(out_path))
        print(f"Wikipedia: already have {n:,} docs")
        return out_path

    existing = [RAW_DIR / f for f in [
        "wikipedia.jsonl", "wikipedia_e2.jsonl", "wikipedia_all.jsonl"
    ] if (RAW_DIR / f).exists()]

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
                    fout.write(json.dumps({"text": text, "title": title,
                                           "source": "wikipedia"}) + "\n")
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
                fout.write(json.dumps({"text": text, "title": title,
                                       "source": "wikipedia"}) + "\n")
                n_docs += 1
                n_chars += len(text)
                if n_docs % 100000 == 0:
                    print(f"    {n_docs:,} docs, ~{n_chars/3.5/1e9:.2f}B tokens")
                if n_chars >= max_chars:
                    break

    print(f"  Wikipedia total: {n_docs:,} docs, ~{n_chars/3.5/1e9:.2f}B tokens")
    return out_path


def _download_cosmopedia(max_tokens):
    """Download Cosmopedia v2, reusing existing raw files."""
    from datasets import load_dataset

    out_path = RAW_DIR / "cosmopedia_v3.jsonl"
    if out_path.exists():
        n = sum(1 for _ in open(out_path))
        print(f"Cosmopedia: already have {n:,} docs")
        return out_path

    existing = [RAW_DIR / f for f in [
        "cosmopedia.jsonl", "cosmopedia_e2.jsonl", "cosmopedia_all.jsonl"
    ] if (RAW_DIR / f).exists()]

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
                    fout.write(json.dumps({"text": text, "source": "cosmopedia"}) + "\n")
                    n_docs += 1
                    n_chars += len(text)

        if n_chars < max_chars:
            print(f"  Downloading more Cosmopedia v2...")
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
                if n_docs % 100000 == 0:
                    print(f"    {n_docs:,} docs, ~{n_chars/3.5/1e9:.2f}B tokens")
                if n_chars >= max_chars:
                    break

    print(f"  Cosmopedia total: {n_docs:,} docs, ~{n_chars/3.5/1e9:.2f}B tokens")
    return out_path


def _download_starcoderdata(max_tokens):
    """Download StarCoder code data, reusing existing raw files."""
    from datasets import load_dataset

    out_path = RAW_DIR / "starcoderdata_v3.jsonl"
    if out_path.exists():
        n = sum(1 for _ in open(out_path))
        print(f"StarCoder: already have {n:,} docs")
        return out_path

    existing = [RAW_DIR / f for f in [
        "code.jsonl", "code_e2.jsonl", "starcoderdata_all.jsonl"
    ] if (RAW_DIR / f).exists()]

    seen = set()
    n_docs = 0
    n_chars = 0
    max_chars = max_tokens * 3.0  # code compresses less

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
    """Download OpenWebMath, reusing existing raw file."""
    from datasets import load_dataset

    out_path = RAW_DIR / "openwebmath_v3.jsonl"
    existing_path = RAW_DIR / "openwebmath.jsonl"

    if out_path.exists():
        n = sum(1 for _ in open(out_path))
        est = n * 2000 / 3.5
        print(f"OpenWebMath: already have {n:,} docs (~{est/1e9:.1f}B tokens)")
        if est >= max_tokens * 0.9:
            return out_path

    # if we have existing data and just need more, copy + extend
    seen = set()
    n_docs = 0
    n_chars = 0
    max_chars = max_tokens * 3.5

    with open(out_path, "w") as fout:
        if existing_path.exists():
            print(f"  Reading existing openwebmath.jsonl...")
            with open(existing_path) as fin:
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
            print(f"  Existing: {n_docs:,} docs, ~{n_chars/3.5/1e9:.2f}B tokens")

        if n_chars < max_chars:
            print(f"  Downloading more OpenWebMath...")
            ds = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
            t0 = time.time()
            for doc in ds:
                text = doc.get("text", "")
                if len(text) < 100:
                    continue
                key = text[:200]
                if key in seen:
                    continue
                seen.add(key)
                fout.write(json.dumps({"text": text, "source": "openwebmath"}) + "\n")
                n_docs += 1
                n_chars += len(text)
                if n_docs % 100000 == 0:
                    print(f"    {n_docs:,} docs, ~{n_chars/3.5/1e9:.2f}B tokens, "
                          f"{time.time()-t0:.0f}s")
                if n_chars >= max_chars:
                    break

    print(f"  OpenWebMath total: {n_docs:,} docs, ~{n_chars/3.5/1e9:.2f}B tokens")
    return out_path


# ============================================================
# Download functions — annealing dataset
# ============================================================

def _download_fineweb_edu_hq(max_tokens):
    """Download FineWeb-Edu with score >= 4 (higher quality threshold)."""
    from datasets import load_dataset

    out_path = RAW_DIR / "fineweb_edu_hq.jsonl"
    if out_path.exists():
        n = sum(1 for _ in open(out_path))
        est = n * 5000 / 3.5
        print(f"FineWeb-Edu HQ: already have {n:,} docs (~{est/1e9:.1f}B tokens)")
        if est >= max_tokens * 0.9:
            return out_path

    print(f"Downloading FineWeb-Edu score>=4 (target: {max_tokens/1e9:.1f}B tokens)...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                      split="train", streaming=True)

    n_docs = 0
    n_chars = 0
    max_chars = max_tokens * 3.5
    t0 = time.time()

    with open(out_path, "w") as f:
        for doc in ds:
            score = doc.get("score", 0)
            if score < 4.0:  # higher threshold
                continue
            text = doc.get("text", "")
            if len(text) < 100:
                continue
            f.write(json.dumps({"text": text, "source": "fineweb_edu_hq",
                                "score": score}) + "\n")
            n_docs += 1
            n_chars += len(text)
            if n_docs % 50000 == 0:
                print(f"  {n_docs:,} docs, ~{n_chars/3.5/1e9:.2f}B tokens, {time.time()-t0:.0f}s")
            if n_chars >= max_chars:
                break

    print(f"  FineWeb-Edu HQ: {n_docs:,} docs, ~{n_chars/3.5/1e9:.2f}B tokens")
    return out_path


def _download_finemath(max_tokens):
    """Download FineMath 4+ (high-quality math reasoning)."""
    from datasets import load_dataset

    out_path = RAW_DIR / "finemath_4plus.jsonl"
    if out_path.exists():
        n = sum(1 for _ in open(out_path))
        est = n * 2000 / 3.5
        print(f"FineMath 4+: already have {n:,} docs (~{est/1e9:.1f}B tokens)")
        if est >= max_tokens * 0.9:
            return out_path

    print(f"Downloading FineMath 4+ (target: {max_tokens/1e9:.1f}B tokens)...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("HuggingFaceTB/finemath", "finemath-4plus",
                      split="train", streaming=True)

    n_docs = 0
    n_chars = 0
    max_chars = max_tokens * 3.5
    t0 = time.time()

    with open(out_path, "w") as f:
        for doc in ds:
            text = doc.get("text", "")
            if len(text) < 100:
                continue
            f.write(json.dumps({"text": text, "source": "finemath"}) + "\n")
            n_docs += 1
            n_chars += len(text)
            if n_docs % 50000 == 0:
                print(f"  {n_docs:,} docs, ~{n_chars/3.5/1e9:.2f}B tokens, {time.time()-t0:.0f}s")
            if n_chars >= max_chars:
                break

    print(f"  FineMath 4+: {n_docs:,} docs, ~{n_chars/3.5/1e9:.2f}B tokens")
    return out_path


def _download_stack_edu(max_tokens):
    """Download Stack-Edu (educational code from The Stack v2)."""
    from datasets import load_dataset

    out_path = RAW_DIR / "stack_edu.jsonl"
    if out_path.exists():
        n = sum(1 for _ in open(out_path))
        est = n * 3000 / 3.0
        print(f"Stack-Edu: already have {n:,} docs (~{est/1e9:.1f}B tokens)")
        if est >= max_tokens * 0.9:
            return out_path

    print(f"Downloading Stack-Edu (target: {max_tokens/1e9:.1f}B tokens)...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # stack-edu has multiple language configs; start with highest-value ones
    langs = ["python", "javascript", "typescript", "java", "c", "cpp", "rust", "go"]
    n_docs = 0
    n_chars = 0
    max_chars = max_tokens * 3.0
    t0 = time.time()

    with open(out_path, "w") as f:
        for lang in langs:
            if n_chars >= max_chars:
                break
            try:
                ds = load_dataset("HuggingFaceTB/stack-edu", data_dir=lang,
                                  split="train", streaming=True)
                lang_docs = 0
                for doc in ds:
                    text = doc.get("text", doc.get("content", ""))
                    if len(text) < 50 or len(text) > 100000:
                        continue
                    f.write(json.dumps({"text": text, "source": "stack_edu",
                                        "lang": lang}) + "\n")
                    n_docs += 1
                    n_chars += len(text)
                    lang_docs += 1
                    if n_docs % 50000 == 0:
                        print(f"    {n_docs:,} docs ({lang}: {lang_docs:,}), "
                              f"~{n_chars/3.0/1e9:.2f}B tokens, {time.time()-t0:.0f}s")
                    if n_chars >= max_chars:
                        break
                print(f"    {lang}: {lang_docs:,} docs")
            except Exception as e:
                print(f"    {lang}: FAILED ({e})")

    print(f"  Stack-Edu total: {n_docs:,} docs, ~{n_chars/3.0/1e9:.2f}B tokens")
    return out_path


# ============================================================
# Download orchestration
# ============================================================

def download_main():
    """Download all main dataset sources."""
    print("=" * 60)
    print("Downloading main dataset sources (target: ~50B tokens)")
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


def download_anneal():
    """Download annealing dataset sources."""
    print("=" * 60)
    print("Downloading annealing dataset sources (target: ~3B tokens)")
    print("=" * 60)

    paths = {}
    paths["fineweb_edu_hq"] = _download_fineweb_edu_hq(ANNEAL_TARGETS["fineweb_edu_hq"])
    paths["finemath"] = _download_finemath(ANNEAL_TARGETS["finemath"])
    paths["stack_edu"] = _download_stack_edu(ANNEAL_TARGETS["stack_edu"])

    return paths


# ============================================================
# Tokenization (shared with v2)
# ============================================================

def _get_tokenizer_path():
    """Find the existing tokenizer."""
    tok_path = DATA_DIR / f"tokenizer_{VOCAB_SIZE}.json"
    if not tok_path.exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tok_path}. Run prepare_data_v2.py first "
            f"to train the tokenizer, or copy it from data/tokens_v2/."
        )
    return tok_path


def _tokenize_source(source_name, raw_path, tok_path, target_tokens, out_dir):
    """Tokenize a source with EOS tokens between documents. Streams to disk."""
    from tokenizers import Tokenizer

    cache_bin = out_dir / f"{source_name}.bin"
    cache_meta = out_dir / f"{source_name}_meta.json"
    if cache_bin.exists() and cache_meta.exists():
        with open(cache_meta) as f:
            meta = json.load(f)
        n = meta["total_tokens"]
        print(f"  {source_name}: cached, {n/1e9:.2f}B tokens")
        return source_name, n

    tok = Tokenizer.from_file(str(tok_path))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Tokenizing {source_name} (target: {target_tokens/1e9:.1f}B tokens)...")
    t0 = time.time()

    total_tokens = 0
    batch_texts = []
    batch_chars = 0
    chunk_chars = 50_000_000  # 50M char batches

    # stream tokens directly to flat binary file (no RAM accumulation)
    with open(cache_bin, "wb") as fout, open(raw_path) as fin:
        for line in fin:
            doc = json.loads(line)
            text = doc.get("text", doc.get("content", ""))
            if len(text) < 50:
                continue
            batch_texts.append(text)
            batch_chars += len(text)

            if batch_chars >= chunk_chars:
                encodings = tok.encode_batch(batch_texts)
                for enc in encodings:
                    ids = enc.ids + [EOS_TOKEN_ID]
                    arr = np.array(ids, dtype=np.int32)
                    arr.tofile(fout)
                    total_tokens += len(arr)
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
                ids = enc.ids + [EOS_TOKEN_ID]
                arr = np.array(ids, dtype=np.int32)
                arr.tofile(fout)
                total_tokens += len(arr)

    # save metadata
    with open(cache_meta, "w") as f:
        json.dump({"total_tokens": total_tokens}, f)

    elapsed = time.time() - t0
    print(f"    {source_name}: {total_tokens/1e9:.2f}B tokens ({elapsed:.0f}s)")
    return source_name, total_tokens


def _combine_tokenized(source_configs, out_dir, dataset_name="main"):
    """Combine tokenized sources into flat binary + shuffled output."""
    print(f"\n--- Combining {dataset_name} dataset ---")
    source_sizes = {}

    for name in source_configs:
        meta_file = out_dir / f"{name}_meta.json"
        if meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            source_sizes[name] = meta["total_tokens"]
            print(f"  {name}: {source_sizes[name]/1e9:.2f}B tokens")

    total_all = sum(source_sizes.values())
    n_val_total = max(int(total_all * VAL_FRACTION), 100000)
    # distribute val tokens proportionally across sources
    val_per_source = {}
    for name, n in source_sizes.items():
        val_per_source[name] = max(int(n_val_total * n / total_all), 10000)

    print(f"  Total: {total_all/1e9:.2f}B tokens, val: {sum(val_per_source.values())/1e6:.1f}M")

    # write val (small — read first N tokens from each source .bin)
    val_parts = []
    for name in source_configs:
        src_bin = out_dir / f"{name}.bin"
        if src_bin.exists() and name in val_per_source:
            src = np.memmap(src_bin, dtype=np.int32, mode="r")
            n_val = min(val_per_source[name], len(src))
            val_parts.append(np.array(src[:n_val]))
            del src
    val_combined = np.concatenate(val_parts)
    del val_parts
    np.save(out_dir / "val.npy", val_combined)

    # write train to flat binary (skip val tokens from each source)
    train_bin = out_dir / "train.bin"
    print(f"Writing train data to flat binary...")
    total_train = 0
    source_stats = {}
    with open(train_bin, "wb") as f:
        for name in source_configs:
            src_bin = out_dir / f"{name}.bin"
            if src_bin.exists() and name in val_per_source:
                src = np.memmap(src_bin, dtype=np.int32, mode="r")
                skip = val_per_source[name]
                # stream in 100M token chunks to avoid loading all into RAM
                chunk = 100_000_000
                written = 0
                for start in range(skip, len(src), chunk):
                    end = min(start + chunk, len(src))
                    np.array(src[start:end]).tofile(f)
                    written += end - start
                source_stats[name] = source_sizes[name]
                total_train += written
                print(f"  wrote {name}: {written/1e9:.2f}B train tokens")
                del src

    # shuffle via memmap
    print("Shuffling (memory-mapped, chunk-by-chunk)...")
    src = np.memmap(train_bin, dtype=np.int32, mode="r")
    chunk_size = 512
    n_chunks = len(src) // chunk_size
    usable = n_chunks * chunk_size

    rng = np.random.default_rng(42)
    perm = rng.permutation(n_chunks)

    shuffled_bin = out_dir / "train_shuffled.bin"
    dst = np.memmap(shuffled_bin, dtype=np.int32, mode="w+", shape=(usable,))
    batch = 100000
    for i in range(0, n_chunks, batch):
        end = min(i + batch, n_chunks)
        idx = perm[i:end]
        for j, ci in enumerate(idx):
            dst[(i+j)*chunk_size:(i+j+1)*chunk_size] = src[ci*chunk_size:(ci+1)*chunk_size]
        if (i // batch) % 10 == 0:
            print(f"  shuffled {end}/{n_chunks} chunks ({end*chunk_size/1e9:.2f}B tokens)")
    dst.flush()
    del src, dst

    train_bin.unlink()
    shuffled_bin.rename(train_bin)

    total_train_tok = usable
    total_val_tok = len(val_combined)

    # save metadata
    tok_path = _get_tokenizer_path()
    meta = {
        "vocab_size": VOCAB_SIZE,
        "tokenizer_path": str(tok_path.relative_to(Path(__file__).parent)),
        "sources": source_stats,
        "total_train_tokens": total_train_tok,
        "total_val_tokens": total_val_tok,
        "val_fraction": VAL_FRACTION,
        "eos_token_id": EOS_TOKEN_ID,
        "has_eos_between_docs": True,
        "format": "flat_binary",
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"FINAL {dataset_name.upper()} DATASET:")
    print(f"  Train: {total_train_tok:,} tokens ({total_train_tok/1e9:.2f}B)")
    print(f"  Val:   {total_val_tok:,} tokens ({total_val_tok/1e6:.1f}M)")
    for name, n in source_stats.items():
        pct = n / sum(source_stats.values()) * 100
        print(f"  {name}: {n/1e9:.2f}B ({pct:.0f}%)")
    print(f"  Format: train.bin (flat int32) + val.npy")
    print(f"{'='*60}")


def tokenize_main():
    """Tokenize all main sources and combine."""
    tok_path = _get_tokenizer_path()

    # raw file paths for each source
    source_configs = {
        "fineweb_edu":   (RAW_DIR / "fineweb_edu_v3.jsonl",  TARGETS["web"]["fineweb_edu"]),
        "wikipedia":     (RAW_DIR / "wikipedia_v3.jsonl",     TARGETS["web"]["wikipedia"]),
        "cosmopedia":    (RAW_DIR / "cosmopedia_v3.jsonl",    TARGETS["web"]["cosmopedia"]),
        "starcoderdata": (RAW_DIR / "starcoderdata_v3.jsonl", TARGETS["code"]["starcoderdata"]),
        "openwebmath":   (RAW_DIR / "openwebmath_v3.jsonl",   TARGETS["math"]["openwebmath"]),
    }

    print("\n--- Tokenizing main dataset ---")
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)

    for name, (raw_path, target) in source_configs.items():
        if not raw_path.exists():
            print(f"  {name}: SKIPPED (no raw data at {raw_path})")
            continue
        _tokenize_source(name, raw_path, str(tok_path), target, TOKEN_DIR)

    _combine_tokenized(source_configs, TOKEN_DIR, "main")


def tokenize_anneal():
    """Tokenize annealing sources and combine."""
    tok_path = _get_tokenizer_path()

    source_configs = {
        "fineweb_edu_hq": (RAW_DIR / "fineweb_edu_hq.jsonl", ANNEAL_TARGETS["fineweb_edu_hq"]),
        "finemath":       (RAW_DIR / "finemath_4plus.jsonl",  ANNEAL_TARGETS["finemath"]),
        "stack_edu":      (RAW_DIR / "stack_edu.jsonl",       ANNEAL_TARGETS["stack_edu"]),
    }

    print("\n--- Tokenizing annealing dataset ---")
    ANNEAL_DIR.mkdir(parents=True, exist_ok=True)

    for name, (raw_path, target) in source_configs.items():
        if not raw_path.exists():
            print(f"  {name}: SKIPPED (no raw data at {raw_path})")
            continue
        _tokenize_source(name, raw_path, str(tok_path), target, ANNEAL_DIR)

    _combine_tokenized(source_configs, ANNEAL_DIR, "annealing")


def show_stats():
    """Show current data statistics for all versions."""
    for label, tdir in [("v2", DATA_DIR / "tokens_v2"),
                        ("v3 main", TOKEN_DIR),
                        ("v3 anneal", ANNEAL_DIR)]:
        meta_path = tdir / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"\n=== {label} ===")
        print(f"  Total train: {meta['total_train_tokens']/1e9:.2f}B tokens")
        print(f"  Total val: {meta['total_val_tokens']/1e6:.1f}M tokens")
        for name, n in meta["sources"].items():
            pct = n / sum(meta["sources"].values()) * 100
            print(f"    {name}: {n/1e9:.2f}B ({pct:.0f}%)")

    print("\n=== Raw data ===")
    if RAW_DIR.exists():
        for f in sorted(RAW_DIR.glob("*.jsonl")):
            n = sum(1 for _ in open(f))
            size = f.stat().st_size / 1e9
            print(f"  {f.name}: {n:,} docs, {size:.2f} GB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenize-only", action="store_true",
                        help="Just retokenize existing raw data")
    parser.add_argument("--anneal-only", action="store_true",
                        help="Prepare annealing dataset only")
    parser.add_argument("--stats", action="store_true",
                        help="Show data statistics")
    args = parser.parse_args()

    if args.stats:
        show_stats()
    elif args.anneal_only:
        download_anneal()
        tokenize_anneal()
    elif args.tokenize_only:
        tokenize_main()
    else:
        download_main()
        tokenize_main()
        print("\n\nTo also prepare annealing data, run:")
        print("  uv run prepare_data_v3.py --anneal-only")
