"""Prepare training data v3 — expanded dataset for continued training.

Main dataset (~50B tokens, same ratios as v2):
  34% FineWeb-Edu (score >= 3)    17.0B
  30% StarCoder code              15.0B — 13 languages
  19% OpenWebMath                  9.5B — math with LaTeX
   9% Wikipedia                    4.5B
   8% Cosmopedia v2                4.0B — synthetic textbooks

Annealing dataset (~3B tokens, highest quality only):
  50% FineWeb-Edu score >= 4       1.5B
  27% FineMath 4+                  0.8B
  23% Stack-Edu (educational code) 0.7B

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
VAL_FRACTION = 0.002
EOS_TOKEN_ID = 1

# --- Main dataset sources ---
MAIN_SOURCES = {
    "fineweb_edu":  {"tokens": 17_000_000_000, "chars_per_tok": 3.5, "min_len": 100},
    "wikipedia":    {"tokens": 4_500_000_000,  "chars_per_tok": 3.5, "min_len": 200},
    "cosmopedia":   {"tokens": 4_000_000_000,  "chars_per_tok": 3.5, "min_len": 100},
    "starcoderdata":{"tokens": 15_000_000_000, "chars_per_tok": 3.0, "min_len": 50},
    "openwebmath":  {"tokens": 9_500_000_000,  "chars_per_tok": 3.5, "min_len": 100},
}

# --- Annealing dataset sources ---
ANNEAL_SOURCES = {
    "fineweb_edu_hq": {"tokens": 1_500_000_000, "chars_per_tok": 3.5, "min_len": 100},
    "finemath":       {"tokens": 800_000_000,   "chars_per_tok": 3.5, "min_len": 100},
    "stack_edu":      {"tokens": 700_000_000,   "chars_per_tok": 3.0, "min_len": 50},
}

# existing raw files to dedup from
EXISTING_FILES = {
    "fineweb_edu":   ["fineweb_edu.jsonl", "fineweb_edu_e2.jsonl", "fineweb_edu_all.jsonl"],
    "wikipedia":     ["wikipedia.jsonl", "wikipedia_e2.jsonl", "wikipedia_all.jsonl"],
    "cosmopedia":    ["cosmopedia.jsonl", "cosmopedia_e2.jsonl", "cosmopedia_all.jsonl"],
    "starcoderdata": ["code.jsonl", "code_e2.jsonl", "starcoderdata_all.jsonl"],
    "openwebmath":   ["openwebmath.jsonl"],
}

# output filenames for v3 raw downloads
RAW_FILENAMES = {
    "fineweb_edu":    "fineweb_edu_v3.jsonl",
    "wikipedia":      "wikipedia_v3.jsonl",
    "cosmopedia":     "cosmopedia_v3.jsonl",
    "starcoderdata":  "starcoderdata_v3.jsonl",
    "openwebmath":    "openwebmath_v3.jsonl",
    "fineweb_edu_hq": "fineweb_edu_hq.jsonl",
    "finemath":       "finemath_4plus.jsonl",
    "stack_edu":      "stack_edu.jsonl",
}


# ============================================================
# Download helpers
# ============================================================

def _dedup_key(doc, source):
    if source == "wikipedia":
        return doc.get("title") or doc.get("text", "")[:200]
    return doc.get("text", doc.get("content", ""))[:200]


def _read_existing(source, seen, fout, min_len):
    """Read existing raw files for dedup. Returns (n_docs, n_chars)."""
    n_docs = n_chars = 0
    for fname in EXISTING_FILES.get(source, []):
        p = RAW_DIR / fname
        if not p.exists():
            continue
        print(f"  Reading {fname}...")
        with open(p) as fin:
            for line in fin:
                doc = json.loads(line)
                text = doc.get("text", doc.get("content", ""))
                if len(text) < min_len:
                    continue
                key = _dedup_key(doc, source)
                if key in seen:
                    continue
                seen.add(key)
                out = {"text": text, "source": source}
                if source == "wikipedia" and "title" in doc:
                    out["title"] = doc["title"]
                fout.write(json.dumps(out) + "\n")
                n_docs += 1
                n_chars += len(text)
    return n_docs, n_chars


def _hf_stream(source):
    """Return HuggingFace streaming dataset for a source."""
    from datasets import load_dataset
    if source == "fineweb_edu":
        return load_dataset("HuggingFaceFW/fineweb-edu", "sample-350BT",
                            split="train", streaming=True)
    if source == "fineweb_edu_hq":
        return load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                            split="train", streaming=True)
    if source == "wikipedia":
        return load_dataset("wikimedia/wikipedia", "20231101.en",
                            split="train", streaming=True)
    if source == "cosmopedia":
        return load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2",
                            split="train", streaming=True)
    if source == "openwebmath":
        return load_dataset("open-web-math/open-web-math", split="train", streaming=True)
    if source == "finemath":
        return load_dataset("HuggingFaceTB/finemath", "finemath-4plus",
                            split="train", streaming=True)
    assert False, f"unknown source: {source}"


def _score_filter(source, doc):
    """Return True if doc passes source-specific score filter."""
    if source == "fineweb_edu":
        return doc.get("score", 0) >= 3.0
    if source == "fineweb_edu_hq":
        return doc.get("score", 0) >= 4.0
    return True


def _download_source(source, cfg):
    """Download a single source, deduplicating against existing raw files."""
    out_path = RAW_DIR / RAW_FILENAMES[source]
    max_chars = cfg["tokens"] * cfg["chars_per_tok"]

    if out_path.exists():
        n = sum(1 for _ in open(out_path))
        print(f"{source}: already have {n:,} docs")
        return out_path

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    seen = set()

    with open(out_path, "w") as fout:
        n_docs, n_chars = _read_existing(source, seen, fout, cfg["min_len"])
        if n_docs:
            print(f"  Existing: {n_docs:,} docs, ~{n_chars/cfg['chars_per_tok']/1e9:.2f}B tokens")

        if n_chars < max_chars:
            print(f"  Downloading {source}...")
            t0 = time.time()
            ds = _hf_stream(source)
            for doc in ds:
                if not _score_filter(source, doc):
                    continue
                text = doc.get("text", doc.get("content", ""))
                if len(text) < cfg["min_len"]:
                    continue
                key = _dedup_key(doc, source)
                if key in seen:
                    continue
                seen.add(key)
                fout.write(json.dumps({"text": text, "source": source}) + "\n")
                n_docs += 1
                n_chars += len(text)
                if n_docs % 100000 == 0:
                    print(f"    {n_docs:,} docs, ~{n_chars/cfg['chars_per_tok']/1e9:.2f}B tokens, "
                          f"{time.time()-t0:.0f}s")
                if n_chars >= max_chars:
                    break

    del seen
    print(f"  {source}: {n_docs:,} docs, ~{n_chars/cfg['chars_per_tok']/1e9:.2f}B tokens")
    return out_path


def _download_multilang(source, cfg, hf_repo, langs, max_doc_len=100000):
    """Download multi-language code source (StarCoder, Stack-Edu)."""
    from datasets import load_dataset

    out_path = RAW_DIR / RAW_FILENAMES[source]
    max_chars = cfg["tokens"] * cfg["chars_per_tok"]

    if out_path.exists():
        n = sum(1 for _ in open(out_path))
        print(f"{source}: already have {n:,} docs")
        return out_path

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    seen = set()

    with open(out_path, "w") as fout:
        n_docs, n_chars = _read_existing(source, seen, fout, cfg["min_len"])
        if n_docs:
            print(f"  Existing: {n_docs:,} docs, ~{n_chars/cfg['chars_per_tok']/1e9:.2f}B tokens")

        print(f"  Downloading {source} ({len(langs)} languages)...")
        t0 = time.time()
        for lang in langs:
            if n_chars >= max_chars:
                break
            try:
                ds = load_dataset(hf_repo, data_dir=lang, split="train", streaming=True)
                lang_docs = 0
                for doc in ds:
                    text = doc.get("content", doc.get("text", ""))
                    if len(text) < cfg["min_len"] or len(text) > max_doc_len:
                        continue
                    key = text[:200]
                    if key in seen:
                        continue
                    seen.add(key)
                    fout.write(json.dumps({"text": text, "source": source}) + "\n")
                    n_docs += 1
                    n_chars += len(text)
                    lang_docs += 1
                    if n_docs % 100000 == 0:
                        print(f"    {n_docs:,} docs ({lang}: {lang_docs:,}), "
                              f"~{n_chars/cfg['chars_per_tok']/1e9:.2f}B tokens, {time.time()-t0:.0f}s")
                    if n_chars >= max_chars:
                        break
                print(f"    {lang}: {lang_docs:,} docs")
            except Exception as e:
                print(f"    {lang}: FAILED ({e})")

    del seen
    print(f"  {source} total: {n_docs:,} docs, ~{n_chars/cfg['chars_per_tok']/1e9:.2f}B tokens")
    return out_path


CODE_LANGS = ["python", "javascript", "typescript", "java", "c", "cpp",
              "rust", "go", "shell", "sql", "html", "css", "markdown"]
EDU_CODE_LANGS = ["python", "javascript", "typescript", "java", "c", "cpp", "rust", "go"]


# ============================================================
# Download orchestration
# ============================================================

def download_main():
    """Download all main dataset sources."""
    print("=" * 60)
    print("Downloading main dataset sources (target: ~50B tokens)")
    print("=" * 60)
    paths = {}
    for src in ["fineweb_edu", "wikipedia", "cosmopedia", "openwebmath"]:
        paths[src] = _download_source(src, MAIN_SOURCES[src])
    paths["starcoderdata"] = _download_multilang(
        "starcoderdata", MAIN_SOURCES["starcoderdata"],
        "bigcode/starcoderdata", CODE_LANGS)
    return paths


def download_anneal():
    """Download annealing dataset sources."""
    print("=" * 60)
    print("Downloading annealing dataset sources (target: ~3B tokens)")
    print("=" * 60)
    paths = {}
    for src in ["fineweb_edu_hq", "finemath"]:
        paths[src] = _download_source(src, ANNEAL_SOURCES[src])
    paths["stack_edu"] = _download_multilang(
        "stack_edu", ANNEAL_SOURCES["stack_edu"],
        "HuggingFaceTB/stack-edu", EDU_CODE_LANGS)
    return paths


# ============================================================
# Tokenization
# ============================================================

def _get_tokenizer_path():
    """Find the existing tokenizer."""
    tok_path = DATA_DIR / f"tokenizer_{VOCAB_SIZE}.json"
    assert tok_path.exists(), \
        f"Tokenizer not found at {tok_path}. Run prepare_data_v2.py first."
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
        if n > 0:
            print(f"  {source_name}: cached, {n/1e9:.2f}B tokens")
            return source_name, n
        cache_bin.unlink()
        cache_meta.unlink()
        print(f"  {source_name}: stale empty cache, re-tokenizing...")

    tok = Tokenizer.from_file(str(tok_path))
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Tokenizing {source_name} (target: {target_tokens/1e9:.1f}B tokens)...")
    t0 = time.time()
    total_tokens = 0
    batch_texts = []
    batch_chars = 0

    with open(cache_bin, "wb") as fout, open(raw_path) as fin:
        for line in fin:
            doc = json.loads(line)
            text = doc.get("text", doc.get("content", ""))
            if len(text) < 50:
                continue
            batch_texts.append(text)
            batch_chars += len(text)

            if batch_chars >= 50_000_000:
                for enc in tok.encode_batch(batch_texts):
                    arr = np.array(enc.ids + [EOS_TOKEN_ID], dtype=np.int32)
                    arr.tofile(fout)
                    total_tokens += len(arr)
                batch_texts = []
                batch_chars = 0
                print(f"    {source_name}: {total_tokens/1e6:.0f}M tokens, {time.time()-t0:.0f}s")
                if total_tokens >= target_tokens:
                    break

        if batch_texts:
            for enc in tok.encode_batch(batch_texts):
                arr = np.array(enc.ids + [EOS_TOKEN_ID], dtype=np.int32)
                arr.tofile(fout)
                total_tokens += len(arr)

    with open(cache_meta, "w") as f:
        json.dump({"total_tokens": total_tokens}, f)

    print(f"    {source_name}: {total_tokens/1e9:.2f}B tokens ({time.time()-t0:.0f}s)")
    return source_name, total_tokens


def _combine_tokenized(source_cfgs, out_dir, dataset_name):
    """Combine tokenized sources into shuffled flat binary + val split."""
    print(f"\n--- Combining {dataset_name} dataset ---")
    source_sizes = {}
    for name in source_cfgs:
        meta_file = out_dir / f"{name}_meta.json"
        if not meta_file.exists():
            continue
        with open(meta_file) as f:
            n = json.load(f)["total_tokens"]
        if n == 0:
            continue
        source_sizes[name] = n
        print(f"  {name}: {n/1e9:.2f}B tokens")

    total = sum(source_sizes.values())
    n_val_total = max(int(total * VAL_FRACTION), 100000)
    val_per_source = {name: max(int(n_val_total * n / total), 10000)
                      for name, n in source_sizes.items()}

    # write val
    val_parts = []
    for name in source_sizes:
        src = np.memmap(out_dir / f"{name}.bin", dtype=np.int32, mode="r")
        val_parts.append(np.array(src[:val_per_source[name]]))
        del src
    val_combined = np.concatenate(val_parts)
    del val_parts
    np.save(out_dir / "val.npy", val_combined)

    # write train (skip val tokens from each source)
    train_bin = out_dir / "train.bin"
    print("Writing train data...")
    total_train = 0
    with open(train_bin, "wb") as f:
        for name in source_sizes:
            src = np.memmap(out_dir / f"{name}.bin", dtype=np.int32, mode="r")
            skip = val_per_source[name]
            chunk = 100_000_000
            written = 0
            for start in range(skip, len(src), chunk):
                np.array(src[start:min(start + chunk, len(src))]).tofile(f)
                written += min(start + chunk, len(src)) - start
            total_train += written
            print(f"  wrote {name}: {written/1e9:.2f}B train tokens")
            del src

    # shuffle via memmap
    print("Shuffling...")
    src = np.memmap(train_bin, dtype=np.int32, mode="r")
    n_chunks = len(src) // 512
    usable = n_chunks * 512
    perm = np.random.default_rng(42).permutation(n_chunks)

    shuffled_bin = out_dir / "train_shuffled.bin"
    dst = np.memmap(shuffled_bin, dtype=np.int32, mode="w+", shape=(usable,))
    batch = 100000
    for i in range(0, n_chunks, batch):
        end = min(i + batch, n_chunks)
        for j, ci in enumerate(perm[i:end]):
            dst[(i+j)*512:(i+j+1)*512] = src[ci*512:(ci+1)*512]
        if (i // batch) % 10 == 0:
            print(f"  shuffled {end}/{n_chunks} chunks ({end*512/1e9:.2f}B tokens)")
    dst.flush()
    del src, dst

    train_bin.unlink()
    shuffled_bin.rename(train_bin)

    tok_path = _get_tokenizer_path()
    meta = {
        "vocab_size": VOCAB_SIZE,
        "tokenizer_path": str(tok_path.relative_to(Path(__file__).parent)),
        "sources": source_sizes,
        "total_train_tokens": usable,
        "total_val_tokens": len(val_combined),
        "val_fraction": VAL_FRACTION,
        "eos_token_id": EOS_TOKEN_ID,
        "has_eos_between_docs": True,
        "format": "flat_binary",
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}")
    print(f"FINAL {dataset_name.upper()} DATASET:")
    print(f"  Train: {usable:,} tokens ({usable/1e9:.2f}B)")
    print(f"  Val:   {len(val_combined):,} tokens ({len(val_combined)/1e6:.1f}M)")
    for name, n in source_sizes.items():
        pct = n / total * 100
        print(f"  {name}: {n/1e9:.2f}B ({pct:.0f}%)")
    print(f"{'='*60}")


def tokenize_main():
    """Tokenize all main sources and combine."""
    tok_path = _get_tokenizer_path()
    print("\n--- Tokenizing main dataset ---")
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    for name, cfg in MAIN_SOURCES.items():
        raw = RAW_DIR / RAW_FILENAMES[name]
        if not raw.exists():
            print(f"  {name}: SKIPPED (no raw data)")
            continue
        _tokenize_source(name, raw, str(tok_path), cfg["tokens"], TOKEN_DIR)
    _combine_tokenized(MAIN_SOURCES, TOKEN_DIR, "main")


def tokenize_anneal():
    """Tokenize annealing sources and combine."""
    tok_path = _get_tokenizer_path()
    print("\n--- Tokenizing annealing dataset ---")
    ANNEAL_DIR.mkdir(parents=True, exist_ok=True)
    for name, cfg in ANNEAL_SOURCES.items():
        raw = RAW_DIR / RAW_FILENAMES[name]
        if not raw.exists():
            print(f"  {name}: SKIPPED (no raw data)")
            continue
        _tokenize_source(name, raw, str(tok_path), cfg["tokens"], ANNEAL_DIR)
    _combine_tokenized(ANNEAL_SOURCES, ANNEAL_DIR, "annealing")


def show_stats():
    """Show current data statistics."""
    for label, tdir in [("v2", DATA_DIR / "tokens_v2"),
                        ("v3 main", TOKEN_DIR),
                        ("v3 anneal", ANNEAL_DIR)]:
        meta_path = tdir / "metadata.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"\n=== {label} ===")
        print(f"  Train: {meta['total_train_tokens']/1e9:.2f}B tokens")
        print(f"  Val: {meta['total_val_tokens']/1e6:.1f}M tokens")
        for name, n in meta["sources"].items():
            pct = n / sum(meta["sources"].values()) * 100
            print(f"    {name}: {n/1e9:.2f}B ({pct:.0f}%)")

    if RAW_DIR.exists():
        print("\n=== Raw data ===")
        for f in sorted(RAW_DIR.glob("*.jsonl")):
            n = sum(1 for _ in open(f))
            print(f"  {f.name}: {n:,} docs, {f.stat().st_size/1e9:.2f} GB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenize-only", action="store_true")
    parser.add_argument("--anneal-only", action="store_true")
    parser.add_argument("--stats", action="store_true")
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
        print("\nTo also prepare annealing data, run:")
        print("  uv run prepare_data_v3.py --anneal-only")
