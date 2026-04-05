# Remote Training Instructions — RTX 4090 (24GB)

## Your mission

Train this transformer model as fast as possible. Minimize wall-clock training time.
Everything is implemented and tested. Your job is:
1. Set up the environment and download data
2. Tune batch size to maximize GPU utilization on 24GB VRAM
3. Profile and optimize any bottlenecks (XLA flags, compilation, etc.)
4. Run the training to completion with minimum ETA

## What this project is

A 306M parameter decoder-only transformer (JAX + Triton), trained from scratch on 7.85B
tokens (web + code + math). The codebase includes custom Triton inference kernels, but
training uses JAX/XLA with cuDNN FlashAttention.

## Architecture

- d_model=1024, n_heads=16, n_kv_heads=4 (GQA), n_layers=24, ctx=512
- RMSNorm, RoPE, SwiGLU FFN (d_ff=2816), no biases, tied embeddings
- Vocab: 32K BPE tokenizer
- 306M params total

## Data (already prepared, 31.4GB)

7.85B unique tokens with EOS between documents:
- 34% FineWeb-Edu (quality-filtered web, score >= 3)
- 30% StarCoder code (13 languages)
- 19% OpenWebMath (math with LaTeX)
-  9% Wikipedia
-  8% Cosmopedia (synthetic textbooks)

Stored as `data/tokens_v2/train.bin` (flat int32 binary, memory-mapped) + `data/tokens_v2/val.npy`.

## Setup

The repo is already cloned. You're running inside it.

```bash
# 1. Install dependencies (uses uv — should already be available)
uv sync

# 2. Download and prepare all training data
# This downloads from HuggingFace, tokenizes everything, and combines.
# Takes 1-2 hours depending on network speed. Needs ~50GB disk, ~16GB RAM.
# Downloads: FineWeb-Edu, Wikipedia, Cosmopedia, StarCoder, OpenWebMath
uv run python -u prepare_data_v2.py

# This creates:
#   data/tokens_v2/train.bin        (31.4 GB — shuffled training tokens)
#   data/tokens_v2/val.npy          (150 MB — validation tokens)
#   data/tokens_v2/metadata.json    (dataset metadata)
#   data/tokenizer_32000.json       (BPE tokenizer, trained on the corpus)
#
# The script is idempotent: if interrupted, re-run and it skips completed steps.
# Each source is cached individually in data/tokens_v2/{source}.npz.
```

### If data download is slow

The data pipeline streams from HuggingFace datasets. If the server has slow internet,
the bottleneck will be downloading ~15GB of raw text. The tokenization and combining
steps are CPU-bound and take ~30 min total.

To speed up downloads, make sure `HF_HUB_ENABLE_HF_TRANSFER=1` is set:
```bash
pip install hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 uv run python -u prepare_data_v2.py
```

## Training command

```bash
# RECOMMENDED: No MTP, no gradient checkpointing, maximize batch size for 24GB
uv run python -u train.py \
  --d-model 1024 --n-heads 16 --n-kv-heads 4 --n-layers 24 \
  --context-len 512 --batch-size 32 --epochs 3 \
  --dataset combined_v2 --curriculum --lr 3e-4 --no-checkpoint \
  2>&1 | tee training.log
```

## Tuning for maximum speed

### --no-checkpoint (gradient checkpointing)
`--no-checkpoint` disables gradient checkpointing. +12% speed, but more VRAM.
On 4080 Super (16GB): 6.0GB with checkpoint → 11.2GB without. Safe.
On 4090 (24GB): easily fits. Always use `--no-checkpoint`.

### Batch size
On 4080 Super (16GB): bs=16 used 11.2GB VRAM (with --no-checkpoint).
On 4090 (24GB): **start with bs=32** (should use ~20GB). If it fits, try bs=48.
If OOM, fall back to bs=24.
NOTE: larger batch doesn't increase tok/s (GPU is compute-saturated), but
fewer optimizer steps may slightly affect convergence.

The curriculum flag multiplies batch size in early phases:
- Phase 1 (10%): ctx=128, bs=bs×4 (e.g., 128 at bs=32)
- Phase 2 (20%): ctx=256, bs=bs×2 (e.g., 64 at bs=32)
- Phase 3 (70%): ctx=512, bs=bs (e.g., 32)

So with bs=32, phase 1 uses bs=128 at ctx=128. Check that this fits in VRAM.
If phase 1 OOMs, reduce base bs until all phases fit.

### MTP (multi-token prediction)
Adding `--mtp-heads 3` improves model quality but adds ~56% training time
(3 extra passes over 32K vocab per step). Skip it for speed. Add it later
if you want better quality.

### JAX cache
Create the cache dir to avoid compilation warnings:
```bash
mkdir -p .jax_cache
```

### Disable XLA debug output
```bash
export XLA_FLAGS="--xla_gpu_autotune_level=0"
export TF_CPP_MIN_LOG_LEVEL=3
```

### Monitor
```bash
# In another terminal:
tail -f training.log | grep "step "
nvidia-smi -l 5  # watch GPU utilization
```

## Expected performance

Baseline on RTX 4080 Super (16GB) with bs=16, no MTP: 28.4K tok/s.
The 4090 has 1.6x compute and 24GB VRAM (larger batch possible).
Your goal: beat the baseline significantly and reach minimum ETA.

Total training: 23.5B tokens (3 epochs × 7.85B).
Loss should reach ~3.0-3.5 by end of training.

## Key files

```
train.py                    — training script (AdamW, cosine LR, curriculum)
model.py                    — JAX model (forward, fused CE, MTP)
data.py                     — data loading (streaming v2 support)
prepare_data_v2.py          — data pipeline (already run, don't need to re-run)
program.md                  — full project history and architecture docs
```

## What NOT to change

- Don't change the model architecture (it's been carefully designed)
- Don't change the data pipeline (already optimized)
- Don't change the tokenizer (data is pre-tokenized with it)
- Don't add quantization (project goal is learning GPU kernels, not shrinking)
- Focus ONLY on training speed: batch size, XLA flags, compilation, etc.

## After training completes

The model saves to `weights.pkl` (~1.2GB). Push it or make it available for download:
```bash
# Option 1: push to a release/artifact
# Option 2: upload to HuggingFace Hub
# Option 3: any file sharing method
```

## Checkpointing

Checkpoints are saved automatically every 2000 steps (configurable via `--checkpoint-interval`)
and at the end of each epoch. Saves to `checkpoint.pkl` (~3.6GB: params + optimizer state).

To resume from a checkpoint:
```bash
uv run python -u train.py \
  --d-model 1024 --n-heads 16 --n-kv-heads 4 --n-layers 24 \
  --context-len 512 --batch-size 32 --epochs 3 \
  --dataset combined_v2 --curriculum --lr 3e-4 --no-checkpoint \
  --resume checkpoint.pkl \
  2>&1 | tee -a training.log
```

This restores params, optimizer state (Adam moments), LR schedule position, and
exact epoch/batch position. Training continues exactly where it left off.
