# Single-GPU Transformer

Training and inference for a decoder-only transformer, built from scratch. Custom Triton kernels for decode. JAX for training. No frameworks, no shortcuts.

Built using [karpathy/autoresearch](https://github.com/karpathy/autoresearch)-style autonomous development — a coding agent is pointed at `program.md` repeatedly. The human steers direction; the agent handles implementation, debugging, benchmarking, and documentation.

See [`repo_explained_from_zero.md`](repo_explained_from_zero.md) for a ground-up explanation of GPU kernels, register pressure, memory hierarchies, and all techniques used.

## Current Model

```
306M params (d=1024, h=16, kv=4, l=24, ctx=512)
  RMSNorm, RoPE, SwiGLU (d_ff=2816), GQA, no biases, tied embeddings
  Vocab: 32K BPE trained on corpus
  Training data: 7.85B tokens (5 sources)
    34% FineWeb-Edu    — quality-filtered web (score >= 3)
    30% StarCoder      — code (13 languages)
    19% OpenWebMath    — math with LaTeX
     9% Wikipedia
     8% Cosmopedia     — synthetic textbooks
```

## Inference Performance (RTX 4080 Super)

```
306M param model (d=1024, l=24, kv_splits=1)

  Multi-SM sync:       235 tok/s  (4.2 ms/tok, 17% BW util)
  Pipelined:           263 tok/s  (3.8 ms/tok, 1.12x)
  Persistent:          265 tok/s  (3.8 ms/tok, 1.13x)
  Prefill (128 tok):   36.9 ms   (3469 tok/s, Triton)
  Weight buffer:       607 MB (9.5x L2 — HBM-bound)
```

The entire decode step — embedding, attention, FFN, output projection — runs in a single GPU kernel call across all 24 layers.

## Training Performance

```
RTX 4080 Super (bs=16):  ~26K tok/s
NVIDIA B200 (bs=256):   ~341K tok/s

3 epochs × 7.85B tokens = 23.5B total
~6h per epoch on B200, ~83h per epoch on 4080 Super
```

## Quick Start

```bash
# generate text (streaming, with sampling)
uv run generate.py --prompt "Once upon a time" --temp 0.7 --top-p 0.95 --rep-penalty 1.2

# greedy decoding
uv run generate.py --prompt "The capital of France is"

# code generation (low temperature)
uv run generate.py --prompt "def fibonacci(n):" --temp 0.3 --top-p 0.9 --rep-penalty 1.1

# batched inference with paged KV cache
uv run serve.py --paged --prompts "Once upon a time" "The cat sat"

# continuous batching (more prompts than batch slots)
uv run serve.py --continuous --batch-size 2 --prompts "A" "B" "C" "D"

# profile decode kernels
uv run profile_kernels.py

# prepare training data (download + tokenize 7.85B tokens from 5 sources)
uv run prepare_data_v2.py

# train (see H100_TRAINING.md for cloud GPU setup)
uv run python -u train.py \
  --d-model 1024 --n-heads 16 --n-kv-heads 4 --n-layers 24 \
  --context-len 512 --batch-size 16 --epochs 3 \
  --curriculum --lr 3e-4 --no-checkpoint
```

## What's In Here

**Training**: JAX model with cuDNN FlashAttention, AdamW with cosine LR schedule, curriculum training (ctx warmup 128→256→512). Fused cross-entropy for 32K vocab without OOM. Multi-source data pipeline with 32K BPE tokenizer.

**Inference kernels**: 7 Triton kernel files covering single-sequence, batched, persistent, and pipelined decode. Multi-SM parallelism with atomic barriers, KV-split attention (FlashDecoding-style), weight-amortized FFN. Projection tiling for D_HEAD=64 shared memory constraints.

**Serving**: Variable-length batched inference server with GPU-accelerated paged KV cache (JIT gather/scatter) and continuous batching.

## Files

```
Training:
  model.py                          JAX transformer (RMSNorm, RoPE, SwiGLU, GQA, fused CE, MTP)
  train.py                          AdamW training (bf16 fwd, cuDNN FlashAttn, curriculum, checkpointing)
  data.py                           streaming data loading (v2 memmap for 8B tokens)
  prepare_data_v2.py                v2 data pipeline: 5-source download, tokenize, shuffle, combine

Inference kernels:
  kernels/multi_sm_decode.py        multi-SM decode with atomic barriers + KV-split
  kernels/batched_decode.py         batched multi-SM decode (B sequences)
  kernels/persistent_decode.py      persistent decode (single launch, all steps)
  kernels/persistent_batched_decode.py  persistent batched (B x N steps)
  kernels/block_prefill.py          multi-block prefill + FlashAttention + GQA
  kernels/fused_decode_nlayer.py    weight/KV packing for all decode kernels
  kernels/paged_kv.py               paged KV cache (GPU gather/scatter)

Serving:
  generate.py                       streaming text generation CLI
  serve.py                          batched server + continuous batching

Benchmarking:
  profile_kernels.py                primary profiling tool
  profile_vram.py                   VRAM profiling for model scaling

Documentation:
  program.md                        agent program / development log
  H100_TRAINING.md                  cloud GPU training setup guide
  repo_explained_from_zero.md       ground-up GPU kernel explanation
```
