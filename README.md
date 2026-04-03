# Single-GPU Transformer

Training and inference for a decoder-only transformer, entirely from scratch on one **RTX 4080 Super (16GB VRAM)**. Custom Triton kernels for decode. JAX for training. No frameworks, no shortcuts.

Built using [karpathy/autoresearch](https://github.com/karpathy/autoresearch)-style autonomous development — a coding agent is pointed at `program.md` repeatedly. The human steers direction; the agent handles implementation, debugging, benchmarking, and documentation.

See [`repo_explained_from_zero.md`](repo_explained_from_zero.md) for a ground-up explanation of GPU kernels, register pressure, memory hierarchies, and all techniques used.

## Current Model

```
d=1024, h=16, l=16, ctx=512, GQA 4 KV heads, 242M params
  Trained on 1.77B tokens (FineWeb-Edu 60%, Cosmopedia 18%, Wikipedia 17%, Code 6%)
  1 epoch, 13.7 hours, 36K tok/s training throughput
  val_loss=3.04, ppl=20.91
  Vocab: 32K BPE trained on combined corpus
```

## Inference Performance

```
RTX 4080 Super | 242M param model (d=1024)

  Multi-SM sync:       501 tok/s  (2.0 ms/tok, 30% BW util)
  Pipelined:           618 tok/s  (1.6 ms/tok, 1.23x)
  Persist B=4:         992 tok/s  (4.0 ms/step, 1.98x)
  Persist B=8:        1097 tok/s  (7.3 ms/step, 2.19x)
  Prefill (128 tok):   23.2 ms   (5516 tok/s, Triton)
  Weight buffer:       485 MB (7.6x L2 — HBM-bound)
```

The entire decode step — embedding, attention, FFN, output projection — runs in a single GPU kernel call across all 16 layers.

## Quick Start

```bash
# generate text (streaming)
uv run generate.py --prompt "Once upon a time"

# batched inference with paged KV cache
uv run serve.py --paged --prompts "Once upon a time" "The cat sat"

# continuous batching (more prompts than batch slots)
uv run serve.py --continuous --batch-size 2 --prompts "A" "B" "C" "D"

# profile decode kernels
uv run profile_kernels.py

# prepare training data (download + tokenize)
uv run prepare_data.py

# train (d=1024, ~14 hours on RTX 4080 Super)
uv run train.py --d-model 1024 --n-heads 16 --n-kv-heads 4 --n-layers 16 \
  --context-len 512 --epochs 1 --batch-size 16 --lr 3e-4 --dataset combined
```

## What's In Here

**Training**: JAX model with bf16 forward pass for tensor core utilization, AdamW with cosine LR schedule. Multi-source data pipeline (FineWeb-Edu, Wikipedia, Cosmopedia, Code) with 32K BPE tokenizer.

**Inference kernels**: 7 Triton kernel files covering single-sequence, batched, persistent, and pipelined decode. Multi-SM parallelism with atomic barriers, KV-split attention (FlashDecoding-style), weight-amortized FFN. Projection tiling for D_HEAD=64 shared memory constraints.

**Serving**: Variable-length batched inference server with GPU-accelerated paged KV cache (JIT gather/scatter) and continuous batching.

## Files

```
Training:
  model.py                          JAX transformer (no FFN biases, GQA)
  data.py                           dataset loading (TinyStories, Shakespeare, combined)
  train.py                          AdamW training with bf16 forward + LR schedule
  prepare_data.py                   multi-source data download + tokenization

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
  repo_explained_from_zero.md       ground-up GPU kernel explanation
```
