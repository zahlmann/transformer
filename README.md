# Fused Inference

Custom Triton kernels for transformer inference on a single **RTX 4080 Super (16GB VRAM, 836 GB/s bandwidth)**. The entire decode step — embedding, attention, FFN, output projection — runs in a **single GPU kernel call** across all layers.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — a coding agent is pointed at `program.md` repeatedly to make progress autonomously. The human only steers which direction to work on next; the agent handles implementation, debugging, benchmarking, and documentation.

See [`repo_explained_from_zero.md`](repo_explained_from_zero.md) for a ground-up explanation of GPU kernels, register pressure, memory hierarchies, and all the techniques used in this project.

## Performance

```
RTX 4080 Super | d=512, h=16, l=8, ctx=512 | GQA 4 KV heads | 26.5M params | ppl=2.96

Single-sequence decode:
  Persistent kernel:    4777 tok/s  (0.209 ms/tok)  ← single launch, all tokens on GPU
  Pipelined:            2624 tok/s  (0.381 ms/tok)  ← no per-token sync
  Sync'd (int/tok):     1869 tok/s  (0.535 ms/tok)

Batched decode (B sequences in parallel):
  B=4  pipelined:       6691 tok/s  (0.598 ms/step, 3.58x vs single-seq sync)
  B=8  pipelined:       7339 tok/s  (1.090 ms/step, 3.93x)
  B=16 sync'd:          6064 tok/s  (2.638 ms/step, 3.24x)

Prefill (128 tokens):   6.1 ms     (21,168 tok/s)

Memory:
  Weight buffer:         53 MB (bf16)
  KV cache:              2.1 MB/seq (bf16, GQA)
  Bandwidth utilization: 16% of 836 GB/s theoretical
```

### Optimization history

```
Phase A1: d=64, 1L  — fused prefill+decode              740 tok/s
Phase A2: d=128, 2L — in-kernel KV updates (3.6x win)  2504 tok/s
Phase A3: d=256, 4L — fused N-layer decode              1396 tok/s
Phase A4: d=512, 8L — tl.dot projections, tiled KV      287 tok/s (2% BW util)
Phase A5: multi-SM   — grid=(32,) + atomic barriers     1937 tok/s (6.8x, 16% BW)
Phase A6: batched     — shared weights, B sequences     3821 tok/s at B=4 (1.97x)
Phase A7: persistent  — single launch, no host sync     4777 tok/s (2.56x vs sync)
          pipelined   — batched + no per-token sync     6691 tok/s at B=4
```

## Quick Start

```bash
# train the model (d=512, 8L, 29.7M params, ~4 hours)
uv run train.py --d-model 512 --n-heads 16 --n-layers 8 \
  --context-len 512 --epochs 3 --lr 1e-4 --batch-size 16

# profile kernels (primary benchmark — run after any kernel change)
uv run profile_kernels.py

# quick inference demo
uv run inference_benchmark.py
```

## How It Works

### Prefill (process full prompt)

Multi-block approach with 3 Triton kernels per layer (projections, attention, FFN). FlashAttention (tiled KV + online softmax) for context > 256. BLOCK_SEQ scales inversely with d_model to fit in registers.

### Decode (generate tokens one at a time)

**Multi-SM kernel** distributes work across all GPU SMs:
- **grid=(N_HEADS × KV_SPLITS,)** — each block handles one attention head (or KV split), all blocks split the FFN
- **Atomic barriers** — release/acquire semantics for cross-block synchronization
- **KV-split parallelism** (FlashDecoding-style) — splits KV cache across 2 blocks per head, merges with online softmax correction
- **Split barrier** — separate counter/done-flag cache lines to reduce L2 contention

**Persistent kernel** runs all decode steps in a single kernel launch:
- Tokens stay on GPU — no per-step host sync
- Fresh barrier slots per step
- In-kernel next-token feedback (block 0 writes, step-sync barrier broadcasts)

**Batched kernel** processes B sequences per launch:
- Weight loads amortized across batch (weight-amortized FFN: outer k-loop / inner b-loop)
- Double-buffered h_buf to avoid cross-block read-write races
- Separate ffn_buf to avoid merge/FFN phase conflicts

### Key techniques

- **Packed weight buffer** — all per-layer weights in one bf16 buffer, kernel computes offsets
- **Packed KV caches** — all layers' caches in one flat buffer, no per-step pack/unpack
- **In-kernel KV cache updates** — kernel writes updated caches directly (biggest single win: 3.6x)
- **tl.dot projections** — avoids register overflow at d=512 by tiling internally
- **Tiled KV decode** — online softmax with KV_TILE=64 for large context

### What was tried but didn't help

- **GQA** — no single-sequence speedup (barrier-limited, not memory-limited). Helps batched inference.
- **Parallel residual** — 9 barriers instead of 17, but only 1.3% speedup (straggler variance dominates)
- **num_warps sweep** — 2/4/8 all within 5%
- **Speculative decoding** — acceptance rate too low at this scale (36-51%), draft/target speed ratio only 2x

## Architecture

```
Decoder-only transformer (d_head=32 for all sizes)

Small:  d=64,  h=2,  l=1, vocab=1024,  189K params  (Shakespeare)
Medium: d=128, h=4,  l=2, vocab=1024,  674K params  (Shakespeare)
Large:  d=256, h=8,  l=4, vocab=4096, 5.3M params   (TinyStories)
XL:     d=512, h=16, l=8, vocab=4096, 29.7M params  (TinyStories)
```

## Files

```
Core:
  model.py                        JAX transformer model
  data.py                         Shakespeare + TinyStories + BPE tokenizer
  train.py               AdamW training with LR schedule

Kernels:
  kernels/fused_prefill.py        fused prefill (d_model <= 64)
  kernels/fused_decode.py         fused decode (d_model <= 64)
  kernels/block_prefill.py        multi-block prefill + FlashAttention (d_model >= 128)
  kernels/block_decode.py         per-layer decode orchestrator (d_model >= 128)
  kernels/fused_decode_nlayer.py  fused N-layer decode (packed weights/caches)
  kernels/multi_sm_decode.py      multi-SM decode with atomic barriers + KV-split
  kernels/batched_decode.py       batched multi-SM decode (B sequences)
  kernels/persistent_decode.py    persistent decode (single launch, all steps)

Benchmarking:
  profile_kernels.py              primary profiling tool (run after every change)
  inference_benchmark.py          quick throughput + text generation demo
  baseline_metrics.txt            current performance numbers

Documentation:
  program.md                      agent program / development log
  repo_explained_from_zero.md     ground-up GPU kernel explanation
  cuda_kernels_docs/              Triton, CUDA, jax-triton reference docs
```
