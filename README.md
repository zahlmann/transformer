# Fused Inference

Custom Triton kernels for transformer inference on a single **RTX 4080 Super (16GB VRAM, 836 GB/s bandwidth)**. The entire decode step (embedding, attention, FFN, output projection) runs in a **single GPU kernel call**.

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch) — a coding agent is pointed at `program.md` repeatedly to make progress autonomously. The human only steers which direction to work on next; the agent handles implementation, debugging, benchmarking, and documentation.

See [`repo_explained_from_zero.md`](repo_explained_from_zero.md) for a ground-up explanation of GPU kernels, register pressure, memory hierarchies, and all the techniques used in this project.

## Current Performance

```
RTX 4080 Super | d=512, h=16, l=8, ctx=512 | 29.7M params | ppl=2.91

Prefill (128 tokens):     6.3 ms  (20,460 tok/s)
Decode (128 tokens):    445.5 ms  (287 tok/s, 3.48 ms/tok)

Memory:
  Weight buffer:         59.3 MB (bf16)
  KV cache:               8.4 MB (bf16, per sequence)
  Total inference:       67.7 MB

Roofline:
  Theoretical min:        0.081 ms/tok (at 836 GB/s bandwidth)
  Achieved:               3.481 ms/tok
  Bandwidth utilization:  2%
```

**The GPU is 98% idle.** The bottleneck is kernel launch overhead and dispatch latency,
not memory bandwidth or compute. This is the main optimization target.

## Metrics

All optimization work is evaluated against these hard metrics (run `uv run profile_kernels.py`):

| Metric | Current | Target | Unit |
|--------|---------|--------|------|
| Decode throughput | 287 | higher is better | tok/s |
| Decode latency | 3.48 | lower is better | ms/tok |
| Bandwidth utilization | 2% | closer to 100% | % of 836 GB/s |
| Prefill latency | 6.3 | lower is better | ms for 128 tokens |

## Quick Start

```bash
# Train the model on TinyStories (default: d=256, 4L, 5.3M params)
uv run train_backprop.py

# Or train the large model (d=512, 8L, 29.7M params, ~4 hours)
uv run train_backprop.py --d-model 512 --n-heads 16 --n-layers 8 \
  --context-len 512 --epochs 3 --lr 1e-4 --batch-size 16

# Profile kernels (primary benchmark)
uv run profile_kernels.py

# Quick inference benchmark
uv run inference_benchmark.py
```

## How It Works

### Prefill (process full prompt)

Multi-block approach with configurable BLOCK_SEQ (32 for d=128, 16 for d=256, 8 for d=512). Three Triton kernels per layer (projections, attention, FFN). FlashAttention (tiled KV + online softmax) for context>256.

### Decode (generate tokens one at a time)

Fully fused N-layer kernel processes all layers in one launch using packed weight buffers and packed KV caches. Key techniques:
- **In-kernel KV cache updates** — kernel writes updated caches directly
- **Packed weight buffer** — all per-layer weights in one bf16 buffer
- **Packed KV caches** — all layers' caches in one flat buffer
- **Multi-layer fusion** — h stays in registers between layers
- **tl.dot projections** — avoids register overflow at d=512
- **Tiled KV decode** — online softmax for large context (512+)

### Training

AdamW with linear warmup + cosine decay. Full TinyStories (487M BPE tokens, vocab=4096).

## Architecture

```
Decoder-only transformer (d_head=32 for all sizes)

Small:  d=64,  h=2,  l=1, vocab=1024,  189K params  (Shakespeare)
Medium: d=128, h=4,  l=2, vocab=1024,  674K params  (Shakespeare)
Large:  d=256, h=8,  l=4, vocab=4096, 5.3M params   (TinyStories)
XL:     d=512, h=16, l=8, vocab=4096, 29.7M params  (TinyStories full)
```

## Files

```
program.md                              agent program (read first for context)
repo_explained_from_zero.md             ground-up explanation of GPU kernels + techniques
profile_kernels.py                      primary profiling tool (run after any change)
model.py                                JAX transformer model
train_backprop.py                       AdamW training with LR schedule
data.py                                 Shakespeare + TinyStories + BPE tokenizer
kernels/fused_prefill.py                fused prefill kernel (d_model<=64)
kernels/fused_decode.py                 fused decode kernel (d_model<=64)
kernels/block_prefill.py                multi-block prefill + FlashAttention (d_model>=128)
kernels/block_decode.py                 per-layer decode (d_model>=128)
kernels/fused_decode_2layer.py          fully fused 2-layer decode
kernels/fused_decode_nlayer.py          fully fused N-layer decode (packed weights/caches)
inference_benchmark.py                  speed comparison benchmark
baseline_metrics.txt                    current baseline numbers to beat
```
