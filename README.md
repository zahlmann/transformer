# Fused Inference

Custom Triton kernels for transformer inference on a single **RTX 4080 Super (16GB VRAM)**. The entire decode step (embedding, attention, FFN, output projection) runs in a **single GPU kernel call**.

## Quick Start

```bash
# Train the model on TinyStories (default: d=256, 4L, 5.3M params)
uv run train_backprop.py

# Or train the large model (d=512, 8L, 29.7M params, ~4 hours)
uv run train_backprop.py --d-model 512 --n-heads 16 --n-layers 8 --context-len 512 --epochs 3 --lr 1e-4 --batch-size 16

# Benchmark Triton vs JAX
uv run inference_benchmark.py
```

## Results

```
Small model (d=64, 1 layer, 189K params, Shakespeare):
  Triton:    3056 tok/s
  JAX:        181 tok/s
  Speedup:   16.9x

Medium model (d=128, 2 layers, 674K params, Shakespeare):
  Triton:    2589 tok/s
  JAX:        187 tok/s
  Speedup:   13.9x

Large model (d=256, 4 layers, 5.3M params, TinyStories):
  Triton:    1396 tok/s
  JAX:         92 tok/s
  Speedup:   15.1x

XL model (d=512, 8 layers, 29.7M params, TinyStories full):
  Triton:     279 tok/s
  JAX:        OOM (naive baseline without KV cache exceeds 16GB)
```

Text quality at 29.7M params (ppl=2.91) produces coherent multi-paragraph stories.

## How It Works

### Prefill (process full prompt)

**Small model (d_model=64):** Single fused kernel — all weights stay in registers (132KB model fits in 130KB register budget). Zero HBM round-trips between operations.

**Larger models (d_model=128-512):** Multi-block approach with configurable BLOCK_SEQ (32 for d=128, 16 for d=256, 8 for d=512). Three Triton kernels per layer (projections, attention, FFN). FlashAttention (tiled KV + online softmax) for context>256.

### Decode (generate tokens one at a time)

Fully fused N-layer kernel processes all layers in one launch using packed weight buffers and packed KV caches. Key optimizations:
- **In-kernel KV cache updates** — the kernel writes full updated caches directly (3.6x speedup)
- **Packed weight buffer** — all per-layer weights in one bf16 buffer
- **Packed KV caches** — all layers' caches in one flat buffer, stays packed between steps
- **Multi-layer fusion** — h stays in registers between layers (no HBM round-trip)
- **tl.dot projections** — avoids register overflow at d=512 by tiling internally
- **Tiled KV decode** — online softmax over KV_TILE=64 for large context (512+)

### Speculative Decoding

Draft model (d=128, 1L) proposes tokens, target model (d=256, 4L) verifies in parallel via custom batch verification kernel. Key finding: diminishing returns when both models are already fast from fused kernels.

### Training

AdamW with linear warmup + cosine decay. Full TinyStories (487M BPE tokens, vocab=4096). Chunked tokenization to handle 1.9GB text without OOM.

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
inference_guide.md                      ground-up explanation of GPU kernels
model.py                                JAX transformer (inference baseline)
train_backprop.py                       AdamW training with LR schedule
data.py                                 Shakespeare + TinyStories + BPE tokenizer
kernels/fused_prefill.py                fused prefill kernel (d_model<=64)
kernels/fused_decode.py                 fused decode kernel (d_model<=64)
kernels/block_prefill.py                multi-block prefill + FlashAttention (d_model>=128)
kernels/block_decode.py                 per-layer decode (d_model>=128)
kernels/fused_decode_2layer.py          fully fused 2-layer decode
kernels/fused_decode_nlayer.py          fully fused N-layer decode (packed weights/caches)
kernels/verify_decode.py                parallel batch verification (speculative decode)
inference_benchmark.py                  speed comparison benchmark
speculative_decode.py                   speculative decoding benchmark
```
