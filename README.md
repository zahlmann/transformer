# Fused Inference

Custom Triton kernels for transformer inference — up to **17x faster** than JAX/XLA baseline. The entire decode step (embedding, attention, FFN, output projection) runs in a **single GPU kernel call**.

## Quick Start

```bash
uv run train_backprop.py --tokenizer trained_bpe --bpe-vocab 1024   # train (6s)
uv run inference_benchmark.py                                        # benchmark
```

## Results

```
Small model (d=64, 1 layer, 189K params, BPE vocab=1024):
  Triton:    3056 tok/s  (20.9 ms)
  JAX:        181 tok/s  (353.8 ms)
  Speedup:   16.9x

Large model (d=128, 2 layers, 674K params, BPE vocab=1024):
  Triton:    2589 tok/s  (24.7 ms)
  JAX:        187 tok/s  (342.4 ms)
  Speedup:   13.9x
```

## How It Works

### Prefill (process full prompt)

**Small model (d_model=64):** Single fused kernel — all weights stay in registers (132KB model fits in 130KB register budget). Zero HBM round-trips between operations.

**Large model (d_model=128):** Multi-block approach with BLOCK_SEQ=32. Three Triton kernels per layer (projections, attention, FFN) enable scaling to any d_model while staying within register limits.

### Decode (generate tokens one at a time)

Fully fused kernel processes all layers in one launch. Key optimizations:
- **In-kernel KV cache updates** — the kernel writes full updated caches directly, avoiding expensive `.at[].set()` scatter ops in Python (this alone gave 3.6x speedup)
- **Precomputed bf16 weights** — dtype conversions done once before the decode loop
- **Multi-layer fusion** — h stays in registers between layers (no HBM round-trip)

### BPE Tokenization

Trained ByteLevel BPE tokenizer on Shakespeare (0% UNK). Tiled output projection handles vocab=1024 in 8 tiles of 128 with zero overhead.

### Key Insight

For small models, **host-side overhead dominates GPU kernel time**. The GPU kernel takes 0.4ms per decode step, but Python/JAX dispatch, dtype conversions, and array scatters added 1.1ms. Eliminating all host overhead gave a 7.5x improvement.

## Architecture

```
Decoder-only transformer on Shakespeare

Small:  d_model=64,  n_heads=2, n_layers=1, vocab=1024, 189K params
Large:  d_model=128, n_heads=4, n_layers=2, vocab=1024, 674K params
```

## Files

```
program.md                              agent program (read first for context)
inference_guide.md                      ground-up explanation of GPU kernels
model.py                                JAX transformer (inference baseline)
train_backprop.py                       backprop+Adam training
data.py                                 Shakespeare dataset + BPE tokenizer
kernels/fused_prefill.py                fused prefill kernel (d_model<=64)
kernels/fused_decode.py                 fused decode kernel (d_model<=64)
kernels/block_prefill.py                multi-block prefill kernels (d_model>=128)
kernels/block_decode.py                 per-layer decode (d_model>=128)
kernels/fused_decode_2layer.py          fully fused 2-layer decode (fastest)
inference_benchmark.py                  speed comparison benchmark
```
