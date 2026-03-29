# EGGROLL Transformer

Training a small transformer with **EGGROLL** (Evolution Strategies with low-rank perturbations) instead of backpropagation. A fused Triton kernel runs the entire transformer forward pass + CE loss in a single GPU kernel call.

## Quick Start

```bash
uv run train_eggroll.py          # EGGROLL training (272s, val_loss ~2.38)
uv run train_backprop.py         # Backprop+Adam baseline (4s, val_loss ~1.84)
uv run benchmark.py              # Side-by-side comparison
uv run validate.py               # 3-seed quality validation
```

## Results (10 epochs, 3-seed average)

| Method | val_loss | ppl | Time | Memory |
|--------|----------|-----|------|--------|
| **EGGROLL** (pop=14336) | **2.38** | **10.8** | **272s** | **103MB** |
| EGGROLL (pop=8192) | 2.49 | 12.1 | 153s | 70MB |
| Backprop+Adam | 1.84 | 6.3 | 4.1s | 160MB |
| Backprop+SGD | 2.45 | 11.6 | 1.3s | 300MB |

EGGROLL beats vanilla SGD backprop (2.38 vs 2.45) but has a 0.54 gap to Adam.

## How EGGROLL Works

1. Generate 7168 random perturbation vectors (rank-1 compressed: 66K params → 2306 dims)
2. Evaluate perturbed model at +sigma and -sigma (14336 forward passes per batch)
3. Estimate gradient from fitness differences (Winsorized z-score, per-subgroup normalization)
4. Update with Adam optimizer

All 14336 forward passes run in a **single fused Triton kernel** — no HBM round-trips between layers.

## Architecture

- Decoder-only transformer: d_model=64, 2 heads, 1 layer, d_ff=256
- Character-level Shakespeare, vocab=65, context=128
- 66,368 parameters

## What's Novel

**The fused Triton kernel** (`kernels/fused_transformer_ce.py`) processes the entire
transformer (embedding → layer norm → multi-head attention → FFN → output projection → CE
loss) in one kernel call. Each thread block handles one (perturbation, sequence) pair with
all weights resident in registers. This architecture is reusable for:
- Fused inference (strip out ES, keep the single-kernel forward pass)
- Forward-mode AD (the dual-number kernel in `kernels/fused_transformer_ce_dual.py`)
- Any gradient-free optimization method that needs many parallel forward passes

**The rank-1 perturbation framework** compresses weight perturbations from O(params) to
O(sqrt(params)) via outer-product structure: delta_W = b ⊗ a for each 2D weight matrix.
This is structurally identical to LoRA adapters.

**The dual-number kernel** (`kernels/fused_transformer_ce_dual.py`) computes exact
directional derivatives via forward-mode AD, fused into the same single-kernel architecture.
Validated against jax.jvp to bf16 precision.

## Where This Could Lead

See `program.md` "Where This Could Lead" for detailed analysis. Key directions:
- **Non-differentiable objectives** (RLHF rewards, discrete sampling, exact-match metrics)
- **Communication-free distributed training** (no gradient sync, linear GPU scaling)
- **Gradient-free LoRA** (rank-1 perturbations = LoRA adapters without backprop)
- **Fused inference** (the kernel is independently valuable)

## Files

```
train_eggroll.py              EGGROLL training script (best config)
train_backprop.py             backprop+Adam baseline
kernels/fused_transformer_ce.py       fused Triton kernel (production)
kernels/fused_transformer_ce_dual.py  dual-number kernel (forward-mode AD)
model.py                      JAX transformer model
data.py                       Shakespeare dataset
validate.py                   3-seed validation
benchmark.py                  comparison benchmark
program.md                    full experiment history + next directions (READ THIS)
```

## Optimization History

Speed: **444s → 153s (2.9×)** via kernel optimizations, then **153s → 272s** trading speed
for quality (HALF_POP 4096 → 7168).

Quality: **2.49 → 2.38** (3-seed avg) via larger population. 30+ algorithmic experiments
(LR schedules, sigma schedules, orthogonal perturbations, guided ES, label smoothing tuning,
Adam HP tuning, forward-mode AD, dual-number kernel, population scheduling, etc.) confirmed
that population size is the only lever — all else is already optimized.

See program.md for the complete experiment log.
