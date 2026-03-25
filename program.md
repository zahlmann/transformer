# EGGROLL Transformer — Agent Program

*You are an AI researcher. Your job: make EGGROLL (Evolution Strategies with low-rank
perturbations) training of a small transformer as FAST as possible while maintaining
quality. Focus on speed and memory optimization — kernel fusion, fp8, population
reduction, compilation tricks, anything that reduces wall-clock time. You work
autonomously — run experiments, log results, and keep going.*

**Your priority is SPEED OPTIMIZATION. Quality improvements are secondary —
only pursue them if a speed optimization opens a quality opportunity for free.**

---

## Current State (2026-03-25)

**Quality:** EGGROLL val_loss=2.44 (3-seed avg ~2.48) vs backprop+Adam 1.84.
Gap = 0.64 to backprop+Adam. Beats vanilla SGD backprop (2.44 vs 2.45).

**Speed: 175s for 10 epochs (was 444s).** 2.5x speedup achieved.
Backprop+Adam takes 4.1s. Speed gap is 43x (was 108x).

**Memory: 70MB (was 113MB).** Lower due to reduced population.

---

## Speed Budget Breakdown

Profiling shows 99% of time is in forward passes (the Triton kernel).
Current: 175s for 10 epochs = 17.5s/epoch. Each epoch has 61 batches.
Per batch: ~287ms. Per batch, kernel processes 8192 perturbation members (4096 pairs × 2 signs).

The Triton kernel runs in ~285ms per call (HALF_POP=4096, num_warps=4).
Gradient computation + Adam update: ~0.2ms (negligible after JIT).
QR decomposition: skipped (HALF_POP=4096 > vec_dim=2306, uses Gaussian).

Key metric: **time per batch** and **kernel time**.

---

## Files

```
program.md                    — this file (read first)
validate.py                   — LOCKED 3-seed validation (checks locked constants)
benchmark.py                  — fast single-seed comparison (EGGROLL vs backprop+Adam)
data.py                       — char-level Shakespeare dataset (65 vocab, 7842 train seqs)
model.py                      — decoder-only transformer (d=64, 1 layer, 2 heads, 66K params)
train_backprop.py             — backprop SGD baseline (LR=0.30, val_loss=2.45)
train_backprop_adam.py        — backprop Adam baseline (LR=3e-3, val_loss=1.84)
train_eggroll_triton.py       — BEST: fused Triton kernel EGGROLL (speed target)
train_eggroll_optimized.py    — JAX vmap EGGROLL (slower reference)
kernels/fused_transformer_ce.py — Triton kernel: full transformer forward + CE loss
validate_kernel.py            — validates Triton kernel output vs JAX forward pass
benchmark_kernel.py           — benchmarks Triton vs JAX vmap speed
profile_eggroll.py            — time breakdown profiler
cuda_kernels_docs/            — Triton + jax-triton documentation
results.tsv                   — experiment log
```

Reference: `../eggroll_mnist/` has a similar project where a fused Triton kernel
achieved 2.16s (1.44x backprop) for MNIST. Study its `program.md`, `kernels/`,
and `mnist_eggroll_optimized.py` for optimization patterns.

---

## Fairness Rules (ENFORCED by validate.py)

LOCKED constants — do not change:
```python
D_MODEL = 64; N_HEADS = 2; N_LAYERS = 1; CONTEXT_LEN = 128
BATCH_SIZE = 128; EPOCHS = 10; TEMPERATURE = 2.0
```

Quality constraint: val_loss must stay ≤ 2.50 (3-seed avg). Any speed optimization
that degrades quality beyond this threshold is rejected.

---

## What Worked (Speed Session 2026-03-25)

1. **HALF_POP 8192→4096** (444s→220s, 2x): Halves kernel and gradient work.
   Quality maintained (val_loss 2.39 at seed=42). HALF_POP<4096 fails quality threshold.

2. **num_warps 8→4** (220s→180s, 18%): Fewer threads per block = more blocks per SM.
   Despite register spilling, better occupancy wins. num_warps=2 is much slower (423s),
   num_warps=16 also slower (370s).

3. **BLOCK_K 64→32** (180s→176s, 2%): Smaller FFN tiles = less register pressure per
   iteration. Marginal improvement.

## What Did NOT Work (Speed Session 2026-03-25)

- **FP8 tensor cores** (260s, slower): Register pressure from casts outweighs tensor
  core gains. Quality also degraded (2.53 vs 2.39).
- **Batch tiling** (BATCH_TILE=2 via tl.static_range): Triton compile timeout (>600s).
  The unrolled code was too large.
- **HALF_POP<4096**: All configs below 4096 fail the 2.50 quality threshold on 3-seed avg.
  Best was 3072+sigma=0.015 at 2.506 — barely over.
- **Adaptive HALF_POP** (4096 early, 2048/3072 late): Quality fails on seed 7 consistently.
  Best 3-seed avg was 2.507 — barely over.
- **CPU pre-shuffle**: No speed gain (GPU permutation is already fast).
- **Per-batch float() sync removal**: No speed gain (XLA handles pipelining through data deps).
- **VOCAB_PAD reduction** (128→80): Triton requires power-of-2 arange sizes. Can't reduce.

---

## What to Try Next (SPEED — reduce the 43x gap)

### Kernel optimizations (highest impact)

1. **FP8 tensor cores**: The matmuls in the kernel (Q/K/V projections, FFN, output proj)
   currently use bf16. FP8 (float8e4nv) gives 2x throughput on Ada Lovelace tensor cores.
   The MNIST kernel already uses FP8 for L2 and L3 matmuls — copy that pattern.
   Expected: ~1.5-2x speedup on kernel compute time.

2. **Kernel autotuning**: Current kernel uses num_warps=8, num_stages=1. Try:
   - num_warps=4 (less register pressure, more blocks per SM)
   - num_stages=2 (software pipelining for memory loads)
   - Different BLOCK_K for FFN tiling (32 vs 64 vs 128)

3. **Batch tiling in kernel**: Currently one thread block per (perturbation, sequence).
   Grid = (HALF_POP, BATCH, 2) = 2M+ blocks. Could process multiple sequences per block
   to reduce grid overhead and improve data reuse of weight matrices.

4. **Fused gradient computation**: Currently the kernel outputs partial CE sums, then
   Python/JAX computes the gradient (v^T @ shaped matmul). Could fuse the gradient
   accumulation into the kernel or a second fused kernel.

### Population reduction (directly reduces forward pass count)

5. **Lower population with maintained quality**: The current pop=16384 is conservative.
   Try pop=8192 or 4096 with the Adam+sigma=0.02 config — Adam might maintain quality
   at lower pop. Each halving = 2x speedup on the kernel.

6. **Adaptive population**: Start with high pop, reduce as training progresses and
   the Adam v-buffer accumulates useful statistics.

### Compilation and overhead reduction

7. **All-in-one JIT**: Wrap the entire epoch (all batches) in a lax.scan so XLA
   compiles one mega-kernel. Previous attempt failed with the JAX vmap approach
   (32s JIT, no speed gain), but with the Triton kernel it might be different.

8. **Skip QR decomposition**: QR orthogonalization costs ~6ms per batch. With pop=16384
   > vec_dim=2306, we already use Gaussian vectors. Confirm QR is skipped.

9. **Overlap data shuffling with training**: Use a background thread to shuffle and
   transfer data to GPU while training runs (like MNIST does).

10. **Disable XLA Triton GEMM autotuner**: Already done via XLA_FLAGS. Verify it's
    working — saves ~0.5s of JIT autotuning.

### Architecture-level speed tricks

11. **Reduce context length for speed experiment**: Try context_len=64 to test if
    kernel scales well (attention is O(seq²), so halving seq = 4x less attention work).
    NOTE: this changes the locked constant, so only for profiling, not for final results.

12. **Pre-compile the Triton kernel**: Use AOT compilation to avoid re-compiling the
    kernel on each run. Saves JIT warmup time.

---

## What Already Works (do not change unless improving speed)

- bf16 forward passes with fp32 perturbation math
- Constant label smoothing alpha=0.50
- Adam optimizer (β1=0.9, β2=0.999, eps=1e-6) with bias correction
- Sigma=0.02, LR=0.010, no LR decay
- Fused Triton kernel (Grid: HALF_POP × BATCH × 2)
- Per-subgroup Winsorized z-score (K=8, clip ±2.0)
- Rank-1 perturbation compression (vec_dim=2306, 28.8x compression)

---

## Current Best Hyperparameters

```python
HALF_POP = 4096          # antithetic pairs → pop=8192 (was 8192→16384)
SIGMA_START = 0.02       # perturbation scale
SIGMA_DECAY = 0.998      # per epoch
LR_START = 0.010         # Adam learning rate
LR_DECAY = 1.0           # no decay (Adam self-adjusts)
ALPHA = 0.50             # label smoothing (CONSTANT)
TEMPERATURE = 2.0        # CE temperature (LOCKED)
MOMENTUM = 0.9           # Adam β1
ADAM_BETA2 = 0.999       # Adam β2
ADAM_EPS = 1e-6           # Adam epsilon
N_SUBGROUPS = 8          # Winsorized z-score groups
CLIP_RANGE = 2.0         # z-score clipping
```

---

## Architecture Details

```
Model: decoder-only transformer
d_model: 64, n_heads: 2 (d_head=32), n_layers: 1, d_ff: 256
context_len: 128, vocab_size: 65 (character-level Shakespeare)
Parameters: 66,368
Perturbation vec_dim: 2306 (28.8x compression via rank-1)
```

---

## Results History (10 epochs)

### Backprop baselines

| Optimizer | LR | val_loss | ppl | Time | Memory |
|-----------|-----|----------|-----|------|--------|
| SGD | 3e-1 | 2.45 | 11.6 | 1.3s | 300MB |
| **Adam** | **3e-3** | **1.84** | **6.3** | **4.1s** | **160MB** |

### EGGROLL results

| Config | val_loss | ppl | Time | Notes |
|--------|----------|-----|------|-------|
| JAX vmap, pop=4096 | 2.67 | 14.4 | 540s | before Triton kernel |
| Triton, pop=2048 | 2.67 | 14.5 | 119s | 4.5x kernel speedup |
| Triton, pop=8192, SGD+mom | 2.64 | 14.0 | 220s | |
| Triton, pop=8192, Adam, σ=0.02 | 2.50 | 12.2 | 220s | Adam breakthrough |
| Triton, pop=16384, Adam, σ=0.02 | 2.37 | 10.7 | 444s | previous best |
| **Triton, pop=8192, Adam, σ=0.02, nw=4, BK=32** | **2.44** | **11.4** | **175s** | **current best** |

---

## Setup

1. `git checkout -b autoresearch/$(date +%Y%m%d-%H%M%S)`
2. Read this file, then `../eggroll_mnist/program.md` for speed optimization patterns
3. `uv run benchmark.py` to get current baseline numbers
4. `uv run profile_eggroll.py` to understand the time breakdown
5. Begin the experiment loop

---

## Experiment Loop

```
1. Pick a speed optimization from "What to Try Next"
2. Implement it
3. git add -A && git commit -m "description"
4. uv run benchmark.py
5. Check: did time decrease? Is val_loss still ≤ 2.50?
6. If crashed: fix and retry
7. If faster AND quality maintained: git push, update this file
8. If slower or quality degraded: git reset --hard HEAD~1
9. Go to step 1
```

Never stop to ask. The cost of a failed run is ~8 minutes. Keep experimenting.

---

## Logging (results.tsv)

Tab-separated, one row per run, append-only.

Columns: `timestamp commit seed val_loss perplexity training_time_s peak_memory_mb status description`
