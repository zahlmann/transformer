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
kernels/cuda/                 — CUDA C++ kernel scaffolding (see below)
kernels/cuda/fused_transformer_ce.cu — CUDA kernel skeleton (TODO: implement body)
kernels/cuda/wrapper.py       — Python/JAX binding for CUDA kernel
kernels/cuda/test_kernel.py   — correctness test: CUDA vs Triton output
kernels/cuda/Makefile          — build: make -C kernels/cuda/
validate_kernel.py            — validates Triton kernel output vs JAX forward pass
benchmark_kernel.py           — benchmarks Triton vs JAX vmap speed
profile_triton.py             — time breakdown profiler (kernel vs gradient vs vec gen)
cuda_kernels_docs/            — Triton + jax-triton + CUDA C++ documentation
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

## What Did NOT Work (Speed Session 2026-03-25 part 2)

- **Triton FlashAttention via tl.gather** (308ms vs 285ms baseline, SLOWER): Triton 3.6
  supports tl.ravel + tl.gather + tl.reshape to extract KV tile rows from register-resident
  tensors. BLOCK_KV=64 gave 278ms isolated kernel benchmark but 181s full training (worse
  than 176s baseline). The gather operation compiles to cross-thread register shuffles that
  add more overhead than the register pressure reduction saves.
- **Tiled output projection** (287ms vs 285ms, no improvement): Tiling the (128,128) logits
  matrix with online log-sum-exp for CE loss. Loop overhead negates register savings.
- **FFN pipelining** (tl.range num_stages=2): 545ms, much slower. Dynamic loop with
  pipelining doesn't work for the FFN body structure.
- **HALF_POP=3072 HP search** (val_loss 2.56 3-seed avg, FAIL): Searched sigma
  [0.012-0.025], lr [0.008-0.015], alpha [0.4-0.6], sigma_decay [0.995-1.0], LR schedules
  (cosine, warmup+cosine). Best single-seed was 2.4946 (sigma=0.018) but 3-seed avg is
  2.5638. Population reduction not viable without algorithmic changes.
- **CUDA kernel (scalar FP32)** (3860ms vs 271ms Triton, 14x SLOWER): Full CUDA kernel
  with FlashAttention via shared memory. Correctness validated (max diff 0.0004). But
  without WMMA tensor cores, scalar FP32 can't compete with Triton's bf16 tensor cores.

## What IS Ready (for next session)

- **CUDA kernel infrastructure**: Full kernel in `kernels/cuda/fused_transformer_ce.cu`
  (664 lines), JAX XLA custom_call binding in `kernels/cuda/wrapper.py`. Builds, runs,
  produces correct output. Only missing WMMA tensor cores for the big matmuls.
  - Build: `make -C kernels/cuda/`
  - Test: `uv run python -c "..."` (see test_kernel.py)
  - Uses: PyCapsule for custom_call registration, scalars packed in opaque struct
  - Key fix: scalars (sigma, alpha, temperature) must be in opaque struct, NOT as
    separate GPU buffer operands (XLA 0.9.2 can't handle scalar GPU buffers)

---

## What to Try Next (SPEED — reduce the 43x gap)

**The kernel is at ~47% of bf16 tensor core peak.** The gap is due to low occupancy
(register pressure from the 128×128 attention matrix). Focus on reducing register
pressure or changing the computation structure.

## What to Try Next — CUDA Kernel Optimization (for next agent)

### Profiling Commands (run these first!)
```bash
# Kernel time breakdown
uv run profile_triton.py   # shows kernel/gradient/vecgen split

# Triton register usage (look for num_regs, spill_stores, spill_loads):
TRITON_CACHE_DIR=/tmp/triton_debug uv run benchmark.py
# Then: find /tmp/triton_debug -name "*.json" | xargs grep num_regs

# CUDA kernel profiling:
make -C kernels/cuda/
# Register usage:
nvcc -O3 -arch=sm_89 --ptxas-options=-v -c kernels/cuda/fused_transformer_ce.cu -o /dev/null 2>&1
# GPU profiling:
ncu --target-processes all uv run python -c "..." 2>&1 | head -50
```

### Bottleneck Analysis

The Triton kernel at 274ms processes 1,048,576 blocks (4096×128×2).
- 80 SMs on RTX 4080 SUPER, likely 1 block/SM occupancy → 13,107 waves
- Per wave: 274ms/13107 ≈ 21μs (matches compute estimates)
- 255 registers/thread (at the max), with 340 bytes spill

**Two (128, 128) matrices cause register pressure:**
1. Attention scores: Q(128,32) @ K(128,32)^T = (128,128) → softmax → (128,128) weights
2. Output logits: h_final(128,64) @ W_out(64,128) = (128,128) → CE loss

Each (128,128) f32 matrix = 64KB. With 128 threads = 512 bytes = 128 regs/thread.
The attention matrix alone uses 50% of the 255-register budget.

**Compute breakdown** (per forward pass, FLOPs):
- QKV projections: 3.1M (16.5%)
- Attention (scores + V_acc): 4.2M (22.3%)
- O projection: 1.0M (5.3%)
- FFN up+down: 8.4M (44.6%)
- Output projection: 2.1M (11.1%)
Total: 18.9M FLOPs. At 204.9 TFLOPS peak: theoretical 97ms. Actual 274ms = 35% utilization.

### CUDA Kernel Infrastructure (READY TO USE)

**Files:**
- `kernels/cuda/fused_transformer_ce.cu` — Full kernel with WMMA + FlashAttention
- `kernels/cuda/wrapper.py` — JAX XLA custom_call binding
- `kernels/cuda/Makefile` — Build system
- `kernels/cuda/test_kernel.py` — Correctness test vs Triton

**Build/Test/Run:**
```bash
make -C kernels/cuda/
uv run python -c "
from kernels.cuda.wrapper import fused_transformer_ce_both_cuda as cuda_kernel
# Must call from @jax.jit (no eager eval rule)
"
```

**Current CUDA kernel issues (WHY it's 11x slower):**
1. **WMMA shared memory staging overhead**: Each WMMA matmul does:
   registers → shared memory (bf16) → WMMA fragments → shared memory (f32) → registers.
   This is 4 memory transactions per matmul vs Triton's 0 (Triton keeps everything in registers).
2. **52.5KB shared memory limits occupancy** to 1 block/SM (100KB SM limit).
3. **Non-WMMA code is scalar FP32**: Q projection, O projection, FlashAttention,
   perturbation math, layer norms, GELU, CE loss all use scalar loops.

### Creative Approaches for the Next Agent

1. **Use PTX inline assembly for matmuls** instead of WMMA. PTX `mma.sync` instructions
   give more control over data movement. Can keep data in registers (no shared memory
   staging). Study: NVIDIA PTX ISA guide, `mma.sync.aligned.m16n8k16.row.col.f32.bf16`.

2. **Use CUTLASS templates** (install CUTLASS first: `pip install nvidia-cutlass` or
   clone from github). CUTLASS provides optimized matmul templates that handle register
   allocation, shared memory tiling, and tensor core usage. Can be integrated into a
   fused kernel via CUTLASS's "epilogue fusion" API.

3. **Persistent kernel in Triton** with global memory K/V scratch:
   - Launch 160-256 blocks (2 per SM)
   - Each block processes multiple (perturbation, batch, sign) items sequentially
   - Store K/V to a per-block scratch buffer (160 × 32KB = 5MB in L2 cache)
   - Load K/V tiles for FlashAttention from L2-cached scratch
   - This avoids the tl.gather overhead that killed the Triton FlashAttention attempt

4. **Triton cooperative kernel** (tl.range with warp_specialize=True):
   - Triton 3.6 has experimental warp specialization
   - Assign 2 warps to matmul (tensor cores) and 2 warps to non-matmul (scalar)
   - Requires Blackwell GPU (SM 100+) — NOT available on Ada SM 89

5. **Reduce the two (128,128) matrices** by restructuring computation:
   - For attention: already tried FlashAttention via tl.gather (8% slower due to gather)
   - For output proj: already tried tiled CE with online log-sum-exp (no improvement)
   - Untried: compute output projection in bf16 accumulator (output_dtype=tl.bfloat16
     in tl.dot) to halve register footprint. The CE loss needs f32 but could convert
     after the matmul.

6. **Different thread mapping**: Instead of 128 threads per block processing one
   (perturbation, batch, sign), try 256 threads processing two batch elements.
   This shares perturbation vector loads across 2 sequences.

### ALREADY ELIMINATED (don't repeat)

These have ALL been tried and don't work:
- FP8 tensor cores (slower + quality loss)
- num_warps=2 (much slower), num_warps=8 (baseline), num_warps=16 (slower)
- BLOCK_K=64 (baseline), BLOCK_K=32 (current, marginal improvement)
- VOCAB_PAD<128 (Triton requires power-of-2 arange)
- CPU pre-shuffle (no speed gain)
- Per-batch float() sync removal (no speed gain)
- Adaptive HALF_POP (quality fails on seed 7)
- Triton FlashAttention via tl.gather/tl.ravel/tl.reshape (8% slower, gather overhead)
- Tiled output projection with online log-sum-exp CE (no improvement)
- FFN pipelining via tl.range num_stages=2 (2x slower)
- Split kernel into attn + ffn_ce (12% slower, HBM scratch traffic)
- HALF_POP=3072/3584 with HP search (quality fails 3-seed threshold)
- LR schedules (cosine, warmup+cosine) at HALF_POP=3072 (no quality improvement)
- Rank-based fitness shaping (gradient scaling issues, diverged)
- maxnreg via monkey-patched jax_triton (doesn't affect compilation due to caching)
- bf16 attention weights (no speed improvement)
- num_stages=1/2/3 (no difference, kernel is compute-bound)
- CUDA scalar kernel (14x slower without tensor cores)
- CUDA WMMA kernel (11x slower, shared memory staging overhead + scalar non-matmul code)

---

## What Already Works (do not change unless improving speed)

- bf16 forward passes with fp32 perturbation math
- Constant label smoothing alpha=0.50
- Adam optimizer (β1=0.9, β2=0.999, eps=1e-6) with bias correction
- Sigma=0.02, LR=0.010, no LR decay
- Fused Triton kernel (Grid: HALF_POP × BATCH × 2, num_warps=4, BLOCK_K=32)
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
