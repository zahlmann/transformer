# EGGROLL Transformer — Agent Program

*You are an AI researcher. Your job: train a small decoder-only transformer using
EGGROLL (Evolution Strategies with low-rank perturbations) instead of backprop.
Match backprop val_loss at the same number of epochs while minimizing time and memory.
You work autonomously — run experiments, log results, and keep going.*

---

## Current State (2026-03-25)

**Goal: match backprop val_loss (2.45) at 10 epochs, same architecture.**
**Best EGGROLL 10ep: val_loss=2.60.** Gap = 0.15.

Best config: `train_eggroll_triton.py` with HALF_POP=4096, LR=0.020, LR_DECAY=0.95,
momentum=0.5, alpha=0.50, Gaussian vectors, N_ACCUM=1.
Uses fused Triton kernel for the full forward pass + CE loss (4.7x speedup over JAX vmap).

---

## Fairness Rules (ENFORCED by validate.py)

These constants are LOCKED. Changing them makes the comparison unfair.

```python
D_MODEL = 64       # model width
N_HEADS = 2        # attention heads
N_LAYERS = 1       # transformer layers
CONTEXT_LEN = 128  # sequence length
BATCH_SIZE = 128   # training batch size
EPOCHS = 10        # LOCKED — same as backprop baseline
TEMPERATURE = 2.0  # CE temperature
```

**What counts as cheating:**
- Changing EPOCHS (more epochs = more gradient steps = unfair)
- Changing architecture (d_model, n_layers, n_heads, d_ff)
- Changing batch size (affects effective learning)
- Subsetting data or changing sequence length
- Different eval methodology

**What is allowed:**
- Any ES algorithm change (population, perturbation strategy, gradient estimation)
- Hyperparameter tuning (LR, sigma, alpha, momentum, etc.)
- Kernel optimizations (Triton, fusion, fp8, etc.)
- Per-layer LR scaling
- Different perturbation structures (orthogonal, guided, etc.)
- Cross-batch state (momentum) — essential for transformer ES

---

## Files

```
program.md                    — this file (read first)
validate.py                   — LOCKED 3-seed validation with locked constants check
benchmark.py                  — fast single-seed comparison (EGGROLL vs backprop)
data.py                       — char-level Shakespeare dataset (65 vocab, 7842 train seqs)
model.py                      — decoder-only transformer (d=64, 1 layer, 2 heads, 66K params)
train_backprop.py             — backprop baseline (SGD, LR=0.30, val_loss=2.45 at 10ep)
train_eggroll_triton.py       — BEST: fused Triton kernel EGGROLL
train_eggroll_optimized.py    — JAX vmap EGGROLL (slower, reference implementation)
train_eggroll.py              — fp32 EGGROLL (superseded)
kernels/fused_transformer_ce.py — Triton kernel: full transformer forward + CE loss
validate_kernel.py            — validates Triton kernel output vs JAX forward pass
benchmark_kernel.py           — benchmarks Triton vs JAX vmap speed
profile_eggroll.py            — time breakdown (forward=99%, QR=1%, grad=0%)
cuda_kernels_docs/            — Triton + jax-triton documentation
results.tsv                   — experiment log (tab-separated)
```

---

## Results — Complete History (10 epochs only)

### Backprop baselines

| LR | val_loss | ppl | Time | Notes |
|----|----------|-----|------|-------|
| 3e-4 | 3.59 | 36.1 | 6.1s | initial under-tuned |
| 2e-2 | 2.70 | 14.9 | 1.3s | |
| 1e-1 | 2.50 | 12.2 | 1.3s | |
| **3e-1** | **2.45** | **11.6** | **1.3s** | **ceiling — target to match** |

### EGGROLL results (10 epochs, bf16)

| Config | val_loss | ppl | Time | Key change |
|--------|----------|-----|------|-----------|
| fp32, pop=512 | 3.94 | 51.4 | 148s | first working |
| bf16, pop=2048, alpha=0.20 | 2.83 | 17.0 | 270s | bf16 breakthrough |
| bf16, pop=2048, alpha=0.50, mom=0.5 | 2.70 | 14.8 | 270s | alpha+momentum |
| bf16, pop=4096 (accum=2), ortho QR | 2.67 | 14.4 | 540s | best JAX vmap |
| triton kernel, pop=2048 | 2.67 | 14.5 | 119s | 4.5x kernel speedup |
| triton, pop=8192, LR=0.020, decay=0.95 | 2.64 | 14.0 | 220s | best SGD+momentum |
| triton, pop=8192, **Adam** (β1=0.6, β2=0.99), LR=0.006, σ=0.03 | **2.60** | **13.5** | **219s** | **best 10ep** |

**Current gap: 2.60 − 2.45 = 0.15**

### What did NOT work (at more epochs — invalid approach)

Running 20/30/50/100 epochs reaches 2.51 but this is cheating: you could give backprop
the same extra epochs and it would improve too (backprop at 30ep reaches 2.17). The
comparison must be at the SAME epoch count (10). Do not increase EPOCHS.

---

## What Worked (apply from day 1)

### 1. bf16 forward passes — biggest quality win (−0.87 loss)
Cast perturbed params to bf16, keep perturbation/gradient math in fp32. bf16 noise
acts as implicit regularization for ES. Does NOT help backprop.

### 2. Constant label smoothing alpha=0.50 (−0.13 loss)
Unlike MNIST (adaptive decay), transformers need CONSTANT high alpha. Smoothing creates
a softer fitness landscape that ES can estimate well. Label smoothing HURTS backprop.

### 3. Momentum beta=0.5 (−0.03 loss)
ES gradients have high variance; momentum smooths across batches. beta=0.9 explodes
(effective LR = LR/(1-beta) = 10x). Sweet spot: beta=0.5.

### 4. Per-layer LR scaling
Small params (<256): 3.0x, attention (<4096): 1.5x, medium: 1.0x, large FFN (>8192): 0.7x.

### 5. Orthogonal QR vectors (−0.02 loss)
QR-orthogonalize perturbation vectors when HALF_POP ≤ vec_dim (2306).

### 6. Temperature T=2.0 in smoothed CE loss
T=1.0 too sharp, T=3.0 too flat. T=2.0 is the sweet spot.

### 7. Fused Triton kernel — 4.7x speedup
Full transformer forward pass + CE fused into one kernel. Grid: (HALF_POP, BATCH, 2).
Eliminates all HBM intermediate writes. Validated: max rel error 0.01%.

### 8. Per-subgroup Winsorized z-score
K=8 subgroups, clip ±2.0. Same as MNIST.

### 10. Adam optimizer with bias correction (−0.04 loss)
Proper Adam (β1=0.6, β2=0.99, eps=1e-6) with bias correction. The earlier attempt
without bias correction diverged because v starts at zero → 1/sqrt(v+eps) is enormous.
Bias correction: m_hat = m/(1-β1^t), v_hat = v/(1-β2^t) handles the cold start.
Key: no LR decay needed (Adam self-adjusts). LR=0.006 is optimal (tested 0.003-0.012).
Note β1=0.6 matches the momentum parameter — Adam's first moment IS the momentum.

### 9. Higher population with kernel speed
Kernel enables pop=8192 in 220s (was 540s for pop=4096 without kernel). Larger pop
gives −0.03 loss at 10 epochs.

---

## What Did NOT Work

1. **Adaptive alpha decay** — oscillation, transformer needs constant alpha
2. **Higher sigma (0.06)** — noisier gradients, worse quality
3. **Momentum beta=0.9** — diverges (10x effective LR)
4. **Weight tying** — hurts both backprop and EGGROLL
5. **Vectorized base+correction forward** — 3x slower, XLA can't optimize
6. **Nested lax.scan for epoch** — 32s JIT, no speed gain
7. **Large POP_CHUNK** — memory saturation at >16
8. **T=3.0** — gradient signal too weak
9. **Adam WITHOUT bias correction** — catastrophic divergence (v starts at zero).
   Adam WITH bias correction works well — see "What Worked" #10.
10. **Cosine LR schedule** — no improvement over exponential decay
11. **More epochs** — invalid comparison; backprop also improves with more epochs
12. **Per-parameter gradient clipping** — clips gradient norms to 1.0, massively degrades
    quality (2.92 vs 2.64). Winsorized z-score already controls gradient magnitude.
13. **Within-batch guided perturbations** — split pop into 25% explore + 75% guided.
    Rank-1 SVD of rough gradient for guide direction. Result: 2.66, within noise of
    baseline. The rough gradient from 1024 explore perturbations is too noisy to guide.

---

## What to Try Next (Quality — close the 0.19 gap)

These are the remaining ideas to improve gradient quality at 10 epochs:

1. **Within-batch guided perturbations**: Split population — use first 25% for rough
   gradient estimate, bias remaining 75% toward that direction. Better gradient quality
   from the same total population.

2. **Natural gradient / Fisher information**: Scale the ES gradient by an estimate of
   the inverse Fisher information matrix. This accounts for the curvature of the loss
   landscape and can dramatically improve convergence.

3. **Antithetic Gaussian with variance reduction**: Instead of plain antithetic pairs,
   use control variates or importance sampling to reduce variance of gradient estimates.

4. **Higher-rank perturbations**: Currently rank-1 (outer(b,a)). Try rank-2 or rank-4
   perturbations for richer gradient information per evaluation.

5. **Population scheduling**: Start with high pop (good exploration) and reduce over
   epochs (exploitation). Invest compute where it matters most.

6. **Gradient clipping on ES updates**: Clip the ES gradient norm before the momentum
   step. May prevent occasional bad updates from high-variance estimates.

7. **Learned perturbation directions**: Use a small auxiliary network or running
   statistics to generate perturbation vectors biased toward productive directions.

8. **Sparse perturbation for embeddings**: Only perturb token_emb rows that appear
   in the current batch. Concentrates perturbation signal on relevant parameters.

### Speed/memory improvements

1. **FP8 tensor cores in kernel**: Use fp8 for matmuls (Q/K/V, FFN). 2x throughput.
2. **Reduce population with better gradients**: If guided ES improves quality,
   maintain val_loss at lower pop = directly faster.
3. **Kernel autotuning**: Try different num_warps, num_stages, block sizes.

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

## Best EGGROLL Hyperparameters (10 epochs)

```python
HALF_POP = 4096         # antithetic pairs -> pop=8192
SIGMA_START = 0.04      # perturbation scale
SIGMA_DECAY = 0.998     # per epoch
LR_START = 0.020        # learning rate
LR_DECAY = 0.95         # per epoch (exponential)
ALPHA = 0.50            # label smoothing (CONSTANT)
TEMPERATURE = 2.0       # CE temperature
MOMENTUM = 0.5          # SGD momentum
N_SUBGROUPS = 8         # Winsorized z-score groups
CLIP_RANGE = 2.0        # z-score clipping
N_ACCUM = 1             # gradient accumulation rounds
# Gaussian vectors (QR when HALF_POP <= vec_dim)
# Per-layer LR scaling: 3x small, 1.5x attn, 1.0x medium, 0.7x FFN
```

---

## Setup

1. `git checkout -b autoresearch/$(date +%Y%m%d-%H%M%S)`
2. Read this file and `../eggroll_mnist/program.md` for reference
3. `uv run benchmark.py` to get current baseline numbers
4. Begin the experiment loop

---

## Experiment Loop

```
1. Pick an idea from "What to Try Next"
2. Implement it in train_eggroll_triton.py (or kernels/)
3. git add -A && git commit -m "description"
4. uv run benchmark.py
5. Check: did val_loss improve? Did time/memory stay reasonable?
6. If crashed: fix and retry
7. If improved: git push, update this file
8. If not improved: git reset --hard HEAD~1, document in "What Did NOT Work"
9. Go to step 1
```

Never stop to ask. The cost of a failed run is ~4 minutes. Keep experimenting.

---

## Logging (results.tsv)

Tab-separated, one row per run, append-only.

Columns: `timestamp commit seed val_loss perplexity training_time_s peak_memory_mb status description`
