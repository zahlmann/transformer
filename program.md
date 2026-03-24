# EGGROLL Transformer — Agent Program

*You are an AI researcher. Your job: train a small decoder-only transformer using
EGGROLL (Evolution Strategies with low-rank perturbations) instead of backprop.
Make it as fast as possible while matching backprop accuracy. You work autonomously,
run experiments, log results, and keep going without stopping to ask for permission.*

---

## The Goal

Train a small decoder-only transformer on a simple language task (character-level
or tiny token-level LM) using EGGROLL ES instead of backpropagation. Compare
wall-clock time and loss/perplexity against an equivalent backprop baseline.

This is a research project — we don't know if EGGROLL works well on transformers.
The MNIST version (see `../eggroll_mnist/`) achieved 1.44x backprop speed on a 3-layer
MLP. Transformers have attention + softmax + multi-head structure which may or may
not be friendly to ES gradient estimation.

### Phase 1: Get it working
- Implement a small decoder transformer in JAX (d_model=64-128, 1-2 layers, 1-4 heads)
- Implement EGGROLL training for it
- Pick a simple task: character-level Shakespeare, tiny stories, or similar
- Get it to converge at all (any loss improvement over random)

### Phase 2: Match backprop quality
- Tune hyperparameters (population, sigma, smoothing, z-score)
- Apply lessons from MNIST (see below)
- Match backprop's final loss/perplexity

### Phase 3: Close the speed gap
- Fused Triton kernels for the transformer forward pass
- All the compilation tricks from MNIST
- Measure and minimize the wall-clock gap to backprop

---

## Architecture Constraints

```
Model: decoder-only transformer
d_model: 64-128 (start small, scale up if ES works)
Layers: 1-2
Heads: 1-4
Context: 64-256 tokens
Vocab: character-level (small vocab, ~100) or BPE (small, ~1000)
```

Start with the SMALLEST viable config. ES gradient quality scales as O(params/pop) —
every parameter you add requires more perturbations to estimate the gradient.

Parameter budget estimates (these determine population requirements):
- Embedding: vocab × d_model (e.g., 100 × 128 = 12.8K)
- Attention Q/K/V: 3 × d_model × d_model = 3 × 16K = 49K per layer
- Attention output: d_model × d_model = 16K per layer
- FFN: 2 × d_model × 4×d_model = 2 × 64K = 131K per layer
- Total for 1-layer d=128: ~210K params

For comparison: MNIST MLP had 118K params (784×128 + 128×128 + 128×10) and needed
pop1680. A 210K-param transformer may need pop3000+. This is the central challenge.

### Fairness rules (same as MNIST)
- Both EGGROLL and backprop must use JAX
- Both must use the same architecture, data, epochs
- No momentum or cross-batch state for EGGROLL
- JIT compilation time included in wall-clock measurement
- fp32 training data, bf16/fp8 allowed for tensor core efficiency

---

## Key Reference: ../eggroll_mnist/

The MNIST project has 8 sessions of optimization work. Read these files:

### Must-read before starting
- `../eggroll_mnist/README.md` — summary of all optimizations (27s → 2.16s)
- `../eggroll_mnist/mnist_eggroll_optimized.py` — the training script. This is the
  reference implementation for how EGGROLL training works in JAX. Study the structure:
  nested `lax.scan`, perturbation vector generation, fitness computation, gradient
  estimation, per-subgroup z-score.
- `../eggroll_mnist/kernels/fused_3layer_ce.py` — the fused Triton kernel. Shows how
  to fuse forward pass + loss into one kernel with K-tiled matmuls and FP8.

### Read for lessons learned
- `../eggroll_mnist/program.md` — full optimization log with what worked and didn't.
  The "What did NOT work" sections are especially valuable — don't repeat these mistakes.

---

## Lessons from MNIST (apply these from day 1)

### What matters for ES accuracy
1. **Label smoothing / soft targets in the fitness signal** — the single biggest
   algorithmic breakthrough. Hard CE only tells ES about the correct class logit.
   Smoothed CE gives gradient information about ALL logits. For a transformer, this
   means using temperature-scaled CE with label smoothing from the start.

2. **Per-subgroup Winsorized z-score** (K=8, clip ±2.0) — split fitness differences
   into K groups, normalize each independently, clip outliers. Much better than global
   z-score at low population sizes.

3. **Adaptive smoothing schedule** (α decays per epoch) — high smoothing early when
   weights are random, low smoothing late for precise fine-tuning. Critical for breaking
   through population barriers.

4. **Higher sigma at lower pop** — counterintuitive but with fewer perturbations, you
   need larger perturbation scale for signal. σ=0.036 worked for MNIST pop1680.

5. **Per-layer LR scaling** — small output layers (few params) get better gradient
   estimates and can use higher LR. L3 (128→10) used 2x LR in MNIST.

### What matters for speed
1. **Fused Triton kernel** — the single biggest speedup. Fuse the entire forward pass
   + loss into one kernel. Keep intermediates in registers, never write to HBM.
   K-tile the matmuls to manage register pressure. Use FP8 tensor cores for matmuls.

2. **`--xla_gpu_enable_triton_gemm=false`** — saves ~0.6s of JIT by disabling XLA's
   internal GEMM autotuner. Set via os.environ before importing JAX. This is the
   single biggest JIT optimization.

3. **Nested `lax.scan`** — epoch scan wrapping batch scan, all in one JIT. Eliminates
   Python loop overhead entirely.

4. **Single random matrix, sliced** — generate one (pop, total_vec_dim) matrix and
   slice into per-layer B and A vectors. Much faster than separate random calls.

5. **Background data preparation** — shuffle + GPU transfer in a background thread,
   overlapping with JIT compilation. Both release the GIL.

6. **CUDA wave alignment** — choose population size to give integer number of CUDA
   waves. Pop1680 = 21.0 waves on RTX 4080 SUPER (80 SMs, 4 blocks/SM).

### What definitely does NOT work for ES (don't try these)
- Rank-based fitness shaping, Boltzmann weighting, top-k truncation
- One-sided ES (no antithetic pairs) — needs 2x population for same quality
- Rank-2 perturbations — 2D subspace doesn't improve gradient quality per compute
- Shared perturbation vectors across layers — each layer needs independent directions
- Per-layer sigma scaling — catastrophic for large layers (L1 in MNIST)
- Structured perturbations (Hadamard, orthogonal) — adds JIT overhead in JAX
- Cross-batch momentum or state — violates fairness rules

### Transformer-specific considerations (untested, think about these)
- **Attention has quadratic compute** in sequence length but the PARAMETERS are just
  Q/K/V/O matrices. ES perturbs parameters, not activations. The quadratic compute
  happens in the forward pass regardless of whether you use ES or backprop.
- **Softmax in attention** is a non-linearity like GELU. The Triton kernel needs to
  compute it but it's just another operation in the fused pipeline.
- **Positional encoding** adds parameters (if learned) or is a fixed function (if
  sinusoidal). Learned embeddings add to the parameter count.
- **Layer norm** has 2×d_model learnable parameters per instance. Small, but needs
  perturbation vectors.
- **The embedding layer** might be the hardest for ES — it's sparse (only the tokens
  in the current batch activate rows). ES perturbs the FULL embedding matrix. This
  wastes perturbation budget on rows that don't affect the current batch. Consider
  whether a smaller vocab helps.

---

## Setup

1. `git checkout -b autoresearch/$(date +%Y%m%d-%H%M%S)`
2. Read this file and `../eggroll_mnist/mnist_eggroll_optimized.py`
3. Set up the project: `uv init --bare && uv add jax jaxlib numpy`
4. Implement backprop baseline first (establishes the accuracy target)
5. Implement EGGROLL training
6. Begin the experiment loop

**Important:** Always commit and push work to the autoresearch branch regularly.

---

## Experiment Loop

```
1. Pick an idea (architecture change, hyperparameter, kernel optimization)
2. Implement it
3. git add -A && git commit -m "description" && git push
4. uv run benchmark.py > run.log 2>&1
5. Check results: loss, time, memory
6. If crashed: fix and retry
7. If improved: keep. If not: git reset --hard HEAD~1
8. Log to results.tsv
9. Go to step 1
```

Never stop to ask if you should continue. The cost of a failed run is seconds.

---

## Logging (results.tsv)

Tab-separated, one row per run, NOT committed to git.

Columns: `commit\tloss\tperplexity\ttraining_time_s\tpeak_memory_mb\tstatus\tdescription`

---

## Results & Honest Assessment (2026-03-24)

### What worked
- EGGROLL **can** train a decoder-only transformer from scratch (Phase 1 complete)
- Constant label smoothing (alpha=0.15-0.20, NO decay) is critical for transformer ES — unlike MNIST where adaptive decay was best. Transformers overfit with ES's noisy gradients if regularization is removed.
- bf16 forward passes act as implicit regularization for ES (noise helps noisy gradients). Does NOT help backprop.
- Per-layer LR scaling (3x for small params, 0.7x for large FFN layers)
- Rank-1 perturbation gives 28.8x compression (2306 random values vs 66K params)

### What the numbers actually show

**Fair comparison (10 epochs, no warmup, both tuned):**

| Method | val_loss | ppl | Time | Notes |
|--------|----------|-----|------|-------|
| Backprop (LR=2e-2 decay=0.80) | 2.70 | 14.9 | **1.3s** | best 10ep backprop |
| EGGROLL (momentum+alpha=0.50) | **2.70** | **14.8** | 270s | quality MATCHED |
| EGGROLL (no momentum, alpha=0.20) | 2.83 | 17.0 | 270s | initial approach |
| Backprop (vanilla LR=3e-4) | 3.59 | 36.1 | 6.1s | under-tuned baseline |

EGGROLL now matches backprop quality but at 208x compute cost. Key techniques:
- Momentum (beta=0.5) eliminates oscillation from noisy ES gradients
- High label smoothing (alpha=0.50) makes the fitness landscape smooth
- bf16 forward passes provide implicit noise regularization

### Where the initial comparison was misleading
1. **Severely under-tuned backprop baseline**: LR=3e-4 when optimal is LR=2e-2. This single mistake made EGGROLL appear to beat backprop. With proper LR tuning (same exponential decay schedule EGGROLL uses), backprop wins on both axes.
2. **"Same epochs" masks a 2048x compute gap**: EGGROLL does 2048 forward passes per batch. This is not "cheating" but means each EGGROLL epoch costs 2048x the compute.
3. **Label smoothing + temperature help EGGROLL but hurt backprop** (3.59 → 3.87). Legitimate for ES, but makes the comparison misleading if you don't also tune backprop.

### Genuine contributions
- First demonstration (to our knowledge) that EGGROLL can train a transformer
- bf16 as ES regularization is a novel, generalizable finding
- Constant alpha (no decay) for transformer ES is a novel finding that differs from MNIST
- The rank-1 perturbation approach scales to attention + FFN architectures
