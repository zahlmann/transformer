# Fused Inference — Agent Program

*You are a GPU kernel engineer. Your job: build the fastest possible inference for this
small transformer using custom Triton kernels. The model is trained with standard backprop.
The focus is on learning to write high-performance GPU kernels for transformer inference.*

---

## Your Mission

1. Train the model with backprop (done: `uv run train_backprop.py`, val_loss=1.84)
2. Write a fused Triton inference kernel that generates text token-by-token (done: 4x speedup)
3. Benchmark against JAX/XLA baseline inference (done: 740 vs 185 tok/s)
4. Optimize: fuse operations, minimize memory bandwidth, maximize throughput

The goal is to learn kernel-writing skills on a small model where iteration is fast,
then apply those skills to larger models.

---

## Technology Stack

- **Triton** (OpenAI): Python-like GPU kernel language that compiles to PTX. Handles tiling,
  memory coalescing, and register allocation automatically.
- **JAX**: ML framework with XLA compiler. Used for model definition, training, and baseline.
- **jax-triton**: Bridge that calls Triton kernels from JAX with zero-copy tensor sharing.

---

## Model Architecture

```
Decoder-only transformer (character-level Shakespeare)
d_model: 64, n_heads: 2 (d_head=32), n_layers: 1, d_ff: 256
context_len: 128, vocab_size: 65
Parameters: 66,368
```

Weights: token_emb (65,64), pos_emb (128,64), Q/K/V/O (64,64 each), FFN up (64,256),
FFN down (256,64), LN scales/biases, output_proj (64,65).

---

## Current Kernel Infrastructure

### `kernels/fused_prefill.py` — Prefill Kernel

Fused Triton kernel that processes the ENTIRE forward pass in one kernel call:
Embedding → LayerNorm → Multi-Head Attention → FFN → LayerNorm → Output Projection

Key techniques:
- **All weights in registers.** Each weight matrix fits in register file. No HBM
  round-trips between layers.
- **bf16 tensor core matmuls with f32 accumulation.** Uses hardware tensor cores
  for 8-16x throughput vs scalar fp32.
- **K-tiled FFN loop.** The (128,256) FFN is tiled into 8 blocks of (128,32) using
  `tl.range(0, 256, 32)` with dynamic range to prevent register pressure from unrolling.
- **Causal attention in-register.** Full (128,128) attention matrix computed and softmaxed
  without going through HBM.
- **num_warps=4 (128 threads/block).** Optimal — fewer warps = more registers/thread.
  255 regs/thread, 0 bytes spill.
- **Outputs KV cache** for the decode phase.

### `kernels/fused_decode.py` — Decode Kernel

Single-token decode using KV cache. Uses element-wise ops (not tensor cores, since M=1).
Reads KV cache, computes attention over all past tokens, outputs logits + new K/V vectors.
JAX wrapper updates the cache via scatter.

---

## Results

```
Prompt: 64 tokens, Generate: 64 tokens

Triton fused:   740 tok/s  (86.5 ms)
JAX baseline:   185 tok/s  (362.6 ms)
Speedup:        4.0x
```

---

## YOUR NEXT TASK: BPE Tokenization + Tiled Output Kernel

The immediate goal is to switch from character-level (vocab=65) to BPE subword tokens.
This is the first scaling step — it improves text quality AND forces you to learn output
tiling, the technique that unlocks everything bigger.

### Why this matters

With vocab=65, the output projection is (64, 65) = 4K params — trivially fits in
registers. With BPE vocab=4096, the output projection is (64, 4096) = 256K params
= 512KB in bf16. That's 4x larger than the entire register file. **You cannot load
this matrix into registers.** You must tile it.

This is exactly the constraint that real models face. Solving it here teaches tiling
patterns that apply everywhere.

### Step-by-step plan

**Step 1: Add a BPE tokenizer.**

Use `tiktoken` (OpenAI's tokenizer library). Specifically use the `gpt2` encoding which
has a vocab of 50257. For our tiny model, this is too large — we'll use a restricted
vocabulary.

Approach: tokenize the Shakespeare corpus with tiktoken, find the top-K most frequent
tokens (start with K=512 or K=1024), and remap to a compact vocabulary. This gives us
real subword tokens while keeping the embedding table manageable.

Modify `data.py`:
- Add tiktoken tokenization alongside the existing char-level path
- Build a restricted vocab from the corpus (top-K frequent tokens + special tokens)
- Save the vocab mapping so inference can decode tokens back to text
- Keep the same `prepare_data` interface: returns train_x, train_y, vocab_size, etc.

**Step 2: Update model.py for variable vocab size.**

The model already parameterizes vocab_size — just make sure init_transformer and
transformer_forward work with vocab_size=512 or 1024. Nothing fundamental changes
in the JAX model.

**Step 3: Retrain with BPE tokens.**

Run train_backprop.py with the new data. Expect lower loss (BPE tokens carry more
information per token than characters). Tune epochs/LR if needed — the model may
need more capacity (increase d_model to 128 or add a layer) to take advantage of
the richer tokenization.

**Step 4: Update the prefill kernel for large vocab.**

The current prefill kernel computes `logits = tl.dot(h_final, output_proj)` where
output_proj is (64, 128_padded). With vocab=1024, output_proj becomes (64, 1024).

The matmul `(128, 64) @ (64, 1024)` produces a (128, 1024) output = 128K f32 values
= 512KB. This doesn't fit in registers.

**Solution: tile the vocab dimension.** Process the output projection in chunks:
```
for v_start in tl.range(0, VOCAB_PAD, VOCAB_TILE):
    # Load a (64, VOCAB_TILE) slice of output_proj
    # Compute (128, VOCAB_TILE) logits
    # Store directly to HBM (don't accumulate — each tile is independent)
```

This is analogous to the existing FFN K-tiling loop. Each tile produces independent
logits that get written to HBM immediately.

Similarly, the embedding lookup may need tiling if the embedding table (vocab × d_model)
gets large, but at vocab=1024 and d_model=64 it's only 128KB — might still fit.

**Step 5: Update the decode kernel for large vocab.**

Same tiling approach for the output projection. The decode kernel already uses
element-wise ops, so the pattern is:
```
for v_start in tl.range(0, VOCAB_PAD, VOCAB_TILE):
    # Load (64, VOCAB_TILE) slice of output_proj
    # Compute (VOCAB_TILE,) logits via element-wise dot
    # Store to output
```

**Step 6: Benchmark.**

Compare Triton vs JAX with the BPE model. The tiled output adds some overhead vs the
register-only version, so the speedup may decrease. Quantify this — it tells you the
cost of leaving the register-only regime.

Generate text and verify it's coherent subword-tokenized English.

### Key design decisions

- **Vocab size**: Start with 512. If the model is too small to learn it well, try 256.
  If it works, try 1024. The sweet spot is where the model produces coherent text and
  the kernel needs real tiling.
- **d_model**: May need to increase from 64 to 128 to give the model enough capacity for
  BPE tokens. If you do, the attention and FFN weight matrices also need tiling (this is
  Phase A2 territory — tackle it only if necessary for BPE to work).
- **VOCAB_TILE size**: Start with 128 (power of 2, fits in registers). This means
  vocab=512 needs 4 tiles, vocab=1024 needs 8 tiles.

### What NOT to change

- Don't change the attention mechanism or FFN structure yet
- Don't add layers yet — keep it 1 layer
- Don't change context_len from 128 yet
- Keep the same kernel structure (one kernel for prefill, one for decode)
- Keep num_warps=4

---

## Future Directions (after BPE)

### Phase A2: Bigger Model

Scale d_model (64→128→256), add layers (1→2→4), increase context (128→512→1024).
Key thresholds:
- d_model=128: weights ~260KB in bf16, exceeds register file. Need multi-block tiling.
- n_layers=2+: can't fuse everything into one kernel. Need one kernel per layer or
  a persistent kernel that loops over layers.
- context=512+: attention matrix (512×512 = 1MB) can't fit in registers. Need
  FlashAttention-style tiled attention.

### Phase B: Kernel Optimizations

- **Quantization**: INT8/FP8 weights for decode (memory-bound, ~1.5-2x speedup)
- **Batched decode**: multiple sequences share weights, enables tensor cores
- **Persistent kernel**: eliminate launch overhead
- **Speculative decoding**: trade compute for latency

### Phase C: Multi-Layer Fusion

For 2+ layers, keep data in fast memory between layers:
- One kernel per layer (simple, HBM between layers)
- Shared memory handoff (stays on-chip, ~164KB limit)
- Persistent kernel (loop over layers in registers, most complex)

---

## Profiling Commands

```bash
uv run train_backprop.py             # train the model (4s)
uv run inference_benchmark.py        # inference speed comparison
```

---

## Files

```
program.md                          — this file (read first)
inference_guide.md                  — ground-up explanation of GPU kernels + this project
README.md                           — project overview
model.py                            — JAX transformer model (inference baseline)
train_backprop.py                   — standard backprop training
data.py                             — char-level Shakespeare dataset
kernels/fused_prefill.py            — fused Triton prefill kernel
kernels/fused_decode.py             — fused Triton decode kernel
inference_benchmark.py              — speed comparison benchmark
```

---

## Kernel Engineering Lessons

1. **num_warps=4 is optimal for this model.** Fewer warps = more registers per thread.
   num_warps=2 is slower (poor occupancy), num_warps=8 is slower (thread overhead).

2. **Dynamic loops (`tl.range`) prevent register-pressure blowup from unrolling.**
   The FFN K-tiling loop MUST use `tl.range`, not `tl.static_range`. Static unrolling
   of 8 iterations causes 340 bytes of register spilling.

3. **bf16 matmuls with f32 accumulation are the sweet spot.** FP8 was tried and was slower
   (register pressure from casts outweighed tensor core gains). fp16 out_dtype didn't help.

4. **The entire model fits in registers.** Total weight storage: ~66K params × 2 bytes =
   132KB. With 128 threads/block and 255 regs/thread, we have 128 × 255 × 4 = 131KB of
   register file. Tight but fits (weights are loaded on-demand, not all live simultaneously).

5. **Attention scores (128×128) in registers are the bottleneck.** This is 16K f32 values
   = 64KB. It works at num_warps=4 but is the largest live tensor.

6. **Decode can't use tensor cores.** With M=1, tensor cores need at least 16×16 tiles.
   Element-wise ops are faster for single-token decode.

7. **maxnreg doesn't help.** The kernel is compute-bound, not occupancy-bound. Forcing
   fewer registers just causes spilling.
