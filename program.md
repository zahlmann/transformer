# Fused Inference — Agent Program

*You are a GPU kernel engineer. Your job: build the fastest possible inference for this
small transformer using custom Triton kernels. The model is trained with standard backprop.
The focus is on learning to write high-performance GPU kernels for transformer inference.*

**IMPORTANT: Commit and push after every meaningful step. Don't batch up changes.**

---

## Mission

Build the fastest possible inference for this transformer using custom Triton kernels.

### Completed

1. Fused Triton prefill + decode kernels (single kernel for small model)
2. BPE tokenization + tiled output projection (vocab=1024, zero overhead)
3. Multi-block prefill for d_model=128 (BLOCK_SEQ=32, 3 kernels/layer)
4. Fused multi-layer decode (all layers + output in one kernel)
5. Host-side optimization: in-kernel cache updates, precomputed bf16 weights
6. TinyStories dataset (14M tokens) + trained BPE tokenizer (vocab=4096)
7. Scaled to d_model=256, 4 layers, 8 heads (5.3M params, ppl=5.40)
8. Fused N-layer decode with packed weights/caches (any layer count)
9. FlashAttention for context>256 (tiled KV + online softmax)
10. AdamW + LR warmup + cosine decay training improvements
11. Speculative decoding with parallel verification kernel
12. Scaled to d=512, 8 layers, 16 heads (29.7M params, ppl=2.91)
13. Multi-SM decode kernel: grid=(N_HEADS,) with atomic barriers (5.2x speedup, 287→1519 tok/s)
14. KV-split parallelism: grid=(N_HEADS*2,) FlashDecoding-style (1734→1851 tok/s, +10%)
15. Split barrier: separate counter/done-flag cache lines (1851→1937 tok/s, +5%)
16. Batched decode: B sequences per kernel with shared weights (1890→3422 tok/s at B=4, +81%)

### Tried but didn't help (single-sequence decode)

- **GQA** (4 KV heads for 16 Q heads): ppl 2.96 (vs 2.91). KV cache 8.4→2.1 MB, total
  data 67.7→55.1 MB (fits in L2). But 0% decode speedup — kernel is barrier-limited,
  not memory-limited. GQA value is for BATCHED inference (KV cache scales with batch).
- **Parallel residual** (attn||FFN, 9 barriers instead of 17): ppl 3.01. Only 1.3% speedup.
  Barrier cost is dominated by straggler VARIANCE (proportional to total work), not
  fixed overhead. Halving barriers saves only ~8µs of fixed overhead.
- **num_warps sweep**: 2/4/8 all within 5%. Not warp-limited.
- **num_stages=2**: marginal (within noise of num_stages=1).

### Current Performance (RTX 4080 Super)

```
XL model (d=512, h=16, l=8, ctx=512, 29.7M params, ppl=2.91):
  Decode (multi-SM):   1937 tok/s (0.52 ms/tok)  ← 6.8x speedup via grid=(32,) + split barrier
  Decode (no sync):    3108 tok/s (0.32 ms/tok)   (pure GPU, no int() per token)
  Decode (grid=16):    1734 tok/s (0.58 ms/tok)   (kv_splits=1, previous)
  Decode (single-SM):  287 tok/s  (3.49 ms/tok)   (original baseline)
  Prefill (128 tok):   6.1 ms    (21,168 tok/s)
  Weight buffer:       59.3 MB   (bf16)
  KV cache:            8.4 MB    (bf16, per sequence)
  Bandwidth util:      16%       (of 836 GB/s theoretical, was 2%)

Previous model sizes:
  d=64,  1L:    3056 tok/s
  d=128, 2L:    2589 tok/s
  d=256, 4L:    1396 tok/s
  d=512, 8L:     287 tok/s  → 1734 → 1851 → 1937 tok/s
```

**Key findings:**
- Multi-SM decode (grid=16) gave 6x speedup over single-SM.
- KV-split parallelism (grid=32, kv_splits=2) adds another 10% (1734→1851 tok/s).
  Splits KV cache tiles across 2 blocks per head (FlashDecoding-style).
  Merge uses online softmax correction (weighted average of normalized O-projections).
  kv_splits=4 was slightly slower (barrier contention with 64 blocks).
- Split barrier: separates arrival counter from done flag on different cache lines.
  Reduces L2 contention during spin-waiting (1851→1937 tok/s, +5%).
  Pure GPU throughput: 3108 tok/s (0.32 ms/tok).
- GPU→CPU sync (int()) costs 34% of total time (0.19 ms per token).
- Bandwidth utilization: 16%. Theoretical minimum: 0.081 ms/tok — 4.0x headroom (pure GPU).

---

## Technology Stack

- **Triton** (OpenAI): Python-like GPU kernel language that compiles to PTX. Handles tiling,
  memory coalescing, and register allocation automatically.
- **JAX**: ML framework with XLA compiler. Used for model definition, training, and baseline.
- **jax-triton**: Bridge that calls Triton kernels from JAX with zero-copy tensor sharing.

---

## Model Architecture

```
Decoder-only transformer
d_model: 64-512, n_heads: 2-16 (d_head=32), n_layers: 1-8, d_ff: 4*d_model
context_len: 128-512

Small:  d=64,  h=2,  l=1, vocab=65,   params=66,368     (Shakespeare, char)
Medium: d=64,  h=2,  l=1, vocab=1024, params=189,120    (Shakespeare, BPE)
Large:  d=128, h=4,  l=2, vocab=1024, params=674,304    (Shakespeare, BPE)
XL:     d=256, h=8,  l=4, vocab=4096, params=5,318,144  (TinyStories, BPE)
XXL:    d=512, h=16, l=8, vocab=4096, params=29,660,160 (TinyStories full, BPE)
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
- **Tiled output projection.** Vocab dimension tiled in VOCAB_TILE=128 chunks via
  `tl.range(0, VOCAB_PAD, VOCAB_TILE)`. Each tile independently computes and stores
  logits to HBM. VOCAB_SIZE/VOCAB_PAD passed as kernel `tl.constexpr` parameters.
- **Outputs KV cache** for the decode phase.

### `kernels/fused_decode.py` — Decode Kernel

Single-token decode using KV cache. Uses element-wise ops (not tensor cores, since M=1).
Reads KV cache, computes attention over all past tokens, outputs logits + new K/V vectors.
Same tiled output projection as prefill. JAX wrapper updates the cache via scatter.

---

## Results (Optimization History)

```
Phase A1: d=64, 1L, vocab=1024 (Shakespeare):
  Initial:                          740 tok/s
  + tiled output projection:        758 tok/s  (zero overhead from tiling)

Phase A2: d=128, 2L, vocab=1024 (Shakespeare):
  Multi-block (3 launches/step):    335 tok/s
  + fused decode (1 launch/step):   454 tok/s  (+35%)
  + precomputed bf16 weights:       713 tok/s  (+57%)
  + in-kernel cache updates:       2504 tok/s  (+3.6x, biggest win)

Phase A3: d=256, 4L, vocab=4096, 5.3M params (TinyStories):
  Fused N-layer decode:            1396 tok/s

Phase A4: d=512, 8L, vocab=4096, 29.7M params (TinyStories full):
  Fused N-layer decode:             287 tok/s  (3.48 ms/tok, 2% BW utilization)

Phase A5: Multi-SM decode optimization:
  Profiling revealed: kernel=93%, host=3%, argmax=4% of step time
  Multi-SM decode (grid=16):       1734 tok/s  (0.58 ms/tok, 14% BW utilization)
  Without int() sync:              2742 tok/s  (0.36 ms/tok)
  Speedup vs single-SM:              6.0x (with sync), 9.6x (without sync)

Phase A6: KV-split parallelism (FlashDecoding):
  KV splits=2, grid=32:           1851 tok/s  (0.54 ms/tok, 15% BW utilization)
  KV splits=4, grid=64:           1991 tok/s  (short-burst, but +barrier contention)
  Improvement over grid=16:         +10% sustained throughput
  Technique: split KV cache tiles across 2 blocks per head, merge with
    online softmax correction (weighted average of normalized O-projections)

Phase A7: Split barrier + profiling:
  Split barrier:                   1937 tok/s  (0.52 ms/tok, 16% BW utilization)
  Without GPU→CPU sync:            3108 tok/s  (0.32 ms/tok)
  Improvement: +5% with sync, +6.5% pure GPU
  Key insight: GPU→CPU int() sync costs 34% of total time (0.19 ms/tok)
  Technique: separate arrival counter and done-flag on different cache lines.
    Last-arriving block sets done flag; others poll done (not counter).

Key findings:
- Tiled output projection has ZERO overhead vs register-only.
- Python dispatch overhead per jt.triton_call is ~0.4ms. Fusing all decode
  into one kernel eliminated 128 launches (35% speedup).
- .astype(bf16) per decode step created 1764 unnecessary allocations.
  Precomputing once gave another 57% speedup.
- In-kernel KV cache updates were the biggest single win (3.6x).
- At d=512 single-SM, bandwidth utilization was only 2%. GPU was 98% idle.
- Multi-SM decode with atomic barriers: 5.2x speedup (287→1519 tok/s).
  Splits attention heads across 16 blocks, distributes FFN across all blocks.
  Uses release/acquire atomics for cross-block synchronization.
  Bandwidth utilization improved from 2% to 12%.
```

---

## COMPLETED: BPE Tokenization + Tiled Output Kernel

Switched from char-level (vocab=65) to BPE subword tokens (vocab=1024).
This was the first scaling step — forces tiled output projection in the kernel.

### What was done

1. **BPE tokenizer**: Trained on Shakespeare using HuggingFace `tokenizers` (ByteLevel
   BPE). GPT-2's tiktoken had 16-25% UNK rate on Shakespeare (11K unique tokens in corpus).
   The trained tokenizer has 0% UNK with vocab=1024.

2. **Tiled output projection**: Both kernels now tile the output projection in VOCAB_TILE=128
   chunks using `tl.range(0, VOCAB_PAD, VOCAB_TILE)`. VOCAB_SIZE and VOCAB_PAD are kernel
   constexpr parameters, so jax-triton JIT-compiles per vocab configuration.

3. **Results**: Zero overhead from tiling (4.35x speedup vs 4.0x baseline). The output
   projection is compute-light — attention dominates the kernel runtime. Text output is
   coherent Shakespeare-like dialogue.

### Design decisions taken

- **Tokenizer**: GPT-2 tiktoken had 24-46% UNK rate on Shakespeare (11K unique tokens).
  Trained a BPE tokenizer directly on the corpus using HuggingFace `tokenizers` — 0% UNK.
- **Vocab size**: 1024. Gives 8 tiles of 128, meaningful tiling work. Both 512 and 1024
  produce similar bits-per-character (~2.2) — the model's capacity is the bottleneck.
- **VOCAB_TILE**: 128 (same as FFN BLOCK_K). Each tile loads (64, 128) bf16 weights,
  produces (128, 128) f32 logits (prefill) or (128,) f32 logits (decode).
- **d_model stays at 64**: model is parameter-inefficient (embedding tables dominate)
  but the kernel engineering goal was achieved without changing the core architecture.

---

## COMPLETED: Phase A2 — Bigger Model

Scaled d_model 64→128, added 2 layers, 4 attention heads. Used multi-block approach:
3 Triton kernels per layer (proj, attn, ffn) + 1 output kernel, grid=(4,) with
BLOCK_SEQ=32. Per-block register peak: ~52KB (vs 127KB budget).

### Block kernels architecture

```
kernels/block_prefill.py:
  _proj_kernel:   LN + Q/K/V projections   (grid=4, each block=32 rows)
  _attn_kernel:   causal attention + O + residual
  _ffn_kernel:    LN + FFN + residual
  _output_kernel: final LN + tiled output projection

kernels/block_decode.py:
  _decode_layer_kernel: per-layer decode (M=1, trivial register pressure)
  _decode_output_kernel: final LN + output projection
```

### What was done (optimization)

- ~~**Persistent kernel**~~ DONE — fused 2-layer decode into single kernel (35% speedup)
- ~~**Precompute weights**~~ DONE — avoid per-step dtype conversion (57% speedup)
- ~~**In-kernel cache updates**~~ DONE — eliminate .at[].set() (3.6x speedup, biggest win)
- ~~**Multi-layer fusion**~~ DONE — h stays in registers between layers
- **Quantization**: NOT NEEDED at this scale (dispatch-bound, not bandwidth-bound)

---

## COMPLETED: Scale Up (Phase A3)

All 5 steps completed. Speedup holds at 15x with 8x larger model.

### What was done

1. **Bigger Training Data**: TinyStories dataset (55M chars, 14M BPE tokens, vocab=4096).
   HuggingFace ByteLevel BPE tokenizer trained on the corpus. Zero UNK rate.

2. **Bigger Model**: d=256, h=8, l=4, d_ff=1024, ctx=256, 5.3M params.
   BLOCK_SEQ=16 in prefill (fits (16,256)=16KB h_block in registers).
   Fused N-layer decode with packed weight buffer + packed KV caches.

3. **FlashAttention**: Tiled KV (KV_TILE=64) with online softmax for context>256.
   Keeps scores at (BLOCK_SEQ, KV_TILE) = (16, 64) = 4KB instead of (16, 512) = 32KB.

4. **Training Quality**: AdamW (wd=0.1) + LR warmup (200 steps) + cosine decay.
   val_loss=1.69, ppl=5.40 in 20 epochs. Coherent multi-paragraph stories.

5. **Benchmark**: 1396 tok/s Triton vs 92 tok/s JAX = 15.1x speedup.

### Key engineering decisions

- **Packed weight buffer**: All per-layer weights concatenated into one bf16 buffer.
  Kernel computes offsets from layer index. Avoids 50+ individual pointer arguments.
- **Packed KV caches**: All layers' K/V caches in one flat buffer. Stays packed between
  decode steps — no per-step pack/unpack overhead.
- **tl.static_range for layers**: Loop unrolled at compile time (4 iterations).
  Compiler can reuse registers between layers. Fast for ≤8 layers.
- **BLOCK_SEQ as kernel parameter**: Not module-level constant. Allows different
  tile sizes per model (32 for d=128, 16 for d=256).

---

## COMPLETED: Speculative Decoding (Phase B)

Implemented speculative decoding with a custom parallel verification kernel.

### What was done

1. **Draft model**: Trained d=128, h=4, l=1 (1.3M params, ppl=8.76) on TinyStories
   with same vocab=4096 as target. Saved to `draft_weights.pkl`.

2. **Speculative decode algorithm**: Draft generates K tokens autoregressively,
   target verifies in parallel, greedy accept/reject with first-disagreement cutoff.

3. **Parallel verification kernel** (`kernels/verify_decode.py`): Processes K draft
   tokens through the full target model in ONE kernel call. Key techniques:
   - Pads to PAD_K=16 internally for `tl.dot` shape compatibility (inner dim >= 16)
   - Online softmax over tiled KV cache (same as FlashAttention) for prefix tokens
   - Causal attention among draft tokens with runtime `real_k` mask
   - Single grid=(1,) launch — all K tokens in one thread block

4. **Correctness verified**: Parallel kernel matches sequential decode within bf16
   tolerance (max logit diff < 0.06).

### Results

```
Standard target decode:                1395 tok/s
Speculative K=2 (sequential verify):    707 tok/s  (0.51x)  acceptance=51%
Speculative K=2 (parallel verify):      715 tok/s  (0.51x)  acceptance=50%
Speculative K=4 (sequential verify):    488 tok/s  (0.35x)  acceptance=37%
Speculative K=4 (parallel verify):      632 tok/s  (0.45x)  acceptance=36%

Parallel kernel speedup over sequential: 1.30x at K=4
```

### Key findings

- **Speculative decoding has diminishing returns at small model scale.**
  With fused kernels, both draft (~2500 tok/s) and target (~1400 tok/s) are
  extremely fast. The speed ratio is only ~2x, not the ~10x needed for
  speculative decoding to be profitable.

- **Acceptance rate is the bottleneck.** With a 1-layer draft (ppl=8.76) vs
  4-layer target (ppl=5.40), greedy acceptance is only 36-51%. Speculative
  decoding needs 80%+ acceptance to break even at this speed ratio.

- **The parallel verification kernel works.** 1.3x faster than K sequential
  decode calls at K=4. The kernel correctly handles prefix cache attention,
  causal draft-draft attention, and online softmax — all in one launch.

- **When speculative decoding helps:** Large, slow target models (e.g., 70B at
  20 tok/s) with a fast draft (7B at 200 tok/s) give a 10x speed ratio.
  At that ratio, even 50% acceptance yields 2-3x speedup.

### Kernel engineering lesson

23. **Batch verification is a "mini-prefill" with existing KV cache.** The
    verification kernel processes K tokens through all layers, attending to P
    cached tokens (via tiled KV + online softmax) plus K draft tokens (via
    causal intra-draft attention). Padding K to 16 satisfies Triton's
    `tl.dot` minimum inner dimension requirement.

---

## COMPLETED: Scale Up (Phase A4)

Scaled to d=512, 8 layers, 29.7M params on full TinyStories (487M tokens, 1.9GB).

### What was done

1. **Full TinyStories**: Downloaded all 2.1M stories (1.9GB). Chunked BPE tokenization
   (50MB chunks) to avoid OOM — encoding 1.9GB at once used 46GB+ RAM. Retrained BPE
   tokenizer on full dataset.

2. **Training**: d=512, h=16, l=8, d_ff=2048, ctx=512, 29.7M params.
   batch=16, lr=1e-4, 3 epochs (~4.4 hours). val_loss=1.068, ppl=2.91.
   Data kept on CPU, batches streamed to GPU to avoid data OOM.

3. **Kernel adaptations for d=512** (register pressure is the main challenge):
   - **BLOCK_SEQ=8** for prefill (h_block = (8,512) = 16KB fits in registers)
   - **tl.dot for projections**: element-wise (512,32) intermediates overflow registers
     (128 regs/thread > 255 limit). tl.dot tiles internally, avoiding materialization.
   - **Tiled KV decode**: (512,32) full cache load overflows shared memory.
     Added online softmax with KV_TILE=64, same technique as FlashAttention.
   - **Smaller output VTILE=32**: (512,128) weight load = 128KB > 130KB register file.
     Reduced to (512,32) = 64KB.

4. **Benchmark**: 287 tok/s, 3.48 ms/tok, 2% bandwidth utilization.

### Key engineering decisions

- **tl.dot vs element-wise at d=512**: At d≤256, element-wise `tl.sum(h[:, None] * W, axis=0)`
  works because (256,32) = 64 regs/thread fits alongside h (2 regs) in 255 budget. At d=512,
  (512,32) = 128 regs alone, product = another 128 → 260 > 255. tl.dot avoids materializing
  the full intermediate by tiling the reduction internally.
- **Triton [0,:] indexing not supported**: Used `.sum(axis=0)` on (1,N) results from tl.dot.
- **Chunked tokenization**: HuggingFace tokenizer uses ~25x text size in working memory.
  1.9GB text → 46GB+ RAM → OOM killed. 50MB chunks cap peak at ~12GB.

---

## COMPLETED: Batched Inference (Step 2c) — DONE

**Results (GQA, d=512, 8L, 4 KV heads, 26.5M params, ppl=2.96):**
```
Single-sequence (B=1):  1935 tok/s  (0.517 ms/tok)
Batched B=4:            3821 tok/s  (1.97x, 1.047 ms/step, 955 tok/s per seq)
Batched B=8:            4641 tok/s  (2.40x, 1.724 ms/step, 580 tok/s per seq)
Batched B=16:           5050 tok/s  (2.61x, 3.169 ms/step, 316 tok/s per seq)
```

**Previous (MHA, 29.7M params):**
```
B=1: 1890 tok/s, B=4: 3422 (1.81x), B=8: 4347 (2.30x)
```

**Why not 4x at B=4:** Per-sequence compute (attention over 512 KV positions, QKV projections,
FFN) adds ~0.17ms per additional sequence regardless of weight sharing. The theoretical 4x assumed
weights dominated kernel time (75%), but in practice the per-sequence attention loop is a
significant fixed cost. GQA helped ~10% over MHA (L2-friendly KV caches: 2.1 vs 8.4 MB per seq).

### What was done

1. **Batched kernel** (`kernels/batched_decode.py`): same grid=(TOTAL_BLOCKS,) as single-seq.
   Each block processes B sequences. Weight loads amortized across batch.

2. **Race condition fixes** (two critical bugs found and fixed):
   - **h_buf read-write race**: all blocks read h, compute h_new, write h_new. A fast block
     can overwrite h before a slow block reads the original, causing double residual addition.
     Fix: double-buffered h_buf (read buf_a, write buf_b, alternate per layer).
   - **partial buffer merge/FFN race**: after the phase 1 barrier, all blocks read o_proj from
     partial (merge) while also writing FFN results. Fix: separate ffn_buf for FFN accumulation.

3. **Race-free design** — h_buf only written in phase 3, read in phases 1-2:
   - Phase 1: LN1 → Q/K/V → attention → O-proj (reads h from buf_in)
   - Phase 2: merge + LN2 + FFN fused per batch element (h_norm in registers, attn_total stored
     to attn_buf, FFN partial to ffn_buf; h_buf NOT modified)
   - Phase 3: h_new = h + attn_total + ffn_total + bias (writes to buf_out)
   - 3 barriers per layer: after phase 1, after phase 2, after phase 3

## COMPLETED: Close the 7.8x Gap — Optimization Round (2026-03-31)

**GPU: RTX 4080 Super (Ada Lovelace, 16GB VRAM, 101KB shared memory, 52 TFLOPS FP16, 836 GB/s HBM, ~3-6 TB/s L2)**

**Targets achieved:**
- Single-sequence: 4777 tok/s (persistent kernel) > 3000 ✓
- Batched B=4: 6691 tok/s (pipelined) > 6000 ✓

**Optimizations applied:**
1. Barrier reduction 25→17 (removed b2 in batched): minimal impact (lesson #37 confirmed)
2. Weight-amortized FFN (batched): outer k-loop / inner b-loop, B=4 +14%, B=8 +22%, B=16 +19%
3. Persistent decode kernel (single-seq): single launch for 128 tokens, 4777 tok/s (2.56x)
4. Pipelined batched decode: tokens stay on GPU, B=4 6691 tok/s (1.59x vs sync'd)

```
CURRENT (GQA, d=512, 8L, 4 KV heads, 26.5M params, ppl=2.96):
  Single-seq (sync):    1869 tok/s  (0.535 ms/tok, 12% BW util)
  Single-seq (pipe):    2624 tok/s  (0.381 ms/tok, no per-token sync)
  Persistent kernel:    4777 tok/s  (0.209 ms/tok, single launch) ✓
  Batched B=4 (sync):   4205 tok/s  (0.951 ms/step, 2.25x)
  Batched B=4 (pipe):   6691 tok/s  (0.598 ms/step, 3.58x) ✓
  Batched B=8 (pipe):   7339 tok/s  (1.090 ms/step, 3.93x)
  Batched B=16 (sync):  6064 tok/s  (2.638 ms/step, 3.24x) ✓
```

**Key insight:** GPU→CPU sync (int() per token) was the #1 bottleneck at 30M params,
costing 30-60% of wall time. Persistent kernel and pipelining eliminate it.

---

## YOUR NEXT TASK: Kernel Optimization + Model Scaling Validation

Two goals in parallel: (A) push kernel performance further, and (B) scale the model up
to verify that kernel optimizations generalize across d_model and context length.

### Part A: Kernel Optimization (on current d=512, 8L model)

Continue closing the gap to theoretical. Current persistent kernel achieves 0.209 ms/tok
vs 0.066 ms/tok theoretical = 3.2x gap. The kernel itself (not sync) runs at ~0.35 ms/tok
pipelined, so the real gap to theoretical is ~5.3x.

**Priority-ordered optimizations:**

**A1. Shared memory workspace (highest expected impact)**

The 101KB shared memory is completely UNUSED. All workspace buffers (partial, attn_buf,
ffn_buf, h_norm, etc.) go through L2 at ~100 cycle latency. Shared memory is ~20 cycles.
Moving the hottest buffers to shared memory could save 50-100µs per step.

Candidate buffers for shared memory:
- `partial`: TOTAL_BLOCKS × D_MODEL f32 = 32 × 512 × 4 = 64KB — fits! This is the
  most-accessed buffer (read by all blocks for merge and FFN reduce).
- `attn_ml`: TOTAL_BLOCKS × 2 f32 = 256 bytes — trivially fits.
- `h_norm`: B × D_MODEL f32 = 4 × 512 × 4 = 8KB — fits alongside partial.

Challenge: shared memory is per-SM, but workspace buffers need to be visible across SMs
(for cross-block barriers). Shared memory is SM-local. So this only works for buffers
that are written and read within the SAME block, not cross-block buffers.

This means only per-block-private data can use shared memory. The `partial` buffer IS
cross-block (all blocks write their part, all blocks read all parts). So it can't use
shared memory directly. But `qkv_tmp` (per-block Q/K/V) and `h_norm` (all blocks write
same value, only need own block's write) could.

**A2. Persistent batched kernel**

Apply the same persistent kernel technique to the batched decode. This eliminates per-step
workspace allocation and int() sync for batched inference, potentially pushing B=4 sync'd
from 4205 to ~6000+ without pipelining.

Design: same as persistent single-seq but with B sequences. Additional complexity:
- h_buf double-buffering across steps (not just layers)
- Larger workspace (more barrier slots for B × steps)
- In-kernel next-token feedback for all B sequences

**A3. In-place KV for persistent kernel (eliminate tile copy)**

The current persistent kernel copies ALL KV tiles during attention (matching original
read-modify-write). This is ~4 MB of unnecessary traffic per step. Writing K_new/V_new
BEFORE attention and reading directly (no tl.where) saves this traffic but introduces a
bf16 round-trip that caused token divergence at step 34. Debug and fix this — the divergence
is a precision issue, not a correctness issue. Options:
- Accept the divergence (both outputs are valid, just different greedy paths)
- Keep K_new/V_new in f32 registers and use tl.where for the CURRENT step only,
  write to kv_out after attention (already tried, still diverges — investigate further)

**A4. Tensor core batched projections (B >= 16)**

With B>=16, QKV/O projections become real matmuls: (B, D_MODEL) @ (D_MODEL, D_HEAD).
This activates tensor cores. Process all B sequences as a single tl.dot.

### Part B: Model Scaling Validation

Train larger models and verify kernel optimizations hold. The key question: do the
techniques (multi-SM, weight amortization, persistent kernel, pipelining) still work
at larger d_model and context length, or do new bottlenecks emerge?

**B1. Scale d_model to 768 (target: ~70M params)**

```
Config: d=768, h=24, l=12, d_ff=3072, ctx=512, vocab=4096
Estimated params: ~70M
KV cache (GQA 6 heads): 2 × 6 × 512 × 32 × 12 layers × 2 bytes ≈ 4.7 MB/seq
Weight buffer: ~140 MB (exceeds L2!)
```

Key challenges at d=768:
- Weight buffer overflows L2 (140 MB > 64 MB). Now truly HBM-bound for weights.
  This is where the kernel becomes bandwidth-bound and the roofline model applies.
- Register pressure: (1, 768) × (768, 32) tl.dot needs careful tiling.
- BLOCK_SEQ for prefill: need BLOCK_SEQ ≤ 8 to keep h_block in registers.

Expected: multi-SM and weight amortization should help MORE (larger weight matrices
mean more L2/HBM traffic to amortize). Barriers should matter LESS (more compute
per step dilutes fixed barrier cost).

**B2. Scale context to 2048**

```
Config: d=512, h=16, l=8, ctx=2048
KV cache (GQA): 2 × 4 × 2048 × 32 × 8 × 2 bytes ≈ 8.4 MB/seq
FlashAttention tiles: 2048/64 = 32 tiles per KV split (was 4)
```

Key challenges at ctx=2048:
- Attention cost grows linearly with context (32 tiles vs 4 at ctx=512).
  Attention becomes a larger fraction of per-seq compute.
- KV_SPLITS should probably increase (4 or 8) to parallelize over more tiles.
- KV cache per sequence grows 4x. At B=4: 33.6 MB KV → doesn't fit in L2 with weights.

Expected: weight amortization matters less (attention dominates), KV-split parallelism
matters more. GQA becomes essential.

**B3. Train and benchmark at each scale**

For each config:
1. Train with `uv run train_backprop.py --d-model D --n-heads H --n-layers L --context-len C`
2. Run `uv run profile_kernels.py` to get throughput numbers
3. Compare persistent/pipelined/sync'd throughput
4. Identify new bottlenecks if optimizations don't transfer

### How to measure progress

Run `uv run profile_kernels.py` after any kernel change. The hard metrics:

```
CURRENT (d=512 baseline to beat):
  Persistent:           4777 tok/s  (0.209 ms/tok)
  Batched B=4 (pipe):   6691 tok/s  (0.598 ms/step)
  Batched B=8 (pipe):   7339 tok/s  (1.090 ms/step)

SCALING TARGETS (new):
  d=768 persistent:     >2000 tok/s  (model is ~3x larger → expect ~3x slower)
  d=768 B=4 pipe:       >4000 tok/s
  ctx=2048 persistent:  >2000 tok/s  (4x more attention work)
  ctx=2048 B=4 pipe:    >4000 tok/s
```

### What NOT to change

- Keep the same technology stack (Triton + JAX + jax-triton)
- Keep BPE tokenization (trained on corpus)
- Keep the multi-block prefill architecture
- Keep the fused multi-layer decode approach
- Keep the in-kernel cache update optimization
- **No quantization** — Quantization (INT8, FP8, etc.) is a deployment optimization
  technique, not a kernel engineering improvement. The goal of this project is to learn
  GPU kernel programming by making the kernel itself fast, not to shrink the model.
  Quantization trades accuracy for bandwidth — it doesn't teach you to write better
  kernels. If the kernel is bandwidth-bound, the right response is to improve the
  kernel's memory access patterns, not to reduce the data size.

---

## Profiling Commands

```bash
uv run profile_kernels.py                   # primary benchmark (run after every change)
uv run profile_kernels.py --detailed        # per-component breakdown
uv run inference_benchmark.py               # quick throughput check

# Nsight Compute for detailed GPU metrics
/usr/local/cuda/bin/ncu --set full uv run profile_kernels.py

# Nsight Systems for timeline view
/usr/local/bin/nsys profile -t cuda uv run profile_kernels.py

# Training
uv run train_backprop.py --d-model 512 --n-heads 16 --n-layers 8 \
  --context-len 512 --epochs 3 --lr 1e-4 --batch-size 16
```

---

## Files

```
program.md                          — this file (read first)
repo_explained_from_zero.md         — ground-up explanation of GPU kernels + this project
README.md                           — project overview
model.py                            — JAX transformer model (inference baseline)
train_backprop.py                   — AdamW training with LR schedule
data.py                             — Shakespeare + TinyStories (char, GPT-2 BPE, trained BPE)
kernels/fused_prefill.py            — fused Triton prefill kernel (d_model≤64)
kernels/fused_decode.py             — fused Triton decode kernel (d_model≤64)
kernels/block_prefill.py            — multi-block prefill + FlashAttention (d_model≥128)
kernels/block_decode.py             — per-layer decode + orchestrator (d_model≥128)
kernels/fused_decode_nlayer.py      — fully fused N-layer decode (packed weights/caches)
kernels/multi_sm_decode.py          — multi-SM decode: grid=(N_HEADS×KV_SPLITS,) with atomic barriers
kernels/batched_decode.py           — batched multi-SM decode: B sequences per launch
kernels/persistent_decode.py        — persistent decode: single launch for all steps
profile_kernels.py                  — primary profiling tool (run after every change)
inference_benchmark.py              — quick throughput benchmark
baseline_metrics.txt                — current numbers to beat
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

8. **Tiled output projection has zero overhead.** Going from register-only (vocab=65)
   to tiled (vocab=1024, 8 tiles of 128) maintained or improved speedup (4.0x → 4.35x).
   The output projection is not the bottleneck — attention is. Tiling is free.

9. **VOCAB_SIZE/VOCAB_PAD as kernel constexpr parameters.** Using `tl.constexpr` function
   parameters (not module-level) lets jax-triton JIT-compile a kernel variant per vocab
   size. Clean separation between kernel logic and model configuration.

10. **Multi-block tiling is the key to scaling d_model.** With d_model=128, h is (128,128) =
    64KB — can't fit with attention scores (also 64KB) in 127KB register file. Tiling
    the sequence into BLOCK_SEQ=32 row blocks reduces h to (32,128) = 16KB, scores to
    (32,128) = 16KB. Peak per block: ~52KB. Scales to any d_model.

11. **Projections and attention need separate kernel launches.** Within a single Triton
    kernel, blocks execute in parallel with no ordering guarantee. Block 3's attention
    can't read K/V written by block 0 in the same launch. Split into proj_kernel (writes
    K/V to HBM) → attn_kernel (reads K/V from HBM).

12. **Decode kernel scales trivially with d_model.** M=1 means all tensors are 1D (d_model,).
    With d_model=128, h is only 512 bytes. A single fused kernel per layer works for any
    d_model. The bottleneck is attention over the KV cache: (max_seq, d_head) per head.

13. **HBM traffic between kernels costs ~2x speedup** initially. Went from 4.3x to 1.9x.
    But fusing decode back into one kernel + precomputing weights recovered most of it
    (1.9x → 3.9x). The real cost was Python dispatch, not HBM.

14. **Python dispatch dominates small-model latency.** Each jt.triton_call has ~0.4ms of
    Python/JAX overhead. With 3 calls per decode step × 63 steps = 189 calls → 75ms of
    pure overhead. Fusing into 1 call/step saved 35%.

15. **Never convert dtypes inside the decode loop.** `.astype(bf16)` creates a new JAX
    array every call. With 28 weight tensors × 63 steps = 1764 allocations → 57% slowdown.
    Precompute bf16 weights once before the loop.

16. **Fold KV cache updates into the kernel.** `.at[:, pos, :].set()` creates a new array
    copy every call. 4 caches × 0.27ms = 1.09ms per step — 73% of decode time!
    Having the kernel write full updated caches directly gave 3.6x speedup.

17. **For small models, host overhead > GPU compute.** The GPU kernel takes 0.4ms per
    decode step, but Python/JAX dispatch, dtype conversions, and array scatters added
    1.1ms. Eliminating all host-side overhead gave 7.5x improvement (335→2504 tok/s).

18. **Packed weight buffers scale to any layer count.** With 4+ layers, passing individual
    weight pointers becomes unwieldy (50+ arguments). Concatenating all per-layer weights
    into one bf16 buffer and computing offsets from layer index is cleaner and equally fast.

19. **Packed KV caches eliminate per-step allocation.** Keep all layers' K/V caches in one
    flat buffer between decode steps. The kernel reads from kv_in and writes to kv_out
    directly. No Python-side pack/unpack between steps.

20. **15x speedup holds across model scales.** d=64/1L: 16.9x, d=128/2L: 14.0x,
    d=256/4L: 15.1x. The kernel fusion advantage is consistent — it's not just a
    small-model artifact. As models grow, compute increases but so does the benefit
    of eliminating host overhead.

21. **BLOCK_SEQ must scale inversely with d_model.** Register file is fixed at ~127KB.
    With d=128: BLOCK_SEQ=32 → h=(32,128)=16KB. With d=256: BLOCK_SEQ=16 → h=(16,256)=16KB.
    The h_block size stays at 16KB by halving BLOCK_SEQ when doubling d_model.

22. **FlashAttention is only needed for context>256.** At context=256 with BLOCK_SEQ=16,
    scores are (16,256)=16KB per head — fits alongside K(32KB)+V(32KB)+accumulators.
    At context=512+, the full scores matrix overflows. Tiled KV with online softmax
    (KV_TILE=64) keeps peak at ~20KB per tile regardless of context length.

24. **At d=512, element-wise projections overflow registers.** Each (D_MODEL, D_HEAD) =
    (512, 32) intermediate uses 128 registers per thread (out of 255 max). The product
    `h[:, None] * W` needs another 128 → 260 > 255, causing spills to shared memory.
    Fix: use `tl.dot` which tiles the reduction internally. With `tl.dot((1, 512) @ (512, 32))`,
    Triton never materializes the full (512, 32) intermediate.

25. **Output projection needs smaller VTILE at d=512.** Loading (D_MODEL, VOCAB_TILE) =
    (512, 128) bf16 = 128KB ≈ entire register file. Reduced to VTILE=32 → (512, 32) = 32KB.
    4x more loop iterations but avoids register spilling.

26. **Tiled KV attention in decode is essential for MAX_SEQ=512.** Loading the full
    (MAX_SEQ, D_HEAD) = (512, 32) cache at once overflows shared memory (139KB > 101KB).
    Online softmax with KV_TILE=64 tiles the cache: each tile loads (64, 32) = 4KB.
    Same algorithm as FlashAttention but for the M=1 single-token case.

27. **At d=512, the GPU kernel is 93% of step time, not host overhead.** Profiling
    showed: kernel=3.35ms (93%), argmax+sync=0.13ms (4%), host=0.11ms (3%). The
    original hypothesis that Python dispatch was the bottleneck was wrong at this
    model scale. Host overhead matters for small models (d≤128) but is negligible
    at d=512 where the kernel dominates.

28. **Multi-SM decode with atomic barriers gives 5.2x speedup.** grid=(N_HEADS=16,)
    instead of grid=(1,) — each block handles one attention head, and all blocks
    split the FFN. Uses `tl.atomic_add` with `sem='release'/'acquire', scope='gpu'`
    for cross-block barriers. Two barriers per layer: one after attention (before
    FFN), one after FFN (before next layer). All blocks redundantly compute LN and
    reductions (32KB from L2, negligible cost). 287→1519 tok/s, 2%→12% BW utilization.

29. **Redundant computation is cheaper than synchronization.** In the multi-SM kernel,
    all 16 blocks independently compute LayerNorm and reduce the 16 partial results
    (16×512 f32 = 32KB reads). This is redundant but costs only ~1µs per block from
    L2 cache. The alternative — having one block compute and broadcast — would need
    an additional barrier (~5µs) plus the serial bottleneck.

30. **Use tl.range (not static_range) for the layer loop in large kernels.** With 8
    layers and the full multi-SM kernel body (attention + FFN + barriers + reductions),
    `tl.static_range` unrolls the loop at compile time, creating enormous IR that takes
    10+ minutes to compile. `tl.range` compiles the loop body once and executes it
    dynamically — identical runtime performance, 10x faster compilation.

31. **KV-split parallelism (FlashDecoding) gives 10% improvement at grid=32.** With
    N_HEADS=16 blocks, only 16/80 SMs were utilized. Splitting KV cache attention
    across 2 blocks per head (KV_SPLITS=2) doubles the grid to 32 blocks. Each block
    handles half the KV tiles and computes a partial online softmax. After the barrier,
    all blocks merge the partials using the log-sum-exp trick: `h_head = Σ(o_proj_s *
    l_s * exp(m_s - m_max)) / Σ(l_s * exp(m_s - m_max))`. The normalized O-projection
    avoids large-value bf16 overflow. Edge case: blocks whose KV range has no valid
    positions (pos < range_start) produce l=0, m=-inf — clamp l before normalization
    to avoid NaN, then the merge naturally weights them to zero.

32. **grid=32 is the sweet spot; grid=64 has diminishing returns.** KV_SPLITS=4
    (grid=64) was slightly slower than KV_SPLITS=2 (grid=32) due to increased barrier
    contention. With 64 blocks, each barrier has more atomic operations and longer
    worst-case spin time. Also, each block has less FFN work (D_FF/64=32 elements,
    only 1 iteration), reducing per-block compute efficiency. The FFN reduction also
    reads from more blocks (64×D_MODEL vs 32×D_MODEL from L2).

33. **Split barrier: separate counter and done-flag on different cache lines.**
    The standard all-spin barrier has all N blocks polling the same counter address
    with atomic_add(0, acquire) while other blocks are still arriving with atomic_add(1,
    release). This causes L2 cache line thrashing (arrivals invalidate the line that
    pollers are reading). Fix: arrivals go to counter[], last-arriving block sets a
    done-flag[] on a different cache line, all blocks poll done[]. The counter line
    sees only 32 writes (no reads during arrival), the done line sees only reads until
    the single write. 6.5% kernel speedup (2918→3108 tok/s pure GPU).

34. **Triton if/else on atomic return values is unreliable.** When branching on the
    return value of `tl.atomic_add`, using `if/else` caused subtle correctness bugs:
    tokens diverged after 10 steps. Replacing `if (...): ... else: ...` with
    `if (...): ...\n while ...:` (no else, all blocks execute the while) fixed it.
    The issue is likely that Triton's compiler generates incorrect predicated code
    when the else branch contains a while loop with atomics.

35. **GPU→CPU sync costs 34% of total decode time.** `int(next_token)` forces a
    GPU→CPU transfer per token (0.19 ms). The in-kernel argmax avoids the argmax
    compute cost, but int() still triggers sync. The pipelined approach (tokens stay
    on device, JAX dispatches next call while current runs) achieves 3108 tok/s vs
    1937 tok/s with sync. For real applications, collect tokens on device and batch-
    transfer to CPU periodically.

36. **GQA doesn't help single-sequence decode when barrier-limited.** With 17
    barriers taking ~50% of kernel time, reducing data from 67.7 MB to 55.1 MB
    (via GQA's 4x smaller KV cache + fewer K/V weights) gives 0% speedup. The
    memory access is already served from L2 at high bandwidth; the bottleneck is
    the barrier spin-wait. GQA's value is for batched inference (batch=8 KV cache:
    16.8 MB GQA vs 67.2 MB MHA) and long-context scenarios. Lesson: always profile
    to identify the actual bottleneck before optimizing.

37. **Barrier count reduction gives minimal speedup because straggler variance
    dominates.** Parallel residual reduces barriers from 17 to 9 (-47%) but only
    gives +1.3% speedup. Each barrier has ~1µs fixed overhead + variable straggler
    wait. The straggler wait is proportional to the TOTAL WORK per step, not the
    number of barriers. With fewer barriers, each remaining barrier has MORE work
    between them → MORE straggler variance → HIGHER per-barrier cost. Net: saving
    8 × 1µs = 8µs of fixed overhead, but the ~80µs of straggler time stays constant.
    To reduce barrier overhead, must either: (a) balance work across blocks better,
    (b) reduce total work, or (c) use hardware cooperative group barriers.

38. **Batched decode with shared buffers requires double-buffering for h.** When all
    blocks read h, compute h_new = h + residual, and write h_new to the SAME buffer,
    a fast block can write h_new before a slow block reads the original h. The slow
    block then computes h_new + residual = h + 2×residual (double residual). Fix:
    double-buffer h — even layers read buf_a/write buf_b, odd layers read buf_b/write
    buf_a. The buffers never overlap because all reads are from one, all writes to
    the other. Cost: 2× h_buf memory (negligible: 2 × B × D_MODEL f32).

39. **Shared workspace buffers must separate read-phase from write-phase data.** In
    the multi-SM batched kernel, the partial buffer stored both attention o_proj
    (written phase 1, read in phase 2 merge) and FFN partial (written phase 2,
    read in phase 3). After the phase 1 barrier, a fast block could finish the merge
    and start writing FFN partials while a slow block was still reading o_proj values.
    Fix: use separate partial and ffn_buf buffers. General rule: if a buffer is read
    by all blocks and also written by all blocks within the same barrier-delimited
    phase, it needs double-buffering or a separate buffer.

40. **Batched decode throughput is L2-limited with MHA.** With MHA (8.4 MB KV per seq),
    B=4 total data = 88.6 MB exceeds 64 MB L2 cache, forcing HBM traffic. Result:
    1.81x throughput (not 4x). GQA (2.1 MB KV per seq) keeps B=4 data at 63.4 MB,
    fitting in L2. Expected: 3-4x throughput with GQA.

41. **Weight-amortized FFN: outer k-loop / inner b-loop saves (B-1)× weight loads.**
    The fused merge+LN2+FFN loaded FFN weights B times per k-tile (once per batch
    element). De-fusing: compute merge+LN2 for all B (store h_norm to buffer), then
    FFN with outer k-loop / inner b-loop loads weights once per tile. Result: B=4
    +14%, B=8 +22%, B=16 +19%. The extra h_norm buffer traffic is negligible vs the
    weight savings. The ffn_buf accumulation uses conditional store (k==0: store,
    else: load+add+store) to avoid a separate initialization pass.

42. **Persistent kernel eliminates ALL per-step host overhead.** The single-launch
    persistent decode kernel runs all N decode steps in one kernel call, using fresh
    barrier slots per step (step * BARRIERS_PER_STEP + barrier_idx). Initial KV copy
    (kv_ptr → kv_out_ptr) is done cooperatively by all blocks before the step loop,
    protected by a barrier. Block 0 writes next_token to workspace; a step-sync
    barrier ensures all blocks see it before the next step's embedding. Result:
    4777 tok/s vs 1869 tok/s sync'd (2.56x). The KV tile copy during attention
    (matching original read-modify-write pattern) is essential for correctness —
    writing K_new BEFORE attention and reading it back introduces a bf16 round-trip
    that causes divergence after ~34 tokens.

43. **GPU→CPU sync (int() per token) is the dominant bottleneck at 30M params.**
    With the kernel itself running at 0.32-0.35 ms/tok and int() adding 0.15-0.20 ms,
    sync accounts for 30-60% of wall time. Pipelining (JAX overlaps dispatch with
    execution) and persistent kernels (tokens stay on GPU) both eliminate this.
    For batched B=4: sync'd 4205 tok/s → pipelined 6691 tok/s (+59%). The lesson:
    benchmark both sync'd and pipelined, and report the number that matches your
    deployment scenario (streaming vs batch generation).
