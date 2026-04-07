# Single-GPU Transformer — Agent Program

*You are building and optimizing a complete transformer — training, custom GPU kernels,
and serving — on a single RTX 4080 Super (16GB VRAM, 836 GB/s, 52 TFLOPS FP16).
The GPU is dedicated to this project (not shared with other workloads).
The primary goal is maximizing training efficiency: highest quality model in the
least wall-clock time. Every wasted GPU cycle is lost training. Squeeze every FLOP
out of this hardware — maximize batch size, minimize kernel overhead, eliminate
memory waste, keep the GPU at 100% utilization.*

**IMPORTANT: Commit and push after every meaningful step. Don't batch up changes.**

---

## Mission

Train a high-quality decoder-only transformer on one GPU, serve it with custom Triton kernels.

---

## Current Model

```
306M params, d=1024, h=16, kv=4, l=24, ctx=512
GQA (4 KV heads), RMSNorm, RoPE, SwiGLU (d_ff=2816), no biases, tied embeddings
Vocab: 32K BPE trained on corpus

Training data: 7.85B tokens (5 sources):
  34% FineWeb-Edu (quality-filtered web, score >= 3)
  30% StarCoder code (13 languages)
  19% OpenWebMath (math with LaTeX)
   9% Wikipedia
   8% Cosmopedia (synthetic textbooks)

Stored as data/tokens_v2/train.bin (flat int32, memory-mapped) + val.npy
```

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
d_model=1024, n_heads=16, n_kv_heads=4 (GQA), n_layers=24, d_ff=2816
context_len=512, vocab=32000

Components:
  - RMSNorm (pre-attention and pre-FFN, no bias)
  - RoPE (base=10000, applied to Q and K after projection)
  - SwiGLU FFN: (SiLU(x @ W_gate) * (x @ W_up)) @ W_down, d_ff=2816
  - GQA: 16 query heads, 4 KV heads (group size 4)
  - No biases anywhere (attention projections, FFN projections)
  - Tied embeddings: output logits = h @ token_emb.T
  - Fused cross-entropy: chunked forward+backward, never materializes full logits
```

---

## Inference Performance (RTX 4080 Super)

```
d=1024, h=16, l=24 (current model):
  Multi-SM sync:       501 tok/s  (2.0 ms/tok, 30% BW util)
  Pipelined:           618 tok/s  (1.6 ms/tok, 1.23x)
  Persist B=4:         992 tok/s  (4.0 ms/step, 1.98x)
  Persist B=8:        1097 tok/s  (7.3 ms/step, 2.19x)
  Prefill (128 tok):   23.2 ms   (5516 tok/s, Triton)
  Weight buffer:       485 MB (7.6x L2 — HBM-bound)
```

The entire decode step — embedding, attention, FFN, output projection — runs in
a single GPU kernel call across all layers.

---

## Training Performance

```
RTX 4080 Super, bs=16, ctx=512, no gradient checkpointing:
  28.4K tok/s with cuDNN FlashAttention + fused CE + 32K vocab
  ~11.2GB VRAM

With --no-checkpoint on RTX 4090 (24GB), bs=32: ~31K tok/s
  3 epochs × 7.85B tokens = 23.5B tokens total
  ~83h per epoch on 4080 Super, ~6h per epoch on B200 (bs=256, ~341K tok/s)

Curriculum training (--curriculum flag):
  Phase 1 (10%): ctx=128, bs=bs×4
  Phase 2 (20%): ctx=256, bs=bs×2
  Phase 3 (70%): ctx=512, bs=bs
```

---

## Completed Phases Summary

### Phase A: Kernel Development (d=64 through d=512)

Developed fused Triton kernels from scratch, scaling from d=64/1L to d=512/8L.
Each scale introduced new register pressure challenges requiring new techniques.

Milestones:
- d=64/1L: Entire model in registers, fused prefill+decode. 740 tok/s.
- d=128/2L: Multi-block tiling (BLOCK_SEQ=32). Fused N-layer decode. In-kernel
  cache updates were the biggest single win (3.6x). 2504 tok/s.
- d=256/4L: FlashAttention for ctx>256. Packed weight/KV buffers. 1396 tok/s.
- d=512/8L: tl.dot for projections (element-wise overflows registers). 287 tok/s
  single-SM → 1734 tok/s multi-SM (6x via atomic barriers) → 1937 tok/s with
  KV-split + split barrier.

Batched and persistent kernels:
- Batched decode: shared weights across B sequences, double-buffered h, separate
  workspace buffers. Weight-amortized FFN (+14-22%).
- Persistent decode: single kernel launch for all N steps, eliminates host sync.
  5129 tok/s at d=512 (2.56x over sync'd).
- Persistent batched: B=4 7351 tok/s, B=8 7862 tok/s.

### Phase A (continued): d=768 and d=1024

Scaled to d=768 (12L, 81M params) and d=1024 (16L, 242M params):
- Non-power-of-2 D_MODEL support via D_BLOCK padding with d_mask
- Projection tiling (PROJ_TILE=512) for D_HEAD=64 shared memory constraints
- Tensor core batched projections (2D tl.dot, +8-12%)
- L2 cache eviction hints (+3%), merged barriers (+2%)

Key result at d=768: pipelined 1472 tok/s, persist-B=8 2687 tok/s.
Key result at d=1024: pipelined 618 tok/s, persist-B=8 1097 tok/s.
At d>=768, HBM bandwidth is the terminal bottleneck (weights exceed L2 cache).

### Phase B: Speculative Decoding

Built parallel verification kernel that processes K draft tokens through the
full target in one kernel call (mini-prefill with existing KV cache).
- Acceptance rates scale with draft quality: 71% at K=4 with d=512 draft
  (ppl=2.91) vs d=768 target (ppl=2.60). Was only 36% with 1-layer draft.
- Not profitable at small scale: draft/target speed ratio is ~2x, need >5x.
  Sequential verify is mathematically always slower (1.55ms > 1.0ms per cycle).
- Valuable at production scale (70B target + 7B draft → 10x speed ratio).

### Phase C: Architecture Modernization

Replaced 2020-era architecture with modern components. Updated model.py and
all 8 Triton kernel files. JAX vs Triton prefill verified within bf16 precision.

- C1: RMSNorm (replaced LayerNorm, ~10-15% faster, no bias)
- C2: RoPE (base=10000, replaced learned pos_emb, better extrapolation)
- C3: SwiGLU FFN (3 matrices, d_ff=2816, ~15% better quality/FLOP)
- C4: Removed all biases, tied output projection to token embedding
- C5: Scaled to 24 layers (from 16), 294M params, 10.6GB at bs=16
  - tl.range for layer loops (tl.static_range takes 10+ min to compile at 24L)
- C6: DeltaNet — tried hybrid 75% DeltaNet + 25% attention (18D+6A pattern).
  Reverted: 35% slower training at ctx=512 (8.6K vs 13.2K tok/s). O(n^2)
  attention is trivially cheap at short context. DeltaNet only helps inference
  at long contexts. Code preserved in git history.
- C7: MTP heads added (optional --mtp-heads flag). 3 extra prediction heads for
  tokens t+2, t+3, t+4. +56% training time but better quality. Skip for speed.
- C8: cuDNN FlashAttention via jax.nn.dot_product_attention (+38% training speed),
  batch prefetch, fused cross-entropy (enables 32K vocab without OOM)

### Phase D: Training Efficiency

Optimized training throughput and data pipeline for the final training run.

- D1: Reverted DeltaNet, pure attention 24L, 274M params
- D7e: cuDNN FlashAttention — 13.3K → 18.4K tok/s (+38%). Uses
  jax.nn.dot_product_attention(implementation='cudnn'). bf16 for cuDNN,
  falls back to XLA for f32 inference.
- D7a: Fused cross-entropy — custom_vjp with chunked forward+backward. Tiles over
  vocab in 4096-token chunks, never materializes full logits. Enables vocab=32K
  (would need 1.1GB for full logits tensor). At vocab=4K, same speed as standard.
- D4: Vocab 4K → 32K — 28.4K tok/s, 12.3GB VRAM, 303M params
- D2: Sequence packing — skipped (data already concatenated without padding)
- D3: Curriculum training — 3 phases: ctx=128 (10%), ctx=256 (20%), ctx=512 (70%).
  Separate JIT compilation per phase.

---

### Previous Training Runs

```
Epoch 1 (d=1024, h=16, l=16, 242M, 1.77B tokens):
  val_loss=3.04, ppl=20.91, 13.7h, 36K tok/s
  Data: FineWeb-Edu 60%, Cosmopedia 18%, Wikipedia 17%, Code 6%

Epoch 2 (same model, 1.72B fresh tokens):
  train_loss=2.705, val_loss=3.338, ppl=28.16, 13.4h
  Data: FineWeb-Edu 51%, StarCoder 18%, Wikipedia 17%, Cosmopedia 14%
  Note: val ppl regressed because val set was from epoch 1 distribution
```

---

## Production Features

- **Streaming generation**: `generate.py` with streaming decode (966 tok/s at d=768)
- **Batched inference**: `serve.py` with per-sequence position tracking
- **Paged KV cache**: PagePool with 64-position pages, JIT gather/scatter
  (87% memory reduction for short seqs; GPU gather 0.07ms, scatter 0.37ms per step)
- **Continuous batching**: auto-refill batch slots as sequences complete

---

## Kernel Architecture

### Prefill (`kernels/block_prefill.py`)

Multi-block prefill with FlashAttention + GQA + D_BLOCK padding:
- 4 kernels per layer: proj, attn, flash_attn, ffn
- GQA: Q loops over N_HEADS, K/V over N_KV_HEADS
- D_BLOCK=1024 (padded from d=1024) with d_mask on all loads/stores
- 2.4x faster than JAX at d=768

### Decode (`kernels/multi_sm_decode.py` + variants)

Multi-SM fused N-layer decode:
- grid=(N_HEADS × KV_SPLITS,) with atomic barriers
- All layers processed in one kernel launch
- Per layer: LN1 → QKV proj → RoPE → attention → O proj → barrier →
  merge + LN2 → SwiGLU FFN → barrier → residual
- KV-split parallelism (FlashDecoding) for head-level parallelism
- Weight-amortized FFN: outer k-loop / inner b-loop
- Projection tiling (PROJ_TILE=512) for D_HEAD=64
- Online softmax with KV_TILE=64 for attention over cache
- All weights packed into one bf16 buffer, offsets computed from layer index
- KV caches packed flat, all layers in one buffer

Variants:
- `batched_decode.py`: B sequences with shared weights, tensor core projections,
  double-buffered h, separate ffn_buf, 3 barriers per layer
- `persistent_decode.py`: single launch for all N steps, fresh barrier slots
  per step, in-kernel argmax, block 0 writes next_token
- `persistent_batched_decode.py`: single launch for B × N steps, combines
  batched + persistent techniques

### Supporting kernels

- `fused_decode_nlayer.py`: weight/KV packing utilities for all decode kernels.
  `prepare_decode_weights_nlayer()` packs all per-layer weights (Q/K/V/O, gate/up/down,
  RMSNorm scales, RoPE cos/sin tables) into one flat bf16 buffer.
- `paged_kv.py`: paged KV cache with GPU gather/scatter
- `block_prefill.py`: multi-block prefill + FlashAttention + GQA
- `fused_prefill.py` / `fused_decode.py`: legacy small-model kernels (d<=64)

---

## Optimization History (Summary)

| Phase | Model | Key Metric | Result |
|-------|-------|-----------|--------|
| A1-A3 | d=64→256 | Triton vs JAX | 15x speedup (consistent across scales) |
| A4 | d=512, 8L, 29.7M | Single-seq decode | 287 tok/s (2% BW util) |
| A5 | d=512 | Multi-SM (grid=16) | 1734 tok/s (6x over single-SM) |
| A6-A7 | d=512 | KV-split + split barrier | 1937 tok/s (+15%) |
| Batched | d=512 | Persist B=8 | 7862 tok/s |
| Scale | d=768, 12L, 81M | Pipelined | 1472 tok/s (20% BW) |
| Scale | d=768 | Persist B=8 | 2687 tok/s |
| Scale | d=1024, 16L, 242M | Pipelined | 618 tok/s (30% BW) |
| Scale | d=1024 | Persist B=8 | 1097 tok/s |
| Training | d=1024, 24L, 306M | Training throughput | 28.4K tok/s (cuDNN FA) |

---

## What NOT to change

- Keep the same technology stack (Triton + JAX + jax-triton)
- Keep BPE tokenization (trained on corpus)
- Keep the multi-block prefill architecture
- Keep the fused multi-layer decode approach
- Keep the in-kernel cache update optimization
- **No quantization** — The goal is to learn GPU kernel programming by making the
  kernel itself fast, not to shrink the model. If bandwidth-bound, improve memory
  access patterns, not data size.

---

## Tried but didn't help (decode)

- **GQA for single-seq**: 0% speedup (barrier-limited, not memory-limited). Value is for batched inference.
- **Parallel residual** (attn||FFN, 9 barriers vs 17): +1.3%. Straggler variance dominates, not barrier count.
- **num_warps sweep**: 2/4/8 all within 5%.
- **num_stages=2**: shared memory overflow at D_BLOCK=1024.
- **In-kernel page table lookups**: 2.5x slower Triton code (indirect addressing defeats compiler).

---

## Files

```
Training:
  model.py                             JAX transformer (RMSNorm, RoPE, SwiGLU, GQA, fused CE, MTP)
  train.py                             AdamW training (bf16 fwd, cuDNN FlashAttn, curriculum, checkpointing)
  data.py                              streaming data loading (v2/v3 memmap, legacy datasets)
  prepare_data_v2.py                   v2 data pipeline: 5-source download, tokenize, shuffle, combine
  prepare_data_v3.py                   v3 data pipeline: 6 sources (28B) + annealing (3B)

Inference kernels:
  kernels/multi_sm_decode.py           multi-SM decode with atomic barriers + KV-split
  kernels/batched_decode.py            batched multi-SM decode (B sequences, tensor core projections)
  kernels/persistent_decode.py         persistent decode (single launch, all steps)
  kernels/persistent_batched_decode.py persistent batched (B x N steps)
  kernels/block_prefill.py             multi-block prefill + FlashAttention + GQA
  kernels/fused_decode_nlayer.py       weight/KV packing for all decode kernels
  kernels/paged_kv.py                  paged KV cache (GPU gather/scatter)
  kernels/fused_prefill.py             legacy fused prefill (d<=64)
  kernels/fused_decode.py              legacy fused decode (d<=64)

Serving:
  generate.py                          streaming text generation CLI
  serve.py                             batched server + continuous batching

Benchmarking:
  profile_kernels.py                   primary profiling tool
  profile_vram.py                      VRAM profiling for model scaling

Documentation:
  program.md                           this file (read first)
  data_research.md                     training data research findings and recommendations
  repo_explained_from_zero.md          ground-up GPU kernel explanation
  README.md                            project overview
  H100_TRAINING.md                     cloud GPU training setup guide
```

---

## GPU Specs (RTX 4080 Super)

```
Ada Lovelace architecture
16 GB VRAM (GDDR6X)
836 GB/s memory bandwidth
52 TFLOPS FP16 (tensor cores)
80 SMs
64 MB L2 cache (~3-6 TB/s L2 bandwidth)
101 KB shared memory per SM
255 registers per thread
```

## Profiling Commands

```bash
uv run profile_kernels.py                   # primary benchmark (run after every change)
uv run profile_kernels.py --detailed        # per-component breakdown

# Nsight Compute for detailed GPU metrics
/usr/local/cuda/bin/ncu --set full uv run profile_kernels.py

# Nsight Systems for timeline view
/usr/local/bin/nsys profile -t cuda uv run profile_kernels.py
```

Run `uv run profile_kernels.py` after any kernel change. Switch models by copying
weights: `cp weights_dXXX.pkl weights.pkl`.

---

## Kernel Engineering Lessons

### Register pressure and tiling

1. **num_warps=4 is optimal.** Fewer warps = more registers per thread. 2 is slower
   (poor occupancy), 8 is slower (thread overhead). 128 threads/block × 255 regs
   × 4 bytes = 131KB register file budget.
2. **Dynamic loops (`tl.range`) prevent register blowup from unrolling.** FFN K-tiling
   and layer loops MUST use `tl.range`, not `tl.static_range`. Static unrolling of
   8+ iterations causes 340+ bytes of register spilling. tl.range compiles the loop
   body once — identical runtime performance, 10x faster compilation at 24 layers.
3. **bf16 matmuls with f32 accumulation are the sweet spot.** FP8 was slower (cast
   overhead). fp16 out_dtype didn't help.
4. **BLOCK_SEQ must scale inversely with d_model** to keep h_block at ~16KB. d=128:
   BLOCK_SEQ=32, d=256: BLOCK_SEQ=16, d=512: BLOCK_SEQ=8.
5. **At d>=512, use `tl.dot` instead of element-wise projections.** Element-wise
   `h[:, None] * W` for (512,32) needs 128+128=256 regs > 255 limit. `tl.dot((1,512)
   @ (512,32))` tiles the reduction internally, avoiding full materialization.
6. **At D_HEAD=64, projection weights (1024,64) overflow shared memory (128KB > 101KB).**
   Fix: tile with PROJ_TILE=512, each tile loads (512,64)=64KB. Adds ~10% overhead
   from extra h_norm loads.
7. **tl.arange requires power-of-2 ranges.** Pad D_MODEL to D_BLOCK=next_pow2 with
   d_mask on every load/store. RMSNorm needs explicit masking: `hc = tl.where(d_mask,
   h - mean, 0.0)` to prevent padded elements from corrupting variance. tl.dot with
   zero-padded operands produces correct results (0 × anything = 0).
8. **Tiled output projection has zero overhead.** Going from register-only (vocab=65)
   to tiled (vocab=1024+, VOCAB_TILE=128 chunks) maintained speedup. The output
   projection is not the bottleneck — attention is.

### Multi-SM and synchronization

9. **Multi-SM decode with atomic barriers: 6x speedup.** grid=(N_HEADS,), each block
   handles one attention head, all blocks split FFN. Uses `tl.atomic_add` with
   `sem='release'/'acquire', scope='gpu'` for cross-block barriers. Two barriers
   per layer: after attention, after FFN.
10. **Redundant computation is cheaper than synchronization.** All blocks independently
    compute LayerNorm (~1us from L2 cache) vs one-block-broadcast + barrier (~5us).
11. **KV-split (FlashDecoding): +10% at grid=32.** Two blocks per head split KV tiles,
    merge with log-sum-exp correction: `h = sum(o_s * l_s * exp(m_s - m_max)) /
    sum(l_s * exp(m_s - m_max))`. Edge case: blocks with no valid positions produce
    l=0, m=-inf — clamp l to avoid NaN. grid=64 has diminishing returns (barrier
    contention, less FFN work per block).
12. **Split barrier: separate counter and done-flag cache lines.** Arrivals write to
    counter[], last-arriving block sets done[], all blocks poll done[]. Eliminates
    L2 thrashing during spin-wait. +5-6.5%.
13. **Barrier count reduction gives minimal speedup** because straggler variance
    dominates fixed barrier overhead. Halving barriers saves ~8us fixed but ~80us
    straggler time stays constant. To reduce barrier overhead: balance work across
    blocks or reduce total work.
14. **GPU→CPU sync (`int()`) costs 30-60% of decode time at small models.** Persistent
    and pipelined kernels eliminate this. At d=1024 (~2ms/step), sync overhead is only
    ~10% so persistent and pipelined are nearly identical.

### Batched decode

15. **Double-buffer h to prevent read-write races.** Fast blocks overwriting h before
    slow blocks read it causes double-residual. Even/odd layers alternate buf_a/buf_b.
16. **Separate workspace buffers for read-phase vs write-phase data.** If a buffer is
    read and written within the same barrier phase, use separate buffers. The partial
    buffer had this bug: merge reads o_proj while FFN writes partials in same phase.
17. **Weight-amortized FFN (outer k-loop / inner b-loop): +14-22%.** Load FFN weights
    once per tile, process all B elements. Saves (B-1)x weight loads. ffn_buf
    accumulation uses conditional store (k==0: store, else: load+add+store).
18. **Batched FFN at D_BLOCK=1024: reduce BLOCK_K from 32 to 16** to fit both up_w and
    down_w in shared memory (64KB each at BLOCK_K=32 > 101KB limit).
19. **Batched projections via 2D tl.dot: +8-12%.** (B, D_BLOCK) @ (D_BLOCK, D_HEAD)
    eliminates B-1 loop iterations. At B>=16, tensor cores activate. O projection
    can't be batched: output (B, D_BLOCK) overflows shared memory.

### Persistent kernels

20. **Persistent kernel eliminates ALL per-step host overhead.** Single launch for N
    steps, fresh barrier slots per step (step * BARRIERS_PER_STEP + idx). Block 0
    writes next_token; step-sync barrier ensures all blocks see it. 2.56x at d=512,
    negligible gain at d=1024 (host dispatch overlaps with long kernel execution).
21. **In-place KV: pos-only store (128 bytes) forces correct compiler ordering.**
    Without a store in the tile loop, Triton reorders K_new store relative to tile
    loads, reading stale data. Causes divergence at step 34. The pos-only store
    forces the dependency chain without significant traffic (+7.1%).

### Scaling observations

22. **HBM bandwidth is the terminal bottleneck at d>=768.** 162MB+ weights exceed
    64MB L2; every step re-fetches from HBM. d=512 (55MB) fits in L2 for much
    better scaling (55% BW util at B=4 vs ~30% at d=768).
23. **ctx=2048 scales gracefully: 4x context costs ~1.6x decode time.** Weight loads
    dominate; attention tiles add incrementally (~0.1ms/tok vs ~0.2ms/tok for weights).
24. **Batched decode scaling is sublinear at d=1024.** B=4: ~2x, B=8: 2.2x, B=16: 2.1x
    (regression). KV overhead (8.4MB × B) grows linearly while 485MB weights dominate.
25. **L2 cache eviction hints give ~3%.** evict_last on KV (reuse across steps),
    evict_first on output projection (single-use).

### Paging and speculative decode

26. **Paged KV saves 87% memory for short sequences** (PAGE_SIZE=64). In-kernel page
    table lookups are 2.5x slower — use JIT gather/scatter instead. Gather indices
    cached per page table config, only rebuild every 64 steps.
27. **JAX `.at[].set()` copies the entire array.** 9KB update in 5MB pool costs 0.37ms
    due to functional immutability. Mutable CUDA buffers would fix this.
28. **Speculative acceptance scales with draft quality, not size.** ppl gap is the
    predictor. Need >5x speed ratio for profitability.

### Miscellaneous

29. **Triton if/else on atomic return values is unreliable.** Replace `if/else` with
    `if + while` pattern to avoid subtle correctness bugs (tokens diverge after ~10 steps).
30. **Never convert dtypes inside the decode loop.** `.astype(bf16)` creates 1764
    unnecessary allocations per decode sequence. Precompute bf16 weights once (57% win).
31. **Python dispatch dominates small-model latency.** Each jt.triton_call has ~0.4ms
    overhead. Fuse everything into one kernel. 35% speedup from eliminating 128
    launches per decode sequence.
32. **Fold KV cache updates into the kernel.** `.at[:, pos, :].set()` creates array
    copies — 73% of decode time at d=128. In-kernel cache writes gave 3.6x speedup.
33. **num_stages=2 infeasible at D_BLOCK=1024.** Double-buffering tiles needs 128KB >
    101KB shared memory.
34. **tl.dot Q/K/V projections must be in separate loops** (not one unrolled loop)
    to avoid 3×64KB=192KB simultaneous shared memory.

---

## Completed: Fix Triton Decode Kernels for Inference

Fixed. `generate.py` now produces coherent text matching JAX reference quality.

Bugs found and fixed in `kernels/multi_sm_decode.py`:
1. **Workspace barrier slots too small**: hardcoded 32, needed 73+ for 24 layers.
   Barriers overflowed into adjacent memory (done flags, argmax region).
2. **Shared buffer race condition**: attention O-projection and FFN partials shared
   the same workspace region. Fast blocks overwrote attention data before slow blocks
   finished reading. Fix: separate `attn_partial` and `ffn_partial` buffers.
3. **Store visibility across SMs**: Triton compiler reorders `tl.store` after atomic
   barriers. Other blocks pass the barrier but read stale data. Fix: `membar.gl`
   (global memory fence via inline PTX) before each barrier.
4. **KV-split non-determinism**: `kv_splits=2` (32 blocks) caused too much barrier
   interaction noise. Fix: use `kv_splits=1` (16 blocks) for reliable synchronization.
5. **Redundant KV cache writes**: multiple blocks wrote identical data to the same
   KV cache positions. Fix: only one block per kv_head writes.

Also fixed: `prepare_data_v2.py` saves relative tokenizer path; `data.py` resolves
relative paths for portability.

Best sampling settings:
- Creative: `--temp 0.7 --top-p 0.95 --rep-penalty 1.2`
- Factual:  `--temp 0.5 --top-p 0.9 --rep-penalty 1.1`
- Code:     `--temp 0.3 --top-p 0.9 --rep-penalty 1.1`

---

## Completed: Research and Prepare Data for Continued Training

Research complete. See `data_research.md` for full findings and rationale.

**Key findings:**
- Tokens/param ratio (25.6x unique) is extremely low vs peers (SmolLM2-360M: 11,000x)
- 3 epochs on current data is fine (up to 4 safe), but more unique data is the lever
- Data mix was too heavy on code (29%) and math (19%) vs general-purpose models (70%+ web)
- Annealing on high-quality data is standard practice (Llama 3, SmolLM2, MiniCPM)
- Instruction tuning degrades quality at 306M scale — skip it

**Implemented:** `prepare_data_v3.py` creates expanded v3 dataset + annealing dataset.
Run `uv run prepare_data_v3.py` to download and prepare.

---

## Current Task: Run v3 Data Preparation and Train

### Step 1: Prepare v3 data

```bash
# Download and tokenize main dataset (~28B tokens, several hours)
uv run prepare_data_v3.py

# Then prepare annealing dataset (~3B tokens)
uv run prepare_data_v3.py --anneal-only
```

### Step 2: Train on v3 data (Phase 1 — extended pretraining)

```bash
uv run python -u train.py \
  --d-model 1024 --n-heads 16 --n-kv-heads 4 --n-layers 24 \
  --context-len 512 --batch-size 16 --epochs 2 \
  --data-dir data/tokens_v3 \
  --curriculum --lr 3e-4 --no-checkpoint \
  --resume checkpoint.pkl \
  2>&1 | tee training_v3.log
```

### Step 3: Anneal (Phase 2 — high-quality cooldown)

```bash
uv run python -u train.py \
  --d-model 1024 --n-heads 16 --n-kv-heads 4 --n-layers 24 \
  --context-len 512 --batch-size 16 --epochs 1 \
  --data-dir data/tokens_v3_anneal \
  --lr 3e-5 --no-checkpoint \
  --resume checkpoint.pkl \
  2>&1 | tee training_anneal.log
```

### Step 4: Context extension (Phase 3 — optional)

```bash
uv run python -u train.py \
  --d-model 1024 --n-heads 16 --n-kv-heads 4 --n-layers 24 \
  --context-len 1024 --batch-size 8 --epochs 1 \
  --data-dir data/tokens_v3 \
  --lr 1e-4 --no-checkpoint \
  --resume checkpoint.pkl \
  2>&1 | tee training_ctx1024.log
```

### v3 Data Mix (~28B unique tokens)

```
Source                  Tokens    Pct    HuggingFace Path
FineWeb-Edu (>= 3)     10.0B    36%    HuggingFaceFW/fineweb-edu sample-10BT
DCLM-Edu                5.0B    18%    HuggingFaceTB/dclm-edu
StarCoder               5.5B    20%    bigcode/starcoderdata
OpenWebMath             3.0B    11%    open-web-math/open-web-math
Wikipedia               2.0B     7%    wikimedia/wikipedia
Cosmopedia v2           2.5B     9%    HuggingFaceTB/smollm-corpus cosmopedia-v2
─────────────────────────────────────
Total                  28.0B   100%

Web: 70% | Code: 20% | Math: 11%
```

### Annealing Data Mix (~3B tokens)

```
FineWeb-Edu score >= 4  1.5B    50%    HuggingFaceFW/fineweb-edu (score filter)
FineMath 4+             0.8B    27%    HuggingFaceTB/finemath finemath-4plus
Stack-Edu               0.7B    23%    HuggingFaceTB/stack-edu
```

---

## Training the model

### Full training run

```bash
# On RTX 4080 Super (16GB), bs=16:
uv run python -u train.py \
  --d-model 1024 --n-heads 16 --n-kv-heads 4 --n-layers 24 \
  --context-len 512 --batch-size 16 --epochs 3 \
  --curriculum --lr 3e-4 --no-checkpoint \
  2>&1 | tee training.log

# On RTX 4090 (24GB), bs=32:
uv run python -u train.py \
  --d-model 1024 --n-heads 16 --n-kv-heads 4 --n-layers 24 \
  --context-len 512 --batch-size 32 --epochs 3 \
  --curriculum --lr 3e-4 --no-checkpoint \
  2>&1 | tee training.log
```

RTX 4080 Super with bs=16: ~26K tok/s, ~83h per epoch.
NVIDIA B200 with bs=256: ~341K tok/s, ~6h per epoch.

`--no-checkpoint` disables gradient checkpointing (+12% speed, more VRAM).
`--curriculum` enables sequence length warmup (ctx=128→256→512).

### Resuming from checkpoint

Checkpoints are saved every 2000 steps and at end of each epoch to `checkpoint.pkl`.

```bash
uv run python -u train.py \
  --d-model 1024 --n-heads 16 --n-kv-heads 4 --n-layers 24 \
  --context-len 512 --batch-size 16 --epochs 3 \
  --curriculum --lr 3e-4 --no-checkpoint \
  --resume checkpoint.pkl \
  2>&1 | tee -a training.log
```

Restores params, optimizer state (Adam moments), LR schedule position, and exact
epoch/batch position. Training continues exactly where it left off.

### Data preparation

```bash
uv run python -u prepare_data_v2.py
```

Downloads all 5 sources from HuggingFace, tokenizes with 32K BPE, shuffles, and
combines into `data/tokens_v2/train.bin` (31.4GB). Idempotent — re-run to resume
interrupted downloads. Takes 1-2 hours depending on network speed.
