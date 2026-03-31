# Custom GPU Kernels for Transformer Inference

How to write Triton kernels that fuse an entire transformer decode step into a single
GPU launch — from register-level data flow to multi-SM parallelism with atomic barriers.
No GPU experience required — just Python and a rough idea of what neural nets do.

---

## Table of Contents

**Part I: The Fundamentals (d=64, 1 layer, 66K params)**

1. [The Problem](#1-the-problem)
2. [How GPUs Actually Work](#2-how-gpus-actually-work)
3. [The Memory Wall](#3-the-memory-wall)
4. [Our Transformer (the Tiny One)](#4-our-transformer-the-tiny-one)
5. [What Happens When JAX Runs Your Model](#5-what-happens-when-jax-runs-your-model)
6. [The Big Idea: One Kernel to Rule Them All](#6-the-big-idea-one-kernel-to-rule-them-all)
7. [Triton: GPU Programming for Humans](#7-triton-gpu-programming-for-humans)
8. [The Prefill Kernel, Line by Line](#8-the-prefill-kernel-line-by-line)
9. [KV Cache: Why We Don't Recompute Everything](#9-kv-cache-why-we-dont-recompute-everything)
10. [The Decode Kernel, Line by Line](#10-the-decode-kernel-line-by-line)
11. [The Generation Loop](#11-the-generation-loop)
12. [Results and Why It's Faster](#12-results-and-why-its-faster)

**Part II: Scaling Up (d=512, 8 layers, 30M params)**

13. [Scaling the Model](#13-scaling-the-model)
14. [Multi-SM Decode: Using the Whole GPU](#14-multi-sm-decode-using-the-whole-gpu)
15. [Batched Decode: Multiple Sequences at Once](#15-batched-decode-multiple-sequences-at-once)
16. [Persistent Decode: Eliminating Host Sync](#16-persistent-decode-eliminating-host-sync)
17. [What Didn't Work](#17-what-didnt-work)
18. [Key Lessons](#18-key-lessons)

---

## 1. The Problem

You have a trained transformer model. You want to generate text with it — feed in a prompt,
get tokens out, one at a time. The standard approach uses JAX (or PyTorch), which compiles
your Python code into GPU operations via XLA. This works, but it's slow.

**How slow?** For our tiny model (66K parameters, character-level Shakespeare), JAX generates
at 185 tokens/second. Our custom Triton kernel does the same job at 740 tokens/second.
**4x faster**, same outputs.

The speed difference isn't about better math. Both do identical matrix multiplications.
The difference is about **where the data lives** during computation.

---

## 2. How GPUs Actually Work

A GPU is not one fast processor — it's thousands of slow processors that all run the same
code simultaneously. An NVIDIA GPU has about 10,000 "CUDA cores" organized into groups.

### The three things that matter

**Threads.** The GPU runs your code on thousands of threads at once. You don't control
individual threads. Instead, you write a "kernel" — a function that says what ONE thread
block should do — and the GPU runs many copies of it in parallel.

**Warps.** Threads are grouped into "warps" of 32 threads that execute in lockstep. If one
thread in a warp takes a branch and the others don't, all 32 wait. A "thread block" typically
has 4-8 warps (128-256 threads).

**Tensor cores.** Special hardware units that multiply small matrices (like 16×16) in a single
clock cycle. They're the reason modern GPUs are so fast for deep learning. To use them, your
data must be in bfloat16 or float16 format.

### The memory hierarchy

This is the single most important concept for GPU performance:

```
┌─────────────────────────────┐
│        Registers            │   ← Fastest. Private to each thread.
│   Speed: ~1 cycle           │     128-256 registers per thread.
│   Size:  ~256 KB per block  │     Data here costs almost nothing to read.
├─────────────────────────────┤
│      Shared Memory          │   ← Fast. Shared within a thread block.
│   Speed: ~5 cycles          │     Up to 164 KB per block (on modern GPUs).
│   Size:  ~164 KB per block  │     Useful for communication between threads.
├─────────────────────────────┤
│     HBM (Global Memory)     │   ← Slow. The GPU's main memory.
│   Speed: ~200-400 cycles    │     Where your tensors normally live.
│   Size:  16-80 GB           │     Every jnp.array sits here.
└─────────────────────────────┘
```

**The speed gap is enormous.** Reading from registers is roughly 200x faster than reading
from HBM. Most GPU programs are "memory-bound" — the math units sit idle, waiting for
data to arrive from HBM.

For our model, every weight matrix fits in registers. The entire model (66K parameters ×
2 bytes = 132 KB) fits in the register file of a single thread block (128 threads × 255
registers × 4 bytes = 130 KB). This is the foundation of everything that follows.

---

## 3. The Memory Wall

When JAX (or PyTorch) runs a transformer, XLA compiles each operation into a separate GPU
kernel:

```
Operation              What happens in memory
─────────────────────  ─────────────────────────────────
token_emb[x]           Load tokens from HBM, load embedding table from HBM,
                       write result to HBM

+ pos_emb[:seq_len]    Load pos_emb from HBM, load previous result from HBM,
                       write sum to HBM

layer_norm(...)        Load from HBM, compute mean+var, write to HBM

x @ Wq                 Load x from HBM, load Wq from HBM, write Q to HBM
x @ Wk                 Load x from HBM, load Wk from HBM, write K to HBM
x @ Wv                 Load x from HBM, load Wv from HBM, write V to HBM

Q @ K^T                Load Q from HBM, load K from HBM, write scores to HBM
softmax(scores)        Load scores from HBM, write attention to HBM
attn @ V               Load attn from HBM, load V from HBM, write out to HBM

... (FFN, final LN, output projection — same pattern)
```

Count the HBM round-trips. **Every intermediate result** gets written to slow global memory
and then read back for the next operation. For a single forward pass with our model, that's
roughly 15-20 separate kernels, each doing a load-compute-store cycle.

The math takes nanoseconds. The memory transfers take microseconds. The kernel launch overhead
(just starting each kernel) takes microseconds. **Most of the GPU's time is spent waiting.**

---

## 4. Our Transformer (the Tiny One)

Before diving into the kernel code, let's understand exactly what our model computes.
It's a decoder-only transformer trained on character-level Shakespeare.

### Architecture

```
vocab_size:   65  (a-z, A-Z, punctuation, newline, space)
d_model:      64  (dimension of token representations)
n_heads:      2   (number of attention heads)
d_head:       32  (= d_model / n_heads)
d_ff:        256  (= 4 × d_model, FFN hidden dimension)
context_len: 128  (maximum sequence length)
parameters: 66,368
```

### The forward pass

Given a sequence of token IDs like `[14, 27, 51, 3, ...]`, the model:

**Step 1: Embedding.** Look up each token in a (65 × 64) table. Token 14 becomes a
64-dimensional vector. Add a positional embedding from a (128 × 64) table — position 0
gets one vector, position 1 gets another. Result: `h` with shape (128, 64).

**Step 2: Layer Norm 1.** For each position independently, normalize the 64 values to
have zero mean and unit variance, then scale and shift by learned parameters.
This stabilizes training. The formula:

```
mean = average of 64 values
var  = average of (value - mean)²
h_norm = scale * (h - mean) / sqrt(var + 1e-5) + bias
```

**Step 3: Multi-Head Attention.** This is where tokens "talk to each other." Split into
2 heads of 32 dimensions each. For each head:

```
Q = h_norm @ Wq    # "what am I looking for?" — shape (128, 32)
K = h_norm @ Wk    # "what do I contain?"     — shape (128, 32)
V = h_norm @ Wv    # "what do I offer?"        — shape (128, 32)

scores = Q @ K^T / sqrt(32)  # how relevant is each position to each other — (128, 128)
scores = mask_future(scores)  # position 5 can't see position 6 (causal)
attn = softmax(scores)        # normalize to probabilities
out = attn @ V                # weighted combination of values — (128, 32)
```

Project back: `h = h + concat(head0_out, head1_out) @ Wo`

The causal mask is crucial: position `i` can only attend to positions `0..i`. This is what
makes it "autoregressive" — each position only sees the past.

**Step 4: Layer Norm 2.** Same as step 2, different learned parameters.

**Step 5: Feed-Forward Network (FFN).** Two matrix multiplications with a nonlinearity:

```
up   = h_norm @ W_up + bias_up     # (128, 64) @ (64, 256) = (128, 256) — expand
act  = gelu(up)                     # element-wise nonlinearity
down = act @ W_down + bias_down     # (128, 256) @ (256, 64) = (128, 64) — compress
h = h + down                        # residual connection
```

GELU is like ReLU but smooth: `gelu(x) ≈ x · sigmoid(1.702x)`.

**Step 6: Final Layer Norm + Output Projection.**

```
h = layer_norm(h)
logits = h @ W_out    # (128, 64) @ (64, 65) = (128, 65) — one score per vocab token
```

The logits at position `i` are scores for what token should come at position `i+1`.
To generate text, we look at the logits of the **last** position, pick the highest
(greedy) or sample from the distribution.

---

## 5. What Happens When JAX Runs Your Model

JAX's `transformer_forward` in `model.py` is clean, readable Python:

```python
def transformer_forward(params, config, x):
    h = params["token_emb"][x] + params["pos_emb"][:seq_len]
    h_norm = layer_norm(h, params["layer0.ln1.scale"], params["layer0.ln1.bias"])
    attn_out = causal_attention(h_norm, wq, wk, wv, wo, n_heads)
    h = h + attn_out
    h_norm = layer_norm(h, params["layer0.ln2.scale"], params["layer0.ln2.bias"])
    h_ff = jax.nn.gelu(h_norm @ params["layer0.ffn.up"] + params["layer0.ffn.up_bias"])
    h_ff = h_ff @ params["layer0.ffn.down"] + params["layer0.ffn.down_bias"]
    h = h + h_ff
    h = layer_norm(h, params["ln_final.scale"], params["ln_final.bias"])
    return h @ params["output_proj"]
```

When you call this with `jax.jit`, XLA compiles it into a sequence of GPU kernels.
Each line becomes one or more kernel launches. Each kernel reads its inputs from HBM,
computes, writes results to HBM, and the next kernel picks up from there.

For **one forward pass**, there are roughly 15-20 HBM round-trips. For **autoregressive
generation** of 64 tokens, JAX calls this function 64 times (once per token, with growing
input length), so that's **~1000 kernel launches** with HBM round-trips each.

This is what we're going to fix.

---

## 6. The Big Idea: One Kernel to Rule Them All

What if we wrote a single GPU kernel that does the **entire forward pass** — embedding,
layer norm, attention, FFN, output projection — without ever writing intermediate results
to HBM?

```
Normal (JAX/XLA):
  HBM → Embedding → HBM → LN → HBM → QKV → HBM → Attention → HBM → FFN → HBM → Logits

Fused kernel:
  HBM → [Embedding → LN → QKV → Attention → FFN → Logits] → HBM
         └──────────── all in registers ────────────────┘
```

The data enters registers once (loading weights + input tokens from HBM), flows through
every operation in registers, and writes the final logits back to HBM once. **One HBM
round-trip instead of fifteen.**

This is possible because our model is small enough to fit in registers:

```
Total parameters: 66,368 × 2 bytes (bf16) = ~132 KB
Register file per block: 128 threads × 255 regs × 4 bytes = ~130 KB
```

It's tight, but it works. The weights aren't all live simultaneously — each phase loads
the weights it needs, uses them, then those registers get reused for the next phase.

For the attention scores matrix (128 × 128 = 16K values at 4 bytes each = 64 KB), this
is the largest live tensor and the binding constraint. It fits because we use 4 warps
(128 threads), giving each thread enough registers.

---

## 7. Triton: GPU Programming for Humans

[Triton](https://openai.com/index/triton/) is a GPU programming language by OpenAI.
It compiles Python-like code to the same PTX machine code that CUDA produces, but it's
much easier to write because:

1. **You think in blocks, not threads.** Instead of "what does thread #47 do?", you think
   "what does this block of 128 values do?"
2. **Memory coalescing is automatic.** Triton figures out how to load data efficiently.
3. **Register allocation is automatic.** You don't manually manage which values go in
   which registers.

### Triton basics

A Triton kernel looks like a Python function with special types:

```python
import triton
import triton.language as tl

@triton.jit
def my_kernel(input_ptr, output_ptr):
    # tl.arange creates a vector of indices: [0, 1, 2, ..., 127]
    idx = tl.arange(0, 128)

    # tl.load reads from GPU memory into registers
    x = tl.load(input_ptr + idx)

    # Math happens in registers (no memory access)
    y = x * 2.0 + 1.0

    # tl.store writes from registers back to GPU memory
    tl.store(output_ptr + idx, y)
```

Key operations we use:

| Operation | What it does | Example |
|-----------|-------------|---------|
| `tl.load(ptr + offsets)` | Read from HBM into registers | Loading a weight matrix |
| `tl.store(ptr + offsets, val)` | Write from registers to HBM | Storing output logits |
| `tl.dot(A, B)` | Matrix multiply using tensor cores | Q @ K^T, h @ W_q |
| `tl.sum(x, axis=)` | Reduce along an axis | Computing mean for layer norm |
| `tl.exp(x)` | Element-wise exponential | Softmax numerator |
| `tl.where(cond, a, b)` | Conditional select | Causal mask |
| `tl.arange(0, N)` | Index vector [0, 1, ..., N-1] | Position offsets |
| `.to(tl.bfloat16)` | Cast to bfloat16 | Preparing for tensor core matmul |

### The bf16 + f32 pattern

Tensor cores require bfloat16 inputs but produce float32 outputs. So the standard pattern is:

```python
# Cast inputs to bf16 for tensor cores, accumulate in f32 for precision
result = tl.dot(A.to(tl.bfloat16), B.to(tl.bfloat16)).to(tl.float32)
```

This gives you the speed of tensor cores (8-16× faster than scalar math) with the precision
of float32 accumulation.

### Calling Triton kernels from JAX

We use `jax_triton`, a bridge library. It passes JAX arrays as pointers to the Triton kernel
with zero-copy sharing (same GPU memory):

```python
import jax_triton as jt

result = jt.triton_call(
    input_array,          # inputs (JAX arrays → pointers)
    kernel=my_kernel,     # the @triton.jit function
    out_shape=[           # shape/dtype of outputs (jax_triton allocates them)
        jax.ShapeDtypeStruct((128,), jnp.float32),
    ],
    grid=(1,),            # how many thread blocks to launch
    num_warps=4,          # threads per block = num_warps × 32
)
```

---

## 8. The Prefill Kernel, Line by Line

"Prefill" processes the entire input prompt at once. All 128 positions are computed in
parallel. This is the same as a normal forward pass — the only difference from the JAX
version is that everything happens in one kernel with data in registers.

The kernel is in `kernels/fused_prefill.py`. Let's walk through it.

### Setup

```python
@triton.jit
def _prefill_kernel(
    token_emb_ptr, pos_emb_ptr,           # embedding tables
    ln1_scale_ptr, ln1_bias_ptr,           # layer norm 1
    wq_ptr, wk_ptr, wv_ptr, wo_ptr,       # attention weights
    ln2_scale_ptr, ln2_bias_ptr,           # layer norm 2
    ffn_up_ptr, ffn_up_bias_ptr,           # FFN
    ffn_down_ptr, ffn_down_bias_ptr,
    ln_final_scale_ptr, ln_final_bias_ptr, # final layer norm
    output_proj_ptr,                       # output projection
    x_ptr,                                 # input token IDs
    logits_ptr, k_cache_ptr, v_cache_ptr,  # outputs
):
```

Every argument is a pointer to GPU memory. Inputs are the weight matrices (in bf16)
and the token IDs. Outputs are logits, K cache, and V cache.

```python
    pos = tl.arange(0, 128)     # [0, 1, 2, ..., 127] — position indices
    d = tl.arange(0, 64)        # [0, 1, 2, ..., 63]  — model dimension indices
```

These index vectors live in registers. Every subsequent load uses them to compute memory
addresses.

### Embedding

```python
    tokens = tl.load(x_ptr + pos)  # load 128 token IDs from HBM
    h = (tl.load(token_emb_ptr + tokens[:, None] * 64 + d[None, :]).to(tl.float32)
       + tl.load(pos_emb_ptr + pos[:, None] * 64 + d[None, :]).to(tl.float32))
```

`tokens[:, None] * 64 + d[None, :]` computes a (128, 64) grid of memory offsets.
For token ID 14 at position 0, it reads row 14 of the embedding table (64 values).
The positional embedding is simpler — row 0 for position 0, row 1 for position 1.

Result: `h` is a (128, 64) matrix in registers. Each of the 128 rows is one token's
64-dimensional representation. **This matrix never touches HBM again until the final
logits.**

### Layer Norm 1

```python
    ln1_s = tl.load(ln1_scale_ptr + d).to(tl.float32)   # (64,) scale
    ln1_b = tl.load(ln1_bias_ptr + d).to(tl.float32)    # (64,) bias
    mean = tl.sum(h, axis=1)[:, None] / 64               # mean per position
    hc = h - mean                                         # center
    h_norm = ln1_s[None, :] * hc * tl.math.rsqrt(        # normalize + scale + shift
        tl.sum(hc * hc, axis=1)[:, None] / 64 + 1e-5
    ) + ln1_b[None, :]
```

`tl.sum(h, axis=1)` sums each row (each position's 64 values), giving a (128,) vector.
`[:, None]` makes it (128, 1) for broadcasting. `rsqrt` is 1/sqrt — faster than
dividing by sqrt.

In JAX, this layer norm would be a separate kernel. Here it's just a few lines that
operate on data already in registers.

### Multi-Head Attention

This is the most complex part. We loop over 2 heads:

```python
    scale = 0.17677669529663689  # 1/sqrt(32) precomputed
    dh = tl.arange(0, 32)       # head dimension indices

    for head in tl.range(2):
        hd = head * 32 + dh      # column offsets for this head
```

`tl.range(2)` is a **dynamic loop** — Triton doesn't unroll it, which saves registers.
`hd` selects columns 0-31 for head 0, columns 32-63 for head 1.

**Q/K/V projections:**

```python
        K = tl.dot(h_norm.to(tl.bfloat16),
                   tl.load(wk_ptr + d[:, None] * 64 + hd[None, :]).to(tl.bfloat16)
            ).to(tl.float32)
```

This loads a (64, 32) slice of the Wk weight matrix and multiplies: (128, 64) @ (64, 32)
= (128, 32). The `.to(tl.bfloat16)` before `tl.dot` enables tensor cores.
Same pattern for Q and V.

**Saving the KV cache:**

```python
        tl.store(k_cache_ptr + head * 128 * 32 + pos[:, None] * 32 + dh[None, :],
                 K.to(tl.bfloat16))
```

This writes K to HBM for the decode phase to use later. It's the one place during
prefill where intermediate data goes to HBM on purpose — because the decode kernel
needs it.

**Causal attention:**

```python
        scores = tl.dot(Q.to(tl.bfloat16), tl.trans(K.to(tl.bfloat16))
                 ).to(tl.float32) * scale
        scores = tl.where(pos[:, None] >= pos[None, :], scores, -1e9)
```

`Q @ K^T` gives a (128, 128) attention matrix. `pos[:, None] >= pos[None, :]` creates
the causal mask: position 5 can see positions 0-5 (True) but not 6-127 (False).
The `-1e9` values become effectively 0 after softmax.

This (128, 128) matrix = 16,384 float32 values = 64 KB is the **largest tensor in the
kernel**. It's the reason we need enough registers — and why 4 warps (128 threads) is
the sweet spot. More warps = fewer registers per thread = this matrix wouldn't fit.

**Softmax:**

```python
        exp_s = tl.exp(scores - tl.max(scores, axis=1)[:, None])
        attn = exp_s / tl.sum(exp_s, axis=1)[:, None]
```

Subtract the max before exponentiating for numerical stability (standard trick).
Then normalize each row to sum to 1.

**Attention output → O projection → residual:**

```python
        attn_out = tl.dot(attn.to(tl.bfloat16), V.to(tl.bfloat16)).to(tl.float32)
        h += tl.dot(attn_out.to(tl.bfloat16),
                    tl.load(wo_ptr + hd[:, None] * 64 + d[None, :]).to(tl.bfloat16)
             ).to(tl.float32)
```

Multiply attention weights by V: (128, 128) @ (128, 32) = (128, 32). Then project back:
(128, 32) @ (32, 64) = (128, 64). Add to `h` (residual connection). Both heads contribute
to the same `h`.

### FFN

The FFN has a (64, 256) up-projection and a (256, 64) down-projection. The intermediate
(128, 256) tensor is too big to hold all at once (128 × 256 = 32K values = 128 KB).
So we tile it:

```python
    ffn_out = tl.zeros((128, 64), dtype=tl.float32)
    for k in tl.range(0, 256, 32):
        kk = k + tl.arange(0, 32)
        up = tl.dot(...) + bias  # (128, 64) @ (64, 32) = (128, 32) — a tile of the FFN
        act = up * tl.sigmoid(1.702 * up)  # GELU on just this tile
        ffn_out += tl.dot(...)   # (128, 32) @ (32, 64) = (128, 64) — accumulate
```

We process 32 FFN hidden units at a time (8 iterations). Each iteration loads a
(64, 32) tile of W_up and a (32, 64) tile of W_down. The intermediate activation
is only (128, 32) = 4K values — small enough for registers.

`tl.range` (not `tl.static_range`) tells Triton to use a dynamic loop. If Triton
unrolled all 8 iterations, it would need registers for all 8 tiles simultaneously,
causing spilling to slow local memory. Dynamic loops reuse registers.

### Final Layer Norm + Output Projection

```python
    # Layer norm (same pattern as before)
    # ...

    # Output projection
    v = tl.arange(0, 128)  # padded vocab indices
    logits = tl.dot(h_final.to(tl.bfloat16),
                    tl.load(output_proj_ptr + d[:, None] * 128 + v[None, :]).to(tl.bfloat16)
             ).to(tl.float32)
    logits = tl.where(v[None, :] < 65, logits, -1e9)  # mask padding
    tl.store(logits_ptr + pos[:, None] * 128 + v[None, :], logits)
```

The vocab has 65 tokens but `tl.dot` needs power-of-2 dimensions, so we pad to 128
and mask the padding to -infinity (which becomes 0 probability after softmax).

The `tl.store` at the end is the only write to HBM in the entire forward pass
(besides the KV cache stores). Everything else happened in registers.

### The Python wrapper

```python
def fused_prefill(params, x):
    assert x.shape == (128,)

    def bf(key):
        return params[key].astype(jnp.bfloat16)

    logits_pad, k_cache, v_cache = jt.triton_call(
        bf("token_emb"), bf("pos_emb"),
        bf("layer0.ln1.scale"), bf("layer0.ln1.bias"),
        bf("layer0.attn.q"), bf("layer0.attn.k"),
        bf("layer0.attn.v"), bf("layer0.attn.o"),
        bf("layer0.ln2.scale"), bf("layer0.ln2.bias"),
        bf("layer0.ffn.up"), bf("layer0.ffn.up_bias"),
        bf("layer0.ffn.down"), bf("layer0.ffn.down_bias"),
        bf("ln_final.scale"), bf("ln_final.bias"),
        jnp.pad(params["output_proj"], [(0, 0), (0, 63)]).astype(jnp.bfloat16),
        x.astype(jnp.int32),
        kernel=_prefill_kernel,
        out_shape=[...],
        grid=(1,),        # one thread block — processes all 128 positions
        num_warps=4,      # 128 threads = sweet spot for registers
        num_stages=1,
    )
    return logits_pad[:, :65], k_cache, v_cache
```

`grid=(1,)` means we launch exactly one thread block. That single block of 128 threads
processes all 128 positions through the entire transformer. `num_warps=4` gives 4 × 32
= 128 threads, which maximizes registers per thread (255 registers, 0 spill).

---

## 9. KV Cache: Why We Don't Recompute Everything

When generating text, we produce tokens one at a time:

```
Prompt:    "The cat sat on the"
Generate:   → "m"     (position 5)
            → "a"     (position 6)
            → "t"     (position 7)
            ...
```

At position 6, the model needs to attend to positions 0-6. The Q/K/V projections for
positions 0-5 are **exactly the same** as they were when we processed the prompt. Only
position 6 is new.

Without caching, we'd recompute Q, K, V for ALL previous positions every time we generate
a token. For 64 generated tokens, that's 64 + 63 + 62 + ... + 1 = 2,080 position
computations instead of just 64.

**The KV cache** stores K and V for all previous positions. When generating position 6,
we only compute K and V for position 6, look up K and V for positions 0-5 from the cache,
and do the attention.

```
Cache layout: (n_heads, max_seq, d_head) = (2, 128, 32) in bf16
              = 2 × 128 × 32 × 2 bytes = 16 KB

After prefill:  positions 0-63 filled (from prompt)
After decode 1: position 64 filled
After decode 2: position 65 filled
...
```

The prefill kernel outputs the KV cache as a side effect. The decode kernel reads
from it and outputs new K/V vectors, which the JAX wrapper inserts at the right position.

---

## 10. The Decode Kernel, Line by Line

The decode kernel (`kernels/fused_decode.py`) processes **one token** instead of 128.
This changes the computational profile dramatically:

| | Prefill | Decode |
|---|---|---|
| Tokens processed | 128 | 1 |
| Matmul shape | (128, 64) @ (64, 64) | (1, 64) @ (64, 64) |
| Can use tensor cores? | Yes (M=128) | No (M=1, too small) |
| Attention shape | (128, 128) | (1, pos) |
| Bottleneck | Compute | Memory (loading KV cache) |

Because we're processing a single token, `tl.dot` can't use tensor cores (they need
at least 16×16 tiles). So the decode kernel uses **element-wise operations** instead:

### Element-wise matmul

Instead of `tl.dot(h[None, :], W)` which would be a (1, 64) @ (64, 32) matmul,
we do:

```python
Q = tl.sum(h_norm[:, None] * Wq, axis=0)    # (64,) * (64, 32) → sum axis 0 → (32,)
```

`h_norm[:, None]` broadcasts (64,) to (64, 1). Multiply element-wise with (64, 32).
Sum across the 64 dimension. Result: (32,) — the Q vector for this single position.

This is mathematically identical to a matrix multiply but uses scalar operations instead
of tensor cores. For M=1, this is actually faster because tensor core setup has overhead.

### Attention with the KV cache

The decode kernel attends to all cached positions plus the current one:

```python
    mask = seq <= pos  # True for positions 0..pos

    for head in tl.range(2):
        # Compute Q, K_new, V_new for this position
        Q = tl.sum(h_norm[:, None] * Wq, axis=0)
        K_new = tl.sum(h_norm[:, None] * Wk, axis=0)
        V_new = tl.sum(h_norm[:, None] * Wv, axis=0)

        # Store K_new, V_new for output (cache update)
        tl.store(k_new_ptr + ..., K_new.to(tl.bfloat16))
        tl.store(v_new_ptr + ..., V_new.to(tl.bfloat16))

        # Load K cache and insert K_new at current position
        K = tl.load(k_cache_ptr + ..., mask=mask[:, None], other=0.0)
        K = tl.where(seq[:, None] == pos, K_new[None, :], K)
```

`tl.where(seq[:, None] == pos, K_new, K)` inserts the newly computed K vector into
the loaded cache at position `pos`. This avoids writing to the input cache (which is
read-only in jax_triton).

**Attention scores:**

```python
        scores = tl.sum(Q[None, :] * K, axis=1) * scale  # (128,)
        scores = tl.where(mask, scores, -1e9)
```

`Q[None, :]` is (1, 32), `K` is (128, 32). Element-wise multiply and sum over the
32 dimension gives (128,) scores — one per cached position. Mask out future positions.

**Weighted sum of V:**

```python
        attn_out = tl.sum(attn_w[:, None] * V, axis=0)  # (32,)
```

`attn_w` is (128,) attention weights. Multiply each V row by its weight, sum
across the 128 positions. Result: a single (32,) vector.

### Why the decode kernel is different

Notice the kernel outputs **new K/V vectors**, not an updated cache:

```python
    logits_pad, k_new, v_new = jt.triton_call(
        ...,
        out_shape=[
            jax.ShapeDtypeStruct((128,), jnp.float32),  # logits (padded)
            jax.ShapeDtypeStruct((2, 32), jnp.bfloat16), # new K per head
            jax.ShapeDtypeStruct((2, 32), jnp.bfloat16), # new V per head
        ],
    )
```

The JAX wrapper then updates the cache:

```python
    return logits_pad[:65], k_cache.at[:, pos, :].set(k_new), v_cache.at[:, pos, :].set(v_new)
```

Why not update the cache inside the kernel? Because `jax_triton` passes inputs as
read-only buffers. The kernel would have to copy the entire cache (16 KB) to a new
output buffer. Instead, we output just 128 bytes of new K/V and let JAX do the
scatter update efficiently.

---

## 11. The Generation Loop

The complete generation pipeline:

```python
def generate_triton(params, prompt, n_tokens):
    # 1. Pad prompt to 128 tokens (causal mask makes padding harmless for early positions)
    x = jnp.pad(prompt, (0, 128 - len(prompt))).astype(jnp.int32)

    # 2. Prefill: one kernel call processes entire prompt
    logits, k_cache, v_cache = fused_prefill(params, x)

    # 3. Sample first generated token from last prompt position's logits
    token = jnp.argmax(logits[len(prompt) - 1])
    tokens = [int(token)]

    # 4. Decode loop: one kernel call per token
    for i in range(n_tokens - 1):
        logits, k_cache, v_cache = fused_decode(
            params, token, len(prompt) + i, k_cache, v_cache
        )
        token = jnp.argmax(logits)
        tokens.append(int(token))

    return tokens
```

**Step 1:** The prompt is padded to 128 tokens. Due to the causal mask, position 63
(the last real token) only attends to positions 0-63, which are all real. The padding
at positions 64-127 can't contaminate positions 0-63.

**Step 2:** One kernel call. In registers: embedding → LN → attention → FFN → LN → logits.
Side effect: KV cache written to HBM.

**Step 3:** `argmax` picks the most probable next token (greedy decoding). You could also
sample with temperature for more creative text.

**Step 4:** Each iteration does one decode kernel call, updates the KV cache in JAX,
and picks the next token. The KV cache grows by one entry per step.

---

## 12. Results and Why It's Faster

### The numbers (d=64, 1 layer)

```
Prompt: 64 tokens, Generate: 64 tokens

Triton fused:   740 tok/s  (86.5 ms total)
JAX baseline:   185 tok/s  (362.6 ms total)
Speedup:        4.0x
```

Both produce identical text, confirming the kernels are correct.

### Where does the speedup come from?

**1. Fewer HBM round-trips.** The prefill does one round-trip instead of ~15. The decode
kernel does one kernel launch instead of ~15. Over 64 decode steps, that's ~900 fewer
kernel launches.

**2. No kernel launch overhead.** Each GPU kernel launch has 5-10 µs of overhead. For JAX's
~15 kernels per forward pass × 64 decode steps = ~960 launches × ~7 µs = ~6.7 ms of pure
overhead. The Triton version has 65 launches (1 prefill + 64 decode) × ~7 µs = ~0.5 ms.

**3. Data stays in registers.** Intermediate activations (the `h` matrix, attention scores,
FFN hidden states) never leave registers. In JAX, each intermediate is written to HBM
(~400 cycles per read/write) and read back. In the fused kernel, it's ~1 cycle per access.

**4. No redundant computation.** The KV cache means the decode kernel only processes 1 token
instead of recomputing all previous tokens. (JAX doesn't implement KV caching in our
baseline, so it processes the full growing sequence each time.)

### What it's NOT

- **Not better math.** Both do the same multiplications.
- **Not parallel tokens.** Both process one token at a time during decode.
- **Not quantization.** Both use bf16 for matmuls.

The speedup is purely from keeping data close to the compute units.

This was the starting point. Everything that follows is about scaling this approach to a
real model and dealing with the problems that emerge at larger scale.

---

## 13. Scaling the Model

The fused-kernel approach works brilliantly at d=64 because the entire model fits in
registers. But what happens when we scale up?

### The scaling journey

```
d=64,  1L,   66K params:    740 tok/s  (4.0x vs JAX)
d=128, 2L,  674K params:   2504 tok/s  (14.0x — in-kernel KV cache updates)
d=256, 4L, 5.3M params:   1396 tok/s  (15.1x — fused N-layer decode)
d=512, 8L,  30M params:    287 tok/s  (single SM, 2% bandwidth utilization)
                          4777 tok/s  (persistent kernel, final)
```

Each scale-up introduced new problems. Here's what happened and how we solved them.

### d=128: The model no longer fits in one block

At d=128, the hidden state `h` is (128, 128) = 64 KB — it can't coexist with the attention
scores matrix (also 128 × 128 = 64 KB) in the 130 KB register file. Solution: **multi-block
prefill**. Tile the sequence dimension into BLOCK_SEQ=32 row blocks. Each of 4 thread blocks
handles 32 positions. h_block is (32, 128) = 16 KB. Scores are (32, 128) = 16 KB. Fits.

The cost is that blocks can't communicate within a single kernel. So projections (which write
K/V to HBM) must be a separate kernel launch from attention (which reads K/V from HBM).
Three kernels per layer instead of one.

The decode kernel stays fused — with M=1, all tensors are 1D and tiny.

The three biggest optimizations at this scale:

1. **In-kernel KV cache updates (3.6x win).** Instead of outputting K_new/V_new and having
   JAX do `.at[pos].set()` (which copies the entire cache per step), the kernel writes the
   updated cache directly to an output buffer. This eliminated the biggest bottleneck.

2. **Precomputed bf16 weights (57% win).** `.astype(bf16)` inside the decode loop created
   a new JAX array per call — 28 weights × 63 steps = 1764 allocations. Converting once
   before the loop: free.

3. **Fused multi-layer decode (35% win).** Instead of one kernel call per layer, a single
   kernel processes all layers with h staying in registers between them.

### d=256: Fused N-layer decode with packed buffers

At 4 layers, passing individual weight pointers becomes unwieldy (50+ arguments). Solution:
**packed weight buffer** — concatenate all per-layer weights into one bf16 buffer. The kernel
computes offsets from the layer index. Same for KV caches: all layers packed into one flat
buffer.

FlashAttention becomes necessary at context > 256. The full scores matrix would be (16, 512)
= 32 KB per head, too large alongside K and V. Tiled KV with online softmax (KV_TILE=64)
keeps peak memory at ~20 KB per tile regardless of context length.

### d=512: Register pressure and the single-SM bottleneck

At d=512, two critical problems emerge:

**Register overflow.** Element-wise projections `tl.sum(h[:, None] * W, axis=0)` create a
(512, 32) intermediate = 128 registers per thread. The product with h needs another 128.
Total: 260 > 255 limit. Fix: `tl.dot` tiles the reduction internally, never materializing
the full intermediate.

**Single-SM bottleneck.** The fused kernel runs on grid=(1,) — one thread block on one SM.
The RTX 4080 Super has 80 SMs. At d=512 with 8 layers, the kernel takes 3.48 ms per token.
Profiling showed: kernel=93% of time, host=3%. The GPU is 98% idle — not because it's
waiting for data, but because 79 of 80 SMs have no work.

This is the fundamental challenge that the next three sections address.

---

## 14. Multi-SM Decode: Using the Whole GPU

### The problem

At d=512, a single thread block runs the entire decode step. It processes 16 attention heads
sequentially, then the full FFN, then layer norm, for each of 8 layers. 79 out of 80 SMs
sit idle. Bandwidth utilization: 2%.

### The solution: one block per attention head

Launch grid=(N_HEADS,) = grid=(16,). Each of the 16 blocks handles one attention head's
Q/K/V projections, attention, and O-projection. Then all 16 blocks split the FFN work
(each handles D_FF/16 = 128 columns of the up-projection).

```
Block 0:  head 0 attention + FFN columns 0-127
Block 1:  head 1 attention + FFN columns 128-255
...
Block 15: head 15 attention + FFN columns 1920-2047
```

### Cross-block synchronization with atomic barriers

The problem: blocks can't communicate within a kernel. After attention, all blocks must
wait for all partial results before anyone can compute the next layer's input.

Solution: **atomic barriers** using GPU-scope atomics in L2 cache.

```python
# Barrier implementation (simplified):
# Each block arrives by atomically incrementing a counter
old = tl.atomic_add(barrier_ptr, 1, sem='release', scope='gpu')
if old == N_BLOCKS - 1:
    # Last block to arrive: set done flag
    tl.atomic_xchg(done_ptr, 1, sem='release', scope='gpu')
# All blocks wait for done flag
while tl.atomic_add(done_ptr, 0, sem='acquire', scope='gpu') == 0:
    pass  # spin-wait
```

Two barriers per layer: one after attention (before FFN), one after FFN (before next layer).
With 8 layers + 1 final: 17 barriers total.

**Key insight: redundant computation is cheaper than synchronization.** All 16 blocks
independently compute LayerNorm and reduce the 16 partial FFN results. This is redundant
(each block reads 32 KB from L2) but costs only ~1 µs. The alternative — one block computes
and broadcasts — would need an additional barrier (~5 µs) plus a serial bottleneck.

### KV-split parallelism (FlashDecoding)

With 16 blocks on 80 SMs, utilization is only 20%. **KV-split parallelism** doubles the
grid to 32 by having 2 blocks per attention head, each handling half the KV cache tiles.

Each block computes a partial online softmax (partial O-projection + running max + running
sum). After the barrier, all blocks merge the partials using the log-sum-exp trick:

$$h_{head} = \frac{\sum_s o_s \cdot l_s \cdot e^{m_s - m_{max}}}{\sum_s l_s \cdot e^{m_s - m_{max}}}$$

where $o_s$ is the normalized partial output, $l_s$ is the partial sum of exponentials, and
$m_s$ is the partial maximum. This is mathematically exact — no approximation.

### Split barrier optimization

The standard barrier has all blocks polling the same counter address while others are still
arriving. This causes **L2 cache line thrashing** — arrivals invalidate the line that pollers
are reading.

Fix: separate the arrival counter and done-flag on different cache lines. Arrivals write to
`counter[]`, the last-arriving block sets `done[]`, all blocks poll `done[]`. Result: +5%
kernel speedup.

### Results

```
Single-SM (grid=1):        287 tok/s  (3.48 ms/tok, 2% BW utilization)
Multi-SM (grid=16):       1734 tok/s  (0.58 ms/tok, 14% BW, 6.0x)
+ KV splits (grid=32):   1851 tok/s  (0.54 ms/tok, 15% BW, +7%)
+ Split barrier:          1937 tok/s  (0.52 ms/tok, 16% BW, +5%)
Without GPU→CPU sync:     3108 tok/s  (0.32 ms/tok — pure GPU speed)
```

The 6.0x speedup from multi-SM is the single largest improvement at d=512.
GPU→CPU sync (`int()` per token) costs 34% of total time — addressed in section 16.

---

## 15. Batched Decode: Multiple Sequences at Once

### The idea

In production, you're often generating for multiple users simultaneously. Instead of
running B separate decode calls, run one kernel that processes B sequences in parallel.
Weight loads are shared across the batch — loaded once, used B times.

### Implementation

Same grid as single-sequence, but each block processes B sequences. The kernel loads each
weight tile once, then loops over B sequences applying it. KV caches are independent per
sequence.

### Race conditions (the hard part)

Two critical race conditions emerged:

**1. h_buf read-write race.** All blocks read h, compute `h_new = h + residual`, and write
back. A fast block can overwrite h before a slow block reads the original, causing
`h + 2 × residual` instead of `h + residual`. Fix: **double-buffered h_buf** — even layers
read buf_a/write buf_b, odd layers read buf_b/write buf_a.

**2. Partial buffer merge/FFN race.** After the phase 1 barrier, all blocks read attention
o_proj from the partial buffer while also writing FFN results to the same buffer. A fast
block can overwrite o_proj before a slow block reads it. Fix: **separate ffn_buf** for FFN
accumulation.

### Weight-amortized FFN

The initial batched kernel loaded FFN weights B times per tile (once per batch element in
a fused merge+LN+FFN loop). De-fusing into separate passes — merge+LN for all B first,
then FFN with outer k-loop / inner b-loop — loads weights once per tile regardless of B.

Result: B=4 +14%, B=8 +22%, B=16 +19%.

### Results

```
Single-sequence:        1935 tok/s
Batched B=4 (sync):     4205 tok/s  (2.17x)
Batched B=4 (pipe):     6691 tok/s  (3.46x)
Batched B=8 (pipe):     7339 tok/s  (3.79x)
Batched B=16 (sync):    6064 tok/s  (3.13x)
```

Why not 4x at B=4? Per-sequence compute (attention over 512 KV positions) adds ~0.17 ms
per additional sequence regardless of weight sharing. The theoretical 4x assumed weights
dominated kernel time, but attention is a significant fixed cost per sequence.

---

## 16. Persistent Decode: Eliminating Host Sync

### The GPU→CPU sync bottleneck

After multi-SM optimization, the kernel itself runs in 0.32 ms per token. But `int(next_token)`
forces a GPU→CPU transfer each step, adding 0.15-0.20 ms. **Sync accounts for 30-60% of
wall time.**

### Persistent kernel: one launch for all tokens

Instead of one kernel call per token, launch a single kernel that runs all N decode steps
internally. Block 0 computes argmax in-kernel and writes the next token to a workspace buffer.
A step-sync barrier ensures all blocks see it before the next step's embedding lookup.

```
Normal decode (N kernel launches):
  Host: launch → sync → read token → launch → sync → read token → ...

Persistent decode (1 kernel launch):
  Host: launch ──────────────────────────────────→ read all N tokens
  GPU:  [step 0 → step 1 → step 2 → ... → step N]
         └── all tokens stay on GPU, no host sync ──┘
```

Fresh barrier slots per step (`step * BARRIERS_PER_STEP + barrier_idx`) avoid reusing
counters. The initial KV copy (input → output buffer) is done cooperatively by all blocks
before the step loop, protected by a barrier.

### Pipelined batched decode

For batched inference, a lighter-weight alternative: tokens stay on GPU as JAX arrays, and
JAX overlaps dispatch of the next step with execution of the current one. Not as fast as
a true persistent kernel, but much simpler.

### Results

```
Single-seq sync'd:      1869 tok/s  (0.535 ms/tok)
Single-seq pipelined:   2624 tok/s  (0.381 ms/tok)
Persistent kernel:      4777 tok/s  (0.209 ms/tok)  ← 2.56x vs sync'd
Batched B=4 sync'd:     4205 tok/s
Batched B=4 pipelined:  6691 tok/s                   ← 1.59x vs sync'd
```

---

## 17. What Didn't Work

Not everything we tried improved performance. Documenting failures is as important as
documenting successes — they reveal what the actual bottlenecks are.

**GQA (Grouped Query Attention).** 4 KV heads instead of 16 Q heads. KV cache shrinks from
8.4 MB to 2.1 MB per sequence. But decode speed was unchanged — the kernel is
barrier-limited, not memory-limited. The data already fits in L2 at high bandwidth. GQA
helps batched inference (where KV cache scales with batch size) but not single-sequence.

**Parallel residual (attn || FFN).** Compute attention and FFN in parallel, reducing
barriers from 17 to 9. Result: +1.3% speedup. Why? Barrier cost is dominated by straggler
*variance* (proportional to total work per step), not fixed overhead. Halving barriers
saves ~8 µs of fixed overhead, but the ~80 µs of straggler time stays constant.

**num_warps sweep.** 2, 4, 8 warps all within 5%. The kernel is not warp-limited.

**Speculative decoding.** With fused kernels, both draft (~2500 tok/s) and target (~1400
tok/s) are fast. The speed ratio is only ~2x, not the ~10x needed. Acceptance rate was
36-51% with a 1-layer draft — you need 80%+ to break even at this ratio.

---

## 18. Key Lessons

### Fundamentals

**1. Memory bandwidth is the bottleneck, not compute.** Modern GPUs do trillions of ops/s
but read ~2 TB/s from HBM. For small models, math finishes instantly — the GPU waits for data.

**2. Kernel fusion is the highest-leverage optimization.** Eliminating HBM round-trips
gives the biggest speedup. In-kernel KV cache updates alone gave 3.6x.

**3. bf16 + f32 accumulation is the sweet spot.** Cast to bf16 for tensor cores, accumulate
in f32. FP8 was slower (register pressure from casts outweighed gains).

**4. Dynamic loops prevent register spilling.** `tl.range` emits a real loop; `tl.static_range`
unrolls, requiring registers for all tiles simultaneously.

### Scaling

**5. BLOCK_SEQ scales inversely with d_model.** Register file is fixed. d=128: BLOCK_SEQ=32.
d=256: BLOCK_SEQ=16. d=512: BLOCK_SEQ=8. The h_block stays at ~16 KB.

**6. Use tl.dot for projections at d >= 512.** Element-wise `h[:, None] * W` materializes the
full (d, d_head) intermediate. tl.dot tiles internally, avoiding register overflow.

**7. The bottleneck shifts with model size.** At d=64, host dispatch dominates (Python overhead).
At d=512, the GPU kernel dominates (93% of step time). Optimizations that help at one scale
may be irrelevant at another. Always profile first.

### Multi-SM and parallelism

**8. Multi-SM with atomic barriers unlocks the full GPU.** Going from grid=(1,) to grid=(32,)
gave 6.8x speedup at d=512. The technique: one block per attention head, all blocks split
the FFN, atomic barriers for cross-block sync.

**9. Redundant computation beats synchronization.** All blocks independently compute LayerNorm
(~1 µs from L2) rather than one block computing and broadcasting (needs a ~5 µs barrier).

**10. GPU→CPU sync is the #1 bottleneck at 30M params.** `int()` per token costs 30-60% of
wall time. Persistent kernels and pipelining eliminate this entirely.

### Batched inference

**11. Weight amortization requires careful loop structure.** Outer k-loop / inner b-loop loads
weights once per tile. The naive fused approach loads them B times.

**12. Shared buffers need double-buffering or phase separation.** Any buffer that is both read
and written by all blocks within a barrier-delimited phase needs either double-buffering or a
separate buffer for the write phase.

### What doesn't help

**13. Reducing barrier count gives minimal speedup.** Straggler variance (proportional to total
work) dominates fixed barrier overhead. Halving barriers from 17 to 9 gave only 1.3%.

**14. GQA doesn't help when barrier-limited.** If data already fits in L2, reducing it further
doesn't matter. GQA helps when KV cache is the memory bottleneck (batched, long-context).

---

## Running It Yourself

```bash
# train the model (d=512, 8 layers, ~4 hours on TinyStories)
uv run train_backprop.py --d-model 512 --n-heads 16 --n-layers 8 \
  --context-len 512 --epochs 3 --lr 1e-4 --batch-size 16

# profile kernels (primary benchmark)
uv run profile_kernels.py

# quick inference demo
uv run inference_benchmark.py
```

### File overview

```
Core:
  model.py                        — JAX transformer model
  data.py                         — Shakespeare + TinyStories + BPE tokenizer
  train_backprop.py               — AdamW training with LR schedule

Kernels:
  kernels/fused_prefill.py        — fused prefill (d_model <= 64, one kernel call)
  kernels/fused_decode.py         — fused decode (d_model <= 64, one kernel call)
  kernels/block_prefill.py        — multi-block prefill + FlashAttention (d_model >= 128)
  kernels/block_decode.py         — per-layer decode orchestrator (d_model >= 128)
  kernels/fused_decode_nlayer.py  — fused N-layer decode (packed weights/caches)
  kernels/multi_sm_decode.py      — multi-SM decode with atomic barriers + KV-split
  kernels/batched_decode.py       — batched multi-SM decode (B sequences)
  kernels/persistent_decode.py    — persistent decode (single launch, all steps)

Benchmarking:
  profile_kernels.py              — primary profiling tool
  inference_benchmark.py          — quick throughput + text generation demo
  baseline_metrics.txt            — current performance numbers
```
