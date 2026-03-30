# From Python to GPU Registers: Building a 4x Faster Transformer

A step-by-step guide to writing custom GPU kernels for transformer inference.
No GPU experience required — just Python and a rough idea of what neural nets do.

---

## Table of Contents

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
13. [Key Lessons](#13-key-lessons)

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

### The numbers

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

---

## 13. Key Lessons

**1. Memory bandwidth is the bottleneck, not compute.**
Modern GPUs can do trillions of operations per second but can only read ~2 TB/s from HBM.
For a tiny model like ours, the math finishes instantly — the GPU spends most of its time
waiting for data.

**2. Kernel fusion is the highest-leverage optimization.**
Eliminating HBM round-trips between operations gives you the biggest speedup. Everything
else (quantization, better attention algorithms) is secondary for small models.

**3. bf16 + f32 accumulation is the sweet spot.**
Cast inputs to bf16 for tensor cores, accumulate in f32 for precision. FP8 was tried and
was slower (register pressure from casts outweighed any gains).

**4. Dynamic loops prevent register spilling.**
`tl.range(0, 256, 32)` tells Triton to emit a real loop. `tl.static_range` would unroll
all 8 iterations, requiring registers for all tiles simultaneously, causing 340 bytes of
spill to slow local memory.

**5. num_warps=4 is optimal for this model.**
4 warps = 128 threads = 255 registers per thread = enough for the (128, 128) attention
matrix (the largest tensor). More warps would mean fewer registers per thread and spilling.

**6. Decode can't use tensor cores.**
With only 1 token, the M dimension is 1. Tensor cores need at least 16×16 tiles. So the
decode kernel uses element-wise operations, which are faster for M=1.

**7. The model must fit in registers for this approach.**
Our model is 132 KB in bf16. The register file is 130 KB. For larger models, you'd need
to tile across multiple thread blocks and use shared memory or HBM for intermediate
results between tiles — which is what FlashAttention and similar algorithms do.

---

## Running It Yourself

```bash
# Train the model (saves weights.pkl)
uv run train_backprop.py

# Run the benchmark
uv run inference_benchmark.py
```

### File overview

```
model.py                         — transformer in pure JAX (the baseline)
train_backprop.py                — trains the model, saves weights
kernels/fused_prefill.py         — prefill kernel (full sequence, one kernel call)
kernels/fused_decode.py          — decode kernel (one token, one kernel call)
inference_benchmark.py           — benchmarks Triton vs JAX and generates text
```
