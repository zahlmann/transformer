"""
Multi-block prefill kernels for d_model >= 128.

With d_model=128, h is (128,128) f32 = 64KB — can't fit in registers alongside
attention intermediates. Solution: tile the sequence dimension into blocks.

BLOCK_SEQ is chosen based on d_model to fit in the 127KB register file:
  d_model=128: BLOCK_SEQ=32 → h_block (32,128) = 16KB, peak ~52KB
  d_model=256: BLOCK_SEQ=16 → h_block (16,256) = 16KB, peak ~116KB

Three Triton kernels per layer (grid=num_blocks, each block handles BLOCK_SEQ rows):
  1. _proj_kernel:  LN1 + Q/K/V projections → writes Q,K,V to HBM
  2. _attn_kernel:  causal attention + O projection + residual → updates h
  3. _ffn_kernel:   LN2 + FFN + residual → updates h

Plus _output_kernel for final LN + tiled output projection.
"""

import triton
import triton.language as tl
import jax
import jax.numpy as jnp
import jax_triton as jt

BLOCK_K    = tl.constexpr(32)
VOCAB_TILE = tl.constexpr(128)


@triton.jit
def _proj_kernel(
    h_ptr,
    ln_scale_ptr, ln_bias_ptr,
    wq_ptr, wk_ptr, wv_ptr,
    q_buf_ptr, k_cache_ptr, v_cache_ptr,
    D_MODEL: tl.constexpr,
    D_HEAD: tl.constexpr,
    N_HEADS: tl.constexpr,
    SEQ: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    """LN1 + Q/K/V projections for a block of rows."""
    bid = tl.program_id(0)
    rows = bid * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    d = tl.arange(0, D_MODEL)
    dh = tl.arange(0, D_HEAD)

    # Load h block: (32, 128) f32 = 16KB
    h = tl.load(h_ptr + rows[:, None] * D_MODEL + d[None, :]).to(tl.float32)

    # Layer Norm
    ln_s = tl.load(ln_scale_ptr + d).to(tl.float32)
    ln_b = tl.load(ln_bias_ptr + d).to(tl.float32)
    mean = tl.sum(h, axis=1)[:, None] / D_MODEL
    hc = h - mean
    h_norm = ln_s[None, :] * hc * tl.math.rsqrt(tl.sum(hc * hc, axis=1)[:, None] / D_MODEL + 1e-5) + ln_b[None, :]

    # Q/K/V projections per head
    for head in tl.range(N_HEADS):
        hd = head * D_HEAD + dh
        head_off = head * SEQ * D_HEAD

        wq = tl.load(wq_ptr + d[:, None] * D_MODEL + hd[None, :]).to(tl.bfloat16)
        Q = tl.dot(h_norm.to(tl.bfloat16), wq).to(tl.float32)
        tl.store(q_buf_ptr + head_off + rows[:, None] * D_HEAD + dh[None, :], Q)

        wk = tl.load(wk_ptr + d[:, None] * D_MODEL + hd[None, :]).to(tl.bfloat16)
        K = tl.dot(h_norm.to(tl.bfloat16), wk).to(tl.float32)
        tl.store(k_cache_ptr + head_off + rows[:, None] * D_HEAD + dh[None, :], K.to(tl.bfloat16))

        wv = tl.load(wv_ptr + d[:, None] * D_MODEL + hd[None, :]).to(tl.bfloat16)
        V = tl.dot(h_norm.to(tl.bfloat16), wv).to(tl.float32)
        tl.store(v_cache_ptr + head_off + rows[:, None] * D_HEAD + dh[None, :], V.to(tl.bfloat16))


@triton.jit
def _attn_kernel(
    h_ptr,
    q_buf_ptr, k_cache_ptr, v_cache_ptr,
    wo_ptr,
    h_out_ptr,
    D_MODEL: tl.constexpr,
    D_HEAD: tl.constexpr,
    N_HEADS: tl.constexpr,
    SEQ: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    """Causal attention + O projection + residual for a block of rows."""
    bid = tl.program_id(0)
    rows = bid * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    all_pos = tl.arange(0, SEQ)
    d = tl.arange(0, D_MODEL)
    dh = tl.arange(0, D_HEAD)

    scale = 0.17677669529663689  # 1/sqrt(32), for D_HEAD=32

    attn_accum = tl.zeros((BLOCK_SEQ, D_MODEL), dtype=tl.float32)  # 16KB

    for head in tl.range(N_HEADS):
        hd = head * D_HEAD + dh
        head_off = head * SEQ * D_HEAD

        # Load Q for this block: (32, 32) = 4KB
        Q = tl.load(q_buf_ptr + head_off + rows[:, None] * D_HEAD + dh[None, :]).to(tl.float32)

        # Load K, V for ALL positions: (128, 32) = 16KB each
        K = tl.load(k_cache_ptr + head_off + all_pos[:, None] * D_HEAD + dh[None, :]).to(tl.float32)
        V = tl.load(v_cache_ptr + head_off + all_pos[:, None] * D_HEAD + dh[None, :]).to(tl.float32)

        # Causal attention: scores (32, 128) = 16KB
        scores = tl.dot(Q.to(tl.bfloat16), tl.trans(K.to(tl.bfloat16))).to(tl.float32) * scale
        mask = rows[:, None] >= all_pos[None, :]
        scores = tl.where(mask, scores, -1e9)
        exp_s = tl.exp(scores - tl.max(scores, axis=1)[:, None])
        attn = exp_s / tl.sum(exp_s, axis=1)[:, None]

        # Attention output: (32, 32) = 4KB
        attn_out = tl.dot(attn.to(tl.bfloat16), V.to(tl.bfloat16)).to(tl.float32)

        # O projection: (32,32) @ (32,128) → (32,128) accumulated
        wo = tl.load(wo_ptr + hd[:, None] * D_MODEL + d[None, :]).to(tl.bfloat16)
        attn_accum += tl.dot(attn_out.to(tl.bfloat16), wo).to(tl.float32)

    # Residual connection
    h = tl.load(h_ptr + rows[:, None] * D_MODEL + d[None, :]).to(tl.float32)
    h = h + attn_accum
    tl.store(h_out_ptr + rows[:, None] * D_MODEL + d[None, :], h)


@triton.jit
def _ffn_kernel(
    h_ptr,
    ln_scale_ptr, ln_bias_ptr,
    ffn_up_ptr, ffn_up_bias_ptr,
    ffn_down_ptr, ffn_down_bias_ptr,
    h_out_ptr,
    D_MODEL: tl.constexpr,
    D_FF: tl.constexpr,
    SEQ: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    """LN2 + FFN + residual for a block of rows."""
    bid = tl.program_id(0)
    rows = bid * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    d = tl.arange(0, D_MODEL)

    # Load h block: (32, 128) = 16KB
    h = tl.load(h_ptr + rows[:, None] * D_MODEL + d[None, :]).to(tl.float32)

    # Layer Norm
    ln_s = tl.load(ln_scale_ptr + d).to(tl.float32)
    ln_b = tl.load(ln_bias_ptr + d).to(tl.float32)
    mean = tl.sum(h, axis=1)[:, None] / D_MODEL
    hc = h - mean
    h_norm = ln_s[None, :] * hc * tl.math.rsqrt(tl.sum(hc * hc, axis=1)[:, None] / D_MODEL + 1e-5) + ln_b[None, :]

    # FFN (tiled over D_FF)
    ffn_out = tl.zeros((BLOCK_SEQ, D_MODEL), dtype=tl.float32)  # 16KB
    for k in tl.range(0, D_FF, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)
        up = tl.dot(h_norm.to(tl.bfloat16), tl.load(ffn_up_ptr + d[:, None] * D_FF + kk[None, :]).to(tl.bfloat16)).to(tl.float32)
        up += tl.load(ffn_up_bias_ptr + kk).to(tl.float32)[None, :]
        act = up * tl.sigmoid(1.702 * up)  # GELU approximation
        ffn_out += tl.dot(act.to(tl.bfloat16), tl.load(ffn_down_ptr + kk[:, None] * D_MODEL + d[None, :]).to(tl.bfloat16)).to(tl.float32)

    # Residual
    h = h + ffn_out + tl.load(ffn_down_bias_ptr + d).to(tl.float32)[None, :]
    tl.store(h_out_ptr + rows[:, None] * D_MODEL + d[None, :], h)


@triton.jit
def _output_kernel(
    h_ptr,
    ln_scale_ptr, ln_bias_ptr,
    output_proj_ptr,
    logits_ptr,
    D_MODEL: tl.constexpr,
    SEQ: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    VOCAB_PAD: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    """Final LN + tiled output projection for a block of rows."""
    bid = tl.program_id(0)
    rows = bid * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    d = tl.arange(0, D_MODEL)

    h = tl.load(h_ptr + rows[:, None] * D_MODEL + d[None, :]).to(tl.float32)

    ln_s = tl.load(ln_scale_ptr + d).to(tl.float32)
    ln_b = tl.load(ln_bias_ptr + d).to(tl.float32)
    mean = tl.sum(h, axis=1)[:, None] / D_MODEL
    hc = h - mean
    h_final = ln_s[None, :] * hc * tl.math.rsqrt(tl.sum(hc * hc, axis=1)[:, None] / D_MODEL + 1e-5) + ln_b[None, :]

    for v_start in tl.range(0, VOCAB_PAD, VOCAB_TILE):
        vv = v_start + tl.arange(0, VOCAB_TILE)
        out_w = tl.load(output_proj_ptr + d[:, None] * VOCAB_PAD + vv[None, :]).to(tl.bfloat16)
        tile_logits = tl.dot(h_final.to(tl.bfloat16), out_w).to(tl.float32)
        tile_logits = tl.where(vv[None, :] < VOCAB_SIZE, tile_logits, -1e9)
        tl.store(logits_ptr + rows[:, None] * VOCAB_PAD + vv[None, :], tile_logits)


# ──────────────────────────────────────────────────────────────────────
# Python orchestrator
# ──────────────────────────────────────────────────────────────────────

def block_prefill(params, config, x, vocab_size):
    """Multi-block prefill for d_model >= 128.

    Args:
        params: weight dict from model.init_transformer
        config: model config dict
        x: (seq_len,) int32 token IDs
        vocab_size: actual vocabulary size

    Returns:
        logits: (seq_len, vocab_size) float32
        k_caches: list of (n_heads, seq_len, d_head) bf16 per layer
        v_caches: list of (n_heads, seq_len, d_head) bf16 per layer
    """
    seq_len = x.shape[0]
    d_model = config["d_model"]
    d_head = config["d_head"]
    n_heads = config["n_heads"]
    n_layers = config["n_layers"]
    d_ff = 4 * d_model
    block_seq = 16 if d_model >= 256 else 32
    num_blocks = seq_len // block_seq
    vocab_pad = ((vocab_size + 127) // 128) * 128

    # Precompute bf16 weights once (avoid repeated .astype in loop)
    w = {k: v.astype(jnp.bfloat16) for k, v in params.items()}

    # Embedding (JAX — simple gather, not worth a kernel)
    h = (params["token_emb"][x] + params["pos_emb"][:seq_len]).astype(jnp.float32)

    all_k_caches = []
    all_v_caches = []

    for layer in range(n_layers):
        p = f"layer{layer}"

        # Kernel 1: LN + Q/K/V projections
        q_buf, k_cache, v_cache = jt.triton_call(
            h,
            w[f"{p}.ln1.scale"], w[f"{p}.ln1.bias"],
            w[f"{p}.attn.q"], w[f"{p}.attn.k"], w[f"{p}.attn.v"],
            kernel=_proj_kernel,
            out_shape=[
                jax.ShapeDtypeStruct((n_heads, seq_len, d_head), jnp.float32),
                jax.ShapeDtypeStruct((n_heads, seq_len, d_head), jnp.bfloat16),
                jax.ShapeDtypeStruct((n_heads, seq_len, d_head), jnp.bfloat16),
            ],
            grid=(num_blocks,),
            num_warps=4, num_stages=1,
            D_MODEL=d_model, D_HEAD=d_head, N_HEADS=n_heads, SEQ=seq_len,
            BLOCK_SEQ=block_seq,
        )
        all_k_caches.append(k_cache)
        all_v_caches.append(v_cache)

        # Kernel 2: Attention + O projection + residual
        (h,) = jt.triton_call(
            h, q_buf, k_cache, v_cache,
            w[f"{p}.attn.o"],
            kernel=_attn_kernel,
            out_shape=[
                jax.ShapeDtypeStruct((seq_len, d_model), jnp.float32),
            ],
            grid=(num_blocks,),
            num_warps=4, num_stages=1,
            D_MODEL=d_model, D_HEAD=d_head, N_HEADS=n_heads, SEQ=seq_len,
            BLOCK_SEQ=block_seq,
        )

        # Kernel 3: LN + FFN + residual
        (h,) = jt.triton_call(
            h,
            w[f"{p}.ln2.scale"], w[f"{p}.ln2.bias"],
            w[f"{p}.ffn.up"], w[f"{p}.ffn.up_bias"],
            w[f"{p}.ffn.down"], w[f"{p}.ffn.down_bias"],
            kernel=_ffn_kernel,
            out_shape=[
                jax.ShapeDtypeStruct((seq_len, d_model), jnp.float32),
            ],
            grid=(num_blocks,),
            num_warps=4, num_stages=1,
            D_MODEL=d_model, D_FF=d_ff, SEQ=seq_len,
            BLOCK_SEQ=block_seq,
        )

    # Kernel 4: Final LN + output projection
    pad_v = vocab_pad - vocab_size
    output_proj_padded = jnp.pad(params["output_proj"], [(0, 0), (0, pad_v)]).astype(jnp.bfloat16)

    (logits_pad,) = jt.triton_call(
        h,
        w["ln_final.scale"], w["ln_final.bias"],
        output_proj_padded,
        kernel=_output_kernel,
        out_shape=[
            jax.ShapeDtypeStruct((seq_len, vocab_pad), jnp.float32),
        ],
        grid=(num_blocks,),
        num_warps=4, num_stages=1,
        D_MODEL=d_model, SEQ=seq_len,
        VOCAB_SIZE=vocab_size, VOCAB_PAD=vocab_pad,
        BLOCK_SEQ=block_seq,
    )

    return logits_pad[:, :vocab_size], all_k_caches, all_v_caches
