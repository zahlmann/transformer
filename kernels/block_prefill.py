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
PROJ_TILE  = tl.constexpr(512)   # tile Q/K/V/O projections when D_BLOCK*D_HEAD > 64KB


@triton.jit
def _proj_kernel(
    h_ptr,
    ln_scale_ptr, ln_bias_ptr,
    wq_ptr, wk_ptr, wv_ptr,
    h_norm_buf_ptr, q_buf_ptr, k_cache_ptr, v_cache_ptr,
    D_MODEL: tl.constexpr,
    D_BLOCK: tl.constexpr,
    D_HEAD: tl.constexpr,
    N_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    D_KV: tl.constexpr,
    SEQ: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    TILE_PROJ: tl.constexpr,
):
    """LN1 + Q/K/V projections for a block of rows. Supports GQA + non-power-of-2 D_MODEL.

    When D_BLOCK*D_HEAD*2 > 101KB (shared memory limit), projection weights are tiled
    along D_BLOCK with TILE_PROJ. h_norm is stored to h_norm_buf and reloaded per tile.
    At (512, 64) bf16 = 64KB, each tile fits in shared memory.
    """
    bid = tl.program_id(0)
    rows = bid * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    d = tl.arange(0, D_BLOCK)
    d_mask = d < D_MODEL
    dh = tl.arange(0, D_HEAD)

    h = tl.load(h_ptr + rows[:, None] * D_MODEL + d[None, :], mask=d_mask[None, :], other=0.0).to(tl.float32)

    # Layer Norm
    ln_s = tl.load(ln_scale_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    ln_b = tl.load(ln_bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    mean = tl.sum(h, axis=1)[:, None] / D_MODEL
    hc = tl.where(d_mask[None, :], h - mean, 0.0)
    h_norm = tl.where(d_mask[None, :],
                      ln_s[None, :] * hc * tl.math.rsqrt(tl.sum(hc * hc, axis=1)[:, None] / D_MODEL + 1e-5) + ln_b[None, :],
                      0.0)

    # Store h_norm to buffer so we can reload per tile for projections
    tl.store(h_norm_buf_ptr + rows[:, None] * D_MODEL + d[None, :],
             h_norm.to(tl.bfloat16), mask=d_mask[None, :])

    # Q projections: N_HEADS heads (tiled along D_BLOCK)
    for head in tl.range(N_HEADS):
        hd = head * D_HEAD + dh
        head_off = head * SEQ * D_HEAD
        Q = tl.zeros((BLOCK_SEQ, D_HEAD), dtype=tl.float32)
        for dd in tl.static_range(0, D_BLOCK, TILE_PROJ):
            dt = dd + tl.arange(0, TILE_PROJ)
            dt_mask = dt < D_MODEL
            h_tile = tl.load(h_norm_buf_ptr + rows[:, None] * D_MODEL + dt[None, :],
                             mask=dt_mask[None, :], other=0.0).to(tl.bfloat16)
            wq_tile = tl.load(wq_ptr + dt[:, None] * D_MODEL + hd[None, :],
                              mask=dt_mask[:, None], other=0.0).to(tl.bfloat16)
            Q += tl.dot(h_tile, wq_tile).to(tl.float32)
        tl.store(q_buf_ptr + head_off + rows[:, None] * D_HEAD + dh[None, :], Q)

    # K/V projections: N_KV_HEADS heads (tiled along D_BLOCK)
    for kv_head in tl.range(N_KV_HEADS):
        kv_hd = kv_head * D_HEAD + dh
        kv_head_off = kv_head * SEQ * D_HEAD

        K = tl.zeros((BLOCK_SEQ, D_HEAD), dtype=tl.float32)
        V = tl.zeros((BLOCK_SEQ, D_HEAD), dtype=tl.float32)
        for dd in tl.static_range(0, D_BLOCK, TILE_PROJ):
            dt = dd + tl.arange(0, TILE_PROJ)
            dt_mask = dt < D_MODEL
            h_tile = tl.load(h_norm_buf_ptr + rows[:, None] * D_MODEL + dt[None, :],
                             mask=dt_mask[None, :], other=0.0).to(tl.bfloat16)
            wk_tile = tl.load(wk_ptr + dt[:, None] * D_KV + kv_hd[None, :],
                              mask=dt_mask[:, None], other=0.0).to(tl.bfloat16)
            K += tl.dot(h_tile, wk_tile).to(tl.float32)
            wv_tile = tl.load(wv_ptr + dt[:, None] * D_KV + kv_hd[None, :],
                              mask=dt_mask[:, None], other=0.0).to(tl.bfloat16)
            V += tl.dot(h_tile, wv_tile).to(tl.float32)
        tl.store(k_cache_ptr + kv_head_off + rows[:, None] * D_HEAD + dh[None, :], K.to(tl.bfloat16))
        tl.store(v_cache_ptr + kv_head_off + rows[:, None] * D_HEAD + dh[None, :], V.to(tl.bfloat16))


@triton.jit
def _attn_kernel(
    h_ptr,
    q_buf_ptr, k_cache_ptr, v_cache_ptr,
    wo_ptr,
    attn_scratch_ptr,
    h_out_ptr,
    D_MODEL: tl.constexpr,
    D_BLOCK: tl.constexpr,
    D_HEAD: tl.constexpr,
    N_HEADS: tl.constexpr,
    GQA_GROUP: tl.constexpr,
    SEQ: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    TILE_PROJ: tl.constexpr,
):
    """Causal attention + O projection + residual. Supports GQA + D_BLOCK padding.

    O projection is tiled along D_BLOCK with TILE_PROJ to fit in shared memory.
    attn_out is stored to attn_scratch then reloaded per tile for the tl.dot.
    """
    bid = tl.program_id(0)
    rows = bid * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    all_pos = tl.arange(0, SEQ)
    dh = tl.arange(0, D_HEAD)

    scale = 0.17677669529663689  # 1/sqrt(32), for D_HEAD=32

    # Initialize h_out with residual h (in tiles to avoid D_BLOCK-wide loads alongside wo)
    for dd in tl.static_range(0, D_BLOCK, TILE_PROJ):
        dt = dd + tl.arange(0, TILE_PROJ)
        dt_mask = dt < D_MODEL
        h_tile = tl.load(h_ptr + rows[:, None] * D_MODEL + dt[None, :],
                         mask=dt_mask[None, :], other=0.0).to(tl.float32)
        tl.store(h_out_ptr + rows[:, None] * D_MODEL + dt[None, :], h_tile, mask=dt_mask[None, :])

    for head in tl.range(N_HEADS):
        hd = head * D_HEAD + dh
        head_off = head * SEQ * D_HEAD
        kv_head = head // GQA_GROUP
        kv_head_off = kv_head * SEQ * D_HEAD

        Q = tl.load(q_buf_ptr + head_off + rows[:, None] * D_HEAD + dh[None, :]).to(tl.float32)
        K = tl.load(k_cache_ptr + kv_head_off + all_pos[:, None] * D_HEAD + dh[None, :]).to(tl.float32)
        V = tl.load(v_cache_ptr + kv_head_off + all_pos[:, None] * D_HEAD + dh[None, :]).to(tl.float32)

        scores = tl.dot(Q.to(tl.bfloat16), tl.trans(K.to(tl.bfloat16))).to(tl.float32) * scale
        mask = rows[:, None] >= all_pos[None, :]
        scores = tl.where(mask, scores, -1e9)
        exp_s = tl.exp(scores - tl.max(scores, axis=1)[:, None])
        attn = exp_s / tl.sum(exp_s, axis=1)[:, None]

        attn_out = tl.dot(attn.to(tl.bfloat16), V.to(tl.bfloat16)).to(tl.float32)

        # Store attn_out to scratch for tiled O projection reload
        tl.store(attn_scratch_ptr + rows[:, None] * D_HEAD + dh[None, :],
                 attn_out.to(tl.bfloat16))

        # Tile O projection along D_BLOCK to fit wo in shared memory
        # At (TILE_PROJ, D_HEAD) bf16 = (512, 64) * 2 = 64KB, fits in 101KB smem
        for dd in tl.static_range(0, D_BLOCK, TILE_PROJ):
            dt = dd + tl.arange(0, TILE_PROJ)
            dt_mask = dt < D_MODEL
            ao = tl.load(attn_scratch_ptr + rows[:, None] * D_HEAD + dh[None, :]).to(tl.bfloat16)
            wo_tile = tl.load(wo_ptr + hd[:, None] * D_MODEL + dt[None, :],
                              mask=dt_mask[None, :], other=0.0).to(tl.bfloat16)
            o_tile = tl.dot(ao, wo_tile).to(tl.float32)
            prev = tl.load(h_out_ptr + rows[:, None] * D_MODEL + dt[None, :],
                           mask=dt_mask[None, :], other=0.0).to(tl.float32)
            tl.store(h_out_ptr + rows[:, None] * D_MODEL + dt[None, :],
                     prev + o_tile, mask=dt_mask[None, :])


@triton.jit
def _flash_attn_kernel(
    h_ptr,
    q_buf_ptr, k_cache_ptr, v_cache_ptr,
    wo_ptr,
    attn_scratch_ptr,
    h_out_ptr,
    D_MODEL: tl.constexpr,
    D_BLOCK: tl.constexpr,
    D_HEAD: tl.constexpr,
    N_HEADS: tl.constexpr,
    GQA_GROUP: tl.constexpr,
    SEQ: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    KV_TILE: tl.constexpr,
    TILE_PROJ: tl.constexpr,
):
    """FlashAttention: tiled KV with online softmax. Supports GQA + D_BLOCK padding.

    O projection is tiled along D_BLOCK with TILE_PROJ to fit in shared memory.
    attn_out is stored to attn_scratch then reloaded per tile for the tl.dot.
    """
    bid = tl.program_id(0)
    rows = bid * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    dh = tl.arange(0, D_HEAD)

    scale = 0.17677669529663689  # 1/sqrt(32), for D_HEAD=32

    # Initialize h_out with residual h (in tiles)
    for dd in tl.static_range(0, D_BLOCK, TILE_PROJ):
        dt = dd + tl.arange(0, TILE_PROJ)
        dt_mask = dt < D_MODEL
        h_tile = tl.load(h_ptr + rows[:, None] * D_MODEL + dt[None, :],
                         mask=dt_mask[None, :], other=0.0).to(tl.float32)
        tl.store(h_out_ptr + rows[:, None] * D_MODEL + dt[None, :], h_tile, mask=dt_mask[None, :])

    for head in tl.range(N_HEADS):
        hd = head * D_HEAD + dh
        head_off = head * SEQ * D_HEAD
        kv_head = head // GQA_GROUP
        kv_head_off = kv_head * SEQ * D_HEAD

        Q = tl.load(q_buf_ptr + head_off + rows[:, None] * D_HEAD + dh[None, :]).to(tl.float32)

        m_i = tl.full((BLOCK_SEQ,), value=-1e9, dtype=tl.float32)
        l_i = tl.zeros((BLOCK_SEQ,), dtype=tl.float32)
        o_i = tl.zeros((BLOCK_SEQ, D_HEAD), dtype=tl.float32)

        for kv_start in tl.range(0, SEQ, KV_TILE):
            kv_pos = kv_start + tl.arange(0, KV_TILE)

            K_tile = tl.load(k_cache_ptr + kv_head_off + kv_pos[:, None] * D_HEAD + dh[None, :]).to(tl.float32)
            V_tile = tl.load(v_cache_ptr + kv_head_off + kv_pos[:, None] * D_HEAD + dh[None, :]).to(tl.float32)

            s = tl.dot(Q.to(tl.bfloat16), tl.trans(K_tile.to(tl.bfloat16))).to(tl.float32) * scale

            causal_mask = rows[:, None] >= kv_pos[None, :]
            s = tl.where(causal_mask, s, -1e9)

            m_ij = tl.max(s, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new[:, None])

            l_i = l_i * alpha + tl.sum(p, axis=1)
            o_i = o_i * alpha[:, None] + tl.dot(p.to(tl.bfloat16), V_tile.to(tl.bfloat16)).to(tl.float32)
            m_i = m_new

        attn_out = o_i / l_i[:, None]

        # Store attn_out to scratch for tiled O projection reload
        tl.store(attn_scratch_ptr + rows[:, None] * D_HEAD + dh[None, :],
                 attn_out.to(tl.bfloat16))

        # Tile O projection along D_BLOCK to fit wo in shared memory
        for dd in tl.static_range(0, D_BLOCK, TILE_PROJ):
            dt = dd + tl.arange(0, TILE_PROJ)
            dt_mask = dt < D_MODEL
            ao = tl.load(attn_scratch_ptr + rows[:, None] * D_HEAD + dh[None, :]).to(tl.bfloat16)
            wo_tile = tl.load(wo_ptr + hd[:, None] * D_MODEL + dt[None, :],
                              mask=dt_mask[None, :], other=0.0).to(tl.bfloat16)
            o_tile = tl.dot(ao, wo_tile).to(tl.float32)
            prev = tl.load(h_out_ptr + rows[:, None] * D_MODEL + dt[None, :],
                           mask=dt_mask[None, :], other=0.0).to(tl.float32)
            tl.store(h_out_ptr + rows[:, None] * D_MODEL + dt[None, :],
                     prev + o_tile, mask=dt_mask[None, :])


@triton.jit
def _ffn_kernel(
    h_ptr,
    ln_scale_ptr, ln_bias_ptr,
    ffn_up_ptr, ffn_down_ptr,
    h_out_ptr,
    D_MODEL: tl.constexpr,
    D_BLOCK: tl.constexpr,
    D_FF: tl.constexpr,
    SEQ: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    """LN2 + FFN + residual for a block of rows. Supports D_BLOCK padding."""
    bid = tl.program_id(0)
    rows = bid * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    d = tl.arange(0, D_BLOCK)
    d_mask = d < D_MODEL

    h = tl.load(h_ptr + rows[:, None] * D_MODEL + d[None, :], mask=d_mask[None, :], other=0.0).to(tl.float32)

    # Layer Norm
    ln_s = tl.load(ln_scale_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    ln_b = tl.load(ln_bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    mean = tl.sum(h, axis=1)[:, None] / D_MODEL
    hc = tl.where(d_mask[None, :], h - mean, 0.0)
    h_norm = tl.where(d_mask[None, :],
                      ln_s[None, :] * hc * tl.math.rsqrt(tl.sum(hc * hc, axis=1)[:, None] / D_MODEL + 1e-5) + ln_b[None, :],
                      0.0)

    # FFN (tiled over D_FF)
    ffn_out = tl.zeros((BLOCK_SEQ, D_BLOCK), dtype=tl.float32)
    for k in tl.range(0, D_FF, BLOCK_K):
        kk = k + tl.arange(0, BLOCK_K)
        up = tl.dot(h_norm.to(tl.bfloat16),
                    tl.load(ffn_up_ptr + d[:, None] * D_FF + kk[None, :], mask=d_mask[:, None], other=0.0).to(tl.bfloat16)).to(tl.float32)
        act = up * tl.sigmoid(1.702 * up)
        ffn_out += tl.dot(act.to(tl.bfloat16),
                          tl.load(ffn_down_ptr + kk[:, None] * D_MODEL + d[None, :], mask=d_mask[None, :], other=0.0).to(tl.bfloat16)).to(tl.float32)

    # Residual
    h = h + ffn_out
    tl.store(h_out_ptr + rows[:, None] * D_MODEL + d[None, :], h, mask=d_mask[None, :])


@triton.jit
def _output_kernel(
    h_ptr,
    ln_scale_ptr, ln_bias_ptr,
    output_proj_ptr,
    logits_ptr,
    D_MODEL: tl.constexpr,
    D_BLOCK: tl.constexpr,
    SEQ: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    VOCAB_PAD: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    VTILE: tl.constexpr,
):
    """Final LN + tiled output projection. Supports D_BLOCK padding."""
    bid = tl.program_id(0)
    rows = bid * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    d = tl.arange(0, D_BLOCK)
    d_mask = d < D_MODEL

    h = tl.load(h_ptr + rows[:, None] * D_MODEL + d[None, :], mask=d_mask[None, :], other=0.0).to(tl.float32)

    ln_s = tl.load(ln_scale_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    ln_b = tl.load(ln_bias_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    mean = tl.sum(h, axis=1)[:, None] / D_MODEL
    hc = tl.where(d_mask[None, :], h - mean, 0.0)
    h_final = tl.where(d_mask[None, :],
                       ln_s[None, :] * hc * tl.math.rsqrt(tl.sum(hc * hc, axis=1)[:, None] / D_MODEL + 1e-5) + ln_b[None, :],
                       0.0)

    for v_start in tl.range(0, VOCAB_PAD, VTILE):
        vv = v_start + tl.arange(0, VTILE)
        out_w = tl.load(output_proj_ptr + d[:, None] * VOCAB_PAD + vv[None, :], mask=d_mask[:, None], other=0.0).to(tl.bfloat16)
        tile_logits = tl.dot(h_final.to(tl.bfloat16), out_w).to(tl.float32)
        tile_logits = tl.where(vv[None, :] < VOCAB_SIZE, tile_logits, -1e9)
        tl.store(logits_ptr + rows[:, None] * VOCAB_PAD + vv[None, :], tile_logits)


def _next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p


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
        k_caches: list of (n_kv_heads, seq_len, d_head) bf16 per layer
        v_caches: list of (n_kv_heads, seq_len, d_head) bf16 per layer
    """
    seq_len = x.shape[0]
    d_model = config["d_model"]
    d_head = config["d_head"]
    n_heads = config["n_heads"]
    n_kv_heads = config.get("n_kv_heads", n_heads)
    d_kv = n_kv_heads * d_head
    gqa_group = n_heads // n_kv_heads
    d_block = _next_power_of_2(d_model)
    n_layers = config["n_layers"]
    d_ff = 4 * d_model
    block_seq = 8 if d_model >= 512 else (16 if d_model >= 256 else 32)
    num_blocks = seq_len // block_seq
    vocab_pad = ((vocab_size + 127) // 128) * 128

    # PROJ_TILE for tiling Q/K/V/O projections when weight matrices exceed shared memory.
    # At D_BLOCK=1024, D_HEAD=64: (1024,64) bf16 = 128KB > 101KB smem limit.
    # PROJ_TILE=512: (512,64) bf16 = 64KB, fits.
    proj_tile = min(d_block, 512)

    # Precompute bf16 weights once (avoid repeated .astype in loop)
    w = {k: v.astype(jnp.bfloat16) for k, v in params.items()}

    # Embedding (JAX — simple gather, not worth a kernel)
    h = (params["token_emb"][x] + params["pos_emb"][:seq_len]).astype(jnp.float32)

    all_k_caches = []
    all_v_caches = []

    for layer in range(n_layers):
        p = f"layer{layer}"

        # Kernel 1: LN + Q/K/V projections (GQA: K/V have N_KV_HEADS)
        # h_norm_buf: scratch buffer for h_norm, reloaded per projection tile
        _h_norm_buf, q_buf, k_cache, v_cache = jt.triton_call(
            h,
            w[f"{p}.ln1.scale"], w[f"{p}.ln1.bias"],
            w[f"{p}.attn.q"], w[f"{p}.attn.k"], w[f"{p}.attn.v"],
            kernel=_proj_kernel,
            out_shape=[
                jax.ShapeDtypeStruct((seq_len, d_model), jnp.bfloat16),
                jax.ShapeDtypeStruct((n_heads, seq_len, d_head), jnp.float32),
                jax.ShapeDtypeStruct((n_kv_heads, seq_len, d_head), jnp.bfloat16),
                jax.ShapeDtypeStruct((n_kv_heads, seq_len, d_head), jnp.bfloat16),
            ],
            grid=(num_blocks,),
            num_warps=4, num_stages=1,
            D_MODEL=d_model, D_BLOCK=d_block, D_HEAD=d_head, N_HEADS=n_heads,
            N_KV_HEADS=n_kv_heads, D_KV=d_kv,
            SEQ=seq_len, BLOCK_SEQ=block_seq, TILE_PROJ=proj_tile,
        )
        all_k_caches.append(k_cache)
        all_v_caches.append(v_cache)

        # Kernel 2: Attention + O projection + residual
        # attn_scratch: scratch buffer for attn_out, reloaded per O projection tile
        use_flash = seq_len > 256
        if use_flash:
            kv_tile = 64
            _attn_scratch, h = jt.triton_call(
                h, q_buf, k_cache, v_cache,
                w[f"{p}.attn.o"],
                kernel=_flash_attn_kernel,
                out_shape=[
                    jax.ShapeDtypeStruct((seq_len, d_head), jnp.bfloat16),
                    jax.ShapeDtypeStruct((seq_len, d_model), jnp.float32),
                ],
                grid=(num_blocks,),
                num_warps=4, num_stages=1,
                D_MODEL=d_model, D_BLOCK=d_block, D_HEAD=d_head, N_HEADS=n_heads,
                GQA_GROUP=gqa_group,
                SEQ=seq_len, BLOCK_SEQ=block_seq, KV_TILE=kv_tile,
                TILE_PROJ=proj_tile,
            )
        else:
            _attn_scratch, h = jt.triton_call(
                h, q_buf, k_cache, v_cache,
                w[f"{p}.attn.o"],
                kernel=_attn_kernel,
                out_shape=[
                    jax.ShapeDtypeStruct((seq_len, d_head), jnp.bfloat16),
                    jax.ShapeDtypeStruct((seq_len, d_model), jnp.float32),
                ],
                grid=(num_blocks,),
                num_warps=4, num_stages=1,
                D_MODEL=d_model, D_BLOCK=d_block, D_HEAD=d_head, N_HEADS=n_heads,
                GQA_GROUP=gqa_group,
                SEQ=seq_len, BLOCK_SEQ=block_seq,
                TILE_PROJ=proj_tile,
            )

        # Kernel 3: LN + FFN + residual
        (h,) = jt.triton_call(
            h,
            w[f"{p}.ln2.scale"], w[f"{p}.ln2.bias"],
            w[f"{p}.ffn.up"], w[f"{p}.ffn.down"],
            kernel=_ffn_kernel,
            out_shape=[
                jax.ShapeDtypeStruct((seq_len, d_model), jnp.float32),
            ],
            grid=(num_blocks,),
            num_warps=4, num_stages=1,
            D_MODEL=d_model, D_BLOCK=d_block, D_FF=d_ff, SEQ=seq_len,
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
        D_MODEL=d_model, D_BLOCK=d_block, SEQ=seq_len,
        VOCAB_SIZE=vocab_size, VOCAB_PAD=vocab_pad,
        BLOCK_SEQ=block_seq,
        VTILE=32 if d_model >= 512 else 128,
    )

    return logits_pad[:, :vocab_size], all_k_caches, all_v_caches
