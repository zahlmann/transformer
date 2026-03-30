"""
Multi-SM fused N-layer decode kernel.

Key insight from profiling: the single-SM kernel (grid=1) uses 1 of 80 SMs on the
RTX 4080 Super, leaving 79 idle. This kernel uses grid=(N_HEADS,) to parallelize
attention across SMs, and also distributes FFN work across all blocks.

Architecture:
  grid = (N_HEADS,)  — each block handles one attention head

  Per layer:
    Phase 1 (all blocks): LN1 + QKV proj + attention + O proj for one head
      → write partial result to scratch buffer
      → atomic barrier
    Phase 2 (all blocks): reduce attention + residual + LN2 + FFN partition
      → each block handles D_FF/N_HEADS elements of intermediate dim
      → write partial result to scratch buffer
      → atomic barrier
    Phase 3 (all blocks): reduce FFN + residual → h ready for next layer

  Output: all blocks participate in tiled output projection

Barriers use atomic counters with release/acquire semantics for cross-SM sync.
With 16 blocks on 80 SMs, all blocks are guaranteed to run concurrently.
"""

import triton
import triton.language as tl
import jax
import jax.numpy as jnp
import jax_triton as jt

BLOCK_K      = tl.constexpr(32)
KV_TILE      = tl.constexpr(64)
OUTPUT_VTILE = tl.constexpr(32)


@triton.jit
def _multi_sm_decode(
    # Inputs (positional args in jt.triton_call, in order)
    token_emb_ptr, pos_emb_ptr,
    packed_w_ptr,
    lnf_s_ptr, lnf_b_ptr,
    output_proj_ptr,
    token_id_ptr, pos_ptr,
    kv_in_ptr,
    partial_ptr,      # (N_HEADS, D_MODEL) f32 — scratch for partial results
    barrier_ptr,      # (2 * N_LAYERS,) int32 — atomic barrier counters
    # Outputs (from out_shape)
    logits_ptr,
    kv_out_ptr,
    # Config
    D_MODEL: tl.constexpr,
    D_HEAD: tl.constexpr,
    D_FF: tl.constexpr,
    N_HEADS: tl.constexpr,
    N_LAYERS: tl.constexpr,
    MAX_SEQ: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    VOCAB_PAD: tl.constexpr,
    FF_PER_BLOCK: tl.constexpr,  # D_FF // N_HEADS
):
    pid = tl.program_id(0)  # 0..N_HEADS-1
    d = tl.arange(0, D_MODEL)

    token_id = tl.load(token_id_ptr)
    pos = tl.load(pos_ptr)

    # ── Embedding (all blocks compute redundantly — 2KB from L2) ──
    h = (tl.load(token_emb_ptr + token_id * D_MODEL + d).to(tl.float32)
       + tl.load(pos_emb_ptr + pos * D_MODEL + d).to(tl.float32))

    # Per-layer weight size in elements (bf16)
    LAYER_W_SIZE: tl.constexpr = (
        D_MODEL + D_MODEL +                      # ln1 scale, bias
        4 * D_MODEL * D_MODEL +                   # wq, wk, wv, wo
        D_MODEL + D_MODEL +                       # ln2 scale, bias
        D_MODEL * D_FF + D_FF +                   # ffn_up, ffn_up_bias
        D_FF * D_MODEL + D_MODEL                  # ffn_down, ffn_down_bias
    )

    # Per-layer KV cache size in elements (bf16)
    LAYER_KV_SIZE: tl.constexpr = 2 * N_HEADS * MAX_SEQ * D_HEAD

    scale = 0.17677669529663689  # 1/sqrt(32)
    dh = tl.arange(0, D_HEAD)

    # Use tl.range (not static_range) to avoid unrolling 8 massive layer bodies
    for layer in tl.range(N_LAYERS):
        w_base = layer * LAYER_W_SIZE
        kv_base = layer * LAYER_KV_SIZE
        kc_base = kv_base
        vc_base = kv_base + N_HEADS * MAX_SEQ * D_HEAD

        # Weight offsets within this layer
        off = w_base
        ln1_s_off = off;          off += D_MODEL
        ln1_b_off = off;          off += D_MODEL
        wq_off = off;             off += D_MODEL * D_MODEL
        wk_off = off;             off += D_MODEL * D_MODEL
        wv_off = off;             off += D_MODEL * D_MODEL
        wo_off = off;             off += D_MODEL * D_MODEL
        ln2_s_off = off;          off += D_MODEL
        ln2_b_off = off;          off += D_MODEL
        up_off = off;             off += D_MODEL * D_FF
        up_b_off = off;           off += D_FF
        down_off = off;           off += D_FF * D_MODEL
        down_b_off = off

        # ════════════════════════════════════════════
        # PHASE 1: LN1 + Attention (each block = 1 head)
        # ════════════════════════════════════════════

        # LN1 (all blocks compute redundantly — cheap, just reductions)
        ln_s = tl.load(packed_w_ptr + ln1_s_off + d).to(tl.float32)
        ln_b = tl.load(packed_w_ptr + ln1_b_off + d).to(tl.float32)
        mean = tl.sum(h) / D_MODEL
        hc = h - mean
        h_norm = ln_s * hc * tl.math.rsqrt(tl.sum(hc * hc) / D_MODEL + 1e-5) + ln_b

        # This block's head
        head = pid
        hd = head * D_HEAD + dh
        cache_off = head * MAX_SEQ * D_HEAD

        h_norm_2d = h_norm[None, :].to(tl.bfloat16)

        # QKV projections via tl.dot
        wq = tl.load(packed_w_ptr + wq_off + d[:, None] * D_MODEL + hd[None, :]).to(tl.bfloat16)
        Q = tl.dot(h_norm_2d, wq).to(tl.float32).sum(axis=0)
        wk = tl.load(packed_w_ptr + wk_off + d[:, None] * D_MODEL + hd[None, :]).to(tl.bfloat16)
        K_new = tl.dot(h_norm_2d, wk).to(tl.float32).sum(axis=0)
        wv = tl.load(packed_w_ptr + wv_off + d[:, None] * D_MODEL + hd[None, :]).to(tl.bfloat16)
        V_new = tl.dot(h_norm_2d, wv).to(tl.float32).sum(axis=0)

        # Write new KV to cache
        tl.store(kv_out_ptr + kc_base + cache_off + pos * D_HEAD + dh, K_new.to(tl.bfloat16))
        tl.store(kv_out_ptr + vc_base + cache_off + pos * D_HEAD + dh, V_new.to(tl.bfloat16))

        # Online softmax attention over tiled KV cache
        m_i = tl.full((1,), value=-1e9, dtype=tl.float32)
        l_i = tl.zeros((1,), dtype=tl.float32)
        o_i = tl.zeros((D_HEAD,), dtype=tl.float32)

        for t in tl.range(0, MAX_SEQ, KV_TILE):
            tile_pos = t + tl.arange(0, KV_TILE)
            tile_mask = tile_pos <= pos

            K_tile = tl.load(kv_in_ptr + kc_base + cache_off + tile_pos[:, None] * D_HEAD + dh[None, :],
                            mask=tile_mask[:, None], other=0.0).to(tl.float32)
            K_tile = tl.where(tile_pos[:, None] == pos, K_new[None, :], K_tile)
            tl.store(kv_out_ptr + kc_base + cache_off + tile_pos[:, None] * D_HEAD + dh[None, :],
                    K_tile.to(tl.bfloat16), mask=tile_mask[:, None])

            V_tile = tl.load(kv_in_ptr + vc_base + cache_off + tile_pos[:, None] * D_HEAD + dh[None, :],
                            mask=tile_mask[:, None], other=0.0).to(tl.float32)
            V_tile = tl.where(tile_pos[:, None] == pos, V_new[None, :], V_tile)
            tl.store(kv_out_ptr + vc_base + cache_off + tile_pos[:, None] * D_HEAD + dh[None, :],
                    V_tile.to(tl.bfloat16), mask=tile_mask[:, None])

            s = tl.sum(Q[None, :] * K_tile, axis=1) * scale
            s = tl.where(tile_mask, s, -1e9)

            m_ij = tl.max(s)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(s - m_new)
            l_i = l_i * alpha + tl.sum(p)
            o_i = o_i * alpha + tl.sum(p[:, None] * V_tile, axis=0)
            m_i = m_new

        attn_out = o_i / l_i

        # O projection: (1, D_HEAD) @ (D_HEAD, D_MODEL) → (D_MODEL,)
        wo = tl.load(packed_w_ptr + wo_off + hd[:, None] * D_MODEL + d[None, :]).to(tl.bfloat16)
        o_proj = tl.dot(attn_out[None, :].to(tl.bfloat16), wo).to(tl.float32).sum(axis=0)

        # Write this head's O-projection to partial buffer
        tl.store(partial_ptr + pid * D_MODEL + d, o_proj)

        # ── Barrier: wait for all heads to finish attention ──
        b0 = layer * 2
        tl.atomic_add(barrier_ptr + b0, 1, sem='release', scope='gpu')
        while tl.atomic_add(barrier_ptr + b0, 0, sem='acquire', scope='gpu') < N_HEADS:
            pass

        # ════════════════════════════════════════════
        # PHASE 2: Reduce attention + LN2 + FFN (all blocks)
        # ════════════════════════════════════════════

        # Reduce attention partials (all blocks read 16×512 f32 = 32KB from L2)
        attn_total = tl.zeros((D_MODEL,), dtype=tl.float32)
        for i in tl.static_range(N_HEADS):
            attn_total += tl.load(partial_ptr + i * D_MODEL + d)
        h = h + attn_total

        # LN2 (all blocks compute redundantly)
        ln_s = tl.load(packed_w_ptr + ln2_s_off + d).to(tl.float32)
        ln_b = tl.load(packed_w_ptr + ln2_b_off + d).to(tl.float32)
        mean = tl.sum(h) / D_MODEL
        hc = h - mean
        h_norm = ln_s * hc * tl.math.rsqrt(tl.sum(hc * hc) / D_MODEL + 1e-5) + ln_b
        h_norm_2d = h_norm[None, :].to(tl.bfloat16)

        # FFN: each block handles FF_PER_BLOCK elements of D_FF
        ff_start = pid * FF_PER_BLOCK
        ffn_partial = tl.zeros((D_MODEL,), dtype=tl.float32)
        for k in tl.range(0, FF_PER_BLOCK, BLOCK_K):
            kk = ff_start + k + tl.arange(0, BLOCK_K)
            up_w = tl.load(packed_w_ptr + up_off + d[:, None] * D_FF + kk[None, :]).to(tl.bfloat16)
            up = tl.dot(h_norm_2d, up_w).to(tl.float32).sum(axis=0)
            up += tl.load(packed_w_ptr + up_b_off + kk).to(tl.float32)
            act = up * tl.sigmoid(1.702 * up)
            down_w = tl.load(packed_w_ptr + down_off + kk[:, None] * D_MODEL + d[None, :]).to(tl.bfloat16)
            ffn_partial += tl.dot(act[None, :].to(tl.bfloat16), down_w).to(tl.float32).sum(axis=0)

        # Write FFN partial to scratch
        tl.store(partial_ptr + pid * D_MODEL + d, ffn_partial)

        # ── Barrier: wait for all blocks to finish FFN ──
        b1 = layer * 2 + 1
        tl.atomic_add(barrier_ptr + b1, 1, sem='release', scope='gpu')
        while tl.atomic_add(barrier_ptr + b1, 0, sem='acquire', scope='gpu') < N_HEADS:
            pass

        # ════════════════════════════════════════════
        # PHASE 3: Reduce FFN + residual (all blocks)
        # ════════════════════════════════════════════

        ffn_total = tl.zeros((D_MODEL,), dtype=tl.float32)
        for i in tl.static_range(N_HEADS):
            ffn_total += tl.load(partial_ptr + i * D_MODEL + d)
        h = h + ffn_total + tl.load(packed_w_ptr + down_b_off + d).to(tl.float32)

    # ════════════════════════════════════════════
    # OUTPUT: tiled output projection (all blocks participate)
    # ════════════════════════════════════════════

    # Final LN (all blocks compute redundantly)
    ln_s = tl.load(lnf_s_ptr + d).to(tl.float32)
    ln_b = tl.load(lnf_b_ptr + d).to(tl.float32)
    mean = tl.sum(h) / D_MODEL
    hc = h - mean
    h_final = ln_s * hc * tl.math.rsqrt(tl.sum(hc * hc) / D_MODEL + 1e-5) + ln_b
    h_final_2d = h_final[None, :].to(tl.bfloat16)

    # Each block handles VOCAB_PAD / N_HEADS tiles of the output
    TILES_PER_BLOCK: tl.constexpr = VOCAB_PAD // (OUTPUT_VTILE * N_HEADS)
    for tile_idx in tl.range(0, TILES_PER_BLOCK):
        v_start = (pid * TILES_PER_BLOCK + tile_idx) * OUTPUT_VTILE
        vv = v_start + tl.arange(0, OUTPUT_VTILE)
        out_w = tl.load(output_proj_ptr + d[:, None] * VOCAB_PAD + vv[None, :]).to(tl.bfloat16)
        tile_logits = tl.dot(h_final_2d, out_w).to(tl.float32).sum(axis=0)
        tile_logits = tl.where(vv < VOCAB_SIZE, tile_logits, -1e9)
        tl.store(logits_ptr + vv, tile_logits)


# ──────────────────────────────────────────────────────────────────────
# Python wrapper
# ──────────────────────────────────────────────────────────────────────

def multi_sm_decode_nlayer(w, config, token_id, pos, kv_packed, vocab_size,
                           workspace=None):
    """Multi-SM fused N-layer decode: one kernel call per token, grid=(N_HEADS,).

    Same interface as fused_decode_nlayer but uses multiple SMs.

    Args:
        w: precomputed weights from prepare_decode_weights_nlayer()
        config: model config
        token_id: scalar token
        pos: scalar position
        kv_packed: flat bf16 buffer
        vocab_size: actual vocabulary size
        workspace: optional pre-allocated workspace dict (avoids per-step alloc)

    Returns: logits (vocab_size,), kv_packed (flat bf16 buffer)
    """
    d_model = config["d_model"]
    d_head = config["d_head"]
    n_heads = config["n_heads"]
    n_layers = config["n_layers"]
    d_ff = 4 * d_model
    max_seq = config["context_len"]
    vocab_pad = w["vocab_pad"]
    total_kv_size = n_layers * 2 * n_heads * max_seq * d_head
    ff_per_block = d_ff // n_heads

    # Workspace: partial results buffer + barrier counters
    # Must be zeroed before each call
    if workspace is None:
        partial = jnp.zeros((n_heads * d_model,), dtype=jnp.float32)
        barriers = jnp.zeros((2 * n_layers,), dtype=jnp.int32)
    else:
        partial = workspace["partial"]
        barriers = workspace["barriers"]

    logits_pad, kv_out = jt.triton_call(
        w["token_emb"], w["pos_emb"],
        w["packed_w"],
        w["lnf_s"], w["lnf_b"],
        w["output_proj_padded"],
        jnp.int32(token_id), jnp.int32(pos),
        kv_packed,
        partial,
        barriers,
        kernel=_multi_sm_decode,
        out_shape=[
            jax.ShapeDtypeStruct((vocab_pad,), jnp.float32),
            jax.ShapeDtypeStruct((total_kv_size,), jnp.bfloat16),
        ],
        grid=(n_heads,),
        num_warps=4, num_stages=1,
        D_MODEL=d_model, D_HEAD=d_head, D_FF=d_ff,
        N_HEADS=n_heads, N_LAYERS=n_layers, MAX_SEQ=max_seq,
        VOCAB_SIZE=vocab_size, VOCAB_PAD=vocab_pad,
        FF_PER_BLOCK=ff_per_block,
    )

    return logits_pad[:vocab_size], kv_out


def allocate_workspace(config):
    """Pre-allocate workspace buffers for multi_sm_decode_nlayer."""
    d_model = config["d_model"]
    n_heads = config["n_heads"]
    n_layers = config["n_layers"]
    return {
        "partial": jnp.zeros((n_heads * d_model,), dtype=jnp.float32),
        "barriers": jnp.zeros((2 * n_layers,), dtype=jnp.int32),
    }
