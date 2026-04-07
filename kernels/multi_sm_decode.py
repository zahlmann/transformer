"""
Multi-SM fused N-layer decode kernel with KV-split parallelism.

Phase C architecture: RMSNorm, RoPE, SwiGLU, no biases, tied embeddings, GQA.

Architecture:
  grid = (N_HEADS * KV_SPLITS,)  — multiple blocks per attention head

  Per layer:
    Phase 1 (all blocks): RMSNorm1 + QKV proj + RoPE + attention (split KV) + O proj
      → write O-proj + online softmax state (m, l) to scratch buffer
      → split barrier (counter + done flag on separate cache lines)
    Phase 2 (all blocks): merge KV splits + reduce attention + residual + RMSNorm2 + SwiGLU FFN
      → write FFN partial to scratch buffer → barrier
    Phase 3 (all blocks): reduce FFN + residual → h ready for next layer

  Output: all blocks participate in tiled output projection + in-kernel argmax.

Supports non-power-of-2 D_MODEL via D_BLOCK padding: tl.arange uses D_BLOCK
(next power of 2) with d_mask = d < D_MODEL for all loads/stores.
"""

import triton
import triton.language as tl
import jax
import jax.numpy as jnp
import jax_triton as jt

BLOCK_K      = tl.constexpr(16)
KV_TILE      = tl.constexpr(64)
OUTPUT_VTILE = tl.constexpr(32)
PROJ_TILE    = tl.constexpr(512)


@triton.jit
def _multi_sm_decode(
    # Inputs
    token_emb_ptr,
    packed_w_ptr,
    lnf_s_ptr,
    output_proj_ptr,
    cos_ptr, sin_ptr,
    token_id_ptr, pos_ptr,
    kv_in_ptr,
    workspace_ptr,
    # Outputs
    logits_ptr,
    kv_out_ptr,
    next_token_ptr,
    # Config
    D_MODEL: tl.constexpr,
    D_BLOCK: tl.constexpr,
    D_HEAD: tl.constexpr,
    D_FF: tl.constexpr,
    N_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    D_KV: tl.constexpr,
    N_LAYERS: tl.constexpr,
    MAX_SEQ: tl.constexpr,
    KV_SPLITS: tl.constexpr,
    TOTAL_BLOCKS: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    VOCAB_PAD: tl.constexpr,
    FF_PER_BLOCK: tl.constexpr,
    FFN_PARTIAL_OFF: tl.constexpr,
    ATTN_ML_OFF: tl.constexpr,
    BARRIER_OFF: tl.constexpr,
    DONE_OFF: tl.constexpr,
    ARGMAX_OFF: tl.constexpr,
):
    pid = tl.program_id(0)
    head_id = pid // KV_SPLITS
    kv_split = pid % KV_SPLITS
    d = tl.arange(0, D_BLOCK)
    d_mask = d < D_MODEL

    token_id = tl.load(token_id_ptr)
    pos = tl.load(pos_ptr)

    attn_partial_ptr = workspace_ptr                     # attention O-proj results
    ffn_partial_ptr = workspace_ptr + FFN_PARTIAL_OFF    # FFN partial results (separate to avoid race)
    attn_ml_ptr = workspace_ptr + ATTN_ML_OFF
    barrier_ptr = workspace_ptr + BARRIER_OFF
    done_ptr = workspace_ptr + DONE_OFF
    argmax_ptr = workspace_ptr + ARGMAX_OFF

    # ── Embedding (no positional — RoPE applied in attention) ──
    h = tl.load(token_emb_ptr + token_id * D_MODEL + d, mask=d_mask, other=0.0).to(tl.float32)

    LAYER_W_SIZE: tl.constexpr = (
        D_MODEL +                                                                    # ln1 scale
        D_MODEL * D_MODEL + D_MODEL * D_KV + D_MODEL * D_KV + D_MODEL * D_MODEL +  # qkvo
        D_MODEL +                                                                    # ln2 scale
        D_MODEL * D_FF + D_MODEL * D_FF + D_FF * D_MODEL                            # gate, up, down
    )
    LAYER_KV_SIZE: tl.constexpr = 2 * N_KV_HEADS * MAX_SEQ * D_HEAD

    scale = 1.0 / (D_HEAD ** 0.5)
    dh = tl.arange(0, D_HEAD)
    GQA_GROUP: tl.constexpr = N_HEADS // N_KV_HEADS
    D_HALF: tl.constexpr = D_HEAD // 2
    rope_lo = tl.arange(0, D_HALF)

    for layer in tl.range(N_LAYERS):
        w_base = layer * LAYER_W_SIZE
        kv_base = layer * LAYER_KV_SIZE
        kc_base = kv_base
        vc_base = kv_base + N_KV_HEADS * MAX_SEQ * D_HEAD

        off = w_base
        ln1_s_off = off;    off += D_MODEL
        wq_off = off;       off += D_MODEL * D_MODEL
        wk_off = off;       off += D_MODEL * D_KV
        wv_off = off;       off += D_MODEL * D_KV
        wo_off = off;       off += D_MODEL * D_MODEL
        ln2_s_off = off;    off += D_MODEL
        gate_off = off;     off += D_MODEL * D_FF
        up_off = off;       off += D_MODEL * D_FF
        down_off = off

        # ── PHASE 1: RMSNorm1 + Attention ──
        ln_s = tl.load(packed_w_ptr + ln1_s_off + d, mask=d_mask, other=0.0).to(tl.float32)
        h_sq = tl.where(d_mask, h * h, 0.0)
        h_norm = tl.where(d_mask,
                          ln_s * h * tl.math.rsqrt(tl.sum(h_sq) / D_MODEL + 1e-5),
                          0.0)

        hd = head_id * D_HEAD + dh
        kv_head = head_id // GQA_GROUP
        kv_hd = kv_head * D_HEAD + dh
        cache_off = kv_head * MAX_SEQ * D_HEAD

        # Store h_norm for tiled projections
        tl.store(attn_partial_ptr + pid * D_BLOCK + d, h_norm, mask=d_mask)

        Q = tl.zeros((D_HEAD,), dtype=tl.float32)
        K_new = tl.zeros((D_HEAD,), dtype=tl.float32)
        V_new = tl.zeros((D_HEAD,), dtype=tl.float32)
        for dd in tl.static_range(0, D_BLOCK, PROJ_TILE):
            dt = dd + tl.arange(0, PROJ_TILE)
            dt_mask = dt < D_MODEL
            h_tile = tl.load(attn_partial_ptr + pid * D_BLOCK + dt,
                             mask=dt_mask, other=0.0).to(tl.bfloat16)
            wq_t = tl.load(packed_w_ptr + wq_off + dt[:, None] * D_MODEL + hd[None, :],
                           mask=dt_mask[:, None], other=0.0).to(tl.bfloat16)
            Q += tl.dot(h_tile[None, :], wq_t).to(tl.float32).sum(axis=0)
            wk_t = tl.load(packed_w_ptr + wk_off + dt[:, None] * D_KV + kv_hd[None, :],
                           mask=dt_mask[:, None], other=0.0).to(tl.bfloat16)
            K_new += tl.dot(h_tile[None, :], wk_t).to(tl.float32).sum(axis=0)
            wv_t = tl.load(packed_w_ptr + wv_off + dt[:, None] * D_KV + kv_hd[None, :],
                           mask=dt_mask[:, None], other=0.0).to(tl.bfloat16)
            V_new += tl.dot(h_tile[None, :], wv_t).to(tl.float32).sum(axis=0)

        # ── RoPE on Q and K_new via scratch buffer ──
        cos_val = tl.load(cos_ptr + pos * D_HALF + rope_lo).to(tl.float32)
        sin_val = tl.load(sin_ptr + pos * D_HALF + rope_lo).to(tl.float32)

        scratch = attn_partial_ptr + pid * D_BLOCK
        # Rotate Q
        tl.store(scratch + dh, Q)
        q_lo = tl.load(scratch + rope_lo)
        q_hi = tl.load(scratch + D_HALF + rope_lo)
        tl.store(scratch + rope_lo, q_lo * cos_val - q_hi * sin_val)
        tl.store(scratch + D_HALF + rope_lo, q_lo * sin_val + q_hi * cos_val)
        Q = tl.load(scratch + dh)

        # Rotate K_new
        tl.store(scratch + dh, K_new)
        k_lo = tl.load(scratch + rope_lo)
        k_hi = tl.load(scratch + D_HALF + rope_lo)
        tl.store(scratch + rope_lo, k_lo * cos_val - k_hi * sin_val)
        tl.store(scratch + D_HALF + rope_lo, k_lo * sin_val + k_hi * cos_val)
        K_new = tl.load(scratch + dh)

        # Store K_new (with RoPE) and V_new to output cache
        # Only one block per kv_head writes to avoid redundant concurrent stores
        is_kv_primary = (head_id == kv_head * GQA_GROUP)
        if is_kv_primary:
            tl.store(kv_out_ptr + kc_base + cache_off + pos * D_HEAD + dh, K_new.to(tl.bfloat16))
            tl.store(kv_out_ptr + vc_base + cache_off + pos * D_HEAD + dh, V_new.to(tl.bfloat16))

        # ── Split KV attention with online softmax ──
        m_i = tl.full((1,), value=-1e9, dtype=tl.float32)
        l_i = tl.zeros((1,), dtype=tl.float32)
        o_i = tl.zeros((D_HEAD,), dtype=tl.float32)

        POS_PER_SPLIT: tl.constexpr = MAX_SEQ // KV_SPLITS
        kv_start = kv_split * POS_PER_SPLIT
        kv_end = kv_start + POS_PER_SPLIT
        for t in tl.range(kv_start, kv_end, KV_TILE):
            tile_pos = t + tl.arange(0, KV_TILE)
            tile_mask = tile_pos <= pos

            K_tile = tl.load(kv_in_ptr + kc_base + cache_off + tile_pos[:, None] * D_HEAD + dh[None, :],
                            mask=tile_mask[:, None], other=0.0,
                            eviction_policy='evict_last').to(tl.float32)
            K_tile = tl.where(tile_pos[:, None] == pos, K_new[None, :], K_tile)
            if is_kv_primary:
                tl.store(kv_out_ptr + kc_base + cache_off + tile_pos[:, None] * D_HEAD + dh[None, :],
                        K_tile.to(tl.bfloat16), mask=tile_mask[:, None])

            V_tile = tl.load(kv_in_ptr + vc_base + cache_off + tile_pos[:, None] * D_HEAD + dh[None, :],
                            mask=tile_mask[:, None], other=0.0,
                            eviction_policy='evict_last').to(tl.float32)
            V_tile = tl.where(tile_pos[:, None] == pos, V_new[None, :], V_tile)
            if is_kv_primary:
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

        attn_out = o_i / tl.maximum(l_i, 1e-10)

        # Tile O projection along D_BLOCK
        for dd in tl.static_range(0, D_BLOCK, PROJ_TILE):
            dt = dd + tl.arange(0, PROJ_TILE)
            dt_mask = dt < D_MODEL
            wo_t = tl.load(packed_w_ptr + wo_off + hd[:, None] * D_MODEL + dt[None, :],
                           mask=dt_mask[None, :], other=0.0).to(tl.bfloat16)
            o_tile = tl.dot(attn_out[None, :].to(tl.bfloat16), wo_t).to(tl.float32).sum(axis=0)
            tl.store(attn_partial_ptr + pid * D_BLOCK + dt, o_tile, mask=dt_mask)
        tl.store(attn_ml_ptr + pid * 2, tl.sum(m_i))
        tl.store(attn_ml_ptr + pid * 2 + 1, tl.sum(l_i))

        # Global memory fence before and after barrier
        tl.debug_barrier()
        tl.inline_asm_elementwise("membar.gl; mov.u32 $0, 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1)
        b0 = layer * 3
        tl.atomic_add(barrier_ptr + b0, 1.0, sem='acq_rel', scope='gpu')
        while tl.atomic_add(barrier_ptr + b0, 0.0, sem='acquire', scope='gpu') < TOTAL_BLOCKS:
            pass
        tl.inline_asm_elementwise("membar.gl; mov.u32 $0, 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1)

        # ── PHASE 2: Merge attention + RMSNorm2 + SwiGLU FFN ──
        attn_total = tl.zeros((D_BLOCK,), dtype=tl.float32)
        for head in tl.range(N_HEADS):
            m_max_val = tl.full((), -1e9, dtype=tl.float32)
            for s in tl.static_range(KV_SPLITS):
                m_s = tl.load(attn_ml_ptr + (head * KV_SPLITS + s) * 2)
                m_max_val = tl.maximum(m_max_val, m_s)
            l_total = tl.full((), 0.0, dtype=tl.float32)
            o_merged = tl.zeros((D_BLOCK,), dtype=tl.float32)
            for s in tl.static_range(KV_SPLITS):
                m_s = tl.load(attn_ml_ptr + (head * KV_SPLITS + s) * 2)
                l_s = tl.load(attn_ml_ptr + (head * KV_SPLITS + s) * 2 + 1)
                o_s = tl.load(attn_partial_ptr + (head * KV_SPLITS + s) * D_BLOCK + d, mask=d_mask, other=0.0)
                w = l_s * tl.exp(m_s - m_max_val)
                l_total = l_total + w
                o_merged = o_merged + o_s * w
            attn_total = attn_total + o_merged / l_total
        h = h + attn_total

        # RMSNorm2
        ln_s = tl.load(packed_w_ptr + ln2_s_off + d, mask=d_mask, other=0.0).to(tl.float32)
        h_sq = tl.where(d_mask, h * h, 0.0)
        h_norm = tl.where(d_mask,
                          ln_s * h * tl.math.rsqrt(tl.sum(h_sq) / D_MODEL + 1e-5),
                          0.0)
        h_norm_2d = h_norm[None, :].to(tl.bfloat16)

        # SwiGLU FFN (distributed across blocks)
        ff_start = pid * FF_PER_BLOCK
        ffn_partial = tl.zeros((D_BLOCK,), dtype=tl.float32)
        for k in tl.range(0, FF_PER_BLOCK, BLOCK_K):
            kk = ff_start + k + tl.arange(0, BLOCK_K)
            ff_mask = kk < D_FF
            # Gate projection
            gate_w = tl.load(packed_w_ptr + gate_off + d[:, None] * D_FF + kk[None, :],
                             mask=d_mask[:, None] & ff_mask[None, :], other=0.0).to(tl.bfloat16)
            gate = tl.dot(h_norm_2d, gate_w).to(tl.float32).sum(axis=0)
            # Up projection
            up_w = tl.load(packed_w_ptr + up_off + d[:, None] * D_FF + kk[None, :],
                           mask=d_mask[:, None] & ff_mask[None, :], other=0.0).to(tl.bfloat16)
            up = tl.dot(h_norm_2d, up_w).to(tl.float32).sum(axis=0)
            # SwiGLU: SiLU(gate) * up
            act = (gate * tl.sigmoid(gate)) * up
            # Down projection
            down_w = tl.load(packed_w_ptr + down_off + kk[:, None] * D_MODEL + d[None, :],
                             mask=ff_mask[:, None] & d_mask[None, :], other=0.0).to(tl.bfloat16)
            ffn_partial += tl.dot(act[None, :].to(tl.bfloat16), down_w).to(tl.float32).sum(axis=0)

        tl.store(ffn_partial_ptr + pid * D_BLOCK + d, ffn_partial, mask=d_mask)

        tl.debug_barrier()
        tl.inline_asm_elementwise("membar.gl; mov.u32 $0, 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1)
        b1 = layer * 3 + 1
        tl.atomic_add(barrier_ptr + b1, 1.0, sem='acq_rel', scope='gpu')
        while tl.atomic_add(barrier_ptr + b1, 0.0, sem='acquire', scope='gpu') < TOTAL_BLOCKS:
            pass
        tl.inline_asm_elementwise("membar.gl; mov.u32 $0, 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1)

        # ── PHASE 3: Reduce FFN + residual ──
        ffn_total = tl.zeros((D_BLOCK,), dtype=tl.float32)
        for i in tl.range(TOTAL_BLOCKS):
            ffn_total += tl.load(ffn_partial_ptr + i * D_BLOCK + d, mask=d_mask, other=0.0)
        h = h + ffn_total

        # Barrier after reduce: prevent next iteration from overwriting before all reads complete
        tl.inline_asm_elementwise("membar.gl; mov.u32 $0, 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1)
        b2 = layer * 3 + 2
        tl.atomic_add(barrier_ptr + b2, 1.0, sem='acq_rel', scope='gpu')
        while tl.atomic_add(barrier_ptr + b2, 0.0, sem='acquire', scope='gpu') < TOTAL_BLOCKS:
            pass

    # ── OUTPUT: final RMSNorm + tied output projection ──
    ln_s = tl.load(lnf_s_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    h_sq = tl.where(d_mask, h * h, 0.0)
    h_final = tl.where(d_mask,
                       ln_s * h * tl.math.rsqrt(tl.sum(h_sq) / D_MODEL + 1e-5),
                       0.0)
    h_final_2d = h_final[None, :].to(tl.bfloat16)

    TILES_PER_BLOCK: tl.constexpr = VOCAB_PAD // (OUTPUT_VTILE * TOTAL_BLOCKS)
    best_val = -1e9
    best_idx = 0.0
    for tile_idx in tl.range(0, TILES_PER_BLOCK):
        v_start = (pid * TILES_PER_BLOCK + tile_idx) * OUTPUT_VTILE
        vv = v_start + tl.arange(0, OUTPUT_VTILE)
        out_w = tl.load(output_proj_ptr + d[:, None] * VOCAB_PAD + vv[None, :],
                        mask=d_mask[:, None], other=0.0,
                        eviction_policy='evict_first').to(tl.bfloat16)
        tile_logits = tl.dot(h_final_2d, out_w).to(tl.float32).sum(axis=0)
        tile_logits = tl.where(vv < VOCAB_SIZE, tile_logits, -1e9)
        tl.store(logits_ptr + vv, tile_logits)
        tile_max = tl.max(tile_logits)
        if tile_max > best_val:
            best_val = tile_max
            best_idx = (v_start + tl.argmax(tile_logits, axis=0)).to(tl.float32)

    tl.store(argmax_ptr + pid * 2, best_val)
    tl.store(argmax_ptr + pid * 2 + 1, best_idx)

    tl.inline_asm_elementwise("membar.gl; mov.u32 $0, 0;", "=r", [], dtype=tl.int32, is_pure=False, pack=1)
    N_BARRIERS: tl.constexpr = 3 * N_LAYERS
    tl.atomic_add(barrier_ptr + N_BARRIERS, 1.0, sem='acq_rel', scope='gpu')
    while tl.atomic_add(barrier_ptr + N_BARRIERS, 0.0, sem='acquire', scope='gpu') < TOTAL_BLOCKS:
        pass

    if pid == 0:
        global_best_val = -1e9
        global_best_idx = 0.0
        for i in tl.range(TOTAL_BLOCKS):
            v = tl.load(argmax_ptr + i * 2)
            idx = tl.load(argmax_ptr + i * 2 + 1)
            if v > global_best_val:
                global_best_val = v
                global_best_idx = idx
        tl.store(next_token_ptr, global_best_idx.to(tl.int32))


def _next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p


# ──────────────────────────────────────────────────────────────────────

def multi_sm_decode_nlayer(w, config, token_id, pos, kv_packed, vocab_size, kv_splits=2):
    """Multi-SM fused N-layer decode with KV-split parallelism and split barriers."""
    d_model = config["d_model"]
    d_block = _next_power_of_2(d_model)
    d_head = config["d_head"]
    n_heads = config["n_heads"]
    n_kv_heads = config.get("n_kv_heads", n_heads)
    d_kv = n_kv_heads * d_head
    n_layers = config["n_layers"]
    d_ff = config["d_ff"]
    max_seq = config["context_len"]
    vocab_pad = w["vocab_pad"]
    total_kv_size = n_layers * 2 * n_kv_heads * max_seq * d_head
    total_blocks = n_heads * kv_splits

    # FF_PER_BLOCK: ceiling division, padded to BLOCK_K for clean loop
    block_k = 16
    raw_ff = (d_ff + total_blocks - 1) // total_blocks
    ff_per_block = ((raw_ff + block_k - 1) // block_k) * block_k

    # Workspace layout
    # Workspace layout: separate attn and FFN partial buffers to avoid race
    n_barriers = 3 * n_layers + 1  # 3 per layer (attn + FFN + post-reduce) + 1 for output
    ffn_partial_off = total_blocks * d_block          # attn partials: [0, ffn_partial_off)
    attn_ml_off = ffn_partial_off + total_blocks * d_block  # FFN partials: [ffn_partial_off, attn_ml_off)
    barrier_off = attn_ml_off + total_blocks * 2
    done_off = barrier_off + n_barriers
    argmax_off = done_off + n_barriers
    workspace_size = argmax_off + 2 * total_blocks

    workspace = jnp.zeros((workspace_size,), dtype=jnp.float32)

    logits_pad, kv_out, next_token = jt.triton_call(
        w["token_emb"],
        w["packed_w"],
        w["lnf_s"],
        w["output_proj_padded"],
        w["cos"], w["sin"],
        jnp.int32(token_id), jnp.int32(pos),
        kv_packed,
        workspace,
        kernel=_multi_sm_decode,
        out_shape=[
            jax.ShapeDtypeStruct((vocab_pad,), jnp.float32),
            jax.ShapeDtypeStruct((total_kv_size,), jnp.bfloat16),
            jax.ShapeDtypeStruct((1,), jnp.int32),
        ],
        grid=(total_blocks,),
        num_warps=4, num_stages=1,
        D_MODEL=d_model, D_BLOCK=d_block, D_HEAD=d_head, D_FF=d_ff,
        N_HEADS=n_heads, N_KV_HEADS=n_kv_heads, D_KV=d_kv,
        N_LAYERS=n_layers, MAX_SEQ=max_seq,
        KV_SPLITS=kv_splits, TOTAL_BLOCKS=total_blocks,
        VOCAB_SIZE=vocab_size, VOCAB_PAD=vocab_pad,
        FF_PER_BLOCK=ff_per_block,
        FFN_PARTIAL_OFF=ffn_partial_off,
        ATTN_ML_OFF=attn_ml_off,
        BARRIER_OFF=barrier_off,
        DONE_OFF=done_off,
        ARGMAX_OFF=argmax_off,
    )

    return next_token[0], logits_pad[:vocab_size], kv_out
