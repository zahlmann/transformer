"""
Multi-SM fused N-layer decode kernel with KV-split parallelism.

Key insight from profiling: the single-SM kernel (grid=1) uses 1 of 80 SMs on the
RTX 4080 Super, leaving 79 idle. This kernel uses grid=(N_HEADS * KV_SPLITS,) to
parallelize attention across SMs, splitting KV cache tiles within each head for
additional parallelism (FlashDecoding-style).

Architecture:
  grid = (N_HEADS * KV_SPLITS,)  — multiple blocks per attention head

  Per layer:
    Phase 1 (all blocks): LN1 + QKV proj + attention (split KV tiles) + O proj
      → each block handles a subset of KV cache tiles for one head
      → write normalized O-proj + online softmax state (m, l) to scratch buffer
      → atomic barrier
    Phase 2 (all blocks): merge KV splits + reduce attention + residual + LN2 + FFN
      → weighted merge of O-proj partials using online softmax correction
      → each block handles D_FF/TOTAL_BLOCKS elements of FFN intermediate dim
      → write FFN partial to scratch buffer
      → atomic barrier
    Phase 3 (all blocks): reduce FFN + residual → h ready for next layer

  Output: all blocks participate in tiled output projection.

Single f32 workspace buffer holds partial results, attention merge state (m/l),
barrier counters, and argmax scratch.
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
    # Inputs (positional args in jt.triton_call)
    token_emb_ptr, pos_emb_ptr,
    packed_w_ptr,
    lnf_s_ptr, lnf_b_ptr,
    output_proj_ptr,
    token_id_ptr, pos_ptr,
    kv_in_ptr,
    workspace_ptr,    # single f32 buffer: [partials | attn_ml | barriers | argmax]
    # Outputs (from out_shape)
    logits_ptr,
    kv_out_ptr,
    next_token_ptr,   # (1,) int32 — argmax result
    # Config
    D_MODEL: tl.constexpr,
    D_HEAD: tl.constexpr,
    D_FF: tl.constexpr,
    N_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,     # GQA: number of KV heads (N_KV_HEADS <= N_HEADS)
    D_KV: tl.constexpr,           # N_KV_HEADS * D_HEAD
    N_LAYERS: tl.constexpr,
    MAX_SEQ: tl.constexpr,
    KV_SPLITS: tl.constexpr,      # KV cache splits per head (1=original, 2+=FlashDecoding)
    TOTAL_BLOCKS: tl.constexpr,   # N_HEADS * KV_SPLITS
    VOCAB_SIZE: tl.constexpr,
    VOCAB_PAD: tl.constexpr,
    FF_PER_BLOCK: tl.constexpr,
    PARALLEL_RESIDUAL: tl.constexpr,  # True: parallel attn||ffn (1 barrier/layer)
    ATTN_ML_OFF: tl.constexpr,    # offset to attention m/l section
    FFN_PARTIAL_OFF: tl.constexpr, # offset to FFN partials (for parallel residual)
    BARRIER_OFF: tl.constexpr,    # offset to barrier counters
    DONE_OFF: tl.constexpr,       # offset to done flags (separate cache line from counters)
    ARGMAX_OFF: tl.constexpr,     # offset to argmax section
):
    pid = tl.program_id(0)  # 0..TOTAL_BLOCKS-1
    head_id = pid // KV_SPLITS
    kv_split = pid % KV_SPLITS
    d = tl.arange(0, D_MODEL)

    token_id = tl.load(token_id_ptr)
    pos = tl.load(pos_ptr)

    # Aliases for workspace sections
    partial_ptr = workspace_ptr                       # attention partials (always)
    ffn_partial_ptr = workspace_ptr + FFN_PARTIAL_OFF # FFN partials (parallel residual only)
    attn_ml_ptr = workspace_ptr + ATTN_ML_OFF
    barrier_ptr = workspace_ptr + BARRIER_OFF
    done_ptr = workspace_ptr + DONE_OFF
    argmax_ptr = workspace_ptr + ARGMAX_OFF

    # ── Embedding (all blocks compute redundantly — 2KB from L2) ──
    h = (tl.load(token_emb_ptr + token_id * D_MODEL + d).to(tl.float32)
       + tl.load(pos_emb_ptr + pos * D_MODEL + d).to(tl.float32))

    # GQA: Q/O use D_MODEL columns, K/V use D_KV columns
    LAYER_W_SIZE: tl.constexpr = (
        D_MODEL + D_MODEL +                   # ln1 scale, bias
        D_MODEL * D_MODEL +                   # wq (n_heads * d_head = d_model)
        D_MODEL * D_KV +                      # wk (n_kv_heads * d_head)
        D_MODEL * D_KV +                      # wv (n_kv_heads * d_head)
        D_MODEL * D_MODEL +                   # wo
        D_MODEL + D_MODEL +                   # ln2 scale, bias
        D_MODEL * D_FF + D_FF +               # ffn up + bias
        D_FF * D_MODEL + D_MODEL              # ffn down + bias
    )
    LAYER_KV_SIZE: tl.constexpr = 2 * N_KV_HEADS * MAX_SEQ * D_HEAD

    scale = 0.17677669529663689
    dh = tl.arange(0, D_HEAD)

    for layer in tl.range(N_LAYERS):
        w_base = layer * LAYER_W_SIZE
        kv_base = layer * LAYER_KV_SIZE
        kc_base = kv_base
        vc_base = kv_base + N_KV_HEADS * MAX_SEQ * D_HEAD

        off = w_base
        ln1_s_off = off;    off += D_MODEL
        ln1_b_off = off;    off += D_MODEL
        wq_off = off;       off += D_MODEL * D_MODEL
        wk_off = off;       off += D_MODEL * D_KV
        wv_off = off;       off += D_MODEL * D_KV
        wo_off = off;       off += D_MODEL * D_MODEL
        ln2_s_off = off;    off += D_MODEL
        ln2_b_off = off;    off += D_MODEL
        up_off = off;       off += D_MODEL * D_FF
        up_b_off = off;     off += D_FF
        down_off = off;     off += D_FF * D_MODEL
        down_b_off = off

        # ── PHASE 1: LN1 + Attention ──
        ln_s = tl.load(packed_w_ptr + ln1_s_off + d).to(tl.float32)
        ln_b = tl.load(packed_w_ptr + ln1_b_off + d).to(tl.float32)
        mean = tl.sum(h) / D_MODEL
        hc = h - mean
        h_norm = ln_s * hc * tl.math.rsqrt(tl.sum(hc * hc) / D_MODEL + 1e-5) + ln_b

        head = head_id
        hd = head * D_HEAD + dh                    # Q head column indices
        GQA_GROUP: tl.constexpr = N_HEADS // N_KV_HEADS
        kv_head = head_id // GQA_GROUP              # which KV head this Q head uses
        kv_hd = kv_head * D_HEAD + dh              # KV head column indices
        cache_off = kv_head * MAX_SEQ * D_HEAD      # KV cache offset (per KV head)
        h_norm_2d = h_norm[None, :].to(tl.bfloat16)

        wq = tl.load(packed_w_ptr + wq_off + d[:, None] * D_MODEL + hd[None, :]).to(tl.bfloat16)
        Q = tl.dot(h_norm_2d, wq).to(tl.float32).sum(axis=0)
        wk = tl.load(packed_w_ptr + wk_off + d[:, None] * D_KV + kv_hd[None, :]).to(tl.bfloat16)
        K_new = tl.dot(h_norm_2d, wk).to(tl.float32).sum(axis=0)
        wv = tl.load(packed_w_ptr + wv_off + d[:, None] * D_KV + kv_hd[None, :]).to(tl.bfloat16)
        V_new = tl.dot(h_norm_2d, wv).to(tl.float32).sum(axis=0)

        tl.store(kv_out_ptr + kc_base + cache_off + pos * D_HEAD + dh, K_new.to(tl.bfloat16))
        tl.store(kv_out_ptr + vc_base + cache_off + pos * D_HEAD + dh, V_new.to(tl.bfloat16))

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

        attn_out = o_i / tl.maximum(l_i, 1e-10)  # clamp to avoid NaN when split has no valid positions

        wo = tl.load(packed_w_ptr + wo_off + hd[:, None] * D_MODEL + d[None, :]).to(tl.bfloat16)
        o_proj = tl.dot(attn_out[None, :].to(tl.bfloat16), wo).to(tl.float32).sum(axis=0)

        tl.store(partial_ptr + pid * D_MODEL + d, o_proj)
        tl.store(attn_ml_ptr + pid * 2, tl.sum(m_i))
        tl.store(attn_ml_ptr + pid * 2 + 1, tl.sum(l_i))

        if PARALLEL_RESIDUAL:
            # ── PARALLEL RESIDUAL: compute FFN from same h (before attention modifies it) ──
            ln_s = tl.load(packed_w_ptr + ln2_s_off + d).to(tl.float32)
            ln_b = tl.load(packed_w_ptr + ln2_b_off + d).to(tl.float32)
            mean = tl.sum(h) / D_MODEL
            hc = h - mean
            h_norm2 = ln_s * hc * tl.math.rsqrt(tl.sum(hc * hc) / D_MODEL + 1e-5) + ln_b
            h_norm2_2d = h_norm2[None, :].to(tl.bfloat16)

            ff_start = pid * FF_PER_BLOCK
            ffn_partial = tl.zeros((D_MODEL,), dtype=tl.float32)
            for k in tl.range(0, FF_PER_BLOCK, BLOCK_K):
                kk = ff_start + k + tl.arange(0, BLOCK_K)
                up_w = tl.load(packed_w_ptr + up_off + d[:, None] * D_FF + kk[None, :]).to(tl.bfloat16)
                up = tl.dot(h_norm2_2d, up_w).to(tl.float32).sum(axis=0)
                up += tl.load(packed_w_ptr + up_b_off + kk).to(tl.float32)
                act = up * tl.sigmoid(1.702 * up)
                down_w = tl.load(packed_w_ptr + down_off + kk[:, None] * D_MODEL + d[None, :]).to(tl.bfloat16)
                ffn_partial += tl.dot(act[None, :].to(tl.bfloat16), down_w).to(tl.float32).sum(axis=0)

            tl.store(ffn_partial_ptr + pid * D_MODEL + d, ffn_partial)

            # SINGLE barrier (both attn and FFN partials written)
            b = layer
            old_cnt = tl.atomic_add(barrier_ptr + b, 1.0, sem='release', scope='gpu')
            if old_cnt >= TOTAL_BLOCKS - 1:
                _ = tl.atomic_add(barrier_ptr + b, 0.0, sem='acquire', scope='gpu')
                tl.atomic_add(done_ptr + b, 1.0, sem='release', scope='gpu')
            while tl.atomic_add(done_ptr + b, 0.0, sem='acquire', scope='gpu') < 1.0:
                pass

            # Reduce both attention and FFN, combine
            attn_total = tl.zeros((D_MODEL,), dtype=tl.float32)
            for head in tl.range(N_HEADS):
                m_max_val = tl.full((), -1e9, dtype=tl.float32)
                for s in tl.static_range(KV_SPLITS):
                    m_s = tl.load(attn_ml_ptr + (head * KV_SPLITS + s) * 2)
                    m_max_val = tl.maximum(m_max_val, m_s)
                l_total = tl.full((), 0.0, dtype=tl.float32)
                o_merged = tl.zeros((D_MODEL,), dtype=tl.float32)
                for s in tl.static_range(KV_SPLITS):
                    m_s = tl.load(attn_ml_ptr + (head * KV_SPLITS + s) * 2)
                    l_s = tl.load(attn_ml_ptr + (head * KV_SPLITS + s) * 2 + 1)
                    o_s = tl.load(partial_ptr + (head * KV_SPLITS + s) * D_MODEL + d)
                    w = l_s * tl.exp(m_s - m_max_val)
                    l_total = l_total + w
                    o_merged = o_merged + o_s * w
                attn_total = attn_total + o_merged / l_total

            ffn_total = tl.zeros((D_MODEL,), dtype=tl.float32)
            for i in tl.range(TOTAL_BLOCKS):
                ffn_total += tl.load(ffn_partial_ptr + i * D_MODEL + d)

            h = h + attn_total + ffn_total + tl.load(packed_w_ptr + down_b_off + d).to(tl.float32)
        else:
            # ── SEQUENTIAL RESIDUAL (standard transformer) ──
            # Barrier 1: attention partials
            b0 = layer * 2
            old_cnt = tl.atomic_add(barrier_ptr + b0, 1.0, sem='release', scope='gpu')
            if old_cnt >= TOTAL_BLOCKS - 1:
                _ = tl.atomic_add(barrier_ptr + b0, 0.0, sem='acquire', scope='gpu')
                tl.atomic_add(done_ptr + b0, 1.0, sem='release', scope='gpu')
            while tl.atomic_add(done_ptr + b0, 0.0, sem='acquire', scope='gpu') < 1.0:
                pass

            # Merge KV splits + reduce attention
            attn_total = tl.zeros((D_MODEL,), dtype=tl.float32)
            for head in tl.range(N_HEADS):
                m_max_val = tl.full((), -1e9, dtype=tl.float32)
                for s in tl.static_range(KV_SPLITS):
                    m_s = tl.load(attn_ml_ptr + (head * KV_SPLITS + s) * 2)
                    m_max_val = tl.maximum(m_max_val, m_s)
                l_total = tl.full((), 0.0, dtype=tl.float32)
                o_merged = tl.zeros((D_MODEL,), dtype=tl.float32)
                for s in tl.static_range(KV_SPLITS):
                    m_s = tl.load(attn_ml_ptr + (head * KV_SPLITS + s) * 2)
                    l_s = tl.load(attn_ml_ptr + (head * KV_SPLITS + s) * 2 + 1)
                    o_s = tl.load(partial_ptr + (head * KV_SPLITS + s) * D_MODEL + d)
                    w = l_s * tl.exp(m_s - m_max_val)
                    l_total = l_total + w
                    o_merged = o_merged + o_s * w
                attn_total = attn_total + o_merged / l_total
            h = h + attn_total

            # LN2 + FFN
            ln_s = tl.load(packed_w_ptr + ln2_s_off + d).to(tl.float32)
            ln_b = tl.load(packed_w_ptr + ln2_b_off + d).to(tl.float32)
            mean = tl.sum(h) / D_MODEL
            hc = h - mean
            h_norm2 = ln_s * hc * tl.math.rsqrt(tl.sum(hc * hc) / D_MODEL + 1e-5) + ln_b
            h_norm2_2d = h_norm2[None, :].to(tl.bfloat16)

            ff_start = pid * FF_PER_BLOCK
            ffn_partial = tl.zeros((D_MODEL,), dtype=tl.float32)
            for k in tl.range(0, FF_PER_BLOCK, BLOCK_K):
                kk = ff_start + k + tl.arange(0, BLOCK_K)
                up_w = tl.load(packed_w_ptr + up_off + d[:, None] * D_FF + kk[None, :]).to(tl.bfloat16)
                up = tl.dot(h_norm2_2d, up_w).to(tl.float32).sum(axis=0)
                up += tl.load(packed_w_ptr + up_b_off + kk).to(tl.float32)
                act = up * tl.sigmoid(1.702 * up)
                down_w = tl.load(packed_w_ptr + down_off + kk[:, None] * D_MODEL + d[None, :]).to(tl.bfloat16)
                ffn_partial += tl.dot(act[None, :].to(tl.bfloat16), down_w).to(tl.float32).sum(axis=0)

            tl.store(partial_ptr + pid * D_MODEL + d, ffn_partial)

            # Barrier 2: FFN partials
            b1 = layer * 2 + 1
            old_cnt = tl.atomic_add(barrier_ptr + b1, 1.0, sem='release', scope='gpu')
            if old_cnt >= TOTAL_BLOCKS - 1:
                _ = tl.atomic_add(barrier_ptr + b1, 0.0, sem='acquire', scope='gpu')
                tl.atomic_add(done_ptr + b1, 1.0, sem='release', scope='gpu')
            while tl.atomic_add(done_ptr + b1, 0.0, sem='acquire', scope='gpu') < 1.0:
                pass

            # Reduce FFN + residual
            ffn_total = tl.zeros((D_MODEL,), dtype=tl.float32)
            for i in tl.range(TOTAL_BLOCKS):
                ffn_total += tl.load(partial_ptr + i * D_MODEL + d)
            h = h + ffn_total + tl.load(packed_w_ptr + down_b_off + d).to(tl.float32)

    # ── OUTPUT ──
    ln_s = tl.load(lnf_s_ptr + d).to(tl.float32)
    ln_b = tl.load(lnf_b_ptr + d).to(tl.float32)
    mean = tl.sum(h) / D_MODEL
    hc = h - mean
    h_final = ln_s * hc * tl.math.rsqrt(tl.sum(hc * hc) / D_MODEL + 1e-5) + ln_b
    h_final_2d = h_final[None, :].to(tl.bfloat16)

    TILES_PER_BLOCK: tl.constexpr = VOCAB_PAD // (OUTPUT_VTILE * TOTAL_BLOCKS)
    best_val = -1e9
    best_idx = 0.0
    for tile_idx in tl.range(0, TILES_PER_BLOCK):
        v_start = (pid * TILES_PER_BLOCK + tile_idx) * OUTPUT_VTILE
        vv = v_start + tl.arange(0, OUTPUT_VTILE)
        out_w = tl.load(output_proj_ptr + d[:, None] * VOCAB_PAD + vv[None, :]).to(tl.bfloat16)
        tile_logits = tl.dot(h_final_2d, out_w).to(tl.float32).sum(axis=0)
        tile_logits = tl.where(vv < VOCAB_SIZE, tile_logits, -1e9)
        tl.store(logits_ptr + vv, tile_logits)

        # Track per-block argmax
        tile_max = tl.max(tile_logits)
        if tile_max > best_val:
            best_val = tile_max
            best_idx = (v_start + tl.argmax(tile_logits, axis=0)).to(tl.float32)

    # Write per-block argmax to workspace
    tl.store(argmax_ptr + pid * 2, best_val)
    tl.store(argmax_ptr + pid * 2 + 1, best_idx)

    # Final barrier: all blocks done with output projection
    N_BARRIERS: tl.constexpr = N_LAYERS if PARALLEL_RESIDUAL else 2 * N_LAYERS
    old_cnt = tl.atomic_add(barrier_ptr + N_BARRIERS, 1.0, sem='release', scope='gpu')
    if old_cnt >= TOTAL_BLOCKS - 1:
        _ = tl.atomic_add(barrier_ptr + N_BARRIERS, 0.0, sem='acquire', scope='gpu')
        tl.atomic_add(done_ptr + N_BARRIERS, 1.0, sem='release', scope='gpu')
    while tl.atomic_add(done_ptr + N_BARRIERS, 0.0, sem='acquire', scope='gpu') < 1.0:
        pass

    # Block 0: find global argmax
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


# ──────────────────────────────────────────────────────────────────────

def multi_sm_decode_nlayer(w, config, token_id, pos, kv_packed, vocab_size, kv_splits=2):
    """Multi-SM fused N-layer decode: one kernel call per token.

    grid=(N_HEADS * kv_splits,). With kv_splits>1, each head's KV attention is
    split across multiple blocks (FlashDecoding-style) for higher SM utilization.

    Uses a single f32 workspace buffer (partials + attn m/l + barriers + argmax).
    Returns next_token from in-kernel argmax (avoids GPU→CPU sync).

    Returns: next_token (scalar int32), logits (vocab_size,), kv_packed (flat bf16)
    """
    d_model = config["d_model"]
    d_head = config["d_head"]
    n_heads = config["n_heads"]
    n_kv_heads = config.get("n_kv_heads", n_heads)
    d_kv = n_kv_heads * d_head
    n_layers = config["n_layers"]
    d_ff = 4 * d_model
    max_seq = config["context_len"]
    vocab_pad = w["vocab_pad"]
    total_kv_size = n_layers * 2 * n_kv_heads * max_seq * d_head
    total_blocks = n_heads * kv_splits
    ff_per_block = d_ff // total_blocks
    parallel_residual = config.get("parallel_residual", False)

    # Workspace layout: attn_partials | [ffn_partials] | attn_ml | barriers | done | argmax
    ffn_partial_off = total_blocks * d_model  # FFN partials start after attn partials
    if parallel_residual:
        # Need separate FFN partial buffer (attn + FFN written before single barrier)
        attn_ml_off = ffn_partial_off + total_blocks * d_model
        n_barriers = n_layers + 1  # 1 per layer + 1 output
    else:
        # Reuse partial buffer (attn and FFN use it sequentially)
        attn_ml_off = ffn_partial_off  # FFN partials overlap attn partials
        n_barriers = 2 * n_layers + 1  # 2 per layer + 1 output
    barrier_off = attn_ml_off + total_blocks * 2
    done_off = barrier_off + 32  # pad to 128 bytes (separate cache line)
    argmax_off = done_off + 32   # pad done flags too
    workspace_size = argmax_off + 2 * total_blocks  # per-block (val, idx)

    workspace = jnp.zeros((workspace_size,), dtype=jnp.float32)

    logits_pad, kv_out, next_token = jt.triton_call(
        w["token_emb"], w["pos_emb"],
        w["packed_w"],
        w["lnf_s"], w["lnf_b"],
        w["output_proj_padded"],
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
        num_warps=4, num_stages=2,
        D_MODEL=d_model, D_HEAD=d_head, D_FF=d_ff,
        N_HEADS=n_heads, N_KV_HEADS=n_kv_heads, D_KV=d_kv,
        N_LAYERS=n_layers, MAX_SEQ=max_seq,
        KV_SPLITS=kv_splits, TOTAL_BLOCKS=total_blocks,
        VOCAB_SIZE=vocab_size, VOCAB_PAD=vocab_pad,
        FF_PER_BLOCK=ff_per_block,
        PARALLEL_RESIDUAL=parallel_residual,
        ATTN_ML_OFF=attn_ml_off,
        FFN_PARTIAL_OFF=ffn_partial_off,
        BARRIER_OFF=barrier_off,
        DONE_OFF=done_off,
        ARGMAX_OFF=argmax_off,
    )

    return next_token[0], logits_pad[:vocab_size], kv_out
