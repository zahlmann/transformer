"""
Persistent multi-SM decode kernel — loops internally across decode steps.

Eliminates per-step overhead:
  - No workspace allocation per step (pre-allocated once)
  - No JAX dispatch per step (single kernel launch)
  - No GPU→CPU sync per step (tokens collected on device)
  - In-place KV cache updates (no copy, just write new K/V at pos)

Barrier handling: each step uses fresh barrier slots (step * BARRIERS_PER_STEP + idx).
Total barrier slots = N_STEPS * BARRIERS_PER_STEP, pre-zeroed in workspace.

Supports non-power-of-2 D_MODEL via D_BLOCK padding: tl.arange uses D_BLOCK
(next power of 2) with d_mask = d < D_MODEL for all loads/stores.

Based on multi_sm_decode.py with same GQA + KV-split + split-barrier design.
"""

import triton
import triton.language as tl
import jax
import jax.numpy as jnp
import jax_triton as jt

BLOCK_K      = tl.constexpr(16)
KV_TILE      = tl.constexpr(64)
OUTPUT_VTILE = tl.constexpr(32)


@triton.jit
def _persistent_decode(
    # Inputs
    token_emb_ptr, pos_emb_ptr,
    packed_w_ptr,
    lnf_s_ptr, lnf_b_ptr,
    output_proj_ptr,
    first_token_ptr,       # int32 scalar — initial token
    start_pos_ptr,         # int32 scalar — initial position
    kv_ptr,                # (TOTAL_KV_SIZE,) bf16 — input KV cache (read-only)
    workspace_ptr,
    # Outputs
    token_out_ptr,         # (N_STEPS,) int32 — generated tokens
    logits_out_ptr,        # (VOCAB_PAD,) f32 — last step's logits only
    kv_out_ptr,            # (TOTAL_KV_SIZE,) bf16 — writable KV cache
    # Config
    D_MODEL: tl.constexpr,
    D_BLOCK: tl.constexpr,   # next power of 2 >= D_MODEL
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
    N_STEPS: tl.constexpr,
    BARRIERS_PER_STEP: tl.constexpr,
    # Workspace offsets
    PARTIAL_OFF: tl.constexpr,
    ATTN_ML_OFF: tl.constexpr,
    BARRIER_OFF: tl.constexpr,
    DONE_OFF: tl.constexpr,
    ARGMAX_OFF: tl.constexpr,
    NEXT_TOK_OFF: tl.constexpr,
):
    pid = tl.program_id(0)
    head_id = pid // KV_SPLITS
    kv_split = pid % KV_SPLITS
    d = tl.arange(0, D_BLOCK)
    d_mask = d < D_MODEL
    dh = tl.arange(0, D_HEAD)

    partial_ptr = workspace_ptr + PARTIAL_OFF
    attn_ml_ptr = workspace_ptr + ATTN_ML_OFF
    barrier_ptr = workspace_ptr + BARRIER_OFF
    done_ptr = workspace_ptr + DONE_OFF
    argmax_ptr = workspace_ptr + ARGMAX_OFF
    next_tok_ptr = workspace_ptr + NEXT_TOK_OFF

    LAYER_W_SIZE: tl.constexpr = (
        D_MODEL + D_MODEL +
        D_MODEL * D_MODEL + D_MODEL * D_KV + D_MODEL * D_KV + D_MODEL * D_MODEL +
        D_MODEL + D_MODEL +
        D_MODEL * D_FF + D_FF * D_MODEL
    )
    LAYER_KV_SIZE: tl.constexpr = 2 * N_KV_HEADS * MAX_SEQ * D_HEAD

    scale = 0.17677669529663689   # 1/sqrt(32)
    GQA_GROUP: tl.constexpr = N_HEADS // N_KV_HEADS
    POS_PER_SPLIT: tl.constexpr = MAX_SEQ // KV_SPLITS

    hd = head_id * D_HEAD + dh
    kv_head = head_id // GQA_GROUP
    kv_hd = kv_head * D_HEAD + dh
    cache_off = kv_head * MAX_SEQ * D_HEAD
    kv_start = kv_split * POS_PER_SPLIT
    kv_end = kv_start + POS_PER_SPLIT

    start_pos = tl.load(start_pos_ptr)
    token_id = tl.load(first_token_ptr)

    # ── Initial KV copy: kv_ptr (read-only input) → kv_out_ptr (writable output) ──
    TOTAL_KV: tl.constexpr = N_LAYERS * 2 * N_KV_HEADS * MAX_SEQ * D_HEAD
    ELEMS_PER_BLOCK: tl.constexpr = TOTAL_KV // TOTAL_BLOCKS
    copy_start = pid * ELEMS_PER_BLOCK
    for ci in tl.range(0, ELEMS_PER_BLOCK, D_BLOCK):
        idx = copy_start + ci + d
        mask = idx < TOTAL_KV
        vals = tl.load(kv_ptr + idx, mask=mask)
        tl.store(kv_out_ptr + idx, vals, mask=mask)

    # Barrier to ensure copy is complete before decode begins
    old_cnt = tl.atomic_add(barrier_ptr + 0, 1.0, sem='release', scope='gpu')
    if old_cnt >= TOTAL_BLOCKS - 1:
        _ = tl.atomic_add(barrier_ptr + 0, 0.0, sem='acquire', scope='gpu')
        tl.atomic_add(done_ptr + 0, 1.0, sem='release', scope='gpu')
    while tl.atomic_add(done_ptr + 0, 0.0, sem='acquire', scope='gpu') < 1.0:
        pass

    for step in tl.range(0, N_STEPS):
        pos = start_pos + step

        # ── Embedding ──
        h = (tl.load(token_emb_ptr + token_id * D_MODEL + d, mask=d_mask, other=0.0).to(tl.float32)
           + tl.load(pos_emb_ptr + pos * D_MODEL + d, mask=d_mask, other=0.0).to(tl.float32))

        b_off = 1 + step * BARRIERS_PER_STEP  # +1 for initial copy barrier

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
            down_off = off;     off += D_FF * D_MODEL

            # ── PHASE 1: LN1 + Attention ──
            ln_s = tl.load(packed_w_ptr + ln1_s_off + d, mask=d_mask, other=0.0).to(tl.float32)
            ln_b = tl.load(packed_w_ptr + ln1_b_off + d, mask=d_mask, other=0.0).to(tl.float32)
            mean = tl.sum(h) / D_MODEL
            hc = tl.where(d_mask, h - mean, 0.0)
            h_norm = tl.where(d_mask,
                              ln_s * hc * tl.math.rsqrt(tl.sum(hc * hc) / D_MODEL + 1e-5) + ln_b,
                              0.0)
            h_norm_2d = h_norm[None, :].to(tl.bfloat16)

            wq = tl.load(packed_w_ptr + wq_off + d[:, None] * D_MODEL + hd[None, :],
                          mask=d_mask[:, None], other=0.0).to(tl.bfloat16)
            Q = tl.dot(h_norm_2d, wq).to(tl.float32).sum(axis=0)
            wk = tl.load(packed_w_ptr + wk_off + d[:, None] * D_KV + kv_hd[None, :],
                          mask=d_mask[:, None], other=0.0).to(tl.bfloat16)
            K_new = tl.dot(h_norm_2d, wk).to(tl.float32).sum(axis=0)
            wv = tl.load(packed_w_ptr + wv_off + d[:, None] * D_KV + kv_hd[None, :],
                          mask=d_mask[:, None], other=0.0).to(tl.bfloat16)
            V_new = tl.dot(h_norm_2d, wv).to(tl.float32).sum(axis=0)

            # Write K_new/V_new at pos before attention.
            tl.store(kv_out_ptr + kc_base + cache_off + pos * D_HEAD + dh, K_new.to(tl.bfloat16))
            tl.store(kv_out_ptr + vc_base + cache_off + pos * D_HEAD + dh, V_new.to(tl.bfloat16))

            m_i = tl.full((1,), value=-1e9, dtype=tl.float32)
            l_i = tl.zeros((1,), dtype=tl.float32)
            o_i = tl.zeros((D_HEAD,), dtype=tl.float32)

            for t in tl.range(kv_start, kv_end, KV_TILE):
                tile_pos = t + tl.arange(0, KV_TILE)
                tile_mask = tile_pos <= pos

                K_tile = tl.load(kv_out_ptr + kc_base + cache_off
                                 + tile_pos[:, None] * D_HEAD + dh[None, :],
                                 mask=tile_mask[:, None], other=0.0,
                                 eviction_policy='evict_last').to(tl.float32)
                K_tile = tl.where(tile_pos[:, None] == pos, K_new[None, :], K_tile)
                # Store only at pos to force compiler commit (not full tile copy)
                pos_mask = tile_pos == pos
                tl.store(kv_out_ptr + kc_base + cache_off
                         + tile_pos[:, None] * D_HEAD + dh[None, :],
                         K_tile.to(tl.bfloat16), mask=pos_mask[:, None])

                V_tile = tl.load(kv_out_ptr + vc_base + cache_off
                                 + tile_pos[:, None] * D_HEAD + dh[None, :],
                                 mask=tile_mask[:, None], other=0.0,
                                 eviction_policy='evict_last').to(tl.float32)
                V_tile = tl.where(tile_pos[:, None] == pos, V_new[None, :], V_tile)
                tl.store(kv_out_ptr + vc_base + cache_off
                         + tile_pos[:, None] * D_HEAD + dh[None, :],
                         V_tile.to(tl.bfloat16), mask=pos_mask[:, None])

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

            wo = tl.load(packed_w_ptr + wo_off + hd[:, None] * D_MODEL + d[None, :],
                          mask=d_mask[None, :], other=0.0).to(tl.bfloat16)
            o_proj = tl.dot(attn_out[None, :].to(tl.bfloat16), wo).to(tl.float32).sum(axis=0)

            tl.store(partial_ptr + pid * D_BLOCK + d, o_proj, mask=d_mask)
            tl.store(attn_ml_ptr + pid * 2, tl.sum(m_i))
            tl.store(attn_ml_ptr + pid * 2 + 1, tl.sum(l_i))

            # Split barrier
            b0 = b_off + layer * 2
            old_cnt = tl.atomic_add(barrier_ptr + b0, 1.0, sem='release', scope='gpu')
            if old_cnt >= TOTAL_BLOCKS - 1:
                _ = tl.atomic_add(barrier_ptr + b0, 0.0, sem='acquire', scope='gpu')
                tl.atomic_add(done_ptr + b0, 1.0, sem='release', scope='gpu')
            while tl.atomic_add(done_ptr + b0, 0.0, sem='acquire', scope='gpu') < 1.0:
                pass

            # ── PHASE 2: Merge attention + LN2 + FFN ──
            attn_total = tl.zeros((D_BLOCK,), dtype=tl.float32)
            for head in tl.range(N_HEADS):
                m_max_val = tl.full((), -1e9, dtype=tl.float32)
                for sv in tl.static_range(KV_SPLITS):
                    m_s = tl.load(attn_ml_ptr + (head * KV_SPLITS + sv) * 2)
                    m_max_val = tl.maximum(m_max_val, m_s)
                l_total = tl.full((), 0.0, dtype=tl.float32)
                o_merged = tl.zeros((D_BLOCK,), dtype=tl.float32)
                for sv in tl.static_range(KV_SPLITS):
                    m_s = tl.load(attn_ml_ptr + (head * KV_SPLITS + sv) * 2)
                    l_s = tl.load(attn_ml_ptr + (head * KV_SPLITS + sv) * 2 + 1)
                    o_s = tl.load(partial_ptr + (head * KV_SPLITS + sv) * D_BLOCK + d, mask=d_mask, other=0.0)
                    w_val = l_s * tl.exp(m_s - m_max_val)
                    l_total = l_total + w_val
                    o_merged = o_merged + o_s * w_val
                attn_total = attn_total + o_merged / l_total
            h = h + attn_total

            ln_s = tl.load(packed_w_ptr + ln2_s_off + d, mask=d_mask, other=0.0).to(tl.float32)
            ln_b = tl.load(packed_w_ptr + ln2_b_off + d, mask=d_mask, other=0.0).to(tl.float32)
            mean = tl.sum(h) / D_MODEL
            hc = tl.where(d_mask, h - mean, 0.0)
            h_norm = tl.where(d_mask,
                              ln_s * hc * tl.math.rsqrt(tl.sum(hc * hc) / D_MODEL + 1e-5) + ln_b,
                              0.0)
            h_norm_2d = h_norm[None, :].to(tl.bfloat16)

            ff_start = pid * FF_PER_BLOCK
            ffn_partial = tl.zeros((D_BLOCK,), dtype=tl.float32)
            for k in tl.range(0, FF_PER_BLOCK, BLOCK_K):
                kk = ff_start + k + tl.arange(0, BLOCK_K)
                up_w = tl.load(packed_w_ptr + up_off + d[:, None] * D_FF + kk[None, :],
                               mask=d_mask[:, None], other=0.0).to(tl.bfloat16)
                up = tl.dot(h_norm_2d, up_w).to(tl.float32).sum(axis=0)
                act = up * tl.sigmoid(1.702 * up)
                down_w = tl.load(packed_w_ptr + down_off + kk[:, None] * D_MODEL + d[None, :],
                                 mask=d_mask[None, :], other=0.0).to(tl.bfloat16)
                ffn_partial += tl.dot(act[None, :].to(tl.bfloat16), down_w).to(tl.float32).sum(axis=0)

            tl.store(partial_ptr + pid * D_BLOCK + d, ffn_partial, mask=d_mask)

            b1 = b_off + layer * 2 + 1
            old_cnt = tl.atomic_add(barrier_ptr + b1, 1.0, sem='release', scope='gpu')
            if old_cnt >= TOTAL_BLOCKS - 1:
                _ = tl.atomic_add(barrier_ptr + b1, 0.0, sem='acquire', scope='gpu')
                tl.atomic_add(done_ptr + b1, 1.0, sem='release', scope='gpu')
            while tl.atomic_add(done_ptr + b1, 0.0, sem='acquire', scope='gpu') < 1.0:
                pass

            # ── PHASE 3: Reduce FFN + residual ──
            ffn_total = tl.zeros((D_BLOCK,), dtype=tl.float32)
            for i in tl.range(TOTAL_BLOCKS):
                ffn_total += tl.load(partial_ptr + i * D_BLOCK + d, mask=d_mask, other=0.0)
            h = h + ffn_total

        # ── OUTPUT: Final LN + tiled output projection + argmax ──
        ln_s = tl.load(lnf_s_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
        ln_b = tl.load(lnf_b_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
        mean = tl.sum(h) / D_MODEL
        hc = tl.where(d_mask, h - mean, 0.0)
        h_final = tl.where(d_mask,
                           ln_s * hc * tl.math.rsqrt(tl.sum(hc * hc) / D_MODEL + 1e-5) + ln_b,
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
            # Only store logits on last step
            if step == N_STEPS - 1:
                tl.store(logits_out_ptr + vv, tile_logits)
            tile_max = tl.max(tile_logits)
            if tile_max > best_val:
                best_val = tile_max
                best_idx = (v_start + tl.argmax(tile_logits, axis=0)).to(tl.float32)

        tl.store(argmax_ptr + pid * 2, best_val)
        tl.store(argmax_ptr + pid * 2 + 1, best_idx)

        # ── Combined output + step-sync barrier ──
        # Last-arriving block knows all argmax values are written,
        # so it does argmax reduction + writes next_tok before signaling.
        bf = b_off + 2 * N_LAYERS
        old_cnt = tl.atomic_add(barrier_ptr + bf, 1.0, sem='release', scope='gpu')
        if old_cnt >= TOTAL_BLOCKS - 1:
            _ = tl.atomic_add(barrier_ptr + bf, 0.0, sem='acquire', scope='gpu')
            # Reduce argmax (this block arrived last → all values are ready)
            global_best_val = -1e9
            global_best_idx = 0.0
            for i in tl.range(TOTAL_BLOCKS):
                v = tl.load(argmax_ptr + i * 2)
                idx = tl.load(argmax_ptr + i * 2 + 1)
                if v > global_best_val:
                    global_best_val = v
                    global_best_idx = idx
            next_tok = global_best_idx.to(tl.int32)
            tl.store(next_tok_ptr, next_tok)
            tl.store(token_out_ptr + step, next_tok)
            tl.atomic_add(done_ptr + bf, 1.0, sem='release', scope='gpu')
        while tl.atomic_add(done_ptr + bf, 0.0, sem='acquire', scope='gpu') < 1.0:
            pass

        # Load next token for next iteration
        token_id = tl.load(next_tok_ptr).to(tl.int32)


def _next_power_of_2(n):
    """Return smallest power of 2 >= n."""
    p = 1
    while p < n:
        p *= 2
    return p


# ──────────────────────────────────────────────────────────────────────

def persistent_decode_nlayer(w, config, first_token, start_pos, kv_packed,
                             vocab_size, n_steps, kv_splits=2):
    """Persistent multi-SM decode: single kernel launch generates n_steps tokens.

    No per-step host overhead. In-place KV cache updates.

    Args:
        w: precomputed weights from prepare_decode_weights_nlayer()
        config: model config dict
        first_token: int — first token to decode
        start_pos: int — starting position in the sequence
        kv_packed: (total_kv_size,) bf16 — initial KV cache (modified in-place)
        vocab_size: actual vocabulary size
        n_steps: number of decode steps
        kv_splits: KV-split parallelism factor (default 2)

    Returns:
        tokens: (n_steps,) int32 — generated tokens
        logits: (vocab_size,) f32 — last step's logits
        kv_out: (total_kv_size,) bf16 — final KV cache
    """
    d_model = config["d_model"]
    d_block = _next_power_of_2(d_model)
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

    # Barriers per step: 2 per layer + 1 combined output/step-sync
    barriers_per_step = 2 * n_layers + 1
    total_barrier_slots = 1 + n_steps * barriers_per_step  # +1 for initial KV copy

    # Workspace layout (all f32):
    #   partial:    TOTAL_BLOCKS * D_BLOCK  (reused each step for attn o_proj then FFN)
    #   attn_ml:    TOTAL_BLOCKS * 2        (m, l for KV-split merge)
    #   barriers:   total_barrier_slots     (one-time, fresh slots per step)
    #   done:       total_barrier_slots     (one-time, fresh slots per step)
    #   argmax:     TOTAL_BLOCKS * 2        (reused each step)
    #   next_tok:   1                       (shared token for next step)
    partial_off = 0
    attn_ml_off = partial_off + total_blocks * d_block
    barrier_off = attn_ml_off + total_blocks * 2
    done_off = barrier_off + total_barrier_slots
    argmax_off = done_off + total_barrier_slots
    next_tok_off = argmax_off + 2 * total_blocks
    workspace_size = next_tok_off + 1

    workspace = jnp.zeros((workspace_size,), dtype=jnp.float32)

    token_out, logits_pad, kv_out = jt.triton_call(
        w["token_emb"], w["pos_emb"],
        w["packed_w"],
        w["lnf_s"], w["lnf_b"],
        w["output_proj_padded"],
        jnp.int32(first_token), jnp.int32(start_pos),
        kv_packed,
        workspace,
        kernel=_persistent_decode,
        out_shape=[
            jax.ShapeDtypeStruct((n_steps,), jnp.int32),
            jax.ShapeDtypeStruct((vocab_pad,), jnp.float32),
            jax.ShapeDtypeStruct((total_kv_size,), jnp.bfloat16),
        ],
        grid=(total_blocks,),
        num_warps=4, num_stages=1,
        D_MODEL=d_model, D_BLOCK=d_block, D_HEAD=d_head, D_FF=d_ff,
        N_HEADS=n_heads, N_KV_HEADS=n_kv_heads, D_KV=d_kv,
        N_LAYERS=n_layers, MAX_SEQ=max_seq,
        KV_SPLITS=kv_splits, TOTAL_BLOCKS=total_blocks,
        VOCAB_SIZE=vocab_size, VOCAB_PAD=vocab_pad,
        FF_PER_BLOCK=ff_per_block,
        N_STEPS=n_steps,
        BARRIERS_PER_STEP=barriers_per_step,
        PARTIAL_OFF=partial_off,
        ATTN_ML_OFF=attn_ml_off,
        BARRIER_OFF=barrier_off,
        DONE_OFF=done_off,
        ARGMAX_OFF=argmax_off,
        NEXT_TOK_OFF=next_tok_off,
    )

    return token_out, logits_pad[:vocab_size], kv_out
