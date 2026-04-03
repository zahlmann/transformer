"""
Batched multi-SM decode kernel.

Processes B sequences in parallel with shared weight loads.
Grid = (N_HEADS * KV_SPLITS,) — same as single-sequence multi-SM kernel.
Each block handles all B sequences: loads weights once, applies to B hidden states.

Weight amortization:
  - Q/K/V/O projections: load (D_MODEL, D_HEAD) once, apply to B sequences
  - FFN up/down: load per k-tile once, apply to B sequences
  - Output projection: load per tile once, apply to B sequences

Per-sequence state:
  - KV caches: B independent caches, each at different positions
  - Hidden states: B vectors stored in workspace between phases
  - Attention: computed independently per sequence with own KV cache

Supports non-power-of-2 D_MODEL via D_BLOCK padding: tl.arange uses D_BLOCK
(next power of 2) with d_mask = d < D_MODEL for all loads/stores.

Workspace layout (all f32):
  h_buf:       2 * B * D_BLOCK                   -- double-buffered hidden states (avoids read-write race)
  partial:     B * TOTAL_BLOCKS * D_BLOCK        -- attention o_proj (phase 1 only)
  ffn_buf:     B * TOTAL_BLOCKS * D_BLOCK        -- FFN partials (phase 2 only)
  attn_buf:    B * D_BLOCK                      -- attention residual (written phase 2, read phase 3)
  h_norm_buf:  B * D_BLOCK                      -- LN1 output (phase 1 only, separate from attn_buf)
  attn_ml:     B * TOTAL_BLOCKS * 2              -- (m, l) for KV-split merge
  qkv_tmp:     TOTAL_BLOCKS * 3 * B * D_HEAD    -- per-block Q/K_new/V_new/attn_out
  barrier:     32                               -- arrival counter (cache line)
  done:        32                               -- done flag (cache line)
  argmax:      B * TOTAL_BLOCKS * 2             -- per-block per-batch argmax state

RACE-FREE DESIGN:
  - h_buf is ONLY READ in phases 1 and 2, ONLY WRITTEN in phase 3.
    This prevents the h_buf read-write race where a fast block could write
    h + attn_total before a slow block reads the original h.
  - partial and ffn_buf are separate buffers (no merge/FFN overlap).
  - attn_buf stores the attention residual (all blocks write same value).
  - h_norm is kept in REGISTERS (fused merge+LN2+FFN per batch element).
"""

import triton
import triton.language as tl
import jax
import jax.numpy as jnp
import jax_triton as jt

BLOCK_K      = tl.constexpr(16)
KV_TILE      = tl.constexpr(64)
OUTPUT_VTILE = tl.constexpr(32)
PROJ_TILE    = tl.constexpr(512)   # tile Q/K/V/O projections when D_BLOCK*D_HEAD > 64KB


@triton.jit
def _batched_decode(
    # Inputs
    token_emb_ptr, pos_emb_ptr,
    packed_w_ptr,
    lnf_s_ptr, lnf_b_ptr,
    output_proj_ptr,
    token_ids_ptr,        # (B,) int32
    positions_ptr,        # (B,) int32
    kv_in_ptr,            # (B * TOTAL_KV_SIZE,) bf16
    workspace_ptr,
    # Outputs
    logits_ptr,           # (B * VOCAB_PAD,) f32
    kv_out_ptr,           # (B * TOTAL_KV_SIZE,) bf16
    next_tokens_ptr,      # (B,) int32
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
    BATCH_SIZE: tl.constexpr,
    TOTAL_KV_SIZE: tl.constexpr,
    # Workspace offsets
    H_BUF_OFF: tl.constexpr,
    PARTIAL_OFF: tl.constexpr,
    FFN_BUF_OFF: tl.constexpr,
    ATTN_ML_OFF: tl.constexpr,
    ATTN_BUF_OFF: tl.constexpr,
    H_NORM_OFF: tl.constexpr,
    QKV_TMP_OFF: tl.constexpr,
    BARRIER_OFF: tl.constexpr,
    DONE_OFF: tl.constexpr,
    ARGMAX_OFF: tl.constexpr,
):
    pid = tl.program_id(0)
    head_id = pid // KV_SPLITS
    kv_split = pid % KV_SPLITS
    d = tl.arange(0, D_BLOCK)
    d_mask = d < D_MODEL
    dh = tl.arange(0, D_HEAD)

    # Workspace pointers
    h_buf_ptr = workspace_ptr + H_BUF_OFF
    partial_ptr = workspace_ptr + PARTIAL_OFF
    ffn_buf_ptr = workspace_ptr + FFN_BUF_OFF
    attn_buf_ptr = workspace_ptr + ATTN_BUF_OFF
    h_norm_ptr = workspace_ptr + H_NORM_OFF
    attn_ml_ptr = workspace_ptr + ATTN_ML_OFF
    qkv_tmp_ptr = workspace_ptr + QKV_TMP_OFF + pid * 3 * BATCH_SIZE * D_HEAD
    barrier_ptr = workspace_ptr + BARRIER_OFF
    done_ptr = workspace_ptr + DONE_OFF
    argmax_ptr = workspace_ptr + ARGMAX_OFF

    # ── Embedding for all B sequences (write to buf_a = offset 0) ──
    H_BUF_SIZE: tl.constexpr = BATCH_SIZE * D_BLOCK
    for b in tl.range(0, BATCH_SIZE):
        token_id = tl.load(token_ids_ptr + b)
        pos = tl.load(positions_ptr + b)
        h = (tl.load(token_emb_ptr + token_id * D_MODEL + d, mask=d_mask, other=0.0).to(tl.float32)
           + tl.load(pos_emb_ptr + pos * D_MODEL + d, mask=d_mask, other=0.0).to(tl.float32))
        tl.store(h_buf_ptr + b * D_BLOCK + d, h, mask=d_mask)

    # ── Layer constants ──
    LAYER_W_SIZE: tl.constexpr = (
        D_MODEL + D_MODEL +
        D_MODEL * D_MODEL + D_MODEL * D_KV + D_MODEL * D_KV + D_MODEL * D_MODEL +
        D_MODEL + D_MODEL +
        D_MODEL * D_FF + D_FF * D_MODEL
    )
    LAYER_KV_SIZE: tl.constexpr = 2 * N_KV_HEADS * MAX_SEQ * D_HEAD

    scale = 0.17677669529663689   # 1/sqrt(32)
    GQA_GROUP: tl.constexpr = N_HEADS // N_KV_HEADS

    hd = head_id * D_HEAD + dh
    kv_head = head_id // GQA_GROUP
    kv_hd = kv_head * D_HEAD + dh
    cache_off = kv_head * MAX_SEQ * D_HEAD

    POS_PER_SPLIT: tl.constexpr = MAX_SEQ // KV_SPLITS
    kv_start = kv_split * POS_PER_SPLIT
    kv_end = kv_start + POS_PER_SPLIT

    for layer in tl.range(N_LAYERS):
        # Double-buffer: even layers read buf_a write buf_b, odd layers vice versa
        h_in_off = (layer % 2) * H_BUF_SIZE
        h_out_off = ((layer + 1) % 2) * H_BUF_SIZE

        w_base = layer * LAYER_W_SIZE
        kv_base = layer * LAYER_KV_SIZE
        kc_base = kv_base
        vc_base = kv_base + N_KV_HEADS * MAX_SEQ * D_HEAD

        # Weight offsets
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
        down_off = off

        # ══════════════════════════════════════════════════
        # PHASE 1: LN1 + Q/K/V projections + Attention + O projection
        # h_norm stored to per-block qkv_tmp section (race-free: each block
        # only reads its own writes). Weights loaded one at a time.
        # ══════════════════════════════════════════════════

        # 1a. LN1 for all B — store to h_norm_buf (separate from attn_buf)
        ln_s = tl.load(packed_w_ptr + ln1_s_off + d, mask=d_mask, other=0.0).to(tl.float32)
        ln_b = tl.load(packed_w_ptr + ln1_b_off + d, mask=d_mask, other=0.0).to(tl.float32)
        for b in tl.range(0, BATCH_SIZE):
            h = tl.load(h_buf_ptr + h_in_off + b * D_BLOCK + d, mask=d_mask, other=0.0)
            mean = tl.sum(h) / D_MODEL
            hc = tl.where(d_mask, h - mean, 0.0)
            h_norm = tl.where(d_mask,
                              ln_s * hc * tl.math.rsqrt(tl.sum(hc * hc) / D_MODEL + 1e-5) + ln_b,
                              0.0)
            tl.store(h_norm_ptr + b * D_BLOCK + d, h_norm, mask=d_mask)

        # 1b-d. Tiled Q/K/V projections along D_BLOCK to fit in shared memory.
        # At D_BLOCK=1024, D_HEAD=64: full (1024,64) bf16 = 128KB > 101KB smem.
        # PROJ_TILE=512: (512,64) bf16 = 64KB, fits.
        # Load one weight matrix tile at a time, apply to all B sequences.

        # Q projection (tiled)
        for b in tl.range(0, BATCH_SIZE):
            tl.store(qkv_tmp_ptr + b * D_HEAD + dh, tl.zeros((D_HEAD,), dtype=tl.float32))
        for dd in tl.static_range(0, D_BLOCK, PROJ_TILE):
            dt = dd + tl.arange(0, PROJ_TILE)
            dt_mask = dt < D_MODEL
            wq_t = tl.load(packed_w_ptr + wq_off + dt[:, None] * D_MODEL + hd[None, :],
                           mask=dt_mask[:, None], other=0.0).to(tl.bfloat16)
            for b in tl.range(0, BATCH_SIZE):
                h_tile = tl.load(h_norm_ptr + b * D_BLOCK + dt,
                                 mask=dt_mask, other=0.0).to(tl.bfloat16)
                Q_acc = tl.load(qkv_tmp_ptr + b * D_HEAD + dh)
                Q_acc += tl.dot(h_tile[None, :], wq_t).to(tl.float32).sum(axis=0)
                tl.store(qkv_tmp_ptr + b * D_HEAD + dh, Q_acc)

        # K projection (tiled) + store to KV cache
        for b in tl.range(0, BATCH_SIZE):
            tl.store(qkv_tmp_ptr + (BATCH_SIZE + b) * D_HEAD + dh, tl.zeros((D_HEAD,), dtype=tl.float32))
        for dd in tl.static_range(0, D_BLOCK, PROJ_TILE):
            dt = dd + tl.arange(0, PROJ_TILE)
            dt_mask = dt < D_MODEL
            wk_t = tl.load(packed_w_ptr + wk_off + dt[:, None] * D_KV + kv_hd[None, :],
                           mask=dt_mask[:, None], other=0.0).to(tl.bfloat16)
            for b in tl.range(0, BATCH_SIZE):
                h_tile = tl.load(h_norm_ptr + b * D_BLOCK + dt,
                                 mask=dt_mask, other=0.0).to(tl.bfloat16)
                K_acc = tl.load(qkv_tmp_ptr + (BATCH_SIZE + b) * D_HEAD + dh)
                K_acc += tl.dot(h_tile[None, :], wk_t).to(tl.float32).sum(axis=0)
                tl.store(qkv_tmp_ptr + (BATCH_SIZE + b) * D_HEAD + dh, K_acc)
        for b in tl.range(0, BATCH_SIZE):
            pos_b = tl.load(positions_ptr + b)
            kv_b = b * TOTAL_KV_SIZE
            K_new = tl.load(qkv_tmp_ptr + (BATCH_SIZE + b) * D_HEAD + dh)
            tl.store(kv_out_ptr + kv_b + kc_base + cache_off + pos_b * D_HEAD + dh,
                     K_new.to(tl.bfloat16))

        # V projection (tiled) + store to KV cache
        for b in tl.range(0, BATCH_SIZE):
            tl.store(qkv_tmp_ptr + (2 * BATCH_SIZE + b) * D_HEAD + dh, tl.zeros((D_HEAD,), dtype=tl.float32))
        for dd in tl.static_range(0, D_BLOCK, PROJ_TILE):
            dt = dd + tl.arange(0, PROJ_TILE)
            dt_mask = dt < D_MODEL
            wv_t = tl.load(packed_w_ptr + wv_off + dt[:, None] * D_KV + kv_hd[None, :],
                           mask=dt_mask[:, None], other=0.0).to(tl.bfloat16)
            for b in tl.range(0, BATCH_SIZE):
                h_tile = tl.load(h_norm_ptr + b * D_BLOCK + dt,
                                 mask=dt_mask, other=0.0).to(tl.bfloat16)
                V_acc = tl.load(qkv_tmp_ptr + (2 * BATCH_SIZE + b) * D_HEAD + dh)
                V_acc += tl.dot(h_tile[None, :], wv_t).to(tl.float32).sum(axis=0)
                tl.store(qkv_tmp_ptr + (2 * BATCH_SIZE + b) * D_HEAD + dh, V_acc)
        for b in tl.range(0, BATCH_SIZE):
            pos_b = tl.load(positions_ptr + b)
            kv_b = b * TOTAL_KV_SIZE
            V_new = tl.load(qkv_tmp_ptr + (2 * BATCH_SIZE + b) * D_HEAD + dh)
            tl.store(kv_out_ptr + kv_b + vc_base + cache_off + pos_b * D_HEAD + dh,
                     V_new.to(tl.bfloat16))

        # 1e. Attention — per-sequence, no weight loads needed
        for b in tl.range(0, BATCH_SIZE):
            Q = tl.load(qkv_tmp_ptr + b * D_HEAD + dh)
            K_new = tl.load(qkv_tmp_ptr + (BATCH_SIZE + b) * D_HEAD + dh)
            V_new = tl.load(qkv_tmp_ptr + (2 * BATCH_SIZE + b) * D_HEAD + dh)
            pos_b = tl.load(positions_ptr + b)
            kv_b = b * TOTAL_KV_SIZE

            m_i = tl.full((1,), value=-1e9, dtype=tl.float32)
            l_i = tl.zeros((1,), dtype=tl.float32)
            o_i = tl.zeros((D_HEAD,), dtype=tl.float32)

            for t in tl.range(kv_start, kv_end, KV_TILE):
                tile_pos = t + tl.arange(0, KV_TILE)
                tile_mask = tile_pos <= pos_b

                K_tile = tl.load(kv_in_ptr + kv_b + kc_base + cache_off
                                 + tile_pos[:, None] * D_HEAD + dh[None, :],
                                 mask=tile_mask[:, None], other=0.0,
                                 eviction_policy='evict_last').to(tl.float32)
                K_tile = tl.where(tile_pos[:, None] == pos_b, K_new[None, :], K_tile)
                tl.store(kv_out_ptr + kv_b + kc_base + cache_off
                         + tile_pos[:, None] * D_HEAD + dh[None, :],
                         K_tile.to(tl.bfloat16), mask=tile_mask[:, None])

                V_tile = tl.load(kv_in_ptr + kv_b + vc_base + cache_off
                                 + tile_pos[:, None] * D_HEAD + dh[None, :],
                                 mask=tile_mask[:, None], other=0.0,
                                 eviction_policy='evict_last').to(tl.float32)
                V_tile = tl.where(tile_pos[:, None] == pos_b, V_new[None, :], V_tile)
                tl.store(kv_out_ptr + kv_b + vc_base + cache_off
                         + tile_pos[:, None] * D_HEAD + dh[None, :],
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

            # Reuse Q slot for attn_out
            tl.store(qkv_tmp_ptr + b * D_HEAD + dh, attn_out)
            tl.store(attn_ml_ptr + (b * TOTAL_BLOCKS + pid) * 2, tl.sum(m_i))
            tl.store(attn_ml_ptr + (b * TOTAL_BLOCKS + pid) * 2 + 1, tl.sum(l_i))

        # 1f. Tiled O projection along D_BLOCK (same shared memory constraint as Q/K/V)
        for dd in tl.static_range(0, D_BLOCK, PROJ_TILE):
            dt = dd + tl.arange(0, PROJ_TILE)
            dt_mask = dt < D_MODEL
            wo_t = tl.load(packed_w_ptr + wo_off + hd[:, None] * D_MODEL + dt[None, :],
                           mask=dt_mask[None, :], other=0.0).to(tl.bfloat16)
            for b in tl.range(0, BATCH_SIZE):
                attn_out = tl.load(qkv_tmp_ptr + b * D_HEAD + dh)
                o_tile = tl.dot(attn_out[None, :].to(tl.bfloat16), wo_t).to(tl.float32).sum(axis=0)
                tl.store(partial_ptr + (b * TOTAL_BLOCKS + pid) * D_BLOCK + dt, o_tile, mask=dt_mask)

        # ── Split barrier (counter + done flag on separate cache lines) ──
        b0 = layer * 2
        old_cnt = tl.atomic_add(barrier_ptr + b0, 1.0, sem='release', scope='gpu')
        if old_cnt >= TOTAL_BLOCKS - 1:
            _ = tl.atomic_add(barrier_ptr + b0, 0.0, sem='acquire', scope='gpu')
            tl.atomic_add(done_ptr + b0, 1.0, sem='release', scope='gpu')
        while tl.atomic_add(done_ptr + b0, 0.0, sem='acquire', scope='gpu') < 1.0:
            pass

        # ══════════════════════════════════════════════════
        # PHASE 2a: Merge attention + LN2 for all B → store h_norm to buffer
        # ══════════════════════════════════════════════════
        # h_buf is READ-ONLY here (no write until phase 3).
        # attn_total stored to attn_buf (all blocks write same value — benign race).
        # h_norm stored to h_norm_buf for FFN weight amortization.

        ln_s = tl.load(packed_w_ptr + ln2_s_off + d, mask=d_mask, other=0.0).to(tl.float32)
        ln_b = tl.load(packed_w_ptr + ln2_b_off + d, mask=d_mask, other=0.0).to(tl.float32)
        ff_start = pid * FF_PER_BLOCK

        for b in tl.range(0, BATCH_SIZE):
            h = tl.load(h_buf_ptr + h_in_off + b * D_BLOCK + d, mask=d_mask, other=0.0)

            # Merge KV splits → attn_total
            attn_total = tl.zeros((D_BLOCK,), dtype=tl.float32)
            for head in tl.range(N_HEADS):
                m_max_val = tl.full((), -1e9, dtype=tl.float32)
                for sv in tl.static_range(KV_SPLITS):
                    m_s = tl.load(attn_ml_ptr + (b * TOTAL_BLOCKS + head * KV_SPLITS + sv) * 2)
                    m_max_val = tl.maximum(m_max_val, m_s)
                l_total = tl.full((), 0.0, dtype=tl.float32)
                o_merged = tl.zeros((D_BLOCK,), dtype=tl.float32)
                for sv in tl.static_range(KV_SPLITS):
                    m_s = tl.load(attn_ml_ptr + (b * TOTAL_BLOCKS + head * KV_SPLITS + sv) * 2)
                    l_s = tl.load(attn_ml_ptr + (b * TOTAL_BLOCKS + head * KV_SPLITS + sv) * 2 + 1)
                    o_s = tl.load(partial_ptr + (b * TOTAL_BLOCKS + head * KV_SPLITS + sv) * D_BLOCK + d,
                                  mask=d_mask, other=0.0)
                    w_val = l_s * tl.exp(m_s - m_max_val)
                    l_total = l_total + w_val
                    o_merged = o_merged + o_s * w_val
                attn_total = attn_total + o_merged / l_total

            # Store attn_total for phase 3 (all blocks write same value)
            tl.store(attn_buf_ptr + b * D_BLOCK + d, attn_total, mask=d_mask)

            # LN2 on (h + attn_total) → store h_norm for weight-amortized FFN
            h_attn = h + attn_total
            mean = tl.sum(h_attn) / D_MODEL
            hc = tl.where(d_mask, h_attn - mean, 0.0)
            h_norm = tl.where(d_mask,
                              ln_s * hc * tl.math.rsqrt(tl.sum(hc * hc) / D_MODEL + 1e-5) + ln_b,
                              0.0)
            tl.store(h_norm_ptr + b * D_BLOCK + d, h_norm, mask=d_mask)

        # ══════════════════════════════════════════════════
        # PHASE 2b: FFN with weight amortization — outer k-loop, inner b-loop
        # ══════════════════════════════════════════════════
        # Load FFN weights once per k-tile, apply to all B sequences.
        # Saves (B-1) * weight_size L2 traffic per tile.

        for k in tl.range(0, FF_PER_BLOCK, BLOCK_K):
            kk = ff_start + k + tl.arange(0, BLOCK_K)
            up_w = tl.load(packed_w_ptr + up_off + d[:, None] * D_FF + kk[None, :],
                           mask=d_mask[:, None], other=0.0).to(tl.bfloat16)
            down_w = tl.load(packed_w_ptr + down_off + kk[:, None] * D_MODEL + d[None, :],
                             mask=d_mask[None, :], other=0.0).to(tl.bfloat16)

            for b in tl.range(0, BATCH_SIZE):
                h_norm = tl.load(h_norm_ptr + b * D_BLOCK + d, mask=d_mask, other=0.0)
                h_norm_2d = h_norm[None, :].to(tl.bfloat16)
                up = tl.dot(h_norm_2d, up_w).to(tl.float32).sum(axis=0)
                act = up * tl.sigmoid(1.702 * up)
                partial = tl.dot(act[None, :].to(tl.bfloat16), down_w).to(tl.float32).sum(axis=0)

                if k == 0:
                    tl.store(ffn_buf_ptr + (b * TOTAL_BLOCKS + pid) * D_BLOCK + d, partial, mask=d_mask)
                else:
                    existing = tl.load(ffn_buf_ptr + (b * TOTAL_BLOCKS + pid) * D_BLOCK + d,
                                       mask=d_mask, other=0.0)
                    tl.store(ffn_buf_ptr + (b * TOTAL_BLOCKS + pid) * D_BLOCK + d,
                             existing + partial, mask=d_mask)

        # ── Barrier ──
        b1 = layer * 2 + 1
        old_cnt = tl.atomic_add(barrier_ptr + b1, 1.0, sem='release', scope='gpu')
        if old_cnt >= TOTAL_BLOCKS - 1:
            _ = tl.atomic_add(barrier_ptr + b1, 0.0, sem='acquire', scope='gpu')
            tl.atomic_add(done_ptr + b1, 1.0, sem='release', scope='gpu')
        while tl.atomic_add(done_ptr + b1, 0.0, sem='acquire', scope='gpu') < 1.0:
            pass

        # ══════════════════════════════════════════════════
        # PHASE 3: h = h_original + attn_total + ffn_total
        # Double-buffer: read from h_in, write to h_out (no read-write race).
        # ══════════════════════════════════════════════════
        for b in tl.range(0, BATCH_SIZE):
            h = tl.load(h_buf_ptr + h_in_off + b * D_BLOCK + d, mask=d_mask, other=0.0)
            attn_total = tl.load(attn_buf_ptr + b * D_BLOCK + d, mask=d_mask, other=0.0)
            ffn_total = tl.zeros((D_BLOCK,), dtype=tl.float32)
            for i in tl.range(TOTAL_BLOCKS):
                ffn_total += tl.load(ffn_buf_ptr + (b * TOTAL_BLOCKS + i) * D_BLOCK + d,
                                     mask=d_mask, other=0.0)
            h = h + attn_total + ffn_total
            tl.store(h_buf_ptr + h_out_off + b * D_BLOCK + d, h, mask=d_mask)

        # No barrier needed after phase 3: all blocks write identical h values
        # (redundant reduce), so each block can safely read its own writes in the
        # next layer's phase 1 without waiting for other blocks.

    # ══════════════════════════════════════════════════
    # OUTPUT: Final LN + tiled output projection + argmax
    # ══════════════════════════════════════════════════
    ln_s = tl.load(lnf_s_ptr + d, mask=d_mask, other=0.0).to(tl.float32)
    ln_b = tl.load(lnf_b_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

    TILES_PER_BLOCK: tl.constexpr = VOCAB_PAD // (OUTPUT_VTILE * TOTAL_BLOCKS)
    # After N_LAYERS, final h is in the output buffer of the last layer
    h_final_off = (N_LAYERS % 2) * H_BUF_SIZE

    for b in tl.range(0, BATCH_SIZE):
        h = tl.load(h_buf_ptr + h_final_off + b * D_BLOCK + d, mask=d_mask, other=0.0)
        mean = tl.sum(h) / D_MODEL
        hc = tl.where(d_mask, h - mean, 0.0)
        h_final = tl.where(d_mask,
                           ln_s * hc * tl.math.rsqrt(tl.sum(hc * hc) / D_MODEL + 1e-5) + ln_b,
                           0.0)
        h_final_2d = h_final[None, :].to(tl.bfloat16)

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
            tl.store(logits_ptr + b * VOCAB_PAD + vv, tile_logits)
            tile_max = tl.max(tile_logits)
            if tile_max > best_val:
                best_val = tile_max
                best_idx = (v_start + tl.argmax(tile_logits, axis=0)).to(tl.float32)

        tl.store(argmax_ptr + (b * TOTAL_BLOCKS + pid) * 2, best_val)
        tl.store(argmax_ptr + (b * TOTAL_BLOCKS + pid) * 2 + 1, best_idx)

    # ── Final barrier ──
    N_BARRIERS: tl.constexpr = 2 * N_LAYERS
    old_cnt = tl.atomic_add(barrier_ptr + N_BARRIERS, 1.0, sem='release', scope='gpu')
    if old_cnt >= TOTAL_BLOCKS - 1:
        _ = tl.atomic_add(barrier_ptr + N_BARRIERS, 0.0, sem='acquire', scope='gpu')
        tl.atomic_add(done_ptr + N_BARRIERS, 1.0, sem='release', scope='gpu')
    while tl.atomic_add(done_ptr + N_BARRIERS, 0.0, sem='acquire', scope='gpu') < 1.0:
        pass

    # Block 0 reduces argmax across all blocks for each batch element
    if pid == 0:
        for b in tl.range(0, BATCH_SIZE):
            global_best_val = -1e9
            global_best_idx = 0.0
            for i in tl.range(TOTAL_BLOCKS):
                v = tl.load(argmax_ptr + (b * TOTAL_BLOCKS + i) * 2)
                idx = tl.load(argmax_ptr + (b * TOTAL_BLOCKS + i) * 2 + 1)
                if v > global_best_val:
                    global_best_val = v
                    global_best_idx = idx
            tl.store(next_tokens_ptr + b, global_best_idx.to(tl.int32))


def _next_power_of_2(n):
    """Return smallest power of 2 >= n."""
    p = 1
    while p < n:
        p *= 2
    return p


# ──────────────────────────────────────────────────────────────────────

def batched_decode_nlayer(w, config, token_ids, positions, kv_packed_batch,
                          vocab_size, batch_size, kv_splits=2):
    """Batched multi-SM decode: B sequences per kernel launch, shared weight loads.

    Args:
        w: precomputed weights from prepare_decode_weights_nlayer()
        config: model config dict
        token_ids: (B,) int32 — current token per sequence
        positions: (B,) int32 — current position per sequence
        kv_packed_batch: (B * total_kv_per_seq,) bf16 — concatenated KV caches
        vocab_size: actual vocabulary size
        batch_size: number of sequences (B)
        kv_splits: KV-split parallelism factor (default 2)

    Returns:
        next_tokens: (B,) int32
        logits: (B, vocab_size) f32
        kv_out: (B * total_kv_per_seq,) bf16
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
    total_kv_size = n_layers * 2 * n_kv_heads * max_seq * d_head  # per sequence
    total_blocks = n_heads * kv_splits
    ff_per_block = d_ff // total_blocks

    # Workspace offsets (all f32) — use d_block for power-of-2 alignment
    h_buf_off = 0
    partial_off = h_buf_off + 2 * batch_size * d_block  # double-buffered h
    ffn_buf_off = partial_off + batch_size * total_blocks * d_block
    attn_buf_off = ffn_buf_off + batch_size * total_blocks * d_block
    h_norm_off = attn_buf_off + batch_size * d_block
    attn_ml_off = h_norm_off + batch_size * d_block
    qkv_tmp_off = attn_ml_off + batch_size * total_blocks * 2
    barrier_off = qkv_tmp_off + total_blocks * 3 * batch_size * d_head
    done_off = barrier_off + 32   # separate cache line
    argmax_off = done_off + 32
    workspace_size = argmax_off + batch_size * total_blocks * 2

    workspace = jnp.zeros((workspace_size,), dtype=jnp.float32)

    logits_pad, kv_out, next_tokens = jt.triton_call(
        w["token_emb"], w["pos_emb"],
        w["packed_w"],
        w["lnf_s"], w["lnf_b"],
        w["output_proj_padded"],
        jnp.asarray(token_ids, dtype=jnp.int32),
        jnp.asarray(positions, dtype=jnp.int32),
        kv_packed_batch,
        workspace,
        kernel=_batched_decode,
        out_shape=[
            jax.ShapeDtypeStruct((batch_size * vocab_pad,), jnp.float32),
            jax.ShapeDtypeStruct((batch_size * total_kv_size,), jnp.bfloat16),
            jax.ShapeDtypeStruct((batch_size,), jnp.int32),
        ],
        grid=(total_blocks,),
        num_warps=4, num_stages=1,
        D_MODEL=d_model, D_BLOCK=d_block, D_HEAD=d_head, D_FF=d_ff,
        N_HEADS=n_heads, N_KV_HEADS=n_kv_heads, D_KV=d_kv,
        N_LAYERS=n_layers, MAX_SEQ=max_seq,
        KV_SPLITS=kv_splits, TOTAL_BLOCKS=total_blocks,
        VOCAB_SIZE=vocab_size, VOCAB_PAD=vocab_pad,
        FF_PER_BLOCK=ff_per_block,
        BATCH_SIZE=batch_size,
        TOTAL_KV_SIZE=total_kv_size,
        H_BUF_OFF=h_buf_off,
        PARTIAL_OFF=partial_off,
        FFN_BUF_OFF=ffn_buf_off,
        ATTN_BUF_OFF=attn_buf_off,
        H_NORM_OFF=h_norm_off,
        ATTN_ML_OFF=attn_ml_off,
        QKV_TMP_OFF=qkv_tmp_off,
        BARRIER_OFF=barrier_off,
        DONE_OFF=done_off,
        ARGMAX_OFF=argmax_off,
    )

    # Reshape logits to (B, vocab_size)
    logits = logits_pad.reshape(batch_size, vocab_pad)[:, :vocab_size]

    return next_tokens, logits, kv_out
