"""
Persistent decode: pre-allocate workspace for all steps, eliminate per-step overhead.

Uses the same multi-SM kernel but with:
1. ONE workspace allocation for all N tokens (avoids N × jnp.zeros calls)
2. Step offset passed as parameter (each step gets fresh barrier slots)
3. In-kernel argmax (token stays on device)
4. Deferred token collection (batch int() at the end)
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
def _multi_sm_decode_v2(
    # Inputs
    token_emb_ptr, pos_emb_ptr,
    packed_w_ptr,
    lnf_s_ptr, lnf_b_ptr,
    output_proj_ptr,
    token_id_ptr, pos_ptr,
    kv_in_ptr,
    workspace_ptr,    # large pre-allocated buffer, offset per step
    step_offset_ptr,  # scalar: offset into workspace for this step
    # Outputs
    logits_ptr,
    kv_out_ptr,
    next_token_ptr,
    # Config
    D_MODEL: tl.constexpr,
    D_HEAD: tl.constexpr,
    D_FF: tl.constexpr,
    N_HEADS: tl.constexpr,
    N_LAYERS: tl.constexpr,
    MAX_SEQ: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    VOCAB_PAD: tl.constexpr,
    FF_PER_BLOCK: tl.constexpr,
    BARRIER_OFF: tl.constexpr,
    ARGMAX_OFF: tl.constexpr,
    WS_STRIDE: tl.constexpr,  # elements per step in workspace
):
    pid = tl.program_id(0)
    d = tl.arange(0, D_MODEL)

    token_id = tl.load(token_id_ptr)
    pos = tl.load(pos_ptr)
    ws_off = tl.load(step_offset_ptr)

    # This step's workspace section
    partial_ptr = workspace_ptr + ws_off
    barrier_ptr = workspace_ptr + ws_off + BARRIER_OFF
    argmax_ptr = workspace_ptr + ws_off + ARGMAX_OFF

    h = (tl.load(token_emb_ptr + token_id * D_MODEL + d).to(tl.float32)
       + tl.load(pos_emb_ptr + pos * D_MODEL + d).to(tl.float32))

    LAYER_W_SIZE: tl.constexpr = (
        D_MODEL + D_MODEL +
        4 * D_MODEL * D_MODEL +
        D_MODEL + D_MODEL +
        D_MODEL * D_FF + D_FF +
        D_FF * D_MODEL + D_MODEL
    )
    LAYER_KV_SIZE: tl.constexpr = 2 * N_HEADS * MAX_SEQ * D_HEAD
    N_BARRIERS: tl.constexpr = 2 * N_LAYERS

    scale = 0.17677669529663689
    dh = tl.arange(0, D_HEAD)

    for layer in tl.range(N_LAYERS):
        w_base = layer * LAYER_W_SIZE
        kv_base = layer * LAYER_KV_SIZE
        kc_base = kv_base
        vc_base = kv_base + N_HEADS * MAX_SEQ * D_HEAD

        off = w_base
        ln1_s_off = off;    off += D_MODEL
        ln1_b_off = off;    off += D_MODEL
        wq_off = off;       off += D_MODEL * D_MODEL
        wk_off = off;       off += D_MODEL * D_MODEL
        wv_off = off;       off += D_MODEL * D_MODEL
        wo_off = off;       off += D_MODEL * D_MODEL
        ln2_s_off = off;    off += D_MODEL
        ln2_b_off = off;    off += D_MODEL
        up_off = off;       off += D_MODEL * D_FF
        up_b_off = off;     off += D_FF
        down_off = off;     off += D_FF * D_MODEL
        down_b_off = off

        # LN1 + Attention
        ln_s = tl.load(packed_w_ptr + ln1_s_off + d).to(tl.float32)
        ln_b = tl.load(packed_w_ptr + ln1_b_off + d).to(tl.float32)
        mean = tl.sum(h) / D_MODEL
        hc = h - mean
        h_norm = ln_s * hc * tl.math.rsqrt(tl.sum(hc * hc) / D_MODEL + 1e-5) + ln_b

        head = pid
        hd = head * D_HEAD + dh
        cache_off = head * MAX_SEQ * D_HEAD
        h_norm_2d = h_norm[None, :].to(tl.bfloat16)

        wq = tl.load(packed_w_ptr + wq_off + d[:, None] * D_MODEL + hd[None, :]).to(tl.bfloat16)
        Q = tl.dot(h_norm_2d, wq).to(tl.float32).sum(axis=0)
        wk = tl.load(packed_w_ptr + wk_off + d[:, None] * D_MODEL + hd[None, :]).to(tl.bfloat16)
        K_new = tl.dot(h_norm_2d, wk).to(tl.float32).sum(axis=0)
        wv = tl.load(packed_w_ptr + wv_off + d[:, None] * D_MODEL + hd[None, :]).to(tl.bfloat16)
        V_new = tl.dot(h_norm_2d, wv).to(tl.float32).sum(axis=0)

        tl.store(kv_out_ptr + kc_base + cache_off + pos * D_HEAD + dh, K_new.to(tl.bfloat16))
        tl.store(kv_out_ptr + vc_base + cache_off + pos * D_HEAD + dh, V_new.to(tl.bfloat16))

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
        wo = tl.load(packed_w_ptr + wo_off + hd[:, None] * D_MODEL + d[None, :]).to(tl.bfloat16)
        o_proj = tl.dot(attn_out[None, :].to(tl.bfloat16), wo).to(tl.float32).sum(axis=0)

        tl.store(partial_ptr + pid * D_MODEL + d, o_proj)

        b0 = layer * 2
        tl.atomic_add(barrier_ptr + b0, 1.0, sem='release', scope='gpu')
        while tl.atomic_add(barrier_ptr + b0, 0.0, sem='acquire', scope='gpu') < N_HEADS:
            pass

        # Reduce attention + LN2 + FFN
        attn_total = tl.zeros((D_MODEL,), dtype=tl.float32)
        for i in tl.static_range(N_HEADS):
            attn_total += tl.load(partial_ptr + i * D_MODEL + d)
        h = h + attn_total

        ln_s = tl.load(packed_w_ptr + ln2_s_off + d).to(tl.float32)
        ln_b = tl.load(packed_w_ptr + ln2_b_off + d).to(tl.float32)
        mean = tl.sum(h) / D_MODEL
        hc = h - mean
        h_norm = ln_s * hc * tl.math.rsqrt(tl.sum(hc * hc) / D_MODEL + 1e-5) + ln_b
        h_norm_2d = h_norm[None, :].to(tl.bfloat16)

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

        tl.store(partial_ptr + pid * D_MODEL + d, ffn_partial)

        b1 = layer * 2 + 1
        tl.atomic_add(barrier_ptr + b1, 1.0, sem='release', scope='gpu')
        while tl.atomic_add(barrier_ptr + b1, 0.0, sem='acquire', scope='gpu') < N_HEADS:
            pass

        ffn_total = tl.zeros((D_MODEL,), dtype=tl.float32)
        for i in tl.static_range(N_HEADS):
            ffn_total += tl.load(partial_ptr + i * D_MODEL + d)
        h = h + ffn_total + tl.load(packed_w_ptr + down_b_off + d).to(tl.float32)

    # Output projection + argmax
    ln_s = tl.load(lnf_s_ptr + d).to(tl.float32)
    ln_b = tl.load(lnf_b_ptr + d).to(tl.float32)
    mean = tl.sum(h) / D_MODEL
    hc = h - mean
    h_final = ln_s * hc * tl.math.rsqrt(tl.sum(hc * hc) / D_MODEL + 1e-5) + ln_b
    h_final_2d = h_final[None, :].to(tl.bfloat16)

    TILES_PER_BLOCK: tl.constexpr = VOCAB_PAD // (OUTPUT_VTILE * N_HEADS)
    best_val = -1e9
    best_idx = 0.0
    for tile_idx in tl.range(0, TILES_PER_BLOCK):
        v_start = (pid * TILES_PER_BLOCK + tile_idx) * OUTPUT_VTILE
        vv = v_start + tl.arange(0, OUTPUT_VTILE)
        out_w = tl.load(output_proj_ptr + d[:, None] * VOCAB_PAD + vv[None, :]).to(tl.bfloat16)
        tile_logits = tl.dot(h_final_2d, out_w).to(tl.float32).sum(axis=0)
        tile_logits = tl.where(vv < VOCAB_SIZE, tile_logits, -1e9)
        tl.store(logits_ptr + vv, tile_logits)
        tile_max = tl.max(tile_logits)
        if tile_max > best_val:
            best_val = tile_max
            best_idx = (v_start + tl.argmax(tile_logits, axis=0)).to(tl.float32)

    tl.store(argmax_ptr + pid * 2, best_val)
    tl.store(argmax_ptr + pid * 2 + 1, best_idx)

    tl.atomic_add(barrier_ptr + N_BARRIERS, 1.0, sem='release', scope='gpu')
    while tl.atomic_add(barrier_ptr + N_BARRIERS, 0.0, sem='acquire', scope='gpu') < N_HEADS:
        pass

    if pid == 0:
        global_best_val = -1e9
        global_best_idx = 0.0
        for i in tl.static_range(N_HEADS):
            v = tl.load(argmax_ptr + i * 2)
            idx = tl.load(argmax_ptr + i * 2 + 1)
            if v > global_best_val:
                global_best_val = v
                global_best_idx = idx
        tl.store(next_token_ptr, global_best_idx.to(tl.int32))


# ──────────────────────────────────────────────────────────────────────

def persistent_decode(w, config, first_token, start_pos, kv_packed, vocab_size, n_tokens):
    """Generate n_tokens with pre-allocated workspace (one allocation for all steps).

    Returns: tokens (list of int), final logits (vocab_size,), final kv_packed
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
    barrier_off = n_heads * d_model
    argmax_off = barrier_off + 2 * n_layers + 1
    ws_stride = argmax_off + 2 * n_heads  # elements per step

    # ONE allocation for ALL steps
    workspace = jnp.zeros((ws_stride * n_tokens,), dtype=jnp.float32)

    kv = kv_packed
    tok = first_token
    token_ids = []

    for step in range(n_tokens):
        step_offset = step * ws_stride

        logits_pad, kv, next_token = jt.triton_call(
            w["token_emb"], w["pos_emb"],
            w["packed_w"],
            w["lnf_s"], w["lnf_b"],
            w["output_proj_padded"],
            jnp.int32(tok), jnp.int32(start_pos + step),
            kv,
            workspace,
            jnp.int32(step_offset),
            kernel=_multi_sm_decode_v2,
            out_shape=[
                jax.ShapeDtypeStruct((vocab_pad,), jnp.float32),
                jax.ShapeDtypeStruct((total_kv_size,), jnp.bfloat16),
                jax.ShapeDtypeStruct((1,), jnp.int32),
            ],
            grid=(n_heads,),
            num_warps=4, num_stages=1,
            D_MODEL=d_model, D_HEAD=d_head, D_FF=d_ff,
            N_HEADS=n_heads, N_LAYERS=n_layers, MAX_SEQ=max_seq,
            VOCAB_SIZE=vocab_size, VOCAB_PAD=vocab_pad,
            FF_PER_BLOCK=ff_per_block,
            BARRIER_OFF=barrier_off,
            ARGMAX_OFF=argmax_off,
            WS_STRIDE=ws_stride,
        )
        tok = next_token[0]  # stays on device
        token_ids.append(tok)

    # Batch collection
    tokens = [int(t) for t in token_ids]
    return tokens, logits_pad[:vocab_size], kv
