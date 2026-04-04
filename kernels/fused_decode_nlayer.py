"""
Fully fused N-layer decode kernel: embedding → N layers → output in ONE launch.

Phase C architecture: RMSNorm, RoPE, SwiGLU, no biases, tied embeddings, GQA.

Weight packing layout (all bf16):
  Per layer:
    ln1_scale:  D_MODEL
    wq:         D_MODEL * D_MODEL
    wk:         D_MODEL * D_KV
    wv:         D_MODEL * D_KV
    wo:         D_MODEL * D_MODEL
    ln2_scale:  D_MODEL
    ffn_gate:   D_MODEL * D_FF
    ffn_up:     D_MODEL * D_FF
    ffn_down:   D_FF * D_MODEL

KV cache packing: all layers concatenated.
  Per layer: k_cache (N_KV_HEADS * MAX_SEQ * D_HEAD) + v_cache (same)
  Note: K values in cache have RoPE already applied.
"""

import triton
import triton.language as tl
import jax
import jax.numpy as jnp
import jax_triton as jt

from model import precompute_rope_table

BLOCK_K      = tl.constexpr(32)
VOCAB_TILE   = tl.constexpr(128)
KV_TILE      = tl.constexpr(64)
OUTPUT_VTILE = tl.constexpr(32)


@triton.jit
def _fused_decode_nlayer(
    # Embedding weights
    token_emb_ptr,
    # Packed per-layer weights (all layers concatenated, bf16)
    packed_w_ptr,
    # Final RMSNorm scale
    lnf_s_ptr,
    # Output projection (tied: token_emb.T, padded)
    output_proj_ptr,
    # RoPE tables
    cos_ptr, sin_ptr,
    # Decode inputs
    token_id_ptr, pos_ptr,
    # Packed KV caches: all layers concatenated (bf16)
    kv_in_ptr,
    # Output
    logits_ptr,
    kv_out_ptr,
    # Config
    D_MODEL: tl.constexpr,
    D_HEAD: tl.constexpr,
    D_FF: tl.constexpr,
    N_HEADS: tl.constexpr,
    N_KV_HEADS: tl.constexpr,
    D_KV: tl.constexpr,
    N_LAYERS: tl.constexpr,
    MAX_SEQ: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    VOCAB_PAD: tl.constexpr,
):
    d = tl.arange(0, D_MODEL)
    token_id = tl.load(token_id_ptr)
    pos = tl.load(pos_ptr)

    # ── Embedding (no positional — RoPE is applied in attention) ──
    h = tl.load(token_emb_ptr + token_id * D_MODEL + d).to(tl.float32)

    # Per-layer weight size in elements (bf16)
    LAYER_W_SIZE: tl.constexpr = (
        D_MODEL +                                                        # ln1 scale
        D_MODEL * D_MODEL + D_MODEL * D_KV + D_MODEL * D_KV + D_MODEL * D_MODEL +  # qkvo
        D_MODEL +                                                        # ln2 scale
        D_MODEL * D_FF + D_MODEL * D_FF + D_FF * D_MODEL                # gate, up, down
    )

    # Per-layer KV cache size in elements (bf16)
    LAYER_KV_SIZE: tl.constexpr = 2 * N_KV_HEADS * MAX_SEQ * D_HEAD

    scale = 1.0 / (D_HEAD ** 0.5)
    dh = tl.arange(0, D_HEAD)
    D_HALF: tl.constexpr = D_HEAD // 2
    dh_lo = tl.arange(0, D_HALF)
    dh_hi = D_HALF + tl.arange(0, D_HALF)
    GQA_GROUP: tl.constexpr = N_HEADS // N_KV_HEADS

    for layer in tl.range(N_LAYERS):
        w_base = layer * LAYER_W_SIZE
        kv_base = layer * LAYER_KV_SIZE
        kc_base = kv_base
        vc_base = kv_base + N_KV_HEADS * MAX_SEQ * D_HEAD

        # Weight offsets within this layer
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

        # ── RMSNorm 1 ──
        ln_s = tl.load(packed_w_ptr + ln1_s_off + d).to(tl.float32)
        h_norm = ln_s * h * tl.math.rsqrt(tl.sum(h * h) / D_MODEL + 1e-5)

        # ── Attention with RoPE and GQA ──
        attn_accum = tl.zeros((D_MODEL,), dtype=tl.float32)
        h_norm_2d = h_norm[None, :].to(tl.bfloat16)

        # Load RoPE cos/sin for current position
        cos_val = tl.load(cos_ptr + pos * D_HALF + dh_lo).to(tl.float32)
        sin_val = tl.load(sin_ptr + pos * D_HALF + dh_lo).to(tl.float32)

        for head in tl.range(N_HEADS):
            kv_head = head // GQA_GROUP
            hd_lo = head * D_HEAD + dh_lo
            hd_hi = head * D_HEAD + dh_hi
            kv_hd_lo = kv_head * D_HEAD + dh_lo
            kv_hd_hi = kv_head * D_HEAD + dh_hi
            cache_off = kv_head * MAX_SEQ * D_HEAD

            # Q projection in two halves for RoPE
            wq_lo = tl.load(packed_w_ptr + wq_off + d[:, None] * D_MODEL + hd_lo[None, :]).to(tl.bfloat16)
            q_lo = tl.dot(h_norm_2d, wq_lo).to(tl.float32).sum(axis=0)
            wq_hi = tl.load(packed_w_ptr + wq_off + d[:, None] * D_MODEL + hd_hi[None, :]).to(tl.bfloat16)
            q_hi = tl.dot(h_norm_2d, wq_hi).to(tl.float32).sum(axis=0)

            # RoPE on Q
            Q_lo = q_lo * cos_val - q_hi * sin_val
            Q_hi = q_lo * sin_val + q_hi * cos_val

            # K projection in two halves for RoPE
            wk_lo = tl.load(packed_w_ptr + wk_off + d[:, None] * D_KV + kv_hd_lo[None, :]).to(tl.bfloat16)
            k_lo = tl.dot(h_norm_2d, wk_lo).to(tl.float32).sum(axis=0)
            wk_hi = tl.load(packed_w_ptr + wk_off + d[:, None] * D_KV + kv_hd_hi[None, :]).to(tl.bfloat16)
            k_hi = tl.dot(h_norm_2d, wk_hi).to(tl.float32).sum(axis=0)

            # RoPE on K_new
            K_new_lo = k_lo * cos_val - k_hi * sin_val
            K_new_hi = k_lo * sin_val + k_hi * cos_val

            # V projection (no RoPE)
            kv_hd = kv_head * D_HEAD + dh
            wv = tl.load(packed_w_ptr + wv_off + d[:, None] * D_KV + kv_hd[None, :]).to(tl.bfloat16)
            V_new = tl.dot(h_norm_2d, wv).to(tl.float32).sum(axis=0)

            # Store K_new (with RoPE) and V_new to output cache
            tl.store(kv_out_ptr + kc_base + cache_off + pos * D_HEAD + dh_lo, K_new_lo.to(tl.bfloat16))
            tl.store(kv_out_ptr + kc_base + cache_off + pos * D_HEAD + dh_hi, K_new_hi.to(tl.bfloat16))
            tl.store(kv_out_ptr + vc_base + cache_off + pos * D_HEAD + dh, V_new.to(tl.bfloat16))

            # Online softmax over tiled KV cache
            m_i = tl.full((1,), value=-1e9, dtype=tl.float32)
            l_i = tl.zeros((1,), dtype=tl.float32)
            o_i = tl.zeros((D_HEAD,), dtype=tl.float32)

            for t in tl.range(0, MAX_SEQ, KV_TILE):
                tile_pos = t + tl.arange(0, KV_TILE)
                tile_mask = tile_pos <= pos

                # Load K tile in two halves (RoPE already applied in cache)
                K_tile_lo = tl.load(kv_in_ptr + kc_base + cache_off + tile_pos[:, None] * D_HEAD + dh_lo[None, :],
                                   mask=tile_mask[:, None], other=0.0).to(tl.float32)
                K_tile_hi = tl.load(kv_in_ptr + kc_base + cache_off + tile_pos[:, None] * D_HEAD + dh_hi[None, :],
                                   mask=tile_mask[:, None], other=0.0).to(tl.float32)
                K_tile_lo = tl.where(tile_pos[:, None] == pos, K_new_lo[None, :], K_tile_lo)
                K_tile_hi = tl.where(tile_pos[:, None] == pos, K_new_hi[None, :], K_tile_hi)
                tl.store(kv_out_ptr + kc_base + cache_off + tile_pos[:, None] * D_HEAD + dh_lo[None, :],
                        K_tile_lo.to(tl.bfloat16), mask=tile_mask[:, None])
                tl.store(kv_out_ptr + kc_base + cache_off + tile_pos[:, None] * D_HEAD + dh_hi[None, :],
                        K_tile_hi.to(tl.bfloat16), mask=tile_mask[:, None])

                # V tile (full D_HEAD)
                V_tile = tl.load(kv_in_ptr + vc_base + cache_off + tile_pos[:, None] * D_HEAD + dh[None, :],
                                mask=tile_mask[:, None], other=0.0).to(tl.float32)
                V_tile = tl.where(tile_pos[:, None] == pos, V_new[None, :], V_tile)
                tl.store(kv_out_ptr + vc_base + cache_off + tile_pos[:, None] * D_HEAD + dh[None, :],
                        V_tile.to(tl.bfloat16), mask=tile_mask[:, None])

                # Score = Q_lo · K_lo + Q_hi · K_hi
                s = (tl.sum(Q_lo[None, :] * K_tile_lo, axis=1)
                   + tl.sum(Q_hi[None, :] * K_tile_hi, axis=1)) * scale
                s = tl.where(tile_mask, s, -1e9)

                m_ij = tl.max(s)
                m_new = tl.maximum(m_i, m_ij)
                alpha = tl.exp(m_i - m_new)
                p = tl.exp(s - m_new)
                l_i = l_i * alpha + tl.sum(p)
                o_i = o_i * alpha + tl.sum(p[:, None] * V_tile, axis=0)
                m_i = m_new

            attn_out = o_i / l_i

            # O projection
            hd = head * D_HEAD + dh
            wo = tl.load(packed_w_ptr + wo_off + hd[:, None] * D_MODEL + d[None, :]).to(tl.bfloat16)
            attn_accum += tl.dot(attn_out[None, :].to(tl.bfloat16), wo).to(tl.float32).sum(axis=0)

        h = h + attn_accum

        # ── RMSNorm 2 + SwiGLU FFN ──
        ln_s = tl.load(packed_w_ptr + ln2_s_off + d).to(tl.float32)
        h_norm = ln_s * h * tl.math.rsqrt(tl.sum(h * h) / D_MODEL + 1e-5)
        h_norm_2d = h_norm[None, :].to(tl.bfloat16)

        ffn_accum = tl.zeros((D_MODEL,), dtype=tl.float32)
        for k in tl.range(0, D_FF, BLOCK_K):
            kk = k + tl.arange(0, BLOCK_K)
            # Gate projection
            gate_w = tl.load(packed_w_ptr + gate_off + d[:, None] * D_FF + kk[None, :]).to(tl.bfloat16)
            gate = tl.dot(h_norm_2d, gate_w).to(tl.float32).sum(axis=0)
            # Up projection
            up_w = tl.load(packed_w_ptr + up_off + d[:, None] * D_FF + kk[None, :]).to(tl.bfloat16)
            up = tl.dot(h_norm_2d, up_w).to(tl.float32).sum(axis=0)
            # SwiGLU: SiLU(gate) * up
            act = (gate * tl.sigmoid(gate)) * up
            # Down projection
            down_w = tl.load(packed_w_ptr + down_off + kk[:, None] * D_MODEL + d[None, :]).to(tl.bfloat16)
            ffn_accum += tl.dot(act[None, :].to(tl.bfloat16), down_w).to(tl.float32).sum(axis=0)
        h = h + ffn_accum

    # ════════════════════════════════════════════
    # OUTPUT: final RMSNorm + tied output projection
    # ════════════════════════════════════════════

    ln_s = tl.load(lnf_s_ptr + d).to(tl.float32)
    h_final = ln_s * h * tl.math.rsqrt(tl.sum(h * h) / D_MODEL + 1e-5)

    h_final_2d = h_final[None, :].to(tl.bfloat16)
    for v_start in tl.range(0, VOCAB_PAD, OUTPUT_VTILE):
        vv = v_start + tl.arange(0, OUTPUT_VTILE)
        out_w = tl.load(output_proj_ptr + d[:, None] * VOCAB_PAD + vv[None, :]).to(tl.bfloat16)
        tile_logits = tl.dot(h_final_2d, out_w).to(tl.float32).sum(axis=0)
        tile_logits = tl.where(vv < VOCAB_SIZE, tile_logits, -1e9)
        tl.store(logits_ptr + vv, tile_logits)


# ──────────────────────────────────────────────────────────────────────

def pack_weights(params, config):
    """Pack per-layer weights into a single bf16 buffer.

    Layout per layer: ln1_s, wq, wk, wv, wo, ln2_s, gate, up, down
    """
    n_layers = config["n_layers"]
    layer_tensors = []
    for i in range(n_layers):
        p = f"layer{i}"
        layer_tensors.extend([
            params[f"{p}.ln1.scale"].reshape(-1),
            params[f"{p}.attn.q"].reshape(-1),
            params[f"{p}.attn.k"].reshape(-1),
            params[f"{p}.attn.v"].reshape(-1),
            params[f"{p}.attn.o"].reshape(-1),
            params[f"{p}.ln2.scale"].reshape(-1),
            params[f"{p}.ffn.gate"].reshape(-1),
            params[f"{p}.ffn.up"].reshape(-1),
            params[f"{p}.ffn.down"].reshape(-1),
        ])
    return jnp.concatenate(layer_tensors).astype(jnp.bfloat16)


def pack_kv_caches(k_caches, v_caches):
    """Pack per-layer KV caches into a single bf16 buffer.

    Input: k_caches[i] shape (n_kv_heads, max_seq, d_head) bf16
    Output: flat buffer with layout [layer0_k, layer0_v, layer1_k, layer1_v, ...]
    """
    parts = []
    for k, v in zip(k_caches, v_caches):
        parts.append(k.reshape(-1))
        parts.append(v.reshape(-1))
    return jnp.concatenate(parts)


def unpack_kv_caches(packed, n_layers, n_kv_heads, max_seq, d_head):
    """Unpack flat KV buffer back to per-layer caches."""
    layer_kv_size = 2 * n_kv_heads * max_seq * d_head
    cache_size = n_kv_heads * max_seq * d_head
    k_caches = []
    v_caches = []
    for i in range(n_layers):
        base = i * layer_kv_size
        k_caches.append(packed[base:base + cache_size].reshape(n_kv_heads, max_seq, d_head))
        v_caches.append(packed[base + cache_size:base + layer_kv_size].reshape(n_kv_heads, max_seq, d_head))
    return k_caches, v_caches


def prepare_decode_weights_nlayer(params, config, vocab_size, kv_splits=2):
    """Precompute packed weights, bf16 embeddings, RoPE tables, tied output proj."""
    n_heads = config["n_heads"]
    d_head = config["d_head"]
    output_vtile = 32
    total_blocks = n_heads * kv_splits
    align = output_vtile * total_blocks
    vocab_pad = ((vocab_size + align - 1) // align) * align
    pad_v = vocab_pad - vocab_size

    cos, sin = precompute_rope_table(config["context_len"], d_head)

    emb_T = params["token_emb"].T  # (d_model, vocab) — tied output projection
    return {
        "token_emb": params["token_emb"].astype(jnp.bfloat16),
        "packed_w": pack_weights(params, config),
        "lnf_s": params["ln_final.scale"].astype(jnp.bfloat16),
        "cos": cos.astype(jnp.bfloat16),
        "sin": sin.astype(jnp.bfloat16),
        "output_proj_padded": jnp.pad(emb_T, [(0, 0), (0, pad_v)]).astype(jnp.bfloat16),
        "vocab_pad": vocab_pad,
    }


def fused_decode_nlayer(w, config, token_id, pos, kv_packed, vocab_size):
    """Fully fused N-layer decode: one kernel call per token.

    Takes and returns packed KV caches to avoid per-step pack/unpack overhead.
    Use pack_kv_caches() once after prefill, then feed the packed buffer through.

    Args:
        w: precomputed weights from prepare_decode_weights_nlayer()
        config: model config
        token_id: scalar token
        pos: scalar position
        kv_packed: flat bf16 buffer from pack_kv_caches() or previous decode call
        vocab_size: actual vocabulary size

    Returns: logits (vocab_size,), kv_packed (flat bf16 buffer)
    """
    d_model = config["d_model"]
    d_head = config["d_head"]
    n_heads = config["n_heads"]
    n_kv_heads = config.get("n_kv_heads", n_heads)
    d_kv = n_kv_heads * d_head
    n_layers = config["n_layers"]
    d_ff = config["d_ff"]
    max_seq = config["context_len"]
    vocab_pad = w["vocab_pad"]
    total_kv_size = n_layers * 2 * n_kv_heads * max_seq * d_head

    logits_pad, kv_out = jt.triton_call(
        w["token_emb"],
        w["packed_w"],
        w["lnf_s"],
        w["output_proj_padded"],
        w["cos"], w["sin"],
        jnp.int32(token_id), jnp.int32(pos),
        kv_packed,
        kernel=_fused_decode_nlayer,
        out_shape=[
            jax.ShapeDtypeStruct((vocab_pad,), jnp.float32),
            jax.ShapeDtypeStruct((total_kv_size,), jnp.bfloat16),
        ],
        grid=(1,),
        num_warps=4, num_stages=1,
        D_MODEL=d_model, D_HEAD=d_head, D_FF=d_ff,
        N_HEADS=n_heads, N_KV_HEADS=n_kv_heads, D_KV=d_kv,
        N_LAYERS=n_layers, MAX_SEQ=max_seq,
        VOCAB_SIZE=vocab_size, VOCAB_PAD=vocab_pad,
    )

    return logits_pad[:vocab_size], kv_out
