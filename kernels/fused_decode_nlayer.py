"""
Fully fused N-layer decode kernel: embedding → N layers → output in ONE launch.

Generalizes fused_decode_2layer.py to any number of layers. Uses a packed weight
buffer and packed KV caches to avoid passing 50+ individual pointers.

With M=1 (single token decode), register pressure is trivial (~35KB peak)
regardless of d_model or n_layers. The kernel processes layers sequentially,
loading each layer's weights from the packed buffer.

Weight packing layout (all bf16):
  Per layer (10 tensors):
    ln1_scale:  D_MODEL
    ln1_bias:   D_MODEL
    wq:         D_MODEL * D_MODEL
    wk:         D_MODEL * D_MODEL
    wv:         D_MODEL * D_MODEL
    wo:         D_MODEL * D_MODEL
    ln2_scale:  D_MODEL
    ln2_bias:   D_MODEL
    ffn_up:     D_MODEL * D_FF
    ffn_down:   D_FF * D_MODEL

KV cache packing: all layers concatenated.
  Per layer: k_cache (N_HEADS * MAX_SEQ * D_HEAD) + v_cache (same)
"""

import triton
import triton.language as tl
import jax
import jax.numpy as jnp
import jax_triton as jt

BLOCK_K      = tl.constexpr(32)
VOCAB_TILE   = tl.constexpr(128)
KV_TILE      = tl.constexpr(64)
# Smaller output tile for large D_MODEL to avoid (D_MODEL, VOCAB_TILE) intermediate overflow
OUTPUT_VTILE = tl.constexpr(32)


@triton.jit
def _fused_decode_nlayer(
    # Embedding weights
    token_emb_ptr, pos_emb_ptr,
    # Packed per-layer weights (all layers concatenated, bf16)
    packed_w_ptr,
    # Final LN + output
    lnf_s_ptr, lnf_b_ptr,
    output_proj_ptr,
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
    N_LAYERS: tl.constexpr,
    MAX_SEQ: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    VOCAB_PAD: tl.constexpr,
):
    d = tl.arange(0, D_MODEL)
    token_id = tl.load(token_id_ptr)
    pos = tl.load(pos_ptr)

    # ── Embedding ──
    h = (tl.load(token_emb_ptr + token_id * D_MODEL + d).to(tl.float32)
       + tl.load(pos_emb_ptr + pos * D_MODEL + d).to(tl.float32))

    # Per-layer weight size in elements (bf16)
    LAYER_W_SIZE: tl.constexpr = (
        D_MODEL + D_MODEL +                      # ln1 scale, bias
        4 * D_MODEL * D_MODEL +                   # wq, wk, wv, wo
        D_MODEL + D_MODEL +                       # ln2 scale, bias
        D_MODEL * D_FF +                          # ffn_up
        D_FF * D_MODEL                            # ffn_down
    )

    # Per-layer KV cache size in elements (bf16)
    LAYER_KV_SIZE: tl.constexpr = 2 * N_HEADS * MAX_SEQ * D_HEAD

    scale = 0.17677669529663689  # 1/sqrt(32), for D_HEAD=32
    dh = tl.arange(0, D_HEAD)

    for layer in tl.static_range(N_LAYERS):
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
        down_off = off

        # ── LN1 ──
        ln_s = tl.load(packed_w_ptr + ln1_s_off + d).to(tl.float32)
        ln_b = tl.load(packed_w_ptr + ln1_b_off + d).to(tl.float32)
        mean = tl.sum(h) / D_MODEL
        hc = h - mean
        h_norm = ln_s * hc * tl.math.rsqrt(tl.sum(hc * hc) / D_MODEL + 1e-5) + ln_b

        # ── Attention (tiled KV with online softmax) ──
        # Use tl.dot for projections: (1, D_MODEL) @ (D_MODEL, D_HEAD) avoids
        # materializing (D_MODEL, D_HEAD) intermediate that overflows registers at d=512.
        attn_accum = tl.zeros((D_MODEL,), dtype=tl.float32)
        h_norm_2d = h_norm[None, :].to(tl.bfloat16)  # (1, D_MODEL) bf16

        for head in tl.range(N_HEADS):
            hd = head * D_HEAD + dh
            cache_off = head * MAX_SEQ * D_HEAD

            # Q/K/V projections via tl.dot: (1, D_MODEL) @ (D_MODEL, D_HEAD) → (1, D_HEAD)
            wq = tl.load(packed_w_ptr + wq_off + d[:, None] * D_MODEL + hd[None, :]).to(tl.bfloat16)
            Q = tl.dot(h_norm_2d, wq).to(tl.float32).sum(axis=0)  # (D_HEAD,)
            wk = tl.load(packed_w_ptr + wk_off + d[:, None] * D_MODEL + hd[None, :]).to(tl.bfloat16)
            K_new = tl.dot(h_norm_2d, wk).to(tl.float32).sum(axis=0)
            wv = tl.load(packed_w_ptr + wv_off + d[:, None] * D_MODEL + hd[None, :]).to(tl.bfloat16)
            V_new = tl.dot(h_norm_2d, wv).to(tl.float32).sum(axis=0)

            # Write new K/V to output cache
            tl.store(kv_out_ptr + kc_base + cache_off + pos * D_HEAD + dh, K_new.to(tl.bfloat16))
            tl.store(kv_out_ptr + vc_base + cache_off + pos * D_HEAD + dh, V_new.to(tl.bfloat16))

            # Online softmax over tiled KV cache
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

            attn_out = o_i / l_i  # (D_HEAD,)

            # O projection via tl.dot: (1, D_HEAD) @ (D_HEAD, D_MODEL) → (1, D_MODEL)
            wo = tl.load(packed_w_ptr + wo_off + hd[:, None] * D_MODEL + d[None, :]).to(tl.bfloat16)
            attn_accum += tl.dot(attn_out[None, :].to(tl.bfloat16), wo).to(tl.float32).sum(axis=0)

        h = h + attn_accum

        # ── LN2 + FFN (also use tl.dot for projections) ──
        ln_s = tl.load(packed_w_ptr + ln2_s_off + d).to(tl.float32)
        ln_b = tl.load(packed_w_ptr + ln2_b_off + d).to(tl.float32)
        mean = tl.sum(h) / D_MODEL
        hc = h - mean
        h_norm = ln_s * hc * tl.math.rsqrt(tl.sum(hc * hc) / D_MODEL + 1e-5) + ln_b
        h_norm_2d = h_norm[None, :].to(tl.bfloat16)

        ffn_accum = tl.zeros((D_MODEL,), dtype=tl.float32)
        for k in tl.range(0, D_FF, BLOCK_K):
            kk = k + tl.arange(0, BLOCK_K)
            up_w = tl.load(packed_w_ptr + up_off + d[:, None] * D_FF + kk[None, :]).to(tl.bfloat16)
            up = tl.dot(h_norm_2d, up_w).to(tl.float32).sum(axis=0)  # (BLOCK_K,)
            act = up * tl.sigmoid(1.702 * up)
            down_w = tl.load(packed_w_ptr + down_off + kk[:, None] * D_MODEL + d[None, :]).to(tl.bfloat16)
            ffn_accum += tl.dot(act[None, :].to(tl.bfloat16), down_w).to(tl.float32).sum(axis=0)
        h = h + ffn_accum

    # ════════════════════════════════════════════
    # OUTPUT
    # ════════════════════════════════════════════

    ln_s = tl.load(lnf_s_ptr + d).to(tl.float32)
    ln_b = tl.load(lnf_b_ptr + d).to(tl.float32)
    mean = tl.sum(h) / D_MODEL
    hc = h - mean
    h_final = ln_s * hc * tl.math.rsqrt(tl.sum(hc * hc) / D_MODEL + 1e-5) + ln_b

    # Output via tl.dot: (1, D_MODEL) @ (D_MODEL, OUTPUT_VTILE) → (1, OUTPUT_VTILE)
    h_final_2d = h_final[None, :].to(tl.bfloat16)
    for v_start in tl.range(0, VOCAB_PAD, OUTPUT_VTILE):
        vv = v_start + tl.arange(0, OUTPUT_VTILE)
        out_w = tl.load(output_proj_ptr + d[:, None] * VOCAB_PAD + vv[None, :]).to(tl.bfloat16)
        tile_logits = tl.dot(h_final_2d, out_w).to(tl.float32).sum(axis=0)
        tile_logits = tl.where(vv < VOCAB_SIZE, tile_logits, -1e9)
        tl.store(logits_ptr + vv, tile_logits)


# ──────────────────────────────────────────────────────────────────────

def pack_weights(params, config):
    """Pack per-layer weights into a single bf16 buffer."""
    n_layers = config["n_layers"]
    d_model = config["d_model"]
    d_ff = 4 * d_model

    layer_tensors = []
    for i in range(n_layers):
        p = f"layer{i}"
        layer_tensors.extend([
            params[f"{p}.ln1.scale"].reshape(-1),
            params[f"{p}.ln1.bias"].reshape(-1),
            params[f"{p}.attn.q"].reshape(-1),
            params[f"{p}.attn.k"].reshape(-1),
            params[f"{p}.attn.v"].reshape(-1),
            params[f"{p}.attn.o"].reshape(-1),
            params[f"{p}.ln2.scale"].reshape(-1),
            params[f"{p}.ln2.bias"].reshape(-1),
            params[f"{p}.ffn.up"].reshape(-1),
            params[f"{p}.ffn.down"].reshape(-1),
        ])

    return jnp.concatenate(layer_tensors).astype(jnp.bfloat16)


def pack_kv_caches(k_caches, v_caches):
    """Pack per-layer KV caches into a single bf16 buffer.

    Input: k_caches[i] shape (n_heads, max_seq, d_head) bf16
    Output: flat buffer with layout [layer0_k, layer0_v, layer1_k, layer1_v, ...]
    """
    parts = []
    for k, v in zip(k_caches, v_caches):
        parts.append(k.reshape(-1))
        parts.append(v.reshape(-1))
    return jnp.concatenate(parts)


def unpack_kv_caches(packed, n_layers, n_heads, max_seq, d_head):
    """Unpack flat KV buffer back to per-layer caches."""
    layer_kv_size = 2 * n_heads * max_seq * d_head
    cache_size = n_heads * max_seq * d_head
    k_caches = []
    v_caches = []
    for i in range(n_layers):
        base = i * layer_kv_size
        k_caches.append(packed[base:base + cache_size].reshape(n_heads, max_seq, d_head))
        v_caches.append(packed[base + cache_size:base + layer_kv_size].reshape(n_heads, max_seq, d_head))
    return k_caches, v_caches


def prepare_decode_weights_nlayer(params, config, vocab_size, kv_splits=2):
    """Precompute packed weights, bf16 embeddings, padded output proj."""
    # vocab_pad must be divisible by OUTPUT_VTILE * TOTAL_BLOCKS for tiled output projection.
    # TOTAL_BLOCKS = n_heads * kv_splits, OUTPUT_VTILE = 32.
    n_heads = config["n_heads"]
    output_vtile = 32
    total_blocks = n_heads * kv_splits
    align = output_vtile * total_blocks  # e.g., 32*32=1024 for d=512, 32*48=1536 for d=768
    vocab_pad = ((vocab_size + align - 1) // align) * align
    pad_v = vocab_pad - vocab_size
    return {
        "token_emb": params["token_emb"].astype(jnp.bfloat16),
        "pos_emb": params["pos_emb"].astype(jnp.bfloat16),
        "packed_w": pack_weights(params, config),
        "lnf_s": params["ln_final.scale"].astype(jnp.bfloat16),
        "lnf_b": params["ln_final.bias"].astype(jnp.bfloat16),
        "output_proj_padded": jnp.pad(params["output_proj"], [(0, 0), (0, pad_v)]).astype(jnp.bfloat16),
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
    n_layers = config["n_layers"]
    d_ff = 4 * d_model
    max_seq = config["context_len"]
    vocab_pad = w["vocab_pad"]
    total_kv_size = n_layers * 2 * n_heads * max_seq * d_head

    logits_pad, kv_out = jt.triton_call(
        w["token_emb"], w["pos_emb"],
        w["packed_w"],
        w["lnf_s"], w["lnf_b"],
        w["output_proj_padded"],
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
        N_HEADS=n_heads, N_LAYERS=n_layers, MAX_SEQ=max_seq,
        VOCAB_SIZE=vocab_size, VOCAB_PAD=vocab_pad,
    )

    return logits_pad[:vocab_size], kv_out
