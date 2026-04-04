"""Small decoder-only transformer in JAX (pure functional).

Phase C architecture: RMSNorm, RoPE, SwiGLU, no biases, tied embeddings.
Hybrid: 75% Gated DeltaNet (linear attention) + 25% standard attention.
"""

import jax
import jax.numpy as jnp


def _swiglu_d_ff(d_model):
    """Compute SwiGLU FFN hidden dim: match param count of standard 4d FFN.

    Standard FFN: 2 * d * 4d = 8d² params
    SwiGLU FFN:   3 * d * d_ff params
    Match: d_ff = 8d/3, rounded up to multiple of 128.
    """
    return ((8 * d_model // 3 + 127) // 128) * 128


def _layer_types(n_layers):
    """Return list of layer types: every 4th layer is 'attn', rest are 'delta'.

    Pattern: [D, D, D, A, D, D, D, A, ...]
    """
    return ["attn" if (i % 4 == 3) else "delta" for i in range(n_layers)]


def init_transformer(key, vocab_size, d_model=64, n_heads=2, n_layers=1, context_len=128,
                     n_kv_heads=None, use_deltanet=True):
    """Initialize a decoder-only transformer. Returns a flat dict of params.

    Architecture: RMSNorm, RoPE, SwiGLU FFN, no biases, tied embeddings.
    Hybrid: 75% Gated DeltaNet + 25% standard attention (Qwen 3.5 pattern).
    use_deltanet: if False, use all standard attention layers (for comparison).
    n_kv_heads: number of KV heads for GQA. If None, defaults to n_heads.
    """
    assert d_model % n_heads == 0
    d_head = d_model // n_heads
    if n_kv_heads is None:
        n_kv_heads = n_heads
    assert n_heads % n_kv_heads == 0

    d_ff = _swiglu_d_ff(d_model)
    layer_types = _layer_types(n_layers) if use_deltanet else ["attn"] * n_layers

    params = {}
    config = {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_kv_heads": n_kv_heads,
        "n_layers": n_layers,
        "d_head": d_head,
        "d_ff": d_ff,
        "context_len": context_len,
        "layer_types": layer_types,
    }

    # token embedding: (vocab_size, d_model) — also used as output projection (tied)
    key, k = jax.random.split(key)
    params["token_emb"] = jax.random.normal(k, (vocab_size, d_model)) * 0.02

    d_kv = n_kv_heads * d_head

    for layer in range(n_layers):
        prefix = f"layer{layer}"
        ltype = layer_types[layer]

        # RMSNorm 1 (pre-attention/deltanet) — scale only
        params[f"{prefix}.ln1.scale"] = jnp.ones(d_model)

        if ltype == "attn":
            # standard attention: Q, O use n_heads; K, V use n_kv_heads (GQA)
            key, k = jax.random.split(key)
            params[f"{prefix}.attn.q"] = jax.random.normal(k, (d_model, d_model)) * (d_model ** -0.5)
            key, k = jax.random.split(key)
            params[f"{prefix}.attn.k"] = jax.random.normal(k, (d_model, d_kv)) * (d_model ** -0.5)
            key, k = jax.random.split(key)
            params[f"{prefix}.attn.v"] = jax.random.normal(k, (d_model, d_kv)) * (d_model ** -0.5)
            key, k = jax.random.split(key)
            params[f"{prefix}.attn.o"] = jax.random.normal(k, (d_model, d_model)) * (d_model ** -0.5)

        elif ltype == "delta":
            # Gated DeltaNet: same Q/K/V/O projections + gates + convolutions
            key, k = jax.random.split(key)
            params[f"{prefix}.attn.q"] = jax.random.normal(k, (d_model, d_model)) * (d_model ** -0.5)
            key, k = jax.random.split(key)
            params[f"{prefix}.attn.k"] = jax.random.normal(k, (d_model, d_kv)) * (d_model ** -0.5)
            key, k = jax.random.split(key)
            params[f"{prefix}.attn.v"] = jax.random.normal(k, (d_model, d_kv)) * (d_model ** -0.5)
            key, k = jax.random.split(key)
            params[f"{prefix}.attn.o"] = jax.random.normal(k, (d_model, d_model)) * (d_model ** -0.5)

            # decay gate: alpha = exp(-A * softplus(a_proj(x) + dt_bias))
            key, k = jax.random.split(key)
            params[f"{prefix}.delta.a_proj"] = jax.random.normal(k, (d_model, n_kv_heads)) * 0.01
            params[f"{prefix}.delta.A_log"] = jnp.log(jax.random.uniform(
                jax.random.split(key)[1], (n_kv_heads,), minval=1.0, maxval=16.0))
            params[f"{prefix}.delta.dt_bias"] = jnp.log(jnp.exp(jax.random.uniform(
                jax.random.split(key)[0], (n_kv_heads,), minval=0.001, maxval=0.1)) - 1.0)

            # beta gate: beta = sigmoid(b_proj(x))
            key, k = jax.random.split(key)
            params[f"{prefix}.delta.b_proj"] = jax.random.normal(k, (d_model, n_kv_heads)) * 0.01

            # output gate: o = rms_norm(o) * silu(g_proj(x))
            key, k = jax.random.split(key)
            params[f"{prefix}.delta.g_proj"] = jax.random.normal(k, (d_model, d_model)) * (d_model ** -0.5)

            # short convolutions (depthwise, kernel_size=4)
            key, k = jax.random.split(key)
            params[f"{prefix}.delta.conv_q"] = jax.random.normal(k, (4, d_model)) * 0.1
            key, k = jax.random.split(key)
            params[f"{prefix}.delta.conv_k"] = jax.random.normal(k, (4, d_kv)) * 0.1
            key, k = jax.random.split(key)
            params[f"{prefix}.delta.conv_v"] = jax.random.normal(k, (4, d_kv)) * 0.1

            # RMSNorm for output (before output gate multiply)
            params[f"{prefix}.delta.o_norm"] = jnp.ones(d_model)

        # RMSNorm 2 (pre-FFN) — scale only
        params[f"{prefix}.ln2.scale"] = jnp.ones(d_model)

        # SwiGLU FFN (same for both layer types)
        key, k = jax.random.split(key)
        params[f"{prefix}.ffn.gate"] = jax.random.normal(k, (d_model, d_ff)) * (d_model ** -0.5)
        key, k = jax.random.split(key)
        params[f"{prefix}.ffn.up"] = jax.random.normal(k, (d_model, d_ff)) * (d_model ** -0.5)
        key, k = jax.random.split(key)
        params[f"{prefix}.ffn.down"] = jax.random.normal(k, (d_ff, d_model)) * (d_ff ** -0.5)

    # final RMSNorm
    params["ln_final.scale"] = jnp.ones(d_model)

    return params, config


def rms_norm(x, scale, eps=1e-5):
    """RMSNorm: scale * x / sqrt(mean(x²) + eps)."""
    rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + eps)
    return scale * x / rms


def precompute_rope_table(context_len, d_head, base=10000.0):
    """Precompute cos/sin tables for RoPE. Returns (context_len, d_head//2) each."""
    half = d_head // 2
    freqs = base ** (-jnp.arange(0, half, dtype=jnp.float32) * 2.0 / d_head)
    positions = jnp.arange(context_len, dtype=jnp.float32)
    angles = positions[:, None] * freqs[None, :]
    return jnp.cos(angles), jnp.sin(angles)


def apply_rope(x, cos, sin):
    """Apply RoPE to x. x: (..., d_head), cos/sin: (seq, d_head//2) or (d_head//2,)."""
    half = x.shape[-1] // 2
    x_even = x[..., :half]
    x_odd = x[..., half:]
    return jnp.concatenate([
        x_even * cos - x_odd * sin,
        x_even * sin + x_odd * cos,
    ], axis=-1)


# ─── causal depthwise conv1d ───

def _causal_conv1d(x, weight):
    """Causal depthwise 1D convolution. x: (seq, channels), weight: (kernel_size, channels)."""
    K = weight.shape[0]
    x_padded = jnp.pad(x, ((K - 1, 0), (0, 0)))
    out = sum(x_padded[K - 1 - i:x.shape[0] + K - 1 - i] * weight[i] for i in range(K))
    return out


# ─── standard causal attention ───

def causal_attention(x, wq, wk, wv, wo, n_heads, n_kv_heads, cos, sin):
    """Multi-head causal self-attention with GQA and RoPE. x: (seq_len, d_model)."""
    seq_len, d_model = x.shape
    d_head = d_model // n_heads

    q = (x @ wq).reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
    k = (x @ wk).reshape(seq_len, n_kv_heads, d_head).transpose(1, 0, 2)
    v = (x @ wv).reshape(seq_len, n_kv_heads, d_head).transpose(1, 0, 2)

    cos_seq = cos[:seq_len]
    sin_seq = sin[:seq_len]
    q = apply_rope(q, cos_seq[None, :, :], sin_seq[None, :, :])
    k = apply_rope(k, cos_seq[None, :, :], sin_seq[None, :, :])

    if n_kv_heads < n_heads:
        repeats = n_heads // n_kv_heads
        k = jnp.repeat(k, repeats, axis=0)
        v = jnp.repeat(v, repeats, axis=0)

    scale = d_head ** -0.5
    attn = (q @ k.transpose(0, 2, 1)) * scale
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    attn = jnp.where(mask, attn, -1e9)
    attn = jax.nn.softmax(attn, axis=-1)

    out = (attn @ v).transpose(1, 0, 2).reshape(seq_len, d_model)
    return out @ wo


# ─── gated deltanet attention ───

def deltanet_attention(x, wq, wk, wv, wo, a_proj, b_proj, A_log, dt_bias, g_proj,
                       conv_q, conv_k, conv_v, o_norm_scale,
                       n_heads, n_kv_heads):
    """Gated DeltaNet linear attention (naive recurrent). x: (seq_len, d_model).

    Uses recurrent state S (d_head × d_head per KV head) instead of KV cache.
    O(T * d²) per layer — linear in sequence length.
    """
    seq_len, d_model = x.shape
    d_head = d_model // n_heads
    d_kv = n_kv_heads * d_head
    gqa_group = n_heads // n_kv_heads

    # project Q, K, V
    q = x @ wq  # (seq, d_model)
    k = x @ wk  # (seq, d_kv)
    v = x @ wv  # (seq, d_kv)

    # short causal convolution + SiLU activation
    q = jax.nn.silu(_causal_conv1d(q, conv_q))
    k = jax.nn.silu(_causal_conv1d(k, conv_k))
    v = jax.nn.silu(_causal_conv1d(v, conv_v))

    # reshape to heads: Q has n_heads, K/V have n_kv_heads
    q = q.reshape(seq_len, n_heads, d_head)      # (seq, n_heads, d_head)
    k = k.reshape(seq_len, n_kv_heads, d_head)   # (seq, n_kv_heads, d_head)
    v = v.reshape(seq_len, n_kv_heads, d_head)   # (seq, n_kv_heads, d_head)

    # L2 normalize Q and K, scale Q
    q = q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-6) * (d_head ** -0.5)
    k = k / (jnp.linalg.norm(k, axis=-1, keepdims=True) + 1e-6)

    # compute gates
    # decay: alpha = exp(-A * softplus(a_proj(x) + dt_bias))
    A = jnp.exp(A_log)  # (n_kv_heads,)
    dt = jax.nn.softplus(x @ a_proj + dt_bias)  # (seq, n_kv_heads)
    decay = jnp.exp(-A[None, :] * dt)  # (seq, n_kv_heads), in (0, 1)
    # beta: update strength
    beta = jax.nn.sigmoid(x @ b_proj)  # (seq, n_kv_heads)

    # output gate
    gate = jax.nn.silu(x @ g_proj)  # (seq, d_model)

    # Chunked recurrence: split into chunks, checkpoint per chunk.
    # Saves only chunk boundary states instead of all T states → O(T/C) memory.
    gqa_idx = jnp.arange(n_heads) // gqa_group  # precompute GQA index mapping

    def step(state, inputs):
        q_t, k_t, v_t, beta_t, decay_t = inputs
        state = state * decay_t[:, None, None]
        retrieved = jnp.einsum('hkv,hk->hv', state, k_t)
        delta = (v_t - retrieved) * beta_t[:, None]
        state = state + jnp.einsum('hk,hv->hkv', k_t, delta)
        o_t = jnp.einsum('hkv,hk->hv', state[gqa_idx], q_t)
        return state, o_t

    CHUNK = 64
    n_chunks = seq_len // CHUNK

    @jax.checkpoint
    def scan_chunk(state, chunk_inputs):
        return jax.lax.scan(step, state, chunk_inputs)

    initial_state = jnp.zeros((n_kv_heads, d_head, d_head), dtype=jnp.float32)
    # reshape inputs to (n_chunks, chunk_size, ...)
    inputs_chunked = jax.tree.map(
        lambda x: x.reshape(n_chunks, CHUNK, *x.shape[1:]),
        (q, k, v, beta, decay))
    _, outputs_chunked = jax.lax.scan(scan_chunk, initial_state, inputs_chunked)
    # outputs_chunked: (n_chunks, chunk_size, n_heads, d_head)
    outputs = outputs_chunked.reshape(seq_len, n_heads, d_head)

    # reshape to (seq, d_model)
    o = outputs.reshape(seq_len, d_model)

    # output: rms_norm(o) * gate, then output projection
    o = rms_norm(o, o_norm_scale) * gate
    return o @ wo


# ─── transformer layers ───

def _attn_layer(h, ln1_s, wq, wk, wv, wo,
                ln2_s, ffn_gate, ffn_up, ffn_down,
                n_heads, n_kv_heads, context_len, d_head):
    """Standard attention transformer layer."""
    cos, sin = precompute_rope_table(context_len, d_head)
    h_norm = rms_norm(h, ln1_s)
    attn_out = causal_attention(h_norm, wq, wk, wv, wo, n_heads, n_kv_heads, cos, sin)
    h = h + attn_out
    h_norm2 = rms_norm(h, ln2_s)
    h_ff = (jax.nn.silu(h_norm2 @ ffn_gate) * (h_norm2 @ ffn_up)) @ ffn_down
    return h + h_ff


def _delta_layer(h, ln1_s, wq, wk, wv, wo,
                 a_proj, b_proj, A_log, dt_bias, g_proj,
                 conv_q, conv_k, conv_v, o_norm_scale,
                 ln2_s, ffn_gate, ffn_up, ffn_down,
                 n_heads, n_kv_heads):
    """Gated DeltaNet transformer layer."""
    h_norm = rms_norm(h, ln1_s)
    delta_out = deltanet_attention(
        h_norm, wq, wk, wv, wo,
        a_proj, b_proj, A_log, dt_bias, g_proj,
        conv_q, conv_k, conv_v, o_norm_scale,
        n_heads, n_kv_heads)
    h = h + delta_out
    h_norm2 = rms_norm(h, ln2_s)
    h_ff = (jax.nn.silu(h_norm2 @ ffn_gate) * (h_norm2 @ ffn_up)) @ ffn_down
    return h + h_ff


def transformer_forward(params, config, x):
    """Forward pass. x: (seq_len,) integer token indices. Returns logits (seq_len, vocab_size)."""
    h = params["token_emb"][x]
    n_heads = config["n_heads"]
    n_kv_heads = config.get("n_kv_heads", n_heads)
    d_head = config["d_head"]
    context_len = config["context_len"]
    layer_types = config.get("layer_types", ["attn"] * config["n_layers"])

    for layer in range(config["n_layers"]):
        p = f"layer{layer}"
        if layer_types[layer] == "delta":
            h = jax.checkpoint(_delta_layer, static_argnums=(19, 20))(
                h,
                params[f"{p}.ln1.scale"],
                params[f"{p}.attn.q"], params[f"{p}.attn.k"],
                params[f"{p}.attn.v"], params[f"{p}.attn.o"],
                params[f"{p}.delta.a_proj"], params[f"{p}.delta.b_proj"],
                params[f"{p}.delta.A_log"], params[f"{p}.delta.dt_bias"],
                params[f"{p}.delta.g_proj"],
                params[f"{p}.delta.conv_q"], params[f"{p}.delta.conv_k"],
                params[f"{p}.delta.conv_v"], params[f"{p}.delta.o_norm"],
                params[f"{p}.ln2.scale"],
                params[f"{p}.ffn.gate"], params[f"{p}.ffn.up"], params[f"{p}.ffn.down"],
                n_heads, n_kv_heads,
            )
        else:
            h = jax.checkpoint(_attn_layer, static_argnums=(10, 11, 12, 13))(
                h,
                params[f"{p}.ln1.scale"],
                params[f"{p}.attn.q"], params[f"{p}.attn.k"],
                params[f"{p}.attn.v"], params[f"{p}.attn.o"],
                params[f"{p}.ln2.scale"],
                params[f"{p}.ffn.gate"], params[f"{p}.ffn.up"], params[f"{p}.ffn.down"],
                n_heads, n_kv_heads, context_len, d_head,
            )

    h = rms_norm(h, params["ln_final.scale"])
    logits = h @ params["token_emb"].T
    return logits


def prefill_with_kv(params, config, x):
    """Forward pass that also returns KV caches for decode.

    For attention layers: returns standard KV caches.
    For DeltaNet layers: returns DeltaNet state matrices as "caches".
    """
    seq_len = x.shape[0]
    n_heads = config["n_heads"]
    n_kv_heads = config.get("n_kv_heads", n_heads)
    d_head = config["d_head"]
    max_seq = config["context_len"]
    layer_types = config.get("layer_types", ["attn"] * config["n_layers"])

    cos, sin = precompute_rope_table(max_seq, d_head)

    h = params["token_emb"][x]
    k_caches, v_caches = [], []

    for layer in range(config["n_layers"]):
        prefix = f"layer{layer}"
        ltype = layer_types[layer]

        if ltype == "attn":
            h_norm = rms_norm(h, params[f"{prefix}.ln1.scale"])

            k_proj = (h_norm @ params[f"{prefix}.attn.k"]).reshape(seq_len, n_kv_heads, d_head)
            v_proj = (h_norm @ params[f"{prefix}.attn.v"]).reshape(seq_len, n_kv_heads, d_head)
            cos_seq = cos[:seq_len]
            sin_seq = sin[:seq_len]
            k_proj = apply_rope(k_proj, cos_seq[:, None, :], sin_seq[:, None, :])

            k_cache = jnp.zeros((n_kv_heads, max_seq, d_head), dtype=jnp.bfloat16)
            v_cache = jnp.zeros((n_kv_heads, max_seq, d_head), dtype=jnp.bfloat16)
            k_cache = k_cache.at[:, :seq_len, :].set(k_proj.transpose(1, 0, 2).astype(jnp.bfloat16))
            v_cache = v_cache.at[:, :seq_len, :].set(v_proj.transpose(1, 0, 2).astype(jnp.bfloat16))
            k_caches.append(k_cache)
            v_caches.append(v_cache)

            attn_out = causal_attention(
                h_norm, params[f"{prefix}.attn.q"], params[f"{prefix}.attn.k"],
                params[f"{prefix}.attn.v"], params[f"{prefix}.attn.o"],
                n_heads, n_kv_heads, cos, sin)

            h = h + attn_out

        elif ltype == "delta":
            h_norm = rms_norm(h, params[f"{prefix}.ln1.scale"])
            delta_out = deltanet_attention(
                h_norm,
                params[f"{prefix}.attn.q"], params[f"{prefix}.attn.k"],
                params[f"{prefix}.attn.v"], params[f"{prefix}.attn.o"],
                params[f"{prefix}.delta.a_proj"], params[f"{prefix}.delta.b_proj"],
                params[f"{prefix}.delta.A_log"], params[f"{prefix}.delta.dt_bias"],
                params[f"{prefix}.delta.g_proj"],
                params[f"{prefix}.delta.conv_q"], params[f"{prefix}.delta.conv_k"],
                params[f"{prefix}.delta.conv_v"], params[f"{prefix}.delta.o_norm"],
                n_heads, n_kv_heads)
            h = h + delta_out
            # DeltaNet layers don't use KV caches — placeholder zeros
            k_caches.append(jnp.zeros((n_kv_heads, max_seq, d_head), dtype=jnp.bfloat16))
            v_caches.append(jnp.zeros((n_kv_heads, max_seq, d_head), dtype=jnp.bfloat16))

        h_norm2 = rms_norm(h, params[f"{prefix}.ln2.scale"])
        h_ff = (jax.nn.silu(h_norm2 @ params[f"{prefix}.ffn.gate"]) *
                (h_norm2 @ params[f"{prefix}.ffn.up"])) @ params[f"{prefix}.ffn.down"]
        h = h + h_ff

    h = rms_norm(h, params["ln_final.scale"])
    logits = h @ params["token_emb"].T
    return logits, k_caches, v_caches


def transformer_forward_batch(params, config, x_batch):
    """Batched forward pass. x_batch: (batch, seq_len). Returns (batch, seq_len, vocab)."""
    return jax.vmap(lambda x: transformer_forward(params, config, x))(x_batch)


def cross_entropy_loss(logits, targets):
    """Mean cross-entropy loss. logits: (batch, seq, vocab), targets: (batch, seq)."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    return -jnp.mean(target_log_probs)


def count_params(params):
    return sum(p.size for p in jax.tree.leaves(params))
