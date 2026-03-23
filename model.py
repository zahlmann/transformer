"""Small decoder-only transformer in JAX (pure functional)."""

import jax
import jax.numpy as jnp


def init_transformer(key, vocab_size, d_model=64, n_heads=2, n_layers=1, context_len=128):
    """Initialize a small decoder-only transformer. Returns a flat dict of params."""
    assert d_model % n_heads == 0
    d_head = d_model // n_heads

    params = {}
    config = {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "d_head": d_head,
        "context_len": context_len,
    }

    # token embedding: (vocab_size, d_model)
    key, k = jax.random.split(key)
    params["token_emb"] = jax.random.normal(k, (vocab_size, d_model)) * 0.02

    # learned positional embedding: (context_len, d_model)
    key, k = jax.random.split(key)
    params["pos_emb"] = jax.random.normal(k, (context_len, d_model)) * 0.02

    for layer in range(n_layers):
        prefix = f"layer{layer}"

        # layer norm 1 (pre-attention)
        params[f"{prefix}.ln1.scale"] = jnp.ones(d_model)
        params[f"{prefix}.ln1.bias"] = jnp.zeros(d_model)

        # attention: Q, K, V, O projections
        for proj in ["q", "k", "v"]:
            key, k = jax.random.split(key)
            params[f"{prefix}.attn.{proj}"] = jax.random.normal(k, (d_model, d_model)) * (d_model ** -0.5)

        key, k = jax.random.split(key)
        params[f"{prefix}.attn.o"] = jax.random.normal(k, (d_model, d_model)) * (d_model ** -0.5)

        # layer norm 2 (pre-FFN)
        params[f"{prefix}.ln2.scale"] = jnp.ones(d_model)
        params[f"{prefix}.ln2.bias"] = jnp.zeros(d_model)

        # FFN: up projection (d_model -> 4*d_model), down projection (4*d_model -> d_model)
        d_ff = 4 * d_model
        key, k = jax.random.split(key)
        params[f"{prefix}.ffn.up"] = jax.random.normal(k, (d_model, d_ff)) * (d_model ** -0.5)
        params[f"{prefix}.ffn.up_bias"] = jnp.zeros(d_ff)

        key, k = jax.random.split(key)
        params[f"{prefix}.ffn.down"] = jax.random.normal(k, (d_ff, d_model)) * (d_ff ** -0.5)
        params[f"{prefix}.ffn.down_bias"] = jnp.zeros(d_model)

    # final layer norm
    params["ln_final.scale"] = jnp.ones(d_model)
    params["ln_final.bias"] = jnp.zeros(d_model)

    # separate output projection (weight tying hurts quality with small models)
    key, k = jax.random.split(key)
    params["output_proj"] = jax.random.normal(k, (d_model, vocab_size)) * (d_model ** -0.5)

    return params, config


def layer_norm(x, scale, bias, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return scale * (x - mean) / jnp.sqrt(var + eps) + bias


def causal_attention(x, wq, wk, wv, wo, n_heads):
    """Multi-head causal self-attention. x: (seq_len, d_model)."""
    seq_len, d_model = x.shape
    d_head = d_model // n_heads

    q = (x @ wq).reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)  # (heads, seq, d_head)
    k = (x @ wk).reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)
    v = (x @ wv).reshape(seq_len, n_heads, d_head).transpose(1, 0, 2)

    # scaled dot-product attention with causal mask
    scale = d_head ** -0.5
    attn = (q @ k.transpose(0, 2, 1)) * scale  # (heads, seq, seq)
    mask = jnp.tril(jnp.ones((seq_len, seq_len)))
    attn = jnp.where(mask, attn, -1e9)
    attn = jax.nn.softmax(attn, axis=-1)

    out = (attn @ v).transpose(1, 0, 2).reshape(seq_len, d_model)  # (seq, d_model)
    return out @ wo


def transformer_forward(params, config, x):
    """Forward pass. x: (seq_len,) integer token indices. Returns logits (seq_len, vocab_size)."""
    seq_len = x.shape[0]

    # embeddings
    h = params["token_emb"][x] + params["pos_emb"][:seq_len]

    for layer in range(config["n_layers"]):
        prefix = f"layer{layer}"

        # pre-norm attention
        h_norm = layer_norm(h, params[f"{prefix}.ln1.scale"], params[f"{prefix}.ln1.bias"])
        attn_out = causal_attention(
            h_norm,
            params[f"{prefix}.attn.q"],
            params[f"{prefix}.attn.k"],
            params[f"{prefix}.attn.v"],
            params[f"{prefix}.attn.o"],
            config["n_heads"],
        )
        h = h + attn_out

        # pre-norm FFN
        h_norm = layer_norm(h, params[f"{prefix}.ln2.scale"], params[f"{prefix}.ln2.bias"])
        h_ff = jax.nn.gelu(h_norm @ params[f"{prefix}.ffn.up"] + params[f"{prefix}.ffn.up_bias"])
        h_ff = h_ff @ params[f"{prefix}.ffn.down"] + params[f"{prefix}.ffn.down_bias"]
        h = h + h_ff

    # final layer norm + output projection
    h = layer_norm(h, params["ln_final.scale"], params["ln_final.bias"])
    logits = h @ params["output_proj"]
    return logits


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
