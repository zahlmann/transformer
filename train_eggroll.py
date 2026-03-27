"""EGGROLL ES training with forward-mode AD (JVP) for exact directional derivatives.

Instead of finite-difference gradient estimation (f(θ+σv) - f(θ-σv)) / 2σ,
uses jax.jvp to compute exact directional derivatives ∇f(θ)·v. This removes
the O(σ²) bias and finite-difference noise from gradient estimation.

Usage: uv run train_eggroll.py [--seed SEED]
"""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import argparse
import time
import jax
import jax.numpy as jnp

from data import prepare_data
from model import init_transformer, transformer_forward_batch, count_params

def build_param_spec(params):
    """Build perturbation vector spec from model params (alphabetical key order)."""
    spec = []
    offset = 0
    for key in sorted(params.keys()):
        shape = params[key].shape
        if len(shape) == 2:
            vec_dim = shape[0] + shape[1]
            spec.append((key, shape, offset, vec_dim, True))
        elif len(shape) == 1:
            vec_dim = shape[0]
            spec.append((key, shape, offset, vec_dim, False))
        else:
            raise ValueError(f"Unexpected: {key} {shape}")
        offset += vec_dim
    return spec, offset

# ══════════════════════════════════════════════════════════════
# LOCKED CONSTANTS — do not change. Validated by validate.py.
# Changing these makes the comparison to backprop unfair.
# ══════════════════════════════════════════════════════════════
D_MODEL = 64
N_HEADS = 2
N_LAYERS = 1
CONTEXT_LEN = 128
BATCH_SIZE = 128
EPOCHS = 10          # LOCKED — same as backprop baseline
TEMPERATURE = 2.0

# ══════════════════════════════════════════════════════════════
# TUNABLE HYPERPARAMETERS — optimize these freely
# ══════════════════════════════════════════════════════════════
HALF_POP = 256       # start small for forward-mode AD testing
SIGMA_START = 0.020  # unused by forward-mode AD (no finite differences)
SIGMA_DECAY = 0.998
LR_START = 0.010
LR_DECAY = 1.0
ALPHA = 0.50
N_SUBGROUPS = 8
CLIP_RANGE = 2.0
MOMENTUM = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-6
JVP_CHUNK = 64       # process JVPs in chunks to limit memory


def winsorized_zscore(dlosses):
    group_size = dlosses.shape[0] // N_SUBGROUPS
    groups = dlosses[:N_SUBGROUPS * group_size].reshape(N_SUBGROUPS, group_size)
    means = jnp.mean(groups, axis=1, keepdims=True)
    stds = jnp.std(groups, axis=1, keepdims=True) + 1e-8
    z = (groups - means) / stds
    z = jnp.clip(z, -CLIP_RANGE, CLIP_RANGE)
    return z.reshape(-1)


def train(seed=42):
    data = prepare_data(context_len=CONTEXT_LEN)
    vocab_size = data["vocab_size"]

    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    params, config = init_transformer(
        init_key, vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, context_len=CONTEXT_LEN,
    )
    spec, total_vec_dim = build_param_spec(params)
    n_batches = len(data["train_x"]) // BATCH_SIZE

    train_x = jnp.array(data["train_x"])
    train_y = jnp.array(data["train_y"])
    val_x = jnp.array(data["val_x"])
    val_y = jnp.array(data["val_y"])

    lr_scale_arr = jnp.ones(len(spec))

    def smoothed_ce_loss(params, x, y):
        """CE loss with label smoothing and temperature, matching Triton kernel."""
        logits = transformer_forward_batch(params, config, x) / TEMPERATURE
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        n_classes = logits.shape[-1]
        one_hot = jax.nn.one_hot(y, n_classes)
        targets = (1 - ALPHA) * one_hot + ALPHA / n_classes
        # Sum over seq and classes, mean over batch
        return -jnp.sum(targets * log_probs) / x.shape[0]

    def vec_to_tangent(v):
        """Convert a perturbation vector to a param-shaped tangent dict."""
        tangent = {}
        for pkey, shape, offset, vec_dim, is_2d in spec:
            if is_2d:
                m, n = shape
                u = v[offset:offset+m]
                w = v[offset+m:offset+m+n]
                tangent[pkey] = jnp.outer(u, w)
            else:
                tangent[pkey] = v[offset:offset+vec_dim]
        return tangent

    def compute_jvp(params, tangent, x, y):
        """Compute exact directional derivative via forward-mode AD."""
        _, dloss = jax.jvp(
            lambda p: smoothed_ce_loss(p, x, y),
            (params,), (tangent,)
        )
        return dloss

    def train_one_batch(params, momentum_buf, v_buf, step, key, x, y, lr):
        key, vec_key = jax.random.split(key)
        vecs = jax.random.normal(vec_key, (HALF_POP, total_vec_dim))

        # Compute directional derivatives via chunked vmap + scan
        n_chunks = HALF_POP // JVP_CHUNK
        vecs_chunked = vecs.reshape(n_chunks, JVP_CHUNK, total_vec_dim)

        def scan_chunk(carry, chunk):
            # Build batched tangent for this chunk
            batched_tangent = {}
            for pkey, shape, offset, vec_dim, is_2d in spec:
                if is_2d:
                    m, n = shape
                    u = chunk[:, offset:offset+m]
                    w = chunk[:, offset+m:offset+m+n]
                    batched_tangent[pkey] = u[:, :, None] * w[:, None, :]
                else:
                    batched_tangent[pkey] = chunk[:, offset:offset+vec_dim]
            chunk_dlosses = jax.vmap(
                lambda t: compute_jvp(carry, t, x, y)
            )(batched_tangent)
            return carry, chunk_dlosses

        _, dlosses_chunks = jax.lax.scan(scan_chunk, params, vecs_chunked)
        dlosses = dlosses_chunks.reshape(-1)

        # Normalize directional derivatives (like winsorized z-score for ES)
        shaped = winsorized_zscore(dlosses)
        scale = 1.0 / HALF_POP

        new_params = {}
        new_momentum = {}
        new_v = {}
        t = step + 1
        for idx, (pkey, shape, offset, vec_dim, is_2d) in enumerate(spec):
            v_pert = vecs[:, offset:offset + vec_dim]
            if is_2d:
                m, n = shape
                g = scale * (v_pert[:, :m] * shaped[:, None]).T @ v_pert[:, m:]
            else:
                g = scale * (v_pert * shaped[:, None]).sum(axis=0)
            lr_s = lr_scale_arr[idx]
            new_momentum[pkey] = MOMENTUM * momentum_buf[pkey] + (1 - MOMENTUM) * g
            new_v[pkey] = ADAM_BETA2 * v_buf[pkey] + (1 - ADAM_BETA2) * g ** 2
            m_hat = new_momentum[pkey] / (1 - MOMENTUM ** t)
            v_hat = new_v[pkey] / (1 - ADAM_BETA2 ** t)
            new_params[pkey] = params[pkey] - lr * lr_s * m_hat / (jnp.sqrt(v_hat) + ADAM_EPS)

        return new_params, new_momentum, new_v, step + 1, key, jnp.mean(dlosses)

    @jax.jit
    def train_batch(params, momentum_buf, v_buf, step, key, x, y, lr):
        return train_one_batch(params, momentum_buf, v_buf, step, key, x, y, lr)

    @jax.jit
    def eval_loss(params, x, y):
        from model import cross_entropy_loss
        logits = transformer_forward_batch(params, config, x)
        return cross_entropy_loss(logits, y)

    # Print locked constants for validate.py
    print("=== CONSTANTS ===")
    print(f"D_MODEL: {D_MODEL}")
    print(f"N_HEADS: {N_HEADS}")
    print(f"N_LAYERS: {N_LAYERS}")
    print(f"CONTEXT_LEN: {CONTEXT_LEN}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"EPOCHS: {EPOCHS}")
    print(f"TEMPERATURE: {TEMPERATURE}")
    print("=" * 20)

    momentum_buf = jax.tree.map(jnp.zeros_like, params)
    v_buf = jax.tree.map(jnp.zeros_like, params)
    step = jnp.int32(0)

    t_start = time.perf_counter()

    lrs_sched = [LR_START * (LR_DECAY ** e) for e in range(EPOCHS)]

    for epoch in range(EPOCHS):
        lr = lrs_sched[epoch]
        key, sk = jax.random.split(key)
        perm = jax.random.permutation(sk, len(data["train_x"]))
        sx, sy = train_x[perm], train_y[perm]
        eloss = jnp.float32(0.0)
        for bi in range(n_batches):
            s = bi * BATCH_SIZE
            params, momentum_buf, v_buf, step, key, pl = train_batch(
                params, momentum_buf, v_buf, step, key,
                sx[s:s+BATCH_SIZE], sy[s:s+BATCH_SIZE], lr)
            eloss = eloss + pl

        vl = eval_loss(params, val_x[:BATCH_SIZE], val_y[:BATCH_SIZE])
        print(f"  Epoch {epoch+1}/{EPOCHS}  proxy={float(eloss)/n_batches:.4f}  val_loss={float(vl):.4f}  ppl={float(jnp.exp(vl)):.2f}  lr={lr:.4f}")

    vl = eval_loss(params, val_x[:BATCH_SIZE], val_y[:BATCH_SIZE])
    vl.block_until_ready()
    total = time.perf_counter() - t_start
    ppl = float(jnp.exp(vl))
    val_loss = float(vl)

    mem_stats = jax.local_devices()[0].memory_stats()
    peak_mb = mem_stats["peak_bytes_in_use"] / 1e6 if mem_stats else 0.0

    print("=== RESULTS ===")
    print(f"val_loss: {val_loss:.4f}")
    print(f"perplexity: {ppl:.2f}")
    print(f"training_time_s: {total:.1f}")
    print(f"peak_memory_mb: {peak_mb:.0f}")
    print("=" * 20)

    return val_loss, ppl, total


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train(seed=args.seed)
