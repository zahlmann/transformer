"""EGGROLL ES training with forward-mode AD (JVP) for exact directional derivatives.

Uses full-rank perturbations (no rank-1 compression) since JVP cost is independent
of tangent structure. Each perturbation direction probes ALL gradient dimensions,
giving much better gradient estimates per direction.

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

# ══════════════════════════════════════════════════════════════
# LOCKED CONSTANTS — do not change. Validated by validate.py.
# ══════════════════════════════════════════════════════════════
D_MODEL = 64
N_HEADS = 2
N_LAYERS = 1
CONTEXT_LEN = 128
BATCH_SIZE = 128
EPOCHS = 10
TEMPERATURE = 2.0

# ══════════════════════════════════════════════════════════════
# TUNABLE HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════
N_DIRS = 256         # number of random directions per batch
LR_START = 0.010
LR_DECAY = 1.0
ALPHA = 0.50         # label smoothing
MOMENTUM = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-6
N_SUBGROUPS = 8
CLIP_RANGE = 2.0


def winsorized_zscore(vals):
    group_size = vals.shape[0] // N_SUBGROUPS
    groups = vals[:N_SUBGROUPS * group_size].reshape(N_SUBGROUPS, group_size)
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

    # Build flat param vector info (for full-rank perturbation)
    param_keys = sorted(params.keys())
    param_shapes = {k: params[k].shape for k in param_keys}
    param_sizes = {k: params[k].size for k in param_keys}
    total_params = sum(param_sizes.values())
    # Build offset table
    offsets = {}
    off = 0
    for k in param_keys:
        offsets[k] = off
        off += param_sizes[k]

    n_batches = len(data["train_x"]) // BATCH_SIZE
    train_x = jnp.array(data["train_x"])
    train_y = jnp.array(data["train_y"])
    val_x = jnp.array(data["val_x"])
    val_y = jnp.array(data["val_y"])

    def smoothed_ce_loss(params, x, y):
        logits = transformer_forward_batch(params, config, x) / TEMPERATURE
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        n_classes = logits.shape[-1]
        one_hot = jax.nn.one_hot(y, n_classes)
        targets = (1 - ALPHA) * one_hot + ALPHA / n_classes
        return -jnp.sum(targets * log_probs) / x.shape[0]

    def flat_to_tangent(v_flat):
        """Convert flat perturbation vector to param-shaped tangent dict."""
        tangent = {}
        for k in param_keys:
            tangent[k] = v_flat[offsets[k]:offsets[k] + param_sizes[k]].reshape(param_shapes[k])
        return tangent

    def compute_jvp_single(params, v_flat, x, y):
        tangent = flat_to_tangent(v_flat)
        _, dloss = jax.jvp(
            lambda p: smoothed_ce_loss(p, x, y),
            (params,), (tangent,)
        )
        return dloss

    def train_one_batch(params, momentum_buf, v_buf, step, key, x, y, lr):
        key, vec_key = jax.random.split(key)
        vecs = jax.random.normal(vec_key, (N_DIRS, total_params))

        # Compute directional derivatives via lax.scan (sequential JVP)
        def scan_body(carry, v):
            dloss = compute_jvp_single(carry, v, x, y)
            return carry, dloss
        _, dlosses = jax.lax.scan(scan_body, params, vecs)

        # Normalize directional derivatives
        shaped = winsorized_zscore(dlosses)
        scale = 1.0 / N_DIRS

        # Compute gradient for each parameter directly (full-rank, no rank-1 compression)
        new_params = {}
        new_momentum = {}
        new_v = {}
        t = step + 1
        for k in param_keys:
            o = offsets[k]
            s = param_sizes[k]
            v_slice = vecs[:, o:o+s]  # (N_DIRS, param_size)
            # g = (1/N) Σ shaped_i * v_i  — gradient in full parameter space
            g = scale * (v_slice * shaped[:, None]).sum(axis=0).reshape(param_shapes[k])
            new_momentum[k] = MOMENTUM * momentum_buf[k] + (1 - MOMENTUM) * g
            new_v[k] = ADAM_BETA2 * v_buf[k] + (1 - ADAM_BETA2) * g ** 2
            m_hat = new_momentum[k] / (1 - MOMENTUM ** t)
            v_hat = new_v[k] / (1 - ADAM_BETA2 ** t)
            new_params[k] = params[k] - lr * m_hat / (jnp.sqrt(v_hat) + ADAM_EPS)

        return new_params, new_momentum, new_v, step + 1, key, jnp.mean(dlosses)

    @jax.jit
    def train_batch(params, momentum_buf, v_buf, step, key, x, y, lr):
        return train_one_batch(params, momentum_buf, v_buf, step, key, x, y, lr)

    @jax.jit
    def eval_loss(params, x, y):
        from model import cross_entropy_loss
        logits = transformer_forward_batch(params, config, x)
        return cross_entropy_loss(logits, y)

    print("=== CONSTANTS ===")
    print(f"D_MODEL: {D_MODEL}")
    print(f"N_HEADS: {N_HEADS}")
    print(f"N_LAYERS: {N_LAYERS}")
    print(f"CONTEXT_LEN: {CONTEXT_LEN}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"EPOCHS: {EPOCHS}")
    print(f"TEMPERATURE: {TEMPERATURE}")
    print("=" * 20)
    print(f"Forward-mode AD: N_DIRS={N_DIRS}, total_params={total_params}")

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
