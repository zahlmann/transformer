"""EGGROLL ES training with dual-number Triton kernel (forward-mode AD).

Uses exact directional derivatives via dual numbers instead of finite differences.
No sigma, no antithetic pairs — each perturbation direction gives an exact JVP.

Usage: uv run train_eggroll.py [--seed SEED]
"""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import argparse
import time
import jax
import jax.numpy as jnp

from data import prepare_data
from model import init_transformer, count_params

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
# LOCKED CONSTANTS
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
N_DIRS = 4096        # perturbation directions (no antithetic pairs needed)
LR_START = 0.010
LR_DECAY = 1.0
ALPHA = 0.50
N_SUBGROUPS = 8
CLIP_RANGE = 2.0
MOMENTUM = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-6


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
    spec, total_vec_dim = build_param_spec(params)
    n_batches = len(data["train_x"]) // BATCH_SIZE

    train_x = jnp.array(data["train_x"])
    train_y = jnp.array(data["train_y"])
    val_x = jnp.array(data["val_x"])
    val_y = jnp.array(data["val_y"])

    lr_scale_arr = jnp.ones(len(spec))

    from kernels.fused_transformer_ce_dual import fused_dual_forward

    def train_one_batch(params, momentum_buf, v_buf, step, key, x, y, lr):
        key, vec_key = jax.random.split(key)
        vecs = jax.random.normal(vec_key, (N_DIRS, total_vec_dim))

        # Dual kernel: exact directional derivatives via forward-mode AD
        ce_primal, ce_tangent = fused_dual_forward(
            params["token_emb"], params["pos_emb"],
            params["layer0.ln1.scale"], params["layer0.ln1.bias"],
            params["layer0.attn.q"], params["layer0.attn.k"],
            params["layer0.attn.v"], params["layer0.attn.o"],
            params["layer0.ln2.scale"], params["layer0.ln2.bias"],
            params["layer0.ffn.up"], params["layer0.ffn.up_bias"],
            params["layer0.ffn.down"], params["layer0.ffn.down_bias"],
            params["ln_final.scale"], params["ln_final.bias"],
            params["output_proj"],
            vecs, x, y, ALPHA, TEMPERATURE,
        )

        # Directional derivatives: mean over batch items
        dlosses = ce_tangent.sum(axis=1) / x.shape[0]  # (N_DIRS,)

        # Normalize and scale to match finite-diff gradient magnitude
        shaped = winsorized_zscore(dlosses)
        scale = 1.0 / (2.0 * 0.020 * N_DIRS)  # match finite-diff scaling for Adam compatibility

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

        proxy = ce_primal.sum(axis=1).mean() / x.shape[0]
        return new_params, new_momentum, new_v, step + 1, key, proxy

    @jax.jit
    def train_batch(params, momentum_buf, v_buf, step, key, x, y, lr):
        return train_one_batch(params, momentum_buf, v_buf, step, key, x, y, lr)

    @jax.jit
    def eval_loss(params, x, y):
        from model import transformer_forward_batch, cross_entropy_loss
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
