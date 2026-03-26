"""EGGROLL ES training with fused Triton kernel for the full forward pass.

Replaces vmap+scan with a single kernel launch per ES round that processes
ALL perturbation members in parallel via the CUDA grid.

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
HALF_POP = 8192
SIGMA_START = 0.022
SIGMA_DECAY = 0.998
LR_START = 0.010
LR_DECAY = 1.0  # no decay for Adam
ALPHA = 0.50
N_SUBGROUPS = 8
CLIP_RANGE = 2.0
MOMENTUM = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-6


def winsorized_zscore(fitness_diffs):
    group_size = fitness_diffs.shape[0] // N_SUBGROUPS
    groups = fitness_diffs[:N_SUBGROUPS * group_size].reshape(N_SUBGROUPS, group_size)
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

    # uniform LR scaling (Adam handles per-param adaptation)
    lr_scale_arr = jnp.ones(len(spec))

    from kernels.fused_transformer_ce import fused_transformer_ce_both

    def train_one_batch(params, momentum_buf, v_buf, step, key, x, y, sigma, lr):
        key, vec_key = jax.random.split(key)
        if HALF_POP <= total_vec_dim:
            raw = jax.random.normal(vec_key, (total_vec_dim, HALF_POP))
            Q, _ = jnp.linalg.qr(raw)
            vecs = Q.T * jnp.sqrt(jnp.float32(total_vec_dim))
        else:
            vecs = jax.random.normal(vec_key, (HALF_POP, total_vec_dim))

        ce_pos, ce_neg = fused_transformer_ce_both(
            params["token_emb"], params["pos_emb"],
            params["layer0.ln1.scale"], params["layer0.ln1.bias"],
            params["layer0.attn.q"], params["layer0.attn.k"],
            params["layer0.attn.v"], params["layer0.attn.o"],
            params["layer0.ln2.scale"], params["layer0.ln2.bias"],
            params["layer0.ffn.up"], params["layer0.ffn.up_bias"],
            params["layer0.ffn.down"], params["layer0.ffn.down_bias"],
            params["ln_final.scale"], params["ln_final.bias"],
            params["output_proj"],
            vecs, x, y, sigma, ALPHA, TEMPERATURE,
        )

        fp = ce_pos.sum(axis=1) / x.shape[0]
        fn = ce_neg.sum(axis=1) / x.shape[0]
        diffs = fp - fn
        shaped = winsorized_zscore(diffs)
        scale = 1.0 / (2.0 * sigma * HALF_POP)

        new_params = {}
        new_momentum = {}
        new_v = {}
        t = step + 1  # 1-indexed for bias correction
        for idx, (pkey, shape, offset, vec_dim, is_2d) in enumerate(spec):
            v_pert = vecs[:, offset:offset + vec_dim]
            if is_2d:
                m, n = shape
                g = scale * (v_pert[:, :m] * shaped[:, None]).T @ v_pert[:, m:]
            else:
                g = scale * (v_pert * shaped[:, None]).sum(axis=0)
            lr_s = lr_scale_arr[idx]
            # First moment (momentum)
            new_momentum[pkey] = MOMENTUM * momentum_buf[pkey] + (1 - MOMENTUM) * g
            # Second moment (Adam)
            new_v[pkey] = ADAM_BETA2 * v_buf[pkey] + (1 - ADAM_BETA2) * g ** 2
            # Bias correction
            m_hat = new_momentum[pkey] / (1 - MOMENTUM ** t)
            v_hat = new_v[pkey] / (1 - ADAM_BETA2 ** t)
            new_params[pkey] = params[pkey] - lr * lr_s * m_hat / (jnp.sqrt(v_hat) + ADAM_EPS)

        return new_params, new_momentum, new_v, step + 1, key, jnp.mean(fp)

    @jax.jit
    def train_batch(params, momentum_buf, v_buf, step, key, x, y, sigma, lr):
        return train_one_batch(params, momentum_buf, v_buf, step, key, x, y, sigma, lr)

    @jax.jit
    def eval_loss(params, x, y):
        from model import transformer_forward_batch, cross_entropy_loss
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

    # Training (JIT warmup happens on first batch — included in timing)
    t_start = time.perf_counter()

    sigmas = [SIGMA_START * (SIGMA_DECAY ** e) for e in range(EPOCHS)]
    lrs_sched = [LR_START * (LR_DECAY ** e) for e in range(EPOCHS)]

    for epoch in range(EPOCHS):
        sigma, lr = sigmas[epoch], lrs_sched[epoch]
        key, sk = jax.random.split(key)
        perm = jax.random.permutation(sk, len(data["train_x"]))
        sx, sy = train_x[perm], train_y[perm]
        eloss = jnp.float32(0.0)
        for bi in range(n_batches):
            s = bi * BATCH_SIZE
            params, momentum_buf, v_buf, step, key, pl = train_batch(
                params, momentum_buf, v_buf, step, key,
                sx[s:s+BATCH_SIZE], sy[s:s+BATCH_SIZE], sigma, lr)
            eloss = eloss + pl  # no float() sync — let XLA pipeline batches

        vl = eval_loss(params, val_x[:BATCH_SIZE], val_y[:BATCH_SIZE])
        print(f"  Epoch {epoch+1}/{EPOCHS}  proxy={float(eloss)/n_batches:.4f}  val_loss={float(vl):.4f}  ppl={float(jnp.exp(vl)):.2f}  lr={lr:.4f}")

    vl = eval_loss(params, val_x[:BATCH_SIZE], val_y[:BATCH_SIZE])
    vl.block_until_ready()
    total = time.perf_counter() - t_start
    ppl = float(jnp.exp(vl))
    val_loss = float(vl)

    # Measure peak memory
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
