"""Quick hyperparameter sweep for 10-epoch EGGROLL."""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import time
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np

from data import prepare_data
from model import init_transformer, count_params
from train_eggroll_optimized import build_param_spec

D_MODEL = 64; N_HEADS = 2; N_LAYERS = 1; CONTEXT_LEN = 128; BATCH_SIZE = 128
EPOCHS = 10; TEMPERATURE = 2.0; ALPHA = 0.50; N_SUBGROUPS = 8; CLIP_RANGE = 2.0


def winsorized_zscore(fitness_diffs):
    group_size = fitness_diffs.shape[0] // N_SUBGROUPS
    groups = fitness_diffs[:N_SUBGROUPS * group_size].reshape(N_SUBGROUPS, group_size)
    means = jnp.mean(groups, axis=1, keepdims=True)
    stds = jnp.std(groups, axis=1, keepdims=True) + 1e-8
    z = (groups - means) / stds
    z = jnp.clip(z, -CLIP_RANGE, CLIP_RANGE)
    return z.reshape(-1)


def run_config(sigma_start, lr_start, lr_decay, momentum, half_pop, seed=42):
    data = prepare_data(context_len=CONTEXT_LEN)
    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    params, config = init_transformer(init_key, data["vocab_size"], d_model=D_MODEL,
                                       n_heads=N_HEADS, n_layers=N_LAYERS, context_len=CONTEXT_LEN)
    spec, total_vec_dim = build_param_spec(params)
    n_batches = len(data["train_x"]) // BATCH_SIZE

    train_x = jnp.array(data["train_x"])
    train_y = jnp.array(data["train_y"])
    val_x = jnp.array(data["val_x"][:BATCH_SIZE])
    val_y = jnp.array(data["val_y"][:BATCH_SIZE])

    lr_scale_arr = []
    for pkey, shape, _, _, _ in spec:
        n_p = int(np.prod(shape))
        if n_p < 256: lr_scale_arr.append(3.0)
        elif n_p < 4096: lr_scale_arr.append(1.5)
        elif n_p < 8192: lr_scale_arr.append(1.0)
        else: lr_scale_arr.append(0.7)
    lr_scale_arr = jnp.array(lr_scale_arr)

    from kernels.fused_transformer_ce import fused_transformer_ce_both

    def train_one_batch(params, momentum_buf, key, x, y, sigma, lr):
        key, vec_key = jax.random.split(key)
        if half_pop <= total_vec_dim:
            raw = jax.random.normal(vec_key, (total_vec_dim, half_pop))
            Q, _ = jnp.linalg.qr(raw)
            vecs = Q.T * jnp.sqrt(jnp.float32(total_vec_dim))
        else:
            vecs = jax.random.normal(vec_key, (half_pop, total_vec_dim))

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
        scale = 1.0 / (2.0 * sigma * half_pop)

        new_params = {}
        new_momentum = {}
        for idx, (pkey, shape, offset, vec_dim, is_2d) in enumerate(spec):
            v = vecs[:, offset:offset + vec_dim]
            if is_2d:
                m, n = shape
                g = scale * (v[:, :m] * shaped[:, None]).T @ v[:, m:]
            else:
                g = scale * (v * shaped[:, None]).sum(axis=0)
            new_momentum[pkey] = momentum * momentum_buf[pkey] + g
            lr_s = lr_scale_arr[idx]
            new_params[pkey] = params[pkey] - lr * lr_s * new_momentum[pkey]

        return new_params, new_momentum, key

    train_batch = jax.jit(train_one_batch)

    @jax.jit
    def eval_loss(params, x, y):
        from model import transformer_forward_batch, cross_entropy_loss
        logits = transformer_forward_batch(params, config, x)
        return cross_entropy_loss(logits, y)

    momentum_buf = jax.tree.map(jnp.zeros_like, params)

    # warmup
    params_t, momentum_buf, key = train_batch(params, momentum_buf, key,
                                               train_x[:BATCH_SIZE], train_y[:BATCH_SIZE],
                                               sigma_start, lr_start)
    _ = jax.block_until_ready(params_t)

    # use original params for actual training
    momentum_buf = jax.tree.map(jnp.zeros_like, params)
    t0 = time.perf_counter()

    for epoch in range(EPOCHS):
        sigma = sigma_start * (0.998 ** epoch)
        lr = lr_start * (lr_decay ** epoch)
        key, sk = jax.random.split(key)
        perm = jax.random.permutation(sk, len(data["train_x"]))
        sx, sy = train_x[perm], train_y[perm]
        for bi in range(n_batches):
            s = bi * BATCH_SIZE
            params, momentum_buf, key = train_batch(
                params, momentum_buf, key,
                sx[s:s+BATCH_SIZE], sy[s:s+BATCH_SIZE], sigma, lr)

    vl = eval_loss(params, val_x, val_y)
    vl.block_until_ready()
    t = time.perf_counter() - t0
    return float(vl), t


def main():
    results = []

    configs = [
        # (sigma, lr, lr_decay, momentum, half_pop, label)
        (0.04, 0.020, 0.95, 0.5, 4096, "baseline"),
        (0.04, 0.025, 0.95, 0.5, 4096, "higher LR"),
        (0.04, 0.030, 0.93, 0.5, 4096, "LR=0.030 decay=0.93"),
        (0.04, 0.020, 0.90, 0.5, 4096, "faster decay"),
        (0.06, 0.020, 0.95, 0.5, 4096, "higher sigma"),
        (0.03, 0.020, 0.95, 0.5, 4096, "lower sigma"),
        (0.04, 0.015, 0.95, 0.7, 4096, "mom=0.7 LR=0.015"),
        (0.04, 0.012, 0.95, 0.7, 4096, "mom=0.7 LR=0.012"),
        (0.04, 0.020, 0.95, 0.6, 4096, "mom=0.6"),
        (0.04, 0.035, 0.90, 0.5, 4096, "aggressive LR+decay"),
    ]

    for sigma, lr, decay, mom, hp, label in configs:
        vl, t = run_config(sigma, lr, decay, mom, hp)
        print(f"  {label:30s}  val_loss={vl:.4f}  time={t:.0f}s")
        results.append((label, vl, t))

    print("\n=== SWEEP RESULTS ===")
    results.sort(key=lambda x: x[1])
    for label, vl, t in results:
        gap = vl - 2.45
        print(f"  {vl:.4f} ({gap:+.4f})  {t:6.0f}s  {label}")


if __name__ == "__main__":
    main()
