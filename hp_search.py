"""Quick hyperparameter search for HALF_POP=2048."""
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import time
import jax
import jax.numpy as jnp
import numpy as np

from data import prepare_data
from model import init_transformer
from train_eggroll_optimized import build_param_spec
from kernels.fused_transformer_ce import fused_transformer_ce_both

D_MODEL = 64
N_HEADS = 2
N_LAYERS = 1
CONTEXT_LEN = 128
BATCH_SIZE = 128
EPOCHS = 10
TEMPERATURE = 2.0

def winsorized_zscore(fitness_diffs, n_subgroups=8, clip_range=2.0):
    group_size = fitness_diffs.shape[0] // n_subgroups
    groups = fitness_diffs[:n_subgroups * group_size].reshape(n_subgroups, group_size)
    means = jnp.mean(groups, axis=1, keepdims=True)
    stds = jnp.std(groups, axis=1, keepdims=True) + 1e-8
    z = (groups - means) / stds
    z = jnp.clip(z, -clip_range, clip_range)
    return z.reshape(-1)


def run_config(half_pop, sigma, lr, alpha, momentum=0.9, beta2=0.999, seed=42):
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
    eps = 1e-6

    def train_one_batch(params, mom_buf, v_buf, step, key, x, y, sigma_val, lr_val):
        key, vec_key = jax.random.split(key)
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
            vecs, x, y, sigma_val, alpha, TEMPERATURE,
        )
        fp = ce_pos.sum(axis=1) / x.shape[0]
        fn = ce_neg.sum(axis=1) / x.shape[0]
        diffs = fp - fn
        shaped = winsorized_zscore(diffs)
        scale = 1.0 / (2.0 * sigma_val * half_pop)
        new_params = {}
        new_mom = {}
        new_v = {}
        t = step + 1
        for idx, (pkey, shape, offset, vec_dim, is_2d) in enumerate(spec):
            v_pert = vecs[:, offset:offset + vec_dim]
            if is_2d:
                m, n = shape
                g = scale * (v_pert[:, :m] * shaped[:, None]).T @ v_pert[:, m:]
            else:
                g = scale * (v_pert * shaped[:, None]).sum(axis=0)
            new_mom[pkey] = momentum * mom_buf[pkey] + (1 - momentum) * g
            new_v[pkey] = beta2 * v_buf[pkey] + (1 - beta2) * g ** 2
            m_hat = new_mom[pkey] / (1 - momentum ** t)
            v_hat = new_v[pkey] / (1 - beta2 ** t)
            new_params[pkey] = params[pkey] - lr_val * lr_scale_arr[idx] * m_hat / (jnp.sqrt(v_hat) + eps)
        return new_params, new_mom, new_v, step + 1, key, jnp.mean(fp)

    train_batch = jax.jit(train_one_batch)

    @jax.jit
    def eval_loss(params, x, y):
        from model import transformer_forward_batch, cross_entropy_loss
        logits = transformer_forward_batch(params, config, x)
        return cross_entropy_loss(logits, y)

    mom_buf = jax.tree.map(jnp.zeros_like, params)
    v_buf = jax.tree.map(jnp.zeros_like, params)
    step = jnp.int32(0)

    t_start = time.perf_counter()
    sigma_decay = 0.998
    for epoch in range(EPOCHS):
        s_val = sigma * (sigma_decay ** epoch)
        key, sk = jax.random.split(key)
        perm = jax.random.permutation(sk, len(data["train_x"]))
        sx, sy = train_x[perm], train_y[perm]
        for bi in range(n_batches):
            s = bi * BATCH_SIZE
            params, mom_buf, v_buf, step, key, pl = train_batch(
                params, mom_buf, v_buf, step, key,
                sx[s:s+BATCH_SIZE], sy[s:s+BATCH_SIZE], s_val, lr)
        vl = eval_loss(params, val_x[:BATCH_SIZE], val_y[:BATCH_SIZE])
        vl.block_until_ready()
    total = time.perf_counter() - t_start
    return float(vl), total


configs = [
    # (half_pop, sigma, lr, alpha)
    (2560, 0.020, 0.010, 0.50),  # baseline sigma/lr
    (2560, 0.015, 0.010, 0.50),  # lower sigma
    (2560, 0.020, 0.012, 0.50),  # slightly higher LR
    (2560, 0.020, 0.010, 0.40),  # less smoothing
    (3072, 0.020, 0.012, 0.50),  # 3072 with higher LR
    (3072, 0.015, 0.010, 0.50),  # 3072 with lower sigma
]

print("Hyperparameter search for HALF_POP=2048")
print(f"{'half_pop':>8s} {'sigma':>8s} {'lr':>8s} {'alpha':>8s} {'val_loss':>10s} {'time':>8s}")
print("-" * 60)

results = []
for hp, sigma, lr, alpha in configs:
    try:
        vl, t = run_config(hp, sigma, lr, alpha)
        status = "OK" if vl <= 2.50 else "FAIL"
        print(f"{hp:8d} {sigma:8.3f} {lr:8.3f} {alpha:8.2f} {vl:10.4f} {t:8.1f}s  {status}")
        results.append((hp, sigma, lr, alpha, vl, t))
    except Exception as e:
        print(f"{hp:8d} {sigma:8.3f} {lr:8.3f} {alpha:8.2f}  ERROR: {e}")

print("\nBest configs (val_loss ≤ 2.50):")
ok = [r for r in results if r[4] <= 2.50]
for r in sorted(ok, key=lambda x: x[5]):
    print(f"  sigma={r[1]}, lr={r[2]}, alpha={r[3]}: val_loss={r[4]:.4f}, time={r[5]:.1f}s")

# Save results
with open("hp_search_results.txt", "w") as f:
    for r in results:
        f.write(f"half_pop={r[0]} sigma={r[1]} lr={r[2]} alpha={r[3]} val_loss={r[4]:.4f} time={r[5]:.1f}s\n")
