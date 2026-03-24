"""EGGROLL ES training with fused Triton kernel for the full forward pass.

Replaces vmap+scan with a single kernel launch per ES round that processes
ALL perturbation members in parallel via the CUDA grid.
"""

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

# === architecture ===
D_MODEL = 64
N_HEADS = 2
N_LAYERS = 1
CONTEXT_LEN = 128
BATCH_SIZE = 128
EPOCHS = 20
SEED = 42

# === EGGROLL hyperparameters ===
HALF_POP = 4096
SIGMA_START = 0.04
SIGMA_DECAY = 0.998
LR_START = 0.020
LR_DECAY = 0.95
ALPHA = 0.50
TEMPERATURE = 2.0
N_SUBGROUPS = 8
CLIP_RANGE = 2.0
MOMENTUM = 0.5
N_ACCUM = 1


def winsorized_zscore(fitness_diffs):
    group_size = fitness_diffs.shape[0] // N_SUBGROUPS
    groups = fitness_diffs[:N_SUBGROUPS * group_size].reshape(N_SUBGROUPS, group_size)
    means = jnp.mean(groups, axis=1, keepdims=True)
    stds = jnp.std(groups, axis=1, keepdims=True) + 1e-8
    z = (groups - means) / stds
    z = jnp.clip(z, -CLIP_RANGE, CLIP_RANGE)
    return z.reshape(-1)


def train():
    print("Preparing data...")
    data = prepare_data(context_len=CONTEXT_LEN)
    vocab_size = data["vocab_size"]
    print(f"Vocab: {vocab_size}, Train: {data['train_x'].shape}, Val: {data['val_x'].shape}")

    key = jax.random.key(SEED)
    key, init_key = jax.random.split(key)
    params, config = init_transformer(
        init_key, vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, context_len=CONTEXT_LEN,
    )
    n_params = count_params(params)
    spec, total_vec_dim = build_param_spec(params)
    n_batches = len(data["train_x"]) // BATCH_SIZE
    print(f"Params: {n_params:,}, Vec dim: {total_vec_dim}, Pop: {HALF_POP*2}, Batches: {n_batches}")

    train_x = jnp.array(data["train_x"])
    train_y = jnp.array(data["train_y"])
    val_x = jnp.array(data["val_x"])
    val_y = jnp.array(data["val_y"])

    # per-layer LR scales
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
        """One batch: N_ACCUM rounds of ES gradient -> average -> momentum update."""

        def one_es_round(carry, _):
            key_r = carry
            key_r, vec_key = jax.random.split(key_r)
            # Orthogonal vectors via QR when possible, else Gaussian
            if HALF_POP <= total_vec_dim:
                raw = jax.random.normal(vec_key, (total_vec_dim, HALF_POP))
                Q, _ = jnp.linalg.qr(raw)
                vecs = Q.T * jnp.sqrt(jnp.float32(total_vec_dim))
            else:
                vecs = jax.random.normal(vec_key, (HALF_POP, total_vec_dim))

            # Fused Triton kernel: all perturbations + both signs in one launch
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
                vecs, x, y,
                sigma, ALPHA, TEMPERATURE,
            )

            # Reduce: mean over batch
            fp = ce_pos.sum(axis=1) / x.shape[0]  # (HALF_POP,)
            fn = ce_neg.sum(axis=1) / x.shape[0]

            diffs = fp - fn
            shaped = winsorized_zscore(diffs)
            scale = 1.0 / (2.0 * sigma * HALF_POP)

            round_grads = {}
            for idx, (pkey, shape, offset, vec_dim, is_2d) in enumerate(spec):
                v = vecs[:, offset:offset + vec_dim]
                if is_2d:
                    m, n = shape
                    round_grads[pkey] = scale * (v[:, :m] * shaped[:, None]).T @ v[:, m:]
                else:
                    round_grads[pkey] = scale * (v * shaped[:, None]).sum(axis=0)
            return key_r, (round_grads, jnp.mean(fp))

        key, (all_grads, all_losses) = lax.scan(one_es_round, key, None, length=N_ACCUM)

        new_params = {}
        new_momentum = {}
        for idx, (pkey, shape, offset, vec_dim, is_2d) in enumerate(spec):
            avg_grad = jnp.mean(all_grads[pkey], axis=0)
            lr_s = lr_scale_arr[idx]
            new_momentum[pkey] = MOMENTUM * momentum_buf[pkey] + avg_grad
            new_params[pkey] = params[pkey] - lr * lr_s * new_momentum[pkey]

        return new_params, new_momentum, key, jnp.mean(all_losses)

    @jax.jit
    def train_batch(params, momentum_buf, key, x, y, sigma, lr):
        return train_one_batch(params, momentum_buf, key, x, y, sigma, lr)

    @jax.jit
    def eval_loss(params, x, y):
        from model import transformer_forward_batch, cross_entropy_loss
        logits = transformer_forward_batch(params, config, x)
        return cross_entropy_loss(logits, y)

    momentum_buf = jax.tree.map(jnp.zeros_like, params)

    # warmup
    print("Warming up JIT...")
    t0 = time.perf_counter()
    _, _, _, wl = train_batch(params, momentum_buf, key, train_x[:BATCH_SIZE], train_y[:BATCH_SIZE], SIGMA_START, LR_START)
    wl.block_until_ready()
    jit_time = time.perf_counter() - t0
    print(f"JIT warmup: {jit_time:.2f}s")

    sigmas = [SIGMA_START * (SIGMA_DECAY ** e) for e in range(EPOCHS)]
    lrs_sched = [LR_START * (LR_DECAY ** e) for e in range(EPOCHS)]

    print("\nTraining...")
    t_start = time.perf_counter()

    for epoch in range(EPOCHS):
        sigma, lr = sigmas[epoch], lrs_sched[epoch]
        key, sk = jax.random.split(key)
        perm = jax.random.permutation(sk, len(data["train_x"]))
        sx, sy = train_x[perm], train_y[perm]
        eloss = 0.0
        for bi in range(n_batches):
            s = bi * BATCH_SIZE
            params, momentum_buf, key, pl = train_batch(
                params, momentum_buf, key,
                sx[s:s+BATCH_SIZE], sy[s:s+BATCH_SIZE], sigma, lr)
            eloss += float(pl)
        eloss /= n_batches

        vl = eval_loss(params, val_x[:BATCH_SIZE], val_y[:BATCH_SIZE])
        print(f"  Epoch {epoch+1}/{EPOCHS}  proxy={eloss:.4f}  val_loss={float(vl):.4f}  ppl={float(jnp.exp(vl)):.2f}  lr={lr:.4f}")

    total = time.perf_counter() - t_start
    vl = eval_loss(params, val_x[:BATCH_SIZE], val_y[:BATCH_SIZE])
    vl.block_until_ready()
    total = time.perf_counter() - t_start
    ppl = float(jnp.exp(vl))
    print(f"\nFinal val_loss={float(vl):.4f}  ppl={ppl:.2f}")
    print(f"Training: {total:.2f}s  (with JIT: {total+jit_time:.2f}s)")

    from train_backprop import generate_sample
    generate_sample(params, config, data, key)
    return float(vl), ppl, total + jit_time


if __name__ == "__main__":
    loss, ppl, t = train()
    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open("results.tsv", "a") as f:
        f.write(f"{commit}\t{loss:.4f}\t{ppl:.2f}\t{t:.2f}\t0\tok\teggroll_triton fused kernel pop={HALF_POP*2} accum={N_ACCUM}\n")
