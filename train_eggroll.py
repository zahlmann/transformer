"""EGGROLL ES training for small decoder-only transformer on character-level Shakespeare.

EGGROLL = Evolution Strategies with low-rank (rank-1) perturbations.
Instead of perturbing each param independently (needing N_params random values),
we perturb each weight matrix W with sigma * outer(B, A), needing only (m+n) values.
For 1D params (biases, norms), we perturb directly.

Key lessons from MNIST applied here:
- Label-smoothed, temperature-scaled CE for richer fitness signal
- Per-subgroup Winsorized z-score normalization
- Adaptive smoothing schedule (high early, low late)
- Antithetic sampling (±sigma pairs)
"""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import time
import functools
import jax
import jax.numpy as jnp
import jax.lax as lax
import numpy as np

from data import prepare_data
from model import (
    init_transformer,
    transformer_forward_batch,
    count_params,
)

# === architecture (must match backprop baseline) ===
D_MODEL = 64
N_HEADS = 2
N_LAYERS = 1
CONTEXT_LEN = 128
BATCH_SIZE = 128
EPOCHS = 10
SEED = 42

# === EGGROLL hyperparameters ===
HALF_POP = 512          # per accumulation round
POP_CHUNK = 16          # process this many perturbation pairs at a time (memory)
N_ACCUM = 4             # gradient accumulation rounds -> effective pop = HALF_POP * N_ACCUM * 2 = 4096
SIGMA_START = 0.04      # perturbation scale
SIGMA_DECAY = 0.998     # per epoch (slow decay)
LR_START = 0.020        # learning rate
LR_DECAY = 0.85         # per epoch
ALPHA_START = 0.20      # label smoothing (constant - provides regularization)
ALPHA_DECAY = 1.00      # no decay - smoothing prevents ES-induced overfitting
TEMPERATURE = 2.0       # CE temperature
N_SUBGROUPS = 8         # for Winsorized z-score
CLIP_RANGE = 2.0        # z-score clipping


def build_param_spec(params):
    """Build a specification of how to slice the random vector into per-param perturbations.

    For 2D params (weight matrices): rank-1 perturbation, need (m+n) random values.
    For 1D params (biases, norms): direct perturbation, need (size,) random values.

    Returns: list of (param_key, shape, slice_start, vec_dim, is_2d)
    """
    spec = []
    offset = 0
    for key in sorted(params.keys()):
        shape = params[key].shape
        if len(shape) == 2:
            m, n = shape
            vec_dim = m + n
            spec.append((key, shape, offset, vec_dim, True))
        elif len(shape) == 1:
            vec_dim = shape[0]
            spec.append((key, shape, offset, vec_dim, False))
        else:
            raise ValueError(f"Unexpected param shape: {key} {shape}")
        offset += vec_dim

    total_vec_dim = offset
    return spec, total_vec_dim


def apply_perturbation(params, spec, vec, sigma):
    """Apply rank-1 perturbation to params using random vector `vec`.

    vec: (vec_dim,) random vector.
    For 2D params: W + sigma * outer(B, A) where B=vec[..m], A=vec[m..m+n]
    For 1D params: p + sigma * vec[..size]
    """
    new_params = {}
    for key, shape, offset, vec_dim, is_2d in spec:
        v = lax.dynamic_slice(vec, (offset,), (vec_dim,))
        if is_2d:
            m, n = shape
            b = v[:m]
            a = v[m:]
            new_params[key] = params[key] + sigma * jnp.outer(b, a)
        else:
            new_params[key] = params[key] + sigma * v
    return new_params


def smoothed_ce_loss(logits, targets, alpha, temperature):
    """Temperature-scaled, label-smoothed cross-entropy.

    This provides richer fitness signal than hard CE by giving gradient
    information about ALL logits, not just the correct class.
    """
    vocab_size = logits.shape[-1]
    # temperature scaling
    scaled_logits = logits / temperature
    log_probs = jax.nn.log_softmax(scaled_logits, axis=-1)

    # smooth labels
    one_hot = jax.nn.one_hot(targets, vocab_size)
    smooth_labels = (1 - alpha) * one_hot + alpha / vocab_size

    # CE
    loss = -jnp.sum(smooth_labels * log_probs, axis=-1)
    return jnp.mean(loss)


def winsorized_zscore(fitness_diffs, n_subgroups, clip_range):
    """Per-subgroup Winsorized z-score normalization.

    Split into subgroups, normalize each independently, clip outliers.
    Much better than global z-score at low population sizes.
    """
    group_size = fitness_diffs.shape[0] // n_subgroups
    groups = fitness_diffs[:n_subgroups * group_size].reshape(n_subgroups, group_size)

    means = jnp.mean(groups, axis=1, keepdims=True)
    stds = jnp.std(groups, axis=1, keepdims=True) + 1e-8
    z = (groups - means) / stds
    z = jnp.clip(z, -clip_range, clip_range)

    return z.reshape(-1)


def eggroll_gradient(params, spec, total_vec_dim, vecs, fitness_pos, fitness_neg, sigma):
    """Estimate gradient from antithetic fitness evaluations.

    gradient_W = (1/2*sigma*N) * sum_i [fitness_diff_i * perturbation_i]
    With rank-1: perturbation = outer(B_i, A_i), so:
    gradient_W = scale * B.T @ (shaped_fitness * A)  [matrix form]
    """
    fitness_diffs = fitness_pos - fitness_neg  # ES gradient: (f(W+σd) - f(W-σd)) / 2σ
    shaped = winsorized_zscore(fitness_diffs, N_SUBGROUPS, CLIP_RANGE)

    half_pop = vecs.shape[0]
    scale = 1.0 / (2.0 * sigma * half_pop)

    grads = {}
    for key, shape, offset, vec_dim, is_2d in spec:
        v = vecs[:, offset:offset + vec_dim]  # (half_pop, vec_dim)
        if is_2d:
            m, n = shape
            B = v[:, :m]   # (half_pop, m)
            A = v[:, m:]   # (half_pop, n)
            # grad = scale * B.T @ diag(shaped) @ A = scale * (B * shaped[:, None]).T @ A
            grads[key] = scale * (B * shaped[:, None]).T @ A
        else:
            # grad = scale * sum_i shaped_i * v_i
            grads[key] = scale * (v * shaped[:, None]).sum(axis=0)

    return grads


def train():
    print("Preparing data...")
    data = prepare_data(context_len=CONTEXT_LEN)
    vocab_size = data["vocab_size"]
    print(f"Vocab: {vocab_size}, Train: {data['train_x'].shape}, Val: {data['val_x'].shape}")

    # init model
    key = jax.random.key(SEED)
    key, init_key = jax.random.split(key)
    params, config = init_transformer(
        init_key, vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, context_len=CONTEXT_LEN,
    )
    n_params = count_params(params)
    print(f"Parameters: {n_params:,}")

    # build perturbation spec
    spec, total_vec_dim = build_param_spec(params)
    print(f"Perturbation vector dim: {total_vec_dim} (vs {n_params} full params, {n_params/total_vec_dim:.1f}x compression)")
    print(f"Half population: {HALF_POP} ({HALF_POP*2} forward passes per batch)")

    # move data to GPU
    train_x = jnp.array(data["train_x"])
    train_y = jnp.array(data["train_y"])
    val_x = jnp.array(data["val_x"])
    val_y = jnp.array(data["val_y"])

    # precompute schedules
    sigmas = jnp.array([SIGMA_START * (SIGMA_DECAY ** e) for e in range(EPOCHS)])
    lrs = jnp.array([LR_START * (LR_DECAY ** e) for e in range(EPOCHS)])
    alphas = jnp.array([ALPHA_START * (ALPHA_DECAY ** e) for e in range(EPOCHS)])

    n_chunks = HALF_POP // POP_CHUNK

    @jax.jit
    def compute_fitness_batch(params, vecs, x, y, sigma, alpha):
        """Compute fitness for all perturbation pairs, chunked to manage memory.

        Process POP_CHUNK perturbation pairs at a time via lax.scan.
        """
        def fitness_one(vec, sign):
            perturbed = apply_perturbation(params, spec, vec * sign, sigma)
            logits = transformer_forward_batch(perturbed, config, x)
            return smoothed_ce_loss(logits, y, alpha, TEMPERATURE)

        def fitness_chunk(carry, chunk_vecs):
            # chunk_vecs: (POP_CHUNK, vec_dim)
            def fitness_pair(vec):
                pos = fitness_one(vec, 1.0)
                neg = fitness_one(vec, -1.0)
                return pos, neg
            fp, fn = jax.vmap(fitness_pair)(chunk_vecs)
            return carry, (fp, fn)

        # reshape vecs into chunks
        vecs_chunked = vecs.reshape(n_chunks, POP_CHUNK, -1)
        _, (fitness_pos, fitness_neg) = lax.scan(fitness_chunk, None, vecs_chunked)
        # flatten back: (n_chunks, POP_CHUNK) -> (HALF_POP,)
        return fitness_pos.reshape(-1), fitness_neg.reshape(-1)

    # per-layer LR scaling: small params get better gradient estimates
    lr_scales = {}
    for pkey, shape, _, _, is_2d in spec:
        n_params = np.prod(shape)
        if n_params < 256:         # biases, layer norms
            lr_scales[pkey] = 3.0
        elif n_params < 4096:      # small matrices (attn)
            lr_scales[pkey] = 1.5
        elif n_params < 8192:      # medium (embeddings, output)
            lr_scales[pkey] = 1.0
        else:                      # large (FFN)
            lr_scales[pkey] = 0.7
    print(f"LR scales: {dict(sorted([(k, v) for k, v in lr_scales.items()], key=lambda x: x[0]))}")

    @jax.jit
    def update_step(params, vecs, fitness_pos, fitness_neg, sigma, lr):
        grads = eggroll_gradient(params, spec, total_vec_dim, vecs, fitness_pos, fitness_neg, sigma)
        new_params = {}
        for key in params:
            new_params[key] = params[key] - lr * lr_scales[key] * grads[key]
        return new_params

    @jax.jit
    def eval_loss(params, x, y):
        logits = transformer_forward_batch(params, config, x)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        target_log_probs = jnp.take_along_axis(log_probs, y[..., None], axis=-1).squeeze(-1)
        return -jnp.mean(target_log_probs)

    n_batches = len(data["train_x"]) // BATCH_SIZE

    # warmup JIT with one fitness eval
    print("Warming up JIT...")
    t_start = time.perf_counter()
    key, warmup_key = jax.random.split(key)
    warmup_vecs = jax.random.normal(warmup_key, (HALF_POP, total_vec_dim))
    fp, fn = compute_fitness_batch(
        params, warmup_vecs,
        train_x[:BATCH_SIZE], train_y[:BATCH_SIZE],
        SIGMA_START, ALPHA_START,
    )
    fp.block_until_ready()
    jit_time = time.perf_counter() - t_start
    print(f"JIT warmup: {jit_time:.2f}s")

    print("\nTraining...")
    t_train_start = time.perf_counter()

    for epoch in range(EPOCHS):
        sigma = float(sigmas[epoch])
        lr = float(lrs[epoch])
        alpha = float(alphas[epoch])

        # shuffle
        key, shuffle_key = jax.random.split(key)
        perm = jax.random.permutation(shuffle_key, len(data["train_x"]))
        shuffled_x = train_x[perm]
        shuffled_y = train_y[perm]

        epoch_loss = 0.0
        for batch_idx in range(n_batches):
            start = batch_idx * BATCH_SIZE
            bx = shuffled_x[start : start + BATCH_SIZE]
            by = shuffled_y[start : start + BATCH_SIZE]

            # gradient accumulation: multiple rounds of perturbation evaluation
            accum_grads = {k: jnp.zeros_like(params[k]) for k in params}
            batch_loss = 0.0
            for accum_idx in range(N_ACCUM):
                key, vec_key = jax.random.split(key)
                vecs = jax.random.normal(vec_key, (HALF_POP, total_vec_dim))
                fitness_pos, fitness_neg = compute_fitness_batch(params, vecs, bx, by, sigma, alpha)
                grads = eggroll_gradient(params, spec, total_vec_dim, vecs, fitness_pos, fitness_neg, sigma)
                accum_grads = {k: accum_grads[k] + grads[k] for k in params}
                batch_loss += float(jnp.mean(fitness_pos))

            # average and apply
            params = {k: params[k] - lr * lr_scales[k] * accum_grads[k] / N_ACCUM for k in params}
            epoch_loss += batch_loss / N_ACCUM

        val_l = eval_loss(params, val_x[:BATCH_SIZE], val_y[:BATCH_SIZE])
        perplexity = jnp.exp(val_l)
        print(f"  Epoch {epoch+1}/{EPOCHS}  proxy_loss={epoch_loss/n_batches:.4f}  val_loss={float(val_l):.4f}  perplexity={float(perplexity):.2f}  sigma={sigma:.4f}  lr={lr:.4f}  alpha={alpha:.4f}")

    train_time = time.perf_counter() - t_train_start
    val_l = eval_loss(params, val_x[:BATCH_SIZE], val_y[:BATCH_SIZE])
    val_l.block_until_ready()
    total_time = time.perf_counter() - t_train_start

    final_perplexity = float(jnp.exp(val_l))
    print(f"\nFinal val_loss={float(val_l):.4f}  perplexity={final_perplexity:.2f}")
    print(f"Training time: {total_time:.2f}s (including JIT: {total_time + jit_time:.2f}s)")

    # generate sample
    from train_backprop import generate_sample
    generate_sample(params, config, data, key)

    return float(val_l), final_perplexity, total_time + jit_time


if __name__ == "__main__":
    loss, ppl, t = train()
    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open("results.tsv", "a") as f:
        if not os.path.exists("results.tsv") or os.path.getsize("results.tsv") == 0:
            f.write("commit\tloss\tperplexity\ttraining_time_s\tpeak_memory_mb\tstatus\tdescription\n")
        f.write(f"{commit}\t{loss:.4f}\t{ppl:.2f}\t{t:.2f}\t0\tok\teggroll pop={HALF_POP*2} d={D_MODEL} L={N_LAYERS}\n")
