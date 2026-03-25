"""Backprop baseline for small decoder-only transformer on character-level Shakespeare."""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import time
import jax
import jax.numpy as jnp
import numpy as np

from data import prepare_data
from model import (
    init_transformer,
    transformer_forward_batch,
    cross_entropy_loss,
    count_params,
)

# hyperparameters
D_MODEL = 64
N_HEADS = 2
N_LAYERS = 1
CONTEXT_LEN = 128
BATCH_SIZE = 128
EPOCHS = 10
LR = 0.30   # tuned: gives val_loss=2.45 at 10 epochs
SEED = 42


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

    # move data to GPU
    train_x = jnp.array(data["train_x"])
    train_y = jnp.array(data["train_y"])
    val_x = jnp.array(data["val_x"])
    val_y = jnp.array(data["val_y"])

    @jax.jit
    def loss_fn(params, x, y):
        logits = transformer_forward_batch(params, config, x)
        return cross_entropy_loss(logits, y)

    @jax.jit
    def train_step(params, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        params = jax.tree.map(lambda p, g: p - LR * g, params, grads)
        return params, loss

    @jax.jit
    def eval_loss(params, x, y):
        logits = transformer_forward_batch(params, config, x)
        return cross_entropy_loss(logits, y)

    n_batches = len(data["train_x"]) // BATCH_SIZE
    print(f"Batches per epoch: {n_batches}")

    # warmup JIT
    print("Warming up JIT...")
    t_start = time.perf_counter()
    dummy_loss = loss_fn(params, train_x[:BATCH_SIZE], train_y[:BATCH_SIZE])
    dummy_loss.block_until_ready()
    jit_time = time.perf_counter() - t_start
    print(f"JIT warmup: {jit_time:.2f}s")

    print("\nTraining...")
    t_train_start = time.perf_counter()

    for epoch in range(EPOCHS):
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
            params, loss = train_step(params, bx, by)
            epoch_loss += loss

        avg_loss = epoch_loss / n_batches
        val_l = eval_loss(params, val_x[:BATCH_SIZE], val_y[:BATCH_SIZE])
        perplexity = jnp.exp(val_l)
        print(f"  Epoch {epoch+1}/{EPOCHS}  train_loss={avg_loss:.4f}  val_loss={val_l:.4f}  perplexity={perplexity:.2f}")

    train_time = time.perf_counter() - t_train_start
    # block for accurate timing
    val_l = eval_loss(params, val_x[:BATCH_SIZE], val_y[:BATCH_SIZE])
    val_l.block_until_ready()
    total_time = time.perf_counter() - t_train_start

    final_perplexity = float(jnp.exp(val_l))
    print(f"\nFinal val_loss={float(val_l):.4f}  perplexity={final_perplexity:.2f}")
    print(f"Training time: {total_time:.2f}s (including JIT: {total_time + jit_time:.2f}s)")

    # generate sample text
    generate_sample(params, config, data, key)

    return float(val_l), final_perplexity, total_time + jit_time


def generate_sample(params, config, data, key, length=200):
    """Generate a sample from the model."""
    char_to_idx = data["char_to_idx"]
    chars = data["chars"]

    # start with newline
    tokens = [char_to_idx["\n"]]

    @jax.jit
    def get_next_logits(tokens_arr):
        from model import transformer_forward
        return transformer_forward(params, config, tokens_arr)

    for _ in range(length):
        ctx = jnp.array(tokens[-config["context_len"]:], dtype=jnp.int32)
        logits = get_next_logits(ctx)
        next_logit = logits[-1]
        # temperature sampling
        key, sample_key = jax.random.split(key)
        next_token = jax.random.categorical(sample_key, next_logit / 0.8)
        tokens.append(int(next_token))

    text = "".join(chars[t] for t in tokens)
    print(f"\n--- Sample ---\n{text}\n--- End ---")


if __name__ == "__main__":
    loss, ppl, t = train()
    # log to results.tsv
    import subprocess
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open("results.tsv", "a") as f:
        if os.path.getsize("results.tsv") == 0 if os.path.exists("results.tsv") else True:
            f.write("commit\tloss\tperplexity\ttraining_time_s\tpeak_memory_mb\tstatus\tdescription\n")
        f.write(f"{commit}\t{loss:.4f}\t{ppl:.2f}\t{t:.2f}\t0\tok\tbackprop baseline d={D_MODEL} L={N_LAYERS} H={N_HEADS}\n")
