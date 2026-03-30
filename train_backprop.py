"""Backprop+Adam baseline.

Usage: uv run train_backprop.py [--seed SEED] [--lr LR] [--tokenizer char|bpe] [--bpe-vocab 512]
"""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import argparse
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

from data import prepare_data
from model import init_transformer, transformer_forward_batch, cross_entropy_loss, count_params

D_MODEL = 128; N_HEADS = 4; N_LAYERS = 2; CONTEXT_LEN = 128; BATCH_SIZE = 64
EPOCHS = 20; SEED = 42


def train(lr=1e-3, seed=SEED, tokenizer="char", bpe_vocab_size=512):
    data = prepare_data(context_len=CONTEXT_LEN, tokenizer=tokenizer, bpe_vocab_size=bpe_vocab_size)
    vocab_size = data["vocab_size"]

    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    params, config = init_transformer(init_key, vocab_size, d_model=D_MODEL,
                                       n_heads=N_HEADS, n_layers=N_LAYERS, context_len=CONTEXT_LEN)

    train_x = jnp.array(data["train_x"])
    train_y = jnp.array(data["train_y"])
    val_x = jnp.array(data["val_x"][:BATCH_SIZE])
    val_y = jnp.array(data["val_y"][:BATCH_SIZE])
    n_batches = len(data["train_x"]) // BATCH_SIZE

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    @jax.jit
    def loss_fn(params, x, y):
        logits = transformer_forward_batch(params, config, x)
        return cross_entropy_loss(logits, y)

    @jax.jit
    def train_step(params, opt_state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def eval_loss(params, x, y):
        logits = transformer_forward_batch(params, config, x)
        return cross_entropy_loss(logits, y)

    print(f"=== Backprop+Adam LR={lr} seed={seed} tokenizer={tokenizer} vocab={vocab_size} ===")
    print(f"Params: {count_params(params):,}")

    t_start = time.perf_counter()
    for epoch in range(EPOCHS):
        key, sk = jax.random.split(key)
        perm = jax.random.permutation(sk, len(data["train_x"]))
        sx, sy = train_x[perm], train_y[perm]
        eloss = 0.0
        for bi in range(n_batches):
            s = bi * BATCH_SIZE
            params, opt_state, loss = train_step(params, opt_state,
                                                  sx[s:s+BATCH_SIZE], sy[s:s+BATCH_SIZE])
            eloss += float(loss)
        eloss /= n_batches
        vl = eval_loss(params, val_x, val_y)
        print(f"  Epoch {epoch+1}/{EPOCHS}  train={eloss:.4f}  val={float(vl):.4f}  ppl={float(jnp.exp(vl)):.2f}")

    vl = eval_loss(params, val_x, val_y)
    vl.block_until_ready()
    total = time.perf_counter() - t_start

    mem_stats = jax.local_devices()[0].memory_stats()
    peak_mb = mem_stats["peak_bytes_in_use"] / 1e6 if mem_stats else 0.0

    print(f"\nFinal: val_loss={float(vl):.4f}  ppl={float(jnp.exp(vl)):.2f}  time={total:.1f}s  mem={peak_mb:.0f}MB")

    # Save weights for inference
    import pickle
    save_path = os.path.join(os.path.dirname(__file__), "weights.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({
            "params": jax.tree.map(np.asarray, params),
            "config": config,
            "tokenizer": tokenizer,
        }, f)
    print(f"Saved weights to weights.pkl (tokenizer={tokenizer})")

    return float(vl), total, peak_mb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", type=str, default="char", choices=["char", "bpe", "trained_bpe"])
    parser.add_argument("--bpe-vocab", type=int, default=512)
    args = parser.parse_args()
    train(lr=args.lr, seed=args.seed, tokenizer=args.tokenizer, bpe_vocab_size=args.bpe_vocab)
