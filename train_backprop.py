"""Backprop+Adam baseline.

Usage: uv run train_backprop.py [--seed SEED] [--lr LR] [--tokenizer char|bpe|trained_bpe]
                                [--bpe-vocab 512] [--dataset shakespeare|tinystories]
                                [--d-model 256] [--n-heads 8] [--n-layers 4]
                                [--context-len 256] [--epochs 20] [--batch-size 64]
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


def train(lr=1e-3, seed=42, tokenizer="char", bpe_vocab_size=512,
          dataset="shakespeare", d_model=128, n_heads=4, n_layers=2,
          context_len=128, batch_size=64, epochs=20, warmup_steps=200,
          weight_decay=0.1):
    data = prepare_data(context_len=context_len, tokenizer=tokenizer,
                        bpe_vocab_size=bpe_vocab_size, dataset=dataset)
    vocab_size = data["vocab_size"]

    key = jax.random.key(seed)
    key, init_key = jax.random.split(key)
    params, config = init_transformer(init_key, vocab_size, d_model=d_model,
                                       n_heads=n_heads, n_layers=n_layers,
                                       context_len=context_len)

    # Keep training data on CPU (numpy) — move batches to GPU on the fly.
    # This avoids OOM when training on large datasets (e.g., full TinyStories).
    train_x = np.array(data["train_x"])
    train_y = np.array(data["train_y"])
    val_x = jnp.array(data["val_x"][:batch_size])
    val_y = jnp.array(data["val_y"][:batch_size])
    n_batches = len(train_x) // batch_size
    total_steps = n_batches * epochs

    # LR schedule: linear warmup + cosine decay
    schedule = optax.join_schedules(
        schedules=[
            optax.linear_schedule(0.0, lr, warmup_steps),
            optax.cosine_decay_schedule(lr, total_steps - warmup_steps, alpha=lr * 0.01),
        ],
        boundaries=[warmup_steps],
    )
    optimizer = optax.adamw(schedule, weight_decay=weight_decay)
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

    print(f"=== Backprop+AdamW LR={lr} wd={weight_decay} warmup={warmup_steps} ===")
    print(f"Model: d={d_model} h={n_heads} l={n_layers} ctx={context_len}")
    print(f"Data: {dataset} tokenizer={tokenizer} vocab={vocab_size}")
    print(f"Params: {count_params(params):,}  steps/epoch={n_batches}  total_steps={total_steps}")

    t_start = time.perf_counter()
    for epoch in range(epochs):
        rng = np.random.default_rng(seed + epoch)
        perm = rng.permutation(len(train_x))
        sx, sy = train_x[perm], train_y[perm]
        eloss = 0.0
        t_epoch = time.perf_counter()
        for bi in range(n_batches):
            s = bi * batch_size
            bx = jnp.array(sx[s:s+batch_size])
            by = jnp.array(sy[s:s+batch_size])
            params, opt_state, loss = train_step(params, opt_state, bx, by)
            eloss += float(loss)
            if bi > 0 and bi % 1000 == 0:
                avg = eloss / bi
                elapsed = time.perf_counter() - t_epoch
                steps_per_sec = bi / elapsed
                eta_min = (n_batches - bi) / steps_per_sec / 60
                print(f"    step {bi}/{n_batches}  loss={avg:.4f}  {steps_per_sec:.1f} steps/s  eta={eta_min:.0f}min")
        eloss /= n_batches
        vl = eval_loss(params, val_x, val_y)
        print(f"  Epoch {epoch+1}/{epochs}  train={eloss:.4f}  val={float(vl):.4f}  ppl={float(jnp.exp(vl)):.2f}")

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
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", type=str, default="trained_bpe",
                        choices=["char", "bpe", "trained_bpe"])
    parser.add_argument("--bpe-vocab", type=int, default=4096)
    parser.add_argument("--dataset", type=str, default="tinystories",
                        choices=["shakespeare", "tinystories"])
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--context-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    args = parser.parse_args()
    train(lr=args.lr, seed=args.seed, tokenizer=args.tokenizer,
          bpe_vocab_size=args.bpe_vocab, dataset=args.dataset,
          d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
          context_len=args.context_len, batch_size=args.batch_size,
          epochs=args.epochs, warmup_steps=args.warmup_steps,
          weight_decay=args.weight_decay)
