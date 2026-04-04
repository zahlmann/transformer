"""Train a small decoder-only transformer.

Usage: uv run train.py --d-model 768 --n-heads 24 --n-kv-heads 6 --n-layers 12 \
                       --context-len 512 --epochs 3 --batch-size 16
"""

import os
os.environ["JAX_COMPILATION_CACHE_DIR"] = os.path.join(os.path.dirname(__file__), ".jax_cache")
os.environ.setdefault("JAX_COMPILATION_CACHE_MAX_SIZE", str(2 * 1024**3))

import argparse
import pickle
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

from data import prepare_data
from model import (init_transformer, transformer_forward_batch, cross_entropy_loss,
                   transformer_loss_fused, count_params)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--n-heads", type=int, required=True)
    parser.add_argument("--n-kv-heads", type=int, required=True)
    parser.add_argument("--n-layers", type=int, required=True)
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--dataset", default="tinystories",
                        choices=["shakespeare", "tinystories", "combined", "combined_epoch2"])
    parser.add_argument("--tokenizer", default="trained_bpe", choices=["char", "bpe", "trained_bpe"])
    parser.add_argument("--bpe-vocab", type=int, default=4096)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to weights.pkl checkpoint to resume from")
    parser.add_argument("--d-ff", type=int, default=None,
                        help="FFN hidden dim (default: auto from SwiGLU formula)")
    parser.add_argument("--use-deltanet", action="store_true",
                        help="Enable DeltaNet hybrid layers (default: pure attention)")
    args = parser.parse_args()

    # data
    data = prepare_data(context_len=args.context_len, tokenizer=args.tokenizer,
                        bpe_vocab_size=args.bpe_vocab, dataset=args.dataset)
    vocab_size = data["vocab_size"]
    train_x = np.array(data["train_x"])  # keep on CPU, batch to GPU
    train_y = np.array(data["train_y"])
    val_x = jnp.array(data["val_x"][:args.batch_size])
    val_y = jnp.array(data["val_y"][:args.batch_size])
    n_batches = len(train_x) // args.batch_size
    total_steps = n_batches * args.epochs
    assert args.warmup_steps < total_steps, \
        f"warmup {args.warmup_steps} >= total steps {total_steps}"

    # model
    if args.resume:
        print(f"Resuming from {args.resume}")
        with open(args.resume, "rb") as f:
            ckpt = pickle.load(f)
        params = jax.tree.map(jnp.array, ckpt["params"])
        config = ckpt["config"]
        assert config["d_model"] == args.d_model, \
            f"checkpoint d_model={config['d_model']} != --d-model={args.d_model}"
    else:
        key, init_key = jax.random.split(jax.random.key(args.seed))
        params, config = init_transformer(
            init_key, vocab_size, d_model=args.d_model, n_heads=args.n_heads,
            n_layers=args.n_layers, context_len=args.context_len, n_kv_heads=args.n_kv_heads,
            use_deltanet=args.use_deltanet)
        if args.d_ff is not None:
            # override auto-computed d_ff
            from model import _swiglu_d_ff
            old_d_ff = config["d_ff"]
            config["d_ff"] = args.d_ff
            # reinit FFN weights with new d_ff if different
            if args.d_ff != old_d_ff:
                for layer in range(args.n_layers):
                    p = f"layer{layer}"
                    key, k1, k2, k3 = jax.random.split(key, 4)
                    params[f"{p}.ffn.gate"] = jax.random.normal(k1, (args.d_model, args.d_ff)) * (args.d_model ** -0.5)
                    params[f"{p}.ffn.up"] = jax.random.normal(k2, (args.d_model, args.d_ff)) * (args.d_model ** -0.5)
                    params[f"{p}.ffn.down"] = jax.random.normal(k3, (args.d_ff, args.d_model)) * (args.d_ff ** -0.5)

    # optimizer: linear warmup + cosine decay
    schedule = optax.join_schedules(
        [optax.linear_schedule(0.0, args.lr, args.warmup_steps),
         optax.cosine_decay_schedule(args.lr, total_steps - args.warmup_steps, alpha=args.lr * 0.01)],
        boundaries=[args.warmup_steps])
    optimizer = optax.adamw(schedule, weight_decay=args.weight_decay)
    opt_state = optimizer.init(params)

    # use fused CE for large vocab (avoids materializing full logits tensor)
    use_fused_ce = config["vocab_size"] >= 8192
    ce_chunk = min(4096, config["vocab_size"])

    @jax.jit
    def train_step(params, opt_state, x, y):
        def loss_fn(params):
            params_bf16 = jax.tree.map(lambda w: w.astype(jnp.bfloat16), params)
            if use_fused_ce:
                return transformer_loss_fused(params_bf16, config, x, y, ce_chunk)
            logits = transformer_forward_batch(params_bf16, config, x)
            return cross_entropy_loss(logits.astype(jnp.float32), y)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def eval_loss(params, x, y):
        params_bf16 = jax.tree.map(lambda w: w.astype(jnp.bfloat16), params)
        if use_fused_ce:
            return transformer_loss_fused(params_bf16, config, x, y, ce_chunk)
        logits = transformer_forward_batch(params_bf16, config, x)
        return cross_entropy_loss(logits.astype(jnp.float32), y)

    layer_types = config.get("layer_types", ["attn"] * args.n_layers)
    n_delta = sum(1 for t in layer_types if t == "delta")
    n_attn = sum(1 for t in layer_types if t == "attn")
    print(f"d={args.d_model} h={args.n_heads} kv_h={args.n_kv_heads} l={args.n_layers} "
          f"ctx={args.context_len} bs={args.batch_size}"
          + (f" ({n_delta}D+{n_attn}A)" if n_delta > 0 else ""))
    print(f"{args.dataset} {args.tokenizer} vocab={vocab_size}")
    print(f"{count_params(params):,} params, {n_batches} steps/epoch, {total_steps} total")

    # train with prefetch: overlap host→device transfer with compute
    t_start = time.perf_counter()
    for epoch in range(args.epochs):
        perm = np.random.default_rng(args.seed + epoch).permutation(len(train_x))
        sx, sy = train_x[perm], train_y[perm]
        eloss = 0.0
        t_epoch = time.perf_counter()
        # prefetch first batch
        s = 0
        next_bx = jax.device_put(jnp.array(sx[s:s + args.batch_size]))
        next_by = jax.device_put(jnp.array(sy[s:s + args.batch_size]))
        for bi in range(n_batches):
            bx, by = next_bx, next_by
            # prefetch next batch while current step runs
            if bi + 1 < n_batches:
                s = (bi + 1) * args.batch_size
                next_bx = jax.device_put(jnp.array(sx[s:s + args.batch_size]))
                next_by = jax.device_put(jnp.array(sy[s:s + args.batch_size]))
            params, opt_state, loss = train_step(params, opt_state, bx, by)
            eloss += float(loss)
            if bi > 0 and bi % 1000 == 0:
                avg = eloss / bi
                sps = bi / (time.perf_counter() - t_epoch)
                eta = (n_batches - bi) / sps / 60
                print(f"    step {bi}/{n_batches}  loss={avg:.4f}  {sps:.1f} steps/s  eta={eta:.0f}min")
        vl = float(eval_loss(params, val_x, val_y))
        print(f"  epoch {epoch+1}/{args.epochs}  train={eloss/n_batches:.4f}  "
              f"val={vl:.4f}  ppl={np.exp(vl):.2f}")

    elapsed = time.perf_counter() - t_start
    mem = jax.local_devices()[0].memory_stats()
    assert mem is not None
    print(f"\n{elapsed:.0f}s total, {mem['peak_bytes_in_use']/1e6:.0f}MB peak VRAM")

    # save
    save_path = os.path.join(os.path.dirname(__file__), "weights.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({"params": jax.tree.map(np.asarray, params), "config": config}, f)
    print(f"saved {save_path}")


if __name__ == "__main__":
    main()
