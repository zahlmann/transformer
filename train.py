"""Train a small decoder-only transformer.

Usage: uv run train.py --d-model 768 --n-heads 24 --n-kv-heads 6 --n-layers 12 \
                       --context-len 512 --epochs 3 --batch-size 16
"""

import os
_jax_cache = os.path.join(os.path.dirname(__file__), ".jax_cache")
os.makedirs(_jax_cache, exist_ok=True)
os.environ["JAX_COMPILATION_CACHE_DIR"] = _jax_cache
os.environ.setdefault("JAX_COMPILATION_CACHE_MAX_SIZE", str(2 * 1024**3))

import argparse
import pickle
import tempfile
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

from data import load_data
from model import init_transformer, transformer_loss_fused, count_params


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
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (checkpoint.pkl or weights.pkl)")
    parser.add_argument("--checkpoint-interval", type=int, default=2000,
                        help="Save checkpoint every N steps (0=disable, default 2000)")
    parser.add_argument("--curriculum", action="store_true",
                        help="Sequence length curriculum: start short (128), grow to full ctx")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Disable gradient checkpointing (faster but uses more VRAM)")
    args = parser.parse_args()

    # data — v2 streaming only (combined_v2, trained_bpe 32K)
    data = load_data(context_len=args.context_len)
    vocab_size = data["vocab_size"]
    assert data.get("streaming", False), "expected v2 streaming dataset"

    train_tokens = data["train_tokens"]  # memmap, not in RAM
    n_train_seqs = (len(train_tokens) - 1) // args.context_len
    n_batches = n_train_seqs // args.batch_size

    val_x = jnp.array(data["val_x"][:args.batch_size])
    val_y = jnp.array(data["val_y"][:args.batch_size])
    total_steps = n_batches * args.epochs
    assert args.warmup_steps < total_steps, \
        f"warmup {args.warmup_steps} >= total steps {total_steps}"

    # model
    resume_step = 0
    resume_epoch = 0
    resume_bi = 0
    resumed_opt_state = None
    if args.resume:
        print(f"Resuming from {args.resume}")
        with open(args.resume, "rb") as f:
            ckpt = pickle.load(f)
        params = jax.tree.map(jnp.array, ckpt["params"])
        config = ckpt["config"]
        assert config["d_model"] == args.d_model, \
            f"checkpoint d_model={config['d_model']} != --d-model={args.d_model}"
        if "opt_state" in ckpt:
            resumed_opt_state = jax.tree.map(jnp.array, ckpt["opt_state"])
            resume_step = ckpt["global_step"]
            resume_epoch = ckpt["epoch"]
            resume_bi = ckpt["batch_index"]
            print(f"  full checkpoint: step {resume_step}, epoch {resume_epoch}, batch {resume_bi}")
        else:
            print("  weights-only checkpoint (optimizer state will be reinitialized)")
    else:
        key, init_key = jax.random.split(jax.random.key(args.seed))
        params, config = init_transformer(
            init_key, vocab_size, d_model=args.d_model, n_heads=args.n_heads,
            n_layers=args.n_layers, context_len=args.context_len,
            n_kv_heads=args.n_kv_heads)

    # optimizer: linear warmup + cosine decay
    schedule = optax.join_schedules(
        [optax.linear_schedule(0.0, args.lr, args.warmup_steps),
         optax.cosine_decay_schedule(args.lr, total_steps - args.warmup_steps, alpha=args.lr * 0.01)],
        boundaries=[args.warmup_steps])
    optimizer = optax.adamw(schedule, weight_decay=args.weight_decay)
    opt_state = resumed_opt_state if resumed_opt_state is not None else optimizer.init(params)

    if args.no_checkpoint:
        config["gradient_checkpoint"] = False

    # fused CE always used (vocab=32K >> 8192 threshold)
    assert config["vocab_size"] >= 8192
    ce_chunk = min(4096, config["vocab_size"])

    def make_train_step(phase_config):
        """Create JIT-compiled train step for a specific context length."""
        pc = {**config, "context_len": phase_config["ctx"]}
        @jax.jit
        def step(params, opt_state, x, y):
            def loss_fn(params):
                params_bf16 = jax.tree.map(lambda w: w.astype(jnp.bfloat16), params)
                return transformer_loss_fused(params_bf16, pc, x, y, ce_chunk)
            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state, loss
        return step

    @jax.jit
    def eval_loss(params, x, y):
        params_bf16 = jax.tree.map(lambda w: w.astype(jnp.bfloat16), params)
        return transformer_loss_fused(params_bf16, config, x, y, ce_chunk)

    # curriculum schedule: phases with (fraction_of_training, ctx, bs_multiplier)
    if args.curriculum and args.context_len >= 256:
        phases = [
            {"frac": 0.10, "ctx": args.context_len // 4, "bs_mult": 4},
            {"frac": 0.20, "ctx": args.context_len // 2, "bs_mult": 2},
            {"frac": 0.70, "ctx": args.context_len,      "bs_mult": 1},
        ]
        print(f"Curriculum: {' -> '.join(f'ctx={p['ctx']}(bs×{p['bs_mult']})' for p in phases)}")
    else:
        phases = [{"frac": 1.0, "ctx": args.context_len, "bs_mult": 1}]

    # pre-compile train steps for each unique context length
    phase_steps = {}
    for p in phases:
        if p["ctx"] not in phase_steps:
            phase_steps[p["ctx"]] = make_train_step(p)

    print(f"d={args.d_model} h={args.n_heads} kv_h={args.n_kv_heads} l={args.n_layers} "
          f"ctx={args.context_len} bs={args.batch_size}")
    print(f"combined_v2 trained_bpe vocab={vocab_size}")
    print(f"{count_params(params):,} params, {n_batches} steps/epoch, {total_steps} total")

    def _get_batch_streaming(seq_indices, ctx, bs, offset):
        """Create (x, y) batch from raw token stream on the fly."""
        indices = seq_indices[offset:offset + bs]
        batch = np.stack([train_tokens[i * args.context_len:i * args.context_len + ctx + 1]
                          for i in indices])
        return batch[:, :ctx], batch[:, 1:ctx + 1]

    ckpt_dir = os.path.dirname(__file__)
    ckpt_path = os.path.join(ckpt_dir, "checkpoint.pkl")

    def save_checkpoint(params, opt_state, config, global_step, epoch, batch_index):
        """Save full checkpoint atomically (write tmp, rename)."""
        ckpt_data = {
            "params": jax.tree.map(np.asarray, params),
            "opt_state": jax.tree.map(np.asarray, opt_state),
            "config": config,
            "global_step": global_step,
            "epoch": epoch,
            "batch_index": batch_index,
        }
        fd, tmp = tempfile.mkstemp(dir=ckpt_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                pickle.dump(ckpt_data, f)
            os.replace(tmp, ckpt_path)
        except BaseException:
            os.unlink(tmp)
            raise
        print(f"    checkpoint saved: step {global_step}, epoch {epoch+1}, batch {batch_index}")

    # train with prefetch: overlap host->device transfer with compute
    t_start = time.perf_counter()
    global_step = resume_step
    for epoch in range(args.epochs):
        if epoch < resume_epoch:
            continue
        rng = np.random.default_rng(args.seed + epoch)
        seq_perm = rng.permutation(n_train_seqs)
        eloss = 0.0
        steps_this_epoch = 0
        t_epoch = time.perf_counter()

        bi = resume_bi if epoch == resume_epoch else 0
        if bi > 0:
            print(f"  resuming epoch {epoch+1} from batch {bi}/{n_batches}")
        while bi < n_batches:
            # determine current phase
            progress = global_step / total_steps
            cur_phase = phases[-1]
            cumfrac = 0.0
            for p in phases:
                cumfrac += p["frac"]
                if progress < cumfrac:
                    cur_phase = p
                    break

            ctx = cur_phase["ctx"]
            bs = args.batch_size * cur_phase["bs_mult"]
            train_step = phase_steps[ctx]

            next_frac = 0.0
            for p in phases:
                next_frac += p["frac"]
                if p is cur_phase:
                    break
            steps_until_phase_end = max(1, int(next_frac * total_steps) - global_step)
            steps_in_epoch = n_batches - bi
            chunk_steps = min(steps_until_phase_end, steps_in_epoch)

            # prefetch first batch
            s = bi * args.batch_size
            bx_np, by_np = _get_batch_streaming(seq_perm, ctx, bs, s)
            next_bx = jax.device_put(jnp.array(bx_np))
            next_by = jax.device_put(jnp.array(by_np))

            for ci in range(chunk_steps):
                bx, by = next_bx, next_by
                # prefetch next batch
                if ci + 1 < chunk_steps:
                    s = (bi + ci + 1) * args.batch_size
                    if s + bs <= n_train_seqs:
                        bx_np, by_np = _get_batch_streaming(seq_perm, ctx, bs, s)
                        next_bx = jax.device_put(jnp.array(bx_np))
                        next_by = jax.device_put(jnp.array(by_np))

                params, opt_state, loss = train_step(params, opt_state, bx, by)
                eloss += float(loss)
                global_step += 1
                steps_this_epoch += 1

                step_in_epoch = bi + ci
                if steps_this_epoch > 0 and step_in_epoch % 1000 == 0:
                    avg = eloss / steps_this_epoch
                    sps = steps_this_epoch / (time.perf_counter() - t_epoch)
                    eta = (n_batches - step_in_epoch) / sps / 60
                    phase_info = f"ctx={ctx}" if len(phases) > 1 else ""
                    print(f"    step {step_in_epoch}/{n_batches}  loss={avg:.4f}  "
                          f"{sps:.1f} steps/s  eta={eta:.0f}min  {phase_info}")

                if args.checkpoint_interval > 0 and global_step > 0 and global_step % args.checkpoint_interval == 0:
                    save_checkpoint(params, opt_state, config, global_step, epoch, bi + ci + 1)

            bi += chunk_steps

        vl = float(eval_loss(params, val_x, val_y))
        avg_train = eloss / steps_this_epoch if steps_this_epoch > 0 else 0
        print(f"  epoch {epoch+1}/{args.epochs}  train={avg_train:.4f}  "
              f"val={vl:.4f}  ppl={np.exp(vl):.2f}")
        save_checkpoint(params, opt_state, config, global_step, epoch + 1, 0)

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
