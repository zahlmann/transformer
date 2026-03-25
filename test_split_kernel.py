"""Test split kernel correctness against the original fused kernel."""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import jax
import jax.numpy as jnp
import numpy as np

from data import prepare_data
from model import init_transformer
from train_eggroll_optimized import build_param_spec

D_MODEL = 64
N_HEADS = 2
N_LAYERS = 1
CONTEXT_LEN = 128
BATCH_SIZE = 128
HALF_POP = 256
SIGMA = 0.04
ALPHA = 0.50
TEMPERATURE = 2.0
SEED = 42


def main():
    print("Setting up...")
    data = prepare_data(context_len=CONTEXT_LEN)
    vocab_size = data["vocab_size"]
    key = jax.random.key(SEED)
    key, init_key = jax.random.split(key)
    params, config = init_transformer(
        init_key, vocab_size, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, context_len=CONTEXT_LEN,
    )
    spec, total_vec_dim = build_param_spec(params)
    print(f"Vec dim: {total_vec_dim}, HALF_POP: {HALF_POP}, BATCH: {BATCH_SIZE}")

    train_x = jnp.array(data["train_x"][:BATCH_SIZE])
    train_y = jnp.array(data["train_y"][:BATCH_SIZE])

    key, vec_key = jax.random.split(key)
    vecs = jax.random.normal(vec_key, (HALF_POP, total_vec_dim))

    from kernels.fused_transformer_ce import fused_transformer_ce_both, fused_transformer_ce_both_split

    args = (
        params["token_emb"], params["pos_emb"],
        params["layer0.ln1.scale"], params["layer0.ln1.bias"],
        params["layer0.attn.q"], params["layer0.attn.k"],
        params["layer0.attn.v"], params["layer0.attn.o"],
        params["layer0.ln2.scale"], params["layer0.ln2.bias"],
        params["layer0.ffn.up"], params["layer0.ffn.up_bias"],
        params["layer0.ffn.down"], params["layer0.ffn.down_bias"],
        params["ln_final.scale"], params["ln_final.bias"],
        params["output_proj"],
        vecs, train_x, train_y,
        SIGMA, ALPHA, TEMPERATURE,
    )

    # === Original fused kernel (reference) ===
    print("Running original fused kernel...")
    ce_pos_orig, ce_neg_orig = fused_transformer_ce_both(*args)
    ce_pos_orig = ce_pos_orig.block_until_ready()
    ce_neg_orig = ce_neg_orig.block_until_ready()
    print(f"  Original CE: pos_mean={float(ce_pos_orig.mean()):.6f}, neg_mean={float(ce_neg_orig.mean()):.6f}")

    pos_orig_np = np.array(ce_pos_orig)
    neg_orig_np = np.array(ce_neg_orig)

    # === Test split kernel with various chunk sizes ===
    all_passed = True
    for chunk_size in [16, 32, 64, 128, 256]:
        print(f"\nSplit kernel (chunk_size={chunk_size})...")
        ce_pos_split, ce_neg_split = fused_transformer_ce_both_split(*args, chunk_size=chunk_size)
        ce_pos_split = ce_pos_split.block_until_ready()
        ce_neg_split = ce_neg_split.block_until_ready()

        pos_split_np = np.array(ce_pos_split)
        neg_split_np = np.array(ce_neg_split)

        pos_diff = np.abs(pos_orig_np - pos_split_np)
        neg_diff = np.abs(neg_orig_np - neg_split_np)
        pos_rel = pos_diff / (np.abs(pos_orig_np) + 1e-8)
        neg_rel = neg_diff / (np.abs(neg_orig_np) + 1e-8)

        max_rel = max(pos_rel.max(), neg_rel.max())
        max_abs = max(pos_diff.max(), neg_diff.max())

        orig_diffs = pos_orig_np - neg_orig_np
        split_diffs = pos_split_np - neg_split_np
        diff_err = np.abs(orig_diffs - split_diffs).max()

        passed = max_rel < 1e-4
        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        print(f"  {status}: max_rel={max_rel:.2e}, max_abs={max_abs:.2e}, fitness_diff_err={diff_err:.2e}")

    print(f"\n{'All tests PASSED' if all_passed else 'Some tests FAILED'}")


if __name__ == "__main__":
    main()
