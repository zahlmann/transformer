"""Validate the Triton kernel against the JAX forward pass."""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import jax
import jax.numpy as jnp
import numpy as np

from data import prepare_data
from model import init_transformer, count_params
from train_eggroll_optimized import build_param_spec, make_perturbed_forward

D_MODEL = 64
N_HEADS = 2
N_LAYERS = 1
CONTEXT_LEN = 128
BATCH_SIZE = 128
HALF_POP = 16  # small for validation
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
    print(f"Vec dim: {total_vec_dim}")

    train_x = jnp.array(data["train_x"][:BATCH_SIZE])
    train_y = jnp.array(data["train_y"][:BATCH_SIZE])

    # Generate random perturbation vectors
    key, vec_key = jax.random.split(key)
    vecs = jax.random.normal(vec_key, (HALF_POP, total_vec_dim))

    # === JAX reference: compute CE for each perturbation ===
    forward_fn = make_perturbed_forward(params, config, spec)

    print("Computing JAX reference...")
    jax_ce_pos = []
    jax_ce_neg = []
    for i in range(HALF_POP):
        ce_p = forward_fn(params, vecs[i], SIGMA, train_x, train_y, ALPHA)
        ce_n = forward_fn(params, vecs[i], -SIGMA, train_x, train_y, ALPHA)
        jax_ce_pos.append(float(ce_p))
        jax_ce_neg.append(float(ce_n))
    jax_ce_pos = np.array(jax_ce_pos)
    jax_ce_neg = np.array(jax_ce_neg)
    print(f"  JAX CE pos: mean={jax_ce_pos.mean():.6f}, range=[{jax_ce_pos.min():.6f}, {jax_ce_pos.max():.6f}]")
    print(f"  JAX CE neg: mean={jax_ce_neg.mean():.6f}")

    # === Triton kernel ===
    print("Computing Triton kernel...")
    from kernels.fused_transformer_ce import fused_transformer_ce_both

    # Extract weight matrices in the order the kernel expects
    partial_ce_pos, partial_ce_neg = fused_transformer_ce_both(
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

    # Reduce: mean over batch sequences
    triton_ce_pos = (partial_ce_pos.sum(axis=1) / BATCH_SIZE).block_until_ready()
    triton_ce_neg = (partial_ce_neg.sum(axis=1) / BATCH_SIZE).block_until_ready()
    triton_ce_pos_np = np.array(triton_ce_pos)
    triton_ce_neg_np = np.array(triton_ce_neg)

    print(f"  Triton CE pos: mean={triton_ce_pos_np.mean():.6f}, range=[{triton_ce_pos_np.min():.6f}, {triton_ce_pos_np.max():.6f}]")
    print(f"  Triton CE neg: mean={triton_ce_neg_np.mean():.6f}")

    # === Compare ===
    pos_diff = np.abs(jax_ce_pos - triton_ce_pos_np)
    neg_diff = np.abs(jax_ce_neg - triton_ce_neg_np)
    pos_rel = pos_diff / (np.abs(jax_ce_pos) + 1e-8)
    neg_rel = neg_diff / (np.abs(jax_ce_neg) + 1e-8)

    print(f"\n=== VALIDATION ===")
    print(f"Pos CE absolute diff: max={pos_diff.max():.6f}, mean={pos_diff.mean():.6f}")
    print(f"Pos CE relative diff: max={pos_rel.max():.6f}, mean={pos_rel.mean():.6f}")
    print(f"Neg CE absolute diff: max={neg_diff.max():.6f}, mean={neg_diff.mean():.6f}")
    print(f"Neg CE relative diff: max={neg_rel.max():.6f}, mean={neg_rel.mean():.6f}")

    # For ES, what matters is the fitness DIFFERENCE (pos - neg)
    jax_diffs = jax_ce_pos - jax_ce_neg
    triton_diffs = triton_ce_pos_np - triton_ce_neg_np
    diff_err = np.abs(jax_diffs - triton_diffs)
    diff_rel = diff_err / (np.abs(jax_diffs) + 1e-8)
    print(f"\nFitness diff error: max_abs={diff_err.max():.6f}, max_rel={diff_rel.max():.6f}")
    print(f"Fitness diff correlation: {np.corrcoef(jax_diffs, triton_diffs)[0,1]:.8f}")

    # Check if differences are within bf16 tolerance
    tol = 0.05  # 5% relative error is acceptable for bf16 + different GELU
    if pos_rel.max() < tol and neg_rel.max() < tol:
        print(f"\n✓ PASSED: max relative error {max(pos_rel.max(), neg_rel.max()):.4f} < {tol}")
    else:
        print(f"\n✗ FAILED: max relative error {max(pos_rel.max(), neg_rel.max()):.4f} >= {tol}")
        # Print per-perturbation comparison for debugging
        for i in range(min(5, HALF_POP)):
            print(f"  [{i}] jax_pos={jax_ce_pos[i]:.6f} tri_pos={triton_ce_pos_np[i]:.6f} "
                  f"jax_neg={jax_ce_neg[i]:.6f} tri_neg={triton_ce_neg_np[i]:.6f}")


if __name__ == "__main__":
    main()
