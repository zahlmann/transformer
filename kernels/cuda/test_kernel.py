"""
Test the CUDA kernel against the Triton kernel for correctness.

Usage: uv run kernels/cuda/test_kernel.py

This tests that the CUDA kernel produces the same CE values as the
Triton kernel (within numerical tolerance). Run this after any
changes to the CUDA kernel.
"""
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import jax
import jax.numpy as jnp
import numpy as np

from data import prepare_data
from model import init_transformer
from train_eggroll_optimized import build_param_spec
from kernels.fused_transformer_ce import fused_transformer_ce_both as triton_kernel

HALF_POP = 64  # small pop for testing
BATCH_SIZE = 128
SIGMA = 0.02
ALPHA = 0.50
TEMPERATURE = 2.0


def test_correctness():
    data = prepare_data(context_len=128)
    vocab_size = data["vocab_size"]
    key = jax.random.key(42)
    key, init_key = jax.random.split(key)
    params, config = init_transformer(
        init_key, vocab_size, d_model=64, n_heads=2,
        n_layers=1, context_len=128,
    )
    spec, total_vec_dim = build_param_spec(params)

    x = jnp.array(data["train_x"][:BATCH_SIZE])
    y = jnp.array(data["train_y"][:BATCH_SIZE])
    key, k = jax.random.split(key)
    vecs = jax.random.normal(k, (HALF_POP, total_vec_dim))

    # Run Triton kernel (ground truth)
    ce_pos_triton, ce_neg_triton = triton_kernel(
        params["token_emb"], params["pos_emb"],
        params["layer0.ln1.scale"], params["layer0.ln1.bias"],
        params["layer0.attn.q"], params["layer0.attn.k"],
        params["layer0.attn.v"], params["layer0.attn.o"],
        params["layer0.ln2.scale"], params["layer0.ln2.bias"],
        params["layer0.ffn.up"], params["layer0.ffn.up_bias"],
        params["layer0.ffn.down"], params["layer0.ffn.down_bias"],
        params["ln_final.scale"], params["ln_final.bias"],
        params["output_proj"],
        vecs, x, y, SIGMA, ALPHA, TEMPERATURE,
    )

    print(f"Triton kernel:")
    print(f"  ce_pos shape: {ce_pos_triton.shape}, mean: {float(ce_pos_triton.mean()):.4f}")
    print(f"  ce_neg shape: {ce_neg_triton.shape}, mean: {float(ce_neg_triton.mean()):.4f}")

    # Run CUDA kernel (must be jit-compiled; custom call has no eager eval rule)
    try:
        from kernels.cuda.wrapper import fused_transformer_ce_both_cuda as cuda_kernel

        @jax.jit
        def run_cuda(te, pe, l1s, l1b, wq, wk, wv, wo, l2s, l2b,
                     fu, fub, fd, fdb, lfs, lfb, op, vecs, x, y):
            return cuda_kernel(te, pe, l1s, l1b, wq, wk, wv, wo,
                             l2s, l2b, fu, fub, fd, fdb, lfs, lfb, op,
                             vecs, x, y, SIGMA, ALPHA, TEMPERATURE)

        ce_pos_cuda, ce_neg_cuda = run_cuda(
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
        )

        print(f"\nCUDA kernel:")
        print(f"  ce_pos shape: {ce_pos_cuda.shape}, mean: {float(ce_pos_cuda.mean()):.4f}")
        print(f"  ce_neg shape: {ce_neg_cuda.shape}, mean: {float(ce_neg_cuda.mean()):.4f}")

        # Compare
        max_diff_pos = float(jnp.max(jnp.abs(ce_pos_triton - ce_pos_cuda)))
        max_diff_neg = float(jnp.max(jnp.abs(ce_neg_triton - ce_neg_cuda)))
        print(f"\nMax absolute difference:")
        print(f"  ce_pos: {max_diff_pos:.6f}")
        print(f"  ce_neg: {max_diff_neg:.6f}")

        tol = 0.01  # allow some numerical difference (bf16 rounding)
        if max_diff_pos < tol and max_diff_neg < tol:
            print(f"\nPASS: CUDA kernel matches Triton within tolerance ({tol})")
        else:
            print(f"\nFAIL: difference exceeds tolerance ({tol})")

    except (ImportError, RuntimeError, NotImplementedError) as e:
        print(f"\nCUDA kernel not available: {e}")
        print("Build it first: make -C kernels/cuda/")
        print("Then implement the kernel body in fused_transformer_ce.cu")


if __name__ == "__main__":
    test_correctness()
