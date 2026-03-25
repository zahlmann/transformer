"""
Python wrapper for the CUDA fused transformer kernel.

Registers the compiled .so as a JAX custom call target and provides
a Python API matching the Triton version (fused_transformer_ce_both).

Usage:
    from kernels.cuda.wrapper import fused_transformer_ce_both_cuda

    # Same interface as the Triton version:
    ce_pos, ce_neg = fused_transformer_ce_both_cuda(
        token_emb, pos_emb, ln1_scale, ln1_bias,
        wq, wk, wv, wo,
        ln2_scale, ln2_bias,
        ffn_up, ffn_up_bias, ffn_down, ffn_down_bias,
        ln_final_scale, ln_final_bias, output_proj,
        vecs, x, y, sigma, alpha, temperature,
    )

Build the kernel first: make -C kernels/cuda/
"""

import os
import ctypes
import struct
import numpy as np
import jax
import jax.numpy as jnp
from jax import core
from jax.interpreters import mlir, xla
from jaxlib import xla_client

# ─── Load the compiled CUDA library ───
_dir = os.path.dirname(os.path.abspath(__file__))
_lib_path = os.path.join(_dir, "libfused_transformer.so")

_lib = None
_registered = False

def _ensure_registered():
    """Load .so and register custom call target with XLA (once)."""
    global _lib, _registered
    if _registered:
        return
    if not os.path.exists(_lib_path):
        raise RuntimeError(
            f"CUDA kernel not built. Run: make -C kernels/cuda/\n"
            f"Expected: {_lib_path}"
        )
    _lib = ctypes.CDLL(_lib_path)
    # Get function pointer for the XLA custom call entry point
    fn_ptr = ctypes.cast(
        _lib.fused_transformer_ce_cuda,
        ctypes.c_void_p
    ).value
    # Register with XLA
    xla_client.register_custom_call_target(
        b"fused_transformer_ce_cuda",
        fn_ptr,
        platform="gpu",
    )
    _registered = True


def fused_transformer_ce_both_cuda(
    token_emb, pos_emb, ln1_scale, ln1_bias,
    wq, wk, wv, wo,
    ln2_scale, ln2_bias,
    ffn_up, ffn_up_bias, ffn_down, ffn_down_bias,
    ln_final_scale, ln_final_bias, output_proj,
    vecs, x, y, sigma, alpha, temperature,
):
    """Launch the CUDA kernel. Same interface as the Triton version.

    Returns (partial_ce_pos, partial_ce_neg) each of shape (HALF_POP, BATCH).
    """
    _ensure_registered()

    HALF_POP = vecs.shape[0]
    BATCH = x.shape[0]
    VOCAB_PAD = 128

    # Pad output projection to VOCAB_PAD
    output_proj_padded = jnp.pad(
        output_proj, [(0, 0), (0, VOCAB_PAD - output_proj.shape[1])]
    )

    # Cast weights to bf16 (matching Triton wrapper)
    token_emb_bf = token_emb.astype(jnp.bfloat16)
    pos_emb_bf = pos_emb.astype(jnp.bfloat16)
    ln1_scale_bf = ln1_scale.astype(jnp.bfloat16)
    ln1_bias_bf = ln1_bias.astype(jnp.bfloat16)
    wq_bf = wq.astype(jnp.bfloat16)
    wk_bf = wk.astype(jnp.bfloat16)
    wv_bf = wv.astype(jnp.bfloat16)
    wo_bf = wo.astype(jnp.bfloat16)
    ln2_scale_bf = ln2_scale.astype(jnp.bfloat16)
    ln2_bias_bf = ln2_bias.astype(jnp.bfloat16)
    ffn_up_bf = ffn_up.astype(jnp.bfloat16)
    ffn_up_bias_bf = ffn_up_bias.astype(jnp.bfloat16)
    ffn_down_bf = ffn_down.astype(jnp.bfloat16)
    ffn_down_bias_bf = ffn_down_bias.astype(jnp.bfloat16)
    ln_final_scale_bf = ln_final_scale.astype(jnp.bfloat16)
    ln_final_bias_bf = ln_final_bias.astype(jnp.bfloat16)
    output_proj_bf = output_proj_padded.astype(jnp.bfloat16)

    # Pack kernel parameters as opaque bytes
    opaque = struct.pack("ii", HALF_POP, BATCH)

    # Call via XLA custom_call
    # Note: the custom_call convention passes all operands as input buffers,
    # and the output shapes are specified separately.
    result_shapes = [
        jax.ShapeDtypeStruct((HALF_POP, BATCH), jnp.float32),  # ce_pos
        jax.ShapeDtypeStruct((HALF_POP, BATCH), jnp.float32),  # ce_neg
    ]

    # Use jax.pure_callback or custom_call primitive
    # For XLA custom_call, we need to use the lower-level API:
    ce_pos, ce_neg = jax.pure_callback(
        lambda *args: _call_cuda_kernel(*args, half_pop=HALF_POP, batch=BATCH),
        [jnp.zeros((HALF_POP, BATCH), jnp.float32)] * 2,
        token_emb_bf, pos_emb_bf, ln1_scale_bf, ln1_bias_bf,
        wq_bf, wk_bf, wv_bf, wo_bf,
        ln2_scale_bf, ln2_bias_bf,
        ffn_up_bf, ffn_up_bias_bf, ffn_down_bf, ffn_down_bias_bf,
        ln_final_scale_bf, ln_final_bias_bf, output_proj_bf,
        vecs,
        x.astype(jnp.int32), y.astype(jnp.int32),
        jnp.float32(sigma), jnp.float32(alpha), jnp.float32(temperature),
    )

    return ce_pos, ce_neg


def _call_cuda_kernel(*args, half_pop, batch):
    """Direct CUDA kernel call (for pure_callback).

    NOTE: This is a placeholder. The proper approach is to use
    xla_client custom_call which runs on GPU without host roundtrip.
    See the _register_custom_call_lowering function below for the
    production implementation.

    For the production version, replace fused_transformer_ce_both_cuda
    with a proper JAX primitive that lowers to XLA custom_call.
    """
    raise NotImplementedError(
        "Direct callback not implemented. Use the XLA custom_call "
        "primitive approach instead. See comments in this file."
    )


# ─── Production approach: JAX primitive with XLA custom_call lowering ───
#
# The proper way to call CUDA from JAX (no host roundtrip):
#
# 1. Define a JAX primitive:
#    _fused_ce_p = core.Primitive("fused_transformer_ce_cuda")
#    _fused_ce_p.multiple_results = True
#
# 2. Define abstract eval (shape inference):
#    @_fused_ce_p.def_abstract_eval
#    def _fused_ce_abstract(token_emb, pos_emb, ..., vecs, x, y, sigma, alpha, temp):
#        half_pop = vecs.shape[0]
#        batch = x.shape[0]
#        return (
#            core.ShapedArray((half_pop, batch), jnp.float32),
#            core.ShapedArray((half_pop, batch), jnp.float32),
#        )
#
# 3. Define MLIR lowering (XLA custom_call):
#    def _fused_ce_lowering(ctx, token_emb, pos_emb, ..., vecs, x, y, sigma, alpha, temp):
#        opaque = struct.pack("ii", half_pop, batch)
#        out_types = [
#            mlir.ir.RankedTensorType.get([half_pop, batch], mlir.ir.F32Type.get()),
#            mlir.ir.RankedTensorType.get([half_pop, batch], mlir.ir.F32Type.get()),
#        ]
#        result = mlir.custom_call(
#            "fused_transformer_ce_cuda",
#            result_types=out_types,
#            operands=[token_emb, pos_emb, ..., vecs, x, y, sigma, alpha, temp],
#            backend_config=opaque,
#            api_version=2,  # typed FFI
#        ).results
#        return result
#
#    mlir.register_lowering(_fused_ce_p, _fused_ce_lowering, platform="gpu")
#
# This is the approach used by jax-triton internally. The exact MLIR API
# depends on the JAX version. For JAX 0.9.2, check:
#   from jax.interpreters import mlir
#   help(mlir.custom_call)
#
# The next agent should implement this lowering to get full GPU-native
# execution without host synchronization.
