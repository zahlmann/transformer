"""
Python wrapper for the CUDA fused transformer kernel.

Registers the compiled .so as a JAX custom call target and provides
a Python API matching the Triton version (fused_transformer_ce_both).

Usage:
    from kernels.cuda.wrapper import fused_transformer_ce_both_cuda

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
from jax._src import core
from jax._src.interpreters import mlir as mlir_impl
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import stablehlo
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
    fn_ptr = ctypes.cast(
        _lib.fused_transformer_ce_cuda,
        ctypes.c_void_p
    ).value
    # Create PyCapsule wrapping the function pointer (required by JAX 0.9+)
    PyCapsule_New = ctypes.pythonapi.PyCapsule_New
    PyCapsule_New.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    PyCapsule_New.restype = ctypes.py_object
    capsule = PyCapsule_New(fn_ptr, b"xla._CUSTOM_CALL_TARGET", None)
    xla_client.register_custom_call_target(
        b"fused_transformer_ce_cuda",
        capsule,
        platform="gpu",
        api_version=0,
    )
    _registered = True


# ─── JAX Primitive ───
_fused_ce_p = core.Primitive("fused_transformer_ce_cuda")
_fused_ce_p.multiple_results = True


def _abstract_eval(
    token_emb, pos_emb, ln1_scale, ln1_bias,
    wq, wk, wv, wo,
    ln2_scale, ln2_bias,
    ffn_up, ffn_up_bias, ffn_down, ffn_down_bias,
    ln_final_scale, ln_final_bias, output_proj,
    vecs, x, y,
    *, half_pop, batch, sigma, alpha, temperature,
):
    return (
        core.ShapedArray((half_pop, batch), jnp.float32),
        core.ShapedArray((half_pop, batch), jnp.float32),
    )

_fused_ce_p.def_abstract_eval(_abstract_eval)


def _jax_dtype_to_ir_type(dtype):
    """Convert JAX dtype to MLIR IR type."""
    if dtype == jnp.bfloat16:
        return ir.BF16Type.get()
    elif dtype == jnp.float32:
        return ir.F32Type.get()
    elif dtype == jnp.int32:
        return ir.IntegerType.get_signless(32)
    raise ValueError(f"Unsupported dtype: {dtype}")


def _lowering(ctx,
    token_emb, pos_emb, ln1_scale, ln1_bias,
    wq, wk, wv, wo,
    ln2_scale, ln2_bias,
    ffn_up, ffn_up_bias, ffn_down, ffn_down_bias,
    ln_final_scale, ln_final_bias, output_proj,
    vecs, x, y,
    *, half_pop, batch, sigma, alpha, temperature,
):
    """MLIR lowering: emit stablehlo.custom_call for the CUDA kernel."""
    # Pack scalars into opaque struct: {half_pop, batch, sigma, alpha, temperature}
    opaque = ir.StringAttr.get(struct.pack("iifff", half_pop, batch, sigma, alpha, temperature))

    f32 = ir.F32Type.get()
    out_types = [
        ir.RankedTensorType.get([half_pop, batch], f32),
        ir.RankedTensorType.get([half_pop, batch], f32),
    ]

    # Only tensor operands (no scalars in the buffer list)
    operands = [
        token_emb, pos_emb, ln1_scale, ln1_bias,
        wq, wk, wv, wo,
        ln2_scale, ln2_bias,
        ffn_up, ffn_up_bias, ffn_down, ffn_down_bias,
        ln_final_scale, ln_final_bias, output_proj,
        vecs, x, y,
    ]

    result = stablehlo.custom_call(
        result=out_types,
        inputs=operands,
        call_target_name="fused_transformer_ce_cuda",
        backend_config=opaque,
        api_version=ir.IntegerAttr.get(ir.IntegerType.get_signless(32), 2),
    )

    return list(result)

mlir_impl.register_lowering(_fused_ce_p, _lowering, platform="gpu")


# ─── Public API ───
def fused_transformer_ce_both_cuda(
    token_emb, pos_emb, ln1_scale, ln1_bias,
    wq, wk, wv, wo,
    ln2_scale, ln2_bias,
    ffn_up, ffn_up_bias, ffn_down, ffn_down_bias,
    ln_final_scale, ln_final_bias, output_proj,
    vecs, x, y, sigma, alpha, temperature,
):
    """Launch the CUDA kernel. Same interface as the Triton version."""
    _ensure_registered()

    VOCAB_PAD = 128
    output_proj_padded = jnp.pad(
        output_proj, [(0, 0), (0, VOCAB_PAD - output_proj.shape[1])]
    )

    HALF_POP = vecs.shape[0]
    BATCH = x.shape[0]

    bf = jnp.bfloat16
    return _fused_ce_p.bind(
        token_emb.astype(bf), pos_emb.astype(bf),
        ln1_scale.astype(bf), ln1_bias.astype(bf),
        wq.astype(bf), wk.astype(bf), wv.astype(bf), wo.astype(bf),
        ln2_scale.astype(bf), ln2_bias.astype(bf),
        ffn_up.astype(bf), ffn_up_bias.astype(bf),
        ffn_down.astype(bf), ffn_down_bias.astype(bf),
        ln_final_scale.astype(bf), ln_final_bias.astype(bf),
        output_proj_padded.astype(bf),
        vecs,
        x.astype(jnp.int32), y.astype(jnp.int32),
        half_pop=HALF_POP, batch=BATCH,
        sigma=float(sigma), alpha=float(alpha), temperature=float(temperature),
    )
