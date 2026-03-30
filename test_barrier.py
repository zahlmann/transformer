"""Minimal test: verify Triton cross-block barrier mechanism works."""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import triton
import triton.language as tl
import jax
import jax.numpy as jnp
import jax_triton as jt


@triton.jit
def _barrier_test(
    # Inputs (positional args in jt.triton_call)
    scratch_ptr,
    barrier_ptr,
    # Output (from out_shape)
    out_ptr,
    N_BLOCKS: tl.constexpr,
):
    """Each block writes its ID to scratch, barrier, then block 0 sums all."""
    pid = tl.program_id(0)

    # Phase 1: each block writes its pid + 1 to scratch
    tl.store(scratch_ptr + pid, (pid + 1).to(tl.float32))

    # Barrier: signal and wait
    tl.atomic_add(barrier_ptr, 1, sem='release', scope='gpu')
    while tl.atomic_add(barrier_ptr, 0, sem='acquire', scope='gpu') < N_BLOCKS:
        pass

    # Phase 2: block 0 sums all values
    if pid == 0:
        total = 0.0
        for i in tl.static_range(N_BLOCKS):
            total += tl.load(scratch_ptr + i)
        tl.store(out_ptr, total)


def test_barrier():
    N = 16
    scratch = jnp.zeros((N,), dtype=jnp.float32)
    barrier = jnp.zeros((1,), dtype=jnp.int32)

    print(f"Testing barrier with {N} blocks...")
    (result,) = jt.triton_call(
        scratch,
        barrier,
        kernel=_barrier_test,
        out_shape=[jax.ShapeDtypeStruct((1,), jnp.float32)],
        grid=(N,),
        num_warps=4, num_stages=1,
        N_BLOCKS=N,
    )

    expected = sum(range(1, N+1))  # 1+2+...+16 = 136
    actual = float(result[0])
    print(f"  Expected: {expected}")
    print(f"  Got:      {actual}")
    print(f"  PASS" if abs(actual - expected) < 0.01 else f"  FAIL")


if __name__ == "__main__":
    test_barrier()
