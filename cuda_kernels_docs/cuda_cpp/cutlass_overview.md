# CUTLASS: Fast Linear Algebra in CUDA C++

Source: https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda/

---

## Overview

CUTLASS (CUDA Templates for Linear Algebra Subroutines) is a collection of C++ templates for implementing high-performance General Matrix Multiplication (GEMM) operations on NVIDIA GPUs. Unlike closed-source libraries like cuBLAS, CUTLASS decomposes GEMM into fundamental components that programmers can customize within their own kernels.

**Key motivation:** Matrix multiplication is a key computation within many scientific applications, particularly those in deep learning. CUTLASS emerged from NVIDIA's work optimizing cuDNN and cuBLAS, enabling developers to adapt GEMM strategies for diverse applications.

**GitHub:** https://github.com/NVIDIA/cutlass

---

## Core Architecture

### Hierarchical Structure

CUTLASS implements a three-level hierarchy mirroring the CUDA programming model:

1. **Thread Block Tiles**: Each thread block computes portions of output matrix C by iteratively loading blocks from input matrices and accumulating matrix products
2. **Warp Tiles**: Within shared memory, warps load data fragments and compute outer products
3. **Thread Tiles**: Individual threads compute 2D tiled outer products using registers

This hierarchy transfers data progressively from slower to faster memory, enabling high reuse rates:

```
Global Memory -> Shared Memory -> Registers -> Compute
   (slow)          (fast)        (fastest)
```

### The Outer Product Approach

Rather than the traditional "inner product" algorithm (dot product of row and column), CUTLASS uses an **accumulating outer product** strategy:

```cuda
// Outer product loop ordering: K is outermost
for (int k = 0; k < K; ++k)
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            C[i][j] += A[i][k] * B[k][j];
```

This permutation maximizes data locality: for each k step, a column of A and a row of B are loaded once and used M*N times in the outer product.

### Thread Block Organization

```
Thread Block Tile (e.g., 128x32):
+-- Shared Memory: Stores A and B tiles
+-- Multiple Warps: Each handles non-overlapping region
+-- Accumulators: Register-resident result tiles
```

### Software Pipelining

Three concurrent instruction streams hide latency:
1. **Global memory loads** (slowest, ~400 cycles)
2. **Shared memory fragment loads** (medium, ~30 cycles)
3. **Math operations on current data** (fast, ~4 cycles)

Double buffering eliminates synchronization overhead between stages: while computing on one set of shared memory tiles, the next set is being loaded from global memory.

---

## Data Type Support

CUTLASS provides specialized implementations for:

| Data Type | Description | Tensor Core Support |
|-----------|-------------|---------------------|
| FP32      | Single precision | SIMT only |
| FP64      | Double precision | SIMT only |
| FP16      | Half precision | Yes (WMMA) |
| BF16      | Brain float 16 | Yes (SM 8.0+) |
| TF32      | Tensor float 32 | Yes (SM 8.0+) |
| INT8      | 8-bit integer (IDP4A) | Yes |
| INT4      | 4-bit integer | Yes (SM 7.5+) |

---

## WMMA Integration (Tensor Cores)

For Volta V100 and later GPUs, CUTLASS implements the Warp Matrix Multiply-Accumulate API:

```cuda
wmma::load_matrix_sync(frag_a, ptr_a, lda);
wmma::load_matrix_sync(frag_b, ptr_b, ldb);
wmma::mma_sync(acc_frag, frag_a, frag_b, acc_frag);
wmma::store_matrix_sync(ptr_c, acc_frag, ldc, wmma::mem_col_major);
```

Each Tensor Core computes 4x4x4 matrix operations (D = A*B + C), where A and B are FP16 and C/D can be FP16 or FP32. The WMMA API exposes these through 16x16x16 warp-level operations.

---

## Code Examples

### Policy Configuration

Policies define tile sizes at compile time, tuned for target processor and GEMM aspect ratio:

```cuda
template <>
struct gemm_policy<float, float, problem_size_t::Tall> :
    block_task_policy<
        128,      // BlockItemsY  -- rows per thread block
        32,       // BlockItemsX  -- cols per thread block
        8,        // ThreadItemsY -- rows per thread
        4,        // ThreadItemsX -- cols per thread
        8,        // BlockItemsK  -- depth per iteration
        true,     // UseDoubleScratchTiles (double buffering)
        grid_raster_strategy::Default>
{};
```

### Basic GEMM Kernel

```cuda
__global__ void gemm_kernel(
    float *C,
    float const *A,
    float const *B,
    int M, int N, int K)
{
    typedef block_task_policy<
        128,  // BlockItemsY
        32,   // BlockItemsX
        8,    // ThreadItemsY
        4,    // ThreadItemsX
        8,    // BlockItemsK
        true, // UseDoubleScratchTiles
        block_raster_enum::Default
    > block_task_policy_t;

    typedef gemm::blas_scaled_epilogue<float, float, float> epilogue_op_t;

    typedef block_task<
        block_task_policy_t,
        float, float,
        matrix_transform_t::NonTranspose, 4,
        matrix_transform_t::NonTranspose, 4,
        epilogue_op_t, 4, true
    > block_task_t;

    __shared__ block_task_t::scratch_storage_t smem;

    block_task_t(
        reinterpret_cast<block_task_t::scratch_storage_t*>(&smem),
        A, B, C,
        epilogue_op_t(1, 0),
        M, N, K).run();
}
```

### Main Loop Structure (Inner Kernel)

```cuda
__device__ void block_matrix_product(int K_dim)
{
    value_t frag_a[ThreadItemsY];
    value_t frag_b[ThreadItemsX];
    accum_t accumulator[ThreadItemsX][ThreadItemsY];

    for (int kblock = 0; kblock < K_dim; kblock += BlockItemsK) {
        // Load A and B from global to shared memory
        __syncthreads();

        #pragma unroll
        for (int warp_k = 0; warp_k < BlockItemsK; warp_k += WarpItemsK) {
            // Fetch fragments from shared memory into registers

            #pragma unroll
            for (int thread_x = 0; thread_x < ThreadItemsX; ++thread_x) {
                #pragma unroll
                for (int thread_y = 0; thread_y < ThreadItemsY; ++thread_y) {
                    accumulator[thread_x][thread_y] +=
                        frag_a[thread_y] * frag_b[thread_x];
                }
            }
        }
        __syncthreads();
    }
}
```

### Fused Bias + ReLU Epilogue

One of CUTLASS's key features: fusing element-wise operations with GEMM to eliminate extra kernel launches and avoid round-trips through global memory:

```cuda
template <typename accum_t, typename scalar_t, typename output_t>
struct fused_bias_relu_epilogue {
    scalar_t const *Bias;
    accum_t threshold;

    inline __device__ __host__
    fused_bias_relu_epilogue(
        scalar_t const *Bias,
        accum_t threshold
    ): Bias(Bias), threshold(threshold) { }

    inline __device__ __host__
    output_t operator()(
        accum_t accumulator,
        output_t c,
        size_t idx
    ) const {
        accum_t result = output_t(
            accumulator + Bias[idx] + c
        );
        return max(threshold, result);
    }
};
```

This fuses `ReLU(alpha * A * B + beta * C + bias)` into a single kernel.

---

## Performance Results

Benchmarks on Tesla V100 for large matrices (M=10240, N=K=4096):

- CUTLASS achieves **within a few percent** of the performance of hand-tuned assembly kernels in cuBLAS
- Performance is comparable across all data types (FP32, FP64, FP16, INT8)
- Works with both row-major and column-major layouts
- WMMA GEMM performance continues to improve with newer CUTLASS versions

---

## Key Design Principles

1. **Extensive Templating**: All tile sizes and data types are compile-time parameters, enabling aggressive compiler optimization and specialization
2. **Register Reuse**: Small fragments maximize compute intensity -- each loaded value is used multiple times in the thread tile
3. **Shared Memory Double-Buffering**: Eliminates pipeline stalls by overlapping global memory loads with computation
4. **Flexible Epilogues**: Support for fused activation functions (ReLU, sigmoid), custom scaling, bias addition
5. **Layout Agnostic**: Support for row-major, column-major, and interleaved layouts

---

## Practical Applications

Beyond standard GEMM, CUTLASS enables:

- **Fused activation functions** (ReLU, GELU, sigmoid) -- no extra kernel launch
- **Custom scaling operations** -- mixed precision, quantization
- **Deep learning convolutions** via implicit GEMM
- **Mixed-precision** computations (FP16 input, FP32 accumulation)
- **Grouped GEMM** -- batch of different-sized matrix multiplies
- **Sparse matrix operations** (structured sparsity on Ampere+)

---

## CUTLASS 2.x vs 3.x

CUTLASS 3.x (for Hopper/SM 9.0) introduces:

- **CuTe layout algebra** -- unified tensor layout description
- **TMA (Tensor Memory Accelerator)** -- hardware-accelerated async copies
- **Warpgroup MMA** -- larger cooperative matrix operations
- **Persistent kernels** -- stream-K decomposition for better load balancing
- **Cluster-level cooperation** -- using thread block clusters

CUTLASS 2.x remains relevant for Volta/Turing/Ampere (SM 7.0 - 8.9).

---

## Resources

- **GitHub**: https://github.com/NVIDIA/cutlass
- **Documentation**: https://nvidia.github.io/cutlass/
- **Examples**: `examples/` directory in the CUTLASS repository
- **Related Libraries**: CUB (collective primitives), cuBLAS, cuDNN
- **GTC Presentations**: Detailed architectural walkthroughs available from NVIDIA GTC
