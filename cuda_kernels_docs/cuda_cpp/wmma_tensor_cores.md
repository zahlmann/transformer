# WMMA (Warp Matrix Multiply Accumulate) - Tensor Core API

Source: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#wmma

---

## Overview

The Warp Matrix Multiply-Accumulate (WMMA) API enables efficient matrix operations on NVIDIA GPUs through cooperative execution by **all 32 threads in a warp**. This feature leverages Tensor Cores for hardware-accelerated matrix multiplication, common in deep learning and scientific computing.

Requires: Compute Capability 7.0+ (Volta and later).

Header: `#include <mma.h>`

Namespace: `nvcuda::wmma`

---

## Fragment Type

The `fragment` template represents a portion of a matrix distributed across warp threads:

```cpp
template <typename Use, int m, int n, int k, typename T, typename Layout = void>
class fragment;
```

**Template Parameters:**

| Parameter | Description |
|-----------|-------------|
| `Use`     | Matrix role: `matrix_a`, `matrix_b`, or `accumulator` |
| `m, n, k` | Dimensions defining the WMMA operation shape |
| `T`       | Data element type (`half`, `float`, `double`, `__nv_bfloat16`, etc.) |
| `Layout`  | Memory layout: `row_major` or `col_major` (not needed for `accumulator`) |

**Fragment Members:**
- `num_elements` - Number of elements in the fragment owned by this thread
- `x[num_elements]` - Array of elements (can be accessed/modified per-thread)

---

## Core Operations

### fill_fragment

Initializes all elements of a fragment to a scalar value:

```cpp
void fill_fragment(fragment<...> &frag, const T &value);
```

**Usage:**
```cpp
wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
wmma::fill_fragment(acc_frag, 0.0f);  // Zero-initialize accumulator
```

### load_matrix_sync

Loads matrix data from memory into a fragment. All threads in the warp must participate.

```cpp
// For matrix_a and matrix_b (layout from template)
void load_matrix_sync(fragment<...> &frag, const T *ptr, unsigned ldm);

// For accumulator (layout specified at runtime)
void load_matrix_sync(fragment<...> &frag, const T *ptr, unsigned ldm, layout_t layout);
```

**Parameters:**
- `frag` - Destination fragment
- `ptr` - Pointer to matrix data in memory (global or shared)
- `ldm` - Leading dimension of the matrix in memory
- `layout` - `wmma::mem_row_major` or `wmma::mem_col_major`

**Example:**
```cpp
wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
wmma::load_matrix_sync(a_frag, a_ptr, K);  // K = leading dimension (number of columns)
```

### store_matrix_sync

Writes fragment data back to memory. All threads in the warp must participate.

```cpp
void store_matrix_sync(T *ptr, const fragment<...> &frag, unsigned ldm, layout_t layout);
```

**Example:**
```cpp
wmma::store_matrix_sync(c_ptr, acc_frag, N, wmma::mem_row_major);
```

### mma_sync

Performs the core matrix multiply-accumulate operation: **D = A x B + C**

```cpp
void mma_sync(fragment<accumulator, m, n, k, float> &d,
              const fragment<matrix_a, m, n, k, half, ...> &a,
              const fragment<matrix_b, m, n, k, half, ...> &b,
              const fragment<accumulator, m, n, k, float> &c);
```

All threads in the warp must participate. The operation computes a matrix multiply of shape `(m x k) * (k x n) -> (m x n)`, accumulated into the result.

**In-place accumulation** (d and c can be the same fragment):
```cpp
wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
```

---

## Supported Matrix Sizes and Element Types

### Half Precision (FP16) - Compute Capability 7.0+

| m  | n  | k  | matrix_a type | matrix_b type | accumulator type |
|----|----|----|---------------|---------------|------------------|
| 16 | 16 | 16 | `half`        | `half`        | `float` or `half` |
| 32 |  8 | 16 | `half`        | `half`        | `float` or `half` |
|  8 | 32 | 16 | `half`        | `half`        | `float` or `half` |

### Tensor Float 32 (TF32) - Compute Capability 8.0+

| m  | n  | k | matrix_a type     | matrix_b type     | accumulator type |
|----|----|---|-------------------|-------------------|------------------|
| 16 | 16 | 8 | `precision::tf32` | `precision::tf32` | `float`          |

TF32 offers intermediate precision suitable for training: 8-bit exponent (same as FP32), 10-bit mantissa (reduced from FP32's 23-bit), with 19 bits total.

### BFloat16 (BF16) - Compute Capability 8.0+

| m  | n  | k  | matrix_a type    | matrix_b type    | accumulator type |
|----|----|----|------------------|------------------|------------------|
| 16 | 16 | 16 | `__nv_bfloat16`  | `__nv_bfloat16`  | `float`          |
| 32 |  8 | 16 | `__nv_bfloat16`  | `__nv_bfloat16`  | `float`          |
|  8 | 32 | 16 | `__nv_bfloat16`  | `__nv_bfloat16`  | `float`          |

### Double Precision (FP64) - Compute Capability 8.0+

| m | n | k | matrix_a type | matrix_b type | accumulator type |
|---|---|---|---------------|---------------|------------------|
| 8 | 8 | 4 | `double`      | `double`      | `double`         |

### Integer Types - Compute Capability 7.2+

| m  | n  | k  | matrix_a type | matrix_b type | accumulator type |
|----|----|----|---------------|---------------|------------------|
| 16 | 16 | 16 | `signed char` / `unsigned char` | `signed char` / `unsigned char` | `int` |
| 32 |  8 | 16 | `signed char` / `unsigned char` | `signed char` / `unsigned char` | `int` |
|  8 | 32 | 16 | `signed char` / `unsigned char` | `signed char` / `unsigned char` | `int` |

### Sub-Byte Operations - Compute Capability 7.5+

| m | n | k  | matrix_a type                  | matrix_b type                  | accumulator type |
|---|---|----|--------------------------------|--------------------------------|------------------|
| 8 | 8 | 32 | `experimental::precision::s4`  | `experimental::precision::s4`  | `int`            |
| 8 | 8 | 32 | `experimental::precision::u4`  | `experimental::precision::u4`  | `int`            |
| 8 | 8 |128 | `experimental::precision::b1`  | `experimental::precision::b1`  | `int`            |

---

## Complete GEMM Example

```cpp
#include <mma.h>
using namespace nvcuda;

// WMMA tile dimensions
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

__global__ void simple_wmma_gemm(half *a, half *b, float *c, float *d,
                                  int M, int N, int K,
                                  float alpha, float beta)
{
    // Each warp computes one 16x16 output tile
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

    // Declare fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;

    // Initialize accumulator to zero
    wmma::fill_fragment(acc_frag, 0.0f);

    // Loop over K dimension in tiles of WMMA_K
    for (int i = 0; i < K; i += WMMA_K) {
        int aRow = warpM * WMMA_M;
        int aCol = i;
        int bRow = i;
        int bCol = warpN * WMMA_N;

        // Bounds checking
        if (aRow < M && aCol < K && bRow < K && bCol < N) {
            // Load tiles from global memory
            wmma::load_matrix_sync(a_frag, a + aRow * K + aCol, K);
            wmma::load_matrix_sync(b_frag, b + bRow * N + bCol, N);

            // Perform matrix multiply-accumulate: acc += a * b
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
    }

    // Load C tile for beta * C
    int cRow = warpM * WMMA_M;
    int cCol = warpN * WMMA_N;

    if (cRow < M && cCol < N) {
        wmma::load_matrix_sync(c_frag, c + cRow * N + cCol, N, wmma::mem_row_major);

        // Apply alpha and beta scaling per-element
        for (int i = 0; i < acc_frag.num_elements; i++) {
            acc_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
        }

        // Store result
        wmma::store_matrix_sync(d + cRow * N + cCol, acc_frag, N, wmma::mem_row_major);
    }
}
```

### Launching the WMMA GEMM

```cpp
// Each warp handles one 16x16 tile
// Use blocks of 128 threads (4 warps per block)
dim3 blockDim(128, 1);
dim3 gridDim((M + WMMA_M - 1) / WMMA_M / (128 / 32),
             (N + WMMA_N - 1) / WMMA_N);

simple_wmma_gemm<<<gridDim, blockDim>>>(a, b, c, d, M, N, K, alpha, beta);
```

---

## Optimized GEMM with Shared Memory + WMMA

For best performance, combine shared memory tiling with WMMA:

```cpp
__global__ void wmma_gemm_shared(half *A, half *B, float *C,
                                  int M, int N, int K)
{
    // Shared memory tiles
    __shared__ half As[BLOCK_K][BLOCK_M];  // Transposed for coalesced access
    __shared__ half Bs[BLOCK_K][BLOCK_N];

    // Fragments for this warp
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;

    wmma::fill_fragment(acc_frag, 0.0f);

    // Iterate over K in blocks
    for (int kBlock = 0; kBlock < K; kBlock += BLOCK_K) {
        // Cooperatively load A and B tiles into shared memory
        // (all threads in block participate)
        load_tile_A(A, As, blockIdx, threadIdx, kBlock, M, K);
        load_tile_B(B, Bs, blockIdx, threadIdx, kBlock, K, N);
        __syncthreads();

        // Each warp processes its WMMA tiles from shared memory
        for (int kTile = 0; kTile < BLOCK_K; kTile += 16) {
            wmma::load_matrix_sync(a_frag, &As[kTile][warpRow * 16], BLOCK_M);
            wmma::load_matrix_sync(b_frag, &Bs[kTile][warpCol * 16], BLOCK_N);
            wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }
        __syncthreads();
    }

    // Store result to global memory
    int cRow = blockIdx.y * BLOCK_M + warpRow * 16;
    int cCol = blockIdx.x * BLOCK_N + warpCol * 16;
    wmma::store_matrix_sync(C + cRow * N + cCol, acc_frag, N, wmma::mem_row_major);
}
```

---

## Key Restrictions

1. **Warp-collective**: All 32 threads in a warp must execute WMMA operations together. Divergent control flow within a warp is not allowed around WMMA calls.
2. **Fragment opacity**: The mapping of fragment elements to threads is unspecified. You can access `frag.x[i]` for element-wise operations but the layout is opaque.
3. **Layout matching**: Fragment layouts specified during `load_matrix_sync` must match the actual data layout in memory.
4. **Compute capability**: Minimum SM 7.0 required. Compile with `-arch=sm_70` or higher.
5. **Alignment**: Pointers passed to load/store should be 256-bit (32-byte) aligned for best performance.
6. **Fragment reuse**: After `mma_sync`, fragments `a` and `b` can be reused. The accumulator `d` can alias `c` for in-place accumulation.

---

## Performance Tips

1. **Use shared memory**: Load data from global to shared memory first, then load fragments from shared memory. This allows data reuse across warps.
2. **Coalesced global loads**: When loading tiles into shared memory, ensure threads access contiguous global memory addresses.
3. **Double buffering**: Pipeline shared memory loads with WMMA computation to hide latency.
4. **Register pressure**: WMMA fragments consume registers. Monitor register usage with `--ptxas-options=-v`.
5. **Occupancy**: Balance tile sizes with register/shared memory usage for good occupancy.
6. **Use FP16 accumulation** when precision permits -- it uses fewer registers than FP32 accumulation.
