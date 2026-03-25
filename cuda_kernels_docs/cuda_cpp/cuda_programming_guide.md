# CUDA C++ Programming Guide: Key Sections

Source: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html

---

## 1. Thread Hierarchy

### Overview

CUDA organizes parallel execution through a hierarchical structure. Threads are grouped into blocks, which are further organized into grids. This design enables scalable parallel computation across GPUs with varying numbers of cores.

### Thread Indexing

Each thread has a unique identifier through the `threadIdx` built-in variable, which is a 3-component vector enabling 1D, 2D, or 3D indexing. For blocks with dimensions (Dx, Dy, Dz):

- **1D block**: Thread ID = `threadIdx.x`
- **2D block** (Dx x Dy): Thread ID = `x + y * Dx`
- **3D block** (Dx x Dy x Dz): Thread ID = `x + y * Dx + z * Dx * Dy`

### Thread Blocks

A thread block is a cooperative group of threads that:

- Execute independently from other blocks
- Can synchronize internally via `__syncthreads()`
- Share low-latency shared memory
- Have a maximum capacity of **1024 threads per block**
- Are identified by `blockIdx` and sized by `blockDim`

### Grids

Grids organize blocks in 1D, 2D, or 3D arrangements. The grid dimension is determined by problem size and accessible via `gridDim`.

#### Example: 2D Matrix Addition

```cuda
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N])
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
        C[i][j] = A[i][j] + B[i][j];
}

int main()
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    MatAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);
}
```

### Thread Block Clusters (Compute Capability 9.0+)

Thread block clusters group multiple blocks within a GPU Processing Cluster (GPC), enabling:

- Hardware-supported synchronization across blocks
- Access to distributed shared memory
- Portable cluster size of up to 8 blocks maximum

#### Compile-Time Cluster Configuration

```cuda
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel(float *input, float* output)
{
    // kernel code
}

int main()
{
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
    cluster_kernel<<<numBlocks, threadsPerBlock>>>(input, output);
}
```

#### Runtime Cluster Configuration

```cuda
__global__ void cluster_kernel(float *input, float* output)
{
    // kernel code
}

int main()
{
    cudaLaunchConfig_t config = {0};
    config.gridDim = numBlocks;
    config.blockDim = threadsPerBlock;

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = 2;
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.attrs = attribute;
    config.numAttrs = 1;

    cudaLaunchKernelEx(&config, cluster_kernel, input, output);
}
```

### Block Launch Specifications

Use the `__block_size__` attribute to explicitly configure thread blocks and clusters:

```cuda
__block_size__((1024, 1, 1), (2, 2, 2)) __global__ void foo();

// Launches 8x8x8 clusters
foo<<<dim3(8, 8, 8)>>>();
```

---

## 2. Memory Hierarchy

CUDA provides a multi-level memory system optimized for different access patterns and persistence requirements.

### Memory Architecture Overview

The hierarchy includes (from fastest/smallest to slowest/largest):

1. **Registers** - Per-thread private memory
2. **Shared Memory** - Per-block cooperative memory
3. **Local Memory** - Per-thread private (spilled registers)
4. **Global Memory** - Device-wide accessible memory
5. **Constant Memory** - Read-only, cached memory space
6. **Texture Memory** - Read-only, specialized addressing

### Memory Characteristics

| Memory Type | Scope   | Lifetime    | Cached | Size     |
|-------------|---------|-------------|--------|----------|
| Registers   | Thread  | Kernel      | N/A    | Limited  |
| Shared      | Block   | Block       | N/A    | ~96KB    |
| Local       | Thread  | Kernel      | Yes    | Limited  |
| Global      | Device  | Application | Yes    | GB       |
| Constant    | Device  | Application | Yes    | 64KB     |
| Texture     | Device  | Application | Yes    | GB       |

### Global Memory

Accessible by all threads across the entire GPU with application-level persistence. Data persists across kernel launches. Memory transfers between host and device occur through this space.

### Local Memory

Per-thread storage for data that exceeds register capacity. Although logically per-thread, it physically resides in global memory with reduced performance due to lack of caching efficiency.

### Constant Memory

Optimized access for read-only data patterns across all threads. All threads access the same addresses simultaneously for optimal bandwidth. Total capacity is 64KB per device.

### Texture Memory

Offers different addressing modes and data filtering capabilities for specific data formats. Optimized for spatial locality in 2D/3D data access patterns (image processing, interpolation).

### Cluster-Level Shared Memory

With thread block clusters (Compute Capability 9.0+), blocks within a cluster access **distributed shared memory**, enabling:

- Read, write, and atomic operations across blocks in a cluster
- Hardware-efficient inter-block communication

---

## 3. Shared Memory

### Declaration and Scope

Shared memory is declared with the `__shared__` qualifier and is local to a thread block:

```cuda
__global__ void kernel()
{
    __shared__ float data[256];
    // All threads in block can access data[]
}
```

### Dynamic Shared Memory

You can also allocate shared memory dynamically at kernel launch time:

```cuda
extern __shared__ float dynamic_smem[];

__global__ void kernel()
{
    // dynamic_smem size determined by third argument in <<<...>>>
    dynamic_smem[threadIdx.x] = 1.0f;
}

// Launch with 256 * sizeof(float) bytes of dynamic shared memory
kernel<<<numBlocks, threadsPerBlock, 256 * sizeof(float)>>>();
```

### Shared Memory Lifetime and Visibility

Shared memory is low-latency memory near each processor core, similar to L1 cache. Memory persists for the block's lifetime and is not visible across blocks.

### Synchronization Requirements

Use `__syncthreads()` as a barrier ensuring all block threads reach the synchronization point before proceeding:

```cuda
__global__ void syncExample(float *input, float *output)
{
    __shared__ float shared[512];
    int idx = threadIdx.x;

    // Load data
    shared[idx] = input[idx];
    __syncthreads();  // All threads wait here

    // Process using shared data -- safe to read other threads' values
    output[idx] = shared[idx] * 2.0f;
}
```

### Bank Conflicts

Shared memory is divided into **32 banks** (one per warp thread). Simultaneous accesses to different addresses within the same bank from different threads cause serialization (bank conflicts). Optimal access patterns avoid bank conflicts.

**Bank mapping**: Address `addr` maps to bank `(addr / 4) % 32` (for 4-byte words).

#### Bank Conflict Avoidance

```cuda
__global__ void noBankConflict()
{
    __shared__ float data[256];
    int tid = threadIdx.x;

    // Stride of 1: consecutive threads access consecutive banks -- NO conflict
    float val = data[tid];

    // Stride of 2: every other bank accessed, 2-way conflict
    float val2 = data[tid * 2];

    // Stride of 32: all threads hit same bank -- 32-way conflict (worst case)
    float val3 = data[tid * 32];
}
```

**Padding trick** to avoid bank conflicts in 2D arrays:

```cuda
// Without padding: column access causes 32-way bank conflicts
__shared__ float tile[32][32];

// With padding: add 1 extra column to shift bank alignment
__shared__ float tile[32][32 + 1];  // Now column access is conflict-free
```

### Cooperative Use Pattern: Tiled Matrix Multiply

```cuda
__global__ void matMulShared(float *A, float *B, float *C, int N)
{
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t < N / TILE_SIZE; ++t) {
        // Collaborative loading: each thread loads one element
        As[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        __syncthreads();

        // Compute partial dot product from shared memory (fast!)
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();  // Wait before loading next tile
    }

    C[row * N + col] = sum;
}
```

### Cooperative Use Pattern: Parallel Reduction

```cuda
__global__ void reductionKernel(float *input, float *output)
{
    __shared__ float sharedData[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Load from global to shared
    sharedData[tid] = input[idx];
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();  // Sync after each reduction step
    }

    // Store result
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}
```

### Distributed Shared Memory (Compute Capability 9.0+)

For thread block clusters, blocks can access each other's shared memory:

```cuda
__global__ void distributedSharedMemory(float *data)
{
    __shared__ float blockData[256];
    int idx = threadIdx.x;
    int blockId = blockIdx.x;

    blockData[idx] = data[blockId * 256 + idx];
    cluster.sync();  // Synchronize across cluster

    // Blocks in cluster can now access each other's blockData
    // through distributed shared memory interface
}
```

---

## Key Concepts Summary

- **Thread Organization**: Hierarchical grouping (threads -> blocks -> grids -> clusters) enables scalable parallelism matching GPU topology.
- **Memory Optimization**: Different memory spaces are optimized for different usage patterns. Select appropriate storage based on access characteristics and scope requirements.
- **Synchronization Discipline**: Shared memory requires explicit synchronization through `__syncthreads()` to prevent race conditions and ensure correct inter-thread coordination.
- **Bank Conflicts**: Shared memory is divided into 32 banks. Avoid stride patterns that cause multiple threads to access the same bank simultaneously.
- **Tiling Pattern**: Load data cooperatively from global memory into shared memory, synchronize, then compute from shared memory. This is the fundamental optimization pattern for memory-bound kernels.
