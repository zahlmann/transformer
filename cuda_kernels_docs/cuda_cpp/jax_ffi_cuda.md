# JAX Foreign Function Interface (FFI) for Custom CUDA Kernels

Sources:
- https://docs.jax.dev/en/latest/ffi.html (official JAX FFI documentation)
- https://openxla.org/xla/custom_call (XLA custom calls)
- https://github.com/dfm/extending-jax (extending JAX tutorial by Dan Foreman-Mackey)

---

## Overview

The JAX Foreign Function Interface (FFI) enables calling external compiled libraries (C++, CUDA) from JAX code. While JAX's built-in `jax.numpy` and `jax.lax` interfaces cover most numerical operations, the FFI is useful when you need to call optimized C/CUDA libraries or implement operations not expressible in JAX.

The FFI comprises two components:
1. A **header-only C++ library** from XLA (included in JAX v0.4.29+) in `xla/ffi/api/`
2. A **Python frontend** via the `jax.ffi` submodule

**Important limitation:** JAX cannot automatically differentiate through foreign functions. You must provide custom VJP (vector-Jacobian product) rules if you need autodiff.

---

## Part 1: C++ Backend Implementation

### Required Headers

```cpp
#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
```

### CPU Handler Example: RMS Normalization

#### Core Computation

```cpp
#include <cmath>
#include <cstdint>

float ComputeRmsNorm(float eps, int64_t size, const float *x, float *y) {
    float sm = 0.0f;
    for (int64_t n = 0; n < size; ++n) {
        sm += x[n] * x[n];
    }
    float scale = 1.0f / std::sqrt(sm / float(size) + eps);
    for (int64_t n = 0; n < size; ++n) {
        y[n] = x[n] * scale;
    }
    return scale;
}
```

#### FFI Handler Definition

```cpp
// Helper: extract dimensions from ffi::Buffer
template <ffi::DataType T>
std::pair<int64_t, int64_t> GetDims(const ffi::Buffer<T> &buffer) {
    auto dims = buffer.dimensions();
    if (dims.size() == 0) {
        return std::make_pair(0, 0);
    }
    return std::make_pair(buffer.element_count(), dims.back());
}

// Implementation handling batch dimensions
ffi::Error RmsNormImpl(float eps, ffi::Buffer<ffi::F32> x,
                       ffi::ResultBuffer<ffi::F32> y) {
    auto [totalSize, lastDim] = GetDims(x);
    if (lastDim == 0) {
        return ffi::Error::InvalidArgument("RmsNorm input must be an array");
    }
    for (int64_t n = 0; n < totalSize; n += lastDim) {
        ComputeRmsNorm(eps, lastDim, &(x.typed_data()[n]), &(y->typed_data()[n]));
    }
    return ffi::Error::Success();
}

// Register handler with XLA FFI
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RmsNorm, RmsNormImpl,
    ffi::Ffi::Bind()
        .Attr<float>("eps")               // Static attribute from Python
        .Arg<ffi::Buffer<ffi::F32>>()     // Input buffer x
        .Ret<ffi::Buffer<ffi::F32>>()     // Output buffer y
);
```

**Key concepts:**
- `ffi::Buffer<T>` - Input buffer with shape info and data pointer
- `ffi::ResultBuffer<T>` - Output buffer (pre-allocated by XLA)
- `.Attr<T>("name")` - Static attribute passed from Python side
- `.Arg<...>()` - Input argument binding
- `.Ret<...>()` - Output return binding

---

## Part 2: GPU/CUDA Handler

### CUDA Handler Signature

The GPU handler receives a `cudaStream_t` for launching kernels:

```cpp
ffi::Error RmsNormCudaImpl(cudaStream_t stream, float eps,
                            ffi::Buffer<ffi::F32> x,
                            ffi::ResultBuffer<ffi::F32> y) {
    auto [totalSize, lastDim] = GetDims(x);

    const int block_dim = 256;
    const int grid_dim = (totalSize / lastDim + block_dim - 1) / block_dim;

    // Launch CUDA kernel on the XLA-provided stream
    rms_norm_kernel<<<grid_dim, block_dim, 0, stream>>>(
        eps, lastDim, totalSize, x.typed_data(), y->typed_data());

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return ffi::Error::Internal(cudaGetErrorString(err));
    }
    return ffi::Error::Success();
}
```

### CUDA Handler Registration

```cpp
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    RmsNormCuda, RmsNormCudaImpl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()  // CUDA stream from XLA
        .Attr<float>("eps")
        .Arg<ffi::Buffer<ffi::F32>>()               // x (device memory)
        .Ret<ffi::Buffer<ffi::F32>>()               // y (device memory)
);
```

**Critical:** The `.Ctx<ffi::PlatformStream<cudaStream_t>>()` binding provides the CUDA stream. All kernel launches MUST use this stream for correct synchronization with XLA's execution.

### Complete GPU Custom Call Example (from XLA docs)

```cpp
__global__ void custom_call_kernel(const float* in0,
                                    const float* in1,
                                    float* out) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = in0[idx % 128] + in1[idx];
}

void do_custom_call(CUstream stream, BufferF32 in0, BufferF32 in1,
                    xla::ffi::Result<BufferF32> out) {
    size_t d0 = in0.dimensions[0];
    size_t d1 = in1.dimensions[0];

    const int64_t block_dim = 64;
    const int64_t grid_dim = 2048 / block_dim;
    custom_call_kernel<<<grid_dim, block_dim, 0, stream>>>(
        in0.data, in1.data, out->data);
}

XLA_FFI_DEFINE_HANDLER(handler, do_custom_call,
                       ffi::Ffi::Bind()
                           .Ctx<xla::ffi::PlatformStream<CUstream>>()
                           .Arg<BufferF32>()
                           .Arg<BufferF32>()
                           .Ret<BufferF32>());

XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "do_custom_call",
                         "CUDA", handler);
```

---

## Part 3: Building and Registering

### CMake Build

```cmake
cmake_minimum_required(VERSION 3.18)
project(my_ffi_ops LANGUAGES CXX CUDA)

# Find JAX/XLA headers (shipped with jaxlib)
find_package(Python REQUIRED)
execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import jaxlib; print(jaxlib.__path__[0])"
    OUTPUT_VARIABLE JAXLIB_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)

add_library(my_ops SHARED cpu_ops.cc gpu_ops.cu)
target_include_directories(my_ops PRIVATE ${JAXLIB_PATH}/include)
```

Build:
```bash
cmake -DCMAKE_BUILD_TYPE=Release -B build .
cmake --build build
cmake --install build
```

### Registration via ctypes

```python
import ctypes
from pathlib import Path
import jax.ffi

# Load the compiled shared library
path = next(Path("build").glob("libmy_ops*"))
my_lib = ctypes.cdll.LoadLibrary(path)

# Register CPU target
jax.ffi.register_ffi_target(
    "rms_norm",
    jax.ffi.pycapsule(my_lib.RmsNorm),
    platform="cpu"
)

# Register GPU target
jax.ffi.register_ffi_target(
    "rms_norm_cuda",
    jax.ffi.pycapsule(my_lib.RmsNormCuda),
    platform="CUDA"
)
```

### Registration via nanobind (alternative)

```cpp
#include <type_traits>
#include "nanobind/nanobind.h"
#include "xla/ffi/api/c_api.h"

namespace nb = nanobind;

template <typename T>
nb::capsule EncapsulateFfiCall(T *fn) {
    static_assert(std::is_invocable_r_v<XLA_FFI_Error *, T, XLA_FFI_CallFrame *>,
                  "Encapsulated function must be an XLA FFI handler");
    return nb::capsule(reinterpret_cast<void *>(fn));
}

NB_MODULE(my_ops, m) {
    m.def("rms_norm", []() { return EncapsulateFfiCall(RmsNorm); });
    m.def("rms_norm_cuda", []() { return EncapsulateFfiCall(RmsNormCuda); });
}
```

Then in Python:
```python
import my_ops
jax.ffi.register_ffi_target("rms_norm", my_ops.rms_norm(), platform="cpu")
jax.ffi.register_ffi_target("rms_norm_cuda", my_ops.rms_norm_cuda(), platform="CUDA")
```

### Registration via pybind11 (older approach, from extending-jax)

```cpp
#include <pybind11/pybind11.h>

template <typename T>
pybind11::capsule EncapsulateFunction(T* fn) {
    return pybind11::capsule((void*)fn, "xla._CUSTOM_CALL_TARGET");
}

pybind11::dict Registrations() {
    pybind11::dict dict;
    dict["gpu_my_op_f32"] = EncapsulateFunction(gpu_my_op<float>);
    dict["gpu_my_op_f64"] = EncapsulateFunction(gpu_my_op<double>);
    return dict;
}

PYBIND11_MODULE(gpu_ops, m) {
    m.def("registrations", &Registrations);
}
```

---

## Part 4: Python Frontend (Calling from JAX)

### Basic FFI Call

```python
import numpy as np
import jax
import jax.numpy as jnp

def rms_norm(x, eps=1e-5):
    if x.dtype != jnp.float32:
        raise ValueError("Only float32 dtype is implemented")

    call = jax.ffi.ffi_call(
        "rms_norm",                                 # Registered target name
        jax.ShapeDtypeStruct(x.shape, x.dtype),     # Output shape/dtype spec
        vmap_method="broadcast_all",                 # Batching behavior
    )

    # eps must be a numpy type, NOT a JAX array (it's a static attribute)
    return call(x, eps=np.float32(eps))
```

### Multiple Outputs

```python
def rms_norm_fwd(x, eps=1e-5):
    y, res = jax.ffi.ffi_call(
        "rms_norm_fwd",
        (
            jax.ShapeDtypeStruct(x.shape, x.dtype),       # Output 1: normalized
            jax.ShapeDtypeStruct(x.shape[:-1], x.dtype),  # Output 2: residual
        ),
        vmap_method="broadcast_all",
    )(x, eps=np.float32(eps))
    return y, (res, x)
```

### Cross-Platform (CPU + GPU)

```python
def rms_norm_cross_platform(x, eps=1e-5):
    assert x.dtype == jnp.float32
    out_type = jax.ShapeDtypeStruct(x.shape, x.dtype)

    def impl(target_name):
        return lambda x: jax.ffi.ffi_call(
            target_name,
            out_type,
            vmap_method="broadcast_all",
        )(x, eps=np.float32(eps))

    # XLA selects the right target at compile time -- zero runtime overhead
    return jax.lax.platform_dependent(
        x,
        cpu=impl("rms_norm"),
        cuda=impl("rms_norm_cuda")
    )
```

---

## Part 5: Custom Differentiation Rules

Since JAX cannot differentiate through opaque foreign functions, you must define custom VJP rules.

### Forward Pass (saves residuals for backward)

```python
@jax.custom_vjp
def rms_norm(x, eps=1e-5):
    # Default implementation (called during tracing)
    return jax.ffi.ffi_call(
        "rms_norm",
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        vmap_method="broadcast_all",
    )(x, eps=np.float32(eps))

def rms_norm_fwd(x, eps=1e-5):
    y, res = jax.ffi.ffi_call(
        "rms_norm_fwd",
        (
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            jax.ShapeDtypeStruct(x.shape[:-1], x.dtype),
        ),
        vmap_method="broadcast_all",
    )(x, eps=np.float32(eps))
    return y, (res, x)  # Return residuals for backward pass
```

### Backward Pass

```python
def rms_norm_bwd(eps, res, ct):
    del eps
    res, x = res
    return (
        jax.ffi.ffi_call(
            "rms_norm_bwd",
            jax.ShapeDtypeStruct(ct.shape, ct.dtype),
            vmap_method="broadcast_all",
        )(res, x, ct),
    )
```

### Register the VJP

```python
rms_norm = jax.custom_vjp(rms_norm, nondiff_argnums=(1,))
rms_norm.defvjp(rms_norm_fwd, rms_norm_bwd)

# Now autodiff works!
x = jnp.ones((4, 8))
grads = jax.grad(lambda x: rms_norm(x).sum())(x)
```

---

## Part 6: Batching with vmap

The `vmap_method` parameter controls how `jax.vmap` interacts with FFI calls:

| Method | Behavior | Performance |
|--------|----------|-------------|
| `"broadcast_all"` | Passes batched arrays directly to FFI (function must handle batch dims) | Best -- no overhead |
| `"sequential"` | Converts vmap to a scan loop, calling FFI once per batch element | Safe but slow |

```python
# broadcast_all: function sees (batch, ...) shaped inputs
call = jax.ffi.ffi_call("my_op", out_type, vmap_method="broadcast_all")

# sequential: JAX loops over batch dimension
call = jax.ffi.ffi_call("my_op", out_type, vmap_method="sequential")
```

---

## Part 7: GPU Kernel Pattern (from extending-jax)

### Project Structure

```
project/
+-- lib/
|   +-- kernels.h             # Descriptor struct
|   +-- kernels.cc.cu         # CUDA kernel + wrapper
|   +-- gpu_ops.cc            # pybind11 module
|   +-- kernel_helpers.h      # Descriptor serialization
+-- src/my_jax_op/
|   +-- __init__.py           # Registration
|   +-- my_op.py              # JAX primitive definition
+-- CMakeLists.txt
+-- pyproject.toml
```

### Descriptor Pattern (for passing metadata to GPU)

```cpp
// kernels.h
struct MyOpDescriptor {
    std::int64_t size;
    float param;
};

// kernel_helpers.h
template <typename T>
std::string PackDescriptorAsString(const T& descriptor) {
    return std::string(
        bit_cast<const char*>(&descriptor),
        sizeof(T));
}

template <typename T>
const T* UnpackDescriptor(const char* opaque, std::size_t opaque_len) {
    if (opaque_len != sizeof(T)) {
        throw std::runtime_error("Invalid opaque object size");
    }
    return bit_cast<const T*>(opaque);
}
```

### CUDA Kernel

```cuda
// kernels.cc.cu
template <typename T>
__global__ void my_kernel(
    std::int64_t size,
    const T* input,
    T* output)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = /* compute something with input[idx] */;
    }
}

template <typename T>
void gpu_my_op(
    cudaStream_t stream,
    void** buffers,
    const char* opaque,
    std::size_t opaque_len)
{
    // Unpack inputs and outputs from buffers array
    const T* input = reinterpret_cast<const T*>(buffers[0]);
    T* output = reinterpret_cast<T*>(buffers[1]);

    // Unpack descriptor
    const MyOpDescriptor& d =
        *UnpackDescriptor<MyOpDescriptor>(opaque, opaque_len);
    const std::int64_t size = d.size;

    // Launch kernel
    const int block_dim = 128;
    const int grid_dim = std::min<int>(1024, (size + block_dim - 1) / block_dim);

    my_kernel<T><<<grid_dim, block_dim, 0, stream>>>(size, input, output);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(error));
    }
}
```

---

## Part 8: Sharding / Distributed Computation

### Using shard_map

```python
from functools import partial
from jax.sharding import PartitionSpec as P

mesh = jax.make_mesh((4,), ("x",))

@partial(
    jax.shard_map,
    mesh=mesh,
    in_specs=P("x", None),
    out_specs=P("x", None)
)
def rms_norm_sharded(x):
    return rms_norm(x)
```

If input/output shardings match the shard_map specs, no communication is required -- the FFI call executes on each shard's local data.

---

## Summary: When to Use FFI vs Alternatives

| Approach | When to Use |
|----------|------------|
| **Pure JAX** (`jax.numpy`, `jax.lax`) | Default choice. XLA compiler optimizes well. |
| **Pallas** | Custom kernels in Python, compiles to GPU code. Easier than C++. |
| **FFI** | Pre-existing optimized C/CUDA library, or operations that XLA/Pallas can't express efficiently. |

The FFI is a **last resort** because:
- You lose automatic differentiation (must implement custom VJP)
- You lose automatic compilation/optimization by XLA
- Build complexity increases (CMake, CUDA compilation)
- Maintenance burden across JAX versions

But it's essential when you need maximum performance from hand-tuned CUDA kernels, especially for operations like custom attention mechanisms, specialized reductions, or Tensor Core operations.
