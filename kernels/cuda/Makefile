# Build the CUDA kernel as a shared library for JAX custom_call
#
# Usage:
#   make          — build kernels/cuda/libfused_transformer.so
#   make clean    — remove build artifacts
#   make test     — build and run correctness test against Triton kernel
#
# Requirements:
#   - CUDA toolkit (nvcc) — found at /usr/local/cuda/bin/nvcc
#   - Python headers (for pybind11 if used)

NVCC := /usr/local/cuda/bin/nvcc
CUDA_HOME := /usr/local/cuda

# Compute capability for RTX 4080 SUPER (Ada Lovelace, SM 8.9)
SM := 89

# Compiler flags
NVCC_FLAGS := \
	-O3 \
	-arch=sm_$(SM) \
	-gencode=arch=compute_$(SM),code=sm_$(SM) \
	--shared \
	-Xcompiler -fPIC \
	-std=c++17 \
	-I$(CUDA_HOME)/include \
	--expt-relaxed-constexpr

# Add maxrregcount to control register usage (tune this)
# Lower values = more blocks per SM but more register spilling
# Try: 128 (default), 96, 80, 64
# NVCC_FLAGS += -maxrregcount=96

TARGET := libfused_transformer.so
SRC := fused_transformer_ce.cu

.PHONY: all clean test

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $<
	@echo "Built $@ (SM $(SM))"
	@echo "Shared memory per block: $$(python3 -c 'print(128*64*2 + 64*32*2 + 128*32*4*2 + 128*4, "bytes =", (128*64*2 + 64*32*2 + 128*32*4*2 + 128*4)/1024, "KB (WMMA layout)")')"

clean:
	rm -f $(TARGET)

test: $(TARGET)
	cd ../.. && uv run kernels/cuda/test_kernel.py
