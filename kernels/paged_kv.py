"""Paged KV cache management for variable-length batched inference.

Manages physical pages of KV cache data. Each page holds PAGE_SIZE positions
worth of KV data for all layers and KV heads. Pages are allocated on demand
as sequences grow and freed when sequences complete.

Two modes of operation:
  1. Python-level paging: to_contiguous() / update_from_contiguous() for kernels
     that expect contiguous KV buffers. Slow (CPU numpy loops).
  2. GPU-accelerated paging: to_contiguous_gpu() / update_page_gpu() use JIT-compiled
     JAX gather/scatter on the GPU pool. 3-4x faster than Python-level paging.
     The decode kernel still receives contiguous KV buffers (unchanged).
"""

import jax
import jax.numpy as jnp
import numpy as np




# ── Python wrappers for GPU copy kernels ──

def _next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p


@jax.jit
def _jit_gather(pool, gather_idx):
    """JIT-compiled gather: pool[gather_idx] → contiguous KV."""
    return pool[gather_idx]


def _build_gather_indices(page_table, config, page_size=64):
    """Build index array mapping contiguous KV positions to pool offsets.

    Returns int32 array of shape (kv_per_seq,) where each element is the
    source index in the flat pool for the corresponding contiguous KV element.
    Fully vectorized with numpy (no Python inner loops).
    """
    n_layers = config["n_layers"]
    n_kv_heads = config.get("n_kv_heads", config["n_heads"])
    d_head = config["d_head"]
    max_seq = config["context_len"]
    page_elems = n_layers * 2 * n_kv_heads * page_size * d_head
    kv_per_seq = n_layers * 2 * n_kv_heads * max_seq * d_head

    pos = np.arange(max_seq)
    page_ids = pos // page_size
    pos_in_page = pos % page_size

    # Map logical pages to physical pages
    pt_array = np.zeros(max_seq // page_size + 1, dtype=np.int32)
    pt_array[:len(page_table)] = page_table
    phys_pages = pt_array[page_ids]  # vectorized lookup

    # Physical base for each position
    page_bases = phys_pages * page_elems  # (max_seq,)
    dh = np.arange(d_head)  # (d_head,)

    pkv = n_kv_heads * page_size * d_head
    ckv = n_kv_heads * max_seq * d_head

    # Build pool offsets: (max_seq, d_head) for each (layer, kv_type, head)
    # pos_offsets[p, d] = page_bases[p] + pos_in_page[p] * d_head + d
    pos_offsets = (page_bases[:, None] + pos_in_page[:, None] * d_head
                   + dh[None, :])  # (max_seq, d_head)

    idx = np.empty(kv_per_seq, dtype=np.int32)
    for layer in range(n_layers):
        for kv_type in range(2):
            for head in range(n_kv_heads):
                pool_off = pos_offsets + (layer * 2 * pkv + kv_type * pkv
                                          + head * page_size * d_head)
                cont_start = layer * 2 * ckv + kv_type * ckv + head * max_seq * d_head
                idx[cont_start:cont_start + max_seq * d_head] = pool_off.ravel()

    return jnp.array(idx, dtype=jnp.int32)


def paged_to_contiguous_gpu(pool_gpu, page_table, config, page_size=64,
                             _gather_cache=None):
    """Convert paged KV to contiguous using JIT-compiled JAX gather.

    Args:
        pool_gpu: flat bf16 GPU pool array
        page_table: Python list of physical page IDs for this sequence
        config: model config dict
        _gather_cache: optional dict for caching gather indices

    Returns:
        contiguous KV buffer: (kv_per_seq,) bf16
    """
    cache_key = tuple(page_table)
    if _gather_cache is not None and cache_key in _gather_cache:
        gather_idx = _gather_cache[cache_key]
    else:
        gather_idx = _build_gather_indices(page_table, config, page_size)
        if _gather_cache is not None:
            _gather_cache[cache_key] = gather_idx

    return _jit_gather(pool_gpu, gather_idx)


def _build_scatter_indices(phys_page, position, config, page_size=64):
    """Precompute source (contiguous) and dest (pool) indices for one position."""
    n_layers = config["n_layers"]
    n_kv_heads = config.get("n_kv_heads", config["n_heads"])
    d_head = config["d_head"]
    max_seq = config["context_len"]
    page_elems = n_layers * 2 * n_kv_heads * page_size * d_head
    pos_in_page = position % page_size

    n_elems = n_layers * 2 * n_kv_heads * d_head
    src_idx = np.empty(n_elems, dtype=np.int32)
    dst_idx = np.empty(n_elems, dtype=np.int32)

    i = 0
    for layer in range(n_layers):
        for kv_type in range(2):
            for head in range(n_kv_heads):
                s = (layer * 2 * n_kv_heads * max_seq * d_head
                     + kv_type * n_kv_heads * max_seq * d_head
                     + head * max_seq * d_head
                     + position * d_head)
                src_idx[i:i + d_head] = np.arange(s, s + d_head)
                d_off = (phys_page * page_elems
                         + layer * 2 * n_kv_heads * page_size * d_head
                         + kv_type * n_kv_heads * page_size * d_head
                         + head * page_size * d_head
                         + pos_in_page * d_head)
                dst_idx[i:i + d_head] = np.arange(d_off, d_off + d_head)
                i += d_head

    return jnp.array(src_idx), jnp.array(dst_idx)


@jax.jit
def _jit_scatter(pool, kv_contiguous, src_idx, dst_idx):
    """JIT-compiled gather + scatter for pool update."""
    values = kv_contiguous[src_idx]
    return pool.at[dst_idx].set(values)


def update_page_gpu(kv_contiguous, pool_gpu, phys_page, position,
                     config, page_size=64, _cache=None):
    """Copy one position's K_new/V_new from contiguous KV to the pool page.

    Uses JIT-compiled JAX gather (from contiguous) + scatter (to pool).
    Only touches n_layers * 2 * n_kv_heads * d_head elements (~9 KB for d=768).
    """
    cache_key = (phys_page, position)
    if _cache is not None and cache_key in _cache:
        src_idx, dst_idx = _cache[cache_key]
    else:
        src_idx, dst_idx = _build_scatter_indices(
            phys_page, position, config, page_size)
        if _cache is not None:
            _cache[cache_key] = (src_idx, dst_idx)

    return _jit_scatter(pool_gpu, kv_contiguous, src_idx, dst_idx)


class PagePool:
    """Manages a pool of physical pages for KV cache data.

    Each page stores PAGE_SIZE positions of KV data for all layers/heads.
    Pages are allocated from a fixed pool and freed back when sequences complete.

    Args:
        config: model config dict
        max_pages: total number of physical pages in the pool
        page_size: positions per page (default 64, matching KV_TILE)
    """

    def __init__(self, config, max_pages, page_size=64):
        self.config = config
        self.page_size = page_size
        self.n_layers = config["n_layers"]
        self.n_kv_heads = config.get("n_kv_heads", config["n_heads"])
        self.d_head = config["d_head"]
        self.max_seq = config["context_len"]
        self.max_pages = max_pages
        self.max_pages_per_seq = (self.max_seq + page_size - 1) // page_size

        # Per-page element count: all layers, K+V, all KV heads, page_size positions
        self.page_elems = self.n_layers * 2 * self.n_kv_heads * page_size * self.d_head

        # Per-sequence contiguous KV size (for kernel interface)
        self.kv_per_seq = self.n_layers * 2 * self.n_kv_heads * self.max_seq * self.d_head

        # Physical page pool (CPU numpy for fast page management)
        self.pool = np.zeros((max_pages, self.page_elems), dtype=np.float16)

        # GPU-resident pool (flat bf16 array, lazily initialized)
        self._gpu_pool = None

        # Free page tracking
        self.free_pages = list(range(max_pages))

        # Per-sequence page tables: seq_idx -> list of physical page IDs
        self.page_tables = {}

        # Caches for GPU operations (invalidated when pages change)
        self._pt_gpu_cache = {}      # seq_idx -> (n_pages, jnp array)
        self._gather_cache = {}      # tuple(page_table) -> gather_idx jnp array
        self._scatter_cache = {}     # (phys_page, position) -> (src_idx, dst_idx) jnp arrays

    def alloc_pages(self, n_pages):
        """Allocate n physical pages from the pool."""
        if len(self.free_pages) < n_pages:
            raise RuntimeError(
                f"Page pool exhausted: need {n_pages}, have {len(self.free_pages)}")
        pages = [self.free_pages.pop() for _ in range(n_pages)]
        return pages

    def free_seq_pages(self, seq_idx):
        """Free all pages belonging to a sequence."""
        if seq_idx in self.page_tables:
            self.free_pages.extend(self.page_tables[seq_idx])
            del self.page_tables[seq_idx]
            self._pt_gpu_cache.pop(seq_idx, None)

    def store_prefill_kv(self, seq_idx, k_caches, v_caches, seq_len):
        """Store prefilled KV caches into pages.

        Args:
            seq_idx: sequence identifier
            k_caches: list of (n_kv_heads, context_len, d_head) bf16 per layer
            v_caches: same
            seq_len: actual prompt length (may be < context_len)
        """
        n_pages = (seq_len + self.page_size - 1) // self.page_size
        pages = self.alloc_pages(n_pages)
        self.page_tables[seq_idx] = pages

        for page_idx, phys_page in enumerate(pages):
            pos_start = page_idx * self.page_size
            pos_end = min(pos_start + self.page_size, seq_len)

            # Pack into page layout: [layer, kv_type, head, pos, d_head]
            page_data = np.zeros(self.page_elems, dtype=np.float16)
            for layer in range(self.n_layers):
                for kv_type, caches in enumerate([k_caches, v_caches]):
                    cache_np = np.array(caches[layer])  # (n_kv_heads, ctx, d_head)
                    for head in range(self.n_kv_heads):
                        off = (layer * 2 * self.n_kv_heads * self.page_size * self.d_head
                               + kv_type * self.n_kv_heads * self.page_size * self.d_head
                               + head * self.page_size * self.d_head)
                        n_pos = pos_end - pos_start
                        page_data[off:off + n_pos * self.d_head] = (
                            cache_np[head, pos_start:pos_end, :].reshape(-1))
            self.pool[phys_page] = page_data

    def ensure_page_for_pos(self, seq_idx, position):
        """Ensure a page exists for the given position. Allocates if needed."""
        page_idx = position // self.page_size
        pages = self.page_tables.get(seq_idx, [])
        while len(pages) <= page_idx:
            new_page = self.alloc_pages(1)[0]
            pages.append(new_page)
        self.page_tables[seq_idx] = pages

    # ── Python-level paging (slow, CPU numpy loops) ──

    def to_contiguous(self, seq_idx):
        """Convert paged KV to contiguous format for the kernel (CPU path)."""
        pages = self.page_tables.get(seq_idx, [])
        kv = np.zeros(self.kv_per_seq, dtype=np.float16)

        for page_idx, phys_page in enumerate(pages):
            page_data = self.pool[phys_page]
            pos_start = page_idx * self.page_size

            for layer in range(self.n_layers):
                for kv_type in range(2):
                    for head in range(self.n_kv_heads):
                        src_off = (layer * 2 * self.n_kv_heads * self.page_size * self.d_head
                                   + kv_type * self.n_kv_heads * self.page_size * self.d_head
                                   + head * self.page_size * self.d_head)
                        dst_off = (layer * 2 * self.n_kv_heads * self.max_seq * self.d_head
                                   + kv_type * self.n_kv_heads * self.max_seq * self.d_head
                                   + head * self.max_seq * self.d_head
                                   + pos_start * self.d_head)
                        n_elems = self.page_size * self.d_head
                        kv[dst_off:dst_off + n_elems] = page_data[src_off:src_off + n_elems]

        return jnp.array(kv, dtype=jnp.bfloat16)

    def update_from_contiguous(self, seq_idx, kv_contiguous, position):
        """Update pages from contiguous KV output (CPU path)."""
        page_idx = position // self.page_size
        pages = self.page_tables.get(seq_idx, [])
        if page_idx >= len(pages):
            return

        phys_page = pages[page_idx]
        kv_np = np.array(kv_contiguous)
        pos_in_page = position % self.page_size

        for layer in range(self.n_layers):
            for kv_type in range(2):
                for head in range(self.n_kv_heads):
                    src_off = (layer * 2 * self.n_kv_heads * self.max_seq * self.d_head
                               + kv_type * self.n_kv_heads * self.max_seq * self.d_head
                               + head * self.max_seq * self.d_head
                               + position * self.d_head)
                    dst_off = (layer * 2 * self.n_kv_heads * self.page_size * self.d_head
                               + kv_type * self.n_kv_heads * self.page_size * self.d_head
                               + head * self.page_size * self.d_head
                               + pos_in_page * self.d_head)
                    self.pool[phys_page, dst_off:dst_off + self.d_head] = (
                        kv_np[src_off:src_off + self.d_head])

    # ── GPU-resident pool for GPU-accelerated paging ──

    def sync_to_gpu(self):
        """Upload the CPU pool to GPU. Call after store_prefill_kv()."""
        self._gpu_pool = jnp.array(self.pool.reshape(-1), dtype=jnp.bfloat16)

    @property
    def gpu_pool(self):
        """Flat bf16 GPU array of shape (max_pages * page_elems,)."""
        if self._gpu_pool is None:
            self.sync_to_gpu()
        return self._gpu_pool

    @gpu_pool.setter
    def gpu_pool(self, value):
        self._gpu_pool = value

    def get_page_table_gpu(self, seq_idx):
        """Return page table for a sequence as a GPU int32 array (cached)."""
        pages = self.page_tables.get(seq_idx, [])
        n_pages = len(pages)
        cached = self._pt_gpu_cache.get(seq_idx)
        if cached is not None and cached[0] == n_pages:
            return cached[1]
        pt = np.full(self.max_pages_per_seq, -1, dtype=np.int32)
        pt[:n_pages] = pages
        pt_gpu = jnp.array(pt, dtype=jnp.int32)
        self._pt_gpu_cache[seq_idx] = (n_pages, pt_gpu)
        return pt_gpu

    def to_contiguous_gpu(self, seq_idx):
        """Convert paged KV to contiguous on GPU using JIT-compiled gather."""
        pages = self.page_tables.get(seq_idx, [])
        return paged_to_contiguous_gpu(
            self.gpu_pool, pages, self.config, self.page_size,
            _gather_cache=self._gather_cache)

    def update_page_gpu(self, seq_idx, kv_contiguous, position):
        """Copy one position from contiguous KV back to pool page on GPU."""
        page_idx = position // self.page_size
        pages = self.page_tables.get(seq_idx, [])
        phys_page = pages[page_idx]
        self._gpu_pool = update_page_gpu(
            kv_contiguous, self.gpu_pool, phys_page, position,
            self.config, self.page_size, _cache=self._scatter_cache)

    @property
    def pages_used(self):
        return self.max_pages - len(self.free_pages)

    @property
    def memory_used_mb(self):
        return self.pages_used * self.page_elems * 2 / 1e6  # bf16 = 2 bytes

    @property
    def memory_allocated_mb(self):
        return self.max_pages * self.page_elems * 2 / 1e6
