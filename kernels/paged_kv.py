"""Paged KV cache: allocate 64-position pages on demand, free on completion.

The decode kernel receives contiguous KV buffers. This module converts between
paged (memory-efficient) and contiguous (kernel-compatible) formats using
JIT-compiled JAX gather/scatter on GPU.
"""

import jax
import jax.numpy as jnp
import numpy as np

PAGE_SIZE = 64  # positions per page, matches KV_TILE in decode kernels


@jax.jit
def _jit_gather(pool, gather_idx):
    return pool[gather_idx]


@jax.jit
def _jit_scatter(pool, kv_contiguous, src_idx, dst_idx):
    return pool.at[dst_idx].set(kv_contiguous[src_idx])


class PagePool:
    """Fixed-size pool of physical pages for KV cache data.

    Each page stores PAGE_SIZE=64 positions of KV data for all layers/heads.
    Layout within a page: [layer, kv_type, head, pos_in_page, d_head].
    """

    def __init__(self, config, max_pages):
        self.n_layers = config["n_layers"]
        self.n_kv_heads = config["n_kv_heads"]
        self.d_head = config["d_head"]
        self.max_seq = config["context_len"]
        self.max_pages = max_pages

        self.page_elems = self.n_layers * 2 * self.n_kv_heads * PAGE_SIZE * self.d_head
        self.kv_per_seq = self.n_layers * 2 * self.n_kv_heads * self.max_seq * self.d_head

        self.pool = np.zeros((max_pages, self.page_elems), dtype=np.float16)
        self._gpu_pool = None
        self.free_pages = list(range(max_pages))
        self.page_tables = {}  # seq_idx -> list[int] (physical page IDs)

        # caches: invalidated when page_tables change
        self._gather_cache = {}   # tuple(page_table) -> jnp gather indices
        self._scatter_cache = {}  # (phys_page, position) -> (src_idx, dst_idx)

    # ── page allocation ──

    def alloc_pages(self, n):
        assert len(self.free_pages) >= n, \
            f"pool exhausted: need {n}, have {len(self.free_pages)}"
        return [self.free_pages.pop() for _ in range(n)]

    def free_seq(self, seq_idx):
        if seq_idx not in self.page_tables:
            return
        self.free_pages.extend(self.page_tables.pop(seq_idx))

    def ensure_page(self, seq_idx, position):
        """Allocate pages up to the one containing `position`."""
        needed = position // PAGE_SIZE + 1
        pages = self.page_tables.get(seq_idx, [])
        while len(pages) < needed:
            pages.append(self.alloc_pages(1)[0])
        self.page_tables[seq_idx] = pages

    # ── prefill: CPU numpy (one-time, not hot path) ──

    def store_prefill_kv(self, seq_idx, k_caches, v_caches, seq_len):
        """Store prefilled KV caches into pages. Called once per sequence."""
        n_pages = (seq_len + PAGE_SIZE - 1) // PAGE_SIZE
        pages = self.alloc_pages(n_pages)
        self.page_tables[seq_idx] = pages

        for page_idx, phys_page in enumerate(pages):
            pos_start = page_idx * PAGE_SIZE
            pos_end = min(pos_start + PAGE_SIZE, seq_len)
            n_pos = pos_end - pos_start

            page_data = np.zeros(self.page_elems, dtype=np.float16)
            for layer in range(self.n_layers):
                for kv_type, caches in enumerate([k_caches, v_caches]):
                    cache_np = np.array(caches[layer])
                    for head in range(self.n_kv_heads):
                        off = (layer * 2 * self.n_kv_heads * PAGE_SIZE * self.d_head
                               + kv_type * self.n_kv_heads * PAGE_SIZE * self.d_head
                               + head * PAGE_SIZE * self.d_head)
                        page_data[off:off + n_pos * self.d_head] = (
                            cache_np[head, pos_start:pos_end, :].reshape(-1))
            self.pool[phys_page] = page_data

    # ── GPU paging: JIT gather/scatter (hot path) ──

    def sync_to_gpu(self):
        self._gpu_pool = jnp.array(self.pool.reshape(-1), dtype=jnp.bfloat16)

    def to_contiguous_gpu(self, seq_idx):
        """Paged pool → contiguous KV buffer on GPU. ~0.07ms."""
        pages = self.page_tables[seq_idx]
        cache_key = tuple(pages)
        if cache_key not in self._gather_cache:
            self._gather_cache[cache_key] = self._build_gather_indices(pages)
        return _jit_gather(self._gpu_pool, self._gather_cache[cache_key])

    def update_page_gpu(self, seq_idx, kv_contiguous, position):
        """Write one position's K/V from contiguous output back to pool. ~0.37ms."""
        pages = self.page_tables[seq_idx]
        phys_page = pages[position // PAGE_SIZE]
        cache_key = (phys_page, position)
        if cache_key not in self._scatter_cache:
            self._scatter_cache[cache_key] = self._build_scatter_indices(
                phys_page, position)
        src_idx, dst_idx = self._scatter_cache[cache_key]
        self._gpu_pool = _jit_scatter(self._gpu_pool, kv_contiguous, src_idx, dst_idx)

    def _build_gather_indices(self, page_table):
        """Map every element in contiguous KV to its source in the pool."""
        n_l, n_kv, d_h, max_s = self.n_layers, self.n_kv_heads, self.d_head, self.max_seq

        pos = np.arange(max_s)
        pt = np.zeros(max_s // PAGE_SIZE + 1, dtype=np.int32)
        pt[:len(page_table)] = page_table
        page_bases = pt[pos // PAGE_SIZE] * self.page_elems
        pos_offsets = page_bases[:, None] + (pos % PAGE_SIZE)[:, None] * d_h + np.arange(d_h)[None, :]

        pkv = n_kv * PAGE_SIZE * d_h
        ckv = n_kv * max_s * d_h
        idx = np.empty(self.kv_per_seq, dtype=np.int32)
        for layer in range(n_l):
            for kv_type in range(2):
                for head in range(n_kv):
                    pool_off = pos_offsets + (layer * 2 * pkv + kv_type * pkv + head * PAGE_SIZE * d_h)
                    cont_start = layer * 2 * ckv + kv_type * ckv + head * max_s * d_h
                    idx[cont_start:cont_start + max_s * d_h] = pool_off.ravel()
        return jnp.array(idx, dtype=jnp.int32)

    def _build_scatter_indices(self, phys_page, position):
        """Source (contiguous) and dest (pool) indices for one position's K/V."""
        n_l, n_kv, d_h, max_s = self.n_layers, self.n_kv_heads, self.d_head, self.max_seq
        pos_in_page = position % PAGE_SIZE
        n_elems = n_l * 2 * n_kv * d_h

        src = np.empty(n_elems, dtype=np.int32)
        dst = np.empty(n_elems, dtype=np.int32)
        i = 0
        for layer in range(n_l):
            for kv_type in range(2):
                for head in range(n_kv):
                    s = (layer * 2 * n_kv * max_s * d_h + kv_type * n_kv * max_s * d_h
                         + head * max_s * d_h + position * d_h)
                    d = (phys_page * self.page_elems + layer * 2 * n_kv * PAGE_SIZE * d_h
                         + kv_type * n_kv * PAGE_SIZE * d_h + head * PAGE_SIZE * d_h
                         + pos_in_page * d_h)
                    src[i:i + d_h] = np.arange(s, s + d_h)
                    dst[i:i + d_h] = np.arange(d, d + d_h)
                    i += d_h
        return jnp.array(src), jnp.array(dst)

    # ── stats ──

    @property
    def pages_used(self):
        return self.max_pages - len(self.free_pages)

    @property
    def memory_used_mb(self):
        return self.pages_used * self.page_elems * 2 / 1e6

    @property
    def memory_allocated_mb(self):
        return self.max_pages * self.page_elems * 2 / 1e6
