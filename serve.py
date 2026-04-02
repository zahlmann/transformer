"""Variable-length batched inference server with optional paged KV + continuous batching.

Usage:
    uv run serve.py --prompts "Once upon a time" "The cat sat"
    uv run serve.py --paged --continuous --batch-size 2 --prompts "A" "B" "C" "D"
"""

import os
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

import jax
import jax.numpy as jnp
import numpy as np

from kernels.fused_decode_nlayer import prepare_decode_weights_nlayer, pack_kv_caches
from kernels.batched_decode import batched_decode_nlayer
from kernels.multi_sm_decode import multi_sm_decode_nlayer
from model import prefill_with_kv


class BatchedServer:
    """Batched decode with contiguous KV cache (one buffer per sequence)."""

    def __init__(self, params, config, batch_size):
        self.params = params
        self.config = config
        self.batch_size = batch_size
        self.vocab_size = config["vocab_size"]
        self.ctx_len = config["context_len"]
        self.n_kv_heads = config["n_kv_heads"]
        self.n_layers = config["n_layers"]
        self.d_head = config["d_head"]
        self.kv_per_seq = self.n_layers * 2 * self.n_kv_heads * self.ctx_len * self.d_head
        self.w = prepare_decode_weights_nlayer(params, config, self.vocab_size)
        self.slots = [None] * batch_size
        self._warmed_up = False

    def _warmup(self):
        if self._warmed_up:
            return
        dummy_kv = jnp.zeros((self.kv_per_seq,), dtype=jnp.bfloat16)
        tok, _, _ = multi_sm_decode_nlayer(self.w, self.config, 0, 0, dummy_kv, self.vocab_size)
        _ = int(tok)
        self._warmed_up = True

    def _prefill(self, prompt_ids):
        """Run prefill, return (first_token, kv_packed)."""
        prompt_ids = jnp.array(prompt_ids, dtype=jnp.int32)
        prompt_len = len(prompt_ids)
        assert prompt_len < self.ctx_len
        x = jnp.pad(prompt_ids, (0, self.ctx_len - prompt_len)).astype(jnp.int32)
        logits, k_caches, v_caches = prefill_with_kv(self.params, self.config, x)
        _ = logits.block_until_ready()
        first_token = int(jnp.argmax(logits[prompt_len - 1]))
        return first_token, prompt_len, pack_kv_caches(k_caches, v_caches), k_caches, v_caches

    def add_sequence(self, prompt_ids):
        """Add a sequence. Returns slot index, or -1 if full."""
        slot = next((i for i, s in enumerate(self.slots) if s is None), -1)
        if slot == -1:
            return -1
        first_token, prompt_len, kv, _, _ = self._prefill(prompt_ids)
        self.slots[slot] = {
            "token": first_token, "position": prompt_len,
            "kv": kv, "done": False, "tokens": [first_token],
        }
        self._warmup()
        return slot

    def _active_indices(self):
        return [i for i, s in enumerate(self.slots) if s is not None and not s["done"]]

    def step(self):
        """One decode step for all active sequences. Returns [(slot, tok, done)]."""
        active = self._active_indices()
        if not active:
            return []

        token_ids = jnp.array([self.slots[i]["token"] for i in active], dtype=jnp.int32)
        positions = jnp.array([self.slots[i]["position"] for i in active], dtype=jnp.int32)
        kv_batch = jnp.concatenate([self.slots[i]["kv"] for i in active])

        next_tokens, _, kv_out = batched_decode_nlayer(
            self.w, self.config, token_ids, positions,
            kv_batch, self.vocab_size, len(active))
        next_toks = next_tokens.tolist()

        results = []
        for j, slot_idx in enumerate(active):
            tok = next_toks[j]
            slot = self.slots[slot_idx]
            slot["token"] = tok
            slot["position"] += 1
            slot["kv"] = kv_out[j * self.kv_per_seq:(j + 1) * self.kv_per_seq]
            slot["tokens"].append(tok)
            slot["done"] = slot["position"] >= self.ctx_len
            results.append((slot_idx, tok, slot["done"]))
        return results

    def generate(self, max_tokens):
        for _ in range(max_tokens):
            results = self.step()
            if not results or all(done for _, _, done in results):
                break
            yield results

    def get_tokens(self, slot_idx):
        return self.slots[slot_idx]["tokens"] if self.slots[slot_idx] else []

    def remove_sequence(self, slot_idx):
        self.slots[slot_idx] = None


class PagedBatchedServer(BatchedServer):
    """Batched decode with paged KV cache (GPU gather/scatter)."""

    def __init__(self, params, config, batch_size):
        super().__init__(params, config, batch_size)
        from kernels.paged_kv import PagePool
        max_pages_per_seq = (self.ctx_len + 63) // 64
        self.page_pool = PagePool(config, batch_size * max_pages_per_seq + 16)

    def add_sequence(self, prompt_ids):
        slot = next((i for i, s in enumerate(self.slots) if s is None), -1)
        if slot == -1:
            return -1
        first_token, prompt_len, _, k_caches, v_caches = self._prefill(prompt_ids)
        self.page_pool.store_prefill_kv(slot, k_caches, v_caches, prompt_len)
        self.page_pool.sync_to_gpu()
        self.slots[slot] = {
            "token": first_token, "position": prompt_len,
            "done": False, "tokens": [first_token],
        }
        self._warmup()
        return slot

    def step(self):
        active = self._active_indices()
        if not active:
            return []

        for i in active:
            self.page_pool.ensure_page(i, self.slots[i]["position"])

        token_ids = jnp.array([self.slots[i]["token"] for i in active], dtype=jnp.int32)
        positions = jnp.array([self.slots[i]["position"] for i in active], dtype=jnp.int32)
        kv_batch = jnp.concatenate([self.page_pool.to_contiguous_gpu(i) for i in active])

        next_tokens, _, kv_out = batched_decode_nlayer(
            self.w, self.config, token_ids, positions,
            kv_batch, self.vocab_size, len(active))
        next_toks = next_tokens.tolist()

        results = []
        for j, slot_idx in enumerate(active):
            tok = next_toks[j]
            slot = self.slots[slot_idx]
            pos = slot["position"]
            kv_seq = kv_out[j * self.kv_per_seq:(j + 1) * self.kv_per_seq]
            self.page_pool.update_page_gpu(slot_idx, kv_seq, pos)
            slot["token"] = tok
            slot["position"] += 1
            slot["tokens"].append(tok)
            slot["done"] = slot["position"] >= self.ctx_len
            results.append((slot_idx, tok, slot["done"]))
        return results

    def remove_sequence(self, slot_idx):
        self.page_pool.free_seq(slot_idx)
        self.slots[slot_idx] = None

    def serve_continuous(self, prompts, max_tokens_per_seq):
        """Process N prompts through B slots. Refills slots as sequences finish.

        Returns (completed, total_tokens) where completed maps prompt_idx → tokens.
        """
        queue = list(enumerate(prompts))
        completed = {}
        slot_to_prompt = {}  # slot_idx -> (prompt_idx, n_generated)

        # fill initial batch
        while queue:
            slot = next((i for i, s in enumerate(self.slots) if s is None), -1)
            if slot == -1:
                break
            prompt_idx, prompt_ids = queue.pop(0)
            assert self.add_sequence(prompt_ids) == slot
            slot_to_prompt[slot] = (prompt_idx, 0)

        total_tokens = 0
        while self._active_indices():
            for slot_idx, tok, is_done in self.step():
                prompt_idx, n_gen = slot_to_prompt[slot_idx]
                n_gen += 1
                total_tokens += 1
                slot_to_prompt[slot_idx] = (prompt_idx, n_gen)

                if n_gen >= max_tokens_per_seq:
                    is_done = True
                    self.slots[slot_idx]["done"] = True

                if is_done:
                    completed[prompt_idx] = self.get_tokens(slot_idx)
                    self.remove_sequence(slot_idx)
                    del slot_to_prompt[slot_idx]
                    if queue:
                        next_idx, next_ids = queue.pop(0)
                        new_slot = self.add_sequence(next_ids)
                        assert new_slot >= 0
                        slot_to_prompt[new_slot] = (next_idx, 0)

        return completed, total_tokens

    @property
    def memory_stats(self):
        pp = self.page_pool
        return {
            "pages_used": pp.pages_used, "pages_total": pp.max_pages,
            "memory_used_mb": pp.memory_used_mb, "memory_pool_mb": pp.memory_allocated_mb,
        }


def main():
    import argparse, pickle, sys, time
    from data import load_bpe_vocab

    parser = argparse.ArgumentParser(description="Batched inference server")
    parser.add_argument("--prompts", nargs="+",
                        default=["Once upon a time", "The cat sat on the",
                                 "In a galaxy far", "She opened the door"])
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--weights", default="weights.pkl")
    parser.add_argument("--paged", action="store_true")
    parser.add_argument("--continuous", action="store_true")
    args = parser.parse_args()

    with open(os.path.join(os.path.dirname(__file__), args.weights), "rb") as f:
        saved = pickle.load(f)
    params = {k: jnp.array(v) for k, v in saved["params"].items()}
    config = saved["config"]
    assert "n_kv_heads" in config, "config must have n_kv_heads"

    bpe_vocab = load_bpe_vocab()
    decode_fn = bpe_vocab["decode_fn"]
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(bpe_vocab["tokenizer_path"])
    encode_fn = lambda text: tok.encode(text).ids

    if args.continuous:
        args.paged = True

    batch_size = args.batch_size or len(args.prompts)
    print(f"Model: d={config['d_model']} h={config['n_heads']} l={config['n_layers']}",
          file=sys.stderr)

    if args.paged:
        server = PagedBatchedServer(params, config, batch_size)
        print(f"Paged KV ({server.page_pool.max_pages} pages)", file=sys.stderr)
    else:
        server = BatchedServer(params, config, batch_size)

    all_prompt_ids = [encode_fn(p) for p in args.prompts]

    if args.continuous:
        t0 = time.perf_counter()
        completed, total_tokens = server.serve_continuous(
            all_prompt_ids, args.max_tokens)
        elapsed = time.perf_counter() - t0

        for idx in sorted(completed):
            text = decode_fn(completed[idx])
            print(f"\n--- Prompt {idx} ({len(completed[idx])} tokens) ---")
            print(f"{args.prompts[idx]}{text}")
        print(f"\n[{total_tokens} tok in {elapsed*1000:.0f}ms = "
              f"{total_tokens/elapsed:.0f} tok/s]", file=sys.stderr)
    else:
        for i, ids in enumerate(all_prompt_ids[:batch_size]):
            server.add_sequence(ids)

        t0 = time.perf_counter()
        total_tokens = 0
        for step_results in server.generate(args.max_tokens):
            total_tokens += len(step_results)
        elapsed = time.perf_counter() - t0

        for i in range(min(batch_size, len(args.prompts))):
            text = decode_fn(server.get_tokens(i))
            print(f"\n--- Slot {i} ({len(server.get_tokens(i))} tokens) ---")
            print(f"{args.prompts[i]}{text}")
        print(f"\n[{total_tokens} tok in {elapsed*1000:.0f}ms = "
              f"{total_tokens/elapsed:.0f} tok/s]", file=sys.stderr)

    if args.paged:
        s = server.memory_stats
        print(f"[Pages: {s['pages_used']}/{s['pages_total']}, "
              f"{s['memory_used_mb']:.1f}/{s['memory_pool_mb']:.1f} MB]", file=sys.stderr)


if __name__ == "__main__":
    main()
