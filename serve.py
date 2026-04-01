"""Variable-length batched inference server.

Manages a batch of sequences at different generation stages using the
batched multi-SM decode kernel (which already supports per-sequence positions).

Usage:
    server = BatchedServer(params, config, batch_size=4)
    server.add_sequence(prompt_ids_1)
    server.add_sequence(prompt_ids_2)

    for step_results in server.generate(max_tokens=128):
        for seq_id, token_id, is_done in step_results:
            if token_id is not None:
                print(decode_fn([token_id]), end='')
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
    """Manages variable-length batched inference.

    Sequences can be at different positions. Each step advances all active
    sequences by one token. Sequences stop when they hit the EOS token or
    max context length.

    Uses the non-persistent batched decode kernel which supports per-sequence
    positions via positions_ptr.
    """

    def __init__(self, params, config, batch_size=4, eos_token=None):
        """Initialize the server.

        Args:
            params: model parameters (JAX arrays)
            config: model config dict
            batch_size: maximum batch size
            eos_token: token ID that signals end of generation (None = no stopping)
        """
        self.params = params
        self.config = config
        self.batch_size = batch_size
        self.eos_token = eos_token
        self.vocab_size = config["vocab_size"]
        self.ctx_len = config["context_len"]
        self.d_head = config["d_head"]
        self.n_kv_heads = config.get("n_kv_heads", config["n_heads"])
        self.n_layers = config["n_layers"]
        self.kv_per_seq = self.n_layers * 2 * self.n_kv_heads * self.ctx_len * self.d_head

        self.w = prepare_decode_weights_nlayer(params, config, self.vocab_size)

        # Per-sequence state
        self.slots = [None] * batch_size  # None = empty slot
        self._warmup_done = False

    def _warmup(self):
        """Run one decode step to trigger JIT compilation."""
        if self._warmup_done:
            return
        dummy_tok = jnp.zeros((1,), dtype=jnp.int32)
        dummy_pos = jnp.zeros((1,), dtype=jnp.int32)
        dummy_kv = jnp.zeros((self.kv_per_seq,), dtype=jnp.bfloat16)
        # Single-sequence warmup via multi_sm_decode
        _t, _, _ = multi_sm_decode_nlayer(
            self.w, self.config, dummy_tok[0], 0, dummy_kv, self.vocab_size)
        _ = int(_t)
        self._warmup_done = True

    def add_sequence(self, prompt_ids):
        """Add a sequence to the batch.

        Runs prefill immediately and stores KV cache.

        Args:
            prompt_ids: list or array of prompt token IDs

        Returns:
            int: slot index, or -1 if batch is full
        """
        # Find empty slot
        slot = -1
        for i, s in enumerate(self.slots):
            if s is None:
                slot = i
                break
        if slot == -1:
            return -1

        prompt_ids = jnp.array(prompt_ids, dtype=jnp.int32)
        prompt_len = len(prompt_ids)

        if prompt_len >= self.ctx_len:
            prompt_ids = prompt_ids[:self.ctx_len - 1]
            prompt_len = len(prompt_ids)

        # Prefill
        x = jnp.pad(prompt_ids, (0, self.ctx_len - prompt_len)).astype(jnp.int32)
        logits, k_caches, v_caches = prefill_with_kv(self.params, self.config, x)
        _ = logits.block_until_ready()

        kv_packed = pack_kv_caches(k_caches, v_caches)
        first_token = int(jnp.argmax(logits[prompt_len - 1]))

        self.slots[slot] = {
            "token": first_token,
            "position": prompt_len,  # next position to use
            "kv": kv_packed,
            "done": False,
            "tokens": [first_token],
        }

        self._warmup()
        return slot

    def _active_indices(self):
        """Return indices of active (non-done, non-empty) slots."""
        return [i for i, s in enumerate(self.slots)
                if s is not None and not s["done"]]

    def step(self):
        """Run one decode step for all active sequences.

        Returns:
            list of (slot_index, token_id, is_done) for each active sequence
        """
        active = self._active_indices()
        if not active:
            return []

        n_active = len(active)

        # Build batched inputs
        token_ids = jnp.array([self.slots[i]["token"] for i in active], dtype=jnp.int32)
        positions = jnp.array([self.slots[i]["position"] for i in active], dtype=jnp.int32)
        kv_batch = jnp.concatenate([self.slots[i]["kv"] for i in active])

        # Run batched decode
        next_tokens, _, kv_out = batched_decode_nlayer(
            self.w, self.config, token_ids, positions,
            kv_batch, self.vocab_size, n_active)

        # Sync results
        next_toks = next_tokens.tolist()

        results = []
        for j, slot_idx in enumerate(active):
            tok = next_toks[j]
            slot = self.slots[slot_idx]

            # Update slot state
            slot["token"] = tok
            slot["position"] += 1
            slot["kv"] = kv_out[j * self.kv_per_seq:(j + 1) * self.kv_per_seq]
            slot["tokens"].append(tok)

            # Check stopping conditions
            is_done = False
            if self.eos_token is not None and tok == self.eos_token:
                is_done = True
            if slot["position"] >= self.ctx_len:
                is_done = True
            slot["done"] = is_done

            results.append((slot_idx, tok, is_done))

        return results

    def generate(self, max_tokens=128):
        """Generate tokens for all active sequences, yielding after each step.

        Yields:
            list of (slot_index, token_id, is_done) after each decode step
        """
        for _ in range(max_tokens):
            results = self.step()
            if not results:
                break
            yield results
            if all(r[2] for r in results):
                break

    def get_tokens(self, slot_idx):
        """Get all generated tokens for a slot."""
        if self.slots[slot_idx] is None:
            return []
        return self.slots[slot_idx]["tokens"]

    def remove_sequence(self, slot_idx):
        """Remove a sequence and free its slot."""
        self.slots[slot_idx] = None


def main():
    import argparse
    import pickle
    import sys
    import time

    from data import load_bpe_vocab

    parser = argparse.ArgumentParser(description="Variable-length batched inference")
    parser.add_argument("--prompts", nargs="+",
                        default=["Once upon a time", "The cat sat on the",
                                 "In a galaxy far", "She opened the door"],
                        help="Prompts to generate from (one per batch slot)")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--weights", type=str, default="weights.pkl")
    args = parser.parse_args()

    # Load model
    with open(os.path.join(os.path.dirname(__file__), args.weights), "rb") as f:
        saved = pickle.load(f)
    params = {k: jnp.array(v) for k, v in saved["params"].items()}
    config = saved["config"]

    # Load tokenizer
    bpe_vocab = load_bpe_vocab()
    decode_fn = bpe_vocab["decode_fn"]
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(bpe_vocab["tokenizer_path"])
    encode_fn = lambda text: tok.encode(text).ids

    batch_size = len(args.prompts)
    print(f"Model: d={config['d_model']} h={config['n_heads']} l={config['n_layers']}",
          file=sys.stderr)
    print(f"Batch size: {batch_size}", file=sys.stderr)
    print(file=sys.stderr)

    server = BatchedServer(params, config, batch_size=batch_size)

    # Add all prompts
    for i, prompt in enumerate(args.prompts):
        prompt_ids = encode_fn(prompt)
        slot = server.add_sequence(prompt_ids)
        print(f"[Slot {slot}] prompt ({len(prompt_ids)} tokens): {prompt}", file=sys.stderr)

    print(file=sys.stderr)

    # Generate
    t0 = time.perf_counter()
    total_tokens = 0
    for step_results in server.generate(max_tokens=args.max_tokens):
        total_tokens += len(step_results)

    elapsed = time.perf_counter() - t0
    tok_per_s = total_tokens / elapsed

    # Print results
    for i in range(batch_size):
        tokens = server.get_tokens(i)
        text = decode_fn(tokens)
        prompt = args.prompts[i]
        print(f"\n--- Slot {i} ({len(tokens)} tokens) ---")
        print(f"{prompt}{text}")

    print(f"\n[{total_tokens} tokens in {elapsed*1000:.0f}ms = {tok_per_s:.0f} tok/s"
          f" ({tok_per_s/batch_size:.0f} per seq)]", file=sys.stderr)


if __name__ == "__main__":
    main()
