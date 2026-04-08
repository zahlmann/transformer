# Training a Transformer from Scratch — Every Line Explained

This document explains the complete training pipeline — data preparation, model
architecture, and the training loop — from first principles. It assumes you know
Python but nothing about machine learning, JAX, or transformers.

Read it straight through, or jump to any section — each one stands alone.

---

## Table of Contents

1. [What Are We Building?](#1-what-are-we-building)
2. [The Libraries](#2-the-libraries)
3. [Preparing the Training Data](#3-preparing-the-training-data)
4. [Loading Data for Training](#4-loading-data-for-training)
5. [The Model Architecture](#5-the-model-architecture)
6. [The Training Loop](#6-the-training-loop)
7. [Glossary](#7-glossary)

---

## 1. What Are We Building?

We are training a **language model** — a program that predicts the next word (more
precisely, the next *token*) given some preceding text. If you give it "The cat sat
on the", it should predict "mat" (or "floor", or "couch") with high probability.

The model is a **decoder-only transformer** with 306 million learned numbers
("parameters"). It reads sequences of 512 tokens and, for every position in the
sequence, predicts what comes next. During training, we show it billions of tokens
from books, code, math, and encyclopedias, and adjust those 306 million parameters
so its predictions get better.

The full pipeline has three stages:

1. **Data preparation** (`prepare_data_v2.py` / `prepare_data_v3.py`): Download
   text from the internet, convert it to numbers (tokenize), shuffle it, and save
   it to disk.
2. **Model definition** (`model.py`): Define the mathematical operations the model
   performs — attention, normalization, feed-forward networks.
3. **Training** (`train.py` + `data.py`): Load the data, run the model forward,
   measure how wrong the predictions are, compute how to adjust each parameter to
   make predictions less wrong, and repeat billions of times.

---

## 2. The Libraries

Before diving into code, here is what each library does and why we use it.

### NumPy (`import numpy as np`)

NumPy is a Python library for working with large arrays of numbers. Instead of
using a Python list of 7 billion integers, you use a NumPy array — it stores the
numbers in a compact block of memory and performs operations on them in fast C code
instead of slow Python loops.

Key operations we use:

```python
np.array([1, 2, 3])           # create array from list
arr.reshape(100, 512)          # reshape 51200 elements into 100 rows of 512
np.concatenate([a, b])         # join arrays end-to-end
np.memmap("file.bin", ...)     # map a file into memory as if it were an array
                                # (reads from disk on demand, doesn't load into RAM)
np.save("file.npy", arr)      # save array to file
np.load("file.npy")           # load array from file
arr.tofile(f)                  # write raw bytes to an open file
```

### JAX (`import jax`, `import jax.numpy as jnp`)

JAX is a machine learning framework by Google. It provides:

1. **GPU arrays**: `jnp.array(...)` creates an array that lives on the GPU, not
   the CPU. GPU operations are 10-100x faster for the matrix multiplications that
   transformers need.
2. **Automatic differentiation**: `jax.value_and_grad(f)(x)` computes both `f(x)`
   and the derivative of `f` with respect to `x`. This is how we figure out which
   direction to adjust each parameter — the derivative (gradient) tells us.
3. **JIT compilation**: `@jax.jit` compiles a Python function into optimized GPU
   machine code the first time it runs. Subsequent calls skip Python entirely.
4. **vmap**: `jax.vmap(f)` takes a function that works on one example and
   automatically makes it work on a batch of examples in parallel.

JAX uses **pure functions** — a function's output depends only on its inputs, with
no hidden state. Model parameters are passed as an explicit argument, not stored
inside an object.

Key operations:

```python
jnp.array(x)                  # create GPU array
x @ y                         # matrix multiplication (same as np.dot(x, y))
jax.nn.silu(x)                # SiLU activation: x * sigmoid(x)
jax.nn.log_softmax(x)         # numerically stable log(softmax(x))
jax.random.key(42)            # create a random number generator seed
jax.random.split(key)         # split one seed into two independent seeds
jax.random.normal(key, shape) # generate random numbers from a normal distribution
jax.tree.map(f, tree)         # apply f to every leaf in a nested dict/list
jax.device_put(x)             # move data to GPU
jax.checkpoint(f)             # recompute f's intermediates during backward pass
                               # instead of storing them (saves memory, costs time)
```

**Number types (dtypes)**:

- `float32` (f32): 32-bit float, standard precision. Used for parameter updates.
- `bfloat16` (bf16): 16-bit float, half the memory, slightly less precise. Used
  for forward pass to save GPU memory and run faster.
- `int32`: 32-bit integer. Used for token IDs.

### Optax (`import optax`)

Optax is a JAX library for **optimizers** — algorithms that decide *how much* to
adjust each parameter given its gradient. We use AdamW, which:

- Keeps a running average of each gradient (momentum) to smooth out noise
- Keeps a running average of squared gradients to adapt the learning rate per-parameter
- Adds weight decay (a small penalty that shrinks parameters toward zero to prevent overfitting)

```python
optimizer = optax.adamw(learning_rate, weight_decay=0.1)
opt_state = optimizer.init(params)  # initialize optimizer state (momentum, etc.)
updates, opt_state = optimizer.update(grads, opt_state, params)  # compute updates
params = optax.apply_updates(params, updates)  # apply updates to params
```

Optax also provides **learning rate schedules**:

```python
optax.linear_schedule(0.0, 0.0003, 200)      # ramp from 0 to 0.0003 over 200 steps
optax.cosine_decay_schedule(0.0003, 10000)    # decay 0.0003 → ~0 in a cosine curve
optax.join_schedules([sched1, sched2], [200]) # use sched1 for first 200 steps, then sched2
```

### HuggingFace `tokenizers`

A tokenizer converts text to numbers and back. We use a **BPE (Byte Pair Encoding)**
tokenizer trained on our corpus. It learns the most common character sequences and
assigns each one an ID number. With a vocabulary of 32,000 tokens:

```python
from tokenizers import Tokenizer
tok = Tokenizer.from_file("tokenizer_32000.json")
tok.encode("Hello world").ids   # → [15496, 995]     (two token IDs)
tok.decode([15496, 995])        # → "Hello world"     (back to text)
tok.encode_batch(["Hello", "World"])  # encode many strings at once (faster)
```

### HuggingFace `datasets`

Used only in data preparation. Streams text datasets from the internet:

```python
from datasets import load_dataset
ds = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
for doc in ds:      # iterate one document at a time (doesn't download everything)
    text = doc["text"]
```

---

## 3. Preparing the Training Data

**Files**: `prepare_data_v2.py`, `prepare_data_v3.py`

These scripts download text, convert it to token IDs, and save a single shuffled
binary file. The v3 script targets 50 billion tokens; v2 targets ~10 billion.
Their structure is nearly identical, so we explain them together.

### 3.1 Constants and Configuration

```python
DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw"           # downloaded text files go here
TOKEN_DIR = DATA_DIR / "tokens_v2"   # tokenized output goes here
VOCAB_SIZE = 32000                   # our tokenizer has 32K tokens
VAL_FRACTION = 0.005                 # 0.5% of data reserved for validation
EOS_TOKEN_ID = 1                     # special token marking end of document
```

The **data mix** is specified as a dict mapping source names to target token counts:

```python
SOURCES = {
    "fineweb_edu":  {"tokens": 3_500_000_000, "chars_per_tok": 3.5, "min_len": 100},
    "wikipedia":    {"tokens": 800_000_000,   "chars_per_tok": 3.5, "min_len": 200},
    ...
}
```

- `tokens`: how many tokens we want from this source
- `chars_per_tok`: rough ratio of characters to tokens (used to estimate how many
  characters to download before we have enough tokens)
- `min_len`: skip documents shorter than this many characters

### 3.2 Downloading

Each source is downloaded with `_download_source()` or `_download_multilang()`
(for code datasets that span multiple programming languages).

The download process:

1. **Check if already done**: If the output file exists, skip downloading.
2. **Dedup existing data**: If we have leftover files from previous versions,
   read them and record the first 200 characters of each document in a `seen` set.
   This prevents duplicate documents.
3. **Stream from HuggingFace**: Iterate through the dataset, skip documents that
   are too short or are duplicates, write each one as a JSON line to the output file.
4. **Stop when we have enough**: Once total characters reaches `tokens * chars_per_tok`,
   we have approximately enough data.

The FineWeb-Edu source applies an extra quality filter — each document has a
quality score, and we only keep documents scoring >= 3 (or >= 4 for the
high-quality annealing subset).

### 3.3 Tokenizer Training (v2 only)

The v2 script trains the BPE tokenizer:

```python
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=["<pad>", "<eos>"],
    min_frequency=2,
    show_progress=True,
)
tokenizer.train_from_iterator(texts, trainer=trainer)
```

BPE works by:
1. Starting with individual bytes as the vocabulary
2. Finding the most frequent pair of adjacent tokens in the training data
3. Merging that pair into a new token
4. Repeating until the vocabulary reaches 32,000 tokens

`ByteLevel` means the base vocabulary is all 256 byte values, so it can represent
any text (no "unknown token" problem). `<pad>` (ID 0) and `<eos>` (ID 1) are
special tokens — `<eos>` marks the boundary between documents.

The v3 script reuses the tokenizer trained by v2.

### 3.4 Tokenization

`_tokenize_source()` converts each raw text file into token IDs:

```python
with open(cache_bin, "wb") as fout, open(raw_path) as fin:
    for line in fin:
        doc = json.loads(line)
        text = doc.get("text", "")
        batch_texts.append(text)

        if batch_chars >= 50_000_000:       # process in 50M-character batches
            encodings = tok.encode_batch(batch_texts)
            for enc in encodings:
                arr = np.array(enc.ids + [EOS_TOKEN_ID], dtype=np.int32)
                arr.tofile(fout)            # write directly to disk
```

Key points:
- Documents are processed in batches of ~50 million characters for speed
  (`encode_batch` is much faster than encoding one at a time).
- After each document's tokens, we append `EOS_TOKEN_ID` (1). This tells the
  model "this document is over, the next tokens are from a different document."
- Tokens are written as raw 32-bit integers directly to a binary file (4 bytes
  per token). This is much faster and more compact than text formats.

### 3.5 Combining and Shuffling

After tokenization, we have separate `.bin` files for each source. These get
combined into one big `train.bin` plus a small `val.npy`:

1. **Split validation data**: Take the first N tokens from each source as
   validation data (used to check model quality, never trained on).
2. **Write training data**: Concatenate the remaining tokens from all sources
   into a single `train.bin`.
3. **Shuffle**: The data is currently ordered by source (all FineWeb, then all
   code, then all math...). We need to shuffle so the model sees a mix during
   training.

The shuffle works on 512-token chunks:

```python
src = np.memmap(train_bin, dtype=np.int32, mode="r")  # map file to memory
n_chunks = len(src) // 512                              # how many 512-token chunks
perm = np.random.default_rng(42).permutation(n_chunks)  # random ordering

dst = np.memmap(shuffled_bin, dtype=np.int32, mode="w+", shape=(usable,))
for i in range(0, n_chunks, batch):
    for j, ci in enumerate(perm[i:end]):
        dst[(i+j)*512:(i+j+1)*512] = src[ci*512:(ci+1)*512]
```

`np.memmap` maps a file into virtual memory — the OS loads pages from disk on
demand rather than reading the entire file into RAM. This lets us shuffle a 30 GB
file using only a few hundred MB of RAM.

`default_rng(42)` creates a random number generator with seed 42, making the
shuffle reproducible.

4. **Save metadata**: A JSON file records vocab size, token counts per source,
   and the path to the tokenizer.

### 3.6 Final Output

```
data/tokens_v2/
  train.bin          # flat binary: 7.85 billion int32 tokens, shuffled
  val.npy            # numpy array: ~39 million validation tokens
  metadata.json      # vocab size, source breakdown, tokenizer path
  fineweb_edu.npz    # per-source cache (can regenerate train.bin from these)
  ...
```

---

## 4. Loading Data for Training

**File**: `data.py`

### 4.1 `load_data(context_len, data_dir=None)`

This function loads the prepared dataset and makes it ready for training.

```python
token_dir = data_dir or os.path.join(DATA_DIR, "tokens_v2")
train_bin = os.path.join(token_dir, "train.bin")
val_npy = os.path.join(token_dir, "val.npy")
meta_path = os.path.join(token_dir, "metadata.json")
```

Paths to the three files. `data_dir` can point to v3 data instead of v2.

```python
assert os.path.exists(train_bin), f"Missing {train_bin}. Run: uv run prepare_data_v3.py"
assert os.path.exists(meta_path), f"Missing {meta_path}"
```

Crash immediately with a helpful message if data hasn't been prepared yet.

```python
train_mmap = np.memmap(train_bin, dtype=np.int32, mode="r")
```

**Memory-mapped loading**: This does NOT read 30 GB into RAM. It creates a
virtual array backed by the file on disk. When training reads a batch of tokens,
only those specific pages get loaded from disk. This is how we train on a dataset
larger than available RAM.

```python
val_data = np.load(val_npy)
n_val = (len(val_data) - 1) // context_len
val_data = val_data[: n_val * context_len + 1]
val_x = val_data[:-1].reshape(n_val, context_len)
val_y = val_data[1:].reshape(n_val, context_len)
```

**Validation data preparation**: The validation data IS loaded fully into RAM
(it's small — ~39M tokens = ~150 MB). We slice it into input/target pairs:

Suppose `context_len = 4` and `val_data = [10, 20, 30, 40, 50, 60, 70, 80, 90]`.
Then `n_val = (9-1) // 4 = 2` sequences. We trim to `9 = 2*4+1 = 9` tokens:

```
val_data = [10, 20, 30, 40, 50, 60, 70, 80, 90]
val_x    = [[10, 20, 30, 40],    ← input: positions 0-3
             [50, 60, 70, 80]]   ← input: positions 4-7
val_y    = [[20, 30, 40, 50],    ← target: positions 1-4  (shifted by 1)
             [60, 70, 80, 90]]   ← target: positions 5-8
```

Each position's target is the next token. The model's job is to predict `val_y`
from `val_x`.

```python
return {
    "train_tokens": train_mmap,   # the full token stream (memory-mapped)
    "val_x": val_x,               # validation inputs
    "val_y": val_y,               # validation targets
    "vocab_size": vocab_size,      # 32000
}
```

Training data is NOT pre-sliced into sequences — that happens on the fly during
training to support curriculum learning (variable context lengths).

### 4.2 `load_bpe_vocab()`

Used by inference scripts (`generate.py`, `serve.py`) to convert token IDs back
to text. During training, `load_data` saves the tokenizer path to a pickle file;
this function reads it back and creates a decode function.

```python
tok = Tokenizer.from_file(saved["tokenizer_path"])
return {
    "tokenizer_path": saved["tokenizer_path"],
    "vocab_size": saved["vocab_size"],
    "decode_fn": lambda ids: tok.decode(list(int(i) for i in ids)),
}
```

The `decode_fn` takes a list of token IDs and returns the text they represent.

---

## 5. The Model Architecture

**File**: `model.py`

This is the core — the mathematical function that maps input tokens to predictions.

### 5.1 High-Level Structure

A decoder-only transformer processes a sequence of tokens in three stages:

1. **Embed**: Convert each token ID to a vector of numbers (a "representation").
2. **Transform**: Pass the representations through 24 identical layers. Each layer
   has an attention mechanism (lets each position look at previous positions) and
   a feed-forward network (processes each position independently).
3. **Predict**: Convert the final representations back to a probability
   distribution over all 32,000 tokens.

```
Input tokens:  [The, cat, sat, on]
       ↓
   Embedding:  each token → a vector of 1024 numbers
       ↓
   Layer 0:    attention + feed-forward
   Layer 1:    attention + feed-forward
   ...
   Layer 23:   attention + feed-forward
       ↓
   Output:     each position → probabilities for all 32K tokens
               position 3 should give high probability to "the"
```

### 5.2 Initialization (`init_transformer`)

```python
def init_transformer(key, vocab_size, d_model=64, n_heads=2, n_layers=1,
                     context_len=128, n_kv_heads=2, n_mtp_heads=0):
```

This creates all the model's parameters as random numbers. Parameters are the
numbers the model learns during training. Our model has ~306 million of them.

**Arguments**:
- `key`: a random seed (JAX requires explicit randomness — no hidden global state)
- `vocab_size`: 32,000 (how many distinct tokens exist)
- `d_model`: 1024 (the size of each token's representation vector)
- `n_heads`: 16 (how many parallel attention "heads" — explained below)
- `n_layers`: 24 (how many transformer layers to stack)
- `context_len`: 512 (how many tokens the model can see at once)
- `n_kv_heads`: 4 (how many Key/Value heads — a memory optimization called GQA)
- `n_mtp_heads`: 0 (multi-token prediction heads — not used in default training)

```python
d_head = d_model // n_heads   # 1024 // 16 = 64 dimensions per attention head
d_ff = _swiglu_d_ff(d_model)  # 2816 (feed-forward hidden size)
```

**d_head**: Each attention head works on a 64-dimensional slice of the full 1024-dimensional
representation. 16 heads, each 64-dimensional, together cover all 1024 dimensions.

**d_ff**: The feed-forward network's hidden layer size. The function `_swiglu_d_ff`
computes this to match the parameter count of a standard FFN:

```python
def _swiglu_d_ff(d_model):
    # Standard FFN has 2 matrices: (d, 4d) and (4d, d) → 8d² params
    # SwiGLU FFN has 3 matrices: (d, d_ff) × 3 → 3·d·d_ff params
    # To match: d_ff = 8d/3, rounded up to multiple of 128 for GPU efficiency
    return ((8 * d_model // 3 + 127) // 128) * 128
```

With `d_model=1024`: `8 * 1024 / 3 ≈ 2731`, rounded up to `2816`.

#### Token Embedding

```python
params["token_emb"] = jax.random.normal(k, (vocab_size, d_model)) * 0.02
```

A matrix of shape `(32000, 1024)` — one row per token in the vocabulary. Each row
is a 1024-dimensional vector representing that token. Initially random; the model
learns meaningful representations during training. The `* 0.02` makes the initial
values small (standard deviation 0.02), which helps training start smoothly.

This same matrix is reused at the end to convert representations back to
vocabulary probabilities (**tied embeddings**). This works because if token A's
embedding is close to position P's final representation, then the dot product
(similarity) between them will be high, giving token A high probability at
position P.

#### Per-Layer Parameters

For each of the 24 layers:

```python
params[f"{prefix}.ln1.scale"] = jnp.ones(d_model)           # (1024,)
```

**RMSNorm scale**: A vector of 1024 ones (initially). Used to normalize and
rescale the representations before attention. Explained in detail below.

```python
params[f"{prefix}.attn.q"] = jax.random.normal(k, (d_model, d_model)) * (d_model ** -0.5)
params[f"{prefix}.attn.k"] = jax.random.normal(k, (d_model, d_kv)) * (d_model ** -0.5)
params[f"{prefix}.attn.v"] = jax.random.normal(k, (d_model, d_kv)) * (d_model ** -0.5)
params[f"{prefix}.attn.o"] = jax.random.normal(k, (d_model, d_model)) * (d_model ** -0.5)
```

**Attention projection matrices**: Four weight matrices per layer.

- `q` (query): `(1024, 1024)` — projects the representation to "what am I looking for?"
- `k` (key): `(1024, 256)` — projects to "what do I contain?" (smaller because of GQA)
- `v` (value): `(1024, 256)` — projects to "what information do I carry?"
- `o` (output): `(1024, 1024)` — recombines the attention results

Why is K/V only 256 wide? That's `n_kv_heads * d_head = 4 * 64 = 256`. This is
**Grouped Query Attention (GQA)** — instead of 16 separate K/V heads, we use only
4 and share each one across 4 query heads. This cuts memory by 4x with minimal
quality loss.

The `* (d_model ** -0.5)` scaling (= `* 0.03125`) initializes weights small
enough that initial attention scores don't blow up.

```python
params[f"{prefix}.ffn.gate"] = jax.random.normal(k, (d_model, d_ff)) * (d_model ** -0.5)
params[f"{prefix}.ffn.up"]   = jax.random.normal(k, (d_model, d_ff)) * (d_model ** -0.5)
params[f"{prefix}.ffn.down"] = jax.random.normal(k, (d_ff, d_model)) * (d_ff ** -0.5)
```

**SwiGLU FFN matrices**: Three weight matrices per layer (explained below).

- `gate`: `(1024, 2816)` — the "gate" projection
- `up`: `(1024, 2816)` — the "up" projection
- `down`: `(2816, 1024)` — projects back to model dimension

```python
params["ln_final.scale"] = jnp.ones(d_model)
```

**Final normalization**: One more RMSNorm applied after all 24 layers, before the
output projection.

### 5.3 RMSNorm (`rms_norm`)

**Problem**: As representations pass through many layers, their magnitudes can
grow or shrink uncontrollably, making training unstable.

**Solution**: Normalize each representation to have a consistent magnitude.

```python
def rms_norm(x, scale, eps=1e-5):
    x_f32 = x.astype(jnp.float32)
    rms = jnp.sqrt(jnp.mean(x_f32 ** 2, axis=-1, keepdims=True) + eps).astype(x.dtype)
    return scale * (x / rms)
```

Line by line:

1. `x_f32 = x.astype(jnp.float32)` — Convert to 32-bit float for precision.
   The input is bfloat16 (16-bit) during training, which doesn't have enough
   precision for computing the mean of squared values.

2. `jnp.mean(x_f32 ** 2, axis=-1, keepdims=True)` — For each position, compute
   the mean of the squared values across all 1024 dimensions. If the vector is
   `[0.5, -1.0, 0.3, ...]`, this computes `mean(0.25 + 1.0 + 0.09 + ...)`.

3. `jnp.sqrt(... + eps)` — Take the square root. The `eps=1e-5` prevents division
   by zero if all values happen to be zero.

4. `scale * (x / rms)` — Divide by the RMS to normalize (magnitude → ~1.0), then
   multiply by a learned scale vector. This lets the model learn what magnitude
   each dimension should have.

**Why RMS, not LayerNorm?** Standard LayerNorm subtracts the mean then divides by
std. RMSNorm skips the mean subtraction — it's 10-15% faster and works just as well.

### 5.4 Rotary Position Embeddings (RoPE)

**Problem**: The model needs to know the *position* of each token ("cat" at
position 2 should behave differently from "cat" at position 50), but the attention
mechanism itself is position-agnostic.

**Solution**: Rotate each token's query and key vectors by an angle that depends
on their position. Tokens at similar positions get similar rotations, so their dot
products (attention scores) are higher.

#### Building the rotation table

```python
def precompute_rope_table(context_len, d_head, base=10000.0):
    half = d_head // 2                    # 32 (we rotate pairs of dimensions)
    freqs = base ** (-jnp.arange(0, half, dtype=jnp.float32) * 2.0 / d_head)
    positions = jnp.arange(context_len, dtype=jnp.float32)
    angles = positions[:, None] * freqs[None, :]
    return jnp.cos(angles), jnp.sin(angles)
```

`freqs`: A vector of 32 frequencies, from fast-rotating (dimension 0) to
slow-rotating (dimension 31). `base=10000` controls the spread.

```
freqs[0]  = 10000^0      = 1.0         # rotates once per token
freqs[1]  = 10000^(-1/32) ≈ 0.72       # rotates ~0.72 times per token
...
freqs[31] = 10000^(-31/32) ≈ 0.001     # rotates ~0.001 times per token
```

`angles`: A `(512, 32)` table where `angles[pos, dim]` = `pos * freqs[dim]`.
Position 0 has angle 0 everywhere. Position 100 has large angles for
fast-rotating dimensions and small angles for slow-rotating ones.

The cos and sin of these angles are used to rotate vectors.

#### Applying the rotation

```python
def apply_rope(x, cos, sin):
    half = x.shape[-1] // 2
    x_even, x_odd = x[..., :half], x[..., half:]
    return jnp.concatenate([
        x_even * cos - x_odd * sin,
        x_even * sin + x_odd * cos,
    ], axis=-1)
```

This is a 2D rotation applied to pairs of dimensions. For each pair `(even, odd)`:

$$\begin{pmatrix} \text{even'} \\ \text{odd'} \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} \text{even} \\ \text{odd} \end{pmatrix}$$

The key insight: when we compute the dot product of a rotated query at position
$i$ with a rotated key at position $j$, the result depends only on the *relative*
distance $i - j$, not the absolute positions. This gives the model position
awareness while being translation-invariant.

### 5.5 Causal Attention

This is the core mechanism that lets each token "attend to" (look at, gather
information from) all previous tokens.

```python
def causal_attention(x, wq, wk, wv, wo, n_heads, n_kv_heads, cos, sin):
    seq_len, d_model = x.shape
    d_head = d_model // n_heads
```

`x` has shape `(512, 1024)` — 512 positions, each with a 1024-dimensional vector.

#### Step 1: Project to Q, K, V

```python
    q = (x @ wq).reshape(seq_len, n_heads, d_head)     # (512, 16, 64)
    k = (x @ wk).reshape(seq_len, n_kv_heads, d_head)  # (512, 4, 64)
    v = (x @ wv).reshape(seq_len, n_kv_heads, d_head)  # (512, 4, 64)
```

`x @ wq` multiplies each 1024-dimensional representation by the `(1024, 1024)`
weight matrix to produce a 1024-dimensional query vector. Then `.reshape(512, 16,
64)` splits it into 16 heads of 64 dimensions each.

K and V only have 4 heads (GQA): each of the 4 K/V heads is shared by 4 query heads.

**Intuition**: Q asks "what am I looking for?", K says "what do I contain?", V says
"here is my information." The attention score between position $i$ and position $j$
is the dot product $Q_i \cdot K_j$ — high when position $i$'s query matches position
$j$'s key.

#### Step 2: Apply RoPE

```python
    q = apply_rope(q.transpose(1, 0, 2), cos_seq[None, :, :], sin_seq[None, :, :]).transpose(1, 0, 2)
    k = apply_rope(k.transpose(1, 0, 2), cos_seq[None, :, :], sin_seq[None, :, :]).transpose(1, 0, 2)
```

Rotate Q and K by position-dependent angles (as explained above). V is NOT rotated —
only the attention scores need position information, not the values being gathered.

#### Step 3: Compute attention

```python
    out = jax.nn.dot_product_attention(q, k, v, is_causal=True, implementation='cudnn')
```

This single call does all of the following:

1. **Attention scores**: For each query head $h$ and position $i$, compute the dot
   product with every key at positions $j \le i$ (causal: can't look at future tokens):

   $$\text{score}_{i,j} = \frac{Q_i^h \cdot K_j^h}{\sqrt{d_\text{head}}}$$

   The $\sqrt{64} = 8$ scaling prevents dot products from getting too large.

2. **Masking**: Set scores where $j > i$ to $-\infty$. After softmax, these
   become 0 — the model cannot cheat by looking at future tokens.

3. **Softmax**: Convert scores to probabilities that sum to 1 across all previous
   positions:

   $$\alpha_{i,j} = \frac{e^{\text{score}_{i,j}}}{\sum_{k \le i} e^{\text{score}_{i,k}}}$$

4. **Weighted sum**: Compute the output as the weighted sum of value vectors:

   $$\text{out}_i^h = \sum_{j \le i} \alpha_{i,j} \cdot V_j^h$$

`is_causal=True` enables the future-masking. `implementation='cudnn'` uses
NVIDIA's FlashAttention implementation, which is much faster because it avoids
materializing the full `(512, 512)` attention matrix.

**GQA detail**: When there are 16 query heads but only 4 KV heads, query heads
0-3 share KV head 0, query heads 4-7 share KV head 1, etc. `dot_product_attention`
handles this automatically.

#### Step 4: Output projection

```python
    return out.reshape(seq_len, d_model) @ wo
```

Concatenate all 16 heads back into a 1024-dimensional vector (`.reshape(512, 1024)`),
then multiply by the output weight matrix `wo` to mix information across heads.

### 5.6 The Feed-Forward Network (SwiGLU)

After attention, each position's representation is processed independently through
a two-layer neural network with a gating mechanism:

```python
h_ff = (jax.nn.silu(h_norm2 @ ffn_gate) * (h_norm2 @ ffn_up)) @ ffn_down
```

Breaking this down:

1. `h_norm2 @ ffn_gate` — Multiply the 1024-dim input by the gate matrix to get
   a 2816-dim vector.
2. `jax.nn.silu(...)` — Apply the SiLU activation function: $\text{SiLU}(x) = x \cdot \sigma(x)$
   where $\sigma$ is the sigmoid function. This introduces non-linearity — without
   it, stacking layers would just be one big matrix multiplication and couldn't
   learn complex patterns.
3. `h_norm2 @ ffn_up` — A second 1024→2816 projection (no activation).
4. `... * ...` — Element-wise multiplication of the gated and ungated projections.
   This is the "gating" — the SiLU output controls how much of each dimension
   passes through.
5. `... @ ffn_down` — Project back from 2816 to 1024 dimensions.

**Why SwiGLU instead of a standard FFN?** A standard FFN does
`relu(x @ W1) @ W2` — one up-projection plus an activation. SwiGLU uses two
up-projections with a multiplicative gate, which consistently produces better
model quality per parameter.

### 5.7 A Full Transformer Layer (`_attn_layer`)

Each of the 24 layers does:

```python
def _attn_layer(h, ln1_s, wq, wk, wv, wo, ln2_s, ffn_gate, ffn_up, ffn_down, ...):
    # Pre-attention normalization
    cos, sin = precompute_rope_table(context_len, d_head)
    h_norm = rms_norm(h, ln1_s)

    # Attention (looks at other positions)
    attn_out = causal_attention(h_norm, wq, wk, wv, wo, n_heads, n_kv_heads, cos, sin)

    # Residual connection: add attention output to input
    h = h + attn_out

    # Pre-FFN normalization
    h_norm2 = rms_norm(h, ln2_s)

    # Feed-forward network (processes each position independently)
    h_ff = (jax.nn.silu(h_norm2 @ ffn_gate) * (h_norm2 @ ffn_up)) @ ffn_down

    # Residual connection: add FFN output
    return h + h_ff
```

The **residual connections** (`h = h + attn_out`, `h = h + h_ff`) are critical.
Without them, gradients would need to flow backward through 24 layers of matrix
multiplications, and the signal would vanish. The residual connection provides a
"highway" — the gradient can flow directly from the output back to any layer.

**Pre-norm**: We normalize *before* each sub-layer (attention, FFN), not after.
This is more stable during training than post-norm (the original transformer design).

### 5.8 The Full Forward Pass

```python
def _transformer_trunk(params, config, x):
    h = params["token_emb"][x]       # (512,) → (512, 1024)  look up embeddings
    ...
    for layer in range(config["n_layers"]):
        p = f"layer{layer}"
        h = maybe_checkpoint(_attn_layer, ...)(h, ...)
    return rms_norm(h, params["ln_final.scale"])
```

1. **Embed**: `params["token_emb"][x]` uses the token IDs as indices into the
   embedding matrix. If `x = [42, 100, 7]`, this grabs rows 42, 100, and 7 from
   the `(32000, 1024)` embedding matrix, producing a `(3, 1024)` matrix.

2. **Transform**: Pass through all 24 layers.

3. **Final norm**: Apply one more RMSNorm to the output.

`maybe_checkpoint = jax.checkpoint if use_checkpoint else lambda f, **kw: f`

**Gradient checkpointing**: During training, the backward pass needs intermediate
values from the forward pass. Normally these are stored in memory, but with 24
layers that requires a lot of GPU memory. `jax.checkpoint` tells JAX to *not*
store intermediates — instead, it recomputes them on the fly during the backward
pass. This trades time for memory (roughly 33% slower but uses much less VRAM).

#### Output projection

```python
def transformer_forward(params, config, x):
    h = _transformer_trunk(params, config, x)
    return h @ params["token_emb"].T       # (512, 1024) @ (1024, 32000) → (512, 32000)
```

The final representations are multiplied by the *transpose* of the embedding
matrix. This produces a `(512, 32000)` matrix — for each of the 512 positions, a
score for each of the 32,000 tokens. Higher scores mean higher predicted probability.

These raw scores are called **logits**. They become probabilities after applying
softmax (which we do inside the loss function).

**Tied embeddings**: We reuse `token_emb` for both input (look up rows) and output
(multiply by transpose). This saves 32 million parameters and works because the
embedding should capture token similarity — tokens with similar embeddings should
be predicted in similar contexts.

### 5.9 The Loss Function

The loss measures how wrong the model's predictions are.

#### Standard cross-entropy

```python
def cross_entropy_loss(logits, targets):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    target_log_probs = jnp.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    return -jnp.mean(target_log_probs)
```

1. `log_softmax`: Convert logits to log-probabilities.
   Softmax: $P(\text{token}_i) = \frac{e^{\text{logit}_i}}{\sum_j e^{\text{logit}_j}}$.
   Log-softmax computes $\log P(\text{token}_i)$ directly (more numerically stable
   than computing softmax then taking log).

2. `take_along_axis`: For each position, extract the log-probability of the
   *correct* token. If position 3's target is token 42, we grab `log_probs[3, 42]`.

3. `-mean(...)`: Average the negative log-probabilities. Lower loss = higher
   probability assigned to correct tokens = better predictions.

**Numerical example**: If the model predicts token 42 with probability 0.9,
its contribution to the loss is $-\log(0.9) \approx 0.105$. If it predicts the
correct token with probability 0.01, the contribution is $-\log(0.01) \approx 4.6$
— a much higher loss, penalizing the bad prediction.

**Perplexity** ($e^{\text{loss}}$) is the exponential of the average loss. A
perplexity of 20 means the model is, on average, as uncertain as choosing uniformly
from 20 options. Lower is better.

#### Fused cross-entropy (memory-efficient)

With a vocabulary of 32,000 tokens and a batch of 16 sequences of 512 tokens,
the full logits tensor would be `(16 * 512, 32000)` = 1 billion float values =
4 GB. That doesn't fit in our 16 GB GPU memory alongside the model.

The solution: **never materialize the full logits tensor**. Instead, compute the
loss in chunks of 4096 tokens at a time:

```python
def _chunked_ce_fwd(h, weight, targets, chunk_size):
    target_emb = weight[targets]                      # look up target embeddings
    target_logits = jnp.sum(h * target_emb, axis=-1)  # dot product with target only

    # accumulate logsumexp across vocabulary chunks
    max_logit = jnp.full(N, -1e30)
    sum_exp = jnp.zeros(N)
    for start in range(0, vocab, chunk_size):
        chunk_logits = h @ weight[start:end].T        # only 4096 logits at a time
        chunk_max = chunk_logits.max(axis=-1)
        new_max = jnp.maximum(max_logit, chunk_max)
        sum_exp = sum_exp * jnp.exp(max_logit - new_max) + \
                  jnp.sum(jnp.exp(chunk_logits - new_max[:, None]), axis=-1)
        max_logit = new_max
    logsumexp = max_logit + jnp.log(sum_exp)
    return jnp.mean(logsumexp - target_logits)
```

Instead of computing all 32,000 logits at once, we compute 4,096 at a time and
accumulate the softmax denominator (logsumexp) using the **online logsumexp trick**:

The trick: $\log \sum_i e^{x_i}$ can be computed incrementally. After processing
each chunk, we maintain `max_logit` (the largest logit seen so far) and `sum_exp`
(the accumulated exponential sum). When a new chunk has a larger maximum, we
rescale the running sum: `sum_exp * exp(old_max - new_max)`.

The numerator (`target_logits`) is cheap — just the dot product of each position's
representation with its target token's embedding.

Peak memory: `(8192, 4096)` instead of `(8192, 32000)` — 8x reduction.

#### Custom backward pass

JAX normally computes gradients automatically. But the fused cross-entropy function
uses a Python `for` loop over chunks, which JAX can't differentiate efficiently.
We provide a custom backward pass:

```python
@functools.partial(jax.custom_vjp, nondiff_argnums=(3,))
def fused_cross_entropy(h, weight, targets, chunk_size):
    return _chunked_ce_fwd(h, weight, targets, chunk_size)
```

`custom_vjp` tells JAX: "I'll provide my own forward and backward functions."

The backward function `_fused_ce_bwd` computes gradients for `h` (hidden states)
and `weight` (embedding matrix) using the same chunked approach — it recomputes
the softmax probabilities one chunk at a time and accumulates gradients.

The gradient of cross-entropy loss with respect to logits has a beautiful form:
$\text{grad}_i = \text{softmax}_i - \mathbb{1}[\text{target} = i]$. That is,
subtract 1 from the correct token's probability and keep the rest. This is what
the backward pass computes, chunk by chunk.

### 5.10 Multi-Token Prediction (MTP)

Standard training predicts token $t+1$ from position $t$. MTP additionally
predicts tokens $t+2$, $t+3$, $t+4$ from position $t$, using extra learned
projection heads. This provides more training signal per sequence but costs
more time. It's optional (disabled by default).

```python
for k_idx in range(n_mtp):
    h_shifted = h_batch[:, :-shift, :]     # remove last positions (no target)
    tgt_shifted = targets[:, shift:]        # targets shifted further ahead
    h_proj = rms_norm(h_shifted @ proj, norm_scale)  # small projection per head
    mtp_loss = fused_output_and_loss(h_proj, token_emb, tgt_shifted, chunk_size)
    loss = loss + mtp_loss
return loss / (1 + n_mtp)   # average across all heads
```

---

## 6. The Training Loop

**File**: `train.py`

### 6.1 Setup

```python
_jax_cache = os.path.join(os.path.dirname(__file__), ".jax_cache")
os.makedirs(_jax_cache, exist_ok=True)
os.environ["JAX_COMPILATION_CACHE_DIR"] = _jax_cache
os.environ.setdefault("JAX_COMPILATION_CACHE_MAX_SIZE", str(2 * 1024**3))
```

JAX compiles Python functions to GPU machine code the first time they run. This
cache persists compiled code between runs so you don't wait 30-60 seconds for
recompilation every time you restart training.

### 6.2 Command-Line Arguments

```python
parser.add_argument("--d-model", type=int, required=True)
parser.add_argument("--n-heads", type=int, required=True)
# ... etc
```

All model hyperparameters are passed on the command line. A typical invocation:

```bash
uv run train.py --d-model 1024 --n-heads 16 --n-kv-heads 4 --n-layers 24 \
                --context-len 512 --epochs 3 --batch-size 16
```

### 6.3 Loading Data

```python
data = load_data(context_len=args.context_len, data_dir=args.data_dir)
vocab_size = data["vocab_size"]
train_tokens = data["train_tokens"]   # memmap — NOT loaded into RAM
n_train_seqs = (len(train_tokens) - 1) // args.context_len
n_batches = n_train_seqs // args.batch_size
```

`n_train_seqs`: How many non-overlapping sequences of length `context_len` fit in
the training data. With 7.85B tokens and context_len=512: `7.85B / 512 ≈ 15.3M` sequences.

`n_batches`: How many batches per epoch. With batch_size=16: `15.3M / 16 ≈ 957K` batches.

```python
val_x = jnp.array(data["val_x"][:args.batch_size])
val_y = jnp.array(data["val_y"][:args.batch_size])
```

Take one batch of validation data and move it to the GPU (`jnp.array` creates a
GPU tensor). This stays on the GPU permanently — we evaluate on it periodically.

### 6.4 Model Initialization or Resume

**Fresh start**:

```python
key, init_key = jax.random.split(jax.random.key(args.seed))
params, config = init_transformer(
    init_key, vocab_size, d_model=args.d_model, n_heads=args.n_heads, ...)
```

`jax.random.key(42)` creates a deterministic random seed. `split` produces two
independent sub-keys — one for initialization, one for future use. This ensures
reproducible results.

**Resume from checkpoint**:

```python
with open(args.resume, "rb") as f:
    ckpt = pickle.load(f)
params = jax.tree.map(jnp.array, ckpt["params"])
```

`pickle.load` reads the saved params (NumPy arrays on CPU). `jax.tree.map(jnp.array, ...)`
converts every parameter from NumPy (CPU) to JAX (GPU). `tree.map` works on the
nested dict structure, applying the function to every leaf value.

If the checkpoint contains optimizer state (a "full checkpoint"), we also resume
the optimizer momentum and the step counter, continuing training exactly where
we left off.

### 6.5 Optimizer Setup

```python
schedule = optax.join_schedules(
    [optax.linear_schedule(0.0, args.lr, args.warmup_steps),
     optax.cosine_decay_schedule(args.lr, total_steps - args.warmup_steps, alpha=args.lr * 0.01)],
    boundaries=[args.warmup_steps])
optimizer = optax.adamw(schedule, weight_decay=args.weight_decay)
```

The **learning rate schedule** controls how much we adjust parameters at each step:

1. **Warmup** (first 200 steps): Linearly ramp from 0 to 3e-4. Starting too large
   causes unstable training. The gradual ramp lets momentum statistics calibrate first.
2. **Cosine decay** (remaining steps): Smoothly decrease from 3e-4 toward near-zero
   following a cosine curve. The idea: large updates early for fast progress, small
   updates later for fine-tuning.

**AdamW** is the optimizer algorithm. It maintains two running averages for each
parameter:
- **First moment** (mean gradient): smooths out noisy gradients
- **Second moment** (mean squared gradient): adapts learning rate per parameter —
  parameters with consistently large gradients get smaller updates

**Weight decay** (0.1): Each step, multiply all parameters by 0.999... This gently
pushes unused parameters toward zero, acting as a regularizer to prevent overfitting.

### 6.6 Training Step

```python
def make_train_step(phase_config):
    pc = {**config, "context_len": phase_config["ctx"]}

    @jax.jit
    def step(params, opt_state, x, y):
        def loss_fn(params):
            params_bf16 = jax.tree.map(lambda w: w.astype(jnp.bfloat16), params)
            return transformer_loss_fused(params_bf16, pc, x, y, ce_chunk)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    return step
```

This is the core function that runs millions of times. Let's break it down:

`@jax.jit` — Compile this function to optimized GPU machine code. The first call
is slow (compilation), but all subsequent calls run at full GPU speed.

`params_bf16 = jax.tree.map(lambda w: w.astype(jnp.bfloat16), params)` — Convert
all parameters from float32 to bfloat16 (16-bit) for the forward pass. This halves
memory usage and doubles speed on GPU tensor cores. Parameters are *stored* in
float32 (for precise gradient updates) but *used* in bfloat16.

`loss, grads = jax.value_and_grad(loss_fn)(params)` — This single line is where
the magic happens:

1. **Forward pass**: Run the transformer on batch `x`, compute predictions,
   compare to targets `y`, compute loss.
2. **Backward pass**: Automatically compute the gradient of the loss with respect
   to every single one of the 306M parameters. The gradient tells us: "if I
   increase this parameter by a tiny amount, how much does the loss increase?"

`jax.value_and_grad` returns both the loss value and the gradients tree. The
gradients tree has the exact same structure as `params` — a dict mapping parameter
names to gradient arrays of the same shape.

`updates, opt_state = optimizer.update(grads, opt_state, params)` — AdamW converts
raw gradients into parameter updates (incorporating momentum, adaptive learning
rates, and weight decay).

`optax.apply_updates(params, updates)` — Apply the updates: `new_param = param + update`
for every parameter. (Updates are negative, so this moves parameters in the
direction that decreases loss.)

### 6.7 Curriculum Learning

```python
if args.curriculum and args.context_len >= 256:
    phases = [
        {"ctx": args.context_len // 4, "bs_mult": 4, "end": int(0.10 * total_steps)},
        {"ctx": args.context_len // 2, "bs_mult": 2, "end": int(0.30 * total_steps)},
        {"ctx": args.context_len,      "bs_mult": 1, "end": total_steps},
    ]
```

Instead of training with 512-token sequences from the start, curriculum training
uses shorter sequences initially:

- First 10% of training: 128-token sequences, batch size × 4 (same total tokens per batch)
- Next 20%: 256-token sequences, batch size × 2
- Final 70%: 512-token sequences, normal batch size

Shorter sequences train faster because attention is cheaper ($O(n^2)$ in sequence
length). The model learns basic patterns on short context first, then learns to use
longer context later. The `bs_mult` keeps the total tokens per batch constant.

Each context length requires a separately compiled train step (because `@jax.jit`
compiles for specific input shapes), so we pre-compile them:

```python
phase_steps = {}
for p in phases:
    if p["ctx"] not in phase_steps:
        phase_steps[p["ctx"]] = make_train_step(p)
```

### 6.8 Batch Construction

```python
def _get_batch_streaming(seq_indices, ctx, bs, offset):
    indices = seq_indices[offset:offset + bs]
    batch = np.stack([train_tokens[i * args.context_len:i * args.context_len + ctx + 1]
                      for i in indices])
    return batch[:, :ctx], batch[:, 1:ctx + 1]
```

The training data is a flat stream of tokens. To create a batch:

1. Pick `bs` random sequence start positions from `seq_indices` (a permutation of all
   possible start positions).
2. For each start position, read `ctx + 1` tokens from the memory-mapped file.
3. Split into input (first `ctx` tokens) and target (last `ctx` tokens, shifted by 1).

The `+1` is because we need both the input token and its target (the next token).
Position $i$'s input is token $i$, and its target is token $i+1$.

### 6.9 The Training Loop

```python
for epoch in range(args.epochs):
    rng = np.random.default_rng(args.seed + epoch)
    seq_perm = rng.permutation(n_train_seqs)     # shuffle sequence order
```

An **epoch** is one pass through the entire training dataset. We do 3 epochs. Each
epoch sees the same data in a different random order.

`seq_perm` is a random permutation of all sequence start indices. With `seed + epoch`,
each epoch gets a different shuffle (but the same shuffle if you re-run with the
same seed).

```python
    while bi < n_batches:
        cur_phase = next(p for p in phases if global_step < p["end"])
        ctx = cur_phase["ctx"]
        bs = args.batch_size * cur_phase["bs_mult"]
        train_step = phase_steps[ctx]
        chunk_steps = min(cur_phase["end"] - global_step, n_batches - bi)
```

Find the current curriculum phase based on the global step count, and compute how
many steps to run before the phase changes (or the epoch ends).

#### Prefetching

```python
        bx_np, by_np = _get_batch_streaming(seq_perm, ctx, bs, s)
        next_bx = jax.device_put(jnp.array(bx_np))
        next_by = jax.device_put(jnp.array(by_np))

        for ci in range(chunk_steps):
            bx, by = next_bx, next_by
            # prefetch next batch while GPU is busy
            if ci + 1 < chunk_steps:
                bx_np, by_np = _get_batch_streaming(seq_perm, ctx, bs, s)
                next_bx = jax.device_put(jnp.array(bx_np))
                next_by = jax.device_put(jnp.array(by_np))
```

**Prefetching** overlaps CPU work with GPU work. While the GPU is computing one
training step, the CPU prepares the next batch (reading from disk, transferring to
GPU). Without this, the GPU would idle waiting for data.

`jax.device_put` starts an asynchronous transfer to the GPU and returns immediately.
By the time the GPU finishes the current step, the next batch is already in GPU memory.

#### The inner loop

```python
            params, opt_state, loss = train_step(params, opt_state, bx, by)
            eloss += float(loss)
            global_step += 1
```

This single line runs the entire forward pass, loss computation, backward pass,
and parameter update on the GPU. `float(loss)` copies the scalar loss value back
to the CPU for logging (this is the only CPU-GPU sync point per step).

#### Logging

```python
            if step_in_epoch % 1000 == 0:
                avg = eloss / steps_this_epoch
                sps = steps_this_epoch / (time.perf_counter() - t_epoch)
                eta = (n_batches - step_in_epoch) / sps / 60
                print(f"    step {step_in_epoch}/{n_batches}  loss={avg:.4f}  "
                      f"{sps:.1f} steps/s  eta={eta:.0f}min  {phase_info}")
```

Every 1000 steps, print progress: average loss, steps per second, and estimated
time remaining.

#### Checkpointing

```python
        if args.checkpoint_interval > 0 and global_step % args.checkpoint_interval == 0:
            save_checkpoint(params, opt_state, config, global_step, epoch, bi + ci + 1)
```

Every 2000 steps, save the full training state to disk. This lets you resume
training if it crashes. The checkpoint includes:
- All 306M parameters
- All optimizer state (momentum, squared gradients — another 612M values)
- The current step, epoch, and batch index

The save is **atomic**: write to a temporary file first, then rename. This
prevents a half-written checkpoint if the process is killed mid-save.

```python
def save_checkpoint(params, opt_state, config, global_step, epoch, batch_index):
    ckpt_data = {
        "params": jax.tree.map(np.asarray, params),       # GPU → CPU (numpy)
        "opt_state": jax.tree.map(np.asarray, opt_state), # GPU → CPU
        ...
    }
    fd, tmp = tempfile.mkstemp(dir=ckpt_dir, suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        pickle.dump(ckpt_data, f)
    os.replace(tmp, ckpt_path)       # atomic rename
```

`jax.tree.map(np.asarray, params)` converts every parameter from a JAX GPU array
to a NumPy CPU array (you can't pickle GPU arrays). `os.replace` is atomic on
Linux — either it fully replaces the file or it doesn't, never a partial state.

### 6.10 Validation and Final Output

```python
    vl = float(eval_loss(params, val_x, val_y))
    avg_train = eloss / steps_this_epoch
    print(f"  epoch {epoch+1}/{args.epochs}  train={avg_train:.4f}  "
          f"val={vl:.4f}  ppl={np.exp(vl):.2f}")
```

At the end of each epoch, evaluate on the held-out validation data.
`eval_loss` runs the model forward on `val_x` and computes the loss against
`val_y` — same as training but without gradient computation (faster).

The validation loss tells us if the model is actually getting better at predicting
unseen text, not just memorizing the training data.

```python
elapsed = time.perf_counter() - t_start
mem = jax.local_devices()[0].memory_stats()
print(f"\n{elapsed:.0f}s total, {mem['peak_bytes_in_use']/1e6:.0f}MB peak VRAM")

save_path = os.path.join(os.path.dirname(__file__), "weights.pkl")
with open(save_path, "wb") as f:
    pickle.dump({"params": jax.tree.map(np.asarray, params), "config": config}, f)
```

After all epochs, print timing and peak GPU memory usage, then save the final
model weights (without optimizer state — that's only needed for resuming training,
not for inference).

---

## 7. Glossary

| Term | Meaning |
|------|---------|
| **Attention** | Mechanism where each position computes a weighted sum of other positions' values, with weights based on query-key dot products |
| **Backward pass** | Computing gradients of the loss with respect to all parameters (reverse-mode automatic differentiation) |
| **Batch** | A group of training examples processed together for efficiency (e.g., 16 sequences at once) |
| **bfloat16** | 16-bit floating point format: same exponent range as float32 but fewer mantissa bits. Half the memory, slightly less precise |
| **BPE** | Byte Pair Encoding — a tokenization algorithm that learns common subword units from data |
| **Causal** | Each position can only attend to previous positions, not future ones (enforced by masking) |
| **Context length** | Maximum number of tokens the model can process at once (512 in our case) |
| **Cross-entropy** | Loss function: $-\log P(\text{correct token})$. Lower = better predictions |
| **d_model** | Dimension of token representations (1024). Every vector in the model has this size |
| **d_head** | Dimension per attention head (64 = 1024 / 16 heads) |
| **Decoder-only** | A transformer that only generates tokens left-to-right (vs encoder-decoder which also reads input bidirectionally) |
| **Embedding** | A learned vector representation for a discrete item (like a token). The embedding matrix has one row per vocabulary item |
| **Epoch** | One complete pass through the training dataset |
| **FlashAttention** | An optimized attention algorithm that avoids materializing the full attention matrix, reducing memory from $O(n^2)$ to $O(n)$ |
| **Forward pass** | Running the model on input data to produce predictions |
| **Fused** | Combining multiple operations into one to reduce memory traffic and kernel launch overhead |
| **GQA** | Grouped Query Attention: multiple query heads share one key/value head, reducing memory |
| **Gradient** | The derivative of the loss with respect to a parameter. Points in the direction of steepest loss increase |
| **JIT** | Just-In-Time compilation: compiling a function to machine code the first time it's called |
| **Logits** | Raw (un-normalized) scores output by the model. Become probabilities after softmax |
| **memmap** | Memory-mapping: a file appears as an array in memory, with the OS loading pages on demand |
| **Perplexity** | $e^{\text{loss}}$. Intuitively: the average number of tokens the model is "choosing between" |
| **Residual connection** | Adding a layer's input to its output: `h = h + layer(h)`. Helps gradients flow through deep networks |
| **RMSNorm** | Normalization: divide by root-mean-square, then scale. Stabilizes training |
| **RoPE** | Rotary Position Embeddings: encode position by rotating Q/K vectors. Enables relative position awareness |
| **Softmax** | Converts logits to probabilities: $P_i = e^{x_i} / \sum_j e^{x_j}$. Output is non-negative and sums to 1 |
| **SwiGLU** | A feed-forward network variant with a gated mechanism: $\text{SiLU}(xW_g) \odot (xW_u)$. Better quality than standard ReLU FFN |
| **Tied embeddings** | Using the same matrix for input token lookup and output logit computation |
| **Token** | A subword unit (could be a whole word like "the", a part like "ing", or a single character). Our vocab has 32,000 tokens |
| **Transformer** | Neural network architecture based on self-attention. Currently the dominant architecture for language models |
| **VRAM** | GPU memory (Video RAM). Our GPU has 16 GB |
| **Weight decay** | Multiplying parameters by a factor slightly less than 1 each step, preventing them from growing too large |
