# Design, measurements and validation

This document holds the detail behind the README: what each option does and costs, how the
engine works, how the code is laid out, and how it was validated. All measurements are from one
machine, an M5 Max with 128 GB.

## Measurements

All figures were measured on this machine. They vary by 10–20% with how warm the machine is:
the GPU slows under sustained load.

**Baseline** (Q8_K_XL, no options), against llama.cpp build `eb5657509` on the same weights:

| Prompt | Prompt processing | llama.cpp | Generation afterwards | llama.cpp |
|---|---|---|---|---|
| 2.4K tokens | 3,150 tok/s | ~870 tok/s | 91 tok/s | 42 tok/s |
| 17.6K tokens | 3,070 tok/s | — | 83 tok/s | — |
| 40K tokens | 2,600 tok/s | 313 tok/s | 72 tok/s | 36 tok/s (reported) |
| 100K tokens | 1,550–1,600 tok/s | — | 54 tok/s | — |

A follow-up turn that adds ~100 tokens to a cached conversation takes about 0.25 s before the
first token. Sending the same prompt again takes a few milliseconds.

**Generation slows as the context grows**, because 8 of the 48 block evaluations per token use
full attention and read the cached keys and values of every position. In real agent sessions it
ran at about 83 tok/s at 15K tokens, 51–54 at 100K and 42–46 at 120–155K. `--draft-model` and
`--kv8` both attack that; see below.

**By model file** (no options):

| File | Generation, short / after 17.6K | Prompt, 17.6K | Memory | with `--low-mem` |
|---|---|---|---|---|
| UD-Q8_K_XL | 90 / 80–81 tok/s | 2,900–3,020 tok/s | 68 GB | 32 GB |
| UD-Q4_K_XL, UD-Q4_K_M | 90 / 80–81 tok/s | 2,900–3,020 tok/s | 68 GB | 25 GB |
| Uncensored-AD-Q4_K | 86–89 / 78 tok/s | 3,030 tok/s | 71 GB | 24 GB |
| UD-Q2_K_XL | 92–94 / 80 tok/s | 2,840 tok/s | 61 GB | 18 GB |

The smaller files are not faster. Prompt processing expands every weight to 16-bit whatever the
file holds, and at generation the experts are only about half of what is read per token; the
rest is attention weights and the output head, which all these files keep at 5–8 bits. What the
smaller files save is disk space and, with `--low-mem`, memory.

## Options in detail

### `--low-mem`

Without it, a 16-bit copy of every expert is kept for prompt processing (49 GB). With it, each
layer's packed experts are expanded into a 2 GB scratch buffer right before the layer runs, 48
times per batch. Output is bit-identical and generation speed is unchanged.

The price is prompt speed and follow-up latency. Prompt processing was 9–21% slower in my runs
(17.6K tokens: 2,330 against 2,940 tok/s on a cool machine; 40K tokens: 2,095 against 2,305). A
follow-up turn that adds ~100 tokens takes about 0.53 s instead of 0.24 s, because the expansion
costs the same however small the batch is. `--batch 8192` wins back a few percent on long
prompts. Q8_K_XL keeps 16-bit copies of its last four expert layers (they are 16-bit in the
file), hence its higher memory figure.

### `--draft-model ../dspark-Inkling-2-Mini-Q8_0.gguf`

The DSpark draft model is a 3-block companion model. It guesses up to 7 next tokens from the main
model's hidden states, and one multi-token pass of the main model checks them. A drafted token is
used only if it is the token the sampler picks from the main model's own logits, so the output
distribution is unchanged for any sampling settings.

| Test (Q4_K_XL, `--low-mem`, greedy) | No drafting | `--kv8` | `--draft-model` |
|---|---|---|---|
| 12.5K-token file, explain + write a helper | 82 tok/s | — | 97 tok/s |
| 12.5K-token file, rewrite a function | 82 tok/s | — | 117–119 tok/s |
| 116K tokens of source, explain + sketch | 51 tok/s | 61 tok/s | 73 tok/s |
| 116K tokens of source, rewrite a function | 51 tok/s | 61 tok/s | 82 tok/s |

- The gain depends on how predictable the output is. Repeating code is the best case; free
  reasoning text gains least. It grows with context.
- With default sampling (temperature 1) the two 12.5K requests ran at 88 and 124 tok/s.
- About half of all drafted tokens were accepted (49%), the same rate llama.cpp gets from this
  draft model on the same requests (48%).
- Costs: about 2 GB more memory at 116K tokens, prompt processing about 1% slower.
- Combining it with `--kv8` gave no further gain: the multi-token pass reads the 16-bit cache.
- Logits from a multi-token pass differ from the single-token path at noise level (mean KL about
  1e-4, the top token changed in up to 3 of 200 steps), so output is not bit-identical to
  generation without drafting.
- `dspark-Inkling-2-Mini-BF16.gguf` is an older layout and is rejected.

Cost model: a multi-token pass takes about 11.5 ms plus 3.9 ms per token (a single token takes
11 ms at short context), a draft run about 6.5 ms. Drafts are kept only while the draft model's
joint confidence in the block stays above 0.5.

`--draft N` without a draft model copies drafts from earlier context instead. It works, but such
drafts were accepted only 30–55% of the time on a coding request and gain little.

### `--kv8`

Generation reads an 8-bit copy of the full-attention cache (one scale per position and head)
instead of the 16-bit one; prompt processing keeps using the 16-bit cache.

| Context | Generation | with `--kv8` |
|---|---|---|
| 2.4K tokens | 90 tok/s | 90 tok/s |
| 40K tokens | 72 tok/s | 78 tok/s |
| 100K tokens | 54 tok/s | 64 tok/s |
| ~155K tokens (real session) | 42–44 tok/s | 50–53 tok/s |

It is an approximation: against exact generation the mean KL over 100 teacher-forced steps was
about 1e-3 (one top-token change in 100 steps at 17.6K tokens, none at 100K). It adds half again
to the cache memory, about 2 GB at 100K tokens.

### `--precise`

By default generation uses 8-bit copies of the weights that Q8_K_XL stores as 16-bit (attention,
output head, last four expert layers). That is the same error level as a Q8_0 GGUF and measured
smaller than the difference between llama.cpp's own Metal and CPU backends. `--precise` turns the
copies off: generation drops to 70 tok/s at short context and 57 tok/s after a 40K prompt. For
the other files generation already uses the file's values as stored, so the flag adds nothing.

## How it works

The model has 24 blocks, of which 12 run three times per token (48 block evaluations), with
128-expert MoE layers, sliding-window attention on 5 of every 6 blocks and full attention on the
rest.

- **Prompt processing** is compute-bound, so it runs on Metal Performance Primitives matmul
  (about 35–55 TFLOPS on this GPU) with FP16 weights: dense GEMMs for the attention projections,
  per-expert GEMMs over tokens sorted by expert, and a flash-attention kernel.
- **Single-token generation** is bound by memory bandwidth (about 500 GB/s) and per-step latency.
  It uses fused mat-vec kernels that read each weight once per token straight from the packed
  data, in one GPU command buffer per token with no CPU round trips. Expert tensors in NVFP4,
  Q4_K, Q5_0, Q2_0 and (gate/up) IQ1_S / IQ2_XXS / IQ2_XS / IQ2_S are read in place; other
  formats fall back to 8-bit copies.
- **Multi-token generation passes** (drafting) run the same computation for up to 8 tokens with a
  low fixed cost: wide per-token kernels, mat-vecs with one right-hand side per token, one read
  of the attention cache for all tokens, and each distinct expert decoded once for all tokens
  that selected it. The convolution state is kept for every prefix of the pass, so the state of
  the accepted tokens can be committed afterwards.
- **The draft model** attends to keys and values derived from the main model's hidden states at
  four depths for every context position; these are written during prompt processing and
  generation. A recurrent head then picks the drafted tokens one after the other on the GPU.
- All paths share one KV cache and convolution state.

## Code layout

| Path | Contents |
|---|---|
| `src/engine.mm` | The engine: loading, prompt processing, generation, drafting, prompt cache |
| `src/kernels.metal` | Kernels for prompt processing and single-token generation |
| `src/kernels_quant.metal` | In-place readers for the low-bit GGUF formats, low-memory expansion, the 8-bit cache, the multi-token pass |
| `src/kernels_dspark.metal` | Kernels of the DSpark draft model |
| `src/quant_tables.*`, `tools/gen_quant_tables.py` | Codebooks of the GGUF IQ formats, generated from ggml's format header (the one place data from ggml is used) |
| `src/gguf.*`, `src/tokenizer.*` | GGUF reader, BPE tokenizer |
| `src/chat.*`, `src/sampler.*`, `src/server.*`, `src/main.cpp` | Template, sampling, HTTP API |
| `src/ink_test.cpp` | Developer tool: timing, logit comparison, cache tests (`build/ink-test`, built by `make`) |
| `tests/` | Tokenizer, chat-template and server tests |
| `tools/convert_ud_to_ref.py`, `ref/` | Builds a llama.cpp-loadable copy of Q8_K_XL and reference logits for validation |
| `bench/` | The GPU micro-benchmarks that drove the kernel design |

`INK_PROF=1` prints a per-stage time breakdown (it forces GPU syncs, so totals are inflated).
`--verbose` prints, at exit, how many drafts were made and accepted.

## Validation

- **Q8_K_XL:** logits compared against llama.cpp (Metal and CPU) at 27, 2.4K and 17.6K tokens.
  Greedy continuations are identical, and the remaining logit differences are the same size as
  those between llama.cpp's two backends.
- **Q4_K_XL, Q4_K_M:** identical greedy continuations at 27 tokens. At 2.4K they part at a near
  tie (34% vs 29%), with logits 2.5% RMS from llama.cpp's Metal backend, whose own distance to
  its CPU backend on that prompt is 4.0%.
- **Uncensored AD-Q4_K:** greedy continuations identical to llama.cpp at 27 and 2.4K tokens.
- **Q2_K_XL:** at 2.4K tokens logits are 6.5% RMS from llama.cpp's Metal backend, whose own
  distance to its CPU backend is 19%; greedy output parts from llama.cpp at near ties. Each
  in-place kernel was also compared with the 16-bit kernel on every layer (agreement to rounding
  level).
- **`--low-mem`:** bit-identical logits to normal mode on all four kinds of file.
- **Drafting:** against single-token generation, mean KL 5e-5 to 4e-4 over 200–300 teacher-forced
  steps on four model files; draft acceptance equal to llama.cpp's with the same draft model.
- **Prompt cache:** bit-identical to fresh evaluation for a repeated prompt, an extended prompt,
  an edited tail and a deep edit, with and without drafting.
- **Tokenizer:** identical ids to `llama-tokenize` on 3.8M tokens. **Chat template:** identical
  to a jinja2 reference on 3,184 cases. **Server:** 27 end-to-end tests against a mock engine.

The local llama.cpp loads every file except Q8_K_XL directly. For Q8_K_XL, regenerate a
loadable copy first (23 GB, about a minute):

```sh
cd .. && python3 inkling-2-mini-metal/tools/convert_ud_to_ref.py Inkling-2-Mini-UD-Q8_K_XL.gguf inkling-2-mini-metal/ref/ud-ref.gguf
```
