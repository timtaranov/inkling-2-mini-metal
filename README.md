# inkling-2-mini-metal

A single-model inference server for [Inkling 2 Mini](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)
on Apple silicon, with an OpenAI-compatible API. It is written from scratch for this one model
and for Metal: it does not link llama.cpp or ggml.

On the machine it was built on (M5 Max, 128 GB) it processes prompts at 2,600–3,150 tokens per
second and generates at 50–120 tokens per second depending on context length and options. On the
same weights llama.cpp does roughly 300–870 and 36–48.

> **Experimental.** It runs one model architecture only, and has only ever been run on one
> machine, an M5 Max. Other Apple silicon Macs should work but are untested, and the kernels are
> tuned for that GPU.

## Requirements

- Any Apple silicon Mac (M1 or later) with macOS 26 or later; the kernels need Metal 4.
- Xcode command-line tools (`clang++`, `make`) and `python3`.
- Memory: 18–32 GB free with `--low-mem`, 61–71 GB without it. See the table below.
- A GGUF file of Inkling 2 Mini.

## Quick start

```sh
git clone <this repo> && cd inkling-2-mini-metal
make                                              # builds ./inkling-server

mkdir -p models                                   # put the two files here:
#   Inkling-2-Mini-UD-Q4_K_XL.gguf     the model   (https://huggingface.co/unsloth/Inkling-2-Mini-GGUF)
#   dspark-Inkling-2-Mini-Q8_0.gguf    its DSpark draft model, 0.5 GB

./inkling-server --model models/Inkling-2-Mini-UD-Q4_K_XL.gguf --low-mem \
                 --draft-model models/dspark-Inkling-2-Mini-Q8_0.gguf
```

The draft model is optional but recommended: it makes generation 1.2–1.6x faster (more on output
that repeats code, and more at long context) without changing the output distribution. Leave
`--draft-model` out to run without it.

```sh
curl localhost:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "Hello"}],
  "reasoning_effort": "low", "stream": true}'
```

Any OpenAI-compatible client works with base URL `http://127.0.0.1:8080/v1` and any API key.

Endpoints: `/v1/chat/completions` (streaming, tools, `reasoning_content`), `/v1/completions`,
`/v1/models`, `/health`, `/tokenize`, `/detokenize`, `/apply-template`. One request runs at a
time; others queue. Each request is logged with its prompt and generation speed.

The server listens on localhost only by default. `--api-key` adds a bearer-token check, but it
is a plain HTTP server meant for local use: put it behind a proxy before exposing it to a network.

`./run.sh [options]` rebuilds if sources changed and then starts the server. Without `--model`
the server looks at `$INKLING_MODEL`, then for `Inkling-2-Mini-UD-Q8_K_XL.gguf` in `./models/`
or next to this folder.

## Model files

All of these are the same model; they differ in how the weights are stored.

| File | Size on disk | Memory | Memory with `--low-mem` |
|---|---|---|---|
| `Inkling-2-Mini-UD-Q8_K_XL.gguf` | 22.9 GB | 68 GB | 32 GB |
| `Inkling-2-Mini-UD-Q4_K_XL.gguf`, `…UD-Q4_K_M.gguf` | 17.6 / 16.6 GB | 68 GB | 25 GB |
| `Inkling-2-Mini-UD-Q2_K_XL.gguf` | 9.1 GB | 61 GB | 18 GB |

The `UD-*` files come from [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF).
Other quantizations of the same model should load too; a conventional Q4_K / Q5_0 file was
tested (71 GB, 24 GB with `--low-mem`). The smaller files are not faster, they only save disk
space and memory. Q4_K_XL with `--low-mem` is a good starting point.

Memory grows with the conversation: a few GB more at 100K tokens (Q4_K_XL with `--low-mem`
measured 31 GB at 116K tokens, 33 GB with a draft model).

## Options

| Option | Default | Meaning |
|---|---|---|
| `--model PATH` | see above | Model file |
| `--low-mem` | off | Much less memory (see the table), prompts 9–21% slower, identical output |
| `--draft-model PATH` | none | DSpark draft model (recommended): faster generation through verified drafts, same output distribution |
| `--draft N` | 7 with a draft model, else 0 | Max tokens drafted per pass (0–7) |
| `--kv8` | off | 8-bit attention cache for generation: faster at long context, slightly approximate |
| `--precise` | off | 16-bit weights at generation instead of 8-bit copies (only matters for Q8_K_XL) |
| `--ctx N` | 262144 | Context limit in tokens |
| `--batch N` | 4096 | Prompt-processing batch size |
| `--host`, `--port` | 127.0.0.1, 8080 | Listen address |
| `--alias`, `--api-key`, `--verbose` | | Model id in the API, required bearer token, engine logging |
| `--keep-warm S`, `--keep-wired S` | 600, 1800 | Keep the GPU awake / its memory wired after a request |

Which options to use:

- **Less than ~90 GB of memory:** `--low-mem`.
- **Always, if you have it:** `--draft-model` with the DSpark draft model for Inkling 2 Mini
  (`dspark-Inkling-2-Mini-Q8_0.gguf`, 0.5 GB). It gives the largest generation speed-up, most
  of all in long conversations and coding agents, and does not change the output distribution.
- **No draft model at hand:** `--kv8` helps at long context, at a small cost in exactness.

Thinking is on by default at high effort, so a short `max_tokens` can be used up before any
answer text. Pass `reasoning_effort` (`"low"` … `"high"`) in the request to control it.

## Speed

Measured on an M5 Max. Expect less on smaller chips, and 10–20% variation with how warm the
machine is.

Prompt processing and generation, default options, against llama.cpp on the same weights:

| Prompt | Prompt processing | llama.cpp | Generation afterwards | llama.cpp |
|---|---|---|---|---|
| 2.4K tokens | 3,150 tok/s | ~870 tok/s | 91 tok/s | 42 tok/s |
| 40K tokens | 2,600 tok/s | 313 tok/s | 72 tok/s | 36 tok/s |
| 100K tokens | 1,550–1,600 tok/s | — | 54 tok/s | — |

Generation with the long-context options, on coding requests (Q4_K_XL, `--low-mem`, greedy):

| Test | No options | `--kv8` | `--draft-model` |
|---|---|---|---|
| 12.5K-token file, explain + write a helper | 82 tok/s | — | 97 tok/s |
| 12.5K-token file, rewrite a function | 82 tok/s | — | 117–119 tok/s |
| 116K tokens of source, explain + sketch | 51 tok/s | 61 tok/s | 73 tok/s |
| 116K tokens of source, rewrite a function | 51 tok/s | 61 tok/s | 82 tok/s |

A follow-up turn that adds ~100 tokens to a cached conversation takes about 0.25 s before the
first token (0.55 s with `--low-mem`).

## Limitations

- One model architecture, one request at a time, text only.
- No grammar-constrained output; `tool_choice: "required"` behaves like `"auto"`.
- Default generation is not bit-exact for Q8_K_XL (8-bit copies of its 16-bit tensors;
  `--precise` turns that off). `--kv8` is an approximation. Drafted generation matches
  non-drafted generation only to rounding noise, though its sampling distribution is the same.
- The older `dspark-Inkling-2-Mini-BF16.gguf` draft model layout is rejected.
- Not done: combining drafting with the 8-bit cache, a cheaper draft run, and a middle memory
  mode between the two existing ones.

## More

- [docs/design.md](docs/design.md): what each option costs, how the engine works, code layout,
  and how it was validated against llama.cpp.
- Tests, all of which need a model file for its vocabulary (`INKLING_MODEL=path`, or a file in
  `./models/`):
  - `sh tests/build_server_mock.sh && python3 tests/server_test.py` runs the API tests against a
    mock engine, with no GPU work.
  - `tests/tok_compare.sh` compares the tokenizer with llama.cpp's and so also needs a built
    llama.cpp checkout (`LLAMA_DIR`) and a model file llama.cpp can load (any except Q8_K_XL).
  - `build/ink-test` (built by `make`) times the engine and compares logits with reference
    dumps made by `ref/tools/ref_logits.cpp`, which links llama.cpp.

## License

MIT, see [LICENSE](LICENSE). Third-party components and data are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The model files are not part of this
repository and have their own license.
