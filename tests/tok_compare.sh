#!/bin/sh
# Builds tests/tok_test and compares ink::Tokenizer against llama.cpp's llama-tokenize (ids must be identical).
#
#   tests/tok_compare.sh [extra tok_compare.py arguments, e.g. --random 20000 --seed 7]
#
# Environment overrides:
#   LLAMA_DIR   llama.cpp checkout with build/bin/llama-tokenize   (default: ../llama.cpp)
#   REF_MODEL   GGUF that llama.cpp can load (any file except UD-Q8_K_XL, whose layout llama.cpp does not
#               read; only the vocabulary is used)  (default: $INKLING_MODEL, else the first UD-Q4 / UD-Q2
#               file found in ./models or next to this folder, else ref/ud-ref.gguf)
#   MODEL       GGUF for our tokenizer                              (default: $REF_MODEL)
#   TOK_BUILD   where to put the test binary                        (default: tests/tok_build)
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMA_DIR="${LLAMA_DIR:-$ROOT/../llama.cpp}"
if [ -z "${REF_MODEL:-}" ]; then
    for cand in "${INKLING_MODEL:-}" \
                "$ROOT/models/Inkling-2-Mini-UD-Q4_K_XL.gguf" "$ROOT/../Inkling-2-Mini-UD-Q4_K_XL.gguf" \
                "$ROOT/models/Inkling-2-Mini-UD-Q4_K_M.gguf" "$ROOT/../Inkling-2-Mini-UD-Q4_K_M.gguf" \
                "$ROOT/models/Inkling-2-Mini-UD-Q2_K_XL.gguf" "$ROOT/../Inkling-2-Mini-UD-Q2_K_XL.gguf" \
                "$ROOT/ref/ud-ref.gguf"; do
        if [ -n "$cand" ] && [ -f "$cand" ]; then REF_MODEL="$cand"; break; fi
    done
fi
if [ -z "${REF_MODEL:-}" ] || [ ! -f "$REF_MODEL" ]; then
    echo "tok_compare.sh: no model file found. Set REF_MODEL to an Inkling 2 Mini GGUF that llama.cpp can load." >&2
    exit 1
fi
if [ ! -x "$LLAMA_DIR/build/bin/llama-tokenize" ]; then
    echo "tok_compare.sh: $LLAMA_DIR/build/bin/llama-tokenize not found. Set LLAMA_DIR to a built llama.cpp checkout." >&2
    exit 1
fi
MODEL="${MODEL:-$REF_MODEL}"
TOK_BUILD="${TOK_BUILD:-$ROOT/tests/tok_build}"

mkdir -p "$TOK_BUILD"
clang++ -std=c++17 -O2 -Wall -Wextra -o "$TOK_BUILD/tok_test" \
    "$ROOT/tests/tok_test.cpp" "$ROOT/src/tokenizer.cpp" "$ROOT/src/gguf.cpp"

"$TOK_BUILD/tok_test" "$MODEL" info

exec python3 "$ROOT/tests/tok_compare.py" \
    --tok-test "$TOK_BUILD/tok_test" \
    --ref-bin "$LLAMA_DIR/build/bin/llama-tokenize" \
    --ref-model "$REF_MODEL" \
    --model "$MODEL" \
    --llama-src "$LLAMA_DIR" \
    "$@"
