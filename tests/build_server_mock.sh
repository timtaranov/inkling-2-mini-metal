#!/bin/sh
# Builds the server against the TEST-ONLY mock engine (no Metal, no GPU): tests/bin/inkling-server-mock
# The vendored httplib is compiled once into tests/bin/httplib.o (it is large and never changes).
set -e
cd "$(dirname "$0")/.."
mkdir -p tests/bin
CXX="${CXX:-clang++}"
FLAGS="-std=c++17 -O2 -Ivendor"
if [ ! -f tests/bin/httplib.o ] || [ vendor/cpp-httplib/httplib.cpp -nt tests/bin/httplib.o ] || [ vendor/cpp-httplib/httplib.h -nt tests/bin/httplib.o ]; then
    $CXX $FLAGS -w -c vendor/cpp-httplib/httplib.cpp -o tests/bin/httplib.o
fi
# every src/*.cpp that belongs to the server, with mock_engine.cpp in place of engine.mm
$CXX $FLAGS -Wall -Wextra \
    src/main.cpp src/server.cpp src/chat.cpp src/sampler.cpp src/tokenizer.cpp src/gguf.cpp src/mock_engine.cpp \
    tests/bin/httplib.o -framework Accelerate -o tests/bin/inkling-server-mock
echo "built tests/bin/inkling-server-mock"
