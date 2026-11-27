// OpenAI-compatible HTTP server around a single-sequence ink::Engine.
#pragma once
#include <string>

namespace ink {

class Engine;
class Tokenizer;
class GGUF;

struct ServerOptions {
    std::string host = "127.0.0.1";
    int port = 8080;                      // 0: pick a free port (the chosen one is printed)
    std::string alias = "inkling-2-mini"; // model id reported by /v1/models and in responses
    int default_max_tokens = -1;          // -1: generate until EOS or the context limit
    std::string api_key;                  // non-empty: require "Authorization: Bearer <key>"
    bool verbose = false;
};

// Blocks serving requests until SIGINT/SIGTERM.  Prints "listening on http://host:port" to stdout
// once the socket is bound.  Returns a process exit code.
int run_server(Engine & engine, Tokenizer & tokenizer, const GGUF & gguf, const ServerOptions & options);

}  // namespace ink
