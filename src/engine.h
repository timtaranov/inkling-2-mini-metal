// Public interface of the Inkling-2-Mini inference engine (Metal, Apple M5 Max).
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace ink {

class GGUF;

struct EngineOptions {
    int max_ctx = 262144;      // hard limit on context length (tokens)
    int batch = 4096;          // prompt-processing batch size (tokens per GPU pass)
    bool q8 = true;            // generation: 8-bit copies of the 16-bit weights (faster, Q8_0-level error)
    bool low_mem = false;      // prompt processing expands the packed experts of a layer to 16-bit on the fly
                               // instead of keeping 16-bit copies of all of them (about 49 GB less memory)
    bool kv8 = false;          // generation reads an 8-bit copy of the full-attention K/V caches: faster at long
                               // context (the cache read halves), slightly approximate, 50% more cache memory
    int draft = 0;             // generation: up to this many tokens (0..7) are drafted and verified in one multi-token
                               // pass; drafts come from `draft_model`, or without one are copied from the context
    std::string draft_model;   // DSpark draft model (GGUF, arch "dflash") for `draft`; empty: drafts are copied from the context
    int keep_wired_s = 1800;   // keep GPU memory wired this long after the last request (0 = never unwire)
    int keep_warm_s = 600;     // keep the GPU out of its idle power state this long after the last request
                               // (a ~20 us workload every 100 ms; 0 = off)
    bool verbose = false;
};

struct EngineTimings {
    double prompt_ms = 0;      // wall time spent in the last set_prompt()
    int prompt_tokens = 0;     // tokens actually evaluated by the last set_prompt()
    int prompt_cached = 0;     // tokens reused from the cache by the last set_prompt()
};

// The engine holds exactly one sequence.  It is NOT thread-safe: callers serialize access.
class Engine {
public:
    static std::unique_ptr<Engine> create(const GGUF & gguf, const EngineOptions & opt);   // throws std::runtime_error
    virtual ~Engine() = default;

    virtual int n_vocab() const = 0;   // number of logits (unpadded vocabulary)
    virtual int max_ctx() const = 0;

    // Make the model state correspond to exactly `tokens` (must be non-empty), reusing the longest
    // prefix shared with the current state.  `progress(done, total)` is called between batches with
    // the number of tokens evaluated so far; returning false aborts (the state stays valid, shorter).
    // Returns false if aborted.  On success logits() is the distribution of the next token.
    virtual bool set_prompt(const std::vector<int32_t> & tokens,
                            const std::function<bool(int done, int total)> & progress = nullptr) = 0;

    // Append one token to the sequence and evaluate it; logits() then predicts the following token.
    virtual void append(int32_t token) = 0;

    // n_vocab() floats, valid until the next set_prompt()/append()/reset().
    virtual const float * logits() = 0;

    virtual int n_ctx_used() const = 0;       // tokens currently in the sequence
    virtual const std::vector<int32_t> & tokens() const = 0;
    virtual void reset() = 0;
    virtual EngineTimings timings() const = 0;
};

}  // namespace ink
