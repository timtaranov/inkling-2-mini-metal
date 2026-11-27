// Token sampler: top-k -> top-p -> min-p -> temperature -> multinomial draw (llama.cpp's default
// chain order, i.e. the truncation filters see the un-tempered distribution).
#pragma once
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace ink {

class GGUF;

struct SamplerParams {
    float temperature = 1.0f;   // <= 0: greedy argmax
    int top_k = 0;              // <= 0: disabled
    float top_p = 1.0f;         // >= 1: disabled
    float min_p = 0.0f;         // <= 0: disabled
    bool has_seed = false;      // false: seed from std::random_device
    uint64_t seed = 0;
    std::vector<std::pair<int32_t, float>> logit_bias;   // added to the logits before anything else

    // Defaults from general.sampling.{temp,top_k,top_p,min_p} when present.
    static SamplerParams from_gguf(const GGUF & gguf);
};

struct TokenProb {
    int32_t id;
    float logprob;
};

class Sampler {
public:
    Sampler(const SamplerParams & params, int n_vocab);

    // Draw the next token from `logits` (n_vocab floats).
    int32_t sample(const float * logits);

    // Natural-log probabilities under the unfiltered, temperature-1 distribution (after logit_bias):
    // the `n` most likely tokens in descending order, plus the logprob of `chosen` if >= 0.
    void top_logprobs(const float * logits, int n, std::vector<TokenProb> & top, int32_t chosen = -1, float * chosen_logprob = nullptr);

    const SamplerParams & params() const { return p_; }

private:
    struct Cand {
        float logit;
        int32_t id;
    };
    const float * biased(const float * logits);
    float block_maxima(const float * l);
    void collect_top_k(const float * l, int k);
    void collect_above(const float * l, float threshold);
    double uniform();

    SamplerParams p_;
    int n_;
    std::mt19937_64 rng_;
    std::vector<float> scratch_;      // n_ floats: exp() workspace
    std::vector<float> biased_;       // n_ floats: logits + bias (only when logit_bias is used)
    std::vector<float> block_max_;    // per-block maxima of the current logits
    std::vector<float> block_sum_;
    std::vector<Cand> cand_;
    std::vector<float> cand_p_;
};

}  // namespace ink
