#include "sampler.h"

#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#endif

namespace ink {

namespace {

// The vocabulary is scanned in blocks so that the (rare) blocks that can contain a candidate are the
// only ones visited by scalar code; everything that touches all ~200K logits is vectorized.
constexpr int BLOCK = 256;

inline float vec_max(const float * x, int n) {
#if defined(__APPLE__)
    float m;
    vDSP_maxv(x, 1, &m, (vDSP_Length) n);
    return m;
#else
    float m = x[0];
    for (int i = 1; i < n; i++) m = x[i] > m ? x[i] : m;
    return m;
#endif
}

inline float vec_sum(const float * x, int n) {
#if defined(__APPLE__)
    float s;
    vDSP_sve(x, 1, &s, (vDSP_Length) n);
    return s;
#else
    float s = 0;
    for (int i = 0; i < n; i++) s += x[i];
    return s;
#endif
}

// out[i] = exp(x[i] * scale + offset)
inline void vec_exp_affine(const float * x, float scale, float offset, float * out, int n) {
#if defined(__APPLE__)
    vDSP_vsmsa(x, 1, &scale, &offset, out, 1, (vDSP_Length) n);
    vvexpf(out, out, &n);
#else
    for (int i = 0; i < n; i++) out[i] = std::exp(x[i] * scale + offset);
#endif
}

}  // namespace

SamplerParams SamplerParams::from_gguf(const GGUF & gguf) {
    SamplerParams p;
    p.temperature = (float) gguf.f64("general.sampling.temp", p.temperature);
    p.top_k = gguf.has("general.sampling.top_k") ? (int) (int64_t) gguf.u64("general.sampling.top_k") : p.top_k;
    p.top_p = (float) gguf.f64("general.sampling.top_p", p.top_p);
    p.min_p = (float) gguf.f64("general.sampling.min_p", p.min_p);
    return p;
}

Sampler::Sampler(const SamplerParams & params, int n_vocab) : p_(params), n_(n_vocab) {
    if (p_.has_seed) {
        rng_.seed(p_.seed);
    } else {
        std::random_device rd;
        rng_.seed(((uint64_t) rd() << 32) ^ rd());
    }
    // drop biases for ids outside the vocabulary once, so the hot path need not check
    p_.logit_bias.erase(std::remove_if(p_.logit_bias.begin(), p_.logit_bias.end(),
                                       [&](const std::pair<int32_t, float> & b) { return b.first < 0 || b.first >= n_; }),
                        p_.logit_bias.end());
    block_max_.resize((size_t) (n_ + BLOCK - 1) / BLOCK);
}

double Sampler::uniform() { return (double) (rng_() >> 11) * (1.0 / 9007199254740992.0); }

const float * Sampler::biased(const float * logits) {
    if (p_.logit_bias.empty()) return logits;
    biased_.resize((size_t) n_);
    memcpy(biased_.data(), logits, sizeof(float) * (size_t) n_);
    for (const auto & b : p_.logit_bias) biased_[(size_t) b.first] += b.second;
    return biased_.data();
}

float Sampler::block_maxima(const float * l) {
    float m = -std::numeric_limits<float>::infinity();
    size_t b = 0;
    for (int off = 0; off < n_; off += BLOCK, b++) {
        float bm = vec_max(l + off, std::min(BLOCK, n_ - off));
        block_max_[b] = bm;
        if (bm > m) m = bm;
    }
    return m;
}

// cand_ <- the k largest logits (unordered).  Requires block_maxima().
void Sampler::collect_top_k(const float * l, int k) {
    cand_.clear();
    auto min_heap = [](const Cand & a, const Cand & b) { return a.logit > b.logit; };
    size_t b = 0;
    for (int off = 0; off < n_; off += BLOCK, b++) {
        if ((int) cand_.size() == k && !(block_max_[b] > cand_.front().logit)) continue;
        int end = std::min(off + BLOCK, n_);
        for (int i = off; i < end; i++) {
            if ((int) cand_.size() < k) {
                cand_.push_back({l[i], i});
                std::push_heap(cand_.begin(), cand_.end(), min_heap);
            } else if (l[i] > cand_.front().logit) {
                std::pop_heap(cand_.begin(), cand_.end(), min_heap);
                cand_.back() = {l[i], i};
                std::push_heap(cand_.begin(), cand_.end(), min_heap);
            }
        }
    }
}

// cand_ <- all tokens with logit >= threshold (unordered).  Requires block_maxima().
void Sampler::collect_above(const float * l, float threshold) {
    cand_.clear();
    size_t b = 0;
    for (int off = 0; off < n_; off += BLOCK, b++) {
        if (!(block_max_[b] >= threshold)) continue;
        int end = std::min(off + BLOCK, n_);
        for (int i = off; i < end; i++) {
            if (l[i] >= threshold) cand_.push_back({l[i], i});
        }
    }
}

int32_t Sampler::sample(const float * logits) {
    const float * l = biased(logits);
    const float m = block_maxima(l);

    auto argmax = [&]() -> int32_t {
        for (size_t b = 0; b < block_max_.size(); b++) {
            if (block_max_[b] == m) {
                int off = (int) b * BLOCK, end = std::min(off + BLOCK, n_);
                for (int i = off; i < end; i++) {
                    if (l[i] == m) return i;
                }
            }
        }
        return 0;   // only reachable with NaN logits
    };

    if (p_.temperature <= 0.0f || !std::isfinite(m)) return argmax();

    const bool use_k = p_.top_k > 0 && p_.top_k < n_;
    const bool use_p = p_.top_p < 1.0f;
    const bool use_min_p = p_.min_p > 0.0f;
    bool have_cand = false;
    auto by_logit_desc = [](const Cand & a, const Cand & b) { return a.logit > b.logit || (a.logit == b.logit && a.id < b.id); };

    if (use_k) {
        collect_top_k(l, p_.top_k);
        std::sort(cand_.begin(), cand_.end(), by_logit_desc);
        have_cand = true;
    }

    if (use_p) {
        const double top_p = std::max(0.0f, p_.top_p);
        double total;   // sum of exp(logit - m) over the current candidate set
        bool cut_done = false;
        if (!have_cand) {
            scratch_.resize((size_t) n_);
            vec_exp_affine(l, 1.0f, -m, scratch_.data(), n_);
            total = 0;
            for (int off = 0; off < n_; off += BLOCK) total += vec_sum(scratch_.data() + off, std::min(BLOCK, n_ - off));
            // Tokens with relative weight below c carry less than n*c in total, so c_min always leaves
            // at least top_p of the mass; start from a much tighter guess and relax only if needed.
            const double c_min = total * (1.0 - top_p) / (double) n_;
            double c = std::max(c_min, total * (1.0 - top_p) / 32.0);
            for (;;) {
                collect_above(l, c > 0 ? m + (float) std::log(c) : -std::numeric_limits<float>::infinity());
                double mass = 0;
                for (const Cand & cd : cand_) mass += scratch_[(size_t) cd.id];
                if (mass >= top_p * total || c <= c_min) break;
                c = c_min;   // the tight guess failed: this is a high-entropy step, take the guaranteed cutoff
            }
            have_cand = true;
            if (cand_.size() > 4096) {
                // High-entropy step: too many candidates to sort.  Histogram them by logit, find the
                // bin where the cumulative mass crosses top_p, and order only that bin.
                enum { NB = 2048 };
                float lo = m;
                for (const Cand & cd : cand_) lo = cd.logit < lo ? cd.logit : lo;
                const float scale = (float) NB / std::max(m - lo, 1e-6f);
                auto bin = [&](float x) {
                    int b = (int) ((m - x) * scale);
                    return b < 0 ? 0 : b >= NB ? NB - 1 : b;
                };
                double hist[NB] = {0};
                for (const Cand & cd : cand_) hist[bin(cd.logit)] += scratch_[(size_t) cd.id];
                double cum = 0;
                int j = 0;
                for (; j < NB - 1; j++) {
                    if (cum + hist[j] >= top_p * total) break;
                    cum += hist[j];
                }
                auto mid = std::partition(cand_.begin(), cand_.end(), [&](const Cand & c) { return bin(c.logit) < j; });
                auto mid2 = std::partition(mid, cand_.end(), [&](const Cand & c) { return bin(c.logit) == j; });
                std::sort(mid, mid2, by_logit_desc);
                size_t keep = (size_t) (mid - cand_.begin());
                for (auto it = mid; it != mid2; ++it) {
                    cum += scratch_[(size_t) it->id];
                    keep++;
                    if (cum >= top_p * total) break;
                }
                cand_.resize(std::max<size_t>(keep, 1));
                cut_done = true;
            } else {
                std::sort(cand_.begin(), cand_.end(), by_logit_desc);
            }
        } else {
            total = 0;
            for (const Cand & cd : cand_) total += std::exp((double) cd.logit - m);
        }
        if (!cut_done) {
            double cum = 0;
            size_t keep = cand_.size();
            for (size_t i = 0; i < cand_.size(); i++) {
                cum += std::exp((double) cand_[i].logit - m);
                if (cum >= top_p * total) {
                    keep = i + 1;
                    break;
                }
            }
            cand_.resize(std::max<size_t>(keep, 1));
        }
    }

    if (use_min_p) {
        const float thr = m + std::log(p_.min_p);
        if (have_cand) {
            cand_.erase(std::remove_if(cand_.begin(), cand_.end(), [&](const Cand & c) { return c.logit < thr; }), cand_.end());
        } else {
            collect_above(l, thr);
            have_cand = true;
        }
    }

    const float inv_t = 1.0f / p_.temperature;

    if (have_cand) {
        if (cand_.empty()) return argmax();
        if (cand_.size() == 1) return cand_[0].id;
        cand_p_.resize(cand_.size());
        for (size_t i = 0; i < cand_.size(); i++) cand_p_[i] = cand_[i].logit;
        vec_exp_affine(cand_p_.data(), inv_t, -m * inv_t, cand_p_.data(), (int) cand_.size());
        double sum = 0;
        for (float pi : cand_p_) sum += pi;
        double r = uniform() * sum;
        for (size_t i = 0; i < cand_.size(); i++) {
            r -= cand_p_[i];
            if (r < 0) return cand_[i].id;
        }
        return cand_.back().id;
    }

    // full-vocabulary draw
    scratch_.resize((size_t) n_);
    vec_exp_affine(l, inv_t, -m * inv_t, scratch_.data(), n_);
    block_sum_.resize(block_max_.size());
    double total = 0;
    {
        size_t b = 0;
        for (int off = 0; off < n_; off += BLOCK, b++) {
            block_sum_[b] = vec_sum(scratch_.data() + off, std::min(BLOCK, n_ - off));
            total += block_sum_[b];
        }
    }
    double r = uniform() * total;
    size_t b = 0;
    for (; b + 1 < block_sum_.size(); b++) {
        if (r < block_sum_[b]) break;
        r -= block_sum_[b];
    }
    for (;; b--) {   // normally a single iteration; walks back only over all-zero blocks
        int off = (int) b * BLOCK, end = std::min(off + BLOCK, n_);
        int last_nonzero = -1;
        for (int i = off; i < end; i++) {
            float pi = scratch_[(size_t) i];
            if (pi > 0) {
                last_nonzero = i;
                r -= pi;
                if (r < 0) return i;
            }
        }
        if (last_nonzero >= 0) return last_nonzero;   // rounding left r slightly above the block sum
        if (b == 0) break;
    }
    return argmax();
}

void Sampler::top_logprobs(const float * logits, int n, std::vector<TokenProb> & top, int32_t chosen, float * chosen_logprob) {
    const float * l = biased(logits);
    const float m = block_maxima(l);
    scratch_.resize((size_t) n_);
    vec_exp_affine(l, 1.0f, -m, scratch_.data(), n_);
    double total = 0;
    for (int off = 0; off < n_; off += BLOCK) total += vec_sum(scratch_.data() + off, std::min(BLOCK, n_ - off));
    const float log_z = m + (float) std::log(total);

    top.clear();
    if (n > 0) {
        collect_top_k(l, std::min(n, n_));
        std::sort(cand_.begin(), cand_.end(), [](const Cand & a, const Cand & b) { return a.logit > b.logit || (a.logit == b.logit && a.id < b.id); });
        for (const Cand & c : cand_) top.push_back({c.id, c.logit - log_z});
    }
    if (chosen_logprob && chosen >= 0 && chosen < n_) *chosen_logprob = l[chosen] - log_z;
}

}  // namespace ink
