// Sampler correctness (against a brute-force reference of the same filter chain) and timing.
//   clang++ -std=c++17 -O2 -Wall -Wextra -Isrc tests/sampler_test.cpp src/sampler.cpp src/gguf.cpp -framework Accelerate -o tests/bin/sampler_test
#include "../src/sampler.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <numeric>
#include <random>
#include <vector>

using namespace ink;

// brute force: probabilities after top_k -> top_p -> min_p -> temperature
static std::vector<double> reference(const std::vector<float> & logits, const SamplerParams & p) {
    int n = (int) logits.size();
    std::vector<double> l(logits.begin(), logits.end());
    for (auto & b : p.logit_bias) l[b.first] += b.second;
    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return l[a] > l[b]; });
    std::vector<double> out(n, 0.0);
    if (p.temperature <= 0) {
        out[idx[0]] = 1;
        return out;
    }
    if (p.top_k > 0 && p.top_k < n) idx.resize(p.top_k);
    double m = l[idx[0]];
    if (p.top_p < 1) {
        double tot = 0, cum = 0;
        for (int i : idx) tot += std::exp(l[i] - m);
        size_t keep = idx.size();
        for (size_t k = 0; k < idx.size(); k++) {
            cum += std::exp(l[idx[k]] - m);
            if (cum >= p.top_p * tot) { keep = k + 1; break; }
        }
        idx.resize(keep);
    }
    if (p.min_p > 0) {
        std::vector<int> kept;
        for (int i : idx) if (l[i] >= m + std::log((double) p.min_p)) kept.push_back(i);
        idx = kept;
    }
    double tot = 0;
    for (int i : idx) tot += std::exp((l[i] - m) / p.temperature);
    for (int i : idx) out[i] = std::exp((l[i] - m) / p.temperature) / tot;
    return out;
}

static int check(const char * name, const std::vector<float> & logits, SamplerParams p, int draws) {
    p.has_seed = true;
    p.seed = 1234;
    Sampler s(p, (int) logits.size());
    std::vector<double> ref = reference(logits, p);
    std::vector<int> count(logits.size(), 0);
    for (int i = 0; i < draws; i++) count[s.sample(logits.data())]++;
    double max_dev = 0;   // in standard deviations of the binomial count
    int outside = 0;
    for (size_t i = 0; i < logits.size(); i++) {
        if (ref[i] == 0 && count[i] > 0) outside++;
        // tokens at the truncation boundary may differ by float rounding of the cutoff; allow tiny mass
        double expect = ref[i] * draws, sd = std::sqrt(std::max(expect * (1 - ref[i]), 1.0));
        max_dev = std::max(max_dev, std::fabs(count[i] - expect) / sd);
    }
    bool ok = outside == 0 && max_dev < 6.0;
    printf("%-44s %s  (max deviation %.2f sd, %d tokens outside support)\n", name, ok ? "ok  " : "FAIL", max_dev, outside);
    return ok ? 0 : 1;
}

static double bench(const char * name, const std::vector<float> & logits, SamplerParams p, bool with_logprobs = false) {
    p.has_seed = true;
    p.seed = 7;
    Sampler s(p, (int) logits.size());
    const int iters = 2000;
    volatile int sink = 0;
    std::vector<TokenProb> top;
    for (int i = 0; i < 50; i++) sink = sink + s.sample(logits.data());
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; i++) {
        int t = s.sample(logits.data());
        if (with_logprobs) {
            float lp;
            s.top_logprobs(logits.data(), 5, top, t, &lp);
        }
        sink = sink + t;
    }
    double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / iters;
    printf("%-58s %8.1f us/token\n", name, us);
    return us;
}

int main() {
    int bad = 0;
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 2.0f);

    // --- correctness on a small vocabulary (many draws per token) ---
    {
        std::vector<float> l(1000);
        for (auto & x : l) x = nd(rng);
        l[17] = 6.0f; l[400] = 5.5f; l[999] = 5.0f; l[0] = 4.0f;
        SamplerParams p;
        bad += check("default (T=1)", l, p, 400000);
        p.temperature = 0.7f; bad += check("T=0.7", l, p, 400000);
        p.temperature = 1.5f; bad += check("T=1.5", l, p, 400000);
        p = {}; p.top_k = 40; bad += check("top_k=40", l, p, 400000);
        p = {}; p.top_k = 1; bad += check("top_k=1", l, p, 1000);
        p = {}; p.top_p = 0.95f; bad += check("top_p=0.95", l, p, 400000);
        p = {}; p.top_p = 0.5f; p.temperature = 2.0f; bad += check("top_p=0.5 T=2", l, p, 400000);
        p = {}; p.top_p = 0.999f; bad += check("top_p=0.999", l, p, 400000);
        p = {}; p.top_p = 0.0f; bad += check("top_p=0", l, p, 1000);
        p = {}; p.min_p = 0.05f; bad += check("min_p=0.05", l, p, 400000);
        p = {}; p.min_p = 0.05f; p.temperature = 3.0f; bad += check("min_p=0.05 T=3", l, p, 400000);
        p = {}; p.top_k = 40; p.top_p = 0.95f; bad += check("top_k=40 top_p=0.95", l, p, 400000);
        p = {}; p.top_k = 40; p.top_p = 0.95f; p.min_p = 0.02f; p.temperature = 0.8f; bad += check("top_k=40 top_p=0.95 min_p=0.02 T=0.8", l, p, 400000);
        p = {}; p.top_p = 0.9f; p.min_p = 0.1f; bad += check("top_p=0.9 min_p=0.1", l, p, 400000);
        p = {}; p.temperature = 0.0f; bad += check("greedy", l, p, 1000);
        p = {}; p.logit_bias = {{17, -100.0f}, {5, 8.0f}, {5000, 3.0f}, {-1, 3.0f}}; bad += check("logit_bias", l, p, 400000);
        p = {}; p.temperature = 0.0f; p.logit_bias = {{17, -100.0f}, {400, -100.0f}}; bad += check("greedy + logit_bias", l, p, 1000);
        // many candidates: exercises the histogram cut in top_p
        {
            std::vector<float> big(20000);
            for (auto & x : big) x = nd(rng) * 0.25f;
            p = {}; p.top_p = 0.9f; bad += check("top_p=0.9, 20000 near-uniform tokens", big, p, 200000);
            p = {}; p.top_p = 0.3f; p.temperature = 1.3f; bad += check("top_p=0.3 T=1.3, 20000 near-uniform", big, p, 200000);
        }
        // flat distribution: top_p has to fall back to the guaranteed cutoff
        std::vector<float> flat(1000, 1.0f);
        p = {}; p.top_p = 0.5f;
        Sampler s(p, 1000);
        std::map<int, int> seen;
        for (int i = 0; i < 20000; i++) seen[s.sample(flat.data())]++;
        bool ok = seen.size() >= 499 && seen.size() <= 501;
        printf("%-44s %s  (%zu distinct tokens)\n", "flat top_p=0.5", ok ? "ok  " : "FAIL", seen.size());
        bad += !ok;
    }

    // --- determinism / seeds ---
    {
        std::vector<float> l(5000);
        for (auto & x : l) x = nd(rng);
        SamplerParams p;
        p.has_seed = true; p.seed = 99;
        Sampler a(p, 5000), b(p, 5000);
        p.seed = 100;
        Sampler c(p, 5000);
        int same_ab = 0, same_ac = 0;
        for (int i = 0; i < 200; i++) {
            int x = a.sample(l.data()), y = b.sample(l.data()), z = c.sample(l.data());
            same_ab += x == y; same_ac += x == z;
        }
        bool ok = same_ab == 200 && same_ac < 50;
        printf("%-44s %s\n", "seed determinism", ok ? "ok  " : "FAIL");
        bad += !ok;
    }

    // --- top_logprobs ---
    {
        std::vector<float> l(3000);
        for (auto & x : l) x = nd(rng);
        l[1234] = 9.0f;
        Sampler s(SamplerParams{}, 3000);
        std::vector<TokenProb> top;
        float lp = 0;
        s.top_logprobs(l.data(), 5, top, 77, &lp);
        double z = 0;
        for (float x : l) z += std::exp((double) x);
        bool ok = top.size() == 5 && top[0].id == 1234 && std::fabs(top[0].logprob - (9.0 - std::log(z))) < 1e-4 &&
                  std::fabs(lp - (l[77] - std::log(z))) < 1e-4 && top[1].logprob >= top[2].logprob;
        printf("%-44s %s\n", "top_logprobs", ok ? "ok  " : "FAIL");
        bad += !ok;
    }

    // --- timing on the real vocabulary size ---
    const int V = 200058;
    std::vector<float> peaked(V), flatish(V);
    for (auto & x : peaked) x = nd(rng);
    peaked[1000] = 18.0f; peaked[150000] = 17.0f; peaked[42] = 15.5f;   // typical LLM step: a few dominant tokens
    for (auto & x : flatish) x = nd(rng);                              // worst case: mass spread over the bulk
    SamplerParams d;
    SamplerParams kp; kp.top_k = 40; kp.top_p = 0.95f;
    SamplerParams tp; tp.top_p = 0.95f;
    SamplerParams mp; mp.min_p = 0.05f;
    SamplerParams g; g.temperature = 0.0f;
    SamplerParams lb; lb.logit_bias = {{5, -100.0f}};
    printf("\n200058-logit timings:\n");
    double t_default = bench("default (T=1, no filters), peaked", peaked, d);
    bench("default (T=1, no filters), spread", flatish, d);
    double t_kp = bench("top_k=40 top_p=0.95, peaked", peaked, kp);
    bench("top_k=40 top_p=0.95, spread", flatish, kp);
    bench("top_p=0.95 only, peaked", peaked, tp);
    bench("top_p=0.95 only, spread (worst case)", flatish, tp);
    bench("min_p=0.05, peaked", peaked, mp);
    bench("greedy", peaked, g);
    bench("default + logit_bias", peaked, lb);
    bench("default + top-5 logprobs", peaked, d, true);
    if (t_default > 200 || t_kp > 200) {
        printf("FAIL: over the 0.2 ms budget\n");
        bad++;
    }
    printf("%s\n", bad ? "SAMPLER TESTS FAILED" : "all sampler tests passed");
    return bad ? 1 : 0;
}
