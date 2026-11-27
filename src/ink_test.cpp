// Developer tool: run the engine on a token-id file, compare logits with a reference dump, time it.
//   ink-test MODEL TOKENS [--ref FILE] [--greedy N] [--batch N] [--repeat N] [--split K]
#include "engine.h"
#include "gguf.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>
#include <unistd.h>

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static void compare(const float * a, const float * b, int n, const char * what) {
    double maxd = 0, sum2 = 0, ref2 = 0;
    for (int i = 0; i < n; ++i) { double d = (double) a[i] - b[i]; maxd = std::max(maxd, fabs(d)); sum2 += d * d; ref2 += (double) b[i] * b[i]; }
    // KL(ref || ours) over softmax
    double ma = *std::max_element(a, a + n), mb = *std::max_element(b, b + n), za = 0, zb = 0;
    for (int i = 0; i < n; ++i) { za += exp(a[i] - ma); zb += exp(b[i] - mb); }
    double kl = 0;
    for (int i = 0; i < n; ++i) { double pb = exp(b[i] - mb) / zb; if (pb > 1e-12) kl += pb * ((b[i] - mb - log(zb)) - (a[i] - ma - log(za))); }
    int aa = (int) (std::max_element(a, a + n) - a), ab = (int) (std::max_element(b, b + n) - b);
    printf("%s: argmax %d vs ref %d (%s)  max|d|=%.4f  rel.rms=%.5f  KL=%.3g\n", what, aa, ab, aa == ab ? "same" : "DIFF", maxd, sqrt(sum2 / ref2), kl);
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf tokens.txt [--ref f] [--greedy n] [--batch n] [--repeat n] [--split k]\n", argv[0]); return 1; }
    std::string ref; int greedy = 0, batch = 4096, repeat = 1, split = 0; bool verbose = false, cachetest = false, q8 = true; std::string tfdump, tfref; int tfn = 64; int aftergen = -1, sleepms = 0; bool lowmem = false, kv8 = false; int draft = 0; std::string dmodel;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--ref") ref = argv[++i];
        else if (a == "--greedy") greedy = atoi(argv[++i]);
        else if (a == "--batch") batch = atoi(argv[++i]);
        else if (a == "--repeat") repeat = atoi(argv[++i]);
        else if (a == "--split") split = atoi(argv[++i]);
        else if (a == "-v") verbose = true;
        else if (a == "--cachetest") cachetest = true;
        else if (a == "--q8") q8 = true;
        else if (a == "--after-gen") aftergen = atoi(argv[++i]);
        else if (a == "--sleep-ms") sleepms = atoi(argv[++i]);
        else if (a == "--precise") q8 = false;
        else if (a == "--low-mem") lowmem = true;
        else if (a == "--kv8") kv8 = true;
        else if (a == "--draft") draft = atoi(argv[++i]);
        else if (a == "--draft-model") dmodel = argv[++i];
        else if (a == "--tf-dump") tfdump = argv[++i];
        else if (a == "--tf-ref") tfref = argv[++i];
        else if (a == "--tf-n") tfn = atoi(argv[++i]);
    }
    std::vector<int32_t> toks; { std::ifstream f(argv[2]); long long t; while (f >> t) toks.push_back((int32_t) t); }
    try {
        ink::GGUF gguf(argv[1]);
        ink::EngineOptions opt; opt.batch = batch; opt.verbose = verbose; opt.q8 = q8; opt.low_mem = lowmem; opt.kv8 = kv8; opt.draft = draft; opt.draft_model = dmodel;
        double t0 = now_s();
        auto eng = ink::Engine::create(gguf, opt);
        printf("load: %.2fs, %zu prompt tokens\n", now_s() - t0, toks.size());
        const int nv = eng->n_vocab();

        std::vector<float> refAll; int refPos = 0, refV = 0; std::vector<int32_t> refGen; std::vector<float> refLast;
        if (!ref.empty()) {
            FILE * f = fopen(ref.c_str(), "rb"); int32_t hdr[2];
            if (!f || fread(hdr, 4, 2, f) != 2) { fprintf(stderr, "bad ref\n"); return 1; }
            refPos = hdr[0]; refV = hdr[1];
            refAll.resize((size_t) refPos * refV);
            if (fread(refAll.data(), 4, refAll.size(), f) != refAll.size()) return 1;
            int32_t ng = 0;
            if (fread(&ng, 4, 1, f) == 1 && ng > 0) {
                refGen.resize(ng); if (fread(refGen.data(), 4, ng, f) != (size_t) ng) return 1;
                refLast.resize(refV); if (fread(refLast.data(), 4, refV, f) != (size_t) refV) refLast.clear();
            }
            fclose(f);
            if (refV != nv) { fprintf(stderr, "vocab mismatch %d vs %d\n", refV, nv); return 1; }
        }

        if (!tfdump.empty() || !tfref.empty()) {
            // teacher-forced decode: dump the token path + logits, or follow a dumped path and report the divergence
            eng->set_prompt(toks);
            if (!tfdump.empty()) {
                FILE * f = fopen(tfdump.c_str(), "wb");
                fwrite(&tfn, 4, 1, f);
                for (int g = 0; g < tfn; ++g) {
                    const float * l = eng->logits();
                    int32_t best = (int32_t) (std::max_element(l, l + nv) - l);
                    fwrite(&best, 4, 1, f); fwrite(l, 4, nv, f);
                    eng->append(best);
                }
                fclose(f);
                printf("dumped %d teacher-forced steps\n", tfn);
            } else {
                FILE * f = fopen(tfref.c_str(), "rb");
                int n = 0; if (!f || fread(&n, 4, 1, f) != 1) return 1;
                std::vector<float> rl(nv); double klsum = 0, klmax = 0; int flips = 0;
                for (int g = 0; g < n; ++g) {
                    int32_t tok; if (fread(&tok, 4, 1, f) != 1 || fread(rl.data(), 4, nv, f) != (size_t) nv) return 1;
                    const float * a = eng->logits();
                    double ma = *std::max_element(a, a + nv), mb = *std::max_element(rl.begin(), rl.end()), za = 0, zb = 0;
                    for (int i = 0; i < nv; ++i) { za += exp(a[i] - ma); zb += exp(rl[i] - mb); }
                    double kl = 0;
                    for (int i = 0; i < nv; ++i) { double pb = exp(rl[i] - mb) / zb; if (pb > 1e-12) kl += pb * ((rl[i] - mb - log(zb)) - (a[i] - ma - log(za))); }
                    klsum += kl; klmax = std::max(klmax, kl);
                    flips += (std::max_element(a, a + nv) - a) != tok;
                    eng->append(tok);
                }
                fclose(f);
                printf("teacher-forced over %d steps: mean KL %.3g, max KL %.3g, argmax flips %d\n", n, klsum / n, klmax, flips);
            }
            return 0;
        }
        if (aftergen >= 0) {
            // prompt minus 100 tokens, generate N, optional idle, then the full prompt (rollback + 100 new tokens)
            std::vector<int32_t> pre(toks.begin(), toks.end() - 100);
            eng->set_prompt(pre);
            for (int g = 0; g < aftergen; ++g) { const float * l = eng->logits(); eng->append((int32_t) (std::max_element(l, l + nv) - l)); }
            if (sleepms > 0) usleep(sleepms * 1000);
            for (int r = 0; r < 3; ++r) {
                double t1 = now_s();
                eng->set_prompt(r == 0 ? toks : (r == 1 ? pre : toks));
                auto tm = eng->timings();
                printf("after %d generated tokens%s: %d evaluated (%d cached) in %.1f ms\n", aftergen, sleepms ? " + idle" : "", tm.prompt_tokens, tm.prompt_cached, (now_s() - t1) * 1e3);
            }
            return 0;
        }
        if (cachetest) {
            auto run = [&](const std::vector<int32_t> & t, const char * what) {
                double t1 = now_s();
                eng->set_prompt(t);
                auto tm = eng->timings();
                printf("%-28s %6d evaluated, %6d cached, %8.1f ms\n", what, tm.prompt_tokens, tm.prompt_cached, (now_s() - t1) * 1e3);
                return std::vector<float>(eng->logits(), eng->logits() + nv);
            };
            auto greedyN = [&](int n) { for (int g = 0; g < n; ++g) { const float * l = eng->logits(); eng->append((int32_t) (std::max_element(l, l + nv) - l)); } };
            auto A = run(toks, "fresh");
            greedyN(40);
            auto B2 = run(toks, "same prompt after gen");
            compare(B2.data(), A.data(), nv, "  vs fresh");
            greedyN(40);
            // continuation: prompt + generated-like tail
            std::vector<int32_t> t2 = toks; for (int i = 0; i < 300; ++i) t2.push_back(toks[100 + i]);
            auto C = run(t2, "extended (rollback to end)");
            greedyN(10);
            // edit near the end: change the last 5 tokens
            std::vector<int32_t> t3(toks.begin(), toks.end() - 5); for (int i = 0; i < 12; ++i) t3.push_back(toks[200 + i]);
            auto Dd = run(t3, "edited tail (light ckpt)");
            // deep edit
            std::vector<int32_t> t4(toks.begin(), toks.begin() + toks.size() / 3); for (int i = 0; i < 50; ++i) t4.push_back(toks[300 + i]);
            auto E = run(t4, "deep edit");
            eng->reset(); auto C0 = run(t2, "fresh t2"); compare(C.data(), C0.data(), nv, "  extended vs fresh");
            eng->reset(); auto D0 = run(t3, "fresh t3"); compare(Dd.data(), D0.data(), nv, "  edited vs fresh");
            eng->reset(); auto E0 = run(t4, "fresh t4"); compare(E.data(), E0.data(), nv, "  deep vs fresh");
            return 0;
        }
        if (refPos > 1) {
            // per-position reference: feed the prompt one prefix at a time through append()
            eng->set_prompt({toks[0]});
            for (int i = 0; i < refPos; ++i) {
                if (i > 0) eng->append(toks[i]);
                if (i < 4 || i == refPos - 1 || i % 8 == 0) { char w[32]; snprintf(w, sizeof w, "pos %d", i); compare(eng->logits(), refAll.data() + (size_t) i * nv, nv, w); }
            }
            eng->reset();
        }
        for (int r = 0; r < repeat; ++r) {
            eng->reset();
            if (split > 0 && split < (int) toks.size()) {
                eng->set_prompt(std::vector<int32_t>(toks.begin(), toks.begin() + split));
            }
            t0 = now_s();
            eng->set_prompt(toks);
            double dt = now_s() - t0;
            auto tm = eng->timings();
            printf("prompt: %d tokens (%d cached) in %.1f ms -> %.1f tok/s\n", tm.prompt_tokens, tm.prompt_cached, dt * 1e3, tm.prompt_tokens / dt);
        }
        if (!refAll.empty()) compare(eng->logits(), refAll.data() + (size_t) (refPos - 1) * nv, nv, "last");
        if (greedy > 0) {
            std::vector<int32_t> gen;
            t0 = now_s();
            for (int g = 0; g < greedy; ++g) {
                const float * l = eng->logits();
                int best = (int) (std::max_element(l, l + nv) - l);
                gen.push_back(best);
                eng->append(best);
            }
            double dt = now_s() - t0;
            printf("greedy: %d tokens in %.1f ms -> %.2f tok/s\n ours:", greedy, dt * 1e3, greedy / dt);
            for (auto t : gen) printf(" %d", t);
            printf("\n");
            if (!refGen.empty()) {
                printf(" ref: "); for (auto t : refGen) printf(" %d", t); printf("\n");
                size_t same = 0; while (same < gen.size() && same < refGen.size() && gen[same] == refGen[same]) ++same;
                printf(" matching prefix: %zu / %zu\n", same, std::min(gen.size(), refGen.size()));
                if (same == refGen.size() && gen.size() == refGen.size() && !refLast.empty()) compare(eng->logits(), refLast.data(), nv, "after greedy");
            }
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
