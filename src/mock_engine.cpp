// TEST-ONLY stand-in for the Metal engine: implements ink::Engine::create() with a deterministic
// fake that "generates" a scripted token sequence, so the HTTP server can be tested end to end
// without a GPU.  Never link this together with engine.mm.
//
// The script is taken from the prompt itself: the text between the last "[[MOCK:" and ":MOCK]]" in
// the (detokenized) prompt is tokenized with parse_special=true and emitted token by token, followed
// by EOS.  "[[DELAY:<ms>]]" sets a per-token delay for that request.  Without a marker the script is
// $INK_MOCK_SCRIPT, or a short default reply.  Other knobs:
//   INK_MOCK_DELAY_MS          default per-token delay (ms)
//   INK_MOCK_PROMPT_DELAY_MS   delay per prompt batch (ms)
// logits() strongly favours the next scripted token (50 vs 0 everywhere else).
#include "engine.h"

#include "gguf.h"
#include "tokenizer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <thread>

namespace ink {

namespace {

double env_ms(const char * name, double def) {
    const char * v = getenv(name);
    return v && *v ? atof(v) : def;
}

void sleep_ms(double ms) {
    if (ms > 0) std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(ms));
}

class MockEngine : public Engine {
public:
    MockEngine(const GGUF & gguf, const EngineOptions & opt) : tok_(gguf), opt_(opt) {
        logits_.assign((size_t) tok_.n_vocab(), 0.0f);
        default_delay_ms_ = env_ms("INK_MOCK_DELAY_MS", 0);
        prompt_delay_ms_ = env_ms("INK_MOCK_PROMPT_DELAY_MS", 0);
    }

    int n_vocab() const override { return tok_.n_vocab(); }
    int max_ctx() const override { return opt_.max_ctx; }

    bool set_prompt(const std::vector<int32_t> & tokens, const std::function<bool(int, int)> & progress) override {
        Guard g(busy_);
        if (tokens.empty()) throw std::runtime_error("mock engine: empty prompt");
        if ((int) tokens.size() > opt_.max_ctx) throw std::runtime_error("mock engine: prompt exceeds context");
        auto t0 = std::chrono::steady_clock::now();
        size_t common = 0;
        while (common < seq_.size() && common < tokens.size() && seq_[common] == tokens[common]) common++;
        if (common == tokens.size()) common--;   // the last token is always re-evaluated to get logits
        seq_.resize(common);
        const int total = (int) (tokens.size() - common);
        bool ok = true;
        for (int done = 0; done < total;) {
            int n = std::min(std::max(opt_.batch, 1), total - done);
            sleep_ms(prompt_delay_ms_);
            seq_.insert(seq_.end(), tokens.begin() + (long) common + done, tokens.begin() + (long) common + done + n);
            done += n;
            if (done < total && progress && !progress(done, total)) {
                ok = false;
                break;
            }
        }
        timings_.prompt_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        timings_.prompt_cached = (int) common;
        timings_.prompt_tokens = (int) (seq_.size() - common);
        if (ok) load_script();
        return ok;
    }

    void append(int32_t token) override {
        Guard g(busy_);
        if ((int) seq_.size() >= opt_.max_ctx) throw std::runtime_error("mock engine: context overflow");
        sleep_ms(delay_ms_);
        seq_.push_back(token);
        if (pos_ < script_.size()) pos_++;
    }

    const float * logits() override {
        if (hot_ >= 0) logits_[(size_t) hot_] = 0.0f;
        hot_ = pos_ < script_.size() ? script_[pos_] : tok_.eos_id();
        logits_[(size_t) hot_] = 50.0f;
        return logits_.data();
    }

    int n_ctx_used() const override { return (int) seq_.size(); }
    const std::vector<int32_t> & tokens() const override { return seq_; }
    void reset() override {
        seq_.clear();
        script_.clear();
        pos_ = 0;
    }
    EngineTimings timings() const override { return timings_; }

private:
    // The real engine is not thread-safe; make overlapping calls a loud failure in tests.
    struct Guard {
        std::atomic<int> & busy;
        explicit Guard(std::atomic<int> & b) : busy(b) {
            if (busy.fetch_add(1) != 0) {
                fprintf(stderr, "mock engine: CONCURRENT ACCESS DETECTED\n");
                abort();
            }
        }
        ~Guard() { busy.fetch_sub(1); }
    };

    void load_script() {
        std::string prompt = tok_.decode(seq_, /*render_special=*/true);
        std::string script;
        const std::string open = "[[MOCK:", close = ":MOCK]]";
        size_t b = prompt.rfind(open);
        size_t e = b == std::string::npos ? b : prompt.find(close, b);
        if (e != std::string::npos) {
            script = prompt.substr(b + open.size(), e - b - open.size());
        } else if (const char * env = getenv("INK_MOCK_SCRIPT")) {
            script = env;
        } else {
            script = "<|message_model|><|content_text|>Hello from the mock engine.<|end_message|>";
        }
        delay_ms_ = default_delay_ms_;
        const std::string dopen = "[[DELAY:";
        size_t d = prompt.rfind(dopen);
        if (d != std::string::npos) delay_ms_ = atof(prompt.c_str() + d + dopen.size());
        script_ = tok_.encode(script, /*parse_special=*/true);
        pos_ = 0;
    }

    Tokenizer tok_;
    EngineOptions opt_;
    std::vector<int32_t> seq_;
    std::vector<int32_t> script_;
    size_t pos_ = 0;
    std::vector<float> logits_;
    int32_t hot_ = -1;
    double default_delay_ms_ = 0, prompt_delay_ms_ = 0, delay_ms_ = 0;
    EngineTimings timings_;
    std::atomic<int> busy_{0};
};

}  // namespace

std::unique_ptr<Engine> Engine::create(const GGUF & gguf, const EngineOptions & opt) {
    fprintf(stderr, "*** MOCK ENGINE: scripted output, no model is evaluated ***\n");
    return std::make_unique<MockEngine>(gguf, opt);
}

}  // namespace ink
