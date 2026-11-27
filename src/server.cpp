#include "server.h"

#include "chat.h"
#include "engine.h"
#include "gguf.h"
#include "sampler.h"
#include "tokenizer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <limits>
#include <csignal>
#include <cstdio>
#include <ctime>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>

#include <cpp-httplib/httplib.h>
#include <nlohmann/json.hpp>

namespace ink {

namespace {

using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------------------------
// errors / JSON helpers
// ---------------------------------------------------------------------------------------------

struct HttpError : std::runtime_error {
    int status;
    const char * type;
    HttpError(int status_, const char * type_, const std::string & msg) : std::runtime_error(msg), status(status_), type(type_) {}
};

[[noreturn]] void bad_request(const std::string & msg) { throw HttpError(400, "invalid_request_error", msg); }

std::string dumps(const ojson & j) { return j.dump(-1, ' ', false, ojson::error_handler_t::replace); }

ojson error_body(int status, const char * type, const std::string & msg) {
    return ojson{{"error", {{"message", msg}, {"type", type}, {"code", status}}}};
}

void set_json(httplib::Response & res, const ojson & j, int status = 200) {
    res.status = status;
    res.set_content(dumps(j), "application/json; charset=utf-8");
}

void set_error(httplib::Response & res, int status, const char * type, const std::string & msg) {
    set_json(res, error_body(status, type, msg), status);
}

// Typed access to optional request fields: returns false when absent or null, 400 on a wrong type.
bool get_number(const ojson & body, const char * key, double & out) {
    auto it = body.find(key);
    if (it == body.end() || it->is_null()) return false;
    if (!it->is_number()) bad_request(std::string("'") + key + "' must be a number");
    out = it->get<double>();
    return true;
}

bool get_int(const ojson & body, const char * key, int64_t & out) {
    auto it = body.find(key);
    if (it == body.end() || it->is_null()) return false;
    if (it->is_number_integer()) {
        out = it->is_number_unsigned() ? (int64_t) std::min<uint64_t>(it->get<uint64_t>(), (uint64_t) INT64_MAX) : it->get<int64_t>();
        return true;
    }
    if (it->is_number_float()) {
        double d = it->get<double>();
        if (d == std::floor(d) && std::fabs(d) < 9e18) {
            out = (int64_t) d;
            return true;
        }
    }
    bad_request(std::string("'") + key + "' must be an integer");
}

bool get_bool(const ojson & body, const char * key, bool def) {
    auto it = body.find(key);
    if (it == body.end() || it->is_null()) return def;
    if (!it->is_boolean()) bad_request(std::string("'") + key + "' must be a boolean");
    return it->get<bool>();
}

std::string random_id(const char * prefix) {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    static const char alphabet[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::string id = prefix;
    for (int i = 0; i < 24; i++) id += alphabet[rng() % 62];
    return id;
}

// ---------------------------------------------------------------------------------------------
// stop strings
// ---------------------------------------------------------------------------------------------

// Streams text through while withholding any tail that could be the beginning of a stop string.
class StopFilter {
public:
    explicit StopFilter(std::vector<std::string> stops) : stops_(std::move(stops)) {}

    bool active() const { return !stops_.empty(); }

    // Appends the emit-safe part of held+text to `out`.  Returns true when a stop string was found
    // (text before it is emitted, the stop string and everything after are dropped).
    bool push(std::string_view text, std::string & out) {
        held_.append(text);
        size_t best = std::string::npos;
        for (const auto & s : stops_) best = std::min(best, held_.find(s));
        if (best != std::string::npos) {
            out.append(held_, 0, best);
            held_.clear();
            return true;
        }
        size_t keep = 0;
        for (const auto & s : stops_) {
            for (size_t len = std::min(s.size() - 1, held_.size()); len > keep; len--) {
                if (held_.compare(held_.size() - len, len, s, 0, len) == 0) {
                    keep = len;
                    break;
                }
            }
        }
        out.append(held_, 0, held_.size() - keep);
        held_.erase(0, held_.size() - keep);
        return false;
    }

    void flush(std::string & out) {
        out += held_;
        held_.clear();
    }

private:
    std::vector<std::string> stops_;
    std::string held_;
};

// ---------------------------------------------------------------------------------------------
// generation
// ---------------------------------------------------------------------------------------------

struct Context {
    Engine & engine;
    Tokenizer & tok;
    ServerOptions opt;
    SamplerParams default_sampling;
    std::timed_mutex engine_mutex;   // the engine holds one sequence: requests run one at a time
    int n_vocab;
    int max_ctx;
    int32_t eos;
};

struct GenParams {
    std::vector<int32_t> prompt;
    SamplerParams sampling;
    int max_tokens = -1;             // -1: until EOS / context limit
    std::vector<std::string> stop;
    bool chat = true;                // parse the output as chat messages; false: raw text
    bool parse_tools = false;
    int top_logprobs = -1;           // >= 0: report per-token logprobs with this many alternatives
    bool stream = false;
    bool include_usage = false;
};

struct GenStats {
    int n_prompt = 0;
    int n_cached = 0;
    int n_prompt_eval = 0;
    double prompt_ms = 0;
    int n_gen = 0;
    double gen_ms = 0;
    const char * finish = "stop";    // "stop" | "length" | "tool_calls"
    bool cancelled = false;
};

using CancelFn = std::function<bool()>;
// Receives each non-empty delta (and, when requested, the logprob entries of the tokens generated
// since the previous call).  Returns false if the client is gone.
using EmitFn = std::function<bool(ChatDelta & delta, ojson & logprobs)>;

std::atomic<bool> g_shutdown{false};

ojson logprob_entry(const Tokenizer & tok, int32_t id, float logprob) {
    const std::string & piece = tok.piece(id);
    ojson bytes = ojson::array();
    for (unsigned char c : piece) bytes.push_back((int) c);
    return ojson{{"token", piece}, {"logprob", logprob}, {"bytes", std::move(bytes)}};
}

GenStats generate(Context & ctx, const GenParams & gp, const CancelFn & cancelled, const EmitFn & emit) {
    GenStats st;
    st.n_prompt = (int) gp.prompt.size();

    // wait in line for the engine, giving up if the client goes away meanwhile
    std::unique_lock<std::timed_mutex> lock(ctx.engine_mutex, std::defer_lock);
    while (!lock.try_lock_for(std::chrono::milliseconds(50))) {
        if (cancelled()) {
            st.cancelled = true;
            return st;
        }
    }
    if (cancelled()) {
        st.cancelled = true;
        return st;
    }

    Engine & engine = ctx.engine;
    const bool prompt_ok = engine.set_prompt(gp.prompt, [&](int, int) { return !cancelled(); });
    const EngineTimings tm = engine.timings();
    st.prompt_ms = tm.prompt_ms;
    st.n_prompt_eval = tm.prompt_tokens;
    st.n_cached = tm.prompt_cached;
    if (!prompt_ok) {
        st.cancelled = true;
        return st;
    }

    const auto t_start = Clock::now();
    Sampler sampler(gp.sampling, ctx.n_vocab);
    ChatOutputParser parser(ctx.tok, gp.parse_tools);
    StopFilter stop(gp.stop);
    ChatDelta delta;
    ojson logprobs = ojson::array();
    std::vector<TokenProb> top;
    std::string raw_pending;   // raw mode: bytes of an unfinished UTF-8 sequence
    std::string filtered;

    int limit = ctx.max_ctx - st.n_prompt;
    if (gp.max_tokens >= 0) limit = std::min(limit, gp.max_tokens);

    // routes delta.content through the stop filter; true when a stop string was hit
    auto apply_stop = [&]() {
        if (!stop.active() || delta.content.empty()) return false;
        filtered.clear();
        bool hit = stop.push(delta.content, filtered);
        delta.content.swap(filtered);
        return hit;
    };
    auto send = [&]() {
        if (delta.empty() && logprobs.empty()) return true;
        bool ok = emit(delta, logprobs);
        logprobs.clear();
        return ok;
    };

    st.finish = "length";
    bool stop_hit = false;
    while (st.n_gen < limit) {
        const float * logits = engine.logits();
        const int32_t id = sampler.sample(logits);
        if (id == ctx.eos) {
            st.finish = "stop";
            break;
        }
        delta.clear();
        bool more = true;
        if (gp.chat) {
            more = parser.feed(id, delta);
        } else {
            const std::string & piece = ctx.tok.piece(id);
            raw_pending += piece;
            size_t n = utf8_complete_len(raw_pending);
            delta.content.assign(raw_pending, 0, n);
            raw_pending.erase(0, n);
        }
        if (!more) {   // the model began a non-assistant message: end of turn
            st.finish = "stop";
            break;
        }
        st.n_gen++;
        if (gp.top_logprobs >= 0) {
            float lp = 0;
            sampler.top_logprobs(logits, gp.top_logprobs, top, id, &lp);
            ojson entry = logprob_entry(ctx.tok, id, lp);
            ojson alts = ojson::array();
            for (const TokenProb & t : top) alts.push_back(logprob_entry(ctx.tok, t.id, t.logprob));
            entry["top_logprobs"] = std::move(alts);
            logprobs.push_back(std::move(entry));
        }
        stop_hit = apply_stop();
        if (!send()) {
            st.cancelled = true;
            break;
        }
        if (stop_hit) {
            st.finish = "stop";
            break;
        }
        if (st.n_gen >= limit) break;
        if (cancelled()) {
            st.cancelled = true;
            break;
        }
        engine.append(id);
    }

    if (!st.cancelled && !stop_hit) {
        // flush an unterminated message / held-back text
        delta.clear();
        if (gp.chat) parser.finish(delta);
        else if (!raw_pending.empty()) delta.content = "\xEF\xBF\xBD";
        if (apply_stop()) {
            st.finish = "stop";
        } else if (stop.active()) {
            stop.flush(delta.content);
        }
        if (!send()) st.cancelled = true;
    }
    if (gp.chat && parser.n_tool_calls() > 0 && std::strcmp(st.finish, "stop") == 0) st.finish = "tool_calls";
    st.gen_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
    return st;
}

ojson usage_json(const GenStats & st) {
    return ojson{{"prompt_tokens", st.n_prompt},
                 {"completion_tokens", st.n_gen},
                 {"total_tokens", st.n_prompt + st.n_gen},
                 {"prompt_tokens_details", {{"cached_tokens", st.n_cached}}}};
}

ojson timings_json(const GenStats & st) {
    auto per_second = [](int n, double ms) { return ms > 0 ? n * 1000.0 / ms : 0.0; };
    return ojson{{"prompt_n", st.n_prompt_eval},
                 {"prompt_ms", st.prompt_ms},
                 {"prompt_per_second", per_second(st.n_prompt_eval, st.prompt_ms)},
                 {"predicted_n", st.n_gen},
                 {"predicted_ms", st.gen_ms},
                 {"predicted_per_second", per_second(st.n_gen, st.gen_ms)},
                 {"cache_n", st.n_cached}};
}

void log_request(const char * path, const GenStats & st) {
    auto per_second = [](int n, double ms) { return ms > 0 ? n * 1000.0 / ms : 0.0; };
    char ts[32];
    time_t now = time(nullptr);
    struct tm tmv;
    localtime_r(&now, &tmv);
    strftime(ts, sizeof(ts), "%H:%M:%S", &tmv);
    fprintf(stderr, "[%s] POST %s | prompt %d tok (cached %d, evaluated %d in %.1f ms, %.1f tok/s) | generated %d tok in %.1f ms (%.1f tok/s) | finish=%s\n",
            ts, path, st.n_prompt, st.n_cached, st.n_prompt_eval, st.prompt_ms, per_second(st.n_prompt_eval, st.prompt_ms), st.n_gen,
            st.gen_ms, per_second(st.n_gen, st.gen_ms), st.cancelled ? "cancelled" : st.finish);
}

// ---------------------------------------------------------------------------------------------
// request parsing
// ---------------------------------------------------------------------------------------------

ojson parse_body(const httplib::Request & req) {
    ojson body = ojson::parse(req.body, nullptr, /*allow_exceptions=*/false);
    if (body.is_discarded()) bad_request("request body is not valid JSON");
    if (!body.is_object()) bad_request("request body must be a JSON object");
    return body;
}

void parse_common(const Context & ctx, const ojson & body, GenParams & gp) {
    SamplerParams & sp = gp.sampling;
    sp = ctx.default_sampling;
    double d;
    int64_t i;
    if (get_number(body, "temperature", d)) sp.temperature = (float) d;
    if (get_number(body, "top_p", d)) sp.top_p = (float) d;
    if (get_number(body, "min_p", d)) sp.min_p = (float) d;
    if (get_int(body, "top_k", i)) sp.top_k = (int) std::clamp<int64_t>(i, 0, INT32_MAX);
    if (get_int(body, "seed", i) && i >= 0) {
        sp.has_seed = true;
        sp.seed = (uint64_t) i;
    }

    auto lb = body.find("logit_bias");
    if (lb != body.end() && !lb->is_null()) {
        auto add = [&](int64_t id, const ojson & v) {
            float bias;
            if (v.is_number()) bias = v.get<float>();
            else if (v.is_boolean() && !v.get<bool>()) bias = -std::numeric_limits<float>::infinity();   // llama.cpp: false bans the token
            else bad_request("'logit_bias' values must be numbers");
            if (id < 0 || id >= ctx.n_vocab) bad_request("'logit_bias' token id " + std::to_string(id) + " is out of range");
            sp.logit_bias.emplace_back((int32_t) id, bias);
        };
        if (lb->is_object()) {   // OpenAI: {"1234": -100}
            for (auto it = lb->begin(); it != lb->end(); ++it) {
                char * end = nullptr;
                long long id = strtoll(it.key().c_str(), &end, 10);
                if (it.key().empty() || *end) bad_request("'logit_bias' keys must be token ids");
                add(id, it.value());
            }
        } else if (lb->is_array()) {   // llama.cpp: [[1234, -100], ...]
            for (const auto & e : *lb) {
                if (!e.is_array() || e.size() != 2 || !e[0].is_number_integer()) bad_request("'logit_bias' entries must be [token_id, bias]");
                add(e[0].get<int64_t>(), e[1]);
            }
        } else {
            bad_request("'logit_bias' must be an object or an array");
        }
    }

    gp.max_tokens = ctx.opt.default_max_tokens;
    for (const char * key : {"n_predict", "max_tokens", "max_completion_tokens"}) {
        if (get_int(body, key, i)) gp.max_tokens = i < 0 ? -1 : (int) std::min<int64_t>(i, INT32_MAX);
    }

    auto stop = body.find("stop");
    if (stop != body.end() && !stop->is_null()) {
        if (stop->is_string()) {
            gp.stop.push_back(stop->get<std::string>());
        } else if (stop->is_array()) {
            for (const auto & s : *stop) {
                if (!s.is_string()) bad_request("'stop' must be a string or an array of strings");
                gp.stop.push_back(s.get<std::string>());
            }
        } else {
            bad_request("'stop' must be a string or an array of strings");
        }
        gp.stop.erase(std::remove(gp.stop.begin(), gp.stop.end(), std::string()), gp.stop.end());
    }

    gp.stream = get_bool(body, "stream", false);
    auto so = body.find("stream_options");
    if (so != body.end() && so->is_object()) gp.include_usage = get_bool(*so, "include_usage", false);
}

void check_context(const Context & ctx, GenParams & gp) {
    const long n = (long) gp.prompt.size();
    if (n == 0) bad_request("the prompt is empty");
    if (n >= ctx.max_ctx)
        bad_request("the prompt is too long: " + std::to_string(n) + " tokens, but the context size is " + std::to_string(ctx.max_ctx) + " tokens");
    // clients often send a very large max_tokens: clamp it to the room left in the context
    if (gp.max_tokens > 0 && n + gp.max_tokens > ctx.max_ctx) gp.max_tokens = (int) (ctx.max_ctx - n);
}

// The template can render an image placeholder, but this server has no vision encoder.
void reject_media(const ojson & body) {
    auto msgs = body.find("messages");
    if (msgs == body.end() || !msgs->is_array()) return;
    for (const auto & m : *msgs) {
        if (!m.is_object()) continue;
        auto c = m.find("content");
        if (c == m.end() || !c->is_array()) continue;
        for (const auto & part : *c) {
            if (!part.is_object()) continue;
            auto t = part.find("type");
            if (t == part.end() || !t->is_string()) continue;
            const std::string & type = t->get_ref<const std::string &>();
            if (type == "image" || type == "input_image" || type == "image_url") bad_request("image input is not supported by this server");
        }
    }
}

std::vector<int32_t> tokenize_parts(const Tokenizer & tok, const std::vector<PromptPart> & parts) {
    std::vector<int32_t> ids;
    for (const auto & p : parts) {
        // request-derived text must not be able to forge control tokens
        std::vector<int32_t> piece = tok.encode(p.text, /*parse_special=*/!p.is_input);
        ids.insert(ids.end(), piece.begin(), piece.end());
    }
    return ids;
}

std::vector<PromptPart> prepare_chat(ojson & body, bool & parse_tools) {
    auto msgs = body.find("messages");
    if (msgs == body.end() || !msgs->is_array()) bad_request("'messages' must be an array");
    parse_tools = canonicalize_chat_request(body);
    reject_media(body);
    return render_chat_parts(body);
}

// ---------------------------------------------------------------------------------------------
// response assembly
// ---------------------------------------------------------------------------------------------

struct ResponseMeta {
    std::string id;
    std::time_t created;
    std::string model;
    std::string path;
};

ojson tool_calls_json(const std::vector<ToolCallDelta> & calls, bool with_index) {
    ojson arr = ojson::array();
    for (const auto & tc : calls) {
        ojson j = ojson::object();
        if (with_index) j["index"] = tc.index;
        j["id"] = tc.id;
        j["type"] = "function";
        j["function"] = {{"name", tc.name}, {"arguments", tc.arguments}};
        arr.push_back(std::move(j));
    }
    return arr;
}

// Runs a blocking (non-streaming) request and fills `res`.
void run_blocking(Context & ctx, const GenParams & gp, const ResponseMeta & meta, const httplib::Request & req, httplib::Response & res) {
    ChatDelta all;
    ojson all_logprobs = ojson::array();
    GenStats st = generate(
        ctx, gp, [&]() { return g_shutdown.load(std::memory_order_relaxed) || req.is_connection_closed(); },
        [&](ChatDelta & d, ojson & lp) {
            all.reasoning += d.reasoning;
            all.content += d.content;
            for (auto & tc : d.tool_calls) all.tool_calls.push_back(std::move(tc));
            for (auto & e : lp) all_logprobs.push_back(std::move(e));
            return true;
        });
    log_request(meta.path.c_str(), st);
    if (st.cancelled) {
        set_error(res, 499, "request_cancelled", "client closed the connection");
        return;
    }

    ojson choice = ojson::object();
    choice["index"] = 0;
    if (gp.chat) {
        ojson msg = ojson::object();
        msg["role"] = "assistant";
        if (all.content.empty() && !all.tool_calls.empty()) msg["content"] = nullptr;
        else msg["content"] = all.content;
        if (!all.reasoning.empty()) msg["reasoning_content"] = all.reasoning;
        if (!all.tool_calls.empty()) msg["tool_calls"] = tool_calls_json(all.tool_calls, false);
        choice["message"] = std::move(msg);
        choice["logprobs"] = gp.top_logprobs >= 0 ? ojson{{"content", std::move(all_logprobs)}} : ojson(nullptr);
    } else {
        choice["text"] = all.content;
        choice["logprobs"] = nullptr;
    }
    choice["finish_reason"] = st.finish;

    ojson out = ojson::object();
    out["id"] = meta.id;
    out["object"] = gp.chat ? "chat.completion" : "text_completion";
    out["created"] = (int64_t) meta.created;
    out["model"] = meta.model;
    out["choices"] = ojson::array({std::move(choice)});
    out["usage"] = usage_json(st);
    out["timings"] = timings_json(st);
    set_json(res, out);
}

// Installs a chunked SSE provider that runs the whole generation, writing one event per delta.
void run_streaming(Context & ctx, std::shared_ptr<GenParams> gp, const ResponseMeta & meta, httplib::Response & res) {
    res.set_header("Cache-Control", "no-cache");
    res.set_header("X-Accel-Buffering", "no");
    auto started = std::make_shared<bool>(false);
    res.set_chunked_content_provider("text/event-stream", [&ctx, gp, meta, started](size_t, httplib::DataSink & sink) {
        if (*started) {   // generation already ran in the first invocation
            sink.done();
            return true;
        }
        *started = true;

        const char * object = gp->chat ? "chat.completion.chunk" : "text_completion";
        const std::string head = "data: {\"id\":" + dumps(meta.id) + ",\"object\":\"" + object + "\",\"created\":" +
                                 std::to_string((int64_t) meta.created) + ",\"model\":" + dumps(meta.model);
        const std::string choice_head = head + ",\"choices\":[{\"index\":0,";
        std::string buf;
        bool alive = true;
        auto write = [&](const std::string & s) {
            if (alive && !sink.write(s.data(), s.size())) alive = false;
            return alive;
        };
        // one event: choice body is `"delta":{...}` (chat) or `"text":"..."` (completions)
        auto write_choice = [&](const std::string & payload, const ojson * logprobs, const char * finish, const std::string & extra) {
            buf.assign(choice_head);
            buf += payload;
            if (logprobs) {
                buf += ",\"logprobs\":";
                buf += dumps(*logprobs);
            } else if (!gp->chat) {
                buf += ",\"logprobs\":null";
            }
            buf += ",\"finish_reason\":";
            if (finish) {
                buf += '"';
                buf += finish;
                buf += '"';
            } else {
                buf += "null";
            }
            buf += "}]";
            buf += extra;
            buf += "}\n\n";
            return write(buf);
        };

        bool first = true;
        ojson d = ojson::object();
        try {
            GenStats st = generate(
                ctx, *gp, [&]() { return g_shutdown.load(std::memory_order_relaxed) || !alive || !sink.is_writable(); },
                [&](ChatDelta & delta, ojson & lp) {
                    ojson lp_obj;
                    const ojson * lp_ptr = nullptr;
                    if (gp->top_logprobs >= 0 && gp->chat) {
                        lp_obj = ojson{{"content", std::move(lp)}};
                        lp = ojson::array();
                        lp_ptr = &lp_obj;
                    }
                    if (!gp->chat) return write_choice("\"text\":" + dumps(delta.content), nullptr, nullptr, "");
                    d.clear();
                    if (first) {
                        d["role"] = "assistant";
                        first = false;
                    }
                    if (!delta.reasoning.empty()) d["reasoning_content"] = delta.reasoning;
                    if (!delta.content.empty()) d["content"] = delta.content;
                    if (!delta.tool_calls.empty()) d["tool_calls"] = tool_calls_json(delta.tool_calls, true);
                    return write_choice("\"delta\":" + dumps(d), lp_ptr, nullptr, "");
                });
            log_request(meta.path.c_str(), st);
            if (st.cancelled || !alive) return false;

            const std::string timings = ",\"timings\":" + dumps(timings_json(st));
            if (gp->chat) {
                // a generation that produced nothing still owes the client the role
                write_choice(first ? "\"delta\":{\"role\":\"assistant\",\"content\":\"\"}" : "\"delta\":{}", nullptr, st.finish, timings);
            } else {
                write_choice("\"text\":\"\"", nullptr, st.finish, timings);
            }
            if (gp->include_usage) write(head + ",\"choices\":[],\"usage\":" + dumps(usage_json(st)) + timings + "}\n\n");
        } catch (const std::exception & e) {
            fprintf(stderr, "error during generation: %s\n", e.what());
            write("data: " + dumps(error_body(500, "server_error", e.what())) + "\n\n");
        }
        if (!write("data: [DONE]\n\n")) return false;
        sink.done();
        return true;
    });
}

void dispatch(Context & ctx, std::shared_ptr<GenParams> gp, const char * id_prefix, const httplib::Request & req, httplib::Response & res) {
    check_context(ctx, *gp);
    ResponseMeta meta{random_id(id_prefix), std::time(nullptr), ctx.opt.alias, req.path};
    if (gp->stream) run_streaming(ctx, std::move(gp), meta, res);
    else run_blocking(ctx, *gp, meta, req, res);
}

// ---------------------------------------------------------------------------------------------
// endpoints
// ---------------------------------------------------------------------------------------------

void handle_chat(Context & ctx, const httplib::Request & req, httplib::Response & res) {
    ojson body = parse_body(req);
    auto gp = std::make_shared<GenParams>();
    parse_common(ctx, body, *gp);
    gp->chat = true;
    gp->prompt = tokenize_parts(ctx.tok, prepare_chat(body, gp->parse_tools));
    if (get_bool(body, "logprobs", false)) {
        int64_t n = 0;
        get_int(body, "top_logprobs", n);
        gp->top_logprobs = (int) std::clamp<int64_t>(n, 0, 20);
    }
    dispatch(ctx, std::move(gp), "chatcmpl-", req, res);
}

void handle_completions(Context & ctx, const httplib::Request & req, httplib::Response & res) {
    ojson body = parse_body(req);
    auto gp = std::make_shared<GenParams>();
    parse_common(ctx, body, *gp);
    gp->chat = false;

    auto p = body.find("prompt");
    if (p == body.end()) bad_request("'prompt' is required");
    auto all_ints = [](const ojson & a) {
        return std::all_of(a.begin(), a.end(), [](const ojson & e) { return e.is_number_integer(); });
    };
    if (p->is_string()) {
        gp->prompt = ctx.tok.encode(p->get_ref<const std::string &>(), /*parse_special=*/true);
    } else if (p->is_array() && !p->empty() && all_ints(*p)) {
        for (const auto & e : *p) {
            int64_t id = e.get<int64_t>();
            if (id < 0 || id >= ctx.n_vocab) bad_request("'prompt' token id " + std::to_string(id) + " is out of range");
            gp->prompt.push_back((int32_t) id);
        }
    } else if (p->is_array() && p->size() == 1 && (*p)[0].is_string()) {
        gp->prompt = ctx.tok.encode((*p)[0].get_ref<const std::string &>(), true);
    } else {
        bad_request("'prompt' must be a string or an array of token ids");
    }
    dispatch(ctx, std::move(gp), "cmpl-", req, res);
}

void handle_tokenize(Context & ctx, const httplib::Request & req, httplib::Response & res) {
    ojson body = parse_body(req);
    auto c = body.find("content");
    if (c == body.end() || !c->is_string()) bad_request("'content' must be a string");
    std::vector<int32_t> ids = ctx.tok.encode(c->get_ref<const std::string &>(), get_bool(body, "parse_special", true));
    ojson out = ojson::object();
    if (get_bool(body, "with_pieces", false)) {
        ojson arr = ojson::array();
        for (int32_t id : ids) arr.push_back({{"id", id}, {"piece", ctx.tok.piece(id)}});
        out["tokens"] = std::move(arr);
    } else {
        out["tokens"] = ids;
    }
    set_json(res, out);
}

void handle_detokenize(Context & ctx, const httplib::Request & req, httplib::Response & res) {
    ojson body = parse_body(req);
    auto t = body.find("tokens");
    if (t == body.end() || !t->is_array()) bad_request("'tokens' must be an array of token ids");
    std::vector<int32_t> ids;
    for (const auto & e : *t) {
        if (!e.is_number_integer()) bad_request("'tokens' must be an array of token ids");
        int64_t id = e.get<int64_t>();
        if (id < 0 || id >= ctx.n_vocab) bad_request("token id " + std::to_string(id) + " is out of range");
        ids.push_back((int32_t) id);
    }
    set_json(res, ojson{{"content", ctx.tok.decode(ids, /*render_special=*/true)}});
}

void handle_apply_template(Context & ctx, const httplib::Request & req, httplib::Response & res) {
    ojson body = parse_body(req);
    bool parse_tools = false;
    std::vector<PromptPart> parts = prepare_chat(body, parse_tools);
    std::string prompt;
    for (const auto & p : parts) prompt += p.text;
    ojson out = ojson{{"prompt", prompt}};
    if (get_bool(body, "tokenize", false)) out["tokens"] = tokenize_parts(ctx.tok, parts);
    set_json(res, out);
}

httplib::Server * g_server = nullptr;

void on_signal(int) {
    if (g_shutdown.exchange(true)) _exit(1);   // second signal: do not wait for a running request
    if (g_server) g_server->stop();
}

}  // namespace

int run_server(Engine & engine, Tokenizer & tokenizer, const GGUF & gguf, const ServerOptions & options) {
    Context ctx{engine, tokenizer, options, SamplerParams::from_gguf(gguf), {}, engine.n_vocab(), engine.max_ctx(), tokenizer.eos_id()};

    httplib::Server svr;
    svr.new_task_queue = [] { return new httplib::ThreadPool(32); };   // queued requests each hold a thread
    svr.set_tcp_nodelay(true);
    svr.set_read_timeout(600);
    svr.set_write_timeout(600);
    svr.set_payload_max_length(1024ull * 1024 * 1024);
    svr.set_default_headers({{"Access-Control-Allow-Origin", "*"}});

    svr.set_pre_routing_handler([&ctx](const httplib::Request & req, httplib::Response & res) {
        if (req.method == "OPTIONS") {
            res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            res.set_header("Access-Control-Allow-Headers", "*");
            res.set_header("Access-Control-Max-Age", "86400");
            res.status = 204;
            return httplib::Server::HandlerResponse::Handled;
        }
        if (!ctx.opt.api_key.empty() && req.path != "/health" && req.path != "/v1/health") {
            std::string auth = req.get_header_value("Authorization");
            const std::string bearer = "Bearer ";
            // compare without an early exit, so the response time does not reveal how much of the key matched
            auto same = [](const std::string & a, const std::string & b) {
                unsigned char diff = a.size() == b.size() ? 0 : 1;
                for (size_t i = 0; i < a.size(); ++i) diff |= (unsigned char) (a[i] ^ b[i % (b.empty() ? 1 : b.size())]);
                return diff == 0;
            };
            bool ok = auth.compare(0, bearer.size(), bearer) == 0 && same(auth.substr(bearer.size()), ctx.opt.api_key);
            if (!ok) ok = same(req.get_header_value("X-Api-Key"), ctx.opt.api_key);
            if (!ok) {
                set_error(res, 401, "authentication_error", "invalid or missing API key");
                return httplib::Server::HandlerResponse::Handled;
            }
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    svr.set_error_handler([](const httplib::Request & req, httplib::Response & res) {
        if (!res.body.empty()) return;   // already a JSON error from a handler
        if (res.status == 404) set_error(res, 404, "not_found_error", "no such endpoint: " + req.method + " " + req.path);
        else set_error(res, res.status, res.status >= 500 ? "server_error" : "invalid_request_error", "HTTP error " + std::to_string(res.status));
    });

    using Handler = void (*)(Context &, const httplib::Request &, httplib::Response &);
    auto wrap = [&ctx](Handler h) {
        return [&ctx, h](const httplib::Request & req, httplib::Response & res) {
            try {
                h(ctx, req, res);
            } catch (const HttpError & e) {
                set_error(res, e.status, e.type, e.what());
            } catch (const ChatError & e) {
                set_error(res, 400, "invalid_request_error", e.what());
            } catch (const nlohmann::json::exception & e) {
                set_error(res, 400, "invalid_request_error", e.what());
            } catch (const std::exception & e) {
                fprintf(stderr, "internal error in %s: %s\n", req.path.c_str(), e.what());
                set_error(res, 500, "server_error", e.what());
            }
        };
    };

    auto health = [](const httplib::Request &, httplib::Response & res) { set_json(res, ojson{{"status", "ok"}}); };
    svr.Get("/health", health);
    svr.Get("/v1/health", health);

    auto models = [&ctx](const httplib::Request &, httplib::Response & res) {
        ojson model = {{"id", ctx.opt.alias}, {"object", "model"}, {"created", (int64_t) std::time(nullptr)}, {"owned_by", "inkling"},
                       {"meta", {{"n_vocab", ctx.n_vocab}, {"n_ctx", ctx.max_ctx}}}};
        set_json(res, ojson{{"object", "list"}, {"data", ojson::array({model})}});
    };
    svr.Get("/v1/models", models);
    svr.Get("/models", models);

    svr.Post("/v1/chat/completions", wrap(handle_chat));
    svr.Post("/chat/completions", wrap(handle_chat));
    svr.Post("/v1/completions", wrap(handle_completions));
    svr.Post("/completions", wrap(handle_completions));
    svr.Post("/tokenize", wrap(handle_tokenize));
    svr.Post("/detokenize", wrap(handle_detokenize));
    svr.Post("/apply-template", wrap(handle_apply_template));

    int port = options.port;
    if (port == 0) {
        port = svr.bind_to_any_port(options.host);
        if (port < 0) {
            fprintf(stderr, "error: cannot bind to %s\n", options.host.c_str());
            return 1;
        }
    } else if (!svr.bind_to_port(options.host, port)) {
        fprintf(stderr, "error: cannot bind to %s:%d\n", options.host.c_str(), port);
        return 1;
    }

    g_server = &svr;
    struct sigaction sa {};
    sa.sa_handler = on_signal;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, nullptr);
    sigaction(SIGTERM, &sa, nullptr);
    signal(SIGPIPE, SIG_IGN);

    printf("listening on http://%s:%d\n", options.host.c_str(), port);
    fflush(stdout);
    const bool ok = svr.listen_after_bind();
    g_server = nullptr;
    return ok || g_shutdown ? 0 : 1;
}

}  // namespace ink
