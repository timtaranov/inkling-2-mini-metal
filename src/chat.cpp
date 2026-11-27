#include "chat.h"

#include "tokenizer.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <random>

#include <nlohmann/json.hpp>

namespace ink {

// ---------------------------------------------------------------------------------------------
// Python-compatible formatting
// ---------------------------------------------------------------------------------------------

std::string py_float_repr(double v) {
    if (std::isnan(v)) return "nan";
    if (std::isinf(v)) return v < 0 ? "-inf" : "inf";
    if (v == 0) return std::signbit(v) ? "-0.0" : "0.0";

    // shortest round-trip digits, as "[-]d[.ddd]e[+-]XX"
    char buf[64];
    auto res = std::to_chars(buf, buf + sizeof(buf), v, std::chars_format::scientific);
    std::string_view s(buf, (size_t) (res.ptr - buf));
    std::string out;
    if (s[0] == '-') {
        out += '-';
        s.remove_prefix(1);
    }
    size_t epos = s.find('e');
    std::string digits;
    for (char c : s.substr(0, epos)) {
        if (c != '.') digits += c;
    }
    int exp10 = 0;
    {
        std::string_view e = s.substr(epos + 1);
        if (!e.empty() && e[0] == '+') e.remove_prefix(1);
        std::from_chars(e.data(), e.data() + e.size(), exp10);
    }
    int nd = (int) digits.size();
    int decpt = exp10 + 1;   // value = 0.DIGITS * 10^decpt
    if (decpt > -4 && decpt <= 16) {
        if (decpt <= 0) {
            out += "0.";
            out.append((size_t) -decpt, '0');
            out += digits;
        } else if (decpt >= nd) {
            out += digits;
            out.append((size_t) (decpt - nd), '0');
            out += ".0";
        } else {
            out.append(digits, 0, (size_t) decpt);
            out += '.';
            out.append(digits, (size_t) decpt, std::string::npos);
        }
    } else {
        out += digits[0];
        if (nd > 1) {
            out += '.';
            out.append(digits, 1, std::string::npos);
        }
        out += 'e';
        out += exp10 < 0 ? '-' : '+';
        int a = std::abs(exp10);
        if (a < 10) out += '0';
        out += std::to_string(a);
    }
    return out;
}

namespace {

void py_json_string(const std::string & s, std::string & out) {
    static const char * hex = "0123456789abcdef";
    out += '"';
    size_t run = 0;
    for (size_t i = 0; i < s.size(); i++) {
        unsigned char c = (unsigned char) s[i];
        const char * esc = nullptr;
        char ubuf[7];
        switch (c) {
            case '"': esc = "\\\""; break;
            case '\\': esc = "\\\\"; break;
            case '\n': esc = "\\n"; break;
            case '\r': esc = "\\r"; break;
            case '\t': esc = "\\t"; break;
            case '\b': esc = "\\b"; break;
            case '\f': esc = "\\f"; break;
            default:
                if (c < 0x20) {
                    ubuf[0] = '\\'; ubuf[1] = 'u'; ubuf[2] = '0'; ubuf[3] = '0';
                    ubuf[4] = hex[c >> 4]; ubuf[5] = hex[c & 15]; ubuf[6] = 0;
                    esc = ubuf;
                }
        }
        if (esc) {
            out.append(s, run, i - run);
            out += esc;
            run = i + 1;
        }
    }
    out.append(s, run, std::string::npos);
    out += '"';
}

void py_json_number(const ojson & v, std::string & out) {
    if (v.is_number_unsigned()) out += std::to_string(v.get<uint64_t>());
    else if (v.is_number_integer()) out += std::to_string(v.get<int64_t>());
    else out += py_float_repr(v.get<double>());
}

void py_json_write(const ojson & v, bool sort_keys, const char * item_sep, const char * key_sep, std::string & out) {
    switch (v.type()) {
        case ojson::value_t::null: out += "null"; break;
        case ojson::value_t::boolean: out += v.get<bool>() ? "true" : "false"; break;
        case ojson::value_t::string: py_json_string(v.get_ref<const std::string &>(), out); break;
        case ojson::value_t::number_integer:
        case ojson::value_t::number_unsigned:
        case ojson::value_t::number_float: py_json_number(v, out); break;
        case ojson::value_t::array: {
            out += '[';
            bool first = true;
            for (const auto & e : v) {
                if (!first) out += item_sep;
                first = false;
                py_json_write(e, sort_keys, item_sep, key_sep, out);
            }
            out += ']';
            break;
        }
        case ojson::value_t::object: {
            out += '{';
            bool first = true;
            auto put = [&](const std::string & k, const ojson & val) {
                if (!first) out += item_sep;
                first = false;
                py_json_string(k, out);
                out += key_sep;
                py_json_write(val, sort_keys, item_sep, key_sep, out);
            };
            if (sort_keys) {
                // Python compares str by code point, which is byte order in UTF-8.
                std::vector<std::pair<const std::string *, const ojson *>> items;
                items.reserve(v.size());
                for (auto it = v.begin(); it != v.end(); ++it) items.emplace_back(&it.key(), &it.value());
                std::stable_sort(items.begin(), items.end(), [](const auto & a, const auto & b) { return *a.first < *b.first; });
                for (const auto & kv : items) put(*kv.first, *kv.second);
            } else {
                for (auto it = v.begin(); it != v.end(); ++it) put(it.key(), it.value());
            }
            out += '}';
            break;
        }
        default: throw ChatError("value is not JSON serializable", false);
    }
}

// --- Jinja value model: a `const ojson *` where nullptr means Undefined --------------------------

using Val = const ojson *;

Val attr(Val v, const char * key) {
    if (!v || !v->is_object()) return nullptr;
    auto it = v->find(key);
    return it == v->end() ? nullptr : &*it;
}

// attribute access on Undefined raises in Jinja
Val attr_strict(Val v, const char * key, const char * what) {
    if (!v) throw ChatError(std::string("'") + what + "' is undefined", false);
    return attr(v, key);
}

bool truthy(Val v) {
    if (!v) return false;
    switch (v->type()) {
        case ojson::value_t::null: return false;
        case ojson::value_t::boolean: return v->get<bool>();
        case ojson::value_t::number_integer: return v->get<int64_t>() != 0;
        case ojson::value_t::number_unsigned: return v->get<uint64_t>() != 0;
        case ojson::value_t::number_float: return v->get<double>() != 0.0;
        case ojson::value_t::string: return !v->get_ref<const std::string &>().empty();
        case ojson::value_t::array:
        case ojson::value_t::object: return !v->empty();
        default: return false;
    }
}

bool is_str(Val v) { return v && v->is_string(); }

bool str_eq(Val v, std::string_view s) { return is_str(v) && v->get_ref<const std::string &>() == s; }

double as_double(const ojson & v) {
    if (v.is_boolean()) return v.get<bool>() ? 1.0 : 0.0;
    return v.get<double>();
}

// Python `==` on JSON-derived values
bool py_eq(const ojson & a, const ojson & b) {
    auto numeric = [](const ojson & x) { return x.is_number() || x.is_boolean(); };
    if (numeric(a) && numeric(b)) {
        if (!a.is_number_float() && !b.is_number_float()) {
            auto as_int = [](const ojson & x) -> __int128 {
                if (x.is_boolean()) return x.get<bool>() ? 1 : 0;
                if (x.is_number_unsigned()) return (__int128) x.get<uint64_t>();
                return (__int128) x.get<int64_t>();
            };
            return as_int(a) == as_int(b);
        }
        return as_double(a) == as_double(b);
    }
    if (a.type() != b.type()) return false;
    if (a.is_array()) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); i++) {
            if (!py_eq(a[i], b[i])) return false;
        }
        return true;
    }
    if (a.is_object()) {
        if (a.size() != b.size()) return false;
        for (auto it = a.begin(); it != a.end(); ++it) {
            auto jt = b.find(it.key());
            if (jt == b.end() || !py_eq(it.value(), *jt)) return false;
        }
        return true;
    }
    return a == b;
}

// str(x) as used by the `~` operator; Undefined renders as "".
std::string py_str(Val v) {
    if (!v) return "";
    switch (v->type()) {
        case ojson::value_t::null: return "None";
        case ojson::value_t::boolean: return v->get<bool>() ? "True" : "False";
        case ojson::value_t::string: return v->get<std::string>();
        case ojson::value_t::number_integer:
        case ojson::value_t::number_unsigned:
        case ojson::value_t::number_float: {
            std::string s;
            py_json_number(*v, s);
            return s;
        }
        default: throw ChatError("cannot use a list or object where a string is expected", false);
    }
}

// Iterate like Jinja's `for x in v`: lists yield items, dicts yield keys, strings yield characters.
template <typename F> void for_each_item(Val v, F && f) {
    if (!v) return;   // iterating Undefined yields nothing
    if (v->is_array()) {
        for (const auto & e : *v) f(e);
    } else if (v->is_object()) {
        for (auto it = v->begin(); it != v->end(); ++it) {
            ojson k = it.key();
            f(k);
        }
    } else if (v->is_string()) {
        const std::string & s = v->get_ref<const std::string &>();
        for (size_t i = 0; i < s.size();) {
            size_t n = 1;
            while (i + n < s.size() && ((unsigned char) s[i + n] & 0xC0) == 0x80) n++;
            ojson ch = s.substr(i, n);
            f(ch);
            i += n;
        }
    } else {
        throw ChatError("value is not iterable", false);
    }
}

// Python str.strip(): whitespace per str.isspace()
std::string py_strip(const std::string & s) {
    auto ws_len_at = [&](size_t i) -> size_t {   // length of a whitespace char starting at i, or 0
        unsigned char c = (unsigned char) s[i];
        if (c == ' ' || (c >= 0x09 && c <= 0x0D) || (c >= 0x1C && c <= 0x1F)) return 1;
        if (c == 0xC2 && i + 1 < s.size()) {
            unsigned char d = (unsigned char) s[i + 1];
            return (d == 0x85 || d == 0xA0) ? 2 : 0;
        }
        if (i + 2 < s.size()) {
            unsigned char d = (unsigned char) s[i + 1], e = (unsigned char) s[i + 2];
            uint32_t cp = 0;
            if ((c & 0xF0) == 0xE0) cp = ((uint32_t) (c & 0x0F) << 12) | ((uint32_t) (d & 0x3F) << 6) | (e & 0x3F);
            if (cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) || cp == 0x2028 || cp == 0x2029 || cp == 0x202F ||
                cp == 0x205F || cp == 0x3000)
                return 3;
        }
        return 0;
    };
    size_t b = 0;
    while (b < s.size()) {
        size_t n = ws_len_at(b);
        if (!n) break;
        b += n;
    }
    size_t e = s.size();
    while (e > b) {
        // step back one code point
        size_t p = e - 1;
        while (p > b && ((unsigned char) s[p] & 0xC0) == 0x80) p--;
        size_t n = ws_len_at(p);
        if (!n || p + n != e) break;
        e = p;
    }
    return s.substr(b, e - b);
}

struct Out {
    std::vector<PromptPart> parts;
    void add(std::string_view s, bool is_input) {
        if (s.empty()) return;
        if (!parts.empty() && parts.back().is_input == is_input) parts.back().text.append(s);
        else parts.push_back({std::string(s), is_input});
    }
    void lit(std::string_view s) { add(s, false); }
    void in(std::string_view s) { add(s, true); }
    void append(Out && o) {
        for (auto & p : o.parts) add(p.text, p.is_input);
    }
};

void emit_thinking_effort(Val reasoning_effort, Out & out) {
    static const std::pair<const char *, double> effort_map[] = {
        {"none", 0.0}, {"minimal", 0.1}, {"low", 0.2}, {"medium", 0.7}, {"high", 0.9}, {"xhigh", 0.99}, {"max", 0.99},
    };
    double num = 0.9;
    if (reasoning_effort && !reasoning_effort->is_null()) {
        const ojson & eff = *reasoning_effort;
        if (eff.is_string()) {
            const std::string & raw = eff.get_ref<const std::string &>();
            std::string key = py_strip(raw);
            bool found = false;
            for (const auto & kv : effort_map) {
                if (key == kv.first) {
                    num = kv.second;
                    found = true;
                    break;
                }
            }
            if (!found) throw ChatError("Unknown reasoning_effort: " + raw);
        } else if (eff.is_number() || eff.is_boolean()) {
            num = as_double(eff);
        } else {
            num = 0.0;   // the `float` filter falls back to 0.0 for lists/objects
        }
    }
    if (num < 0.0 || num > 0.99) throw ChatError("reasoning_effort must be in [0.0, 0.99]");
    out.lit("<|message_system|><|content_text|>Thinking effort level: ");
    out.lit(num == 0.0 ? std::string("0") : py_float_repr(num));
    out.lit("<|end_message|>");
}

const char * role_token(Val role) {
    if (!is_str(role)) return nullptr;
    const std::string & r = role->get_ref<const std::string &>();
    if (r == "user") return "<|message_user|>";
    if (r == "assistant") return "<|message_model|>";
    if (r == "system") return "<|message_system|>";
    if (r == "tool") return "<|message_tool|>";
    return nullptr;
}

bool type_in(Val type, std::initializer_list<const char *> names) {
    for (const char * n : names) {
        if (str_eq(type, n)) return true;
    }
    return false;
}

}  // namespace

std::string py_json_dumps(const ojson & v, bool sort_keys, bool compact) {
    std::string out;
    py_json_write(v, sort_keys, compact ? "," : ", ", compact ? ":" : ": ", out);
    return out;
}

// ---------------------------------------------------------------------------------------------
// Template port.  Comments quote the corresponding template lines.
// ---------------------------------------------------------------------------------------------

std::vector<PromptPart> render_chat_parts(const ojson & request) {
    if (!request.is_object()) throw ChatError("request must be a JSON object", false);
    Val messages = attr(&request, "messages");
    Val tools = attr(&request, "tools");
    Val reasoning_effort = attr(&request, "reasoning_effort");
    if (messages && !messages->is_array()) throw ChatError("'messages' must be an array", false);

    Out out;

    // {%- if tools -%}
    if (truthy(tools)) {
        if (!tools->is_array()) throw ChatError("'tools' must be an array", false);
        ojson specs = ojson::array();
        for (const auto & tool : *tools) {
            // {%- set fn = tool.function if tool.function is defined else tool -%}
            Val fn = attr(&tool, "function");
            if (!fn) fn = &tool;
            Val name = attr(fn, "name");
            if (!name) throw ChatError("tool is missing a function name", false);   // tojson(Undefined) fails
            Val desc = attr(fn, "description");
            Val params = attr(fn, "parameters");
            Val type = attr(&tool, "type");
            ojson spec = ojson::object();
            spec["description"] = truthy(desc) ? *desc : ojson("");
            spec["name"] = *name;
            spec["parameters"] = truthy(params) ? *params : ojson::object();
            spec["type"] = truthy(type) ? *type : ojson("function");
            specs.push_back(std::move(spec));
        }
        out.lit("<|message_system|>tool_declare<|content_xml|>");
        out.in(py_json_dumps(specs, /*sort_keys=*/true, /*compact=*/true));
        out.lit("<|end_message|>");
    }

    bool effort_emitted = false;
    if (messages) {
        for (const auto & message_ref : *messages) {
            Val message = &message_ref;
            Val role = attr(message, "role");
            // {%- if message.role not in role_token -%}
            const char * rt = role_token(role);
            if (!rt) {
                if (role && (role->is_array() || role->is_object())) throw ChatError("unhashable message role", false);
                throw ChatError("Unknown message role: " + py_str(role));
            }
            const std::string & role_s = role->get_ref<const std::string &>();
            const bool is_assistant = role_s == "assistant";

            if (!effort_emitted && role_s != "system") {
                emit_thinking_effort(reasoning_effort, out);
                effort_emitted = true;
            }

            // header = role token + optional name
            std::string header_name;
            Val msg_name = attr(message, "name");
            if (role_s == "tool") {
                Val tcid = attr(message, "tool_call_id");
                if (truthy(msg_name)) {
                    header_name = py_str(msg_name);
                } else if (truthy(tcid)) {
                    for (const auto & prev_ref : *messages) {
                        Val prev = &prev_ref;
                        Val prev_calls = attr(prev, "tool_calls");
                        if (str_eq(attr(prev, "role"), "assistant") && truthy(prev_calls)) {
                            for_each_item(prev_calls, [&](const ojson & tc) {
                                Val id = attr(&tc, "id");
                                if (id && py_eq(*id, *tcid)) {
                                    Val fname = attr_strict(attr(&tc, "function"), "name", "function");
                                    if (fname) header_name += py_str(fname);
                                }
                            });
                        }
                    }
                }
            } else if (is_assistant && truthy(msg_name)) {
                header_name = py_str(msg_name);
            }

            Out r;   // {%- set rendered -%}
            auto rtok = [&]() {
                r.lit(rt);
                r.in(header_name);
            };

            Val reasoning = attr(message, "reasoning_content");
            Val content = attr(message, "content");
            Val tool_calls = attr(message, "tool_calls");

            if (is_assistant && truthy(reasoning)) {
                rtok();
                r.lit("<|content_thinking|>");
                r.in(py_str(reasoning));
                r.lit("<|end_message|>");
            }

            if (is_str(content) && (truthy(content) || !(is_assistant && (truthy(tool_calls) || truthy(reasoning))))) {
                rtok();
                r.lit("<|content_text|>");
                r.in(content->get_ref<const std::string &>());
                r.lit("<|end_message|>");
            } else if (truthy(content)) {
                for_each_item(content, [&](const ojson & part) {
                    Val type = attr(&part, "type");
                    if (part.is_string()) {
                        rtok();
                        r.lit("<|content_text|>");
                        r.in(part.get_ref<const std::string &>());
                        r.lit("<|end_message|>");
                    } else if (!type || type_in(type, {"text", "input_text"})) {
                        Val text = attr(&part, "text");
                        rtok();
                        r.lit("<|content_text|>");
                        if (is_str(text)) r.in(text->get_ref<const std::string &>());
                        r.lit("<|end_message|>");
                    } else if (type_in(type, {"image", "input_image", "image_url"})) {
                        if (role_s != "user" && role_s != "tool")
                            throw ChatError("Image content is only supported for user and tool messages");
                        rtok();
                        r.lit("<|content_image|><|unused_200054|><|end_message|>");
                    } else if (type_in(type, {"audio", "input_audio", "audio_url"})) {
                        throw ChatError("Inkling 2 does not accept audio input");
                    } else {
                        throw ChatError("Unsupported content part type: " + py_str(type));
                    }
                });
            }

            if (is_assistant && truthy(tool_calls)) {
                for_each_item(tool_calls, [&](const ojson & tc) {
                    Val fn = attr(&tc, "function");
                    Val fname = attr_strict(fn, "name", "function");
                    if (!is_str(fname)) throw ChatError("tool call function name must be a string");
                    if (truthy(msg_name) && !py_eq(*msg_name, *fname))
                        throw ChatError("assistant name " + py_str(msg_name) + " does not match its tool call " + py_str(fname));
                    Val args = attr(fn, "arguments");
                    static const ojson empty_obj = ojson::object();
                    if (!truthy(args)) args = &empty_obj;
                    if (args->is_string())
                        throw ChatError("tool call arguments must be a parsed object, not a JSON string; canonicalize upstream");
                    if (!args->is_object()) throw ChatError("tool call arguments must be an object");

                    const std::string & name = fname->get_ref<const std::string &>();
                    r.lit("<|message_model|>");
                    r.in(name);
                    r.lit("<|content_invoke_tool_json|>");
                    std::string payload = "{\"name\":";
                    py_json_string(name, payload);
                    payload += ",\"args\":{";
                    // {%- for key, value in args | dictsort(case_sensitive=true) -%}
                    std::vector<std::pair<const std::string *, const ojson *>> items;
                    for (auto it = args->begin(); it != args->end(); ++it) items.emplace_back(&it.key(), &it.value());
                    std::stable_sort(items.begin(), items.end(), [](const auto & a, const auto & b) { return *a.first < *b.first; });
                    for (size_t i = 0; i < items.size(); i++) {
                        if (i) payload += ',';
                        py_json_string(*items[i].first, payload);
                        payload += ':';
                        py_json_write(*items[i].second, false, ",", ":", payload);
                    }
                    payload += "}}";
                    r.in(payload);
                    r.lit("<|end_message|>");
                });
            }

            if (is_assistant) r.lit("<|content_model_end_sampling|>");
            out.append(std::move(r));
        }
    }

    if (!effort_emitted) emit_thinking_effort(reasoning_effort, out);
    return std::move(out.parts);
}

std::string render_chat(const ojson & request) {
    std::string s;
    for (const auto & p : render_chat_parts(request)) s += p.text;
    return s;
}

bool canonicalize_chat_request(ojson & request) {
    if (!request.is_object()) throw ChatError("request body must be a JSON object", false);

    auto msgs = request.find("messages");
    if (msgs != request.end() && msgs->is_array()) {
        for (auto & m : *msgs) {
            if (!m.is_object()) continue;
            auto tcs = m.find("tool_calls");
            if (tcs == m.end() || !tcs->is_array()) continue;
            for (auto & tc : *tcs) {
                if (!tc.is_object()) continue;
                auto fn = tc.find("function");
                if (fn == tc.end() || !fn->is_object()) continue;
                auto args = fn->find("arguments");
                if (args == fn->end() || !args->is_string()) continue;
                const std::string & text = args->get_ref<const std::string &>();
                if (text.find_first_not_of(" \t\r\n") == std::string::npos) {
                    *args = ojson::object();
                    continue;
                }
                ojson parsed = ojson::parse(text, nullptr, /*allow_exceptions=*/false);
                if (parsed.is_discarded()) {
                    auto nm = fn->find("name");
                    std::string name = nm != fn->end() && nm->is_string() ? nm->get<std::string>() : std::string();
                    throw ChatError("tool call arguments for '" + name + "' are not valid JSON", false);
                }
                if (parsed.is_string()) throw ChatError("tool call arguments must be an object", false);
                *args = std::move(parsed);
            }
        }
    }

    if (!request.contains("reasoning_effort")) {
        auto kw = request.find("chat_template_kwargs");
        if (kw != request.end() && kw->is_object() && kw->contains("reasoning_effort")) {
            ojson v = (*kw)["reasoning_effort"];
            request["reasoning_effort"] = std::move(v);
        }
    }

    auto tc = request.find("tool_choice");
    if (tc != request.end() && tc->is_string() && tc->get_ref<const std::string &>() == "none") request.erase("tools");

    auto tools = request.find("tools");
    if (tools == request.end()) return false;
    if (tools->is_null()) return false;
    if (!tools->is_array()) throw ChatError("'tools' must be an array", false);
    return !tools->empty();
}

// ---------------------------------------------------------------------------------------------
// Output parser
// ---------------------------------------------------------------------------------------------

size_t utf8_complete_len(std::string_view s) {
    size_t n = s.size();
    // look at up to the last 3 bytes for the lead byte of an unfinished sequence
    for (size_t back = 1; back <= 3 && back <= n; back++) {
        unsigned char c = (unsigned char) s[n - back];
        if ((c & 0xC0) == 0x80) continue;   // continuation byte, keep looking for the lead
        size_t need = (c & 0x80) == 0 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : (c & 0xF8) == 0xF0 ? 4 : 1;
        return need > back ? n - back : n;
    }
    return n;
}

static std::string random_call_id() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    static const char alphabet[] = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
    std::string id = "call_";
    for (int i = 0; i < 24; i++) id += alphabet[rng() % 62];
    return id;
}

ChatOutputParser::ChatOutputParser(const Tokenizer & tok, bool parse_tool_calls) : tok_(tok), parse_tools_(parse_tool_calls) {
    id_model_ = tok.token_id("<|message_model|>");
    id_think_ = tok.token_id("<|content_thinking|>");
    id_text_ = tok.token_id("<|content_text|>");
    id_tool_json_ = tok.token_id("<|content_invoke_tool_json|>");
    id_end_ = tok.token_id("<|end_message|>");
    id_eos_ = tok.eos_id();
    id_eos2_ = tok.token_id("<|content_model_end_sampling|>");
}

void ChatOutputParser::emit(std::string_view bytes, ChatDelta & out) {
    if (state_ == State::Tool) {
        tool_buf_.append(bytes);
        return;
    }
    std::string & dst = state_ == State::Thinking ? out.reasoning : out.content;
    if (pending_.empty()) {
        size_t n = utf8_complete_len(bytes);
        dst.append(bytes.substr(0, n));
        pending_.assign(bytes.substr(n));
    } else {
        pending_.append(bytes);
        size_t n = utf8_complete_len(pending_);
        dst.append(pending_, 0, n);
        pending_.erase(0, n);
    }
}

void ChatOutputParser::flush_pending(ChatDelta & out) {
    if (pending_.empty()) return;
    // an unfinished code point at the end of a message: substitute U+FFFD
    (state_ == State::Thinking ? out.reasoning : out.content) += "\xEF\xBF\xBD";
    pending_.clear();
}

void ChatOutputParser::enter_body(State s) {
    state_ = s;
    tool_buf_.clear();
}

void ChatOutputParser::close_message(ChatDelta & out, bool terminated) {
    (void) terminated;
    if (state_ == State::Tool) {
        // payload: {"name":"fn","args":{...}}
        bool ok = false;
        ojson j = ojson::parse(tool_buf_, nullptr, false);
        if (j.is_object()) {
            std::string name;
            auto n = j.find("name");
            if (n != j.end() && n->is_string()) name = n->get<std::string>();
            else name = py_strip(header_name_);
            ojson args = ojson::object();
            bool args_ok = true;
            for (const char * key : {"args", "arguments", "parameters"}) {
                auto a = j.find(key);
                if (a == j.end()) continue;
                if (a->is_object()) {
                    args = *a;
                } else if (a->is_string()) {
                    ojson inner = ojson::parse(a->get_ref<const std::string &>(), nullptr, false);
                    if (inner.is_object()) args = std::move(inner);
                    else args_ok = false;
                } else if (!a->is_null()) {
                    args_ok = false;
                }
                break;
            }
            if (!name.empty() && args_ok) {
                ToolCallDelta tc;
                tc.index = n_tool_calls_++;
                tc.id = random_call_id();
                tc.name = std::move(name);
                tc.arguments = args.dump(-1, ' ', false, ojson::error_handler_t::replace);
                out.tool_calls.push_back(std::move(tc));
                ok = true;
            }
        }
        if (!ok) out.content += tool_buf_;   // malformed call: surface the raw payload as text
        tool_buf_.clear();
    } else {
        flush_pending(out);
    }
    state_ = State::Outside;
    header_name_.clear();
}

bool ChatOutputParser::feed(int32_t id, ChatDelta & out) {
    if (id == id_eos_ || id == id_eos2_) return false;

    if (id == id_model_) {
        if (state_ != State::Outside && state_ != State::Header) close_message(out, false);
        state_ = State::Header;
        header_name_.clear();
        return true;
    }
    if (id == id_end_) {
        if (state_ == State::Header) {
            state_ = State::Outside;
            header_name_.clear();
        } else if (state_ != State::Outside) {
            close_message(out, true);
        }
        return true;
    }

    if (tok_.is_special(id)) {
        const std::string & piece = tok_.piece(id);
        const bool in_body = state_ == State::Thinking || state_ == State::Text || state_ == State::Tool;
        if (piece.compare(0, 10, "<|message_") == 0) return false;   // the model started someone else's turn
        if (in_body) return true;                                    // stray control token inside a payload: drop
        if (id == id_think_) enter_body(State::Thinking);
        else if (id == id_text_) enter_body(State::Text);
        else if (id == id_tool_json_) enter_body(parse_tools_ ? State::Tool : State::Text);
        else if (piece.compare(0, 10, "<|content_") == 0) enter_body(State::Text);   // unknown kind: show as text
        return true;
    }

    const std::string & piece = tok_.piece(id);
    switch (state_) {
        case State::Header:
            header_name_ += piece;
            break;
        case State::Outside:
            // text outside any message: treat as answer text
            state_ = State::Text;
            emit(piece, out);
            break;
        default:
            emit(piece, out);
    }
    return true;
}

void ChatOutputParser::finish(ChatDelta & out) {
    if (state_ == State::Thinking || state_ == State::Text || state_ == State::Tool) close_message(out, false);
    state_ = State::Outside;
}

}  // namespace ink
