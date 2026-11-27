// Chat formatting for Inkling 2: a hand-written port of the model's Jinja chat template (prompt
// rendering) and a streaming parser for the model's output messages.
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json_fwd.hpp>

namespace ink {

class Tokenizer;

// Insertion-ordered JSON: the template serializes tool-call arguments in the caller's key order.
using ojson = nlohmann::ordered_json;

// Any failure to render the request.  `raised` is true when the template itself would have called
// raise_exception(message); false for errors Jinja/Python would raise implicitly (bad types, ...).
// Both map to HTTP 400.
struct ChatError : std::runtime_error {
    bool raised;
    explicit ChatError(const std::string & msg, bool raised_ = true) : std::runtime_error(msg), raised(raised_) {}
};

// One span of the rendered prompt.  `is_input` marks text that came from the request (message
// content, names, tool JSON) as opposed to template literals; input spans should be tokenized with
// parse_special=false so that user text cannot forge control tokens.
struct PromptPart {
    std::string text;
    bool is_input = false;
};

// Renders {messages, tools, reasoning_effort} exactly like the GGUF's tokenizer.chat_template under
// Hugging Face transformers.  `request` is an object; missing keys are "undefined" in the template.
// Tool-call arguments must already be objects (see canonicalize_chat_request).  Throws ChatError.
std::vector<PromptPart> render_chat_parts(const ojson & request);
std::string render_chat(const ojson & request);

// Request preprocessing applied before rendering:
//  - assistant tool_calls[].function.arguments given as a JSON string (OpenAI wire format) are parsed
//    into objects (invalid JSON -> ChatError);
//  - `tools` is dropped when tool_choice is "none";
//  - chat_template_kwargs.reasoning_effort is honoured when no top-level reasoning_effort is given.
// Returns true if tools will be rendered (i.e. tool calls should be parsed from the output).
bool canonicalize_chat_request(ojson & request);

// Python-compatible helpers (exposed for tests).
std::string py_float_repr(double v);                                        // repr(float)
std::string py_json_dumps(const ojson & v, bool sort_keys, bool compact);   // json.dumps(ensure_ascii=False)

// Length of the longest prefix of `s` that does not end inside a UTF-8 sequence.
size_t utf8_complete_len(std::string_view s);

struct ToolCallDelta {
    int index = 0;
    std::string id;
    std::string name;
    std::string arguments;   // JSON text of the arguments object
};

struct ChatDelta {
    std::string reasoning;
    std::string content;
    std::vector<ToolCallDelta> tool_calls;
    bool empty() const { return reasoning.empty() && content.empty() && tool_calls.empty(); }
    void clear() { reasoning.clear(); content.clear(); tool_calls.clear(); }
};

// Incremental parser for the model's output:
//   (<|message_model|>[name]<|content_KIND|>payload<|end_message|>)* <|content_model_end_sampling|>
// KIND thinking -> reasoning, text -> content, invoke_tool_json -> tool call; any other kind is
// surfaced as content.  Text deltas never end inside a UTF-8 code point.
class ChatOutputParser {
public:
    ChatOutputParser(const Tokenizer & tok, bool parse_tool_calls);

    // Feed one generated token; appends to `out`.  Returns false when generation must stop (EOS, or
    // the model started a non-assistant message); that token produces no output.
    bool feed(int32_t id, ChatDelta & out);

    // Flush whatever is buffered (unterminated message, incomplete tool call).  Call once at the end.
    void finish(ChatDelta & out);

    int n_tool_calls() const { return n_tool_calls_; }

private:
    enum class State { Outside, Header, Thinking, Text, Tool };

    void emit(std::string_view bytes, ChatDelta & out);
    void flush_pending(ChatDelta & out);
    void close_message(ChatDelta & out, bool terminated);
    void enter_body(State s);

    const Tokenizer & tok_;
    bool parse_tools_;
    int32_t id_model_, id_think_, id_text_, id_tool_json_, id_end_, id_eos_, id_eos2_;
    State state_ = State::Outside;
    std::string header_name_;
    std::string tool_buf_;
    std::string pending_;   // bytes of an incomplete UTF-8 sequence for the current channel
    int n_tool_calls_ = 0;
};

}  // namespace ink
