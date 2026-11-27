// Byte-level BPE tokenizer for Inkling 2 (GGUF tokenizer.ggml.model = "gpt2", pre = "inkling2").
#pragma once
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace ink {

class GGUF;

class Tokenizer {
public:
    explicit Tokenizer(const GGUF & gguf);   // throws std::runtime_error on unsupported vocab
    ~Tokenizer();

    // Text -> ids.  With parse_special, literal special-token text such as "<|end_message|>" maps to
    // its single id; otherwise it is tokenized as plain text.  No BOS/EOS is ever added.
    std::vector<int32_t> encode(std::string_view text, bool parse_special = true) const;

    // ids -> bytes.  Special (control) tokens are rendered as their literal text when render_special,
    // otherwise skipped.  The result may end in an incomplete UTF-8 sequence if the ids do.
    std::string decode(const std::vector<int32_t> & ids, bool render_special = true) const;

    // Raw bytes of one token (for special tokens: the literal text).
    const std::string & piece(int32_t id) const;

    bool is_special(int32_t id) const;                    // token_type != normal
    int32_t token_id(std::string_view text) const;        // exact vocabulary lookup, -1 if absent
    int n_vocab() const;
    int32_t eos_id() const;                               // tokenizer.ggml.eos_token_id

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ink
