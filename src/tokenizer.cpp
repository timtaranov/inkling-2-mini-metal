// Byte-level BPE tokenizer for Inkling 2 (tokenizer.ggml.model = "gpt2", tokenizer.ggml.pre = "inkling2").
//
// Behaviour is modelled on the llama.cpp reference for this vocabulary:
//   * special tokens are cut out of the text first, longest token text first (when parse_special);
//   * each remaining fragment is split into words by the "gpt-4o" pre-tokenizer pattern, evaluated the way the
//     reference evaluates it (see Splitter below for the exact pattern and its quirks);
//   * a word that is itself a vocabulary entry becomes that single token ("ignore_merges"), otherwise it is
//     merged bottom-up from single bytes using the ranked merge list.
//
// Everything here works on raw bytes.  The GGUF stores token and merge strings in the GPT-2 "bytes as
// printable code points" form (e.g. U+0120 for a space); they are converted to raw bytes once at load.

#include "tokenizer.h"

#include "gguf.h"
#include "unicode_tables.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace ink {

namespace {

// ---------------------------------------------------------------------------------------------------------
// UTF-8
// ---------------------------------------------------------------------------------------------------------

// Strict-ish UTF-8 decoder with the same accept/reject rules as the reference: overlong forms and values
// above U+10FFFF are rejected, surrogates are accepted.  Returns the sequence length, or 0 if invalid.
inline int utf8_decode(const uint8_t * p, size_t avail, uint32_t & cpt) {
    const uint8_t b0 = p[0];
    if (b0 < 0x80) {
        cpt = b0;
        return 1;
    }
    if (b0 < 0xC0) return 0;
    if (b0 < 0xE0) {
        if (avail < 2 || (p[1] & 0xC0) != 0x80) return 0;
        cpt = ((uint32_t) (b0 & 0x1F) << 6) | (p[1] & 0x3F);
        return cpt < 0x80 ? 0 : 2;
    }
    if (b0 < 0xF0) {
        if (avail < 3 || (p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80) return 0;
        cpt = ((uint32_t) (b0 & 0x0F) << 12) | ((uint32_t) (p[1] & 0x3F) << 6) | (p[2] & 0x3F);
        return cpt < 0x800 ? 0 : 3;
    }
    if (b0 < 0xF8) {
        if (avail < 4 || (p[1] & 0xC0) != 0x80 || (p[2] & 0xC0) != 0x80 || (p[3] & 0xC0) != 0x80) return 0;
        cpt = ((uint32_t) (b0 & 0x07) << 18) | ((uint32_t) (p[1] & 0x3F) << 12) | ((uint32_t) (p[2] & 0x3F) << 6) |
              (p[3] & 0x3F);
        return (cpt < 0x10000 || cpt > 0x10FFFF) ? 0 : 4;
    }
    return 0;
}

void utf8_append(std::string & out, uint32_t cpt) {
    if (cpt < 0x80) {
        out.push_back((char) cpt);
    } else if (cpt < 0x800) {
        out.push_back((char) (0xC0 | (cpt >> 6)));
        out.push_back((char) (0x80 | (cpt & 0x3F)));
    } else if (cpt < 0x10000) {
        out.push_back((char) (0xE0 | (cpt >> 12)));
        out.push_back((char) (0x80 | ((cpt >> 6) & 0x3F)));
        out.push_back((char) (0x80 | (cpt & 0x3F)));
    } else {
        out.push_back((char) (0xF0 | (cpt >> 18)));
        out.push_back((char) (0x80 | ((cpt >> 12) & 0x3F)));
        out.push_back((char) (0x80 | ((cpt >> 6) & 0x3F)));
        out.push_back((char) (0x80 | (cpt & 0x3F)));
    }
}

// ---------------------------------------------------------------------------------------------------------
// Hashing
// ---------------------------------------------------------------------------------------------------------

inline uint64_t mix64(uint64_t x) {
    x ^= x >> 32;
    x *= 0xD6E8FEB86659FD93ull;
    x ^= x >> 32;
    x *= 0xD6E8FEB86659FD93ull;
    x ^= x >> 32;
    return x;
}

inline uint64_t hash_bytes(const uint8_t * p, size_t n) {
    uint64_t h = 0x9E3779B97F4A7C15ull ^ (n * 0xFF51AFD7ED558CCDull);
    while (n >= 8) {
        uint64_t v;
        memcpy(&v, p, 8);
        h = (h ^ v) * 0x9FB21C651E98DF25ull;
        h ^= h >> 29;
        p += 8;
        n -= 8;
    }
    if (n) {
        uint64_t v = 0;
        memcpy(&v, p, n);
        h = (h ^ v) * 0x9FB21C651E98DF25ull;
    }
    return mix64(h);
}

// Character classes used by the pre-tokenizer.
enum : uint8_t {
    C_OTHER = 0,   // anything that is not whitespace, a letter, a mark or a number
    C_LOWER,       // a-z
    C_UPPER,       // A-Z
    C_LETTER,      // any non-ASCII letter
    C_MARK,        // \p{M}
    C_NUMBER,      // \p{N}
    C_SPACE,       // whitespace other than \r \n
    C_NEWLINE,     // \r or \n
};

constexpr uint32_t NO_RANK = 0xFFFFFFFFu;
constexpr uint64_t EMPTY_KEY = ~0ull;

}  // namespace

// ---------------------------------------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------------------------------------

struct Tokenizer::Impl {
    // token attributes (same meaning as GGUF token_type)
    enum : uint8_t { T_UNDEFINED = 0, T_NORMAL = 1, T_UNKNOWN = 2, T_CONTROL = 3, T_USER_DEFINED = 4, T_UNUSED = 5, T_BYTE = 6 };

    int n_vocab = 0;
    int32_t eos = -1;
    std::vector<std::string> texts;     // token text as stored in the GGUF
    std::vector<std::string> pieces;    // raw bytes rendered for each token
    std::vector<uint8_t> types;
    std::string empty;

    // --- symbols -------------------------------------------------------------------------------------------
    // A symbol is a distinct byte string that can appear during BPE merging.  Symbols [0, n_vocab) are the
    // vocabulary tokens themselves (for duplicate strings the highest id is the canonical one, as in the
    // reference's text->id map); symbols >= n_vocab are merge operands/results that are not in the vocabulary.
    std::string arena;                  // all symbol byte strings, concatenated
    std::vector<uint32_t> sym_off;      // per symbol: offset into arena
    std::vector<uint32_t> sym_len;      // per symbol: length (0 = no byte string, never matches)
    struct StrSlot { uint32_t tag; uint32_t sym1; };   // sym1 = symbol + 1, 0 = empty
    std::vector<StrSlot> str_slots;
    uint64_t str_mask = 0;
    uint32_t byte_sym[256];

    // --- merges --------------------------------------------------------------------------------------------
    struct MergeSlot { uint64_t key; uint32_t rank; uint32_t merged; };
    std::vector<MergeSlot> merge_slots;
    uint64_t merge_mask = 0;

    // --- special tokens ------------------------------------------------------------------------------------
    std::vector<int32_t> specials;      // ids, in the reference's matching priority order (longest text first)
    bool special_first[256];
    int special_first_count = 0;
    uint8_t special_first_byte = 0;
    bool any_user_defined = false;

    // --- unicode -------------------------------------------------------------------------------------------
    std::vector<uint8_t> cls_table;     // class per code point

    explicit Impl(const GGUF & gguf);

    // string table
    int64_t find_sym(const uint8_t * p, size_t n) const {
        const uint64_t h = hash_bytes(p, n);
        const uint32_t tag = (uint32_t) (h >> 32);
        for (uint64_t i = h & str_mask;; i = (i + 1) & str_mask) {
            const StrSlot & s = str_slots[i];
            if (s.sym1 == 0) return -1;
            if (s.tag == tag) {
                const uint32_t sym = s.sym1 - 1;
                if (sym_len[sym] == n && memcmp(arena.data() + sym_off[sym], p, n) == 0) return sym;
            }
        }
    }
    void grow_str_table();
    uint32_t add_string(const std::string & bytes);                 // appends to the arena, returns its offset
    void set_token_sym(uint32_t id, const std::string & bytes);     // vocabulary entry: later ids replace earlier
    uint32_t intern(const std::string & bytes);                     // find or create a symbol
    size_t n_str_entries = 0;

    // merge table
    inline const MergeSlot * find_merge(uint32_t l, uint32_t r) const {
        const uint64_t key = ((uint64_t) l << 32) | r;
        for (uint64_t i = mix64(key) & merge_mask;; i = (i + 1) & merge_mask) {
            const MergeSlot & s = merge_slots[i];
            if (s.key == key) return &s;
            if (s.key == EMPTY_KEY) return nullptr;
        }
    }

    // unicode
    inline uint8_t cls_at(const uint8_t * p, int & len) const {     // p must point at a valid sequence
        const uint8_t b0 = p[0];
        if (b0 < 0x80) {
            len = 1;
            return cls_table[b0];
        }
        if (b0 < 0xE0) {
            len = 2;
            return cls_table[((uint32_t) (b0 & 0x1F) << 6) | (p[1] & 0x3F)];
        }
        if (b0 < 0xF0) {
            len = 3;
            return cls_table[((uint32_t) (b0 & 0x0F) << 12) | ((uint32_t) (p[1] & 0x3F) << 6) | (p[2] & 0x3F)];
        }
        len = 4;
        return cls_table[((uint32_t) (b0 & 0x07) << 18) | ((uint32_t) (p[1] & 0x3F) << 12) |
                         ((uint32_t) (p[2] & 0x3F) << 6) | (p[3] & 0x3F)];
    }

    // encoding
    void emit(uint32_t sym, std::vector<int32_t> & out) const;
    void bpe_small(const uint8_t * w, size_t n, std::vector<int32_t> & out) const;
    void bpe_large(const uint8_t * w, size_t n, std::vector<int32_t> & out) const;
    inline void encode_word(const uint8_t * w, size_t n, std::vector<int32_t> & out) const {
        const int64_t whole = find_sym(w, n);
        if (whole >= 0 && whole < n_vocab) {
            out.push_back((int32_t) whole);
            return;
        }
        if (n <= kSmallWord) bpe_small(w, n, out);
        else bpe_large(w, n, out);
    }
    static constexpr size_t kSmallWord = 48;

    const uint8_t * letters(const uint8_t * p, const uint8_t * e) const;
    const uint8_t * next_word(const uint8_t * p, const uint8_t * e) const;
    void encode_fragment(const uint8_t * s, size_t n, std::vector<int32_t> & out) const;
    void encode(std::string_view text, bool parse_special, std::vector<int32_t> & out) const;
};

// ---------------------------------------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------------------------------------

void Tokenizer::Impl::grow_str_table() {
    const size_t n = str_slots.empty() ? (size_t) 1 << 16 : str_slots.size() * 2;
    std::vector<StrSlot> fresh(n, StrSlot{0, 0});
    const uint64_t mask = n - 1;
    for (const StrSlot & s : str_slots) {
        if (s.sym1 == 0) continue;
        const uint32_t sym = s.sym1 - 1;
        const uint64_t h = hash_bytes((const uint8_t *) arena.data() + sym_off[sym], sym_len[sym]);
        uint64_t i = h & mask;
        while (fresh[i].sym1 != 0) i = (i + 1) & mask;
        fresh[i] = s;
    }
    str_slots.swap(fresh);
    str_mask = mask;
}

uint32_t Tokenizer::Impl::add_string(const std::string & bytes) {
    if (arena.size() + bytes.size() > 0xFFFFFFF0u) throw std::runtime_error("tokenizer: vocabulary too large");
    const uint32_t off = (uint32_t) arena.size();
    arena.append(bytes);
    return off;
}

void Tokenizer::Impl::set_token_sym(uint32_t id, const std::string & bytes) {
    if (bytes.empty()) return;
    if ((n_str_entries + 1) * 2 > str_slots.size()) grow_str_table();
    const uint64_t h = hash_bytes((const uint8_t *) bytes.data(), bytes.size());
    const uint32_t tag = (uint32_t) (h >> 32);
    for (uint64_t i = h & str_mask;; i = (i + 1) & str_mask) {
        StrSlot & s = str_slots[i];
        if (s.sym1 == 0) {
            sym_off[id] = add_string(bytes);
            sym_len[id] = (uint32_t) bytes.size();
            s = StrSlot{tag, id + 1};
            ++n_str_entries;
            return;
        }
        if (s.tag == tag) {
            const uint32_t old = s.sym1 - 1;
            if (sym_len[old] == bytes.size() && memcmp(arena.data() + sym_off[old], bytes.data(), bytes.size()) == 0) {
                // duplicate text: the later id wins, the earlier one can no longer be produced by encode()
                sym_off[id] = sym_off[old];
                sym_len[id] = sym_len[old];
                sym_len[old] = 0;
                s.sym1 = id + 1;
                return;
            }
        }
    }
}

uint32_t Tokenizer::Impl::intern(const std::string & bytes) {
    const int64_t found = find_sym((const uint8_t *) bytes.data(), bytes.size());
    if (found >= 0) return (uint32_t) found;
    if ((n_str_entries + 1) * 2 > str_slots.size()) grow_str_table();
    const uint32_t sym = (uint32_t) sym_off.size();
    sym_off.push_back(add_string(bytes));
    sym_len.push_back((uint32_t) bytes.size());
    const uint64_t h = hash_bytes((const uint8_t *) bytes.data(), bytes.size());
    uint64_t i = h & str_mask;
    while (str_slots[i].sym1 != 0) i = (i + 1) & str_mask;
    str_slots[i] = StrSlot{(uint32_t) (h >> 32), sym + 1};
    ++n_str_entries;
    return sym;
}

Tokenizer::Impl::Impl(const GGUF & gguf) {
    const std::string model = gguf.str("tokenizer.ggml.model", "");
    if (model != "gpt2") throw std::runtime_error("tokenizer: unsupported tokenizer.ggml.model '" + model + "'");
    const std::string pre = gguf.str("tokenizer.ggml.pre", "");
    if (pre != "inkling2") throw std::runtime_error("tokenizer: unsupported tokenizer.ggml.pre '" + pre + "'");

    const std::vector<std::string> & toks = gguf.str_array("tokenizer.ggml.tokens");
    if (toks.empty() || toks.size() > 0x7FFFFFF0u) throw std::runtime_error("tokenizer: bad vocabulary size");
    n_vocab = (int) toks.size();

    std::vector<int64_t> toktypes;
    if (gguf.has("tokenizer.ggml.token_type")) toktypes = gguf.int_array("tokenizer.ggml.token_type");
    if (!toktypes.empty() && toktypes.size() != toks.size()) throw std::runtime_error("tokenizer: token_type size mismatch");

    eos = gguf.has("tokenizer.ggml.eos_token_id") ? (int32_t) gguf.u64("tokenizer.ggml.eos_token_id") : -1;

    // --- unicode class table ------------------------------------------------------------------------------
    cls_table.assign(0x110000, C_OTHER);
    {
        static const uint8_t map[5] = {C_OTHER, C_LETTER, C_MARK, C_NUMBER, C_SPACE};
        for (unsigned i = 0; i < kUniRunCount; ++i) {
            const uint32_t first = kUniRuns[i].first;
            const uint32_t last = i + 1 < kUniRunCount ? kUniRuns[i + 1].first : 0x110000;
            const uint8_t c = map[kUniRuns[i].cls];
            if (c != C_OTHER) memset(cls_table.data() + first, c, last - first);
        }
        for (int c = 0; c < 0x80; ++c) cls_table[c] = C_OTHER;
        for (int c = 'a'; c <= 'z'; ++c) cls_table[c] = C_LOWER;
        for (int c = 'A'; c <= 'Z'; ++c) cls_table[c] = C_UPPER;
        for (int c = '0'; c <= '9'; ++c) cls_table[c] = C_NUMBER;
        cls_table['\t'] = cls_table['\v'] = cls_table['\f'] = cls_table[' '] = C_SPACE;
        cls_table['\r'] = cls_table['\n'] = C_NEWLINE;
    }

    // --- GPT-2 byte <-> code point mapping -------------------------------------------------------------------
    // Printable Latin-1 bytes map to themselves, the other 68 bytes map to U+0100.. in ascending byte order.
    int cpt_to_byte[0x180];
    for (int & v : cpt_to_byte) v = -1;
    {
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            const bool self = (b >= 0x21 && b <= 0x7E) || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
            cpt_to_byte[self ? b : 256 + n++] = b;
        }
    }
    // Converts a GGUF token/merge string to raw bytes; false if it contains a code point outside the mapping.
    auto to_bytes = [&](const char * s, size_t n, std::string & out) -> bool {
        out.clear();
        const uint8_t * p = (const uint8_t *) s;
        size_t i = 0;
        while (i < n) {
            uint32_t cpt;
            const int len = utf8_decode(p + i, n - i, cpt);
            if (len == 0 || cpt >= 0x180 || cpt_to_byte[cpt] < 0) return false;
            out.push_back((char) cpt_to_byte[cpt]);
            i += len;
        }
        return true;
    };

    // --- tokens ----------------------------------------------------------------------------------------------
    texts.resize(n_vocab);
    pieces.resize(n_vocab);
    types.resize(n_vocab);
    sym_off.assign(n_vocab, 0);
    sym_len.assign(n_vocab, 0);
    arena.reserve(toks.size() * 8);
    str_slots.assign((size_t) 1 << 20, StrSlot{0, 0});
    str_mask = str_slots.size() - 1;

    std::string bytes;
    for (int i = 0; i < n_vocab; ++i) {
        texts[i] = toks[i].empty() ? "[EMPTY_" + std::to_string(i) + "]" : toks[i];
        const std::string & text = texts[i];
        const int64_t tt = toktypes.empty() ? (int64_t) T_NORMAL : toktypes[i];
        types[i] = (tt >= 0 && tt <= 6) ? (uint8_t) tt : (uint8_t) T_UNDEFINED;

        const bool ok = to_bytes(text.data(), text.size(), bytes);
        if (ok) set_token_sym((uint32_t) i, bytes);

        switch (types[i]) {
            case T_UNKNOWN:
            case T_CONTROL:
            case T_USER_DEFINED:
                pieces[i] = text;
                break;
            case T_NORMAL:
            case T_BYTE:
                if (ok) {
                    pieces[i] = bytes;
                } else {
                    // same rendering as the reference for code points that are not byte stand-ins
                    std::string piece;
                    const uint8_t * p = (const uint8_t *) text.data();
                    size_t k = 0;
                    while (k < text.size()) {
                        uint32_t cpt;
                        int len = utf8_decode(p + k, text.size() - k, cpt);
                        if (len == 0) {
                            cpt = 0xFFFD;
                            len = 1;
                        }
                        k += len;
                        if (cpt < 0x180 && cpt_to_byte[cpt] >= 0) {
                            piece.push_back((char) cpt_to_byte[cpt]);
                        } else {
                            std::string u;
                            utf8_append(u, cpt);
                            piece += "[UNK_BYTE_0x";
                            for (unsigned char c : u) {
                                char hex[3];
                                snprintf(hex, sizeof(hex), "%02x", c);
                                piece += hex;
                            }
                            piece += text + "]";
                        }
                    }
                    pieces[i] = piece;
                }
                break;
            default:
                break;   // unused / undefined: rendered as nothing
        }
        if (types[i] == T_USER_DEFINED) any_user_defined = true;
    }

    // --- single bytes ----------------------------------------------------------------------------------------
    for (int b = 0; b < 256; ++b) byte_sym[b] = intern(std::string(1, (char) b));

    // --- merges ----------------------------------------------------------------------------------------------
    if (gguf.has("tokenizer.ggml.merges")) {
        const std::vector<std::string> & merges = gguf.str_array("tokenizer.ggml.merges");
        size_t cap = 1024;
        while (cap < merges.size() * 2 + 16) cap *= 2;
        merge_slots.assign(cap, MergeSlot{EMPTY_KEY, 0, 0});
        merge_mask = cap - 1;

        std::string left, right;
        for (size_t i = 0; i < merges.size(); ++i) {
            const std::string & m = merges[i];
            // "<left> <right>", split at the first space after the first character
            const size_t sep = m.size() > 1 ? m.find(' ', 1) : std::string::npos;
            if (sep == std::string::npos) continue;
            if (!to_bytes(m.data(), sep, left) || !to_bytes(m.data() + sep + 1, m.size() - sep - 1, right)) continue;
            if (left.empty() || right.empty()) continue;
            const uint32_t l = intern(left);
            const uint32_t r = intern(right);
            const uint32_t merged = intern(left + right);
            const uint64_t key = ((uint64_t) l << 32) | r;
            uint64_t slot = mix64(key) & merge_mask;
            while (merge_slots[slot].key != EMPTY_KEY && merge_slots[slot].key != key) slot = (slot + 1) & merge_mask;
            if (merge_slots[slot].key == EMPTY_KEY) merge_slots[slot] = MergeSlot{key, (uint32_t) i, merged};   // first rank wins
        }
    } else {
        merge_slots.assign(16, MergeSlot{EMPTY_KEY, 0, 0});
        merge_mask = 15;
    }

    // --- special tokens ------------------------------------------------------------------------------------
    for (int i = 0; i < n_vocab; ++i) {
        if (types[i] == T_CONTROL || types[i] == T_USER_DEFINED || types[i] == T_UNKNOWN) specials.push_back(i);
    }
    // Same comparator and algorithm as the reference so that equal-length tokens end up in the same order.
    std::sort(specials.begin(), specials.end(),
              [&](int32_t a, int32_t b) { return texts[a].size() > texts[b].size(); });
    memset(special_first, 0, sizeof(special_first));
    for (int32_t id : specials) {
        const uint8_t b = (uint8_t) texts[id][0];
        if (!special_first[b]) {
            special_first[b] = true;
            special_first_byte = b;
            ++special_first_count;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------
// BPE
// ---------------------------------------------------------------------------------------------------------

void Tokenizer::Impl::emit(uint32_t sym, std::vector<int32_t> & out) const {
    if (sym < (uint32_t) n_vocab) {
        out.push_back((int32_t) sym);
        return;
    }
    // Not a vocabulary entry.  The reference falls back to per-character lookups on the GPT-2 encoded text,
    // which can only succeed for printable ASCII; other bytes are dropped.
    const uint8_t * p = (const uint8_t *) arena.data() + sym_off[sym];
    for (uint32_t i = 0; i < sym_len[sym]; ++i) {
        const uint8_t b = p[i];
        if (b >= 0x21 && b <= 0x7E && byte_sym[b] < (uint32_t) n_vocab) out.push_back((int32_t) byte_sym[b]);
    }
}

// Repeatedly merges the adjacent pair with the lowest rank (leftmost on ties).  n <= kSmallWord.
void Tokenizer::Impl::bpe_small(const uint8_t * w, size_t n, std::vector<int32_t> & out) const {
    uint32_t sym[kSmallWord];
    uint32_t rank[kSmallWord];     // rank[i]: rank of the pair (sym[i], sym[i+1])
    uint32_t merged[kSmallWord];
    for (size_t i = 0; i < n; ++i) sym[i] = byte_sym[w[i]];
    auto look = [&](size_t i) {
        const MergeSlot * m = find_merge(sym[i], sym[i + 1]);
        if (m) {
            rank[i] = m->rank;
            merged[i] = m->merged;
        } else {
            rank[i] = NO_RANK;
        }
    };
    for (size_t i = 0; i + 1 < n; ++i) look(i);

    while (n > 1) {
        size_t best = 0;
        uint32_t best_rank = rank[0];
        for (size_t i = 1; i + 1 < n; ++i) {
            if (rank[i] < best_rank) {
                best_rank = rank[i];
                best = i;
            }
        }
        if (best_rank == NO_RANK) break;
        sym[best] = merged[best];
        for (size_t i = best + 1; i + 1 < n; ++i) sym[i] = sym[i + 1];
        for (size_t i = best + 1; i + 2 < n; ++i) {
            rank[i] = rank[i + 1];
            merged[i] = merged[i + 1];
        }
        --n;
        if (best > 0) look(best - 1);
        if (best + 1 < n) look(best);
    }
    for (size_t i = 0; i < n; ++i) emit(sym[i], out);
}

// Same result as bpe_small, for long words: linked list + heap of candidate pairs with lazy invalidation.
void Tokenizer::Impl::bpe_large(const uint8_t * w, size_t n, std::vector<int32_t> & out) const {
    struct Cand {
        uint32_t rank, left, right, lsym, rsym, merged;
    };
    auto worse = [](const Cand & a, const Cand & b) { return a.rank > b.rank || (a.rank == b.rank && a.left > b.left); };

    std::vector<uint32_t> sym(n);
    std::vector<int32_t> prev(n), next(n);
    std::vector<Cand> heap;
    heap.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        sym[i] = byte_sym[w[i]];
        prev[i] = (int32_t) i - 1;
        next[i] = i + 1 < n ? (int32_t) i + 1 : -1;
    }
    constexpr uint32_t DEAD = 0xFFFFFFFFu;
    auto push = [&](int32_t l, int32_t r) {
        if (l < 0 || r < 0) return;
        const MergeSlot * m = find_merge(sym[l], sym[r]);
        if (!m) return;
        heap.push_back(Cand{m->rank, (uint32_t) l, (uint32_t) r, sym[l], sym[r], m->merged});
        std::push_heap(heap.begin(), heap.end(), worse);
    };
    for (size_t i = 0; i + 1 < n; ++i) push((int32_t) i, (int32_t) i + 1);

    while (!heap.empty()) {
        std::pop_heap(heap.begin(), heap.end(), worse);
        const Cand c = heap.back();
        heap.pop_back();
        if (sym[c.left] != c.lsym || sym[c.right] != c.rsym || next[c.left] != (int32_t) c.right) continue;   // stale
        sym[c.left] = c.merged;
        sym[c.right] = DEAD;
        next[c.left] = next[c.right];
        if (next[c.right] >= 0) prev[next[c.right]] = (int32_t) c.left;
        push(prev[c.left], (int32_t) c.left);
        push((int32_t) c.left, next[c.left]);
    }
    for (int32_t i = 0; i >= 0; i = next[i]) emit(sym[i], out);
}

// ---------------------------------------------------------------------------------------------------------
// Pre-tokenizer (word splitter)
// ---------------------------------------------------------------------------------------------------------
//
// The reference runs this pattern (leftmost match, alternatives tried in order, greedy with backtracking):
//
//   [^\r\n\p{L}\p{N}]? U* W+ C?  |  [^\r\n\p{L}\p{N}]? U+ W* C?  |  \p{N}{1,3}  |
//    ?[^\s\p{L}\p{N}]+[\r\n/]*   |  \s*[\r\n]+  |  \s+(?!\S)  |  \s+
//
// where C is a contraction ('s 't 're 've 'm 'll 'd, either case) and, because of how the reference maps
// Unicode categories onto a byte alphabet before matching:
//   U = a letter or mark that is not an ASCII lower-case letter  (so every non-ASCII letter/mark is "upper")
//   W = a letter or mark that is not an ASCII upper-case letter  (and also "lower")
//   \s = \t \n \v \f \r space, and all Unicode whitespace
// The functions below evaluate exactly that, without a regex engine.

namespace {

inline const uint8_t * contraction(const uint8_t * p, const uint8_t * e) {
    if (e - p >= 2 && p[0] == '\'') {
        const uint8_t c = p[1] | 0x20;
        if (c == 's' || c == 't' || c == 'm' || c == 'd') return p + 2;
        if (e - p >= 3) {
            const uint8_t d = p[2] | 0x20;
            if ((c == 'r' && d == 'e') || (c == 'v' && d == 'e') || (c == 'l' && d == 'l')) return p + 3;
        }
    }
    return p;
}

}  // namespace

// Matches "U* W+ C?" and, failing that, "U+ W* C?" at p (no optional prefix).  Returns the end or nullptr.
const uint8_t * Tokenizer::Impl::letters(const uint8_t * p, const uint8_t * e) const {
    const uint8_t * q = p;
    const uint8_t * last_w = nullptr;   // end of the last character of the U-run that is also a W
    int len;
    while (q < e) {
        const uint8_t c = cls_at(q, len);
        if (c == C_UPPER) {
            q += len;
        } else if (c == C_LETTER || c == C_MARK) {
            q += len;
            last_w = q;
        } else {
            break;
        }
    }
    const uint8_t * r = q;
    while (r < e) {
        const uint8_t c = cls_at(r, len);
        if (c != C_LOWER && c != C_LETTER && c != C_MARK) break;
        r += len;
    }
    if (r > q) return contraction(r, e);          // U* W+
    if (last_w) return contraction(last_w, e);    // U* gives characters back until W+ can match
    if (q > p) return contraction(q, e);          // U+ W* (all ASCII upper case)
    return nullptr;
}

// Returns the end of the word starting at p (p < e).
const uint8_t * Tokenizer::Impl::next_word(const uint8_t * p, const uint8_t * e) const {
    int n0;
    const uint8_t c0 = cls_at(p, n0);
    int len;

    switch (c0) {
        case C_LOWER:
        case C_UPPER:
        case C_LETTER:
            return letters(p, e);   // cannot fail: p starts a letter run

        case C_NUMBER: {
            const uint8_t * q = p + n0;
            for (int k = 1; k < 3 && q < e && cls_at(q, len) == C_NUMBER; ++k) q += len;
            return q;
        }

        case C_MARK: {
            // as a prefix of a letter run, or else as the start of one
            if (p + n0 < e) {
                // the alternatives are tried in order: (prefix, U*W+), (no prefix, U*W+), ...; the second always
                // matches for a mark, so only the first form of the prefixed attempt is relevant.
                const uint8_t * q = p + n0;
                const uint8_t * u = q;
                const uint8_t * last_w = nullptr;
                while (u < e) {
                    const uint8_t c = cls_at(u, len);
                    if (c == C_UPPER) {
                        u += len;
                    } else if (c == C_LETTER || c == C_MARK) {
                        u += len;
                        last_w = u;
                    } else {
                        break;
                    }
                }
                const uint8_t * r = u;
                while (r < e) {
                    const uint8_t c = cls_at(r, len);
                    if (c != C_LOWER && c != C_LETTER && c != C_MARK) break;
                    r += len;
                }
                if (r > u) return contraction(r, e);
                if (last_w) return contraction(last_w, e);
            }
            return letters(p, e);
        }

        case C_NEWLINE:
        case C_SPACE:
        case C_OTHER:
        default:
            break;
    }

    if (c0 != C_NEWLINE) {
        // optional prefix character followed by a letter run
        if (p + n0 < e) {
            const uint8_t * r = letters(p + n0, e);
            if (r) return r;
        }
        // " ?[^\s\p{L}\p{N}]+[\r\n/]*"
        const uint8_t * q = p;
        if (*p == ' ' && p + 1 < e) {
            const uint8_t c = cls_at(p + 1, len);
            if (c == C_OTHER || c == C_MARK) q = p + 1;
        }
        const uint8_t * r = q;
        while (r < e) {
            const uint8_t c = cls_at(r, len);
            if (c != C_OTHER && c != C_MARK) break;
            r += len;
        }
        if (r > q) {
            while (r < e && (*r == '\r' || *r == '\n' || *r == '/')) ++r;
            return r;
        }
    }

    // whitespace: "\s*[\r\n]+", then "\s+(?!\S)", then "\s+"
    const uint8_t * q = p;
    const uint8_t * last_nl = nullptr;      // end of the last \r or \n in the run
    const uint8_t * last_start = p;         // start of the last character in the run
    while (q < e) {
        const uint8_t c = cls_at(q, len);
        if (c == C_NEWLINE) {
            last_start = q;
            q += len;
            last_nl = q;
        } else if (c == C_SPACE) {
            last_start = q;
            q += len;
        } else {
            break;
        }
    }
    if (last_nl) return last_nl;
    if (q == e) return q;                   // run reaches the end of the fragment
    if (last_start > p) return last_start;  // leave the last whitespace character for the next word
    return q > p ? q : p + n0;
}

void Tokenizer::Impl::encode_fragment(const uint8_t * s, size_t n, std::vector<int32_t> & out) const {
    if (n == 0) return;

    // Invalid UTF-8: each offending byte is replaced by U+FFFD, like the reference.
    std::string fixed;
    {
        size_t i = 0;
        while (i < n) {
            if (s[i] < 0x80) {
                ++i;
                continue;
            }
            uint32_t cpt;
            const int len = utf8_decode(s + i, n - i, cpt);
            if (len == 0) break;
            i += len;
        }
        if (i < n) {
            fixed.reserve(n + 64);
            fixed.append((const char *) s, i);
            while (i < n) {
                uint32_t cpt;
                const int len = utf8_decode(s + i, n - i, cpt);
                if (len == 0) {
                    fixed.append("\xEF\xBF\xBD");
                    ++i;
                } else {
                    fixed.append((const char *) s + i, len);
                    i += len;
                }
            }
            s = (const uint8_t *) fixed.data();
            n = fixed.size();
        }
    }

    const uint8_t * p = s;
    const uint8_t * e = s + n;
    while (p < e) {
        const uint8_t * q = next_word(p, e);
        encode_word(p, (size_t) (q - p), out);
        p = q;
    }
}

void Tokenizer::Impl::encode(std::string_view text, bool parse_special, std::vector<int32_t> & out) const {
    const uint8_t * s = (const uint8_t *) text.data();
    const size_t n = text.size();
    if (n == 0) return;
    out.reserve(out.size() + n / 3 + 8);

    // --- locate special tokens -----------------------------------------------------------------------------
    // The reference takes the special tokens one at a time in priority order and, for each, cuts every
    // left-to-right non-overlapping occurrence out of the text that is still unclaimed.  Equivalent here:
    // collect all occurrences, then accept them in (priority, position) order unless they overlap an
    // accepted one.
    struct Hit { uint32_t prio; uint32_t id; size_t pos; size_t end; };
    std::vector<Hit> hits;
    const bool want_specials = !specials.empty() && (parse_special || any_user_defined);
    if (want_specials) {
        auto probe = [&](size_t pos) {
            for (size_t k = 0; k < specials.size(); ++k) {
                const int32_t id = specials[k];
                if (!parse_special && types[id] != T_USER_DEFINED) continue;
                const std::string & t = texts[id];
                if ((uint8_t) t[0] == s[pos] && t.size() <= n - pos && memcmp(t.data(), s + pos, t.size()) == 0) {
                    hits.push_back(Hit{(uint32_t) k, (uint32_t) id, pos, pos + t.size()});
                }
            }
        };
        if (special_first_count == 1) {
            const uint8_t * p = s;
            const uint8_t * e = s + n;
            while (p < e && (p = (const uint8_t *) memchr(p, special_first_byte, (size_t) (e - p))) != nullptr) {
                probe((size_t) (p - s));
                ++p;
            }
        } else {
            for (size_t i = 0; i < n; ++i) {
                if (special_first[s[i]]) probe(i);
            }
        }
    }

    if (hits.empty()) {
        encode_fragment(s, n, out);
        return;
    }

    // hits are in position order (ties: priority order).  Visit them by priority and accept each one that
    // does not overlap an already accepted hit; only neighbours within one token length can overlap.
    std::vector<uint32_t> order(hits.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = (uint32_t) i;
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return hits[a].prio < hits[b].prio; });
    const size_t max_len = texts[specials[0]].size();   // specials are sorted longest first
    std::vector<uint8_t> taken(hits.size(), 0);
    for (uint32_t i : order) {
        const Hit & h = hits[i];
        bool overlap = false;
        for (size_t j = i; j-- > 0 && hits[j].pos + max_len > h.pos;) {
            if (taken[j] && hits[j].end > h.pos) {
                overlap = true;
                break;
            }
        }
        for (size_t j = i + 1; !overlap && j < hits.size() && hits[j].pos < h.end; ++j) {
            if (taken[j]) overlap = true;
        }
        if (!overlap) taken[i] = 1;
    }
    std::vector<Hit> accepted;
    for (size_t i = 0; i < hits.size(); ++i) {
        if (taken[i]) accepted.push_back(hits[i]);
    }

    size_t at = 0;
    for (const Hit & h : accepted) {
        if (h.pos > at) encode_fragment(s + at, h.pos - at, out);
        out.push_back((int32_t) h.id);
        at = h.end;
    }
    if (at < n) encode_fragment(s + at, n - at, out);
}

// ---------------------------------------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------------------------------------

Tokenizer::Tokenizer(const GGUF & gguf) : impl_(new Impl(gguf)) {}
Tokenizer::~Tokenizer() = default;

std::vector<int32_t> Tokenizer::encode(std::string_view text, bool parse_special) const {
    std::vector<int32_t> out;
    impl_->encode(text, parse_special, out);
    return out;
}

std::string Tokenizer::decode(const std::vector<int32_t> & ids, bool render_special) const {
    const Impl & m = *impl_;
    std::string out;
    size_t total = 0;
    for (int32_t id : ids) {
        if (id >= 0 && id < m.n_vocab) total += m.pieces[id].size();
    }
    out.reserve(total);
    for (int32_t id : ids) {
        if (id < 0 || id >= m.n_vocab) continue;
        if (!render_special && (m.types[id] == Impl::T_CONTROL || m.types[id] == Impl::T_UNKNOWN)) continue;
        out += m.pieces[id];
    }
    return out;
}

const std::string & Tokenizer::piece(int32_t id) const {
    if (id < 0 || id >= impl_->n_vocab) return impl_->empty;
    return impl_->pieces[id];
}

bool Tokenizer::is_special(int32_t id) const {
    if (id < 0 || id >= impl_->n_vocab) return false;
    return impl_->types[id] != Impl::T_NORMAL;
}

int32_t Tokenizer::token_id(std::string_view text) const {
    const Impl & m = *impl_;
    if (text.empty()) return -1;
    // special tokens are matched on their literal text
    if (m.special_first[(uint8_t) text[0]]) {
        for (int32_t id : m.specials) {
            if (m.texts[id] == text) return id;
        }
    }
    const int64_t sym = m.find_sym((const uint8_t *) text.data(), text.size());
    return (sym >= 0 && sym < m.n_vocab) ? (int32_t) sym : -1;
}

int Tokenizer::n_vocab() const { return impl_->n_vocab; }
int32_t Tokenizer::eos_id() const { return impl_->eos; }

}  // namespace ink
