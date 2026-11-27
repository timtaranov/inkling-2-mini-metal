// Small CLI around ink::Tokenizer, used by tests/tok_compare.py.
//
//   tok_test <model.gguf> encode [--no-special] [--lines] <file|->   ids of the input (space separated)
//   tok_test <model.gguf> decode [--no-special] <file|->             bytes for whitespace separated ids
//   tok_test <model.gguf> roundtrip <file|->                         checks decode(encode(x)) == x
//   tok_test <model.gguf> batch [--no-special] <listfile>            one line of ids per file named in listfile,
//                                                                    followed by " | rt=0/1" (round-trip result)
//   tok_test <model.gguf> pieces <file|->                            "id<TAB>escaped piece" per token
//   tok_test <model.gguf> dump [--no-special] <file|->               "%6d -> 'piece'" per token (llama-tokenize layout)
//   tok_test <model.gguf> bench <file> [iterations]                  timing
//   tok_test <model.gguf> info                                       vocabulary summary
//   tok_test <model.gguf> specials                                   "id<TAB>text" of every special token

#include "../src/gguf.h"
#include "../src/tokenizer.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

static bool read_all(const std::string & path, std::string & out) {
    if (path == "-") {
        out.assign(std::istreambuf_iterator<char>(std::cin), std::istreambuf_iterator<char>());
        return true;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return true;
}

static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

static void print_ids(const std::vector<int32_t> & ids, char sep) {
    std::string line;
    line.reserve(ids.size() * 7);
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i) line.push_back(sep);
        line += std::to_string(ids[i]);
    }
    fwrite(line.data(), 1, line.size(), stdout);
}

static int usage() {
    fprintf(stderr,
            "usage: tok_test <model.gguf> encode [--no-special] [--lines] <file|->\n"
            "       tok_test <model.gguf> decode [--no-special] <file|->\n"
            "       tok_test <model.gguf> roundtrip <file|->\n"
            "       tok_test <model.gguf> batch [--no-special] <listfile>\n"
            "       tok_test <model.gguf> pieces <file|->\n"
            "       tok_test <model.gguf> bench <file> [iterations]\n"
            "       tok_test <model.gguf> info\n");
    return 2;
}

int main(int argc, char ** argv) {
    if (argc < 3) return usage();
    const std::string model = argv[1];
    const std::string mode = argv[2];
    bool special = true, lines = false;
    std::vector<std::string> args;
    for (int i = 3; i < argc; ++i) {
        if (!strcmp(argv[i], "--no-special")) special = false;
        else if (!strcmp(argv[i], "--lines")) lines = true;
        else args.push_back(argv[i]);
    }

    try {
        const double t0 = now_ms();
        ink::GGUF gguf(model);
        const double t1 = now_ms();
        ink::Tokenizer tok(gguf);
        const double t2 = now_ms();

        if (mode == "info") {
            int n_special = 0;
            for (int i = 0; i < tok.n_vocab(); ++i) n_special += tok.is_special(i);
            printf("n_vocab %d, special %d, eos %d ('%s')\n", tok.n_vocab(), n_special, tok.eos_id(),
                   tok.piece(tok.eos_id()).c_str());
            printf("gguf parse %.1f ms, tokenizer construction %.1f ms\n", t1 - t0, t2 - t1);
            return 0;
        }

        if (mode == "specials") {   // "id<TAB>text" for every non-normal token
            for (int i = 0; i < tok.n_vocab(); ++i) {
                if (tok.is_special(i)) printf("%d\t%s\n", i, tok.piece(i).c_str());
            }
            return 0;
        }

        if (args.empty()) return usage();
        std::string text;

        if (mode == "batch") {
            std::ifstream list(args[0]);
            if (!list) {
                fprintf(stderr, "cannot open %s\n", args[0].c_str());
                return 1;
            }
            std::string path;
            while (std::getline(list, path)) {
                if (path.empty()) continue;
                if (!read_all(path, text)) {
                    fprintf(stderr, "cannot open %s\n", path.c_str());
                    return 1;
                }
                const std::vector<int32_t> ids = tok.encode(text, special);
                print_ids(ids, ' ');
                printf(" | rt=%d\n", tok.decode(ids, true) == text ? 1 : 0);
            }
            return 0;
        }

        if (!read_all(args[0], text)) {
            fprintf(stderr, "cannot open %s\n", args[0].c_str());
            return 1;
        }

        if (mode == "encode") {
            print_ids(tok.encode(text, special), lines ? '\n' : ' ');
            printf("\n");
        } else if (mode == "decode") {
            std::vector<int32_t> ids;
            std::istringstream in(text);
            long long v;
            while (in >> v) ids.push_back((int32_t) v);
            const std::string out = tok.decode(ids, special);
            fwrite(out.data(), 1, out.size(), stdout);
        } else if (mode == "roundtrip") {
            const std::vector<int32_t> ids = tok.encode(text, special);
            const bool ok = tok.decode(ids, true) == text;
            printf("%s (%zu bytes, %zu tokens)\n", ok ? "OK" : "FAIL", text.size(), ids.size());
            return ok ? 0 : 1;
        } else if (mode == "pieces") {
            for (int32_t id : tok.encode(text, special)) {
                printf("%d\t", id);
                for (unsigned char c : tok.piece(id)) {
                    if (c >= 0x20 && c < 0x7F && c != '\\') putchar(c);
                    else printf("\\x%02x", c);
                }
                printf("\n");
            }
        } else if (mode == "dump") {   // same layout as llama-tokenize's default output, for diffing pieces
            for (int32_t id : tok.encode(text, special)) {
                printf("%6d -> '", id);
                const std::string & p = tok.piece(id);
                fwrite(p.data(), 1, p.size(), stdout);
                printf("'\n");
            }
        } else if (mode == "bench") {
            const int iters = args.size() > 1 ? atoi(args[1].c_str()) : 20;
            size_t n_tok = 0;
            double best = 1e30, total = 0;
            for (int i = 0; i < iters; ++i) {
                const double a = now_ms();
                const std::vector<int32_t> ids = tok.encode(text, special);
                const double b = now_ms();
                n_tok = ids.size();
                best = b - a < best ? b - a : best;
                total += b - a;
            }
            const double mb = text.size() / 1e6;
            printf("gguf parse %.1f ms, tokenizer construction %.1f ms\n", t1 - t0, t2 - t1);
            printf("%zu bytes -> %zu tokens: best %.3f ms (%.1f MB/s), mean %.3f ms (%.1f MB/s) over %d runs\n",
                   text.size(), n_tok, best, mb / (best / 1e3), total / iters, mb / (total / iters / 1e3), iters);
            const std::vector<int32_t> ids = tok.encode(text, special);
            const double a = now_ms();
            const std::string back = tok.decode(ids, true);
            const double b = now_ms();
            printf("decode %.3f ms, round-trip %s\n", b - a, back == text ? "OK" : "differs");
        } else {
            return usage();
        }
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
