// Renders chat-template test cases with ink::render_chat and prints results in the same JSON shape
// as tests/chat_ref.py, so tests/chat_test.py can diff the two.  Also runs a few self-checks of the
// Python-formatting helpers.
//   chat_test CASES.json > RESULTS.json
#include "../src/chat.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>

#include <nlohmann/json.hpp>

using ink::ojson;

static int self_check() {
    struct { double v; const char * want; } floats[] = {
        {0.9, "0.9"}, {0.99, "0.99"}, {0.2, "0.2"}, {0.1, "0.1"}, {0.7, "0.7"}, {1e-05, "1e-05"}, {1.5e-05, "1.5e-05"},
        {0.0001, "0.0001"}, {0.00012345, "0.00012345"}, {1.0, "1.0"}, {100.0, "100.0"}, {1e16, "1e+16"},
        {1234567890123456.0, "1234567890123456.0"}, {1e22, "1e+22"}, {-2.5, "-2.5"}, {0.30000000000000004, "0.30000000000000004"},
        {1e-310, "1e-310"}, {123456.789, "123456.789"}, {5e-324, "5e-324"}, {1.7976931348623157e308, "1.7976931348623157e+308"},
    };
    int bad = 0;
    for (auto & f : floats) {
        std::string got = ink::py_float_repr(f.v);
        if (got != f.want) {
            fprintf(stderr, "py_float_repr(%g): got %s want %s\n", f.v, got.c_str(), f.want);
            bad++;
        }
    }
    struct { const char * s; size_t want; } utf[] = {
        {"abc", 3}, {"ab\xC3", 2}, {"ab\xC3\xA9", 4}, {"\xE2\x82", 0}, {"\xE2\x82\xAC", 3}, {"x\xF0\x9F\x98", 1},
        {"x\xF0\x9F\x98\x80", 5}, {"", 0}, {"\x80\x80\x80\x80", 4},
    };
    for (auto & u : utf) {
        size_t got = ink::utf8_complete_len(u.s);
        if (got != u.want) {
            fprintf(stderr, "utf8_complete_len case failed: got %zu want %zu\n", got, u.want);
            bad++;
        }
    }
    return bad;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s CASES.json\n", argv[0]);
        return 2;
    }
    if (int bad = self_check()) {
        fprintf(stderr, "%d self-check failures\n", bad);
        return 1;
    }
    std::ifstream in(argv[1]);
    std::stringstream ss;
    ss << in.rdbuf();
    ojson cases = ojson::parse(ss.str());
    ojson results = ojson::array();
    for (auto & c : cases) {
        ojson r = ojson::object();
        try {
            ojson request = c.at("request");
            if (c.value("canonicalize", false)) ink::canonicalize_chat_request(request);
            // the parts API must concatenate to the same text
            std::string text = ink::render_chat(request);
            std::string joined;
            for (const auto & p : ink::render_chat_parts(request)) joined += p.text;
            if (joined != text) throw std::logic_error("parts/text mismatch");
            r["ok"] = true;
            r["text"] = text;
        } catch (const ink::ChatError & e) {
            r["ok"] = false;
            r["raised"] = e.raised;
            r["message"] = e.what();
        }
        r["name"] = c.at("name");
        results.push_back(std::move(r));
    }
    std::cout << results.dump(-1, ' ', false, ojson::error_handler_t::replace);
    return 0;
}
