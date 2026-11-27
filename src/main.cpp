// inkling-server: OpenAI-compatible HTTP server for Inkling 2 Mini.
#include "engine.h"
#include "gguf.h"
#include "server.h"
#include "tokenizer.h"

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#include <sys/sysctl.h>
#endif
#include <unistd.h>

namespace {

const char * DEFAULT_MODEL_NAME = "Inkling-2-Mini-UD-Q8_K_XL.gguf";

// Directory containing the running executable ("." if it cannot be determined).
std::string executable_dir(const char * argv0) {
    char raw[PATH_MAX] = {0};
#if defined(__APPLE__)
    uint32_t size = sizeof(raw);
    if (_NSGetExecutablePath(raw, &size) != 0) raw[0] = 0;
#endif
    if (!raw[0] && argv0) snprintf(raw, sizeof(raw), "%s", argv0);
    char resolved[PATH_MAX];
    std::string path = realpath(raw, resolved) ? resolved : raw;
    size_t slash = path.rfind('/');
    return slash == std::string::npos ? "." : path.substr(0, slash);
}

const char * MODEL_HINT =
    "Model files (GGUF): https://huggingface.co/unsloth/Inkling-2-Mini-GGUF\n"
    "Pass one with --model PATH, or set INKLING_MODEL, or put Inkling-2-Mini-UD-Q8_K_XL.gguf\n"
    "in ./models/ or next to this folder.\n";

void usage(const char * prog, const std::string & default_model) {
    printf("usage: %s --model PATH [options]\n\n"
           "  --model PATH     GGUF model file of Inkling 2 Mini.  Without it: $INKLING_MODEL, then\n"
           "                   %s in ./models/ or next to this folder\n"
           "  --host HOST      address to listen on (default: 127.0.0.1)\n"
           "  --port N         port to listen on, 0 = any free port (default: 8080)\n"
           "  --ctx N          maximum context length in tokens (default: engine default)\n"
           "  --batch N        prompt-processing batch size (default: engine default)\n"
           "  --precise        generate with the original 16-bit attention/output weights instead of\n"
           "                   8-bit copies (about 25%% slower generation, exact weights)\n"
           "  --low-mem        expand expert weights to 16-bit per layer during prompt processing instead of\n"
           "                   keeping 16-bit copies of all layers (much less memory, slower prompts)\n"
           "  --kv8            generate from an 8-bit copy of the attention cache: faster at long context\n"
           "                   (tens of thousands of tokens), slightly approximate\n"
           "  --draft-model PATH, -md PATH\n"
           "                   DSpark draft model (e.g. ../dspark-Inkling-2-Mini-Q8_0.gguf): it drafts the next\n"
           "                   tokens and one multi-token pass verifies them.  Faster generation, most of all\n"
           "                   at long context and on output that repeats code; same output distribution\n"
           "  --draft N        max tokens drafted per pass, 0-7 (default 7 with --draft-model, else 0).  Without\n"
           "                   a draft model, drafts are copied from the context (experimental, little gain)\n"
           "  --keep-wired S   keep GPU memory wired for S seconds after the last request, 0 = forever\n"
           "                   (default: 1800; the first request after that pays ~0.5 s to re-map)\n"
           "  --keep-warm S    keep the GPU out of its idle power state for S seconds after the last\n"
           "                   request (default: 600, 0 = off; avoids ~0.3 s of latency after idling)\n"
           "  --alias NAME     model id reported by the API (default: inkling-2-mini)\n"
           "  --api-key KEY    require 'Authorization: Bearer KEY'\n"
           "  --verbose        verbose engine logging\n"
           "  --help           show this help\n",
           prog, default_model.c_str());
}

bool parse_int(const char * s, int & out) {
    char * end = nullptr;
    long v = strtol(s, &end, 10);
    if (!*s || *end || v < 0 || v > INT_MAX) return false;
    out = (int) v;
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    const std::string default_model = DEFAULT_MODEL_NAME;
    const std::string exe_dir = executable_dir(argv[0]);
    std::string model;
    bool draft_given = false;   // --draft N was passed (so --draft 0 with a draft model means no drafting)
    ink::EngineOptions eopt;
    ink::ServerOptions sopt;

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        auto value = [&]() -> const char * {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s requires a value\n", arg.c_str());
                exit(2);
            }
            return argv[++i];
        };
        auto int_value = [&](int & out, int min) {
            const char * v = value();
            if (!parse_int(v, out) || out < min) {
                fprintf(stderr, "error: invalid value for %s: %s\n", arg.c_str(), v);
                exit(2);
            }
        };
        if (arg == "--help" || arg == "-h") {
            usage(argv[0], default_model);
            return 0;
        } else if (arg == "--model" || arg == "-m") {
            model = value();
        } else if (arg == "--host") {
            sopt.host = value();
        } else if (arg == "--port") {
            int_value(sopt.port, 0);
            if (sopt.port > 65535) {
                fprintf(stderr, "error: invalid port\n");
                return 2;
            }
        } else if (arg == "--ctx" || arg == "-c") {
            int_value(eopt.max_ctx, 1);
        } else if (arg == "--batch" || arg == "-b") {
            int_value(eopt.batch, 1);
        } else if (arg == "--keep-wired") {
            int_value(eopt.keep_wired_s, 0);
        } else if (arg == "--keep-warm") {
            int_value(eopt.keep_warm_s, 0);
        } else if (arg == "--low-mem") {
            eopt.low_mem = true;
        } else if (arg == "--kv8") {
            eopt.kv8 = true;
        } else if (arg == "--draft-model" || arg == "-md") {
            eopt.draft_model = value();
        } else if (arg == "--draft") {
            int_value(eopt.draft, 0);
            if (eopt.draft > 7) eopt.draft = 7;
            draft_given = true;
        } else if (arg == "--precise") {
            eopt.q8 = false;
        } else if (arg == "--alias") {
            sopt.alias = value();
        } else if (arg == "--api-key") {
            sopt.api_key = value();
        } else if (arg == "--verbose" || arg == "-v") {
            eopt.verbose = true;
            sopt.verbose = true;
        } else {
            fprintf(stderr, "error: unknown argument: %s\n\n", arg.c_str());
            usage(argv[0], default_model);
            return 2;
        }
    }

    if (!eopt.draft_model.empty() && !draft_given) eopt.draft = 7;

    // no --model: the environment, then the default file in the usual places
    if (model.empty()) {
        const char * env = getenv("INKLING_MODEL");
        if (env && *env) model = env;
    }
    if (model.empty()) {
        for (const char * dir : {"/models/", "/../", "/"}) {
            const std::string cand = exe_dir + dir + DEFAULT_MODEL_NAME;
            if (access(cand.c_str(), R_OK) == 0) { model = cand; break; }
        }
    }
    if (model.empty() || access(model.c_str(), R_OK) != 0) {
        if (model.empty()) fprintf(stderr, "error: no model file given\n\n%s", MODEL_HINT);
        else fprintf(stderr, "error: cannot read model file %s\n\n%s", model.c_str(), MODEL_HINT);
        return 1;
    }
    if (!eopt.draft_model.empty() && access(eopt.draft_model.c_str(), R_OK) != 0) {
        fprintf(stderr, "error: cannot read draft model file %s\n", eopt.draft_model.c_str());
        return 1;
    }
    // memory: the full-speed mode keeps 16-bit copies of all expert weights
    {
        uint64_t mem = 0;
        size_t len = sizeof(mem);
        if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) {
            const double gb = (double) mem / (1ull << 30);
            if (!eopt.low_mem && gb < 90)
                fprintf(stderr, "warning: this Mac has %.0f GB of memory; without --low-mem the server needs 61-71 GB.\n"
                                "         Consider --low-mem (18-32 GB, prompts 9-21%% slower).\n", gb);
            else if (eopt.low_mem && gb < 30)
                fprintf(stderr, "warning: this Mac has %.0f GB of memory; --low-mem needs 18-32 GB depending on the model file.\n", gb);
        }
    }

    try {
        char resolved[PATH_MAX];
        if (realpath(model.c_str(), resolved)) model = resolved;
        fprintf(stderr, "loading %s\n", model.c_str());
        ink::GGUF gguf(model);
        {
            const std::string arch = gguf.str("general.architecture", "");
            if (arch != "inkling2" && arch != "inkling2_loop") {
                fprintf(stderr, "error: this server only runs Inkling 2 Mini; %s is a '%s' model\n\n%s", model.c_str(), arch.c_str(), MODEL_HINT);
                return 1;
            }
        }
        ink::Tokenizer tokenizer(gguf);
        std::unique_ptr<ink::Engine> engine = ink::Engine::create(gguf, eopt);
        fprintf(stderr, "model ready: vocab %d, context %d tokens\n", engine->n_vocab(), engine->max_ctx());
        return ink::run_server(*engine, tokenizer, gguf, sopt);
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
