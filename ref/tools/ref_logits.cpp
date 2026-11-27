// Dev-only reference tool: runs upstream llama.cpp on a token-id file and dumps logits.
// Not part of the engine; used to validate it.
#include "llama.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
int main(int argc, char ** argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s model.gguf tokens.txt out.bin [n_greedy] [n_batch] [dump_all 0/1]\n", argv[0]); return 1; }
    int n_greedy = argc > 4 ? atoi(argv[4]) : 0;
    int n_batch = argc > 5 ? atoi(argv[5]) : 2048;
    bool dump_all = argc > 6 && atoi(argv[6]);
    std::vector<llama_token> toks; { std::ifstream f(argv[2]); long long t; while (f >> t) toks.push_back((llama_token) t); }
    llama_backend_init();
    auto mp = llama_model_default_params(); mp.n_gpu_layers = getenv("NGL") ? atoi(getenv("NGL")) : 999;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    auto cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) toks.size() + n_greedy + 64; cp.n_batch = n_batch; cp.n_ubatch = n_batch;
    llama_context * ctx = llama_init_from_model(model, cp);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    int nv = llama_vocab_n_tokens(vocab);
    FILE * out = fopen(argv[3], "wb");
    int32_t hdr[2] = { dump_all ? (int32_t) toks.size() : 1, nv };
    fwrite(hdr, 4, 2, out);
    llama_batch b = llama_batch_init(n_batch, 0, 1);
    int pos = 0;
    while (pos < (int) toks.size()) {
        int n = std::min(n_batch, (int) toks.size() - pos);
        b.n_tokens = n;
        for (int i = 0; i < n; ++i) { b.token[i] = toks[pos + i]; b.pos[i] = pos + i; b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = dump_all || (pos + i == (int) toks.size() - 1); }
        if (llama_decode(ctx, b)) { fprintf(stderr, "decode failed\n"); return 1; }
        if (dump_all) for (int i = 0; i < n; ++i) fwrite(llama_get_logits_ith(ctx, i), 4, nv, out);
        pos += n;
    }
    const float * lg = llama_get_logits_ith(ctx, -1);
    if (!dump_all) fwrite(lg, 4, nv, out);
    std::vector<llama_token> gen;
    for (int g = 0; g < n_greedy; ++g) {
        lg = llama_get_logits_ith(ctx, -1);
        int best = (int) (std::max_element(lg, lg + nv) - lg);
        gen.push_back(best);
        b.n_tokens = 1; b.token[0] = best; b.pos[0] = pos++; b.n_seq_id[0] = 1; b.seq_id[0][0] = 0; b.logits[0] = 1;
        if (llama_decode(ctx, b)) return 1;
    }
    int32_t ng = (int32_t) gen.size(); fwrite(&ng, 4, 1, out); fwrite(gen.data(), 4, gen.size(), out);
    if (n_greedy) { lg = llama_get_logits_ith(ctx, -1); fwrite(lg, 4, nv, out); }
    fclose(out);
    printf("greedy:"); for (auto t : gen) printf(" %d", t); printf("\n");
    return 0;
}
