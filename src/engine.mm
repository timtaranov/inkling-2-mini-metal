// Inkling-2-Mini inference engine for Apple M5 Max (Metal).
//
// Two execution paths share one set of weights and one recurrent state:
//   * prefill  - batched prompt processing built on Metal Performance Primitives GEMM (FP16 weights)
//   * decode   - single-token generation with fused, bandwidth-bound mat-vec kernels (NVFP4 / FP16 weights)
//
// Architecture notes (matching the reference graph):
//   24 physical blocks; blocks [loopStart, loopEnd) are executed loopCount times ("visits": 48 in total),
//   each visit has its own KV cache and convolution state.  5 of 6 blocks use sliding-window attention.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "engine.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include "quant_tables.h"
#include "kernels_metal.h"   // generated: kernels_metal_src[] / kernels_metal_src_len

namespace ink {

namespace {

double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---- argument structs (must match kernels.metal) -------------------------------------------------
struct CountArgs { uint32_t w, h; float f; uint32_t pad; };
struct NvArgs { uint32_t rows, K, rowBytes, pad; };
struct GemmArgs { uint32_t N, M, K, ldb, ldc, pad0, pad1, pad2; };
struct MoeArgs { uint32_t M, K, nTiles, pad; };
struct EmbedArgs { uint32_t N; float eps; };
struct NormArgs { uint32_t N; float eps; };
struct MixArgs {
    uint32_t N, nStates, nq, mode;
    uint32_t slot0, slot1, slot2, slot3;
    uint32_t nbStride, outStride, wantF32;
    float eps, mixEps, qscale;
    uint32_t wStride, pad;
};
struct CtrlArgs { uint32_t N; float eps; uint32_t wantF32, pad; };
struct PrepQArgs { uint32_t N, pos0, PW, qCol, rCol; float qThr, rThr, tauAlpha, tauFloor; uint32_t useTau; float qScale; uint32_t pad0, pad1, pad2, pad3, pad4; };
struct PrepKVArgs {
    uint32_t N, pos0, PW, col, dynCol, useDyn, goff, isV;
    float thr, eps, invs;
    uint32_t slotMask, stStride, pad0, pad1, pad2;
};
struct ConvArgs { uint32_t N, C, goff, mode; float invs; uint32_t ldx, xcol, stStride; };
struct AttnArgs { uint32_t N, pos0, swa, slotMask, window, extent; float scale; uint32_t PW, sinkCol, gateCol, pad0, pad1; };
struct RouteArgs { uint32_t N; float wscale, eps; uint32_t pad; };
struct RowsArgs { uint32_t R, N; float beta, lbeta; };

// decode kernels
struct DEmbedArgs { uint32_t tok; float eps; };
struct DNormArgs { uint32_t nj, srcStride, dstStride, wOff0, wOff1, wOff2, useW; float eps; };
struct DMixArgs {
    uint32_t nStates, nq, mode, nbStride;
    uint32_t slot0, slot1, slot2, slot3;
    uint32_t outStride, wStride; float eps, mixEps, qscale; uint32_t pad0, pad1, pad2;
};
struct DGateArgs { uint32_t nj, xStride, msgStride; float eps; };
struct DCtrlArgs { uint32_t nj, rank, ctrlBase, per, xStride, msgStride, pad0, pad1; };
struct DProjArgs { uint32_t rows, kCol, vCol, xStride; };
struct DPrepArgs {
    uint32_t slot, qCol, rCol, kCol, vCol, kDynCol, vDynCol, useDyn;
    float qThr, rThr, tau, eps, invs; uint32_t stStride, pad0, pad1;
};
struct DAttnArgs { uint32_t pos, lo, nk, slotMask, extent; float scale; uint32_t nChunks, pad; };
struct KV8Args { uint32_t pos0, N, pad0, pad1; };
struct MAttnArgs { uint32_t N, pos0, nChunks, chunkKeys, extent, PW, sinkCol, gateCol, window, slotMask, base0, flags, chunk0, pad0, pad1, pad2; };
struct DsMixArgs { uint32_t N, tapStride, wantF; float eps, m0, m1, m2, m3; };
struct DsEmbedArgs { uint32_t tok; float eps; uint32_t pad0, pad1; };
struct DsPrepArgs { uint32_t N, ldp, qCol, kCol, vCol, u0, slot0; float qscale, eps; uint32_t pad0, pad1, pad2; };
struct DsStepArgs { uint32_t step, vocab, hStride; float cap; };
struct MMvArgs { uint32_t K, ldx, ldy; float cap; };
const int MINI_MAX = 8;          // tokens per multi-token generation pass
const int MINI_CHUNK = 1024;     // keys per threadgroup of the multi-token attention
struct DMergeArgs { uint32_t nIn, nOut, final, sinkCol, gateCol, pad0, pad1, pad2; };
struct DOutArgs { uint32_t goff, mode; float invs; uint32_t stStride; };
struct DRouteArgs { float wscale, eps; uint32_t pad0, pad1; };
struct DExpArgs { float beta, lbeta; uint32_t xStride, pad1; };
struct DLogitArgs { uint32_t rows; float cap; };

struct B {
    id<MTLBuffer> buf;
    size_t off = 0;
    B(id<MTLBuffer> b, size_t o = 0) : buf(b), off(o) {}
};

float ue4m3(int x) {
    if (x == 0 || x == 0x7f) return 0.0f;
    const int e = (x >> 3) & 15, m = x & 7;
    const float raw = e == 0 ? ldexpf((float) m, -9) : ldexpf(1.0f + (float) m / 8.0f, e - 7);
    return raw * 0.5f;
}

// ---- GGUF weight formats -> f32 (CPU).  Elements [i0, i1) of a tensor's data; for block formats both
// bounds must be multiples of the block size (32 or 256).
bool dequant_supported(uint32_t type) {
    switch (type) {
        case GG_F32: case GG_F16: case GG_BF16: case GG_Q8_0: case GG_Q4_0: case GG_Q5_0: case GG_Q4_K: case GG_Q5_K: case GG_Q6_K:
        case GG_IQ2_XXS: case GG_IQ2_XS: case GG_IQ2_S: case GG_IQ3_XXS: case GG_IQ1_S: case GG_IQ4_XS: case GG_MXFP4: case GG_NVFP4: case GG_Q2_0:
            return true;
        default: return false;
    }
}

size_t dequant_block(uint32_t type) {
    switch (type) {
        case GG_Q8_0: case GG_Q4_0: case GG_Q5_0: case GG_MXFP4: return 32;
        case GG_NVFP4: case GG_Q2_0: return 64;
        case GG_Q4_K: case GG_Q5_K: case GG_Q6_K:
        case GG_IQ2_XXS: case GG_IQ2_XS: case GG_IQ2_S: case GG_IQ3_XXS: case GG_IQ1_S: case GG_IQ4_XS: return 256;
        default: return 1;
    }
}

const int8_t kFP4c[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};   // doubled E2M1 values

float half_at(const uint8_t * p) {
    __fp16 h;
    memcpy(&h, p, 2);
    return (float) h;
}

// 6-bit scale / min number k of a K-quant super-block
void k4_scale_min(int k, const uint8_t * sc, uint8_t * s6, uint8_t * m6) {
    if (k < 4) { *s6 = sc[k] & 63; *m6 = sc[k + 4] & 63; }
    else { *s6 = (uint8_t) ((sc[k + 4] & 0xF) | ((sc[k - 4] >> 6) << 4)); *m6 = (uint8_t) ((sc[k + 4] >> 4) | ((sc[k] >> 6) << 4)); }
}

void dequant_range(uint32_t type, const uint8_t * src, size_t i0, size_t i1, float * y) {
    switch (type) {
        case GG_F32: memcpy(y, src + i0 * 4, (i1 - i0) * 4); break;
        case GG_F16: for (size_t i = i0; i < i1; ++i) *y++ = half_at(src + i * 2); break;
        case GG_BF16:
            for (size_t i = i0; i < i1; ++i) {
                uint16_t b; memcpy(&b, src + i * 2, 2);
                const uint32_t u = (uint32_t) b << 16;
                memcpy(y++, &u, 4);
            }
            break;
        case GG_Q8_0:
            for (size_t b = i0 / 32; b < i1 / 32; ++b) {
                const uint8_t * blk = src + b * 34;
                const float d = half_at(blk);
                const int8_t * q = (const int8_t *) (blk + 2);
                for (int k = 0; k < 32; ++k) *y++ = d * (float) q[k];
            }
            break;
        case GG_Q4_0:
            for (size_t b = i0 / 32; b < i1 / 32; ++b, y += 32) {
                const uint8_t * blk = src + b * 18;
                const float d = half_at(blk);
                for (int j = 0; j < 16; ++j) {
                    y[j] = (float) ((blk[2 + j] & 0x0F) - 8) * d;
                    y[j + 16] = (float) ((blk[2 + j] >> 4) - 8) * d;
                }
            }
            break;
        case GG_Q5_0:
            for (size_t b = i0 / 32; b < i1 / 32; ++b, y += 32) {
                const uint8_t * blk = src + b * 22;
                const float d = half_at(blk);
                uint32_t qh; memcpy(&qh, blk + 2, 4);
                const uint8_t * qs = blk + 6;
                for (int j = 0; j < 16; ++j) {
                    const uint8_t h0 = (uint8_t) (((qh >> j) << 4) & 0x10), h1 = (uint8_t) ((qh >> (j + 12)) & 0x10);
                    y[j] = (float) (((qs[j] & 0x0F) | h0) - 16) * d;
                    y[j + 16] = (float) (((qs[j] >> 4) | h1) - 16) * d;
                }
            }
            break;
        case GG_Q4_K:   // {d, dmin, 12 bytes of 6-bit scales/mins, 128 bytes of nibbles}
            for (size_t b = i0 / 256; b < i1 / 256; ++b) {
                const uint8_t * blk = src + b * 144;
                const float d = half_at(blk), mn = half_at(blk + 2);
                const uint8_t * sc = blk + 4, * q = blk + 16;
                for (int j = 0, is = 0; j < 4; ++j, is += 2, q += 32) {
                    uint8_t s6, m6;
                    k4_scale_min(is, sc, &s6, &m6);
                    const float d1 = d * s6, m1 = mn * m6;
                    k4_scale_min(is + 1, sc, &s6, &m6);
                    const float d2 = d * s6, m2 = mn * m6;
                    for (int l = 0; l < 32; ++l) *y++ = d1 * (float) (q[l] & 0xF) - m1;
                    for (int l = 0; l < 32; ++l) *y++ = d2 * (float) (q[l] >> 4) - m2;
                }
            }
            break;
        case GG_Q5_K:   // {d, dmin, 12 bytes of scales/mins, 32 bytes of high bits, 128 bytes of low nibbles}
            for (size_t b = i0 / 256; b < i1 / 256; ++b) {
                const uint8_t * blk = src + b * 176;
                const float d = half_at(blk), mn = half_at(blk + 2);
                const uint8_t * sc = blk + 4, * qh = blk + 16, * ql = blk + 48;
                uint8_t u1 = 1, u2 = 2;
                for (int j = 0, is = 0; j < 4; ++j, is += 2, ql += 32, u1 <<= 2, u2 <<= 2) {
                    uint8_t s6, m6;
                    k4_scale_min(is, sc, &s6, &m6);
                    const float d1 = d * s6, m1 = mn * m6;
                    k4_scale_min(is + 1, sc, &s6, &m6);
                    const float d2 = d * s6, m2 = mn * m6;
                    for (int l = 0; l < 32; ++l) *y++ = d1 * (float) ((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
                    for (int l = 0; l < 32; ++l) *y++ = d2 * (float) ((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
                }
            }
            break;
        case GG_Q6_K:   // {128 bytes low nibbles, 64 bytes high 2 bits, 16 int8 scales, d}
            for (size_t b = i0 / 256; b < i1 / 256; ++b) {
                const uint8_t * blk = src + b * 210;
                const uint8_t * ql = blk, * qh = blk + 128;
                const int8_t * sc = (const int8_t *) (blk + 192);
                const float d = half_at(blk + 208);
                for (int n = 0; n < 256; n += 128, y += 128, ql += 64, qh += 32, sc += 8) {
                    for (int l = 0; l < 32; ++l) {
                        const int is = l / 16;
                        const int q1 = (int) ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        const int q2 = (int) ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        const int q3 = (int) ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        const int q4 = (int) ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        y[l] = d * (float) sc[is] * (float) q1;
                        y[l + 32] = d * (float) sc[is + 2] * (float) q2;
                        y[l + 64] = d * (float) sc[is + 4] * (float) q3;
                        y[l + 96] = d * (float) sc[is + 6] * (float) q4;
                    }
                }
            }
            break;
        case GG_Q2_0:   // {d, 16 bytes of 2-bit codes}: 00 = -1, 01 = 0, 10 = +1, 11 = +2
            for (size_t b = i0 / 64; b < i1 / 64; ++b) {
                const uint8_t * blk = src + b * 18;
                const float d = half_at(blk);
                for (int j = 0; j < 64; ++j) *y++ = (float) ((int) ((blk[2 + j / 4] >> ((j % 4) * 2)) & 3) - 1) * d;
            }
            break;
        case GG_MXFP4:  // {E8M0 scale, 16 nibble bytes}
            for (size_t b = i0 / 32; b < i1 / 32; ++b, y += 32) {
                const uint8_t * blk = src + b * 17;
                const uint32_t bits = blk[0] < 2 ? (0x00200000u << blk[0]) : ((uint32_t) (blk[0] - 1) << 23);
                float d; memcpy(&d, &bits, 4);
                for (int j = 0; j < 16; ++j) {
                    y[j] = (float) kFP4c[blk[1 + j] & 0x0F] * d;
                    y[j + 16] = (float) kFP4c[blk[1 + j] >> 4] * d;
                }
            }
            break;
        case GG_NVFP4:  // {4 UE4M3 scales, 32 nibble bytes}: four 16-weight sub-blocks
            for (size_t b = i0 / 64; b < i1 / 64; ++b) {
                const uint8_t * blk = src + b * 36;
                for (int sb = 0; sb < 4; ++sb, y += 16) {
                    const float d = ue4m3(blk[sb]);
                    for (int j = 0; j < 8; ++j) {
                        y[j] = (float) kFP4c[blk[4 + sb * 8 + j] & 0x0F] * d;
                        y[j + 8] = (float) kFP4c[blk[4 + sb * 8 + j] >> 4] * d;
                    }
                }
            }
            break;
        case GG_IQ2_XXS:   // {d, 8 x (4 codebook bytes, 4 x 7-bit sign codes + 4-bit scale)}
            for (size_t b = i0 / 256; b < i1 / 256; ++b) {
                const uint8_t * blk = src + b * 66;
                const float d = half_at(blk);
                for (int ib = 0; ib < 8; ++ib) {
                    uint32_t aux[2]; memcpy(aux, blk + 2 + 8 * ib, 8);
                    const float db = d * (0.5f + (float) (aux[1] >> 28)) * 0.25f;
                    for (int l = 0; l < 4; ++l) {
                        const uint64_t grid = qt::iq2xxs_grid[(aux[0] >> (8 * l)) & 255];
                        const uint8_t signs = qt::ksigns_iq2xs[(aux[1] >> (7 * l)) & 127];
                        for (int j = 0; j < 8; ++j) *y++ = db * (float) ((grid >> (8 * j)) & 255) * ((signs >> j) & 1 ? -1.0f : 1.0f);
                    }
                }
            }
            break;
        case GG_IQ2_XS:    // {d, 32 x (9-bit codebook index + 7-bit sign code), 8 bytes of 4-bit scales}
            for (size_t b = i0 / 256; b < i1 / 256; ++b) {
                const uint8_t * blk = src + b * 74;
                const float d = half_at(blk);
                const uint8_t * sc = blk + 66;
                for (int ib = 0; ib < 8; ++ib) {
                    const float db[2] = {d * (0.5f + (float) (sc[ib] & 0xF)) * 0.25f, d * (0.5f + (float) (sc[ib] >> 4)) * 0.25f};
                    for (int l = 0; l < 4; ++l) {
                        uint16_t q; memcpy(&q, blk + 2 + 2 * (4 * ib + l), 2);
                        const uint64_t grid = qt::iq2xs_grid[q & 511];
                        const uint8_t signs = qt::ksigns_iq2xs[q >> 9];
                        for (int j = 0; j < 8; ++j) *y++ = db[l / 2] * (float) ((grid >> (8 * j)) & 255) * ((signs >> j) & 1 ? -1.0f : 1.0f);
                    }
                }
            }
            break;
        case GG_IQ2_S:     // {d, 32 codebook bytes, 32 sign bytes, 8 bytes of high index bits, 8 bytes of scales}
            for (size_t b = i0 / 256; b < i1 / 256; ++b) {
                const uint8_t * blk = src + b * 82;
                const float d = half_at(blk);
                const uint8_t * qs = blk + 2, * sg = blk + 34, * qh = blk + 66, * sc = blk + 74;
                for (int ib = 0; ib < 8; ++ib) {
                    const float db[2] = {d * (0.5f + (float) (sc[ib] & 0xF)) * 0.25f, d * (0.5f + (float) (sc[ib] >> 4)) * 0.25f};
                    for (int l = 0; l < 4; ++l) {
                        const uint64_t grid = qt::iq2s_grid[qs[4 * ib + l] | (((uint32_t) qh[ib] << (8 - 2 * l)) & 0x300)];
                        const uint8_t signs = sg[4 * ib + l];
                        for (int j = 0; j < 8; ++j) *y++ = db[l / 2] * (float) ((grid >> (8 * j)) & 255) * ((signs >> j) & 1 ? -1.0f : 1.0f);
                    }
                }
            }
            break;
        case GG_IQ3_XXS:   // {d, 64 codebook bytes (4 weights each), 8 x (4 x 7-bit sign codes + 4-bit scale)}
            for (size_t b = i0 / 256; b < i1 / 256; ++b) {
                const uint8_t * blk = src + b * 98;
                const float d = half_at(blk);
                const uint8_t * qs = blk + 2;
                for (int ib = 0; ib < 8; ++ib, qs += 8) {
                    uint32_t aux; memcpy(&aux, blk + 66 + 4 * ib, 4);
                    const float db = d * (0.5f + (float) (aux >> 28)) * 0.5f;
                    for (int l = 0; l < 4; ++l) {
                        const uint8_t signs = qt::ksigns_iq2xs[(aux >> (7 * l)) & 127];
                        const uint32_t g1 = qt::iq3xxs_grid[qs[2 * l]], g2 = qt::iq3xxs_grid[qs[2 * l + 1]];
                        for (int j = 0; j < 4; ++j) y[j] = db * (float) ((g1 >> (8 * j)) & 255) * ((signs >> j) & 1 ? -1.0f : 1.0f);
                        for (int j = 0; j < 4; ++j) y[4 + j] = db * (float) ((g2 >> (8 * j)) & 255) * ((signs >> (4 + j)) & 1 ? -1.0f : 1.0f);
                        y += 8;
                    }
                }
            }
            break;
        case GG_IQ1_S:     // {d, 32 codebook bytes, 8 x uint16 (4 x 3 high index bits, 3-bit scale, sign of the offset)}
            for (size_t b = i0 / 256; b < i1 / 256; ++b) {
                const uint8_t * blk = src + b * 50;
                const float d = half_at(blk);
                for (int ib = 0; ib < 8; ++ib) {
                    uint16_t qh; memcpy(&qh, blk + 34 + 2 * ib, 2);
                    const float dl = d * (float) (2 * ((qh >> 12) & 7) + 1);
                    const float delta = (qh & 0x8000) ? -0.125f : 0.125f;
                    for (int l = 0; l < 4; ++l) {
                        const uint64_t grid = qt::iq1s_grid[blk[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7) << 8)];
                        for (int j = 0; j < 8; ++j) *y++ = dl * ((float) (int8_t) ((grid >> (8 * j)) & 255) + delta);
                    }
                }
            }
            break;
        case GG_IQ4_XS:    // {d, 16 bits of high scale bits, 4 bytes of low scale nibbles, 128 nibble bytes}
            for (size_t b = i0 / 256; b < i1 / 256; ++b) {
                const uint8_t * blk = src + b * 136;
                const float d = half_at(blk);
                uint16_t sh; memcpy(&sh, blk + 2, 2);
                const uint8_t * sl = blk + 4, * qs = blk + 8;
                for (int ib = 0; ib < 8; ++ib, y += 32, qs += 16) {
                    const int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((sh >> (2 * ib)) & 3) << 4);
                    const float dl = d * (float) (ls - 32);
                    for (int j = 0; j < 16; ++j) {
                        y[j] = dl * (float) qt::kvalues_iq4nl[qs[j] & 0xF];
                        y[j + 16] = dl * (float) qt::kvalues_iq4nl[qs[j] >> 4];
                    }
                }
            }
            break;
        default: break;
    }
}

// bytes per row of the repacked ("native") layout of an expert format, 0 if it has none (see kernels_quant.metal)
size_t native_row_bytes(uint32_t type, size_t K) {
    const size_t nu = K / 32, ng = K / 8, hdr = ((K / 256) * 2 + 3) & ~(size_t) 3;
    switch (type) {
        case GG_Q4_K: return K / 64 + nu * 2 + K / 2;
        case GG_Q5_0: return nu * 2 + nu * 4 + K / 2;
        case GG_IQ2_XXS: return hdr + nu * 8;
        case GG_IQ2_XS: return hdr + nu + ng * 2;
        case GG_IQ2_S: return hdr + 2 * nu + 2 * ng;
        case GG_IQ1_S: return hdr + nu * 2 + ng;
        case GG_Q2_0: return ((K / 64 * 2 + 3) & ~(size_t) 3) + K / 4;
        default: return 0;
    }
}

const char * expand_kernel(uint32_t type) {
    switch (type) {
        case GG_Q4_K: return "k_x_q4k";
        case GG_Q5_0: return "k_x_q50";
        case GG_Q2_0: return "k_x_q20";
        case GG_IQ2_XXS: return "k_x_iq2xxs";
        case GG_IQ2_XS: return "k_x_iq2xs";
        case GG_IQ2_S: return "k_x_iq2s";
        case GG_IQ1_S: return "k_x_iq1s";
        default: return nullptr;
    }
}

// decode kernels that read a GGUF expert format in place (nullptr = none; 8-bit copies are used instead)
const char * native_kernel(bool gateUp, uint32_t type) {
    switch (type) {
        case GG_Q4_K: return gateUp ? "d_gate_up_q4k" : "d_down_q4k";
        case GG_Q5_0: return gateUp ? "d_gate_up_q50" : "d_down_q50";
        case GG_Q2_0: return gateUp ? "d_gate_up_q20" : "d_down_q20";
        case GG_IQ2_XXS: return gateUp ? "d_gate_up_iq2xxs" : nullptr;
        case GG_IQ2_XS: return gateUp ? "d_gate_up_iq2xs" : nullptr;
        case GG_IQ2_S: return gateUp ? "d_gate_up_iq2s" : nullptr;
        case GG_IQ1_S: return gateUp ? "d_gate_up_iq1s" : nullptr;
        default: return nullptr;
    }
}

// ---- model description -----------------------------------------------------------------------------
struct HParams {
    int D = 0, NH = 0, NKV = 0, DH = 0, NREL = 0, NEXP = 0, NUSED = 0, FF = 0, NCONV = 0;
    int nLayers = 0, loopStart = 0, loopEnd = 0, loopCount = 1, gateRank = 0, blockSize = 6;
    int window = 0, extent = 0, vocab = 0, vocabPadded = 0;
    float eps = 1e-6f, qkClip = 0, relClip = 0, logFloor = 0, logAlpha = 0, wScale = 1, situBeta = 0, situLBeta = 0, softcap = 0;
    std::vector<bool> swa;
};

struct LayerW {
    bool swa = true;
    bool nv = false;                 // experts stored as NVFP4 (decode path)
    id<MTLBuffer> guP, dnP;          // experts repacked from Q4_K / Q5_0, read in place by the decode path
    uint32_t guFmt = 0, dnFmt = 0;   // GGUF type of guP / dnP (0 = none)
    bool loop = false;
    id<MTLBuffer> attn;              // half [PW x D]
    id<MTLBuffer> wo;                // half [D x DQ]
    id<MTLBuffer> small;             // f32 parameters, element offsets below
    size_t oNormQ = 0, oNormK = 0, oNormV = 0, oQClip = 0, oRClip = 0, oRelBias = 0, oKConv = 0, oVConv = 0, oAConv = 0, oFConv = 0;
    size_t oAttnResQ = 0, oFfnResQ = 0, oFfnNorm = 0, oGateInp = 0, oProbsB = 0, oScaleGU = 0, oScaleDN = 0, oMsgQ = 0;
    bool hasAttnResQ = false;
    id<MTLBuffer> guH, dnH;          // half experts [NEXP][2FF][D], [NEXP][D][FF]
    id<MTLBuffer> guNV, dnNV;        // packed NVFP4 experts (rows of [K/16 scales][K/2 nibbles])
    id<MTLBuffer> attnQ8, woQ8, guQ8, dnQ8;   // optional 8-bit copies for the decode path
    id<MTLBuffer> loopW;             // half: 4 x { down [R x 2D], gain|forget [2D x R] }
    std::vector<float> scaleGU, scaleDN;
};

struct Visit { int layer; int pass; };

}  // namespace

// ====================================================================================================
class EngineImpl final : public Engine {
public:
    EngineImpl(const GGUF & gguf, const EngineOptions & opt);
    ~EngineImpl() override;

    int n_vocab() const override { return hp.vocab; }
    int max_ctx() const override { return opt.max_ctx; }
    bool set_prompt(const std::vector<int32_t> & tokens, const std::function<bool(int, int)> & progress) override;
    void append(int32_t token) override;
    const float * logits() override {
        return specN ? (const float *) bLogitsM.contents + (size_t) specRow * hp.vocabPadded : (const float *) bLogits.contents;
    }
    int n_ctx_used() const override { return (int) seq.size(); }
    const std::vector<int32_t> & tokens() const override { return seq; }
    void reset() override;
    EngineTimings timings() const override { return tm; }

private:
    // ---- setup
    void load_hparams(const GGUF & g);
    void build_library();
    void load_weights(const GGUF & g);
    void alloc_state();
    id<MTLBuffer> new_buf(size_t bytes, bool touch = false);
    id<MTLComputePipelineState> pipe(const char * name);
    // specialised pipeline: function constants {S, Q, MODE, NJ}
    id<MTLComputePipelineState> pipe_fc(const char * name, uint32_t S, uint32_t Q, uint32_t mode, uint32_t nj);
    uint32_t fc[4] = {0, 0, 0, 0};
    bool fcOn = false;

    // ---- dispatch helpers
    void begin();
    void flush(bool wait);
    template <typename T>
    void enc_setup(const char * name, const T & args, std::initializer_list<B> bufs, id<MTLComputePipelineState> * out);
    template <typename T>
    void groups(const char * name, const T & args, std::initializer_list<B> bufs, size_t gx, size_t gy, size_t threads);
    template <typename T>
    void threads(const char * name, const T & args, std::initializer_list<B> bufs, size_t tx, size_t ty);
    void blit(id<MTLBuffer> src, size_t so, id<MTLBuffer> dst, size_t d_o, size_t n);
    void gemm(const char * name, B bB, B bA, B bC, uint32_t N, uint32_t M, uint32_t K, uint32_t ldb, uint32_t ldc);

    // ---- forward
    void ensure_kv(int upto);
    void forward_batch(const int32_t * toks, int N, int pos0, bool wantLogits);
    void decode_token(int32_t tok, int pos);
    void control(int N, const LayerW & L, int c, B x, B msg, size_t normOff, B outH, bool wantF32);
    void moe(int N, const LayerW & L);
    void moe_mini(int N, const LayerW & L);
    void forward_mini(const int32_t * toks, int N, int pos0);
    void mmv(const char * kern, int N, B W, uint32_t rows, uint32_t K, B x, uint32_t ldx, B y, uint32_t ldy, float cap = 0);
    // multi-token generation passes: the token to append plus drafts, verified by append()
    bool miniMode = false;           // forward_batch is running such a pass
    int specPos = 0, specN = 0, specRow = 0;
    int32_t specTok[MINI_MAX] = {0};
    bool convPending = false;        // the conv state of the accepted prefix is still in bConvVar
    void settle();
    id<MTLBuffer> bLogitsM, bConvVar, bMRoute, bMA, bMPart, bMAttn, bMGroup;
    int mAttnChunks = 0;
    // n-gram index over the sequence for drafting: hash of the last NGRAM tokens -> end position
    std::unordered_map<uint64_t, int> ngLast;
    int ngIndexed = 0, ngMin = 4;
    void ng_sync();
    int ng_draft(int32_t token, int32_t * out, int maxn);
    long statPasses = 0, statDrafted = 0, statAccepted = 0, statSingles = 0;

    // ---- DSpark draft model (see kernels_dspark.metal)
    struct DsMat { id<MTLBuffer> w; bool q8 = false; };
    struct DsLayer {
        DsMat wq, wk, wv, wo, wg, wu, wd;
        id<MTLBuffer> wkc, wvc;      // context K / V projections, FP16
        id<MTLBuffer> small;         // f32: attn_norm | ffn_norm | ctx_norm (D each) | q_norm | k_norm | k_ctx_norm (DH each)
        float mix[4] = {0, 0, 0, 0};
    };
    bool dsOn = false;
    std::vector<DsLayer> dsL;
    std::vector<int> dsTapVisit;     // visit whose input is tap i; nVisits = the final normalized output
    std::vector<bool> dsSwa;
    int dsWindow = 0, dsBlock = 8;
    float dsMMix[4] = {0, 0, 0, 0}, dsRopeBase = 1e6f, dsCap = 0, dsEps = 1e-6f;
    id<MTLBuffer> dsJoint, dsW1, dsW2, dsSmall;
    size_t dsoJB = 0, dsoW2B = 0, dsoMask = 0, dsoOutN = 0, dsoTokN = 0, dsoMCN = 0;
    std::vector<id<MTLBuffer>> dsK, dsV;   // injected context K / V per decoder layer [positions][DKV] half
    id<MTLBuffer> bTaps, bDsX, bDsP, bDsRope, bDsFrnn;
    static const int FR_SLOTS = 24;
    int frnnPos[FR_SLOTS], frnnCur = 0;
    id<MTLBuffer> bDsCur, bDsXh, bDsQP, bDsQh, bDsBK, bDsBV, bDsO, bDsAO, bDsG, bDsU, bDsAct, bDsH, bDsBase, bDsU5, bDsJ, bDsState,
                  bDsMarkov, bDsBias, bDsCol, bDsArg, bDsTok, bDsProb, bDsRopeD;
    void ds_load(const std::string & path);
    void ds_alloc();
    void ds_tap(int v, int N, B cur);                     // called at the start of visit v (v = nVisits: final output)
    void ds_inject(int N, int pos0);
    int ds_frnn_slot(int pos) const { for (int i = 0; i < FR_SLOTS; ++i) if (frnnPos[i] == pos) return i; return -1; }
    int ds_draft(int32_t token, int pos, int kmax, int32_t * out, float * prob);
    void ds_mv(int N, const DsMat & m, uint32_t rows, uint32_t K, B x, uint32_t ldx, B y, uint32_t ldy);
    float draftMin = 0.5f;          // drafts are kept while the joint probability of the block stays above this
    long statDraftCalls = 0;
    id<MTLBuffer> expert_f16(const LayerW & L, bool gateUp);
    id<MTLBuffer> bXgu, bXdn;      // low-memory mode: FP16 scratch for the experts of the layer being processed

    // optional stage profiler (INK_PROF=1): forces a GPU sync after each stage
    int attnQ = 32, attnC = 128;   // prompt attention tile: queries x keys per threadgroup
    bool useBatchDecode = false;   // debug: run generation through the batched path (INK_BATCH_DECODE=1)
    bool profOn = false;
    double profT = 0;
    std::map<std::string, double> profAcc;
    void prof(const char * label) {
        if (!profOn) return;
        flush(true);
        const double t = now_s();
        profAcc[label] += t - profT;
        profT = t;
    }
    void prof_report(const char * what, int n);

    // Residency set: without it macOS un-wires idle GPU buffers within a second or two and the next
    // large batch pays hundreds of milliseconds to map ~60 GB again.  A heartbeat keeps requesting
    // residency until the engine has been idle for opt.keep_wired_s.
    id<MTLResidencySet> rset;
    std::mutex rsetMu;
    bool rsetDirty = false;
    std::thread heartbeat;
    std::atomic<bool> hbStop{false};
    std::atomic<double> lastUse{0};
    void rset_commit();
    void drop_buf(id<MTLBuffer> b);
    void touch_use() { lastUse.store(now_s()); }

    EngineOptions opt;
    HParams hp;
    EngineTimings tm;

    id<MTLDevice> dev;
    id<MTLCommandQueue> queue;
    id<MTLLibrary> lib;
    std::map<std::string, id<MTLComputePipelineState>> pipes;
    id<MTLCommandBuffer> cb;
    id<MTLComputeCommandEncoder> enc;

    // weights
    std::vector<LayerW> layers;
    id<MTLBuffer> bEmb, bOut, bOutQ8, bGlobalSmall, bScaleTab;
    size_t oTokNorm = 0, oOutNorm = 0, oOutResQ = 0;
    std::vector<Visit> visits;

    // P (projection scratch) column layout
    uint32_t PW = 0, qCol = 0, gateCol = 0, rCol = 0, sinkCol = 0, kCol = 0, kDynCol = 0, vCol = 0, vDynCol = 0;
    uint32_t CH = 0;   // conv state channels per visit

    // state
    int NB_ = 0;       // batch capacity
    uint32_t swaCap = 0;
    int kvCap = 0;     // capacity (positions) of the full-attention caches
    std::vector<id<MTLBuffer>> kvK, kvV;   // per visit
    id<MTLBuffer> convSt[2];
    int convCur = 0;
    std::vector<int32_t> seq;

    // prompt-cache checkpoints.  A checkpoint restores the recurrent state at a position: the conv
    // state of every visit and (if "full") the sliding-window K/V of the last `window` positions.
    // A light checkpoint relies on the SWA rings still holding that window.
    struct Checkpoint {
        int pos = 0;
        bool full = false;
        bool promptEnd = false;
        std::vector<float> conv;
        std::vector<uint8_t> swa;
        std::vector<float> logits;   // prompt-end checkpoints: the next-token logits
    };
    std::vector<Checkpoint> ckpts;   // sorted by pos
    int hiWater = 0;                 // highest sequence length reached since the last reset
    int lastFullPos = 0;
    void save_checkpoint(bool full, bool promptEnd);
    bool restore_checkpoint(int maxPos, bool needLogits = false);
    bool light_valid(int pos) const { return hiWater - (pos - (hp.window - 1)) <= (int) swaCap; }

    // scratch
    id<MTLBuffer> bTok, bCur, bPrelude, bBankOut, bNB, bMixed, bMsgs, bXin, bXff32, bGateIn, bG64, bGF, bP, bQh, bRc, bT, bO, bAO;
    id<MTLBuffer> bDX, bDMsg, bDMixed, bDGate, bDG, bDXp, bDQ, bDO, bDRoute, bDA, bDPart, bPart[2];
    int partCap = 0;   // chunks per head the partial buffers can hold
    std::vector<id<MTLBuffer>> kvK8, kvV8;   // --kv8: 8-bit copies of the full-attention caches (per visit)
    id<MTLBuffer> bRLog, bIds, bRW, bPairTok, bRowOfPair, bRowScale, bTiles, bXg, bH, bA, bOe, bY, bLogits;
};

// ----------------------------------------------------------------------------------------------------
id<MTLBuffer> EngineImpl::new_buf(size_t bytes, bool touch) {
    bytes = (bytes + 16383) & ~(size_t) 16383;
    id<MTLBuffer> b = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (!b) throw std::runtime_error("engine: failed to allocate a " + std::to_string(bytes >> 20) + " MiB Metal buffer");
    if (rset) {
        std::lock_guard<std::mutex> lk(rsetMu);
        [rset addAllocation:b];
        rsetDirty = true;
    }
    // fault the pages in from the CPU: first-touch faults taken by GPU writes are far slower
    if (touch) {
        char * p = (char *) b.contents;
        const size_t chunk = 16u << 20, nch = (bytes + chunk - 1) / chunk;
        dispatch_apply(nch, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
            memset(p + i * chunk, 0, std::min(chunk, bytes - i * chunk));
        });
    }
    return b;
}

void EngineImpl::rset_commit() {
    if (!rset) return;
    std::lock_guard<std::mutex> lk(rsetMu);
    if (rsetDirty) {
        [rset commit];
        [rset requestResidency];
        rsetDirty = false;
    }
}

void EngineImpl::drop_buf(id<MTLBuffer> b) {
    if (!rset || !b) return;
    std::lock_guard<std::mutex> lk(rsetMu);
    [rset removeAllocation:b];
    rsetDirty = true;
}

EngineImpl::~EngineImpl() {
    if (opt.verbose && statPasses + statSingles)
        fprintf(stderr, "engine: %ld single-token steps, %ld multi-token passes, %ld drafted, %ld accepted, %ld draft-model runs\n", statSingles, statPasses, statDrafted, statAccepted, statDraftCalls);
    hbStop.store(true);
    if (heartbeat.joinable()) heartbeat.join();
    if (rset) [rset endResidency];
}

id<MTLComputePipelineState> EngineImpl::pipe(const char * name) {
    auto it = pipes.find(name);
    if (it != pipes.end()) return it->second;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
    if (!fn) throw std::runtime_error(std::string("engine: missing kernel ") + name);
    NSError * err = nil;
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!ps) throw std::runtime_error(std::string("engine: pipeline ") + name + ": " + err.localizedDescription.UTF8String);
    pipes[name] = ps;
    return ps;
}

id<MTLComputePipelineState> EngineImpl::pipe_fc(const char * name, uint32_t S, uint32_t Q, uint32_t mode, uint32_t nj) {
    char key[96];
    snprintf(key, sizeof key, "%s/%u/%u/%u/%u", name, S, Q, mode, nj);
    auto it = pipes.find(key);
    if (it != pipes.end()) return it->second;
    MTLFunctionConstantValues * cv = [MTLFunctionConstantValues new];
    const uint32_t vals[4] = {S, Q, mode, nj};
    for (int i = 0; i < 4; ++i) [cv setConstantValue:&vals[i] type:MTLDataTypeUInt atIndex:i];
    NSError * err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name] constantValues:cv error:&err];
    if (!fn) throw std::runtime_error(std::string("engine: missing kernel ") + name);
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
    if (!ps) throw std::runtime_error(std::string("engine: pipeline ") + name + ": " + err.localizedDescription.UTF8String);
    pipes[key] = ps;
    return ps;
}

void EngineImpl::begin() {
    if (!cb) {
        rset_commit();
        cb = [queue commandBuffer];
        enc = [cb computeCommandEncoder];
    }
}

void EngineImpl::flush(bool wait) {
    if (!cb) return;
    [enc endEncoding];
    [cb commit];
    if (wait) {
        [cb waitUntilCompleted];
        if (cb.status == MTLCommandBufferStatusError) {
            std::string msg = cb.error ? cb.error.localizedDescription.UTF8String : "unknown";
            cb = nil; enc = nil;
            throw std::runtime_error("engine: GPU command buffer failed: " + msg);
        }
    }
    cb = nil;
    enc = nil;
}

template <typename T>
void EngineImpl::enc_setup(const char * name, const T & args, std::initializer_list<B> bufs, id<MTLComputePipelineState> * out) {
    begin();
    id<MTLComputePipelineState> ps = fcOn ? pipe_fc(name, fc[0], fc[1], fc[2], fc[3]) : pipe(name);
    fcOn = false;
    [enc setComputePipelineState:ps];
    [enc setBytes:&args length:sizeof(T) atIndex:0];
    int i = 1;
    for (const B & b : bufs) [enc setBuffer:b.buf offset:b.off atIndex:i++];
    *out = ps;
}

template <typename T>
void EngineImpl::groups(const char * name, const T & args, std::initializer_list<B> bufs, size_t gx, size_t gy, size_t nthreads) {
    if (gx == 0 || gy == 0) return;
    id<MTLComputePipelineState> ps;
    enc_setup(name, args, bufs, &ps);
    [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1) threadsPerThreadgroup:MTLSizeMake(nthreads, 1, 1)];
}

template <typename T>
void EngineImpl::threads(const char * name, const T & args, std::initializer_list<B> bufs, size_t tx, size_t ty) {
    if (tx == 0 || ty == 0) return;
    id<MTLComputePipelineState> ps;
    enc_setup(name, args, bufs, &ps);
    const size_t w = ps.threadExecutionWidth;
    const size_t h = std::max<size_t>(1, ps.maxTotalThreadsPerThreadgroup / w);
    [enc dispatchThreads:MTLSizeMake(tx, ty, 1) threadsPerThreadgroup:MTLSizeMake(std::min(w, tx), std::min(h, ty), 1)];
}

void EngineImpl::blit(id<MTLBuffer> src, size_t so, id<MTLBuffer> dst, size_t d_o, size_t n) {
    // keep ordering with compute work: end the compute encoder, blit, resume
    begin();
    [enc endEncoding];
    id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
    [bl copyFromBuffer:src sourceOffset:so toBuffer:dst destinationOffset:d_o size:n];
    [bl endEncoding];
    enc = [cb computeCommandEncoder];
}

void EngineImpl::gemm(const char * name, B bB, B bA, B bC, uint32_t N, uint32_t M, uint32_t K, uint32_t ldb, uint32_t ldc) {
    GemmArgs a{N, M, K, ldb, ldc, 0, 0, 0};
    groups(name, a, {bB, bA, bC}, (M + 63) / 64, (N + 63) / 64, 128);
}

// ----------------------------------------------------------------------------------------------------
void EngineImpl::load_hparams(const GGUF & g) {
    const std::string arch = g.str("general.architecture");
    if (arch != "inkling2" && arch != "inkling2_loop")
        throw std::runtime_error("engine: this server only runs Inkling 2 Mini (GGUF architecture inkling2 / inkling2_loop); the file is '" + arch + "'");
    auto key = [&](const char * k) { return arch + "." + k; };
    hp.D = (int) g.u64(key("embedding_length"));
    hp.NH = (int) g.int_array(key("attention.head_count"))[0];
    hp.NKV = (int) g.int_array(key("attention.head_count_kv"))[0];
    hp.DH = (int) g.u64(key("attention.key_length"));
    hp.NREL = (int) g.u64(key("attention.relative_length"));
    hp.NEXP = (int) g.u64(key("expert_count"));
    hp.NUSED = (int) g.u64(key("expert_used_count"));
    hp.FF = (int) g.u64(key("expert_feed_forward_length"));
    hp.NCONV = (int) g.u64(key("shortconv.l_cache"));
    hp.blockSize = (int) g.u64(key("attn_res.block_size"));
    hp.window = (int) g.u64(key("attention.sliding_window"));
    hp.extent = (int) g.u64(key("attention.relative_extent"));
    hp.vocab = (int) g.u64(key("vocab_size_unpadded"));
    hp.vocabPadded = (int) g.u64(key("vocab_size"));
    hp.eps = (float) g.f64(key("attention.layer_norm_rms_epsilon"));
    hp.qkClip = (float) g.f64(key("attention.qk_clip"));
    hp.relClip = (float) g.f64(key("attention.relative_clip"));
    hp.logFloor = (float) g.f64(key("attention.log_scale_floor"));
    hp.logAlpha = (float) g.f64(key("attention.log_scale_alpha"));
    hp.wScale = (float) g.f64(key("expert_weights_scale"), 1.0);
    hp.situBeta = (float) g.f64(key("activation.situ_beta"));
    hp.situLBeta = (float) g.f64(key("activation.situ_linear_beta"));
    hp.softcap = (float) g.f64(key("final_logit_softcapping"), 0.0);
    if (arch == "inkling2") {
        hp.loopStart = (int) g.u64(key("loop.core_start"));
        hp.loopEnd = (int) g.u64(key("loop.core_end"));
        hp.loopCount = (int) g.u64(key("num_loops"));
    } else {
        hp.loopStart = (int) g.u64(key("loop.start"));
        hp.loopEnd = (int) g.u64(key("loop.end"));
        hp.loopCount = (int) g.u64(key("loop.count"));
    }
    hp.gateRank = (int) g.u64(key("loop.gate_rank"));
    // physical layer count: count the blocks present in the file
    int n = 0;
    while (g.find("blk." + std::to_string(n) + ".attn_q.weight")) ++n;
    hp.nLayers = n;
    auto pat = g.int_array(key("attention.sliding_window_pattern"));
    hp.swa.resize(n);
    for (int i = 0; i < n; ++i) hp.swa[i] = pat[i] != 0;

    if (hp.D != 2304 || hp.NH != 24 || hp.NKV != 8 || hp.DH != 128 || hp.NREL != 16 || hp.NEXP != 128 || hp.NUSED != 8 ||
        hp.FF != 1152 || hp.NCONV != 4 || hp.window != hp.extent || hp.NH % hp.NKV || n != 24) {
        throw std::runtime_error("engine: this server only runs Inkling 2 Mini; the file is an Inkling model with other dimensions");
    }
    if (hp.loopStart % hp.blockSize || hp.loopEnd % hp.blockSize || hp.loopStart <= 0 || hp.loopEnd > n || hp.loopCount < 1) {
        throw std::runtime_error("engine: unsupported loop configuration");
    }

    for (int i = 0; i < hp.loopStart; ++i) visits.push_back({i, 0});
    for (int p = 0; p < hp.loopCount; ++p)
        for (int i = hp.loopStart; i < hp.loopEnd; ++i) visits.push_back({i, p});
    for (int i = hp.loopEnd; i < n; ++i) visits.push_back({i, 0});

    const uint32_t DQ = hp.NH * hp.DH, DKV = hp.NKV * hp.DH;
    qCol = 0;
    gateCol = DQ;
    rCol = 2 * DQ;
    sinkCol = rCol + hp.NH * hp.NREL;
    kCol = sinkCol + hp.NH;
    kDynCol = kCol + DKV;
    vCol = kDynCol + hp.NKV * hp.NCONV;
    vDynCol = vCol + DKV;
    PW = vDynCol + hp.NKV * hp.NCONV;
    CH = 2 * DKV + 2 * hp.D;
}

void EngineImpl::build_library() {
    NSString * src = [[NSString alloc] initWithBytes:kernels_metal_src length:kernels_metal_src_len encoding:NSUTF8StringEncoding];
    MTLCompileOptions * o = [MTLCompileOptions new];
    o.languageVersion = MTLLanguageVersion4_0;
    o.preprocessorMacros = @{
        @"D": @(hp.D), @"NH": @(hp.NH), @"NKV": @(hp.NKV), @"DH": @(hp.DH), @"NREL": @(hp.NREL), @"NEXP": @(hp.NEXP),
        @"NUSED": @(hp.NUSED), @"FF": @(hp.FF), @"NCONV": @(hp.NCONV),
        @"AC": @(attnC), @"AQ": @(attnQ),
    };
    NSError * err = nil;
    lib = [dev newLibraryWithSource:src options:o error:&err];
    if (!lib) throw std::runtime_error(std::string("engine: GPU kernel compilation failed. This server needs Metal 4 (Apple silicon, macOS 26 or later). Compiler output: ") + err.localizedDescription.UTF8String);
}

// ----------------------------------------------------------------------------------------------------
void EngineImpl::load_weights(const GGUF & g) {
    const double t0 = now_s();
    const size_t D = hp.D, DQ = hp.NH * hp.DH, DKV = hp.NKV * hp.DH, FF = hp.FF, NE = hp.NEXP;

    // parallel copy from the mmap'ed file
    auto pcopy = [](void * dst, const void * src, size_t n) {
        const size_t chunk = 32u << 20;
        const size_t nch = (n + chunk - 1) / chunk;
        dispatch_apply(nch, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
            const size_t o = i * chunk;
            memcpy((char *) dst + o, (const char *) src + o, std::min(chunk, n - o));
        });
    };
    // convert a BF16 region to FP16 in place on the GPU
    auto bf16_fix = [&](id<MTLBuffer> buf, size_t offBytes, size_t elems) {
        const uint32_t w = 16384;
        CountArgs a{w, (uint32_t) ((elems + w - 1) / w), 0, 0};
        // the grid may overshoot the tensor by < w elements: buffers are padded to cover that
        threads("k_bf16_to_f16", a, {B(buf, offBytes)}, w, a.h);
    };
    // small tensor of any supported type -> f32 values
    auto read_f32 = [&](const GGTensor & t, float * dst) {
        const size_t n = t.nelements();
        if (!dequant_supported(t.type) || n % dequant_block(t.type))
            throw std::runtime_error("engine: unsupported type " + std::to_string(t.type) + " for tensor " + t.name);
        dequant_range(t.type, t.data, 0, n, dst);
    };

    // FP16 rows -> 8-bit rows of [K/32 half scales][K int8] (GGUF Q8_0 blocks, de-interleaved)
    auto make_q8 = [&](id<MTLBuffer> src, size_t rows, size_t K) {
        const size_t rowBytes = K + K / 16;
        id<MTLBuffer> out = new_buf(rows * rowBytes + 65536);
        NvArgs a{(uint32_t) rows, (uint32_t) K, (uint32_t) rowBytes, 0};
        threads("k_f16_to_q8", a, {B(src), B(out)}, K / 32, rows);
        return out;
    };
    // Quantized variants of the file store the 16-bit tensors (and, in some, all experts) in GGUF block
    // formats.  Any supported tensor type -> FP16 values, on the CPU.
    auto to_f16 = [&](const GGTensor & t, void * dstv) {
        __fp16 * dst = (__fp16 *) dstv;
        const size_t n = t.nelements();
        const uint8_t * src = t.data;
        const uint32_t type = t.type;
        if (!dequant_supported(type) || n % dequant_block(type))
            throw std::runtime_error("engine: unsupported weight type " + std::to_string(type) + " for tensor " + t.name);
        const size_t chunk = 1u << 18;   // a multiple of every block size
        dispatch_apply((n + chunk - 1) / chunk, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t ci) {
            const size_t i0 = ci * chunk, i1 = std::min(n, i0 + chunk);
            if (type == GG_F16) { memcpy(dst + i0, src + i0 * 2, (i1 - i0) * 2); return; }
            std::vector<float> tmp(i1 - i0);
            dequant_range(type, src, i0, i1, tmp.data());
            for (size_t i = i0; i < i1; ++i) dst[i] = (__fp16) tmp[i - i0];
        });
    };
    // A tensor into an FP16 buffer region: BF16 stays on the fast GPU path when `gpuBF16` (the in-place
    // conversion kernel may run past the region, so it is only used when nothing else follows it).
    auto load_f16 = [&](const GGTensor & t, id<MTLBuffer> buf, size_t offBytes, bool gpuBF16) {
        if (t.type == GG_BF16 && gpuBF16) {
            pcopy((char *) buf.contents + offBytes, t.data, t.nbytes);
            bf16_fix(buf, offBytes, t.nelements());
        } else {
            to_f16(t, (char *) buf.contents + offBytes);
        }
    };
    // Q8_0 tensor rows -> our 8-bit rows, verbatim.  Blocks never straddle rows (K is a multiple of 32).
    auto q8_exact = [&](const GGTensor & t, uint8_t * dst, size_t rows, size_t K) {
        const size_t nb = K / 32, rowB = K + K / 16;
        const uint8_t * src = t.data;
        const size_t per = 256;
        dispatch_apply((rows + per - 1) / per, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t ci) {
            for (size_t r = ci * per; r < std::min(rows, (ci + 1) * per); ++r) {
                const uint8_t * s = src + r * nb * 34;
                uint8_t * d = dst + r * rowB;
                for (size_t u = 0; u < nb; ++u) {
                    memcpy(d + u * 2, s + u * 34, 2);
                    memcpy(d + nb * 2 + u * 32, s + u * 34 + 2, 32);
                }
            }
        });
    };
    // 8-bit copy of an FP16 buffer holding one tensor: verbatim for Q8_0 sources
    auto q8_of = [&](const GGTensor & t, id<MTLBuffer> f16, size_t rows, size_t K) {
        if (t.type != GG_Q8_0) return make_q8(f16, rows, K);
        id<MTLBuffer> out = new_buf(rows * (K + K / 16) + 65536);
        q8_exact(t, (uint8_t *) out.contents, rows, K);
        return out;
    };

    // ue4m3 scale table
    bScaleTab = new_buf(256 * 4);
    for (int i = 0; i < 256; ++i) ((float *) bScaleTab.contents)[i] = ue4m3(i);

    // embeddings / output head
    {
        const GGTensor & te = g.get("token_embd.weight");
        const GGTensor & to = g.get("output.weight");
        if (te.nelements() != (size_t) hp.vocabPadded * D || to.nelements() != (size_t) hp.vocabPadded * D)
            throw std::runtime_error("engine: unexpected embedding shape");
        bEmb = new_buf(te.nelements() * 2 + 65536);
        bOut = new_buf(to.nelements() * 2 + 65536);
        load_f16(te, bEmb, 0, true);
        load_f16(to, bOut, 0, true);
        if (opt.q8) bOutQ8 = q8_of(to, bOut, hp.vocabPadded, D);
        bGlobalSmall = new_buf(3 * D * 4);
        float * s = (float *) bGlobalSmall.contents;
        oTokNorm = 0; oOutNorm = D; oOutResQ = 2 * D;
        read_f32(g.get("token_embd_norm.weight"), s + oTokNorm);
        read_f32(g.get("output_norm.weight"), s + oOutNorm);
        read_f32(g.get("output_res_query.weight"), s + oOutResQ);
    }

    const bool udNames = g.find("loop_res_query.weight") != nullptr;
    std::vector<float> loopQ, loopKVQ;
    if (udNames) {
        loopQ.resize(g.get("loop_res_query.weight").nelements());
        loopKVQ.resize(g.get("loop_kv_res_query.weight").nelements());
        read_f32(g.get("loop_res_query.weight"), loopQ.data());
        read_f32(g.get("loop_kv_res_query.weight"), loopKVQ.data());
    }

    layers.resize(hp.nLayers);
    for (int il = 0; il < hp.nLayers; ++il) {
        LayerW & L = layers[il];
        const std::string p = "blk." + std::to_string(il) + ".";
        auto T = [&](const char * n) -> const GGTensor & { return g.get(p + n); };
        L.swa = hp.swa[il];
        L.loop = il >= hp.loopStart && il < hp.loopEnd && hp.loopCount > 1;

        // --- attention projections, packed in P-column order
        L.attn = new_buf((size_t) PW * D * 2 + 65536);
        memset(L.attn.contents, 0, L.attn.length);
        struct Part { const GGTensor * t; uint32_t row; size_t rows; };
        std::vector<Part> parts;
        bool allBF16 = true;
        auto put = [&](const char * name, uint32_t row, size_t rows) {
            const GGTensor & t = T(name);
            if (t.nelements() != rows * D) throw std::runtime_error("engine: unexpected shape for tensor " + t.name);
            parts.push_back({&t, row, rows});
            allBF16 = allBF16 && t.type == GG_BF16;
        };
        put("attn_q.weight", qCol, DQ);
        put("attn_gate.weight", gateCol, DQ);
        put("attn_rel.weight", rCol, hp.NH * hp.NREL);
        put("attn_sink_proj.weight", sinkCol, hp.NH);
        put("attn_k.weight", kCol, DKV);
        put("attn_v.weight", vCol, DKV);
        if (!L.swa) {
            put("attn_k_conv_proj.weight", kDynCol, hp.NKV * hp.NCONV);
            put("attn_v_conv_proj.weight", vDynCol, hp.NKV * hp.NCONV);
        }
        if (allBF16) {
            for (const Part & pt : parts) pcopy((char *) L.attn.contents + (size_t) pt.row * D * 2, pt.t->data, pt.t->nbytes);
            bf16_fix(L.attn, 0, (size_t) PW * D);
        } else {
            for (const Part & pt : parts) to_f16(*pt.t, (char *) L.attn.contents + (size_t) pt.row * D * 2);
        }

        const GGTensor & two = T("attn_output.weight");
        if (two.nelements() != D * DQ) throw std::runtime_error("engine: unexpected shape for tensor " + two.name);
        L.wo = new_buf(two.nelements() * 2 + 65536);
        load_f16(two, L.wo, 0, true);
        if (opt.q8) {
            L.attnQ8 = make_q8(L.attn, PW, D);
            L.woQ8 = q8_of(two, L.wo, D, DQ);
            if (!allBF16) {
                // the rows that are Q8_0 in the file are used verbatim
                flush(true);
                for (const Part & pt : parts)
                    if (pt.t->type == GG_Q8_0) q8_exact(*pt.t, (uint8_t *) L.attnQ8.contents + (size_t) pt.row * (D + D / 16), pt.rows, D);
            }
        }

        // --- small f32 parameters
        std::vector<float> sm;
        auto add = [&](const GGTensor & t, size_t elems) {
            if (t.nelements() != elems) throw std::runtime_error("engine: unexpected shape for " + t.name);
            const size_t o = sm.size();
            sm.resize(o + elems);
            read_f32(t, sm.data() + o);
            return o;
        };
        if (L.swa) {
            L.oNormQ = L.oNormK = L.oNormV = add(T("attn_norm.weight"), D);
        } else {
            L.oNormQ = add(T("attn_q_inp_norm.weight"), D);
            L.oNormK = add(T("attn_k_inp_norm.weight"), D);
            L.oNormV = add(T("attn_v_inp_norm.weight"), D);
        }
        L.oQClip = add(T("attn_q_clip.weight"), DQ);
        L.oRClip = add(T("attn_rel_clip.weight"), hp.NH * hp.NREL);
        L.oRelBias = add(T("attn_rel_bias.weight"), (size_t) hp.NREL * hp.extent);
        L.oKConv = add(T("attn_k_conv.weight"), DKV * hp.NCONV);
        L.oVConv = add(T("attn_v_conv.weight"), DKV * hp.NCONV);
        L.oAConv = add(T("attn_post_conv.weight"), D * hp.NCONV);
        L.oFConv = add(T("ffn_post_conv.weight"), D * hp.NCONV);
        if (il > 0) {
            L.hasAttnResQ = true;
            L.oAttnResQ = add(T("attn_res_query.weight"), L.swa ? D : 3 * D);
        }
        L.oFfnResQ = add(T("ffn_res_query.weight"), D);
        L.oFfnNorm = add(T("ffn_norm.weight"), D);
        L.oGateInp = add(T("ffn_gate_inp.weight"), NE * D);
        {
            const GGTensor * t = g.find(p + "exp_probs_b.bias");
            if (t) L.oProbsB = add(*t, NE);
            else { L.oProbsB = sm.size(); sm.resize(sm.size() + NE, 0.0f); }
        }
        L.scaleGU.assign(NE, 1.0f);
        L.scaleDN.assign(NE, 1.0f);
        if (const GGTensor * t = g.find(p + "ffn_gate_up_exps.scale")) read_f32(*t, L.scaleGU.data());
        if (const GGTensor * t = g.find(p + "ffn_down_exps.scale")) read_f32(*t, L.scaleDN.data());
        L.oScaleGU = sm.size(); sm.insert(sm.end(), L.scaleGU.begin(), L.scaleGU.end());
        L.oScaleDN = sm.size(); sm.insert(sm.end(), L.scaleDN.begin(), L.scaleDN.end());

        // --- loop controllers
        if (L.loop) {
            const size_t R = hp.gateRank;
            L.oMsgQ = sm.size();
            sm.resize(sm.size() + 4 * D);
            float * mq = sm.data() + L.oMsgQ;
            if (udNames) {
                const int i = il - hp.loopStart;
                memcpy(mq + 0 * D, loopQ.data() + (size_t) (2 * i) * D, D * 4);        // q
                memcpy(mq + 1 * D, loopKVQ.data() + (size_t) (2 * i) * D, D * 4);      // k
                memcpy(mq + 2 * D, loopKVQ.data() + (size_t) (2 * i + 1) * D, D * 4);  // v
                memcpy(mq + 3 * D, loopQ.data() + (size_t) (2 * i + 1) * D, D * 4);    // ffn
            } else {
                read_f32(T("loop_msg_query.weight"), mq);
            }
            static const char * ud[4] = {"loop_attn", "loop_k", "loop_v", "loop_mlp"};
            static const char * up[4] = {"loop_q_ctrl", "loop_k_ctrl", "loop_v_ctrl", "loop_ffn_ctrl"};
            const size_t per = R * 2 * D + 2 * D * R;   // elements per controller
            L.loopW = new_buf(4 * per * 2);
            std::vector<float> tmp(2 * D * R);
            for (int c = 0; c < 4; ++c) {
                const std::string base = p + (udNames ? ud[c] : up[c]);
                _Float16 * dst = (_Float16 *) L.loopW.contents + c * per;
                const GGTensor & td = g.get(base + "_down.weight");
                const GGTensor & tg = g.get(base + "_gain.weight");
                const GGTensor & tf = g.get(base + "_forget.weight");
                if (td.nelements() != R * 2 * D || tg.nelements() != D * R || tf.nelements() != D * R)
                    throw std::runtime_error("engine: unexpected loop tensor shape in " + base);
                read_f32(td, tmp.data());
                for (size_t i = 0; i < R * 2 * D; ++i) dst[i] = (_Float16) tmp[i];
                read_f32(tg, tmp.data());
                for (size_t i = 0; i < D * R; ++i) dst[R * 2 * D + i] = (_Float16) tmp[i];
                read_f32(tf, tmp.data());
                for (size_t i = 0; i < D * R; ++i) dst[R * 2 * D + D * R + i] = (_Float16) tmp[i];
            }
        }
        L.small = new_buf(sm.size() * 4);
        memcpy(L.small.contents, sm.data(), sm.size() * 4);

        // --- experts
        const GGTensor & tgu = T("ffn_gate_up_exps.weight");
        const GGTensor & tdn = T("ffn_down_exps.weight");
        if (tgu.nelements() != NE * 2 * FF * D || tdn.nelements() != NE * D * FF)
            throw std::runtime_error("engine: unexpected expert tensor shape in layer " + std::to_string(il));
        // Q4_K and Q5_0 are read in place by the decode kernels: rows of 32-weight units, 16 nibble bytes
        // each (byte j = weights j and 16 + j), see kernels.metal.  Blocks never straddle rows.
        auto pack_native = [&](const GGTensor & t, size_t K, size_t rows) -> id<MTLBuffer> {
            const uint32_t type = t.type;
            const size_t nu = K / 32, ng = K / 8, nsb = K / 256, hdr = (nsb * 2 + 3) & ~(size_t) 3;
            const size_t rowB = native_row_bytes(type, K);
            if (!rowB) return nil;
            // 256-weight formats need whole super-blocks per row (and 8 units per super-block alignment)
            if (type == GG_Q2_0 ? K % 64 != 0 : (type == GG_Q5_0 ? K % 32 != 0 : K % 256 != 0)) return nil;
            id<MTLBuffer> out = new_buf(rows * rowB + 65536);
            uint8_t * o = (uint8_t *) out.contents;
            const uint8_t * src = t.data;
            const size_t per = 256;
            dispatch_apply((rows + per - 1) / per, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t ci) {
                for (size_t r = ci * per; r < std::min(rows, (ci + 1) * per); ++r) {
                    uint8_t * d = o + r * rowB;
                    if (type == GG_Q5_0) {
                        const uint8_t * s = src + r * nu * 22;
                        for (size_t u = 0; u < nu; ++u) {
                            memcpy(d + u * 2, s + u * 22, 2);
                            memcpy(d + nu * 2 + u * 4, s + u * 22 + 2, 4);
                            memcpy(d + nu * 6 + u * 16, s + u * 22 + 6, 16);
                        }
                    } else if (type == GG_Q2_0) {
                        const size_t nb = K / 64, h2 = (nb * 2 + 3) & ~(size_t) 3;
                        const uint8_t * s = src + r * nb * 18;
                        for (size_t b = 0; b < nb; ++b) {
                            memcpy(d + b * 2, s + b * 18, 2);
                            memcpy(d + h2 + b * 16, s + b * 18 + 2, 16);
                        }
                    } else if (type == GG_Q4_K) {
                        const uint8_t * s = src + r * nsb * 144;
                        uint8_t * sm = d + nsb * 4, * nib = d + nsb * 4 + nu * 2;
                        for (size_t b = 0; b < nsb; ++b) {
                            const uint8_t * blk = s + b * 144;
                            memcpy(d + b * 4, blk, 4);   // d, dmin
                            for (int j = 0; j < 8; ++j) k4_scale_min(j, blk + 4, &sm[(b * 8 + j) * 2], &sm[(b * 8 + j) * 2 + 1]);
                            // source: byte l of pair p holds weight l of unit 2p (low) and of unit 2p + 1 (high)
                            for (int pr = 0; pr < 4; ++pr) {
                                const uint8_t * q = blk + 16 + pr * 32;
                                uint8_t * u0 = nib + (b * 8 + 2 * pr) * 16, * u1 = u0 + 16;
                                for (int j = 0; j < 16; ++j) {
                                    u0[j] = (uint8_t) ((q[j] & 0xF) | ((q[16 + j] & 0xF) << 4));
                                    u1[j] = (uint8_t) ((q[j] >> 4) | ((q[16 + j] >> 4) << 4));
                                }
                            }
                        }
                    } else {
                        // codebook formats: split the fields of every super-block into per-row arrays
                        const size_t bs = type == GG_IQ2_XXS ? 66 : type == GG_IQ2_XS ? 74 : type == GG_IQ2_S ? 82 : 50;
                        const uint8_t * s = src + r * nsb * bs;
                        uint8_t * body = d + hdr;
                        for (size_t b = 0; b < nsb; ++b) {
                            const uint8_t * blk = s + b * bs;
                            memcpy(d + b * 2, blk, 2);
                            if (type == GG_IQ2_XXS) {
                                memcpy(body + b * 64, blk + 2, 64);
                            } else if (type == GG_IQ2_XS) {
                                memcpy(body + b * 8, blk + 66, 8);                 // scales
                                memcpy(body + nu + b * 64, blk + 2, 64);            // 32 x ushort
                            } else if (type == GG_IQ2_S) {
                                memcpy(body + b * 8, blk + 74, 8);                 // scales
                                memcpy(body + nu + b * 8, blk + 66, 8);            // high index bits
                                memcpy(body + 2 * nu + b * 32, blk + 2, 32);       // codebook bytes
                                memcpy(body + 2 * nu + ng + b * 32, blk + 34, 32); // sign bytes
                            } else {
                                memcpy(body + b * 16, blk + 34, 16);               // 8 x ushort
                                memcpy(body + nu * 2 + b * 32, blk + 2, 32);       // codebook bytes
                            }
                        }
                    }
                }
            });
            return out;
        };
        // NVFP4 rows: 36-byte blocks {4 scales, 32 nibble bytes} -> [all scales of the row][all nibbles of the row]
        auto pack_nv = [&](const GGTensor & t, size_t K, size_t rows) {
            const size_t nblk = K / 64, rowBytes = K / 16 + K / 2;
            id<MTLBuffer> out = new_buf(rows * rowBytes + 65536);
            uint8_t * o = (uint8_t *) out.contents;
            const uint8_t * sp = t.data;
            dispatch_apply(rows / 256, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t ch) {
                for (size_t r = ch * 256; r < (ch + 1) * 256; ++r) {
                    const uint8_t * sr = sp + r * nblk * 36;
                    uint8_t * dr = o + r * rowBytes;
                    for (size_t bb = 0; bb < nblk; ++bb) {
                        memcpy(dr + bb * 4, sr + bb * 36, 4);
                        memcpy(dr + K / 16 + bb * 32, sr + bb * 36 + 4, 32);
                    }
                }
            });
            return out;
        };
        // Each expert tensor on its own: the decode path reads NVFP4, Q4_K and Q5_0 in place and uses 8-bit
        // copies for every other format; prompt processing always gets the FP16 expansion H.
        auto load_expert = [&](const GGTensor & t, size_t K, size_t rows, id<MTLBuffer> __strong & H, id<MTLBuffer> __strong & NV,
                               id<MTLBuffer> __strong & P, uint32_t & fmt, id<MTLBuffer> __strong & Q8) {
            // INK_NO_NATIVE=types (comma separated GGUF type ids, or 'all'): debugging aid, use the 8-bit copies instead
            static const char * noNat = getenv("INK_NO_NATIVE");
            const bool skip = noNat && (strstr(noNat, "all") || strstr((std::string(",") + noNat + ",").c_str(), ("," + std::to_string(t.type) + ",").c_str()));
            if (t.type == GG_NVFP4) {
                NV = pack_nv(t, K, rows);
            } else if (!skip && native_kernel(K == D, t.type)) {
                P = pack_native(t, K, rows);
                if (P) fmt = t.type;
            }
            // low-memory mode: a packed form that holds the file's values exactly is expanded per layer at
            // prompt time, so no FP16 copy is kept (Q8_0 data is such a form too)
            if (opt.low_mem && !NV && !P && t.type == GG_Q8_0) Q8 = q8_of(t, nil, rows, K);
            if (opt.low_mem && (NV || P || Q8)) return;
            H = new_buf(t.nelements() * 2 + 65536);
            if (NV) {
                NvArgs na{(uint32_t) rows, (uint32_t) K, (uint32_t) (K / 16 + K / 2), 0};
                threads("k_nvfp4_to_f16", na, {B(NV), B(bScaleTab), B(H)}, K / 16, na.rows);
                return;
            }
            load_f16(t, H, 0, true);
            if (!P && opt.q8) Q8 = q8_of(t, H, rows, K);
        };
        load_expert(tgu, D, NE * 2 * FF, L.guH, L.guNV, L.guP, L.guFmt, L.guQ8);
        load_expert(tdn, FF, NE * D, L.dnH, L.dnNV, L.dnP, L.dnFmt, L.dnQ8);
        L.nv = L.guNV && L.dnNV;
        flush(true);
        if (opt.verbose) fprintf(stderr, "engine: loaded layer %d (%s, %s experts) %.1fs\n", il, L.swa ? "swa" : "full",
                                 L.nv ? "nvfp4" : "f16", now_s() - t0);
    }
    flush(true);
    if (opt.verbose) fprintf(stderr, "engine: weights ready in %.1fs\n", now_s() - t0);
}

// ----------------------------------------------------------------------------------------------------
void EngineImpl::alloc_state() {
    if (opt.low_mem) {
        bXgu = new_buf((size_t) hp.NEXP * 2 * hp.FF * hp.D * 2 + 65536, true);
        bXdn = new_buf((size_t) hp.NEXP * hp.D * hp.FF * 2 + 65536, true);
    }
    const size_t D = hp.D, DQ = hp.NH * hp.DH, DKV = hp.NKV * hp.DH, FF = hp.FF;
    NB_ = std::max(1, opt.batch);
    const size_t N = NB_;
    swaCap = 8192;
    while (swaCap < (uint32_t) (hp.window + NB_)) swaCap <<= 1;

    kvK.resize(visits.size());
    kvV.resize(visits.size());
    kvCap = 0;
    for (size_t v = 0; v < visits.size(); ++v) {
        if (layers[visits[v].layer].swa) {
            kvK[v] = new_buf((size_t) swaCap * DKV * 2, true);
            kvV[v] = new_buf((size_t) swaCap * DKV * 2, true);
        }
    }
    convSt[0] = new_buf(visits.size() * (hp.NCONV - 1) * CH * 4, true);
    convSt[1] = new_buf(visits.size() * (hp.NCONV - 1) * CH * 4, true);

    bTok = new_buf(N * 4, true);
    bCur = new_buf(N * D * 4, true);
    bPrelude = new_buf(N * D * 4, true);
    bBankOut = new_buf(N * D * 4, true);
    bNB = new_buf(5 * N * D * 4, true);
    bMixed = new_buf(3 * N * D * 4, true);
    bMsgs = new_buf(4 * N * D * 4, true);
    bXin = new_buf(3 * N * D * 2, true);
    bXff32 = new_buf(N * D * 4, true);
    bGateIn = new_buf(N * 2 * D * 2, true);
    bG64 = new_buf(N * hp.gateRank * 4, true);
    bGF = new_buf(N * 2 * D * 4, true);
    bP = new_buf(N * PW * 4, true);
    bQh = new_buf(N * DQ * 2, true);
    bRc = new_buf(N * hp.NH * hp.NREL * 4, true);
    bT = new_buf(N * hp.NH * hp.extent * 4, true);
    bO = new_buf(N * DQ * 2, true);
    bAO = new_buf(N * D * 4, true);
    bRLog = new_buf(N * hp.NEXP * 4, true);
    bIds = new_buf(N * hp.NUSED * 4, true);
    bRW = new_buf(N * hp.NUSED * 4, true);
    const size_t R = N * hp.NUSED;
    bPairTok = new_buf(R * 4, true);
    bRowOfPair = new_buf(R * 4, true);
    bRowScale = new_buf(R * 4, true);
    bTiles = new_buf((R / 64 + hp.NEXP + 1) * 3 * 4, true);
    bXg = new_buf(R * D * 2, true);
    bH = new_buf(R * 2 * FF * 4, true);
    bA = new_buf(R * FF * 2, true);
    bOe = new_buf(R * D * 4, true);
    bY = new_buf(N * D * 4, true);
    bLogits = new_buf((size_t) hp.vocabPadded * 4, true);
    if (opt.draft > 0) {
        if (dsOn) ds_alloc();
        bLogitsM = new_buf((size_t) MINI_MAX * hp.vocabPadded * 4, true);
        bConvVar = new_buf((size_t) MINI_MAX * visits.size() * (hp.NCONV - 1) * CH * 4, true);
        bMRoute = new_buf((size_t) MINI_MAX * 3 * hp.NUSED * 4, true);
        bMA = new_buf((size_t) MINI_MAX * hp.NUSED * FF * 4, true);
        bMPart = new_buf((size_t) MINI_MAX * hp.NUSED * D * 4, true);
        bMGroup = new_buf((size_t) MINI_MAX * hp.NUSED * 12 * 4, true);
    }
    bDX = new_buf(3 * D * 4, true);
    bDMsg = new_buf(4 * D * 4, true);
    bDMixed = new_buf(3 * D * 4, true);
    bDGate = new_buf(3 * 2 * D * 4, true);
    bDG = new_buf(3 * hp.gateRank * 4, true);
    bDXp = new_buf(3 * D * 4, true);
    bDQ = new_buf(DQ * 4, true);
    bDO = new_buf(DQ * 4, true);
    bDRoute = new_buf(3 * hp.NUSED * 4, true);
    bDA = new_buf(hp.NUSED * FF * 4, true);
    bDPart = new_buf(hp.NUSED * D * 4, true);
    reset();
}

void EngineImpl::save_checkpoint(bool full, bool promptEnd) {
    const int pos = (int) seq.size();
    if (pos == 0) return;
    for (size_t i = 0; i < ckpts.size();) {
        if (ckpts[i].pos == pos) ckpts.erase(ckpts.begin() + i);
        else ++i;
    }
    Checkpoint c;
    c.pos = pos;
    c.full = full;
    c.promptEnd = promptEnd;
    if (promptEnd) c.logits.assign((const float *) bLogits.contents, (const float *) bLogits.contents + hp.vocab);
    const size_t nconv = visits.size() * (hp.NCONV - 1) * CH;
    c.conv.resize(nconv);
    memcpy(c.conv.data(), convSt[convCur].contents, nconv * 4);
    if (full) {
        const size_t DKV = hp.NKV * hp.DH, rowB = DKV * 2;
        const int w = std::min(pos, hp.window);
        size_t nswa = 0;
        for (auto & vv : visits) nswa += layers[vv.layer].swa ? 1 : 0;
        c.swa.resize(nswa * 2 * (size_t) w * rowB);
        uint8_t * dst = c.swa.data();
        for (size_t v = 0; v < visits.size(); ++v) {
            if (!layers[visits[v].layer].swa) continue;
            for (id<MTLBuffer> b : {kvK[v], kvV[v]}) {
                const uint8_t * src = (const uint8_t *) b.contents;
                for (int q = pos - w; q < pos; ++q, dst += rowB) memcpy(dst, src + (size_t) (q & (swaCap - 1)) * rowB, rowB);
            }
        }
        lastFullPos = pos;
    }
    ckpts.push_back(std::move(c));
    std::sort(ckpts.begin(), ckpts.end(), [](const Checkpoint & a, const Checkpoint & b) { return a.pos < b.pos; });
    // bound memory: at most 64 light and 12 full checkpoints; drop the oldest positions first, but keep
    // the newest prompt-end checkpoints
    auto count = [&](bool f) { int n = 0; for (auto & k : ckpts) n += k.full == f; return n; };
    while (count(false) > 64) {
        for (size_t i = 0; i < ckpts.size(); ++i) if (!ckpts[i].full) { ckpts.erase(ckpts.begin() + i); break; }
    }
    while (count(true) > 12) {
        for (size_t i = 0; i < ckpts.size(); ++i) if (ckpts[i].full) { ckpts.erase(ckpts.begin() + i); break; }
    }
}

// restore the newest usable checkpoint at or below maxPos; false if there is none
bool EngineImpl::restore_checkpoint(int maxPos, bool needLogits) {
    int best = -1;
    for (int i = (int) ckpts.size() - 1; i >= 0; --i) {
        if (ckpts[i].pos > maxPos) continue;
        if (needLogits && (ckpts[i].pos != maxPos || ckpts[i].logits.empty())) break;
        if (ckpts[i].full || light_valid(ckpts[i].pos)) { best = i; break; }
    }
    if (best < 0) return false;
    const Checkpoint & c = ckpts[best];
    if (needLogits) memcpy(bLogits.contents, c.logits.data(), c.logits.size() * 4);
    memcpy(convSt[convCur].contents, c.conv.data(), c.conv.size() * 4);
    if (c.full) {
        const size_t DKV = hp.NKV * hp.DH, rowB = DKV * 2;
        const int w = std::min(c.pos, hp.window);
        const uint8_t * src = c.swa.data();
        for (size_t v = 0; v < visits.size(); ++v) {
            if (!layers[visits[v].layer].swa) continue;
            for (id<MTLBuffer> b : {kvK[v], kvV[v]}) {
                uint8_t * dst = (uint8_t *) b.contents;
                for (int q = c.pos - w; q < c.pos; ++q, src += rowB) memcpy(dst + (size_t) (q & (swaCap - 1)) * rowB, src, rowB);
            }
        }
        // the ring now only guarantees this window: light checkpoints below it must not be trusted
        hiWater = std::max(hiWater, c.pos + (int) swaCap);
    }
    seq.resize(c.pos);
    ckpts.erase(ckpts.begin() + best + 1, ckpts.end());
    return true;
}

void EngineImpl::reset() {
    ckpts.clear();
    hiWater = 0;
    lastFullPos = 0;
    seq.clear();
    memset(convSt[0].contents, 0, convSt[0].length);
    memset(convSt[1].contents, 0, convSt[1].length);
    convCur = 0;
    specN = 0;
    convPending = false;
    ngLast.clear();
    ngIndexed = 0;
    for (int i = 0; i < FR_SLOTS; ++i) frnnPos[i] = -1;
}

void EngineImpl::ensure_kv(int upto) {
    if (upto <= kvCap) return;
    int cap = std::max(kvCap, 8192);
    while (cap < upto + 256) cap *= 2;   // slack: attention reads whole key tiles
    const size_t DKV = hp.NKV * hp.DH;
    flush(true);
    for (size_t v = 0; v < visits.size(); ++v) {
        if (layers[visits[v].layer].swa) continue;
        id<MTLBuffer> k = new_buf((size_t) cap * DKV * 2, true), vv = new_buf((size_t) cap * DKV * 2, true);
        if (kvCap > 0) {
            memcpy(k.contents, kvK[v].contents, (size_t) kvCap * DKV * 2);
            memcpy(vv.contents, kvV[v].contents, (size_t) kvCap * DKV * 2);
        }
        drop_buf(kvK[v]);
        drop_buf(kvV[v]);
        kvK[v] = k;
        kvV[v] = vv;
        if (opt.kv8) {
            const size_t rowB = hp.NKV * 2 + DKV;
            kvK8.resize(visits.size());
            kvV8.resize(visits.size());
            id<MTLBuffer> k8 = new_buf((size_t) cap * rowB, true), v8 = new_buf((size_t) cap * rowB, true);
            if (kvCap > 0) {
                memcpy(k8.contents, kvK8[v].contents, (size_t) kvCap * rowB);
                memcpy(v8.contents, kvV8[v].contents, (size_t) kvCap * rowB);
            }
            drop_buf(kvK8[v]);
            drop_buf(kvV8[v]);
            kvK8[v] = k8;
            kvV8[v] = v8;
        }
    }
    for (size_t l = 0; l < dsK.size(); ++l) {
        // the draft model's context caches grow with the target's
        id<MTLBuffer> dk = new_buf((size_t) cap * DKV * 2, true), dv = new_buf((size_t) cap * DKV * 2, true);
        if (kvCap > 0 && dsK[l]) {
            memcpy(dk.contents, dsK[l].contents, (size_t) kvCap * DKV * 2);
            memcpy(dv.contents, dsV[l].contents, (size_t) kvCap * DKV * 2);
        }
        drop_buf(dsK[l]);
        drop_buf(dsV[l]);
        dsK[l] = dk;
        dsV[l] = dv;
    }
    kvCap = cap;
    partCap = std::max(cap, hp.window) / 32 + 1;
    const size_t psz = (size_t) hp.NH * partCap * (hp.DH + 4) * 4;
    drop_buf(bPart[0]);
    drop_buf(bPart[1]);
    bPart[0] = new_buf(psz, true);
    bPart[1] = new_buf(psz, true);
    if (opt.draft > 0) {
        mAttnChunks = std::max(cap / MINI_CHUNK + 1, hp.window / attnC + 4) + 2;
        drop_buf(bMAttn);
        bMAttn = new_buf((size_t) hp.NKV * mAttnChunks * attnQ * (hp.DH + 4) * 4, true);
    }
}

// ----------------------------------------------------------------------------------------------------
// single-token decode: one command buffer, no CPU round trips
void EngineImpl::decode_token(int32_t tok, int pos) {
    const uint32_t D = hp.D, DKV = hp.NKV * hp.DH, FF = hp.FF;
    const uint32_t nbStride = (uint32_t) NB_ * D;
    ensure_kv(pos + 1);
    if (profOn) { flush(true); profT = now_s(); }

    auto one = [&](const char * name, auto args, std::initializer_list<B> bufs, size_t nthreads) {
        groups(name, args, bufs, 1, 1, nthreads);
    };
    auto copy = [&](B src, B dst) {
        CountArgs a{D, 1, 0, 0};
        threads("d_copy", a, {src, dst}, D, 1);
    };
    const float qscale = 1.0f / sqrtf((float) D);
    auto mix = [&](B curB, const std::vector<uint32_t> & slots, B queries, int nq, int mode, B normw, uint32_t wStride, B out) {
        DMixArgs a{};
        a.nStates = (uint32_t) slots.size(); a.nq = nq; a.mode = mode; a.nbStride = nbStride;
        a.slot0 = slots.size() > 0 ? slots[0] : 0; a.slot1 = slots.size() > 1 ? slots[1] : 0;
        a.slot2 = slots.size() > 2 ? slots[2] : 0; a.slot3 = slots.size() > 3 ? slots[3] : 0;
        a.outStride = D; a.wStride = wStride; a.eps = hp.eps; a.mixEps = 1e-6f; a.qscale = qscale;
        fc[0] = a.nStates; fc[1] = a.nq; fc[2] = a.mode; fc[3] = 0; fcOn = true;
        one("d_mix", a, {curB, B(bNB), queries, normw, out}, 768);
    };

    DEmbedArgs ea{(uint32_t) tok, hp.eps};
    one("d_embed", ea, {B(bEmb), B(bGlobalSmall, oTokNorm * 4), B(bCur)}, 768);

    std::vector<uint32_t> blocks;
    const uint32_t bankSlot = 4;
    id<MTLBuffer> stIn = convSt[convCur], stOut = convSt[1 - convCur];
    const float invsStatic = 1.0f / sqrtf((float) hp.NCONV + 1.0f), invsDyn = 1.0f / sqrtf(2.0f * hp.NCONV + 1.0f);
    const uint32_t R = hp.gateRank, per = R * 2 * D + 2 * D * R;

    // loop controller chain: x_j (stride xStride in bDMixed) + msgs -> normalized inputs in bDX
    auto ctrl = [&](const LayerW & L, int nj, uint32_t ctrlBase, uint32_t xStride, size_t msgOff, const size_t * normOff) {
        DGateArgs ga{(uint32_t) nj, xStride, D, 1e-6f};
        fc[0] = fc[1] = fc[2] = 0; fc[3] = (uint32_t) nj; fcOn = true;
        one("d_gate_in", ga, {B(bDMixed), B(bDMsg, msgOff), B(bDGate)}, 768);
        DCtrlArgs ca{(uint32_t) nj, R, ctrlBase, per, xStride, D, 0, 0};
        groups("d_ctrl_down", ca, {B(L.loopW), B(bDGate), B(bDG)}, nj * R, 1, 128);
        threads("d_ctrl_apply", ca, {B(L.loopW), B(bDG), B(bDMixed), B(bDMsg, msgOff), B(bDXp)}, D, nj);
        DNormArgs na{(uint32_t) nj, D, D, (uint32_t) normOff[0], (uint32_t) (nj > 1 ? normOff[1] : 0),
                     (uint32_t) (nj > 2 ? normOff[2] : 0), 1, hp.eps};
        fc[0] = fc[1] = fc[2] = 0; fc[3] = (uint32_t) nj; fcOn = true;
        one("d_norm", na, {B(bDXp), B(L.small), B(bDX)}, 768);
    };

    for (size_t v = 0; v < visits.size(); ++v) {
        const int il = visits[v].layer, pass = visits[v].pass;
        const LayerW & L = layers[il];
        const bool looped = L.loop && pass > 0;
        const size_t stOff = v * (hp.NCONV - 1) * CH * 4;

        if (L.loop && il == hp.loopStart) {
            if (pass == 0) {
                copy(B(bCur), B(bPrelude));
            } else {
                copy(B(bNB, (size_t) 2 * nbStride * 4), B(bNB, (size_t) bankSlot * nbStride * 4));
                copy(B(bCur), B(bBankOut));
                copy(B(bPrelude), B(bCur));
                blocks.assign(1, 0u);
            }
        }
        if (dsOn) ds_tap((int) v, 1, B(bCur));
        if (looped) mix(B(bBankOut), {bankSlot}, B(L.small, L.oMsgQ * 4), 4, 2, B(L.small), 0, B(bDMsg));

        // ---- attention inputs
        uint32_t xStride = 0;
        if (!L.hasAttnResQ) {
            DNormArgs na{1, D, D, (uint32_t) L.oNormQ, 0, 0, 1, hp.eps};
            fc[0] = fc[1] = fc[2] = 0; fc[3] = 1; fcOn = true;
            one("d_norm", na, {B(bCur), B(L.small), B(bDX)}, 768);
        } else if (!looped) {
            mix(B(bCur), blocks, B(L.small, L.oAttnResQ * 4), L.swa ? 1 : 3, 1, B(L.small, L.oNormQ * 4), L.swa ? 0 : D, B(bDX));
            xStride = L.swa ? 0 : D;
        } else {
            mix(B(bCur), blocks, B(L.small, L.oAttnResQ * 4), L.swa ? 1 : 3, 2, B(L.small), 0, B(bDMixed));
            const size_t normOff[3] = {L.oNormQ, L.oNormK, L.oNormV};
            ctrl(L, 3, 0, L.swa ? 0 : D, 0, normOff);
            xStride = D;
        }
        prof("d_pre");

        DProjArgs pa{PW, kCol, vCol, xStride};
        if (opt.q8) groups("d_proj_q8", pa, {B(L.attnQ8), B(bDX), B(bP)}, PW / 2, 1, 32);
        else groups("d_proj", pa, {B(L.attn), B(bDX), B(bP)}, PW / 2, 1, 32);
        prof("d_proj");

        const uint32_t slotMask = L.swa ? swaCap - 1 : 0xffffffffu;
        DPrepArgs pp{};
        pp.slot = (uint32_t) pos & slotMask; pp.qCol = qCol; pp.rCol = rCol; pp.kCol = kCol; pp.vCol = vCol;
        pp.kDynCol = kDynCol; pp.vDynCol = vDynCol; pp.useDyn = L.swa ? 0 : 1;
        pp.qThr = hp.qkClip; pp.rThr = hp.relClip;
        pp.tau = L.swa ? 1.0f : 1.0f + hp.logAlpha * logf(std::max(1.0f, (float) (pos + 1) / hp.logFloor));
        pp.eps = hp.eps; pp.invs = L.swa ? invsStatic : invsDyn; pp.stStride = CH;
        groups("d_attn_prep", pp, {B(bP), B(L.small, L.oQClip * 4), B(L.small, L.oRClip * 4), B(L.small, L.oKConv * 4),
                                    B(L.small, L.oVConv * 4), B(stIn, stOff), B(stOut, stOff), B(bDQ), B(bRc), B(kvK[v]), B(kvV[v])},
               2 * hp.NH + 2 * hp.NKV, 1, 32);

        const uint32_t lo = (L.swa && pos >= hp.window) ? (uint32_t) (pos - (hp.window - 1)) : 0;
        const uint32_t nk = (uint32_t) pos - lo + 1;
        uint32_t n = (nk + 31) / 32;
        DAttnArgs aa{(uint32_t) pos, lo, nk, slotMask, (uint32_t) hp.extent, 1.0f / sqrtf((float) hp.DH), n, 0};
        if (opt.kv8 && !L.swa) {
            KV8Args ka{(uint32_t) pos, 1, 0, 0};
            groups("k_kv_q8", ka, {B(kvK[v]), B(kvK8[v])}, hp.NKV, 1, 32);
            groups("k_kv_q8", ka, {B(kvV[v]), B(kvV8[v])}, hp.NKV, 1, 32);
            groups("d_attn8", aa, {B(bDQ), B(bRc), B(L.small, L.oRelBias * 4), B(kvK8[v]), B(kvV8[v]), B(bPart[0])}, n, hp.NKV, 32);
        } else {
            groups("d_attn", aa, {B(bDQ), B(bRc), B(L.small, L.oRelBias * 4), B(kvK[v]), B(kvV[v]), B(bPart[0])}, n, hp.NKV, 32);
        }
        int src = 0;
        while (n > 32) {
            const uint32_t nOut = (n + 31) / 32;
            DMergeArgs ma{n, nOut, 0, sinkCol, gateCol, 0, 0, 0};
            groups("d_attn_merge", ma, {B(bPart[src]), B(bPart[1 - src]), B(bP), B(bDO)}, nOut, hp.NH, 32);
            src = 1 - src;
            n = nOut;
        }
        DMergeArgs ma{n, 1, 1, sinkCol, gateCol, 0, 0, 0};
        groups("d_attn_merge", ma, {B(bPart[src]), B(bPart[1 - src]), B(bP), B(bDO)}, 1, hp.NH, 32);
        prof(L.swa ? "d_attn_swa" : "d_attn_full");

        const bool blockStart = il % hp.blockSize == 0;
        if (blockStart) {
            const uint32_t slot = (uint32_t) (il / hp.blockSize);
            DNormArgs na{1, D, D, 0, 0, 0, 0, 1e-6f};
            fc[0] = fc[1] = fc[2] = 0; fc[3] = 1; fcOn = true;
            one("d_norm", na, {B(bCur), B(L.small), B(bNB, (size_t) slot * nbStride * 4)}, 768);
            blocks.push_back(slot);
        }
        DOutArgs oa{2 * DKV, blockStart ? 0u : 1u, invsStatic, CH};
        if (opt.q8) groups("d_wo_q8", oa, {B(L.woQ8), B(bDO), B(L.small, L.oAConv * 4), B(stIn, stOff), B(stOut, stOff), B(bCur)}, D, 1, 32);
        else groups("d_wo", oa, {B(L.wo), B(bDO), B(L.small, L.oAConv * 4), B(stIn, stOff), B(stOut, stOff), B(bCur)}, D, 1, 32);
        prof("d_wo");

        // ---- MoE feed-forward
        if (!looped) {
            mix(B(bCur), blocks, B(L.small, L.oFfnResQ * 4), 1, 1, B(L.small, L.oFfnNorm * 4), 0, B(bDX));
        } else {
            mix(B(bCur), blocks, B(L.small, L.oFfnResQ * 4), 1, 2, B(L.small), 0, B(bDMixed));
            const size_t normOff[3] = {L.oFfnNorm, 0, 0};
            ctrl(L, 1, 3, 0, (size_t) 3 * D * 4, normOff);
        }
        prof("d_pre");
        {
            id<MTLComputePipelineState> ps;
            // d_route_mv has no args struct: bind buffers from index 0
            begin();
            ps = pipe("d_route_mv");
            [enc setComputePipelineState:ps];
            [enc setBuffer:L.small offset:L.oGateInp * 4 atIndex:0];
            [enc setBuffer:bDX offset:0 atIndex:1];
            [enc setBuffer:bRLog offset:0 atIndex:2];
            [enc dispatchThreadgroups:MTLSizeMake(hp.NEXP, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        }
        DRouteArgs ra{hp.wScale, 1e-20f, 0, 0};
        one("d_route_topk", ra, {B(bRLog), B(L.small, L.oProbsB * 4), B(L.small, L.oScaleGU * 4), B(L.small, L.oScaleDN * 4), B(bDRoute)},
            hp.NEXP);
        prof("d_route");
        DExpArgs xa{hp.situBeta, hp.situLBeta, 0, 0};
        DOutArgs fa{2 * DKV + D, 1, invsStatic, CH};
        if (L.guNV) groups("d_gate_up_nv", xa, {B(L.guNV), B(bDX), B(bScaleTab), B(bDRoute), B(bDA)}, FF, hp.NUSED, 32);
        else if (L.guFmt) groups(native_kernel(true, L.guFmt), xa, {B(L.guP), B(bDX), B(bDRoute), B(bDA)}, FF, hp.NUSED, 32);
        else if (L.guQ8) groups("d_gate_up_q8", xa, {B(L.guQ8), B(bDX), B(bDRoute), B(bDA)}, FF, hp.NUSED, 32);
        else groups("d_gate_up_h", xa, {B(L.guH), B(bDX), B(bDRoute), B(bDA)}, FF, hp.NUSED, 32);
        prof("d_gate_up");
        if (L.dnNV) groups("d_down_nv", fa, {B(L.dnNV), B(bDA), B(bScaleTab), B(bDRoute), B(bDPart)}, D / 2, hp.NUSED, 32);
        else if (L.dnFmt) groups(native_kernel(false, L.dnFmt), fa, {B(L.dnP), B(bDA), B(bDRoute), B(bDPart)}, D / 2, hp.NUSED, 32);
        else if (L.dnQ8) groups("d_down_q8", fa, {B(L.dnQ8), B(bDA), B(bDRoute), B(bDPart)}, D / 2, hp.NUSED, 32);
        else groups("d_down_h", fa, {B(L.dnH), B(bDA), B(bDRoute), B(bDPart)}, D / 2, hp.NUSED, 32);
        threads("d_down_fin", fa, {B(bDPart), B(bDRoute), B(L.small, L.oFConv * 4), B(stIn, stOff), B(stOut, stOff), B(bCur)}, D, 1);
        prof("d_down");
    }

    mix(B(bCur), blocks, B(bGlobalSmall, oOutResQ * 4), 1, 1, B(bGlobalSmall, oOutNorm * 4), 0, B(bDX));
    if (dsOn) {
        ds_tap((int) visits.size(), 1, B(bDX));
        ds_inject(1, pos);
    }
    DLogitArgs la{(uint32_t) hp.vocab, hp.softcap};
    if (opt.q8) groups("d_logits_q8", la, {B(bOutQ8), B(bDX), B(bLogits)}, (hp.vocab + 1) / 2, 1, 32);
    else groups("d_logits", la, {B(bOut), B(bDX), B(bLogits)}, (hp.vocab + 1) / 2, 1, 32);
    flush(true);
    prof("d_logits");
    convCur = 1 - convCur;
}

// ----------------------------------------------------------------------------------------------------
// x' = control(x, msg) followed by rmsnorm * w  ->  outH (half) [+ bXff32]
void EngineImpl::control(int N, const LayerW & L, int c, B x, B msg, size_t normOff, B outH, bool wantF32) {
    const size_t D = hp.D, R = hp.gateRank;
    const size_t per = (R * 2 * D + 2 * D * R) * 2;   // bytes per controller
    NormArgs na{(uint32_t) N, 1e-6f};
    groups("k_ctrl_gate_in", na, {x, msg, B(bGateIn)}, N, 1, 32);
    gemm("k_gemm_hh_f", B(bGateIn), B(L.loopW, c * per), B(bG64), N, (uint32_t) R, (uint32_t) (2 * D), (uint32_t) (2 * D), (uint32_t) R);
    gemm("k_gemm_fh_f", B(bG64), B(L.loopW, c * per + R * 2 * D * 2), B(bGF), N, (uint32_t) (2 * D), (uint32_t) R, (uint32_t) R, (uint32_t) (2 * D));
    CtrlArgs ca{(uint32_t) N, hp.eps, wantF32 ? 1u : 0u, 0};
    groups("k_ctrl_apply_norm", ca, {x, msg, B(bGF), B(L.small, normOff * 4), outH, B(bXff32)}, N, 1, 32);
}

// MoE feed-forward: bXin[0] (half) / bXff32 (f32) -> bY
// FP16 expert weights of a layer for the prompt GEMMs: the persistent copy, or (low-memory mode) the packed
// weights expanded into a scratch buffer
id<MTLBuffer> EngineImpl::expert_f16(const LayerW & L, bool gateUp) {
    id<MTLBuffer> H = gateUp ? L.guH : L.dnH;
    if (H) return H;
    id<MTLBuffer> dst = gateUp ? bXgu : bXdn, NV = gateUp ? L.guNV : L.dnNV, P = gateUp ? L.guP : L.dnP, Q8 = gateUp ? L.guQ8 : L.dnQ8;
    const uint32_t K = gateUp ? hp.D : hp.FF, rows = (uint32_t) hp.NEXP * (gateUp ? 2 * hp.FF : hp.D);
    if (NV) {
        NvArgs a{rows, K, K / 16 + K / 2, 0};
        threads("k_nvfp4_to_f16", a, {B(NV), B(bScaleTab), B(dst)}, K / 16, rows);
    } else if (P) {
        const uint32_t fmt = gateUp ? L.guFmt : L.dnFmt;
        NvArgs a{rows, K, (uint32_t) native_row_bytes(fmt, K), 0};
        threads(expand_kernel(fmt), a, {B(P), B(dst)}, fmt == GG_Q2_0 ? K / 16 : K / 32, rows);
    } else {
        NvArgs a{rows, K, K + K / 16, 0};
        threads("k_x_q8", a, {B(Q8), B(dst)}, K / 32, rows);
    }
    return dst;
}

void EngineImpl::moe(int N, const LayerW & L) {
    const size_t D = hp.D, FF = hp.FF;
    const int U = hp.NUSED, NE = hp.NEXP;
    gemm("k_gemm_ff_f", B(bXff32), B(L.small, L.oGateInp * 4), B(bRLog), N, NE, (uint32_t) D, (uint32_t) D, NE);
    RouteArgs ra{(uint32_t) N, hp.wScale, 1e-20f, 0};
    threads("k_route", ra, {B(bRLog), B(L.small, L.oProbsB * 4), B(bIds), B(bRW)}, N, 1);
    flush(true);
    prof("route");

    // sort (token, slot) pairs by expert
    const uint32_t * ids = (const uint32_t *) bIds.contents;
    float * rw = (float *) bRW.contents;
    uint32_t * pairTok = (uint32_t *) bPairTok.contents;
    uint32_t * rowOfPair = (uint32_t *) bRowOfPair.contents;
    float * rowScale = (float *) bRowScale.contents;
    uint32_t * tiles = (uint32_t *) bTiles.contents;
    const int R = N * U;
    std::vector<int> cnt(NE + 1, 0);
    for (int i = 0; i < R; ++i) cnt[ids[i] + 1]++;
    for (int e = 0; e < NE; ++e) cnt[e + 1] += cnt[e];
    std::vector<int> fill(cnt.begin(), cnt.end() - 1);
    for (int i = 0; i < R; ++i) {
        const uint32_t e = ids[i];
        const int r = fill[e]++;
        pairTok[r] = i / U;
        rowOfPair[i] = r;
        rowScale[r] = L.scaleGU[e];
        rw[i] *= L.scaleDN[e];
    }
    uint32_t nTiles = 0;
    for (int e = 0; e < NE; ++e) {
        for (int r0 = cnt[e]; r0 < cnt[e + 1]; r0 += 64) {
            tiles[3 * nTiles] = e;
            tiles[3 * nTiles + 1] = r0;
            tiles[3 * nTiles + 2] = std::min(64, cnt[e + 1] - r0);
            ++nTiles;
        }
    }

    prof("moe_sort_cpu");
    RowsArgs rows{(uint32_t) R, (uint32_t) N, hp.situBeta, hp.situLBeta};
    threads("k_gather", rows, {B(bXin), B(bPairTok), B(bXg)}, D / 4, R);
    prof("moe_gather");
    MoeArgs m1{(uint32_t) (2 * FF), (uint32_t) D, nTiles, 0};
    id<MTLBuffer> guW = expert_f16(L, true);
    prof("moe_expand");
    groups("k_gemm_moe", m1, {B(bXg), B(guW), B(bH), B(bTiles)}, nTiles, (2 * FF + 63) / 64, 128);
    prof("moe_gemm_gu");
    threads("k_act", rows, {B(bH), B(bRowScale), B(bA)}, FF, R);
    prof("moe_act");
    MoeArgs m2{(uint32_t) D, (uint32_t) FF, nTiles, 0};
    id<MTLBuffer> dnW = expert_f16(L, false);
    prof("moe_expand");
    groups("k_gemm_moe", m2, {B(bA), B(dnW), B(bOe), B(bTiles)}, nTiles, (D + 63) / 64, 128);
    prof("moe_gemm_dn");
    threads("k_combine", rows, {B(bOe), B(bRowOfPair), B(bRW), B(bY)}, D, N);
    prof("moe_combine");
}

// MoE feed-forward of a multi-token pass: every (token, selected expert) pair runs through the generation
// kernels, which read the packed experts in place.  bXff32 (f32) -> bY
void EngineImpl::moe_mini(int N, const LayerW & L) {
    const uint32_t D = hp.D, FF = hp.FF, NE = hp.NEXP, NP = (uint32_t) (N * hp.NUSED);
    mmv("m_mv_f_f", N, B(L.small, L.oGateInp * 4), NE, D, B(bXff32), D, B(bRLog), NE);
    DRouteArgs ra{hp.wScale, 1e-20f, 0, 0};
    groups("d_route_topk", ra, {B(bRLog), B(L.small, L.oProbsB * 4), B(L.small, L.oScaleGU * 4), B(L.small, L.oScaleDN * 4), B(bMRoute)},
           N, 1, hp.NEXP);
    // group the (token, expert) pairs by expert, then one read per distinct expert
    struct { uint32_t nPairs, pad0, pad1, pad2; } ga{NP, 0, 0, 0};
    groups("m_group", ga, {B(bMRoute), B(bMGroup)}, 1, 1, 64);
    DExpArgs xa{hp.situBeta, hp.situLBeta, D, 0};
    auto gname = [&](bool gateUp, id<MTLBuffer> NV, uint32_t fmt, id<MTLBuffer> Q8) {
        std::string k = native_kernel(gateUp, fmt) ? native_kernel(gateUp, fmt) : (gateUp ? "d_gate_up_h" : "d_down_h");
        if (NV) k = gateUp ? "d_gate_up_nv" : "d_down_nv";
        else if (!fmt && Q8) k = gateUp ? "d_gate_up_q8" : "d_down_q8";
        k[0] = 'g';   // d_gate_up_x -> g_gate_up_x
        return k;
    };
    id<MTLBuffer> guW = L.guNV ? L.guNV : (L.guFmt ? L.guP : (L.guQ8 ? L.guQ8 : L.guH));
    id<MTLBuffer> dnW = L.dnNV ? L.dnNV : (L.dnFmt ? L.dnP : (L.dnQ8 ? L.dnQ8 : L.dnH));
    groups(gname(true, L.guNV, L.guFmt, L.guQ8).c_str(), xa, {B(guW), B(bXff32), B(bMRoute), B(bMGroup), B(bMA)}, FF, NP, 32);
    groups(gname(false, L.dnNV, L.dnFmt, L.dnQ8).c_str(), xa, {B(dnW), B(bMA), B(bMGroup), B(bMPart)}, D / 2, NP, 32);
    CountArgs ca{(uint32_t) N, 1, 0, 0};
    threads("k_combine_mini", ca, {B(bMPart), B(bMRoute), B(bY)}, D, N);
    prof("moe_mini");
}

// mat-vec with one right-hand side per token of the pass
void EngineImpl::mmv(const char * kern, int N, B W, uint32_t rows, uint32_t K, B x, uint32_t ldx, B y, uint32_t ldy, float cap) {
    MMvArgs a{K, ldx, ldy, cap};
    fc[0] = fc[1] = fc[2] = 0; fc[3] = (uint32_t) N; fcOn = true;
    groups(kern, a, {W, x, y}, rows / 2, 1, 32);
}

// Multi-token generation pass: N tokens (the one being appended plus drafts) at positions pos0.., logits of
// every row in bLogitsM.  Same computation as forward_batch, built for a low fixed cost at small N: wide
// per-token kernels, multi-right-hand-side mat-vecs on the generation weights instead of GEMMs, one cache
// read for all tokens, and the generation kernels for the experts.  The conv state is left to settle().
void EngineImpl::forward_mini(const int32_t * toks, int N, int pos0) {
    const uint32_t D = hp.D, DQ = hp.NH * hp.DH, DKV = hp.NKV * hp.DH;
    const size_t nbStride = (size_t) NB_ * D;
    ensure_kv(pos0 + N);
    memcpy(bTok.contents, toks, (size_t) N * 4);
    if (profOn) { flush(true); profT = now_s(); }

    const size_t TG = 768;
    EmbedArgs ea{(uint32_t) N, hp.eps};
    groups("m_embed", ea, {B(bTok), B(bEmb), B(bGlobalSmall, oTokNorm * 4), B(bCur)}, N, 1, TG);

    std::vector<uint32_t> blocks;
    const uint32_t bankSlot = 4;
    id<MTLBuffer> stIn = convSt[convCur];
    const float invsStatic = 1.0f / sqrtf((float) hp.NCONV + 1.0f), invsDyn = 1.0f / sqrtf(2.0f * hp.NCONV + 1.0f);
    const float qscale = 1.0f / sqrtf((float) D);
    const size_t R = hp.gateRank, per = (R * 2 * D + 2 * D * R) * 2;
    const uint32_t rbq = D + D / 16;   // bytes of an 8-bit row with K = D
    const uint32_t convVar = (uint32_t) (visits.size() * (hp.NCONV - 1) * CH);

    auto copy = [&](B src, B dst) {
        CountArgs a{(uint32_t) (N * D), 1, 0, 0};
        threads("d_copy", a, {src, dst}, (size_t) N * D, 1);
    };
    auto mix = [&](B curB, const std::vector<uint32_t> & slots, B queries, int nq, int mode, B normw, uint32_t wStride,
                   B outH, B outF, bool wantF32) {
        MixArgs a{};
        a.N = N; a.nStates = (uint32_t) slots.size(); a.nq = nq; a.mode = mode;
        a.slot0 = slots.size() > 0 ? slots[0] : 0; a.slot1 = slots.size() > 1 ? slots[1] : 0;
        a.slot2 = slots.size() > 2 ? slots[2] : 0; a.slot3 = slots.size() > 3 ? slots[3] : 0;
        a.nbStride = (uint32_t) nbStride; a.outStride = (uint32_t) nbStride; a.wantF32 = wantF32;
        a.eps = hp.eps; a.mixEps = 1e-6f; a.qscale = qscale; a.wStride = wStride;
        groups("m_mix_norm", a, {curB, B(bNB), queries, normw, outH, outF}, N, 1, TG);
    };
    auto control = [&](const LayerW & L, int c, B x, B msg, size_t normOff, B outH, bool wantF32) {
        NormArgs na{(uint32_t) N, 1e-6f};
        groups("m_ctrl_gate_in", na, {x, msg, B(bGateIn)}, N, 1, TG);
        mmv("m_mv_h_h", N, B(L.loopW, c * per), (uint32_t) R, 2 * D, B(bGateIn), 2 * D, B(bG64), (uint32_t) R);
        mmv("m_mv_h_f", N, B(L.loopW, c * per + R * 2 * D * 2), 2 * D, (uint32_t) R, B(bG64), (uint32_t) R, B(bGF), 2 * D);
        CtrlArgs ca{(uint32_t) N, hp.eps, wantF32 ? 1u : 0u, 0};
        groups("m_ctrl_apply_norm", ca, {x, msg, B(bGF), B(L.small, normOff * 4), outH, B(bXff32)}, N, 1, TG);
    };

    for (size_t v = 0; v < visits.size(); ++v) {
        const int il = visits[v].layer, pass = visits[v].pass;
        const LayerW & L = layers[il];
        const bool looped = L.loop && pass > 0;
        const size_t stOff = v * (hp.NCONV - 1) * CH * 4;
        // conv state after every prefix of the pass (see settle())
        auto conv_state = [&](ConvArgs cv, B X, size_t C) {
            cv.mode = convVar;
            id<MTLComputePipelineState> ps;
            enc_setup("k_conv_state_var", cv, {X, B(stIn, stOff), B(bConvVar, stOff)}, &ps);
            [enc dispatchThreads:MTLSizeMake(C, hp.NCONV - 1, N) threadsPerThreadgroup:MTLSizeMake(std::min<size_t>(ps.threadExecutionWidth, C), 1, 1)];
        };

        if (L.loop && il == hp.loopStart) {
            if (pass == 0) {
                copy(B(bCur), B(bPrelude));
            } else {
                copy(B(bNB, 2 * nbStride * 4), B(bNB, bankSlot * nbStride * 4));
                copy(B(bCur), B(bBankOut));
                copy(B(bPrelude), B(bCur));
                blocks.assign(1, 0u);
            }
        }
        if (dsOn) ds_tap((int) v, N, B(bCur));
        if (looped) mix(B(bBankOut), {bankSlot}, B(L.small, L.oMsgQ * 4), 4, 2, B(L.small), 0, B(bXin), B(bMsgs), false);

        // ---- attention inputs
        bool sameInputs = true;
        if (!L.hasAttnResQ) {
            mix(B(bCur), {}, B(L.small), 1, 0, B(L.small, L.oNormQ * 4), 0, B(bXin), B(bXff32), false);
        } else if (!looped) {
            mix(B(bCur), blocks, B(L.small, L.oAttnResQ * 4), L.swa ? 1 : 3, 1, B(L.small, L.oNormQ * 4), L.swa ? 0 : D,
                B(bXin), B(bXff32), false);
            sameInputs = L.swa;
        } else {
            mix(B(bCur), blocks, B(L.small, L.oAttnResQ * 4), L.swa ? 1 : 3, 2, B(L.small), 0, B(bXin), B(bMixed), false);
            const size_t normOff[3] = {L.oNormQ, L.oNormK, L.oNormV};
            for (int j = 0; j < 3; ++j)
                control(L, j, B(bMixed, (L.swa ? 0 : j) * nbStride * 4), B(bMsgs, j * nbStride * 4), normOff[j], B(bXin, j * nbStride * 2), false);
            sameInputs = false;
        }
        prof("m_pre");

        // ---- projections (the generation weights: 8-bit copies unless --precise)
        {
            const bool q8 = L.attnQ8 != nil;
            const char * kern = q8 ? "m_mv_q8_h" : "m_mv_h_h";
            id<MTLBuffer> W = q8 ? L.attnQ8 : L.attn;
            const size_t rowB = q8 ? rbq : (size_t) D * 2;
            if (sameInputs) {
                mmv(kern, N, B(W), PW, D, B(bXin), D, B(bP), PW);
            } else {
                const uint32_t dyn = L.swa ? 0 : hp.NKV * hp.NCONV;
                mmv(kern, N, B(W), kCol, D, B(bXin), D, B(bP), PW);
                mmv(kern, N, B(W, (size_t) kCol * rowB), DKV + dyn, D, B(bXin, nbStride * 2), D, B(bP, kCol * 4), PW);
                mmv(kern, N, B(W, (size_t) vCol * rowB), DKV + dyn, D, B(bXin, 2 * nbStride * 2), D, B(bP, vCol * 4), PW);
            }
        }
        prof("m_proj");

        // ---- Q / R / K / V post-processing, caches, conv state
        PrepQArgs pq{(uint32_t) N, (uint32_t) pos0, PW, qCol, rCol, hp.qkClip, hp.relClip, hp.logAlpha, hp.logFloor, L.swa ? 0u : 1u,
                     1.0f / sqrtf((float) hp.DH), 0, 0, 0, 0, 0};
        groups("m_prep_q", pq, {B(bP), B(L.small, L.oQClip * 4), B(L.small, L.oRClip * 4), B(bQh), B(bRc)}, hp.NH, N, 32);
        const uint32_t slotMask = L.swa ? swaCap - 1 : 0xffffffffu;
        PrepKVArgs pk{};
        pk.N = N; pk.pos0 = pos0; pk.PW = PW; pk.col = kCol; pk.dynCol = kDynCol; pk.useDyn = L.swa ? 0 : 1; pk.goff = 0; pk.isV = 0;
        pk.thr = hp.qkClip; pk.eps = hp.eps; pk.invs = L.swa ? invsStatic : invsDyn; pk.slotMask = slotMask; pk.stStride = CH;
        groups("m_prep_kv", pk, {B(bP), B(L.small, L.oKConv * 4), B(stIn, stOff), B(kvK[v])}, hp.NKV, N, 32);
        PrepKVArgs pv = pk;
        pv.col = vCol; pv.dynCol = vDynCol; pv.goff = DKV; pv.isV = 1;
        groups("m_prep_kv", pv, {B(bP), B(L.small, L.oVConv * 4), B(stIn, stOff), B(kvV[v])}, hp.NKV, N, 32);
        if (opt.kv8 && !L.swa) {
            KV8Args ka{(uint32_t) pos0, (uint32_t) N, 0, 0};
            groups("k_kv_q8", ka, {B(kvK[v]), B(kvK8[v])}, hp.NKV, N, 32);
            groups("k_kv_q8", ka, {B(kvV[v]), B(kvV8[v])}, hp.NKV, N, 32);
        }
        ConvArgs cs{(uint32_t) N, DKV, 0, 0, 0, PW, kCol, CH};
        conv_state(cs, B(bP), DKV);
        cs.goff = DKV; cs.xcol = vCol;
        conv_state(cs, B(bP), DKV);
        prof("m_prep");

        // ---- attention: one cache read for all tokens
        {
            MAttnArgs ma{(uint32_t) N, (uint32_t) pos0, 0, 0, (uint32_t) hp.extent, PW, sinkCol, gateCol, 0, slotMask, 0, 0, 0, 0, 0, 0};
            if (L.swa) {
                const uint32_t lo0 = pos0 >= hp.window ? (uint32_t) (pos0 - (hp.window - 1)) : 0;
                ma.window = (uint32_t) hp.window;
                ma.base0 = (lo0 / attnC) * attnC;
                ma.chunkKeys = (uint32_t) attnC;
            } else {
                ma.chunkKeys = MINI_CHUNK;
            }
            ma.nChunks = ((uint32_t) (pos0 + N) - ma.base0 + ma.chunkKeys - 1) / ma.chunkKeys;
            groups("m_attn_chunk", ma, {B(bQh), B(kvK[v]), B(kvV[v]), B(bRc), B(L.small, L.oRelBias * 4), B(bMAttn)}, ma.nChunks, hp.NKV, 256);
            groups("m_attn_merge", ma, {B(bMAttn), B(bP), B(bO)}, hp.NH, N, 32);
        }
        prof(L.swa ? "m_attn_swa" : "m_attn_full");
        if (L.woQ8) mmv("m_mv_q8_h", N, B(L.woQ8), D, DQ, B(bO), DQ, B(bAO), D);
        else mmv("m_mv_h_h", N, B(L.wo), D, DQ, B(bO), DQ, B(bAO), D);

        const bool blockStart = il % hp.blockSize == 0;
        if (blockStart) {
            const uint32_t slot = (uint32_t) (il / hp.blockSize);
            NormArgs na{(uint32_t) N, 1e-6f};
            groups("m_norm_store", na, {B(bCur), B(bNB, slot * nbStride * 4)}, N, 1, TG);
            blocks.push_back(slot);
        }
        ConvArgs ca{(uint32_t) N, D, 2 * DKV, blockStart ? 0u : 1u, invsStatic, D, 0, CH};
        threads("k_conv_residual", ca, {B(bAO), B(L.small, L.oAConv * 4), B(stIn, stOff), B(bCur)}, D / 4, N);
        conv_state(ca, B(bAO), D);
        prof("m_wo");

        // ---- MoE feed-forward
        if (!looped) {
            mix(B(bCur), blocks, B(L.small, L.oFfnResQ * 4), 1, 1, B(L.small, L.oFfnNorm * 4), 0, B(bXin), B(bXff32), true);
        } else {
            mix(B(bCur), blocks, B(L.small, L.oFfnResQ * 4), 1, 2, B(L.small), 0, B(bXin), B(bMixed), false);
            control(L, 3, B(bMixed), B(bMsgs, 3 * nbStride * 4), L.oFfnNorm, B(bXin), true);
        }
        prof("m_pre");
        moe_mini(N, L);
        ConvArgs cf{(uint32_t) N, D, 2 * DKV + D, 1, invsStatic, D, 0, CH};
        threads("k_conv_residual", cf, {B(bY), B(L.small, L.oFConv * 4), B(stIn, stOff), B(bCur)}, D / 4, N);
        conv_state(cf, B(bY), D);
        prof("m_post");
    }

    mix(B(bCur), blocks, B(bGlobalSmall, oOutResQ * 4), 1, 1, B(bGlobalSmall, oOutNorm * 4), 0, B(bXin), B(bXff32), dsOn);
    if (dsOn) {
        ds_tap((int) visits.size(), N, B(bXff32));
        ds_inject(N, pos0);
    }
    if (bOutQ8) mmv("m_mv_q8_h", N, B(bOutQ8), (uint32_t) hp.vocab, D, B(bXin), D, B(bLogitsM), (uint32_t) hp.vocabPadded, hp.softcap);
    else mmv("m_mv_h_h", N, B(bOut), (uint32_t) hp.vocab, D, B(bXin), D, B(bLogitsM), (uint32_t) hp.vocabPadded, hp.softcap);
    flush(true);
    prof("m_logits");
}

// ====================================================================================================
// DSpark draft model
// ====================================================================================================
void EngineImpl::ds_load(const std::string & path) {
    GGUF g(path);
    if (g.str("general.architecture") != "dflash") throw std::runtime_error("engine: the draft model is not a DSpark (dflash) model");
    const size_t D = hp.D, DQ = hp.NH * hp.DH, DKV = hp.NKV * hp.DH, DFF = 4 * D, R = 256;
    const int nl = (int) g.u64("dflash.block_count");
    if ((int) g.u64("dflash.embedding_length") != hp.D || (int) g.u64("dflash.attention.head_count") != hp.NH ||
        (int) g.u64("dflash.attention.head_count_kv") != hp.NKV || (int) g.u64("dflash.attention.key_length") != hp.DH ||
        g.u64("dflash.feed_forward_length") != DFF || nl != 3 || (int) g.u64("dflash.rope.dimension_count") != hp.DH ||
        !g.find("markov_w1.weight") || !g.find("joint_proj.weight") || !g.find("markov_ctx_mix.weight") || !g.find("blk.0.ctx_mix.weight") ||
        !g.find("mask_embd.weight") || g.boolean("dflash.sample_from_anchor", true) || g.boolean("dflash.attention.causal", false))
        throw std::runtime_error("engine: unsupported DSpark draft model layout");
    dsBlock = (int) g.u64("dflash.block_size");
    dsRopeBase = (float) g.f64("dflash.rope.freq_base");
    dsCap = (float) g.f64("dflash.final_logit_softcapping", 0.0);
    dsEps = (float) g.f64("dflash.attention.layer_norm_rms_epsilon");
    dsWindow = (int) g.u64("dflash.attention.sliding_window", 0);
    auto pat = g.int_array("dflash.attention.sliding_window_pattern");
    for (auto id : g.int_array("dflash.target_layers")) dsTapVisit.push_back((int) id);
    if (dsTapVisit.size() != 4 || dsBlock < 2 || dsBlock > MINI_MAX) throw std::runtime_error("engine: unsupported DSpark configuration");
    for (int id : dsTapVisit) if (id < 0 || id > (int) visits.size()) throw std::runtime_error("engine: DSpark target layer out of range");

    auto f32v = [&](const std::string & name, float * dst, size_t n) {
        const GGTensor & t = g.get(name);
        if (t.nelements() != n || !dequant_supported(t.type) || n % dequant_block(t.type)) throw std::runtime_error("engine: unexpected draft tensor " + name);
        dequant_range(t.type, t.data, 0, n, dst);
    };
    // matrix [rows][K]: Q8_0 rows are kept as they are, anything else becomes FP16
    auto mat = [&](const std::string & name, size_t rows, size_t K, bool allowQ8) {
        const GGTensor & t = g.get(name);
        if (t.dims.size() != 2 || t.dims[0] != K || t.dims[1] != rows || !dequant_supported(t.type)) throw std::runtime_error("engine: unexpected draft tensor " + name);
        DsMat m;
        const uint8_t * src = t.data;
        if (t.type == GG_Q8_0 && allowQ8) {
            m.q8 = true;
            const size_t nb = K / 32, rowB = K + K / 16;
            m.w = new_buf(rows * rowB + 65536);
            uint8_t * dst = (uint8_t *) m.w.contents;
            dispatch_apply(rows, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t r) {
                const uint8_t * sp = src + r * nb * 34;
                uint8_t * d = dst + r * rowB;
                for (size_t u = 0; u < nb; ++u) { memcpy(d + u * 2, sp + u * 34, 2); memcpy(d + nb * 2 + u * 32, sp + u * 34 + 2, 32); }
            });
        } else {
            m.w = new_buf(rows * K * 2 + 65536);
            __fp16 * dst = (__fp16 *) m.w.contents;
            const uint32_t type = t.type;
            const size_t n = rows * K, chunk = 1u << 18;
            dispatch_apply((n + chunk - 1) / chunk, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t ci) {
                const size_t i0 = ci * chunk, i1 = std::min(n, i0 + chunk);
                std::vector<float> tmp(i1 - i0);
                dequant_range(type, src, i0, i1, tmp.data());
                for (size_t i = i0; i < i1; ++i) dst[i] = (__fp16) tmp[i - i0];
            });
        }
        return m;
    };

    dsL.resize(nl);
    dsSwa.resize(nl);
    for (int il = 0; il < nl; ++il) {
        DsLayer & L = dsL[il];
        const std::string p = "blk." + std::to_string(il) + ".";
        dsSwa[il] = dsWindow > 0 && il < (int) pat.size() && pat[il] != 0;
        L.wq = mat(p + "attn_q.weight", DQ, D, true);
        L.wk = mat(p + "attn_k.weight", DKV, D, true);
        L.wv = mat(p + "attn_v.weight", DKV, D, true);
        L.wo = mat(p + "attn_output.weight", D, DQ, true);
        L.wg = mat(p + "ffn_gate.weight", DFF, D, true);
        L.wu = mat(p + "ffn_up.weight", DFF, D, true);
        L.wd = mat(p + "ffn_down.weight", D, DFF, true);
        L.wkc = mat(p + "attn_k_ctx.weight", DKV, D, false).w;
        L.wvc = mat(p + "attn_v_ctx.weight", DKV, D, false).w;
        L.small = new_buf((3 * D + 3 * hp.DH) * 4);
        float * sm = (float *) L.small.contents;
        f32v(p + "attn_norm.weight", sm, D);
        f32v(p + "ffn_norm.weight", sm + D, D);
        f32v(p + "ctx_norm.weight", sm + 2 * D, D);
        f32v(p + "attn_q_norm.weight", sm + 3 * D, hp.DH);
        f32v(p + "attn_k_norm.weight", sm + 3 * D + hp.DH, hp.DH);
        f32v(p + "attn_k_ctx_norm.weight", sm + 3 * D + 2 * hp.DH, hp.DH);
        f32v(p + "ctx_mix.weight", L.mix, 4);
    }
    f32v("markov_ctx_mix.weight", dsMMix, 4);
    const GGTensor & tm = g.get("mask_embd.weight");
    const size_t nMask = tm.nelements() / D;
    if (tm.nelements() % D || (int) nMask < dsBlock - 1) throw std::runtime_error("engine: unexpected draft tensor mask_embd.weight");
    dsJoint = mat("joint_proj.weight", 3 * R, 2 * R + 2 * D, false).w;
    dsW1 = mat("markov_w1.weight", (size_t) hp.vocab, R, false).w;
    dsW2 = mat("markov_w2.weight", (size_t) hp.vocab, R, false).w;
    dsoJB = 0; dsoW2B = 3 * R; dsoMask = dsoW2B + (size_t) hp.vocabPadded; dsoOutN = dsoMask + nMask * D; dsoTokN = dsoOutN + D; dsoMCN = dsoTokN + D;
    dsSmall = new_buf((dsoMCN + D) * 4);
    float * sm = (float *) dsSmall.contents;
    memset(sm, 0, (dsoMCN + D) * 4);
    f32v("joint_proj.bias", sm + dsoJB, 3 * R);
    if (g.find("markov_w2.bias")) f32v("markov_w2.bias", sm + dsoW2B, (size_t) hp.vocab);
    f32v("mask_embd.weight", sm + dsoMask, nMask * D);
    f32v("output_norm.weight", sm + dsoOutN, D);
    if (g.find("token_embd_norm.weight")) f32v("token_embd_norm.weight", sm + dsoTokN, D);
    else for (size_t i = 0; i < D; ++i) sm[dsoTokN + i] = 1.0f;
    f32v("markov_ctx_norm.weight", sm + dsoMCN, D);
    dsK.resize(nl);
    dsV.resize(nl);
    dsOn = true;
    if (opt.verbose) fprintf(stderr, "engine: DSpark draft model loaded (%d blocks, block size %d)\n", nl, dsBlock);
}

void EngineImpl::ds_alloc() {
    const size_t D = hp.D, DQ = hp.NH * hp.DH, DKV = hp.NKV * hp.DH, DFF = 4 * D, N = NB_, T = MINI_MAX;
    bTaps = new_buf(4 * N * D * 4, true);
    bDsX = new_buf(N * D * 2, true);
    bDsP = new_buf(N * 2 * DKV * 4, true);
    bDsRope = new_buf(N * hp.DH * 4, true);
    bDsFrnn = new_buf((size_t) FR_SLOTS * D * 4, true);
    bDsCur = new_buf(T * D * 4, true);
    bDsXh = new_buf(T * D * 2, true);
    bDsQP = new_buf(T * (DQ + 2 * DKV) * 4, true);
    bDsQh = new_buf(T * DQ * 2, true);
    bDsBK = new_buf(256 * DKV * 2, true);
    bDsBV = new_buf(256 * DKV * 2, true);
    bDsO = new_buf(T * DQ * 2, true);
    bDsAO = new_buf(T * D * 4, true);
    bDsG = new_buf(T * DFF * 4, true);
    bDsU = new_buf(T * DFF * 4, true);
    bDsAct = new_buf(T * DFF * 2, true);
    bDsH = new_buf(T * D * 4, true);
    bDsBase = new_buf(T * (size_t) hp.vocabPadded * 4, true);
    bDsU5 = new_buf((2 * 256 + 2 * D) * 4, true);
    bDsJ = new_buf(3 * 256 * 4, true);
    bDsState = new_buf(256 * 4, true);
    bDsMarkov = new_buf(256 * 4, true);
    bDsBias = new_buf((size_t) hp.vocabPadded * 4, true);
    bDsCol = new_buf((size_t) hp.vocabPadded * 4, true);
    bDsArg = new_buf(((size_t) hp.vocabPadded / 1024 + 2) * 16, true);
    bDsTok = new_buf(64 * 4, true);
    bDsProb = new_buf(64 * 4, true);
    bDsRopeD = new_buf(T * hp.DH * 4, true);
    for (int i = 0; i < FR_SLOTS; ++i) frnnPos[i] = -1;
}

// remember the target's hidden state of N tokens at a tap point
void EngineImpl::ds_tap(int v, int N, B cur) {
    for (size_t i = 0; i < dsTapVisit.size(); ++i) {
        if (dsTapVisit[i] != v) continue;
        CountArgs a{(uint32_t) (N * hp.D), 1, 0, 0};
        threads("d_copy", a, {cur, B(bTaps, i * (size_t) NB_ * hp.D * 4)}, (size_t) N * hp.D, 1);
    }
}

void EngineImpl::ds_mv(int N, const DsMat & m, uint32_t rows, uint32_t K, B x, uint32_t ldx, B y, uint32_t ldy) {
    mmv(m.q8 ? "m_mv_q8_h" : "m_mv_h_h", N, B(m.w), rows, K, x, ldx, y, ldy);
}

static void ds_fill_rope(float * r, int pos0, int n, int dh, float base) {
    const int half = dh / 2;
    for (int t = 0; t < n; ++t)
        for (int i = 0; i < half; ++i) {
            const double th = (double) (pos0 + t) * pow((double) base, -(double) i / half);
            r[(size_t) t * dh + 2 * i] = (float) cos(th);
            r[(size_t) t * dh + 2 * i + 1] = (float) sin(th);
        }
}

// Inject the context K / V of N positions (taps already recorded) into the draft model's caches, and keep
// the Markov head's context state (f_rnn) of the last positions.
void EngineImpl::ds_inject(int N, int pos0) {
    const uint32_t D = hp.D, DKV = hp.NKV * hp.DH;
    const uint32_t tapStride = (uint32_t) NB_ * D;
    ds_fill_rope((float *) bDsRope.contents, pos0, N, hp.DH, dsRopeBase);
    for (size_t il = 0; il < dsL.size(); ++il) {
        const DsLayer & L = dsL[il];
        DsMixArgs ma{(uint32_t) N, tapStride, 0, dsEps, L.mix[0], L.mix[1], L.mix[2], L.mix[3]};
        groups("ds_mix_norm", ma, {B(bTaps), B(L.small, 2 * D * 4), B(bDsX), B(bDsX)}, N, 1, 768);
        if (N > MINI_MAX) {
            gemm("k_gemm_hh_f", B(bDsX), B(L.wkc), B(bDsP), N, DKV, D, D, 2 * DKV);
            gemm("k_gemm_hh_f", B(bDsX), B(L.wvc), B(bDsP, DKV * 4), N, DKV, D, D, 2 * DKV);
        } else {
            mmv("m_mv_h_h", N, B(L.wkc), DKV, D, B(bDsX), D, B(bDsP), 2 * DKV);
            mmv("m_mv_h_h", N, B(L.wvc), DKV, D, B(bDsX), D, B(bDsP, DKV * 4), 2 * DKV);
        }
        DsPrepArgs pa{(uint32_t) N, 2 * DKV, 0, 0, DKV, (uint32_t) hp.NH, (uint32_t) pos0, 1.0f, dsEps, 0, 0, 0};
        groups("ds_prep", pa, {B(bDsP), B(L.small), B(L.small, (3 * D + 2 * hp.DH) * 4), B(bDsRope), B(bDsX), B(dsK[il]), B(dsV[il])},
               2 * hp.NKV, N, 32);
    }
    // f_rnn of the last (up to MINI_MAX) positions of this batch
    const int nf = std::min(N, MINI_MAX), n0 = N - nf;
    if (frnnCur + nf > FR_SLOTS) frnnCur = 0;
    DsMixArgs fa{(uint32_t) nf, tapStride, 1, dsEps, dsMMix[0], dsMMix[1], dsMMix[2], dsMMix[3]};
    groups("ds_mix_norm", fa, {B(bTaps, (size_t) n0 * D * 4), B(dsSmall, dsoMCN * 4), B(bDsX), B(bDsFrnn, (size_t) frnnCur * D * 4)}, nf, 1, 768);
    for (int i = 0; i < nf; ++i) {
        for (int k = 0; k < FR_SLOTS; ++k) if (frnnPos[k] == pos0 + n0 + i) frnnPos[k] = -1;
        frnnPos[frnnCur + i] = pos0 + n0 + i;
    }
    frnnCur += nf;
}

// Draft up to kmax tokens that follow `token` (about to be appended at position pos).  Returns the number
// drafted, 0 if the Markov context of position pos - 1 is not available.
int EngineImpl::ds_draft(int32_t token, int pos, int kmax, int32_t * out, float * prob) {
    kmax = std::min(kmax, dsBlock - 1);
    const int fs = ds_frnn_slot(pos - 1);
    if (kmax <= 0 || fs < 0 || pos < 1) return 0;
    const uint32_t D = hp.D, DQ = hp.NH * hp.DH, DKV = hp.NKV * hp.DH, DFF = 4 * D, PWd = DQ + 2 * DKV;
    const int T = kmax + 1;
    ensure_kv(pos + T);
    ds_fill_rope((float *) bDsRopeD.contents, pos, T, hp.DH, dsRopeBase);
    memset(bDsState.contents, 0, 256 * 4);
    ((uint32_t *) bDsTok.contents)[0] = (uint32_t) token;

    DsEmbedArgs ea{(uint32_t) token, dsEps, 0, 0};
    groups("ds_embed", ea, {B(bEmb), B(dsSmall, dsoTokN * 4), B(dsSmall, dsoMask * 4), B(bDsCur)}, T, 1, 768);
    auto norm = [&](B w, bool wantF) {
        MixArgs a{};
        a.N = T; a.mode = 0; a.nq = 1; a.eps = dsEps; a.wantF32 = wantF ? 1 : 0;
        groups("m_mix_norm", a, {B(bDsCur), B(bDsCur), B(bDsCur), w, B(bDsXh), B(bDsH)}, T, 1, 768);
    };
    auto add = [&](B x) {
        CountArgs a{(uint32_t) (T * D), 1, 0, 0};
        threads("m_add", a, {x, B(bDsCur)}, (size_t) T * D, 1);
    };
    for (size_t il = 0; il < dsL.size(); ++il) {
        const DsLayer & L = dsL[il];
        norm(B(L.small), false);
        ds_mv(T, L.wq, DQ, D, B(bDsXh), D, B(bDsQP), PWd);
        ds_mv(T, L.wk, DKV, D, B(bDsXh), D, B(bDsQP, DQ * 4), PWd);
        ds_mv(T, L.wv, DKV, D, B(bDsXh), D, B(bDsQP, (DQ + DKV) * 4), PWd);
        DsPrepArgs pa{(uint32_t) T, PWd, 0, DQ, DQ + DKV, 0, 0, 1.0f / sqrtf((float) hp.DH), dsEps, 0, 0, 0};
        groups("ds_prep", pa, {B(bDsQP), B(L.small, 3 * D * 4), B(L.small, (3 * D + hp.DH) * 4), B(bDsRopeD), B(bDsQh), B(bDsBK), B(bDsBV)},
               hp.NH + 2 * hp.NKV, T, 32);
        // attention: the context cache (every row sees positions < pos, within the window on windowed layers),
        // then the block itself (non-causal) as one more chunk
        MAttnArgs ma{(uint32_t) T, (uint32_t) (pos - 1), 0, 0, 0, 0, 0, 0, 0, 0xffffffffu, 0, 3, 0, 0, 0, 0};
        if (dsSwa[il]) {
            const uint32_t w = (uint32_t) (dsWindow - 1);       // keys p1 with pos - p1 < dsWindow
            ma.window = w;
            const uint32_t lo = (uint32_t) pos > w ? (uint32_t) pos - w : 0;
            ma.base0 = (lo / attnC) * attnC;
            ma.chunkKeys = (uint32_t) attnC;
        } else {
            ma.chunkKeys = MINI_CHUNK;
        }
        const uint32_t nCtx = ((uint32_t) pos - ma.base0 + ma.chunkKeys - 1) / ma.chunkKeys;
        ma.nChunks = nCtx + 1;
        groups("m_attn_chunk", ma, {B(bDsQh), B(dsK[il]), B(dsV[il]), B(bRc), B(dsSmall), B(bMAttn)}, nCtx, hp.NKV, 256);
        MAttnArgs mb = ma;
        mb.pos0 = (uint32_t) (T - 1); mb.window = 0; mb.base0 = 0; mb.chunkKeys = (uint32_t) attnC; mb.chunk0 = nCtx;
        groups("m_attn_chunk", mb, {B(bDsQh), B(bDsBK), B(bDsBV), B(bRc), B(dsSmall), B(bMAttn)}, 1, hp.NKV, 256);
        groups("m_attn_merge", ma, {B(bMAttn), B(bDsQP), B(bDsO)}, hp.NH, T, 32);
        ds_mv(T, L.wo, D, DQ, B(bDsO), DQ, B(bDsAO), D);
        add(B(bDsAO));
        norm(B(L.small, D * 4), false);
        ds_mv(T, L.wg, DFF, D, B(bDsXh), D, B(bDsG), DFF);
        ds_mv(T, L.wu, DFF, D, B(bDsXh), D, B(bDsU), DFF);
        CountArgs ca{(uint32_t) (T * DFF), 1, 0, 0};
        threads("ds_act", ca, {B(bDsG), B(bDsU), B(bDsAct)}, (size_t) T * DFF, 1);
        ds_mv(T, L.wd, D, DFF, B(bDsAct), DFF, B(bDsAO), D);
        add(B(bDsAO));
    }
    norm(B(dsSmall, dsoOutN * 4), true);
    if (bOutQ8) mmv("m_mv_q8_h", T, B(bOutQ8), (uint32_t) hp.vocab, D, B(bDsXh), D, B(bDsBase), (uint32_t) hp.vocabPadded);
    else mmv("m_mv_h_h", T, B(bOut), (uint32_t) hp.vocab, D, B(bDsXh), D, B(bDsBase), (uint32_t) hp.vocabPadded);
    // Markov head: one block position after the other, each conditioned on the token just chosen
    const uint32_t nch = ((uint32_t) hp.vocab + 1023) / 1024;
    for (int i = 1; i <= kmax; ++i) {
        DsStepArgs sa{(uint32_t) i, (uint32_t) hp.vocab, D, dsCap};
        threads("ds_joint_in", sa, {B(bDsState), B(dsW1), B(bDsTok), B(bDsH), B(bDsFrnn, (size_t) fs * D * 4), B(bDsU5)}, 2 * 256 + 2 * D, 1);
        mmv("m_mv_h_f", 1, B(dsJoint), 3 * 256, 2 * 256 + 2 * D, B(bDsU5), 2 * 256 + 2 * D, B(bDsJ), 3 * 256);
        threads("ds_gate", sa, {B(bDsJ), B(dsSmall, dsoJB * 4), B(bDsState), B(bDsMarkov)}, 256, 1);
        mmv("m_mv_h_f", 1, B(dsW2), (uint32_t) hp.vocab, 256, B(bDsMarkov), 256, B(bDsBias), (uint32_t) hp.vocabPadded);
        DsStepArgs sc{(uint32_t) i, (uint32_t) hp.vocab, (uint32_t) hp.vocabPadded, dsCap};
        groups("ds_col", sc, {B(bDsBase), B(bDsBias), B(dsSmall, dsoW2B * 4), B(bDsCol), B(bDsArg)}, nch, 1, 32);
        groups("ds_pick", sc, {B(bDsArg), B(bDsTok), B(bDsProb)}, 1, 1, 32);
    }
    flush(true);
    const uint32_t * tk = (const uint32_t *) bDsTok.contents;
    const float * pr = (const float *) bDsProb.contents;
    for (int i = 0; i < kmax; ++i) { out[i] = (int32_t) tk[i + 1]; prob[i] = pr[i + 1]; }
    ++statDraftCalls;
    return kmax;
}

// End a multi-token pass: keep the conv state of the tokens that were accepted.
void EngineImpl::settle() {
    if (convPending) {
        flush(true);
        const size_t bytes = visits.size() * (hp.NCONV - 1) * CH * 4;
        memcpy(convSt[convCur].contents, (const char *) bConvVar.contents + (size_t) specRow * bytes, bytes);
        convPending = false;
    }
    specN = 0;
}

static inline uint64_t ng_hash(const int32_t * t, int n) {
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < n; ++i) { h ^= (uint32_t) t[i]; h *= 1099511628211ull; h ^= h >> 29; }
    return h;
}

// index every NGRAM-token window of the sequence by its end position (the last occurrence wins)
void EngineImpl::ng_sync() {
    const int NG = 3;
    const int n = (int) seq.size();
    if (n < ngIndexed) { ngLast.clear(); ngIndexed = 0; }
    for (int e = std::max(ngIndexed, NG); e <= n; ++e) {
        if (e > ngIndexed) ngLast[ng_hash(seq.data() + e - NG, NG)] = e;
    }
    ngIndexed = n;
}

// Drafts for the tokens after `token` (about to be appended): what followed the most recent earlier
// occurrence of the current ending, if at least ngMin tokens match.  Returns the number of drafts.
int EngineImpl::ng_draft(int32_t token, int32_t * out, int maxn) {
    const int NG = 3;
    const int n = (int) seq.size();
    if (maxn <= 0 || n + 1 < ngMin) return 0;
    ng_sync();
    int32_t tail[NG];
    for (int i = 0; i < NG - 1; ++i) tail[i] = seq[n - (NG - 1) + i];
    tail[NG - 1] = token;
    auto it = ngLast.find(ng_hash(tail, NG));
    if (it == ngLast.end()) return 0;
    const int e = it->second;   // tokens [e - NG, e) are the candidate match; drafts start at e
    if (e > n || e < NG) return 0;
    // verify and measure the match length backwards (seq[e - 1] must equal token)
    int len = 0;
    while (len < 32 && e - 1 - len >= 0) {
        const int32_t a = seq[e - 1 - len];
        const int32_t b = len == 0 ? token : seq[n - len];
        if (a != b) break;
        ++len;
    }
    if (len < ngMin) return 0;
    int k = 0;
    while (k < maxn && e + k < n) { out[k] = seq[e + k]; ++k; }
    return k;
}

void EngineImpl::forward_batch(const int32_t * toks, int N, int pos0, bool wantLogits) {
    const size_t D = hp.D, DQ = hp.NH * hp.DH, DKV = hp.NKV * hp.DH;
    const size_t ND4 = (size_t) N * D * 4;
    const size_t nbStride = (size_t) NB_ * D;
    ensure_kv(pos0 + N);
    memcpy(bTok.contents, toks, (size_t) N * 4);

    if (profOn) { flush(true); profT = now_s(); }
    EmbedArgs ea{(uint32_t) N, hp.eps};
    groups("k_embed", ea, {B(bTok), B(bEmb), B(bGlobalSmall, oTokNorm * 4), B(bCur)}, N, 1, 32);

    std::vector<uint32_t> blocks;   // NB slots of the mixed block inputs
    const uint32_t bankSlot = 4;
    id<MTLBuffer> stIn = convSt[convCur], stOut = convSt[1 - convCur];
    const float invsStatic = 1.0f / sqrtf((float) hp.NCONV + 1.0f), invsDyn = 1.0f / sqrtf(2.0f * hp.NCONV + 1.0f);
    const float qscale = 1.0f / sqrtf((float) D);

    auto mix = [&](B curB, const std::vector<uint32_t> & slots, B queries, int nq, int mode, B normw, uint32_t wStride,
                   B outH, B outF, bool wantF32) {
        MixArgs a{};
        a.N = N; a.nStates = (uint32_t) slots.size(); a.nq = nq; a.mode = mode;
        a.slot0 = slots.size() > 0 ? slots[0] : 0; a.slot1 = slots.size() > 1 ? slots[1] : 0;
        a.slot2 = slots.size() > 2 ? slots[2] : 0; a.slot3 = slots.size() > 3 ? slots[3] : 0;
        a.nbStride = (uint32_t) nbStride; a.outStride = (uint32_t) nbStride; a.wantF32 = wantF32;
        a.eps = hp.eps; a.mixEps = 1e-6f; a.qscale = qscale; a.wStride = wStride;
        groups("k_mix_norm", a, {curB, B(bNB), queries, normw, outH, outF}, N, 1, 32);
    };

    for (size_t v = 0; v < visits.size(); ++v) {
        const int il = visits[v].layer, pass = visits[v].pass;
        const LayerW & L = layers[il];
        const bool looped = L.loop && pass > 0;
        const size_t stOff = v * (hp.NCONV - 1) * CH * 4;

        if (L.loop && il == hp.loopStart) {
            if (pass == 0) {
                blit(bCur, 0, bPrelude, 0, ND4);
            } else {
                // bank of the previous pass: its second core block (slot 2) and its output
                blit(bNB, 2 * nbStride * 4, bNB, bankSlot * nbStride * 4, ND4);
                blit(bCur, 0, bBankOut, 0, ND4);
                blit(bPrelude, 0, bCur, 0, ND4);
                blocks.assign(1, 0u);
            }
        }
        if (dsOn) ds_tap((int) v, N, B(bCur));
        if (looped) {
            mix(B(bBankOut), {bankSlot}, B(L.small, L.oMsgQ * 4), 4, 2, B(L.small), 0, B(bXin), B(bMsgs), false);
        }

        // ---- attention inputs
        bool sameInputs = true;
        if (!L.hasAttnResQ) {
            mix(B(bCur), {}, B(L.small), 1, 0, B(L.small, L.oNormQ * 4), 0, B(bXin), B(bXff32), false);
        } else if (!looped) {
            const int nq = L.swa ? 1 : 3;
            mix(B(bCur), blocks, B(L.small, L.oAttnResQ * 4), nq, 1, B(L.small, L.oNormQ * 4), L.swa ? 0 : (uint32_t) D,
                B(bXin), B(bXff32), false);
            sameInputs = L.swa;
        } else {
            const int nq = L.swa ? 1 : 3;
            mix(B(bCur), blocks, B(L.small, L.oAttnResQ * 4), nq, 2, B(L.small), 0, B(bXin), B(bMixed), false);
            const size_t normOff[3] = {L.oNormQ, L.oNormK, L.oNormV};
            for (int j = 0; j < 3; ++j) {
                control(N, L, j, B(bMixed, (L.swa ? 0 : j) * nbStride * 4), B(bMsgs, j * nbStride * 4), normOff[j],
                        B(bXin, j * nbStride * 2), false);
            }
            sameInputs = false;
        }

        prof("mix/ctrl");
        // ---- projections
        if (sameInputs) {
            gemm("k_gemm_hh_f", B(bXin), B(L.attn), B(bP), N, PW, (uint32_t) D, (uint32_t) D, PW);
        } else {
            const uint32_t dyn = L.swa ? 0 : hp.NKV * hp.NCONV;
            gemm("k_gemm_hh_f", B(bXin), B(L.attn), B(bP), N, kCol, (uint32_t) D, (uint32_t) D, PW);
            gemm("k_gemm_hh_f", B(bXin, nbStride * 2), B(L.attn, (size_t) kCol * D * 2), B(bP, kCol * 4), N,
                 (uint32_t) DKV + dyn, (uint32_t) D, (uint32_t) D, PW);
            gemm("k_gemm_hh_f", B(bXin, 2 * nbStride * 2), B(L.attn, (size_t) vCol * D * 2), B(bP, vCol * 4), N,
                 (uint32_t) DKV + dyn, (uint32_t) D, (uint32_t) D, PW);
        }

        prof("gemm_attn");
        // ---- Q / R / K / V post-processing, caches, conv state
        PrepQArgs pq{(uint32_t) N, (uint32_t) pos0, PW, qCol, rCol, hp.qkClip, hp.relClip, hp.logAlpha, hp.logFloor, L.swa ? 0u : 1u,
                     1.0f / sqrtf((float) hp.DH), 0, 0, 0, 0, 0};
        groups("k_prep_q", pq, {B(bP), B(L.small, L.oQClip * 4), B(L.small, L.oRClip * 4), B(bQh), B(bRc)}, N, 1, 32);
        const uint32_t slotMask = L.swa ? swaCap - 1 : 0xffffffffu;
        PrepKVArgs pk{};
        pk.N = N; pk.pos0 = pos0; pk.PW = PW; pk.col = kCol; pk.dynCol = kDynCol; pk.useDyn = L.swa ? 0 : 1; pk.goff = 0; pk.isV = 0;
        pk.thr = hp.qkClip; pk.eps = hp.eps; pk.invs = L.swa ? invsStatic : invsDyn; pk.slotMask = slotMask; pk.stStride = CH;
        groups("k_prep_kv", pk, {B(bP), B(L.small, L.oKConv * 4), B(stIn, stOff), B(kvK[v])}, N, 1, 32);
        PrepKVArgs pv = pk;
        pv.col = vCol; pv.dynCol = vDynCol; pv.goff = (uint32_t) DKV; pv.isV = 1;
        groups("k_prep_kv", pv, {B(bP), B(L.small, L.oVConv * 4), B(stIn, stOff), B(kvV[v])}, N, 1, 32);
        if (opt.kv8 && !L.swa) {
            KV8Args ka{(uint32_t) pos0, (uint32_t) N, 0, 0};
            groups("k_kv_q8", ka, {B(kvK[v]), B(kvK8[v])}, hp.NKV, N, 32);
            groups("k_kv_q8", ka, {B(kvV[v]), B(kvV8[v])}, hp.NKV, N, 32);
        }
        // new conv state; a multi-token pass stores the state after every prefix instead (see settle())
        auto conv_state = [&](ConvArgs cv, B X, size_t C) {
            if (!miniMode) { threads("k_conv_state", cv, {X, B(stIn, stOff), B(stOut, stOff)}, C, hp.NCONV - 1); return; }
            cv.mode = (uint32_t) (visits.size() * (hp.NCONV - 1) * CH);
            id<MTLComputePipelineState> ps;
            enc_setup("k_conv_state_var", cv, {X, B(stIn, stOff), B(bConvVar, stOff)}, &ps);
            [enc dispatchThreads:MTLSizeMake(C, hp.NCONV - 1, N) threadsPerThreadgroup:MTLSizeMake(std::min<size_t>(ps.threadExecutionWidth, C), 1, 1)];
        };
        ConvArgs cs{(uint32_t) N, (uint32_t) DKV, 0, 0, 0, PW, kCol, CH};
        conv_state(cs, B(bP), DKV);
        cs.goff = (uint32_t) DKV; cs.xcol = vCol;
        conv_state(cs, B(bP), DKV);

        prof("prep_qkv");
        // ---- relative position table and attention
        gemm("k_gemm_ff_f", B(bRc), B(L.small, L.oRelBias * 4), B(bT), N * hp.NH, hp.extent, hp.NREL, hp.NREL, hp.extent);
        prof("rel_table");
        AttnArgs aa{(uint32_t) N, (uint32_t) pos0, L.swa ? 1u : 0u, slotMask, (uint32_t) hp.window, (uint32_t) hp.extent,
                    1.0f / sqrtf((float) hp.DH), PW, sinkCol, gateCol, 0, 0};
        if (miniMode && !L.swa) {
            // long linear cache: one read for all tokens of the pass
            const int nCh = (pos0 + N + MINI_CHUNK - 1) / MINI_CHUNK;
            MAttnArgs ma{(uint32_t) N, (uint32_t) pos0, (uint32_t) nCh, MINI_CHUNK, (uint32_t) hp.extent, PW, sinkCol, gateCol, 0, 0xffffffffu, 0, 0, 0, 0, 0, 0};
            groups("m_attn_chunk", ma, {B(bQh), B(kvK[v]), B(kvV[v]), B(bRc), B(L.small, L.oRelBias * 4), B(bMAttn)}, nCh, hp.NKV, 256);
            groups("m_attn_merge", ma, {B(bMAttn), B(bP), B(bO)}, hp.NH, N, 32);
        } else {
            groups("k_attn_mpp", aa, {B(bQh), B(kvK[v]), B(kvV[v]), B(bT), B(bP), B(bO)}, (N + attnQ - 1) / attnQ, hp.NH, 256);
        }
        prof(L.swa ? "attn_swa" : "attn_full");
        gemm("k_gemm_hh_f", B(bO), B(L.wo), B(bAO), N, (uint32_t) D, (uint32_t) DQ, (uint32_t) DQ, (uint32_t) D);
        prof("gemm_wo");

        // ---- residual; the first layer of a block starts a new stream and stores the old one
        const bool blockStart = il % hp.blockSize == 0;
        if (blockStart) {
            const uint32_t slot = (uint32_t) (il / hp.blockSize);
            NormArgs na{(uint32_t) N, 1e-6f};
            groups("k_norm_store", na, {B(bCur), B(bNB, slot * nbStride * 4)}, N, 1, 32);
            blocks.push_back(slot);
        }
        ConvArgs ca{(uint32_t) N, (uint32_t) D, (uint32_t) (2 * DKV), blockStart ? 0u : 1u, invsStatic, (uint32_t) D, 0, CH};
        threads("k_conv_residual", ca, {B(bAO), B(L.small, L.oAConv * 4), B(stIn, stOff), B(bCur)}, D / 4, N);
        conv_state(ca, B(bAO), D);

        prof("post_attn");
        // ---- MoE feed-forward
        if (!looped) {
            mix(B(bCur), blocks, B(L.small, L.oFfnResQ * 4), 1, 1, B(L.small, L.oFfnNorm * 4), 0, B(bXin), B(bXff32), true);
        } else {
            mix(B(bCur), blocks, B(L.small, L.oFfnResQ * 4), 1, 2, B(L.small), 0, B(bXin), B(bMixed), false);
            control(N, L, 3, B(bMixed), B(bMsgs, 3 * nbStride * 4), L.oFfnNorm, B(bXin), true);
        }
        prof("mix/ctrl");
        if (miniMode) moe_mini(N, L);
        else moe(N, L);
        ConvArgs cf{(uint32_t) N, (uint32_t) D, (uint32_t) (2 * DKV + D), 1, invsStatic, (uint32_t) D, 0, CH};
        threads("k_conv_residual", cf, {B(bY), B(L.small, L.oFConv * 4), B(stIn, stOff), B(bCur)}, D / 4, N);
        conv_state(cf, B(bY), D);
        prof("post_ffn");
    }

    if (miniMode) {
        // logits of every row
        mix(B(bCur), blocks, B(bGlobalSmall, oOutResQ * 4), 1, 1, B(bGlobalSmall, oOutNorm * 4), 0, B(bXin), B(bXff32), false);
        gemm("k_gemm_hh_f", B(bXin), B(bOut), B(bLogitsM), N, (uint32_t) hp.vocab, (uint32_t) D, (uint32_t) D, (uint32_t) hp.vocabPadded);
        if (hp.softcap > 0) {
            CountArgs sc{(uint32_t) (N * hp.vocabPadded), 1, hp.softcap, 0};
            threads("k_softcap", sc, {B(bLogitsM)}, (size_t) N * hp.vocabPadded, 1);
        }
        flush(true);
        prof("logits");
        return;   // the conv state stays with the caller's choice of prefix
    }
    if (wantLogits || dsOn) {
        mix(B(bCur), blocks, B(bGlobalSmall, oOutResQ * 4), 1, 1, B(bGlobalSmall, oOutNorm * 4), 0, B(bXin), B(bXff32), dsOn);
        if (dsOn) {
            ds_tap((int) visits.size(), N, B(bXff32));
            ds_inject(N, pos0);
        }
    }
    if (wantLogits) {
        gemm("k_gemm_hh_f", B(bXin, (size_t) (N - 1) * D * 2), B(bOut), B(bLogits), 1, (uint32_t) hp.vocab, (uint32_t) D, (uint32_t) D,
             (uint32_t) hp.vocabPadded);
        if (hp.softcap > 0) {
            CountArgs sc{(uint32_t) hp.vocab, 1, hp.softcap, 0};
            threads("k_softcap", sc, {B(bLogits)}, hp.vocab, 1);
        }
    }
    flush(true);
    prof("logits");
    convCur = 1 - convCur;
}

void EngineImpl::prof_report(const char * what, int n) {
    if (!profOn) return;
    double tot = 0;
    for (auto & kv : profAcc) tot += kv.second;
    fprintf(stderr, "[prof] %s: %d tokens, %.1f ms\n", what, n, tot * 1e3);
    std::vector<std::pair<double, std::string>> v;
    for (auto & kv : profAcc) v.push_back({kv.second, kv.first});
    std::sort(v.rbegin(), v.rend());
    for (auto & e : v) fprintf(stderr, "[prof]   %-14s %8.2f ms  %5.1f%%\n", e.second.c_str(), e.first * 1e3, 100 * e.first / tot);
    profAcc.clear();
}

// ----------------------------------------------------------------------------------------------------
bool EngineImpl::set_prompt(const std::vector<int32_t> & tokens, const std::function<bool(int, int)> & progress) {
    if (tokens.empty()) throw std::runtime_error("engine: empty prompt");
    if ((int) tokens.size() > opt.max_ctx) throw std::runtime_error("engine: prompt exceeds the context limit");
    @autoreleasepool {
    touch_use();
    const double t0 = now_s();
    size_t common = 0;
    while (common < seq.size() && common < tokens.size() && seq[common] == tokens[common]) ++common;
    flush(true);
    settle();
    tm = EngineTimings{};
    // the same prompt again (regenerate): the prompt-end checkpoint carries its logits
    if (common == tokens.size() && restore_checkpoint((int) common, true)) {
        tm.prompt_cached = (int) common;
        tm.prompt_ms = (now_s() - t0) * 1e3;
        return true;
    }
    // otherwise at least the last prompt token must be evaluated to produce logits
    const size_t keep = std::min(common, tokens.size() - 1);
    if (keep < seq.size()) {
        const double tr = now_s();
        if (keep == 0 || !restore_checkpoint((int) keep)) {
            reset();
        }
        if (opt.verbose) fprintf(stderr, "engine: rollback %zu -> %zu tokens (common prefix %zu) in %.1f ms\n",
                                 (size_t) hiWater, seq.size(), common, (now_s() - tr) * 1e3);
    }
    const size_t start = seq.size();
    const int total = (int) (tokens.size() - start);
    tm.prompt_cached = (int) start;
    int done = 0;
    while (done < total) {
        const int left = total - done;
        const bool viaDecode = left <= 8;
        const int n = viaDecode ? 1 : std::min(NB_, left);
        const bool last = done + n == total;
        const double tb = now_s();
        if (viaDecode) decode_token(tokens[start + done], (int) seq.size());
        else forward_batch(tokens.data() + start + done, n, (int) seq.size(), last);
        seq.insert(seq.end(), tokens.begin() + start + done, tokens.begin() + start + done + n);
        hiWater = std::max(hiWater, (int) seq.size());
        done += n;
        prof_report(viaDecode ? "decode (prompt tail)" : "prefill batch", n);
        const double tc = now_s();
        if (last) {
            save_checkpoint(true, true);
        } else if (!viaDecode) {
            save_checkpoint((int) seq.size() - lastFullPos >= 8192, false);
        }
        if (opt.verbose) fprintf(stderr, "engine: %s %d tokens in %.1f ms, checkpoint %.1f ms\n", viaDecode ? "decoded" : "batch of", n,
                                 (tc - tb) * 1e3, (now_s() - tc) * 1e3);
        tm.prompt_tokens = done;
        tm.prompt_ms = (now_s() - t0) * 1e3;
        if (!last && progress && !progress(done, total)) return false;
    }
    tm.prompt_ms = (now_s() - t0) * 1e3;
    return true;
    }
}

void EngineImpl::append(int32_t token) {
    if ((int) seq.size() >= opt.max_ctx) throw std::runtime_error("engine: context is full");
    touch_use();
    const int pos = (int) seq.size();
    // a verified draft: its logits are already there
    if (specN && pos == specPos + specRow + 1 && specRow + 1 < specN && specTok[specRow + 1] == token) {
        ++specRow;
        ++statAccepted;
        seq.push_back(token);
        return;
    }
    settle();
    @autoreleasepool {
        int32_t toks[MINI_MAX];
        toks[0] = token;
        int n = 1;
        const int room = std::min({opt.draft, MINI_MAX - 1, opt.max_ctx - pos - 1});
        if (dsOn && room > 0) {
            // DSpark drafts, kept while the block is likely to be accepted as a whole
            float pr[MINI_MAX];
            const int k = ds_draft(token, pos, room, toks + 1, pr);
            float joint = 1.0f;
            for (int i = 0; i < k; ++i) {
                joint *= pr[i];
                if (joint < draftMin) break;
                ++n;
            }
            prof("draft");
        } else if (opt.draft > 0) {
            n += ng_draft(token, toks + 1, room);
        }
        // INK_FORCE_MINI=n: timing aid, always run an n-token pass (drafts copied from earlier text)
        static const int forceMini = getenv("INK_FORCE_MINI") ? atoi(getenv("INK_FORCE_MINI")) : 0;
        if (forceMini > 0 && opt.draft > 0 && pos > 200) { n = forceMini; for (int i = 1; i < n; ++i) toks[i] = seq[pos - 150 + i]; }
        if (n > 1) {
            forward_mini(toks, n, pos);
            specPos = pos; specN = n; specRow = 0;
            memcpy(specTok, toks, sizeof(int32_t) * n);
            convPending = true;
            ++statPasses;
            statDrafted += n - 1;
            hiWater = std::max(hiWater, pos + n);
            prof_report("multi-token pass", n);
        } else {
            if (useBatchDecode) forward_batch(&token, 1, pos, true);
            else decode_token(token, pos);
            ++statSingles;
            prof_report("decode", 1);
        }
    }
    seq.push_back(token);
    hiWater = std::max(hiWater, (int) seq.size());
}

EngineImpl::EngineImpl(const GGUF & gguf, const EngineOptions & o) : opt(o) {
    profOn = getenv("INK_PROF") != nullptr;
    if (getenv("INK_NG_MIN")) ngMin = atoi(getenv("INK_NG_MIN"));
    if (getenv("INK_DRAFT_MIN")) draftMin = (float) atof(getenv("INK_DRAFT_MIN"));
    useBatchDecode = getenv("INK_BATCH_DECODE") != nullptr;
    dev = MTLCreateSystemDefaultDevice();
    if (!dev) throw std::runtime_error("engine: no Metal device (this server needs an Apple silicon Mac)");
    queue = [dev newCommandQueue];
    {
        MTLResidencySetDescriptor * d = [MTLResidencySetDescriptor new];
        d.label = @"inkling";
        d.initialCapacity = 512;
        NSError * err = nil;
        rset = [dev newResidencySetWithDescriptor:d error:&err];
        if (rset) [queue addResidencySet:rset];
        else if (opt.verbose) fprintf(stderr, "engine: no residency set (%s)\n", err.localizedDescription.UTF8String);
    }
    load_hparams(gguf);
    build_library();
    load_weights(gguf);
    if (opt.draft > 0 && !opt.draft_model.empty()) ds_load(opt.draft_model);
    alloc_state();
    // warm up: build every pipeline and fault in the scratch buffers so the first request is not slow
    {
        const double t0 = now_s();
        std::vector<int32_t> warm(96);
        for (size_t i = 0; i < warm.size(); ++i) warm[i] = (int32_t) (1000 + 37 * i);
        @autoreleasepool {
            forward_batch(warm.data(), (int) warm.size(), 0, true);
            seq = warm;
            for (int i = 0; i < 3; ++i) { decode_token(warm[i], (int) seq.size()); seq.push_back(warm[i]); }
        }
        reset();
        if (opt.verbose) fprintf(stderr, "engine: warm-up %.2fs\n", now_s() - t0);
    }
    touch_use();
    if (rset) {
        // warm-keeping workload: a small GEMM on its own queue and buffers
        id<MTLCommandQueue> hbQueue = [dev newCommandQueue];
        id<MTLComputePipelineState> hbPipe = pipe("k_gemm_hh_f");
        static const uint32_t hn = 256, hm = 512, hk = 2304;
        id<MTLBuffer> hbB = [dev newBufferWithLength:hn * hk * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> hbA = [dev newBufferWithLength:hm * hk * 2 options:MTLResourceStorageModeShared];
        id<MTLBuffer> hbC = [dev newBufferWithLength:hn * hm * 4 options:MTLResourceStorageModeShared];
        heartbeat = std::thread([this, hbQueue, hbPipe, hbB, hbA, hbC] {
            int tick = 0;
            while (!hbStop.load()) {
                const double idle = now_s() - lastUse.load();
                const bool active = opt.keep_wired_s <= 0 || idle < (double) opt.keep_wired_s;
                if (active && tick % 5 == 0) {
                    std::lock_guard<std::mutex> lk(rsetMu);
                    [rset requestResidency];
                }
                // after idling the GPU drops to a low power state and the next batch runs ~2x slower for a
                // few hundred milliseconds; a tiny periodic workload prevents that
                if (idle > 0.05 && idle < (double) opt.keep_warm_s) {
                    @autoreleasepool {
                        id<MTLCommandBuffer> c = [hbQueue commandBuffer];
                        id<MTLComputeCommandEncoder> e = [c computeCommandEncoder];
                        [e setComputePipelineState:hbPipe];
                        GemmArgs a{hn, hm, hk, hk, hm, 0, 0, 0};
                        [e setBytes:&a length:sizeof(a) atIndex:0];
                        [e setBuffer:hbB offset:0 atIndex:1];
                        [e setBuffer:hbA offset:0 atIndex:2];
                        [e setBuffer:hbC offset:0 atIndex:3];
                        [e dispatchThreadgroups:MTLSizeMake(hm / 64, hn / 64, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
                        [e endEncoding];
                        [c commit];
                    }
                }
                ++tick;
                usleep(100 * 1000);
            }
        });
    }
}

std::unique_ptr<Engine> Engine::create(const GGUF & gguf, const EngineOptions & opt) {
    @autoreleasepool {
        return std::make_unique<EngineImpl>(gguf, opt);
    }
}

}  // namespace ink
