// Metal kernels for the Inkling-2-Mini engine.
// Model dimensions are injected as preprocessor macros by the host (see engine.mm):
//   D (2304)  NH (24)  NKV (8)  DH (128)  NREL (16)  NEXP (128)  NUSED (8)  FF (1152)  NCONV (4)
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

#define SW 32                      // threads per SIMD group
#define DKV (NKV * DH)             // K / V width per position (1024)
#define DQ (NH * DH)               // Q width (3072)
#define GQA (NH / NKV)             // query heads per KV head

// ------------------------------------------------------------------------------------------------
// load-time conversion
// ------------------------------------------------------------------------------------------------

struct CountArgs { uint w; uint h; float f; uint pad; };

// BF16 -> FP16, in place (grid w x h covers the buffer)
kernel void k_bf16_to_f16(constant CountArgs & a, device ushort * buf, uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.w || gid.y >= a.h) return;
    const ulong i = (ulong) gid.y * a.w + gid.x;
    const float f = as_type<float>(uint(buf[i]) << 16);
    buf[i] = as_type<ushort>(half(f));
}

constant float kFP4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

struct NvArgs { uint rows; uint K; uint rowBytes; uint pad; };

// packed NVFP4 rows ([K/16 scale bytes][K/2 nibble bytes]) -> FP16 rows.  One thread per 16-weight sub-block.
kernel void k_nvfp4_to_f16(constant NvArgs & a, device const uchar * src, device const float * stab, device half * dst,
                           uint2 gid [[thread_position_in_grid]]) {
    const uint nsub = a.K / 16;
    if (gid.x >= nsub || gid.y >= a.rows) return;
    device const uchar * row = src + (ulong) gid.y * a.rowBytes;
    const float d = stab[row[gid.x]];
    device const uchar * q = row + nsub + gid.x * 8;
    device half * o = dst + (ulong) gid.y * a.K + gid.x * 16;
    for (uint j = 0; j < 8; ++j) {
        const uint b = q[j];
        o[j] = half(kFP4[b & 15u] * d);
        o[j + 8] = half(kFP4[b >> 4] * d);
    }
}

// ------------------------------------------------------------------------------------------------
// GEMM (prompt processing): C[n, m] = sum_k B[n, k] * A[m, k]
// ------------------------------------------------------------------------------------------------

struct GemmArgs { uint N; uint M; uint K; uint ldb; uint ldc; uint pad0; uint pad1; uint pad2; };

#define TILE 64
#define GEMM_KERNEL(NAME, TB, TA, TC) \
kernel void NAME(constant GemmArgs & a, device const TB * Bp, device const TA * Ap, device TC * Cp, \
                 uint3 tg [[threadgroup_position_in_grid]]) { \
    const int n0 = tg.y * TILE, m0 = tg.x * TILE; \
    const int tn = min((int) TILE, (int) a.N - n0), tm = min((int) TILE, (int) a.M - m0); \
    auto tB = tensor((device TB *) Bp + (ulong) n0 * a.ldb, dextents<int32_t, 2>((int) a.K, tn), array<int, 2>({1, (int) a.ldb})); \
    auto tA = tensor((device TA *) Ap + (ulong) m0 * a.K, dextents<int32_t, 2>((int) a.K, tm), array<int, 2>({1, (int) a.K})); \
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(TILE, TILE, static_cast<int>(dynamic_extent), false, true, false, \
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply), execution_simdgroups<4>> mm; \
    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), TC>(); \
    mm.run(tB, tA, cT); \
    auto tD = tensor(Cp + (ulong) n0 * a.ldc + m0, dextents<int32_t, 2>(tm, tn), array<int, 2>({1, (int) a.ldc})); \
    cT.store(tD); \
}

GEMM_KERNEL(k_gemm_hh_f, half, half, float)
GEMM_KERNEL(k_gemm_fh_f, float, half, float)
GEMM_KERNEL(k_gemm_ff_f, float, float, float)

// per-expert GEMM over rows sorted by expert.  tiles[3*t] = {expert, first row, row count (<= TILE)}
struct MoeArgs { uint M; uint K; uint nTiles; uint pad; };

kernel void k_gemm_moe(constant MoeArgs & a, device const half * Xg, device const half * W, device float * H,
                       device const uint * tiles, uint3 tg [[threadgroup_position_in_grid]]) {
    // x (fastest) walks the token tiles so that neighbouring threadgroups share a weight tile
    const uint e = tiles[3 * tg.x], r0 = tiles[3 * tg.x + 1];
    const int tn = (int) tiles[3 * tg.x + 2];
    const int m0 = tg.y * TILE;
    const int tm = min((int) TILE, (int) a.M - m0);
    auto tB = tensor((device half *) Xg + (ulong) r0 * a.K, dextents<int32_t, 2>((int) a.K, tn), array<int, 2>({1, (int) a.K}));
    auto tA = tensor((device half *) W + ((ulong) e * a.M + m0) * a.K, dextents<int32_t, 2>((int) a.K, tm), array<int, 2>({1, (int) a.K}));
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(TILE, TILE, static_cast<int>(dynamic_extent), false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply), execution_simdgroups<4>> mm;
    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();
    mm.run(tB, tA, cT);
    auto tD = tensor(H + (ulong) r0 * a.M + m0, dextents<int32_t, 2>(tm, tn), array<int, 2>({1, (int) a.M}));
    cT.store(tD);
}

// ------------------------------------------------------------------------------------------------
// per-token kernels (prompt processing).  One SIMD group (32 threads) per token unless noted.
// ------------------------------------------------------------------------------------------------

struct EmbedArgs { uint N; float eps; };

kernel void k_embed(constant EmbedArgs & a, device const int * tok, device const half * emb, device const float * w,
                    device float * cur, uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_threadgroup]]) {
    const uint n = tg.x;
    device const half * e = emb + (ulong) tok[n] * D;
    float ss = 0;
    for (uint i = ti; i < D; i += SW) { const float v = e[i]; ss += v * v; }
    ss = simd_sum(ss);
    const float inv = 1.0f / sqrt(ss / D + a.eps);
    device float * o = cur + (ulong) n * D;
    for (uint i = ti; i < D; i += SW) o[i] = float(e[i]) * inv * w[i];
}

// rms-normalize rows of src into dst (no weight)
struct NormArgs { uint N; float eps; };

kernel void k_norm_store(constant NormArgs & a, device const float * src, device float * dst,
                         uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_threadgroup]]) {
    device const float * x = src + (ulong) tg.x * D;
    device float * o = dst + (ulong) tg.x * D;
    float ss = 0;
    for (uint i = ti; i < D; i += SW) ss += x[i] * x[i];
    ss = simd_sum(ss);
    const float inv = 1.0f / sqrt(ss / D + a.eps);
    for (uint i = ti; i < D; i += SW) o[i] = x[i] * inv;
}

// depth mix + norm.
//   mode 0: out = rmsnorm(cur) * w                               (nq == 1)
//   mode 1: out[q] = rmsnorm(mix_q(states, cur)) * w[q]
//   mode 2: outF[q] = mix_q(states, cur)                          (raw, f32)
// mix_q = sum_s p[s] * state_s, p = softmax_s(<state_s, query_q> * qscale), states = normalized blocks + rmsnorm(cur)
struct MixArgs {
    uint N; uint nStates; uint nq; uint mode;
    uint slot0; uint slot1; uint slot2; uint slot3;
    uint nbStride;      // elements between NB slots
    uint outStride;     // elements between per-query outputs
    uint wantF32;       // also write the normalized result as f32 (query 0 only) to outF
    float eps; float mixEps; float qscale;
    uint wStride;       // elements between per-query norm weights (0 = shared)
    uint pad;
};

kernel void k_mix_norm(constant MixArgs & a, device const float * cur, device const float * NB, device const float * queries,
                       device const float * normw, device half * outH, device float * outF,
                       uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_threadgroup]]) {
    const uint n = tg.x;
    device const float * c = cur + (ulong) n * D;
    if (a.mode == 0) {
        float ss = 0;
        for (uint i = ti; i < D; i += SW) ss += c[i] * c[i];
        ss = simd_sum(ss);
        const float inv = 1.0f / sqrt(ss / D + a.eps);
        device half * o = outH + (ulong) n * D;
        for (uint i = ti; i < D; i += SW) {
            const float v = c[i] * inv * normw[i];
            o[i] = half(v);
            if (a.wantF32) outF[(ulong) n * D + i] = v;
        }
        return;
    }
    const uint slots[4] = {a.slot0, a.slot1, a.slot2, a.slot3};
    device const float * st[4];
    for (uint s = 0; s < 4; ++s) st[s] = NB + (ulong) slots[s] * a.nbStride + (ulong) n * D;
    const uint S = a.nStates, Q = a.nq;

    float ss = 0;
    float dc[4] = {0, 0, 0, 0};
    float ds[4][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}, {0, 0, 0, 0}};
    for (uint i = ti; i < D; i += SW) {
        const float v = c[i];
        ss += v * v;
        for (uint q = 0; q < Q; ++q) {
            const float qv = queries[q * D + i];
            dc[q] += v * qv;
            for (uint s = 0; s < S; ++s) ds[s][q] += st[s][i] * qv;
        }
    }
    ss = simd_sum(ss);
    const float inv = 1.0f / sqrt(ss / D + a.mixEps);
    float pc[4];
    float ps[4][4];
    for (uint q = 0; q < Q; ++q) {
        const float lc = simd_sum(dc[q]) * inv * a.qscale;
        float mx = lc;
        float ls[4];
        for (uint s = 0; s < S; ++s) { ls[s] = simd_sum(ds[s][q]) * a.qscale; mx = max(mx, ls[s]); }
        float den = exp(lc - mx);
        pc[q] = den;
        for (uint s = 0; s < S; ++s) { ps[s][q] = exp(ls[s] - mx); den += ps[s][q]; }
        pc[q] = pc[q] / den * inv;       // folded: weight applied to raw cur
        for (uint s = 0; s < S; ++s) ps[s][q] /= den;
    }
    for (uint q = 0; q < Q; ++q) {
        if (a.mode == 2) {
            device float * o = outF + (ulong) q * a.outStride + (ulong) n * D;
            for (uint i = ti; i < D; i += SW) {
                float m = pc[q] * c[i];
                for (uint s = 0; s < S; ++s) m += ps[s][q] * st[s][i];
                o[i] = m;
            }
        } else {
            float ss2 = 0;
            for (uint i = ti; i < D; i += SW) {
                float m = pc[q] * c[i];
                for (uint s = 0; s < S; ++s) m += ps[s][q] * st[s][i];
                ss2 += m * m;
            }
            ss2 = simd_sum(ss2);
            const float inv2 = 1.0f / sqrt(ss2 / D + a.eps);
            device half * o = outH + (ulong) q * a.outStride + (ulong) n * D;
            device const float * w = normw + q * a.wStride;
            for (uint i = ti; i < D; i += SW) {
                float m = pc[q] * c[i];
                for (uint s = 0; s < S; ++s) m += ps[s][q] * st[s][i];
                const float v = m * inv2 * w[i];
                o[i] = half(v);
                if (a.wantF32 && q == 0) outF[(ulong) n * D + i] = v;
            }
        }
    }
}

// loop controller, step 1: gate input [rmsnorm(x) ; rmsnorm(msg)] (half, 2*D per token)
kernel void k_ctrl_gate_in(constant NormArgs & a, device const float * x, device const float * msg, device half * out,
                           uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_threadgroup]]) {
    const uint n = tg.x;
    device const float * xv = x + (ulong) n * D;
    device const float * mv = msg + (ulong) n * D;
    float s1 = 0, s2 = 0;
    for (uint i = ti; i < D; i += SW) { s1 += xv[i] * xv[i]; s2 += mv[i] * mv[i]; }
    s1 = simd_sum(s1); s2 = simd_sum(s2);
    const float i1 = 1.0f / sqrt(s1 / D + a.eps), i2 = 1.0f / sqrt(s2 / D + a.eps);
    device half * o = out + (ulong) n * 2 * D;
    for (uint i = ti; i < D; i += SW) { o[i] = half(xv[i] * i1); o[D + i] = half(mv[i] * i2); }
}

// loop controller, step 2: x' = x - tanh(forget) * x + tanh(gain) * msg, then rmsnorm(x') * w
// GF holds [gain (D) | forget (D)] per token
struct CtrlArgs { uint N; float eps; uint wantF32; uint pad; };

kernel void k_ctrl_apply_norm(constant CtrlArgs & a, device const float * x, device const float * msg, device const float * GF,
                              device const float * normw, device half * outH, device float * outF,
                              uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_threadgroup]]) {
    const uint n = tg.x;
    device const float * xv = x + (ulong) n * D;
    device const float * mv = msg + (ulong) n * D;
    device const float * gf = GF + (ulong) n * 2 * D;
    float ss = 0;
    for (uint i = ti; i < D; i += SW) {
        const float v = xv[i] - tanh(gf[D + i]) * xv[i] + tanh(gf[i]) * mv[i];
        ss += v * v;
    }
    ss = simd_sum(ss);
    const float inv = 1.0f / sqrt(ss / D + a.eps);
    device half * o = outH + (ulong) n * D;
    for (uint i = ti; i < D; i += SW) {
        const float v = (xv[i] - tanh(gf[D + i]) * xv[i] + tanh(gf[i]) * mv[i]) * inv * normw[i];
        o[i] = half(v);
        if (a.wantF32) outF[(ulong) n * D + i] = v;
    }
}

// Q and R post-processing: per-head scale + L2 clip (+ log position scale on full-attention layers)
struct PrepQArgs {
    uint N; uint pos0; uint PW; uint qCol; uint rCol;
    float qThr; float rThr; float tauAlpha; float tauFloor; uint useTau;
    float qScale; uint pad0; uint pad1; uint pad2; uint pad3; uint pad4;
};

static inline float pos_tau(uint pos, float alpha, float floorv) {
    return 1.0f + alpha * log(max(1.0f, float(pos + 1) / floorv));
}

kernel void k_prep_q(constant PrepQArgs & a, device const float * P, device const float * qclip, device const float * rclip,
                     device half * Qh, device float * Rc,
                     uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_threadgroup]]) {
    const uint n = tg.x;
    const float tau = a.useTau ? pos_tau(a.pos0 + n, a.tauAlpha, a.tauFloor) : 1.0f;
    device const float * row = P + (ulong) n * a.PW;
    for (uint h = 0; h < NH; ++h) {
        const uint o = h * DH + ti * 4;
        float4 v = float4(row[a.qCol + o], row[a.qCol + o + 1], row[a.qCol + o + 2], row[a.qCol + o + 3]) *
                   float4(qclip[o], qclip[o + 1], qclip[o + 2], qclip[o + 3]);
        const float ss = simd_sum(dot(v, v));
        // stored pre-scaled and grouped by KV head: QG[hk][n][g][DH], so the query heads that share a
        // KV head form one contiguous (N*GQA) x DH matrix
        const float s = a.qThr / max(sqrt(ss), a.qThr) * tau * a.qScale;
        device half * q = Qh + ((ulong) (h / GQA) * a.N + n) * (GQA * DH) + (h % GQA) * DH + ti * 4;
        v *= s;
        q[0] = half(v.x); q[1] = half(v.y); q[2] = half(v.z); q[3] = half(v.w);
    }
    for (uint h = 0; h < NH; ++h) {
        float4 v = 0;
        const uint o = h * NREL + ti * 4;
        if (ti < NREL / 4) {
            v = float4(row[a.rCol + o], row[a.rCol + o + 1], row[a.rCol + o + 2], row[a.rCol + o + 3]) *
                float4(rclip[o], rclip[o + 1], rclip[o + 2], rclip[o + 3]);
        }
        const float ss = simd_sum(dot(v, v));
        const float s = a.rThr / max(sqrt(ss), a.rThr) * tau;
        if (ti < NREL / 4) {
            device float * r = Rc + (ulong) n * NH * NREL + o;
            v *= s;
            r[0] = v.x; r[1] = v.y; r[2] = v.z; r[3] = v.w;
        }
    }
}

// K / V post-processing: causal short convolution (+ optional input-dependent taps), then per-head
// L2 clip (K) or rms norm (V), stored as half into the cache at the token's slot.
struct PrepKVArgs {
    uint N; uint pos0; uint PW; uint col; uint dynCol; uint useDyn;
    uint goff;          // channel offset of this group in the conv state
    uint isV;
    float thr; float eps; float invs;
    uint slotMask;      // ring mask for SWA caches, 0xffffffff for linear caches
    uint stStride;      // conv state row length (channels)
    uint pad0; uint pad1; uint pad2;
};

kernel void k_prep_kv(constant PrepKVArgs & a, device const float * P, device const float * convw, device const float * state,
                      device half * cache, uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_threadgroup]]) {
    const uint n = tg.x;
    const uint slot = (a.pos0 + n) & a.slotMask;
    device const float * row = P + (ulong) n * a.PW;
    for (uint hk = 0; hk < NKV; ++hk) {
        const uint c = hk * DH + ti * 4;
        float4 x[NCONV];   // x[j] = input at time t - j
        for (uint j = 0; j < NCONV; ++j) {
            if (n >= j) {
                device const float * r = P + (ulong) (n - j) * a.PW + a.col + c;
                x[j] = float4(r[0], r[1], r[2], r[3]);
            } else {
                device const float * r = state + (ulong) (NCONV - 1 - j + n) * a.stStride + a.goff + c;
                x[j] = float4(r[0], r[1], r[2], r[3]);
            }
        }
        float4 conv = 0;
        for (uint k = 0; k < 4; ++k) {
            device const float * w = convw + (c + k) * NCONV;
            float s = 0;
            for (uint j = 0; j < NCONV; ++j) s += w[NCONV - 1 - j] * x[j][k];
            conv[k] = s;
        }
        if (a.useDyn) {
            for (uint j = 0; j < NCONV; ++j) conv += row[a.dynCol + hk * NCONV + j] * x[j];
        }
        float4 y = (x[0] + conv) * a.invs;
        const float ss = simd_sum(dot(y, y));
        const float s = a.isV ? 1.0f / sqrt(ss / DH + a.eps) : a.thr / max(sqrt(ss), a.thr);
        y *= s;
        device half * o = cache + (ulong) slot * DKV + c;
        o[0] = half(y.x); o[1] = half(y.y); o[2] = half(y.z); o[3] = half(y.w);
    }
}

// causal short convolution + residual for the attention / FFN outputs (one thread per 4 channels)
struct ConvArgs { uint N; uint C; uint goff; uint mode; float invs; uint ldx; uint xcol; uint stStride; };

kernel void k_conv_residual(constant ConvArgs & a, device const float * X, device const float * convw, device const float * state,
                            device float * cur, uint2 gid [[thread_position_in_grid]]) {
    const uint n = gid.y, c = gid.x * 4;
    if (n >= a.N || c >= a.C) return;
    float4 x[NCONV];
    for (uint j = 0; j < NCONV; ++j) {
        if (n >= j) {
            device const float * r = X + (ulong) (n - j) * a.ldx + a.xcol + c;
            x[j] = float4(r[0], r[1], r[2], r[3]);
        } else {
            device const float * r = state + (ulong) (NCONV - 1 - j + n) * a.stStride + a.goff + c;
            x[j] = float4(r[0], r[1], r[2], r[3]);
        }
    }
    float4 conv = 0;
    for (uint k = 0; k < 4; ++k) {
        device const float * w = convw + (c + k) * NCONV;
        float s = 0;
        for (uint j = 0; j < NCONV; ++j) s += w[NCONV - 1 - j] * x[j][k];
        conv[k] = s;
    }
    const float4 y = (x[0] + conv) * a.invs;
    device float * o = cur + (ulong) n * a.C + c;
    if (a.mode) { o[0] += y.x; o[1] += y.y; o[2] += y.z; o[3] += y.w; }
    else        { o[0]  = y.x; o[1]  = y.y; o[2]  = y.z; o[3]  = y.w; }
}

// new conv state = last NCONV-1 inputs of the sequence (taken from the batch, or the old state when the batch is short)
kernel void k_conv_state(constant ConvArgs & a, device const float * X, device const float * stIn, device float * stOut,
                         uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.y, c = gid.x;
    if (i >= NCONV - 1 || c >= a.C) return;
    const int t = (int) a.N - (NCONV - 1) + (int) i;
    stOut[(ulong) i * a.stStride + a.goff + c] =
        t >= 0 ? X[(ulong) t * a.ldx + a.xcol + c] : stIn[(ulong) (i + a.N) * a.stStride + a.goff + c];
}

// ------------------------------------------------------------------------------------------------
// attention (prompt processing): flash attention on Metal Performance Primitives matmul.
// One threadgroup (8 SIMD groups) per (32-query tile, head).  Per key tile: S = Q*K^T into threadgroup
// memory, online softmax per row across one SIMD group, P*V accumulated in a cooperative tensor.
// Key tiles are aligned in position space so that they are contiguous in the (ring) cache.
// ------------------------------------------------------------------------------------------------

struct AttnArgs {
    uint N; uint pos0; uint swa; uint slotMask; uint window; uint extent;
    float scale; uint PW; uint sinkCol; uint gateCol; uint pad0; uint pad1;
};

#ifndef AQ
#define AQ 32
#endif
#ifndef AC
#define AC 128
#endif
#define ANSG 8
#define ANQ (AQ / ANSG)

kernel void k_attn_mpp(constant AttnArgs & a, device const half * QG, device const half * Kc, device const half * Vc,
                       device const float * T, device const float * P, device half * Out,
                       uint3 tg [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]],
                       ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float ss[AQ * AC];
    threadgroup half sp[AQ * AC];
    threadgroup float sr[AQ];
    threadgroup int sf[1];

    const uint q0 = tg.x * AQ, h = tg.y, hk = h / GQA;
    const ushort tiitg = sgitg * SW + tiisg;
    const uint nq = min((uint) AQ, a.N - q0);
    const uint pFirst = a.pos0 + q0, pLast = pFirst + nq - 1;

    float M[ANQ], S[ANQ];
    for (uint jj = 0; jj < ANQ; ++jj) { M[jj] = -FLT_MAX / 2; S[jj] = 0; }

    // the pre-scaled queries of this head: rows (tokens) are GQA*DH apart in the grouped layout
    auto tq = tensor((device half *) QG + ((ulong) hk * a.N + q0) * (GQA * DH) + (h % GQA) * DH,
                     dextents<int32_t, 2>(DH, (int) nq), array<int, 2>({1, GQA * DH}));
    auto ts = tensor(ss, dextents<int32_t, 2>(AC, (int) nq), array<int, 2>({1, AC}));
    auto tp = tensor(sp, dextents<int32_t, 2>(AC, (int) nq), array<int, 2>({1, AC}));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(AQ, AC, DH, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply),
        execution_simdgroups<ANSG>> mm_qk;
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(AQ, DH, AC, false, false, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<ANSG>> mm_pv;

    auto tv0 = tensor((device half *) Vc, dextents<int32_t, 2>(DH, AC), array<int, 2>({1, DKV}));
    auto co = mm_pv.get_destination_cooperative_tensor<decltype(tp), decltype(tv0), float>();
    for (ushort i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) co[i] = 0.0f;
    }
    if (tiitg == 0) sf[0] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint loFirst = (a.swa && pFirst >= a.window) ? pFirst - (a.window - 1) : 0;
    int it = 0;
    for (uint kb = (loFirst / AC) * AC; kb <= pLast; kb += AC) {
        ++it;
        const ulong slot = kb & a.slotMask;
        {
            auto tk = tensor((device half *) Kc + slot * DKV + hk * DH, dextents<int32_t, 2>(DH, AC), array<int, 2>({1, DKV}));
            mm_qk.run(tq, tk, ts);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint jj = 0; jj < ANQ; ++jj) {
            const uint j = jj * ANSG + sgitg;
            const uint p = pFirst + j;
            const uint lo = (a.swa && p >= a.window) ? p - (a.window - 1) : 0;
            device const float * Tb = T + ((ulong) (q0 + j) * NH + h) * a.extent;
            float s[AC / SW];
            bool ok[AC / SW];
            float m = M[jj];
            for (uint ii = 0; ii < AC / SW; ++ii) {
                const uint c = ii * SW + tiisg;
                const uint kp = kb + c;
                ok[ii] = j < nq && kp <= p && kp >= lo;
                s[ii] = -FLT_MAX / 2;
                if (ok[ii]) {
                    const uint d = p - kp;
                    s[ii] = ss[j * AC + c] + (d < a.extent ? Tb[d] : 0.0f);
                    m = max(m, s[ii]);
                }
            }
            m = simd_max(m);
            // lazy rescaling: move the running max only when it grows by more than 8 (e^8 fits in half)
            float ms = 1.0f;
            if (m > M[jj] + 8.0f) {
                ms = exp(M[jj] - m);
                M[jj] = m;
                if (tiisg == 0) sf[0] = it;
            }
            float sum = 0;
            for (uint ii = 0; ii < AC / SW; ++ii) {
                const half pr = ok[ii] ? half(exp(s[ii] - M[jj])) : half(0);
                sp[j * AC + ii * SW + tiisg] = pr;
                sum += float(pr);
            }
            S[jj] = S[jj] * ms + simd_sum(sum);
            if (tiisg == 0) sr[j] = ms;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sf[0] == it) {
            for (ushort i = 0; i < co.get_capacity(); ++i) {
                if (co.is_valid_element(i)) co[i] *= sr[co.get_multidimensional_index(i)[1]];
            }
        }
        {
            auto tv = tensor((device half *) Vc + slot * DKV + hk * DH, dextents<int32_t, 2>(DH, AC), array<int, 2>({1, DKV}));
            mm_pv.run(tp, tv, co);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // fold in the sink (denominator only) and normalize
    for (uint jj = 0; jj < ANQ; ++jj) {
        const uint j = jj * ANSG + sgitg;
        float r = 0;
        if (j < nq) {
            const float sink = P[(ulong) (q0 + j) * a.PW + a.sinkCol + h];
            const float m = max(M[jj], sink);
            const float ms = exp(M[jj] - m);
            r = ms / (S[jj] * ms + exp(sink - m));
        }
        if (tiisg == 0) sr[j] = r;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) {
            const auto idx = co.get_multidimensional_index(i);
            const uint j = idx[1];
            if (j < nq) {
                const float g = P[(ulong) (q0 + j) * a.PW + a.gateCol + h * DH + idx[0]];
                Out[(ulong) (q0 + j) * DQ + h * DH + idx[0]] = half(co[i] * sr[j] / (1.0f + exp(-g)));
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------
// MoE helpers
// ------------------------------------------------------------------------------------------------

struct RouteArgs { uint N; float wscale; float eps; uint pad; };

// one thread per token: sigmoid gate, biased top-k selection, normalized weights
kernel void k_route(constant RouteArgs & a, device const float * logits, device const float * bias,
                    device uint * ids, device float * wts, uint gid [[thread_position_in_grid]]) {
    if (gid >= a.N) return;
    device const float * l = logits + (ulong) gid * NEXP;
    float pr[NEXP];
    float sel[NEXP];
    for (uint e = 0; e < NEXP; ++e) { pr[e] = 1.0f / (1.0f + exp(-l[e])); sel[e] = pr[e] + bias[e]; }
    float w[NUSED];
    uint id[NUSED];
    float sum = 0;
    for (uint k = 0; k < NUSED; ++k) {
        uint best = 0; float bv = sel[0];
        for (uint e = 1; e < NEXP; ++e) if (sel[e] > bv) { bv = sel[e]; best = e; }
        sel[best] = -1e30f;
        id[k] = best; w[k] = pr[best]; sum += pr[best];
    }
    for (uint k = 0; k < NUSED; ++k) {
        ids[(ulong) gid * NUSED + k] = id[k];
        wts[(ulong) gid * NUSED + k] = w[k] / (sum + a.eps) * a.wscale;
    }
}

struct RowsArgs { uint R; uint N; float beta; float lbeta; };

// Xg[r] = xff[pairTok[r]]
kernel void k_gather(constant RowsArgs & a, device const half * xff, device const uint * pairTok, device half * Xg,
                     uint2 gid [[thread_position_in_grid]]) {
    if (gid.y >= a.R || gid.x >= D / 4) return;
    device const half4 * s = (device const half4 *) (xff + (ulong) pairTok[gid.y] * D);
    device half4 * d = (device half4 *) (Xg + (ulong) gid.y * D);
    d[gid.x] = s[gid.x];
}

// situ(gate, up) = beta*tanh(gate/beta)*sigmoid(gate) * lb*tanh(up/lb); H rows are [gate (FF) | up (FF)], scaled per row
kernel void k_act(constant RowsArgs & a, device const float * H, device const float * rowScale, device half * A,
                  uint2 gid [[thread_position_in_grid]]) {
    if (gid.y >= a.R || gid.x >= FF) return;
    const float s = rowScale[gid.y];
    device const float * h = H + (ulong) gid.y * 2 * FF;
    const float g = h[gid.x] * s, u = h[FF + gid.x] * s;
    const float act = a.beta * tanh(g / a.beta) / (1.0f + exp(-g)) * a.lbeta * tanh(u / a.lbeta);
    A[(ulong) gid.y * FF + gid.x] = half(act);
}

// Y[n] = sum_j w[n][j] * O[rowOfPair[n*NUSED + j]]
kernel void k_combine(constant RowsArgs & a, device const float * O, device const uint * rowOfPair, device const float * wts,
                      device float * Y, uint2 gid [[thread_position_in_grid]]) {
    if (gid.y >= a.N || gid.x >= D) return;
    float y = 0;
    for (uint j = 0; j < NUSED; ++j) {
        y += wts[(ulong) gid.y * NUSED + j] * O[(ulong) rowOfPair[(ulong) gid.y * NUSED + j] * D + gid.x];
    }
    Y[(ulong) gid.y * D + gid.x] = y;
}

// logits soft cap: cap * tanh(x / cap)
kernel void k_softcap(constant CountArgs & a, device float * x, uint gid [[thread_position_in_grid]]) {
    if (gid >= a.w) return;
    x[gid] = a.f * tanh(x[gid] / a.f);
}

// ================================================================================================
// decode path (single token).  Small kernels are wide and shallow: a GPU thread is slow at long
// sequential loops, so vector work is spread over many threads and reduced through SIMD-group
// sums plus one hop through threadgroup memory.
// ================================================================================================

// Loop counts of the small vector kernels are function constants: the host builds one specialised
// pipeline per combination, so the compiler can fully unroll and keep everything in registers.
constant uint FC_S [[function_constant(0)]];      // number of mixed block states
constant uint FC_Q [[function_constant(1)]];      // number of queries
constant uint FC_MODE [[function_constant(2)]];   // d_mix: 1 = normalised output, 2 = raw
constant uint FC_NJ [[function_constant(3)]];     // number of parallel inputs (d_norm, d_gate_in)

#define TGW 768                 // threads in the single-threadgroup vector kernels
#define TGSG (TGW / SW)         // SIMD groups in such a threadgroup (24)
#define RMAX 24                 // max values per reduction

// Reduce nv per-thread partial values across the whole threadgroup; on return v[] holds the totals.
static inline void tg_reduce(thread float * v, const uint nv, threadgroup float * part, threadgroup float * tot,
                             const ushort sg, const ushort tis, const ushort nsg) {
    for (uint k = 0; k < nv; ++k) {
        const float s = simd_sum(v[k]);
        if (tis == 0) part[sg * RMAX + k] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && tis < nv) {
        float t = 0;
        for (uint g = 0; g < nsg; ++g) t += part[g * RMAX + tis];
        tot[tis] = t;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < nv; ++k) v[k] = tot[k];
}

static inline float3 ld3(device const float * p) { return float3(p[0], p[1], p[2]); }
static inline void st3(device float * p, float3 v) { p[0] = v.x; p[1] = v.y; p[2] = v.z; }

struct DEmbedArgs { uint tok; float eps; };

kernel void d_embed(constant DEmbedArgs & a, device const half * emb, device const float * w, device float * cur,
                    ushort ti [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                    ushort tis [[thread_index_in_simdgroup]]) {
    threadgroup float part[TGSG * RMAX];
    threadgroup float tot[RMAX];
    const uint i = ti * 3;
    device const half * e = emb + (ulong) a.tok * D + i;
    const float3 x = float3(e[0], e[1], e[2]);
    float v[1] = {dot(x, x)};
    tg_reduce(v, 1, part, tot, sg, tis, TGSG);
    const float inv = 1.0f / sqrt(v[0] / D + a.eps);
    st3(cur + i, x * inv * ld3(w + i));
}

// out_j = rmsnorm(src_j) [* w_j], j < nj
struct DNormArgs { uint nj; uint srcStride; uint dstStride; uint wOff0; uint wOff1; uint wOff2; uint useW; float eps; };

kernel void d_norm(constant DNormArgs & a, device const float * src, device const float * w, device float * dst,
                   ushort ti [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                   ushort tis [[thread_index_in_simdgroup]]) {
    threadgroup float part[TGSG * RMAX];
    threadgroup float tot[RMAX];
    const uint i = ti * 3;
    const uint wOff[3] = {a.wOff0, a.wOff1, a.wOff2};
    float3 x[3];
    float v[3];
    for (uint j = 0; j < FC_NJ; ++j) { x[j] = ld3(src + j * a.srcStride + i); v[j] = dot(x[j], x[j]); }
    tg_reduce(v, FC_NJ, part, tot, sg, tis, TGSG);
    for (uint j = 0; j < FC_NJ; ++j) {
        float3 y = x[j] * (1.0f / sqrt(v[j] / D + a.eps));
        if (a.useW) y *= ld3(w + wOff[j] + i);
        st3(dst + j * a.dstStride + i, y);
    }
}

// depth mix (see k_mix_norm): mode 1 -> rmsnorm(mix_q) * w_q, mode 2 -> raw mix_q; f32 outputs
struct DMixArgs {
    uint nStates; uint nq; uint mode; uint nbStride;
    uint slot0; uint slot1; uint slot2; uint slot3;
    uint outStride; uint wStride; float eps; float mixEps; float qscale; uint pad0; uint pad1; uint pad2;
};

kernel void d_mix(constant DMixArgs & a, device const float * cur, device const float * NB, device const float * queries,
                  device const float * normw, device float * out,
                  ushort ti [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                  ushort tis [[thread_index_in_simdgroup]]) {
    threadgroup float part[TGSG * RMAX];
    threadgroup float tot[RMAX];
    const uint i = ti * 3;
    const uint S = FC_S, Q = FC_Q;
    const uint slots[4] = {a.slot0, a.slot1, a.slot2, a.slot3};
    const float3 c = ld3(cur + i);
    float3 st[4];
    for (uint s = 0; s < S; ++s) st[s] = ld3(NB + (ulong) slots[s] * a.nbStride + i);
    float v[RMAX];
    v[0] = dot(c, c);
    for (uint q = 0; q < Q; ++q) {
        const float3 qv = ld3(queries + q * D + i);
        v[1 + q] = dot(c, qv);
        for (uint s = 0; s < S; ++s) v[5 + s * 4 + q] = dot(st[s], qv);
    }
    // compact the used values: [ss, dc[0..Q), ds[s][q]...]
    float r[RMAX];
    uint nv = 0;
    r[nv++] = v[0];
    for (uint q = 0; q < Q; ++q) r[nv++] = v[1 + q];
    for (uint s = 0; s < S; ++s) for (uint q = 0; q < Q; ++q) r[nv++] = v[5 + s * 4 + q];
    tg_reduce(r, nv, part, tot, sg, tis, TGSG);
    const float inv = 1.0f / sqrt(r[0] / D + a.mixEps);
    float3 m[4];
    float ss2[4];
    for (uint q = 0; q < Q; ++q) {
        const float lc = r[1 + q] * inv * a.qscale;
        float mx = lc;
        float ls[4];
        for (uint s = 0; s < S; ++s) { ls[s] = r[1 + Q + s * Q + q] * a.qscale; mx = max(mx, ls[s]); }
        float pc = exp(lc - mx);
        float den = pc;
        float ps[4];
        for (uint s = 0; s < S; ++s) { ps[s] = exp(ls[s] - mx); den += ps[s]; }
        float3 acc = c * (pc / den * inv);
        for (uint s = 0; s < S; ++s) acc += st[s] * (ps[s] / den);
        m[q] = acc;
        ss2[q] = dot(acc, acc);
    }
    if (FC_MODE == 2) {
        for (uint q = 0; q < Q; ++q) st3(out + q * a.outStride + i, m[q]);
        return;
    }
    tg_reduce(ss2, Q, part, tot, sg, tis, TGSG);
    for (uint q = 0; q < Q; ++q) {
        st3(out + q * a.outStride + i, m[q] * (1.0f / sqrt(ss2[q] / D + a.eps)) * ld3(normw + q * a.wStride + i));
    }
}

// gateIn_j = [rmsnorm(x_j) ; rmsnorm(msg_j)], j < nj
struct DGateArgs { uint nj; uint xStride; uint msgStride; float eps; };

kernel void d_gate_in(constant DGateArgs & a, device const float * x, device const float * msg, device float * out,
                      ushort ti [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                      ushort tis [[thread_index_in_simdgroup]]) {
    threadgroup float part[TGSG * RMAX];
    threadgroup float tot[RMAX];
    const uint i = ti * 3;
    float3 xv[3], mv[3];
    float v[6];
    for (uint j = 0; j < FC_NJ; ++j) {
        xv[j] = ld3(x + j * a.xStride + i);
        mv[j] = ld3(msg + j * a.msgStride + i);
        v[2 * j] = dot(xv[j], xv[j]);
        v[2 * j + 1] = dot(mv[j], mv[j]);
    }
    tg_reduce(v, 2 * FC_NJ, part, tot, sg, tis, TGSG);
    for (uint j = 0; j < FC_NJ; ++j) {
        st3(out + j * 2 * D + i, xv[j] * (1.0f / sqrt(v[2 * j] / D + a.eps)));
        st3(out + j * 2 * D + D + i, mv[j] * (1.0f / sqrt(v[2 * j + 1] / D + a.eps)));
    }
}

// g_j[r] = down_j[r] . gateIn_j   (grid: nj*rank threadgroups of 128 threads)
struct DCtrlArgs { uint nj; uint rank; uint ctrlBase; uint per; uint xStride; uint msgStride; uint pad0; uint pad1; };

kernel void d_ctrl_down(constant DCtrlArgs & a, device const half * W, device const float * gateIn, device float * g,
                        uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]], ushort tis [[thread_index_in_simdgroup]]) {
    threadgroup float part[4];
    const uint j = tg.x / a.rank, r = tg.x % a.rank;
    device const half4 * w = (device const half4 *) (W + (ulong) (a.ctrlBase + j) * a.per + (ulong) r * 2 * D);
    device const float4 * x = (device const float4 *) (gateIn + (ulong) j * 2 * D);
    float s = 0;
    for (uint i = ti; i < 2 * D / 4; i += 128) s += dot(float4(w[i]), x[i]);
    s = simd_sum(s);
    if (tis == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ti == 0) g[j * a.rank + r] = part[0] + part[1] + part[2] + part[3];
}

// x'_j[i] = x_j[i] * (1 - tanh(forget_j[i] . g_j)) + tanh(gain_j[i] . g_j) * msg_j[i]   (one thread per (i, j))
kernel void d_ctrl_apply(constant DCtrlArgs & a, device const half * W, device const float * g, device const float * x,
                         device const float * msg, device float * out, uint2 gid [[thread_position_in_grid]]) {
    const uint i = gid.x, j = gid.y;
    if (i >= D || j >= a.nj) return;
    device const half4 * base = (device const half4 *) (W + (ulong) (a.ctrlBase + j) * a.per + (ulong) a.rank * 2 * D);
    device const half4 * wg = base + (ulong) i * a.rank / 4;
    device const half4 * wf = base + ((ulong) D * a.rank + (ulong) i * a.rank) / 4;
    device const float4 * gv = (device const float4 *) (g + j * a.rank);
    float sgain = 0, sforget = 0;
    for (uint k = 0; k < a.rank / 4; ++k) {
        const float4 gg = gv[k];
        sgain += dot(float4(wg[k]), gg);
        sforget += dot(float4(wf[k]), gg);
    }
    const float xv = x[j * a.xStride + i];
    out[j * D + i] = xv - tanh(sforget) * xv + tanh(sgain) * msg[j * a.msgStride + i];
}

// fused attention projections: P[r] = W[r] . xin[group(r)], two rows per SIMD group
struct DProjArgs { uint rows; uint kCol; uint vCol; uint xStride; };

kernel void d_proj(constant DProjArgs & a, device const half * W, device const float * xin, device float * P,
                   uint3 tg [[threadgroup_position_in_grid]], ushort tis [[thread_index_in_threadgroup]]) {
    const uint r0 = tg.x * 2;
    const uint sel = r0 < a.kCol ? 0 : (r0 < a.vCol ? 1 : 2);
    device const float4 * x = (device const float4 *) (xin + sel * a.xStride);
    device const half4 * w0 = (device const half4 *) (W + (ulong) r0 * D);
    device const half4 * w1 = w0 + D / 4;
    float s0 = 0, s1 = 0;
    for (uint i = tis; i < D / 4; i += SW) {
        const float4 xv = x[i];
        s0 += dot(float4(w0[i]), xv);
        s1 += dot(float4(w1[i]), xv);
    }
    s0 = simd_sum(s0);
    s1 = simd_sum(s1);
    if (tis == 0) { P[r0] = s0; P[r0 + 1] = s1; }
}

// Q / R / K / V post-processing for one token.  Grid: 2*NH + 2*NKV SIMD groups:
//   [0, NH) Q head, [NH, 2NH) R head, [2NH, 2NH+NKV) K head, [2NH+NKV, 2NH+2NKV) V head
struct DPrepArgs {
    uint slot; uint qCol; uint rCol; uint kCol; uint vCol; uint kDynCol; uint vDynCol; uint useDyn;
    float qThr; float rThr; float tau; float eps; float invs; uint stStride; uint pad0; uint pad1;
};

kernel void d_attn_prep(constant DPrepArgs & a, device const float * P, device const float * qclip, device const float * rclip,
                        device const float * kconv, device const float * vconv, device const float * stIn, device float * stOut,
                        device float * Q, device float * R, device half * Kc, device half * Vc,
                        uint3 tg [[threadgroup_position_in_grid]], ushort tis [[thread_index_in_threadgroup]]) {
    const uint role = tg.x;
    if (role < NH) {
        const uint o = role * DH + tis * 4;
        float4 v = float4(P[a.qCol + o], P[a.qCol + o + 1], P[a.qCol + o + 2], P[a.qCol + o + 3]) *
                   float4(qclip[o], qclip[o + 1], qclip[o + 2], qclip[o + 3]);
        const float ss = simd_sum(dot(v, v));
        v *= a.qThr / max(sqrt(ss), a.qThr) * a.tau;
        Q[o] = v.x; Q[o + 1] = v.y; Q[o + 2] = v.z; Q[o + 3] = v.w;
    } else if (role < 2 * NH) {
        const uint o = (role - NH) * NREL + tis * 4;
        float4 v = 0;
        if (tis < NREL / 4) {
            v = float4(P[a.rCol + o], P[a.rCol + o + 1], P[a.rCol + o + 2], P[a.rCol + o + 3]) *
                float4(rclip[o], rclip[o + 1], rclip[o + 2], rclip[o + 3]);
        }
        const float ss = simd_sum(dot(v, v));
        v *= a.rThr / max(sqrt(ss), a.rThr) * a.tau;
        if (tis < NREL / 4) { R[o] = v.x; R[o + 1] = v.y; R[o + 2] = v.z; R[o + 3] = v.w; }
    } else {
        const bool isV = role >= 2 * NH + NKV;
        const uint hk = role - 2 * NH - (isV ? NKV : 0);
        const uint c = hk * DH + tis * 4;
        const uint col = isV ? a.vCol : a.kCol;
        const uint goff = isV ? DKV : 0;
        device const float * convw = isV ? vconv : kconv;
        float4 x[NCONV];
        x[0] = float4(P[col + c], P[col + c + 1], P[col + c + 2], P[col + c + 3]);
        for (uint j = 1; j < NCONV; ++j) {
            device const float * r = stIn + (ulong) (NCONV - 1 - j) * a.stStride + goff + c;
            x[j] = float4(r[0], r[1], r[2], r[3]);
        }
        float4 conv = 0;
        for (uint k = 0; k < 4; ++k) {
            device const float * w = convw + (c + k) * NCONV;
            float s = 0;
            for (uint j = 0; j < NCONV; ++j) s += w[NCONV - 1 - j] * x[j][k];
            conv[k] = s;
        }
        if (a.useDyn) {
            const uint dc = (isV ? a.vDynCol : a.kDynCol) + hk * NCONV;
            for (uint j = 0; j < NCONV; ++j) conv += P[dc + j] * x[j];
        }
        float4 y = (x[0] + conv) * a.invs;
        const float ss = simd_sum(dot(y, y));
        y *= isV ? 1.0f / sqrt(ss / DH + a.eps) : a.qThr / max(sqrt(ss), a.qThr);
        device half * o = (isV ? Vc : Kc) + (ulong) a.slot * DKV + c;
        o[0] = half(y.x); o[1] = half(y.y); o[2] = half(y.z); o[3] = half(y.w);
        // shift the conv state: [t-2, t-1, t]
        for (uint j = 0; j < NCONV - 1; ++j) {
            device float * so = stOut + (ulong) j * a.stStride + goff + c;
            const float4 nv = x[NCONV - 2 - j];
            so[0] = nv.x; so[1] = nv.y; so[2] = nv.z; so[3] = nv.w;
        }
    }
}

// attention partials: one SIMD group per (32-key chunk, KV head); each thread scores one key for the
// GQA query heads sharing that KV head, then the group accumulates V with chunk-local normalization.
// partial layout (PSZ floats): [0] max, [1] sum, [4..4+DH) weighted V
#define PSZ (DH + 4)
struct DAttnArgs { uint pos; uint lo; uint nk; uint slotMask; uint extent; float scale; uint nChunks; uint pad; };

kernel void d_attn(constant DAttnArgs & a, device const float * Q, device const float * R, device const float * relbias,
                   device const half * Kc, device const half * Vc, device float * part,
                   uint3 tg [[threadgroup_position_in_grid]], ushort tis [[thread_index_in_threadgroup]]) {
    threadgroup float pw[GQA * SW];
    const uint chunk = tg.x, hk = tg.y;
    const uint j = chunk * SW + tis;
    const bool valid = j < a.nk;
    float s[GQA];
    for (uint g = 0; g < GQA; ++g) s[g] = 0;
    if (valid) {
        const uint kp = a.lo + j;
        device const half4 * kr = (device const half4 *) (Kc + (ulong) (kp & a.slotMask) * DKV + hk * DH);
        device const float4 * q = (device const float4 *) (Q + hk * GQA * DH);
        for (uint i = 0; i < DH / 4; ++i) {
            const float4 kv = float4(kr[i]);
            for (uint g = 0; g < GQA; ++g) s[g] += dot(kv, q[g * (DH / 4) + i]);
        }
        const uint d = a.pos - kp;
        if (d < a.extent) {
            device const float4 * rb = (device const float4 *) (relbias + d * NREL);
            device const float4 * r = (device const float4 *) (R + hk * GQA * NREL);
            for (uint g = 0; g < GQA; ++g) {
                float b = 0;
                for (uint i = 0; i < NREL / 4; ++i) b += dot(rb[i], r[g * (NREL / 4) + i]);
                s[g] = s[g] * a.scale + b;
            }
        } else {
            for (uint g = 0; g < GQA; ++g) s[g] *= a.scale;
        }
    }
    float m[GQA], S[GQA];
    for (uint g = 0; g < GQA; ++g) {
        m[g] = simd_max(valid ? s[g] : -1e30f);
        const float p = valid ? exp(s[g] - m[g]) : 0.0f;
        S[g] = simd_sum(p);
        pw[g * SW + tis] = p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 acc[GQA];
    for (uint g = 0; g < GQA; ++g) acc[g] = 0;
    const uint cn = min((uint) SW, a.nk - chunk * SW);
    for (uint jj = 0; jj < cn; ++jj) {
        const uint kp = a.lo + chunk * SW + jj;
        device const half4 * vr = (device const half4 *) (Vc + (ulong) (kp & a.slotMask) * DKV + hk * DH);
        const float4 v = float4(vr[tis]);
        for (uint g = 0; g < GQA; ++g) acc[g] += pw[g * SW + jj] * v;
    }
    for (uint g = 0; g < GQA; ++g) {
        device float * o = part + ((ulong) (hk * GQA + g) * a.nChunks + chunk) * PSZ;
        if (tis == 0) { o[0] = m[g]; o[1] = S[g]; }
        o[4 + tis * 4] = acc[g].x; o[5 + tis * 4] = acc[g].y; o[6 + tis * 4] = acc[g].z; o[7 + tis * 4] = acc[g].w;
    }
}

// merge up to 32 partials into one (grid: nOut x NH SIMD groups).  The final level folds in the
// attention sink, normalizes and applies the output gate.
struct DMergeArgs { uint nIn; uint nOut; uint final; uint sinkCol; uint gateCol; uint pad0; uint pad1; uint pad2; };

kernel void d_attn_merge(constant DMergeArgs & a, device const float * in, device float * out, device const float * P,
                         device float * O, uint3 tg [[threadgroup_position_in_grid]], ushort tis [[thread_index_in_threadgroup]]) {
    threadgroup float fw[SW];
    const uint co = tg.x, h = tg.y;
    const uint c0 = co * SW;
    const uint cnt = min((uint) SW, a.nIn - c0);
    device const float * src = in + ((ulong) h * a.nIn + c0) * PSZ;
    const bool valid = tis < cnt;
    const float mc = valid ? src[(ulong) tis * PSZ] : -1e30f;
    const float Sc = valid ? src[(ulong) tis * PSZ + 1] : 0.0f;
    float M = simd_max(mc);
    float sink = 0;
    if (a.final) { sink = P[a.sinkCol + h]; M = max(M, sink); }
    const float f = valid ? exp(mc - M) : 0.0f;
    float S = simd_sum(f * Sc);
    fw[tis] = f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float4 acc = 0;
    for (uint c = 0; c < cnt; ++c) {
        device const float * p = src + (ulong) c * PSZ + 4 + tis * 4;
        acc += fw[c] * float4(p[0], p[1], p[2], p[3]);
    }
    if (a.final) {
        S += exp(sink - M);
        const uint o = h * DH + tis * 4;
        const float4 g = float4(P[a.gateCol + o], P[a.gateCol + o + 1], P[a.gateCol + o + 2], P[a.gateCol + o + 3]);
        const float4 r = acc / S / (1.0f + exp(-g));
        O[o] = r.x; O[o + 1] = r.y; O[o + 2] = r.z; O[o + 3] = r.w;
    } else {
        device float * o = out + ((ulong) h * a.nOut + co) * PSZ;
        if (tis == 0) { o[0] = M; o[1] = S; }
        o[4 + tis * 4] = acc.x; o[5 + tis * 4] = acc.y; o[6 + tis * 4] = acc.z; o[7 + tis * 4] = acc.w;
    }
}

// shared epilogue: causal short conv with state shift + residual, for output channel r with value y
struct DOutArgs { uint goff; uint mode; float invs; uint stStride; };

static inline void conv_epilogue(const uint r, const float y, constant DOutArgs & a, device const float * convw,
                                 device const float * stIn, device float * stOut, device float * cur) {
    const float s0 = stIn[a.goff + r], s1 = stIn[a.stStride + a.goff + r], s2 = stIn[2 * a.stStride + a.goff + r];
    device const float * w = convw + r * NCONV;
    const float v = (y + w[3] * y + w[2] * s2 + w[1] * s1 + w[0] * s0) * a.invs;
    cur[r] = a.mode ? cur[r] + v : v;
    stOut[a.goff + r] = s1;
    stOut[a.stStride + a.goff + r] = s2;
    stOut[2 * a.stStride + a.goff + r] = y;
}

// attention output projection (FP16, K = DQ) + conv + residual; one SIMD group per row
kernel void d_wo(constant DOutArgs & a, device const half * W, device const float * O, device const float * convw,
                 device const float * stIn, device float * stOut, device float * cur,
                 uint3 tg [[threadgroup_position_in_grid]], ushort tis [[thread_index_in_threadgroup]]) {
    const uint r = tg.x;
    device const half4 * w = (device const half4 *) (W + (ulong) r * DQ);
    device const float4 * x = (device const float4 *) O;
    float s = 0;
    for (uint i = tis; i < DQ / 4; i += SW) s += dot(float4(w[i]), x[i]);
    s = simd_sum(s);
    if (tis == 0) conv_epilogue(r, s, a, convw, stIn, stOut, cur);
}

// router logits: one SIMD group per expert
kernel void d_route_mv(device const float * W, device const float * x, device float * logits,
                       uint3 tg [[threadgroup_position_in_grid]], ushort tis [[thread_index_in_threadgroup]]) {
    device const float4 * w = (device const float4 *) (W + (ulong) tg.x * D);
    device const float4 * xv = (device const float4 *) x;
    float s = 0;
    for (uint i = tis; i < D / 4; i += SW) s += dot(w[i], xv[i]);
    s = simd_sum(s);
    if (tis == 0) logits[tg.x] = s;
}

// top-k routing in one threadgroup of NEXP threads (one per expert): rank by counting larger scores.
// route[]: ids (NUSED uint) | weights * down scale (NUSED float) | gate_up scale (NUSED float)
struct DRouteArgs { float wscale; float eps; uint pad0; uint pad1; };

kernel void d_route_topk(constant DRouteArgs & a, device const float * logits, device const float * bias,
                         device const float * scaleGU, device const float * scaleDN, device uint * routeAll,
                         uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_threadgroup]]) {
    // one threadgroup per token (generation: a single one)
    logits += (ulong) tg.x * NEXP;
    device uint * route = routeAll + (ulong) tg.x * 3 * NUSED;
    threadgroup float sel[NEXP];
    threadgroup float wsel[NUSED];
    threadgroup uint esel[NUSED];
    const uint e = ti;
    const float pr = 1.0f / (1.0f + exp(-logits[e]));
    const float mine = pr + bias[e];
    sel[e] = mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint rank = 0;
    for (uint o = 0; o < NEXP; ++o) {
        const float other = sel[o];
        rank += (other > mine || (other == mine && o < e)) ? 1 : 0;
    }
    if (rank < NUSED) { wsel[rank] = pr; esel[rank] = e; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (ti < NUSED) {
        float sum = 0;
        for (uint k = 0; k < NUSED; ++k) sum += wsel[k];
        const uint ex = esel[ti];
        device float * rf = (device float *) route;
        route[ti] = ex;
        rf[NUSED + ti] = wsel[ti] / (sum + a.eps) * a.wscale * scaleDN[ex];
        rf[2 * NUSED + ti] = scaleGU[ex];
    }
}

static inline float situ(float g, float u, float beta, float lbeta) {
    return beta * tanh(g / beta) / (1.0f + exp(-g)) * lbeta * tanh(u / lbeta);
}

// e2m1 value pairs indexed by a packed byte: .x = low nibble, .y = high nibble
constant float2 kFP4B[256] = {
    float2(0, 0), float2(1, 0), float2(2, 0), float2(3, 0), float2(4, 0), float2(6, 0), float2(8, 0), float2(12, 0),
    float2(0, 0), float2(-1, 0), float2(-2, 0), float2(-3, 0), float2(-4, 0), float2(-6, 0), float2(-8, 0), float2(-12, 0),
    float2(0, 1), float2(1, 1), float2(2, 1), float2(3, 1), float2(4, 1), float2(6, 1), float2(8, 1), float2(12, 1),
    float2(0, 1), float2(-1, 1), float2(-2, 1), float2(-3, 1), float2(-4, 1), float2(-6, 1), float2(-8, 1), float2(-12, 1),
    float2(0, 2), float2(1, 2), float2(2, 2), float2(3, 2), float2(4, 2), float2(6, 2), float2(8, 2), float2(12, 2),
    float2(0, 2), float2(-1, 2), float2(-2, 2), float2(-3, 2), float2(-4, 2), float2(-6, 2), float2(-8, 2), float2(-12, 2),
    float2(0, 3), float2(1, 3), float2(2, 3), float2(3, 3), float2(4, 3), float2(6, 3), float2(8, 3), float2(12, 3),
    float2(0, 3), float2(-1, 3), float2(-2, 3), float2(-3, 3), float2(-4, 3), float2(-6, 3), float2(-8, 3), float2(-12, 3),
    float2(0, 4), float2(1, 4), float2(2, 4), float2(3, 4), float2(4, 4), float2(6, 4), float2(8, 4), float2(12, 4),
    float2(0, 4), float2(-1, 4), float2(-2, 4), float2(-3, 4), float2(-4, 4), float2(-6, 4), float2(-8, 4), float2(-12, 4),
    float2(0, 6), float2(1, 6), float2(2, 6), float2(3, 6), float2(4, 6), float2(6, 6), float2(8, 6), float2(12, 6),
    float2(0, 6), float2(-1, 6), float2(-2, 6), float2(-3, 6), float2(-4, 6), float2(-6, 6), float2(-8, 6), float2(-12, 6),
    float2(0, 8), float2(1, 8), float2(2, 8), float2(3, 8), float2(4, 8), float2(6, 8), float2(8, 8), float2(12, 8),
    float2(0, 8), float2(-1, 8), float2(-2, 8), float2(-3, 8), float2(-4, 8), float2(-6, 8), float2(-8, 8), float2(-12, 8),
    float2(0, 12), float2(1, 12), float2(2, 12), float2(3, 12), float2(4, 12), float2(6, 12), float2(8, 12), float2(12, 12),
    float2(0, 12), float2(-1, 12), float2(-2, 12), float2(-3, 12), float2(-4, 12), float2(-6, 12), float2(-8, 12), float2(-12, 12),
    float2(0, 0), float2(1, 0), float2(2, 0), float2(3, 0), float2(4, 0), float2(6, 0), float2(8, 0), float2(12, 0),
    float2(0, 0), float2(-1, 0), float2(-2, 0), float2(-3, 0), float2(-4, 0), float2(-6, 0), float2(-8, 0), float2(-12, 0),
    float2(0, -1), float2(1, -1), float2(2, -1), float2(3, -1), float2(4, -1), float2(6, -1), float2(8, -1), float2(12, -1),
    float2(0, -1), float2(-1, -1), float2(-2, -1), float2(-3, -1), float2(-4, -1), float2(-6, -1), float2(-8, -1), float2(-12, -1),
    float2(0, -2), float2(1, -2), float2(2, -2), float2(3, -2), float2(4, -2), float2(6, -2), float2(8, -2), float2(12, -2),
    float2(0, -2), float2(-1, -2), float2(-2, -2), float2(-3, -2), float2(-4, -2), float2(-6, -2), float2(-8, -2), float2(-12, -2),
    float2(0, -3), float2(1, -3), float2(2, -3), float2(3, -3), float2(4, -3), float2(6, -3), float2(8, -3), float2(12, -3),
    float2(0, -3), float2(-1, -3), float2(-2, -3), float2(-3, -3), float2(-4, -3), float2(-6, -3), float2(-8, -3), float2(-12, -3),
    float2(0, -4), float2(1, -4), float2(2, -4), float2(3, -4), float2(4, -4), float2(6, -4), float2(8, -4), float2(12, -4),
    float2(0, -4), float2(-1, -4), float2(-2, -4), float2(-3, -4), float2(-4, -4), float2(-6, -4), float2(-8, -4), float2(-12, -4),
    float2(0, -6), float2(1, -6), float2(2, -6), float2(3, -6), float2(4, -6), float2(6, -6), float2(8, -6), float2(12, -6),
    float2(0, -6), float2(-1, -6), float2(-2, -6), float2(-3, -6), float2(-4, -6), float2(-6, -6), float2(-8, -6), float2(-12, -6),
    float2(0, -8), float2(1, -8), float2(2, -8), float2(3, -8), float2(4, -8), float2(6, -8), float2(8, -8), float2(12, -8),
    float2(0, -8), float2(-1, -8), float2(-2, -8), float2(-3, -8), float2(-4, -8), float2(-6, -8), float2(-8, -8), float2(-12, -8),
    float2(0, -12), float2(1, -12), float2(2, -12), float2(3, -12), float2(4, -12), float2(6, -12), float2(8, -12), float2(12, -12),
    float2(0, -12), float2(-1, -12), float2(-2, -12), float2(-3, -12), float2(-4, -12), float2(-6, -12), float2(-8, -12), float2(-12, -12),
};

// dot of one 16-weight NVFP4 sub-block (two packed uints) with x[0..16)
static inline float nv_dot16(uint2 q, device const float4 * x) {
    const float2 p0 = kFP4B[q.x & 0xff], p1 = kFP4B[(q.x >> 8) & 0xff], p2 = kFP4B[(q.x >> 16) & 0xff], p3 = kFP4B[q.x >> 24];
    const float2 p4 = kFP4B[q.y & 0xff], p5 = kFP4B[(q.y >> 8) & 0xff], p6 = kFP4B[(q.y >> 16) & 0xff], p7 = kFP4B[q.y >> 24];
    return dot(float4(p0.x, p1.x, p2.x, p3.x), x[0]) + dot(float4(p4.x, p5.x, p6.x, p7.x), x[1]) +
           dot(float4(p0.y, p1.y, p2.y, p3.y), x[2]) + dot(float4(p4.y, p5.y, p6.y, p7.y), x[3]);
}

// xStride: elements between the input rows of consecutive tokens (multi-token passes; grid y = token * NUSED + slot)
struct DExpArgs { float beta; float lbeta; uint xStride; uint pad1; };

// expert gate/up projection + SiTU activation, NVFP4 weights.  Grid: FF x NUSED SIMD groups; each computes
// the gate row i and the up row FF+i of one selected expert.
kernel void d_gate_up_nv(constant DExpArgs & a, device const uchar * W, device const float * x, device const float * stab,
                         device const uint * route, device float * A,
                         uint3 tg [[threadgroup_position_in_grid]], ushort tis [[thread_index_in_threadgroup]]) {
    const uint i = tg.x, k = tg.y;
    route += (k / NUSED) * 3 * NUSED;
    x += (ulong) (k / NUSED) * a.xStride;
    const uint e = route[k % NUSED];
    const uint nsub = D / 16, rowBytes = D / 16 + D / 2;
    device const uchar * rg = W + ((ulong) e * 2 * FF + i) * rowBytes;
    device const uchar * ru = rg + (ulong) FF * rowBytes;
    device const uint2 * ng = (device const uint2 *) (rg + nsub);
    device const uint2 * nu = (device const uint2 *) (ru + nsub);
    device const float4 * xv = (device const float4 *) x;
    float sg = 0, su = 0;
    for (uint b = tis; b < nsub; b += SW) {
        sg += nv_dot16(ng[b], xv + 4 * b) * stab[rg[b]];
        su += nv_dot16(nu[b], xv + 4 * b) * stab[ru[b]];
    }
    sg = simd_sum(sg);
    su = simd_sum(su);
    if (tis == 0) {
        const float s = ((device const float *) route)[2 * NUSED + k % NUSED];
        A[k * FF + i] = situ(sg * s, su * s, a.beta, a.lbeta);
    }
}

// same with FP16 weights
kernel void d_gate_up_h(constant DExpArgs & a, device const half * W, device const float * x,
                        device const uint * route, device float * A,
                        uint3 tg [[threadgroup_position_in_grid]], ushort tis [[thread_index_in_threadgroup]]) {
    const uint i = tg.x, k = tg.y;
    route += (k / NUSED) * 3 * NUSED;
    x += (ulong) (k / NUSED) * a.xStride;
    const uint e = route[k % NUSED];
    device const half4 * wg = (device const half4 *) (W + ((ulong) e * 2 * FF + i) * D);
    device const half4 * wu = wg + (ulong) FF * D / 4;
    device const float4 * xv = (device const float4 *) x;
    float sg = 0, su = 0;
    for (uint b = tis; b < D / 4; b += SW) {
        const float4 v = xv[b];
        sg += dot(float4(wg[b]), v);
        su += dot(float4(wu[b]), v);
    }
    sg = simd_sum(sg);
    su = simd_sum(su);
    if (tis == 0) {
        const float s = ((device const float *) route)[2 * NUSED + k % NUSED];
        A[k * FF + i] = situ(sg * s, su * s, a.beta, a.lbeta);
    }
}

// expert down projection, stage 1: one SIMD group per (row pair, selected expert) -> unweighted partials [NUSED][D]
kernel void d_down_nv(constant DOutArgs & a, device const uchar * W, device const float * A, device const float * stab,
                      device const uint * route, device float * part,
                      uint3 tg [[threadgroup_position_in_grid]], ushort tis [[thread_index_in_threadgroup]]) {
    const uint r0 = tg.x * 2, k = tg.y;
    const uint nsub = FF / 16, rowBytes = FF / 16 + FF / 2;
    device const uchar * rowA = W + ((ulong) route[(k / NUSED) * 3 * NUSED + k % NUSED] * D + r0) * rowBytes;
    device const uchar * rowB = rowA + rowBytes;
    device const uint2 * na = (device const uint2 *) (rowA + nsub);
    device const uint2 * nb = (device const uint2 *) (rowB + nsub);
    device const float4 * av = (device const float4 *) (A + k * FF);
    float sa = 0, sb = 0;
    for (uint b = tis; b < nsub; b += SW) {
        sa += nv_dot16(na[b], av + 4 * b) * stab[rowA[b]];
        sb += nv_dot16(nb[b], av + 4 * b) * stab[rowB[b]];
    }
    sa = simd_sum(sa);
    sb = simd_sum(sb);
    if (tis == 0) { part[k * D + r0] = sa; part[k * D + r0 + 1] = sb; }
}

kernel void d_down_h(constant DOutArgs & a, device const half * W, device const float * A,
                     device const uint * route, device float * part,
                     uint3 tg [[threadgroup_position_in_grid]], ushort tis [[thread_index_in_threadgroup]]) {
    const uint r0 = tg.x * 2, k = tg.y;
    device const half4 * wa = (device const half4 *) (W + ((ulong) route[(k / NUSED) * 3 * NUSED + k % NUSED] * D + r0) * FF);
    device const half4 * wb = wa + FF / 4;
    device const float4 * av = (device const float4 *) (A + k * FF);
    float sa = 0, sb = 0;
    for (uint b = tis; b < FF / 4; b += SW) {
        const float4 v = av[b];
        sa += dot(float4(wa[b]), v);
        sb += dot(float4(wb[b]), v);
    }
    sa = simd_sum(sa);
    sb = simd_sum(sb);
    if (tis == 0) { part[k * D + r0] = sa; part[k * D + r0 + 1] = sb; }
}

// stage 2: weighted sum over the experts + conv + residual, one thread per output row
kernel void d_down_fin(constant DOutArgs & a, device const float * part, device const uint * route, device const float * convw,
                       device const float * stIn, device float * stOut, device float * cur, uint gid [[thread_position_in_grid]]) {
    if (gid >= D) return;
    device const float * w = (device const float *) route + NUSED;
    float y = 0;
    for (uint k = 0; k < NUSED; ++k) y += part[k * D + gid] * w[k];
    conv_epilogue(gid, y, a, convw, stIn, stOut, cur);
}

// output head (FP16) with soft cap; two vocabulary rows per SIMD group
struct DLogitArgs { uint rows; float cap; };

kernel void d_logits(constant DLogitArgs & a, device const half * W, device const float * x, device float * logits,
                     uint3 tg [[threadgroup_position_in_grid]], ushort tis [[thread_index_in_threadgroup]]) {
    const uint r0 = tg.x * 2;
    device const half4 * w0 = (device const half4 *) (W + (ulong) r0 * D);
    device const half4 * w1 = w0 + D / 4;
    device const float4 * xv = (device const float4 *) x;
    float s0 = 0, s1 = 0;
    for (uint i = tis; i < D / 4; i += SW) {
        const float4 v = xv[i];
        s0 += dot(float4(w0[i]), v);
        s1 += dot(float4(w1[i]), v);
    }
    s0 = simd_sum(s0);
    s1 = simd_sum(s1);
    if (tis == 0) {
        logits[r0] = a.cap > 0 ? a.cap * tanh(s0 / a.cap) : s0;
        logits[r0 + 1] = a.cap > 0 ? a.cap * tanh(s1 / a.cap) : s1;
    }
}

kernel void d_copy(constant CountArgs & a, device const float * src, device float * dst, uint gid [[thread_position_in_grid]]) {
    if (gid < a.w) dst[gid] = src[gid];
}

// ================================================================================================
// 8-bit decode weights ("q8"): rows of [K/32 half scales][K int8], one scale per 32 weights, i.e. GGUF
// Q8_0 blocks de-interleaved.  Q8_0 tensors of the file are used verbatim; 16-bit tensors are converted
// (Q8_0-level error).  Only the bandwidth-bound decode mat-vecs use them.
// ================================================================================================

kernel void k_f16_to_q8(constant NvArgs & a, device const half * src, device uchar * dst, uint2 gid [[thread_position_in_grid]]) {
    const uint nblk = a.K / 32;
    if (gid.x >= nblk || gid.y >= a.rows) return;
    device const half4 * w = (device const half4 *) (src + (ulong) gid.y * a.K + gid.x * 32);
    float4 m = 0;
    for (uint i = 0; i < 8; ++i) m = max(m, abs(float4(w[i])));
    const float mx = max(max(m.x, m.y), max(m.z, m.w));
    device uchar * row = dst + (ulong) gid.y * a.rowBytes;
    const half hs = half(mx / 127.0f);
    ((device half *) row)[gid.x] = hs;
    const float sc = float(hs);
    const float inv = sc > 0 ? 1.0f / sc : 0.0f;
    device packed_char4 * q = (device packed_char4 *) (row + nblk * 2 + gid.x * 32);
    for (uint i = 0; i < 8; ++i) q[i] = char4(clamp(round(float4(w[i]) * inv), -127.0f, 127.0f));
}

// dot of one int8 row with x (K multiple of 32); each thread walks 32-weight units
static inline float q8_row_dot(device const uchar * row, const uint K, device const float4 * x, const ushort tis) {
    device const half * sc = (device const half *) row;
    device const packed_char4 * q = (device const packed_char4 *) (row + K / 16);
    float acc = 0;
    for (uint u = tis; u < K / 32; u += SW) {
        float s = 0;
        for (uint i = 0; i < 8; ++i) s += dot(float4(char4(q[u * 8 + i])), x[u * 8 + i]);
        acc += s * float(sc[u]);
    }
    return acc;
}

kernel void d_proj_q8(constant DProjArgs & a, device const uchar * W, device const float * xin, device float * P,
                      uint3 tg [[threadgroup_position_in_grid]], ushort tis [[thread_index_in_threadgroup]]) {
    const uint r0 = tg.x * 2;
    const uint sel = r0 < a.kCol ? 0 : (r0 < a.vCol ? 1 : 2);
    device const float4 * x = (device const float4 *) (xin + sel * a.xStride);
    const uint rowBytes = D + D / 16;
    const float s0 = simd_sum(q8_row_dot(W + (ulong) r0 * rowBytes, D, x, tis));
    const float s1 = simd_sum(q8_row_dot(W + (ulong) (r0 + 1) * rowBytes, D, x, tis));
    if (tis == 0) { P[r0] = s0; P[r0 + 1] = s1; }
}

kernel void d_wo_q8(constant DOutArgs & a, device const uchar * W, device const float * O, device const float * convw,
                    device const float * stIn, device float * stOut, device float * cur,
                    uint3 tg [[threadgroup_position_in_grid]], ushort tis [[thread_index_in_threadgroup]]) {
    const uint r = tg.x;
    const float s = simd_sum(q8_row_dot(W + (ulong) r * (DQ + DQ / 16), DQ, (device const float4 *) O, tis));
    if (tis == 0) conv_epilogue(r, s, a, convw, stIn, stOut, cur);
}

kernel void d_logits_q8(constant DLogitArgs & a, device const uchar * W, device const float * x, device float * logits,
                        uint3 tg [[threadgroup_position_in_grid]], ushort tis [[thread_index_in_threadgroup]]) {
    const uint r0 = tg.x * 2;
    const uint rowBytes = D + D / 16;
    device const float4 * xv = (device const float4 *) x;
    const float s0 = simd_sum(q8_row_dot(W + (ulong) r0 * rowBytes, D, xv, tis));
    const float s1 = simd_sum(q8_row_dot(W + (ulong) (r0 + 1) * rowBytes, D, xv, tis));
    if (tis == 0) {
        logits[r0] = a.cap > 0 ? a.cap * tanh(s0 / a.cap) : s0;
        logits[r0 + 1] = a.cap > 0 ? a.cap * tanh(s1 / a.cap) : s1;
    }
}

kernel void d_gate_up_q8(constant DExpArgs & a, device const uchar * W, device const float * x,
                         device const uint * route, device float * A,
                         uint3 tg [[threadgroup_position_in_grid]], ushort tis [[thread_index_in_threadgroup]]) {
    const uint i = tg.x, k = tg.y;
    route += (k / NUSED) * 3 * NUSED;
    x += (ulong) (k / NUSED) * a.xStride;
    const uint rowBytes = D + D / 16;
    device const uchar * rg = W + ((ulong) route[k % NUSED] * 2 * FF + i) * rowBytes;
    device const float4 * xv = (device const float4 *) x;
    const float sg = simd_sum(q8_row_dot(rg, D, xv, tis));
    const float su = simd_sum(q8_row_dot(rg + (ulong) FF * rowBytes, D, xv, tis));
    if (tis == 0) {
        const float s = ((device const float *) route)[2 * NUSED + k % NUSED];
        A[k * FF + i] = situ(sg * s, su * s, a.beta, a.lbeta);
    }
}

kernel void d_down_q8(constant DOutArgs & a, device const uchar * W, device const float * A,
                      device const uint * route, device float * part,
                      uint3 tg [[threadgroup_position_in_grid]], ushort tis [[thread_index_in_threadgroup]]) {
    const uint r0 = tg.x * 2, k = tg.y;
    const uint rowBytes = FF + FF / 16;
    device const uchar * rowA = W + ((ulong) route[(k / NUSED) * 3 * NUSED + k % NUSED] * D + r0) * rowBytes;
    device const float4 * av = (device const float4 *) (A + k * FF);
    const float sa = simd_sum(q8_row_dot(rowA, FF, av, tis));
    const float sb = simd_sum(q8_row_dot(rowA + rowBytes, FF, av, tis));
    if (tis == 0) { part[k * D + r0] = sa; part[k * D + r0 + 1] = sb; }
}


// ================================================================================================
// experts in GGUF Q4_K / Q5_0, read in place by the decode mat-vecs (no 8-bit copies needed).
// Both are repacked per row at load time into 32-weight units of 16 bytes: byte j holds weight j
// (low nibble) and weight 16 + j (high nibble).
//   Q4_K row: [K/256 x (d, dmin) halves][K/32 x (scale, min) bytes][K/2 nibble bytes];  w = d*scale*q - dmin*min
//   Q5_0 row: [K/32 halves d][K/32 uints of high bits][K/2 nibble bytes];                w = d*(q - 16)
// ================================================================================================

// sum of (a*q + b) * x over one 32-weight unit; `hi` carries a fifth bit per weight (0 for 4-bit data)
static inline float unit_dot32(device const uint * nib, const uint hi, const float a, const float b, device const float4 * x) {
    float s = 0;
    for (uint w = 0; w < 4; ++w) {
        const uint u = nib[w];
        const uint4 sh = uint4(0, 1, 2, 3) + 4 * w;
        const float4 lo = float4(as_type<uchar4>(u & 0x0F0F0F0Fu)) + float4((uint4(hi) >> sh) & 1u) * 16.0f;
        const float4 hh = float4(as_type<uchar4>((u >> 4) & 0x0F0F0F0Fu)) + float4((uint4(hi) >> (sh + 16)) & 1u) * 16.0f;
        s += dot(lo * a + b, x[w]) + dot(hh * a + b, x[4 + w]);
    }
    return s;
}

static inline float q4k_row_dot(device const uchar * row, const uint K, device const float4 * x, const ushort tis) {
    const uint nu = K / 32;
    device const half * dm = (device const half *) row;
    device const uchar * sm = row + K / 64;
    device const uint * nib = (device const uint *) (row + K / 64 + nu * 2);
    float acc = 0;
    for (uint u = tis; u < nu; u += SW) {
        const float a = float(dm[2 * (u / 8)]) * float(sm[2 * u]);
        const float b = -float(dm[2 * (u / 8) + 1]) * float(sm[2 * u + 1]);
        acc += unit_dot32(nib + u * 4, 0u, a, b, x + u * 8);
    }
    return acc;
}

static inline float q50_row_dot(device const uchar * row, const uint K, device const float4 * x, const ushort tis) {
    const uint nu = K / 32;
    device const half * d = (device const half *) row;
    device const uint * qh = (device const uint *) (row + nu * 2);
    device const uint * nib = qh + nu;
    float acc = 0;
    for (uint u = tis; u < nu; u += SW) {
        const float a = float(d[u]);
        acc += unit_dot32(nib + u * 4, qh[u], a, -16.0f * a, x + u * 8);
    }
    return acc;
}

#define Q4K_ROW(K) ((K) / 64 + (K) / 16 + (K) / 2)
#define Q50_ROW(K) ((K) / 16 + (K) / 8 + (K) / 2)

#define GATE_UP_KERNEL(NAME, DOT, ROWB) \
kernel void NAME(constant DExpArgs & a, device const uchar * W, device const float * x, \
                 device const uint * route, device float * A, \
                 uint3 tg [[threadgroup_position_in_grid]], ushort tis [[thread_index_in_threadgroup]]) { \
    const uint i = tg.x, k = tg.y; \
    route += (k / NUSED) * 3 * NUSED; \
    x += (ulong) (k / NUSED) * a.xStride; \
    const uint rowBytes = ROWB(D); \
    device const uchar * rg = W + ((ulong) route[k % NUSED] * 2 * FF + i) * rowBytes; \
    device const float4 * xv = (device const float4 *) x; \
    const float sg = simd_sum(DOT(rg, D, xv, tis)); \
    const float su = simd_sum(DOT(rg + (ulong) FF * rowBytes, D, xv, tis)); \
    if (tis == 0) { \
        const float s = ((device const float *) route)[2 * NUSED + k % NUSED]; \
        A[k * FF + i] = situ(sg * s, su * s, a.beta, a.lbeta); \
    } \
}

#define DOWN_KERNEL(NAME, DOT, ROWB) \
kernel void NAME(constant DOutArgs & a, device const uchar * W, device const float * A, \
                 device const uint * route, device float * part, \
                 uint3 tg [[threadgroup_position_in_grid]], ushort tis [[thread_index_in_threadgroup]]) { \
    const uint r0 = tg.x * 2, k = tg.y; \
    const uint rowBytes = ROWB(FF); \
    device const uchar * rowA = W + ((ulong) route[(k / NUSED) * 3 * NUSED + k % NUSED] * D + r0) * rowBytes; \
    device const float4 * av = (device const float4 *) (A + k * FF); \
    const float sa = simd_sum(DOT(rowA, FF, av, tis)); \
    const float sb = simd_sum(DOT(rowA + rowBytes, FF, av, tis)); \
    if (tis == 0) { part[k * D + r0] = sa; part[k * D + r0 + 1] = sb; } \
}

GATE_UP_KERNEL(d_gate_up_q4k, q4k_row_dot, Q4K_ROW)
GATE_UP_KERNEL(d_gate_up_q50, q50_row_dot, Q50_ROW)
DOWN_KERNEL(d_down_q4k, q4k_row_dot, Q4K_ROW)
DOWN_KERNEL(d_down_q50, q50_row_dot, Q50_ROW)
