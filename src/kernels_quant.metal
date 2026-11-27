
// ================================================================================================
// experts in the low-bit GGUF formats IQ1_S / IQ2_XXS / IQ2_XS / IQ2_S (codebook formats: 8 weights
// per codebook entry) and Q2_0, read in place by the decode mat-vecs.  Rows are repacked at load time
// into separate, aligned arrays (HDR = the row's super-block scales, K/256 halves padded to 4 bytes):
//   IQ2_XXS: [HDR][K/32 x (uint: 4 codebook bytes, uint: 4 x 7-bit sign codes | 4-bit scale)]
//   IQ2_XS:  [HDR][K/32 scale bytes][K/8 x ushort: 9-bit codebook index | 7-bit sign code]
//   IQ2_S:   [HDR][K/32 scale bytes][K/32 bytes of high index bits][K/8 codebook bytes][K/8 sign bytes]
//   IQ1_S:   [HDR][K/32 x ushort: 4 x 3 high index bits | 3-bit scale | offset sign][K/8 codebook bytes]
//   Q2_0:    [K/64 halves d, padded to 4 bytes][K/4 bytes of 2-bit codes];  w = (code - 1) * d
// Each thread walks 8-weight groups g = tis, tis + 32, ...
// ================================================================================================

#define QHDR(K) ((((K) / 256) * 2 + 3) & ~3u)

// a * sum of (+-grid byte) * x over one 8-weight group; bit j of `signs` negates weight j
static inline float grp_dot(const uint2 g, const uint signs, const float a, device const float4 * x) {
    const uint4 sh = uint4(0, 1, 2, 3);
    const float4 s0 = 1.0f - 2.0f * float4((uint4(signs) >> sh) & 1u);
    const float4 s1 = 1.0f - 2.0f * float4((uint4(signs) >> (sh + 4)) & 1u);
    return a * (dot(float4(as_type<uchar4>(g.x)) * s0, x[0]) + dot(float4(as_type<uchar4>(g.y)) * s1, x[1]));
}

static inline float iq2xxs_row_dot(device const uchar * row, const uint K, device const float4 * x, const ushort tis) {
    device const half * d = (device const half *) row;
    device const uint * q = (device const uint *) (row + QHDR(K));
    float acc = 0;
    for (uint g = tis; g < K / 8; g += SW) {
        const uint u = g / 4, l = g % 4;
        const uint a0 = q[2 * u], a1 = q[2 * u + 1];
        const float db = float(d[u / 8]) * (0.5f + float(a1 >> 28)) * 0.25f;
        acc += grp_dot(kIQ2XXS[(a0 >> (8 * l)) & 255u], kSigns[(a1 >> (7 * l)) & 127u], db, x + 2 * g);
    }
    return acc;
}

static inline float iq2xs_row_dot(device const uchar * row, const uint K, device const float4 * x, const ushort tis) {
    device const half * d = (device const half *) row;
    device const uchar * sc = row + QHDR(K);
    device const ushort * q = (device const ushort *) (sc + K / 32);
    float acc = 0;
    for (uint g = tis; g < K / 8; g += SW) {
        const uint u = g / 4, l = g % 4;
        const uint s = (uint(sc[u]) >> (4 * (l / 2))) & 15u;
        const float db = float(d[u / 8]) * (0.5f + float(s)) * 0.25f;
        const uint qq = q[g];
        acc += grp_dot(kIQ2XS[qq & 511u], kSigns[qq >> 9], db, x + 2 * g);
    }
    return acc;
}

static inline float iq2s_row_dot(device const uchar * row, const uint K, device const float4 * x, const ushort tis) {
    device const half * d = (device const half *) row;
    device const uchar * sc = row + QHDR(K);
    device const uchar * qh = sc + K / 32;
    device const uchar * qs = qh + K / 32;
    device const uchar * sg = qs + K / 8;
    float acc = 0;
    for (uint g = tis; g < K / 8; g += SW) {
        const uint u = g / 4, l = g % 4;
        const uint s = (uint(sc[u]) >> (4 * (l / 2))) & 15u;
        const float db = float(d[u / 8]) * (0.5f + float(s)) * 0.25f;
        const uint idx = uint(qs[g]) | ((uint(qh[u]) << (8 - 2 * l)) & 0x300u);
        acc += grp_dot(kIQ2S[idx], sg[g], db, x + 2 * g);
    }
    return acc;
}

static inline float iq1s_row_dot(device const uchar * row, const uint K, device const float4 * x, const ushort tis) {
    device const half * d = (device const half *) row;
    device const ushort * qh = (device const ushort *) (row + QHDR(K));
    device const uchar * qs = (device const uchar *) (qh + K / 32);
    float acc = 0;
    for (uint g = tis; g < K / 8; g += SW) {
        const uint u = g / 4, l = g % 4;
        const uint h = qh[u];
        const float dl = float(d[u / 8]) * float(2 * ((h >> 12) & 7u) + 1);
        const float delta = (h & 0x8000u) ? -0.125f : 0.125f;
        const uint2 gr = kIQ1S[uint(qs[g]) | (((h >> (3 * l)) & 7u) << 8)];
        acc += dl * (dot(float4(as_type<char4>(gr.x)) + delta, x[2 * g]) + dot(float4(as_type<char4>(gr.y)) + delta, x[2 * g + 1]));
    }
    return acc;
}

// Q2_0: each thread walks 16-weight pieces (one uint of codes)
static inline float q20_row_dot(device const uchar * row, const uint K, device const float4 * x, const ushort tis) {
    device const half * d = (device const half *) row;
    device const uint * q = (device const uint *) (row + ((K / 64 * 2 + 3) & ~3u));
    const uint4 sh = uint4(0, 2, 4, 6);
    float acc = 0;
    for (uint p = tis; p < K / 16; p += SW) {
        const uint c = q[p];
        float s = 0;
        for (uint k = 0; k < 4; ++k) s += dot(float4(((uint4(c) >> (sh + 8 * k)) & 3u)) - 1.0f, x[4 * p + k]);
        acc += s * float(d[p / 4]);
    }
    return acc;
}

#define IQ2XXS_ROW(K) (QHDR(K) + (K) / 4)
#define IQ2XS_ROW(K) (QHDR(K) + (K) / 32 + (K) / 4)
#define IQ2S_ROW(K) (QHDR(K) + (K) / 16 + (K) / 4)
#define IQ1S_ROW(K) (QHDR(K) + (K) / 16 + (K) / 8)
#define Q20_ROW(K) ((((K) / 64 * 2 + 3) & ~3u) + (K) / 4)

GATE_UP_KERNEL(d_gate_up_iq2xxs, iq2xxs_row_dot, IQ2XXS_ROW)
GATE_UP_KERNEL(d_gate_up_iq2xs, iq2xs_row_dot, IQ2XS_ROW)
GATE_UP_KERNEL(d_gate_up_iq2s, iq2s_row_dot, IQ2S_ROW)
GATE_UP_KERNEL(d_gate_up_iq1s, iq1s_row_dot, IQ1S_ROW)
GATE_UP_KERNEL(d_gate_up_q20, q20_row_dot, Q20_ROW)
DOWN_KERNEL(d_down_q20, q20_row_dot, Q20_ROW)

// ================================================================================================
// low-memory mode: prompt processing expands the packed experts of a layer to FP16 into a scratch
// buffer right before the layer's GEMMs, instead of keeping FP16 copies of every layer.  One kernel
// per packed row format (see above and kernels.metal); one thread per 32-weight unit unless noted.
// ================================================================================================

static inline void put4(device half * o, const float4 v) { *(device half4 *) o = half4(v); }

#define EXPAND_HEAD(NPER) \
    const uint u = gid.x, r = gid.y; \
    if (u >= a.K / (NPER) || r >= a.rows) return; \
    device const uchar * row = src + (ulong) r * a.rowBytes; \
    device half * o = dst + (ulong) r * a.K + u * (NPER);

kernel void k_x_q8(constant NvArgs & a, device const uchar * src, device half * dst, uint2 gid [[thread_position_in_grid]]) {
    EXPAND_HEAD(32)
    const float s = float(((device const half *) row)[u]);
    device const packed_char4 * q = (device const packed_char4 *) (row + a.K / 16 + u * 32);
    for (uint i = 0; i < 8; ++i) put4(o + 4 * i, float4(char4(q[i])) * s);
}

// 16 nibble bytes of a unit (+ optional fifth bits) -> a*q + b
static inline void put_nib32(device half * o, device const uint * nib, const uint hi, const float a, const float b) {
    for (uint w = 0; w < 4; ++w) {
        const uint v = nib[w];
        const uint4 sh = uint4(0, 1, 2, 3) + 4 * w;
        put4(o + 4 * w, (float4(as_type<uchar4>(v & 0x0F0F0F0Fu)) + float4((uint4(hi) >> sh) & 1u) * 16.0f) * a + b);
        put4(o + 16 + 4 * w, (float4(as_type<uchar4>((v >> 4) & 0x0F0F0F0Fu)) + float4((uint4(hi) >> (sh + 16)) & 1u) * 16.0f) * a + b);
    }
}

kernel void k_x_q4k(constant NvArgs & a, device const uchar * src, device half * dst, uint2 gid [[thread_position_in_grid]]) {
    EXPAND_HEAD(32)
    const uint nu = a.K / 32;
    device const half * dm = (device const half *) row;
    device const uchar * sm = row + a.K / 64;
    device const uint * nib = (device const uint *) (row + a.K / 64 + nu * 2);
    put_nib32(o, nib + u * 4, 0u, float(dm[2 * (u / 8)]) * float(sm[2 * u]), -float(dm[2 * (u / 8) + 1]) * float(sm[2 * u + 1]));
}

kernel void k_x_q50(constant NvArgs & a, device const uchar * src, device half * dst, uint2 gid [[thread_position_in_grid]]) {
    EXPAND_HEAD(32)
    const uint nu = a.K / 32;
    device const uint * qh = (device const uint *) (row + nu * 2);
    const float d = float(((device const half *) row)[u]);
    put_nib32(o, qh + nu + u * 4, qh[u], d, -16.0f * d);
}

static inline void put_grp(device half * o, const uint2 g, const uint signs, const float a) {
    const uint4 sh = uint4(0, 1, 2, 3);
    put4(o, float4(as_type<uchar4>(g.x)) * (1.0f - 2.0f * float4((uint4(signs) >> sh) & 1u)) * a);
    put4(o + 4, float4(as_type<uchar4>(g.y)) * (1.0f - 2.0f * float4((uint4(signs) >> (sh + 4)) & 1u)) * a);
}

kernel void k_x_iq2xxs(constant NvArgs & a, device const uchar * src, device half * dst, uint2 gid [[thread_position_in_grid]]) {
    EXPAND_HEAD(32)
    device const uint * q = (device const uint *) (row + QHDR(a.K));
    const uint a0 = q[2 * u], a1 = q[2 * u + 1];
    const float db = float(((device const half *) row)[u / 8]) * (0.5f + float(a1 >> 28)) * 0.25f;
    for (uint l = 0; l < 4; ++l) put_grp(o + 8 * l, kIQ2XXS[(a0 >> (8 * l)) & 255u], kSigns[(a1 >> (7 * l)) & 127u], db);
}

kernel void k_x_iq2xs(constant NvArgs & a, device const uchar * src, device half * dst, uint2 gid [[thread_position_in_grid]]) {
    EXPAND_HEAD(32)
    device const uchar * sc = row + QHDR(a.K);
    device const ushort * q = (device const ushort *) (sc + a.K / 32);
    const float d = float(((device const half *) row)[u / 8]);
    for (uint l = 0; l < 4; ++l) {
        const float db = d * (0.5f + float((uint(sc[u]) >> (4 * (l / 2))) & 15u)) * 0.25f;
        const uint qq = q[4 * u + l];
        put_grp(o + 8 * l, kIQ2XS[qq & 511u], kSigns[qq >> 9], db);
    }
}

kernel void k_x_iq2s(constant NvArgs & a, device const uchar * src, device half * dst, uint2 gid [[thread_position_in_grid]]) {
    EXPAND_HEAD(32)
    device const uchar * sc = row + QHDR(a.K);
    device const uchar * qh = sc + a.K / 32;
    device const uchar * qs = qh + a.K / 32;
    device const uchar * sg = qs + a.K / 8;
    const float d = float(((device const half *) row)[u / 8]);
    for (uint l = 0; l < 4; ++l) {
        const float db = d * (0.5f + float((uint(sc[u]) >> (4 * (l / 2))) & 15u)) * 0.25f;
        put_grp(o + 8 * l, kIQ2S[uint(qs[4 * u + l]) | ((uint(qh[u]) << (8 - 2 * l)) & 0x300u)], sg[4 * u + l], db);
    }
}

kernel void k_x_iq1s(constant NvArgs & a, device const uchar * src, device half * dst, uint2 gid [[thread_position_in_grid]]) {
    EXPAND_HEAD(32)
    device const ushort * qh = (device const ushort *) (row + QHDR(a.K));
    device const uchar * qs = (device const uchar *) (qh + a.K / 32);
    const uint h = qh[u];
    const float dl = float(((device const half *) row)[u / 8]) * float(2 * ((h >> 12) & 7u) + 1);
    const float delta = (h & 0x8000u) ? -0.125f : 0.125f;
    for (uint l = 0; l < 4; ++l) {
        const uint2 gr = kIQ1S[uint(qs[4 * u + l]) | (((h >> (3 * l)) & 7u) << 8)];
        put4(o + 8 * l, (float4(as_type<char4>(gr.x)) + delta) * dl);
        put4(o + 8 * l + 4, (float4(as_type<char4>(gr.y)) + delta) * dl);
    }
}

// Q2_0: one thread per 16 weights (one uint of 2-bit codes)
kernel void k_x_q20(constant NvArgs & a, device const uchar * src, device half * dst, uint2 gid [[thread_position_in_grid]]) {
    EXPAND_HEAD(16)
    const uint c = ((device const uint *) (row + ((a.K / 64 * 2 + 3) & ~3u)))[u];
    const float d = float(((device const half *) row)[u / 4]);
    const uint4 sh = uint4(0, 2, 4, 6);
    for (uint k = 0; k < 4; ++k) put4(o + 4 * k, (float4((uint4(c) >> (sh + 8 * k)) & 3u) - 1.0f) * d);
}

// ================================================================================================
// optional 8-bit copy of the full-attention K / V caches for generation (--kv8).  At long context most
// of a token's time is reading these caches; 8-bit rows halve that.  Row per position:
// [NKV half scales][DKV int8], one scale per KV head.  Prompt processing keeps using the FP16 caches.
// ================================================================================================

#define KV8_ROW (NKV * 2 + DKV)
struct KV8Args { uint pos0; uint N; uint pad0; uint pad1; };

// FP16 cache rows [pos0, pos0 + N) -> 8-bit rows.  One SIMD group per (KV head, position).
kernel void k_kv_q8(constant KV8Args & a, device const half * src, device uchar * dst,
                    uint3 tg [[threadgroup_position_in_grid]], ushort tis [[thread_index_in_threadgroup]]) {
    const uint hk = tg.x;
    const ulong pos = a.pos0 + tg.y;
    const float4 f = float4(((device const half4 *) (src + pos * DKV + hk * DH))[tis]);
    const float4 af = abs(f);
    const float m = simd_max(max(max(af.x, af.y), max(af.z, af.w)));
    const half hs = half(m / 127.0f);
    const float sc = float(hs);
    const float inv = sc > 0 ? 1.0f / sc : 0.0f;
    device uchar * row = dst + pos * KV8_ROW;
    if (tis == 0) ((device half *) row)[hk] = hs;
    ((device packed_char4 *) (row + NKV * 2 + hk * DH))[tis] = char4(clamp(round(f * inv), -127.0f, 127.0f));
}

// d_attn on the 8-bit caches (linear caches only: key j lives in row a.lo + j)
kernel void d_attn8(constant DAttnArgs & a, device const float * Q, device const float * R, device const float * relbias,
                    device const uchar * Kc, device const uchar * Vc, device float * part,
                    uint3 tg [[threadgroup_position_in_grid]], ushort tis [[thread_index_in_threadgroup]]) {
    threadgroup float pw[GQA * SW];
    const uint chunk = tg.x, hk = tg.y;
    const uint j = chunk * SW + tis;
    const bool valid = j < a.nk;
    float s[GQA];
    for (uint g = 0; g < GQA; ++g) s[g] = 0;
    if (valid) {
        const uint kp = a.lo + j;
        device const uchar * row = Kc + (ulong) kp * KV8_ROW;
        device const packed_char4 * kr = (device const packed_char4 *) (row + NKV * 2 + hk * DH);
        device const float4 * q = (device const float4 *) (Q + hk * GQA * DH);
        for (uint i = 0; i < DH / 4; ++i) {
            const float4 kv = float4(char4(kr[i]));
            for (uint g = 0; g < GQA; ++g) s[g] += dot(kv, q[g * (DH / 4) + i]);
        }
        const float ks = float(((device const half *) row)[hk]) * a.scale;
        const uint d = a.pos - kp;
        if (d < a.extent) {
            device const float4 * rb = (device const float4 *) (relbias + d * NREL);
            device const float4 * r = (device const float4 *) (R + hk * GQA * NREL);
            for (uint g = 0; g < GQA; ++g) {
                float b = 0;
                for (uint i = 0; i < NREL / 4; ++i) b += dot(rb[i], r[g * (NREL / 4) + i]);
                s[g] = s[g] * ks + b;
            }
        } else {
            for (uint g = 0; g < GQA; ++g) s[g] *= ks;
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
        device const uchar * row = Vc + (ulong) (a.lo + chunk * SW + jj) * KV8_ROW;
        const float4 v = float4(char4(((device const packed_char4 *) (row + NKV * 2 + hk * DH))[tis])) * float(((device const half *) row)[hk]);
        for (uint g = 0; g < GQA; ++g) acc[g] += pw[g * SW + jj] * v;
    }
    for (uint g = 0; g < GQA; ++g) {
        device float * o = part + ((ulong) (hk * GQA + g) * a.nChunks + chunk) * PSZ;
        if (tis == 0) { o[0] = m[g]; o[1] = S[g]; }
        o[4 + tis * 4] = acc[g].x; o[5 + tis * 4] = acc[g].y; o[6 + tis * 4] = acc[g].z; o[7 + tis * 4] = acc[g].w;
    }
}


// ================================================================================================
// multi-token generation passes ("mini-batch"): a few tokens (the sampled one plus drafts) go through the
// batch pipeline, but with the generation kernels for the experts (above, grid y = token * NUSED + slot)
// and with the attention below, which reads the long caches once for all tokens.
// ================================================================================================

// weighted sum of the expert outputs of each token: Y[t] = sum_k w[t][k] * part[t * NUSED + k]
kernel void k_combine_mini(constant CountArgs & a, device const float * part, device const uint * route, device float * Y,
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= D || gid.y >= a.w) return;
    device const float * w = (device const float *) (route + gid.y * 3 * NUSED) + NUSED;
    float y = 0;
    for (uint k = 0; k < NUSED; ++k) y += part[(ulong) (gid.y * NUSED + k) * D + gid.x] * w[k];
    Y[(ulong) gid.y * D + gid.x] = y;
}

// conv state after each prefix of the pass: variant z = state after z + 1 tokens (a.mode = floats per variant)
kernel void k_conv_state_var(constant ConvArgs & a, device const float * X, device const float * stIn, device float * stOut,
                             uint3 gid [[thread_position_in_grid]]) {
    const uint i = gid.y, c = gid.x, n = gid.z + 1;
    if (i >= NCONV - 1 || c >= a.C || n > a.N) return;
    const int t = (int) n - (NCONV - 1) + (int) i;
    stOut[(ulong) gid.z * a.mode + (ulong) i * a.stStride + a.goff + c] =
        t >= 0 ? X[(ulong) t * a.ldx + a.xcol + c] : stIn[(ulong) (i + n) * a.stStride + a.goff + c];
}

// attention of N tokens over a linear (full-attention) cache, split over key chunks.  One threadgroup per
// (chunk, KV head) handles the N * GQA query rows that share the KV head; row j = token j / GQA, which sees
// keys <= pos0 + j / GQA.  Partial per row: [max, sum, 0, 0, acc (DH)].
// window = 0: linear cache; otherwise a ring cache (slot = position & slotMask) holding a sliding window, and
// chunks start at position base0 (a multiple of the key tile)
struct MAttnArgs { uint N; uint pos0; uint nChunks; uint chunkKeys; uint extent; uint PW; uint sinkCol; uint gateCol;
                   uint window; uint slotMask; uint base0; uint flags;
                   uint chunk0; uint pad0; uint pad1; uint pad2; };   // flags: 1 = every row sees keys <= pos0, 2 = no sink / gate

kernel void m_attn_chunk(constant MAttnArgs & a, device const half * QG, device const half * Kc, device const half * Vc,
                         device const float * Rc, device const float * relbias, device float * part,
                         uint3 tg [[threadgroup_position_in_grid]], ushort tiisg [[thread_index_in_simdgroup]],
                         ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float ss[AQ * AC];
    threadgroup half sp[AQ * AC];
    threadgroup float sr[AQ];
    threadgroup int sf[1];

    const uint chunk = tg.x, hk = tg.y;
    const ushort tiitg = sgitg * SW + tiisg;
    const uint nq = a.N * GQA;
    const uint pLast = a.pos0 + a.N - 1;

    float M[ANQ], S[ANQ];
    for (uint jj = 0; jj < ANQ; ++jj) { M[jj] = -FLT_MAX / 2; S[jj] = 0; }

    auto tq = tensor((device half *) QG + (ulong) hk * a.N * (GQA * DH), dextents<int32_t, 2>(DH, (int) nq), array<int, 2>({1, DH}));
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

    const uint kEnd = min(a.base0 + (chunk + 1) * a.chunkKeys, pLast + 1);
    int it = 0;
    for (uint kb = a.base0 + chunk * a.chunkKeys; kb < kEnd; kb += AC) {
        ++it;
        const ulong slot = kb & a.slotMask;
        {
            auto tk = tensor((device half *) Kc + slot * DKV + hk * DH, dextents<int32_t, 2>(DH, AC), array<int, 2>({1, DKV}));
            mm_qk.run(tq, tk, ts);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint jj = 0; jj < ANQ; ++jj) {
            const uint j = jj * ANSG + sgitg;
            const uint n = j / GQA, h = hk * GQA + j % GQA;
            const uint p = (a.flags & 1u) ? a.pos0 : a.pos0 + n;
            const uint lo = (a.window && p >= a.window) ? p - (a.window - 1) : 0;
            float s[AC / SW];
            bool ok[AC / SW];
            float m = M[jj];
            for (uint ii = 0; ii < AC / SW; ++ii) {
                const uint c = ii * SW + tiisg;
                const uint kp = kb + c;
                ok[ii] = j < nq && kp <= p && kp >= lo;
                s[ii] = -FLT_MAX / 2;
                if (ok[ii]) {
                    float v = ss[j * AC + c];
                    const uint d = p - kp;
                    if (d < a.extent) {
                        device const float4 * rb = (device const float4 *) (relbias + d * NREL);
                        device const float4 * r = (device const float4 *) (Rc + ((ulong) n * NH + h) * NREL);
                        for (uint i = 0; i < NREL / 4; ++i) v += dot(rb[i], r[i]);
                    }
                    s[ii] = v;
                    m = max(m, v);
                }
            }
            m = simd_max(m);
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

    // chunk0: index of this dispatch's first chunk among the partials (a second dispatch can add chunks from another cache)
    device float * base = part + ((ulong) hk * a.nChunks + a.chunk0 + chunk) * AQ * PSZ;
    for (uint jj = 0; jj < ANQ; ++jj) {
        const uint j = jj * ANSG + sgitg;
        if (tiisg == 0 && j < nq) { base[(ulong) j * PSZ] = M[jj]; base[(ulong) j * PSZ + 1] = S[jj]; }
    }
    for (ushort i = 0; i < co.get_capacity(); ++i) {
        if (co.is_valid_element(i)) {
            const auto idx = co.get_multidimensional_index(i);
            const uint j = idx[1];
            if (j < nq) base[(ulong) j * PSZ + 4 + idx[0]] = co[i];
        }
    }
}

// merge the chunk partials of one (head, token) row, fold in the sink, normalize, apply the output gate
kernel void m_attn_merge(constant MAttnArgs & a, device const float * part, device const float * P, device half * O,
                         uint3 tg [[threadgroup_position_in_grid]], ushort tis [[thread_index_in_threadgroup]]) {
    const uint h = tg.x, n = tg.y;
    const uint hk = h / GQA, j = n * GQA + (h % GQA);
    device const float * base = part + (ulong) hk * a.nChunks * AQ * PSZ + (ulong) j * PSZ;
    const ulong cs = (ulong) AQ * PSZ;
    const bool plain = (a.flags & 2u) != 0;
    const float sink = plain ? -1e30f : P[(ulong) n * a.PW + a.sinkCol + h];
    float M = sink;
    for (uint c = 0; c < a.nChunks; ++c) M = max(M, base[c * cs]);
    float S = exp(sink - M);
    float4 acc = 0;
    for (uint c = 0; c < a.nChunks; ++c) {
        device const float * p = base + c * cs;
        const float f = exp(p[0] - M);
        S += f * p[1];
        acc += f * float4(p[4 + tis * 4], p[5 + tis * 4], p[6 + tis * 4], p[7 + tis * 4]);
    }
    const uint o = h * DH + tis * 4;
    device const float * gp = P + (plain ? 0 : (ulong) n * a.PW + a.gateCol + o);
    const float4 r = plain ? acc / S : acc / S / (1.0f + exp(-float4(gp[0], gp[1], gp[2], gp[3])));
    *(device half4 *) (O + (ulong) n * DQ + o) = half4(r);
}


// ---- wide per-token kernels of the multi-token pass.  Same maths and buffer layouts as the batch kernels
// (k_embed, k_norm_store, k_mix_norm, k_ctrl_*, k_prep_*), but one threadgroup of TGW threads per token
// (3 elements per thread) instead of one SIMD group with long loops: a GPU thread is slow at long loops.

kernel void m_embed(constant EmbedArgs & a, device const int * tok, device const half * emb, device const float * w, device float * cur,
                    uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_threadgroup]],
                    ushort sg [[simdgroup_index_in_threadgroup]], ushort tis [[thread_index_in_simdgroup]]) {
    threadgroup float part[TGSG * RMAX];
    threadgroup float tot[RMAX];
    const uint n = tg.x, i = ti * 3;
    device const half * e = emb + (ulong) tok[n] * D + i;
    const float3 x = float3(e[0], e[1], e[2]);
    float v[1] = {dot(x, x)};
    tg_reduce(v, 1, part, tot, sg, tis, TGSG);
    st3(cur + (ulong) n * D + i, x * (1.0f / sqrt(v[0] / D + a.eps)) * ld3(w + i));
}

kernel void m_norm_store(constant NormArgs & a, device const float * src, device float * dst,
                         uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]], ushort tis [[thread_index_in_simdgroup]]) {
    threadgroup float part[TGSG * RMAX];
    threadgroup float tot[RMAX];
    const uint n = tg.x, i = ti * 3;
    const float3 x = ld3(src + (ulong) n * D + i);
    float v[1] = {dot(x, x)};
    tg_reduce(v, 1, part, tot, sg, tis, TGSG);
    st3(dst + (ulong) n * D + i, x * (1.0f / sqrt(v[0] / D + a.eps)));
}

static inline void sth3(device half * p, float3 v) { p[0] = half(v.x); p[1] = half(v.y); p[2] = half(v.z); }

kernel void m_mix_norm(constant MixArgs & a, device const float * cur, device const float * NB, device const float * queries,
                       device const float * normw, device half * outH, device float * outF,
                       uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_threadgroup]],
                       ushort sg [[simdgroup_index_in_threadgroup]], ushort tis [[thread_index_in_simdgroup]]) {
    threadgroup float part[TGSG * RMAX];
    threadgroup float tot[RMAX];
    const uint n = tg.x, i = ti * 3;
    const float3 c = ld3(cur + (ulong) n * D + i);
    if (a.mode == 0) {
        float v[1] = {dot(c, c)};
        tg_reduce(v, 1, part, tot, sg, tis, TGSG);
        const float3 y = c * (1.0f / sqrt(v[0] / D + a.eps)) * ld3(normw + i);
        sth3(outH + (ulong) n * D + i, y);
        if (a.wantF32) st3(outF + (ulong) n * D + i, y);
        return;
    }
    const uint slots[4] = {a.slot0, a.slot1, a.slot2, a.slot3};
    const uint S = a.nStates, Q = a.nq;
    float3 st[4];
    for (uint s = 0; s < 4; ++s) st[s] = s < S ? ld3(NB + (ulong) slots[s] * a.nbStride + (ulong) n * D + i) : float3(0);
    // v = [ss, dc[q], ds[s][q]]
    float v[RMAX];
    v[0] = dot(c, c);
    for (uint q = 0; q < Q; ++q) {
        const float3 qv = ld3(queries + q * D + i);
        v[1 + q] = dot(c, qv);
        for (uint s = 0; s < S; ++s) v[1 + Q + s * Q + q] = dot(st[s], qv);
    }
    tg_reduce(v, 1 + Q + S * Q, part, tot, sg, tis, TGSG);
    const float inv = 1.0f / sqrt(v[0] / D + a.mixEps);
    float pc[4];
    float ps[4][4];
    for (uint q = 0; q < Q; ++q) {
        const float lc = v[1 + q] * inv * a.qscale;
        float mx = lc;
        float ls[4];
        for (uint s = 0; s < S; ++s) { ls[s] = v[1 + Q + s * Q + q] * a.qscale; mx = max(mx, ls[s]); }
        float den = exp(lc - mx);
        pc[q] = den;
        for (uint s = 0; s < S; ++s) { ps[s][q] = exp(ls[s] - mx); den += ps[s][q]; }
        pc[q] = pc[q] / den * inv;
        for (uint s = 0; s < S; ++s) ps[s][q] /= den;
    }
    float3 m[4];
    float v2[4];
    for (uint q = 0; q < Q; ++q) {
        m[q] = pc[q] * c;
        for (uint s = 0; s < S; ++s) m[q] += ps[s][q] * st[s];
        v2[q] = dot(m[q], m[q]);
    }
    if (a.mode == 2) {
        for (uint q = 0; q < Q; ++q) st3(outF + (ulong) q * a.outStride + (ulong) n * D + i, m[q]);
        return;
    }
    tg_reduce(v2, Q, part, tot, sg, tis, TGSG);
    for (uint q = 0; q < Q; ++q) {
        const float3 y = m[q] * (1.0f / sqrt(v2[q] / D + a.eps)) * ld3(normw + q * a.wStride + i);
        sth3(outH + (ulong) q * a.outStride + (ulong) n * D + i, y);
        if (a.wantF32 && q == 0) st3(outF + (ulong) n * D + i, y);
    }
}

kernel void m_ctrl_gate_in(constant NormArgs & a, device const float * x, device const float * msg, device half * out,
                           uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_threadgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]], ushort tis [[thread_index_in_simdgroup]]) {
    threadgroup float part[TGSG * RMAX];
    threadgroup float tot[RMAX];
    const uint n = tg.x, i = ti * 3;
    const float3 xv = ld3(x + (ulong) n * D + i), mv = ld3(msg + (ulong) n * D + i);
    float v[2] = {dot(xv, xv), dot(mv, mv)};
    tg_reduce(v, 2, part, tot, sg, tis, TGSG);
    device half * o = out + (ulong) n * 2 * D;
    sth3(o + i, xv * (1.0f / sqrt(v[0] / D + a.eps)));
    sth3(o + D + i, mv * (1.0f / sqrt(v[1] / D + a.eps)));
}

kernel void m_ctrl_apply_norm(constant CtrlArgs & a, device const float * x, device const float * msg, device const float * GF,
                              device const float * normw, device half * outH, device float * outF,
                              uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_threadgroup]],
                              ushort sg [[simdgroup_index_in_threadgroup]], ushort tis [[thread_index_in_simdgroup]]) {
    threadgroup float part[TGSG * RMAX];
    threadgroup float tot[RMAX];
    const uint n = tg.x, i = ti * 3;
    const float3 xv = ld3(x + (ulong) n * D + i), mv = ld3(msg + (ulong) n * D + i);
    device const float * gf = GF + (ulong) n * 2 * D;
    const float3 y0 = xv - tanh(ld3(gf + D + i)) * xv + tanh(ld3(gf + i)) * mv;
    float v[1] = {dot(y0, y0)};
    tg_reduce(v, 1, part, tot, sg, tis, TGSG);
    const float3 y = y0 * (1.0f / sqrt(v[0] / D + a.eps)) * ld3(normw + i);
    sth3(outH + (ulong) n * D + i, y);
    if (a.wantF32) st3(outF + (ulong) n * D + i, y);
}

// Q and R post-processing, one SIMD group per (head, token)
kernel void m_prep_q(constant PrepQArgs & a, device const float * P, device const float * qclip, device const float * rclip,
                     device half * Qh, device float * Rc,
                     uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_threadgroup]]) {
    const uint h = tg.x, n = tg.y;
    const float tau = a.useTau ? pos_tau(a.pos0 + n, a.tauAlpha, a.tauFloor) : 1.0f;
    device const float * row = P + (ulong) n * a.PW;
    {
        const uint o = h * DH + ti * 4;
        float4 v = float4(row[a.qCol + o], row[a.qCol + o + 1], row[a.qCol + o + 2], row[a.qCol + o + 3]) *
                   float4(qclip[o], qclip[o + 1], qclip[o + 2], qclip[o + 3]);
        const float ss = simd_sum(dot(v, v));
        const float s = a.qThr / max(sqrt(ss), a.qThr) * tau * a.qScale;
        device half * q = Qh + ((ulong) (h / GQA) * a.N + n) * (GQA * DH) + (h % GQA) * DH + ti * 4;
        v *= s;
        q[0] = half(v.x); q[1] = half(v.y); q[2] = half(v.z); q[3] = half(v.w);
    }
    {
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

// K / V post-processing, one SIMD group per (KV head, token)
kernel void m_prep_kv(constant PrepKVArgs & a, device const float * P, device const float * convw, device const float * state,
                      device half * cache, uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_threadgroup]]) {
    const uint hk = tg.x, n = tg.y;
    const uint slot = (a.pos0 + n) & a.slotMask;
    device const float * row = P + (ulong) n * a.PW;
    const uint c = hk * DH + ti * 4;
    float4 x[NCONV];
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
    y *= a.isV ? 1.0f / sqrt(ss / DH + a.eps) : a.thr / max(sqrt(ss), a.thr);
    device half * o = cache + (ulong) slot * DKV + c;
    o[0] = half(y.x); o[1] = half(y.y); o[2] = half(y.z); o[3] = half(y.w);
}

// ---- mat-vec with FC_NJ right-hand sides (the tokens of the pass): y[t * ldy + r] = <w_r, x_t>.
// One SIMD group per 2 weight rows, each thread walks 32-weight blocks.
struct MMvArgs { uint K; uint ldx; uint ldy; float cap; };

#define MMV_KERNEL(NAME, WDECL, XT, XV4, ROWSETUP, LOADW, SCALE) \
kernel void NAME(constant MMvArgs & a, WDECL, device const XT * x, device float * y, \
                 uint3 tg [[threadgroup_position_in_grid]], ushort tis [[thread_index_in_threadgroup]]) { \
    const uint r0 = tg.x * 2; \
    const uint nb = a.K / 32; \
    float acc[2][8]; \
    for (uint r = 0; r < 2; ++r) { \
        ROWSETUP \
        for (uint t = 0; t < FC_NJ; ++t) acc[r][t] = 0; \
        for (uint u = tis; u < nb; u += SW) { \
            float4 w[8]; \
            for (uint i = 0; i < 8; ++i) w[i] = LOADW; \
            for (uint t = 0; t < FC_NJ; ++t) { \
                device const XV4 * xv = (device const XV4 *) (x + (ulong) t * a.ldx) + u * 8; \
                float4 s4 = 0; \
                for (uint i = 0; i < 8; ++i) s4 += w[i] * float4(xv[i]); \
                acc[r][t] += (s4.x + s4.y + s4.z + s4.w) SCALE; \
            } \
        } \
    } \
    for (uint t = 0; t < FC_NJ; ++t) { \
        float v0 = simd_sum(acc[0][t]), v1 = simd_sum(acc[1][t]); \
        if (a.cap > 0) { v0 = a.cap * tanh(v0 / a.cap); v1 = a.cap * tanh(v1 / a.cap); } \
        if (tis == 0) { y[(ulong) t * a.ldy + r0] = v0; y[(ulong) t * a.ldy + r0 + 1] = v1; } \
    } \
}

#define ROW_Q8 device const uchar * row = W + (ulong) (r0 + r) * (a.K + a.K / 16); device const half * sc = (device const half *) row; \
               device const packed_char4 * q = (device const packed_char4 *) (row + nb * 2);
#define ROW_H  device const half4 * q = (device const half4 *) (W + (ulong) (r0 + r) * a.K);
#define ROW_F  device const float4 * q = (device const float4 *) (W + (ulong) (r0 + r) * a.K);

MMV_KERNEL(m_mv_q8_h, device const uchar * W, half, half4, ROW_Q8, float4(char4(q[u * 8 + i])), * float(sc[u]))
MMV_KERNEL(m_mv_h_h, device const half * W, half, half4, ROW_H, float4(q[u * 8 + i]), )
MMV_KERNEL(m_mv_h_f, device const half * W, float, float4, ROW_H, float4(q[u * 8 + i]), )
MMV_KERNEL(m_mv_f_f, device const float * W, float, float4, ROW_F, q[u * 8 + i], )

// ---- experts of a multi-token pass, grouped: tokens of a pass share many experts, so each distinct expert
// is read once and applied to every token that selected it.  G entry (12 uints) = {expert, n, pair ids...},
// pair id = token * NUSED + slot; entries past the number of distinct experts have n = 0.
#define GSTRIDE 12

struct GroupArgs { uint nPairs; uint pad0; uint pad1; uint pad2; };

kernel void m_group(constant GroupArgs & a, device const uint * route, device uint * G, ushort p [[thread_index_in_threadgroup]]) {
    threadgroup uint ex[64];
    threadgroup uint lead[64];
    const bool live = p < a.nPairs;
    const uint e = live ? route[(p / NUSED) * 3 * NUSED + p % NUSED] : 0xffffffffu;
    ex[p] = e;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool leader = live;
    for (uint q = 0; q < p; ++q) leader = leader && ex[q] != e;
    lead[p] = leader ? 1u : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint gi = 0, total = 0;
    for (uint q = 0; q < 64; ++q) { if (q < p) gi += lead[q]; total += lead[q]; }
    if (leader) {
        device uint * g = G + gi * GSTRIDE;
        uint n = 0;
        for (uint q = p; q < a.nPairs; ++q) {
            if (ex[q] == e) { g[2 + n] = q; ++n; }
        }
        g[0] = e;
        g[1] = n;
    }
    if (live && p >= total) G[p * GSTRIDE + 1] = 0;
}

static inline float nv_row_dot(device const uchar * row, const uint K, device const float4 * x, const ushort tis) {
    const uint nsub = K / 16;
    device const uint2 * nib = (device const uint2 *) (row + nsub);
    float acc = 0;
    for (uint b = tis; b < nsub; b += SW) acc += nv_dot16(nib[b], x + 4 * b) * kUE4M3[row[b]];
    return acc;
}

static inline float half_row_dot(device const uchar * row, const uint K, device const float4 * x, const ushort tis) {
    device const half4 * w = (device const half4 *) row;
    float acc = 0;
    for (uint b = tis; b < K / 4; b += SW) acc += dot(float4(w[b]), x[b]);
    return acc;
}

#define NV_ROW(K) ((K) / 16 + (K) / 2)
#define Q8_ROW(K) ((K) + (K) / 16)
#define H_ROW(K) ((K) * 2)

// A row applied to the n tokens of a group with one decode: units of U float4 are decoded once and
// dotted with each token's input.  tk[j] = input row of token j (rows xs floats apart).
#define ROW_MULTI(NAME, UNITS, U, DECODE) \
static inline void NAME(device const uchar * row, const uint K, device const float * xb, const uint xs, thread uint * tk, \
                        const uint n, const ushort tis, thread float * acc) { \
    for (uint j = 0; j < n; ++j) acc[j] = 0; \
    for (uint u = tis; u < (UNITS); u += SW) { \
        float4 w[U]; \
        DECODE \
        for (uint j = 0; j < n; ++j) { \
            device const float4 * x = (device const float4 *) (xb + (ulong) tk[j] * xs) + u * (U); \
            float s = 0; \
            for (uint i = 0; i < (U); ++i) s += dot(w[i], x[i]); \
            acc[j] += s; \
        } \
    } \
}

ROW_MULTI(nv_row_multi, K / 16, 4,
    const uint2 q = ((device const uint2 *) (row + K / 16))[u];
    const float d = kUE4M3[row[u]];
    const float2 p0 = kFP4B[q.x & 0xff]; const float2 p1 = kFP4B[(q.x >> 8) & 0xff]; const float2 p2 = kFP4B[(q.x >> 16) & 0xff]; const float2 p3 = kFP4B[q.x >> 24];
    const float2 p4 = kFP4B[q.y & 0xff]; const float2 p5 = kFP4B[(q.y >> 8) & 0xff]; const float2 p6 = kFP4B[(q.y >> 16) & 0xff]; const float2 p7 = kFP4B[q.y >> 24];
    w[0] = float4(p0.x, p1.x, p2.x, p3.x) * d; w[1] = float4(p4.x, p5.x, p6.x, p7.x) * d;
    w[2] = float4(p0.y, p1.y, p2.y, p3.y) * d; w[3] = float4(p4.y, p5.y, p6.y, p7.y) * d;)

ROW_MULTI(half_row_multi, K / 32, 8,
    for (uint i = 0; i < 8; ++i) w[i] = float4(((device const half4 *) row)[u * 8 + i]);)

ROW_MULTI(q8_row_multi, K / 32, 8,
    const float sc = float(((device const half *) row)[u]);
    for (uint i = 0; i < 8; ++i) w[i] = float4(char4(((device const packed_char4 *) (row + K / 16))[u * 8 + i])) * sc;)

#define NIB32_DECODE(NIBP, HI, AV, BV) \
    for (uint k = 0; k < 4; ++k) { \
        const uint v = (NIBP)[k]; \
        const uint4 sh = uint4(0, 1, 2, 3) + 4 * k; \
        w[k] = (float4(as_type<uchar4>(v & 0x0F0F0F0Fu)) + float4((uint4(HI) >> sh) & 1u) * 16.0f) * (AV) + (BV); \
        w[4 + k] = (float4(as_type<uchar4>((v >> 4) & 0x0F0F0F0Fu)) + float4((uint4(HI) >> (sh + 16)) & 1u) * 16.0f) * (AV) + (BV); \
    }

ROW_MULTI(q4k_row_multi, K / 32, 8,
    device const half * dm = (device const half *) row;
    device const uchar * sm = row + K / 64;
    const float av = float(dm[2 * (u / 8)]) * float(sm[2 * u]);
    const float bv = -float(dm[2 * (u / 8) + 1]) * float(sm[2 * u + 1]);
    NIB32_DECODE((device const uint *) (row + K / 64 + K / 16) + u * 4, 0u, av, bv))

ROW_MULTI(q50_row_multi, K / 32, 8,
    device const uint * qh = (device const uint *) (row + K / 16);
    const float av = float(((device const half *) row)[u]);
    NIB32_DECODE(qh + K / 32 + u * 4, qh[u], av, -16.0f * av))

ROW_MULTI(q20_row_multi, K / 16, 4,
    const uint c = ((device const uint *) (row + ((K / 64 * 2 + 3) & ~3u)))[u];
    const float d = float(((device const half *) row)[u / 4]);
    for (uint k = 0; k < 4; ++k) w[k] = (float4((uint4(c) >> (uint4(0, 2, 4, 6) + 8 * k)) & 3u) - 1.0f) * d;)

#define GRP_DECODE(GRID, SIGNS, AV) \
    { const uint2 gr = (GRID); const uint sgn = (SIGNS); const uint4 sh = uint4(0, 1, 2, 3); \
      w[0] = float4(as_type<uchar4>(gr.x)) * (1.0f - 2.0f * float4((uint4(sgn) >> sh) & 1u)) * (AV); \
      w[1] = float4(as_type<uchar4>(gr.y)) * (1.0f - 2.0f * float4((uint4(sgn) >> (sh + 4)) & 1u)) * (AV); }

ROW_MULTI(iq2xxs_row_multi, K / 8, 2,
    device const uint * q = (device const uint *) (row + QHDR(K));
    const uint a0 = q[2 * (u / 4)]; const uint a1 = q[2 * (u / 4) + 1]; const uint l = u % 4;
    GRP_DECODE(kIQ2XXS[(a0 >> (8 * l)) & 255u], kSigns[(a1 >> (7 * l)) & 127u],
               float(((device const half *) row)[u / 32]) * (0.5f + float(a1 >> 28)) * 0.25f))

ROW_MULTI(iq2xs_row_multi, K / 8, 2,
    device const uchar * sc = row + QHDR(K);
    const uint qq = ((device const ushort *) (sc + K / 32))[u]; const uint l = u % 4;
    GRP_DECODE(kIQ2XS[qq & 511u], kSigns[qq >> 9],
               float(((device const half *) row)[u / 32]) * (0.5f + float((uint(sc[u / 4]) >> (4 * (l / 2))) & 15u)) * 0.25f))

ROW_MULTI(iq2s_row_multi, K / 8, 2,
    device const uchar * sc = row + QHDR(K);
    device const uchar * qh = sc + K / 32; device const uchar * qs = qh + K / 32; const uint l = u % 4;
    GRP_DECODE(kIQ2S[uint(qs[u]) | ((uint(qh[u / 4]) << (8 - 2 * l)) & 0x300u)], (qs + K / 8)[u],
               float(((device const half *) row)[u / 32]) * (0.5f + float((uint(sc[u / 4]) >> (4 * (l / 2))) & 15u)) * 0.25f))

ROW_MULTI(iq1s_row_multi, K / 8, 2,
    device const ushort * qh = (device const ushort *) (row + QHDR(K));
    const uint h = qh[u / 4]; const uint l = u % 4;
    const float dl = float(((device const half *) row)[u / 32]) * float(2 * ((h >> 12) & 7u) + 1);
    const float delta = (h & 0x8000u) ? -0.125f : 0.125f;
    const uint2 gr = kIQ1S[uint(((device const uchar *) (qh + K / 32))[u]) | (((h >> (3 * l)) & 7u) << 8)];
    w[0] = (float4(as_type<char4>(gr.x)) + delta) * dl; w[1] = (float4(as_type<char4>(gr.y)) + delta) * dl;)

#define GATE_UP_G(NAME, MULTI, ROWB) \
kernel void NAME(constant DExpArgs & a, device const uchar * W, device const float * x, device const uint * route, \
                 device const uint * G, device float * A, \
                 uint3 tg [[threadgroup_position_in_grid]], ushort tis [[thread_index_in_threadgroup]]) { \
    const uint i = tg.x; \
    device const uint * g = G + tg.y * GSTRIDE; \
    const uint n = g[1]; \
    if (n == 0) return; \
    const uint rowBytes = ROWB(D); \
    device const uchar * rg = W + ((ulong) g[0] * 2 * FF + i) * rowBytes; \
    uint tk[8]; \
    float ag[8], au[8]; \
    for (uint j = 0; j < n; ++j) tk[j] = g[2 + j] / NUSED; \
    MULTI(rg, D, x, a.xStride, tk, n, tis, ag); \
    MULTI(rg + (ulong) FF * rowBytes, D, x, a.xStride, tk, n, tis, au); \
    for (uint j = 0; j < n; ++j) { \
        const float sg = simd_sum(ag[j]), su = simd_sum(au[j]); \
        if (tis == 0) { \
            const uint pair = g[2 + j]; \
            const float s = ((device const float *) (route + (pair / NUSED) * 3 * NUSED))[2 * NUSED + pair % NUSED]; \
            A[(ulong) pair * FF + i] = situ(sg * s, su * s, a.beta, a.lbeta); \
        } \
    } \
}

#define DOWN_G(NAME, MULTI, ROWB) \
kernel void NAME(constant DExpArgs & a, device const uchar * W, device const float * A, device const uint * G, device float * part, \
                 uint3 tg [[threadgroup_position_in_grid]], ushort tis [[thread_index_in_threadgroup]]) { \
    const uint r0 = tg.x * 2; \
    device const uint * g = G + tg.y * GSTRIDE; \
    const uint n = g[1]; \
    if (n == 0) return; \
    const uint rowBytes = ROWB(FF); \
    device const uchar * rowA = W + ((ulong) g[0] * D + r0) * rowBytes; \
    uint tk[8]; \
    float sa[8], sb[8]; \
    for (uint j = 0; j < n; ++j) tk[j] = g[2 + j]; \
    MULTI(rowA, FF, A, FF, tk, n, tis, sa); \
    MULTI(rowA + rowBytes, FF, A, FF, tk, n, tis, sb); \
    for (uint j = 0; j < n; ++j) { \
        const float va = simd_sum(sa[j]), vb = simd_sum(sb[j]); \
        if (tis == 0) { part[(ulong) tk[j] * D + r0] = va; part[(ulong) tk[j] * D + r0 + 1] = vb; } \
    } \
}

GATE_UP_G(g_gate_up_nv, nv_row_multi, NV_ROW)
GATE_UP_G(g_gate_up_h, half_row_multi, H_ROW)
GATE_UP_G(g_gate_up_q8, q8_row_multi, Q8_ROW)
GATE_UP_G(g_gate_up_q4k, q4k_row_multi, Q4K_ROW)
GATE_UP_G(g_gate_up_q50, q50_row_multi, Q50_ROW)
GATE_UP_G(g_gate_up_q20, q20_row_multi, Q20_ROW)
GATE_UP_G(g_gate_up_iq2xxs, iq2xxs_row_multi, IQ2XXS_ROW)
GATE_UP_G(g_gate_up_iq2xs, iq2xs_row_multi, IQ2XS_ROW)
GATE_UP_G(g_gate_up_iq2s, iq2s_row_multi, IQ2S_ROW)
GATE_UP_G(g_gate_up_iq1s, iq1s_row_multi, IQ1S_ROW)
DOWN_G(g_down_nv, nv_row_multi, NV_ROW)
DOWN_G(g_down_h, half_row_multi, H_ROW)
DOWN_G(g_down_q8, q8_row_multi, Q8_ROW)
DOWN_G(g_down_q4k, q4k_row_multi, Q4K_ROW)
DOWN_G(g_down_q50, q50_row_multi, Q50_ROW)
DOWN_G(g_down_q20, q20_row_multi, Q20_ROW)
