
#include <metal_stdlib>
using namespace metal;
struct Args { uint rows; uint K; uint nr; uint rowBytes; };
constant float kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
constant half kvh[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
constant float2 kBf[256] = {float2(0,0),float2(1,0),float2(2,0),float2(3,0),float2(4,0),float2(6,0),float2(8,0),float2(12,0),float2(0,0),float2(-1,0),float2(-2,0),float2(-3,0),float2(-4,0),float2(-6,0),float2(-8,0),float2(-12,0),float2(0,1),float2(1,1),float2(2,1),float2(3,1),float2(4,1),float2(6,1),float2(8,1),float2(12,1),float2(0,1),float2(-1,1),float2(-2,1),float2(-3,1),float2(-4,1),float2(-6,1),float2(-8,1),float2(-12,1),float2(0,2),float2(1,2),float2(2,2),float2(3,2),float2(4,2),float2(6,2),float2(8,2),float2(12,2),float2(0,2),float2(-1,2),float2(-2,2),float2(-3,2),float2(-4,2),float2(-6,2),float2(-8,2),float2(-12,2),float2(0,3),float2(1,3),float2(2,3),float2(3,3),float2(4,3),float2(6,3),float2(8,3),float2(12,3),float2(0,3),float2(-1,3),float2(-2,3),float2(-3,3),float2(-4,3),float2(-6,3),float2(-8,3),float2(-12,3),float2(0,4),float2(1,4),float2(2,4),float2(3,4),float2(4,4),float2(6,4),float2(8,4),float2(12,4),float2(0,4),float2(-1,4),float2(-2,4),float2(-3,4),float2(-4,4),float2(-6,4),float2(-8,4),float2(-12,4),float2(0,6),float2(1,6),float2(2,6),float2(3,6),float2(4,6),float2(6,6),float2(8,6),float2(12,6),float2(0,6),float2(-1,6),float2(-2,6),float2(-3,6),float2(-4,6),float2(-6,6),float2(-8,6),float2(-12,6),float2(0,8),float2(1,8),float2(2,8),float2(3,8),float2(4,8),float2(6,8),float2(8,8),float2(12,8),float2(0,8),float2(-1,8),float2(-2,8),float2(-3,8),float2(-4,8),float2(-6,8),float2(-8,8),float2(-12,8),float2(0,12),float2(1,12),float2(2,12),float2(3,12),float2(4,12),float2(6,12),float2(8,12),float2(12,12),float2(0,12),float2(-1,12),float2(-2,12),float2(-3,12),float2(-4,12),float2(-6,12),float2(-8,12),float2(-12,12),float2(0,0),float2(1,0),float2(2,0),float2(3,0),float2(4,0),float2(6,0),float2(8,0),float2(12,0),float2(0,0),float2(-1,0),float2(-2,0),float2(-3,0),float2(-4,0),float2(-6,0),float2(-8,0),float2(-12,0),float2(0,-1),float2(1,-1),float2(2,-1),float2(3,-1),float2(4,-1),float2(6,-1),float2(8,-1),float2(12,-1),float2(0,-1),float2(-1,-1),float2(-2,-1),float2(-3,-1),float2(-4,-1),float2(-6,-1),float2(-8,-1),float2(-12,-1),float2(0,-2),float2(1,-2),float2(2,-2),float2(3,-2),float2(4,-2),float2(6,-2),float2(8,-2),float2(12,-2),float2(0,-2),float2(-1,-2),float2(-2,-2),float2(-3,-2),float2(-4,-2),float2(-6,-2),float2(-8,-2),float2(-12,-2),float2(0,-3),float2(1,-3),float2(2,-3),float2(3,-3),float2(4,-3),float2(6,-3),float2(8,-3),float2(12,-3),float2(0,-3),float2(-1,-3),float2(-2,-3),float2(-3,-3),float2(-4,-3),float2(-6,-3),float2(-8,-3),float2(-12,-3),float2(0,-4),float2(1,-4),float2(2,-4),float2(3,-4),float2(4,-4),float2(6,-4),float2(8,-4),float2(12,-4),float2(0,-4),float2(-1,-4),float2(-2,-4),float2(-3,-4),float2(-4,-4),float2(-6,-4),float2(-8,-4),float2(-12,-4),float2(0,-6),float2(1,-6),float2(2,-6),float2(3,-6),float2(4,-6),float2(6,-6),float2(8,-6),float2(12,-6),float2(0,-6),float2(-1,-6),float2(-2,-6),float2(-3,-6),float2(-4,-6),float2(-6,-6),float2(-8,-6),float2(-12,-6),float2(0,-8),float2(1,-8),float2(2,-8),float2(3,-8),float2(4,-8),float2(6,-8),float2(8,-8),float2(12,-8),float2(0,-8),float2(-1,-8),float2(-2,-8),float2(-3,-8),float2(-4,-8),float2(-6,-8),float2(-8,-8),float2(-12,-8),float2(0,-12),float2(1,-12),float2(2,-12),float2(3,-12),float2(4,-12),float2(6,-12),float2(8,-12),float2(12,-12),float2(0,-12),float2(-1,-12),float2(-2,-12),float2(-3,-12),float2(-4,-12),float2(-6,-12),float2(-8,-12),float2(-12,-12)};
static inline float4 dec_lut(uint4 n) { return float4(kv[n.x], kv[n.y], kv[n.z], kv[n.w]); }
static inline half4 dec_luth(uint4 n) { return half4(kvh[n.x], kvh[n.y], kvh[n.z], kvh[n.w]); }
static inline float nvdot(uint2 q, device const float4 * x) {
    uint4 b0 = (uint4(q.x) >> uint4(0, 8, 16, 24)) & 0xffu;
    uint4 b1 = (uint4(q.y) >> uint4(0, 8, 16, 24)) & 0xffu;
    return dot(dec_lut(b0 & 15u), x[0]) + dot(dec_lut(b1 & 15u), x[1]) + dot(dec_lut(b0 >> 4u), x[2]) + dot(dec_lut(b1 >> 4u), x[3]);
}
static inline float nvdoth(uint2 q, device const half4 * x) {
    uint4 b0 = (uint4(q.x) >> uint4(0, 8, 16, 24)) & 0xffu;
    uint4 b1 = (uint4(q.y) >> uint4(0, 8, 16, 24)) & 0xffu;
    return float(dot(dec_luth(b0 & 15u), x[0]) + dot(dec_luth(b1 & 15u), x[1]) + dot(dec_luth(b0 >> 4u), x[2]) + dot(dec_luth(b1 >> 4u), x[3]));
}
static inline float nvdotb(uint2 q, device const float4 * x) {
    const float2 p0 = kBf[q.x & 0xff], p1 = kBf[(q.x >> 8) & 0xff], p2 = kBf[(q.x >> 16) & 0xff], p3 = kBf[q.x >> 24];
    const float2 p4 = kBf[q.y & 0xff], p5 = kBf[(q.y >> 8) & 0xff], p6 = kBf[(q.y >> 16) & 0xff], p7 = kBf[q.y >> 24];
    return dot(float4(p0.x, p1.x, p2.x, p3.x), x[0]) + dot(float4(p4.x, p5.x, p6.x, p7.x), x[1])
         + dot(float4(p0.y, p1.y, p2.y, p3.y), x[2]) + dot(float4(p4.y, p5.y, p6.y, p7.y), x[3]);
}
// pair of rows interleaved (gate_up shape); DOT selects the sub-block kernel, XT the x type
#define PAIR(NAME, DOT, XT) \
kernel void NAME(constant Args & a, device const uchar * W, device const XT * x, device const float * stab, device float * y, \
               uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_simdgroup]]) { \
    const uint r0 = tg.x * 2; const uint nsub = a.K / 16; \
    device const uchar * rowA = W + (ulong) r0 * a.rowBytes; \
    device const uchar * rowB = rowA + a.rowBytes; \
    device const uint2 * na = (device const uint2 *)(rowA + nsub); \
    device const uint2 * nbb = (device const uint2 *)(rowB + nsub); \
    float sa = 0, sb = 0; \
    for (uint i = ti; i < nsub; i += 32) { \
        sa += DOT(na[i], x + 4 * i) * stab[rowA[i]]; \
        sb += DOT(nbb[i], x + 4 * i) * stab[rowB[i]]; \
    } \
    sa = simd_sum(sa); sb = simd_sum(sb); \
    if (ti == 0) { y[r0] = sa; y[r0+1] = sb; } \
}
PAIR(p_lut, nvdot, float4)
PAIR(p_half, nvdoth, half4)
PAIR(p_byte, nvdotb, float4)

// "down" shape A: one simdgroup per output row, 8 experts looped (rows from 8 different matrices)
kernel void dn_a(constant Args & a, device const uchar * W, device const float4 * x, device const float * stab, device float * y,
               uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_simdgroup]]) {
    const uint r = tg.x; const uint nsub = a.K / 16;
    float acc = 0;
    for (uint k = 0; k < 8; ++k) {
        device const uchar * row = W + ((ulong) (k * 13 % 40) * a.rows + r) * a.rowBytes;
        device const uint2 * nb = (device const uint2 *)(row + nsub);
        float s = 0;
        for (uint i = ti; i < nsub; i += 32) s += nvdot(nb[i], x + 4 * i) * stab[row[i]];
        acc += s;
    }
    acc = simd_sum(acc);
    if (ti == 0) y[r] = acc;
}
// "down" shape C: one simdgroup per (row pair, expert)
kernel void dn_c(constant Args & a, device const uchar * W, device const float4 * x, device const float * stab, device float * y,
               uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_simdgroup]]) {
    const uint r0 = tg.x * 2, k = tg.y; const uint nsub = a.K / 16;
    device const uchar * rowA = W + ((ulong) (k * 13 % 40) * a.rows + r0) * a.rowBytes;
    device const uchar * rowB = rowA + a.rowBytes;
    device const uint2 * na = (device const uint2 *)(rowA + nsub);
    device const uint2 * nbb = (device const uint2 *)(rowB + nsub);
    float sa = 0, sb = 0;
    for (uint i = ti; i < nsub; i += 32) {
        sa += nvdot(na[i], x + 4 * i) * stab[rowA[i]];
        sb += nvdot(nbb[i], x + 4 * i) * stab[rowB[i]];
    }
    sa = simd_sum(sa); sb = simd_sum(sb);
    if (ti == 0) { y[k * a.rows + r0] = sa; y[k * a.rows + r0 + 1] = sb; }
}
// shape D: one simdgroup per (4 rows, expert)
kernel void dn_d(constant Args & a, device const uchar * W, device const float4 * x, device const float * stab, device float * y,
               uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_simdgroup]]) {
    const uint r0 = tg.x * 4, k = tg.y; const uint nsub = a.K / 16;
    device const uchar * row0 = W + ((ulong) (k * 13 % 40) * a.rows + r0) * a.rowBytes;
    float s[4] = {0, 0, 0, 0};
    for (uint i = ti; i < nsub; i += 32) {
        for (uint r = 0; r < 4; ++r) {
            device const uchar * row = row0 + r * a.rowBytes;
            s[r] += nvdot(((device const uint2 *)(row + nsub))[i], x + 4 * i) * stab[row[i]];
        }
    }
    for (uint r = 0; r < 4; ++r) { const float v = simd_sum(s[r]); if (ti == 0) y[k * a.rows + r0 + r] = v; }
}
