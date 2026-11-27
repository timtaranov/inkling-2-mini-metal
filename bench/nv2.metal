
#include <metal_stdlib>
using namespace metal;
struct Args { uint rows; uint K; uint nr; uint rowBytes; };
constant float kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
constant float2 kB[256] = {float2(0,0),float2(1,0),float2(2,0),float2(3,0),float2(4,0),float2(6,0),float2(8,0),float2(12,0),float2(0,0),float2(-1,0),float2(-2,0),float2(-3,0),float2(-4,0),float2(-6,0),float2(-8,0),float2(-12,0),float2(0,1),float2(1,1),float2(2,1),float2(3,1),float2(4,1),float2(6,1),float2(8,1),float2(12,1),float2(0,1),float2(-1,1),float2(-2,1),float2(-3,1),float2(-4,1),float2(-6,1),float2(-8,1),float2(-12,1),float2(0,2),float2(1,2),float2(2,2),float2(3,2),float2(4,2),float2(6,2),float2(8,2),float2(12,2),float2(0,2),float2(-1,2),float2(-2,2),float2(-3,2),float2(-4,2),float2(-6,2),float2(-8,2),float2(-12,2),float2(0,3),float2(1,3),float2(2,3),float2(3,3),float2(4,3),float2(6,3),float2(8,3),float2(12,3),float2(0,3),float2(-1,3),float2(-2,3),float2(-3,3),float2(-4,3),float2(-6,3),float2(-8,3),float2(-12,3),float2(0,4),float2(1,4),float2(2,4),float2(3,4),float2(4,4),float2(6,4),float2(8,4),float2(12,4),float2(0,4),float2(-1,4),float2(-2,4),float2(-3,4),float2(-4,4),float2(-6,4),float2(-8,4),float2(-12,4),float2(0,6),float2(1,6),float2(2,6),float2(3,6),float2(4,6),float2(6,6),float2(8,6),float2(12,6),float2(0,6),float2(-1,6),float2(-2,6),float2(-3,6),float2(-4,6),float2(-6,6),float2(-8,6),float2(-12,6),float2(0,8),float2(1,8),float2(2,8),float2(3,8),float2(4,8),float2(6,8),float2(8,8),float2(12,8),float2(0,8),float2(-1,8),float2(-2,8),float2(-3,8),float2(-4,8),float2(-6,8),float2(-8,8),float2(-12,8),float2(0,12),float2(1,12),float2(2,12),float2(3,12),float2(4,12),float2(6,12),float2(8,12),float2(12,12),float2(0,12),float2(-1,12),float2(-2,12),float2(-3,12),float2(-4,12),float2(-6,12),float2(-8,12),float2(-12,12),float2(0,0),float2(1,0),float2(2,0),float2(3,0),float2(4,0),float2(6,0),float2(8,0),float2(12,0),float2(0,0),float2(-1,0),float2(-2,0),float2(-3,0),float2(-4,0),float2(-6,0),float2(-8,0),float2(-12,0),float2(0,-1),float2(1,-1),float2(2,-1),float2(3,-1),float2(4,-1),float2(6,-1),float2(8,-1),float2(12,-1),float2(0,-1),float2(-1,-1),float2(-2,-1),float2(-3,-1),float2(-4,-1),float2(-6,-1),float2(-8,-1),float2(-12,-1),float2(0,-2),float2(1,-2),float2(2,-2),float2(3,-2),float2(4,-2),float2(6,-2),float2(8,-2),float2(12,-2),float2(0,-2),float2(-1,-2),float2(-2,-2),float2(-3,-2),float2(-4,-2),float2(-6,-2),float2(-8,-2),float2(-12,-2),float2(0,-3),float2(1,-3),float2(2,-3),float2(3,-3),float2(4,-3),float2(6,-3),float2(8,-3),float2(12,-3),float2(0,-3),float2(-1,-3),float2(-2,-3),float2(-3,-3),float2(-4,-3),float2(-6,-3),float2(-8,-3),float2(-12,-3),float2(0,-4),float2(1,-4),float2(2,-4),float2(3,-4),float2(4,-4),float2(6,-4),float2(8,-4),float2(12,-4),float2(0,-4),float2(-1,-4),float2(-2,-4),float2(-3,-4),float2(-4,-4),float2(-6,-4),float2(-8,-4),float2(-12,-4),float2(0,-6),float2(1,-6),float2(2,-6),float2(3,-6),float2(4,-6),float2(6,-6),float2(8,-6),float2(12,-6),float2(0,-6),float2(-1,-6),float2(-2,-6),float2(-3,-6),float2(-4,-6),float2(-6,-6),float2(-8,-6),float2(-12,-6),float2(0,-8),float2(1,-8),float2(2,-8),float2(3,-8),float2(4,-8),float2(6,-8),float2(8,-8),float2(12,-8),float2(0,-8),float2(-1,-8),float2(-2,-8),float2(-3,-8),float2(-4,-8),float2(-6,-8),float2(-8,-8),float2(-12,-8),float2(0,-12),float2(1,-12),float2(2,-12),float2(3,-12),float2(4,-12),float2(6,-12),float2(8,-12),float2(12,-12),float2(0,-12),float2(-1,-12),float2(-2,-12),float2(-3,-12),float2(-4,-12),float2(-6,-12),float2(-8,-12),float2(-12,-12)};
static inline float4 dec_lut(uint4 n) { return float4(kv[n.x], kv[n.y], kv[n.z], kv[n.w]); }

// V0: current
kernel void v0(constant Args & a, device const uchar * W, device const float * x, device const float * stab, device const float2 * tab2, device float * y,
               uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_simdgroup]]) {
    const uint r0 = tg.x * 2; const uint nsub = a.K / 16;
    float acc[2];
    for (uint r = 0; r < 2; ++r) {
        device const uchar * row = W + (ulong)(r0 + r) * a.rowBytes;
        device const uint2 * nb = (device const uint2 *)(row + nsub);
        device const float4 * xv = (device const float4 *) x;
        float s = 0;
        for (uint i = ti; i < nsub; i += 32) {
            uint2 q = nb[i];
            uint4 b0 = (uint4(q.x) >> uint4(0, 8, 16, 24)) & 0xffu;
            uint4 b1 = (uint4(q.y) >> uint4(0, 8, 16, 24)) & 0xffu;
            float t = dot(dec_lut(b0 & 15u), xv[4*i]) + dot(dec_lut(b1 & 15u), xv[4*i+1]) + dot(dec_lut(b0 >> 4u), xv[4*i+2]) + dot(dec_lut(b1 >> 4u), xv[4*i+3]);
            s += t * stab[row[i]];
        }
        acc[r] = simd_sum(s);
    }
    if (ti == 0) { y[r0] = acc[0]; y[r0+1] = acc[1]; }
}
// byte-pair LUT, generic over the table source
#define BYTELUT(NAME, TAB) \
kernel void NAME(constant Args & a, device const uchar * W, device const float * x, device const float * stab, device const float2 * tab2, device float * y, \
               uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_simdgroup]]) { \
    const uint r0 = tg.x * 2; const uint nsub = a.K / 16; \
    float acc[2]; \
    for (uint r = 0; r < 2; ++r) { \
        device const uchar * row = W + (ulong)(r0 + r) * a.rowBytes; \
        device const uint2 * nb = (device const uint2 *)(row + nsub); \
        device const float4 * xv = (device const float4 *) x; \
        float s = 0; \
        for (uint i = ti; i < nsub; i += 32) { \
            const uint2 q = nb[i]; \
            const float4 x0 = xv[4*i], x1 = xv[4*i+1], x2 = xv[4*i+2], x3 = xv[4*i+3]; \
            const float2 p0 = TAB[q.x & 0xff], p1 = TAB[(q.x >> 8) & 0xff], p2 = TAB[(q.x >> 16) & 0xff], p3 = TAB[q.x >> 24]; \
            const float2 p4 = TAB[q.y & 0xff], p5 = TAB[(q.y >> 8) & 0xff], p6 = TAB[(q.y >> 16) & 0xff], p7 = TAB[q.y >> 24]; \
            float t = dot(float4(p0.x, p1.x, p2.x, p3.x), x0) + dot(float4(p4.x, p5.x, p6.x, p7.x), x1) \
                    + dot(float4(p0.y, p1.y, p2.y, p3.y), x2) + dot(float4(p4.y, p5.y, p6.y, p7.y), x3); \
            s += t * stab[row[i]]; \
        } \
        acc[r] = simd_sum(s); \
    } \
    if (ti == 0) { y[r0] = acc[0]; y[r0+1] = acc[1]; } \
}
BYTELUT(v1, tab2)
BYTELUT(v2, kB)

// V3: V0 with both rows interleaved in one loop (share x loads)
kernel void v3(constant Args & a, device const uchar * W, device const float * x, device const float * stab, device const float2 * tab2, device float * y,
               uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_simdgroup]]) {
    const uint r0 = tg.x * 2; const uint nsub = a.K / 16;
    device const uchar * rowA = W + (ulong) r0 * a.rowBytes;
    device const uchar * rowB = rowA + a.rowBytes;
    device const uint2 * na = (device const uint2 *)(rowA + nsub);
    device const uint2 * nbb = (device const uint2 *)(rowB + nsub);
    device const float4 * xv = (device const float4 *) x;
    float sa = 0, sb = 0;
    for (uint i = ti; i < nsub; i += 32) {
        const float4 x0 = xv[4*i], x1 = xv[4*i+1], x2 = xv[4*i+2], x3 = xv[4*i+3];
        uint2 q = na[i];
        uint4 b0 = (uint4(q.x) >> uint4(0, 8, 16, 24)) & 0xffu;
        uint4 b1 = (uint4(q.y) >> uint4(0, 8, 16, 24)) & 0xffu;
        sa += (dot(dec_lut(b0 & 15u), x0) + dot(dec_lut(b1 & 15u), x1) + dot(dec_lut(b0 >> 4u), x2) + dot(dec_lut(b1 >> 4u), x3)) * stab[rowA[i]];
        q = nbb[i];
        b0 = (uint4(q.x) >> uint4(0, 8, 16, 24)) & 0xffu;
        b1 = (uint4(q.y) >> uint4(0, 8, 16, 24)) & 0xffu;
        sb += (dot(dec_lut(b0 & 15u), x0) + dot(dec_lut(b1 & 15u), x1) + dot(dec_lut(b0 >> 4u), x2) + dot(dec_lut(b1 >> 4u), x3)) * stab[rowB[i]];
    }
    sa = simd_sum(sa); sb = simd_sum(sb);
    if (ti == 0) { y[r0] = sa; y[r0+1] = sb; }
}
// V4: one row per 16 threads (two rows per simdgroup, half the loop depth), reduce with quad-swizzle free trick: use simd_sum on masked values
kernel void v4(constant Args & a, device const uchar * W, device const float * x, device const float * stab, device const float2 * tab2, device float * y,
               uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_simdgroup]]) {
    const uint r0 = tg.x * 2; const uint nsub = a.K / 16;
    const uint r = ti >> 4, t16 = ti & 15;
    device const uchar * row = W + (ulong)(r0 + r) * a.rowBytes;
    device const uint2 * nb = (device const uint2 *)(row + nsub);
    device const float4 * xv = (device const float4 *) x;
    float s = 0;
    for (uint i = t16; i < nsub; i += 16) {
        uint2 q = nb[i];
        uint4 b0 = (uint4(q.x) >> uint4(0, 8, 16, 24)) & 0xffu;
        uint4 b1 = (uint4(q.y) >> uint4(0, 8, 16, 24)) & 0xffu;
        float t = dot(dec_lut(b0 & 15u), xv[4*i]) + dot(dec_lut(b1 & 15u), xv[4*i+1]) + dot(dec_lut(b0 >> 4u), xv[4*i+2]) + dot(dec_lut(b1 >> 4u), xv[4*i+3]);
        s += t * stab[row[i]];
    }
    const float sa = simd_sum(r == 0 ? s : 0.0f), sb = simd_sum(r == 1 ? s : 0.0f);
    if (ti == 0) { y[r0] = sa; y[r0+1] = sb; }
}
// V5: fp16 scales stored separately is emulated by skipping the scale table (upper bound on what removing the dependent load buys)
kernel void v5(constant Args & a, device const uchar * W, device const float * x, device const float * stab, device const float2 * tab2, device float * y,
               uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_simdgroup]]) {
    const uint r0 = tg.x * 2; const uint nsub = a.K / 16;
    float acc[2];
    for (uint r = 0; r < 2; ++r) {
        device const uchar * row = W + (ulong)(r0 + r) * a.rowBytes;
        device const uint2 * nb = (device const uint2 *)(row + nsub);
        device const float4 * xv = (device const float4 *) x;
        float s = 0;
        for (uint i = ti; i < nsub; i += 32) {
            uint2 q = nb[i];
            uint4 b0 = (uint4(q.x) >> uint4(0, 8, 16, 24)) & 0xffu;
            uint4 b1 = (uint4(q.y) >> uint4(0, 8, 16, 24)) & 0xffu;
            float t = dot(dec_lut(b0 & 15u), xv[4*i]) + dot(dec_lut(b1 & 15u), xv[4*i+1]) + dot(dec_lut(b0 >> 4u), xv[4*i+2]) + dot(dec_lut(b1 >> 4u), xv[4*i+3]);
            s += t * float(row[i]);
        }
        acc[r] = simd_sum(s);
    }
    if (ti == 0) { y[r0] = acc[0]; y[r0+1] = acc[1]; }
}
