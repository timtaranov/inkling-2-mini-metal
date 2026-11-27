#include <metal_stdlib>
using namespace metal;
struct Args { uint rows; uint K; uint nr; uint rowBytes; };
// row layout: [K/16 scale bytes][K/2 nibble bytes]; within a 16-weight sub-block byte j: lo -> w[j], hi -> w[j+8]
constant float kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

static inline float4 dec_lut(uint4 n) { return float4(kv[n.x], kv[n.y], kv[n.z], kv[n.w]); }
static inline float4 dec_ar(uint4 n) {
    // magnitude index m = n&7: 0,1,2,3,4,6,8,12 ; sign = n>>3
    uint4 m = n & 7u;
    uint4 e = m >> 1;                       // 0..3
    uint4 mag = select(m, (2u + (m & 1u)) << (e - 1u), m >= 4u);
    float4 f = float4(mag);
    return select(f, -f, (n & 8u) != 0u);
}
#define MV_NV(NAME, DEC) \
kernel void NAME(constant Args & a, device const uchar * W, device const half * x, device const float * stab, device float * y, \
                 uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_simdgroup]]) { \
    const uint r0 = tg.x * a.nr; const uint nsub = a.K / 16; \
    float acc[4] = {0, 0, 0, 0}; \
    for (uint r = 0; r < a.nr; ++r) { \
        device const uchar * row = W + (ulong)(r0 + r) * a.rowBytes; \
        device const uint2 * nb = (device const uint2 *)(row + nsub); \
        device const half4 * xv = (device const half4 *) x; \
        float s = 0; \
        for (uint i = ti; i < nsub; i += 32) { \
            uint2 q = nb[i]; \
            uint4 b0 = (uint4(q.x) >> uint4(0, 8, 16, 24)) & 0xffu; \
            uint4 b1 = (uint4(q.y) >> uint4(0, 8, 16, 24)) & 0xffu; \
            float t = dot(DEC(b0 & 15u), (float4) xv[4*i]) + dot(DEC(b1 & 15u), (float4) xv[4*i+1]) \
                    + dot(DEC(b0 >> 4u), (float4) xv[4*i+2]) + dot(DEC(b1 >> 4u), (float4) xv[4*i+3]); \
            s += t * stab[row[i]]; \
        } \
        acc[r] = simd_sum(s); \
    } \
    if (ti == 0) for (uint r = 0; r < a.nr; ++r) y[r0 + r] = acc[r]; \
}
MV_NV(nv_lut, dec_lut)
MV_NV(nv_ar, dec_ar)

// table variant: T[(bytepos)*256 + byte] = kv[lo]*x[j] + kv[hi]*x[j+8]
#define MV_TAB(NAME, TT) \
kernel void NAME(constant Args & a, device const uchar * W, device const TT * T, device const float * stab, device float * y, \
                 uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_simdgroup]]) { \
    const uint r0 = tg.x * a.nr; const uint nsub = a.K / 16; \
    float acc[4] = {0, 0, 0, 0}; \
    for (uint r = 0; r < a.nr; ++r) { \
        device const uchar * row = W + (ulong)(r0 + r) * a.rowBytes; \
        device const uint2 * nb = (device const uint2 *)(row + nsub); \
        float s = 0; \
        for (uint i = ti; i < nsub; i += 32) { \
            uint2 q = nb[i]; \
            device const TT * t = T + i * 8 * 256; \
            float u = (float) t[q.x & 0xff] + (float) t[256 + ((q.x >> 8) & 0xff)] + (float) t[512 + ((q.x >> 16) & 0xff)] + (float) t[768 + (q.x >> 24)] \
                    + (float) t[1024 + (q.y & 0xff)] + (float) t[1280 + ((q.y >> 8) & 0xff)] + (float) t[1536 + ((q.y >> 16) & 0xff)] + (float) t[1792 + (q.y >> 24)]; \
            s += u * stab[row[i]]; \
        } \
        acc[r] = simd_sum(s); \
    } \
    if (ti == 0) for (uint r = 0; r < a.nr; ++r) y[r0 + r] = acc[r]; \
}
MV_TAB(nv_tab32, float)
MV_TAB(nv_tab16, half)
