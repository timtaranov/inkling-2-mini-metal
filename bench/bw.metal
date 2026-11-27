#include <metal_stdlib>
using namespace metal;
struct Args { uint rows; uint K; uint nr; uint pad; };

// variant A: 1 simdgroup per threadgroup, NR rows per threadgroup, half4 reads
kernel void mv_a(constant Args & a, device const half * W, device const half * x, device float * y,
                 uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_simdgroup]]) {
    const uint r0 = tg.x * a.nr;
    float acc[4] = {0, 0, 0, 0};
    for (uint r = 0; r < a.nr; ++r) {
        device const half4 * w = (device const half4 *)(W + (ulong)(r0 + r) * a.K);
        device const half4 * xv = (device const half4 *) x;
        float s = 0;
        for (uint i = ti; i < a.K/4; i += 32) {
            s += dot((float4) w[i], (float4) xv[i]);
        }
        acc[r] = simd_sum(s);
    }
    if (ti == 0) for (uint r = 0; r < a.nr; ++r) y[r0 + r] = acc[r];
}
// variant B: each thread owns a contiguous K/32 chunk of the row (sequential access per thread)
kernel void mv_b(constant Args & a, device const half * W, device const half * x, device float * y,
                 uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_simdgroup]]) {
    const uint r0 = tg.x * a.nr;
    const uint chunk = a.K / 4 / 32;
    float acc[4] = {0, 0, 0, 0};
    for (uint r = 0; r < a.nr; ++r) {
        device const half4 * w = (device const half4 *)(W + (ulong)(r0 + r) * a.K) + ti * chunk;
        device const half4 * xv = (device const half4 *) x + ti * chunk;
        float4 s = 0;
        for (uint i = 0; i < chunk; ++i) {
            s += (float4) w[i] * (float4) xv[i];
        }
        acc[r] = simd_sum(s.x + s.y + s.z + s.w);
    }
    if (ti == 0) for (uint r = 0; r < a.nr; ++r) y[r0 + r] = acc[r];
}
// variant C: one thread per row (no simd reduction), threads in a simdgroup do 32 adjacent rows
kernel void mv_c(constant Args & a, device const half * W, device const half * x, device float * y,
                 uint3 gid [[thread_position_in_grid]]) {
    const uint r = gid.x;
    if (r >= a.rows) return;
    device const half4 * w = (device const half4 *)(W + (ulong) r * a.K);
    device const half4 * xv = (device const half4 *) x;
    float4 s0 = 0, s1 = 0;
    for (uint i = 0; i < a.K/4; i += 2) {
        s0 += (float4) w[i] * (float4) xv[i];
        s1 += (float4) w[i+1] * (float4) xv[i+1];
    }
    s0 += s1;
    y[r] = s0.x + s0.y + s0.z + s0.w;
}
// variant D: int8 weights, per-row scale applied outside; 1 simdgroup, nr rows
kernel void mv_i8(constant Args & a, device const char * W, device const half * x, device float * y,
                 uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_simdgroup]]) {
    const uint r0 = tg.x * a.nr;
    float acc[4] = {0, 0, 0, 0};
    for (uint r = 0; r < a.nr; ++r) {
        device const packed_char4 * w = (device const packed_char4 *)(W + (ulong)(r0 + r) * a.K);
        device const half4 * xv = (device const half4 *) x;
        float s = 0;
        for (uint i = ti; i < a.K/4; i += 32) {
            s += dot((float4) char4(w[i]), (float4) xv[i]);
        }
        acc[r] = simd_sum(s);
    }
    if (ti == 0) for (uint r = 0; r < a.nr; ++r) y[r0 + r] = acc[r];
}
