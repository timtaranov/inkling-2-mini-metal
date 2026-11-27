#include <metal_stdlib>
using namespace metal;
struct A { uint n; uint pad; };
// one simdgroup, n elements strided
kernel void tiny(constant A & a, device const float * x, device float * y, ushort ti [[thread_index_in_threadgroup]]) {
    float s = 0; for (uint i = ti; i < a.n; i += 32) s += x[i] * x[i];
    s = simd_sum(s);
    for (uint i = ti; i < a.n; i += 32) y[i] = x[i] * s * 1e-6f;
}
// one threadgroup of 768 threads, 3 elements each, cross-simdgroup reduction through threadgroup memory
kernel void tg768(constant A & a, device const float * x, device float * y, ushort ti [[thread_index_in_threadgroup]],
                  ushort sg [[simdgroup_index_in_threadgroup]], ushort tis [[thread_index_in_simdgroup]]) {
    threadgroup float part[32];
    float3 v = float3(x[ti * 3], x[ti * 3 + 1], x[ti * 3 + 2]);
    float s = simd_sum(dot(v, v));
    if (tis == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0; for (uint i = 0; i < 24; ++i) tot += part[i];
    v *= tot * 1e-6f;
    y[ti * 3] = v.x; y[ti * 3 + 1] = v.y; y[ti * 3 + 2] = v.z;
}
// same with three reduction rounds
kernel void tg768x3(constant A & a, device const float * x, device float * y, ushort ti [[thread_index_in_threadgroup]],
                  ushort sg [[simdgroup_index_in_threadgroup]], ushort tis [[thread_index_in_simdgroup]]) {
    threadgroup float part[32];
    float3 v = float3(x[ti * 3], x[ti * 3 + 1], x[ti * 3 + 2]);
    for (uint r = 0; r < 3; ++r) {
        float s = simd_sum(dot(v, v));
        if (tis == 0) part[sg] = s;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot = 0; for (uint i = 0; i < 24; ++i) tot += part[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        v *= rsqrt(tot / 2304 + 1e-6f);
    }
    y[ti * 3] = v.x; y[ti * 3 + 1] = v.y; y[ti * 3 + 2] = v.z;
}
