#include <metal_stdlib>
using namespace metal;
struct A { uint n; uint pad; };
// tiny kernel: one simdgroup, reduce 2304 floats
kernel void tiny(constant A & a, device const float * x, device float * y, ushort ti [[thread_index_in_threadgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
    float s = 0; for (uint i = ti; i < 2304; i += 32) s += x[i] * x[i];
    s = simd_sum(s);
    for (uint i = ti; i < 2304; i += 32) y[i] = x[i] * s * 1e-6f;
}
// chained dependency: y -> x alternate
kernel void wide(constant A & a, device const float * x, device float * y, uint gid [[thread_position_in_grid]]) {
    if (gid < a.n) y[gid] = x[gid] * 1.0001f;
}
