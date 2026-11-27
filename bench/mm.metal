#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
struct GArgs { uint N; uint M; uint K; uint pad; };
// C[n, m] = sum_k B[n, k] * A[m, k];  B: N x K (activations), A: M x K (weights), C: N x M
#define GEMM(NAME, TN, TM, SG, CT, RELAX) \
kernel void NAME(constant GArgs & a, device const half * Bp, device const half * Ap, device CT * Cp, \
                 uint3 tg [[threadgroup_position_in_grid]]) { \
    const int n0 = tg.y * TN, m0 = tg.x * TM; \
    const int K = a.K; \
    auto tB = tensor((device half *) Bp + (ulong) n0 * K, dextents<int32_t, 2>(K, min((int) TN, (int) a.N - n0)), array<int, 2>({1, K})); \
    auto tA = tensor((device half *) Ap + (ulong) m0 * K, dextents<int32_t, 2>(K, min((int) TM, (int) a.M - m0)), array<int, 2>({1, K})); \
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(TN, TM, static_cast<int>(dynamic_extent), false, true, RELAX, \
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply), execution_simdgroups<SG>> mm; \
    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), CT>(); \
    mm.run(tB, tA, cT); \
    auto tD = tensor(Cp + (ulong) n0 * a.M + m0, dextents<int32_t, 2>(min((int) TM, (int) a.M - m0), min((int) TN, (int) a.N - n0)), array<int, 2>({1, (int) a.M})); \
    cT.store(tD); \
}
GEMM(g_32_64_4_f, 32, 64, 4, float, true)
GEMM(g_32_64_4_h, 32, 64, 4, half, true)
GEMM(g_64_64_4_f, 64, 64, 4, float, true)
GEMM(g_64_64_8_f, 64, 64, 8, float, true)
GEMM(g_32_32_4_f, 32, 32, 4, float, true)
GEMM(g_64_128_8_f, 64, 128, 8, float, true)
GEMM(g_32_64_4_fx, 32, 64, 4, float, false)
GEMM(g_16_64_4_f, 16, 64, 4, float, true)
GEMM(g_32_128_8_f, 32, 128, 8, float, true)
GEMM(g_64_32_4_f, 64, 32, 4, float, true)
GEMM(g_128_32_4_f, 128, 32, 4, float, true)
GEMM(g_128_32_8_f, 128, 32, 8, float, true)
GEMM(g_64_32_4_h, 64, 32, 4, half, true)
GEMM(g_64_32_4_fx, 64, 32, 4, float, false)
GEMM(g_64_32_2_f, 64, 32, 2, float, true)
GEMM(g_64_16_4_f, 64, 16, 4, float, true)
GEMM(g_128_64_8_f, 128, 64, 8, float, true)
GEMM(g_64_64_4_h, 64, 64, 4, half, true)
