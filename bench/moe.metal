#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
struct MoeArgs { uint M; uint K; uint nTiles; uint swap; };
#define MOE(NAME, TN, TM, SG) \
kernel void NAME(constant MoeArgs & a, device const half * Xg, device const half * W, device float * H, \
                 device const uint * tiles, uint3 tg [[threadgroup_position_in_grid]]) { \
    const uint t = a.swap ? tg.x : tg.y, mt = a.swap ? tg.y : tg.x; \
    const uint e = tiles[3 * t], r0 = tiles[3 * t + 1]; \
    const int tn = (int) tiles[3 * t + 2]; \
    const int m0 = mt * TM; \
    const int tm = min((int) TM, (int) a.M - m0); \
    auto tB = tensor((device half *) Xg + (ulong) r0 * a.K, dextents<int32_t, 2>((int) a.K, tn), array<int, 2>({1, (int) a.K})); \
    auto tA = tensor((device half *) W + ((ulong) e * a.M + m0) * a.K, dextents<int32_t, 2>((int) a.K, tm), array<int, 2>({1, (int) a.K})); \
    mpp::tensor_ops::matmul2d<mpp::tensor_ops::matmul2d_descriptor(TN, TM, static_cast<int>(dynamic_extent), false, true, false, \
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply), execution_simdgroups<SG>> mm; \
    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>(); \
    mm.run(tB, tA, cT); \
    auto tD = tensor(H + (ulong) r0 * a.M + m0, dextents<int32_t, 2>(tm, tn), array<int, 2>({1, (int) a.M})); \
    cT.store(tD); \
}
MOE(m_64_64_4, 64, 64, 4)
MOE(m_128_32_4, 128, 32, 4)
MOE(m_128_64_8, 128, 64, 8)
MOE(m_256_32_8, 256, 32, 8)
MOE(m_256_16_4, 256, 16, 4)
MOE(m_128_16_4, 128, 16, 4)
MOE(m_192_32_8, 192, 32, 8)
MOE(m_256_64_16, 256, 64, 16)
MOE(m_512_16_8, 512, 16, 8)
