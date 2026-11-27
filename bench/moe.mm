#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <vector>
#include <chrono>
#include <cstdio>
#include <random>
#include <algorithm>
struct MoeArgs { uint32_t M, K, nTiles, swap; };
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char ** argv) { @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError * err = nil;
    MTLCompileOptions * opt = [MTLCompileOptions new]; opt.languageVersion = MTLLanguageVersion4_0;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithContentsOfFile:@"moe.metal" encoding:NSUTF8StringEncoding error:nil] options:opt error:&err];
    if (!lib) { printf("%s\n", err.description.UTF8String); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    const uint32_t K = 2304, M = 2304, NE = 128, NL = 6;   // NL "layers" of distinct expert banks to defeat caching across reps
    id<MTLBuffer> W = [dev newBufferWithLength:(size_t) NL * NE * M * K * 2 options:MTLResourceStorageModeShared];
    { uint16_t * w = (uint16_t *) W.contents; size_t n = (size_t) NL * NE * M * K; for (size_t i = 0; i < n; i += 1) w[i] = 0x2c00 + (uint16_t)((i * 2654435761u) >> 22); }
    for (int N : {2048, 4096}) {
        const uint32_t R = N * 8;
        id<MTLBuffer> X = [dev newBufferWithLength:(size_t) R * K * 2 options:MTLResourceStorageModeShared];
        { uint16_t * x = (uint16_t *) X.contents; for (size_t i = 0; i < (size_t) R * K; ++i) x[i] = 0x3800 + (uint16_t)((i * 40503u) >> 6 & 0x3ff); }
        id<MTLBuffer> H = [dev newBufferWithLength:(size_t) R * M * 4 options:MTLResourceStorageModeShared];
        // routing: each token picks 8 distinct experts, mildly skewed
        std::mt19937 rng(1); std::vector<int> cnt(NE, 0);
        std::vector<double> wts(NE); for (int e = 0; e < (int) NE; ++e) wts[e] = 1.0 / (1.0 + 0.02 * e);
        for (int t = 0; t < N; ++t) { std::vector<double> w = wts; for (int k = 0; k < 8; ++k) { std::discrete_distribution<int> d(w.begin(), w.end()); int e = d(rng); cnt[e]++; w[e] = 0; } }
        struct V { const char * name; int tn, tm, sg; };
        std::vector<V> vs = { {"m_64_64_4", 64, 64, 4}, {"m_128_32_4", 128, 32, 4}, {"m_128_64_8", 128, 64, 8}, {"m_256_32_8", 256, 32, 8}, {"m_256_16_4", 256, 16, 4}, {"m_128_16_4", 128, 16, 4}, {"m_192_32_8", 192, 32, 8}, {"m_256_64_16", 256, 64, 16}, {"m_512_16_8", 512, 16, 8} };
        for (auto & v : vs) for (uint32_t swap : {0u, 1u}) {
            std::vector<uint32_t> tiles; uint32_t r = 0, nT = 0;
            for (uint32_t e = 0; e < NE; ++e) { for (int o = 0; o < cnt[e]; o += v.tn) { tiles.push_back(e); tiles.push_back(r + o); tiles.push_back(std::min(v.tn, cnt[e] - o)); ++nT; } r += cnt[e]; }
            id<MTLBuffer> Tb = [dev newBufferWithBytes:tiles.data() length:tiles.size() * 4 options:MTLResourceStorageModeShared];
            id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:v.name]] error:&err];
            if (!ps) { printf("%s: %s\n", v.name, err.description.UTF8String); continue; }
            double best = 1e9;
            for (int rep = 0; rep < 3; ++rep) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:ps];
                for (uint32_t l = 0; l < NL; ++l) {
                    MoeArgs a = { M, K, nT, swap };
                    [enc setBytes:&a length:sizeof(a) atIndex:0];
                    [enc setBuffer:X offset:0 atIndex:1]; [enc setBuffer:W offset:(size_t) l * NE * M * K * 2 atIndex:2]; [enc setBuffer:H offset:0 atIndex:3]; [enc setBuffer:Tb offset:0 atIndex:4];
                    const uint32_t mt = (M + v.tm - 1) / v.tm;
                    [enc dispatchThreadgroups:(swap ? MTLSizeMake(nT, mt, 1) : MTLSizeMake(mt, nT, 1)) threadsPerThreadgroup:MTLSizeMake(32 * v.sg, 1, 1)];
                }
                [enc endEncoding];
                double t0 = now(); [cb commit]; [cb waitUntilCompleted]; double t = now() - t0;
                if (t < best) best = t;
            }
            printf("N=%d %-12s swap=%u tiles=%5u: %.2f ms/layer  %.1f TFLOPS\n", N, v.name, swap, nT, best * 1e3 / NL, 2.0 * R * M * K * NL / best / 1e12);
        }
    }
} return 0; }
