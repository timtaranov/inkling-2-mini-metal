#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cmath>
struct Args { uint32_t rows, K, nr, rowBytes; };
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main() { @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSString * src = [NSString stringWithContentsOfFile:@"nv.metal" encoding:NSUTF8StringEncoding error:nil];
    NSError * err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:[MTLCompileOptions new] error:&err];
    if (!lib) { printf("compile: %s\n", err.description.UTF8String); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    for (uint32_t K : {2304u, 1152u}) {
        const uint32_t rows = 2304 * 8, nmat = 40, rowBytes = K/16 + K/2;
        size_t mb = (size_t) rows * rowBytes;
        id<MTLBuffer> W = [dev newBufferWithLength:mb * nmat options:MTLResourceStorageModeShared];
        uint8_t * wp = (uint8_t *) W.contents; uint32_t s = 12345;
        for (size_t i = 0; i < mb * nmat; ++i) { s = s * 1664525u + 1013904223u; wp[i] = s >> 24; }
        for (size_t r = 0; r < (size_t) rows * nmat; ++r) for (uint32_t i = 0; i < K/16; ++i) wp[r * rowBytes + i] = 40 + (wp[r * rowBytes + i] & 31);
        id<MTLBuffer> X = [dev newBufferWithLength:K * 2 options:MTLResourceStorageModeShared];
        std::vector<float> xf(K);
        for (uint32_t i = 0; i < K; ++i) { xf[i] = sinf(i * 0.37f); ((_Float16 *) X.contents)[i] = xf[i]; xf[i] = (float)((_Float16 *) X.contents)[i]; }
        id<MTLBuffer> ST = [dev newBufferWithLength:256 * 4 options:MTLResourceStorageModeShared];
        for (int x = 0; x < 256; ++x) { int e = (x >> 3) & 15, m = x & 7; float raw = e == 0 ? ldexpf(m, -9) : ldexpf(1.0f + m / 8.0f, e - 7); ((float *) ST.contents)[x] = (x == 0 || x == 0x7f) ? 0 : raw * 0.5f; }
        static const float kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
        id<MTLBuffer> T32 = [dev newBufferWithLength:(K/2) * 256 * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> T16 = [dev newBufferWithLength:(K/2) * 256 * 2 options:MTLResourceStorageModeShared];
        for (uint32_t p = 0; p < K/2; ++p) { uint32_t sb = p / 8, j = p % 8; for (int b = 0; b < 256; ++b) {
            float v = kv[b & 15] * xf[sb * 16 + j] + kv[b >> 4] * xf[sb * 16 + j + 8];
            ((float *) T32.contents)[p * 256 + b] = v; ((_Float16 *) T16.contents)[p * 256 + b] = v; } }
        id<MTLBuffer> Y = [dev newBufferWithLength:rows * 4 * nmat options:MTLResourceStorageModeShared];
        struct V { const char * name; int nr; int tab; };
        std::vector<V> vs = { {"nv_lut", 1, 0}, {"nv_lut", 2, 0}, {"nv_ar", 1, 0}, {"nv_ar", 2, 0}, {"nv_tab32", 1, 1}, {"nv_tab32", 2, 1}, {"nv_tab32", 4, 1}, {"nv_tab16", 1, 2}, {"nv_tab16", 2, 2}, {"nv_tab16", 4, 2} };
        for (auto & v : vs) {
            id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:v.name]] error:&err];
            if (!ps) { printf("pipeline: %s\n", err.description.UTF8String); return 1; }
            double best = 1e9;
            for (int rep = 0; rep < 6; ++rep) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:ps];
                for (uint32_t m = 0; m < nmat; ++m) {
                    Args a = { rows, K, (uint32_t) v.nr, rowBytes };
                    [enc setBytes:&a length:sizeof(a) atIndex:0];
                    [enc setBuffer:W offset:(size_t) m * mb atIndex:1];
                    [enc setBuffer:(v.tab == 0 ? X : v.tab == 1 ? T32 : T16) offset:0 atIndex:2];
                    [enc setBuffer:ST offset:0 atIndex:3];
                    [enc setBuffer:Y offset:m * rows * 4 atIndex:4];
                    [enc dispatchThreadgroups:MTLSizeMake(rows / v.nr, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                }
                [enc endEncoding];
                double t0 = now(); [cb commit]; [cb waitUntilCompleted]; double t = now() - t0;
                if (t < best) best = t;
            }
            double gw = (double) rows * K * nmat / 1e9;
            printf("K=%u %-8s nr=%d: %.2f ms %.2f Gw -> %.0f Gw/s (%.0f GB/s) y=%g %g\n", K, v.name, v.nr, best * 1e3, gw, gw / best, gw * 0.5625 / best, ((float *) Y.contents)[5], ((float *) Y.contents)[rows + 77]);
        }
    }
} return 0; }
