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
    NSError * err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithContentsOfFile:@"nv2.metal" encoding:NSUTF8StringEncoding error:nil] options:[MTLCompileOptions new] error:&err];
    if (!lib) { printf("compile: %s\n", err.description.UTF8String); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    static const float kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    id<MTLBuffer> ST = [dev newBufferWithLength:256 * 4 options:MTLResourceStorageModeShared];
    for (int x = 0; x < 256; ++x) { int e = (x >> 3) & 15, m = x & 7; float raw = e == 0 ? ldexpf(m, -9) : ldexpf(1.0f + m / 8.0f, e - 7); ((float *) ST.contents)[x] = (x == 0 || x == 0x7f) ? 0 : raw * 0.5f; }
    id<MTLBuffer> T2 = [dev newBufferWithLength:256 * 8 options:MTLResourceStorageModeShared];
    for (int b = 0; b < 256; ++b) { ((float *) T2.contents)[2*b] = kv[b & 15]; ((float *) T2.contents)[2*b+1] = kv[b >> 4]; }
    for (uint32_t K : {2304u, 1152u}) {
        const uint32_t rows = 2304 * 8, nmat = 40, rowBytes = K/16 + K/2;
        size_t mb = (size_t) rows * rowBytes;
        id<MTLBuffer> W = [dev newBufferWithLength:mb * nmat options:MTLResourceStorageModeShared];
        uint8_t * wp = (uint8_t *) W.contents; uint32_t s = 12345;
        for (size_t i = 0; i < mb * nmat; ++i) { s = s * 1664525u + 1013904223u; wp[i] = s >> 24; }
        for (size_t r = 0; r < (size_t) rows * nmat; ++r) for (uint32_t i = 0; i < K/16; ++i) wp[r * rowBytes + i] = 40 + (wp[r * rowBytes + i] & 31);
        id<MTLBuffer> X = [dev newBufferWithLength:K * 4 options:MTLResourceStorageModeShared];
        for (uint32_t i = 0; i < K; ++i) ((float *) X.contents)[i] = sinf(i * 0.37f);
        id<MTLBuffer> Y = [dev newBufferWithLength:rows * 4 * nmat options:MTLResourceStorageModeShared];
        for (const char * name : {"v0", "v1", "v2", "v3", "v4", "v5"}) {
            id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:name]] error:&err];
            if (!ps) { printf("pipeline: %s\n", err.description.UTF8String); return 1; }
            for (int hot = 0; hot < 2; ++hot) {
                const uint32_t rr = hot ? 256 : rows; const uint32_t nd = hot ? 2000 : nmat;
                double best = 1e9;
                for (int rep = 0; rep < 5; ++rep) {
                    id<MTLCommandBuffer> cb = [q commandBuffer];
                    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                    [enc setComputePipelineState:ps];
                    for (uint32_t m = 0; m < nd; ++m) {
                        Args a = { rr, K, 2, rowBytes };
                        [enc setBytes:&a length:sizeof(a) atIndex:0];
                        [enc setBuffer:W offset:hot ? 0 : (size_t) m * mb atIndex:1];
                        [enc setBuffer:X offset:0 atIndex:2]; [enc setBuffer:ST offset:0 atIndex:3]; [enc setBuffer:T2 offset:0 atIndex:4];
                        [enc setBuffer:Y offset:hot ? 0 : m * rows * 4 atIndex:5];
                        [enc dispatchThreadgroups:MTLSizeMake(rr / 2, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    }
                    [enc endEncoding];
                    double t0 = now(); [cb commit]; [cb waitUntilCompleted]; double t = now() - t0;
                    if (t < best) best = t;
                }
                double gw = (double) rr * K * nd / 1e9;
                printf("K=%u %s %s: %.0f Gw/s  (y=%g)\n", K, name, hot ? "hot " : "cold", gw / best, ((float *) Y.contents)[5]);
            }
        }
    }
} return 0; }
