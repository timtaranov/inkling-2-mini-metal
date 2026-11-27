#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstring>
struct Args { uint32_t rows, K, nr, rowBytes; };
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main() { @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError * err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithContentsOfFile:@"nv3.metal" encoding:NSUTF8StringEncoding error:nil] options:[MTLCompileOptions new] error:&err];
    if (!lib) { printf("compile: %s\n", err.description.UTF8String); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLBuffer> ST = [dev newBufferWithLength:256 * 4 options:MTLResourceStorageModeShared];
    for (int x = 0; x < 256; ++x) { int e = (x >> 3) & 15, m = x & 7; float raw = e == 0 ? ldexpf(m, -9) : ldexpf(1.0f + m / 8.0f, e - 7); ((float *) ST.contents)[x] = (x == 0 || x == 0x7f) ? 0 : raw * 0.5f; }
    struct T { const char * name; uint32_t K; int shape; bool half; };   // shape 0: pair rows over [rows]; 1: dn_a; 2: dn_c; 3: dn_d
    T tests[] = { {"p_lut", 2304, 0, false}, {"p_half", 2304, 0, true}, {"p_byte", 2304, 0, false}, {"p_lut", 1152, 0, false}, {"p_half", 1152, 0, true}, {"p_byte", 1152, 0, false},
                  {"dn_a", 1152, 1, false}, {"dn_c", 1152, 2, false}, {"dn_d", 1152, 3, false} };
    for (int pass = 0; pass < 2; ++pass) for (auto & t : tests) {
        const uint32_t K = t.K, rowBytes = K/16 + K/2;
        const uint32_t rows = t.shape == 0 ? 2304 * 8 : 2304, nmat = 40;
        const size_t mb = (size_t) (t.shape == 0 ? rows : rows * 1) * rowBytes;   // per matrix
        static id<MTLBuffer> W = nil; static size_t wlen = 0;
        size_t need = (size_t) 2304 * 8 * (2304/16 + 2304/2) * nmat;
        if (!W) { W = [dev newBufferWithLength:need options:MTLResourceStorageModeShared]; wlen = need; uint8_t * wp = (uint8_t *) W.contents; uint32_t s = 12345; for (size_t i = 0; i < need; ++i) { s = s * 1664525u + 1013904223u; wp[i] = 40 + ((s >> 24) & 31); } }
        id<MTLBuffer> X = [dev newBufferWithLength:K * 4 options:MTLResourceStorageModeShared];
        for (uint32_t i = 0; i < K; ++i) { if (t.half) ((_Float16 *) X.contents)[i] = sinf(i * 0.37f); else ((float *) X.contents)[i] = sinf(i * 0.37f); }
        id<MTLBuffer> Y = [dev newBufferWithLength:2304 * 8 * 4 * 40 options:MTLResourceStorageModeShared];
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:t.name]] error:&err];
        if (!ps) { printf("pipeline: %s\n", err.description.UTF8String); return 1; }
        double best = 1e9;
        for (int rep = 0; rep < 6; ++rep) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ps];
            for (uint32_t m = 0; m < nmat; ++m) {
                Args a = { rows, K, 2, rowBytes };
                [enc setBytes:&a length:sizeof(a) atIndex:0];
                // shape 0: 40 distinct matrices of 8*2304 rows; shapes 1-3: the kernel picks 8 of 40 sub-matrices of 2304 rows, offset varies per dispatch
                [enc setBuffer:W offset:t.shape == 0 ? (size_t) m * mb : (size_t) (m % 8) * 2304 * rowBytes * 40 atIndex:1];
                [enc setBuffer:X offset:0 atIndex:2]; [enc setBuffer:ST offset:0 atIndex:3]; [enc setBuffer:Y offset:0 atIndex:4];
                if (t.shape == 0) [enc dispatchThreadgroups:MTLSizeMake(rows / 2, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                else if (t.shape == 1) [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                else if (t.shape == 2) [enc dispatchThreadgroups:MTLSizeMake(rows / 2, 8, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                else [enc dispatchThreadgroups:MTLSizeMake(rows / 4, 8, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
            }
            [enc endEncoding];
            double t0 = now(); [cb commit]; [cb waitUntilCompleted]; double tt = now() - t0;
            if (tt < best) best = tt;
        }
        double gw = (double) (t.shape == 0 ? rows : rows * 8) * K * nmat / 1e9;
        if (pass == 1) printf("K=%u %-7s: %.0f Gw/s  %.1f us/dispatch (y=%g)\n", K, t.name, gw / best, best * 1e6 / nmat, ((float *) Y.contents)[5]);
    }
} return 0; }
