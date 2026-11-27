#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <vector>
#include <string>
#include <chrono>
#include <cstdio>
struct Args { uint32_t rows, K, nr, pad; };
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main(int argc, char ** argv) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        printf("device: %s maxBuffer=%.1f GB\n", dev.name.UTF8String, dev.maxBufferLength / 1e9);
        NSString * src = [NSString stringWithContentsOfFile:@"bw.metal" encoding:NSUTF8StringEncoding error:nil];
        MTLCompileOptions * opt = [MTLCompileOptions new];
        NSError * err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opt error:&err];
        if (!lib) { printf("compile: %s\n", err.description.UTF8String); return 1; }
        id<MTLCommandQueue> q = [dev newCommandQueue];
        const uint32_t K = 2304, rows = 8192, nmat = 48;
        size_t wbytes = (size_t) rows * K * 2;
        id<MTLBuffer> W = [dev newBufferWithLength:wbytes * nmat options:MTLResourceStorageModeShared];
        uint16_t * wp = (uint16_t *) W.contents;
        for (size_t i = 0; i < wbytes * nmat / 2; ++i) wp[i] = 0x3000 + (i * 2654435761u & 0x3ff);
        id<MTLBuffer> X = [dev newBufferWithLength:K * 2 options:MTLResourceStorageModeShared];
        for (int i = 0; i < (int) K; ++i) ((uint16_t *) X.contents)[i] = 0x3c00;
        id<MTLBuffer> Y = [dev newBufferWithLength:rows * 4 * nmat options:MTLResourceStorageModeShared];
        struct V { const char * name; int nr; bool tg; int bytes; };
        std::vector<V> vs = { {"mv_a", 1, true, 2}, {"mv_a", 2, true, 2}, {"mv_a", 4, true, 2}, {"mv_b", 1, true, 2}, {"mv_b", 2, true, 2}, {"mv_b", 4, true, 2}, {"mv_c", 1, false, 2}, {"mv_i8", 2, true, 1}, {"mv_i8", 4, true, 1} };
        for (auto & v : vs) {
            id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:v.name]];
            id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
            if (!ps) { printf("pipeline: %s\n", err.description.UTF8String); return 1; }
            double best = 1e9;
            for (int rep = 0; rep < 6; ++rep) {
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                [enc setComputePipelineState:ps];
                for (uint32_t m = 0; m < nmat; ++m) {
                    Args a = { rows, K, (uint32_t) v.nr, 0 };
                    [enc setBytes:&a length:sizeof(a) atIndex:0];
                    [enc setBuffer:W offset:(size_t) m * rows * K * v.bytes atIndex:1];
                    [enc setBuffer:X offset:0 atIndex:2];
                    [enc setBuffer:Y offset:m * rows * 4 atIndex:3];
                    if (v.tg) [enc dispatchThreadgroups:MTLSizeMake(rows / v.nr, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    else [enc dispatchThreads:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                }
                [enc endEncoding];
                double t0 = now();
                [cb commit]; [cb waitUntilCompleted];
                double t = now() - t0;
                if (t < best) best = t;
            }
            double gb = (double) rows * K * v.bytes * nmat / 1e9;
            printf("%-6s nr=%d: %.2f ms for %.2f GB -> %.0f GB/s (%.1f Gw/s) y0=%g\n", v.name, v.nr, best * 1e3, gb, gb / best, gb / v.bytes / best, ((float *) Y.contents)[5]);
        }
    }
    return 0;
}
