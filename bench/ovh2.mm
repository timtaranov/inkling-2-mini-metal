#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <chrono>
#include <cstdio>
struct A { uint32_t n, pad; };
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main() { @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError * err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithContentsOfFile:@"ovh2.metal" encoding:NSUTF8StringEncoding error:nil] options:[MTLCompileOptions new] error:&err];
    if (!lib) { printf("%s\n", err.description.UTF8String); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLBuffer> X = [dev newBufferWithLength:1 << 20 options:MTLResourceStorageModeShared], Y = [dev newBufferWithLength:1 << 20 options:MTLResourceStorageModeShared];
    for (int i = 0; i < 2304; ++i) ((float *) X.contents)[i] = 1;
    struct T { const char * fn; uint32_t n; int threads; };
    T tests[] = { {"tiny", 32, 32}, {"tiny", 256, 32}, {"tiny", 2304, 32}, {"tiny", 9216, 32}, {"tg768", 0, 768}, {"tg768x3", 0, 768} };
    for (auto & t : tests) {
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:t.fn]] error:&err];
        if (!ps) { printf("%s\n", err.description.UTF8String); return 1; }
        double best = 1e9; const int nd = 1000;
        for (int rep = 0; rep < 20; ++rep) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ps];
            for (int d = 0; d < nd; ++d) {
                A a = {t.n, 0};
                [enc setBytes:&a length:sizeof(a) atIndex:0];
                [enc setBuffer:(d & 1) ? Y : X offset:0 atIndex:1];
                [enc setBuffer:(d & 1) ? X : Y offset:0 atIndex:2];
                [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(t.threads, 1, 1)];
            }
            [enc endEncoding];
            double t0 = now(); [cb commit]; [cb waitUntilCompleted]; double tt = now() - t0;
            if (tt < best) best = tt;
        }
        printf("%-8s n=%5u threads=%d: %.2f us/dispatch\n", t.fn, t.n, t.threads, best * 1e6 / nd);
    }
} return 0; }
