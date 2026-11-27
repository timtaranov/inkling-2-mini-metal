#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <chrono>
#include <cstdio>
struct A { uint32_t n, pad; };
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main() { @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSError * err = nil;
    id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithContentsOfFile:@"ovh.metal" encoding:NSUTF8StringEncoding error:nil] options:[MTLCompileOptions new] error:&err];
    if (!lib) { printf("%s\n", err.description.UTF8String); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLBuffer> X = [dev newBufferWithLength:1 << 20 options:MTLResourceStorageModeShared], Y = [dev newBufferWithLength:1 << 20 options:MTLResourceStorageModeShared];
    for (int i = 0; i < 2304; ++i) ((float *) X.contents)[i] = 1;
    id<MTLComputePipelineState> pt = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"tiny"] error:&err];
    id<MTLComputePipelineState> pw = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"wide"] error:&err];
    for (int mode = 0; mode < 4; ++mode) {
        for (int nd : {1, 100, 1000}) {
            double best = 1e9, bestEnc = 1e9;
            for (int rep = 0; rep < 20; ++rep) {
                double te = now();
                id<MTLCommandBuffer> cb = [q commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                for (int d = 0; d < nd; ++d) {
                    A a = {2304, 0};
                    bool sw = (mode & 2) ? (d & 1) : 0;   // mode>=2: alternate pipelines
                    [enc setComputePipelineState:(mode & 1) ^ sw ? pw : pt];
                    [enc setBytes:&a length:sizeof(a) atIndex:0];
                    [enc setBuffer:(d & 1) ? Y : X offset:0 atIndex:1];
                    [enc setBuffer:(d & 1) ? X : Y offset:0 atIndex:2];
                    if ((mode & 1) ^ sw) [enc dispatchThreads:MTLSizeMake(2304, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                    else [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
                }
                [enc endEncoding];
                double t0 = now(); [cb commit]; [cb waitUntilCompleted]; double t = now() - t0;
                if (t < best) best = t;
                if (t0 - te < bestEnc) bestEnc = t0 - te;
            }
            printf("mode %d: %4d dispatches: encode %.1f us, run %.1f us (%.2f us/dispatch)\n", mode, nd, bestEnc * 1e6, best * 1e6, best * 1e6 / nd);
        }
    }
} return 0; }
