#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Foundation/Foundation.h>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <cstring>
struct GArgs { uint32_t N, M, K, pad; };
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main() { @autoreleasepool {
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    NSString * src = [NSString stringWithContentsOfFile:@"mm.metal" encoding:NSUTF8StringEncoding error:nil];
    NSError * err = nil;
    MTLCompileOptions * opt = [MTLCompileOptions new];
    opt.languageVersion = MTLLanguageVersion4_0;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opt error:&err];
    if (!lib) { printf("compile: %s\n", err.description.UTF8String); return 1; }
    id<MTLCommandQueue> q = [dev newCommandQueue];
    const uint32_t N = 8192, M = 8192, K = 2304;
    id<MTLBuffer> A = [dev newBufferWithLength:(size_t) M * K * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> B = [dev newBufferWithLength:(size_t) N * K * 2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> C = [dev newBufferWithLength:(size_t) N * M * 4 options:MTLResourceStorageModeShared];
    _Float16 * ap = (_Float16 *) A.contents, * bp = (_Float16 *) B.contents;
    uint32_t s = 1;
    for (size_t i = 0; i < (size_t) M * K; ++i) { s = s * 1664525u + 1013904223u; ap[i] = ((int)(s >> 20) - 2048) / 4096.0f; }
    for (size_t i = 0; i < (size_t) N * K; ++i) { s = s * 1664525u + 1013904223u; bp[i] = ((int)(s >> 20) - 2048) / 2048.0f; }
    auto ref = [&](int n, int m) { double r = 0; for (uint32_t k = 0; k < K; ++k) r += (double) bp[(size_t) n * K + k] * (double) ap[(size_t) m * K + k]; return r; };
    struct V { const char * name; int tn, tm, sg; bool half; };
    std::vector<V> vs = { {"g_32_64_4_f", 32, 64, 4, false}, {"g_32_64_4_h", 32, 64, 4, true}, {"g_64_64_4_f", 64, 64, 4, false}, {"g_64_64_8_f", 64, 64, 8, false},
        {"g_32_32_4_f", 32, 32, 4, false}, {"g_64_128_8_f", 64, 128, 8, false}, {"g_32_64_4_fx", 32, 64, 4, false}, {"g_16_64_4_f", 16, 64, 4, false}, {"g_32_128_8_f", 32, 128, 8, false}, {"g_64_32_4_f", 64, 32, 4, false}, {"g_128_32_4_f",128,32,4,false}, {"g_128_32_8_f",128,32,8,false}, {"g_64_32_4_h",64,32,4,true}, {"g_64_32_4_fx",64,32,4,false}, {"g_64_32_2_f",64,32,2,false}, {"g_64_16_4_f",64,16,4,false}, {"g_128_64_8_f",128,64,8,false}, {"g_64_64_4_h",64,64,4,true}, {"g_32_64_4_f", 32, 64, 4, false}, {"g_64_64_4_f", 64, 64, 4, false} };
    for (auto & v : vs) {
        id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:v.name]];
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
        if (!ps) { printf("pipeline %s: %s\n", v.name, err.description.UTF8String); continue; }
        double best = 1e9;
        for (int rep = 0; rep < 12; ++rep) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:ps];
            GArgs a = { N, M, K, 0 };
            [enc setBytes:&a length:sizeof(a) atIndex:0];
            [enc setBuffer:B offset:0 atIndex:1]; [enc setBuffer:A offset:0 atIndex:2]; [enc setBuffer:C offset:0 atIndex:3];
            [enc dispatchThreadgroups:MTLSizeMake((M + v.tm - 1) / v.tm, (N + v.tn - 1) / v.tn, 1) threadsPerThreadgroup:MTLSizeMake(ps.threadExecutionWidth * v.sg, 1, 1)];
            [enc endEncoding];
            memset(C.contents, 0, C.length); double t0 = now(); [cb commit]; [cb waitUntilCompleted]; double t = now() - t0;
            if (t < best) best = t;
        }
        double c1, c2;
        if (v.half) { c1 = ((_Float16 *) C.contents)[(size_t) 37 * M + 4099]; c2 = ((_Float16 *) C.contents)[(size_t) 8191 * M + 8190]; }
        else { c1 = ((float *) C.contents)[(size_t) 37 * M + 4099]; c2 = ((float *) C.contents)[(size_t) 8191 * M + 8190]; }
        printf("%-14s %.2f ms -> %.2f TFLOPS  (chk %.4f/%.4f %.4f/%.4f)\n", v.name, best * 1e3, 2.0 * N * M * K / best / 1e12, c1, ref(37, 4099), c2, ref(8191, 8190));
    }
    // MPS reference
    {
        MPSMatrixDescriptor * dB = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:K rowBytes:K * 2 dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor * dA = [MPSMatrixDescriptor matrixDescriptorWithRows:M columns:K rowBytes:K * 2 dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor * dC = [MPSMatrixDescriptor matrixDescriptorWithRows:N columns:M rowBytes:M * 2 dataType:MPSDataTypeFloat16];
        MPSMatrix * mB = [[MPSMatrix alloc] initWithBuffer:B descriptor:dB], * mA = [[MPSMatrix alloc] initWithBuffer:A descriptor:dA], * mC = [[MPSMatrix alloc] initWithBuffer:C descriptor:dC];
        MPSMatrixMultiplication * mm = [[MPSMatrixMultiplication alloc] initWithDevice:dev transposeLeft:NO transposeRight:YES resultRows:N resultColumns:M interiorColumns:K alpha:1.0 beta:0.0];
        double best = 1e9;
        for (int rep = 0; rep < 4; ++rep) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            [mm encodeToCommandBuffer:cb leftMatrix:mB rightMatrix:mA resultMatrix:mC];
            double t0 = now(); [cb commit]; [cb waitUntilCompleted]; double t = now() - t0;
            if (t < best) best = t;
        }
        printf("MPS f16        %.2f ms -> %.2f TFLOPS (chk %.4f/%.4f)\n", best * 1e3, 2.0 * N * M * K / best / 1e12, (double)((_Float16 *) C.contents)[(size_t) 37 * M + 4099], ref(37, 4099));
    }
} return 0; }
