#import <Metal/Metal.h>
#include <cstdio>
int main() { @autoreleasepool { id<MTLDevice> d = MTLCreateSystemDefaultDevice(); printf("maxThreadgroupMemoryLength=%lu maxThreadsPerThreadgroup=%lu\n", (unsigned long) d.maxThreadgroupMemoryLength, (unsigned long) d.maxThreadsPerThreadgroup.width); } }
