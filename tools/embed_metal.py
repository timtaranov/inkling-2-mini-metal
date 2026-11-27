#!/usr/bin/env python3
"""Embed kernels.metal into a C header as a byte array."""
import sys
src = open(sys.argv[1], 'rb').read()
with open(sys.argv[2], 'w') as f:
    f.write("// generated from %s - do not edit\n" % sys.argv[1])
    f.write("static const unsigned char kernels_metal_src[] = {\n")
    for i in range(0, len(src), 24):
        f.write(",".join(str(b) for b in src[i:i+24]) + ",\n")
    f.write("0};\nstatic const unsigned long kernels_metal_src_len = %d;\n" % len(src))
