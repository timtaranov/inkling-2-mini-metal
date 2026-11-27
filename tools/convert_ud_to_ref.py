#!/usr/bin/env python3
"""Rewrite the UD ("inkling2", 48-slot) GGUF into the layout upstream llama.cpp expects
("inkling2_loop").  Weights are copied bit-for-bit; only names/metadata change.  The result
is used purely as a reference to validate the standalone engine."""
import sys, numpy as np
sys.path.insert(0, "llama.cpp/gguf-py")
import gguf
src, dst = sys.argv[1], sys.argv[2]
r = gguf.GGUFReader(src)
w = gguf.GGUFWriter(dst, arch="inkling2_loop", endianess=r.endianess)
skip = {"inkling2.num_loops", "inkling2.loop.core_start", "inkling2.loop.core_end", "inkling2.loop.gate_rank", "inkling2.block_count"}
get = lambda k: r.fields[k].contents()
n_phys = 24
for f in r.fields.values():
    if f.name == gguf.Keys.General.ARCHITECTURE or f.name.startswith("GGUF.") or f.name in skip:
        continue
    vt = f.types[0]; st = f.types[-1] if vt == gguf.GGUFValueType.ARRAY else None
    val = f.contents(); name = f.name
    if name.startswith("inkling2."):
        name = "inkling2_loop." + name[len("inkling2."):]
        if vt == gguf.GGUFValueType.ARRAY and len(val) == 48:
            val = val[:n_phys]
    w.add_key_value(name, val, vt, sub_type=st)
w.add_uint32("inkling2_loop.block_count", n_phys)
w.add_uint32("inkling2_loop.loop.start", get("inkling2.loop.core_start"))
w.add_uint32("inkling2_loop.loop.end", get("inkling2.loop.core_end"))
w.add_uint32("inkling2_loop.loop.count", get("inkling2.num_loops"))
w.add_uint32("inkling2_loop.loop.gate_rank", get("inkling2.loop.gate_rank"))
w.add_bool("inkling2_loop.loop.varying", False)
ren = {"loop_attn_": "loop_q_ctrl_", "loop_k_": "loop_k_ctrl_", "loop_v_": "loop_v_ctrl_", "loop_mlp_": "loop_ffn_ctrl_"}
def bf16(t): return (np.asarray(t.data).reshape(-1).view(np.uint16).astype(np.uint32) << 16).view(np.float32)
def f32(t):
    return bf16(t) if t.tensor_type.name == "BF16" else np.asarray(t.data).reshape(-1).view(np.float32)
T = {t.name: t for t in r.tensors}
q = f32(T["loop_res_query.weight"]).reshape(-1, 2304); kv = f32(T["loop_kv_res_query.weight"]).reshape(-1, 2304)
start = get("inkling2.loop.core_start")
out = []
for t in r.tensors:
    if t.name in ("loop_res_query.weight", "loop_kv_res_query.weight"):
        continue
    name = t.name
    for a, b in ren.items():
        name = name.replace("." + a, "." + b)
    data = t.data; ty = t.tensor_type
    if "_ctrl_" in name:  # upstream keeps the loop tensors in F32
        data = f32(t).reshape([int(x) for x in reversed(t.shape)]).copy(); ty = gguf.GGMLQuantizationType.F32
    out.append((name, data, ty))
for i in range(q.shape[0] // 2):
    m = np.stack([q[2*i], kv[2*i], kv[2*i+1], q[2*i+1]]).astype(np.float32)
    out.append((f"blk.{start+i}.loop_msg_query.weight", m, gguf.GGMLQuantizationType.F32))
for name, data, ty in out:
    w.add_tensor_info(name, data.shape, data.dtype, data.nbytes, ty)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for name, data, ty in out:
    w.write_tensor_data(data)
w.close()
print("wrote", dst)
