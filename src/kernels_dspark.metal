
// ================================================================================================
// DSpark draft model (GGUF arch "dflash" with a Markov head): a small 3-block non-causal decoder that
// drafts a block of tokens after the last committed one.  It attends to K / V "injected" for every
// context position from the target model's hidden states at four depths, plus the block itself.
// Dimensions equal the target's (D, NH, NKV, DH); DFF = 4 * D.
// ================================================================================================

#define DFF (4 * D)

// out[n] = rmsnorm(sum_i mix[i] * tap_i[n]) * w  (half, and f32 if wantF).  taps: [4][tapStride] f32.
struct DsMixArgs { uint N; uint tapStride; uint wantF; float eps; float m0; float m1; float m2; float m3; };

kernel void ds_mix_norm(constant DsMixArgs & a, device const float * taps, device const float * w, device half * outH, device float * outF,
                        uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_threadgroup]],
                        ushort sg [[simdgroup_index_in_threadgroup]], ushort tis [[thread_index_in_simdgroup]]) {
    threadgroup float part[TGSG * RMAX];
    threadgroup float tot[RMAX];
    const uint n = tg.x, i = ti * 3;
    device const float * t0 = taps + (ulong) n * D + i;
    const float3 x = a.m0 * ld3(t0) + a.m1 * ld3(t0 + a.tapStride) + a.m2 * ld3(t0 + 2 * (ulong) a.tapStride) + a.m3 * ld3(t0 + 3 * (ulong) a.tapStride);
    float v[1] = {dot(x, x)};
    tg_reduce(v, 1, part, tot, sg, tis, TGSG);
    const float3 y = x * (1.0f / sqrt(v[0] / D + a.eps)) * ld3(w + i);
    sth3(outH + (ulong) n * D + i, y);
    if (a.wantF) st3(outF + (ulong) n * D + i, y);
}

// block input: row 0 = rmsnorm(embedding of the anchor token) * w, row n > 0 = mask embedding n - 1
struct DsEmbedArgs { uint tok; float eps; uint pad0; uint pad1; };

kernel void ds_embed(constant DsEmbedArgs & a, device const half * emb, device const float * w, device const float * masks, device float * cur,
                     uint3 tg [[threadgroup_position_in_grid]], ushort ti [[thread_index_in_threadgroup]],
                     ushort sg [[simdgroup_index_in_threadgroup]], ushort tis [[thread_index_in_simdgroup]]) {
    threadgroup float part[TGSG * RMAX];
    threadgroup float tot[RMAX];
    const uint n = tg.x, i = ti * 3;
    if (n > 0) { st3(cur + (ulong) n * D + i, ld3(masks + (ulong) (n - 1) * D + i)); return; }
    device const half * e = emb + (ulong) a.tok * D + i;
    const float3 x = float3(e[0], e[1], e[2]);
    float v[1] = {dot(x, x)};
    tg_reduce(v, 1, part, tot, sg, tis, TGSG);
    st3(cur + i, x * (1.0f / sqrt(v[0] / D + a.eps)) * ld3(w + i));
}

// Q / K / V of the decoder.  One SIMD group per (unit, row); units u0.. : NH query heads, then NKV key heads,
// then NKV value heads.  Q and K: rmsnorm over the head * w, rotary embedding on all dims (pairs i, i + DH/2);
// each thread holds dims {2t, 2t+1, 2t+DH/2, 2t+1+DH/2}, so both of its pairs are thread-local.
// rope[n] = DH/2 x (cos, sin).  Q is stored pre-scaled, grouped by KV head: QG[hk][n][g][DH].
// K / V rows go to row (slot0 + n) of Kc / Vc.
struct DsPrepArgs { uint N; uint ldp; uint qCol; uint kCol; uint vCol; uint u0; uint slot0; float qscale; float eps; uint pad0; uint pad1; uint pad2; };

kernel void ds_prep(constant DsPrepArgs & a, device const float * P, device const float * qn, device const float * kn,
                    device const float * rope, device half * QG, device half * Kc, device half * Vc,
                    uint3 tg [[threadgroup_position_in_grid]], ushort tis [[thread_index_in_threadgroup]]) {
    const uint u = tg.x + a.u0, n = tg.y;
    device const float * row = P + (ulong) n * a.ldp;
    const uint i0 = tis * 2, i1 = tis * 2 + DH / 2;
    if (u < NH + NKV) {
        const bool isq = u < NH;
        const uint h = isq ? u : u - NH;
        device const float * src = row + (isq ? a.qCol : a.kCol) + h * DH;
        device const float * w = isq ? qn : kn;
        float4 x = float4(src[i0], src[i0 + 1], src[i1], src[i1 + 1]);
        const float ss = simd_sum(dot(x, x));
        x *= float4(w[i0], w[i0 + 1], w[i1], w[i1 + 1]) * (1.0f / sqrt(ss / DH + a.eps) * (isq ? a.qscale : 1.0f));
        device const float * r = rope + (ulong) n * DH + tis * 4;   // (cos, sin) of pairs i0 and i0 + 1
        const float4 y = float4(x.x * r[0] - x.z * r[1], x.y * r[2] - x.w * r[3], x.x * r[1] + x.z * r[0], x.y * r[3] + x.w * r[2]);
        device half * o = isq ? QG + (((ulong) (h / GQA) * a.N + n) * GQA + (h % GQA)) * DH
                              : Kc + (ulong) (a.slot0 + n) * DKV + h * DH;
        o[i0] = half(y.x); o[i0 + 1] = half(y.y); o[i1] = half(y.z); o[i1 + 1] = half(y.w);
    } else {
        const uint h = u - NH - NKV;
        device const float * src = row + a.vCol + h * DH;
        device half * o = Vc + (ulong) (a.slot0 + n) * DKV + h * DH;
        o[i0] = half(src[i0]); o[i0 + 1] = half(src[i0 + 1]); o[i1] = half(src[i1]); o[i1 + 1] = half(src[i1 + 1]);
    }
}

// y += x
kernel void m_add(constant CountArgs & a, device const float * x, device float * y, uint gid [[thread_position_in_grid]]) {
    if (gid < a.w) y[gid] += x[gid];
}

// A = silu(G) * U as half
kernel void ds_act(constant CountArgs & a, device const float * G, device const float * U, device half * A, uint gid [[thread_position_in_grid]]) {
    if (gid >= a.w) return;
    const float g = G[gid];
    A[gid] = half(clamp(g / (1.0f + exp(-g)) * U[gid], -65000.0f, 65000.0f));
}

// ---- Markov head, one block position per step (all on the GPU: the chosen token feeds the next step)
#define DSR 256                 // rank of the Markov head
struct DsStepArgs { uint step; uint vocab; uint hStride; float cap; };

// joint input u = [state (R) | w1[prev] (R) | hidden of this block row (D) | f_rnn (D)]
kernel void ds_joint_in(constant DsStepArgs & a, device const float * state, device const half * w1, device const uint * toks,
                        device const float * hidden, device const float * frnn, device float * u, uint gid [[thread_position_in_grid]]) {
    if (gid >= 2 * DSR + 2 * D) return;
    float v;
    if (gid < DSR) v = state[gid];
    else if (gid < 2 * DSR) v = float(w1[(ulong) toks[a.step - 1] * DSR + gid - DSR]);
    else if (gid < 2 * DSR + D) v = hidden[(ulong) a.step * a.hStride + gid - 2 * DSR];
    else v = frnn[gid - 2 * DSR - D];
    u[gid] = v;
}

// joint = [gate | cand | markov] pre-activations (+ bias): state = gate * state + (1 - gate) * cand
kernel void ds_gate(constant DsStepArgs & a, device const float * joint, device const float * bias, device float * state, device float * markov,
                    uint gid [[thread_position_in_grid]]) {
    if (gid >= DSR) return;
    const float gate = 1.0f / (1.0f + exp(-(joint[gid] + bias[gid])));
    const float cand = tanh(joint[DSR + gid] + bias[DSR + gid]);
    state[gid] = cand + gate * (state[gid] - cand);
    markov[gid] = tanh(joint[2 * DSR + gid] + bias[2 * DSR + gid]);
}

// col = softcap(base + w2 * markov + b); partial arg-max and sum of exp per chunk of 1024 (one SIMD group each)
kernel void ds_col(constant DsStepArgs & a, device const float * base, device const float * wb, device const float * b2,
                   device float * col, device float * part,
                   uint3 tg [[threadgroup_position_in_grid]], ushort tis [[thread_index_in_threadgroup]]) {
    const uint c0 = tg.x * 1024;
    float m = -1e30f;
    uint am = 0;
    float v[32];
    for (uint k = 0; k < 32; ++k) {
        const uint i = c0 + k * SW + tis;
        float x = -1e30f;
        if (i < a.vocab) {
            x = base[(ulong) a.step * a.hStride + i] + wb[i] + b2[i];
            x = a.cap * tanh(x / a.cap);
            col[i] = x;
        }
        v[k] = x;
        if (x > m) { m = x; am = i; }
    }
    const float M = simd_max(m);
    const uint AM = simd_min(m == M ? am : 0xffffffffu);
    float s = 0;
    for (uint k = 0; k < 32; ++k) s += exp(v[k] - M);
    s = simd_sum(s);
    if (tis == 0) { device float * o = part + tg.x * 4; o[0] = M; o[1] = s; ((device uint *) o)[2] = AM; }
}

// reduce the chunk partials: token of this step and its probability
kernel void ds_pick(constant DsStepArgs & a, device const float * part, device uint * toks, device float * probs,
                    ushort tis [[thread_index_in_threadgroup]]) {
    const uint nch = (a.vocab + 1023) / 1024;
    float m = -1e30f;
    uint am = 0xffffffffu;
    for (uint c = tis; c < nch; c += SW) {
        const float x = part[c * 4];
        const uint id = ((device const uint *) part)[c * 4 + 2];
        if (x > m || (x == m && id < am)) { m = x; am = id; }
    }
    const float M = simd_max(m);
    const uint AM = simd_min(m == M ? am : 0xffffffffu);
    float s = 0;
    for (uint c = tis; c < nch; c += SW) s += part[c * 4 + 1] * exp(part[c * 4] - M);
    s = simd_sum(s);
    if (tis == 0) { toks[a.step] = AM; probs[a.step] = 1.0f / s; }
}
