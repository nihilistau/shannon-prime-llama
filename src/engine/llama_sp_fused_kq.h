// Shannon-Prime VHT2: Fused decompress-matmul ggml custom op (Phase 1.6 / Path A.2).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
// Licensed under AGPLv3. Commercial license: raydaniels@gmail.com
//
// Replaces ggml_mul_mat(ctx, k, q) in attention with a fused op that:
//   1. reads K from a per-(layer,head) packed-byte buffer (10× smaller than fp32 K)
//   2. decompresses each K row on-the-fly via sp_band_dequantize + VHT2 self-inverse
//   3. dot-products against Q rows to produce KQ scores
//
// The decompress stays in L2 cache across rows (compressed K is ~10× smaller than
// fp32 K), so memory bandwidth — the gating cost in vanilla mul_mat — is replaced
// by ALU cycles. CPU bench at n_kv=4096 hd=64 n_q=8: 1.79× faster than vanilla
// scalar matmul on Snapdragon 8 Gen 1 ARM cores.
//
// This file declares the ggml_custom2_op_t callback and the userdata struct.
// The callback is registered via ggml_map_custom2 in build_attn_mha when
// SHANNON_PRIME_FUSED_KQ=1 is set.

#ifndef LLAMA_SP_FUSED_KQ_H
#define LLAMA_SP_FUSED_KQ_H

#ifdef __cplusplus
extern "C" {
#endif

#include "ggml.h"

// Userdata passed through ggml_map_custom2 to the callback. Lifetime: must
// outlive the ggml graph compute call that invokes the op.
//
// Today the post-decode hook writes K-roundtripped values back into the fp16
// ggml KV cache, so the K tensor passed to ggml_mul_mat IS already an
// SP-decompressed view. Phase 1.6 will switch the source-of-truth to
// sp_state.hexagon_cache.k_cache[layer*nhk+head] (compressed bytes, persistent
// archive — requires the post_compute slot=0 → slot=position change). Until
// that lands, the callback reads the fp16 K tensor's `data` directly and
// performs the decompress/recompress round-trip per call. Net cost: extra
// compress/decompress per attention call, but validates the integration.
//
// k_packed_buf_per_head: pointer to compressed K bytes, layout
//   [head][position][total_bytes]
// where total_bytes = bc.total_bytes for the K config (4 bands @ 5/5/4/3
// for K, 1 band @ 3 for V). Set to NULL until the persistent archive lands;
// callback then falls back to compressing the fp16 K tensor on-the-fly.
typedef struct {
    int    layer_idx;          // for diagnostics + cache lookup
    int    n_heads_kv;
    int    head_dim;
    int    n_kv;               // attention window (positions to dot against)
    int    is_v;               // 0 = K config (4-band), 1 = V config (1-band)

    // Direct-from-archive pointer (Phase 1.6 ready). NULL = fall back to
    // on-the-fly compress of the fp16 K tensor data passed via 'b' arg.
    const unsigned char * k_packed_buf_per_head;
    int                   total_bytes_per_pos;
} llama_sp_kq_userdata;

// ggml_custom2_op_t signature: (dst, a=Q, b=K, ith, nth, userdata)
//   dst: KQ scores tensor — shape [n_kv, n_head_q, n_seq, n_batch]
//   a:   Q tensor       — shape [head_dim, n_head_q, n_seq, n_batch]
//   b:   K tensor       — shape [head_dim, n_kv,       n_head_kv, n_batch]
//                          (whose data we may ignore if userdata->k_packed_buf_per_head set)
//
// The op uses GGML's per-thread tile sharing: ith in [0, nth) splits the n_kv
// dimension across worker threads. Each thread iterates its slice, decompresses
// K rows, dot-products against all Q heads.
// Phase 1.6 fast-path: thread-local accessors set by process_ubatch and
// read by build_attn_mha. Declared here so llama-graph.cpp (which includes
// this header under LLAMA_SHANNON_PRIME) can wire userdata directly.
// Implementations live in llama-shannon-prime.cpp.
const void * llama_sp_get_current_hexagon_k_cache_slot(int layer, int head);
int          llama_sp_get_current_hexagon_k_total_bytes(void);

// Phase 1.7: SP-archive-IS-the-path gate. When SHANNON_PRIME_FAST_PATH=1 the
// SP compressed K archive is treated as the SOLE source of truth for K. The
// fp16 ggml K cache is still allocated (so ggml graph build doesn't break),
// but the K_cur copy step in build_attn is skipped — those zero pages stay
// uncommitted by the Linux kernel, freeing ~150 MB at n_ctx=4096 on a 3B
// model. Combined with FUSED_KQ=1 (which already routes attention reads to
// the SP archive via the custom op), this fully decouples K storage from
// the fp16 path.
//
// Implementation: 1-byte cache primed from getenv on first call. Returns 0/1.
// Static inline so each TU gets its own copy (no ODR hazard with extern "C").
#include <stdlib.h>
static inline int llama_sp_fast_path_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char * v = getenv("SHANNON_PRIME_FAST_PATH");
        cached = (v && v[0] == '1') ? 1 : 0;
    }
    return cached;
}

void llama_sp_kq_compute(
        struct ggml_tensor * dst,
        const struct ggml_tensor * a,    // Q
        const struct ggml_tensor * b,    // K
        int ith, int nth, void * userdata);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_SP_FUSED_KQ_H
