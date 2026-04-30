// Shannon-Prime VHT2: K-capture custom ggml op (Phase 1.7).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
// Licensed under AGPLv3. Commercial license: raydaniels@gmail.com
//
// Phase 1.7 problem statement
// ──────────────────────────────────────────────────────────────────
// With Phase 1.6 / Path A.2 active, attention reads K from the SP
// compressed archive (sp_state.hexagon_cache.k_cache) via the FUSED_KQ
// custom op. The fp16 ggml K cache is no longer the read source — but
// it's still allocated AND still written by ggml_cpy in build_attn, AND
// the post_compute hook reads from it as the GATHER SOURCE for the SP
// archive write.
//
// To "own the entire path" — skip cpy_k entirely so kernel pages stay
// zero-mapped and we recover ~150 MB at n_ctx=4096 — we must source the
// SP archive write from somewhere other than the fp16 cache. The only
// other place the live K values exist is k_cur (the K-projection output
// tensor, before cpy_k spills it into the cache). But k_cur lives in
// the graph allocator's reusable buffer and may be overwritten the
// moment its last consumer (cpy_k, which we're skipping) finishes —
// reading it from the post_compute hook (which runs AFTER graph_compute)
// would be a use-after-free in the steady state.
//
// This op is the fix: a ggml_map_custom2 op that runs DURING graph
// compute as a consumer of k_cur. By being a consumer it keeps k_cur
// alive until the op runs; inside the op we gather k_cur values and
// dispatch sp_llama_write_k_batch to populate the SP archive directly.
// post_compute's K loop becomes a no-op when the fast path is active.
//
// Side benefit: FastRPC into the cDSP can now pipeline against the
// rest of the graph compute (next layer's Q proj etc.) instead of
// being serial after graph_compute returns.

#ifndef LLAMA_SP_KCAP_H
#define LLAMA_SP_KCAP_H

#ifdef __cplusplus
extern "C" {
#endif

#include "ggml.h"

// Userdata passed through ggml_map_custom2 to the callback. Lifetime
// must outlive the ggml graph compute call.
typedef struct {
    int   layer_idx;
    int   n_heads_kv;
    int   head_dim;
    int   n_tokens;       // ubatch size
    void *sp_ctx;          // sp_llama_ctx_t * (opaque)
} llama_sp_kcap_userdata;

// ggml_custom2_op_t signature: (dst, a=k_cur, b=k_idxs, ith, nth, userdata)
//   dst: dummy passthrough (we don't compute a meaningful output)
//   a:   k_cur tensor   — shape [head_dim, n_heads_kv, n_tokens]
//   b:   k_idxs tensor  — shape [n_tokens] of int32 (slot indices)
//
// The op iterates (head, token), gathers k_cur[token, head, :] into a
// per-thread scratch, and dispatches sp_llama_write_k_batch into the
// persistent SP archive at slots given by k_idxs.
//
// Threads split work along the head axis (n_heads_kv is small for GQA
// models — typically 2-8 — so per-thread chunks are coarse-grained;
// FastRPC dispatch overhead dominates the per-call cost so finer
// splits would just multiply overhead).
void llama_sp_kcap_compute(struct ggml_tensor * dst,
                            const struct ggml_tensor * a,
                            const struct ggml_tensor * b,
                            int ith, int nth, void * userdata);

// Convenience emitter — wires up userdata + ggml_map_custom2 + adds the
// resulting node to the forward graph. Returns 1 if the op was emitted,
// 0 if it was skipped (e.g., sp_ctx not active or k_cur shape unexpected).
//
// Called from llm_graph_context::build_attn at sites where Phase 1.7
// fast path bypasses cpy_k. Lives here (not inline at the call site) to
// keep the patch hunk for llama-graph.cpp small.
int llama_sp_kcap_emit(struct ggml_context * ctx0,
                        struct ggml_cgraph * gf,
                        struct ggml_tensor * k_cur,
                        struct ggml_tensor * k_idxs,
                        int layer_idx);

// Returns nonzero if the post_compute K loop should be skipped this
// ubatch (i.e., FAST_PATH=1 and at least one layer has a registered
// kcap dispatch). Cleared at process_ubatch entry.
int llama_sp_kcap_skip_post_compute_k(void);
void llama_sp_kcap_clear_post_compute_skip(void);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_SP_KCAP_H
