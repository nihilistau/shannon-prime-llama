/*
 * Shannon-Prime QNN dispatch wrapper for shannon-prime-llama integration.
 *
 * This is the C entry point that the existing FUSED_KQ custom op slot in
 * shannon-prime-llama (src/engine/llama_sp_fused_kq.cpp) calls when the
 * SP_USE_QNN_KQ env / preprocessor flag is set. The wrapper:
 *
 *   1. Lazy-creates a per-(M_q, K_dim, N_kv) sp_qnn_handle on first call
 *      (graphFinalize is the slow step — ~50-100 ms — paid once per shape)
 *   2. Caches the handle indexed by shape, reuses across calls
 *   3. Pseudo-persists the K tensor across calls (production path is
 *      memhandle once Phase 2.5b lands rpcmem; today it preserves clientBuf)
 *   4. Per-call: bind Q, optionally rebind K if changed, execute, return
 *      attention weights
 *
 * Design constraint: the existing llama_sp_kq_compute callback in
 * llama_sp_fused_kq.cpp has signature:
 *
 *   void llama_sp_kq_compute(struct ggml_tensor * dst,
 *                            const struct ggml_tensor * a,    // Q
 *                            const struct ggml_tensor * b,    // K
 *                            int ith, int nth, void * userdata);
 *
 * The QNN dispatch path replaces the existing DSP fast-path branch in that
 * callback with a call to sp_llama_qnn_kq_dispatch(). The integration point
 * is the existing `if (sp_ctx_v && !u->is_v && u->n_kv > 0) { rc = ... }`
 * block — swap sp_llama_kq_matmul_fused for our wrapper.
 *
 * Copyright (C) 2026 Ray Daniels. AGPLv3.
 */
#ifndef SP_LLAMA_QNN_H
#define SP_LLAMA_QNN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque shape-keyed cache. One entry per unique (M_q, K_dim, N_kv) shape
 * the model uses. Persistent across all attention calls in a session. */
typedef struct sp_llama_qnn_kq_cache sp_llama_qnn_kq_cache;

/* Create the global cache once at session start. Returns NULL on
 * allocation failure. Threading: the cache is mutex-protected internally;
 * concurrent callers (e.g., per-thread KQ ops in ggml's tile-share pattern)
 * are safe but serialize on the mutex. Most LLM inference paths are
 * single-threaded at the QNN dispatch boundary. */
sp_llama_qnn_kq_cache *sp_llama_qnn_kq_cache_create(void);
void sp_llama_qnn_kq_cache_destroy(sp_llama_qnn_kq_cache **c);

/* Dispatch one KQ + Softmax via runtime QNN graph.
 *
 * Inputs (all pointers caller-owned, must remain valid until this call returns):
 *   M_q     — number of Q rows (typically n_seq * n_head_q after layout)
 *   K_dim   — head_dim
 *   N_kv    — number of K positions in the attention window
 *   q_data  — fp16 [M_q, K_dim], M_q*K_dim*2 bytes
 *   k_data  — fp16 [N_kv, K_dim], N_kv*K_dim*2 bytes
 *
 * Output:
 *   attn_out — fp16 [M_q, N_kv], M_q*N_kv*2 bytes (caller-allocated)
 *
 * Optional:
 *   exec_us — wall-clock execute time in microseconds (skip via NULL)
 *
 * Returns 0 on success, negative on failure (caller should fall back to
 * the existing CPU/DSP path).
 */
int sp_llama_qnn_kq_dispatch(sp_llama_qnn_kq_cache *cache,
                             uint32_t M_q,
                             uint32_t K_dim,
                             uint32_t N_kv,
                             const void *q_data, size_t q_bytes,
                             void *k_data,       size_t k_bytes,
                             void *attn_out,     size_t out_bytes,
                             uint64_t *exec_us);

#ifdef __cplusplus
}
#endif

#endif /* SP_LLAMA_QNN_H */
