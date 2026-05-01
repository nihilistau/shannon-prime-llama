// Shannon-Prime VHT2: Fused decompress-matmul ggml custom op — implementation.
// See llama_sp_fused_kq.h for design notes and integration plan.

#include "llama_sp_fused_kq.h"
#include "shannon_prime.h"   // sp_band_config_t, sp_band_quantize, sp_band_dequantize
                             // sp_vht2_forward_f32 (self-inverse)
#include "ggml.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include "shannon_prime_llama.h"  // sp_llama_kq_matmul_fused (DSP fast path)

#include <atomic>

// Phase 1.7: forward decl from llama-context.cpp patch — exposes the
// sp_llama_ctx_t for the current ubatch. Used to dispatch the FastRPC
// fused kernel directly to the DSP, bypassing the per-thread scalar
// inner loop.
extern "C" void * llama_sp_get_current_sp_ctx_void(void);

// Phase 2.5+: optional QNN HTP dispatch path. When SHANNON_PRIME_QNN_KQ=1
// is set in env AND we're on aarch64-android (where libQnnHtp.so + the
// shared backend exist), the QNN runtime-graph dispatch fires before the
// DSP fast-path. On non-Android builds (desktop CI etc.) the QNN path
// compiles as a stub that always returns -1, falling through to the
// existing DSP+CPU paths.
//
// The QNN path is documented in sp_llama_qnn.h. It uses runtime graph
// build (no .bin AOT compile) at the architectural primitive validated
// in test_sp_llama_qnn_smoke.c on the S22U: 314 µs steady at Qwen3-4B
// head shape (Q[64,128] x K^T[2048,128] -> softmax -> attn[64,2048]).
#if defined(__ANDROID__) && defined(__aarch64__) && defined(LLAMA_SHANNON_PRIME_QNN)
  #include "sp_llama_qnn.h"
  #define SP_QNN_KQ_AVAILABLE 1
#else
  #define SP_QNN_KQ_AVAILABLE 0
#endif

namespace sp_qnn_kq_local {
    // Lazily-initialized QNN cache. Created on first call when env opt-in
    // is set; destroyed at process exit via atexit handler. Single global
    // shared across all FUSED_KQ ops since it's shape-keyed internally.
    static std::atomic<bool>     s_qnn_init_attempted{false};
    static std::atomic<int>      s_qnn_enabled{-1};  // -1 unknown, 0 disabled, 1 enabled
#if SP_QNN_KQ_AVAILABLE
    static sp_llama_qnn_kq_cache * s_qnn_cache = nullptr;

    static void qnn_atexit_cleanup(void) {
        if (s_qnn_cache) sp_llama_qnn_kq_cache_destroy(&s_qnn_cache);
    }
#endif

    // Returns 1 if QNN dispatch is available + opted-in; 0 otherwise.
    static inline int try_init_qnn(void) {
        int cached = s_qnn_enabled.load(std::memory_order_acquire);
        if (cached >= 0) return cached;

        // Race-safe init: only the first thread to flip s_qnn_init_attempted
        // does the actual setup; others spin until s_qnn_enabled is published.
        bool expected = false;
        if (s_qnn_init_attempted.compare_exchange_strong(expected, true)) {
#if SP_QNN_KQ_AVAILABLE
            const char * v = std::getenv("SHANNON_PRIME_QNN_KQ");
            if (v && v[0] == '1') {
                s_qnn_cache = sp_llama_qnn_kq_cache_create();
                if (s_qnn_cache) {
                    std::atexit(qnn_atexit_cleanup);
                    fprintf(stderr, "[sp_qnn_kq] enabled — runtime QNN HTP dispatch active\n");
                    s_qnn_enabled.store(1, std::memory_order_release);
                    return 1;
                }
                fprintf(stderr, "[sp_qnn_kq] cache_create failed; staying disabled\n");
            }
#endif
            s_qnn_enabled.store(0, std::memory_order_release);
            return 0;
        }
        // Lost the race — busy-wait for the leader to publish the result.
        while ((cached = s_qnn_enabled.load(std::memory_order_acquire)) < 0) {
            // empty — initialization is microsecond-scale
        }
        return cached;
    }

    // Dispatch wrapper that handles the SP_QNN_KQ_AVAILABLE=0 case as a
    // null-op so the call site stays clean.
    static inline int dispatch(uint32_t M_q, uint32_t K_dim, uint32_t N_kv,
                                const void * q, size_t q_bytes,
                                void * k,        size_t k_bytes,
                                void * out,      size_t out_bytes) {
#if SP_QNN_KQ_AVAILABLE
        return sp_llama_qnn_kq_dispatch(s_qnn_cache,
                                         M_q, K_dim, N_kv,
                                         q, q_bytes, k, k_bytes,
                                         out, out_bytes, nullptr);
#else
        (void)M_q; (void)K_dim; (void)N_kv;
        (void)q; (void)q_bytes; (void)k; (void)k_bytes;
        (void)out; (void)out_bytes;
        return -1;
#endif
    }
}  // namespace sp_qnn_kq_local


namespace {

inline void sp_band_config_for_k(sp_band_config_t * bc, int head_dim) {
    int bits[4] = {5, 5, 4, 3};
    sp_band_config_init(bc, head_dim, 4, bits);
}
inline void sp_band_config_for_v(sp_band_config_t * bc, int head_dim) {
    int bits[1] = {3};
    sp_band_config_init(bc, head_dim, 1, bits);
}

// Decompress a single K (or V) row from packed bytes back to fp32, in-place
// into `out` (size = head_dim floats). Mirrors the inner loop of
// sp_hex_kq_matmul_bench in scaffold/src_app/sp_hex_ext.c.
inline void decompress_one_row(const unsigned char * packed,
                               int head_dim, int is_v,
                               float * out) {
    sp_band_config_t bc;
    if (is_v) sp_band_config_for_v(&bc, head_dim);
    else      sp_band_config_for_k(&bc, head_dim);
    sp_band_dequantize(packed, out, &bc);
    sp_vht2_forward_f32(out, head_dim);  // self-inverse
}

// Fall-back path: compress an fp16 K row to bytes, then decompress (the same
// round-trip the post-decode hook already does, just inline at attention time).
// Used only when userdata->k_packed_buf_per_head == NULL.
inline void compress_then_decompress(const float * k_row_f32,
                                     int head_dim, int is_v,
                                     float * out) {
    sp_band_config_t bc;
    if (is_v) sp_band_config_for_v(&bc, head_dim);
    else      sp_band_config_for_k(&bc, head_dim);

    // VHT2 forward (in-place coeff scratch)
    float coeffs[1024] __attribute__((aligned(64)));
    std::memcpy(coeffs, k_row_f32, sizeof(float) * head_dim);
    sp_vht2_forward_f32(coeffs, head_dim);

    unsigned char packed[256] __attribute__((aligned(64)));  // K total_bytes ≤ 64 at hd≤256
    sp_band_quantize(coeffs, packed, &bc);

    // Decompress (band_dequantize + VHT2 self-inverse)
    sp_band_dequantize(packed, out, &bc);
    sp_vht2_forward_f32(out, head_dim);
}

// fp32 → fp16 (round-to-nearest-even, matches IEEE 754 binary16). Inline
// fallback for when ggml's helper isn't visible in this TU. Subnormals
// flushed to zero — fine for KV cache weights (they're far from zero).
inline uint16_t sp_fp32_to_fp16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    const uint32_t sign = (u >> 16) & 0x8000;
    const int32_t  exp  = (int32_t)((u >> 23) & 0xFF) - 127 + 15;
    const uint32_t mant = u & 0x7FFFFF;
    if (exp <= 0)        return (uint16_t)sign;          // underflow → 0
    if (exp >= 31) {
        if ((u & 0x7FFFFFFF) > 0x7F800000) return (uint16_t)(sign | 0x7E00); // NaN
        return (uint16_t)(sign | 0x7C00);                // ±Inf
    }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

// fp16 → fp32 (matches ggml_fp16_to_fp32). Keeping minimal local copy so we
// don't pull in ggml-impl.h here.
inline float sp_fp16_to_fp32(uint16_t h) {
    const uint32_t s = (uint32_t)(h & 0x8000) << 16;
    const uint32_t e = (h >> 10) & 0x1f;
    const uint32_t m = h & 0x3ff;
    uint32_t f;
    if (e == 0) {
        if (m == 0) f = s;
        else { uint32_t mm = m, ee = 1; while (!(mm & 0x400)) { mm <<= 1; ++ee; }
               f = s | (((127 - 15 - ee + 1) << 23)) | ((mm & 0x3ff) << 13); }
    } else if (e == 31) {
        f = s | 0x7f800000 | (m << 13);
    } else {
        f = s | ((e + (127 - 15)) << 23) | (m << 13);
    }
    float r;
    std::memcpy(&r, &f, sizeof(r));
    return r;
}

} // namespace

extern "C"
void llama_sp_kq_compute(struct ggml_tensor * dst,
                         const struct ggml_tensor * a,
                         const struct ggml_tensor * b,
                         int ith, int nth, void * userdata) {
    const llama_sp_kq_userdata * u = (const llama_sp_kq_userdata *) userdata;
    if (!u || !dst || !a || !b) return;

    // Phase 2.6b debug: confirm FUSED_KQ slot fires at all and what state
    // userdata carries. One-shot only (per process) so it can't flood logs.
    static std::atomic<int> s_kq_log_once{0};
    int prev = s_kq_log_once.fetch_add(1, std::memory_order_relaxed);
    if (prev == 0) {
        const void * accessor_k =
            llama_sp_get_current_hexagon_k_cache_slot(u->layer_idx, 0);
        int          accessor_bytes = llama_sp_get_current_hexagon_k_total_bytes();
        fprintf(stderr,
            "[fused_kq] FIRST CALL: layer=%d n_kv=%d head_dim=%d is_v=%d "
            "k_packed_userdata=%p k_packed_accessor=%p (bytes/pos=%d) "
            "a.type=%d a.dims=[%lld,%lld,%lld,%lld] "
            "b.type=%d b.dims=[%lld,%lld,%lld,%lld] dst.type=%d\n",
            u->layer_idx, u->n_kv, u->head_dim, u->is_v,
            (const void *)u->k_packed_buf_per_head,
            accessor_k, accessor_bytes,
            (int)a->type, (long long)a->ne[0], (long long)a->ne[1],
                          (long long)a->ne[2], (long long)a->ne[3],
            (int)b->type, (long long)b->ne[0], (long long)b->ne[1],
                          (long long)b->ne[2], (long long)b->ne[3],
            (int)dst->type);
    }

    // ── Fast-path dispatch (Phase 1.7 + Phase 2.5+ QNN HTP) ────────────
    // ith==0 tries the fastest available silicon, in order:
    //   1. QNN HTP runtime-graph (if SHANNON_PRIME_QNN_KQ=1 + Android aarch64)
    //   2. cDSP FastRPC (existing Phase 1.7 path)
    //   3. fall through to per-thread scalar loop
    //
    // Whichever succeeds fills the entire kq tensor; ith>0 short-circuit.
    static std::atomic<int> g_dsp_status{0};   // 0=pending, 1=ok, 2=fallback
    if (ith == 0) {
        g_dsp_status.store(0, std::memory_order_relaxed);
        int rc = -1;

        // [1] QNN HTP path. fp16 KQ + softmax in one runtime graph dispatch.
        // Active when SHANNON_PRIME_QNN_KQ=1 + Android aarch64 + the QNN
        // libs are present at runtime. Decompresses K from SP-banded form
        // to a thread-local fp16 scratch buffer once per call, then hands
        // the fp16 [N_kv, head_dim] tensor to the runtime graph.
        // Phase 2.6b: if the userdata-cached k_packed pointer is null
        // (graph-build populates lazily and may miss layers), call the
        // archive accessor directly. Same physical archive — different
        // entry point. Lets QNN dispatch fire even when the userdata
        // wiring is incomplete upstream.
        const unsigned char * k_packed_eff = u->k_packed_buf_per_head;
        int                   k_total_bytes_eff = u->total_bytes_per_pos;
        if (!k_packed_eff) {
            k_packed_eff = (const unsigned char *)
                llama_sp_get_current_hexagon_k_cache_slot(u->layer_idx, 0);
            if (!k_total_bytes_eff)
                k_total_bytes_eff = llama_sp_get_current_hexagon_k_total_bytes();
        }

        if (sp_qnn_kq_local::try_init_qnn()
            && !u->is_v && u->n_kv > 0
            && k_packed_eff != nullptr) {
            const int n_q_local = (int) b->ne[1];
            const int hd        = u->head_dim;
            const int nk        = u->n_kv;

            // Per-thread fp16 scratch for K. Sized to max(seen) to avoid
            // re-alloc churn across calls. Total: nk * hd * 2 bytes.
            // For Qwen3-4B head_dim=128 n_kv=4096: 1 MB. Fine.
            static thread_local std::vector<uint16_t> k_fp16_scratch;
            const size_t k_fp16_count = (size_t)nk * (size_t)hd;
            if (k_fp16_scratch.size() < k_fp16_count) {
                k_fp16_scratch.resize(k_fp16_count);
            }
            // Q is also fp16 (or fp32) in `b`. If fp32, convert to fp16
            // scratch as well. Most production paths have b->type=fp16
            // already (KV cache fp16 + the Q projection output is fp16).
            static thread_local std::vector<uint16_t> q_fp16_scratch;
            const size_t q_count = (size_t)n_q_local * (size_t)hd;
            const uint16_t * q_fp16_ptr = nullptr;
            if (b->type == GGML_TYPE_F16) {
                q_fp16_ptr = (const uint16_t *) b->data;
            } else if (b->type == GGML_TYPE_F32) {
                if (q_fp16_scratch.size() < q_count) q_fp16_scratch.resize(q_count);
                const float * q_f32 = (const float *) b->data;
                for (size_t i = 0; i < q_count; ++i) {
                    q_fp16_scratch[i] = sp_fp32_to_fp16(q_f32[i]);
                }
                q_fp16_ptr = q_fp16_scratch.data();
            }
            // For dst, QNN writes fp16; if dst is fp32 we need a scratch
            // and convert back. Most dst's from ggml_map_custom2 are fp32.
            static thread_local std::vector<uint16_t> out_fp16_scratch;
            const size_t out_count = (size_t)n_q_local * (size_t)nk;
            if (out_fp16_scratch.size() < out_count) out_fp16_scratch.resize(out_count);

            if (q_fp16_ptr) {
                // Decompress every K row from the SP archive into fp16 scratch.
                // Layout: k_fp16_scratch[kv * hd + h] for kv in [0,nk), h in [0,hd).
                float k_row_f32[1024] __attribute__((aligned(64)));
                for (int kv = 0; kv < nk; ++kv) {
                    const unsigned char * packed = k_packed_eff +
                        (size_t)kv * (size_t)k_total_bytes_eff;
                    decompress_one_row(packed, hd, /*is_v=*/0, k_row_f32);
                    for (int h = 0; h < hd; ++h) {
                        k_fp16_scratch[(size_t)kv * (size_t)hd + h] =
                            sp_fp32_to_fp16(k_row_f32[h]);
                    }
                }

                int qnn_rc = sp_qnn_kq_local::dispatch(
                    /*M_q=*/   (uint32_t)n_q_local,
                    /*K_dim=*/ (uint32_t)hd,
                    /*N_kv=*/  (uint32_t)nk,
                    /*q=*/     q_fp16_ptr,            /*q_bytes=*/   q_count   * 2,
                    /*k=*/     k_fp16_scratch.data(), /*k_bytes=*/   k_fp16_count * 2,
                    /*out=*/   out_fp16_scratch.data(),/*out_bytes=*/out_count * 2);

                if (qnn_rc == 0) {
                    // Convert QNN's fp16 attn output to dst's fp32 layout.
                    // dst row stride is dst->nb[1]; one row per Q head.
                    if (dst->type == GGML_TYPE_F32) {
                        for (int qh = 0; qh < n_q_local; ++qh) {
                            float * dst_row = (float *)
                                ((uint8_t *)dst->data + (size_t)qh * dst->nb[1]);
                            const uint16_t * src_row =
                                &out_fp16_scratch[(size_t)qh * (size_t)nk];
                            for (int kv = 0; kv < nk; ++kv) {
                                dst_row[kv] = sp_fp16_to_fp32(src_row[kv]);
                            }
                        }
                    } else if (dst->type == GGML_TYPE_F16) {
                        // direct copy by row (respect dst stride)
                        for (int qh = 0; qh < n_q_local; ++qh) {
                            uint16_t * dst_row = (uint16_t *)
                                ((uint8_t *)dst->data + (size_t)qh * dst->nb[1]);
                            std::memcpy(dst_row,
                                        &out_fp16_scratch[(size_t)qh * (size_t)nk],
                                        (size_t)nk * 2);
                        }
                    }
                    rc = 0;  // QNN path handled it
                }
            }
        }

        // [2] cDSP FastRPC path (existing Phase 1.7).
        if (rc != 0) {
            void * sp_ctx_v = llama_sp_get_current_sp_ctx_void();
            if (sp_ctx_v && !u->is_v && u->n_kv > 0) {
                // Q: [head_dim, n_head_q, ...] in `b`. We hand the q rows
                // straight through; the bridge marshals into rpcmem.
                int n_q_local = (int) b->ne[1];
                // dst layout matches mul_mat result [n_kv, n_q, ...]; we
                // dispatch one head_kv worth at a time. For now use head=0
                // (matches existing FUSED_KQ wiring's known limitation —
                // GQA n_heads_kv > 1 fix is a follow-up).
                const sp_llama_ctx_t * sp_ctx =
                    (const sp_llama_ctx_t *) sp_ctx_v;
                rc = sp_llama_kq_matmul_fused(sp_ctx,
                                              u->layer_idx, /*head=*/0,
                                              /*start_pos=*/0, u->n_kv,
                                              (const float *) b->data,
                                              n_q_local,
                                              (float *) dst->data);
            }
        }
        g_dsp_status.store(rc == 0 ? 1 : 2, std::memory_order_release);
    }
    // All threads wait for ith==0 to publish the status.
    int dsp_status;
    do {
        dsp_status = g_dsp_status.load(std::memory_order_acquire);
    } while (dsp_status == 0);
    if (dsp_status == 1) {
        // DSP wrote the whole kq tensor; this thread is done.
        return;
    }
    // dsp_status == 2: fall through to scalar fallback below.


    const int head_dim   = u->head_dim;
    const int n_kv       = u->n_kv;
    const int is_v       = u->is_v;
    const int n_head_kv  = u->n_heads_kv;

    // Phase 1.7 wiring: ggml_map_custom2_inplace(ctx, kq, q, ...) so:
    //   dst == a == kq (the mul_mat-shaped output tensor — we overwrite values)
    //   b   == q  (read Q rows from here)
    // The K source comes from userdata->k_packed_buf_per_head (SP archive).
    // q tensor 'b': [head_dim, n_head_q, n_seq, n_batch].
    const enum ggml_type qt = b->type;
    const size_t q_row_stride = b->nb[1];  // bytes per (one Q row across head_dim)
    const int    n_head_q    = (int) b->ne[1];

    // dst: KQ scores. [n_kv, n_head_q, n_seq, n_batch]. fp32.
    float * dst_data = (float *) dst->data;
    const size_t dst_row_bytes = dst->nb[1];   // bytes per kq row across n_kv

    // Q raw bytes; we'll read a row at a time
    const uint8_t * q_data = (const uint8_t *) b->data;

    // Fallback K source (when userdata->k_packed_buf_per_head is null) was
    // the original cache view passed as `b`. With inplace wiring `b` is q,
    // so the fallback path no longer has access to the cache view K. Until
    // archive-only is fully wired, fallback is degraded: zeros.
    const uint8_t * k_data_fp16 = nullptr;
    const size_t k_row_bytes_fp16 = 0;

    // Multi-thread split along n_kv
    const int kv_per_thread = (n_kv + nth - 1) / nth;
    const int kv_lo = ith * kv_per_thread;
    const int kv_hi = (kv_lo + kv_per_thread) > n_kv ? n_kv : (kv_lo + kv_per_thread);

    float k_row_f32[1024] __attribute__((aligned(64)));
    float q_row_f32[1024] __attribute__((aligned(64)));

    for (int kv = kv_lo; kv < kv_hi; ++kv) {
        // Decompress K row for this kv position.
        if (u->k_packed_buf_per_head) {
            // Fast path: read from persistent SP archive (Phase 1.6).
            const unsigned char * packed = u->k_packed_buf_per_head +
                (size_t) kv * (size_t) u->total_bytes_per_pos;
            decompress_one_row(packed, head_dim, is_v, k_row_f32);
        } else {
            // Fallback: re-roundtrip the fp16 K row through SP. Costs extra
            // compress per attn call until persistent archive lands.
            const uint8_t * k_row_bytes = k_data_fp16 + (size_t) kv * k_row_bytes_fp16;
            for (int h = 0; h < head_dim; ++h) {
                if (b->type == GGML_TYPE_F32) {
                    k_row_f32[h] = ((const float *) k_row_bytes)[h];
                } else if (b->type == GGML_TYPE_F16) {
                    k_row_f32[h] = sp_fp16_to_fp32(((const uint16_t *) k_row_bytes)[h]);
                } else {
                    // Other types (Q4_0, Q8_0 etc.): not supported in v0.
                    k_row_f32[h] = 0.0f;
                }
            }
            float coeffs[1024] __attribute__((aligned(64)));
            std::memcpy(coeffs, k_row_f32, sizeof(float) * head_dim);
            compress_then_decompress(coeffs, head_dim, is_v, k_row_f32);
        }

        // Dot product against each Q head.
        for (int qh = 0; qh < n_head_q; ++qh) {
            const uint8_t * q_row_bytes = q_data + (size_t) qh * q_row_stride;
            // Convert Q row to fp32 once
            for (int h = 0; h < head_dim; ++h) {
                if (qt == GGML_TYPE_F32) {
                    q_row_f32[h] = ((const float *) q_row_bytes)[h];
                } else if (qt == GGML_TYPE_F16) {
                    q_row_f32[h] = sp_fp16_to_fp32(((const uint16_t *) q_row_bytes)[h]);
                } else {
                    q_row_f32[h] = 0.0f;
                }
            }
            float s = 0.0f;
            for (int h = 0; h < head_dim; ++h) s += k_row_f32[h] * q_row_f32[h];
            // Write into dst[kv][qh] (which is dst_data + qh*dst_row_bytes/4 + kv)
            ((float *) (((uint8_t *) dst_data) + (size_t) qh * dst_row_bytes))[kv] = s;
        }
    }
}
