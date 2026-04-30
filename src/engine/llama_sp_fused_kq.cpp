// Shannon-Prime VHT2: Fused decompress-matmul ggml custom op — implementation.
// See llama_sp_fused_kq.h for design notes and integration plan.

#include "llama_sp_fused_kq.h"
#include "shannon_prime.h"   // sp_band_config_t, sp_band_quantize, sp_band_dequantize
                             // sp_vht2_forward_f32 (self-inverse)
#include "ggml.h"

#include <cstdio>
#include <cstring>

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

    const int head_dim   = u->head_dim;
    const int n_kv       = u->n_kv;
    const int is_v       = u->is_v;
    const int n_head_kv  = u->n_heads_kv;

    // Q tensor 'a': [head_dim, n_head_q, n_seq, n_batch]. We assume fp32 or
    // fp16 (most common). Read its strides from ggml.
    const enum ggml_type qt = a->type;
    const size_t q_row_stride = a->nb[1];  // bytes per (one Q row across head_dim)
    const int    n_head_q    = (int) a->ne[1];

    // dst: KQ scores. [n_kv, n_head_q, n_seq, n_batch]. fp32.
    float * dst_data = (float *) dst->data;
    const size_t dst_row_bytes = dst->nb[1];   // bytes per kq row across n_kv

    // Q raw bytes; we'll read a row at a time
    const uint8_t * q_data = (const uint8_t *) a->data;

    // K raw bytes (used in fallback path when userdata has no archive ptr).
    const uint8_t * k_data_fp16 = (const uint8_t *) b->data;
    const size_t k_row_bytes_fp16 = b->nb[1];

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
