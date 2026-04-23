// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

#ifndef SHANNON_PRIME_CUDA_H
#define SHANNON_PRIME_CUDA_H

#include "../../core/shannon_prime.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// CUDA Shadow Cache
// ============================================================================
//
// GPU-resident compressed KV cache. The write path (VHT2 → Möbius → quantize)
// runs entirely on GPU. The read path (dequantize → unreorder → VHT2) runs
// on GPU and produces fp16 K/V vectors ready for attention.
//
// Integration point for llama.cpp:
//   ggml_backend_cuda_set_kv_cache_hooks(sp_cuda_write_k, sp_cuda_read_k, ...)

typedef struct {
    sp_config_t     config;
    sp_band_config_t k_bands;
    sp_band_config_t v_bands;

    // GPU-resident compressed storage
    void           *d_k_cache;       // Compressed K: [n_layers * n_heads][max_seq][k_bytes]
    void           *d_v_cache;       // Compressed V: [n_layers * n_heads][max_seq][v_bytes]
    int             max_seq_len;

    // GPU-resident Möbius permutation tables
    int            *d_mobius_order;   // [head_dim] forward permutation
    int            *d_mobius_inv;     // [head_dim] inverse permutation

    // GPU scratch (per-stream)
    float          *d_scratch;        // [head_dim] working buffer
    void           *stream;           // CUDA stream
} sp_cuda_cache_t;

// Initialize CUDA shadow cache. Allocates GPU memory.
// stream: CUDA stream for async operations (NULL for default stream).
int sp_cuda_cache_init(sp_cuda_cache_t *cc, const sp_config_t *cfg,
                       int max_seq_len, void *stream);
void sp_cuda_cache_free(sp_cuda_cache_t *cc);

// ============================================================================
// Write path: raw KV → GPU VHT2 → Möbius reorder → band quantize → store
// ============================================================================
//
// d_k_vec: device pointer to raw K vector (head_dim floats, already RoPE'd)
// All operations run on the cache's CUDA stream.

void sp_cuda_write_k(sp_cuda_cache_t *cc,
                     int layer, int head, int pos,
                     const float *d_k_vec);

void sp_cuda_write_v(sp_cuda_cache_t *cc,
                     int layer, int head, int pos,
                     const float *d_v_vec);

// ============================================================================
// Read path: load → band dequantize → Möbius unreorder → VHT2 (self-inverse) → KV
// ============================================================================
//
// d_k_out: device pointer for reconstructed K vector (head_dim floats)

void sp_cuda_read_k(const sp_cuda_cache_t *cc,
                    int layer, int head, int pos,
                    float *d_k_out);

void sp_cuda_read_v(const sp_cuda_cache_t *cc,
                    int layer, int head, int pos,
                    float *d_v_out);

// ============================================================================
// Batch operations — process entire sequence positions at once
// ============================================================================
//
// For prefill: compress all tokens in one kernel launch.
// n_pos: number of positions to process
// d_k_vecs: [n_pos][head_dim] contiguous K vectors on device

void sp_cuda_write_k_batch(sp_cuda_cache_t *cc,
                           int layer, int head,
                           int start_pos, int n_pos,
                           const float *d_k_vecs);

void sp_cuda_write_v_batch(sp_cuda_cache_t *cc,
                           int layer, int head,
                           int start_pos, int n_pos,
                           const float *d_v_vecs);

// Read batch: reconstruct n_pos K vectors into contiguous output
void sp_cuda_read_k_batch(const sp_cuda_cache_t *cc,
                          int layer, int head,
                          int start_pos, int n_pos,
                          float *d_k_out);

void sp_cuda_read_v_batch(const sp_cuda_cache_t *cc,
                          int layer, int head,
                          int start_pos, int n_pos,
                          float *d_v_out);

// ============================================================================
// CUDA kernel launchers (exposed for testing)
// ============================================================================

// In-place VHT2 on GPU. Self-inverse — call twice to recover the input.
// Processes n_vecs independent vectors of length n. At p=2 (power-of-2 n)
// this is the 1/√2-per-stage Hartley butterfly; non-power-of-2 dims
// dispatch the staged Vilenkin kernel from shannon_prime_sqfree.cu.
void sp_cuda_vht2_forward(float *d_data, int n, int n_vecs, void *stream);

// Apply Möbius permutation on GPU
void sp_cuda_mobius_reorder(float *d_data, const int *d_order,
                            int n, int n_vecs, void *stream);
void sp_cuda_mobius_unreorder(float *d_data, const int *d_order,
                              int n, int n_vecs, void *stream);

// Banded quantize/dequantize on GPU
void sp_cuda_band_quantize(const float *d_input, void *d_output,
                           const sp_band_config_t *bc,
                           int n_vecs, void *stream);
void sp_cuda_band_dequantize(const void *d_input, float *d_output,
                             const sp_band_config_t *bc,
                             int n_vecs, void *stream);

// ============================================================================
// Diagnostics
// ============================================================================

// Print GPU memory usage
void sp_cuda_print_memory(const sp_cuda_cache_t *cc);

// ============================================================================
// Sqfree GPU cache (step 3 MVP — no spinor yet)
// ============================================================================
//
// GPU-resident variant of sp_sqfree_cache_t. Compressed K/V blocks live
// in VRAM; compress/decompress pipelines (sqfree_pad → Vilenkin →
// Knight extract → band quantize + Möbius predict + residual quantize)
// run as CUDA kernels. Spinor sheet bit storage is deferred (full scope
// in docs/STEP3-GPU-SQFREE-CACHE.md).

typedef struct {
    sp_config_t     config;
    int             pad_dim;
    int             sk_k;
    int             n_res;
    int             n_terms;
    int             residual_bits;
    int             use_spinor;
    int             use_skel_mobius;    // unused in MVP

    sp_band_config_t k_bands;
    sp_band_config_t v_bands;

    int            *d_skeleton_idx;
    int            *d_residual_idx;
    int            *d_csr_offsets;
    int            *d_csr_skel_slot;
    int            *d_csr_mu_sign;   // int32 on GPU (converted from int8_t on init)

    int            *d_vilenkin_factors;
    int             n_factors;

    unsigned char  *d_k_cache;
    unsigned char  *d_v_cache;
    int             bytes_per_pos_k;
    int             bytes_per_pos_v;
    int             max_seq_len;
    int             n_slots;

    float          *d_pad_scratch;
    float          *d_coeff_scratch;
    float          *d_skel_scratch;
    float          *d_pred_scratch;
    float          *d_dev_scratch;
    unsigned char  *d_levels_scratch;
    float          *d_mag_scratch;
    // Spinor scratches. Always allocated so use_spinor can be selected at
    // runtime without reallocating the cache. Sizes are tiny compared to
    // d_{k,v}_cache; cost is a few hundred KB at pad_dim=154, max_seq=1k.
    float          *d_actual_scratch;   // [n_res * max_seq_len]  fp32
    unsigned char  *d_sheet_scratch;    // [max_seq_len * ((n_res+7)/8)]
    void           *stream;
} sp_cuda_sqfree_cache_t;

int  sp_cuda_sqfree_cache_init(sp_cuda_sqfree_cache_t *cc,
                                const sp_config_t *cfg,
                                int max_seq_len,
                                int residual_bits,
                                int use_spinor,
                                void *stream);
void sp_cuda_sqfree_cache_free(sp_cuda_sqfree_cache_t *cc);

void sp_cuda_sqfree_write_k(sp_cuda_sqfree_cache_t *cc,
                             int layer, int head, int pos,
                             const float *d_k_vec);
void sp_cuda_sqfree_write_v(sp_cuda_sqfree_cache_t *cc,
                             int layer, int head, int pos,
                             const float *d_v_vec);
void sp_cuda_sqfree_read_k(const sp_cuda_sqfree_cache_t *cc,
                            int layer, int head, int pos,
                            float *d_k_out);
void sp_cuda_sqfree_read_v(const sp_cuda_sqfree_cache_t *cc,
                            int layer, int head, int pos,
                            float *d_v_out);

// Batched read: process n_pos contiguous positions for one
// (layer, head) in a single kernel-dispatch series. Output layout
// is [n_pos × head_dim] contiguous (vec-major). ~9 total kernel
// launches vs ~9*n_pos for the per-vec path — the step-3
// kernel-launch-overhead fix.
void sp_cuda_sqfree_read_k_batch(const sp_cuda_sqfree_cache_t *cc,
                                  int layer, int head,
                                  int start_pos, int n_pos,
                                  float *d_k_out);
void sp_cuda_sqfree_read_v_batch(const sp_cuda_sqfree_cache_t *cc,
                                  int layer, int head,
                                  int start_pos, int n_pos,
                                  float *d_v_out);

// ============================================================================
// Cold storage layer — tiered GPU ↔ CPU ↔ disk offload
// ============================================================================
//
// Each layer's compressed K/V data can be staged through three tiers:
//   1. Hot:  GPU VRAM (d_k_cache / d_v_cache in sp_cuda_cache_t)
//   2. Warm: CPU pinned RAM (h_pinned below, cudaHostAlloc'd)
//   3. Cold: Disk (via sp_*_cache_save/load in core library)
//
// The cold layer tracks per-layer ring-buffer state for incremental
// GPU→CPU writeback. When cold_max_mb is set, the CPU buffer wraps;
// evicted GPU positions are zeroed to reclaim VRAM working set.

typedef struct {
    uint8_t *h_pinned;           // cudaHostAlloc'd CPU staging buffer
    int64_t  capacity;           // h_pinned bytes allocated
    int      packed_stride;      // bytes per position per head (from band config)
    int      n_heads;            // heads per layer (needed for stride calc)

    // Ring-buffer tracking
    int64_t  write_head;         // byte offset for next write in h_pinned
    int64_t  oldest_pos;         // oldest valid position in ring; -1 = unknown
    int64_t  newest_pos;         // newest position written
    bool     ring_mode;          // true after first wrap-around

    bool     initialized;
} sp_cuda_cold_layer_t;

// Allocate a cold layer with `cap_bytes` of pinned CPU RAM.
// packed_stride = bc->total_bytes for the relevant band config.
int  sp_cuda_cold_layer_init(sp_cuda_cold_layer_t *cl,
                             int64_t cap_bytes,
                             int packed_stride, int n_heads);
void sp_cuda_cold_layer_free(sp_cuda_cold_layer_t *cl);

// Async GPU→CPU copy of positions [start_pos, start_pos+n_pos) for one
// layer. d_packed points into the GPU cache at the layer's K or V slab.
// On ring-buffer wrap, oldest_pos advances. Returns 0 on success.
int  sp_cuda_cold_writeback(sp_cuda_cold_layer_t *cl,
                            const void *d_packed,
                            int start_pos, int n_pos,
                            void *stream);

// Async CPU→GPU restore: copy positions [0, n_pos) from h_pinned back
// to d_packed on the GPU. Returns 0 on success.
int  sp_cuda_cold_restore(const sp_cuda_cold_layer_t *cl,
                          void *d_packed,
                          int n_pos,
                          void *stream);

// Convenience: D2H / H2D async wrappers for arbitrary buffers.
void sp_cuda_d2h_async(void *dst_host, const void *src_device,
                       int64_t n_bytes, void *stream);
void sp_cuda_h2d_async(void *dst_device, const void *src_host,
                       int64_t n_bytes, void *stream);

// ============================================================================
// Hierarchical GPU cache — maximum compression, GPU-resident
// ============================================================================
//
// GPU-resident variant of sp_hier_cache_t. Compressed K/V blocks and
// predictor W matrices live in VRAM. Write pipeline:
//   pad → Vilenkin → gather skeleton → band-quantize skeleton
//   → W·skeleton prediction → residual deviation → quantize residual
//   → pack bits → store
// Read pipeline is the reverse. Per-slot W matrices are uploaded from
// the CPU hier_cache after calibration via sp_cuda_hier_cache_upload_W.

typedef struct {
    sp_config_t      config;
    int              pad_dim;
    int              n_skeleton;
    int              n_target;
    int              target_res_bits;
    int              n_slots;           // n_layers × n_heads_kv
    int              max_seq_len;

    // Per-slot predictor W matrices in VRAM [n_slots][n_target × n_skeleton] fp16
    // Stored as a single flat allocation; slot s starts at offset
    // s * n_target * n_skeleton * sizeof(uint16_t).
    uint16_t        *d_W;               // [n_slots × n_target × n_skeleton] fp16

    // Index arrays (shared across all slots)
    int             *d_skeleton_idx;    // [n_skeleton] indices into [0, pad_dim)
    int             *d_target_idx;      // [n_target]   indices into [0, pad_dim)

    // Vilenkin factors for in-place transform
    int             *d_vilenkin_factors; // staged prime factors
    int              n_factors;

    // Band config for skeleton quantisation
    sp_band_config_t skel_bands;
    int             *d_skel_band_bits;  // band_bits array in device memory

    // Compressed storage in VRAM
    unsigned char   *d_k_cache;         // [n_slots][max_seq × bytes_per_pos_k]
    unsigned char   *d_v_cache;         // [n_slots][max_seq × bytes_per_pos_v]
    int              bytes_per_pos_k;   // skel_bands.total_bytes + 4 + res_bytes
    int              bytes_per_pos_v;

    // GPU scratch buffers (allocated once, reused every write/read)
    float           *d_pad_scratch;     // [pad_dim]
    float           *d_coeff_scratch;   // [pad_dim]
    float           *d_skel_scratch;    // [n_skeleton]
    float           *d_pred_scratch;    // [n_target]
    float           *d_dev_scratch;     // [n_target] deviation = actual - predicted
    float           *d_mag_scratch;     // [1] magnitude scalar
    unsigned char   *d_levels_scratch;  // [n_target]
    void            *stream;
} sp_cuda_hier_cache_t;

// Initialize a GPU-resident hierarchical cache. The CPU-side
// sp_hier_cache_t must already be initialised (for pad_dim, skeleton/
// target indices, skel_bands, and Vilenkin factors). This function
// copies the structural metadata to VRAM — W matrices are uploaded
// separately after calibration.
int  sp_cuda_hier_cache_init(sp_cuda_hier_cache_t *hc,
                              const sp_config_t *cfg,
                              int pad_dim, int n_skeleton, int n_target,
                              int target_res_bits,
                              const int *skeleton_idx,
                              const int *target_idx,
                              const sp_band_config_t *skel_bands,
                              int max_seq_len, int n_slots,
                              void *stream);
void sp_cuda_hier_cache_free(sp_cuda_hier_cache_t *hc);

// Upload calibrated W matrices from CPU hier_cache. Call once after
// calibration. W_all is a flat [n_slots][n_target × n_skeleton]
// uint16_t array (row-major, same layout as sp_hier_predictor_t::W).
int  sp_cuda_hier_cache_upload_W(sp_cuda_hier_cache_t *hc,
                                  const uint16_t *W_all);

// Write/read single vector (layer, head, pos)
void sp_cuda_hier_write_k(sp_cuda_hier_cache_t *hc,
                           int layer, int head, int pos,
                           const float *d_k_vec);
void sp_cuda_hier_write_v(sp_cuda_hier_cache_t *hc,
                           int layer, int head, int pos,
                           const float *d_v_vec);
void sp_cuda_hier_read_k(const sp_cuda_hier_cache_t *hc,
                          int layer, int head, int pos,
                          float *d_k_out);
void sp_cuda_hier_read_v(const sp_cuda_hier_cache_t *hc,
                          int layer, int head, int pos,
                          float *d_v_out);

#ifdef __cplusplus
}
#endif

#endif // SHANNON_PRIME_CUDA_H
