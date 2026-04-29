// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available - contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

#ifndef SHANNON_PRIME_HEXAGON_H
#define SHANNON_PRIME_HEXAGON_H

#include "shannon_prime.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Hexagon Backend - Qualcomm DSP for KV Cache Compression
// ============================================================================
//
// Targets the Hexagon DSP block on Snapdragon SoCs. Co-processor: runs in
// parallel with the ARM cores and the Adreno GPU at low power. SP's
// banded quantize/dequantize maps cleanly onto HVX vector intrinsics; the
// VHT2 butterfly maps cleanly onto HMX matrix intrinsics on V73+ (S8G3+).
//
// Hardware target ladder:
//
//   Hexagon V69 (Snapdragon 8 Gen 1)     HVX 1024-bit only.        Primary.
//   Hexagon V73 (Snapdragon 8 Gen 2)     HVX + HMX 256x256 int8.   Future.
//   Hexagon V75 (Snapdragon 8 Gen 3)     HVX + HMX + tensor cores. Future.
//
// Primary validation device: Samsung Galaxy S22 Ultra (SM8450, V69).
// SDK requirement: Qualcomm Hexagon SDK 5.x (toolv87 / DSP_ARCH=v69).

// ============================================================================
// Capability / feature detection
// ============================================================================

typedef struct {
    int has_dsp;                 // Hexagon DSP accessible via FastRPC
    int dsp_version;             // V69, V73, V75 (raw int)
    int has_hvx;                 // HVX vector eXtension
    int hvx_width_bits;          // 1024 on V69+
    int has_hmx;                 // HMX matrix eXtension (V73+)
    int max_threads;             // DSP hardware threads (typically 4-6)
    long long shared_mem_bytes;  // Shared CPU<->DSP physical buffer budget
} sp_hexagon_caps_t;

int sp_hexagon_caps_probe(sp_hexagon_caps_t *caps);
void sp_hexagon_caps_print(const sp_hexagon_caps_t *caps);

// ============================================================================
// Lifecycle
// ============================================================================

typedef struct sp_hexagon_ctx_s sp_hexagon_ctx_t;

sp_hexagon_ctx_t *sp_hexagon_init(const sp_config_t *cfg);
void sp_hexagon_free(sp_hexagon_ctx_t *ctx);

// ============================================================================
// Round-trip operations (single-vector smoke / engine-API path)
// ============================================================================

void *sp_hexagon_alloc(sp_hexagon_ctx_t *ctx, size_t n_bytes);
void  sp_hexagon_free_shared(sp_hexagon_ctx_t *ctx, void *ptr);

int sp_hexagon_round_trip_k(sp_hexagon_ctx_t *ctx,
                             const uint16_t *in_fp16,
                             uint16_t *out_fp16);
int sp_hexagon_round_trip_v(sp_hexagon_ctx_t *ctx,
                             const uint16_t *in_fp16,
                             uint16_t *out_fp16);
int sp_hexagon_round_trip_k_batch(sp_hexagon_ctx_t *ctx,
                                   const uint16_t *in_fp16,
                                   uint16_t *out_fp16,
                                   int n_vectors);
int sp_hexagon_band_dequantize_partial(sp_hexagon_ctx_t *ctx,
                                        const uint8_t *in_packed,
                                        uint16_t *out_fp16,
                                        int max_bands);

// ============================================================================
// Diagnostics
// ============================================================================

size_t sp_hexagon_memory_in_use(const sp_hexagon_ctx_t *ctx);
long long sp_hexagon_last_call_cycles(const sp_hexagon_ctx_t *ctx);

// ============================================================================
// sp_hexagon_cache_t - per-position packed-byte storage with DSP offload
// ============================================================================
//
// Mirrors the sp_shadow_cache_t / sp_adreno_cache_t shape: per-(layer,head)
// slots, each a contiguous max_seq * bc.total_bytes byte array indexed by
// position. The compress/decompress work for each per-position write or
// read goes through ONE FastRPC dispatch into compress_f32 /
// decompress_f32 on the cDSP, with the packed bytes living in
// rpcmem-backed pages so the marshal copy is bypassed (the DSP gets the
// same physical pages via the SMMU).
//
// Lifecycle: ctx = sp_hexagon_init; sp_hexagon_cache_init(cache, ctx,...);
//   ... per-position writes/reads ...; sp_hexagon_cache_free(cache);
//   sp_hexagon_free(ctx). Cache borrows ctx - ctx must outlive cache.

typedef struct {
    sp_hexagon_ctx_t  *ctx;
    sp_config_t        cfg_snapshot;
    int                max_seq_len;
    int                n_slots;
    sp_band_config_t   k_bands;
    sp_band_config_t   v_bands;
    uint8_t          **k_cache;
    uint8_t          **v_cache;
    float             *vec_in_f32;
    float             *vec_out_f32;
} sp_hexagon_cache_t;

int  sp_hexagon_cache_init(sp_hexagon_cache_t *cache,
                            sp_hexagon_ctx_t *ctx,
                            const sp_config_t *cfg,
                            int max_seq_len);
void sp_hexagon_cache_free(sp_hexagon_cache_t *cache);

void sp_hexagon_cache_write_k(sp_hexagon_cache_t *cache,
                               int layer, int head, int pos,
                               const float *k_vec);
void sp_hexagon_cache_write_v(sp_hexagon_cache_t *cache,
                               int layer, int head, int pos,
                               const float *v_vec);
void sp_hexagon_cache_read_k(const sp_hexagon_cache_t *cache,
                              int layer, int head, int pos,
                              float *k_out);
void sp_hexagon_cache_read_v(const sp_hexagon_cache_t *cache,
                              int layer, int head, int pos,
                              float *v_out);
void sp_hexagon_cache_read_k_partial(const sp_hexagon_cache_t *cache,
                                      int layer, int head, int pos,
                                      float *k_out, int max_bands);
void sp_hexagon_cache_read_v_partial(const sp_hexagon_cache_t *cache,
                                      int layer, int head, int pos,
                                      float *v_out, int max_bands);
void sp_hexagon_cache_write_k_batch(sp_hexagon_cache_t *cache,
                                     int layer, int head,
                                     int start_pos, int n_pos,
                                     const float *k_vecs);
void sp_hexagon_cache_write_v_batch(sp_hexagon_cache_t *cache,
                                     int layer, int head,
                                     int start_pos, int n_pos,
                                     const float *v_vecs);
void sp_hexagon_cache_read_k_batch(const sp_hexagon_cache_t *cache,
                                    int layer, int head,
                                    int start_pos, int n_pos,
                                    float *k_out);
void sp_hexagon_cache_read_v_batch(const sp_hexagon_cache_t *cache,
                                    int layer, int head,
                                    int start_pos, int n_pos,
                                    float *v_out);
void sp_hexagon_cache_clear_range(sp_hexagon_cache_t *cache,
                                   int start_pos, int end_pos);

#ifdef __cplusplus
}
#endif

#endif // SHANNON_PRIME_HEXAGON_H
