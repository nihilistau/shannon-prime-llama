// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

#ifndef SHANNON_PRIME_ADRENO_H
#define SHANNON_PRIME_ADRENO_H

#include "../../core/shannon_prime.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Mobile Backend — Qualcomm Snapdragon
// ============================================================================
//
// Targets the full Snapdragon compute stack:
//
// ┌─────────────────────────────────────────────────────────────────────┐
// │ Compute Unit         │ VHT2 Role        │ API                      │
// │──────────────────────│──────────────────│──────────────────────────│
// │ Cortex-X2 (prime)    │ Decode writeback │ NEON intrinsics          │
// │ Cortex-A710 (gold)   │ Parallel layers  │ NEON + thread affinity   │
// │ Cortex-A510 (silver) │ Background work  │ NEON + low-power         │
// │ Adreno 730 GPU       │ Prefill batch    │ Vulkan compute shaders   │
// │ Hexagon V69 DSP      │ VHT2 butterfly   │ HVX intrinsics (V73+)   │
// └─────────────────────────────────────────────────────────────────────┘
//
// Primary target: Samsung S22 Ultra (SM-S908U)
//   SoC: Snapdragon 8 Gen 1 (SM8450)
//   CPU: 1×X2 @3.0GHz + 3×A710 @2.5GHz + 4×A510 @1.8GHz (ARMv9)
//   GPU: Adreno 730, 8 SPs, Vulkan 1.1, 900 MHz
//   DSP: Hexagon V69, HVX 1024-bit (NO HMX — requires V73+)
//   RAM: 12 GB LPDDR5 @ 3200 MHz (51.2 GB/s)
//
// Pipeline:
//   Prefill (batch):  Adreno 730 Vulkan compute → batch VHT2 + quant
//   Decode (single):  CPU NEON on prime core → single-vector VHT2 + quant
//   Background:       silver cores → async cache maintenance
//   Future (8G2+):    Hexagon HVX for VHT2 butterfly
//
// Production results (Dolphin 1B Q8_0):
//   PPL:      14.24 → 13.20 (Möbius 5/4/4)
//   K corr:   0.9972
//   V corr:   0.9960
//   Speed:    1.79 → 3.57 tok/s (2× speedup)
//   Writeback: 37–42 ms/batch (all 16 layers)

// ============================================================================
// Feature detection
// ============================================================================

typedef struct {
    // ARM CPU features
    int has_neon;           // ARMv7+ NEON (baseline, always on ARMv8+)
    int has_fp16_arith;     // ARMv8.2 FP16 arithmetic (FEAT_FP16)
    int has_dotprod;        // ARMv8.2 dot product (sdot/udot)
    int has_i8mm;           // ARMv8.6 int8 matrix multiply (smmla)
    int has_sve;            // ARMv9 SVE (Scalable Vector Extension)
    int has_sve2;           // ARMv9 SVE2

    // CPU topology
    int n_big_cores;        // X2/X3/X4 prime + A710/A715 gold
    int n_little_cores;     // A510/A520 silver
    int prime_core_id;      // CPU ID of the prime (fastest) core

    // Qualcomm-specific
    int has_adreno;         // Adreno GPU detected (via Vulkan/OpenCL)
    int adreno_model;       // 730, 740, 750, etc.
    int has_hexagon;        // Hexagon DSP accessible
    int hexagon_version;    // V69, V73, V75, etc. (0 if unavailable)
    int has_hvx;            // HVX (Hexagon Vector eXtension)
    int hvx_width;          // HVX register width in bits (1024 on V69+)
    int has_hmx;            // HMX (Hexagon Matrix eXtension, V73+ only)
} sp_mobile_caps_t;

// Detect hardware capabilities at runtime.
void sp_mobile_detect_caps(sp_mobile_caps_t *caps);
void sp_mobile_print_caps(const sp_mobile_caps_t *caps);

// ============================================================================
// NEON SIMD Operations
// ============================================================================
//
// Three tiers of NEON implementation:
//
// Tier 1 (baseline): float32x4_t — 4 f32 per op
//   Works on ALL ARMv8+ devices. 128-bit NEON registers.
//   VHT2 butterfly: 4 elements per add/sub cycle.
//
// Tier 2 (fp16 arith): float16x8_t — 8 f16 per op
//   Requires FEAT_FP16 (ARMv8.2+, all Cortex-A76 and newer).
//   Snapdragon 8 Gen 1 X2/A710 cores have this.
//   VHT2 butterfly on fp16: HALVES memory bandwidth, 8 elements/op.
//   head_dim=128 × 2 bytes = 256 bytes = two 128-bit loads.
//
// Tier 3 (dotprod/i8mm): for banded quant inner loops
//   Quantized coefficient operations with sdot/smmla.
//   Available on Cortex-X2 (8 Gen 1) and newer.

// VHT2 on f32 (Tier 1 — universal). Self-inverse; call twice to invert.
void sp_neon_vht2_f32(float *data, int n);

// VHT2 on f16 (Tier 2 — requires FEAT_FP16)
// Operates directly on __fp16 / _Float16 arrays.
// Avoids fp16↔fp32 conversion overhead entirely. Self-inverse.
void sp_neon_vht2_f16(void *data, int n);

// Absmax reduction
float sp_neon_absmax_f32(const float *data, int n);

// fp16 conversion (Tier 1 — uses vcvt even without FEAT_FP16)
void sp_neon_f16_to_f32(const void *src, float *dst, int n);
void sp_neon_f32_to_f16(const float *src, void *dst, int n);

// Banded quantize/dequantize (dispatches to best available tier)
void sp_neon_band_quantize(const float *vht2_coeffs, uint8_t *out,
                           const sp_band_config_t *bc);
void sp_neon_band_dequantize(const uint8_t *in, float *vht2_coeffs,
                             const sp_band_config_t *bc);

// ============================================================================
// Hexagon HVX Operations (V69+, V73+ for full feature set)
// ============================================================================
//
// HVX registers are 1024 bits = 128 bytes.
// head_dim=128 × 4 bytes (f32) = 512 bytes = half an HVX register.
// head_dim=128 × 2 bytes (f16) = 256 bytes = quarter HVX register.
// head_dim=64  × 4 bytes (f32) = 256 bytes = quarter HVX register.
//
// The VHT2 p=2 butterfly is a perfect fit: each pass does add/sub on
// vector halves, mapping directly to HVX vadd/vsub.
//
// Availability:
//   V69 (8 Gen 1):  HVX only, no HMX. Good for VHT2 butterfly.
//   V73 (8 Gen 2):  HVX + HMX fp16. Full acceleration.
//   V75 (8 Gen 3+): HVX + HMX improved. Best performance.
//
// Access: requires Hexagon SDK + FastRPC. On non-rooted devices,
// needs signed DSP binaries (skel library).

#ifdef SP_HEXAGON_ENABLED

// Hexagon HVX entry points — currently stubs; implementation lives
// behind the same SP_HEXAGON_ENABLED guard and a signed FastRPC skel.
void sp_hvx_vht2_f32(float *data, int n);
void sp_hvx_vht2_f16(void *data, int n);
void sp_hvx_band_quantize(const float *vht2_coeffs, uint8_t *out,
                          const sp_band_config_t *bc);
void sp_hvx_band_dequantize(const uint8_t *in, float *vht2_coeffs,
                            const sp_band_config_t *bc);

#endif // SP_HEXAGON_ENABLED

// ============================================================================
// Thread affinity — big.LITTLE aware scheduling
// ============================================================================
//
// Snapdragon 8 Gen 1 CPU topology:
//   Core 0-3: Cortex-A510 (silver, 1.8 GHz, power-efficient)
//   Core 4-6: Cortex-A710 (gold, 2.5 GHz, balanced)
//   Core 7:   Cortex-X2 (prime, 3.0 GHz, highest single-thread)
//
// For VHT2:
//   Decode writeback (latency-critical): pin to prime core (7)
//   Parallel layer compression: spread across gold cores (4-6)
//   Background maintenance: use silver cores (0-3)

typedef enum {
    SP_AFFINITY_PRIME  = 0,  // Fastest single core (X2/X3/X4)
    SP_AFFINITY_GOLD   = 1,  // Performance cores (A710/A715/A720)
    SP_AFFINITY_SILVER = 2,  // Efficiency cores (A510/A520)
    SP_AFFINITY_ANY    = 3,  // OS scheduler decides
} sp_core_affinity_t;

// Set current thread's CPU affinity.
// Returns 0 on success, -1 on failure (non-Linux, permissions).
int sp_set_thread_affinity(sp_core_affinity_t affinity,
                           const sp_mobile_caps_t *caps);

// ============================================================================
// Mobile Shadow Cache — full pipeline
// ============================================================================

typedef enum {
    SP_COMPUTE_CPU_NEON  = 0,  // CPU with NEON (always available)
    SP_COMPUTE_CPU_FP16  = 1,  // CPU with native fp16 arithmetic
    SP_COMPUTE_VULKAN    = 2,  // Adreno GPU via Vulkan compute
    SP_COMPUTE_HEXAGON   = 3,  // Hexagon DSP via HVX
    SP_COMPUTE_AUTO      = 4,  // Auto-select best for each operation
} sp_compute_target_t;

typedef struct {
    sp_config_t         config;
    sp_band_config_t    k_bands;
    sp_band_config_t    v_bands;
    sp_mobius_mask_t     mobius_mask;
    sp_mobile_caps_t    caps;

    // Compressed storage (CPU-resident)
    uint8_t            *k_cache;
    uint8_t            *v_cache;
    int                 max_seq_len;

    // NEON-aligned scratch buffers (16-byte aligned for NEON, 128-byte for HVX)
    float              *scratch_a;   // Primary working buffer [head_dim]
    float              *scratch_b;   // Secondary (for permutation) [head_dim]
    void               *scratch_f16; // fp16 buffer [head_dim] (__fp16*)

    // Compute dispatch
    sp_compute_target_t decode_target;   // What to use for single-vector decode
    sp_compute_target_t prefill_target;  // What to use for batch prefill

    // Performance counters
    uint64_t            n_writes;
    uint64_t            n_reads;
    double              total_write_us;
    double              total_read_us;
} sp_adreno_cache_t;

int  sp_adreno_cache_init(sp_adreno_cache_t *ac, const sp_config_t *cfg,
                          int max_seq_len);
void sp_adreno_cache_free(sp_adreno_cache_t *ac);

// Write path (dispatches to best compute target)
void sp_adreno_write_k(sp_adreno_cache_t *ac,
                       int layer, int head, int pos,
                       const float *k_vec);
void sp_adreno_write_v(sp_adreno_cache_t *ac,
                       int layer, int head, int pos,
                       const float *v_vec);

// Read path
void sp_adreno_read_k(const sp_adreno_cache_t *ac,
                      int layer, int head, int pos,
                      float *k_out);
void sp_adreno_read_v(const sp_adreno_cache_t *ac,
                      int layer, int head, int pos,
                      float *v_out);

// fp16 write path — avoids fp16→f32 conversion when model runs in fp16
void sp_adreno_write_k_f16(sp_adreno_cache_t *ac,
                           int layer, int head, int pos,
                           const void *k_vec_f16);
void sp_adreno_write_v_f16(sp_adreno_cache_t *ac,
                           int layer, int head, int pos,
                           const void *v_vec_f16);

// Batch variants — contiguous [n_pos × head_dim] input/output, single-
// dispatch into a tight loop that reuses the persistent scratch buffers
// on sp_adreno_cache_t (no malloc per vector, NEON pipeline stays warm).
void sp_adreno_write_k_batch(sp_adreno_cache_t *ac,
                             int layer, int head,
                             int start_pos, int n_pos,
                             const float *k_vecs);
void sp_adreno_write_v_batch(sp_adreno_cache_t *ac,
                             int layer, int head,
                             int start_pos, int n_pos,
                             const float *v_vecs);
void sp_adreno_read_k_batch (const sp_adreno_cache_t *ac,
                             int layer, int head,
                             int start_pos, int n_pos,
                             float *k_out);
void sp_adreno_read_v_batch (const sp_adreno_cache_t *ac,
                             int layer, int head,
                             int start_pos, int n_pos,
                             float *v_out);

// ============================================================================
// Diagnostics
// ============================================================================

float sp_adreno_bench_writeback(sp_adreno_cache_t *ac);
void  sp_adreno_print_stats(const sp_adreno_cache_t *ac);
int   sp_adreno_check_neon(void);

#ifdef __cplusplus
}
#endif

#endif // SHANNON_PRIME_ADRENO_H
