// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

// Mobile backend: NEON SIMD, Hexagon HVX stubs, big.LITTLE affinity.
// Compiles on any platform — NEON/Hexagon paths activate via feature detection.

#define _POSIX_C_SOURCE 199309L
#define _GNU_SOURCE  // For sched_setaffinity on Linux/Android

#include "shannon_prime_adreno.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// Platform detection
// ============================================================================

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  #include <arm_neon.h>
  #define HAS_NEON 1
#else
  #define HAS_NEON 0
#endif

// ARMv8.2 fp16 arithmetic: __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  #define HAS_FP16_ARITH 1
#else
  #define HAS_FP16_ARITH 0
#endif

// ARMv8.2 dot product: sdot/udot
#if defined(__ARM_FEATURE_DOTPROD)
  #define HAS_DOTPROD 1
#else
  #define HAS_DOTPROD 0
#endif

// ARMv8.6 int8 matrix multiply: smmla
#if defined(__ARM_FEATURE_MATMUL_INT8)
  #define HAS_I8MM 1
#else
  #define HAS_I8MM 0
#endif

// SVE / SVE2
#if defined(__ARM_FEATURE_SVE)
  #define HAS_SVE 1
#else
  #define HAS_SVE 0
#endif

#if defined(__ARM_FEATURE_SVE2)
  #define HAS_SVE2 1
#else
  #define HAS_SVE2 0
#endif

// Linux thread affinity
#if defined(__linux__) || defined(__ANDROID__)
  #include <sched.h>
  #include <unistd.h>
  #define HAS_AFFINITY 1
#else
  #define HAS_AFFINITY 0
#endif

// Android-specific CPU topology detection
#if defined(__ANDROID__)
  #define HAS_ANDROID_CPUINFO 1
#else
  #define HAS_ANDROID_CPUINFO 0
#endif

// ============================================================================
// Feature detection
// ============================================================================

// Parse /sys/devices/system/cpu/ to detect big.LITTLE topology on Linux/Android
static void detect_cpu_topology(sp_mobile_caps_t *caps) {
    caps->n_big_cores = 0;
    caps->n_little_cores = 0;
    caps->prime_core_id = -1;

#if HAS_AFFINITY
    long n_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (n_cpus <= 0) n_cpus = 4;

    int max_freq = 0;

    for (int i = 0; i < (int)n_cpus && i < 16; i++) {
        char path[256];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/cpufreq/cpuinfo_max_freq", i);
        FILE *f = fopen(path, "r");
        int freq = 0;
        if (f) {
            if (fscanf(f, "%d", &freq) != 1) freq = 0;
            fclose(f);
        }

        // Heuristic: >2.2 GHz = big, <=2.2 GHz = little
        // Snapdragon 8 Gen 1: prime=3.0, gold=2.5, silver=1.8
        if (freq > 2200000) {
            caps->n_big_cores++;
            if (freq > max_freq) {
                max_freq = freq;
                caps->prime_core_id = i;
            }
        } else if (freq > 0) {
            caps->n_little_cores++;
        }
    }

    // Fallback if cpufreq not readable
    if (caps->n_big_cores == 0 && caps->n_little_cores == 0) {
        caps->n_big_cores = (int)(n_cpus > 4 ? 4 : n_cpus);
        caps->n_little_cores = (int)(n_cpus - caps->n_big_cores);
        caps->prime_core_id = (int)(n_cpus - 1); // Highest ID is usually prime
    }
#else
    // Non-Linux: assume 4 big cores
    caps->n_big_cores = 4;
    caps->n_little_cores = 0;
    caps->prime_core_id = 0;
#endif
}

void sp_mobile_detect_caps(sp_mobile_caps_t *caps) {
    memset(caps, 0, sizeof(*caps));

    // Compile-time features
    caps->has_neon       = HAS_NEON;
    caps->has_fp16_arith = HAS_FP16_ARITH;
    caps->has_dotprod    = HAS_DOTPROD;
    caps->has_i8mm       = HAS_I8MM;
    caps->has_sve        = HAS_SVE;
    caps->has_sve2       = HAS_SVE2;

    // CPU topology
    detect_cpu_topology(caps);

    // GPU/DSP detection would require Vulkan/OpenCL/Hexagon SDK queries.
    // For now, mark as unavailable — runtime init will probe.
    caps->has_adreno      = 0;
    caps->adreno_model    = 0;
    caps->has_hexagon     = 0;
    caps->hexagon_version = 0;
    caps->has_hvx         = 0;
    caps->hvx_width       = 0;
    caps->has_hmx         = 0;

#ifdef SP_HEXAGON_ENABLED
    caps->has_hexagon = 1;
    caps->has_hvx     = 1;
    caps->hvx_width   = 1024;
    // V69 = 8 Gen 1, V73 = 8 Gen 2, V75 = 8 Gen 3
    // This would be queried via Hexagon SDK at runtime
#endif
}

void sp_mobile_print_caps(const sp_mobile_caps_t *caps) {
    fprintf(stderr, "[Shannon-Prime Mobile] Hardware capabilities:\n");
    fprintf(stderr, "  NEON:         %s\n", caps->has_neon ? "yes" : "no");
    fprintf(stderr, "  FP16 arith:   %s\n", caps->has_fp16_arith ? "yes (Tier 2)" : "no (Tier 1)");
    fprintf(stderr, "  Dot product:  %s\n", caps->has_dotprod ? "yes" : "no");
    fprintf(stderr, "  I8MM:         %s\n", caps->has_i8mm ? "yes" : "no");
    fprintf(stderr, "  SVE/SVE2:     %s/%s\n",
            caps->has_sve ? "yes" : "no", caps->has_sve2 ? "yes" : "no");
    fprintf(stderr, "  CPU cores:    %d big + %d little (prime=#%d)\n",
            caps->n_big_cores, caps->n_little_cores, caps->prime_core_id);
    fprintf(stderr, "  Adreno:       %s", caps->has_adreno ? "yes" : "no");
    if (caps->adreno_model > 0)
        fprintf(stderr, " (model %d)", caps->adreno_model);
    fprintf(stderr, "\n");
    fprintf(stderr, "  Hexagon:      %s", caps->has_hexagon ? "yes" : "no");
    if (caps->hexagon_version > 0)
        fprintf(stderr, " (V%d, HVX=%d-bit%s)",
                caps->hexagon_version, caps->hvx_width,
                caps->has_hmx ? ", HMX" : "");
    fprintf(stderr, "\n");
}

// ============================================================================
// Thread affinity
// ============================================================================

int sp_set_thread_affinity(sp_core_affinity_t affinity,
                           const sp_mobile_caps_t *caps) {
#if HAS_AFFINITY
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);

    long n_cpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (n_cpus <= 0) return -1;

    switch (affinity) {
    case SP_AFFINITY_PRIME:
        if (caps->prime_core_id >= 0) {
            CPU_SET(caps->prime_core_id, &cpuset);
        } else {
            CPU_SET((int)(n_cpus - 1), &cpuset);
        }
        break;

    case SP_AFFINITY_GOLD:
        // Gold cores: typically cores n_little .. n_cpus-2
        for (int i = caps->n_little_cores;
             i < (int)n_cpus - 1 && i < 16; i++) {
            CPU_SET(i, &cpuset);
        }
        if (CPU_COUNT(&cpuset) == 0) {
            // Fallback: use all big cores
            for (int i = caps->n_little_cores; i < (int)n_cpus && i < 16; i++)
                CPU_SET(i, &cpuset);
        }
        break;

    case SP_AFFINITY_SILVER:
        for (int i = 0; i < caps->n_little_cores && i < 16; i++) {
            CPU_SET(i, &cpuset);
        }
        if (CPU_COUNT(&cpuset) == 0) {
            // No little cores detected — use any
            for (int i = 0; i < (int)n_cpus && i < 16; i++)
                CPU_SET(i, &cpuset);
        }
        break;

    case SP_AFFINITY_ANY:
    default:
        for (int i = 0; i < (int)n_cpus && i < 16; i++)
            CPU_SET(i, &cpuset);
        break;
    }

    return sched_setaffinity(0, sizeof(cpuset), &cpuset);
#else
    (void)affinity; (void)caps;
    return -1;
#endif
}

// ============================================================================
// NEON VHT2 — Tier 1 (f32, 4 elements/op, p=2 stages orthonormal via 1/√N)
// ============================================================================
//
// At p=2 VHT2 is the Hadamard butterfly with 1/√2 per stage, producing
// an orthonormal self-inverse transform (no /N on the inverse). The NEON
// code below runs the unnormalised butterfly for speed, then applies a
// single 1/√N multiply at the end — numerically equivalent to the per-stage
// 1/√2 (the end-multiply keeps NEON pipelines saturated in the butterfly
// loop). Non-power-of-2 dims dispatch to the scalar core sp_vht2_forward_f32
// which handles the staged Hartley for primes {2,3,5,7,11}.

#if HAS_NEON

void sp_neon_vht2_f32(float *data, int n) {
    // Guard: for non-power-of-2, fall back to the core staged VHT2
    // (scalar; Adreno's sqfree call sites are rare compared to p=2).
    if (n <= 0 || (n & (n - 1)) != 0) {
        sp_vht2_forward_f32(data, n);
        return;
    }

    // Unnormalised p=2 Hartley butterfly
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            int j = 0;
            // NEON: 4 float32 per cycle
            for (; j + 3 < len; j += 4) {
                float32x4_t u = vld1q_f32(&data[i + j]);
                float32x4_t v = vld1q_f32(&data[i + j + len]);
                vst1q_f32(&data[i + j],       vaddq_f32(u, v));
                vst1q_f32(&data[i + j + len],  vsubq_f32(u, v));
            }
            // Scalar tail (len < 4 in early passes)
            for (; j < len; j++) {
                float u = data[i + j];
                float v = data[i + j + len];
                data[i + j]       = u + v;
                data[i + j + len] = u - v;
            }
        }
    }

    // Orthonormal normalisation: multiply by 1/√N so the transform is
    // self-inverse (VHT2 semantics). Two applications reproduce the input
    // without any further scaling.
    const float inv_sqrt_n = 1.0f / sqrtf((float)n);
    const float32x4_t vinv = vdupq_n_f32(inv_sqrt_n);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(&data[i]);
        vst1q_f32(&data[i], vmulq_f32(v, vinv));
    }
    for (; i < n; i++) data[i] *= inv_sqrt_n;
}

float sp_neon_absmax_f32(const float *data, int n) {
    float32x4_t vmax = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t v = vld1q_f32(&data[i]);
        vmax = vmaxq_f32(vmax, vabsq_f32(v));
    }
    // Horizontal max — pairwise reduction
    float32x2_t vmax2 = vpmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
    vmax2 = vpmax_f32(vmax2, vmax2);
    float amax = vget_lane_f32(vmax2, 0);
    // Scalar tail
    for (; i < n; i++) {
        float a = fabsf(data[i]);
        if (a > amax) amax = a;
    }
    return amax;
}

#else // Scalar fallback — route through the core staged VHT2

void sp_neon_vht2_f32(float *data, int n) {
    sp_vht2_forward_f32(data, n);
}

float sp_neon_absmax_f32(const float *data, int n) {
    float amax = 0.0f;
    for (int i = 0; i < n; i++) {
        float a = fabsf(data[i]);
        if (a > amax) amax = a;
    }
    return amax;
}

#endif // HAS_NEON

// ============================================================================
// NEON VHT2 — Tier 2 (f16, 8 elements/op, requires FEAT_FP16)
// ============================================================================

#if HAS_NEON && HAS_FP16_ARITH

void sp_neon_vht2_f16(void *data, int n) {
    // Non-power-of-2: fall back via the f32 path (which dispatches the
    // core staged VHT2 for sqfree dims).
    if (n <= 0 || (n & (n - 1)) != 0) {
        float *tmp = (float *)malloc(n * sizeof(float));
        if (!tmp) return;
        sp_neon_f16_to_f32(data, tmp, n);
        sp_vht2_forward_f32(tmp, n);
        sp_neon_f32_to_f16(tmp, data, n);
        free(tmp);
        return;
    }

    __fp16 *d = (__fp16 *)data;

    // Unnormalised butterfly in fp16
    for (int len = 1; len < n; len <<= 1) {
        for (int i = 0; i < n; i += len << 1) {
            int j = 0;
            // NEON fp16: 8 elements per cycle
            for (; j + 7 < len; j += 8) {
                float16x8_t u = vld1q_f16(&d[i + j]);
                float16x8_t v = vld1q_f16(&d[i + j + len]);
                vst1q_f16(&d[i + j],       vaddq_f16(u, v));
                vst1q_f16(&d[i + j + len],  vsubq_f16(u, v));
            }
            // 4-wide tail
            for (; j + 3 < len; j += 4) {
                float16x4_t u = vld1_f16(&d[i + j]);
                float16x4_t v = vld1_f16(&d[i + j + len]);
                vst1_f16(&d[i + j],       vadd_f16(u, v));
                vst1_f16(&d[i + j + len],  vsub_f16(u, v));
            }
            // Scalar tail
            for (; j < len; j++) {
                __fp16 u = d[i + j];
                __fp16 v = d[i + j + len];
                d[i + j]       = u + v;
                d[i + j + len] = u - v;
            }
        }
    }

    // VHT2 end-normalisation: 1/√N so the transform is self-inverse.
    const __fp16 inv_sqrt_n = (__fp16)(1.0f / sqrtf((float)n));
    float16x8_t vinv = vdupq_n_f16(inv_sqrt_n);
    int i = 0;
    for (; i + 7 < n; i += 8) {
        float16x8_t v = vld1q_f16(&d[i]);
        vst1q_f16(&d[i], vmulq_f16(v, vinv));
    }
    for (; i < n; i++) d[i] *= inv_sqrt_n;
}

#else // No native fp16 arithmetic — convert through f32

void sp_neon_vht2_f16(void *data, int n) {
    // Convert fp16 → f32, VHT2 in f32, convert back
    float *tmp = (float *)malloc(n * sizeof(float));
    if (!tmp) return;
    sp_neon_f16_to_f32(data, tmp, n);
    sp_neon_vht2_f32(tmp, n);
    sp_neon_f32_to_f16(tmp, data, n);
    free(tmp);
}

#endif // HAS_FP16_ARITH

// ============================================================================
// fp16 ↔ fp32 conversion
// ============================================================================

void sp_neon_f16_to_f32(const void *src, float *dst, int n) {
#if HAS_NEON
    const uint16_t *s = (const uint16_t *)src;
    int i = 0;
    // NEON vcvt_f32_f16: converts 4 fp16 → 4 fp32
    for (; i + 3 < n; i += 4) {
        // Load 4 × 16-bit values, reinterpret as float16x4_t
        uint16x4_t raw = vld1_u16(&s[i]);
        float16x4_t h = vreinterpret_f16_u16(raw);
        float32x4_t f = vcvt_f32_f16(h);
        vst1q_f32(&dst[i], f);
    }
    // Scalar tail
    for (; i < n; i++) {
        // Use core conversion for remaining elements
        uint16_t h = s[i];
        uint32_t sign = ((uint32_t)(h >> 15)) << 31;
        uint32_t exp  = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        uint32_t f;
        if (exp == 0) f = sign;
        else if (exp == 31) f = sign | 0x7F800000 | (mant << 13);
        else f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        memcpy(&dst[i], &f, sizeof(float));
    }
#else
    const uint16_t *s = (const uint16_t *)src;
    for (int i = 0; i < n; i++) {
        uint16_t h = s[i];
        uint32_t sign = ((uint32_t)(h >> 15)) << 31;
        uint32_t exp  = (h >> 10) & 0x1F;
        uint32_t mant = h & 0x3FF;
        uint32_t f;
        if (exp == 0) f = sign;
        else if (exp == 31) f = sign | 0x7F800000 | (mant << 13);
        else f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        memcpy(&dst[i], &f, sizeof(float));
    }
#endif
}

void sp_neon_f32_to_f16(const float *src, void *dst, int n) {
#if HAS_NEON
    uint16_t *d = (uint16_t *)dst;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t f = vld1q_f32(&src[i]);
        float16x4_t h = vcvt_f16_f32(f);
        uint16x4_t raw = vreinterpret_u16_f16(h);
        vst1_u16(&d[i], raw);
    }
    for (; i < n; i++) {
        uint32_t f;
        memcpy(&f, &src[i], sizeof(uint32_t));
        uint16_t sign = (f >> 16) & 0x8000;
        int exp = ((f >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = f & 0x7FFFFF;
        if (exp <= 0) d[i] = sign;
        else if (exp >= 31) d[i] = sign | 0x7C00;
        else d[i] = sign | (exp << 10) | (mant >> 13);
    }
#else
    uint16_t *d = (uint16_t *)dst;
    for (int i = 0; i < n; i++) {
        uint32_t f;
        memcpy(&f, &src[i], sizeof(uint32_t));
        uint16_t sign = (f >> 16) & 0x8000;
        int exp = ((f >> 23) & 0xFF) - 127 + 15;
        uint32_t mant = f & 0x7FFFFF;
        if (exp <= 0) d[i] = sign;
        else if (exp >= 31) d[i] = sign | 0x7C00;
        else d[i] = sign | (exp << 10) | (mant >> 13);
    }
#endif
}

// ============================================================================
// Banded quantization — dispatches to core (NEON acceleration in absmax)
// ============================================================================

void sp_neon_band_quantize(const float *vht2_coeffs, uint8_t *out,
                           const sp_band_config_t *bc) {
    // The core quantization is bit-packing which is inherently sequential.
    // NEON helps with the absmax reduction (above) but the packing loop
    // is the same. Use the validated core implementation.
    sp_band_quantize(vht2_coeffs, out, bc);
}

void sp_neon_band_dequantize(const uint8_t *in, float *vht2_coeffs,
                             const sp_band_config_t *bc) {
    sp_band_dequantize(in, vht2_coeffs, bc);
}

// ============================================================================
// Hexagon HVX stubs — compiled only with SP_HEXAGON_ENABLED
// ============================================================================
//
// These are the function signatures that would be implemented using
// Hexagon SDK intrinsics (hexagon_protos.h, HVX_Vector types).
//
// The VHT2 p=2 butterfly on HVX is particularly elegant:
//   HVX register = 1024 bits = 128 bytes = 32 × float32
//   head_dim=128 = 4 HVX registers of f32
//   head_dim=128 = 2 HVX registers of f16
//   Each butterfly pass: Q6_V_vadd_VV / Q6_V_vsub_VV
//
// For Snapdragon 8 Gen 1 (V69): HVX only, no HMX.
// For 8 Gen 2+ (V73+): HVX + HMX. HMX can do fp16 matrix ops
// which could accelerate the entire VHT2 as a single matrix multiply.

#ifdef SP_HEXAGON_ENABLED

// These would include:
// #include <hexagon_protos.h>
// #include <hvx_hexagon_protos.h>

void sp_hvx_vht2_f32(float *data, int n) {
    // HVX implementation:
    // - Load 32 floats per HVX_Vector
    // - Butterfly with Q6_V_vadd_VV, Q6_V_vsub_VV
    // - For hd=128: 4 vectors, log2(128)=7 passes
    // - Each pass shuffles vector halves (Q6_V_vdeal_VV)
    //
    // Pseudocode:
    //   HVX_Vector v[4]; // 128 floats in 4 HVX registers
    //   for each butterfly pass:
    //     HVX_VectorPair p = Q6_W_vshuff_VVR(v[hi], v[lo], stride);
    //     v[lo] = Q6_V_vadd_VV(p.v0, p.v1);
    //     v[hi] = Q6_V_vsub_VV(p.v0, p.v1);
    //
    // Falls back to NEON for now:
    sp_neon_vht2_f32(data, n);
}

void sp_hvx_vht2_f16(void *data, int n) {
    // HVX fp16: 64 elements per HVX_Vector
    // head_dim=128 = 2 HVX registers
    // Even more efficient than f32 path
    sp_neon_vht2_f16(data, n);
}

void sp_hvx_band_quantize(const float *vht2_coeffs, uint8_t *out,
                          const sp_band_config_t *bc) {
    // HVX can accelerate the absmax reduction via Q6_V_vmax_VV
    // and the quantize-and-pack via vector shift/mask operations.
    sp_band_quantize(vht2_coeffs, out, bc);
}

void sp_hvx_band_dequantize(const uint8_t *in, float *vht2_coeffs,
                            const sp_band_config_t *bc) {
    sp_band_dequantize(in, vht2_coeffs, bc);
}

#endif // SP_HEXAGON_ENABLED

// ============================================================================
// Mobile Shadow Cache — full implementation
// ============================================================================

static sp_compute_target_t choose_decode_target(const sp_mobile_caps_t *caps) {
#ifdef SP_HEXAGON_ENABLED
    if (caps->has_hvx) return SP_COMPUTE_HEXAGON;
#endif
    if (caps->has_fp16_arith) return SP_COMPUTE_CPU_FP16;
    if (caps->has_neon) return SP_COMPUTE_CPU_NEON;
    return SP_COMPUTE_CPU_NEON;
}

static sp_compute_target_t choose_prefill_target(const sp_mobile_caps_t *caps) {
    if (caps->has_adreno) return SP_COMPUTE_VULKAN;
    return choose_decode_target(caps);
}

int sp_adreno_cache_init(sp_adreno_cache_t *ac, const sp_config_t *cfg,
                         int max_seq_len) {
    memset(ac, 0, sizeof(*ac));
    memcpy(&ac->config, cfg, sizeof(sp_config_t));
    ac->max_seq_len = max_seq_len;

    sp_band_config_init(&ac->k_bands, cfg->head_dim,
                        cfg->k_n_bands, cfg->k_band_bits);
    sp_band_config_init(&ac->v_bands, cfg->head_dim,
                        cfg->v_n_bands, cfg->v_band_bits);

    if (cfg->use_mobius_mask) {
        sp_mobius_mask_init(&ac->mobius_mask, cfg->head_dim);
    }

    // Detect hardware
    sp_mobile_detect_caps(&ac->caps);

    // Choose compute targets
    ac->decode_target  = choose_decode_target(&ac->caps);
    ac->prefill_target = choose_prefill_target(&ac->caps);

    // Allocate compressed storage
    int n_slots = cfg->n_layers * cfg->n_heads_kv;
    size_t k_total = (size_t)n_slots * max_seq_len * ac->k_bands.total_bytes;
    size_t v_total = (size_t)n_slots * max_seq_len * ac->v_bands.total_bytes;
    ac->k_cache = (uint8_t *)calloc(1, k_total);
    ac->v_cache = (uint8_t *)calloc(1, v_total);

    // Allocate aligned scratch buffers
    // 128-byte alignment for potential HVX compatibility (1024-bit = 128 bytes)
    size_t align = 128;
    size_t sz_f32 = (size_t)cfg->head_dim * sizeof(float);
    size_t sz_f16 = (size_t)cfg->head_dim * sizeof(uint16_t);

    // posix_memalign for 128-byte aligned buffers
#if HAS_AFFINITY
    if (posix_memalign((void **)&ac->scratch_a, align, sz_f32) != 0)
        ac->scratch_a = (float *)malloc(sz_f32);
    if (posix_memalign((void **)&ac->scratch_b, align, sz_f32) != 0)
        ac->scratch_b = (float *)malloc(sz_f32);
    if (posix_memalign((void **)&ac->scratch_f16, align, sz_f16) != 0)
        ac->scratch_f16 = malloc(sz_f16);
#else
    ac->scratch_a   = (float *)malloc(sz_f32);
    ac->scratch_b   = (float *)malloc(sz_f32);
    ac->scratch_f16 = malloc(sz_f16);
#endif

    if (!ac->k_cache || !ac->v_cache || !ac->scratch_a || !ac->scratch_b)
        return -1;

    // Print configuration
    float compression = sp_compression_ratio(cfg);
    fprintf(stderr, "[Shannon-Prime Mobile] Cache: %.2f MB "
            "(vs %.2f MB fp16, %.1f× compression)\n",
            (k_total + v_total) / (1024.0 * 1024.0),
            (size_t)n_slots * max_seq_len * cfg->head_dim * 4 / (1024.0 * 1024.0),
            compression);

    const char *target_names[] = {"CPU/NEON", "CPU/FP16", "Vulkan", "Hexagon", "auto"};
    fprintf(stderr, "[Shannon-Prime Mobile] Decode: %s, Prefill: %s\n",
            target_names[ac->decode_target], target_names[ac->prefill_target]);

    return 0;
}

void sp_adreno_cache_free(sp_adreno_cache_t *ac) {
    free(ac->k_cache);
    free(ac->v_cache);
    free(ac->scratch_a);
    free(ac->scratch_b);
    free(ac->scratch_f16);
    if (ac->config.use_mobius_mask) {
        sp_mobius_mask_free(&ac->mobius_mask);
    }
}

// ============================================================================
// Write path — dispatches VHT2 to best available compute unit
// ============================================================================

static inline void do_vht2_f32(float *data, int n, sp_compute_target_t target) {
    switch (target) {
#ifdef SP_HEXAGON_ENABLED
    case SP_COMPUTE_HEXAGON:
        sp_hvx_vht2_f32(data, n);
        return;
#endif
    case SP_COMPUTE_CPU_FP16:
    case SP_COMPUTE_CPU_NEON:
    default:
        sp_neon_vht2_f32(data, n);
        return;
    }
}

void sp_adreno_write_k(sp_adreno_cache_t *ac,
                       int layer, int head, int pos,
                       const float *k_vec) {
    int hd = ac->config.head_dim;
    float *scratch = ac->scratch_a;

    memcpy(scratch, k_vec, hd * sizeof(float));

    // VHT2 forward
    do_vht2_f32(scratch, hd, ac->decode_target);

    // Möbius reorder — use ac->scratch_b as tmp (no malloc)
    if (ac->config.use_mobius_mask) {
        sp_mobius_reorder_ex(scratch, &ac->mobius_mask, ac->scratch_b);
    }

    // Quantize into cache
    int slot = layer * ac->config.n_heads_kv + head;
    size_t offset = ((size_t)slot * ac->max_seq_len + pos) * ac->k_bands.total_bytes;
    sp_neon_band_quantize(scratch, ac->k_cache + offset, &ac->k_bands);

    ac->n_writes++;
}

void sp_adreno_write_v(sp_adreno_cache_t *ac,
                       int layer, int head, int pos,
                       const float *v_vec) {
    int hd = ac->config.head_dim;
    float *scratch = ac->scratch_a;

    memcpy(scratch, v_vec, hd * sizeof(float));
    do_vht2_f32(scratch, hd, ac->decode_target);

    int slot = layer * ac->config.n_heads_kv + head;
    size_t offset = ((size_t)slot * ac->max_seq_len + pos) * ac->v_bands.total_bytes;
    sp_neon_band_quantize(scratch, ac->v_cache + offset, &ac->v_bands);

    ac->n_writes++;
}

// ============================================================================
// fp16 write path — zero conversion overhead when model is fp16
// ============================================================================

void sp_adreno_write_k_f16(sp_adreno_cache_t *ac,
                           int layer, int head, int pos,
                           const void *k_vec_f16) {
    int hd = ac->config.head_dim;

#if HAS_FP16_ARITH
    // Tier 2: VHT2 directly in fp16, then convert for quantization
    memcpy(ac->scratch_f16, k_vec_f16, hd * sizeof(uint16_t));
    sp_neon_vht2_f16(ac->scratch_f16, hd);
    // Convert to f32 for banded quantization
    sp_neon_f16_to_f32(ac->scratch_f16, ac->scratch_a, hd);
#else
    // Tier 1: convert fp16→f32 first, then VHT2 in f32
    sp_neon_f16_to_f32(k_vec_f16, ac->scratch_a, hd);
    do_vht2_f32(ac->scratch_a, hd, ac->decode_target);
#endif

    if (ac->config.use_mobius_mask) {
        sp_mobius_reorder_ex(ac->scratch_a, &ac->mobius_mask, ac->scratch_b);
    }

    int slot = layer * ac->config.n_heads_kv + head;
    size_t offset = ((size_t)slot * ac->max_seq_len + pos) * ac->k_bands.total_bytes;
    sp_neon_band_quantize(ac->scratch_a, ac->k_cache + offset, &ac->k_bands);

    ac->n_writes++;
}

void sp_adreno_write_v_f16(sp_adreno_cache_t *ac,
                           int layer, int head, int pos,
                           const void *v_vec_f16) {
    int hd = ac->config.head_dim;

#if HAS_FP16_ARITH
    memcpy(ac->scratch_f16, v_vec_f16, hd * sizeof(uint16_t));
    sp_neon_vht2_f16(ac->scratch_f16, hd);
    sp_neon_f16_to_f32(ac->scratch_f16, ac->scratch_a, hd);
#else
    sp_neon_f16_to_f32(v_vec_f16, ac->scratch_a, hd);
    do_vht2_f32(ac->scratch_a, hd, ac->decode_target);
#endif

    int slot = layer * ac->config.n_heads_kv + head;
    size_t offset = ((size_t)slot * ac->max_seq_len + pos) * ac->v_bands.total_bytes;
    sp_neon_band_quantize(ac->scratch_a, ac->v_cache + offset, &ac->v_bands);

    ac->n_writes++;
}

// ============================================================================
// Read path
// ============================================================================

void sp_adreno_read_k(const sp_adreno_cache_t *ac,
                      int layer, int head, int pos,
                      float *k_out) {
    int hd = ac->config.head_dim;

    int slot = layer * ac->config.n_heads_kv + head;
    size_t offset = ((size_t)slot * ac->max_seq_len + pos) * ac->k_bands.total_bytes;

    sp_neon_band_dequantize(ac->k_cache + offset, k_out, &ac->k_bands);

    if (ac->config.use_mobius_mask) {
        sp_mobius_unreorder_ex(k_out, &ac->mobius_mask,
                               ((sp_adreno_cache_t *)ac)->scratch_b);
    }

    do_vht2_f32(k_out, hd, ac->decode_target);  // self-inverse

    ((sp_adreno_cache_t *)ac)->n_reads++;
}

void sp_adreno_read_v(const sp_adreno_cache_t *ac,
                      int layer, int head, int pos,
                      float *v_out) {
    int hd = ac->config.head_dim;

    int slot = layer * ac->config.n_heads_kv + head;
    size_t offset = ((size_t)slot * ac->max_seq_len + pos) * ac->v_bands.total_bytes;

    sp_neon_band_dequantize(ac->v_cache + offset, v_out, &ac->v_bands);
    do_vht2_f32(v_out, hd, ac->decode_target);  // self-inverse

    ((sp_adreno_cache_t *)ac)->n_reads++;
}

// ============================================================================
// Batch variants — tight loop over singletons, reusing ac->scratch_{a,b}.
// Amortizes one level of function-call / dispatch overhead across n_pos
// vectors. The NEON inner loops (VHT2, Möbius gather, band quantize) stay
// warm in icache across iterations.
// ============================================================================

void sp_adreno_write_k_batch(sp_adreno_cache_t *ac,
                             int layer, int head,
                             int start_pos, int n_pos,
                             const float *k_vecs) {
    int hd = ac->config.head_dim;
    for (int i = 0; i < n_pos; i++) {
        sp_adreno_write_k(ac, layer, head, start_pos + i, k_vecs + (size_t)i * hd);
    }
}

void sp_adreno_write_v_batch(sp_adreno_cache_t *ac,
                             int layer, int head,
                             int start_pos, int n_pos,
                             const float *v_vecs) {
    int hd = ac->config.head_dim;
    for (int i = 0; i < n_pos; i++) {
        sp_adreno_write_v(ac, layer, head, start_pos + i, v_vecs + (size_t)i * hd);
    }
}

void sp_adreno_read_k_batch(const sp_adreno_cache_t *ac,
                            int layer, int head,
                            int start_pos, int n_pos,
                            float *k_out) {
    int hd = ac->config.head_dim;
    for (int i = 0; i < n_pos; i++) {
        sp_adreno_read_k(ac, layer, head, start_pos + i, k_out + (size_t)i * hd);
    }
}

void sp_adreno_read_v_batch(const sp_adreno_cache_t *ac,
                            int layer, int head,
                            int start_pos, int n_pos,
                            float *v_out) {
    int hd = ac->config.head_dim;
    for (int i = 0; i < n_pos; i++) {
        sp_adreno_read_v(ac, layer, head, start_pos + i, v_out + (size_t)i * hd);
    }
}

// ============================================================================
// Diagnostics
// ============================================================================

int sp_adreno_check_neon(void) {
    return HAS_NEON;
}

float sp_adreno_bench_writeback(sp_adreno_cache_t *ac) {
    int hd = ac->config.head_dim;
    float *dummy = (float *)malloc(hd * sizeof(float));
    for (int i = 0; i < hd; i++) dummy[i] = (float)i / hd;

    // Warm up
    sp_adreno_write_k(ac, 0, 0, 0, dummy);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int l = 0; l < ac->config.n_layers; l++) {
        for (int h = 0; h < ac->config.n_heads_kv; h++) {
            sp_adreno_write_k(ac, l, h, 0, dummy);
            sp_adreno_write_v(ac, l, h, 0, dummy);
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec - t0.tv_sec) * 1000.0 +
                (t1.tv_nsec - t0.tv_nsec) / 1e6;

    free(dummy);
    return (float)ms;
}

void sp_adreno_print_stats(const sp_adreno_cache_t *ac) {
    fprintf(stderr, "[Shannon-Prime Mobile] Stats:\n");
    fprintf(stderr, "  Writes: %llu\n", (unsigned long long)ac->n_writes);
    fprintf(stderr, "  Reads:  %llu\n", (unsigned long long)ac->n_reads);
    sp_mobile_print_caps(&ac->caps);
}
