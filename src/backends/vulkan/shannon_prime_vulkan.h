// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

#ifndef SHANNON_PRIME_VULKAN_H
#define SHANNON_PRIME_VULKAN_H

#include "../../core/shannon_prime.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Vulkan Shadow Cache
// ============================================================================
//
// Compute shader pipeline for VHT2 KV cache compression.
// Integration target: llama.cpp Vulkan backend, mobile inference (Adreno, Mali).
//
// Pipeline stages (each is a compute shader dispatch):
//   1. VHT2 staged Hartley (log_p(n) passes at p=2; Vilenkin for sqfree dims)
//   2. Möbius permutation (scatter/gather)
//   3. Band quantize (per-band abs-max reduction + pack)
//   4. Band dequantize (unpack + scale)
//   5. VHT2 (self-inverse — same kernel, no 1/N scale)

// Opaque handle — hides Vulkan implementation details
typedef struct sp_vulkan_cache_s sp_vulkan_cache_t;

// ============================================================================
// Lifecycle
// ============================================================================

// Initialize Vulkan compute pipeline and allocate cache buffers.
// vk_device, vk_queue: existing Vulkan device/queue from the inference engine.
// If NULL, creates its own device (standalone mode).
// gpu_index: which VkPhysicalDevice to use (0 = first, 1 = second, ...).
//            Only used in standalone mode (vk_device == NULL).
//
// Returns 0 on success, -1 on failure.
int sp_vulkan_cache_init(sp_vulkan_cache_t **cc,
                         const sp_config_t *cfg,
                         int max_seq_len,
                         void *vk_device,    // VkDevice or NULL
                         void *vk_queue,     // VkQueue or NULL
                         int gpu_index);     // physical device index (standalone)

void sp_vulkan_cache_free(sp_vulkan_cache_t *cc);

// ============================================================================
// Write path
// ============================================================================

// Write K/V from host memory (CPU → GPU → compress → store)
void sp_vulkan_write_k(sp_vulkan_cache_t *cc,
                       int layer, int head, int pos,
                       const float *k_vec);  // host pointer

void sp_vulkan_write_v(sp_vulkan_cache_t *cc,
                       int layer, int head, int pos,
                       const float *v_vec);

// Write K/V from existing Vulkan buffer (zero-copy if same device)
void sp_vulkan_write_k_buffer(sp_vulkan_cache_t *cc,
                              int layer, int head, int pos,
                              void *vk_buffer,       // VkBuffer
                              size_t offset);        // byte offset

void sp_vulkan_write_v_buffer(sp_vulkan_cache_t *cc,
                              int layer, int head, int pos,
                              void *vk_buffer,
                              size_t offset);

// ============================================================================
// Read path
// ============================================================================

// Read K/V to host memory (decompress → GPU → CPU)
void sp_vulkan_read_k(const sp_vulkan_cache_t *cc,
                      int layer, int head, int pos,
                      float *k_out);         // host pointer

void sp_vulkan_read_v(const sp_vulkan_cache_t *cc,
                      int layer, int head, int pos,
                      float *v_out);

// Read K/V into existing Vulkan buffer
void sp_vulkan_read_k_buffer(const sp_vulkan_cache_t *cc,
                             int layer, int head, int pos,
                             void *vk_buffer,
                             size_t offset);

void sp_vulkan_read_v_buffer(const sp_vulkan_cache_t *cc,
                             int layer, int head, int pos,
                             void *vk_buffer,
                             size_t offset);

// ============================================================================
// Batch operations
// ============================================================================

void sp_vulkan_write_k_batch(sp_vulkan_cache_t *cc,
                             int layer, int head,
                             int start_pos, int n_pos,
                             const float *k_vecs);

void sp_vulkan_read_k_batch(const sp_vulkan_cache_t *cc,
                            int layer, int head,
                            int start_pos, int n_pos,
                            float *k_out);

// ============================================================================
// Diagnostics
// ============================================================================

void sp_vulkan_print_memory(const sp_vulkan_cache_t *cc);

// Check device capabilities (shared memory size, max workgroup, etc.)
int sp_vulkan_check_device(const sp_vulkan_cache_t *cc);

// Diagnostic: dispatch only the vilenkin.comp stage on the GPU for one
// vector. Used by tests to isolate shader correctness from the rest of
// the pipeline (band quant, Möbius). `inout` is a host pointer with `hd`
// floats; replaced in-place with the GPU-computed VHT2-forward result.
// Returns 0 on success, negative on Vulkan / device error.
int sp_vulkan_diag_vht2_forward(sp_vulkan_cache_t *cc, float *inout, int hd);

// Diagnostic: run GPU band_quantize + band_dequantize round-trip on `in`
// (hd floats) using the band-config layout implied by the cache. Writes
// the reconstructed vector to `out`. Used to isolate quant-shader
// correctness from the VHT2/Möbius stages.
//   which = 0 -> use cc->k_bands
//   which = 1 -> use cc->v_bands
int sp_vulkan_diag_band_roundtrip(sp_vulkan_cache_t *cc, int which,
                                   const float *in, float *out, int hd);

#ifdef __cplusplus
}
#endif

#endif // SHANNON_PRIME_VULKAN_H
