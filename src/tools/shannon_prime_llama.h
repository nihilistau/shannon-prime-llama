// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

#ifndef SHANNON_PRIME_LLAMA_H
#define SHANNON_PRIME_LLAMA_H

// ============================================================================
// llama.cpp Integration Layer
// ============================================================================
//
// This header provides the hook points for integrating VHT2 compressed
// KV cache into llama.cpp. It wraps the core shadow cache API into
// the llama.cpp KV cache interface.
//
// Integration approach:
//   1. Shannon-Prime intercepts KV writes AFTER RoPE is applied
//   2. Raw K/V vectors are compressed into the shadow cache
//   3. On attention computation, K/V are reconstructed from shadow cache
//   4. The original KV cache can be disabled or used as a fallback
//
// Environment variables (no rebuild required):
//   SHANNON_PRIME_ENABLED=1         Enable VHT2 compression
//   SHANNON_PRIME_K_BITS=5,5,4,3   K band bit allocation
//   SHANNON_PRIME_V_BITS=3          V bit allocation (flat)
//   SHANNON_PRIME_MOBIUS=1          Enable Möbius mask (default on)
//   SHANNON_PRIME_VERBOSE=1         Print diagnostics
//
// Activation:
//   #include "shannon_prime_llama.h"
//
//   // In llama_kv_cache_init() or equivalent:
//   sp_llama_ctx_t *sp = sp_llama_init(model_params);
//
//   // In the KV write path (after RoPE):
//   sp_llama_write_kv(sp, layer, head, pos, k_vec, v_vec);
//
//   // In the attention path (when reading cached KV):
//   sp_llama_read_k(sp, layer, head, pos, k_out);
//   sp_llama_read_v(sp, layer, head, pos, v_out);
//
//   // Cleanup:
//   sp_llama_free(sp);

#include "../core/shannon_prime.h"

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Model parameters (extracted from llama_model)
// ============================================================================

typedef struct {
    int head_dim;        // hparams.n_embd_head_k (typically 64 or 128)
    int n_layers;        // hparams.n_layer
    int n_heads_kv;      // hparams.n_head_kv (GQA-aware)
    int max_seq_len;     // n_ctx from context params

    // Backend selection
    enum {
        SP_BACKEND_CPU  = 0,  // Core C (universal fallback)
        SP_BACKEND_CUDA = 1,  // CUDA kernels
        SP_BACKEND_VULKAN = 2, // Vulkan compute shaders
        SP_BACKEND_ADRENO = 3, // ARM NEON (mobile)
    } backend;

    // Optional: CUDA stream or Vulkan queue for GPU backends
    void *gpu_context;   // cudaStream_t or VkQueue, NULL for CPU
} sp_llama_params_t;

// ============================================================================
// Context (opaque)
// ============================================================================

typedef struct sp_llama_ctx_s sp_llama_ctx_t;

// ============================================================================
// Lifecycle
// ============================================================================

// Initialize from environment variables + model params.
// Returns NULL if SHANNON_PRIME_ENABLED is not set.
sp_llama_ctx_t *sp_llama_init(const sp_llama_params_t *params);

// Initialize with explicit config (for programmatic use).
sp_llama_ctx_t *sp_llama_init_config(const sp_llama_params_t *params,
                                     const sp_config_t *cfg);

void sp_llama_free(sp_llama_ctx_t *ctx);

// ============================================================================
// KV Write Path
// ============================================================================
//
// Called AFTER RoPE is applied to K, before storing in the KV cache.
// The shadow cache compresses and stores the K/V vectors.
//
// In llama.cpp, this hooks into:
//   llama_kv_cache_seq_add() or the equivalent write path
//   after ggml_rope_ext() has been applied to K

// Write a single K+V pair
void sp_llama_write_kv(sp_llama_ctx_t *ctx,
                       int layer, int head, int pos,
                       const float *k_vec, const float *v_vec);

// Write K and V separately (for architectures where they're computed separately)
void sp_llama_write_k(sp_llama_ctx_t *ctx,
                      int layer, int head, int pos,
                      const float *k_vec);

void sp_llama_write_v(sp_llama_ctx_t *ctx,
                      int layer, int head, int pos,
                      const float *v_vec);

// Batch write for prefill (all positions at once)
void sp_llama_write_k_batch(sp_llama_ctx_t *ctx,
                            int layer, int head,
                            int start_pos, int n_pos,
                            const float *k_vecs);

void sp_llama_write_v_batch(sp_llama_ctx_t *ctx,
                            int layer, int head,
                            int start_pos, int n_pos,
                            const float *v_vecs);

// ============================================================================
// KV Read Path
// ============================================================================
//
// Called during attention computation to reconstruct K/V from the shadow cache.
//
// In llama.cpp, this hooks into:
//   The attention kernel where K^T is computed for Q·K^T scoring

void sp_llama_read_k(const sp_llama_ctx_t *ctx,
                     int layer, int head, int pos,
                     float *k_out);

void sp_llama_read_v(const sp_llama_ctx_t *ctx,
                     int layer, int head, int pos,
                     float *v_out);

// Batch read (reconstruct n_pos vectors into contiguous output)
void sp_llama_read_k_batch(const sp_llama_ctx_t *ctx,
                           int layer, int head,
                           int start_pos, int n_pos,
                           float *k_out);

void sp_llama_read_v_batch(const sp_llama_ctx_t *ctx,
                           int layer, int head,
                           int start_pos, int n_pos,
                           float *v_out);

// ============================================================================
// Cache Management
// ============================================================================

// Clear cache for a sequence range (for KV cache eviction)
void sp_llama_clear_range(sp_llama_ctx_t *ctx,
                          int start_pos, int end_pos);

// Get current memory usage
typedef struct {
    size_t compressed_bytes;   // K+V compressed total
    size_t baseline_bytes;     // What fp16 would use
    float  compression_ratio;  // baseline / compressed
    int    n_positions;        // Current sequence length
} sp_llama_memory_t;

sp_llama_memory_t sp_llama_memory(const sp_llama_ctx_t *ctx);

// ============================================================================
// Diagnostics
// ============================================================================

// Validate a K vector: compress → decompress → measure correlation
float sp_llama_validate_k(sp_llama_ctx_t *ctx,
                          const float *k_vec, int head_dim);

// Print configuration to stderr
void sp_llama_print_config(const sp_llama_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif // SHANNON_PRIME_LLAMA_H
