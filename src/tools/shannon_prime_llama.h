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
//   SHANNON_PRIME_HIERARCHICAL=1    Enable hierarchical Kronecker sub-projection
//                                   (maximum compression path, requires >=24 token prompt)
//   SHANNON_PRIME_HIER_W_PATH=<f>  Path to precomputed W sidecar (.sp_hier_W.bin)
//                                   Skips cold-start calibration when provided
//   SHANNON_PRIME_FP8=1             Use FP8 (E4M3) for band quantization
//                                   (advisory at the bridge layer; honoured
//                                   only by backends with an fp8 path —
//                                   currently the engine's CUDA backend
//                                   built with SP_ENGINE_FP8=ON. CPU /
//                                   Adreno bridge logs a warning and
//                                   falls back to int.)
//   SHANNON_PRIME_PE=1              Enable PrimePE lattice RoPE injection (default: on)
//   SHANNON_PRIME_PE=0              Disable PrimePE (use standard geometric RoPE)
//   SHANNON_PRIME_PE_ALPHA=0.17     Lattice blend ratio (default: 0.17, range 0.15-0.22)
//   SHANNON_PRIME_FREQ_BASE=10000   Override rope_freq_base (auto-detected when possible)
//
// Speculative-decoding draft overrides (see sp_llama_init_with_role and
// docs/SPECULATIVE-DECODING.md). When a context is initialised with role
// SP_LLAMA_ROLE_DRAFT, each SHANNON_PRIME_X lookup tries the DRAFT_-prefixed
// version first and falls back to the global one if unset:
//   SHANNON_PRIME_DRAFT_K_BITS=2,1     Aggressive draft K bands
//   SHANNON_PRIME_DRAFT_V_BITS=1       Aggressive draft V bits
//   SHANNON_PRIME_DRAFT_MOBIUS=1       Override Möbius for draft only
//   SHANNON_PRIME_DRAFT_PE=0           Disable PrimePE on draft only
//   SHANNON_PRIME_DRAFT_PRESET=aggressive   Shortcut — picks K=2,1 V=1
//                                            ("ternary"=2,2 / "ship"=defaults)
//
// Ternary noise-tail (5/5/4/1.58 — band 3 stored as {-1,0,+1} at 2 bpp
// regardless of K_BITS[3]). Comma-separated band indices, 0-based:
//   SHANNON_PRIME_K_TERNARY_BANDS=3       Ternary on K band 3 only
//   SHANNON_PRIME_V_TERNARY_BANDS=        (empty / unset = no V ternary)
//   SHANNON_PRIME_DRAFT_K_TERNARY_BANDS=2,3   Differential: ternary on
//                                              the draft only, more bands
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

// Context role — disambiguates the target and draft contexts in a
// speculative-decoding pair. Used by sp_llama_init_with_role() to look
// up role-specific environment variables (e.g. SHANNON_PRIME_DRAFT_K_BITS)
// before falling back to the global SHANNON_PRIME_* names.
//
// SP_LLAMA_ROLE_DEFAULT preserves the historical behaviour: only the
// global SHANNON_PRIME_* env vars are read. sp_llama_init() is now a
// thin wrapper that calls sp_llama_init_with_role(p, ROLE_DEFAULT).
//
// SP_LLAMA_ROLE_TARGET behaves identically to ROLE_DEFAULT today —
// reserved for future use if target-only env vars are ever introduced.
//
// SP_LLAMA_ROLE_DRAFT routes through the role-aware getenv path: each
// SHANNON_PRIME_X lookup first tries SHANNON_PRIME_DRAFT_X, then falls
// back to SHANNON_PRIME_X. This lets a speculative deployment compress
// the draft cache more aggressively than the target without affecting
// the target's defaults. Also enables the SHANNON_PRIME_DRAFT_PRESET
// shortcut.
typedef enum {
    SP_LLAMA_ROLE_DEFAULT = 0,
    SP_LLAMA_ROLE_TARGET  = 1,
    SP_LLAMA_ROLE_DRAFT   = 2,
} sp_llama_role_t;

// ============================================================================
// Lifecycle
// ============================================================================

// Initialize from environment variables + model params.
// Returns NULL if SHANNON_PRIME_ENABLED is not set.
//
// Equivalent to sp_llama_init_with_role(params, SP_LLAMA_ROLE_DEFAULT).
sp_llama_ctx_t *sp_llama_init(const sp_llama_params_t *params);

// Role-aware initialiser. Same as sp_llama_init() except env-var lookups
// honour `role` — when role==SP_LLAMA_ROLE_DRAFT, each lookup tries
// SHANNON_PRIME_DRAFT_X before falling back to SHANNON_PRIME_X. Intended
// for speculative-decoding integrations that want differential KV
// compression on the draft context.
//
// Today's caller-side workflow:
//   sp_llama_ctx_t *target = sp_llama_init_with_role(&p, SP_LLAMA_ROLE_TARGET);
//   sp_llama_ctx_t *draft  = sp_llama_init_with_role(&q, SP_LLAMA_ROLE_DRAFT);
//
// The llama.cpp speculative-init patch surgery to actually call this
// from the draft path is a separate work item (see FUTURE-WORK §8a).
// Until that lands, behaviour is identical to sp_llama_init().
sp_llama_ctx_t *sp_llama_init_with_role(const sp_llama_params_t *params,
                                        sp_llama_role_t role);

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

// ============================================================================
// PrimePE — Lattice-Aligned RoPE Frequency Auto-Injection
// ============================================================================
//
// Returns a malloc'd float array of length n_freqs containing per-dimension
// frequency multipliers for ggml_rope_ext. The caller (llama.cpp patch)
// writes these into model.layers[*].rope_freqs at model load time.
//
// When SHANNON_PRIME_ENABLED=1, this is called automatically during
// sp_llama_init() and the factors are stored on the context. The llama.cpp
// patch retrieves them via sp_llama_get_freq_factors() and injects them
// into the model's RoPE path — zero user action required.
//
// Environment variables:
//   SHANNON_PRIME_PE=1          Enable PrimePE injection (default: on when SP enabled)
//   SHANNON_PRIME_PE=0          Disable (use pure geometric RoPE)
//   SHANNON_PRIME_PE_ALPHA=0.17 Blend ratio (default: 0.17)
//
// Returns NULL if PrimePE is disabled or context has no freq factors.
// The returned pointer is owned by the context — do NOT free it.
const float *sp_llama_get_freq_factors(const sp_llama_ctx_t *ctx);

// Number of frequency factors (= head_dim / 2). Returns 0 if ctx is NULL.
int sp_llama_get_n_freqs(const sp_llama_ctx_t *ctx);

#ifdef __cplusplus
}
#endif

#endif // SHANNON_PRIME_LLAMA_H
