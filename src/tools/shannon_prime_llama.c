// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

#include "shannon_prime_llama.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Build-time backend gates. Define SP_HAVE_ADRENO when linking against
// backends/adreno/shannon_prime_adreno.c. The other backends remain
// commented out here until their bridge cases are written.
#ifdef SP_HAVE_ADRENO
  #include "../backends/adreno/shannon_prime_adreno.h"
#endif

// ============================================================================
// Internal context
// ============================================================================

struct sp_llama_ctx_s {
    sp_llama_params_t  params;
    sp_config_t        config;

    // Backend-specific cache (exactly one is active).
    sp_shadow_cache_t  cpu_cache;      // CPU backend — always compiled
#ifdef SP_HAVE_ADRENO
    sp_adreno_cache_t  adreno_cache;   // ARM NEON (Tier 1/2) backend
#endif
    // sp_cuda_cache_t    cuda_cache;  // CUDA backend (when linked)
    // sp_vulkan_cache_t *vulkan_cache; // Vulkan backend

    int active_backend;  // Which backend is in use
    int n_positions;     // Current max written position
};

// ============================================================================
// Environment variable parsing
// ============================================================================

static int parse_env_bool(const char *name, int default_val) {
    const char *v = getenv(name);
    if (!v) return default_val;
    return (v[0] == '1' || v[0] == 'y' || v[0] == 'Y');
}

static void parse_env_bits(const char *name, int *bits, int n, const int *defaults) {
    const char *v = getenv(name);
    if (!v) {
        memcpy(bits, defaults, n * sizeof(int));
        return;
    }

    // Parse comma-separated: "5,5,4,3"
    int i = 0;
    const char *p = v;
    while (i < n && *p) {
        bits[i++] = atoi(p);
        while (*p && *p != ',') p++;
        if (*p == ',') p++;
    }
    // Fill remaining with last value
    while (i < n) {
        bits[i] = bits[i-1];
        i++;
    }
}

// Variant that infers the band count from the env string's comma count.
// Returns the number of values parsed (1..max_n), or 0 when the env var
// is unset. Callers can then use the returned count to override the
// default n_bands.
static int parse_env_bits_autocount(const char *name, int *bits, int max_n) {
    const char *v = getenv(name);
    if (!v || !*v) return 0;

    int i = 0;
    const char *p = v;
    while (i < max_n && *p) {
        bits[i++] = atoi(p);
        while (*p && *p != ',') p++;
        if (*p == ',') p++;
    }
    return i;
}

// ============================================================================
// Lifecycle
// ============================================================================

sp_llama_ctx_t *sp_llama_init(const sp_llama_params_t *params) {
    if (!parse_env_bool("SHANNON_PRIME_ENABLED", 0)) {
        return NULL;
    }

    sp_config_t cfg;
    sp_config_init(&cfg, params->head_dim, params->n_layers, params->n_heads_kv);

    // Adaptive K band count. Paper §3.5: band_size below ~32 starves the
    // band with too few coefficients to meaningfully split the energy
    // decay. hd>=128 → 4 bands (5/5/4/3, ~32 elts/band, paper's ship);
    // hd==64 → 3 bands (5/4/4, ~21 elts/band, paper's "mobile safe").
    //
    // Adaptive V bit width. Paper's flat 3-bit V is validated on hd=128
    // (Qwen3-8B); on hd=64 (Dolphin 1B) measured PPL at flat 3-bit is
    // catastrophic (≈119 on wikitext-2 32-chunk vs baseline ~11.6). A
    // bit-sweep at hd=64 shows 5-bit is the minimum that preserves
    // quality (PPL 13.4); banding V adds nothing on top (5/5 == flat 5).
    // So for hd=64 we keep V flat but bump to 5-bit. The "flat beats
    // banded" invariant is preserved; only the bit count is head-dim
    // adaptive.
    int k_n_bands_default;
    int v_bits_default;
    const int * k_defaults;
    int k_defaults_hd128[] = {5, 5, 4, 3};
    int k_defaults_hd64 [] = {5, 4, 4, 4}; // 4th entry unused unless env overrides
    if (params->head_dim >= 128) {
        k_n_bands_default = 4;
        k_defaults        = k_defaults_hd128;
        v_bits_default    = 3;  // paper ship
    } else {
        k_n_bands_default = 3;
        k_defaults        = k_defaults_hd64;
        v_bits_default    = 5;  // hd=64 minimum; 3-bit is catastrophic here
    }
    int v_defaults[] = { v_bits_default };
    cfg.k_n_bands = k_n_bands_default;

    // Environment overrides. If SHANNON_PRIME_K_BITS or _V_BITS are set,
    // the number of comma-separated values determines the band count
    // (capped by SP_MAX_BANDS). E.g. `SHANNON_PRIME_K_BITS=3,3,3,3,3`
    // → 5 bands at 3 bits each. When the env var is unset we fall back
    // to the head-dim-adaptive defaults.
    int k_override_n = parse_env_bits_autocount(
        "SHANNON_PRIME_K_BITS", cfg.k_band_bits, SP_MAX_BANDS);
    if (k_override_n > 0) {
        cfg.k_n_bands = k_override_n;
    } else {
        memcpy(cfg.k_band_bits, k_defaults, k_n_bands_default * sizeof(int));
    }

    int v_override_n = parse_env_bits_autocount(
        "SHANNON_PRIME_V_BITS", cfg.v_band_bits, SP_MAX_BANDS);
    if (v_override_n > 0) {
        cfg.v_n_bands = v_override_n;
    } else {
        cfg.v_band_bits[0] = v_bits_default;
        cfg.v_n_bands      = 1;
    }
    cfg.use_mobius_mask = parse_env_bool("SHANNON_PRIME_MOBIUS", 1);

    // Allow the caller to request a specific backend via env var.
    // Valid values: "cpu" (default), "adreno". Unknown values → caller's
    // params->backend is used unchanged.
    sp_llama_params_t p = *params;
    const char *b = getenv("SHANNON_PRIME_BACKEND");
    if (b) {
        if (strcmp(b, "cpu") == 0)            p.backend = SP_BACKEND_CPU;
#ifdef SP_HAVE_ADRENO
        else if (strcmp(b, "adreno") == 0)    p.backend = SP_BACKEND_ADRENO;
#endif
    }

    return sp_llama_init_config(&p, &cfg);
}

sp_llama_ctx_t *sp_llama_init_config(const sp_llama_params_t *params,
                                     const sp_config_t *cfg) {
    sp_llama_ctx_t *ctx = (sp_llama_ctx_t *)calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;

    memcpy(&ctx->params, params, sizeof(*params));
    memcpy(&ctx->config, cfg, sizeof(*cfg));

    // Initialize the appropriate backend
    ctx->active_backend = params->backend;

    switch (params->backend) {
    case SP_BACKEND_CPU:
    default: {
        if (sp_shadow_cache_init(&ctx->cpu_cache, cfg) != 0) {
            free(ctx);
            return NULL;
        }

        // Allocate cache storage
        int n_slots = cfg->n_layers * cfg->n_heads_kv;
        ctx->cpu_cache.k_cache = (uint8_t **)calloc(n_slots, sizeof(uint8_t *));
        ctx->cpu_cache.v_cache = (uint8_t **)calloc(n_slots, sizeof(uint8_t *));

        for (int i = 0; i < n_slots; i++) {
            ctx->cpu_cache.k_cache[i] = (uint8_t *)calloc(
                params->max_seq_len, ctx->cpu_cache.k_bands.total_bytes);
            ctx->cpu_cache.v_cache[i] = (uint8_t *)calloc(
                params->max_seq_len, ctx->cpu_cache.v_bands.total_bytes);
        }
        break;
    }
#ifdef SP_HAVE_ADRENO
    case SP_BACKEND_ADRENO: {
        if (sp_adreno_cache_init(&ctx->adreno_cache, cfg,
                                  params->max_seq_len) != 0) {
            free(ctx);
            return NULL;
        }
        break;
    }
#endif
    // case SP_BACKEND_CUDA: ...
    // case SP_BACKEND_VULKAN: ...
    }

    if (parse_env_bool("SHANNON_PRIME_VERBOSE", 0)) {
        sp_llama_print_config(ctx);
    }

    return ctx;
}

void sp_llama_free(sp_llama_ctx_t *ctx) {
    if (!ctx) return;

    switch (ctx->active_backend) {
    case SP_BACKEND_CPU:
    default: {
        int n_slots = ctx->config.n_layers * ctx->config.n_heads_kv;
        for (int i = 0; i < n_slots; i++) {
            free(ctx->cpu_cache.k_cache[i]);
            free(ctx->cpu_cache.v_cache[i]);
        }
        free(ctx->cpu_cache.k_cache);
        free(ctx->cpu_cache.v_cache);
        sp_shadow_cache_free(&ctx->cpu_cache);
        break;
    }
#ifdef SP_HAVE_ADRENO
    case SP_BACKEND_ADRENO:
        sp_adreno_cache_free(&ctx->adreno_cache);
        break;
#endif
    }

    free(ctx);
}

// ============================================================================
// Write path
// ============================================================================

void sp_llama_write_kv(sp_llama_ctx_t *ctx,
                       int layer, int head, int pos,
                       const float *k_vec, const float *v_vec) {
    sp_llama_write_k(ctx, layer, head, pos, k_vec);
    sp_llama_write_v(ctx, layer, head, pos, v_vec);
}

void sp_llama_write_k(sp_llama_ctx_t *ctx,
                      int layer, int head, int pos,
                      const float *k_vec) {
    switch (ctx->active_backend) {
    case SP_BACKEND_CPU:
    default:
        sp_shadow_write_k(&ctx->cpu_cache, layer, head, pos, k_vec);
        break;
#ifdef SP_HAVE_ADRENO
    case SP_BACKEND_ADRENO:
        sp_adreno_write_k(&ctx->adreno_cache, layer, head, pos, k_vec);
        break;
#endif
    }
    if (pos >= ctx->n_positions) ctx->n_positions = pos + 1;
}

void sp_llama_write_v(sp_llama_ctx_t *ctx,
                      int layer, int head, int pos,
                      const float *v_vec) {
    switch (ctx->active_backend) {
    case SP_BACKEND_CPU:
    default:
        sp_shadow_write_v(&ctx->cpu_cache, layer, head, pos, v_vec);
        break;
#ifdef SP_HAVE_ADRENO
    case SP_BACKEND_ADRENO:
        sp_adreno_write_v(&ctx->adreno_cache, layer, head, pos, v_vec);
        break;
#endif
    }
}

void sp_llama_write_k_batch(sp_llama_ctx_t *ctx,
                            int layer, int head,
                            int start_pos, int n_pos,
                            const float *k_vecs) {
    switch (ctx->active_backend) {
    case SP_BACKEND_CPU:
    default:
        sp_shadow_write_k_batch(&ctx->cpu_cache, layer, head, start_pos, n_pos, k_vecs);
        break;
#ifdef SP_HAVE_ADRENO
    case SP_BACKEND_ADRENO:
        sp_adreno_write_k_batch(&ctx->adreno_cache, layer, head, start_pos, n_pos, k_vecs);
        break;
#endif
    }
    int last_pos = start_pos + n_pos - 1;
    if (last_pos >= ctx->n_positions) ctx->n_positions = last_pos + 1;
}

void sp_llama_write_v_batch(sp_llama_ctx_t *ctx,
                            int layer, int head,
                            int start_pos, int n_pos,
                            const float *v_vecs) {
    switch (ctx->active_backend) {
    case SP_BACKEND_CPU:
    default:
        sp_shadow_write_v_batch(&ctx->cpu_cache, layer, head, start_pos, n_pos, v_vecs);
        break;
#ifdef SP_HAVE_ADRENO
    case SP_BACKEND_ADRENO:
        sp_adreno_write_v_batch(&ctx->adreno_cache, layer, head, start_pos, n_pos, v_vecs);
        break;
#endif
    }
}

// ============================================================================
// Read path
// ============================================================================

void sp_llama_read_k(const sp_llama_ctx_t *ctx,
                     int layer, int head, int pos,
                     float *k_out) {
    switch (ctx->active_backend) {
    case SP_BACKEND_CPU:
    default:
        sp_shadow_read_k(&ctx->cpu_cache, layer, head, pos, k_out);
        break;
#ifdef SP_HAVE_ADRENO
    case SP_BACKEND_ADRENO:
        sp_adreno_read_k(&ctx->adreno_cache, layer, head, pos, k_out);
        break;
#endif
    }
}

void sp_llama_read_v(const sp_llama_ctx_t *ctx,
                     int layer, int head, int pos,
                     float *v_out) {
    switch (ctx->active_backend) {
    case SP_BACKEND_CPU:
    default:
        sp_shadow_read_v(&ctx->cpu_cache, layer, head, pos, v_out);
        break;
#ifdef SP_HAVE_ADRENO
    case SP_BACKEND_ADRENO:
        sp_adreno_read_v(&ctx->adreno_cache, layer, head, pos, v_out);
        break;
#endif
    }
}

void sp_llama_read_k_batch(const sp_llama_ctx_t *ctx,
                           int layer, int head,
                           int start_pos, int n_pos,
                           float *k_out) {
    switch (ctx->active_backend) {
    case SP_BACKEND_CPU:
    default:
        sp_shadow_read_k_batch(&ctx->cpu_cache, layer, head, start_pos, n_pos, k_out);
        break;
#ifdef SP_HAVE_ADRENO
    case SP_BACKEND_ADRENO:
        sp_adreno_read_k_batch(&ctx->adreno_cache, layer, head, start_pos, n_pos, k_out);
        break;
#endif
    }
}

void sp_llama_read_v_batch(const sp_llama_ctx_t *ctx,
                           int layer, int head,
                           int start_pos, int n_pos,
                           float *v_out) {
    switch (ctx->active_backend) {
    case SP_BACKEND_CPU:
    default:
        sp_shadow_read_v_batch(&ctx->cpu_cache, layer, head, start_pos, n_pos, v_out);
        break;
#ifdef SP_HAVE_ADRENO
    case SP_BACKEND_ADRENO:
        sp_adreno_read_v_batch(&ctx->adreno_cache, layer, head, start_pos, n_pos, v_out);
        break;
#endif
    }
}

// ============================================================================
// Cache management
// ============================================================================

void sp_llama_clear_range(sp_llama_ctx_t *ctx,
                          int start_pos, int end_pos) {
    if (ctx->active_backend != SP_BACKEND_CPU) {
        // TODO: adreno/cuda/vulkan clear_range. The eval-callback hook doesn't
        // need this during inference (round-trip is in-place), so leave as
        // no-op for non-CPU backends until a caller actually needs it.
        return;
    }

    // Zero out the compressed cache in the given range
    int n_slots = ctx->config.n_layers * ctx->config.n_heads_kv;

    for (int s = 0; s < n_slots; s++) {
        size_t k_off = (size_t)start_pos * ctx->cpu_cache.k_bands.total_bytes;
        size_t k_len = (size_t)(end_pos - start_pos) * ctx->cpu_cache.k_bands.total_bytes;
        memset(ctx->cpu_cache.k_cache[s] + k_off, 0, k_len);

        size_t v_off = (size_t)start_pos * ctx->cpu_cache.v_bands.total_bytes;
        size_t v_len = (size_t)(end_pos - start_pos) * ctx->cpu_cache.v_bands.total_bytes;
        memset(ctx->cpu_cache.v_cache[s] + v_off, 0, v_len);
    }
}

sp_llama_memory_t sp_llama_memory(const sp_llama_ctx_t *ctx) {
    sp_llama_memory_t mem;
    int n_slots = ctx->config.n_layers * ctx->config.n_heads_kv;
    int n = ctx->n_positions;
    int hd = ctx->config.head_dim;

    // Both backends expose .k_bands / .v_bands with the same total_bytes
    // derived from config. Read whichever is active.
    int k_total_bytes = 0;
    int v_total_bytes = 0;
#ifdef SP_HAVE_ADRENO
    if (ctx->active_backend == SP_BACKEND_ADRENO) {
        k_total_bytes = ctx->adreno_cache.k_bands.total_bytes;
        v_total_bytes = ctx->adreno_cache.v_bands.total_bytes;
    } else
#endif
    {
        k_total_bytes = ctx->cpu_cache.k_bands.total_bytes;
        v_total_bytes = ctx->cpu_cache.v_bands.total_bytes;
    }

    mem.compressed_bytes = (size_t)n_slots * n * (k_total_bytes + v_total_bytes);
    mem.baseline_bytes = (size_t)n_slots * n * hd * 2 * 2; // K+V × fp16
    mem.compression_ratio = (mem.compressed_bytes > 0)
        ? (float)mem.baseline_bytes / (float)mem.compressed_bytes
        : 0.0f;
    mem.n_positions = n;
    return mem;
}

// ============================================================================
// Diagnostics
// ============================================================================

float sp_llama_validate_k(sp_llama_ctx_t *ctx,
                          const float *k_vec, int head_dim) {
    float *recon = (float *)malloc(head_dim * sizeof(float));
    sp_llama_write_k(ctx, 0, 0, 0, k_vec);
    sp_llama_read_k(ctx, 0, 0, 0, recon);
    float corr = sp_correlation_f32(k_vec, recon, head_dim);
    free(recon);
    return corr;
}

void sp_llama_print_config(const sp_llama_ctx_t *ctx) {
    fprintf(stderr, "[Shannon-Prime] llama.cpp integration\n");
    fprintf(stderr, "  Backend:  %s\n",
            ctx->active_backend == SP_BACKEND_CPU    ? "CPU" :
            ctx->active_backend == SP_BACKEND_CUDA   ? "CUDA" :
            ctx->active_backend == SP_BACKEND_VULKAN  ? "Vulkan" :
            ctx->active_backend == SP_BACKEND_ADRENO  ? "Adreno" : "unknown");
    sp_config_print(&ctx->config);
}
