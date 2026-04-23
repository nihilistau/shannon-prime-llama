// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

//
// llama.cpp integration for the sqfree + spinor aggressive compression path.
//
// Additive to tools/shannon_prime_llama.c — does NOT modify the VHT2 ship path.
// Link this alongside the existing llama integration when building with sqfree.
//
// New environment variables (all opt-in, VHT2 ship config remains default):
//
//   SHANNON_PRIME_SQFREE=1       Enable sqfree prime-Hartley basis
//   SHANNON_PRIME_SPINOR=1       Enable spinor sheet bit (requires SQFREE)
//   SHANNON_PRIME_RESIDUAL_BITS=3  Residual quantization (1-4, default 3)
//   SHANNON_PRIME_SK_FRAC=0.75   Skeleton fraction of pad_dim
//
// When SQFREE is enabled, the shadow cache uses:
//   - Vilenkin-Hartley transform on sqfree-padded head_dim
//   - Knight-ranked skeleton with Möbius CSR predictor
//   - N-bit quantized residual + optional spinor sheet bit
//
// Validated result (Qwen3-8B Q8 hd=128):
//   SPINOR + 3/3/3/3/3:  PPL 7.32 @ 3.3× (matches MOBIUS default 7.31 @ 2.6×)

#include "../core/shannon_prime.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// Configuration from environment
// ============================================================================

typedef struct {
    bool     sqfree_enabled;     // SHANNON_PRIME_SQFREE
    bool     spinor_enabled;     // SHANNON_PRIME_SPINOR
    int      residual_bits;      // SHANNON_PRIME_RESIDUAL_BITS (1-4)
    float    sk_frac;            // SHANNON_PRIME_SK_FRAC (0.0-1.0)
    int      k_band_bits[SP_MAX_BANDS]; // SHANNON_PRIME_K_BITS (reused)
    int      n_k_bands;
} sp_llama_sqfree_config_t;

static sp_llama_sqfree_config_t sp_llama_sqfree_config_from_env(void) {
    sp_llama_sqfree_config_t cfg = {0};

    const char *sqfree = getenv("SHANNON_PRIME_SQFREE");
    cfg.sqfree_enabled = (sqfree && strcmp(sqfree, "1") == 0);

    const char *spinor = getenv("SHANNON_PRIME_SPINOR");
    cfg.spinor_enabled = (spinor && strcmp(spinor, "1") == 0);

    // Spinor requires sqfree basis
    if (cfg.spinor_enabled && !cfg.sqfree_enabled) {
        fprintf(stderr, "[Shannon-Prime] SPINOR=1 requires SQFREE=1. Enabling SQFREE.\n");
        cfg.sqfree_enabled = true;
    }

    const char *rb = getenv("SHANNON_PRIME_RESIDUAL_BITS");
    cfg.residual_bits = rb ? atoi(rb) : 3;
    if (cfg.residual_bits < 1) cfg.residual_bits = 1;
    if (cfg.residual_bits > 4) cfg.residual_bits = 4;

    const char *sf = getenv("SHANNON_PRIME_SK_FRAC");
    cfg.sk_frac = sf ? (float)atof(sf) : 0.75f;
    if (cfg.sk_frac < 0.1f) cfg.sk_frac = 0.1f;
    if (cfg.sk_frac > 1.0f) cfg.sk_frac = 1.0f;

    // Parse K band bits (default: 5,4,4,4,5 for sqfree, 5,5,4,3 for ship VHT2)
    const char *kb = getenv("SHANNON_PRIME_K_BITS");
    if (kb) {
        cfg.n_k_bands = 0;
        const char *p = kb;
        while (*p && cfg.n_k_bands < SP_MAX_BANDS) {
            cfg.k_band_bits[cfg.n_k_bands++] = atoi(p);
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
        }
    } else if (cfg.sqfree_enabled) {
        // Default for sqfree: 5-band torus-aligned
        cfg.n_k_bands = 5;
        cfg.k_band_bits[0] = 5;
        cfg.k_band_bits[1] = 4;
        cfg.k_band_bits[2] = 4;
        cfg.k_band_bits[3] = 4;
        cfg.k_band_bits[4] = 5;
    } else {
        // Default for ship VHT2: 4-band
        cfg.n_k_bands = 4;
        cfg.k_band_bits[0] = 5;
        cfg.k_band_bits[1] = 5;
        cfg.k_band_bits[2] = 4;
        cfg.k_band_bits[3] = 3;
    }

    return cfg;
}

// ============================================================================
// Sqfree context — extends the base llama context
// ============================================================================

typedef struct {
    sp_sqfree_cache_t  cache;
    sp_llama_sqfree_config_t env_cfg;
    bool               initialized;
    // SHANNON_PRIME_DUMP_K=<path> appends raw K vectors (head_dim × fp32) to
    // <path> so tools/sp_auto_bands.py can compute per-band VHT2 energy and
    // emit a fitted K_BITS allocation. Write-only, append mode. When unset,
    // dump_fp is NULL and the dump path is a no-op.
    FILE              *dump_fp;
    int                dump_head_dim;
    long long          dump_count;   // number of K vectors written so far
    long long          dump_limit;   // stop after N vectors (0 = unlimited)
} sp_llama_sqfree_ctx_t;

// Initialize sqfree context. Call after model params are known.
// Returns NULL if sqfree is not enabled (caller should use ship VHT2 path).
sp_llama_sqfree_ctx_t *sp_llama_sqfree_init(int head_dim, int n_layers,
                                             int n_heads_kv, int max_seq_len) {
    sp_llama_sqfree_config_t env = sp_llama_sqfree_config_from_env();
    if (!env.sqfree_enabled) return NULL;

    sp_llama_sqfree_ctx_t *ctx = (sp_llama_sqfree_ctx_t *)calloc(1, sizeof(*ctx));
    ctx->env_cfg = env;

    sp_config_t cfg;
    sp_config_init(&cfg, head_dim, n_layers, n_heads_kv);

    // Override band bits from env
    cfg.k_n_bands = env.n_k_bands;
    for (int i = 0; i < env.n_k_bands; i++) {
        cfg.k_band_bits[i] = env.k_band_bits[i];
    }
    // V gets same allocation for sqfree (cross-attn style)
    cfg.v_n_bands = env.n_k_bands;
    for (int i = 0; i < env.n_k_bands; i++) {
        cfg.v_band_bits[i] = env.k_band_bits[i];
    }

    int rc = sp_sqfree_cache_init(&ctx->cache, &cfg, max_seq_len,
                                   env.residual_bits, env.spinor_enabled);
    if (rc != 0) {
        free(ctx);
        return NULL;
    }

    ctx->initialized = true;

    // Optional: open K-dump file for offline auto-bands analysis.
    const char *dump_path = getenv("SHANNON_PRIME_DUMP_K");
    if (dump_path && *dump_path) {
        ctx->dump_fp = fopen(dump_path, "wb");
        if (ctx->dump_fp) {
            ctx->dump_head_dim = head_dim;
            const char *limit_s = getenv("SHANNON_PRIME_DUMP_K_LIMIT");
            ctx->dump_limit = limit_s ? atoll(limit_s) : 8192;
            fprintf(stderr, "[Shannon-Prime] K dump → %s (limit=%lld vectors)\n",
                    dump_path, ctx->dump_limit);
        } else {
            fprintf(stderr, "[Shannon-Prime] warning: failed to open K dump '%s'\n",
                    dump_path);
        }
    }

    if (getenv("SHANNON_PRIME_VERBOSE")) {
        int pd = ctx->cache.pad_dim;
        fprintf(stderr, "[Shannon-Prime SQFREE] head_dim=%d → pad_dim=%d\n",
                head_dim, pd);
        fprintf(stderr, "[Shannon-Prime SQFREE] skeleton=%d residual=%d\n",
                ctx->cache.mask.sk_k, ctx->cache.mask.n_res);
        fprintf(stderr, "[Shannon-Prime SQFREE] residual_bits=%d spinor=%s\n",
                env.residual_bits, env.spinor_enabled ? "on" : "off");
        fprintf(stderr, "[Shannon-Prime SQFREE] K bits=");
        for (int i = 0; i < env.n_k_bands; i++) {
            fprintf(stderr, "%d%s", env.k_band_bits[i],
                    i < env.n_k_bands - 1 ? "," : "\n");
        }

        // Print scaling-law prediction if model size is known
        float params_b = (float)(head_dim * n_layers * n_heads_kv * 4) / 1e9f;
        float floor = sp_min_k_corr_for_budget(params_b, 8, 0.03f);
        fprintf(stderr, "[Shannon-Prime SQFREE] scaling law: safe K_corr floor=%.4f "
                "(est. %.1fB params @ Q8)\n", floor, params_b);
    }

    return ctx;
}

void sp_llama_sqfree_free(sp_llama_sqfree_ctx_t *ctx) {
    if (!ctx) return;
    if (ctx->dump_fp) {
        fprintf(stderr, "[Shannon-Prime] K dump closed (%lld vectors written)\n",
                ctx->dump_count);
        fclose(ctx->dump_fp);
    }
    if (ctx->initialized) {
        sp_sqfree_cache_free(&ctx->cache);
    }
    free(ctx);
}

// Write/read wrappers — same interface as the ship VHT2 path
void sp_llama_sqfree_write_kv(sp_llama_sqfree_ctx_t *ctx,
                               int layer, int head, int pos,
                               const float *k_vec, const float *v_vec) {
    if (ctx->dump_fp &&
        (ctx->dump_limit == 0 || ctx->dump_count < ctx->dump_limit)) {
        fwrite(k_vec, sizeof(float), ctx->dump_head_dim, ctx->dump_fp);
        ctx->dump_count++;
    }
    sp_sqfree_write_k(&ctx->cache, layer, head, pos, k_vec);
    sp_sqfree_write_v(&ctx->cache, layer, head, pos, v_vec);
}

void sp_llama_sqfree_read_k(const sp_llama_sqfree_ctx_t *ctx,
                             int layer, int head, int pos,
                             float *k_out) {
    sp_sqfree_read_k(&ctx->cache, layer, head, pos, k_out);
}

void sp_llama_sqfree_read_v(const sp_llama_sqfree_ctx_t *ctx,
                             int layer, int head, int pos,
                             float *v_out) {
    sp_sqfree_read_v(&ctx->cache, layer, head, pos, v_out);
}

// Validate a K vector by compressing and reconstructing, returning correlation
float sp_llama_sqfree_validate_k(sp_llama_sqfree_ctx_t *ctx,
                                  const float *k_vec, int head_dim) {
    // Write to scratch position
    sp_sqfree_write_k(&ctx->cache, 0, 0, 0, k_vec);
    float k_out[SP_MAX_HEAD_DIM];
    sp_sqfree_read_k(&ctx->cache, 0, 0, 0, k_out);
    return sp_correlation_f32(k_vec, k_out, head_dim);
}