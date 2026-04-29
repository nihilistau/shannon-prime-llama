// Shannon-Prime VHT2: Exact Spectral KV Cache Compression
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// See LICENSE in the project root for full terms.

#include "shannon_prime_llama.h"
#include "shannon_prime_modelpack.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// Build-time backend gates. Define SP_HAVE_ADRENO when linking against
// backends/adreno/shannon_prime_adreno.c, SP_HAVE_HEXAGON when linking
// the FastRPC engine implementation at
// backends/hexagon/shannon_prime_hexagon.c (which itself requires
// SP_HEXAGON_FASTRPC at its compile site, since it pulls in rpcmem.h /
// remote.h / sp_hex.h from the qaic-generated FastRPC stubs).
#ifdef SP_HAVE_ADRENO
  #include "../backends/adreno/shannon_prime_adreno.h"
#endif
#ifdef SP_HAVE_HEXAGON
  #include "../backends/hexagon/shannon_prime_hexagon.h"
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
#ifdef SP_HAVE_HEXAGON
    // Snapdragon cDSP backend (Hexagon V69+ via FastRPC + HVX). The
    // engine-side ctx (sp_hexagon_ctx_t) holds the FastRPC session and
    // rpcmem-backed scratch; per-position packed-bytes storage is
    // shared with cpu_cache for now, with the DSP doing the
    // compress/decompress work via the engine API. A dedicated
    // sp_hexagon_cache_t with rpcmem-backed packed-byte storage is the
    // next layer once write/read decoupling lands.
    struct sp_hexagon_ctx_s *hexagon_ctx;
#endif
    // sp_cuda_cache_t    cuda_cache;  // CUDA backend (when linked)
    // sp_vulkan_cache_t *vulkan_cache; // Vulkan backend

    int active_backend;  // Which backend is in use
    int n_positions;     // Current max written position

    // PrimePE frequency factors (owned by context, freed on sp_llama_free)
    float *freq_factors;   // length = head_dim/2, or NULL if disabled
    int    n_freqs;        // head_dim / 2
};

// ============================================================================
// Environment variable parsing
// ============================================================================

static int parse_env_bool(const char *name, int default_val) {
    const char *v = getenv(name);
    if (!v) return default_val;
    return (v[0] == '1' || v[0] == 'y' || v[0] == 'Y');
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

// ── Role-aware env lookup ───────────────────────────────────────────
//
// When a context is initialised with role SP_LLAMA_ROLE_DRAFT, each
// SHANNON_PRIME_<base> lookup first tries SHANNON_PRIME_DRAFT_<base>;
// only if that's unset does it fall back to the global name. ROLE_DEFAULT
// and ROLE_TARGET both bypass the prefixed lookup, preserving legacy
// behaviour.
//
// These wrappers compute the right env-var name for the role and then
// delegate to the existing parse_env_* helpers — keeps the parsing
// logic in one place.

// Resolves the env-var name for a base + role combination. When role==
// SP_LLAMA_ROLE_DRAFT and SHANNON_PRIME_DRAFT_<base> is set, returns
// the DRAFT-prefixed name. Otherwise returns the global name. The
// returned pointer aliases `out` (caller-provided storage).
static const char *resolve_role_name(const char *base, sp_llama_role_t role,
                                     char *out, size_t out_sz) {
    if (role == SP_LLAMA_ROLE_DRAFT) {
        snprintf(out, out_sz, "SHANNON_PRIME_DRAFT_%s", base);
        const char *v = getenv(out);
        if (v && *v) return out;
    }
    snprintf(out, out_sz, "SHANNON_PRIME_%s", base);
    return out;
}

static const char *role_getenv(const char *base, sp_llama_role_t role) {
    char name[160];
    return getenv(resolve_role_name(base, role, name, sizeof(name)));
}

static int parse_role_bool(const char *base, sp_llama_role_t role, int default_val) {
    char name[160];
    return parse_env_bool(resolve_role_name(base, role, name, sizeof(name)),
                          default_val);
}

static int parse_role_bits_autocount(const char *base, sp_llama_role_t role,
                                     int *bits, int max_n) {
    char name[160];
    return parse_env_bits_autocount(
        resolve_role_name(base, role, name, sizeof(name)), bits, max_n);
}

// Parse a comma-separated list of band indices (e.g. "3" or "2,3") into
// a uint32 mask suitable for sp_config_t::{k,v}_ternary_mask. Indices >= 32
// are silently dropped (SP_MAX_BANDS is bounded well below 32). Returns 0
// when the role-resolved env var is unset.
static uint32_t parse_role_ternary_mask(const char *base, sp_llama_role_t role) {
    const char *v = role_getenv(base, role);
    if (!v || !*v) return 0u;
    uint32_t mask = 0u;
    const char *p = v;
    while (*p) {
        // Skip whitespace + leading separators.
        while (*p == ',' || *p == ' ' || *p == '\t') p++;
        if (!*p) break;
        if (*p < '0' || *p > '9') {
            // Bad token — skip until next separator and continue.
            while (*p && *p != ',') p++;
            continue;
        }
        int idx = atoi(p);
        if (idx >= 0 && idx < 32) mask |= (1u << idx);
        while (*p && *p != ',') p++;
    }
    return mask;
}

// Apply a draft preset's bit allocations as if the user had set
// SHANNON_PRIME_DRAFT_K_BITS / V_BITS. Real env vars still win — this
// only fills in defaults. Returns 1 if a known preset was applied,
// 0 otherwise (unknown preset string is ignored with a warning).
//
// Presets (chosen for speculative-decoding draft contexts where
// occasional acceptance loss is recoverable):
//   "aggressive" — K=2,1 V=1   (~10× compression, expect 5-15% accept dip)
//   "ternary"    — K=2,2 V=2   (~7× compression, expect 2-8% accept dip)
//   "ship"       — defer to ship defaults (no-op, useful as explicit "off")
static int apply_draft_preset(int *k_bits, int *k_n_bands,
                              int *v_bits, int *v_n_bands) {
    const char *preset = getenv("SHANNON_PRIME_DRAFT_PRESET");
    if (!preset || !*preset) return 0;

    if (strcmp(preset, "aggressive") == 0) {
        k_bits[0] = 2; k_bits[1] = 1; *k_n_bands = 2;
        v_bits[0] = 1;                *v_n_bands = 1;
        return 1;
    }
    if (strcmp(preset, "ternary") == 0) {
        k_bits[0] = 2; k_bits[1] = 2; *k_n_bands = 2;
        v_bits[0] = 2;                *v_n_bands = 1;
        return 1;
    }
    if (strcmp(preset, "ship") == 0) {
        // Explicit no-op — caller's existing defaults stand.
        return 1;
    }
    fprintf(stderr, "[Shannon-Prime] unknown SHANNON_PRIME_DRAFT_PRESET '%s' "
                    "(known: aggressive, ternary, ship); ignoring\n", preset);
    return 0;
}

// ============================================================================
// Lifecycle
// ============================================================================

sp_llama_ctx_t *sp_llama_init(const sp_llama_params_t *params) {
    return sp_llama_init_with_role(params, SP_LLAMA_ROLE_DEFAULT);
}

sp_llama_ctx_t *sp_llama_init_with_role(const sp_llama_params_t *params,
                                        sp_llama_role_t role) {
    // SHANNON_PRIME_ENABLED is a process-wide kill switch. We honour the
    // role-aware path here too so a caller can disable SP for the draft
    // only via SHANNON_PRIME_DRAFT_ENABLED=0 — useful when the draft is
    // small enough that the SP overhead exceeds the compression payoff.
    if (!parse_role_bool("ENABLED", role, 0)) {
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
    cfg.k_n_bands = k_n_bands_default;
    memcpy(cfg.k_band_bits, k_defaults, k_n_bands_default * sizeof(int));
    cfg.v_band_bits[0] = v_bits_default;
    cfg.v_n_bands      = 1;

    // Draft-preset shortcut: when role==DRAFT and SHANNON_PRIME_DRAFT_PRESET
    // is set to a known value, pre-fill the band tables. Explicit env vars
    // below still override.
    if (role == SP_LLAMA_ROLE_DRAFT) {
        apply_draft_preset(cfg.k_band_bits, &cfg.k_n_bands,
                           cfg.v_band_bits, &cfg.v_n_bands);
    }

    // Environment overrides. If SHANNON_PRIME_K_BITS or _V_BITS are set,
    // the number of comma-separated values determines the band count
    // (capped by SP_MAX_BANDS). E.g. `SHANNON_PRIME_K_BITS=3,3,3,3,3`
    // → 5 bands at 3 bits each. When the env var is unset we keep the
    // head-dim-adaptive defaults (or draft-preset values if applied).
    int k_override_n = parse_role_bits_autocount(
        "K_BITS", role, cfg.k_band_bits, SP_MAX_BANDS);
    if (k_override_n > 0) {
        cfg.k_n_bands = k_override_n;
    }

    int v_override_n = parse_role_bits_autocount(
        "V_BITS", role, cfg.v_band_bits, SP_MAX_BANDS);
    if (v_override_n > 0) {
        cfg.v_n_bands = v_override_n;
    }

    cfg.use_mobius_mask = parse_role_bool("MOBIUS", role, 1);

    // Ternary noise-tail bands. Mask is applied at sp_band_config_init_ext
    // inside sp_shadow_cache_init (sp_config_t carries the mask through).
    // Empty / unset env var = 0u = no ternary bands (existing behaviour).
    cfg.k_ternary_mask = parse_role_ternary_mask("K_TERNARY_BANDS", role);
    cfg.v_ternary_mask = parse_role_ternary_mask("V_TERNARY_BANDS", role);

    // FP8 (E4M3FN) banded quantisation. Advisory: the CPU bridge has no
    // fp8 path, so requesting fp8 here just logs a warning and falls
    // through to the int path. The engine's CUDA backend honours
    // sp_config_t::use_fp8 when SP_ENGINE_FP8 was set at compile time
    // (see shannon-prime-engine/src/kv_cache.cpp). The CPU/Adreno
    // backend warning is emitted below once the active backend has
    // been resolved (after env-override on the BACKEND var).
    cfg.use_fp8 = (parse_role_bool("FP8", role, 0) != 0);

    // Allow the caller to request a specific backend via env var.
    // Valid values: "cpu" (default), "adreno". Unknown values → caller's
    // params->backend is used unchanged.
    sp_llama_params_t p = *params;
    const char *b = role_getenv("BACKEND", role);
    if (b) {
        if (strcmp(b, "cpu") == 0)            p.backend = SP_BACKEND_CPU;
        else if (strcmp(b, "hexagon") == 0)   p.backend = SP_BACKEND_HEXAGON;
#ifdef SP_HAVE_ADRENO
        else if (strcmp(b, "adreno") == 0)    p.backend = SP_BACKEND_ADRENO;
#endif
    }

    // Now that the active backend is resolved, warn if fp8 was requested
    // but the active backend has no fp8 path. CPU + Adreno fall through
    // to int regardless; only the engine's CUDA backend (linked via a
    // separate build path, NOT through the CPU bridge) honours use_fp8.
    if (cfg.use_fp8 && (p.backend == SP_BACKEND_CPU
#ifdef SP_HAVE_ADRENO
                       || p.backend == SP_BACKEND_ADRENO
#endif
                       )) {
        const char *role_tag =
            (role == SP_LLAMA_ROLE_DRAFT)  ? " [draft]"  :
            (role == SP_LLAMA_ROLE_TARGET) ? " [target]" : "";
        fprintf(stderr, "[Shannon-Prime]%s SHANNON_PRIME_FP8=1 requested but the "
                        "active bridge backend has no fp8 path; falling back to "
                        "int. Use shannon-prime-engine with SP_ENGINE_FP8=ON for "
                        "the actual fp8 dispatch on CUDA.\n", role_tag);
    }

    sp_llama_ctx_t *ctx = sp_llama_init_config(&p, &cfg);
    if (!ctx) return ctx;

    // ── PrimePE auto-injection ──────────────────────────────────────────
    // Compute lattice-aligned freq_factors at init. The llama.cpp patch
    // retrieves these via sp_llama_get_freq_factors() and writes them
    // into model.layers[*].rope_freqs. Zero per-token cost.
    int pe_enabled = parse_role_bool("PE", role, 1);  // on by default
    if (pe_enabled && params->head_dim > 0) {
        float pe_alpha = 0.17f;
        const char *alpha_str = role_getenv("PE_ALPHA", role);
        if (alpha_str) pe_alpha = (float)atof(alpha_str);

        // freq_base: default 10000.0, but models vary (500000 for Qwen, etc.)
        // The caller should set this from hparams.rope_freq_base. If not
        // available, 10000.0 is safe — the factors are multipliers on
        // whatever base the model uses, and the lattice normalization
        // adapts to the geometric range.
        float freq_base = 10000.0f;
        const char *fb_str = role_getenv("FREQ_BASE", role);
        if (fb_str) freq_base = (float)atof(fb_str);

        int n_freqs = params->head_dim / 2;
        ctx->freq_factors = sp_prime_pe_freq_factors(n_freqs, freq_base, pe_alpha);
        ctx->n_freqs = (ctx->freq_factors != NULL) ? n_freqs : 0;

        if (ctx->freq_factors) {
            int verbose = parse_role_bool("VERBOSE", role, 0);
            if (verbose) {
                const char *role_tag =
                    (role == SP_LLAMA_ROLE_DRAFT)  ? " [draft]"  :
                    (role == SP_LLAMA_ROLE_TARGET) ? " [target]" : "";
                fprintf(stderr, "[Shannon-Prime]%s PrimePE enabled: α=%.2f, "
                        "freq_base=%.0f, %d freq pairs\n",
                        role_tag, pe_alpha, freq_base, n_freqs);
                fprintf(stderr, "[Shannon-Prime]%s PrimePE factor range: "
                        "[%.4f, %.4f]\n",
                        role_tag,
                        ctx->freq_factors[0],
                        ctx->freq_factors[n_freqs - 1]);
            }
        }
    }

    // Speculative-decoding hint: when verbose AND we know the arch AND the
    // resolved preset has a suggested_draft, log a one-line tip. We only
    // emit this for ROLE_TARGET / ROLE_DEFAULT — emitting it for the draft
    // would be circular noise. The hint is purely advisory; it doesn't
    // load anything or alter behaviour.
    if (params->arch_name && role != SP_LLAMA_ROLE_DRAFT
        && parse_role_bool("VERBOSE", role, 0))
    {
        const sp_model_preset_t *preset = sp_model_preset_resolve(
            params->arch_name, params->head_dim, params->n_layers, params->n_heads_kv);
        if (preset) {
            float accept = 0.0f;
            const char *draft = sp_model_preset_suggested_draft(preset, &accept);
            if (draft && *draft) {
                const char *role_tag =
                    (role == SP_LLAMA_ROLE_TARGET) ? " [target]" : "";
                fprintf(stderr, "[Shannon-Prime]%s preset '%s' matched. "
                        "Suggested draft for speculative decoding: %s "
                        "(expected acceptance ~%.0f%%). Pass it with "
                        "llama-cli's -md flag, or ignore for single-model use.\n",
                        role_tag, preset->name, draft, accept * 100.0f);
            }
        }
    }

    return ctx;
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
#ifdef SP_HAVE_HEXAGON
    case SP_BACKEND_HEXAGON: {
        // Engine API: sp_hexagon_init opens a FastRPC session against
        // libsp_hex_skel.so on the cDSP, enables unsigned PD (required
        // for unsigned developer builds), pre-allocates rpcmem-backed
        // ping-pong scratch (RPCMEM_TRY_MAP_STATIC pre-maps to DSP),
        // acquires VTCM. Returns NULL if the DSP is unreachable —
        // we fall back to CPU in that case so the model still runs.
        ctx->hexagon_ctx = sp_hexagon_init(cfg);
        if (!ctx->hexagon_ctx) {
            fprintf(stderr,
                "[Shannon-Prime] Hexagon backend requested but DSP "
                "unreachable; falling back to CPU\n");
            // Init the CPU cache so reads/writes still work via
            // shadow_cache. active_backend is updated below.
            if (sp_shadow_cache_init(&ctx->cpu_cache, cfg) != 0) {
                free(ctx);
                return NULL;
            }
            int n_slots = cfg->n_layers * cfg->n_heads_kv;
            ctx->cpu_cache.k_cache = (uint8_t **)calloc(n_slots, sizeof(uint8_t *));
            ctx->cpu_cache.v_cache = (uint8_t **)calloc(n_slots, sizeof(uint8_t *));
            for (int i = 0; i < n_slots; i++) {
                ctx->cpu_cache.k_cache[i] = (uint8_t *)calloc(
                    params->max_seq_len, ctx->cpu_cache.k_bands.total_bytes);
                ctx->cpu_cache.v_cache[i] = (uint8_t *)calloc(
                    params->max_seq_len, ctx->cpu_cache.v_bands.total_bytes);
            }
            ctx->active_backend = SP_BACKEND_CPU;
            break;
        }
        // Hexagon backend uses CPU's packed-byte storage; the DSP just
        // does compress/decompress work. Init the CPU cache too — it
        // owns the per-position packed-bytes for now. A dedicated
        // sp_hexagon_cache_t with rpcmem-backed packed storage is the
        // next layer (TODO: add band_quantize-only IDL method + cache
        // wrapper functions in the engine API).
        if (sp_shadow_cache_init(&ctx->cpu_cache, cfg) != 0) {
            sp_hexagon_free(ctx->hexagon_ctx);
            free(ctx);
            return NULL;
        }
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
#ifdef SP_HAVE_HEXAGON
    case SP_BACKEND_HEXAGON: {
        // Tear down the cDSP session first (closes FastRPC handle,
        // releases VTCM, frees rpcmem ping-pong scratch). Then free
        // the CPU-side packed-byte storage we share with the CPU
        // backend.
        if (ctx->hexagon_ctx) sp_hexagon_free(ctx->hexagon_ctx);
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
#endif
    }

    free(ctx->freq_factors);
    free(ctx);
}

// ============================================================================
// PrimePE getters
// ============================================================================

const float *sp_llama_get_freq_factors(const sp_llama_ctx_t *ctx) {
    if (!ctx) return NULL;
    return ctx->freq_factors;
}

int sp_llama_get_n_freqs(const sp_llama_ctx_t *ctx) {
    if (!ctx) return 0;
    return ctx->n_freqs;
}

// ============================================================================
// Write path
// ============================================================================
//
// SP_BACKEND_HEXAGON note: there's no explicit case for HEXAGON in the
// read/write switch sites below. That's deliberate — the `default:`
// label at each switch routes to the CPU path (sp_shadow_*), and the
// Hexagon backend currently shares the CPU's packed-byte storage. The
// DSP-side compress/decompress work isn't on the per-vector hot path
// yet; it'll be wired in when the engine API grows
// sp_hexagon_cache_write_k / read_k wrappers backed by rpcmem-resident
// packed storage. Until then, the HEXAGON case is "init the engine
// session for future use, but write/read still go through CPU."
//
// This means installing SP_BACKEND_HEXAGON in production today gives
// you a working session with the FastRPC handle warmed up + VTCM
// acquired + ping-pong scratch ready, but no actual DSP offload on
// each KV write/read. That's the next layer of integration work.

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

void sp_llama_read_k_partial(const sp_llama_ctx_t *ctx,
                             int layer, int head, int pos,
                             float *k_out, int max_bands) {
    switch (ctx->active_backend) {
    case SP_BACKEND_CPU:
    default:
        sp_shadow_read_k_partial(&ctx->cpu_cache, layer, head, pos, k_out, max_bands);
        break;
#ifdef SP_HAVE_ADRENO
    case SP_BACKEND_ADRENO: {
        // Adreno doesn't have a partial path yet; fall back to full read.
        // One-shot warning so the user knows they're not getting the IO win
        // until that backend gets its partial dispatch.
        static int warned = 0;
        if (!warned) {
            fprintf(stderr, "[Shannon-Prime] adreno backend has no partial read; "
                            "falling back to full read (max_bands=%d ignored)\n",
                            max_bands);
            warned = 1;
        }
        sp_adreno_read_k(&ctx->adreno_cache, layer, head, pos, k_out);
        break;
    }
#endif
    }
}

void sp_llama_read_v_partial(const sp_llama_ctx_t *ctx,
                             int layer, int head, int pos,
                             float *v_out, int max_bands) {
    switch (ctx->active_backend) {
    case SP_BACKEND_CPU:
    default:
        sp_shadow_read_v_partial(&ctx->cpu_cache, layer, head, pos, v_out, max_bands);
        break;
#ifdef SP_HAVE_ADRENO
    case SP_BACKEND_ADRENO: {
        static int warned = 0;
        if (!warned) {
            fprintf(stderr, "[Shannon-Prime] adreno backend has no partial read; "
                            "falling back to full read (max_bands=%d ignored)\n",
                            max_bands);
            warned = 1;
        }
        sp_adreno_read_v(&ctx->adreno_cache, layer, head, pos, v_out);
        break;
    }
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
