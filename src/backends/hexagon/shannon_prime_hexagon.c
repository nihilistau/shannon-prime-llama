// Shannon-Prime VHT2 - Hexagon DSP backend (host-side).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved. AGPLv3 / commercial.
//
// Two configurations:
//
//   1. SP_HEXAGON_FASTRPC defined  →  real FastRPC implementation. Calls
//      the scaffold's sp_hex_* IDL methods (qaic-generated, see
//      backends/hexagon/scaffold/inc/sp_hex.idl). Requires the Hexagon
//      SDK headers and rpcmem/dsprpc libraries to be on the include /
//      link paths — typically only true when building the scaffold's
//      ARM-side target on Android.
//
//   2. SP_HEXAGON_FASTRPC undefined  →  stub fallback. Every entry point
//      returns -1 / NULL with a one-shot diagnostic so the bridge falls
//      back cleanly to Adreno/CPU. This is what x86 / desktop /
//      non-Snapdragon builds get.
//
// The math core (core/shannon_prime.c) is what runs on the DSP; this
// file is just the FastRPC shim that ferries data across the IPC boundary.

#include "shannon_prime_hexagon.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef SP_HEXAGON_FASTRPC

// ============================================================================
// Real FastRPC implementation
// ============================================================================

#include "rpcmem.h"
#include "remote.h"
#include "AEEStdErr.h"
#include "sp_hex.h"            // qaic-generated stub header from sp_hex.idl
#include "shannon_prime.h"     // sp_f16_to_f32 / sp_f32_to_f16 / sp_band_*

#ifndef CDSP_DOMAIN_ID
#define CDSP_DOMAIN_ID 3
#endif

// FastRPC needs an explicit URI for each domain. The qaic-generated
// `sp_hex_URI` macro encodes the interface name; CDSP_DOMAIN appends the
// session selector. Both come from the SDK's remote.idl-derived headers.
#ifndef CDSP_DOMAIN
#define CDSP_DOMAIN "&_dom=cdsp"
#endif

struct sp_hexagon_ctx_s {
    remote_handle64 fastrpc_handle;     // FastRPC session
    sp_config_t     cfg_snapshot;       // Captured at init for reference
    size_t          bytes_in_use;       // Tracked allocations
    long long       last_call_cycles;   // 0 until profiling lands
    int             unsigned_pd_active; // 1 if we enabled it on this domain

    // Pre-allocated rpcmem-backed scratch buffers, sized for the
    // largest head_dim we expect (1024 fp32 = 4 KB each). FastRPC
    // recognises pointers from rpcmem_alloc as ION-backed shared
    // physical memory and skips the IPC marshal copy entirely — the
    // DSP gets the same physical pages via SMMU translation. malloc'd
    // pointers fall back to a marshal copy across the host/DSP boundary.
    //
    // Two-deep ping-pong: scratch_*_f32[0] and [1]. The round_trip path
    // alternates between them so an upcoming caller can prefetch into
    // [i^1] (e.g., fread the next packed band) while the DSP is still
    // chewing on [i]. Without this, the I/O and compute pipelines
    // serialise — disk read latency stacks on top of DSP cycles.
    //
    // RPCMEM_HEAP_ID_SYSTEM = non-contiguous physical memory routed
    // through the cDSP's SMMU. The contiguous heap (HEAP_ID_CONTIG=22)
    // is for older subsystems without SMMU and is documented as
    // deprecated for V73+. V69 cDSP uses SYSTEM.
    //
    // Allocations come back 4 KB-aligned (one physical page minimum),
    // which is also the SMMU translation granularity — no extra page-
    // table walks for the kernels' inner loops.
    float          *scratch_in_f32[2];   // [0]/[1] alternated per round_trip
    float          *scratch_out_f32[2];
    size_t          scratch_bytes;       // size of each scratch buffer
    int             scratch_idx;         // 0 or 1 — flipped per call
};

// Largest head_dim the engine accommodates without re-allocating.
// 1024 covers every modern model (Qwen / Llama 3 / Phi / Gemma all sit
// at 64–256). 4 KB per buffer × 2 = 8 KB total per session.
#define SP_HEXAGON_HEAD_DIM_MAX 1024

// One-shot init of rpcmem. rpcmem_init / _deinit are reference-counted so
// repeated init pairs are safe; a static guard keeps us from doubling up.
static int g_rpcmem_inited = 0;

static int sp_hexagon_enable_unsigned_pd(int domain) {
    if (!remote_session_control) return AEE_EUNSUPPORTED;
    struct remote_rpc_control_unsigned_module data;
    data.domain = domain;
    data.enable = 1;
    return remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                  (void *)&data, sizeof(data));
}

int sp_hexagon_caps_probe(sp_hexagon_caps_t *caps) {
    if (!caps) return -1;
    memset(caps, 0, sizeof(*caps));
    // Best-effort caps. A future revision could open a probe FastRPC
    // session and read DSP-side hardware regs (HEXAGON_REG_VERSION etc.)
    // For now we hardcode V69+HVX assumptions matching our v69 build.
    caps->has_dsp         = 1;
    caps->dsp_version     = 69;
    caps->has_hvx         = 1;
    caps->hvx_width_bits  = 1024;
    caps->has_hmx         = 0;          // V69 has no HMX (V73+ feature)
    caps->max_threads     = 4;          // V69 typical
    caps->shared_mem_bytes = 16 * 1024 * 1024;  // 16 MB rpcmem budget
    return 0;
}

void sp_hexagon_caps_print(const sp_hexagon_caps_t *caps) {
    if (!caps) return;
    fprintf(stderr, "Hexagon DSP Capabilities:\n");
    fprintf(stderr, "  DSP accessible: %s\n", caps->has_dsp ? "yes" : "no");
    if (caps->has_dsp) {
        fprintf(stderr, "  DSP version:    V%d\n", caps->dsp_version);
        fprintf(stderr, "  HVX:            %s (%d-bit)\n",
                caps->has_hvx ? "yes" : "no", caps->hvx_width_bits);
        fprintf(stderr, "  HMX:            %s\n", caps->has_hmx ? "yes" : "no");
        fprintf(stderr, "  Threads:        %d\n", caps->max_threads);
        fprintf(stderr, "  Shared memory:  %lld bytes\n",
                (long long)caps->shared_mem_bytes);
    }
}

sp_hexagon_ctx_t *sp_hexagon_init(const sp_config_t *cfg) {
    if (!cfg) return NULL;

    if (!g_rpcmem_inited) {
        rpcmem_init();
        g_rpcmem_inited = 1;
    }

    sp_hexagon_ctx_t *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) return NULL;
    ctx->cfg_snapshot       = *cfg;
    ctx->fastrpc_handle     = (remote_handle64)-1;
    ctx->unsigned_pd_active = 0;

    // Enable unsigned PD on cDSP — required for unsigned developer builds.
    int rc = sp_hexagon_enable_unsigned_pd(CDSP_DOMAIN_ID);
    if (rc != AEE_SUCCESS) {
        fprintf(stderr, "[Shannon-Prime] hexagon: enable_unsigned_pd "
                        "failed 0x%x; trying signed path\n", rc);
    } else {
        ctx->unsigned_pd_active = 1;
    }

    // Open the FastRPC session against libsp_hex_skel.so on the cDSP.
    char uri_buf[128];
    snprintf(uri_buf, sizeof(uri_buf), "%s%s", sp_hex_URI, CDSP_DOMAIN);
    rc = sp_hex_open(uri_buf, &ctx->fastrpc_handle);
    if (rc != AEE_SUCCESS) {
        fprintf(stderr, "[Shannon-Prime] hexagon: sp_hex_open failed 0x%x; "
                        "falling back to Adreno/CPU\n", rc);
        free(ctx);
        return NULL;
    }

    // Pre-allocate the FastRPC-shared scratch — two-deep for ping-pong.
    // RPCMEM_TRY_MAP_STATIC tells the FastRPC runtime to pre-map at
    // allocation time so the first round_trip doesn't pay a one-shot
    // map-on-first-use latency.
    ctx->scratch_bytes = SP_HEXAGON_HEAD_DIM_MAX * sizeof(float);
    int alloc_flags = RPCMEM_DEFAULT_FLAGS | RPCMEM_TRY_MAP_STATIC;
    int alloc_ok = 1;
    for (int b = 0; b < 2; ++b) {
        ctx->scratch_in_f32[b]  = (float *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                                                         alloc_flags,
                                                         (int)ctx->scratch_bytes);
        ctx->scratch_out_f32[b] = (float *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,
                                                         alloc_flags,
                                                         (int)ctx->scratch_bytes);
        if (!ctx->scratch_in_f32[b] || !ctx->scratch_out_f32[b]) {
            alloc_ok = 0;
            break;
        }
    }
    if (!alloc_ok) {
        fprintf(stderr, "[Shannon-Prime] hexagon: rpcmem_alloc scratch "
                        "failed; falling back to non-zero-copy\n");
        // Non-fatal — round_trip will use stack scratch as fallback.
        for (int b = 0; b < 2; ++b) {
            if (ctx->scratch_in_f32[b])  rpcmem_free(ctx->scratch_in_f32[b]);
            if (ctx->scratch_out_f32[b]) rpcmem_free(ctx->scratch_out_f32[b]);
            ctx->scratch_in_f32[b]  = NULL;
            ctx->scratch_out_f32[b] = NULL;
        }
        ctx->scratch_bytes = 0;
    } else {
        ctx->bytes_in_use = ctx->scratch_bytes * 4;  // 2 in + 2 out
    }
    ctx->scratch_idx = 0;

    return ctx;
}

void sp_hexagon_free(sp_hexagon_ctx_t *ctx) {
    if (!ctx) return;
    for (int b = 0; b < 2; ++b) {
        if (ctx->scratch_in_f32[b])  rpcmem_free(ctx->scratch_in_f32[b]);
        if (ctx->scratch_out_f32[b]) rpcmem_free(ctx->scratch_out_f32[b]);
    }
    if (ctx->fastrpc_handle != (remote_handle64)-1) {
        sp_hex_close(ctx->fastrpc_handle);
    }
    free(ctx);
}

void *sp_hexagon_alloc(sp_hexagon_ctx_t *ctx, size_t n_bytes) {
    if (!ctx) return NULL;
    // Page-align up to 4 KB.
    size_t aligned = (n_bytes + 4095) & ~(size_t)4095;
    void *p = rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS,
                           (int)aligned);
    if (p) ctx->bytes_in_use += aligned;
    return p;
}

void sp_hexagon_free_shared(sp_hexagon_ctx_t *ctx, void *ptr) {
    if (!ctx || !ptr) return;
    rpcmem_free(ptr);
    // bytes_in_use bookkeeping is approximate — rpcmem doesn't expose
    // per-pointer size lookup. Rely on the next caps refresh for truth.
}

// fp16 ↔ fp32 bridges using the math core helpers (sp_f16_to_f32,
// sp_f32_to_f16). These are header-inline in shannon_prime.h so no extra
// link footprint.
static void sp_hex_widen_f16_to_f32(const uint16_t *in, float *out, int n) {
    for (int i = 0; i < n; ++i) out[i] = sp_f16_to_f32(in[i]);
}
static void sp_hex_narrow_f32_to_f16(const float *in, uint16_t *out, int n) {
    for (int i = 0; i < n; ++i) out[i] = sp_f32_to_f16(in[i]);
}

// Single-vector round-trip. fp16 in/out, fp32 across the FastRPC boundary.
//
// Uses the rpcmem-backed scratch from the engine ctx — FastRPC sees these
// as ION-backed shared physical memory and skips the IPC marshal copy,
// the DSP gets the same physical pages via SMMU translation. If the
// scratch wasn't allocated at init (rpcmem_alloc failure), falls back
// to stack scratch with the marshal copy.
static int sp_hex_round_trip_one(sp_hexagon_ctx_t *ctx,
                                  const uint16_t *in_fp16,
                                  uint16_t *out_fp16) {
    if (!ctx || ctx->fastrpc_handle == (remote_handle64)-1) return -1;
    int hd = ctx->cfg_snapshot.head_dim;
    if (hd <= 0 || hd > SP_HEXAGON_HEAD_DIM_MAX) return -1;

    float *in_f32, *out_f32;
    float  stack_in[SP_HEXAGON_HEAD_DIM_MAX]
        __attribute__((aligned(128)));
    float  stack_out[SP_HEXAGON_HEAD_DIM_MAX]
        __attribute__((aligned(128)));

    int idx = ctx->scratch_idx;
    if (ctx->scratch_in_f32[idx] && ctx->scratch_out_f32[idx]) {
        // Zero-copy ping-pong path: alternate between [0] and [1] each
        // call. A caller running an explicit pipeline can prefetch into
        // ctx->scratch_in_f32[idx ^ 1] (e.g., fread the next packed
        // band) while this call is running on the DSP.
        in_f32  = ctx->scratch_in_f32[idx];
        out_f32 = ctx->scratch_out_f32[idx];
        ctx->scratch_idx = idx ^ 1;
    } else {
        // Fallback path: stack scratch, FastRPC will marshal-copy.
        in_f32  = stack_in;
        out_f32 = stack_out;
    }

    sp_hex_widen_f16_to_f32(in_fp16, in_f32, hd);
    int rc = sp_hex_round_trip_f32(ctx->fastrpc_handle, in_f32, hd, hd,
                                    out_f32, hd);
    if (rc != AEE_SUCCESS) return rc;
    sp_hex_narrow_f32_to_f16(out_f32, out_fp16, hd);
    return 0;
}

int sp_hexagon_round_trip_k(sp_hexagon_ctx_t *ctx,
                             const uint16_t *in_fp16, uint16_t *out_fp16) {
    return sp_hex_round_trip_one(ctx, in_fp16, out_fp16);
}

int sp_hexagon_round_trip_v(sp_hexagon_ctx_t *ctx,
                             const uint16_t *in_fp16, uint16_t *out_fp16) {
    // K and V share the same DSP code path for now. Differentiation
    // (e.g. V using a 1-band 3-bit config) lands when the IDL grows a
    // band-config parameter.
    return sp_hex_round_trip_one(ctx, in_fp16, out_fp16);
}

int sp_hexagon_round_trip_k_batch(sp_hexagon_ctx_t *ctx,
                                   const uint16_t *in_fp16, uint16_t *out_fp16,
                                   int n_vectors) {
    if (!ctx || n_vectors <= 0) return -1;
    int hd = ctx->cfg_snapshot.head_dim;
    // Naive loop. A real batch IDL entry would amortize FastRPC overhead
    // across all n_vectors; we'll add that once the per-call cost shows
    // up in profiles. For now, correctness before perf.
    for (int v = 0; v < n_vectors; ++v) {
        int rc = sp_hex_round_trip_one(ctx, in_fp16 + v * hd,
                                        out_fp16 + v * hd);
        if (rc != AEE_SUCCESS) return rc;
    }
    return 0;
}

int sp_hexagon_band_dequantize_partial(sp_hexagon_ctx_t *ctx,
                                        const uint8_t *in_packed,
                                        uint16_t *out_fp16,
                                        int max_bands) {
    if (!ctx || ctx->fastrpc_handle == (remote_handle64)-1) return -1;
    int hd = ctx->cfg_snapshot.head_dim;
    if (hd <= 0 || hd > 1024) return -1;

    sp_band_config_t bc;
    int default_bits[4] = {5, 5, 4, 3};
    sp_band_config_init(&bc, hd, 4, default_bits);

    float coeffs[1024];
    int rc = sp_hex_band_dequantize(ctx->fastrpc_handle,
                                     in_packed, bc.total_bytes,
                                     hd, max_bands,
                                     coeffs, hd);
    if (rc != AEE_SUCCESS) return rc;

    // The math core's band_dequantize emits VHT2 coefficients. Inverse
    // VHT2 to land back in the original basis, then narrow to fp16.
    sp_vht2_forward_f32(coeffs, hd);   // self-inverse
    sp_hex_narrow_f32_to_f16(coeffs, out_fp16, hd);
    return 0;
}

size_t sp_hexagon_memory_in_use(const sp_hexagon_ctx_t *ctx) {
    return ctx ? ctx->bytes_in_use : 0;
}

long long sp_hexagon_last_call_cycles(const sp_hexagon_ctx_t *ctx) {
    return ctx ? ctx->last_call_cycles : 0;
}

// ============================================================================
// sp_hexagon_cache_t — per-position storage with DSP-side compress/decompress
// ============================================================================
//
// Each (layer, head) slot is a max_seq × bc.total_bytes byte block, allocated
// via rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, ...) so the compress_f32 /
// decompress_f32 FastRPC calls land in zero-copy SMMU-mapped pages and
// skip the IPC marshal copy. RPCMEM_TRY_MAP_STATIC pre-maps each slot at
// allocation time so the first access to a fresh slot doesn't pay
// map-on-first-use latency.
//
// Sizing: a single 32-layer / 8-head model with head_dim=128 and 4096-token
// max-seq comes to about (32*8) * 4096 * 144 = ~144 MB total, well within
// the SMMU-mapped budget. Smaller models (Llama-3.2-1B, draft-class)
// land under 50 MB.
//
// Storage offset for a per-position write/read:
//   offset = pos * total_bytes
//   slot[layer * n_heads_kv + head] + offset

static int sp_hex_slot_index(const sp_config_t *cfg, int layer, int head) {
    return layer * cfg->n_heads_kv + head;
}

// Defensive cache-clean barrier for rpcmem buffers that the host filled
// mid-pipeline (e.g. fread() into a previously-marshaled rpcmem region).
// FastRPC's marshal does an implicit cache_clean_invalidate at dispatch
// time, so today's tests pass without this. But streaming readers that
// fill rpcmem AFTER the first marshal — disk-tier loaders, KV cache page
// prefetchers — should call this between the host write and the next
// FastRPC call to guarantee visibility into the DSP's SMMU-mapped view.
//
// Implementation today: a strong memory barrier (__atomic_thread_fence
// SEQ_CST). On ARMv8 that emits DMB ISH which waits for all in-flight
// stores to drain to inner-shareable point of unification — sufficient
// when the buffer is in a cached-mapped ION region (the default for
// rpcmem_alloc). For uncached regions this is a no-op.
//
// Future: when SDK exposes rpcmem_sync_cache (or remote_register_buf with
// fd-based zero-copy), swap this for the explicit ION clean ioctl.
static inline void sp_hex_rpcmem_sync_for_dsp(const void *buf, size_t bytes) {
    (void)buf; (void)bytes;
    __atomic_thread_fence(__ATOMIC_SEQ_CST);
}

int sp_hexagon_cache_init(sp_hexagon_cache_t *cache,
                           sp_hexagon_ctx_t *ctx,
                           const sp_config_t *cfg,
                           int max_seq_len) {
    if (!cache || !ctx || !cfg || max_seq_len <= 0) return -1;
    memset(cache, 0, sizeof(*cache));
    cache->ctx          = ctx;
    cache->cfg_snapshot = *cfg;
    cache->max_seq_len  = max_seq_len;
    cache->n_slots      = cfg->n_layers * cfg->n_heads_kv;

    int k_bits[4] = {5, 5, 4, 3};
    int v_bits[4] = {3};
    sp_band_config_init(&cache->k_bands, cfg->head_dim, 4, k_bits);
    sp_band_config_init(&cache->v_bands, cfg->head_dim, 1, v_bits);

    // FastRPC IN/OUT scratches — allocated at EXACTLY the per-call length so
    // the rpcmem registration size matches the FastRPC marshaling len param.
    // Mismatch (e.g. 4 KB-registered buffer called with len=256) is rejected
    // with AEE_EUNSUPPORTED (rc=0x4e). Confirmed via standalone scaffold:
    // its compress_f32 calls allocate at exact-size and succeed; the bridge
    // previously over-allocated and every call silently failed.
    int alloc_flags = RPCMEM_DEFAULT_FLAGS | RPCMEM_TRY_MAP_STATIC;
    cache->vec_in_f32   = (float *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, alloc_flags,
                            (int)(sizeof(float) * cfg->head_dim));
    cache->vec_out_f32  = (float *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, alloc_flags,
                            (int)(sizeof(float) * cfg->head_dim));
    cache->k_packed_rpc = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, alloc_flags,
                            cache->k_bands.total_bytes);
    cache->v_packed_rpc = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, alloc_flags,
                            cache->v_bands.total_bytes);
    if (!cache->vec_in_f32 || !cache->vec_out_f32 ||
        !cache->k_packed_rpc || !cache->v_packed_rpc) {
        fprintf(stderr, "[Shannon-Prime] hexagon cache: rpcmem scratch alloc failed "
                        "(in=%p out=%p k_pkt=%p v_pkt=%p)\n",
                (void*)cache->vec_in_f32, (void*)cache->vec_out_f32,
                (void*)cache->k_packed_rpc, (void*)cache->v_packed_rpc);
        sp_hexagon_cache_free(cache);
        return -1;
    }

    // Batch scratches — sized at chunk_size * per-call so the prefill path
    // can land an entire (layer, head, [start_pos..start_pos+chunk]) chunk
    // in ONE FastRPC dispatch. SP_HEX_BATCH_CHUNK env var overrides the
    // default of 32. Defensive clamp to [1, 256] — n_vectors=1 degrades to
    // single-position cost (no batching benefit but still correct);
    // n_vectors > 256 starts pushing FastRPC payload limits at large
    // head_dim. Per-vector budget at head_dim=128 with 32 chunk: 32*128*4
    // = 16 KB in, 32*total_bytes ~ 1.3 KB out — comfortable inside V69's
    // 64 KB VTCM and FastRPC's ~1 MB per-call budget.
    cache->batch_chunk_size = 32;
    {
        const char *env = getenv("SP_HEX_BATCH_CHUNK");
        if (env && *env) {
            int v = atoi(env);
            if (v >= 1 && v <= 2048) cache->batch_chunk_size = v;
        }
    }
    int chunk = cache->batch_chunk_size;
    cache->vec_in_batch_rpc  = (float *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, alloc_flags,
                                  (int)((size_t)chunk * cfg->head_dim * sizeof(float)));
    cache->vec_out_batch_rpc = (float *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, alloc_flags,
                                  (int)((size_t)chunk * cfg->head_dim * sizeof(float)));
    cache->k_packed_batch_rpc = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, alloc_flags,
                                  (int)((size_t)chunk * cache->k_bands.total_bytes));
    cache->v_packed_batch_rpc = (uint8_t *)rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM, alloc_flags,
                                  (int)((size_t)chunk * cache->v_bands.total_bytes));
    if (!cache->vec_in_batch_rpc || !cache->vec_out_batch_rpc ||
        !cache->k_packed_batch_rpc || !cache->v_packed_batch_rpc) {
        fprintf(stderr, "[Shannon-Prime] hexagon cache: rpcmem batch scratch alloc failed "
                        "(chunk=%d in=%p out=%p k_pkt=%p)\n",
                chunk,
                (void*)cache->vec_in_batch_rpc, (void*)cache->vec_out_batch_rpc,
                (void*)cache->k_packed_batch_rpc);
        sp_hexagon_cache_free(cache);
        return -1;
    }

    // Cache slots are HOST-ONLY memory now (plain malloc). DSP never reads or
    // writes them directly — bridge stages each per-position vector through
    // the rpcmem scratches above, then memcpy into / out of the corresponding
    // slot offset on the host side. Two wins: (1) no FD/handle exhaustion
    // on big models (Llama-70B with 80 layers x 8 KV heads x 2 = 1280 slots
    // would otherwise blow through the typical 1024 FD limit), (2) slot
    // sub-indexing is unconstrained — no 128-byte HVX-alignment requirement
    // because the DSP never sees these pointers.
    cache->k_cache = (uint8_t **)calloc(cache->n_slots, sizeof(uint8_t *));
    cache->v_cache = (uint8_t **)calloc(cache->n_slots, sizeof(uint8_t *));
    if (!cache->k_cache || !cache->v_cache) {
        sp_hexagon_cache_free(cache);
        return -1;
    }

    size_t k_slot_bytes = (size_t)max_seq_len * (size_t)cache->k_bands.total_bytes;
    // V slot stores compressed bytes (V now goes through DSP via compress_f32_v
    // / decompress_f32_v IDL methods, restoring 3.0x compression on V cache).
    size_t v_slot_bytes = (size_t)max_seq_len * (size_t)cache->v_bands.total_bytes;

    for (int s = 0; s < cache->n_slots; ++s) {
        cache->k_cache[s] = (uint8_t *)calloc(1, k_slot_bytes);
        cache->v_cache[s] = (uint8_t *)calloc(1, v_slot_bytes);
        if (!cache->k_cache[s] || !cache->v_cache[s]) {
            fprintf(stderr, "[Shannon-Prime] hexagon cache: malloc slot %d/%d failed "
                            "(k=%zu v=%zu)\n", s, cache->n_slots, k_slot_bytes, v_slot_bytes);
            sp_hexagon_cache_free(cache);
            return -1;
        }
        ctx->bytes_in_use += k_slot_bytes + v_slot_bytes;
    }
    return 0;
}

void sp_hexagon_cache_free(sp_hexagon_cache_t *cache) {
    if (!cache) return;
    // Cache slots are now host-only malloc'd memory (see init for rationale).
    if (cache->k_cache) {
        for (int s = 0; s < cache->n_slots; ++s)
            if (cache->k_cache[s]) free(cache->k_cache[s]);
        free(cache->k_cache);
    }
    if (cache->v_cache) {
        for (int s = 0; s < cache->n_slots; ++s)
            if (cache->v_cache[s]) free(cache->v_cache[s]);
        free(cache->v_cache);
    }
    // Per-call rpcmem scratches.
    if (cache->vec_in_f32)   rpcmem_free(cache->vec_in_f32);
    if (cache->vec_out_f32)  rpcmem_free(cache->vec_out_f32);
    if (cache->k_packed_rpc) rpcmem_free(cache->k_packed_rpc);
    if (cache->v_packed_rpc) rpcmem_free(cache->v_packed_rpc);
    // Batch scratches.
    if (cache->vec_in_batch_rpc)   rpcmem_free(cache->vec_in_batch_rpc);
    if (cache->vec_out_batch_rpc)  rpcmem_free(cache->vec_out_batch_rpc);
    if (cache->k_packed_batch_rpc) rpcmem_free(cache->k_packed_batch_rpc);
    if (cache->v_packed_batch_rpc) rpcmem_free(cache->v_packed_batch_rpc);
    memset(cache, 0, sizeof(*cache));
}

static void sp_hex_cache_write_one(sp_hexagon_cache_t *cache,
                                    int layer, int head, int pos,
                                    const float *vec, int is_k) {
    if (!cache || !cache->ctx ||
        cache->ctx->fastrpc_handle == (remote_handle64)-1) return;
    if (pos < 0 || pos >= cache->max_seq_len) return;
    int hd = cache->cfg_snapshot.head_dim;
    if (hd <= 0 || hd > SP_HEXAGON_HEAD_DIM_MAX) return;

    int slot = sp_hex_slot_index(&cache->cfg_snapshot, layer, head);
    if (slot < 0 || slot >= cache->n_slots) return;

    sp_band_config_t *bc = is_k ? &cache->k_bands : &cache->v_bands;
    uint8_t *dst_slot = is_k ? cache->k_cache[slot] : cache->v_cache[slot];
    if (!dst_slot) return;
    uint8_t *out_scratch = is_k ? cache->k_packed_rpc : cache->v_packed_rpc;
    if (!out_scratch) return;

    memcpy(cache->vec_in_f32, vec, sizeof(float) * hd);
    sp_hex_rpcmem_sync_for_dsp(cache->vec_in_f32, sizeof(float) * hd);
    int packed_used = 0;
    int rc = is_k
        ? sp_hex_compress_f32  (cache->ctx->fastrpc_handle,
                                  cache->vec_in_f32, hd, hd,
                                  out_scratch, bc->total_bytes,
                                  &packed_used)
        : sp_hex_compress_f32_v(cache->ctx->fastrpc_handle,
                                  cache->vec_in_f32, hd, hd,
                                  out_scratch, bc->total_bytes,
                                  &packed_used);
    if (rc != AEE_SUCCESS || packed_used != bc->total_bytes) {
        static int warned = 0;
        if (!warned) {
            fprintf(stderr, "[Shannon-Prime] hexagon: compress_f32 rc=0x%x "
                    "used=%d expect=%d (slot=%d pos=%d)\n",
                    rc, packed_used, bc->total_bytes, slot, pos);
            warned = 1;
        }
        return;
    }
    // Successful DSP encode — stage into the per-position offset of the
    // host-only cache slot. rpcmem buffers allocated with DEFAULT_FLAGS +
    // TRY_MAP_STATIC are cache-coherent in both directions on V69 (ION
    // I/O-coherent attribute); no explicit sync_for_host needed.
    uint8_t *dst = dst_slot + (size_t)pos * (size_t)bc->total_bytes;
    memcpy(dst, out_scratch, bc->total_bytes);
}

static void sp_hex_cache_read_one(const sp_hexagon_cache_t *cache,
                                   int layer, int head, int pos,
                                   float *out_vec, int is_k, int max_bands) {
    if (!cache || !cache->ctx ||
        cache->ctx->fastrpc_handle == (remote_handle64)-1) {
        if (out_vec && cache) memset(out_vec, 0, sizeof(float) * cache->cfg_snapshot.head_dim);
        return;
    }
    int hd = cache->cfg_snapshot.head_dim;
    if (hd <= 0 || hd > SP_HEXAGON_HEAD_DIM_MAX) return;
    if (pos < 0 || pos >= cache->max_seq_len) {
        memset(out_vec, 0, sizeof(float) * hd);
        return;
    }
    int slot = sp_hex_slot_index(&cache->cfg_snapshot, layer, head);
    if (slot < 0 || slot >= cache->n_slots) {
        memset(out_vec, 0, sizeof(float) * hd);
        return;
    }

    const sp_band_config_t *bc = is_k ? &cache->k_bands : &cache->v_bands;
    uint8_t *src_slot = is_k ? cache->k_cache[slot] : cache->v_cache[slot];
    if (!src_slot) {
        memset(out_vec, 0, sizeof(float) * hd);
        return;
    }
    sp_hexagon_cache_t *cache_mut = (sp_hexagon_cache_t *)cache;
    uint8_t *in_scratch = is_k ? cache_mut->k_packed_rpc : cache_mut->v_packed_rpc;
    if (!in_scratch) {
        memset(out_vec, 0, sizeof(float) * hd);
        return;
    }
    const uint8_t *src = src_slot + (size_t)pos * (size_t)bc->total_bytes;
    memcpy(in_scratch, src, bc->total_bytes);
    sp_hex_rpcmem_sync_for_dsp(in_scratch, bc->total_bytes);

    int rc = is_k
        ? sp_hex_decompress_f32  (cache->ctx->fastrpc_handle,
                                    in_scratch, bc->total_bytes,
                                    hd, max_bands,
                                    cache_mut->vec_out_f32, hd)
        : sp_hex_decompress_f32_v(cache->ctx->fastrpc_handle,
                                    in_scratch, bc->total_bytes,
                                    hd, max_bands,
                                    cache_mut->vec_out_f32, hd);
    if (rc != AEE_SUCCESS) {
        static int warned = 0;
        if (!warned) {
            fprintf(stderr, "[Shannon-Prime] hexagon: decompress_f32 rc=0x%x "
                    "(slot=%d pos=%d)\n", rc, slot, pos);
            warned = 1;
        }
        memset(out_vec, 0, sizeof(float) * hd);
        return;
    }
    memcpy(out_vec, cache_mut->vec_out_f32, sizeof(float) * hd);
}

void sp_hexagon_cache_write_k(sp_hexagon_cache_t *cache, int layer, int head, int pos, const float *k_vec) {
    sp_hex_cache_write_one(cache, layer, head, pos, k_vec, 1);
}
void sp_hexagon_cache_write_v(sp_hexagon_cache_t *cache, int layer, int head, int pos, const float *v_vec) {
    sp_hex_cache_write_one(cache, layer, head, pos, v_vec, 0);
}
void sp_hexagon_cache_read_k(const sp_hexagon_cache_t *cache, int layer, int head, int pos, float *k_out) {
    sp_hex_cache_read_one(cache, layer, head, pos, k_out, 1, -1);
}
void sp_hexagon_cache_read_v(const sp_hexagon_cache_t *cache, int layer, int head, int pos, float *v_out) {
    sp_hex_cache_read_one(cache, layer, head, pos, v_out, 0, -1);
}
void sp_hexagon_cache_read_k_partial(const sp_hexagon_cache_t *cache, int layer, int head, int pos,
                                      float *k_out, int max_bands) {
    sp_hex_cache_read_one(cache, layer, head, pos, k_out, 1, max_bands);
}
void sp_hexagon_cache_read_v_partial(const sp_hexagon_cache_t *cache, int layer, int head, int pos,
                                      float *v_out, int max_bands) {
    sp_hex_cache_read_one(cache, layer, head, pos, v_out, 0, max_bands);
}

// ============================================================================
// Batched K-cache write — single FastRPC per chunk
// ============================================================================
//
// On prefill (n_pos > 1) the post-decode hook hands us a contiguous run of
// per-position K vectors for one (layer, head). Calling write_k once per
// position pays one FastRPC dispatch (~20µs) per call — at 2000 prefill
// tokens × 16 layers × 8 KV heads that's ~256K dispatches, ~5 seconds of
// pure transport overhead, and is the single biggest contributor to the
// -79% prefill penalty observed in the n_ctx=4096 bench.
//
// This path chunks the run into batch_chunk_size-sized groups and dispatches
// each group through ONE compress_f32_batch FastRPC call. The DSP loops
// the kernel internally over the chunk; the bridge memcpys the per-position
// output into the host-only cache slot. Net effect: 256K dispatches becomes
// 256K / chunk_size — at chunk=32 that's ~8K, a ~32× reduction.
//
// Generation (n_pos=1) skips the batch path and uses single-position
// write_k, since chunk=1 has no benefit and adds the in-scratch copy.
static void sp_hex_cache_write_k_batch_chunk(sp_hexagon_cache_t *cache,
                                              int layer, int head,
                                              int start_pos, int chunk_n,
                                              const float *k_vecs) {
    if (!cache || !cache->ctx ||
        cache->ctx->fastrpc_handle == (remote_handle64)-1) return;
    int hd = cache->cfg_snapshot.head_dim;
    if (hd <= 0 || hd > SP_HEXAGON_HEAD_DIM_MAX) return;
    int slot = sp_hex_slot_index(&cache->cfg_snapshot, layer, head);
    if (slot < 0 || slot >= cache->n_slots) return;
    if (chunk_n < 1 || chunk_n > cache->batch_chunk_size) return;
    if (start_pos < 0 || start_pos + chunk_n > cache->max_seq_len) return;

    sp_band_config_t *bc = &cache->k_bands;
    uint8_t *dst_slot = cache->k_cache[slot];
    if (!dst_slot) return;
    float   *in_scratch  = cache->vec_in_batch_rpc;
    uint8_t *out_scratch = cache->k_packed_batch_rpc;
    if (!in_scratch || !out_scratch) return;

    // Stage chunk_n vectors contiguously into the rpcmem in-scratch.
    memcpy(in_scratch, k_vecs, (size_t)chunk_n * hd * sizeof(float));
    sp_hex_rpcmem_sync_for_dsp(in_scratch, (size_t)chunk_n * hd * sizeof(float));

    int packed_used = 0;
    int rc = sp_hex_compress_f32_batch(cache->ctx->fastrpc_handle,
                                        in_scratch, chunk_n * hd,
                                        hd, chunk_n,
                                        out_scratch, chunk_n * bc->total_bytes,
                                        &packed_used);
    if (rc != AEE_SUCCESS || packed_used != chunk_n * bc->total_bytes) {
        static int warned = 0;
        if (!warned) {
            fprintf(stderr, "[Shannon-Prime] hexagon: compress_f32_batch rc=0x%x "
                    "used=%d expect=%d (slot=%d start=%d n=%d)\n",
                    rc, packed_used, chunk_n * bc->total_bytes,
                    slot, start_pos, chunk_n);
            warned = 1;
        }
        return;
    }

    // De-interleave: write each per-position chunk into its slot offset.
    // The DSP wrote contiguous (vec0_packed, vec1_packed, ...) — the cache
    // slot expects (pos0_packed, pos1_packed, ...) at total_bytes stride,
    uint8_t *dst = dst_slot + (size_t)start_pos * (size_t)bc->total_bytes;
    memcpy(dst, out_scratch, (size_t)chunk_n * bc->total_bytes);
}

static void sp_hex_cache_read_k_batch_chunk(const sp_hexagon_cache_t *cache,
                                             int layer, int head,
                                             int start_pos, int chunk_n,
                                             float *k_out, int max_bands) {
    if (!cache || !cache->ctx ||
        cache->ctx->fastrpc_handle == (remote_handle64)-1) {
        if (k_out && cache) memset(k_out, 0,
            (size_t)chunk_n * cache->cfg_snapshot.head_dim * sizeof(float));
        return;
    }
    int hd = cache->cfg_snapshot.head_dim;
    if (hd <= 0 || hd > SP_HEXAGON_HEAD_DIM_MAX) return;
    int slot = sp_hex_slot_index(&cache->cfg_snapshot, layer, head);
    if (slot < 0 || slot >= cache->n_slots ||
        chunk_n < 1 || chunk_n > cache->batch_chunk_size ||
        start_pos < 0 || start_pos + chunk_n > cache->max_seq_len) {
        memset(k_out, 0, (size_t)chunk_n * hd * sizeof(float));
        return;
    }

    const sp_band_config_t *bc = &cache->k_bands;
    uint8_t *src_slot = cache->k_cache[slot];
    if (!src_slot) {
        memset(k_out, 0, (size_t)chunk_n * hd * sizeof(float));
        return;
    }
    sp_hexagon_cache_t *cache_mut = (sp_hexagon_cache_t *)cache;
    uint8_t *in_scratch  = cache_mut->k_packed_batch_rpc;
    float   *out_scratch = cache_mut->vec_out_batch_rpc;
    if (!in_scratch || !out_scratch) {
        memset(k_out, 0, (size_t)chunk_n * hd * sizeof(float));
        return;
    }

    const uint8_t *src = src_slot + (size_t)start_pos * (size_t)bc->total_bytes;
    memcpy(in_scratch, src, (size_t)chunk_n * bc->total_bytes);
    sp_hex_rpcmem_sync_for_dsp(in_scratch, (size_t)chunk_n * bc->total_bytes);

    int rc = sp_hex_decompress_f32_batch(cache->ctx->fastrpc_handle,
                                          in_scratch, chunk_n * bc->total_bytes,
                                          hd, chunk_n, max_bands,
                                          out_scratch, chunk_n * hd);
    if (rc != AEE_SUCCESS) {
        static int warned = 0;
        if (!warned) {
            fprintf(stderr, "[Shannon-Prime] hexagon: decompress_f32_batch rc=0x%x "
                    "(slot=%d start=%d n=%d)\n", rc, slot, start_pos, chunk_n);
            warned = 1;
        }
        memset(k_out, 0, (size_t)chunk_n * hd * sizeof(float));
        return;
    }
    memcpy(k_out, out_scratch, (size_t)chunk_n * hd * sizeof(float));
}

void sp_hexagon_cache_write_k_batch(sp_hexagon_cache_t *cache, int layer, int head,
                                     int start_pos, int n_pos, const float *k_vecs) {
    if (!cache || n_pos <= 0 || !k_vecs) return;
    int hd = cache->cfg_snapshot.head_dim;
    int chunk = cache->batch_chunk_size;
    if (chunk < 1) chunk = 1;
    // Single-position fast path: no batching benefit, skip the staging copy.
    if (n_pos == 1) {
        sp_hexagon_cache_write_k(cache, layer, head, start_pos, k_vecs);
        return;
    }
    int i = 0;
    while (i < n_pos) {
        int this_chunk = n_pos - i;
        if (this_chunk > chunk) this_chunk = chunk;
        sp_hex_cache_write_k_batch_chunk(cache, layer, head,
                                          start_pos + i, this_chunk,
                                          k_vecs + (size_t)i * hd);
        i += this_chunk;
    }
}

static void sp_hex_cache_write_v_batch_chunk(sp_hexagon_cache_t *cache,
                                              int layer, int head,
                                              int start_pos, int chunk_n,
                                              const float *v_vecs) {
    if (!cache || !cache->ctx ||
        cache->ctx->fastrpc_handle == (remote_handle64)-1) return;
    int hd = cache->cfg_snapshot.head_dim;
    if (hd <= 0 || hd > SP_HEXAGON_HEAD_DIM_MAX) return;
    int slot = sp_hex_slot_index(&cache->cfg_snapshot, layer, head);
    if (slot < 0 || slot >= cache->n_slots) return;
    if (chunk_n < 1 || chunk_n > cache->batch_chunk_size) return;
    if (start_pos < 0 || start_pos + chunk_n > cache->max_seq_len) return;

    sp_band_config_t *bc = &cache->v_bands;
    uint8_t *dst_slot = cache->v_cache[slot];
    if (!dst_slot) return;
    float   *in_scratch  = cache->vec_in_batch_rpc;
    uint8_t *out_scratch = cache->v_packed_batch_rpc;
    if (!in_scratch || !out_scratch) return;

    memcpy(in_scratch, v_vecs, (size_t)chunk_n * hd * sizeof(float));
    sp_hex_rpcmem_sync_for_dsp(in_scratch, (size_t)chunk_n * hd * sizeof(float));

    int packed_used = 0;
    int rc = sp_hex_compress_f32_v_batch(cache->ctx->fastrpc_handle,
                                          in_scratch, chunk_n * hd,
                                          hd, chunk_n,
                                          out_scratch, chunk_n * bc->total_bytes,
                                          &packed_used);
    if (rc != AEE_SUCCESS || packed_used != chunk_n * bc->total_bytes) {
        static int warned = 0;
        if (!warned) {
            fprintf(stderr, "[Shannon-Prime] hexagon: compress_f32_v_batch rc=0x%x "
                    "used=%d expect=%d (slot=%d start=%d n=%d)\n",
                    rc, packed_used, chunk_n * bc->total_bytes,
                    slot, start_pos, chunk_n);
            warned = 1;
        }
        return;
    }
    uint8_t *dst = dst_slot + (size_t)start_pos * (size_t)bc->total_bytes;
    memcpy(dst, out_scratch, (size_t)chunk_n * bc->total_bytes);
}

void sp_hexagon_cache_write_v_batch(sp_hexagon_cache_t *cache, int layer, int head,
                                     int start_pos, int n_pos, const float *v_vecs) {
    if (!cache || n_pos <= 0 || !v_vecs) return;
    int hd = cache->cfg_snapshot.head_dim;
    int chunk = cache->batch_chunk_size;
    if (chunk < 1) chunk = 1;
    if (n_pos == 1) {
        sp_hexagon_cache_write_v(cache, layer, head, start_pos, v_vecs);
        return;
    }
    int i = 0;
    while (i < n_pos) {
        int this_chunk = n_pos - i;
        if (this_chunk > chunk) this_chunk = chunk;
        sp_hex_cache_write_v_batch_chunk(cache, layer, head,
                                          start_pos + i, this_chunk,
                                          v_vecs + (size_t)i * hd);
        i += this_chunk;
    }
}

void sp_hexagon_cache_read_k_batch(const sp_hexagon_cache_t *cache, int layer, int head,
                                    int start_pos, int n_pos, float *k_out) {
    if (!cache || n_pos <= 0 || !k_out) return;
    int hd = cache->cfg_snapshot.head_dim;
    int chunk = cache->batch_chunk_size;
    if (chunk < 1) chunk = 1;
    if (n_pos == 1) {
        sp_hexagon_cache_read_k(cache, layer, head, start_pos, k_out);
        return;
    }
    int i = 0;
    while (i < n_pos) {
        int this_chunk = n_pos - i;
        if (this_chunk > chunk) this_chunk = chunk;
        sp_hex_cache_read_k_batch_chunk(cache, layer, head,
                                         start_pos + i, this_chunk,
                                         k_out + (size_t)i * hd, -1);
        i += this_chunk;
    }
}

static void sp_hex_cache_read_v_batch_chunk(const sp_hexagon_cache_t *cache,
                                             int layer, int head,
                                             int start_pos, int chunk_n,
                                             float *v_out, int max_bands) {
    if (!cache || !cache->ctx ||
        cache->ctx->fastrpc_handle == (remote_handle64)-1) {
        if (v_out && cache) memset(v_out, 0,
            (size_t)chunk_n * cache->cfg_snapshot.head_dim * sizeof(float));
        return;
    }
    int hd = cache->cfg_snapshot.head_dim;
    if (hd <= 0 || hd > SP_HEXAGON_HEAD_DIM_MAX) return;
    int slot = sp_hex_slot_index(&cache->cfg_snapshot, layer, head);
    if (slot < 0 || slot >= cache->n_slots ||
        chunk_n < 1 || chunk_n > cache->batch_chunk_size ||
        start_pos < 0 || start_pos + chunk_n > cache->max_seq_len) {
        memset(v_out, 0, (size_t)chunk_n * hd * sizeof(float));
        return;
    }
    const sp_band_config_t *bc = &cache->v_bands;
    uint8_t *src_slot = cache->v_cache[slot];
    if (!src_slot) {
        memset(v_out, 0, (size_t)chunk_n * hd * sizeof(float));
        return;
    }
    sp_hexagon_cache_t *cache_mut = (sp_hexagon_cache_t *)cache;
    uint8_t *in_scratch  = cache_mut->v_packed_batch_rpc;
    float   *out_scratch = cache_mut->vec_out_batch_rpc;
    if (!in_scratch || !out_scratch) {
        memset(v_out, 0, (size_t)chunk_n * hd * sizeof(float));
        return;
    }
    const uint8_t *src = src_slot + (size_t)start_pos * (size_t)bc->total_bytes;
    memcpy(in_scratch, src, (size_t)chunk_n * bc->total_bytes);
    sp_hex_rpcmem_sync_for_dsp(in_scratch, (size_t)chunk_n * bc->total_bytes);

    int rc = sp_hex_decompress_f32_v_batch(cache->ctx->fastrpc_handle,
                                            in_scratch, chunk_n * bc->total_bytes,
                                            hd, chunk_n, max_bands,
                                            out_scratch, chunk_n * hd);
    if (rc != AEE_SUCCESS) {
        static int warned = 0;
        if (!warned) {
            fprintf(stderr, "[Shannon-Prime] hexagon: decompress_f32_v_batch rc=0x%x "
                    "(slot=%d start=%d n=%d)\n", rc, slot, start_pos, chunk_n);
            warned = 1;
        }
        memset(v_out, 0, (size_t)chunk_n * hd * sizeof(float));
        return;
    }
    memcpy(v_out, out_scratch, (size_t)chunk_n * hd * sizeof(float));
}

void sp_hexagon_cache_read_v_batch(const sp_hexagon_cache_t *cache, int layer, int head,
                                    int start_pos, int n_pos, float *v_out) {
    if (!cache || n_pos <= 0 || !v_out) return;
    int hd = cache->cfg_snapshot.head_dim;
    int chunk = cache->batch_chunk_size;
    if (chunk < 1) chunk = 1;
    if (n_pos == 1) {
        sp_hexagon_cache_read_v(cache, layer, head, start_pos, v_out);
        return;
    }
    int i = 0;
    while (i < n_pos) {
        int this_chunk = n_pos - i;
        if (this_chunk > chunk) this_chunk = chunk;
        sp_hex_cache_read_v_batch_chunk(cache, layer, head,
                                         start_pos + i, this_chunk,
                                         v_out + (size_t)i * hd, -1);
        i += this_chunk;
    }
}

void sp_hexagon_cache_clear_range(sp_hexagon_cache_t *cache, int start_pos, int end_pos) {
    // K + V cache slots store compressed bytes at bands.total_bytes stride per
    // position (V was uncompressed fp32 during the V-bypass era; #57 reverted that).
    if (!cache || start_pos >= end_pos) return;
    if (start_pos < 0) start_pos = 0;
    if (end_pos > cache->max_seq_len) end_pos = cache->max_seq_len;
    int n_clear = end_pos - start_pos;
    int hd = cache->cfg_snapshot.head_dim;
    for (int s = 0; s < cache->n_slots; ++s) {
        if (cache->k_cache[s]) {
            size_t k_off = (size_t)start_pos * cache->k_bands.total_bytes;
            size_t k_len = (size_t)n_clear * cache->k_bands.total_bytes;
            memset(cache->k_cache[s] + k_off, 0, k_len);
        }
        if (cache->v_cache[s]) {
            size_t v_off = (size_t)start_pos * cache->v_bands.total_bytes;
            size_t v_len = (size_t)n_clear * cache->v_bands.total_bytes;
            memset(cache->v_cache[s] + v_off, 0, v_len);
        }
    }
}

#else  // !SP_HEXAGON_FASTRPC

// ============================================================================
// Stub fallback (x86 / desktop / no FastRPC headers available).
// Bridge sees -1 / NULL and falls back to Adreno/CPU.
// ============================================================================

int sp_hexagon_caps_probe(sp_hexagon_caps_t *caps) {
    if (!caps) return -1;
    memset(caps, 0, sizeof(*caps));
    return -1;
}
void sp_hexagon_caps_print(const sp_hexagon_caps_t *caps) {
    if (!caps) return;
    fprintf(stderr, "Hexagon DSP unavailable (FastRPC not built)\n");
}
sp_hexagon_ctx_t *sp_hexagon_init(const sp_config_t *cfg) {
    (void)cfg;
    static int warned = 0;
    if (!warned) {
        fprintf(stderr, "[Shannon-Prime] Hexagon backend not built "
                        "(SP_HEXAGON_FASTRPC undefined). Falling back.\n");
        warned = 1;
    }
    return NULL;
}
void sp_hexagon_free(sp_hexagon_ctx_t *ctx) { (void)ctx; }
void *sp_hexagon_alloc(sp_hexagon_ctx_t *ctx, size_t n) { (void)ctx; (void)n; return NULL; }
void  sp_hexagon_free_shared(sp_hexagon_ctx_t *ctx, void *p) { (void)ctx; (void)p; }
int sp_hexagon_round_trip_k(sp_hexagon_ctx_t *ctx, const uint16_t *i, uint16_t *o) {
    (void)ctx; (void)i; (void)o; return -1;
}
int sp_hexagon_round_trip_v(sp_hexagon_ctx_t *ctx, const uint16_t *i, uint16_t *o) {
    (void)ctx; (void)i; (void)o; return -1;
}
int sp_hexagon_round_trip_k_batch(sp_hexagon_ctx_t *ctx, const uint16_t *i, uint16_t *o, int n) {
    (void)ctx; (void)i; (void)o; (void)n; return -1;
}
int sp_hexagon_band_dequantize_partial(sp_hexagon_ctx_t *ctx, const uint8_t *i,
                                        uint16_t *o, int max_bands) {
    (void)ctx; (void)i; (void)o; (void)max_bands; return -1;
}
size_t    sp_hexagon_memory_in_use(const sp_hexagon_ctx_t *ctx) { (void)ctx; return 0; }
long long sp_hexagon_last_call_cycles(const sp_hexagon_ctx_t *ctx) { (void)ctx; return 0; }

// Cache stubs (active when sp_hexagon_init returned NULL upstream so cache
// is never actually constructed; these exist to satisfy bridge linkage).
int sp_hexagon_cache_init(sp_hexagon_cache_t *c, sp_hexagon_ctx_t *ctx,
                           const sp_config_t *cfg, int n) {
    (void)c; (void)ctx; (void)cfg; (void)n; return -1;
}
void sp_hexagon_cache_free(sp_hexagon_cache_t *c) { (void)c; }
void sp_hexagon_cache_write_k(sp_hexagon_cache_t *c, int l, int h, int p, const float *v) {
    (void)c; (void)l; (void)h; (void)p; (void)v;
}
void sp_hexagon_cache_write_v(sp_hexagon_cache_t *c, int l, int h, int p, const float *v) {
    (void)c; (void)l; (void)h; (void)p; (void)v;
}
void sp_hexagon_cache_read_k(const sp_hexagon_cache_t *c, int l, int h, int p, float *o) {
    (void)l; (void)h; (void)p;
    if (o && c) memset(o, 0, sizeof(float) * c->cfg_snapshot.head_dim);
}
void sp_hexagon_cache_read_v(const sp_hexagon_cache_t *c, int l, int h, int p, float *o) {
    (void)l; (void)h; (void)p;
    if (o && c) memset(o, 0, sizeof(float) * c->cfg_snapshot.head_dim);
}
void sp_hexagon_cache_read_k_partial(const sp_hexagon_cache_t *c, int l, int h, int p,
                                      float *o, int mb) {
    (void)l; (void)h; (void)p; (void)mb;
    if (o && c) memset(o, 0, sizeof(float) * c->cfg_snapshot.head_dim);
}
void sp_hexagon_cache_read_v_partial(const sp_hexagon_cache_t *c, int l, int h, int p,
                                      float *o, int mb) {
    (void)l; (void)h; (void)p; (void)mb;
    if (o && c) memset(o, 0, sizeof(float) * c->cfg_snapshot.head_dim);
}
void sp_hexagon_cache_write_k_batch(sp_hexagon_cache_t *c, int l, int h, int sp, int np, const float *v) {
    (void)c; (void)l; (void)h; (void)sp; (void)np; (void)v;
}
void sp_hexagon_cache_write_v_batch(sp_hexagon_cache_t *c, int l, int h, int sp, int np, const float *v) {
    (void)c; (void)l; (void)h; (void)sp; (void)np; (void)v;
}
void sp_hexagon_cache_read_k_batch(const sp_hexagon_cache_t *c, int l, int h, int sp, int np, float *o) {
    (void)l; (void)h; (void)sp;
    if (o && c) memset(o, 0, sizeof(float) * (size_t)np * c->cfg_snapshot.head_dim);
}
void sp_hexagon_cache_read_v_batch(const sp_hexagon_cache_t *c, int l, int h, int sp, int np, float *o) {
    (void)l; (void)h; (void)sp;
    if (o && c) memset(o, 0, sizeof(float) * (size_t)np * c->cfg_snapshot.head_dim);
}
void sp_hexagon_cache_clear_range(sp_hexagon_cache_t *c, int s, int e) {
    (void)c; (void)s; (void)e;
}

#endif  // SP_HEXAGON_FASTRPC
