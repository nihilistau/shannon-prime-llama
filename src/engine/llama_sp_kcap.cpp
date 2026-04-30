// Shannon-Prime VHT2: K-capture custom ggml op implementation (Phase 1.7).
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
// Licensed under AGPLv3. Commercial license: raydaniels@gmail.com

#include "llama_sp_kcap.h"
#include "shannon_prime_llama.h"
#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <vector>

// Forward declaration — defined in llama-context.cpp inside the
// LLAMA_SHANNON_PRIME block. Returns the sp_llama_ctx_t pointer for
// the current ubatch's model (set by llama_sp_set_current_state at the
// top of process_ubatch). Returns nullptr if SP isn't active for this
// model — emitter then no-ops, falling back to standard cpy_k semantics
// up the call chain.
extern "C" void * llama_sp_get_current_sp_ctx_void(void);

// Per-ubatch flag: post_compute should skip the K-side roundtrip when
// FAST_PATH=1 has emitted kcap ops at graph build. Cleared at
// process_ubatch entry; set by the emitter on first emit.
static std::atomic<int> g_kcap_skip_post_compute_k{0};

// Userdata pool — one slot per layer index modulo SP_MAX_LAYERS, lives
// in static storage so the pointer in ggml_map_custom2's userdata
// outlives graph build and stays valid through graph compute. Single
// active llama_context per thread => no slot collision.
static constexpr int SP_KCAP_MAX_LAYERS = 256;
static thread_local llama_sp_kcap_userdata g_kcap_ud_pool[SP_KCAP_MAX_LAYERS];

extern "C" void llama_sp_kcap_compute(struct ggml_tensor * dst,
                                       const struct ggml_tensor * a,
                                       const struct ggml_tensor * b,
                                       int ith, int nth, void * userdata) {
    (void)dst;

    auto * ud = (llama_sp_kcap_userdata *)userdata;
    if (!ud || !ud->sp_ctx) return;
    if (!a || !a->data) return;
    if (!b || !b->data) return;

    const int nhk = ud->n_heads_kv;
    const int hd  = ud->head_dim;
    const int nt  = ud->n_tokens;
    const int il  = ud->layer_idx;

    if (nhk <= 0 || hd <= 0 || nt <= 0) return;

    auto * sp_ctx = (sp_llama_ctx_t *)ud->sp_ctx;

    // k_idxs is a 1D int32 vector of slot indices, one per ubatch token.
    const int32_t * slots = (const int32_t *)b->data;

    // Phase 1.7 diagnostic: log once per process to confirm op fired and
    // sniff the data we're about to compress. Only thread 0 logs.
    if (ith == 0) {
        static std::atomic<int> compute_count{0};
        static std::atomic<bool> verbose_init{false};
        static bool verbose = false;
        if (!verbose_init.exchange(true)) {
            const char * v = std::getenv("SHANNON_PRIME_VERBOSE");
            verbose = (v && v[0] == '1');
        }
        if (verbose && compute_count.fetch_add(1) < 4) {
            const float * head0 = (const float *)a->data;
            std::fprintf(stderr,
                "[SP] kcap_compute(il=%d): nhk=%d hd=%d nt=%d slots[0]=%d "
                "k_cur[0..3]=[%.4f %.4f %.4f %.4f] type=%d nb=[%lld %lld %lld]\n",
                il, nhk, hd, nt, slots[0],
                head0[0], head0[1], head0[2], head0[3], (int)a->type,
                (long long)a->nb[0], (long long)a->nb[1], (long long)a->nb[2]);
        }
    }

    // Split heads across threads. n_heads_kv is small (2-8 for GQA), so
    // for nth > nhk many threads idle — that's fine; FastRPC dispatch
    // dominates compute and over-parallelising would just multiply
    // bridge dispatches.
    const int per_thread = (nhk + nth - 1) / nth;
    const int h_start    = std::min(ith * per_thread, nhk);
    const int h_end      = std::min(h_start + per_thread, nhk);
    if (h_start >= h_end) return;

    // Per-thread scratch — sized once at first call, reused across
    // (layer × head × ubatch) invocations on the same worker thread.
    thread_local std::vector<float> tl_batch;
    const size_t need = (size_t)nt * (size_t)hd;
    if (tl_batch.size() < need) tl_batch.resize(need);

    // a layout: [head_dim (ne[0]), n_heads_kv (ne[1]), n_tokens (ne[2])]
    // Strides via a->nb[].
    for (int h = h_start; h < h_end; ++h) {
        for (int t = 0; t < nt; ++t) {
            const float * src = (const float *)((const char *)a->data
                + (size_t)t * a->nb[2]
                + (size_t)h * a->nb[1]);
            std::memcpy(tl_batch.data() + (size_t)t * hd, src,
                        (size_t)hd * sizeof(float));
        }

        const int first_slot = slots[0];
        if (nt > 1) {
            // Prefill / spec-decode wide ubatch — single batched FastRPC
            // (the bridge chunks internally per SP_HEX_BATCH_CHUNK).
            sp_llama_write_k_batch(sp_ctx, il, h, first_slot, nt,
                                    tl_batch.data());
        } else {
            // Single-token decode — no batching benefit, single FastRPC.
            sp_llama_write_k(sp_ctx, il, h, first_slot, tl_batch.data());
        }
    }
}

extern "C" int llama_sp_kcap_emit(struct ggml_context * ctx0,
                                   struct ggml_cgraph * gf,
                                   struct ggml_tensor * k_cur,
                                   struct ggml_tensor * k_idxs,
                                   int layer_idx) {
    static std::atomic<int> emit_count{0};
    static std::atomic<bool> verbose_init{false};
    static bool verbose = false;
    if (!verbose_init.exchange(true)) {
        const char * v = std::getenv("SHANNON_PRIME_VERBOSE");
        verbose = (v && v[0] == '1');
    }
    if (verbose && emit_count.fetch_add(1) < 3) {
        std::fprintf(stderr,
            "[SP] kcap_emit(il=%d): k_cur=%p type=%d ne=[%lld,%lld,%lld,%lld] k_idxs=%p type=%d ne=[%lld,%lld]\n",
            layer_idx, (void *)k_cur, k_cur ? (int)k_cur->type : -1,
            k_cur ? (long long)k_cur->ne[0] : -1, k_cur ? (long long)k_cur->ne[1] : -1,
            k_cur ? (long long)k_cur->ne[2] : -1, k_cur ? (long long)k_cur->ne[3] : -1,
            (void *)k_idxs, k_idxs ? (int)k_idxs->type : -1,
            k_idxs ? (long long)k_idxs->ne[0] : -1, k_idxs ? (long long)k_idxs->ne[1] : -1);
    }
    if (!ctx0 || !gf || !k_cur || !k_idxs) return 0;
    if (layer_idx < 0 || layer_idx >= SP_KCAP_MAX_LAYERS) return 0;

    // k_cur shape sanity. We expect [head_dim, n_heads_kv, n_tokens].
    // build_attn variants that pass a 2D k_cur would need a different
    // dispatch — bail and let the caller fall through to cpy_k.
    if (k_cur->ne[0] <= 0 || k_cur->ne[1] <= 0 || k_cur->ne[2] <= 0) return 0;

    void * sp_ctx_v = llama_sp_get_current_sp_ctx_void();
    if (!sp_ctx_v) return 0;

    llama_sp_kcap_userdata & ud = g_kcap_ud_pool[layer_idx];
    ud.layer_idx  = layer_idx;
    ud.n_heads_kv = (int)k_cur->ne[1];
    ud.head_dim   = (int)k_cur->ne[0];
    ud.n_tokens   = (int)k_cur->ne[2];
    ud.sp_ctx     = sp_ctx_v;

    // Single-threaded: FastRPC dispatch is serial per cDSP context, so
    // splitting the head loop across worker threads just queues up
    // concurrent FastRPC calls that serialise in the bridge anyway.
    // Keeps the per-layer FastRPC hot-loop on one thread.
    struct ggml_tensor * cap = ggml_map_custom2(ctx0, k_cur, k_idxs,
                                                 llama_sp_kcap_compute,
                                                 1, &ud);
    if (!cap) return 0;
    ggml_build_forward_expand(gf, cap);

    g_kcap_skip_post_compute_k.store(1, std::memory_order_relaxed);
    return 1;
}

extern "C" int llama_sp_kcap_skip_post_compute_k(void) {
    return g_kcap_skip_post_compute_k.load(std::memory_order_relaxed);
}

extern "C" void llama_sp_kcap_clear_post_compute_skip(void) {
    g_kcap_skip_post_compute_k.store(0, std::memory_order_relaxed);
}
