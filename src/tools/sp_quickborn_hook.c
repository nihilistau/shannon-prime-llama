// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

// SP-QuickBorn outbound side-channel — capture + env gate.
// HTTP POST itself lives in sp_quickborn_http.cpp (uses vendored cpp-httplib).

#include "sp_quickborn_hook.h"

#include "ggml.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_MSC_VER)
#  define SP_THREAD_LOCAL __declspec(thread)
#else
#  define SP_THREAD_LOCAL __thread
#endif

// ----------------------------------------------------------------------
// Per-decode capture state.
// ----------------------------------------------------------------------
static SP_THREAD_LOCAL struct sp_quickborn_capture *g_sp_capture = NULL;

void sp_quickborn_capture_set(struct sp_quickborn_capture *cap) {
    g_sp_capture = cap;
}

void sp_quickborn_capture_post_layer(int layer_idx,
                                     const struct ggml_tensor *cur) {
    // One-shot stderr proof-of-life when env is on, even with no capture
    // active. Lets Phase G.1 verify the patched dll is the one LM Studio
    // actually loaded, without needing the full spec-decode wire-in (G.2).
    if (sp_quickborn_enabled()) {
        static int announced = 0;
        if (!announced) {
            announced = 1;
            fprintf(stderr,
                "[sp-quickborn] hook live - first layer=%d url=%s timeout=%dms\n",
                layer_idx, sp_quickborn_url(), sp_quickborn_timeout_ms());
            fflush(stderr);
        }
    }

    struct sp_quickborn_capture *cap = g_sp_capture;
    if (!cap || !cur) return;

    int slot = -1;
    for (int i = 0; i < cap->n_layers; ++i) {
        if (cap->layer_ids[i] == layer_idx) { slot = i; break; }
    }
    if (slot < 0) return;

    const int hidden = (int)cur->ne[0];
    const int n_tok  = (int)cur->ne[1];
    if (cap->hidden_size == 0) cap->hidden_size = hidden;
    if (n_tok < cap->n_positions || !cap->bufs[slot]) return;

    // Copy the trailing `n_positions` columns. ggml stores [hidden, n_tok]
    // contiguous so the trailing chunk is a single memcpy.
    const float *src = (const float *)cur->data
        + (size_t)(n_tok - cap->n_positions) * (size_t)hidden;
    memcpy(cap->bufs[slot], src,
           (size_t)hidden * (size_t)cap->n_positions * sizeof(float));
}

// ----------------------------------------------------------------------
// Env gate.
// ----------------------------------------------------------------------
int sp_quickborn_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("SP_QUICKBORN_ENABLE");
        cached = (e && e[0] == '1') ? 1 : 0;
    }
    return cached;
}

const char *sp_quickborn_url(void) {
    static const char *cached = NULL;
    if (!cached) {
        const char *e = getenv("SP_QUICKBORN_URL");
        cached = (e && e[0]) ? e
                              : "http://127.0.0.1:9988/v1/spec-draft-raw";
    }
    return cached;
}

int sp_quickborn_timeout_ms(void) {
    static int cached = -1;
    if (cached < 0) {
        const char *e = getenv("SP_QUICKBORN_TIMEOUT_MS");
        int v = e ? atoi(e) : 0;
        cached = (v > 0) ? v : 5000;
    }
    return cached;
}
