// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

// SP-QuickBorn outbound side-channel — capture + env gate.
// HTTP POST itself lives in sp_quickborn_http.cpp (uses vendored cpp-httplib).

#include "sp_quickborn_hook.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ----------------------------------------------------------------------
// Per-decode capture state.
//
// NOTE: this was thread-local originally, but llama.cpp's server runs
// the spec-decode subclass init on the slot launcher thread while the
// target's graph-build (where our layer hook fires) runs on a different
// worker thread. Thread-local broke the wire-up — the hook saw NULL.
// Process-global is correct for single-slot serving; multi-slot will
// race (TODO: keyed on llama_context*).
// ----------------------------------------------------------------------
static struct sp_quickborn_capture *g_sp_capture = NULL;

void sp_quickborn_capture_set(struct sp_quickborn_capture *cap) {
    g_sp_capture = cap;
}

void sp_quickborn_capture_post_layer(int layer_idx,
                                     const struct ggml_tensor *cur) {
    if (!sp_quickborn_enabled()) return;

    static int announced = 0;
    if (!announced) {
        announced = 1;
        fprintf(stderr,
            "[sp-quickborn] hook live - first layer=%d url=%s timeout=%dms\n",
            layer_idx, sp_quickborn_url(), sp_quickborn_timeout_ms());
        fflush(stderr);
    }

    struct sp_quickborn_capture *cap = g_sp_capture;
    if (!cap || !cur) {
        // Diagnostic: cap-null is the bug we just fixed for thread-local;
        // print the first few times so we can see if the global reaches.
        static int cap_null_count = 0;
        if (cap_null_count < 3) {
            cap_null_count++;
            fprintf(stderr, "[sp-quickborn] hook called with cap=%p cur=%p (layer=%d)\n",
                (void*)cap, (void*)cur, layer_idx);
            fflush(stderr);
        }
        return;
    }

    int slot = -1;
    for (int i = 0; i < cap->n_layers; ++i) {
        if (cap->layer_ids[i] == layer_idx) { slot = i; break; }
    }

    static int seen_layers = 0;
    if (seen_layers < 6) {
        seen_layers++;
        fprintf(stderr, "[sp-quickborn] capture: layer=%d slot=%d cap_n_layers=%d\n",
            layer_idx, slot, cap->n_layers);
        fflush(stderr);
    }

    if (slot < 0) return;

    const int hidden = (int)cur->ne[0];
    const int n_tok  = (int)cur->ne[1];
    if (cap->hidden_size == 0) {
        cap->hidden_size = hidden;
        fprintf(stderr, "[sp-quickborn] hidden_size set to %d (layer %d, n_tok=%d)\n",
            hidden, layer_idx, n_tok);
        fflush(stderr);
    }
    if (n_tok < cap->n_positions || !cap->bufs[slot]) return;

    // Tensor data may live on a CUDA device buffer; reading it as host
    // memory would access-violate. Use the backend API to copy the tail
    // n_positions × hidden floats into our host-side scratch.
    const size_t bytes = (size_t)hidden * (size_t)cap->n_positions * sizeof(float);
    const size_t offs  = (size_t)(n_tok - cap->n_positions) * (size_t)hidden * sizeof(float);
    if (cur->buffer && !ggml_backend_buffer_is_host(cur->buffer)) {
        ggml_backend_tensor_get(cur, cap->bufs[slot], offs, bytes);
    } else if (cur->data) {
        memcpy(cap->bufs[slot], (const char *)cur->data + offs, bytes);
    }
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
