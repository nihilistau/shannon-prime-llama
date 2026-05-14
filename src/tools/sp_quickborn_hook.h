// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

// SP-QuickBorn outbound side-channel for shannon-prime-llama (Phase G).
//
// Direction: the patched llama.dll captures residual stream at configured
// layer indices during its own forward pass, posts them to the lean engine
// at /v1/spec-draft-raw, and gets back N draft tokens to feed into its
// existing speculative-decode loop. LM Studio sees no API change.
//
// What landed in the OLD version of this file (server-side /v1/sp/step +
// /v1/sp/verify endpoints) was discarded — LM Studio's HTTP server is
// closed-source and can't be extended, so the dll has to do the POST
// itself. See SHANNON_PRIME_LLAMA_PATCH.md for the full architectural
// rationale.
//
// Three deliverables here:
//   1. sp_quickborn_capture_* : thread-local capture state set/cleared
//      around a forward pass; per-layer hook copies the residual stream.
//   2. sp_quickborn_request_draft : pack captured states into the binary
//      wire format, POST, parse response, return N draft tokens.
//   3. sp_quickborn_enabled : gate everything on SP_QUICKBORN_ENABLE=1
//      so the patch is a strict no-op when not requested.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Export/import dance: when this header is consumed inside the unit that
// builds INTO llama.dll (sp_quickborn_hook.c, sp_quickborn_http.cpp), the
// CMake target sets SP_QB_INSIDE_DLL so the public functions are
// dllexport. When consumed outside (common/speculative.cpp, which links
// into llama-server.exe), SP_QB_INSIDE_DLL is unset and the functions
// resolve via dllimport from llama.dll. This keeps `g_sp_capture` as a
// single global living inside llama.dll — the previous version relinked
// the static lib into both binaries, so each got its own copy and the
// hook+subclass talked past each other.
#if defined(_WIN32) && defined(SP_QB_INSIDE_DLL)
#  define SP_QB_API __declspec(dllexport)
#elif defined(_WIN32)
#  define SP_QB_API __declspec(dllimport)
#else
#  define SP_QB_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_tensor;

#define SP_QUICKBORN_MAX_CAPTURE_LAYERS 16

// ----------------------------------------------------------------------
// Capture state set per-decode (thread-local). Caller pre-allocates the
// per-layer float buffers. After llama_decode returns, bufs[i] contains
// the captured residual stream for layer_ids[i].
// ----------------------------------------------------------------------
struct sp_quickborn_capture {
    int   layer_ids[SP_QUICKBORN_MAX_CAPTURE_LAYERS];
    int   n_layers;
    int   n_positions;          // typically 1 — the newest position
    int   hidden_size;          // populated by hook on first call
    float *bufs[SP_QUICKBORN_MAX_CAPTURE_LAYERS];
};

SP_QB_API void sp_quickborn_capture_set(struct sp_quickborn_capture *cap);

// The patch's only intrusion into llama-graph.cpp: a single call to this
// after each layer's body. No-op when no capture is active or layer_idx
// isn't in the captured set.
SP_QB_API void sp_quickborn_capture_post_layer(int layer_idx,
                                                const struct ggml_tensor *cur);

// ----------------------------------------------------------------------
// Outbound draft request.
// Wire shape (matches the lean engine's /v1/spec-draft-raw):
//   POST /v1/spec-draft-raw
//   Content-Type: application/octet-stream
//   X-SP-Shape:   "n_layers,seq,hidden"   ← from `cap`
//   X-SP-Dtype:   "fp32"
//   X-SP-Block:   "<block_size>"
//   Body:         n_layers × seq × hidden × sizeof(float), layer-major
// Response:
//   200 + body = block * int32 token IDs.
//
// Returns the number of draft tokens written to `out_tokens` (≤ max_out),
// or a negative value on failure:
//   -1  not enabled
//   -2  no capture state set
//   -3  HTTP transport error
//   -4  HTTP non-200
//   -5  malformed response
// On success, out_tokens[0..n) are the draft IDs.
// ----------------------------------------------------------------------
SP_QB_API int sp_quickborn_request_draft(const struct sp_quickborn_capture *cap,
                                          int   block_size,
                                          int  *out_tokens,
                                          int   max_out);

// ----------------------------------------------------------------------
// Env gate. Returns 1 iff SP_QUICKBORN_ENABLE=1 (cached on first call).
// The URL/timeout are sourced from env too:
//   SP_QUICKBORN_URL       default http://127.0.0.1:9988/v1/spec-draft-raw
//   SP_QUICKBORN_TIMEOUT_MS default 5000
// ----------------------------------------------------------------------
SP_QB_API int sp_quickborn_enabled(void);
SP_QB_API const char *sp_quickborn_url(void);
SP_QB_API int sp_quickborn_timeout_ms(void);

#ifdef __cplusplus
}
#endif
