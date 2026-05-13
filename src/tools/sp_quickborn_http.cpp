// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

// SP-QuickBorn outbound HTTP — wraps vendored cpp-httplib so the C hook
// can stay C. Header parsing is hand-rolled (4 fields, all known).
//
// Build expectations:
//   - cpp-httplib is HEADER-ONLY (vendored at
//     llama_cpp/vendor/cpp-httplib/httplib.h).
//   - This TU compiles in C++17 and exposes one extern "C" symbol:
//     sp_quickborn_request_draft(...). Link this together with
//     sp_quickborn_hook.c into the patched llama target.

#include "sp_quickborn_hook.h"

// cpp-httplib pulled in implementation here (its header is HTTPLIB_HEADER_ONLY
// by default — just including it is enough; no httplib.cpp link needed).
#include "httplib.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

namespace {

// Parse "scheme://host:port/path" into (host_with_scheme, path).
// cpp-httplib's Client takes "scheme://host[:port]" and POST takes path.
struct ParsedUrl {
    std::string base;     // e.g. "http://127.0.0.1:9988"
    std::string path;     // e.g. "/v1/spec-draft-raw"
    bool        ok;
};
ParsedUrl parse_url(const char *url) {
    ParsedUrl r{};
    if (!url) return r;
    std::string s(url);
    auto scheme_end = s.find("://");
    if (scheme_end == std::string::npos) return r;
    auto path_start = s.find('/', scheme_end + 3);
    if (path_start == std::string::npos) {
        r.base = s; r.path = "/"; r.ok = true; return r;
    }
    r.base = s.substr(0, path_start);
    r.path = s.substr(path_start);
    r.ok   = true;
    return r;
}

}  // namespace

extern "C" int sp_quickborn_request_draft(
        const struct sp_quickborn_capture *cap,
        int   block_size,
        int  *out_tokens,
        int   max_out)
{
    if (!sp_quickborn_enabled())            return -1;
    if (!cap || cap->n_layers <= 0
            || cap->hidden_size <= 0
            || cap->n_positions <= 0
            || !out_tokens || max_out <= 0) return -2;

    // Pack the body — layer-major, then position-major, then hidden.
    const size_t per_layer = (size_t)cap->n_positions * (size_t)cap->hidden_size;
    const size_t total     = (size_t)cap->n_layers * per_layer;
    std::string body;
    body.resize(total * sizeof(float));
    char *dst = body.data();
    for (int i = 0; i < cap->n_layers; ++i) {
        if (!cap->bufs[i]) return -2;
        std::memcpy(dst, cap->bufs[i], per_layer * sizeof(float));
        dst += per_layer * sizeof(float);
    }

    // URL parsing.
    ParsedUrl u = parse_url(sp_quickborn_url());
    if (!u.ok) return -3;

    httplib::Client cli(u.base);
    cli.set_connection_timeout(0, sp_quickborn_timeout_ms() * 1000);  // sec, usec
    cli.set_read_timeout      (0, sp_quickborn_timeout_ms() * 1000);
    cli.set_write_timeout     (0, sp_quickborn_timeout_ms() * 1000);

    char shape_hdr[64];
    std::snprintf(shape_hdr, sizeof(shape_hdr), "%d,%d,%d",
                  cap->n_layers, cap->n_positions, cap->hidden_size);
    char block_hdr[16];
    std::snprintf(block_hdr, sizeof(block_hdr), "%d", block_size);

    httplib::Headers headers = {
        {"X-SP-Shape",  shape_hdr},
        {"X-SP-Dtype",  "fp32"},
        {"X-SP-Block",  block_hdr},
    };

    auto res = cli.Post(u.path, headers, body, "application/octet-stream");
    if (!res) {
        std::fprintf(stderr, "[sp-quickborn] http error: %s\n",
                     httplib::to_string(res.error()).c_str());
        return -3;
    }
    if (res->status != 200) {
        std::fprintf(stderr, "[sp-quickborn] http %d: %s\n",
                     res->status,
                     res->get_header_value("X-SP-Error").c_str());
        return -4;
    }

    const std::string &out = res->body;
    if (out.size() % sizeof(int32_t) != 0) return -5;
    int n_tok = (int)(out.size() / sizeof(int32_t));
    if (n_tok > max_out) n_tok = max_out;
    std::memcpy(out_tokens, out.data(), (size_t)n_tok * sizeof(int32_t));
    return n_tok;
}
