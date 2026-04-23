// Shannon-Prime Engine — Gated DeltaNet recurrent state cache
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// Companion to KvCache for hybrid architectures (currently qwen35moe).
// Where standard attention layers accumulate per-token K/V slots and
// scale linearly in sequence length, Gated DeltaNet layers maintain
// two FIXED-SIZE recurrent state tensors that are updated in place:
//
//   conv_state  : rolling window of the last (conv_kernel - 1) values
//                 fed through the causal 1D depthwise conv, shape
//                 [conv_kernel - 1, conv_channels, n_seqs]
//
//   ssm_state   : delta-rule recurrent memory, shape
//                 [head_v_dim, head_v_dim, num_v_heads, n_seqs]
//
// Total footprint per GDN layer on Qwen3.6-35B-A3B (single sequence,
// fp16 internal storage — the in-graph tensors stay f32 because the
// CPU kernels require f32; conversion happens at the read/write
// boundary):
//   conv: 3 × 8192 × 2 bytes = 48 KiB
//   ssm:  128 × 128 × 32 × 2 bytes = 1 MiB
// → ~31 MiB across all 30 GDN layers. Immaterial next to weights.
//
// Unlike KvCache the state is bounded and persists across the entire
// decode chain — no max_seq dimension. reset() zeros everything for a
// fresh sequence.

#pragma once

#include <cstddef>
#include <cstdio>
#include <memory>
#include <vector>

namespace sp::engine {

class GdnStateCache {
public:
    // `layer_is_gdn[il]` marks which layers own recurrent state. Layers
    // with `false` allocate zero bytes and read/write is a no-op that
    // returns success — lets the forward builder call uniformly across
    // the n_layer loop without a per-step kind check.
    //
    // The four shape params come from the GGUF's qwen35moe.ssm.* keys:
    //   conv_kernel   = ssm.conv_kernel        (typically 4)
    //   conv_channels = ssm.inner_size + 2 * ssm.group_count * ssm.state_size
    //   head_v_dim    = ssm.inner_size / ssm.time_step_rank
    //   num_v_heads   = ssm.time_step_rank
    //
    // `n_seqs` is the batch dimension (1 for single-sequence decode).
    static std::unique_ptr<GdnStateCache> create(
            const std::vector<bool>& layer_is_gdn,
            int conv_kernel,
            int conv_channels,
            int head_v_dim,
            int num_v_heads,
            int n_seqs = 1);

    ~GdnStateCache();
    GdnStateCache(const GdnStateCache&) = delete;
    GdnStateCache& operator=(const GdnStateCache&) = delete;

    // Read/write the current conv state for `layer` into/out of a host
    // buffer sized `conv_state_floats() * n_seqs`. For non-GDN layers
    // both calls are no-ops and return true — callers don't need to
    // branch on layer kind.
    bool read_conv (int layer, std::vector<float>& buf) const;
    bool write_conv(int layer, const float* src);

    // Same for the ssm_state — buffer size `ssm_state_floats() * n_seqs`.
    bool read_ssm  (int layer, std::vector<float>& buf) const;
    bool write_ssm (int layer, const float* src);

    // Zero all state. Called at the start of a fresh prefill so the
    // recurrence starts from the canonical zero history.
    void reset();

    // --- shape accessors ---
    int n_layer()       const;
    int conv_kernel()   const;
    int conv_channels() const;
    int head_v_dim()    const;
    int num_v_heads()   const;
    int n_seqs()        const;

    // Per-layer state tensor sizes in floats, exclusive of n_seqs
    // (i.e. what you'd pass to ggml_new_tensor for a single sequence).
    int  conv_state_floats() const;  // (conv_kernel - 1) * conv_channels
    int  ssm_state_floats()  const;  // head_v_dim * head_v_dim * num_v_heads

    // True if `layer` was marked as GDN at creation — useful for
    // summary / describe output.
    bool is_gdn_layer(int layer) const;
    int  n_gdn_layers() const;

    // Diagnostics / introspection. Size breakdown written to stderr.
    void print_summary(std::FILE* f) const;

private:
    GdnStateCache();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sp::engine
