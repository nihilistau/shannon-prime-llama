// Shannon-Prime Engine — public API
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com

#pragma once

#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>

namespace sp::engine {

struct Config {
    std::string model_path;   // GGUF on disk
    int         n_ctx   = 2048;
    int         n_batch = 512;

    // Shannon-Prime compression switches. One struct, no env-var scavenging —
    // the caller picks the composition explicitly.
    bool        sqfree      = false;   // Enable sqfree + Knight skeleton
    bool        spinor      = false;   // SU(2) sheet bit (requires sqfree)
    bool        mobius      = true;    // Ship-path Möbius reorder
    int         residual_bits = 3;     // Sqfree residual depth
    std::string k_bits_csv  = "5,5,4,3"; // Per-band K bit allocation
    std::string v_bits_csv  = "3";       // Per-band V (default flat)

    // Model-pack preset selection — arch-aware defaults.
    //   ""    / "off"  — use shipping defaults / explicit flags (default)
    //   "auto"         — resolve from model's GGUF arch_name at load time
    //   "<preset>"     — force a specific preset (e.g. "qwen3-moe")
    // Preset overlays apply only when k_bits_csv/v_bits_csv/residual_bits
    // are still at their shipping defaults — any explicit user flag wins.
    std::string model_preset = "";
    // Populated from GGUF general.architecture at model load; used by
    // KvCache::create_gpu when model_preset == "auto".
    std::string arch_name = "";

    // Hierarchical Vilenkin predictor — maximum compression path.
    // Uses Kronecker sub-projection as a small skeleton (~9% of pad_dim)
    // and a calibrated linear map to predict the remaining coefficients.
    // Requires calibration (first prefill). Mutually exclusive with sqfree.
    bool        hierarchical    = false;
    int         hier_level      = 0;       // 0 = auto (second-to-last prime grouping)
    int         hier_res_bits   = 2;       // 1-4 bits for target residuals
    std::string hier_skel_bits  = "5,5";   // Band bits for skeleton quantisation

    // Backend selection.
    enum class Backend { CPU, CUDA, Vulkan };
    Backend     backend = Backend::CPU;
    int         n_gpu_layers = 0;

    // Multi-GPU sharding — distribute transformer layers across GPUs.
    //
    //   n_gpus = 0 → auto-detect all available GPUs (default)
    //   n_gpus = 1 → single GPU (current behaviour, no sharding)
    //   n_gpus > 1 → shard layers across that many GPUs
    //
    // Layer L is assigned to GPU[ L * n_gpus / n_layer ]. Non-layer
    // tensors (tok_embd, output_norm, output) go to GPU 0 when fully
    // offloaded, or stay CPU-mapped under partial offload.
    //
    // The scheduler handles cross-GPU copies transparently — when a
    // tensor produced on GPU i is consumed by an op on GPU j, it gets
    // an automatic copy node inserted.
    int         n_gpus = 0;

    // Positional-encoding mode.
    enum class PeMode { Standard, PrimePe, PrimePeAlibi, AlibiOnly };
    PeMode      pe_mode  = PeMode::Standard;
    float       pe_alpha = 0.17f;
    int         pe_tier  = 0;

    // Cauchy reset system — decode-chain causal stability.
    int         cauchy_mode     = 0;
    int         cauchy_fixed_n  = 512;
    int         cauchy_cooldown = 64;
    int         cauchy_warmup   = 64;
    bool        cauchy_use_ricci = false;
    bool        cauchy_ricci_only = false;
    bool        cauchy_mertens_only = false;
    float       params_b        = 0.0f;

    // Hot/cold tiered storage — GPU VRAM → CPU pinned RAM → disk.
    int         cold_mb      = 0;
    int         evict_keep   = 0;
    bool        enable_cold  = false;

    // Disk serialisation — save/load compressed KV cache state.
    std::string save_cache_path;
    std::string load_cache_path;

    // System 1↔2 switching — entropy-gated dynamic cache routing.
    //
    // When enabled, the engine maintains two caches:
    //   System 1: ship path (fast, moderate compression)
    //   System 2: hier or sqfree path (slower, maximum fidelity)
    //
    // During decode, the output logit entropy after each token determines
    // which cache stores the NEXT token's K/V. High entropy (model is
    // uncertain, distributing probability mass widely) → System 2 for
    // maximum reconstruction fidelity on these "hard" tokens. Low entropy
    // (model is confident) → System 1 for speed.
    //
    // The threshold is in nats (natural log). Typical softmax entropy for
    // an 8B model ranges from ~0.3 (very confident) to ~8 (very uncertain).
    // Default threshold 2.0 routes ~15-25% of tokens to System 2.
    //
    // On read, the DualKvCache merges positions from both caches
    // transparently — the decode graph sees a single unified K/V history.
    bool        system12          = false;
    float       s12_threshold     = 2.0f;  // entropy threshold (nats)
    // System 2 cache type: "hier" (default) or "sqfree"
    std::string s12_sys2          = "hier";
};

// Seed Config fields from environment variables. Called by each CLI verb
// immediately after Config construction, so the precedence ordering stays:
//   Config default → env var → CLI flag.
inline void seed_config_from_env(Config& cfg) {
    if (cfg.model_preset.empty()) {
        if (const char* s = std::getenv("SHANNON_PRIME_MODEL_PRESET")) {
            cfg.model_preset = s;
        }
    }
    if (!cfg.enable_cold) {
        if (const char* s = std::getenv("SP_ENGINE_COLD_MB")) {
            cfg.cold_mb = std::atoi(s);
            cfg.enable_cold = (cfg.cold_mb > 0);
        }
    }
    if (cfg.evict_keep == 0) {
        if (const char* s = std::getenv("SP_ENGINE_EVICT_KEEP")) {
            cfg.evict_keep = std::atoi(s);
        }
    }
    if (cfg.save_cache_path.empty()) {
        if (const char* s = std::getenv("SP_ENGINE_SAVE_CACHE")) {
            cfg.save_cache_path = s;
        }
    }
    if (cfg.load_cache_path.empty()) {
        if (const char* s = std::getenv("SP_ENGINE_LOAD_CACHE")) {
            cfg.load_cache_path = s;
        }
    }
    if (!cfg.system12) {
        if (const char* s = std::getenv("SP_ENGINE_SYSTEM12")) {
            cfg.system12 = (std::atoi(s) != 0);
        }
    }
    if (cfg.s12_threshold == 2.0f) {
        if (const char* s = std::getenv("SP_ENGINE_S12_THRESHOLD")) {
            cfg.s12_threshold = (float)std::atof(s);
        }
    }
    if (cfg.s12_sys2.empty() || cfg.s12_sys2 == "hier") {
        if (const char* s = std::getenv("SP_ENGINE_S12_SYS2")) {
            cfg.s12_sys2 = s;
        }
    }
    if (cfg.n_gpus == 0) {
        if (const char* s = std::getenv("SP_ENGINE_N_GPUS")) {
            cfg.n_gpus = std::atoi(s);
        }
    }
}

class Engine {
public:
    Engine();
    ~Engine();

    // Load model + build compute graph. Returns 0 on success.
    int load(const Config& cfg);

    // Run perplexity over a tokenised input file. Returns PPL on success,
    // negative on error. Writes per-chunk values to stderr when verbose.
    float perplexity(const std::string& wikitext_path,
                     int n_chunks, bool verbose = false);

    // Greedy generate: tokenise prompt, prefill via ForwardContext with a
    // KvCache bound (compression mode controlled by cfg), then argmax-decode
    // n_predict tokens (or until EOS). Sampling temperature is zero; richer
    // sampling hooks would layer on top of ForwardContext::decode directly.
    int generate(const std::string& prompt, int n_predict,
                 std::string& out);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace sp::engine
