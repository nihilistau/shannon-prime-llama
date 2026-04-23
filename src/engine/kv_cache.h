// Shannon-Prime Engine — compressed KV cache (public API)
// Copyright (C) 2026 Ray Daniels. All Rights Reserved.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPLv3).
// Commercial license available — contact raydaniels@gmail.com
//
// Wraps the C-level sp_shadow_cache_t (ship path) or sp_sqfree_cache_t
// (aggressive path) behind a single typed C++ interface. The cache owns
// the per-(layer, head) compressed storage and the scratch buffers; the
// engine writes raw fp32 K/V vectors in and reads reconstructed fp32
// vectors out. Compression happens on the write path by construction.
//
// Stage 4: the cache is a passive store. Stage 5 will swap the inline
// K/V projection in build_block for read-from-cache so decode can run
// over a real compressed history.

#pragma once

#include "engine.h"

#include <memory>
#include <string>
#include <vector>

namespace sp::engine {

class KvCache {
public:
    // Allocate a cache sized for `n_layer × n_head_kv × max_seq` slots.
    // `cfg.sqfree`, `cfg.spinor`, `cfg.mobius`, `cfg.k_bits_csv`,
    // `cfg.v_bits_csv`, `cfg.residual_bits` select the compression.
    // Returns nullptr on init failure (bad bit allocation, OOM, etc.).
    static std::unique_ptr<KvCache> create(int n_layer, int n_head_kv,
                                           int head_dim, int max_seq,
                                           const Config& cfg);

    // GPU-resident variant (ship path only, MVP). Compressed K/V blocks
    // live in VRAM; compress / decompress run as CUDA kernels — no host
    // round-trip on read/write. `stream` is a cudaStream_t; pass nullptr
    // for default. Returns nullptr if built without SP_ENGINE_WITH_CUDA
    // or if cfg selects sqfree / hierarchical (not yet supported on GPU).
    static std::unique_ptr<KvCache> create_gpu(int n_layer, int n_head_kv,
                                                int head_dim, int max_seq,
                                                const Config& cfg,
                                                void* stream);

    ~KvCache();
    KvCache(const KvCache&) = delete;
    KvCache& operator=(const KvCache&) = delete;

    // Write a contiguous batch of n_tokens K and V vectors at sequence
    // positions [pos_offset, pos_offset + n_tokens) for the given layer.
    //
    // Layout (matches what ggml_backend_tensor_get returns for a
    // [head_dim, n_head_kv, n_tokens] tensor):
    //   K_flat[(q * n_head_kv + h) * head_dim + d]
    //
    // Returns false if positions overflow max_seq.
    bool write(int layer, int pos_offset, int n_tokens,
               const float* K_flat, const float* V_flat);

    // Read positions [0, kv_len) for the layer back into K_out / V_out
    // using the same layout. Buffers are resized as needed.
    bool read(int layer, int kv_len,
              std::vector<float>& K_out,
              std::vector<float>& V_out) const;

    // GPU-native write: d_K_flat / d_V_flat are DEVICE pointers in the
    // same [head_dim, n_head_kv, n_tokens] layout as write(). Compress
    // kernels run on GPU; no host round-trip. Only valid on caches
    // created via create_gpu(). Returns false otherwise.
    bool write_gpu(int layer, int pos_offset, int n_tokens,
                   const float* d_K_flat, const float* d_V_flat);

    // GPU-native read: d_K_out / d_V_out are DEVICE pointers with the
    // same [head_dim, n_head_kv, kv_len] layout as read(). Decompress
    // kernels run on GPU; caller-provided device buffers are written
    // in place. Only valid on caches created via create_gpu().
    bool read_gpu(int layer, int kv_len,
                  float* d_K_out, float* d_V_out) const;

    // Query whether this cache is GPU-resident.
    bool is_gpu() const;

    // --- adaptive calibration ---
    //
    // Calibration feeds raw KV vectors through the spectral transform
    // and accumulates per-coefficient variance. calibrate_end() rebuilds
    // the internal masks so write/read use variance-ranked ordering
    // (sqfree: Knight mask with L/2 skeleton; ship: variance-ranked
    // reorder into banded quantizer).
    //
    // Feed ALL calibration vectors between begin/end. Typical use: feed
    // K vectors from the first forward pass (warmup), then end.
    //
    // For hierarchical mode, calibrate_feed takes a slot index (layer * H + head)
    // so each predictor is trained on its own head's data. The single-arg
    // overload feeds ALL slots (used by sqfree/shadow which have shared masks).
    bool calibrate_begin();
    void calibrate_feed(const float* vec);
    void calibrate_feed(int slot, const float* vec);  // hierarchical per-slot
    bool calibrate_end();
    // Sticky-EMA variant for hierarchical mode only. keep_frac in [0,1];
    // 0 is equivalent to calibrate_end() (full replacement). On non-hier
    // caches this falls through to calibrate_end() — ship / sqfree masks
    // don't need blended updates.
    bool calibrate_end_ema(float keep_frac);
    bool is_calibrated() const;
    bool is_hierarchical() const;

    // --- hot/cold offload (tiered GPU ↔ CPU ↔ disk) ---
    //
    // Enable cold storage for GPU-resident caches. Allocates `cold_mb`
    // megabytes of pinned CPU RAM per K and V per layer. When the cache
    // grows past `evict_keep` positions, the oldest GPU positions are
    // copied to CPU (and optionally zeroed on GPU to reclaim VRAM).
    //
    // cold_mb=0 means unlimited (allocate enough for max_seq).
    // evict_keep=0 means no GPU eviction — just mirror to CPU.
    bool enable_cold_storage(int cold_mb = 0, int evict_keep = 0);

    // Writeback new positions to cold storage. Call after write/write_gpu.
    // Only copies positions that haven't been written back yet.
    bool cold_writeback(int current_pos);

    // Restore cold storage back to GPU. Used after disk load or eviction
    // recovery. Returns the number of positions restored, or -1 on error.
    int  cold_restore(int n_pos);

    // Query cold storage state.
    bool has_cold_storage() const;

    // --- disk serialisation (VHT2 v2 binary format) ---
    //
    // Save compressed cache contents [0, n_pos) to disk. Writes per-layer
    // files {prefix}.L{n}.bin with a 64-byte VHT2 header. For hierarchical
    // caches the predictor W matrices are saved in {prefix}.hier_w.bin.
    // `model_hash` is an FNV-1a hash of the model path — used to detect
    // loads against a different model. Returns 0 on success, -1 on error.
    int save_to_disk(const std::string& prefix, int n_pos,
                     uint64_t model_hash) const;

    // Load cache state from disk. Validates the VHT2 header magic and
    // model hash. Returns the number of positions loaded (the n_pos
    // stored in the header), or -1 on error. The cache must already be
    // created with matching dimensions.
    int load_from_disk(const std::string& prefix, uint64_t expected_hash);

    // --- diagnostics / introspection ---
    int  n_layer()           const;
    int  n_head_kv()         const;
    int  head_dim()          const;
    int  max_seq()           const;
    bool is_sqfree()         const;
    float compression_ratio() const;
    std::string describe()    const;

    // --- Cauchy reset system (decode-chain causal stability) ---
    //
    // Layers:
    //   Layer 1: Zeta Schedule  (pre-computed at init_cauchy)
    //   Layer 2: Mertens Oracle (proactive, arithmetic per-position)
    //   Layer 3: Ricci Sentinel (reactive drift, opt-in — measured
    //            contribution is 0 incremental PPL on Qwen3-8B-Q8)
    //
    // init_cauchy() sets up the stack. cauchy_check(pos) is called per decode
    // step; it returns 0 = no reset, 1 = full reset needed, 2 = partial reset
    // OK (hierarchical only). After performing a reset, call
    // cauchy_record_reset(pos) to update the scheduler state.
    //
    // mode: 0=off, 1=fixed-N (reset every fixed_n tokens), 2=dynamic
    // (Mertens schedule). use_ricci: add the reactive drift sentinel
    // alongside. params_b sizes the Ricci threshold (params_b^0.45); only
    // relevant when use_ricci=true.
    //
    // Default use_ricci=false because the measured contribution of Ricci
    // is 0 incremental PPL over Mertens-only on Qwen3-8B-Q8 ctx=1024
    // (full system 11.92, Mertens-only 11.92, Ricci-only 12.02). Keep
    // Ricci as an opt-in for research / drift diagnostics.
    bool init_cauchy(int mode, int fixed_n, float params_b,
                     bool use_ricci = false);

    // Check whether a reset should fire at this decode position.
    int  cauchy_check(int pos);

    // Override the cooldown (minimum positions between resets). Lower
    // values let the controller fire more often — useful with partial
    // reset, risky with full reset. Default is 64.
    void cauchy_set_cooldown(int n);

    // Ablation hook: free the Mertens oracle and null its pointer in
    // the controller, so mode 2 behaves as Ricci-only. Call after
    // init_cauchy. No-op if Mertens was never allocated.
    void cauchy_disable_mertens();

    // Symmetric ablation: free the Ricci sentinel so mode 2 behaves
    // as Mertens-only. Call after init_cauchy.
    void cauchy_disable_ricci();

    // Manual feed path for the Ricci sentinel (when the caller wants to
    // feed a VHT2-domain K vector from some place other than the normal
    // write() call — e.g. per-token during decode without going through
    // the cache). Safe to call when Cauchy is off (no-op).
    void ricci_feed(const float* vht2_coeffs, int hd);

    // Record that a reset was performed at `pos`.
    void cauchy_record_reset(int pos);

    // Current Ricci drift |1 - p3_ema|; 0 if not initialized.
    double ricci_drift() const;

    // Print Cauchy system stats to stderr.
    void cauchy_print_stats() const;

private:
    KvCache();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ── System 1↔2 dual cache ─────────────────────────────────────────
//
// DualKvCache wraps two KvCache instances: a ship-path "System 1"
// cache (fast, moderate fidelity) and a hier or sqfree "System 2"
// cache (slower, maximum fidelity). The engine's decode loop calls
// route_position() with the output-logit entropy after each token;
// high-entropy positions are stored in System 2, low-entropy in
// System 1.
//
// read_merged() produces a single contiguous K/V buffer for the
// decode graph by reading both caches and interleaving positions
// according to the routing table.
//
// Calibration, Cauchy, disk I/O, and cold storage are delegated to
// whichever cache owns the relevant position (Cauchy uses the System 1
// cache; calibration feeds both).

class DualKvCache {
public:
    // Create both caches. sys2_type is "hier" or "sqfree". The System 1
    // cache is always ship-path. Returns nullptr on failure. The Config's
    // sqfree/hierarchical flags are overridden internally — the caller
    // should set them to false (DualKvCache manages its own composition).
    static std::unique_ptr<DualKvCache> create(int n_layer, int n_head_kv,
                                                int head_dim, int max_seq,
                                                const Config& cfg,
                                                const std::string& sys2_type,
                                                float entropy_threshold);

    // GPU-resident variant (System 1 = GPU ship, System 2 = GPU hier).
    static std::unique_ptr<DualKvCache> create_gpu(int n_layer, int n_head_kv,
                                                    int head_dim, int max_seq,
                                                    const Config& cfg,
                                                    const std::string& sys2_type,
                                                    float entropy_threshold,
                                                    void* stream);

    ~DualKvCache();

    // Route a position to System 1 or System 2 based on entropy. Call
    // BEFORE writing the position. Returns 1 if routed to System 2,
    // 0 if System 1.
    int route_position(int pos, float entropy);

    // Write to whichever cache pos is routed to. The layer's full
    // n_tokens batch is split internally per-position. For single-token
    // decode (n_tokens=1), the routing of pos_offset is used directly.
    bool write(int layer, int pos_offset, int n_tokens,
               const float* K_flat, const float* V_flat);

    // GPU-native write.
    bool write_gpu(int layer, int pos_offset, int n_tokens,
                   const float* d_K_flat, const float* d_V_flat);

    // Read merged K/V for [0, kv_len) into contiguous host buffers.
    // Positions come from whichever cache they were routed to.
    bool read_merged(int layer, int kv_len,
                     std::vector<float>& K_out,
                     std::vector<float>& V_out) const;

    // GPU-native merged read.
    bool read_merged_gpu(int layer, int kv_len,
                         float* d_K_out, float* d_V_out) const;

    // Access underlying caches for calibration, Cauchy, etc.
    KvCache* sys1()       { return sys1_.get(); }
    KvCache* sys2()       { return sys2_.get(); }
    const KvCache* sys1() const { return sys1_.get(); }
    const KvCache* sys2() const { return sys2_.get(); }

    // Query which system owns a position. Returns 1 (sys1) or 2 (sys2).
    int owner_of(int pos) const;

    // Stats: how many positions routed to each system.
    int sys1_count() const;
    int sys2_count() const;

    // Diagnostics.
    int  n_layer()      const;
    int  n_head_kv()    const;
    int  head_dim()     const;
    int  max_seq()      const;
    bool is_gpu()       const;
    float threshold()   const { return threshold_; }
    std::string describe() const;

private:
    DualKvCache() = default;
    std::unique_ptr<KvCache> sys1_;   // ship path (System 1)
    std::unique_ptr<KvCache> sys2_;   // hier or sqfree (System 2)
    float threshold_ = 2.0f;
    // Per-position routing: false = System 1, true = System 2.
    // Index is sequence position. Sized to max_seq at creation.
    std::vector<bool> route_;
    int max_seq_ = 0;
    int n_layer_ = 0;
    int n_head_kv_ = 0;
    int head_dim_ = 0;
};

} // namespace sp::engine
