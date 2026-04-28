# Changelog

All notable changes to shannon-prime-llama since the first public release.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning tracks llama.cpp's upstream tag plus an `-spN` suffix —
`v2.14.0-sp1` is built on llama.cpp `b8861` with the first Shannon-Prime
patch revision; `-sp2` keeps the same upstream and bumps the SP layer.

## [Unreleased]

Heading toward `v2.14.0-sp3`. Open backlog items live in
`FUTURE-WORK.md` at the workspace root, and design specs for in-flight
work live alongside the math core (`docs/PHASE-3-ATTENTION-DESIGN.md`).

### In progress

- **Phase 3: attention short-circuit** — wire the partial-band-read
  primitive (sp2) into llama.cpp's attention path. Entropy-gated band
  count, 3-tier extension of System 1/2 routing. Design doc shipped;
  implementation deferred to sp3.
- **Hexagon backend (Snapdragon NPU)** — V69 HVX kernels for the SP
  math primitives so the S22 Ultra and other 8 Gen 1 phones get NPU
  acceleration alongside CPU/GPU/CUDA. Scaffolding work; depends on
  Qualcomm Hexagon SDK 5.x.
- **sqfree / hier migration to v3 disk format** — currently those
  paths still write v2 (per-vec interleaved bands). Migrating them
  unlocks partial-load IO wins for sqfree+spinor and hierarchical
  modes too.

## [v2.14.0-sp2] — 2026-04-29

### Fixed

- **Speculative decoding correctness (`-md` flag).** The `v2.14.0-sp1`
  patch maintained a single global SP context per process. With
  `-md` (target + draft), only the first-loaded model received SP
  compression; the second either bypassed SP or — if its head
  dimensions matched the target's — silently corrupted the target's
  cache. Replaced the global with a
  `std::unordered_map<const llama_model*, sp_per_model>` so each
  context has its own SP state keyed on the model pointer.
- **Disk-cache hash-guard.** `sp_read_cache_header` was warn-only on
  hash mismatch, which let users load the wrong model's cache and
  see silent garbage output. Now strict by default — mismatch
  returns -1 and aborts the load. The Archimedes-style warn-only
  behaviour is preserved as an opt-in escape hatch via
  `SP_DISK_HASH_STRICT=0`.

### Added

- **Disk-tier scaffold (phases 1+2 of progressive band loading).**
  - Phase 1 (math core): `sp_band_dequantize_partial(in, out, bc, max_bands)`
    reconstructs only the first N bands of a banded vector, zeroing the
    rest. Energy concentration in the early bands means band 0 alone
    gets ~30% reconstruction correlation, bands 0+1 get ~85% on smooth
    signals.
  - Phase 2 (disk format): bumped cache disk format v2 → v3 with
    band-major layout. New `sp_shadow_cache_load_partial(prefix, hash,
    max_bands)` reads only the first N bands' contiguous regions from
    disk. Engine wrapper `KvCache::load_from_disk_partial(...)`. Format
    is backward-compatible — v2 files still readable via fallback path.
  - Architecture: `DISK-TIER-ARCHITECTURE.md` covers the three-phase
    plan, energy concentration, Granite/Sand/Jazz tier mapping, and
    the S22 Ultra phone deployment walkthrough including the Hexagon
    NPU backend story.
  - Bench: `bench_disk_partial` measures the IO win — 1.80× on K-side
    band-0 reads on commodity NVMe (V cache stays full because it's
    single-band by design).
- **Auto-apply K-quant getrows patch.** `cmake -S . -B build` now
  auto-applies `patches/ggml-cuda-getrows-kquant.patch` to `ggml/`
  on first configure, idempotent via SP-marker comment probe. Opt-out
  via `-DSP_AUTO_PATCH=OFF`. Eliminates the #1 setup-friction point
  on Gemma-3 / Phi-4 / Phi-3.1-mini-128k loads.
- **FP8 advisory wiring.** `SHANNON_PRIME_FP8=1` env var (and
  `SHANNON_PRIME_DRAFT_FP8`) parsed in the bridge, populates
  `sp_config_t::use_fp8`. CPU/Adreno bridge logs a warning and falls
  back to int (no fp8 path on those backends today). The engine's
  CUDA backend honours the flag when `SP_ENGINE_FP8=ON` was set at
  compile time.
- **Model-pack `suggested_draft` field.** Each per-arch preset now
  carries a recommended draft model hint and expected-acceptance
  number for speculative decoding. Populated for all 7 shipping
  presets (Qwen / Llama / Mistral / Phi / Gemma).
- **Auto-detect-draft hint in bridge.** When `SHANNON_PRIME_VERBOSE=1`
  and the caller sets `arch_name` on `sp_llama_params_t`, the bridge
  resolves the matching preset and logs a one-line draft suggestion.
- **Speculative-decoding bench harness.** `scripts/bench-spec-decode.ps1`
  runs llama-cli through 5 SP configurations and outputs CSV with
  tok/sec, acceptance rate, and output edit-distance vs vanilla.
- **Per-model SP context map.** Foundation for `-md` correctness
  above. New `llama_sp_free(model)` /
  `llama_sp_is_enabled(model)` /
  `llama_sp_post_compute(model, mctx, ubatch)` signatures (plus a
  `llama_sp_free_all()` for shutdown). Existing single-model use is
  unaffected — every callsite in llama.cpp's injection points
  already had `&model` available locally.
- **Role-aware bridge initialisation.** `sp_llama_init_with_role(params, role)`
  with `SP_LLAMA_ROLE_{DEFAULT, TARGET, DRAFT}`. When
  `SHANNON_PRIME_SPEC=1`, the first init becomes target and the
  second becomes draft; draft init reads `SHANNON_PRIME_DRAFT_X`
  env vars first and falls back to `SHANNON_PRIME_X`. Existing
  `sp_llama_init` is now a thin `ROLE_DEFAULT` wrapper for
  back-compat.
- **`SHANNON_PRIME_DRAFT_PRESET` shortcut.** Three values:
  `aggressive` (K=2,1 V=1, ~10× compression), `ternary` (K=2,2 V=2,
  ~7×), `ship` (no-op). Useful for speculative deployments that want
  the draft compressed harder without per-band knob fiddling.
- **Ternary noise-tail (5/5/4/1.58) primitive.**
  `sp_band_config_init_ext()` accepts a `uint32_t ternary_band_mask`;
  bit b set ⇒ band b is quantised to {-1, 0, +1} at 2 bpp regardless
  of `band_bits[b]`. Bridge env vars
  `SHANNON_PRIME_K_TERNARY_BANDS=3` /
  `SHANNON_PRIME_V_TERNARY_BANDS=` populate the mask through
  `sp_config_t::{k,v}_ternary_mask`. `DRAFT_K_TERNARY_BANDS` works
  for differential draft compression on top of `SPEC=1`.
- **Patch-drift CI workflow** (`.github/workflows/patch-drift.yml`).
  Runs on every push and PR; validates that
  `patches/llama-cpp-b8861-full-engine.patch` still applies cleanly
  to a fresh `b8861` checkout. Three checks: `git apply --check`,
  `--whitespace=error` (CRLF guard), and a surface-verification step
  that confirms the expected files were touched. Fast (<2 min),
  separate from the heavy build matrix.
- **`SPECULATIVE-DECODING.md`** integration guide
  (`lib/shannon-prime/docs/`). Recommended draft/target pairs
  across Qwen / Llama / Mistral / Phi / Gemma with observed
  acceptance rates, suggested SP settings, the differential-
  compression workflow, the upgrade caveat for users on `sp1`.
- **`tests/test_disk_cache_roundtrip.cpp`** in shannon-prime-engine.
  Three properties: round-trip exactness, strict hash mismatch
  rejection, `SP_DISK_HASH_STRICT=0` escape hatch.
- **PrimePE unit tests** in `lib/shannon-prime/tests/test_core.c`.
  Seven properties covering bad inputs, alpha=0 identity, finite
  positive output, determinism, low-index factor bounds, alpha
  effect, and freq_base robustness.

### Changed

- **CUDA toolkit bumped** from 12.1.1 to 12.4.1 in CI release
  workflow. CUDA 12.1 rejected the MSVC shipping on the
  windows-2022 GitHub runners; 12.4 supports it natively. Binaries
  are forward-compatible with users on CUDA 12.1+ drivers via
  CUDA's minor-version compat guarantee. Artifact filenames
  updated `cuda12.1` → `cuda12.4`.
- **Artifact naming gained `-sp-` infix.** Lets users keep these
  bundles side-by-side with vanilla llama.cpp LM Studio runtimes
  without filename collisions. New names:
  `lmstudio-runtime-sp-windows-cuda12.4-*.zip`,
  `llama-cli-sp-{windows,linux}-{cuda,vulkan}-*`,
  `shannon-prime-llama-cpp-b8861-patched-source.tar.gz`.

### Notes for sp1 users

If you're on `v2.14.0-sp1` and using single-model inference (no
`-md`), there's no urgency — your config keeps working as is.
**If you're using `-md` for speculative decoding, upgrade to
`v2.14.0-sp2`.** Your sp1 binaries silently produced wrong results
on the draft side; sp2 fixes it. See `BUILD-AND-DISTRIBUTE.md`
"Release notes & known limitations" for the full sp1 caveat.

## [v2.14.0-sp1] — 2026-04-28

First public release.

### Added

- **GitHub Actions release pipeline** (`.github/workflows/release.yml`).
  Four-target build matrix (Windows × Linux × CUDA × Vulkan) plus a
  source-archive job. Tag pushes matching `v*-sp*` mint a GitHub
  Release with all artifacts attached.
- **Pre-built binaries shipped:**
  - `lmstudio-runtime-sp-windows-cuda12.4-<sha>.zip` — LM Studio
    drop-in DLLs.
  - `llama-cli-sp-{windows,linux}-{cuda,vulkan}-<sha>.{zip,tar.gz}` —
    Standalone llama-cli binaries.
  - `shannon-prime-llama-cpp-b8861-patched-source.tar.gz` — Source
    archive for users on unsupported GPU archs.
- **`BUILD-AND-DISTRIBUTE.md`** — install instructions, source-build
  recipe, compute-capability flags, license matrix.

### Known limitations

- Speculative decoding (`-md`) silently corrupts the draft KV cache
  due to a single-global SP context (see sp2 fix). Single-model use
  is unaffected.
