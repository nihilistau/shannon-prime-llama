# Changelog

All notable changes to shannon-prime-llama since the first public release.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning tracks llama.cpp's upstream tag plus an `-spN` suffix —
`v2.14.0-sp1` is built on llama.cpp `b8861` with the first Shannon-Prime
patch revision; `-sp2` keeps the same upstream and bumps the SP layer.

## [Unreleased]

Nothing on the road to `v2.14.0-sp3` yet — open backlog items live in
`FUTURE-WORK.md` at the workspace root.

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
