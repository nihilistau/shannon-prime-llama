# Building and Distributing Shannon-Prime for llama.cpp / LM Studio

This document covers two audiences:

1. **Users** who want a working LM Studio runtime or a llama-cli binary without setting up a build toolchain. Skip to "Pre-built binaries" below.
2. **Builders** who want to compile from source — for a GPU compute capability the binaries don't cover, for a different llama.cpp version, or for distribution under a custom license. The "Building from source" section is for you.

A short licensing note at the end clarifies what can and can't be redistributed.

---

## Release notes & known limitations

### v2.14.0-sp1 (current release)

First public release. Working: single-model llama-cli + LM Studio runtime, all four backends (Windows/Linux × CUDA/Vulkan), all 24+ supported architectures, hierarchical/sqfree compression paths, multi-GPU sharding, hot/cold tiered KV.

**Known limitation — speculative decoding (`-md`):** the patch in this release uses a single global SP context per process, so when llama.cpp loads two models (target + draft under `-md`), only the first-loaded model receives Shannon-Prime compression. The draft either bypasses SP entirely or — if its head dimensions match the target's by coincidence — corrupts the target's compressed cache. **Do not rely on speculative decoding numbers under v2.14.0-sp1.** The fix is on the `feat/per-model-sp-context` branch and will ship in v2.14.0-sp2.

If you specifically need speculative decoding today, build from source on `feat/per-model-sp-context`:

```bash
git clone --branch feat/per-model-sp-context --recursive \
    https://github.com/nihilistau/shannon-prime-llama.git
# then follow the Building from source recipe below
```

The architecture rewrite is documented in `lib/shannon-prime/docs/SPECULATIVE-DECODING.md`. Single-model use is unaffected.

---

## Pre-built binaries

GitHub Releases on this repository ship four artifact families per release tag (`v2.14.0-sp1`, `v2.14.0-sp2`, etc.):

| Artifact | Target | Use case |
|---|---|---|
| `lmstudio-runtime-sp-windows-cuda12.4-*.zip` | LM Studio drop-in DLLs | The button-click install for LM Studio users on Windows + CUDA. |
| `llama-cli-sp-windows-{cuda,vulkan}-*.zip` | Standalone Windows binaries | Run llama.cpp directly without LM Studio. |
| `llama-cli-sp-linux-{cuda,vulkan}-*.tar.gz` | Standalone Linux binaries | Same on Linux. |
| `shannon-prime-llama-cpp-b8861-patched-source.tar.gz` | Source archive | Build for a specific GPU arch or distribute as a derivative work. |

The `-sp-` infix is intentional — it lets you keep these bundles side-by-side with vanilla llama.cpp LM Studio runtimes without filename collisions.

### Installing the LM Studio runtime (Windows + CUDA)

1. Download `lmstudio-runtime-sp-windows-cuda12.4-<sha>.zip` from the latest release.
2. Extract the archive into:

       %LOCALAPPDATA%\LM-Studio\extensions\backends\llama.cpp-sp-cuda\

3. Restart LM Studio.
4. In the runtime picker (Settings → Runtime → llama.cpp), select `llama.cpp-sp-cuda`.
5. Reload your model. The console will show `[shannon-prime] kv-cache: VHT2 ship 5/5/4/3` (or a sqfree variant if configured) — that means SP compression is live.

### CUDA compute capabilities included

The shipped CUDA binaries are compiled for `sm_75;sm_86;sm_89` — covers RTX 20-series (2060/2070/2080), 30-series (3050–3090), and 40-series (4060–4090). If you have an A100 (sm_80), H100 (sm_90), or older hardware (sm_61, GTX 10-series), the shipped binaries won't load and you need to build from source with the right `-DCMAKE_CUDA_ARCHITECTURES=...` flag.

The Vulkan binaries don't have this restriction — they work on any GPU with Vulkan 1.3 support, including AMD, Intel UHD, and older NVIDIA cards.

---

## Building from source

### Quick recipe (Windows, CUDA, Ninja)

```bash
# 1. Clone shannon-prime-llama (with submodule)
git clone --recursive https://github.com/nihilistau/shannon-prime-llama.git
cd shannon-prime-llama

# 2. Clone llama.cpp at the patched tag
git clone --branch b8861 --depth 1 https://github.com/ggml-org/llama.cpp llama-cpp-sp
cd llama-cpp-sp

# 3. Apply the SP patch
git apply ../patches/llama-cpp-b8861-full-engine.patch

# 4. Configure (Shannon-Prime ON by default; SHANNON_PRIME_DIR auto-detected
#    when the parent dir is the shannon-prime-llama clone)
cmake -S . -B build -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_CUDA=ON ^
  -DSP_CUDA=ON ^
  -DCMAKE_CUDA_ARCHITECTURES=75;86;89 ^
  -DLLAMA_BUILD_TESTS=OFF

# 5. Build
cmake --build build --config Release -j
```

The resulting `build/bin/llama.dll`, `build/bin/ggml.dll`, and friends are the LM Studio runtime DLLs. Drop them into the LM Studio runtime directory as described above.

### Compute-capability flags

| GPU | `-DCMAKE_CUDA_ARCHITECTURES=` |
|---|---|
| RTX 2060 / 2070 / 2080 | `75` |
| RTX 3050 / 3060 / 3070 / 3080 / 3090 | `86` |
| RTX 4060 / 4070 / 4080 / 4090 | `89` |
| A100 / A40 datacenter | `80` |
| H100 datacenter | `90` |
| Mix (e.g., 2060 + 4090) | `75;89` (semicolon-separated) |
| Everything modern | `75;80;86;89;90` (default for production builds) |

Match the flag to the cards in the box. CUDA binaries baked for one arch won't load on another.

### Vulkan path (no CUDA toolchain needed)

```bash
cmake -S . -B build -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_VULKAN=ON ^
  -DGGML_CUDA=OFF ^
  -DLLAMA_BUILD_TESTS=OFF
cmake --build build --config Release -j
```

Requires the [LunarG Vulkan SDK](https://vulkan.lunarg.com/) on Windows or `libvulkan-dev` + `vulkan-validationlayers-dev` on Linux. Slightly slower than CUDA on the same hardware, but works on AMD and Intel iGPUs and across mixed-vendor multi-GPU setups (the dual-GPU case validated at K=0.9920 / V=0.9730 fidelity).

### LM Studio runtime builder (Windows convenience script)

`lmstudio/build.bat` automates the above for the LM Studio target specifically — it auto-detects the Vulkan SDK, copies the produced DLLs into the right LM Studio directory, and skips the `vcvars` redirection if `cl.exe` is already on `PATH`. Run from a fresh `cmd` shell:

    .\lmstudio\build.bat

It produces a working LM Studio runtime in roughly 10 minutes on a 2060 + 16 cores.

---

## Continuous integration

`.github/workflows/release.yml` runs the four-target matrix (Windows × Linux × CUDA × Vulkan) on:

- **Tag push** matching `v*-sp*` — produces a tagged GitHub Release with all artifacts attached.
- **Branch push** to `release/*` — staging build; artifacts are uploaded as workflow artifacts (no release minted).
- **Manual workflow dispatch** — same as the staging build, useful when iterating.

To cut a new release locally:

```bash
git tag v2.14.0-sp1
git push origin v2.14.0-sp1
```

The workflow will build all four targets, package them, and create the release page automatically.

---

## License terms for redistribution

This project combines code under three licenses. Anyone redistributing a built binary needs to know how they interact.

| Component | License | Source |
|---|---|---|
| `llama.cpp` core | MIT | https://github.com/ggml-org/llama.cpp |
| `ggml` library | MIT | bundled in llama.cpp |
| `shannon-prime` math core (this repo's submodule) | AGPLv3 with commercial dual-license available | https://github.com/nihilistau/shannon-prime |
| Shannon-Prime patch + integration code (this repo) | AGPLv3 with commercial dual-license available | this repo |

**The combined binary inherits AGPLv3** unless you have a commercial license from the Shannon-Prime author. AGPLv3 means: if you offer the binary as a network service, you must make the corresponding source code available to your users. For private use, redistribution among consenting peers, or local-only deployments, AGPLv3 imposes no restriction beyond preserving the license notices.

**Commercial license** is available for proprietary integration. Contact: Ray Daniels (raydaniels@gmail.com).

---

## Pre-flight checklist before tagging a release

- [ ] All four CI matrix builds pass on the staging branch (`release/test`)
- [ ] LM Studio runtime DLLs load and produce non-zero `[shannon-prime]` log output
- [ ] At least one Qwen3.6-35B-A3B run measurably hits the 26.92 tok/sec mark on a 2060 with the new bundle
- [ ] Patch applies cleanly to a fresh `git clone --branch b8861` (no merge conflicts)
- [ ] CHANGELOG entry exists describing what changed since the previous tag
- [ ] License files (LICENSE, LICENSE.third_party) updated if any new dependencies were added

If all six are checked, push the tag. The release notes are auto-generated by the workflow but you can enrich them via `body:` or by editing the release page after creation.

---

*Maintained alongside `README.md`. Build issues and missed configurations are welcome via GitHub issues.*
