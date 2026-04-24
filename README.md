# shannon-prime-llama

**Shannon-Prime full engine integration for llama.cpp**

Adds spectral KV cache compression to llama.cpp via the full Shannon-Prime
engine. The recommended patch (`llama-cpp-b8861-full-engine.patch`) compiles
the complete VHT2 stack — ship, sqfree+spinor, hierarchical Vilenkin, System
1/2 switching, and multi-GPU sharding — into `llama.dll` / `libllama.so` as
internal static libraries. Four backends: CPU, CUDA, Vulkan, and Adreno.

- **Ship path**: 3.4–3.8× KV compression at <1.25% PPL cost, zero retraining.
- **Sqfree+spinor aggressive**: 2.8× at MOBIUS-default quality on Q8+ backbones.
- **LM Studio validated**: Qwen3.6-35B-A3B (MoE) at **26.92 tok/sec** with the
  Shannon-Prime KV cache active (LM Studio v2.13.0, custom runtime DLLs).

### Current (2026-04-25)

1. **Full-engine b8861 patch.** Targets b8861 (commit cf8b0db, LM Studio v2.14.0 base). Supersedes the b8733 patch (v2.13.0). Integrates VHT2 ship + sqfree+spinor + hierarchical + System 1/2 + multi-GPU.
2. **Qwen 3.6 27B support.** b8861 includes upstream improvements for Qwen 3.6 architecture handling, complementing our existing Qwen3.6-35B-A3B (MoE) validation.
3. **LM Studio runtime builder.** `lmstudio/build.bat` produces drop-in `llama.dll` + `ggml.dll` for LM Studio v2.14.0. SP_CUDA compiles independently of GGML_CUDA (avoids MSVC template errors).
4. **Engine + backend sources ported.** All engine subsystems (KV cache manager, GDN state) and all 4 backend implementations live in `src/`.
5. **Dual-GPU Vulkan.** RTX 2060 (K=0.9920, V=0.9730) + Intel UHD (identical fidelity), cross-device correlation 1.0000.
6. **Model-pack registry.** Per-architecture compression defaults. phi3 CALIBRATED, 7 architectures PROVISIONAL.

### Previous (b8733, LM Studio v2.13.0)

The b8733 patch remains in `patches/` for users on LM Studio v2.13.0.

## Quick Start

```bash
# Clone with submodule
git clone --recursive https://github.com/nihilistau/shannon-prime-llama.git

# Fetch llama.cpp at the patch's target tag
git clone --branch b8861 --depth 1 https://github.com/ggml-org/llama.cpp llama-cpp-sp
cd llama-cpp-sp

# Apply the full-engine patch
git apply /path/to/shannon-prime-llama/patches/llama-cpp-b8861-full-engine.patch

# Build (SP_CUDA=ON is auto-detected if nvcc is found)
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_SHANNON_PRIME=ON \
  -DSHANNON_PRIME_DIR=/path/to/shannon-prime-llama
cmake --build build -j

# Ship path
SHANNON_PRIME_ENABLED=1 ./build/bin/llama-server -m model.gguf -c 32768
```

### LM Studio (Windows)

```cmd
REM See lmstudio/README.md for full instructions
lmstudio\build.bat C:\llama-cpp-sp C:\shannon-prime-llama
REM Copy output\llama.dll and output\ggml.dll into LM Studio v2.14.0 runtime folder
```

## Configuration

### Ship path (default)
| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `SHANNON_PRIME_ENABLED` | 0 | Enable the VHT2 shadow cache hook |
| `SHANNON_PRIME_K_BITS` | 5,5,4,3 | K band bit allocation |
| `SHANNON_PRIME_V_BITS` | 3 | V flat bit allocation |
| `SHANNON_PRIME_MOBIUS` | 1 | Möbius squarefree-first reorder (K only) |
| `SHANNON_PRIME_VERBOSE` | 0 | Print config + init |

### Sqfree+spinor aggressive path (opt-in, Q8+)
| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `SHANNON_PRIME_SQFREE` | 0 | Enable sqfree prime-Hartley basis |
| `SHANNON_PRIME_SPINOR` | 0 | Enable SU(2) sheet bit (auto-enables SQFREE) |
| `SHANNON_PRIME_RESIDUAL_BITS` | 3 | Residual depth (1–4; 3 is the Pareto point) |
| `SHANNON_PRIME_SK_FRAC` | 0.75 | Skeleton fraction of pad_dim (Knight skeleton size) |
| `SHANNON_PRIME_K_BITS` | 5,4,4,4,5 | 5-band torus-aligned (override as needed) |

## Supported Models

The engine works with any GGUF model via llama.cpp. Tested architectures include Qwen 3.6 (27B dense and 35B-A3B MoE), Phi-3, Llama 3, Gemma 3/4, and Dolphin 1B. The model-pack registry provides per-architecture compression defaults keyed by GGUF `general.architecture`. See [shannon-prime/docs/MODEL-PACK.md](https://github.com/nihilistau/shannon-prime/blob/main/docs/MODEL-PACK.md) for the full registry and calibration status.

## Android / mobile

```bash
# Cross-compile for aarch64, API 28 (posix_spawn requires API 28+)
API=28 bash android_build/build_llama_cli.sh

# Push and run
bash android_build/push_and_run.sh
adb shell 'cd /data/local/tmp/sp && LD_LIBRARY_PATH=./lib \
  SHANNON_PRIME_ENABLED=1 SHANNON_PRIME_MOBIUS=1 \
  ./llama-cli -m Dolphin.gguf -p "The capital of France is" -n 8 --no-warmup'
```

See [patches/README.md](patches/README.md) for build+run recipes and the
full env-var table, [lmstudio/README.md](lmstudio/README.md) for the LM Studio
runtime builder.

## Project Structure

```
shannon-prime-llama/
├── lib/shannon-prime/          ← git submodule → nihilistau/shannon-prime (core math)
├── src/
│   ├── backends/
│   │   ├── cuda/               CUDA kernels (ship, sqfree, hierarchical)
│   │   ├── vulkan/             Vulkan compute shaders
│   │   └── adreno/             Qualcomm NEON + Hexagon HVX
│   ├── engine/
│   │   ├── kv_cache.{h,cpp}    KV cache manager (wraps sp_shadow_cache_t)
│   │   └── gdn_state.{h,cpp}   GDN state for System 1/2 switching
│   └── tools/
│       ├── shannon_prime_llama.{h,c}       Bridge to llama.cpp
│       └── shannon_prime_llama_sqfree.c    Sqfree bridge
├── patches/
│   ├── llama-cpp-b8861-full-engine.patch   ← CURRENT: full engine, b8861 (LMS v2.14.0)
│   ├── llama-cpp-b8733-full-engine.patch   ← previous: full engine, b8733 (LMS v2.13.0)
│   └── llama-cpp-b8799-*.patch             ← legacy patches
├── lmstudio/
│   ├── build.bat               Windows build script for LM Studio DLLs
│   └── README.md               LM Studio integration guide
├── android_build/              Cross-compile scripts for aarch64
└── ppl_logs/                   Reference perplexity measurements
```

## License

**AGPLv3** for open-source, academic, and non-proprietary use.
Everyone can use it and benefit. Derivative works must share alike.

**Dual License** — the primary goal is that the work belongs to the
commons and is protected from closure. A commercial license is
available for proprietary integration. See [LICENSE](LICENSE).

## Contact

Email: raydaniels@gmail.com
