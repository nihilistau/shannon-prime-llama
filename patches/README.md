# llama.cpp patches

Each patch in this directory applies cleanly against upstream
`https://github.com/ggml-org/llama.cpp` at the tag indicated in its filename.

## Recommended patch

**`llama-cpp-b8733-full-engine.patch`** is the current recommended patch. It
targets b8733 (the commit behind LM Studio v2.13.0) and integrates the full
Shannon-Prime engine: VHT2 ship + sqfree+spinor + hierarchical Vilenkin +
System 1/2 switching + multi-GPU sharding. All four backends (CPU, CUDA,
Vulkan, Adreno) are compiled from sources in `shannon-prime-llama/src/`.

The older b8799 patches below are retained for reference but are superseded.

## How to apply (full-engine patch, b8733)

```bash
git clone --branch b8733 --depth 1 https://github.com/ggml-org/llama.cpp /path/to/llama-cpp-sp
cd /path/to/llama-cpp-sp
git apply /path/to/shannon-prime-llama/patches/llama-cpp-b8733-full-engine.patch
```

## How to build

```bash
# CPU + SP CUDA (recommended — GGML_CUDA=OFF avoids MSVC template errors)
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_SHANNON_PRIME=ON \
  -DSHANNON_PRIME_DIR=/path/to/shannon-prime-llama \
  -DSP_CUDA=ON
cmake --build build -j

# CPU only (no CUDA)
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_SHANNON_PRIME=ON \
  -DSHANNON_PRIME_DIR=/path/to/shannon-prime-llama \
  -DSP_CUDA=OFF
cmake --build build -j
```

For LM Studio runtime builds (Windows DLLs), see [`../lmstudio/README.md`](../lmstudio/README.md).

## How to run

VHT2 is gated by `SHANNON_PRIME_ENABLED=1`. With the env var unset, the patched
binary is byte-identical in behaviour to vanilla llama.cpp.

```bash
# baseline
./build/bin/llama-perplexity -m model.gguf -f wikitext-2-raw/wiki.test.raw \
    -c 2048 -b 512 -t 16 -fa off

# VHT2 (ship config)
SHANNON_PRIME_ENABLED=1 \
SHANNON_PRIME_K_BITS=5,5,4,3 \
SHANNON_PRIME_V_BITS=3 \
SHANNON_PRIME_MOBIUS=1 \
./build/bin/llama-perplexity -m model.gguf -f wikitext-2-raw/wiki.test.raw \
    -c 2048 -b 512 -t 16 -fa off
```

## Architecture (full-engine patch)

The full-engine patch adds `LLAMA_SHANNON_PRIME=ON` and `SHANNON_PRIME_DIR`
to the CMake build. When enabled, three static libraries compile from
`shannon-prime-llama/` sources and link into `llama.dll` / `libllama.so`:

| Library | Language | Sources |
|---------|----------|---------|
| `shannon_prime_core` | C | `lib/shannon-prime/core/` (VHT2, Möbius, sqfree, modelpack) + `src/tools/` (bridge) |
| `shannon_prime_cuda` | CUDA | `src/backends/cuda/` (ship, sqfree, hierarchical GPU kernels) |
| `shannon_prime_engine` | C++ | `src/engine/` (KV cache manager, GDN state for System 1/2) |

The bridge file `llama-shannon-prime.cpp` (added by the patch) hooks into
`llama-context.cpp` at three points: constructor init, destructor cleanup,
and post-compute. Backend-aware gather/scatter handles both host and
GPU-resident KV buffers.

**SP_CUDA vs GGML_CUDA:** The build uses `SP_CUDA=ON` independently of
`GGML_CUDA`. Our `.cu` files compile standalone against the CUDA toolkit
without touching ggml-cuda internals, avoiding MSVC 2019 template errors.

## Validation env vars

### Ship path (VHT2 on power-of-2 head_dim)
| Variable | Default | Effect |
|---|---|---|
| `SHANNON_PRIME_ENABLED` | unset | `1` enables the post-decode hook. |
| `SHANNON_PRIME_K_BITS` | `5,5,4,3` | Per-band K bit allocation (4 bands). |
| `SHANNON_PRIME_V_BITS` | `3` | Flat V bit width (1 band). |
| `SHANNON_PRIME_MOBIUS` | `1` | Möbius squarefree-first reordering on K. |
| `SHANNON_PRIME_VERBOSE` | `0` | Print Shannon-Prime config + init line at startup. |

### Aggressive path (sqfree + spinor)
| Variable | Default | Effect |
|---|---|---|
| `SHANNON_PRIME_SQFREE` | `0` | `1` enables squarefree prime-Hartley basis (pads hd to 66/154/330). |
| `SHANNON_PRIME_SPINOR` | `0` | `1` enables the SU(2) sheet bit; auto-enables SQFREE. |
| `SHANNON_PRIME_RESIDUAL_BITS` | `3` | N-bit residual quantization (1–4; 3 is the Pareto point). |
| `SHANNON_PRIME_K_BITS` | `5,4,4,4,5` | 5-band torus-aligned allocation (when SQFREE=1). |

## Patches in this directory

| Patch | Upstream tag | Date | Notes |
|---|---|---|---|
| **`llama-cpp-b8733-full-engine.patch`** | **b8733** | **2026-04-22** | **Current.** Full Shannon-Prime engine: VHT2 ship + sqfree+spinor + hierarchical + System 1/2 + multi-GPU. 4 backends (CPU, CUDA, Vulkan, Adreno). SP_CUDA independent of GGML_CUDA. Validated: Qwen3.6-35B-A3B at 26.92 tok/sec in LM Studio. |
| `llama-cpp-b8799.patch` | b8799 | 2026-04-16 | Legacy. First end-to-end VHT2 hook + CMake integration (VHT2 ship path only). CPU-only. |
| `llama-cpp-b8799-sqfree.patch` | b8799 | 2026-04-17 | Legacy. Ship path plus sqfree + spinor wire-up. |
| `llama-cpp-v1.03-sidecar-kdump.patch` | b8799 | 2026-04-18 | Legacy. Additive to sqfree patch — sidecar freq_factors + K-dump for auto-band tools. |
| `llama-cpp-v1.05-gpu-kv.patch` | b8799 | 2026-04-18 | Legacy. Backend-aware gather/scatter for GPU-resident KV. Superseded by the b8733 full-engine patch. |
