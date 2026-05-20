# shannon-prime-llama

**Shannon-Prime integration for llama.cpp / LM Studio**

This repository patches the full Shannon-Prime VHT2 spectral compression stack into llama.cpp. The patch compiles the complete engine — ship path, sqfree+spinor, hierarchical Vilenkin, System 1/2, multi-GPU sharding, and PrimePE — into `llama.dll` / `libllama.so` as internal static libraries.

This is a **bridge**, not the primary deployment path. llama.cpp was not designed for compressed KV caches, and maintaining a patch against a moving upstream target is inherently fragile. The [Shannon-Prime Engine](https://github.com/nihilistau/shannon-prime-engine) is the reference implementation that owns the KV data path end-to-end. This repository exists because LM Studio and the broader llama.cpp ecosystem represent a large installed base that benefits from Shannon-Prime compression today.

The long-term plan is to phase this out as the engine matures and gains its own ecosystem integrations..

---

## What It Does

The recommended patch (`llama-cpp-b8861-full-engine.patch`) targets llama.cpp b8861 (commit cf8b0db, LM Studio v2.14.0 base). It intercepts KV cache writes and reads in the llama.cpp forward pass, routing them through Shannon-Prime's shadow cache. From the perspective of llama.cpp's attention kernel, K and V vectors appear as normal fp32/fp16 data — the compression and decompression are transparent.

Four backends: CPU, CUDA, Vulkan, and Adreno. All are compiled as internal static libraries within the llama.cpp build.

---

## Validated Results

| Model | Configuration | Result | Platform |
|---|---|---|---|
| Qwen3.6-35B-A3B (MoE) | Ship + PrimePE | **26.92 tok/sec** | LM Studio v2.14.0, custom DLLs |
| Qwen2.5-Coder-3B + 0.5B | Spec-decode --draft 8 + FUSED_KQ | **43.72 t/s (3.58×)** | S22U phone CPU |
| RTX 2060 + Intel UHD | Dual-GPU Vulkan | K=0.9920, V=0.9730, cross-device 1.0000 | Desktop |
| phi3 | Ship, calibrated | +2.44% PPL (within budget) | Desktop |

---

## Quick Start

### Apply the Patch

```bash
# Clone llama.cpp at b8861
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
git checkout cf8b0db

# Apply Shannon-Prime patch
git apply /path/to/shannon-prime-llama/patches/llama-cpp-b8861-full-engine.patch
```

### Build

```bash
# Linux / macOS
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
cmake --build build -j

# Windows (LM Studio runtime)
cd lmstudio
.\build.bat    # Produces llama.dll + ggml.dll
```

### Install in LM Studio

```powershell
# Copy built DLLs to LM Studio runtime directory
copy build\bin\Release\llama.dll "C:\Users\<you>\.cache\lm-studio\runtimes\<version>\"
copy build\bin\Release\ggml.dll  "C:\Users\<you>\.cache\lm-studio\runtimes\<version>\"
```

### Run

Set environment variables before launching LM Studio:

```powershell
$env:SHANNON_PRIME = "1"
$env:SHANNON_PRIME_PRIME_PE = "1"
& "C:\...\LM Studio.exe"
```

Or for llama-cli:

```bash
SHANNON_PRIME=1 SHANNON_PRIME_PRIME_PE=1 \
  ./build/bin/llama-cli -m model.gguf -p "Hello, world"
```

### Speculative Decoding

```bash
SHANNON_PRIME=1 \
  ./build/bin/llama-cli \
  -m target.gguf \
  -md draft.gguf \
  --draft-max 8 --draft-min 2 \
  -p "Write a function that sorts an array"
```

---

## Configuration

All configuration via environment variables (same as the engine):

```bash
SHANNON_PRIME=1                      # Master enable
SHANNON_PRIME_K_BITS=5,5,4,3        # K band allocation
SHANNON_PRIME_V_BITS=3              # V band allocation (flat)
SHANNON_PRIME_MOBIUS=1              # Möbius reorder
SHANNON_PRIME_SQFREE=1              # Sqfree + spinor path
SHANNON_PRIME_PRIME_PE=1            # PrimePE frequency injection
SHANNON_PRIME_PRIME_PE_ALPHA=0.17   # PrimePE blend ratio
SHANNON_PRIME_CAUCHY=2              # Dynamic Cauchy reset
SHANNON_PRIME_ROLE=target           # For spec-decode: "target" or "draft"
SHANNON_PRIME_PRESET=auto           # Model-pack preset
```

---

## What This Is Not

This is not a lightweight plugin. The patch modifies llama.cpp's KV cache allocation, write, and read paths. It adds ~6,500 lines of Shannon-Prime source code as internal static libraries. It changes the memory layout of the KV cache from contiguous fp16 to compressed banded format.

Every upstream llama.cpp update requires re-validating and potentially rebasing the patch. The b8861 patch supersedes the b8733 patch (v2.13.0).

For new projects, use the [Shannon-Prime Engine](https://github.com/nihilistau/shannon-prime-engine) instead. It was built specifically so we don't have to fight llama.cpp's assumptions about KV cache layout.

---

## Ecosystem

Shannon-Prime's VHT2 compression extends beyond LLMs:

| Repository | Purpose |
|---|---|
| [shannon-prime](https://github.com/nihilistau/shannon-prime) | Core math library. Vendored here as `lib/shannon-prime/`. |
| [shannon-prime-engine](https://github.com/nihilistau/shannon-prime-engine) | The reference inference engine. **Use this for new work.** |
| [shannon-prime-comfyui](https://github.com/nihilistau/shannon-prime-comfyui) | 16 ComfyUI nodes for video/image/audio/TTS. |

**Voxtral TTS forks:**
[Python](https://github.com/nihilistau/ComfyUI-FL-VoxtralTTS),
[Rust](https://github.com/nihilistau/voxtral-mini-realtime-rs),
[C](https://github.com/nihilistau/voxtral-tts.c).

---

## License

Copyright (C) 2026 Ray Daniels. All Rights Reserved.

Licensed under the [GNU Affero General Public License v3.0](LICENSE) (AGPLv3).
Commercial license available — contact raydaniels@gmail.com.
