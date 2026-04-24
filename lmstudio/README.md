# LM Studio Integration

**Drop-in Shannon-Prime runtime for LM Studio v2.14.0+**

Builds a custom `llama.dll` and `ggml.dll` that replace the stock LM Studio
runtime with a Shannon-Prime-enabled build. The full VHT2 engine (ship,
sqfree+spinor, hierarchical, System 1/2 switching, multi-GPU) compiles into
`llama.dll` as internal static libraries — no extra DLLs to manage.

**Validated:** Qwen3.6-35B-A3B (MoE) running at **26.92 tok/sec** in
LM Studio with the Shannon-Prime KV cache active.

## What gets replaced

Only two DLLs are replaced. Everything else stays stock:

| File | Source | Notes |
|------|--------|-------|
| `llama.dll` | **Built here** | Contains Shannon-Prime engine linked statically |
| `ggml.dll` | **Built here** | Rebuilt alongside llama.dll |
| `ggml-base.dll` | Stock LM Studio | Keep as-is (661 exports) |
| `ggml-cpu.dll` | Stock LM Studio | Keep as-is |
| `ggml-cuda.dll` | Stock LM Studio | Keep as-is (157 MB) |

## Prerequisites

- **Visual Studio 2019+ Build Tools** — `cl.exe`, `link.exe`, `rc.exe`
  (the build script auto-detects VS 2019 or 2022)
- **CUDA Toolkit 12.x+** — `nvcc` (optional; builds without CUDA if not found)
- **CMake 3.14+** and **Ninja**
- **llama.cpp checkout at b8861** with the full-engine patch applied
  (see instructions below)

## Quick start

```cmd
REM 1. Clone llama.cpp at the LM Studio base tag
git clone --branch b8861 --depth 1 https://github.com/ggml-org/llama.cpp C:\llama-cpp-sp
cd C:\llama-cpp-sp

REM 2. Clone shannon-prime-llama (with submodule)
git clone --recursive https://github.com/nihilistau/shannon-prime-llama C:\shannon-prime-llama

REM 3. Apply the full-engine patch
git apply C:\shannon-prime-llama\patches\llama-cpp-b8861-full-engine.patch

REM 4. Build the runtime DLLs
C:\shannon-prime-llama\lmstudio\build.bat C:\llama-cpp-sp C:\shannon-prime-llama

REM 5. Copy output DLLs into LM Studio's runtime folder
copy /y output\llama.dll "%USERPROFILE%\.cache\lm-studio\extensions\backends\llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.13.0\"
copy /y output\ggml.dll "%USERPROFILE%\.cache\lm-studio\extensions\backends\llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.13.0\"
```

## Build script usage

```
build.bat <llama-cpp-dir> <shannon-prime-llama-dir> [output-dir]
```

| Argument | Description |
|----------|-------------|
| `llama-cpp-dir` | Path to patched llama.cpp checkout (b8861 + full-engine patch) |
| `shannon-prime-llama-dir` | Path to this repo (shannon-prime-llama) |
| `output-dir` | Where to put the DLLs (default: `.\output`) |

The script:
1. Auto-detects Visual Studio Build Tools (2019 or 2022)
2. Auto-detects CUDA Toolkit (v12.4, v12.6, v13.0, v13.2)
3. Configures with `GGML_CUDA=OFF` + `SP_CUDA=ON` (our CUDA backends
   compile independently of ggml's CUDA code, which avoids MSVC template
   errors in ggml-cuda)
4. Builds only `llama.dll` via selective Ninja target
5. Copies `llama.dll` and `ggml.dll` to the output directory

## How it works

The patch adds `LLAMA_SHANNON_PRIME=ON` and `SHANNON_PRIME_DIR=<path>` to
llama.cpp's CMake. When enabled, three static libraries are built from
shannon-prime-llama sources and linked into `llama.dll`:

| Library | Language | Contents |
|---------|----------|----------|
| `shannon_prime_core` | C | VHT2, Möbius, banded quant, sqfree, modelpack |
| `shannon_prime_cuda` | CUDA | GPU kernels (ship, sqfree, hierarchical) |
| `shannon_prime_engine` | C++ | KV cache manager, GDN state (System 1/2) |

The bridge file `llama-shannon-prime.cpp` hooks into `llama-context.cpp` at
three points: constructor init, destructor cleanup, and post-compute. When
`SHANNON_PRIME_ENABLED=1` is set in the environment, the engine intercepts
KV writes and routes them through the VHT2 compression pipeline.

## SP_CUDA vs GGML_CUDA

The build uses `SP_CUDA=ON` independently of `GGML_CUDA`. This is deliberate:
ggml's own CUDA code has template specializations in `common.cuh` that fail
to compile on MSVC 2019. Our `.cu` files compile standalone against the CUDA
toolkit without touching ggml-cuda internals. The `--use-local-env` nvcc flag
avoids path-length issues with vcvars on Windows.

## LM Studio runtime folder

The runtime folder is typically at:

```
%USERPROFILE%\.cache\lm-studio\extensions\backends\llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.13.0\
```

Back up the stock `llama.dll` and `ggml.dll` before replacing them. To revert,
restore the backups.

## Configuration

Once the custom runtime is installed, configure Shannon-Prime via environment
variables before launching LM Studio:

```cmd
set SHANNON_PRIME_ENABLED=1
set SHANNON_PRIME_K_BITS=5,5,4,3
set SHANNON_PRIME_V_BITS=3
set SHANNON_PRIME_MOBIUS=1
start "" "%LOCALAPPDATA%\LM Studio\LM Studio.exe"
```

See the main [README.md](../README.md) for the full environment variable
reference including sqfree+spinor and hierarchical paths.

## KV cache memory with VHT2

| Context length | VHT2 KV cache size | Stock fp16 KV size | Compression |
|---------------|--------------------|--------------------|-------------|
| 8K | 39 MB | ~150 MB | ~3.8× |
| 32K | 156 MB | ~600 MB | ~3.8× |
| 131K | 624 MB | ~2.4 GB | ~3.8× |
| 262K | 4.88 GB | ~18.7 GB | ~3.8× |

## License

AGPLv3. See [LICENSE](../LICENSE).
