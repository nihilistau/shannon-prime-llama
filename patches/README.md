# llama.cpp patches

Each patch in this directory is named `llama-cpp-b<NNNN>.patch` and applies
cleanly against upstream `https://github.com/ggml-org/llama.cpp` at the tag
indicated by `b<NNNN>`.

## How to apply

```bash
git clone --branch b8799 --depth 1 https://github.com/ggml-org/llama.cpp /path/to/llama-cpp-sp
cd /path/to/llama-cpp-sp
git apply /path/to/shannon-prime-llama/patches/llama-cpp-b8799.patch
```

## How to build (CPU-only is supported today)

```bash
cmake -S . -B build \
  -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_SHANNON_PRIME=ON \
  -DSHANNON_PRIME_DIR=/path/to/shannon-prime-llama
cmake --build build --target llama-perplexity -j
```

## How to run

VHT2 is gated by `SHANNON_PRIME_ENABLED=1`. With the env var unset, the patched
binary is byte-identical in behaviour to vanilla llama.cpp (the eval callback
isn't installed).

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

## Hook strategy

The patch installs a `ggml_backend_sched_set_eval_callback` that watches for
post-RoPE `Kcur-N` / `Vcur-N` tensors. When one fires (post-eval), the data is
read out, round-tripped through Shannon-Prime's CPU shadow cache (compress →
decompress in-place), and written back. The existing `cpy_k` / `cpy_v` graph
nodes then commit the round-tripped values into the live KV cache. Attention
reads the cache normally — no separate read hook is needed because the cache
already holds reconstructed (compressed-then-decompressed) values.

This is the simplest correct shape for the first PPL number. It does NOT save
memory in this build (the live ggml KV cache is unchanged in size); the live
cache holds reconstructed values, and the PPL delta measures the quality cost
of compression on this model. Memory savings is a follow-up: the integration
needs to make the live cache itself compressed and reconstruct on demand,
which requires backend-side hooks (CUDA/Vulkan/Metal) — out of scope for the
first patch.

CPU-only build (`-DGGML_CUDA=OFF`). fp32 tensors only. Other types are passed
through untouched (with a one-shot warning to stderr).

## Validation env vars

### Ship path (VHT2 on power-of-2 head_dim)
| Variable | Default | Effect |
|---|---|---|
| `SHANNON_PRIME_ENABLED` | unset | `1` enables the post-decode hook. |
| `SHANNON_PRIME_K_BITS` | `5,5,4,3` | Per-band K bit allocation (4 bands). |
| `SHANNON_PRIME_V_BITS` | `3` | Flat V bit width (1 band). |
| `SHANNON_PRIME_MOBIUS` | `1` | Möbius squarefree-first reordering on K. |
| `SHANNON_PRIME_VERBOSE` | `0` | Print Shannon-Prime config + init line at startup. |

### Aggressive path (sqfree + spinor, opt-in, requires `llama-cpp-b8799-sqfree.patch`)
| Variable | Default | Effect |
|---|---|---|
| `SHANNON_PRIME_SQFREE` | `0` | `1` enables squarefree prime-Hartley basis (pads hd to 66/154/330). |
| `SHANNON_PRIME_SPINOR` | `0` | `1` enables the SU(2) sheet bit; auto-enables SQFREE. |
| `SHANNON_PRIME_RESIDUAL_BITS` | `3` | N-bit residual quantization (1–4; 3 is the Pareto point). |
| `SHANNON_PRIME_K_BITS` | `5,4,4,4,5` | 5-band torus-aligned allocation (when SQFREE=1). |

## Patches in this directory

| Patch | Upstream tag | Date | Notes |
|---|---|---|---|
| `llama-cpp-b8799.patch` | b8799 | 2026-04-16 | First end-to-end VHT2 hook + CMake integration (VHT2 ship path only). CPU-only. |
| `llama-cpp-b8799-sqfree.patch` | b8799 | 2026-04-17 | Superset of the above — same VHT2 ship path plus sqfree + spinor wire-up, gated on `SHANNON_PRIME_SQFREE=1`. Links `core/shannon_prime_sqfree.c` from the submodule and branches the post-decode hook on the env var. Validated on Qwen3-8B Q8 hd=128 (`SQFREE=1 SPINOR=1 K_BITS=3,3,3,3,3 RESIDUAL_BITS=3`) at 2-chunk PPL 10.20 (on trajectory for the 32-chunk target 7.32 @ 3.3×). |
| `llama-cpp-v1.03-sidecar-kdump.patch` | b8799 | 2026-04-18 | Additive to the sqfree patch. Adds `SHANNON_PRIME_SIDECAR=<path>` which loads an `.sp_freq_factors.bin` emitted by `tools/sp_inject_freqs.py` and overwrites the `rope_freqs.weight` tensor via `ggml_backend_tensor_set` at context init (optional `SHANNON_PRIME_ALPHA=0..1` blends against the current factors). Adds `SHANNON_PRIME_DUMP_K=<path>` inside the post-decode hook so `tools/sp_auto_bands.py` can compute per-band VHT2 energy-weighted K allocations from a warmup run. No behaviour change when either env is unset. |
| `llama-cpp-v1.05-gpu-kv.patch` | b8799 | 2026-04-18 | Supersedes v1.03. Same sidecar + K-dump features, plus backend-aware `gather_vec` / `scatter_vec` so the post-decode hook works when KV is offloaded to GPU (`-ngl 99` without `-nkvo`). Pre-v1.05 the hook segfaulted silently mid-warmup whenever it dereferenced a CUDA pointer with `std::memcpy`. The fix routes per-vector reads/writes through `ggml_backend_tensor_get` / `ggml_backend_tensor_set` when the tensor's buffer reports non-host. Correct but slow on GPU-resident KV (per-vector PCIe round-trip); for fast benches keep using `-nkvo`. The `v_trans=true` strided-V path still requires host-buffer V and skips silently otherwise. |
