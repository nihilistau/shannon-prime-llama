# CLAUDE.md — shannon-prime-llama

## What This Repo Is

Integration layer for Shannon-Prime VHT2 KV cache compression in llama.cpp.
Contains the bridge code and upstream patch files — **not** the core math.
Core library lives in the `shannon-prime` repo, pulled in as a submodule at
`lib/shannon-prime/`. Never duplicate core code here; import it.

VHT2 (Vilenkin-Hartley Transform) is the single transform used throughout
the project — at n=2^k it reduces to the classical Walsh-Hadamard Transform
butterfly scaled by 1/√2 per stage (self-inverse, no 1/N), and at sqfree-padded
dimensions it extends naturally to primes {2, 3, 5, 7, 11}. Two runtime configurations use
the same pipeline:

- **Ship path** (default): VHT2 → Möbius reorder → 5/5/4/3 band quantize → store.
  Gated on `SHANNON_PRIME_ENABLED=1`.
- **Sqfree+spinor aggressive path** (opt-in): sqfree_pad → VHT2 → Knight
  skeleton → Möbius CSR predict → 3-bit residual → SU(2) spinor sheet bit →
  store. Gated on `SHANNON_PRIME_SQFREE=1` (auto-on with `_SPINOR=1`).

## Structure

```
shannon-prime-llama/
├── lib/
│   └── shannon-prime/             Submodule → shannon-prime repo
├── src/
│   ├── shannon_prime_llama.h      Integration API (bridge to core)
│   └── shannon_prime_llama.c      Bridge impl
├── patches/
│   ├── llama-cpp-b8799.patch      Ship-path-only patch (CMake + post-decode hook)
│   └── llama-cpp-b8799-sqfree.patch
│                                   Superset: ship path + sqfree/spinor wire-up
│   └── README.md                  Env vars, build + run recipes
├── examples/
├── tests/
│   └── test_integration.c         Integration tests
├── android_build/
│   ├── build_llama_cli.sh         NDK r27d aarch64 build (API 28)
│   ├── build_test_adreno.sh       test_adreno cross-compile
│   ├── push_and_run.sh            adb push model + binary to /data/local/tmp/sp
│   └── phone_*.log                Reference phone-runtime logs
├── ppl_logs/                      32-chunk wiki.test reference numbers
├── docs/
└── README.md
```

## Rules

- Core math is in `lib/shannon-prime/`. NEVER copy-paste it here.
- The integration layer (src/) wraps the core API for llama.cpp's specific
  hooks. The hook is a **post-graph_compute walk** of the KV cache — one sync
  per decode call, not per-tensor — installed on the llama_context.
- Patches are versioned against specific llama.cpp tags. Keep one patch per
  configuration (ship / sqfree) so users can pick.
- Run the integration tests after any change to src/.
- On Windows git-bash, `adb push` to `/data/local/tmp/...` needs
  `MSYS_NO_PATHCONV=1` set, or adb will rewrite the Unix target to a bundled
  Git-for-Windows path.

## Applying Patches

Both patches target the same upstream llama.cpp tag (`b8799`):

```bash
git clone --branch b8799 --depth 1 https://github.com/ggml-org/llama.cpp /path/to/llama-cpp-sp
cd /path/to/llama-cpp-sp

# Ship path only (smaller, VHT2 ship config)
git apply /path/to/shannon-prime-llama/patches/llama-cpp-b8799.patch

# OR ship + sqfree+spinor opt-in (superset)
git apply /path/to/shannon-prime-llama/patches/llama-cpp-b8799-sqfree.patch
```

## Building

```bash
cmake -S . -B build \
  -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_SHANNON_PRIME=ON \
  -DSHANNON_PRIME_DIR=/path/to/shannon-prime-llama
cmake --build build --target llama-perplexity -j
```

## Running

### Ship path
```bash
SHANNON_PRIME_ENABLED=1 \
SHANNON_PRIME_K_BITS=5,5,4,3 \
SHANNON_PRIME_V_BITS=3 \
SHANNON_PRIME_MOBIUS=1 \
./build/bin/llama-perplexity -m model.gguf -f wikitext-2-raw/wiki.test.raw \
    -c 2048 -b 512 -t 16 -fa off
```

### Sqfree+spinor aggressive path (Q8+ backbones)
```bash
SHANNON_PRIME_ENABLED=1 \
SHANNON_PRIME_SQFREE=1 \
SHANNON_PRIME_SPINOR=1 \
SHANNON_PRIME_RESIDUAL_BITS=3 \
SHANNON_PRIME_K_BITS=3,3,3,3,3 \
./build/bin/llama-perplexity -m Qwen3-8B-Q8_0.gguf -f wikitext-2-raw/wiki.test.raw \
    -c 2048 -b 512 -t 16 -fa off
```

## Generating New Patches

```bash
cd /path/to/llama-cpp-fork
# Intent-to-add tracks NEW files (llama-shannon-prime.cpp/.h) in the diff
git add -N src/llama-shannon-prime.cpp src/llama-shannon-prime.h
git diff --no-color > /path/to/shannon-prime-llama/patches/llama-cpp-$(git describe --tags).patch
git reset HEAD -- src/llama-shannon-prime.cpp src/llama-shannon-prime.h
```
