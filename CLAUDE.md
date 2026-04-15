# CLAUDE.md — Project Instructions for AI Agents

## What This Repo Is

This is the llama.cpp integration layer for Shannon-Prime VHT2 KV cache compression.
It contains ONLY the bridge code and patch files — not the core math.

The core library lives in the `shannon-prime` repo (submodule at `lib/shannon-prime/`).
Never duplicate core code here. Import it.

## Structure

```
shannon-prime-llama/
├── lib/
│   └── shannon-prime/          Git submodule → shannon-prime repo
├── src/
│   ├── shannon_prime_llama.h   Integration API (from shannon-prime/tools/)
│   └── shannon_prime_llama.c   Integration impl (from shannon-prime/tools/)
├── patches/
│   └── llama-cpp-XXXX.patch    Patch file against specific llama.cpp version
├── examples/
│   ├── run_with_vht2.sh        Usage example
│   └── benchmark.sh            PPL comparison script
├── tests/
│   └── test_integration.c      Integration tests
├── docs/
│   └── INTEGRATION.md          Full integration guide
├── CLAUDE.md
├── LICENSE
└── README.md
```

## Rules

- The core math is in `lib/shannon-prime/`. NEVER copy-paste it here.
- The integration layer (src/) wraps the core API for llama.cpp's specific hooks.
- Patches are versioned against specific llama.cpp tags/commits.
- Run the integration tests after any change to src/.
- The hook point is AFTER RoPE is applied to K and BEFORE KV enters the cache.

## Generating Patches

After modifying a llama.cpp fork to integrate Shannon-Prime:

```bash
cd /path/to/llama-cpp-fork
git diff upstream/master > /path/to/shannon-prime-llama/patches/llama-cpp-$(git describe).patch
```

## Applying Patches

```bash
cd /path/to/llama.cpp
git apply /path/to/shannon-prime-llama/patches/llama-cpp-XXXX.patch
```
