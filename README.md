# shannon-prime-llama

**Shannon-Prime VHT2 integration for llama.cpp**

Adds spectral KV cache compression to llama.cpp via a post-decode hook that
round-trips K (and V when contiguous per position) through the VHT2 shadow
cache in place. Single sync per decode call — no per-tensor eval callback
penalty. The core transform is **VHT2** (Vilenkin-Hartley Transform),
self-inverse with 1/√p per stage; at n=2^k it reduces to the classical
Walsh-Hadamard butterfly.

- **Ship path**: 3.4–3.8× KV compression at <1.25% PPL cost, zero retraining.
- **Sqfree+spinor aggressive**: 3.3× at MOBIUS-default quality on Q8+ backbones
  (e.g. Qwen3-8B Q8 hd=128 → PPL 7.32 matching 7.31 baseline).

Validated on desktop CPU (32-chunk wiki.test PPL matches reference logs
bit-identical) and on a Samsung S22 Ultra via wireless adb (baseline 121.4 t/s
prompt, 17.7 t/s gen; post-decode VHT2 at parity +2–3%).

## Quick Start

```bash
# Clone with submodule
git clone --recursive https://github.com/nihilistau/shannon-prime-llama.git

# Fetch llama.cpp at the patch's target tag
git clone --branch b8799 --depth 1 https://github.com/ggml-org/llama.cpp llama-cpp-sp
cd llama-cpp-sp

# Pick a patch:
#   llama-cpp-b8799.patch         — ship path only
#   llama-cpp-b8799-sqfree.patch  — ship path + sqfree+spinor wire-up
git apply /path/to/shannon-prime-llama/patches/llama-cpp-b8799-sqfree.patch

cmake -S . -B build \
  -DGGML_CUDA=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_SHANNON_PRIME=ON \
  -DSHANNON_PRIME_DIR=/path/to/shannon-prime-llama
cmake --build build --target llama-perplexity -j

# Ship path
SHANNON_PRIME_ENABLED=1 ./build/bin/llama-server -m model.gguf -c 32768
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
| `SHANNON_PRIME_K_BITS` | 5,4,4,4,5 | 5-band torus-aligned (override as needed) |

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
full env-var table, and [docs/INTEGRATION.md](docs/INTEGRATION.md) for the
hook architecture, GQA support, and validation numbers.

## License

**AGPLv3** for open-source, academic, and non-proprietary use.
Everyone can use it and benefit. Derivative works must share alike.

**Dual License** — the primary goal is that the work belongs to the
commons and is protected from closure. A commercial license is
available for proprietary integration. See [LICENSE](LICENSE).

## Contact

Email: raydaniels@gmail.com
