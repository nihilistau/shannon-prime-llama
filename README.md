# shannon-prime-llama

**Shannon-Prime VHT2 integration for llama.cpp**

Adds spectral KV cache compression to llama.cpp. 3.4–3.8× KV memory reduction at <1.25% perplexity cost. Zero retraining required.

## Quick Start

```bash
# Clone with submodule
git clone --recursive https://github.com/YOUR_USER/shannon-prime-llama.git

# Apply patch to your llama.cpp
cd /path/to/llama.cpp
git apply /path/to/shannon-prime-llama/patches/llama-cpp-XXXX.patch

# Rebuild llama.cpp
cmake -B build && cmake --build build

# Run with VHT2 compression
SHANNON_PRIME_ENABLED=1 ./build/bin/llama-server -m model.gguf -c 32768
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `SHANNON_PRIME_ENABLED` | 0 | Enable VHT2 shadow cache |
| `SHANNON_PRIME_K_BITS` | 5,5,4,3 | K band bit allocation |
| `SHANNON_PRIME_V_BITS` | 3 | V flat bit allocation |
| `SHANNON_PRIME_MOBIUS` | 1 | Möbius squarefree-first reorder |

See [docs/INTEGRATION.md](docs/INTEGRATION.md) for full guide including architecture diagrams, GQA support, and validation.

## License

AGPLv3 / Commercial dual license. See [LICENSE](LICENSE).
