# Shannon-Prime on Android (arm64) — S22 Ultra reference run

This directory contains everything needed to reproduce the Shannon-Prime
VHT2 run on a Samsung Galaxy S22 Ultra (Snapdragon 8 Gen 1, arm64).

## What was built

| File | Target | Notes |
|---|---|---|
| `test_adreno` | arm64 Android (API 24+) | Standalone 14-test smoke for the Adreno backend. Built with `-march=armv8.2-a+fp16+dotprod` → NEON Tier 2 fp16 path active. 42 KB static-ish ELF. |
| `llama-cli` | arm64 Android (API 28+) | Patched llama.cpp b8799 with Shannon-Prime CPU shadow-cache hook. 83 MB ELF, dynamically linked to `lib/*.so`. |
| `lib/libggml*.so`, `libllama.so`, `libmtmd.so` | arm64 Android | Runtime deps. Push alongside `llama-cli` and set `LD_LIBRARY_PATH`. |

## Build it yourself

Prereqs:
- Android NDK r27d at `D:/Files/Android/android-ndk-r27d` (or override `NDK=...`).
- Latest `adb` (platform-tools 36+) for wireless debug.
- `cmake` + `ninja` in `PATH`.

```bash
bash android_build/build_test_adreno.sh
bash android_build/build_llama_cli.sh
```

## Reference run (2026-04-16, S22 Ultra / SM-S908E / Android 15)

Detected silicon features at runtime via `test_adreno`:
```
  NEON:         yes
  FP16 arith:   yes (Tier 2)
  Dot product:  yes
  I8MM:         no             (phone supports, not enabled at compile)
  SVE/SVE2:     no/no
  CPU cores:    4 big + 4 little (prime=#7)
```

`test_adreno` on-device result: **14 / 14 tests passed**, including:
- NEON VHT2 exactly matches C core on hd=32/64/128 (max_err = 0.00e+00)
- fp16 round-trip error 2.42e-04 (~fp16 precision)
- hd=64 K correlation 0.9936 (paper reference 0.9972)
- fp16 write path K correlation 0.9868 (>0.980 threshold)

Full log: `test_adreno_phone.log`.

## Dolphin 1B Q8_0 inference on-device

Model: `Dolphin3.0-Llama3.2-1B.Q8_0.gguf` pushed to `/data/local/tmp/sp/Dolphin.gguf`.

Prompt: `"The capital of France is"`, n=30, ctx=512, threads=6, seed=42,
`--single-turn`, `-ngl 0` (no GPU offload).

### Baseline (vanilla path, SHANNON_PRIME_ENABLED unset)

```
> The capital of France is Paris.
[ Prompt: 119.3 t/s | Generation: 17.8 t/s ]
```

### VHT2 (5/5/4/3 K bits, flat 3-bit V, Möbius on)

```
[Shannon-Prime] enabled: head_dim_k=64 head_dim_v=64 n_layers=16 n_heads_kv=8 n_ctx=512
Shannon-Prime VHT2 Configuration:
  head_dim:     64
  n_layers:     16
  n_heads_kv:   8
  K bands:      4 (5/5/4/3)
  V bands:      1 (3)
  Möbius mask:  on
  Compression:  3.8×

> The capital of France is Paris.
[ Prompt: 11.0 t/s | Generation: 0.4 t/s ]
```

**Same token produced by both paths, byte-for-byte ("Paris.").**

Full logs: `phone_baseline.log`, `phone_vht2.log`.

### Throughput

| Stage | Baseline | VHT2 | Ratio |
|---|---|---|---|
| Prompt eval (t/s) | 119.3 | 11.0 | 10.8× slower |
| Token gen (t/s) | 17.8 | 0.4 | 45× slower |

The slowdown is the expected consequence of the first-integration choice:
the shannon-prime-llama bridge layer (`lib/shannon-prime/tools/shannon_prime_llama.c`)
currently wires **only** the scalar CPU core (`sp_shadow_cache_t`). The
Adreno NEON-accelerated backend (`sp_adreno_cache_t`) is built, tested
(14/14) and exists at `lib/shannon-prime/backends/adreno/`, but is **not
yet routed** from `sp_llama_init_config()` when `params->backend ==
SP_BACKEND_ADRENO`. Wiring that is a separate patch against
`lib/shannon-prime/tools/shannon_prime_llama.{c,h}` (explicitly out of scope
for today's first boot).

## Run it yourself

```bash
ADB=D:/Files/Android/pt-latest/platform-tools/adb.exe
$ADB connect 192.168.8.110:33131
bash android_build/push_and_run.sh   # pushes binaries + model

$ADB shell
cd /data/local/tmp/sp
export LD_LIBRARY_PATH=/data/local/tmp/sp/lib:$LD_LIBRARY_PATH

# baseline
./llama-cli -m ./Dolphin.gguf -p "Hello" -n 30 -t 6 -c 512 --single-turn --no-warmup -ngl 0

# VHT2 (5/5/4/3 K + flat 3-bit V + Möbius)
SHANNON_PRIME_ENABLED=1 SHANNON_PRIME_K_BITS=5,5,4,3 SHANNON_PRIME_V_BITS=3 \
SHANNON_PRIME_MOBIUS=1 SHANNON_PRIME_VERBOSE=1 \
./llama-cli -m ./Dolphin.gguf -p "Hello" -n 30 -t 6 -c 512 --single-turn --no-warmup -ngl 0
```

## Gotchas

- `adb push` from Git Bash on Windows rewrites Unix-style destination paths
  into MSYS2-translated Windows paths. Either prefix the command with
  `MSYS_NO_PATHCONV=1` or quote the source path with backslashes (`"D:\\..."`).
- `llama-cli` uses `posix_spawn` (via the vendored `subprocess.h`); requires
  Android API 28+ at build time. The script defaults to API 28.
- `-no-cnv` alone still dropped into an interactive prompt in my test; use
  `--single-turn` to force a one-shot exit. Dolphin has a chat template, so
  the default is conversation mode.
- Storage: ensure `/data/local/tmp` has >2 GB free (1.6 GB for the model +
  ~100 MB for binaries + libs).
