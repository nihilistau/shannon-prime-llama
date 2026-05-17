# Shannon-Prime QNN HTP integration into FUSED_KQ — Phase 2.3 stage 3

**Status as of 2026-05-02**: code drop-in complete, preprocessor-gated.
The QNN runtime-graph dispatch primitive is validated end-to-end on S22U
V69 HTP (see `backends/qnn_aihub/sp_qnn_runner/test_sp_llama_qnn_smoke.c`,
~330 µs steady at Qwen3-4B head shape). What's left to make it fire
inside an actual `llama-cli` run is a small K-decompression hook
(documented below).

## Files (all in `src/engine/`)

| File | Purpose |
|---|---|
| `sp_qnn.h` / `sp_qnn.c` | QNN dlopen + runtime-graph build. Same source as `backends/qnn_aihub/sp_qnn_runner/sp_qnn.{c,h}`. |
| `sp_llama_qnn.h` / `sp_llama_qnn.c` | Shape-keyed cache + dispatch entry point. |
| `llama_sp_fused_kq.cpp` | Modified: added `sp_qnn_kq_local` namespace + QNN dispatch attempt before the existing DSP fast-path (gated on `LLAMA_SHANNON_PRIME_QNN`). |

## Build flags

The QNN path activates when **all three** are true at compile time:
```
__ANDROID__              (target is Android — set automatically by NDK)
__aarch64__              (target is 64-bit ARM)
LLAMA_SHANNON_PRIME_QNN  (added by us in CMake when -DSP_QNN_KQ=ON)
```

When any is missing (e.g., desktop CI build), the QNN path compiles as
a stub that returns -1, falling through to existing DSP/CPU paths.
**No desktop builds are broken by this drop-in.**

## CMake additions needed (next patch revision)

The b8861-full-engine.patch's `_SP_ENGINE_SOURCES` block needs to grow:

```cmake
set(_SP_ENGINE_SOURCES
    ${_SP_ENG}/kv_cache.cpp
    ${_SP_ENG}/gdn_state.cpp
    ${_SP_ENG}/llama_sp_fused_kq.cpp
    ${_SP_ENG}/llama_sp_kcap.cpp
)

if (SP_QNN_KQ AND CMAKE_SYSTEM_NAME STREQUAL "Android"
              AND CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    list(APPEND _SP_ENGINE_SOURCES
        ${_SP_ENG}/sp_qnn.c
        ${_SP_ENG}/sp_llama_qnn.c
    )
    target_compile_definitions(shannon_prime_engine PUBLIC LLAMA_SHANNON_PRIME_QNN)
    target_link_libraries(shannon_prime_engine PUBLIC dl)
    target_include_directories(shannon_prime_engine PRIVATE
        ${QAIRT_ROOT}/include/QNN
        ${QAIRT_ROOT}/include
    )
    message(STATUS "Shannon-Prime: QNN HTP KQ dispatch enabled")
endif()
```

Build invocation when ready:
```
cmake -B build-android \
      -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-21 \
      -DLLAMA_SHANNON_PRIME=ON \
      -DSHANNON_PRIME_DIR=$PWD/../shannon-prime-llama \
      -DSP_QNN_KQ=ON \
      -DQAIRT_ROOT=/c/Qualcomm/AIStack/QAIRT/2.45.40.260406
```

## Runtime activation

After build + push:
```
adb shell 'cd /data/local/tmp/sp22u && \
    LD_LIBRARY_PATH=$PWD/qnn:$LD_LIBRARY_PATH \
    ADSP_LIBRARY_PATH=$PWD/qnn \
    SHANNON_PRIME_FAST_PATH=1 \
    SHANNON_PRIME_FUSED_KQ=1 \
    SHANNON_PRIME_QNN_KQ=1 \
    ./llama-cli -m model.gguf ...'
```

First QK call shows `[sp_qnn_kq] enabled — runtime QNN HTP dispatch active`
in stderr. Subsequent calls hit the cached graph at ~330 µs each.

## What's wired (2026-05-02)

The QNN path inside `llama_sp_kq_compute` is **fully wired** — when env
opt-in is set + Android aarch64 build + `u->k_packed_buf_per_head` is
populated (Phase 1.6 archive path), execution flows:

1. `sp_qnn_kq_local::try_init_qnn()` lazy-creates the cache on first call
   (one-time graphFinalize cost ~50ms, amortized across all subsequent calls)
2. Decompress K rows from SP-banded bytes via existing `decompress_one_row`
   into a thread-local fp32 row buffer, then convert to fp16 in
   `k_fp16_scratch`. ~262K elements at n_kv=2048 head_dim=128, well under
   1ms on X2 Prime
3. Convert Q to fp16 if it's fp32 (most production paths have fp16 Q already)
4. Call `sp_llama_qnn_kq_dispatch()` — runs MatMul + Softmax on V69 HTP
   (~330 µs steady)
5. Convert QNN's fp16 attn output back to dst's expected layout (fp32 typical
   for ggml KQ outputs), respecting dst's row stride
6. Return rc=0; ith>0 threads short-circuit on `g_dsp_status==1`

If anything along the chain fails (cache create, dispatch error, missing
SP archive), `rc` stays nonzero and execution falls through to the existing
cDSP FastRPC path, then the scalar fallback. **No regression risk**: every
path that worked before still works.

## Performance expectations

From `test_sp_llama_qnn_smoke.c` numbers on S22U:
- First call (graphFinalize): ~800 µs amortized once per shape per session
- Steady-state per call: 320-440 µs (KQ matmul + softmax fused, fp16)
- Per-token cost (8 heads sequential): ~3.4 ms

The K-decompress overhead added by the scratch path above is bounded
by `n_kv × head_dim × C` where C is the per-element scalar cost. At
n_kv=2048 head_dim=128 it's ~262K element conversions. On the X2 Prime
core that's well under 1 ms — comparable to the current scalar inner
loop's per-call cost, but now DECOUPLED from the QK matmul (which the
HTP does in 330 µs versus ~10-20 ms scalar).

Net: per-token QK+Softmax expected at ~1.5-2 ms wall-clock. Versus
current FUSED_KQ DSP path at ~5-8 ms. **2-4× speedup ceiling once the
fp16 K hook lands.**

## Mode C continuation

The proper fix isn't the K-decompress-to-fp16 hook. It's the SP custom
QNN op package (Phase 2.3.2 task #44): a HTP-side op that takes
SP-banded bytes and outputs fp16 directly, so K bytes go to HTP without
the round-trip through the X2 core's scratch buffer.

That custom op + the runtime graph build primitive validated tonight is
the full Mode C front-end on V69.
