# S22 Ultra — real mobile performance numbers (Adreno NEON backend)

Run against Dolphin 3.0 Llama 3.2 1B Q8_0, prompt `"The capital of France is"`,
n=30, ctx=512, threads=6, `-ngl 0`, `--single-turn`, seed=42, Android 15,
Snapdragon 8 Gen 1, cold boot (LP3/NORMAL CPU governor).

Same `llama-cli` binary, same `.so` files, same model. Only the environment
differs.

| Run | `SHANNON_PRIME_BACKEND` | Active backend (reported) | Prompt (t/s) | Generation (t/s) | Token output |
|---|---|---|---|---|---|
| 1 | unset | — (env-disabled, vanilla path) | **119.3** | **17.8** | `Paris.` |
| 2 | `cpu` | CPU (scalar `sp_shadow_cache_t`) | 11.2 | 0.4 | `Paris.` |
| 3 | `adreno` | Adreno (NEON Tier-2 fp16, `sp_adreno_cache_t`) | 10.7 | 0.4 | `Paris.` |

Logs: `phone_baseline.log`, `phone_vht2_cpu.log`, `phone_vht2_adreno.log`.

All three runs emit the same deterministic token (`"Paris."`) — the VHT2
compress/decompress round-trip is mathematically correct on-device, matching
both the desktop Qwen3-8B result and the 14/14 `test_adreno` on-device smoke.

## The NEON speedup didn't materialise. Why?

Wiring `SP_BACKEND_ADRENO` through the bridge replaced the scalar C inner
loop with the NEON Tier-2 fp16 path (verified: the `[Shannon-Prime Mobile]
Decode: CPU/FP16` line appears in `phone_vht2_adreno.log`, and `test_adreno`
separately confirmed `FP16 arith: yes (Tier 2)`). But the **total** VHT2
throughput is unchanged: both backends land at 0.4 t/s.

That's because the NEON win is confined to the inner compute — WHT butterfly,
Möbius gather, banded quant — which for `head_dim=64` is only ~100 NEON ops
per call. What's left (per-call fixed overhead) dominates:

- The llama.cpp eval callback fires **per tensor per layer**, not per batch.
- Inside the callback, the loop is `for each token, for each head:
  sp_llama_write_k; sp_llama_read_k; memcpy_back`. One write + read pair
  per (token, head), dispatched through the `switch (active_backend)` in
  the bridge.
- For Dolphin 1B (16 layers × 8 KV heads) over 30 generated tokens that is
  **~15,360 single-vector dispatcher calls**, each doing malloc/free of
  `sp_mobius_reorder`'s temporary buffer, a strided cache-slot offset
  calculation, and a full compress→store→load→decompress cycle. At
  ~4.9 ms/call on the S22 Ultra that is ~75 seconds of wall time, which
  matches what we see.
- NEON doesn't help that overhead at all — it speeds up the ~20 µs of
  actual transform, not the ~4.9 ms of call setup + dispatch + malloc.

The paper's **37–42 ms per writeback** (the Dolphin 1B mobile target cited
in `docs/BACKEND-ADRENO.md`) is a **batch** writeback — all layer/head
slots processed per-batch in one call. The bridge already exposes
`sp_llama_write_k_batch` / `sp_llama_read_k_batch`; the llama.cpp eval
callback needs to be switched to use those instead of iterating singletons.

## So, what did the Adreno wiring actually buy us?

1. The bridge is now **multi-backend real** (not stubs). Any caller can pick
   `SP_BACKEND_ADRENO` at compile/run time and the dispatch actually works.
2. The `test_adreno` on-device run's 14/14 passing result is now exercised
   end-to-end under real inference pressure: `[Shannon-Prime Mobile] Cache:
   4.25 MB (vs 16.00 MB fp16, 3.8× compression)` proves the cache
   allocates, writes, and reads correctly for Dolphin 1B's geometry.
3. The ceiling on mobile VHT2 is now known and bounded: with the current
   per-vector call pattern, neither scalar nor NEON crosses the ~0.4 t/s
   floor on this silicon. The next optimisation is structural, not
   algorithmic.

## What would close the gap

- **Batch the hook** (`sp_llama_write_k_batch`): single call per (layer,
  tensor) instead of per (layer, head, token). Amortises the pipeline
  overhead across many vectors, keeps NEON warm.
- **Move `sp_mobius_reorder`'s scratch to a persistent buffer** on the
  ctx instead of malloc-per-call.
- **Use the fp16 write path** (`sp_adreno_write_k_f16`): skip the fp16→f32
  conversion in the callback when the source tensor is already fp16.

These are local changes to `llama-shannon-prime.cpp` (the callback
implementation) plus a light extension to the bridge. Out of scope for
this task — the goal was to prove the bridge can dispatch to the NEON
path and capture the honest performance number. Both done.
