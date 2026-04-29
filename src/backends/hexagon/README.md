# Hexagon DSP Backend

**Status: scaffolding. The header declares the API; the implementation is a stub. See [BACKEND-HEXAGON.md](../../docs/BACKEND-HEXAGON.md) for the build-out plan.**

This directory will hold the Hexagon-DSP-specific implementation of Shannon-Prime's KV cache compression kernels. The structure when complete:

```
hexagon/
  shannon_prime_hexagon.h    Public API (this commit)
  shannon_prime_hexagon.c    Host-side dispatch (this commit, stub)
  shannon_prime_hexagon.idl  FastRPC interface definition (M1)
  dsp/
    shannon_prime_dsp.c      Hexagon-side scalar dispatcher (M2)
    sp_band_hvx.c            HVX-vectorised band quantize/dequantize (M3)
    sp_vht2_hvx.c            HVX-vectorised butterfly (M3)
  Makefile / CMakeLists.txt  Build glue for hexagon-clang (M1)
```

The header is wired into the existing capability-detection plumbing in `backends/adreno/shannon_prime_adreno.h::sp_mobile_caps_t` (the `has_hexagon` / `hexagon_version` / `has_hvx` fields). When the DSP is unavailable or the kernels aren't built, `sp_hexagon_init()` returns `NULL` and consumers fall back to the Adreno or CPU path.

Target SDK: Qualcomm Hexagon SDK 5.x (validated on 5.5.6.0). Target hardware ladder: V69 (Snapdragon 8 Gen 1, S22 Ultra) on M3, V73+ (8 Gen 2+) on follow-up milestones.

For a step-by-step pickup guide — what to install, how to set up the environment, how to verify the toolchain, what each milestone delivers — read `docs/BACKEND-HEXAGON.md` in the parent.
