#!/usr/bin/env bash
# Cross-compile test_adreno for arm64 Android with NEON Tier 2 (fp16) enabled.
set -euo pipefail

NDK="${NDK:-D:/Files/Android/android-ndk-r27d}"
SP="${SP:-D:/F/shannon-prime-repos/shannon-prime-llama/lib/shannon-prime}"
OUT="${OUT:-D:/F/shannon-prime-repos/shannon-prime-llama/android_build}"
API="${API:-24}"

TOOL="$NDK/toolchains/llvm/prebuilt/windows-x86_64"
CC="$TOOL/bin/aarch64-linux-android${API}-clang.cmd"

echo "NDK:       $NDK"
echo "SP:        $SP"
echo "compiler:  $CC"

mkdir -p "$OUT"

# -march=armv8.2-a+fp16+dotprod enables:
#   __ARM_NEON
#   __ARM_FEATURE_FP16_VECTOR_ARITHMETIC  (Tier 2 fp16 WHT + absmax)
#   __ARM_FEATURE_DOTPROD                 (sdot/udot for i8 quant)
"$CC" \
    -O2 \
    -march=armv8.2-a+fp16+dotprod \
    -o "$OUT/test_adreno" \
    "$SP/tests/test_adreno.c" \
    "$SP/backends/adreno/shannon_prime_adreno.c" \
    "$SP/core/shannon_prime.c" \
    -lm -lc

file "$OUT/test_adreno" 2>/dev/null || true
ls -la "$OUT/test_adreno"
