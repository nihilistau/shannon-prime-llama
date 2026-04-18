#!/usr/bin/env bash
# Cross-compile the patched llama.cpp llama-perplexity for arm64 Android.
# Mirrors build_llama_cli.sh but targets the perplexity binary so we can
# run the KV-compression sweep on-device.
set -euo pipefail

NDK="${NDK:-D:/Files/Android/android-ndk-r27d}"
SRC="${SRC:-D:/F/llama-cpp-sp}"
SP="${SP:-D:/F/shannon-prime-repos/shannon-prime-llama}"
BUILD="${BUILD:-D:/F/llama-cpp-sp/build-android}"
API="${API:-24}"

echo "NDK:    $NDK"
echo "SRC:    $SRC"
echo "SP:     $SP"
echo "BUILD:  $BUILD"

# Reuse existing build-android dir when possible (CMake incremental),
# only nuke if the toolchain/ABI config has changed.
if [ ! -f "$BUILD/CMakeCache.txt" ] || ! grep -q "^ANDROID_ABI:STRING=arm64-v8a" "$BUILD/CMakeCache.txt"; then
  rm -rf "$BUILD"
  cmake -S "$SRC" -B "$BUILD" \
    -G "Ninja" \
    -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-$API \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-march=armv8.2-a+fp16+dotprod" \
    -DCMAKE_CXX_FLAGS="-march=armv8.2-a+fp16+dotprod" \
    -DGGML_CUDA=OFF \
    -DGGML_VULKAN=OFF \
    -DGGML_OPENMP=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_SHANNON_PRIME=ON \
    -DSHANNON_PRIME_DIR="$SP"
fi

cmake --build "$BUILD" --target llama-perplexity -j

BIN="$BUILD/bin/llama-perplexity"
if [ -f "$BIN" ]; then
  echo "Built: $BIN"
  ls -la "$BIN"
else
  echo "NOT FOUND: searching for llama-perplexity"
  find "$BUILD" -name 'llama-perplexity*' -type f
fi
