#!/usr/bin/env bash
# Cross-compile the patched llama.cpp llama-cli for arm64 Android.
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
    -DGGML_OPENMP=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_SHANNON_PRIME=ON \
    -DSHANNON_PRIME_DIR="$SP"

cmake --build "$BUILD" --target llama-cli -j

ls -la "$BUILD/bin/llama-cli" 2>/dev/null || ls -la "$BUILD/bin/llama-cli.exe" 2>/dev/null || {
  echo "NOT FOUND: searching for llama-cli"
  find "$BUILD" -name 'llama-cli*' -type f
}
