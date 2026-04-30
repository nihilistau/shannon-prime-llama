#!/usr/bin/env bash
# Cross-compile the patched llama.cpp llama-cli for arm64 Android with the
# Shannon-Prime Hexagon cDSP backend wired in.
#
# Differs from build_llama_cli.sh in three places:
#   1. SRC defaults to D:/F/shannon-prime-repos/sp-model-test/llama-cpp (the
#      fresh ggml-org/llama.cpp@b8861 tree we now patch into) instead of the
#      old D:/F/llama-cpp-sp working dir.
#   2. Each invocation resets SRC to upstream HEAD and re-applies the patch
#      so the build is reproducible from clean state.
#   3. -DSP_HAVE_HEXAGON=ON + Hexagon SDK paths + librpcmem.a / libcdsprpc.so
#      passed at link time.
#
# Outputs a llama-cli binary and the .so deps it dynamically links against.
# To run on-device you also need libsp_hex_skel.so (built separately via the
# Hexagon SDK's build.cmd at C:/Qualcomm/Hexagon_IDE/S22U) pushed somewhere
# on ADSP_LIBRARY_PATH on the phone.

set -euo pipefail

NDK="${NDK:-D:/Files/Android/android-ndk-r27d}"
SP="${SP:-D:/F/shannon-prime-repos/shannon-prime-llama}"
SRC="${SRC:-D:/F/shannon-prime-repos/sp-model-test/llama-cpp}"
HEXAGON_SDK="${HEXAGON_SDK:-C:/Qualcomm/Hexagon_SDK/5.5.6.0}"
BUILD="${BUILD:-${SRC}/build-android-hexagon}"
API="${API:-28}"

PATCH="$SP/patches/llama-cpp-b8861-full-engine.patch"

# ── Sanity-check inputs ─────────────────────────────────────────────
[ -d "$NDK" ]         || { echo "ERROR: NDK not found at $NDK"; exit 1; }
[ -d "$SP" ]          || { echo "ERROR: shannon-prime-llama checkout not found at $SP"; exit 1; }
[ -d "$SRC" ]         || { echo "ERROR: llama.cpp source not found at $SRC"; exit 1; }
[ -d "$HEXAGON_SDK" ] || { echo "ERROR: Hexagon SDK not found at $HEXAGON_SDK"; exit 1; }
[ -f "$PATCH" ]       || { echo "ERROR: patch not found at $PATCH"; exit 1; }

RPCMEM_INC="$HEXAGON_SDK/ipc/fastrpc/rpcmem/inc"
RPCMEM_LIB="$HEXAGON_SDK/ipc/fastrpc/rpcmem/prebuilt/android_aarch64/rpcmem.a"
CDSPRPC_LIB="$HEXAGON_SDK/ipc/fastrpc/remote/ship/android_aarch64/libcdsprpc.so"
[ -f "$RPCMEM_LIB" ]  || { echo "ERROR: rpcmem.a not found at $RPCMEM_LIB"; exit 1; }
[ -f "$CDSPRPC_LIB" ] || { echo "ERROR: libcdsprpc.so not found at $CDSPRPC_LIB"; exit 1; }

echo "NDK:         $NDK"
echo "SP:          $SP"
echo "SRC:         $SRC"
echo "HEXAGON_SDK: $HEXAGON_SDK"
echo "BUILD:       $BUILD"
echo "API:         $API"
echo "PATCH:       $PATCH"
echo ""

# ── Step 1: reset SRC + apply patch ─────────────────────────────────
echo "[1/3] Reset $SRC to upstream HEAD and apply Shannon-Prime patch"
cd "$SRC"
git reset --hard HEAD >/dev/null
git clean -fd >/dev/null
git apply "$PATCH"
echo "      patch applied OK"
echo ""

# ── Step 2: CMake configure ─────────────────────────────────────────
echo "[2/3] Configure CMake (Hexagon enabled)"
rm -rf "$BUILD"

# `-march=armv8.2-a+fp16+dotprod` — same flags as build_llama_cli.sh; needed
# for the Adreno NEON backend (FP16 vector arithmetic + sdot/udot for i8 quant).
# The rpcmem inc path goes onto the global C/CXX flags so any TU that
# pulls in shannon_prime_hexagon.h transitively finds rpcmem.h.
EXTRA_C="-march=armv8.2-a+fp16+dotprod -I${RPCMEM_INC}"

# `RPCMEM_LIB` resolves rpcmem_alloc/free and is a static archive.
# `CDSPRPC_LIB` resolves remote_handle64 etc. — dynamic .so on-device.
# `-llog` resolves __android_log_print which the Hexagon SDK's bundled
# rpcmem_android.c uses for diagnostic logging (linker error otherwise:
# "undefined symbol: __android_log_print referenced by rpcmem_android.c").
# Both go on the executable AND shared linker flags so they're picked up
# when the final llama-cli binary or libllama.so is linked.
LINK_LIBS="${RPCMEM_LIB} ${CDSPRPC_LIB} -llog"

cmake -S "$SRC" -B "$BUILD" \
    -G "Ninja" \
    -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-$API \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="$EXTRA_C" \
    -DCMAKE_CXX_FLAGS="$EXTRA_C" \
    -DCMAKE_EXE_LINKER_FLAGS="$LINK_LIBS" \
    -DCMAKE_SHARED_LINKER_FLAGS="$LINK_LIBS" \
    -DGGML_CUDA=OFF \
    -DGGML_VULKAN=OFF \
    -DGGML_OPENMP=OFF \
    -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_SHANNON_PRIME=ON \
    -DSHANNON_PRIME_DIR="$SP" \
    -DSP_CUDA=OFF \
    -DSP_HAVE_HEXAGON=ON \
    -DHEXAGON_SDK_ROOT="$HEXAGON_SDK"
echo ""

# ── Step 3: build llama-cli ─────────────────────────────────────────
echo "[3/3] Build llama-cli"
cmake --build "$BUILD" --target llama-cli -j
echo ""

# ── Verify output ───────────────────────────────────────────────────
BIN="$BUILD/bin/llama-cli"
if [ -f "$BIN" ]; then
  echo "OK: $BIN"
  ls -la "$BIN"
  echo ""
  echo "Next steps:"
  echo "  1. Build libsp_hex_skel.so via Hexagon SDK build.cmd (skel=DSP-side)"
  echo "  2. adb push $BIN /data/local/tmp/sp/"
  echo "  3. adb push $BUILD/bin/lib*.so /data/local/tmp/sp/lib/"
  echo "  4. adb push libsp_hex_skel.so /data/local/tmp/sp/  (or to ADSP_LIBRARY_PATH)"
  echo "  5. SHANNON_PRIME_ENABLED=1 SHANNON_PRIME_BACKEND=hexagon \\"
  echo "     SHANNON_PRIME_VERBOSE=1 ADSP_LIBRARY_PATH=/data/local/tmp/sp \\"
  echo "     ./llama-cli -m model.gguf -p 'The capital of France is' -n 30"
else
  echo "ERROR: llama-cli not found at $BIN"
  echo "Searching:"
  find "$BUILD" -name 'llama-cli*' -type f
  exit 1
fi
