#!/usr/bin/env bash
export NDK=/mnt/d/Files/Android/android-ndk-r27d
export SP=/mnt/d/F/shannon-prime-repos/shannon-prime-llama
export SRC=/mnt/d/F/shannon-prime-repos/sp-model-test/llama-cpp
export HEXAGON_SDK=/mnt/c/Qualcomm/Hexagon_SDK/5.5.6.0
exec ./build_llama_cli_hexagon.sh "$@"