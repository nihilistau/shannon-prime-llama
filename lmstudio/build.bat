@echo off
REM Shannon-Prime LM Studio runtime builder
REM Copyright (C) 2026 Ray Daniels. All Rights Reserved.
REM Licensed under AGPLv3.
REM
REM Produces a drop-in llama.dll + ggml.dll for LM Studio v2.14.0+.
REM Stock ggml-base.dll, ggml-cpu.dll, and ggml-cuda.dll from LM Studio
REM are retained — only llama.dll and ggml.dll are replaced.
REM
REM Prerequisites:
REM   - Visual Studio 2019+ Build Tools (cl.exe, link.exe, rc.exe)
REM   - CUDA Toolkit 12.x+ (nvcc)
REM   - CMake 3.14+ and Ninja
REM   - llama.cpp checkout at b8861 with the full-engine patch applied
REM
REM Usage:
REM   build.bat <llama-cpp-dir> <shannon-prime-llama-dir> [output-dir]
REM
REM Example:
REM   build.bat C:\llama-cpp-sp C:\shannon-prime-llama C:\lmstudio-runtime

setlocal enabledelayedexpansion

if "%~1"=="" (
    echo Usage: build.bat ^<llama-cpp-dir^> ^<shannon-prime-llama-dir^> [output-dir]
    echo.
    echo   llama-cpp-dir          Path to patched llama.cpp checkout (b8861 + full-engine patch)
    echo   shannon-prime-llama-dir Path to this repo (shannon-prime-llama)
    echo   output-dir             Where to put the DLLs (default: .\output)
    exit /b 1
)

set LLAMA_DIR=%~1
set SP_DIR=%~2
set OUT_DIR=%~3
if "%OUT_DIR%"=="" set OUT_DIR=%~dp0output

echo [SP] Shannon-Prime LM Studio builder
echo [SP] llama.cpp:          %LLAMA_DIR%
echo [SP] shannon-prime-llama: %SP_DIR%
echo [SP] output:             %OUT_DIR%
echo.

REM Set up VS developer environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
if errorlevel 1 (
    call "C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
)
if errorlevel 1 (
    echo [SP] ERROR: Could not find Visual Studio Build Tools. Install VS 2019+ BuildTools.
    exit /b 1
)

REM Find CUDA
if defined CUDA_PATH (
    set "PATH=%CUDA_PATH%\bin;%PATH%"
) else (
    for %%d in (v13.2 v13.1 v13.0 v12.6 v12.4 v12.1) do (
        if exist "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%d\bin\nvcc.exe" (
            set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\%%d"
            set "PATH=!CUDA_PATH!\bin;%PATH%"
            goto :cuda_found
        )
    )
    echo [SP] WARNING: No CUDA toolkit found. Building without SP_CUDA.
    set SP_CUDA_FLAG=-DSP_CUDA=OFF
    goto :configure
)
:cuda_found
echo [SP] CUDA: %CUDA_PATH%
set SP_CUDA_FLAG=-DSP_CUDA=ON "-DCMAKE_CUDA_COMPILER=%CUDA_PATH%/bin/nvcc.exe" -DCMAKE_CUDA_ARCHITECTURES=native "-DCMAKE_CUDA_FLAGS=--use-local-env"

:configure
set BUILD_DIR=%LLAMA_DIR%\build_sp_lmstudio

echo [SP] Configuring...
cmake -G Ninja ^
  -B "%BUILD_DIR%" ^
  -S "%LLAMA_DIR%" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_C_COMPILER=cl ^
  -DCMAKE_CXX_COMPILER=cl ^
  -DGGML_CUDA=OFF ^
  -DLLAMA_SHANNON_PRIME=ON ^
  "-DSHANNON_PRIME_DIR=%SP_DIR%" ^
  %SP_CUDA_FLAG% ^
  -DLLAMA_BUILD_TESTS=OFF ^
  -DLLAMA_BUILD_EXAMPLES=OFF ^
  -DLLAMA_BUILD_TOOLS=OFF ^
  -DLLAMA_BUILD_SERVER=OFF ^
  -DBUILD_SHARED_LIBS=ON
if errorlevel 1 (
    echo [SP] ERROR: CMake configure failed.
    exit /b 1
)

echo [SP] Building llama.dll...
ninja -C "%BUILD_DIR%" -j%NUMBER_OF_PROCESSORS% bin/llama.dll
if errorlevel 1 (
    echo [SP] ERROR: Build failed.
    exit /b 1
)

echo [SP] Copying DLLs to %OUT_DIR%...
if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
copy /y "%BUILD_DIR%\bin\llama.dll" "%OUT_DIR%\" >nul
copy /y "%BUILD_DIR%\bin\ggml.dll" "%OUT_DIR%\" >nul

echo.
echo [SP] Build complete. Drop these into your LM Studio runtime folder:
echo [SP]   %OUT_DIR%\llama.dll
echo [SP]   %OUT_DIR%\ggml.dll
echo.
echo [SP] LM Studio runtime folder is typically:
echo [SP]   %%USERPROFILE%%\.cache\lm-studio\extensions\backends\llama.cpp-win-x86_64-nvidia-cuda12-avx2-2.14.0\
echo.
echo [SP] Keep the stock ggml-base.dll, ggml-cpu.dll, and ggml-cuda.dll from LM Studio.
echo [SP] Only replace llama.dll and ggml.dll with the ones built here.
