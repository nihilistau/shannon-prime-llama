#!/usr/bin/env bash
# Push binaries + model to phone, then run.
set -euo pipefail

ADB="${ADB:-D:/Files/Android/pt-latest/platform-tools/adb.exe}"
OUT="${OUT:-D:/F/shannon-prime-repos/shannon-prime-llama/android_build}"
MODEL="${MODEL:-D:/Files/Models/Dolphin3.0-Llama3.2-1B.Q8_0.gguf}"
DEV="/data/local/tmp/sp"

echo "ADB: $ADB"
"$ADB" devices -l
"$ADB" shell "mkdir -p $DEV"

if [ -f "$OUT/test_adreno" ]; then
  echo "=== pushing test_adreno ==="
  "$ADB" push "$OUT/test_adreno" $DEV/
  "$ADB" shell "chmod 755 $DEV/test_adreno"
fi

if [ -f "$OUT/llama-cli" ]; then
  echo "=== pushing llama-cli ==="
  "$ADB" push "$OUT/llama-cli" $DEV/
  "$ADB" shell "chmod 755 $DEV/llama-cli"
fi

# Push model unless already there (and size matches)
MODEL_NAME=$(basename "$MODEL")
LOCAL_SZ=$(stat -c %s "$MODEL")
REMOTE_SZ=$("$ADB" shell "stat -c %s $DEV/$MODEL_NAME 2>/dev/null" | tr -d '\r')
if [ "$LOCAL_SZ" != "$REMOTE_SZ" ]; then
  echo "=== pushing model (${LOCAL_SZ} bytes) ==="
  "$ADB" push "$MODEL" "$DEV/$MODEL_NAME"
else
  echo "=== model already on device (${LOCAL_SZ} bytes) ==="
fi

echo "=== ready on device: $DEV ==="
"$ADB" shell "ls -la $DEV"
