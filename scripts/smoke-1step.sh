#!/bin/sh
# 1-step regression check for the 310B1 backend.
#
# Expected latent md5:
#   b747de1e3b97409172bbcc2d8fba39ad
#
# Usage:
#   sh scripts/smoke-1step.sh <sd-cpp-dir> <models-dir> [output.png]

set -e

if [ $# -lt 2 ]; then
    echo "usage: smoke-1step.sh <sd-cpp-dir> <models-dir> [output.png]"
    exit 1
fi

SDCPP=$1
MODELS=$2
OUT=${3:-$(pwd)/smoke-1step.png}
DUMP=$(mktemp -d /tmp/qwen-smoke.XXXXXX) || exit 2
BIN=$SDCPP/build-cann/bin/sd-cli

if [ ! -x "$BIN" ]; then
    echo "missing binary: $BIN"
    exit 3
fi

if pgrep -x sd-cli >/dev/null; then
    echo "busy: sd-cli is already running"
    exit 4
fi
if pgrep -x sd-server >/dev/null; then
    echo "busy: sd-server is already running"
    exit 4
fi

. /usr/local/Ascend/ascend-toolkit/set_env.sh

SD_DUMP_DIR="$DUMP" "$BIN" \
  --diffusion-model "$MODELS/diffusion_models/qwen_image_2.1_UC_f16.gguf" \
  --vae             "$MODELS/vae/qwen_image_2.1_vae_f16.safetensors" \
  --llm             "$MODELS/text_encoders/Qwen3VL-8B-Instruct-Q4_K_M.gguf" \
  -p "a lovely cat holding a sign says qwen2.1.cpp" \
  --steps 1 --cfg-scale 6.0 --sampling-method euler --threads 3 \
  --vae-conv-direct --fa -s 42 -v -W 256 -H 256 \
  --backend        "te=cpu,diffusion=CANN0,vae=CANN0" \
  --params-backend "diffusion=CANN0,te=cpu,vae=CANN0" \
  -o "$OUT"

GOT=$(md5sum "$DUMP/latent.bin" | cut -d' ' -f1)
echo "latent md5 = $GOT"
echo "dump dir   = $DUMP"

if [ "$GOT" = "b747de1e3b97409172bbcc2d8fba39ad" ]; then
    echo "SMOKE PASS"
    exit 0
fi

echo "SMOKE FAIL (expected b747de1e3b97409172bbcc2d8fba39ad)"
exit 5
