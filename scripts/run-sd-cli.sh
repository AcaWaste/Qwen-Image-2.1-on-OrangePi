#!/bin/sh
# Run one Qwen-Image-2.1 generation on the 310B1 NPU.
#
# Usage:
#   sh scripts/run-sd-cli.sh <sd-cpp-dir> <models-dir> <output.png> \
#                            <width> <height> <steps> <cfg-scale> <seed> <prompt>
#
# Example:
#   sh scripts/run-sd-cli.sh build-cann models output/cat.png \
#      512 512 20 6.0 42 "a lovely cat holding a sign says qwen2.1.cpp"

set -e

if [ $# -lt 9 ]; then
    echo "usage: run-sd-cli.sh <sd-cpp-dir> <models-dir> <out.png> <W> <H> <steps> <cfg> <seed> <prompt>"
    exit 1
fi

SDCPP=$1
MODELS=$2
OUT=$3
WIDTH=$4
HEIGHT=$5
STEPS=$6
CFG=$7
SEED=$8
PROMPT=$9

BIN=$SDCPP/build-cann/bin/sd-cli
if [ ! -x "$BIN" ]; then
    echo "missing binary: $BIN"
    exit 2
fi

# Only one sd-cli / sd-server may run on this board at a time.
if pgrep -x sd-cli >/dev/null; then
    echo "busy: sd-cli is already running"
    exit 3
fi
if pgrep -x sd-server >/dev/null; then
    echo "busy: sd-server is already running"
    exit 3
fi

. /usr/local/Ascend/ascend-toolkit/set_env.sh

mkdir -p "$(dirname "$OUT")" || exit 4

"$BIN" \
  --diffusion-model "$MODELS/diffusion_models/qwen_image_2.1_UC_f16.gguf" \
  --vae             "$MODELS/vae/qwen_image_2.1_vae_f16.safetensors" \
  --llm             "$MODELS/text_encoders/Qwen3VL-8B-Instruct-Q4_K_M.gguf" \
  -p "$PROMPT" \
  --steps "$STEPS" --cfg-scale "$CFG" --sampling-method euler --threads 3 \
  --vae-conv-direct --fa -s "$SEED" -v -W "$WIDTH" -H "$HEIGHT" \
  --backend        "te=cpu,diffusion=CANN0,vae=CANN0" \
  --params-backend "diffusion=CANN0,te=cpu,vae=CANN0" \
  -o "$OUT"
