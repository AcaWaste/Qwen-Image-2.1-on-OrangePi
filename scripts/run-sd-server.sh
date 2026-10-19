#!/bin/sh
# Start sd-server on the 310B1 NPU with parameters kept resident between requests.
#
# Usage:
#   sh scripts/run-sd-server.sh <sd-cpp-dir> <models-dir> [listen-port]
#
# Then:
#   curl -s http://127.0.0.1:<port>/sdapi/v1/txt2img -H 'Content-Type: application/json' -d @request.json

set -e

if [ $# -lt 2 ]; then
    echo "usage: run-sd-server.sh <sd-cpp-dir> <models-dir> [listen-port]"
    exit 1
fi

SDCPP=$1
MODELS=$2
PORT=${3:-1241}
BIN=$SDCPP/build-cann/bin/sd-server

if [ ! -x "$BIN" ]; then
    echo "missing binary: $BIN"
    exit 2
fi

if pgrep -x sd-cli >/dev/null; then
    echo "busy: sd-cli is already running"
    exit 3
fi
if pgrep -x sd-server >/dev/null; then
    echo "busy: sd-server is already running"
    exit 3
fi

. /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2.6 GB stays resident; the 3rd 1-step 256x256 request drops 44.58 -> 28.08 s.
SD_KEEP_PARAMS=1 "$BIN" -l 127.0.0.1 --listen-port "$PORT" \
  --diffusion-model "$MODELS/diffusion_models/qwen_image_2.1_UC_f16.gguf" \
  --vae             "$MODELS/vae/qwen_image_2.1_vae_f16.safetensors" \
  --llm             "$MODELS/text_encoders/Qwen3VL-8B-Instruct-Q4_K_M.gguf" \
  --threads 3 --vae-conv-direct --fa --sampling-method euler --cfg-scale 6.0 \
  --backend        "te=cpu,diffusion=CANN0,vae=CANN0" \
  --params-backend "diffusion=CANN0,te=cpu,vae=CANN0"
