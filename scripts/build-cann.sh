#!/bin/sh
# Build stable-diffusion.cpp with the Ascend 310B1 patches.
#
# Usage:
#   sh scripts/build-cann.sh /path/to/stable-diffusion.cpp [repo-dir]
#
# repo-dir defaults to the parent of this script's directory.

set -e

SDCPP=${1:?usage: build-cann.sh <sd-cpp-dir> [repo-dir]}
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO=${2:-$(cd "$SCRIPT_DIR/.." && pwd)}

PATCH_MAIN=$REPO/patches/0001-sd-cpp-310b-backend.patch
PATCH_GGML=$REPO/patches/0002-ggml-cann-310b-backend.patch
SHIM_DIR=$REPO/files/ggml/src/ggml-cann/aclnnop_shim

for f in "$PATCH_MAIN" "$PATCH_GGML"; do
    if [ ! -f "$f" ]; then
        echo "missing patch: $f"
        exit 1
    fi
done
if [ ! -d "$SHIM_DIR" ]; then
    echo "missing shim dir: $SHIM_DIR"
    exit 1
fi

cd "$SDCPP" || exit 1

if [ -n "$(git status --porcelain)" ]; then
    echo "working tree is not clean; refuse to patch"
    exit 2
fi

git apply "$PATCH_MAIN" || exit 3
git -C ggml apply "$PATCH_GGML" || exit 4

# aclnnop_shim is a new untracked file, so it cannot be carried by a diff.
cp -r "$SHIM_DIR" ggml/src/ggml-cann/ || exit 5

. /usr/local/Ascend/ascend-toolkit/set_env.sh

cmake -B build-cann \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CANN=ON \
      -DGGML_NATIVE=ON \
      -DUSE_ACL_GRAPH=OFF || exit 6

cmake --build build-cann -j4 --target sd-cli sd-server test-backend-ops || exit 7

ls -l build-cann/bin/sd-cli build-cann/bin/sd-server build-cann/bin/test-backend-ops
