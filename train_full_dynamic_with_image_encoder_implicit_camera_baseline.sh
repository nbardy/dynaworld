#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export RUN_NAME="${RUN_NAME:-dynamic-image-encoder-implicit-camera-baseline}"
echo "Alias: preserved image-encoder implicit-camera baseline"

exec bash ./train_full_dynamic_with_implicit_camera_all_frames.sh "$@"
