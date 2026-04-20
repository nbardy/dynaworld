#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

echo "Alias: preserved image-encoder implicit-camera baseline"

exec bash ./src/train_scripts/train_full_dynamic_with_implicit_camera_all_frames.sh "$@"
