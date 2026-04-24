#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${1:-src/dataset_configs/neural_3d_video_seed.jsonc}"

.venv/bin/python src/dataset_pipeline/neural_3d_video.py list-assets --config "$CONFIG"
.venv/bin/python src/dataset_pipeline/neural_3d_video.py download --config "$CONFIG"
.venv/bin/python src/dataset_pipeline/neural_3d_video.py extract --config "$CONFIG"
.venv/bin/python src/dataset_pipeline/neural_3d_video.py inspect --config "$CONFIG"
