#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 <source-video-or-directory> [extra build_clip_dataset.py args...]" >&2
  exit 1
fi

uv run python src/train/build_clip_dataset.py \
  --input "$1" \
  --output-dir data/clip_sets/local_100_128_4fps_46f \
  --dataset-name local_100_128_4fps_46f \
  --target-count 100 \
  --clip-frames 46 \
  --fps 4 \
  --target-size 128 \
  "${@:2}"
