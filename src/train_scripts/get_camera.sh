#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

DEFAULT_VIDEO="test_data/test_video_small.mp4"
DEFAULT_OUTPUT="test_data/dust3r_outputs/test_video_small_all_frames"

uv run python src/train/run_dust3r_video.py \
  --video "$DEFAULT_VIDEO" \
  --frame-stride 1 \
  --max-frames 0 \
  --scene-graph swin \
  --winsize 5 \
  --noncyclic \
  --output-dir "$DEFAULT_OUTPUT" \
  "$@"
