#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="src/dataset_configs/youtube_scene_distinct_30_64_4fps_16f.jsonc"
STAGE="${1:-all}"

uv run --with yt-dlp python src/dataset_pipeline/youtube_ingest.py "$STAGE" \
  --config "$CONFIG" \
  --overwrite
