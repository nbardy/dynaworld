#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-all}"
CONFIG="src/dataset_configs/youtube_curated_spans_64_4fps_16f.jsonc"

uv run --with yt-dlp python src/dataset_pipeline/youtube_curated_spans.py "$STAGE" \
  --config "$CONFIG" \
  --overwrite
