#!/usr/bin/env bash
# Cleanup intermediates for the curated YouTube spans pipeline.
# Default is DRY RUN (prints reclaimable bytes). Pass --execute to actually
# delete. Raw mp4 spans are NOT deleted by default since reproducing them
# requires yt-dlp + the network.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="src/dataset_configs/youtube_curated_spans_64_4fps_16f.jsonc"

uv run python src/dataset_pipeline/youtube_curated_spans.py \
  cleanup-intermediates --config "$CONFIG" "$@"
