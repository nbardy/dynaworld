#!/usr/bin/env bash
# Cleanup intermediates for the high-camera-motion YouTube pipeline.
# Default is DRY RUN. Pass --execute to actually delete.
# Raw mp4s are NOT deleted by default; pass --include-raw to also wipe them
# (requires yt-dlp + network to recover).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="src/dataset_configs/youtube_high_camera_motion_seed.jsonc"

uv run python src/dataset_pipeline/youtube_ingest.py \
  cleanup-intermediates --config "$CONFIG" "$@"
