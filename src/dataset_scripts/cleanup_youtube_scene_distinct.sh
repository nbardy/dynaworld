#!/usr/bin/env bash
# Cleanup intermediates for the scene-distinct YouTube pipeline.
# Default is DRY RUN. Pass --execute to actually delete.
# Both 64px and 256px clip sets share data/youtube_scene_distinct/, so this
# script runs cleanup against BOTH dataset configs in turn -- they share raw/
# and segments/ but each has its own clip_sets/<dataset_name>/ subdirectory.
# Raw mp4s are NOT deleted by default; pass --include-raw to also wipe them
# (requires yt-dlp + network to recover). Pass --include-segments to also
# delete segments/ (recoverable from raw via the segment stage).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

for CONFIG in \
  "src/dataset_configs/youtube_scene_distinct_30_64_4fps_16f.jsonc" \
  "src/dataset_configs/youtube_scene_distinct_30_256_4fps_16f.jsonc"; do
  echo "## $CONFIG"
  uv run python src/dataset_pipeline/youtube_ingest.py \
    cleanup-intermediates --config "$CONFIG" "$@"
done
