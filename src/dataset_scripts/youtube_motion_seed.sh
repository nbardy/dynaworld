#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${1:-src/dataset_configs/youtube_high_camera_motion_seed.jsonc}"

uv run --with yt-dlp python src/dataset_pipeline/youtube_ingest.py search --config "$CONFIG"
uv run --with yt-dlp python src/dataset_pipeline/youtube_ingest.py download --config "$CONFIG"
uv run --with yt-dlp python src/dataset_pipeline/youtube_ingest.py segment --config "$CONFIG"
uv run --with yt-dlp python src/dataset_pipeline/youtube_ingest.py build-clips --config "$CONFIG" --overwrite
