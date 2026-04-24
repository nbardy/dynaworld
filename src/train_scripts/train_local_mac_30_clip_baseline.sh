#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$#" -gt 1 ]]; then
  echo "Usage: $0 [config.jsonc]" >&2
  exit 1
fi

CONFIG_PATH="${1:-src/train_configs/local_mac_tiny_30_video_token_smoke.jsonc}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config: $CONFIG_PATH" >&2
  exit 1
fi

if [[ ! -f data/youtube_scene_distinct/clip_sets/youtube_scene_distinct_30_64_4fps_16f/manifest.jsonl ]]; then
  echo "Missing scene-distinct local Mac 30-clip dataset." >&2
  echo "Build it with: ./src/dataset_scripts/youtube_scene_distinct_30_seed.sh" >&2
  exit 1
fi

echo "Config: $CONFIG_PATH"
uv run python src/train/train_video_token_implicit_dynamic.py "$CONFIG_PATH"
