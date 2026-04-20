#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$#" -gt 1 ]]; then
  echo "Usage: $0 [config.jsonc]" >&2
  exit 1
fi

CONFIG_PATH="${1:-src/train_configs/video_token_implicit_camera_full.jsonc}"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Missing config: $CONFIG_PATH" >&2
  exit 1
fi

echo "Config: $CONFIG_PATH"
uv run python src/train/train_video_token_implicit_dynamic.py "$CONFIG_PATH"
