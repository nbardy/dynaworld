#!/usr/bin/env bash
# Reclaim disk by deleting DeepView raw archive zips whose `extracted/<scene>/`
# tree has a populated `models.json`. Default is dry-run; pass `--execute` to
# actually delete. Always prints reclaimable bytes and per-archive verdicts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="src/dataset_configs/deepview_video_seed.jsonc"
ARGS=(cleanup-zips --config "$CONFIG")
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --execute)
      ARGS+=(--execute)
      ;;
    --all-scenes)
      ARGS+=(--all-scenes)
      ;;
    *)
      echo "Usage: $0 [--execute] [--all-scenes]" >&2
      exit 1
      ;;
  esac
  shift
done

uv run python src/dataset_pipeline/deepview_video.py "${ARGS[@]}"
