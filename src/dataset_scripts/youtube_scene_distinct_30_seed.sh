#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="src/dataset_configs/youtube_scene_distinct_30_64_4fps_16f.jsonc"
STAGE="${1:-all}"
if [[ "$#" -gt 0 ]]; then
  shift
fi

OVERWRITE_RAW="${OVERWRITE_RAW:-0}"
OVERWRITE_CLIPS="${OVERWRITE_CLIPS:-1}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --overwrite)
      OVERWRITE_RAW=1
      OVERWRITE_CLIPS=1
      ;;
    --overwrite-raw)
      OVERWRITE_RAW=1
      ;;
    --no-overwrite-clips)
      OVERWRITE_CLIPS=0
      ;;
    *)
      echo "Usage: $0 [all|search|download|segment|build-clips] [--overwrite] [--overwrite-raw] [--no-overwrite-clips]" >&2
      exit 1
      ;;
  esac
  shift
done

run_stage() {
  local stage="$1"
  local overwrite_mode="${2:-none}"
  local args=("$stage" --config "$CONFIG")
  if [[ "$overwrite_mode" == "raw" && "$OVERWRITE_RAW" == "1" ]]; then
    args+=(--overwrite)
  elif [[ "$overwrite_mode" == "clips" && "$OVERWRITE_CLIPS" == "1" ]]; then
    args+=(--overwrite)
  fi
  uv run --with yt-dlp python src/dataset_pipeline/youtube_ingest.py "${args[@]}"
}

case "$STAGE" in
  all)
    run_stage search
    run_stage download raw
    run_stage segment
    run_stage build-clips clips
    ;;
  search)
    run_stage search
    ;;
  download)
    run_stage download raw
    ;;
  segment)
    run_stage segment
    ;;
  build-clips)
    run_stage build-clips clips
    ;;
  *)
    echo "Unknown stage: $STAGE" >&2
    echo "Usage: $0 [all|search|download|segment|build-clips] [--overwrite] [--overwrite-raw] [--no-overwrite-clips]" >&2
    exit 1
    ;;
esac
