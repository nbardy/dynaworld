#!/usr/bin/env bash
set -euo pipefail

STAGE="${1:-all}"
if [[ "$#" -gt 0 ]]; then
  shift
fi
CONFIG="src/dataset_configs/youtube_curated_spans_64_4fps_16f.jsonc"

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
      echo "Usage: $0 [all|materialize|download|build-clips] [--overwrite] [--overwrite-raw] [--no-overwrite-clips]" >&2
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
  uv run --with yt-dlp python src/dataset_pipeline/youtube_curated_spans.py "${args[@]}"
}

case "$STAGE" in
  all)
    run_stage materialize
    run_stage download raw
    run_stage build-clips clips
    ;;
  materialize)
    run_stage materialize
    ;;
  download)
    run_stage download raw
    ;;
  build-clips)
    run_stage build-clips clips
    ;;
  *)
    echo "Unknown stage: $STAGE" >&2
    echo "Usage: $0 [all|materialize|download|build-clips] [--overwrite] [--overwrite-raw] [--no-overwrite-clips]" >&2
    exit 1
    ;;
esac
