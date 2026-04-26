#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="src/dataset_configs/multicam_val_v1_128_4fps_16f.jsonc"
STAGE="${1:-all}"
if [[ "$#" -gt 0 ]]; then
  shift
fi

OVERWRITE="${OVERWRITE:-1}"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --overwrite)
      OVERWRITE=1
      ;;
    --no-overwrite)
      OVERWRITE=0
      ;;
    *)
      echo "Usage: $0 [all|download-aist|build|inspect] [--overwrite] [--no-overwrite]" >&2
      exit 1
      ;;
  esac
  shift
done

ARGS=("$STAGE" --config "$CONFIG")
if [[ "$OVERWRITE" == "1" ]]; then
  ARGS+=(--overwrite)
fi

uv run python src/dataset_pipeline/multicam_val.py "${ARGS[@]}"
