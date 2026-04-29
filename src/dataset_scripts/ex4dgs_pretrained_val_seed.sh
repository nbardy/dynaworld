#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="src/dataset_configs/ex4dgs_pretrained_val_seed.jsonc"
STAGE="${1:-all}"
if [[ "$#" -gt 0 ]]; then
  shift
fi

ARGS=("$STAGE" --config "$CONFIG")
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --overwrite)
      ARGS+=(--overwrite)
      ;;
    --execute)
      ARGS+=(--execute)
      ;;
    *)
      echo "Usage: $0 [list-assets|download|extract|inspect|validate-local|cleanup-zips|all] [--overwrite] [--execute]" >&2
      exit 1
      ;;
  esac
  shift
done

uv run python src/dataset_pipeline/ex4dgs_pretrained.py "${ARGS[@]}"
