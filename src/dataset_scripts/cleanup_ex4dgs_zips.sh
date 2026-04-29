#!/usr/bin/env bash
# Reclaim disk by deleting Ex4DGS pretrained-checkpoint raw archive zips whose
# `extracted/<bundle>/` directory has a populated `cameras.json` +
# `mean_metrics.json`. Default is dry-run; pass `--execute` to actually delete.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="src/dataset_configs/ex4dgs_pretrained_val_seed.jsonc"
ARGS=(cleanup-zips --config "$CONFIG")
while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --execute)
      ARGS+=(--execute)
      ;;
    *)
      echo "Usage: $0 [--execute]" >&2
      exit 1
      ;;
  esac
  shift
done

uv run python src/dataset_pipeline/ex4dgs_pretrained.py "${ARGS[@]}"
