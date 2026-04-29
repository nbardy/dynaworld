#!/usr/bin/env bash
# Reclaim derived ViVo staging artifacts (compacted MP4s, frame manifests,
# scene inventory, logs). NEVER touches raw/ or extracted/ which require the
# MS-Form upstream access flow to recreate.
#
# Default behavior: --execute is hardcoded so this script is a one-shot
# cleanup. Use the python entrypoint directly for a dry-run preview:
#   uv run python src/dataset_pipeline/vivo.py cleanup-staging
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${1:-src/dataset_configs/vivo_seed.jsonc}"

uv run python src/dataset_pipeline/vivo.py cleanup-staging \
  --config "$CONFIG" \
  --execute
