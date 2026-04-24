#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="${1:-src/dataset_configs/vivo_seed.jsonc}"

.venv/bin/python src/dataset_pipeline/vivo.py info --config "$CONFIG"
.venv/bin/python src/dataset_pipeline/vivo.py inspect --config "$CONFIG"
