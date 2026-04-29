#!/usr/bin/env bash
# Delete Neural 3D Video raw/*.zip archives once the matching extracted scene
# is on disk. Non-interactive: this script always runs with --execute. Run the
# python stage directly (without --execute) for a dry-run / size preview:
#
#     uv run python src/dataset_pipeline/neural_3d_video.py cleanup-zips \
#         --config src/dataset_configs/neural_3d_video_seed.jsonc
#
# Re-extraction requires re-downloading from the GitHub release; the pipeline
# can do that with the `download` stage.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG="src/dataset_configs/neural_3d_video_seed.jsonc"
# Tolerate --execute (other cleanup scripts accept it); a positional arg is
# treated as a config path override.
for arg in "$@"; do
    case "$arg" in
        --execute) ;;
        *) CONFIG="$arg" ;;
    esac
done

uv run python src/dataset_pipeline/neural_3d_video.py cleanup-zips \
    --config "$CONFIG" \
    --execute
