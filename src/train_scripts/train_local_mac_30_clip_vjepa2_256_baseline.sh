#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

LOCAL_CONFIG="src/train_configs/local_mac_scene_distinct_30_local_encoder_256_fast_mac_2048splats.jsonc"
VJEPA_CONFIG="src/train_configs/local_mac_scene_distinct_30_vjepa2_vitl_fpc16_256_frozen_256_fast_mac_2048splats.jsonc"
MANIFEST="data/youtube_scene_distinct/clip_sets/youtube_scene_distinct_30_256_4fps_16f/manifest.jsonl"

MODE="${1:-both}"
case "$MODE" in
  local)
    CONFIGS=("$LOCAL_CONFIG")
    ;;
  vjepa)
    CONFIGS=("$VJEPA_CONFIG")
    ;;
  both)
    CONFIGS=("$LOCAL_CONFIG" "$VJEPA_CONFIG")
    ;;
  *)
    echo "Usage: $0 [local|vjepa|both]" >&2
    exit 1
    ;;
esac

if [[ ! -f "$MANIFEST" ]]; then
  echo "Missing 256px scene-distinct clip dataset." >&2
  echo "Build it with: ./src/dataset_scripts/youtube_scene_distinct_30_256_seed.sh build-clips" >&2
  exit 1
fi

for config in "${CONFIGS[@]}"; do
  if [[ ! -f "$config" ]]; then
    echo "Missing config: $config" >&2
    exit 1
  fi
  echo "Config: $config"
  PYTHONPATH=src/train uv run python src/train/train.py "$config"
done
