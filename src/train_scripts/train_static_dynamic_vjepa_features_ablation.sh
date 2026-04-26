#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_250="src/train_configs/local_mac_ablate_time_static_dynamic_96_32_crossattn4_precomputed_vjepa2_1_vitb_384_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc"
CONFIG_1000="src/train_configs/local_mac_ablate_time_static_dynamic_96_32_crossattn4_precomputed_vjepa2_1_vitb_384_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats_1000step.jsonc"

MODE="${1:-250}"
case "$MODE" in
  250|base)
    CONFIG="$CONFIG_250"
    ;;
  1000|long)
    CONFIG="$CONFIG_1000"
    ;;
  *)
    echo "Usage: $0 [250|1000]" >&2
    exit 1
    ;;
esac

PYTHONPATH=src/train uv run python src/train/train_precomputed_feature_implicit_dynamic.py "$CONFIG"
