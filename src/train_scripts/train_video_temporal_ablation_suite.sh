#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

BASE_CONFIG="src/train_configs/local_mac_ablate_time_crossattn1_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc"
CROSS2_CONFIG="src/train_configs/local_mac_ablate_time_crossattn2_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc"
CROSS4_CONFIG="src/train_configs/local_mac_ablate_time_crossattn4_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc"
SIN4_CONFIG="src/train_configs/local_mac_ablate_time_sinusoidal_crossattn4_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc"
SPLIT4_CONFIG="src/train_configs/local_mac_ablate_time_static_dynamic_96_32_crossattn4_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc"

MODE="${1:-all}"
case "$MODE" in
  base)
    CONFIGS=("$BASE_CONFIG")
    ;;
  cross2)
    CONFIGS=("$CROSS2_CONFIG")
    ;;
  cross4)
    CONFIGS=("$CROSS4_CONFIG")
    ;;
  sin4|sinusoidal4)
    CONFIGS=("$SIN4_CONFIG")
    ;;
  split4|static-dynamic4)
    CONFIGS=("$SPLIT4_CONFIG")
    ;;
  depth)
    CONFIGS=("$CROSS2_CONFIG" "$CROSS4_CONFIG")
    ;;
  all)
    CONFIGS=("$BASE_CONFIG" "$CROSS2_CONFIG" "$CROSS4_CONFIG" "$SIN4_CONFIG" "$SPLIT4_CONFIG")
    ;;
  *)
    echo "Usage: $0 [base|cross2|cross4|sin4|split4|depth|all]" >&2
    exit 1
    ;;
esac

for config in "${CONFIGS[@]}"; do
  if [[ ! -f "$config" ]]; then
    echo "Missing config: $config" >&2
    exit 1
  fi
  echo "Config: $config"
  PYTHONPATH=src/train uv run python src/train/train.py "$config"
done
