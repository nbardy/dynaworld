#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

CONFIG_250="src/train_configs/local_mac_ablate_time_static_dynamic_96_32_crossattn4_precomputed_vjepa2_1_vitb_384_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc"
CONFIG_1000="src/train_configs/local_mac_ablate_time_static_dynamic_96_32_crossattn4_precomputed_vjepa2_1_vitb_384_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats_1000step.jsonc"
LOCAL_CONFIG_250="src/train_configs/local_mac_ablate_time_static_dynamic_96_32_crossattn4_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats.jsonc"
LOCAL_CONFIG_1000="src/train_configs/local_mac_ablate_time_static_dynamic_96_32_crossattn4_rgb_uniform_strong_video_implicit_128_fast_mac_8192splats_1000step.jsonc"
UNCONDITIONED_CONFIG_250="src/train_configs/local_mac_ablate_time_static_dynamic_96_32_unconditioned_strong_video_implicit_128_fast_mac_8192splats.jsonc"
UNCONDITIONED_CONFIG_1000="src/train_configs/local_mac_ablate_time_static_dynamic_96_32_unconditioned_strong_video_implicit_128_fast_mac_8192splats_1000step.jsonc"

run_video_token() {
  local config="$1"
  PYTHONPATH=src/train uv run python src/train/train_video_token_implicit_dynamic.py "$config"
}

run_precomputed() {
  local config="$1"
  PYTHONPATH=src/train uv run python src/train/train_precomputed_feature_implicit_dynamic.py "$config"
}

MODE="${1:-250}"
case "$MODE" in
  250|base|vjepa|vjepa-250)
    run_precomputed "$CONFIG_250"
    ;;
  1000|long|vjepa-1000)
    run_precomputed "$CONFIG_1000"
    ;;
  local|local-250)
    run_video_token "$LOCAL_CONFIG_250"
    ;;
  local-1000)
    run_video_token "$LOCAL_CONFIG_1000"
    ;;
  unconditioned|unconditioned-250|tokens|tokens-250)
    run_video_token "$UNCONDITIONED_CONFIG_250"
    ;;
  unconditioned-1000|tokens-1000)
    run_video_token "$UNCONDITIONED_CONFIG_1000"
    ;;
  matrix-250)
    run_video_token "$UNCONDITIONED_CONFIG_250"
    run_video_token "$LOCAL_CONFIG_250"
    run_precomputed "$CONFIG_250"
    ;;
  matrix-1000)
    run_video_token "$UNCONDITIONED_CONFIG_1000"
    run_video_token "$LOCAL_CONFIG_1000"
    run_precomputed "$CONFIG_1000"
    ;;
  all)
    run_video_token "$UNCONDITIONED_CONFIG_250"
    run_video_token "$UNCONDITIONED_CONFIG_1000"
    run_video_token "$LOCAL_CONFIG_1000"
    run_precomputed "$CONFIG_1000"
    ;;
  *)
    echo "Usage: $0 [250|1000|local|local-1000|unconditioned|unconditioned-1000|matrix-250|matrix-1000|all]" >&2
    exit 1
    ;;
esac
