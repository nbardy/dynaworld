#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

LOCAL_CONFIG="src/train_configs/local_mac_compare_local_video_encoder_16f_implicit_camera_128_fast_mac_8192splats.jsonc"
VJEPA_CONFIG="src/train_configs/local_mac_compare_vjepa2_vitl_fpc16_256_frozen_16f_implicit_camera_128_fast_mac_8192splats.jsonc"
KNOWN_CAMERA_CONFIG="src/train_configs/local_mac_compare_local_video_encoder_16f_known_camera_128_fast_mac_8192splats.jsonc"
FREE_SPLATS_CONFIG="src/train_configs/local_mac_compare_free_splats_16f_implicit_camera_128_fast_mac_8192splats.jsonc"
FREE_LINEAR_TIME_SPLATS_CONFIG="src/train_configs/local_mac_compare_free_linear_time_splats_16f_implicit_camera_128_fast_mac_8192splats.jsonc"
UNCONDITIONED_TOKENS_CONFIG="src/train_configs/local_mac_compare_unconditioned_tokens_16f_implicit_camera_128_fast_mac_8192splats.jsonc"
RESIDUAL_LOCAL_CONFIG="src/train_configs/local_mac_compare_residual_free_bank_local_video_encoder_16f_implicit_camera_128_fast_mac_8192splats.jsonc"
RESIDUAL_VJEPA_CONFIG="src/train_configs/local_mac_compare_residual_free_bank_vjepa2_vitl_fpc16_256_frozen_16f_implicit_camera_128_fast_mac_8192splats.jsonc"
RESIDUAL_TOKENS_CONFIG="src/train_configs/local_mac_compare_unconditioned_residual_free_bank_16f_implicit_camera_128_fast_mac_8192splats.jsonc"

MODE="${1:-both}"
case "$MODE" in
  local)
    CONFIGS=("$LOCAL_CONFIG")
    ;;
  vjepa)
    CONFIGS=("$VJEPA_CONFIG")
    ;;
  known|known-camera|camera)
    CONFIGS=("$KNOWN_CAMERA_CONFIG")
    ;;
  free|free-splats|modelless)
    CONFIGS=("$FREE_SPLATS_CONFIG")
    ;;
  free-linear|free-linear-time|linear-free|linear-free-splats)
    CONFIGS=("$FREE_LINEAR_TIME_SPLATS_CONFIG")
    ;;
  tokens|unconditioned|unconditioned-tokens)
    CONFIGS=("$UNCONDITIONED_TOKENS_CONFIG")
    ;;
  residual-local|residual-free-local)
    CONFIGS=("$RESIDUAL_LOCAL_CONFIG")
    ;;
  residual-vjepa|residual-free-vjepa)
    CONFIGS=("$RESIDUAL_VJEPA_CONFIG")
    ;;
  residual-tokens|residual-unconditioned|residual-free-tokens)
    CONFIGS=("$RESIDUAL_TOKENS_CONFIG")
    ;;
  residual-both)
    CONFIGS=("$RESIDUAL_LOCAL_CONFIG" "$RESIDUAL_VJEPA_CONFIG")
    ;;
  residual|residual-matrix)
    CONFIGS=("$RESIDUAL_TOKENS_CONFIG" "$RESIDUAL_LOCAL_CONFIG" "$RESIDUAL_VJEPA_CONFIG")
    ;;
  no-conditioning|controls)
    CONFIGS=("$FREE_SPLATS_CONFIG" "$FREE_LINEAR_TIME_SPLATS_CONFIG" "$UNCONDITIONED_TOKENS_CONFIG" "$RESIDUAL_TOKENS_CONFIG")
    ;;
  both)
    CONFIGS=("$LOCAL_CONFIG" "$VJEPA_CONFIG")
    ;;
  matrix)
    CONFIGS=(
      "$LOCAL_CONFIG"
      "$VJEPA_CONFIG"
      "$UNCONDITIONED_TOKENS_CONFIG"
      "$FREE_LINEAR_TIME_SPLATS_CONFIG"
      "$FREE_SPLATS_CONFIG"
      "$RESIDUAL_TOKENS_CONFIG"
      "$RESIDUAL_LOCAL_CONFIG"
      "$RESIDUAL_VJEPA_CONFIG"
    )
    ;;
  all)
    CONFIGS=(
      "$LOCAL_CONFIG"
      "$VJEPA_CONFIG"
      "$KNOWN_CAMERA_CONFIG"
      "$FREE_SPLATS_CONFIG"
      "$FREE_LINEAR_TIME_SPLATS_CONFIG"
      "$UNCONDITIONED_TOKENS_CONFIG"
      "$RESIDUAL_TOKENS_CONFIG"
      "$RESIDUAL_LOCAL_CONFIG"
      "$RESIDUAL_VJEPA_CONFIG"
    )
    ;;
  *)
    echo "Usage: $0 [local|vjepa|known|free|free-linear|tokens|residual-local|residual-vjepa|residual-tokens|residual|matrix|controls|both|all]" >&2
    exit 1
    ;;
esac

for config in "${CONFIGS[@]}"; do
  if [[ ! -f "$config" ]]; then
    echo "Missing config: $config" >&2
    exit 1
  fi
  echo "Config: $config"
  uv run python src/train/train_video_token_implicit_dynamic.py "$config"
done
