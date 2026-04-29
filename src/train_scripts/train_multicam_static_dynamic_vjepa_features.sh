#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_static_dynamic_96_32_precomputed_vjepa2_1_vitb_384_128_16f_8192splats.jsonc}"

uv run python src/train/train_multicam_precomputed_feature_implicit_dynamic.py "$CONFIG_PATH"
