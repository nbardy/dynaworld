#!/usr/bin/env bash
set -euo pipefail

uv run python src/train/train_precomputed_feature_implicit_dynamic.py \
  src/train_configs/local_mac_overfit_ltx_feature_implicit_camera_128_4fps_fast_mac_8192splats.jsonc
