#!/usr/bin/env bash
set -euo pipefail

uv run python src/train/train_precomputed_feature_implicit_dynamic.py \
  src/train_configs/local_mac_overfit_precomputed_vjepa2_1_torchhub_vitb_384.jsonc
