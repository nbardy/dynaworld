#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODE="${1:-help}"
PYTHON="${PYTHON:-.venv/bin/python}"

STAR_CONFIG_256="${STAR_CONFIG_256:-src/train_configs/star_uvt_highmotion_hlaZbH_64f_256_directatomic_200step.jsonc}"
STAR_CONFIG_512="${STAR_CONFIG_512:-src/train_configs/star_uvt_highmotion_hlaZbH_64f_512_directatomic_multires256c200_50fine.jsonc}"
STAR_FEATURE_FAST_CONFIG="${STAR_FEATURE_FAST_CONFIG:-src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_sparseforward_batchedvjp_checkpoint_media.jsonc}"
STAR_FEATURE_VISUAL_CONFIG="${STAR_FEATURE_VISUAL_CONFIG:-src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_compact_currentbuild_from1500_lr001_50step_media.jsonc}"
STAR_FEATURE_NATIVE_FULLCELL_CONFIG="${STAR_FEATURE_NATIVE_FULLCELL_CONFIG:-src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_sparsevisual_targetarea64_fullcell8_nativehidden_vec4wt_from1500_lr001_50step_media.jsonc}"
STAR_FEATURE_RGB_FAST_CONFIG="${STAR_FEATURE_RGB_FAST_CONFIG:-src/train_configs/star_uvt_feature_testvideo_64f_512_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_20step_media.jsonc}"
GSPLAT_CONFIG="${GSPLAT_CONFIG:-src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_overfit1_400step_multires_256to512.jsonc}"

usage() {
  cat >&2 <<'EOF'
Usage:
  train_fast_overfit_star_uvt_and_dynamic_gsplat.sh star-256
  train_fast_overfit_star_uvt_and_dynamic_gsplat.sh star-512
  train_fast_overfit_star_uvt_and_dynamic_gsplat.sh star-feature-512-fast
  train_fast_overfit_star_uvt_and_dynamic_gsplat.sh star-feature-512-visual
  train_fast_overfit_star_uvt_and_dynamic_gsplat.sh star-feature-512-native-fullcell
  train_fast_overfit_star_uvt_and_dynamic_gsplat.sh star-feature-512-rgbfast
  train_fast_overfit_star_uvt_and_dynamic_gsplat.sh gsplat-smoke
  train_fast_overfit_star_uvt_and_dynamic_gsplat.sh gsplat-overfit

Environment:
  STAR_CONFIG_256        Override the first-class 256px STAR UVT config.
  STAR_CONFIG_512        Override the first-class 512px multires STAR UVT config.
  STAR_FEATURE_FAST_CONFIG
                         Override the 512px STAR UVT batched sparse-forward V-JEPA target fast config.
  STAR_FEATURE_VISUAL_CONFIG
                         Override the 512px compact target-area visual overfit config.
  STAR_FEATURE_NATIVE_FULLCELL_CONFIG
                         Override the 512px full-cell8 native target-area vec4 W^T config.
  STAR_FEATURE_RGB_FAST_CONFIG
                         Override the older 512px STAR UVT RGB-target feature fast config.
  GSPLAT_CONFIG          Override the dynamic gsplat precomputed-feature overfit config.
  PROBE_STEPS            For gsplat-smoke only. Default: 5.

Notes:
  STAR UVT uses direct_atomic/index_add as the current practical overfit lane.
  STAR feature fast mode is the current batched sparse-forward V-JEPA target
  diagnostic. star-feature-512-visual is the better compact visual target-area
  route today. star-feature-512-native-fullcell runs the promoted exact
  full-support native target-area vec4 W^T shader, which is a correctness/speed
  baseline rather than the best visual objective. star-feature-512-rgbfast keeps
  the older RGB-target speed row.
  The gsplat overfit config uses cached V-JEPA conditioning and disables
  differentiable V-JEPA prediction-side feature loss.
EOF
}

case "$MODE" in
  star-256)
    PYTHONPATH=src/train "$PYTHON" src/train/train.py "$STAR_CONFIG_256"
    ;;
  star-512)
    PYTHONPATH=src/train "$PYTHON" src/train/train.py "$STAR_CONFIG_512"
    ;;
  star-feature-512-fast)
    PYTHONPATH=src/train "$PYTHON" src/train/train.py "$STAR_FEATURE_FAST_CONFIG"
    ;;
  star-feature-512-visual)
    PYTHONPATH=src/train "$PYTHON" src/train/train.py "$STAR_FEATURE_VISUAL_CONFIG"
    ;;
  star-feature-512-native-fullcell)
    PYTHONPATH=src/train "$PYTHON" src/train/train.py "$STAR_FEATURE_NATIVE_FULLCELL_CONFIG"
    ;;
  star-feature-512-rgbfast)
    PYTHONPATH=src/train "$PYTHON" src/train/train.py "$STAR_FEATURE_RGB_FAST_CONFIG"
    ;;
  gsplat-smoke)
    PROBE_STEPS="${PROBE_STEPS:-5}" \
      WANDB_MODE="${WANDB_MODE:-disabled}" \
      TRAIN_CONFIG="$GSPLAT_CONFIG" \
      ./src/train_scripts/train_single_video_pretrain_300_64f.sh probe
    ;;
  gsplat-overfit)
    PYTHONPATH=src/train "$PYTHON" src/train/train.py "$GSPLAT_CONFIG"
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac
