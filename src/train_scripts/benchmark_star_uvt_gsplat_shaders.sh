#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_DIR="${OUT_DIR:-outputs/benchmarks/$(date +%Y-%m-%d)_shader_audit}"
VIDEO="${VIDEO:-data/youtube_curated_spans/raw/hlaZbH_OFBU_seg_003_s00131000_e00138000.mp4}"
PYTHON="${PYTHON:-.venv/bin/python}"
GS_WARMUP="${GS_WARMUP:-1}"
GS_ITERS="${GS_ITERS:-2}"
UVT_STEPS="${UVT_STEPS:-2}"
UVT_WARMUP_STEPS="${UVT_WARMUP_STEPS:-1}"
RUN_GSPLAT="${RUN_GSPLAT:-1}"
RUN_UVT="${RUN_UVT:-1}"

mkdir -p "$OUT_DIR"

run_gsplat_case() {
  local size="$1"
  local gaussians="$2"
  local batch="$3"
  local out="$OUT_DIR/gsplat_projected_${size}_g${gaussians}_b${batch}_backward.csv"
  echo "== gsplat size=${size} gaussians=${gaussians} batch=${batch}"
  PYTHONPATH=src/train "$PYTHON" src/benchmarks/mac_renderer_stack_compare.py \
    --height "$size" \
    --width "$size" \
    --gaussians "$gaussians" \
    --batch-size "$batch" \
    --warmup "$GS_WARMUP" \
    --iters "$GS_ITERS" \
    --backward \
    --no-include-torch \
    --no-check-outputs \
    --renderers v5,v6 \
    --csv "$out"
}

run_uvt_case() {
  local size="$1"
  local frames="$2"
  local tubes="$3"
  local emission="$4"
  local reduction="$5"
  local out="$OUT_DIR/star_uvt_${size}_${frames}f_${tubes}_${emission}_${reduction}_trainstep.json"
  echo "== star_uvt size=${size} frames=${frames} tubes=${tubes} emission=${emission} reduction=${reduction}"
  "$PYTHON" third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/uvt_train_step_timing_probe.py \
    "$VIDEO" \
    --target-size "$size" \
    --max-frames "$frames" \
    --tube-count "$tubes" \
    --seed 5 \
    --spatial-precision 0.125 \
    --temporal-precision 2.0 \
    --opacity 0.7 \
    --uvt-tile-t 1 \
    --uvt-tile-capacity 256 \
    --lr 0.12 \
    --steps "$UVT_STEPS" \
    --warmup-steps "$UVT_WARMUP_STEPS" \
    --sample-count-every 0 \
    --pair-count-every 0 \
    --uvt-sample-emission-mode "$emission" \
    --uvt-reduction-mode "$reduction" \
    --out-json "$out"
}

if [[ "$RUN_GSPLAT" == "1" ]]; then
  run_gsplat_case 256 8192 16
  run_gsplat_case 256 8192 64
  run_gsplat_case 512 8192 16
  run_gsplat_case 512 8192 64
  run_gsplat_case 512 32768 16
fi

if [[ "$RUN_UVT" == "1" ]]; then
  run_uvt_case 256 16 32768 direct_atomic index_add
  run_uvt_case 256 64 32768 direct_atomic index_add
  run_uvt_case 512 16 32768 direct_atomic index_add
  run_uvt_case 512 64 32768 direct_atomic index_add
  run_uvt_case 256 64 8192 direct_atomic index_add
  run_uvt_case 256 64 8192 tile_pair_suffix key_sort_segmented_metal
fi

echo "shader_audit_out=$OUT_DIR"
