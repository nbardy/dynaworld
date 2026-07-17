#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

MODE="${1:-dry-run}"
DATASET_NAME="${DATASET_NAME:-dynaworld_scale_1k_256_4fps_16f}"
OUTPUT_DIR="${OUTPUT_DIR:-data/clip_sets/${DATASET_NAME}}"
TARGET_COUNT="${TARGET_COUNT:-1000}"
CLIP_FRAMES="${CLIP_FRAMES:-16}"
FPS="${FPS:-4}"
TARGET_SIZE="${TARGET_SIZE:-256}"
STRIDE_SECONDS="${STRIDE_SECONDS:-0.5}"
SOURCE_SCHEDULE="${SOURCE_SCHEDULE:-round_robin}"
MAX_CLIPS_PER_SOURCE="${MAX_CLIPS_PER_SOURCE:-0}"

usage() {
  cat >&2 <<'EOF'
Usage:
  build_dynaworld_scale_1k_clip_dataset.sh dry-run
  build_dynaworld_scale_1k_clip_dataset.sh build

Environment:
  TARGET_COUNT          Requested train clips. Default: 1000.
  OUTPUT_DIR            Output dataset directory. Default: data/clip_sets/dynaworld_scale_1k_256_4fps_16f.
  STRIDE_SECONDS        Seconds between source windows. Default: 0.5.
  MAX_CLIPS_PER_SOURCE  Per-video cap. Default: 0, meaning unlimited.
  OVERWRITE_CLIPS=1     Required to replace an existing build output.

The build emits loader-compatible frame clips and manifest.jsonl for the
single-video scale-pretrain lane. It scans prepared YouTube clips and the
prepared AIST/Neural3D/ViVo/DeepView video datasets; empty synthetic folders are
skipped by the scanner because they contain no videos.
EOF
}

case "$MODE" in
  dry-run|build)
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage
    exit 1
    ;;
esac

candidate_inputs=(
  "data/youtube_scene_distinct/raw"
  "data/youtube_scene_distinct/segments"
  "data/youtube_curated_spans/raw"
  "data/youtube_curated_spans/high_motion_smokes"
  "data/external/aist_dance_db/raw/refined_10M_sBM"
  "data/external/neural_3d_video/extracted"
  "data/external/vivo/rgb_mp4/athlete_rows/train"
  "data/external/vivo/rgb_mp4/athlete_rows/test"
  "data/external/deepview_video/extracted"
  "data/blender_synthetic"
)

inputs=()
for path in "${candidate_inputs[@]}"; do
  if [[ -e "$path" ]]; then
    inputs+=("$path")
  fi
done

if [[ "${#inputs[@]}" -eq 0 ]]; then
  echo "No source video inputs found." >&2
  exit 1
fi

output_args=(--output-dir "$OUTPUT_DIR")
if [[ "$MODE" == "dry-run" ]]; then
  output_args=(--output-dir "/tmp/${DATASET_NAME}_dryrun")
fi

cmd=(uv run python src/train/build_clip_dataset.py
  --input "${inputs[@]}" \
  "${output_args[@]}" \
  --dataset-name "$DATASET_NAME" \
  --target-count "$TARGET_COUNT" \
  --clip-frames "$CLIP_FRAMES" \
  --fps "$FPS" \
  --target-size "$TARGET_SIZE" \
  --stride-seconds "$STRIDE_SECONDS" \
  --source-schedule "$SOURCE_SCHEDULE" \
  --max-clips-per-source "$MAX_CLIPS_PER_SOURCE")
if [[ "$MODE" == "dry-run" ]]; then
  cmd+=(--dry-run)
elif [[ "${OVERWRITE_CLIPS:-0}" == "1" ]]; then
  cmd+=(--overwrite)
fi

PYTHONPATH=src/train "${cmd[@]}"
