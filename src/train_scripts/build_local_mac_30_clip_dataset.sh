#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

NEURAL_INPUT="data/external/neural_3d_video/extracted/coffee_martini/coffee_martini"
VIVO_TRAIN_INPUT="data/external/vivo/rgb_mp4/athlete_rows/train"
VIVO_TEST_INPUT="data/external/vivo/rgb_mp4/athlete_rows/test"

if [[ ! -d "$NEURAL_INPUT" || ! -d "$VIVO_TRAIN_INPUT" || ! -d "$VIVO_TEST_INPUT" ]]; then
  echo "Missing local source video folders." >&2
  echo "Expected:" >&2
  echo "  $NEURAL_INPUT" >&2
  echo "  $VIVO_TRAIN_INPUT" >&2
  echo "  $VIVO_TEST_INPUT" >&2
  echo "Run the Neural 3D Video seed extract and ViVo compact-rgb steps first." >&2
  exit 1
fi

TRAIN_INPUTS=()
TEST_INPUTS=()

while IFS= read -r path; do
  TRAIN_INPUTS+=("$path")
done < <(find "$VIVO_TRAIN_INPUT" -maxdepth 1 -type f -name '*.mp4' | sort)

while IFS= read -r path; do
  TRAIN_INPUTS+=("$path")
done < <(find "$NEURAL_INPUT" -maxdepth 1 -type f -name '*.mp4' | sort | head -n 10)

while IFS= read -r path; do
  TEST_INPUTS+=("$path")
done < <(find "$VIVO_TEST_INPUT" -maxdepth 1 -type f -name '*.mp4' | sort)

while IFS= read -r path; do
  TEST_INPUTS+=("$path")
done < <(find "$NEURAL_INPUT" -maxdepth 1 -type f -name '*.mp4' | sort | tail -n +11 | head -n 6)

if [[ "${#TRAIN_INPUTS[@]}" -ne 20 || "${#TEST_INPUTS[@]}" -ne 10 ]]; then
  echo "Expected exactly 20 train source videos and 10 test source videos." >&2
  echo "Found train=${#TRAIN_INPUTS[@]} test=${#TEST_INPUTS[@]}" >&2
  exit 1
fi

uv run python src/train/build_clip_dataset.py \
  --train-input "${TRAIN_INPUTS[@]}" \
  --test-input "${TEST_INPUTS[@]}" \
  --output-dir data/clip_sets/local_mac_30_64_4fps_16f \
  --dataset-name local_mac_30_64_4fps_16f \
  --target-count 30 \
  --train-count 20 \
  --test-count 10 \
  --clip-frames 16 \
  --fps 4 \
  --target-size 64 \
  --source-schedule round_robin \
  --max-clips-per-source 1 \
  --require-target-count \
  "$@"
