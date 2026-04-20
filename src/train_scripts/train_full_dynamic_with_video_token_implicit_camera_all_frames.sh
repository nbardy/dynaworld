#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$#" -ne 0 ]]; then
  echo "Pass overrides as environment variables, not CLI args." >&2
  exit 1
fi

SEQUENCE_DIR="${SEQUENCE_DIR:-test_data}"
VIDEO_PATH="${VIDEO_PATH:-test_data/test_video_384_3fps.mp4}"
STEPS="${STEPS:-1000}"
LOG_EVERY="${LOG_EVERY:-25}"
IMAGE_LOG_EVERY="${IMAGE_LOG_EVERY:-100}"
VIDEO_LOG_EVERY="${VIDEO_LOG_EVERY:-200}"
SIZE="${SIZE:-384}"
TRAIN_FRAME_COUNT="${TRAIN_FRAME_COUNT:-16}"
TOKENS="${TOKENS:-8}"
GAUSSIANS_PER_TOKEN="${GAUSSIANS_PER_TOKEN:-64}"
MODEL_DIM="${MODEL_DIM:-128}"
BOTTLENECK_DIM="${BOTTLENECK_DIM:-256}"
NUM_HEADS="${NUM_HEADS:-8}"
MLP_RATIO="${MLP_RATIO:-4.0}"
SCENE_EXTENT="${SCENE_EXTENT:-1.0}"
TUBELET_SIZE_T="${TUBELET_SIZE_T:-4}"
PATCH_COMPRESSION="${PATCH_COMPRESSION:-16}"
ENCODER_SELF_ATTN_LAYERS="${ENCODER_SELF_ATTN_LAYERS:-1}"
BOTTLENECK_SELF_ATTN_LAYERS="${BOTTLENECK_SELF_ATTN_LAYERS:-4}"
CROSS_ATTN_LAYERS="${CROSS_ATTN_LAYERS:-1}"
BASE_FOV_DEGREES="${BASE_FOV_DEGREES:-60.0}"
BASE_RADIUS="${BASE_RADIUS:-3.0}"
MAX_FOV_DELTA_DEGREES="${MAX_FOV_DELTA_DEGREES:-15.0}"
MAX_RADIUS_SCALE="${MAX_RADIUS_SCALE:-1.5}"
MAX_ROTATION_DEGREES="${MAX_ROTATION_DEGREES:-5.0}"
MAX_TRANSLATION_RATIO="${MAX_TRANSLATION_RATIO:-0.2}"
RENDERER="${RENDERER:-dense}"
LR="${LR:-0.005}"
AMP="${AMP:-false}"
TRAIN_BACKWARD_STRATEGY="${TRAIN_BACKWARD_STRATEGY:-framewise}"
TEMPORAL_MICROBATCH_SIZE="${TEMPORAL_MICROBATCH_SIZE:-4}"
CAMERA_MOTION_WEIGHT="${CAMERA_MOTION_WEIGHT:-0.01}"
CAMERA_TEMPORAL_WEIGHT="${CAMERA_TEMPORAL_WEIGHT:-0.02}"
CAMERA_GLOBAL_WEIGHT="${CAMERA_GLOBAL_WEIGHT:-0.005}"
ALWAYS_LOG_LAST_STEP="${ALWAYS_LOG_LAST_STEP:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-dynamic-tokengs-overfit}"
RUN_NAME="${RUN_NAME:-dynamic-video-token-implicit-camera}"

if [[ ! -f "$VIDEO_PATH" ]]; then
  echo "Missing video: $VIDEO_PATH" >&2
  exit 1
fi

read -r FRAME_COUNT VIDEO_FPS VIDEO_WIDTH VIDEO_HEIGHT <<EOF
$(uv run python - "$VIDEO_PATH" <<'PY'
import cv2
import sys

video_path = sys.argv[1]
capture = cv2.VideoCapture(video_path)
if not capture.isOpened():
    raise SystemExit("0 0 0 0")
frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
capture.release()
print(frame_count, fps, width, height)
PY
)
EOF

if (( TRAIN_FRAME_COUNT % TUBELET_SIZE_T != 0 )); then
  echo "TRAIN_FRAME_COUNT=$TRAIN_FRAME_COUNT must be divisible by TUBELET_SIZE_T=$TUBELET_SIZE_T" >&2
  exit 1
fi

if [[ "$FRAME_COUNT" -lt "$TRAIN_FRAME_COUNT" ]]; then
  echo "Need at least TRAIN_FRAME_COUNT=$TRAIN_FRAME_COUNT frames in $VIDEO_PATH, got $FRAME_COUNT" >&2
  exit 1
fi

echo "Sequence dir: $SEQUENCE_DIR"
echo "Video path: $VIDEO_PATH"
echo "Video fps: $VIDEO_FPS"
echo "Video size: ${VIDEO_WIDTH}x${VIDEO_HEIGHT}"
echo "Video frames: $FRAME_COUNT"
echo "Train frame count: $TRAIN_FRAME_COUNT"
echo "Temporal tubelets per clip: $(( TRAIN_FRAME_COUNT / TUBELET_SIZE_T ))"
echo "3DGS tokens: $TOKENS"
echo "Gaussians per token: $GAUSSIANS_PER_TOKEN"
echo "Model dim: $MODEL_DIM"
echo "Bottleneck dim: $BOTTLENECK_DIM"
echo "Heads: $NUM_HEADS"
echo "MLP ratio: $MLP_RATIO"
echo "Scene extent: $SCENE_EXTENT"
echo "Tubelet temporal size: $TUBELET_SIZE_T"
echo "Patch compression: $PATCH_COMPRESSION"
echo "Encoder self-attn layers: $ENCODER_SELF_ATTN_LAYERS"
echo "Bottleneck self-attn layers: $BOTTLENECK_SELF_ATTN_LAYERS"
echo "Cross-attn layers: $CROSS_ATTN_LAYERS"
echo "Base FOV degrees: $BASE_FOV_DEGREES"
echo "Base radius: $BASE_RADIUS"
echo "Max FOV delta degrees: $MAX_FOV_DELTA_DEGREES"
echo "Max radius scale: $MAX_RADIUS_SCALE"
echo "Max rotation degrees: $MAX_ROTATION_DEGREES"
echo "Max translation ratio: $MAX_TRANSLATION_RATIO"
echo "Renderer: $RENDERER"
echo "LR: $LR"
echo "AMP: $AMP"
echo "Recon backward strategy: $TRAIN_BACKWARD_STRATEGY"
echo "Temporal microbatch size: $TEMPORAL_MICROBATCH_SIZE"
echo "Steps: $STEPS"
echo "Scalar log every: $LOG_EVERY"
echo "Image log every: $IMAGE_LOG_EVERY"
echo "Video log every: $VIDEO_LOG_EVERY"
echo "Always log last step: $ALWAYS_LOG_LAST_STEP"
echo "Input/render size: $SIZE"
if [[ "$FRAME_COUNT" -gt "$TRAIN_FRAME_COUNT" ]]; then
  echo "Sampling: random contiguous chunks of $TRAIN_FRAME_COUNT frames per step"
else
  echo "Sampling: full video each step"
fi

export SEQUENCE_DIR
export VIDEO_PATH
export FRAME_SOURCE="explicit_video"
export STEPS
export SIZE
export TRAIN_FRAME_COUNT
export TOKENS
export GAUSSIANS_PER_TOKEN
export MODEL_DIM
export BOTTLENECK_DIM
export NUM_HEADS
export MLP_RATIO
export SCENE_EXTENT
export TUBELET_SIZE_T
export PATCH_COMPRESSION
export ENCODER_SELF_ATTN_LAYERS
export BOTTLENECK_SELF_ATTN_LAYERS
export CROSS_ATTN_LAYERS
export BASE_FOV_DEGREES
export BASE_RADIUS
export MAX_FOV_DELTA_DEGREES
export MAX_RADIUS_SCALE
export MAX_ROTATION_DEGREES
export MAX_TRANSLATION_RATIO
export RENDERER
export LR
export AMP
export TRAIN_BACKWARD_STRATEGY
export TEMPORAL_MICROBATCH_SIZE
export CAMERA_MOTION_WEIGHT
export CAMERA_TEMPORAL_WEIGHT
export CAMERA_GLOBAL_WEIGHT
export LOG_EVERY
export IMAGE_LOG_EVERY
export VIDEO_LOG_EVERY
export ALWAYS_LOG_LAST_STEP
export WANDB_PROJECT
export RUN_NAME

uv run python - <<'PY'
import sys

sys.path.insert(0, "src/train")
from train_video_token_implicit_dynamic import config_from_env, main

main(config_from_env())
PY
