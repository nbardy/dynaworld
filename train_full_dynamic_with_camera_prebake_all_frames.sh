#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SEQUENCE_DIR="${SEQUENCE_DIR:-test_data/dust3r_outputs/test_video_small_all_frames}"
CAMERA_JSON="${CAMERA_JSON:-$SEQUENCE_DIR/per_frame_cameras.json}"
STEPS="${STEPS:-1000}"
LOG_EVERY="${LOG_EVERY:-25}"
SIZE="${SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
RENDERER="${RENDERER:-dense}"
RUN_NAME="${RUN_NAME:-dynamic-all-frames-prebaked-camera}"

if [[ ! -f "$CAMERA_JSON" ]]; then
  echo "Missing camera JSON: $CAMERA_JSON" >&2
  exit 1
fi

FRAME_COUNT="$(uv run python - "$CAMERA_JSON" <<'PY'
import json
import sys
from pathlib import Path

camera_json = Path(sys.argv[1])
records = json.loads(camera_json.read_text())
print(len(records))
PY
)"

if [[ "$FRAME_COUNT" -lt 2 ]]; then
  echo "Need at least 2 frames in $CAMERA_JSON, got $FRAME_COUNT" >&2
  exit 1
fi

echo "Sequence dir: $SEQUENCE_DIR"
echo "Camera JSON: $CAMERA_JSON"
echo "Frames per step: $FRAME_COUNT"
echo "Steps: $STEPS"
echo "Log/Image/Video every: $LOG_EVERY"
echo "Render size: $SIZE"
echo "Eval batch size: $EVAL_BATCH_SIZE"
echo "Renderer: $RENDERER"

uv run python train_scripts/dynamicTokenGS.py \
  --sequence-dir "$SEQUENCE_DIR" \
  --camera-json "$CAMERA_JSON" \
  --renderer "$RENDERER" \
  --max-frames 0 \
  --frames-per-step "$FRAME_COUNT" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --steps "$STEPS" \
  --size "$SIZE" \
  --log-every "$LOG_EVERY" \
  --image-log-every "$LOG_EVERY" \
  --video-log-every "$LOG_EVERY" \
  --wandb-run-name "$RUN_NAME" \
  "$@"
