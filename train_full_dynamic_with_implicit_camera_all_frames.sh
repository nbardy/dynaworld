#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

SEQUENCE_DIR="${SEQUENCE_DIR:-test_data/dust3r_outputs/test_video_small_all_frames}"
FRAMES_DIR="${FRAMES_DIR:-$SEQUENCE_DIR/frames}"
STEPS="${STEPS:-1000}"
LOG_EVERY="${LOG_EVERY:-25}"
IMAGE_LOG_EVERY="${IMAGE_LOG_EVERY:-100}"
VIDEO_LOG_EVERY="${VIDEO_LOG_EVERY:-200}"
SIZE="${SIZE:-32}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
RENDERER="${RENDERER:-dense}"
RUN_NAME="${RUN_NAME:-dynamic-all-frames-implicit-camera}"

read -r FRAME_SOURCE FRAME_COUNT SOURCE_TOTAL FRAMES_DIR_PNG_COUNT <<EOF
$(uv run python - "$SEQUENCE_DIR" "$FRAMES_DIR" <<'PY'
import cv2
import json
import sys
from pathlib import Path

sequence_dir = Path(sys.argv[1])
frames_dir = Path(sys.argv[2])
png_count = len(sorted(frames_dir.glob("*.png"))) if frames_dir.exists() else 0
summary_path = sequence_dir / "summary.json"
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
    video_value = summary.get("video")
    expected_total = summary.get("frame_sampling", {}).get("total_frames")
    if video_value:
        video_path = Path(video_value)
        if video_path.exists():
            capture = cv2.VideoCapture(str(video_path))
            if capture.isOpened():
                video_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                capture.release()
                if video_count >= 2 and (expected_total is None or int(expected_total) == video_count):
                    print("summary_video", video_count, video_count, png_count)
                    raise SystemExit
    sampled = [Path(item["path"]) for item in summary.get("frame_sampling", {}).get("sampled_frames", []) if item.get("path")]
    sampled = [path for path in sampled if path.exists()]
    if len(sampled) >= 2:
        print("summary_sampled", len(sampled), len(sampled), png_count)
        raise SystemExit
if png_count >= 2:
    print("all_frames", png_count, png_count, png_count)
    raise SystemExit
print("missing", 0, 0, png_count)
PY
)
EOF

if [[ "$FRAME_COUNT" -lt 2 ]]; then
  echo "Need at least 2 frames from summary video or $FRAMES_DIR, got $FRAME_COUNT" >&2
  exit 1
fi

echo "Sequence dir: $SEQUENCE_DIR"
echo "Frames dir: $FRAMES_DIR"
echo "Frame source: $FRAME_SOURCE"
echo "Frames per step: $FRAME_COUNT"
echo "Source total frames: $SOURCE_TOTAL"
echo "PNGs in frames dir: $FRAMES_DIR_PNG_COUNT"
echo "Steps: $STEPS"
echo "Scalar log every: $LOG_EVERY"
echo "Image log every: $IMAGE_LOG_EVERY"
echo "Video log every: $VIDEO_LOG_EVERY"
echo "Render size: $SIZE"
echo "Eval batch size: $EVAL_BATCH_SIZE"
echo "Renderer: $RENDERER"
echo "Model: image-encoder implicit-camera baseline (no plucker conditioning)"

uv run python train_scripts/train_camera_implicit_dynamic.py \
  --sequence-dir "$SEQUENCE_DIR" \
  --frames-dir "$FRAMES_DIR" \
  --frame-source "$FRAME_SOURCE" \
  --renderer "$RENDERER" \
  --max-frames 0 \
  --frames-per-step "$FRAME_COUNT" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --steps "$STEPS" \
  --size "$SIZE" \
  --log-every "$LOG_EVERY" \
  --image-log-every "$IMAGE_LOG_EVERY" \
  --video-log-every "$VIDEO_LOG_EVERY" \
  --wandb-run-name "$RUN_NAME" \
  "$@"
