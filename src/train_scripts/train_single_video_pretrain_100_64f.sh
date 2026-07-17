#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MANIFEST_CONFIG="${MANIFEST_CONFIG:-src/dataset_configs/single_video_pretrain_100_64f_manifest.jsonc}"
TRAIN_CONFIG="${TRAIN_CONFIG:-src/train_configs/local_mac_single_video_pretrain_100_precomputed_vjepa2_1_vitb_256crop_64f_fast_proof_64render_512splats.jsonc}"
FULL_TRAIN_CONFIG="${FULL_TRAIN_CONFIG:-src/train_configs/local_mac_single_video_pretrain_100_precomputed_vjepa2_1_vitb_256crop_64f_implicit_camera_128_fast_mac_8192splats.jsonc}"
DIRECT_TRAIN_CONFIG="${DIRECT_TRAIN_CONFIG:-src/train_configs/local_mac_single_video_pretrain_100_vjepa2_1_vitb_256crop_64f_implicit_camera_128_fast_mac_8192splats.jsonc}"
SMOKE_CONFIG="${SMOKE_CONFIG:-src/train_configs/local_mac_single_video_pretrain_100_local_encoder_64f_tiny_smoke.jsonc}"
MANIFEST="${MANIFEST:-data/single_video_pretrain/dynaworld_single_video_pretrain_100_64f_v0/train_manifest.jsonl}"

case "${1:-}" in
  build)
    uv run python src/dataset_scripts/build_single_video_pretrain_manifest.py --config "$MANIFEST_CONFIG"
    ;;
  audit)
    PYTHONPATH=src/train uv run python - "$MANIFEST" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

from json_io import load_jsonl_objects

manifest = Path(sys.argv[1])
records = load_jsonl_objects(manifest)
counts = Counter(record.get("source_label", "unknown") for record in records)
bad = [
    record for record in records
    if record.get("frame_source") != "explicit_video_window" or int(record.get("frame_count", 0)) != 64
]
missing = [record for record in records if not Path(record["video_path"]).exists()]
print(json.dumps({
    "manifest": str(manifest),
    "count": len(records),
    "source_counts": dict(sorted(counts.items())),
    "bad_record_count": len(bad),
    "missing_video_count": len(missing),
}, indent=2, sort_keys=True))
if len(records) != 100 or bad or missing:
    raise SystemExit(1)
PY
    ;;
  load-check)
    PYTHONPATH=src/train uv run python - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

import torch

from json_io import load_jsonl_objects
from sequence_data import load_manifest_sequence

manifest = Path(sys.argv[1])
records = load_jsonl_objects(manifest)
data_cfg = {
    "frames_dir": None,
    "frame_source": "explicit_video_window",
    "max_frames": 0,
    "camera_json": None,
    "camera_image_size": 224,
    "camera_focal_mode": "median",
}
model_cfg = {"size": 64, "train_frame_count": 64}
for index, record in enumerate(records):
    sequence = load_manifest_sequence(record, data_cfg=data_cfg, model_cfg=model_cfg, device=torch.device("cpu"))
    if sequence.frame_count != 64:
        raise RuntimeError(f"{index}: expected 64 frames, got {sequence.frame_count}")
print(json.dumps({"loaded_records": len(records), "frames_per_record": 64}, indent=2, sort_keys=True))
PY
    ;;
  smoke)
    WANDB_MODE="${WANDB_MODE:-disabled}" PYTHONPATH=src/train uv run python src/train/train.py "$SMOKE_CONFIG"
    ;;
  run)
    PYTHONPATH=src/train uv run python src/train/train.py "$TRAIN_CONFIG"
    ;;
  run-full)
    PYTHONPATH=src/train uv run python src/train/train.py "$FULL_TRAIN_CONFIG"
    ;;
  run-direct)
    PYTHONPATH=src/train uv run python src/train/train.py "$DIRECT_TRAIN_CONFIG"
    ;;
  *)
    echo "Usage: $0 {build|audit|load-check|smoke|run|run-full|run-direct}" >&2
    exit 2
    ;;
esac
