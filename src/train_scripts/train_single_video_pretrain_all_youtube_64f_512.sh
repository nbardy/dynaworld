#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MANIFEST_CONFIG="${MANIFEST_CONFIG:-src/dataset_configs/single_video_pretrain_all_youtube_64f_512_manifest.jsonc}"
TRAIN_CONFIG="${TRAIN_CONFIG:-src/train_configs/local_mac_single_video_pretrain_all_youtube_64f_512render_vjepa_loss_fastproof.jsonc}"
MANIFEST="${MANIFEST:-data/single_video_pretrain/dynaworld_single_video_pretrain_all_youtube_64f_512_v0/train_manifest.jsonl}"
LOAD_CHECK_LIMIT="${LOAD_CHECK_LIMIT:-8}"
PROBE_STEPS="${PROBE_STEPS:-1}"

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
    if (
        record.get("frame_source") != "explicit_video_window"
        or int(record.get("frame_count", 0)) != 64
        or int(record.get("target_size", 0)) != 512
        or str(record.get("image_crop_mode", "")) != "center_square"
    )
]
missing = [record for record in records if not Path(record["video_path"]).exists()]
fps_values = [float(record["fps"]) for record in records]
duration_values = [float(record["duration_seconds"]) for record in records]
payload = {
    "manifest": str(manifest),
    "count": len(records),
    "source_counts": dict(sorted(counts.items())),
    "bad_record_count": len(bad),
    "missing_video_count": len(missing),
    "fps_min": min(fps_values) if fps_values else None,
    "fps_max": max(fps_values) if fps_values else None,
    "duration_min": min(duration_values) if duration_values else None,
    "duration_max": max(duration_values) if duration_values else None,
}
print(json.dumps(payload, indent=2, sort_keys=True))
if not records or bad or missing:
    raise SystemExit(1)
PY
    ;;
  load-check)
    PYTHONPATH=src/train uv run python - "$MANIFEST" "$LOAD_CHECK_LIMIT" <<'PY'
import json
import sys
from pathlib import Path

import torch

from json_io import load_jsonl_objects
from sequence_data import load_manifest_sequence

manifest = Path(sys.argv[1])
limit = int(sys.argv[2])
records = load_jsonl_objects(manifest)
selected = records if limit <= 0 else records[:limit]
data_cfg = {
    "frames_dir": None,
    "frame_source": "explicit_video_window",
    "image_crop_mode": "center_square",
    "max_frames": 0,
    "camera_json": None,
    "camera_image_size": 224,
    "camera_focal_mode": "median",
}
model_cfg = {"size": 512, "train_frame_count": 64}
for index, record in enumerate(selected):
    sequence = load_manifest_sequence(record, data_cfg=data_cfg, model_cfg=model_cfg, device=torch.device("cpu"))
    if sequence.frame_count != 64 or sequence.image_size != 512 or sequence.image_crop_mode != "center_square":
        raise RuntimeError(
            f"{index}: expected 64 frames at 512 center_square, got "
            f"frames={sequence.frame_count} size={sequence.image_size} crop={sequence.image_crop_mode}"
        )
print(json.dumps({
    "loaded_records": len(selected),
    "total_records": len(records),
    "frames_per_record": 64,
    "image_size": 512,
    "image_crop_mode": "center_square",
}, indent=2, sort_keys=True))
PY
    ;;
  probe)
    WANDB_MODE="${WANDB_MODE:-disabled}" PYTHONPATH=src/train uv run python - "$TRAIN_CONFIG" "$PROBE_STEPS" <<'PY'
import sys
from copy import deepcopy
from pathlib import Path

from config_utils import load_config_file
from trainer_registry import run_config_dict

config = deepcopy(load_config_file(Path(sys.argv[1])))
config["train"]["steps"] = int(sys.argv[2])
config["logging"]["log_every"] = 1
config["logging"]["image_log_every"] = 1000000
config["logging"]["video_log_every"] = 1000000
config["logging"]["always_log_last_step"] = False
config["logging"]["wandb_run_name"] = f"{config['logging']['wandb_run_name']}-probe-{config['train']['steps']}step"
run_config_dict(config, Path(sys.argv[1]))
PY
    ;;
  run)
    PYTHONPATH=src/train uv run python src/train/train.py "$TRAIN_CONFIG"
    ;;
  *)
    echo "Usage: $0 {build|audit|load-check|probe|run}" >&2
    exit 2
    ;;
esac
