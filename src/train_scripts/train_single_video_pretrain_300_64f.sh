#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MANIFEST_CONFIG="${MANIFEST_CONFIG:-src/dataset_configs/single_video_pretrain_300_youtube_64f_512_manifest.jsonc}"
MANIFEST="${MANIFEST:-data/single_video_pretrain/dynaworld_single_video_pretrain_300_youtube_64f_512_v0/train_manifest.jsonl}"
TRAIN_CONFIG="${TRAIN_CONFIG:-src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_overfit1_400step.jsonc}"
LOAD_CHECK_LIMIT="${LOAD_CHECK_LIMIT:-8}"
PROBE_STEPS="${PROBE_STEPS:-1}"
PREBAKE_LIMIT="${PREBAKE_LIMIT:-0}"
REQUIRE_FULL_CACHE="${REQUIRE_FULL_CACHE:-0}"
CACHE_POLL_SECONDS="${CACHE_POLL_SECONDS:-60}"
RUN_LOG_DIR="${RUN_LOG_DIR:-outputs/run_logs}"

default_bench_configs=(
  "src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_profile.jsonc"
  "src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_8192splats_token_capacity.jsonc"
  "src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_overfit1_400step.jsonc"
)

log_path_for() {
  local label="$1"
  local config="$2"
  local stem
  stem="$(basename "$config" .jsonc)"
  mkdir -p "$RUN_LOG_DIR"
  printf '%s/%s_%s_%s.log' "$RUN_LOG_DIR" "$stem" "$label" "$(date +%Y%m%d_%H%M%S)"
}

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
    "window_duration_min": min(duration_values) if duration_values else None,
    "window_duration_max": max(duration_values) if duration_values else None,
}
print(json.dumps(payload, indent=2, sort_keys=True))
if len(records) != 300 or bad or missing:
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
  resolve)
    PYTHONPATH=src/train uv run python - "$TRAIN_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

from config_utils import load_config_file
from render_dispatch import decoded_token_count_from_model_config
from trainer_registry import resolve_config_for_arch

cfg = resolve_config_for_arch(load_config_file(Path(sys.argv[1])), Path(sys.argv[1]))
model = cfg["model"]
layout = model.get("token_layout")
payload = {
    "config": sys.argv[1],
    "manifest_path": str(cfg["data"]["manifest_path"]),
    "model_size": model["size"],
    "train_frame_count": model["train_frame_count"],
    "render_size": cfg["render"]["render_size"],
    "tokens_total_non_camera": model["tokens"],
    "decoded_tokens_active": decoded_token_count_from_model_config(model),
    "static_tokens_full_capacity": model["static_tokens"],
    "dynamic_tokens_full_capacity": model["dynamic_tokens"],
    "token_layout": layout,
    "gaussians_per_token": model["gaussians_per_token"],
    "approx_decoded_gaussians": decoded_token_count_from_model_config(model) * model["gaussians_per_token"],
    "camera_refine_with_decode_time": model["camera_refine_with_decode_time"],
    "vjepa_crop_size": cfg["features"]["vjepa_crop_size"],
    "vjepa_feature_loss_crop_size": cfg["losses"].get("vjepa_feature_crop_size"),
    "vjepa_feature_temporal_stride": cfg["losses"].get("vjepa_feature_temporal_stride"),
    "fast_mac": cfg["render"]["fast_mac"],
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
    ;;
  probe)
    log_path="$(log_path_for "probe${PROBE_STEPS}step" "$TRAIN_CONFIG")"
    WANDB_MODE="${WANDB_MODE:-disabled}" PYTHONPATH=src/train uv run python - "$TRAIN_CONFIG" "$PROBE_STEPS" <<'PY' 2>&1 | tee "$log_path"
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
    echo "probe_log=$log_path"
    ;;
  bench)
    shift || true
    configs=("$@")
    if [ "${#configs[@]}" -eq 0 ]; then
      configs=("${default_bench_configs[@]}")
    fi
    for config in "${configs[@]}"; do
      echo "== probe config: $config"
      TRAIN_CONFIG="$config" PROBE_STEPS="$PROBE_STEPS" "$0" resolve
      TRAIN_CONFIG="$config" PROBE_STEPS="$PROBE_STEPS" "$0" probe
    done
    ;;
  cache-status)
    STATUS_LOG="${STATUS_LOG:-}"
    if [ -z "$STATUS_LOG" ]; then
      STATUS_LOG="$(ls -t "$RUN_LOG_DIR"/*prebake_*.log 2>/dev/null | head -1 || true)"
    fi
    PYTHONPATH=src/train uv run python - "$TRAIN_CONFIG" "$STATUS_LOG" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

from config_utils import load_config_file
from json_io import load_jsonl_objects

config_path = Path(sys.argv[1])
log_path = Path(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2] else None
cfg = load_config_file(config_path)
cache_dir = Path(cfg["features"]["cache_dir"])
manifest_path = Path(cfg["data"]["manifest_path"])
manifest_count = len(load_jsonl_objects(manifest_path))
cache_files = len(list(cache_dir.glob("*.pt"))) if cache_dir.exists() else 0

last_progress = None
if log_path is not None and log_path.exists():
    for line in log_path.read_text(errors="replace").splitlines():
        stripped = line.strip()
        if not (stripped.startswith("{") and stripped.endswith("}")):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if "record" in payload or payload.get("done"):
            last_progress = payload

screen_result = subprocess.run(["screen", "-ls"], check=False, capture_output=True, text=True)
screen_lines = [
    line.strip()
    for line in screen_result.stdout.splitlines()
    if "dynaworld_300_64f_512_prebake" in line
]

payload = {
    "config": str(config_path),
    "cache_dir": str(cache_dir),
    "feature_cache_files": cache_files,
    "manifest_records": manifest_count,
    "feature_cache_coverage": cache_files / manifest_count if manifest_count else None,
    "prebake_log": str(log_path) if log_path is not None else None,
    "last_prebake_progress": last_progress,
    "prebake_screen_sessions": screen_lines,
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
    ;;
  prebake)
    log_path="$(log_path_for "prebake" "$TRAIN_CONFIG")"
    PYTHONPATH=src/train uv run python -u - "$TRAIN_CONFIG" "$PREBAKE_LIMIT" <<'PY' 2>&1 | tee "$log_path"
import gc
import json
import sys
import time
from pathlib import Path

import torch

from config_utils import load_config_file
from json_io import load_jsonl_objects
from sequence_data import load_manifest_entries, load_manifest_sequence
from trainer_registry import resolve_config_for_arch
from video_feature_cache import VideoFeatureCache

config_path = Path(sys.argv[1])
limit = int(sys.argv[2])
cfg = resolve_config_for_arch(load_config_file(config_path), config_path)
feature_cfg = dict(cfg["features"])
feature_cfg["keep_in_memory"] = False

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
cache = VideoFeatureCache(feature_cfg, device)
entries = load_manifest_entries(Path(cfg["data"]["manifest_path"]), split=cfg["data"]["split"])
if limit > 0:
    entries = entries[:limit]

total_manifest_records = len(load_jsonl_objects(Path(cfg["data"]["manifest_path"])))

payload = {
    "config": str(config_path),
    "manifest_path": str(cfg["data"]["manifest_path"]),
    "split": cfg["data"]["split"],
    "records_selected": len(entries),
    "manifest_records": total_manifest_records,
    "prebake_limit": limit,
    "device": str(device),
    "cache_dir": str(cache.cache_dir),
}
print(json.dumps(payload, indent=2, sort_keys=True))

baked = 0
skipped = 0
started = time.time()
try:
    for index, entry in enumerate(entries, start=1):
        sequence = load_manifest_sequence(
            entry,
            data_cfg=cfg["data"],
            model_cfg=cfg["model"],
            device=torch.device("cpu"),
        )
        key = cache.cache_key(sequence)
        path = cache.cache_path(sequence)
        if path.exists() and not cache.force_rebake:
            skipped += 1
        else:
            cache.load_or_bake(sequence)
            baked += 1
        del sequence
        if device.type == "mps":
            torch.mps.empty_cache()
        gc.collect()
        if index == 1 or index == len(entries) or index % 10 == 0 or baked and baked % 5 == 0:
            elapsed = max(time.time() - started, 1e-6)
            rate = index / elapsed
            remaining = (len(entries) - index) / rate if rate > 0 else None
            print(json.dumps({
                "record": index,
                "selected_records": len(entries),
                "baked": baked,
                "skipped_existing": skipped,
                "last_cache_key": key,
                "last_cache_path": str(path),
                "cache_files": len(list(cache.cache_dir.glob("*.pt"))),
                "elapsed_seconds": round(elapsed, 2),
                "records_per_second": round(rate, 4),
                "eta_seconds": None if remaining is None else round(remaining, 2),
            }, sort_keys=True))
finally:
    cache.release_extractor()

cache_files = len(list(cache.cache_dir.glob("*.pt")))
print(json.dumps({
    "done": True,
    "selected_records": len(entries),
    "baked": baked,
    "skipped_existing": skipped,
    "cache_files": cache_files,
    "manifest_records": total_manifest_records,
    "feature_cache_coverage": cache_files / total_manifest_records if total_manifest_records else None,
    "elapsed_seconds": round(time.time() - started, 2),
}, indent=2, sort_keys=True))
PY
    echo "prebake_log=$log_path"
    ;;
  wait-cache-run)
    if [ "$CACHE_POLL_SECONDS" -lt 1 ]; then
      echo "CACHE_POLL_SECONDS must be >= 1, got $CACHE_POLL_SECONDS" >&2
      exit 2
    fi
    while true; do
      cache_payload="$(
        PYTHONPATH=src/train uv run python - "$TRAIN_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

from config_utils import load_config_file
from json_io import load_jsonl_objects

cfg = load_config_file(Path(sys.argv[1]))
cache_dir = Path(cfg["features"]["cache_dir"])
manifest_path = Path(cfg["data"]["manifest_path"])
manifest_count = len(load_jsonl_objects(manifest_path))
cache_files = len(list(cache_dir.glob("*.pt"))) if cache_dir.exists() else 0
print(json.dumps({
    "cache_dir": str(cache_dir),
    "feature_cache_files": cache_files,
    "manifest_records": manifest_count,
    "feature_cache_coverage": cache_files / manifest_count if manifest_count else None,
}))
PY
      )"
      echo "$cache_payload"
      ready="$(
        CACHE_PAYLOAD="$cache_payload" uv run python - <<'PY'
import json
import os

payload = json.loads(os.environ["CACHE_PAYLOAD"])
print("1" if payload["feature_cache_files"] >= payload["manifest_records"] else "0")
PY
      )"
      if [ "$ready" = "1" ]; then
        session="dynaworld_300_64f_512_train_$(date +%Y%m%d_%H%M%S)"
        screen_log="$RUN_LOG_DIR/${session}_screen.log"
        echo "Feature cache is complete; launching cache-hot training in screen session $session."
        screen -dmS "$session" bash -lc "cd '$ROOT' && WANDB_MODE='${WANDB_MODE:-offline}' REQUIRE_FULL_CACHE=1 TRAIN_CONFIG='$TRAIN_CONFIG' RUN_LOG_DIR='$RUN_LOG_DIR' ./src/train_scripts/train_single_video_pretrain_300_64f.sh run > '$screen_log' 2>&1"
        echo "training_session=$session"
        echo "training_screen_log=$screen_log"
        exit 0
      fi
      sleep "$CACHE_POLL_SECONDS"
    done
    ;;
  run)
    if [ "$REQUIRE_FULL_CACHE" = "1" ]; then
      PYTHONPATH=src/train uv run python - "$TRAIN_CONFIG" <<'PY'
import sys
from pathlib import Path

from config_utils import load_config_file
from json_io import load_jsonl_objects

cfg = load_config_file(Path(sys.argv[1]))
cache_dir = Path(cfg["features"]["cache_dir"])
manifest_path = Path(cfg["data"]["manifest_path"])
manifest_count = len(load_jsonl_objects(manifest_path))
cache_files = len(list(cache_dir.glob("*.pt"))) if cache_dir.exists() else 0
if cache_files < manifest_count:
    raise SystemExit(
        f"Feature cache is not complete: {cache_files}/{manifest_count} files in {cache_dir}. "
        "Run the prebake action first, or set REQUIRE_FULL_CACHE=0 to allow lazy baking."
    )
print(f"Feature cache ready: {cache_files}/{manifest_count} files in {cache_dir}.")
PY
    fi
    log_path="$(log_path_for "run" "$TRAIN_CONFIG")"
    PYTHONPATH=src/train uv run python -u - "$TRAIN_CONFIG" <<'PY' 2>&1 | tee "$log_path"
import sys
from pathlib import Path

from config_utils import load_config_file
from trainer_registry import run_config_dict

run_config_dict(load_config_file(Path(sys.argv[1])), Path(sys.argv[1]))
PY
    echo "run_log=$log_path"
    ;;
  status)
    STATUS_LOG="${STATUS_LOG:-}"
    if [ -z "$STATUS_LOG" ]; then
      config_stem="$(basename "$TRAIN_CONFIG" .jsonc)"
      STATUS_LOG="$(ls -t "$RUN_LOG_DIR"/"${config_stem}"_run_*.log 2>/dev/null | head -1 || true)"
    fi
    if [ -z "$STATUS_LOG" ]; then
      echo "No 300-run log found under $RUN_LOG_DIR" >&2
      exit 1
    fi
    PYTHONPATH=src/train uv run python - "$TRAIN_CONFIG" "$STATUS_LOG" <<'PY'
import json
import re
import subprocess
import sys
from pathlib import Path

from config_utils import load_config_file
from json_io import load_jsonl_objects

config_path = Path(sys.argv[1])
log_path = Path(sys.argv[2])
cfg = load_config_file(config_path)
text = log_path.read_text(errors="replace") if log_path.exists() else ""

step_matches = list(re.finditer(r"\|\s*(\d+)/(\d+)\s*\[", text))
last_step = int(step_matches[-1].group(1)) if step_matches else None
total_steps = int(step_matches[-1].group(2)) if step_matches else int(cfg["train"]["steps"])

metric_matches = list(
    re.finditer(
        r"Loss:\s*([0-9.]+)\s+recon:\s*([0-9.]+)\s+fov:\s*([0-9.]+)\s+r:\s*([0-9.]+)",
        text,
    )
)
last_metrics = None
if metric_matches:
    match = metric_matches[-1]
    last_metrics = {
        "loss": float(match.group(1)),
        "recon": float(match.group(2)),
        "fov": float(match.group(3)),
        "radius": float(match.group(4)),
    }

def parse_duration_seconds(value: str) -> int:
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        minutes, seconds = parts
        return minutes * 60 + seconds
    hours, minutes, seconds = parts[-3:]
    return hours * 3600 + minutes * 60 + seconds


rate_matches = list(re.finditer(r"(\d+(?::\d+){1,2})<([^,>\]]+),\s*([0-9.]+)s/it\]", text))
last_rate = None
if rate_matches:
    match = rate_matches[-1]
    last_rate = {
        "elapsed_seconds": parse_duration_seconds(match.group(1)),
        "eta_text": match.group(2).strip(),
        "seconds_per_step": float(match.group(3)),
    }

timing_matches = list(re.finditer(r"Timing step\s+(\d+):\s+([^\n\r]+)", text))
last_timing = None
if timing_matches:
    match = timing_matches[-1]
    terms = {}
    for item in match.group(2).split():
        if "=" not in item or not item.endswith("s"):
            continue
        key, value = item[:-1].split("=", 1)
        try:
            terms[key] = float(value)
        except ValueError:
            continue
    last_timing = {
        "step": int(match.group(1)),
        "terms_seconds": terms,
    }

cache_dir = Path(cfg["features"]["cache_dir"])
manifest_path = Path(cfg["data"]["manifest_path"])
manifest_count = 0
if manifest_path.exists():
    manifest_count = len(load_jsonl_objects(manifest_path))
cache_files = len(list(cache_dir.glob("*.pt"))) if cache_dir.exists() else 0

screen_result = subprocess.run(["screen", "-ls"], check=False, capture_output=True, text=True)
all_screen_lines = [
    line.strip()
    for line in screen_result.stdout.splitlines()
    if "dynaworld_" in line
]

def screen_matches_config(line: str) -> bool:
    config_name = config_path.stem
    is_capacity_screen = "_2048_train_" in line
    if "2048splats_capacity" in config_name:
        return is_capacity_screen
    return not is_capacity_screen

config_screen_lines = [line for line in all_screen_lines if screen_matches_config(line)]
utility_screen_lines = [
    line
    for line in config_screen_lines
    if "dynaworld_300_64f_512_prebake" in line or "dynaworld_300_64f_512_wait_cache_run" in line
]
train_screen_lines = [
    line
    for line in config_screen_lines
    if line not in utility_screen_lines
]

payload = {
    "config": str(config_path),
    "log": str(log_path),
    "screen_sessions": config_screen_lines,
    "all_dynaworld_screen_sessions": all_screen_lines,
    "training_screen_sessions": train_screen_lines,
    "utility_screen_sessions": utility_screen_lines,
    "last_step": last_step,
    "total_steps": total_steps,
    "last_metrics": last_metrics,
    "last_rate": last_rate,
    "last_timing": last_timing,
    "cache_hits_logged": text.count("[features] cache hit"),
    "cache_misses_logged": text.count("[features] cache miss"),
    "feature_cache_dir": str(cache_dir),
    "feature_cache_files": cache_files,
    "manifest_records": manifest_count,
    "feature_cache_coverage": (cache_files / manifest_count) if manifest_count else None,
    "wandb_offline_run": next(
        (line.rsplit(" ", 1)[-1] for line in text.splitlines() if "Run data is saved locally in " in line),
        None,
    ),
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
    ;;
  timing-summary)
    STATUS_LOG="${STATUS_LOG:-}"
    if [ -z "$STATUS_LOG" ]; then
      config_stem="$(basename "$TRAIN_CONFIG" .jsonc)"
      STATUS_LOG="$(ls -t "$RUN_LOG_DIR"/"${config_stem}"_run_*.log 2>/dev/null | head -1 || true)"
    fi
    if [ -z "$STATUS_LOG" ]; then
      echo "No 300-run log found under $RUN_LOG_DIR" >&2
      exit 1
    fi
    uv run python - "$TRAIN_CONFIG" "$STATUS_LOG" <<'PY'
import json
import re
import statistics
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
log_path = Path(sys.argv[2])
text = log_path.read_text(errors="replace") if log_path.exists() else ""

rows = []
for match in re.finditer(r"Timing step\s+(\d+):\s+([^\n\r]+)", text):
    terms = {}
    for item in match.group(2).split():
        if "=" not in item or not item.endswith("s"):
            continue
        key, value = item[:-1].split("=", 1)
        try:
            terms[key] = float(value)
        except ValueError:
            continue
    if terms:
        rows.append((int(match.group(1)), terms))

def stats(values):
    if not values:
        return None
    return {
        "last": round(values[-1], 4),
        "median": round(statistics.median(values), 4),
        "mean": round(statistics.mean(values), 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
    }

keys = (
    "step_total",
    "backward",
    "forward_decode",
    "vjepa_feature_loss",
    "render/rasterize",
    "render_view_total",
    "sample_clip",
)
payload = {
    "config": str(config_path),
    "log": str(log_path),
    "timing_record_count": len(rows),
    "first_timing_step": rows[0][0] if rows else None,
    "last_timing_step": rows[-1][0] if rows else None,
    "terms_seconds": {
        key: stats([terms[key] for _, terms in rows if key in terms])
        for key in keys
    },
}
latest = rows[-1][1] if rows else {}
latest_step_total = latest.get("step_total")
if latest_step_total:
    payload["latest_fraction_of_step_total"] = {
        "backward": round(latest.get("backward", 0.0) / latest_step_total, 4),
        "render/rasterize": round(latest.get("render/rasterize", 0.0) / latest_step_total, 4),
        "vjepa_feature_loss": round(latest.get("vjepa_feature_loss", 0.0) / latest_step_total, 4),
        "forward_decode": round(latest.get("forward_decode", 0.0) / latest_step_total, 4),
    }
print(json.dumps(payload, indent=2, sort_keys=True))
PY
    ;;
  *)
    echo "Usage: $0 {build|audit|load-check|resolve|probe|bench|cache-status|prebake|wait-cache-run|run|status|timing-summary}" >&2
    exit 2
    ;;
esac
