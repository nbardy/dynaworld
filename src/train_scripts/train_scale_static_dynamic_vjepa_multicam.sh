#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT_DIR"

BASE_CONFIG="src/train_configs/local_mac_scale_static_dynamic_vjepa_multicam_train2_holdout1_F32_256_16f_8192splats.jsonc"
TRAINER="src/train/train.py"
DEFAULT_MANIFEST="src/dataset_configs/multicam_train2_holdout1_5sample_128_4fps_16f_manifest.jsonl"
DEFAULT_SPLIT="${MULTICAM_SPLIT:-train2_holdout1}"

usage() {
  cat >&2 <<'EOF'
Usage:
  train_scale_static_dynamic_vjepa_multicam.sh check [manifest]
  train_scale_static_dynamic_vjepa_multicam.sh sample <filtered-index> [manifest]
  train_scale_static_dynamic_vjepa_multicam.sh sweep <start-index> <count> [manifest]

Environment:
  MULTICAM_SPLIT     Manifest split to select. Default: train2_holdout1.

Notes:
  This is a per-record launcher for the existing single-record multicam trainer.
  It patches train/heldout cameras from each manifest row into a temp config.
EOF
}

manifest_path() {
  local maybe_manifest="${1:-}"
  if [[ -n "$maybe_manifest" ]]; then
    printf '%s\n' "$maybe_manifest"
  else
    printf '%s\n' "${MULTICAM_MANIFEST:-$DEFAULT_MANIFEST}"
  fi
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing $label: $path" >&2
    exit 1
  fi
}

write_sample_config() {
  local manifest="$1"
  local split="$2"
  local sample_index="$3"
  local output_path="$4"
  PYTHONPATH=src/train uv run python - "$BASE_CONFIG" "$manifest" "$split" "$sample_index" "$output_path" <<'PY'
import json
import re
import sys
from pathlib import Path

from config_utils import load_config_file
from train_artifacts import write_json

base_config, manifest, split, sample_index, output_path = sys.argv[1:6]
sample_index = int(sample_index)

records = []
with Path(manifest).open() as handle:
    for line in handle:
        if not line.strip():
            continue
        record = json.loads(line)
        if str(record.get("split", "")) == split:
            records.append(record)

if sample_index < 0 or sample_index >= len(records):
    raise SystemExit(
        f"sample index {sample_index} is out of range for {len(records)} records "
        f"in split {split!r} from {manifest}"
    )

record = records[sample_index]
dataset_to_rig = {
    "aist_dance_db": "aist",
    "deepview_video": "deepview",
    "neural_3d_video": "neural_3d_video",
    "vivo": "vivo",
}
dataset = str(record.get("dataset", ""))
rig_init = str(record.get("rig_init") or dataset_to_rig.get(dataset, ""))
if not rig_init:
    raise SystemExit(
        f"record {record.get('sample_id', sample_index)!r} has dataset={dataset!r}; "
        "add record.rig_init or use a supported calibrated multicam dataset"
    )

train_cameras = record.get("train_cameras") or [record["source_camera"]]
heldout_cameras = record.get("heldout_cameras") or [record["target_camera"]]
anchor_camera = record.get("anchor_camera") or train_cameras[0]
condition_camera = record.get("condition_camera") or anchor_camera
sample_id = str(record.get("sample_id", f"sample_{sample_index:04d}"))
safe_sample_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", sample_id)[:96]

cfg = load_config_file(base_config)
cfg["data"].update(
    {
        "multicam_manifest": manifest,
        "multicam_split": split,
        "multicam_sample_id": None,
        "multicam_sample_index": sample_index,
        "multicam_train_cameras": [str(camera) for camera in train_cameras],
        "multicam_heldout_cameras": [str(camera) for camera in heldout_cameras],
        "multicam_heldout_camera": None,
        "multicam_anchor_camera": str(anchor_camera),
        "multicam_condition_camera": str(condition_camera),
    }
)
cfg["camera"]["rig_init"] = rig_init

base_cache_key = str(cfg["features"]["sample_cache_key"])
cfg["features"]["sample_cache_key"] = f"{base_cache_key}-{safe_sample_id}-idx{sample_index:04d}"

base_run_name = str(cfg["logging"]["wandb_run_name"])
cfg["logging"]["wandb_run_name"] = f"{base_run_name}-{safe_sample_id}-idx{sample_index:04d}"
tags = list(cfg["logging"].get("wandb_tags", []))
tags.extend([f"manifest-split-{split}", f"sample-index-{sample_index:04d}", f"dataset-{dataset}"])
cfg["logging"]["wandb_tags"] = tags

write_json(Path(output_path), cfg, sort_keys=False)
print(
    f"Wrote {output_path} for sample_index={sample_index} sample_id={sample_id} "
    f"dataset={dataset} rig_init={rig_init} train={train_cameras} heldout={heldout_cameras}"
)
PY
}

check_config() {
  local manifest="$1"
  local tmp_config
  tmp_config="$(mktemp -t dynaworld_scale_vjepa_config.XXXXXX)"
  trap 'rm -f "$tmp_config"' RETURN
  write_sample_config "$manifest" "$DEFAULT_SPLIT" 0 "$tmp_config"
  PYTHONPATH=src/train uv run python - "$tmp_config" <<'PY'
import sys
from config_utils import load_config_file
from trainer_registry import resolve_config_for_arch

cfg = load_config_file(sys.argv[1])
resolved = resolve_config_for_arch(cfg, sys.argv[1])
print(
    "Resolved config: "
    f"arch={resolved['arch']} "
    f"manifest={resolved['data']['multicam_manifest']} "
    f"split={resolved['data']['multicam_split']} "
    f"sample_index={resolved['data']['multicam_sample_index']}"
)
PY
}

run_sample() {
  local manifest="$1"
  local sample_index="$2"
  local tmp_config
  tmp_config="$(mktemp -t "dynaworld_scale_vjepa_sample_${sample_index}.XXXXXX")"
  trap 'rm -f "$tmp_config"' RETURN
  write_sample_config "$manifest" "$DEFAULT_SPLIT" "$sample_index" "$tmp_config"
  echo "Config: $tmp_config"
  PYTHONPATH=src/train uv run python "$TRAINER" "$tmp_config"
}

MODE="${1:-check}"
case "$MODE" in
  check)
    MANIFEST="$(manifest_path "${2:-}")"
    require_file "$BASE_CONFIG" "base config"
    require_file "$TRAINER" "trainer"
    require_file "$MANIFEST" "manifest"
    check_config "$MANIFEST"
    ;;
  sample)
    if [[ "$#" -lt 2 || "$#" -gt 3 ]]; then
      usage
      exit 1
    fi
    MANIFEST="$(manifest_path "${3:-}")"
    require_file "$BASE_CONFIG" "base config"
    require_file "$TRAINER" "trainer"
    require_file "$MANIFEST" "manifest"
    run_sample "$MANIFEST" "$2"
    ;;
  sweep)
    if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
      usage
      exit 1
    fi
    MANIFEST="$(manifest_path "${4:-}")"
    require_file "$BASE_CONFIG" "base config"
    require_file "$TRAINER" "trainer"
    require_file "$MANIFEST" "manifest"
    START_INDEX="$2"
    COUNT="$3"
    for ((offset = 0; offset < COUNT; offset++)); do
      run_sample "$MANIFEST" "$((START_INDEX + offset))"
    done
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac
