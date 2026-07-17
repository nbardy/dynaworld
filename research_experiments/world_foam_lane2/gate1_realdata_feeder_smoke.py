#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


DYNAWORLD = Path(__file__).resolve().parents[2]
TRAIN_SRC = DYNAWORLD / "src" / "train"
DEFAULT_CONFIG = (
    DYNAWORLD
    / "src"
    / "train_configs"
    / "local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_32_smoke.jsonc"
)

if str(TRAIN_SRC) not in sys.path:
    sys.path.insert(0, str(TRAIN_SRC))

from powerfoam_training_data import load_powerfoam_training_data  # noqa: E402


def _shape(value: torch.Tensor | None) -> list[int] | None:
    return None if value is None else list(value.shape)


def _finite(value: torch.Tensor | None) -> bool | None:
    return None if value is None else bool(torch.isfinite(value).all().item())


def _load_config(path: Path, *, max_frames: int | None, render_size: int | None) -> dict[str, Any]:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["data"] = dict(cfg["data"])
    cfg["render"] = dict(cfg["render"])
    manifest = Path(cfg["data"]["multicam_manifest"])
    if not manifest.is_absolute():
        cfg["data"]["multicam_manifest"] = str(DYNAWORLD / manifest)
    if max_frames is not None:
        cfg["data"]["max_frames"] = int(max_frames)
    if render_size is not None:
        cfg["render"]["render_size"] = int(render_size)
    return cfg


def run_smoke(*, config_path: Path, max_frames: int | None, render_size: int | None) -> dict[str, Any]:
    cfg = _load_config(config_path, max_frames=max_frames, render_size=render_size)
    data = load_powerfoam_training_data(cfg, torch.device("cpu"))
    targets = data["targets"]
    sample_rays = data["sample_rays"]
    heldout_targets = data["heldout_targets"]
    heldout_rays = data["heldout_rays"]
    frame_count = int(data["frame_count"])
    train_views = list(data["train_views"])
    heldout_views = list(data["heldout_views"])

    if len(train_views) > 1 and sample_rays.shape[0] >= 2 * frame_count:
        first_view = sample_rays[0]
        second_view = sample_rays[frame_count]
        origin_delta = float((first_view[..., :3] - second_view[..., :3]).abs().max().item())
        direction_delta = float((first_view[..., 3:] - second_view[..., 3:]).abs().max().item())
    else:
        origin_delta = 0.0
        direction_delta = 0.0

    train_ray_dir_norm = sample_rays[..., 3:].norm(dim=-1)
    heldout_ray_dir_norm = None if heldout_rays is None else heldout_rays[..., 3:].norm(dim=-1)
    acceptance = {
        "loaded_real_multicam_bundle": str(cfg["data"]["frame_source"]) == "multicam_val",
        "train_targets_shape_is_expected": list(targets.shape)
        == [len(train_views) * frame_count, 3, int(cfg["render"]["render_size"]), int(cfg["render"]["render_size"])],
        "train_rays_shape_is_expected": list(sample_rays.shape)
        == [len(train_views) * frame_count, int(cfg["render"]["render_size"]), int(cfg["render"]["render_size"]), 6],
        "heldout_targets_present": heldout_targets is not None,
        "heldout_rays_present": heldout_rays is not None,
        "all_targets_finite": bool(torch.isfinite(targets).all().item()) and _finite(heldout_targets) is not False,
        "all_rays_finite": bool(torch.isfinite(sample_rays).all().item()) and _finite(heldout_rays) is not False,
        "train_ray_directions_nonzero": bool((train_ray_dir_norm > 0.0).all().item()),
        "heldout_ray_directions_nonzero": bool(
            True if heldout_ray_dir_norm is None else (heldout_ray_dir_norm > 0.0).all().item()
        ),
        "train_views_have_distinct_rays": origin_delta > 0.0 or direction_delta > 0.0,
    }
    return {
        "benchmark": "world_foam_lane2_gate1_realdata_feeder_smoke",
        "status": "ok" if all(acceptance.values()) else "failed",
        "gate": "1_feeder",
        "config_path": str(config_path),
        "sample_id": data["source_label"],
        "train_views": train_views,
        "heldout_views": heldout_views,
        "pose_source": data["pose_source"],
        "frame_count": frame_count,
        "video_fps": data["video_fps"],
        "render_size": int(cfg["render"]["render_size"]),
        "target_shapes": {
            "train_targets": _shape(targets),
            "train_sample_rays": _shape(sample_rays),
            "heldout_targets": _shape(heldout_targets),
            "heldout_sample_rays": _shape(heldout_rays),
        },
        "finite_checks": {
            "train_targets": _finite(targets),
            "train_sample_rays": _finite(sample_rays),
            "heldout_targets": _finite(heldout_targets),
            "heldout_sample_rays": _finite(heldout_rays),
        },
        "ray_checks": {
            "train_ray_direction_min_norm": float(train_ray_dir_norm.min().item()),
            "train_ray_direction_max_norm": float(train_ray_dir_norm.max().item()),
            "heldout_ray_direction_min_norm": None
            if heldout_ray_dir_norm is None
            else float(heldout_ray_dir_norm.min().item()),
            "heldout_ray_direction_max_norm": None
            if heldout_ray_dir_norm is None
            else float(heldout_ray_dir_norm.max().item()),
            "train_view_origin_max_abs_delta": origin_delta,
            "train_view_direction_max_abs_delta": direction_delta,
        },
        "world_foam_renderer_status": "not_connected_full_frame_shader_missing_u_v_t_image_op",
        "acceptance": acceptance,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="World Foam Lane 2 real-data feeder smoke.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--render-size", type=int)
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_smoke(config_path=args.config, max_frames=args.max_frames, render_size=args.render_size)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
