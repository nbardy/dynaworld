#!/usr/bin/env python3
"""Visualize dataset camera rigs in the trainer's anchor-relative frame."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = REPO_ROOT / "src" / "train"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from config_utils import load_config_file  # noqa: E402
from multicam_video_data import (  # noqa: E402
    CAMXTIME_DATASETS,
    camera_from_K_w2c,
    camxtime_camera_count,
    load_camxtime_camera_data,
    make_aist_multiview_cameras,
    make_camxtime_multiview_cameras,
    make_deepview_multiview_cameras,
    make_neural_3d_multiview_cameras,
    make_orthogonal_origin_multiview_cameras,
    make_vivo_multiview_cameras,
    select_multicam_record,
    validate_multicam_camera_split,
)
from train_multicam_precomputed_feature_implicit_dynamic import (  # noqa: E402
    MulticamPrecomputedFeatureImplicitTrainer,
)


@dataclass(frozen=True)
class CameraRigPayload:
    metadata: dict[str, Any]
    cameras: list[dict[str, Any]]


def _repo_text(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _as_camera_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def _camxtime_camera_names(record: dict[str, Any]) -> list[str]:
    camera_count = camxtime_camera_count(load_camxtime_camera_data(record))
    return [f"camera_{index:03d}" for index in range(camera_count)]


def _resolve_split(
    data_cfg: dict[str, Any],
    record: dict[str, Any],
    *,
    input_camera: str | None,
    train_cameras_override: str | None,
    heldout_cameras_override: str | None,
    all_camxtime_cameras: bool,
    all_camxtime_role: str,
) -> tuple[list[str], list[str], str, str]:
    train_cameras = (
        _as_camera_list(train_cameras_override)
        or _as_camera_list(data_cfg.get("multicam_train_cameras"))
        or _as_camera_list(record.get("train_cameras"))
        or [str(record["source_camera"])]
    )
    heldout_cameras = (
        _as_camera_list(heldout_cameras_override)
        or _as_camera_list(data_cfg.get("multicam_heldout_cameras"))
        or _as_camera_list(record.get("heldout_cameras"))
        or [str(data_cfg.get("multicam_heldout_camera") or record.get("heldout_camera") or record["target_camera"])]
    )
    condition_camera = str(
        input_camera
        or data_cfg.get("multicam_condition_camera")
        or record.get("condition_camera")
        or data_cfg.get("multicam_anchor_camera")
        or record.get("anchor_camera")
        or train_cameras[0]
    )
    anchor_camera = str(data_cfg.get("multicam_anchor_camera") or record.get("anchor_camera") or condition_camera)
    if input_camera is not None and input_camera not in train_cameras:
        train_cameras = [input_camera] + [camera for camera in train_cameras if camera != input_camera]
    if all_camxtime_cameras:
        dataset = str(record.get("dataset") or "")
        if dataset not in CAMXTIME_DATASETS:
            raise ValueError("--all-camxtime-cameras requires a CamXTime record.")
        all_names = _camxtime_camera_names(record)
        if condition_camera not in all_names:
            raise ValueError(f"condition/input camera {condition_camera!r} is not present in CamXTime camera_data.json.")
        if all_camxtime_role == "heldout_except_input":
            train_cameras = [condition_camera]
            heldout_cameras = [camera for camera in all_names if camera != condition_camera]
        elif all_camxtime_role == "train_except_heldout":
            heldout_set = set(heldout_cameras)
            train_cameras = [camera for camera in all_names if camera not in heldout_set]
            if condition_camera not in train_cameras:
                raise ValueError(
                    f"condition/input camera {condition_camera!r} is in the heldout set; "
                    "choose a different --input-camera or --heldout-cameras."
                )
        else:
            raise ValueError(
                "--all-camxtime-role must be 'heldout_except_input' or 'train_except_heldout'."
            )
        anchor_camera = condition_camera
    validate_multicam_camera_split(
        train_cameras=train_cameras,
        heldout_cameras=heldout_cameras,
        anchor_camera=anchor_camera,
        condition_camera=condition_camera,
    )
    return train_cameras, heldout_cameras, anchor_camera, condition_camera


def _make_rig_tensors(
    record: dict[str, Any],
    camera_cfg: dict[str, Any],
    *,
    train_cameras: list[str],
    heldout_cameras: list[str],
    anchor_camera: str,
    frame_count: int,
    target_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    rig_init = str(camera_cfg.get("rig_init", "deepview")).lower()
    kwargs = {
        "record": record,
        "train_cameras": train_cameras,
        "heldout_cameras": heldout_cameras,
        "anchor_camera": anchor_camera,
        "T": int(frame_count),
        "H": int(target_size),
        "W": int(target_size),
        "device": device,
    }
    if rig_init == "camxtime":
        return make_camxtime_multiview_cameras(**kwargs)
    if rig_init == "deepview":
        return make_deepview_multiview_cameras(**kwargs)
    if rig_init == "aist":
        return make_aist_multiview_cameras(
            **kwargs,
            translation_scale=float(camera_cfg.get("aist_translation_scale", 1.0)),
        )
    if rig_init == "neural_3d_video":
        return make_neural_3d_multiview_cameras(
            **kwargs,
            translation_scale=float(camera_cfg.get("n3d_translation_scale", 1.0)),
        )
    if rig_init == "vivo":
        return make_vivo_multiview_cameras(
            **kwargs,
            translation_scale=float(camera_cfg.get("vivo_translation_scale", 1.0)),
        )
    if rig_init == "orthogonal_origin":
        return make_orthogonal_origin_multiview_cameras(
            view_count=len(train_cameras),
            heldout_count=len(heldout_cameras),
            T=int(frame_count),
            H=int(target_size),
            W=int(target_size),
            radius=float(camera_cfg.get("rig_radius", camera_cfg.get("base_radius", 3.0))),
            fov_degrees=float(camera_cfg.get("base_fov_degrees", 60.0)),
            device=device,
        )
    raise ValueError(f"Unsupported camera.rig_init for visualization: {rig_init!r}")


def _camera_entry(
    *,
    name: str,
    role: str,
    is_condition: bool,
    K: torch.Tensor,
    w2c: torch.Tensor,
) -> dict[str, Any]:
    camera = camera_from_K_w2c(K.detach().cpu().float(), w2c.detach().cpu().float())
    c2w = camera.camera_to_world.detach().cpu().float()
    return {
        "name": name,
        "role": role,
        "is_condition": bool(is_condition),
        "center": c2w[:3, 3].tolist(),
        "right": c2w[:3, 0].tolist(),
        "up": c2w[:3, 1].tolist(),
        "forward": c2w[:3, 2].tolist(),
        "fx": float(torch.as_tensor(camera.fx)),
        "fy": float(torch.as_tensor(camera.fy)),
        "cx": float(torch.as_tensor(camera.cx)),
        "cy": float(torch.as_tensor(camera.cy)),
        "camera_to_world": c2w.tolist(),
        "world_to_camera": w2c.detach().cpu().float().tolist(),
    }


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _norm(values: list[float]) -> float:
    return float(sum(value * value for value in values) ** 0.5)


def _sub(left: list[float], right: list[float]) -> list[float]:
    return [a - b for a, b in zip(left, right)]


def _cross(left: list[float], right: list[float]) -> list[float]:
    return [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _camera_rotation(entry: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(
        [
            [entry["right"][0], entry["up"][0], entry["forward"][0]],
            [entry["right"][1], entry["up"][1], entry["forward"][1]],
            [entry["right"][2], entry["up"][2], entry["forward"][2]],
        ],
        dtype=torch.float64,
    )


def _axis_focus_point(centers: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    directions = directions / torch.clamp(torch.linalg.norm(directions, dim=1, keepdim=True), min=1e-12)
    identity = torch.eye(3, dtype=torch.float64)
    projectors = identity.unsqueeze(0) - directions.unsqueeze(2) @ directions.unsqueeze(1)
    lhs = torch.sum(projectors, dim=0)
    rhs = torch.sum(projectors @ centers.unsqueeze(2), dim=0).squeeze(1)
    return torch.linalg.pinv(lhs) @ rhs


def _add_pose_diagnostics(
    cameras: list[dict[str, Any]],
    *,
    condition_camera: str,
) -> dict[str, Any]:
    if not cameras:
        return {}
    condition = next((camera for camera in cameras if camera["name"] == condition_camera), cameras[0])
    condition_center = [float(value) for value in condition["center"]]
    condition_forward = [float(value) for value in condition["forward"]]

    centers = torch.tensor([camera["center"] for camera in cameras], dtype=torch.float64)
    directions = torch.tensor([camera["forward"] for camera in cameras], dtype=torch.float64)
    rotations = torch.stack([_camera_rotation(camera) for camera in cameras], dim=0)
    identity = torch.eye(3, dtype=torch.float64)
    determinants = torch.linalg.det(rotations)
    orthogonality = torch.linalg.norm(rotations.transpose(1, 2) @ rotations - identity, dim=(1, 2))
    radii = torch.linalg.norm(centers, dim=1)
    focus = _axis_focus_point(centers, directions)
    focus_depths = []
    focus_misses = []

    for camera in cameras:
        center = [float(value) for value in camera["center"]]
        forward = [float(value) for value in camera["forward"]]
        to_anchor_origin = [-value for value in center]
        to_focus = [float(value) for value in (focus - torch.tensor(center, dtype=torch.float64)).tolist()]
        center_norm = _norm(center)
        forward_norm = max(_norm(forward), 1e-12)
        anchor_origin_depth = _dot(to_anchor_origin, forward) / forward_norm
        anchor_origin_ray_miss = 0.0 if center_norm < 1e-9 else _norm(_cross(forward, to_anchor_origin)) / forward_norm
        focus_depth = _dot(to_focus, forward) / forward_norm
        focus_ray_miss = _norm(_cross(forward, to_focus)) / forward_norm
        distance_to_condition = _norm(_sub(center, condition_center))
        forward_dot = _dot(forward, condition_forward) / (forward_norm * max(_norm(condition_forward), 1e-12))
        camera["anchor_origin_depth_along_forward"] = anchor_origin_depth
        camera["anchor_origin_ray_miss_distance"] = anchor_origin_ray_miss
        camera["focus_depth_along_forward"] = focus_depth
        camera["focus_ray_miss_distance"] = focus_ray_miss
        camera["distance_to_condition"] = distance_to_condition
        camera["forward_angle_from_condition_degrees"] = float(
            torch.rad2deg(torch.acos(torch.tensor(_clamp(forward_dot, -1.0, 1.0)))).item()
        )
        focus_depths.append(focus_depth)
        focus_misses.append(focus_ray_miss)

    pairwise = torch.cdist(centers, centers)
    if len(cameras) > 1:
        pairwise.fill_diagonal_(float("inf"))
        nearest = torch.min(pairwise)
    else:
        nearest = torch.tensor(0.0, dtype=torch.float64)

    role_counts: dict[str, int] = {}
    for camera in cameras:
        role = str(camera["role"])
        role_counts[role] = role_counts.get(role, 0) + 1

    return {
        "camera_count": len(cameras),
        "role_counts": role_counts,
        "position_min": torch.min(centers, dim=0).values.tolist(),
        "position_max": torch.max(centers, dim=0).values.tolist(),
        "radius_min": float(torch.min(radii).item()),
        "radius_max": float(torch.max(radii).item()),
        "radius_mean": float(torch.mean(radii).item()),
        "nearest_camera_distance": float(nearest.item()),
        "rotation_det_min": float(torch.min(determinants).item()),
        "rotation_det_max": float(torch.max(determinants).item()),
        "rotation_orthogonality_error_max": float(torch.max(orthogonality).item()),
        "condition_origin_error": _norm(condition_center),
        "condition_rotation_from_identity_fro": float(torch.linalg.norm(_camera_rotation(condition) - identity).item()),
        "axis_focus_point": focus.tolist(),
        "axis_focus_depth_min": min(focus_depths),
        "axis_focus_depth_max": max(focus_depths),
        "axis_focus_rms_miss_distance": float((sum(value * value for value in focus_misses) / len(focus_misses)) ** 0.5),
        "axis_focus_max_miss_distance": max(focus_misses),
        "cameras_with_focus_behind": [
            camera["name"] for camera in cameras if float(camera["focus_depth_along_forward"]) < -1e-6
        ],
    }


def build_payload(
    config_path: Path,
    *,
    input_camera: str | None,
    train_cameras_override: str | None,
    heldout_cameras_override: str | None,
    all_camxtime_cameras: bool,
    all_camxtime_role: str,
    frame_count: int,
    target_size: int | None,
    frame_index: int,
) -> CameraRigPayload:
    config = MulticamPrecomputedFeatureImplicitTrainer.resolve_config(load_config_file(config_path))
    data_cfg = config["data"]
    camera_cfg = config["camera"]
    record = select_multicam_record(data_cfg)
    resolved_size = int(target_size or config["model"]["size"])
    train_cameras, heldout_cameras, anchor_camera, condition_camera = _resolve_split(
        data_cfg,
        record,
        input_camera=input_camera,
        train_cameras_override=train_cameras_override,
        heldout_cameras_override=heldout_cameras_override,
        all_camxtime_cameras=all_camxtime_cameras,
        all_camxtime_role=all_camxtime_role,
    )
    train_K, train_w2c, heldout_K, heldout_w2c, pose_source = _make_rig_tensors(
        record,
        camera_cfg,
        train_cameras=train_cameras,
        heldout_cameras=heldout_cameras,
        anchor_camera=anchor_camera,
        frame_count=frame_count,
        target_size=resolved_size,
        device=torch.device("cpu"),
    )
    frame = max(0, min(int(frame_index), int(frame_count) - 1))
    cameras = [
        _camera_entry(
            name=name,
            role="train",
            is_condition=name == condition_camera,
            K=train_K[index],
            w2c=train_w2c[index, frame],
        )
        for index, name in enumerate(train_cameras)
    ]
    cameras.extend(
        _camera_entry(
            name=name,
            role="heldout",
            is_condition=False,
            K=heldout_K[index],
            w2c=heldout_w2c[index, frame],
        )
        for index, name in enumerate(heldout_cameras)
    )
    diagnostics = _add_pose_diagnostics(cameras, condition_camera=condition_camera)
    metadata = {
        "config": _repo_text(config_path),
        "sample_id": record.get("sample_id"),
        "dataset": record.get("dataset"),
        "scene": record.get("scene"),
        "rig_init": camera_cfg.get("rig_init"),
        "pose_source": pose_source,
        "frame_index": frame,
        "frame_count": int(frame_count),
        "target_size": resolved_size,
        "train_cameras": train_cameras,
        "heldout_cameras": heldout_cameras,
        "anchor_camera": anchor_camera,
        "condition_camera": condition_camera,
        "coordinate_frame": "anchor_relative_opencv_plus_z_forward",
        "diagnostics": diagnostics,
    }
    return CameraRigPayload(metadata=metadata, cameras=cameras)


def _html(payload: CameraRigPayload) -> str:
    data = json.dumps({"metadata": payload.metadata, "cameras": payload.cameras}, sort_keys=True)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Multicam Rig Diagnostic</title>
<style>
body {{ margin: 0; font-family: system-ui, -apple-system, sans-serif; background: #f7f6f0; color: #202020; }}
#wrap {{ display: grid; grid-template-columns: minmax(0, 1fr) 360px; min-height: 100vh; }}
canvas {{ width: 100%; height: 100vh; display: block; background: #fbfaf6; }}
aside {{ border-left: 1px solid #d5d1c4; padding: 16px; overflow: auto; background: #f0eee6; }}
h1 {{ font-size: 18px; margin: 0 0 10px; }}
pre {{ white-space: pre-wrap; word-break: break-word; font-size: 12px; line-height: 1.4; }}
.legend {{ display: flex; gap: 12px; margin: 12px 0; font-size: 13px; }}
.swatch {{ display: inline-block; width: 12px; height: 12px; margin-right: 5px; vertical-align: -1px; }}
@media (max-width: 900px) {{ #wrap {{ grid-template-columns: 1fr; }} canvas {{ height: 70vh; }} aside {{ border-left: 0; border-top: 1px solid #d5d1c4; }} }}
</style>
</head>
<body>
<div id="wrap">
<canvas id="scene"></canvas>
<aside>
<h1>Multicam Rig Diagnostic</h1>
<div class="legend">
  <span><i class="swatch" style="background:#1f77b4"></i>train</span>
  <span><i class="swatch" style="background:#d62728"></i>heldout</span>
  <span><i class="swatch" style="background:#2ca02c"></i>condition</span>
</div>
<p>Drag to orbit. Wheel to zoom. The coordinate frame is the trainer frame: the anchor/input camera is at identity when it is the anchor.</p>
<pre id="meta"></pre>
</aside>
</div>
<script>
const data = {data};
const canvas = document.getElementById('scene');
const ctx = canvas.getContext('2d');
document.getElementById('meta').textContent = JSON.stringify(data.metadata, null, 2);
let yaw = -0.85, pitch = 0.45, zoom = 1.0;
let dragging = false, lastX = 0, lastY = 0;
function resize() {{
  const dpr = Math.max(window.devicePixelRatio || 1, 1);
  canvas.width = Math.floor(canvas.clientWidth * dpr);
  canvas.height = Math.floor(canvas.clientHeight * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}}
function add(a,b) {{ return [a[0]+b[0], a[1]+b[1], a[2]+b[2]]; }}
function sub(a,b) {{ return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]; }}
function mul(a,s) {{ return [a[0]*s, a[1]*s, a[2]*s]; }}
const centers = data.cameras.map(c => c.center);
const mins = [0,1,2].map(i => Math.min(...centers.map(p => p[i]), -0.2));
const maxs = [0,1,2].map(i => Math.max(...centers.map(p => p[i]), 0.2));
const mid = [0,1,2].map(i => (mins[i] + maxs[i]) * 0.5);
const extent = Math.max(...[0,1,2].map(i => maxs[i] - mins[i]), 1e-3);
const frustumLength = Math.max(extent * 0.07, 0.08);
const frustumWidth = frustumLength * 0.45;
function rotate(p) {{
  const q = sub(p, mid);
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const x1 = cy * q[0] + sy * q[2];
  const z1 = -sy * q[0] + cy * q[2];
  const y2 = cp * q[1] - sp * z1;
  const z2 = sp * q[1] + cp * z1;
  return [x1, y2, z2];
}}
function project(p) {{
  const r = rotate(p);
  const scale = Math.min(canvas.clientWidth, canvas.clientHeight) * 0.78 * zoom / extent;
  return [canvas.clientWidth * 0.5 + r[0] * scale, canvas.clientHeight * 0.52 - r[1] * scale, r[2]];
}}
function colorFor(c) {{ return c.is_condition ? '#2ca02c' : (c.role === 'train' ? '#1f77b4' : '#d62728'); }}
function line(a, b, color, width=1) {{
  const pa = project(a), pb = project(b);
  ctx.strokeStyle = color; ctx.lineWidth = width;
  ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
}}
function label(p, text, color) {{
  const pp = project(p);
  ctx.fillStyle = color;
  ctx.font = '12px system-ui, sans-serif';
  ctx.fillText(text, pp[0] + 5, pp[1] - 5);
}}
function drawCamera(c) {{
  const center = c.center;
  const far = add(center, mul(c.forward, frustumLength));
  const rw = mul(c.right, frustumWidth), uw = mul(c.up, frustumWidth);
  const corners = [sub(sub(far,rw),uw), sub(add(far,rw),uw), add(add(far,rw),uw), add(sub(far,rw),uw)];
  const color = colorFor(c);
  for (const corner of corners) line(center, corner, color, c.is_condition ? 2.8 : 1.4);
  for (let i = 0; i < 4; i++) line(corners[i], corners[(i+1)%4], color, c.is_condition ? 2.4 : 1.2);
  line(center, add(center, mul(c.forward, frustumLength * 1.4)), color, c.is_condition ? 3 : 1.8);
  const pc = project(center);
  ctx.fillStyle = color; ctx.beginPath(); ctx.arc(pc[0], pc[1], c.is_condition ? 5 : 3, 0, Math.PI*2); ctx.fill();
  if (c.is_condition || data.cameras.length <= 24) label(center, c.name, color);
}}
function drawAxes() {{
  const o = [0,0,0], s = extent * 0.16;
  line(o, [s,0,0], '#b23a3a', 2); label([s,0,0], '+x', '#b23a3a');
  line(o, [0,s,0], '#287d3c', 2); label([0,s,0], '+y', '#287d3c');
  line(o, [0,0,s], '#2e5aac', 2); label([0,0,s], '+z', '#2e5aac');
}}
function draw() {{
  ctx.clearRect(0,0,canvas.clientWidth,canvas.clientHeight);
  ctx.fillStyle = '#fbfaf6'; ctx.fillRect(0,0,canvas.clientWidth,canvas.clientHeight);
  drawAxes();
  [...data.cameras].sort((a,b) => project(b.center)[2] - project(a.center)[2]).forEach(drawCamera);
  ctx.fillStyle = '#555'; ctx.font = '13px system-ui, sans-serif';
  ctx.fillText(`${{data.cameras.length}} cameras | ${{data.metadata.pose_source}}`, 14, 24);
}}
canvas.addEventListener('mousedown', e => {{ dragging = true; lastX = e.clientX; lastY = e.clientY; }});
window.addEventListener('mouseup', () => dragging = false);
window.addEventListener('mousemove', e => {{
  if (!dragging) return;
  yaw += (e.clientX - lastX) * 0.008;
  pitch = Math.max(-1.45, Math.min(1.45, pitch + (e.clientY - lastY) * 0.008));
  lastX = e.clientX; lastY = e.clientY; draw();
}});
canvas.addEventListener('wheel', e => {{ e.preventDefault(); zoom *= Math.exp(-e.deltaY * 0.001); zoom = Math.max(0.2, Math.min(10, zoom)); draw(); }}, {{passive:false}});
window.addEventListener('resize', resize);
resize();
</script>
</body>
</html>
"""


def write_outputs(payload: CameraRigPayload, output_path: Path, json_path: Path | None) -> tuple[Path, Path]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_html(payload), encoding="utf-8")
    resolved_json = json_path or output_path.with_suffix(".json")
    resolved_json.parent.mkdir(parents=True, exist_ok=True)
    resolved_json.write_text(
        json.dumps({"metadata": payload.metadata, "cameras": payload.cameras}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path, resolved_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a camera-frustum HTML diagnostic for a multicam train config.")
    parser.add_argument("--config", type=Path, required=True, help="Multicam train JSONC config.")
    parser.add_argument("--output", type=Path, required=True, help="HTML output path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional JSON pose dump path.")
    parser.add_argument("--input-camera", default=None, help="Condition/input camera override.")
    parser.add_argument("--train-cameras", default=None, help="Comma-separated train camera override.")
    parser.add_argument("--heldout-cameras", default=None, help="Comma-separated heldout camera override.")
    parser.add_argument(
        "--all-camxtime-cameras",
        action="store_true",
        help="For CamXTime, draw the input camera plus every other camera in camera_data.json.",
    )
    parser.add_argument(
        "--all-camxtime-role",
        choices=("heldout_except_input", "train_except_heldout"),
        default="heldout_except_input",
        help=(
            "How to label all CamXTime cameras when --all-camxtime-cameras is set. "
            "Use train_except_heldout to model one input camera with losses on every "
            "camera except the configured heldout split."
        ),
    )
    parser.add_argument("--frame-count", type=int, default=1, help="Pose sequence length to materialize.")
    parser.add_argument("--frame-index", type=int, default=0, help="Frame index to draw for moving-camera trajectories.")
    parser.add_argument("--target-size", type=int, default=None, help="Camera intrinsic viewport size override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_payload(
        args.config,
        input_camera=args.input_camera,
        train_cameras_override=args.train_cameras,
        heldout_cameras_override=args.heldout_cameras,
        all_camxtime_cameras=bool(args.all_camxtime_cameras),
        all_camxtime_role=str(args.all_camxtime_role),
        frame_count=int(args.frame_count),
        target_size=args.target_size,
        frame_index=int(args.frame_index),
    )
    html_path, json_path = write_outputs(payload, args.output, args.json_output)
    print(
        json.dumps(
            {
                "html": _repo_text(html_path),
                "json": _repo_text(json_path),
                "camera_count": len(payload.cameras),
                **payload.metadata,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
