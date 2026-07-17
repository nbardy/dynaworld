from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

try:
    from .report_artifacts import ensure_train_path, write_report_json
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import ensure_train_path, write_report_json

ensure_train_path()
from multicam_video_data import neural_3d_camera_from_poses_bounds
from powerfoam_point_cloud import load_point_cloud_xyz_rgb


def load_manifest_record(path: Path, sample_id: str) -> dict[str, Any]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if str(record.get("sample_id")) == str(sample_id):
            return record
    raise ValueError(f"No manifest record with sample_id={sample_id!r} in {path}.")


def camera_matrices(
    record: dict[str, Any],
    camera_name: str,
    *,
    render_size: int,
    translation_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return neural_3d_camera_from_poses_bounds(
        record,
        camera_name,
        H=int(render_size),
        W=int(render_size),
        device=torch.device("cpu"),
        translation_scale=float(translation_scale),
    )


def project_points(
    points_anchor: torch.Tensor,
    *,
    K: torch.Tensor,
    camera_c2w: torch.Tensor,
    anchor_c2w: torch.Tensor,
    render_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rel_w2c = torch.linalg.inv(camera_c2w) @ anchor_c2w
    points_h = torch.cat([points_anchor, torch.ones((points_anchor.shape[0], 1), dtype=points_anchor.dtype)], dim=-1)
    points_camera = (rel_w2c @ points_h.T).T[:, :3]
    z = points_camera[:, 2]
    u = K[0, 0] * points_camera[:, 0] / z.clamp_min(1.0e-6) + K[0, 2]
    v = K[1, 1] * points_camera[:, 1] / z.clamp_min(1.0e-6) + K[1, 2]
    inside = (z > 1.0e-5) & (u >= 0.0) & (u < float(render_size)) & (v >= 0.0) & (v < float(render_size))
    return u, v, z, inside


def write_ascii_ply(path: Path, points: torch.Tensor, colors: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    colors_u8 = (colors.clamp(0.0, 1.0) * 255.0).round().to(torch.int64)
    with path.open("w", encoding="ascii") as fh:
        fh.write("ply\n")
        fh.write("format ascii 1.0\n")
        fh.write(f"element vertex {int(points.shape[0])}\n")
        fh.write("property float x\n")
        fh.write("property float y\n")
        fh.write("property float z\n")
        fh.write("property uchar red\n")
        fh.write("property uchar green\n")
        fh.write("property uchar blue\n")
        fh.write("end_header\n")
        for point, color in zip(points.tolist(), colors_u8.tolist()):
            fh.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def tensor_summary(values: torch.Tensor) -> dict[str, list[float]]:
    if values.numel() == 0:
        return {"min": [], "max": [], "median": []}
    return {
        "min": [float(v) for v in values.min(dim=0).values],
        "max": [float(v) for v in values.max(dim=0).values],
        "median": [float(v) for v in values.median(dim=0).values],
    }


def build_anchor_point_cloud(args: argparse.Namespace) -> dict[str, Any]:
    record = load_manifest_record(Path(args.manifest), str(args.sample_id))
    if record.get("dataset") != "neural_3d_video":
        raise ValueError(f"Expected a Neural 3D Video record, got dataset={record.get('dataset')!r}.")
    train_cameras = list(args.train_camera or record.get("train_cameras") or [])
    if not train_cameras:
        raise ValueError("No train cameras provided and record has no train_cameras.")
    heldout_cameras = list(args.heldout_camera or record.get("heldout_cameras") or [])
    anchor_camera = str(args.anchor_camera or record.get("anchor_camera") or train_cameras[0])

    points_world, colors = load_point_cloud_xyz_rgb(Path(args.input_ply))
    _anchor_K, anchor_c2w = camera_matrices(
        record,
        anchor_camera,
        render_size=int(args.render_size),
        translation_scale=float(args.translation_scale),
    )
    points_h = torch.cat([points_world, torch.ones((points_world.shape[0], 1), dtype=points_world.dtype)], dim=-1)
    points_anchor = (torch.linalg.inv(anchor_c2w) @ points_h.T).T[:, :3].contiguous()

    finite = torch.isfinite(points_anchor).all(dim=-1) & torch.isfinite(colors).all(dim=-1)
    box = (
        (points_anchor[:, 0].abs() <= float(args.xy_extent))
        & (points_anchor[:, 1].abs() <= float(args.xy_extent))
        & (points_anchor[:, 2] >= float(args.z_min))
        & (points_anchor[:, 2] <= float(args.z_max))
    )
    base_mask = finite & box

    camera_info = {}
    visible_votes = torch.zeros(points_anchor.shape[0], dtype=torch.int64)
    for camera_name in train_cameras + heldout_cameras:
        K, c2w = camera_matrices(
            record,
            camera_name,
            render_size=int(args.render_size),
            translation_scale=float(args.translation_scale),
        )
        _u, _v, z, inside = project_points(
            points_anchor,
            K=K,
            camera_c2w=c2w,
            anchor_c2w=anchor_c2w,
            render_size=int(args.render_size),
        )
        if camera_name in train_cameras:
            visible_votes += inside.to(torch.int64)
        camera_info[camera_name] = {
            "inside_count_before_box": int(inside.sum().item()),
            "inside_fraction_before_box": float(inside.float().mean().item()),
            "inside_count_after_box": int((inside & base_mask).sum().item()),
            "inside_fraction_after_box": float((inside & base_mask).float().mean().item()),
            "positive_depth_fraction": float((z > 1.0e-5).float().mean().item()),
        }

    visible = visible_votes >= int(args.min_visible_train_views)
    keep = base_mask & visible
    if int(keep.sum().item()) == 0:
        raise RuntimeError("Filtering removed every point; relax z/xy/visibility thresholds.")
    out_points = points_anchor[keep].contiguous()
    out_colors = colors[keep].contiguous()
    write_ascii_ply(Path(args.output), out_points, out_colors)

    summary = {
        "sample_id": str(args.sample_id),
        "source_ply": str(args.input_ply),
        "output_ply": str(args.output),
        "anchor_camera": anchor_camera,
        "train_cameras": train_cameras,
        "heldout_cameras": heldout_cameras,
        "render_size": int(args.render_size),
        "translation_scale": float(args.translation_scale),
        "xy_extent": float(args.xy_extent),
        "z_min": float(args.z_min),
        "z_max": float(args.z_max),
        "min_visible_train_views": int(args.min_visible_train_views),
        "source_count": int(points_world.shape[0]),
        "box_count": int(base_mask.sum().item()),
        "kept_count": int(out_points.shape[0]),
        "source_anchor_bounds": tensor_summary(points_anchor),
        "kept_anchor_bounds": tensor_summary(out_points),
        "camera_projection": camera_info,
    }
    summary_path = Path(args.output).with_suffix(".json")
    write_report_json(summary_path, summary, sort_keys=False)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_ply", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-camera", action="append")
    parser.add_argument("--heldout-camera", action="append")
    parser.add_argument("--anchor-camera")
    parser.add_argument("--render-size", type=int, default=128)
    parser.add_argument("--translation-scale", type=float, default=1.0)
    parser.add_argument("--xy-extent", type=float, default=24.0)
    parser.add_argument("--z-min", type=float, default=4.0)
    parser.add_argument("--z-max", type=float, default=120.0)
    parser.add_argument("--min-visible-train-views", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    summary = build_anchor_point_cloud(parse_args())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
