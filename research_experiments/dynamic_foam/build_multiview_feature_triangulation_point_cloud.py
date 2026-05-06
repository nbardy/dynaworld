from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TRAIN_SRC = ROOT / "src" / "train"
if str(TRAIN_SRC) not in sys.path:
    sys.path.insert(0, str(TRAIN_SRC))

from config_utils import load_config_file
from multicam_video_data import load_multicam_video_bundle
from train_powerfoam_metal import resolve_config


def image_to_gray_u8(image: torch.Tensor) -> np.ndarray:
    rgb = (image.detach().cpu().permute(1, 2, 0).clamp(0.0, 1.0).numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def make_feature_detector(method: str, max_features: int) -> tuple[Any, int]:
    method = str(method).lower()
    if method == "sift" and hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=int(max_features)), cv2.NORM_L2
    if method in {"sift", "orb"}:
        return cv2.ORB_create(nfeatures=int(max_features), fastThreshold=7), cv2.NORM_HAMMING
    raise ValueError("feature method must be 'sift' or 'orb'.")


def detect_features(image: torch.Tensor, *, method: str, max_features: int) -> tuple[list[Any], np.ndarray | None, str]:
    detector, _ = make_feature_detector(method, max_features)
    actual_method = "sift" if str(method).lower() == "sift" and detector.__class__.__name__.lower().startswith("sift") else "orb"
    keypoints, descriptors = detector.detectAndCompute(image_to_gray_u8(image), None)
    return keypoints, descriptors, actual_method


def ratio_matches(
    descriptors_a: np.ndarray | None,
    descriptors_b: np.ndarray | None,
    *,
    norm_type: int,
    ratio: float,
) -> list[cv2.DMatch]:
    if descriptors_a is None or descriptors_b is None or len(descriptors_a) < 2 or len(descriptors_b) < 2:
        return []
    matcher = cv2.BFMatcher(norm_type)
    raw = matcher.knnMatch(descriptors_a, descriptors_b, k=2)
    out = []
    for pair in raw:
        if len(pair) != 2:
            continue
        first, second = pair
        if first.distance <= float(ratio) * second.distance:
            out.append(first)
    return out


def symmetric_ratio_matches(
    descriptors_a: np.ndarray | None,
    descriptors_b: np.ndarray | None,
    *,
    norm_type: int,
    ratio: float,
) -> list[cv2.DMatch]:
    forward = ratio_matches(descriptors_a, descriptors_b, norm_type=norm_type, ratio=ratio)
    reverse = ratio_matches(descriptors_b, descriptors_a, norm_type=norm_type, ratio=ratio)
    reverse_pairs = {(int(match.queryIdx), int(match.trainIdx)) for match in reverse}
    return [
        match
        for match in forward
        if (int(match.trainIdx), int(match.queryIdx)) in reverse_pairs
    ]


def projection_matrix(K: torch.Tensor, w2c: torch.Tensor) -> np.ndarray:
    return (K.detach().cpu().numpy().astype(np.float64) @ w2c[:3].detach().cpu().numpy().astype(np.float64)).astype(
        np.float64
    )


def project_points(points: torch.Tensor, K: torch.Tensor, w2c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    points_h = torch.cat([points, torch.ones((points.shape[0], 1), dtype=points.dtype)], dim=-1)
    camera = (points_h @ w2c.detach().cpu().to(dtype=points.dtype).T)[:, :3]
    z = camera[:, 2]
    u = K[0, 0].to(dtype=points.dtype).cpu() * camera[:, 0] / z.clamp_min(1.0e-6) + K[0, 2].to(dtype=points.dtype).cpu()
    v = K[1, 1].to(dtype=points.dtype).cpu() * camera[:, 1] / z.clamp_min(1.0e-6) + K[1, 2].to(dtype=points.dtype).cpu()
    return torch.stack([u, v], dim=-1), z


def sample_rgb_nearest(image: torch.Tensor, points_xy: torch.Tensor) -> torch.Tensor:
    _, height, width = image.shape
    xy = points_xy.round().to(dtype=torch.long)
    x = xy[:, 0].clamp(0, width - 1)
    y = xy[:, 1].clamp(0, height - 1)
    return image.detach().cpu()[:, y, x].T.contiguous()


def triangulation_angle_deg(points: torch.Tensor, w2c_a: torch.Tensor, w2c_b: torch.Tensor) -> torch.Tensor:
    c2w_a = torch.linalg.inv(w2c_a.detach().cpu().to(dtype=points.dtype))
    c2w_b = torch.linalg.inv(w2c_b.detach().cpu().to(dtype=points.dtype))
    center_a = c2w_a[:3, 3]
    center_b = c2w_b[:3, 3]
    ray_a = torch.nn.functional.normalize(points - center_a, dim=-1)
    ray_b = torch.nn.functional.normalize(points - center_b, dim=-1)
    cos = (ray_a * ray_b).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.acos(cos))


def triangulate_pair(
    *,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    K_a: torch.Tensor,
    K_b: torch.Tensor,
    w2c_a: torch.Tensor,
    w2c_b: torch.Tensor,
    view_a: str,
    view_b: str,
    frame_index: int,
    method: str,
    max_features: int,
    ratio: float,
    reproj_error_px: float,
    min_angle_deg: float,
    xy_extent: float,
    z_min: float,
    z_max: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    detector, norm_type = make_feature_detector(method, max_features)
    actual_method = "sift" if str(method).lower() == "sift" and detector.__class__.__name__.lower().startswith("sift") else "orb"
    gray_a = image_to_gray_u8(image_a)
    gray_b = image_to_gray_u8(image_b)
    keypoints_a, descriptors_a = detector.detectAndCompute(gray_a, None)
    keypoints_b, descriptors_b = detector.detectAndCompute(gray_b, None)
    matches = symmetric_ratio_matches(descriptors_a, descriptors_b, norm_type=norm_type, ratio=ratio)
    matches = sorted(matches, key=lambda match: (float(match.distance), int(match.queryIdx), int(match.trainIdx)))
    if len(matches) == 0:
        stats = {
            "frame_index": int(frame_index),
            "view_a": view_a,
            "view_b": view_b,
            "method": actual_method,
            "keypoints_a": len(keypoints_a),
            "keypoints_b": len(keypoints_b),
            "matches": 0,
            "kept": 0,
        }
        empty_points = torch.empty((0, 3), dtype=torch.float32)
        empty_colors = torch.empty((0, 3), dtype=torch.float32)
        empty_errors = torch.empty((0,), dtype=torch.float32)
        return empty_points, empty_colors, empty_errors, stats

    pts_a_np = np.array([keypoints_a[match.queryIdx].pt for match in matches], dtype=np.float64)
    pts_b_np = np.array([keypoints_b[match.trainIdx].pt for match in matches], dtype=np.float64)
    P_a = projection_matrix(K_a, w2c_a)
    P_b = projection_matrix(K_b, w2c_b)
    points_h = cv2.triangulatePoints(P_a, P_b, pts_a_np.T, pts_b_np.T).T
    denom = points_h[:, 3:4]
    valid_h = np.isfinite(points_h).all(axis=1) & (np.abs(denom[:, 0]) > 1.0e-8)
    points_np = points_h[:, :3] / denom
    points = torch.from_numpy(points_np).to(dtype=torch.float32)
    pts_a = torch.from_numpy(pts_a_np).to(dtype=torch.float32)
    pts_b = torch.from_numpy(pts_b_np).to(dtype=torch.float32)

    reproj_a, depth_a = project_points(points, K_a, w2c_a)
    reproj_b, depth_b = project_points(points, K_b, w2c_b)
    err_a = torch.linalg.vector_norm(reproj_a - pts_a, dim=-1)
    err_b = torch.linalg.vector_norm(reproj_b - pts_b, dim=-1)
    error = 0.5 * (err_a + err_b)
    angle = triangulation_angle_deg(points, w2c_a, w2c_b)
    valid = (
        torch.from_numpy(valid_h)
        & torch.isfinite(points).all(dim=-1)
        & (depth_a > 1.0e-5)
        & (depth_b > 1.0e-5)
        & (error <= float(reproj_error_px))
        & (angle >= float(min_angle_deg))
        & (points[:, 0].abs() <= float(xy_extent))
        & (points[:, 1].abs() <= float(xy_extent))
        & (points[:, 2] >= float(z_min))
        & (points[:, 2] <= float(z_max))
    )
    kept_points = points[valid].contiguous()
    kept_colors = sample_rgb_nearest(image_a, pts_a[valid])
    kept_errors = error[valid].contiguous()
    stats = {
        "frame_index": int(frame_index),
        "view_a": view_a,
        "view_b": view_b,
        "method": actual_method,
        "keypoints_a": len(keypoints_a),
        "keypoints_b": len(keypoints_b),
        "matches": len(matches),
        "finite_homogeneous": int(torch.from_numpy(valid_h).sum().item()),
        "positive_depth": int(((depth_a > 1.0e-5) & (depth_b > 1.0e-5)).sum().item()),
        "reprojection_kept": int((error <= float(reproj_error_px)).sum().item()),
        "angle_kept": int((angle >= float(min_angle_deg)).sum().item()),
        "box_kept": int(
            (
                (points[:, 0].abs() <= float(xy_extent))
                & (points[:, 1].abs() <= float(xy_extent))
                & (points[:, 2] >= float(z_min))
                & (points[:, 2] <= float(z_max))
            ).sum().item()
        ),
        "kept": int(kept_points.shape[0]),
        "median_reproj_error_px": None if kept_errors.numel() == 0 else float(kept_errors.median().item()),
    }
    return kept_points, kept_colors, kept_errors, stats


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


def parse_frame_indices(raw: str, frame_count: int) -> list[int]:
    if str(raw).lower() == "all":
        return list(range(frame_count))
    out = [int(item.strip()) for item in str(raw).split(",") if item.strip()]
    if not out:
        raise ValueError("At least one frame index is required.")
    for frame_index in out:
        if frame_index < 0 or frame_index >= frame_count:
            raise IndexError(f"frame index {frame_index} out of range for {frame_count} frames.")
    return out


def build_feature_triangulation_cloud(args: argparse.Namespace) -> dict[str, Any]:
    cfg = resolve_config(load_config_file(args.config))
    if str(cfg["data"]["frame_source"]) != "multicam_val":
        raise ValueError("Feature triangulation point cloud builder expects data.frame_source='multicam_val'.")
    render_size = int(args.target_size or cfg["render"]["render_size"])
    bundle = load_multicam_video_bundle(
        data_cfg=cfg["data"],
        camera_cfg=cfg["camera"],
        target_size=render_size,
        device=torch.device("cpu"),
    )
    if bundle.train_view_count < 2:
        raise ValueError("Feature triangulation requires at least two train cameras.")
    frame_indices = parse_frame_indices(str(args.frame_indices), bundle.frame_count)
    xy_extent = float(args.xy_extent if args.xy_extent is not None else cfg["model"]["xy_extent"])
    z_min = float(args.z_min if args.z_min is not None else cfg["model"]["z_min"])
    z_max = float(args.z_max if args.z_max is not None else cfg["model"]["z_max"])

    all_points = []
    all_colors = []
    all_errors = []
    pair_stats = []
    for frame_index in frame_indices:
        for view_a, view_b in itertools.combinations(range(bundle.train_view_count), 2):
            points, colors, errors, stats = triangulate_pair(
                image_a=bundle.train_frames[view_a, frame_index].to(dtype=torch.float32),
                image_b=bundle.train_frames[view_b, frame_index].to(dtype=torch.float32),
                K_a=bundle.train_K[view_a],
                K_b=bundle.train_K[view_b],
                w2c_a=bundle.train_w2c[view_a, frame_index],
                w2c_b=bundle.train_w2c[view_b, frame_index],
                view_a=str(bundle.train_camera_names[view_a]),
                view_b=str(bundle.train_camera_names[view_b]),
                frame_index=frame_index,
                method=str(args.method),
                max_features=int(args.max_features),
                ratio=float(args.ratio),
                reproj_error_px=float(args.reproj_error_px),
                min_angle_deg=float(args.min_angle_deg),
                xy_extent=xy_extent,
                z_min=z_min,
                z_max=z_max,
            )
            pair_stats.append(stats)
            if points.numel() > 0:
                all_points.append(points)
                all_colors.append(colors)
                all_errors.append(errors)
    if not all_points:
        summary = {
            "config": str(args.config),
            "output": str(args.output),
            "sample_id": str(bundle.metadata.get("sample_id")) if bundle.metadata else None,
            "train_cameras": list(bundle.train_camera_names),
            "heldout_cameras": list(bundle.heldout_camera_names or []),
            "pose_source": bundle.pose_source,
            "coordinate_frame": "model",
            "frame_indices": frame_indices,
            "target_size": render_size,
            "method_requested": str(args.method),
            "ratio": float(args.ratio),
            "reproj_error_px": float(args.reproj_error_px),
            "min_angle_deg": float(args.min_angle_deg),
            "xy_extent": xy_extent,
            "z_min": z_min,
            "z_max": z_max,
            "raw_pair_count": len(pair_stats),
            "raw_valid_points": 0,
            "point_count": 0,
            "pairs": pair_stats,
        }
        Path(args.output).with_suffix(".json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError("Feature triangulation produced no valid points; wrote diagnostic JSON next to output path.")

    points = torch.cat(all_points, dim=0)
    colors = torch.cat(all_colors, dim=0)
    errors = torch.cat(all_errors, dim=0)
    order = torch.argsort(errors, stable=True)
    if int(args.max_points) > 0:
        order = order[: int(args.max_points)]
    points = points.index_select(0, order).contiguous()
    colors = colors.index_select(0, order).contiguous()
    errors = errors.index_select(0, order).contiguous()
    write_ascii_ply(Path(args.output), points, colors)

    summary = {
        "config": str(args.config),
        "output": str(args.output),
        "sample_id": str(bundle.metadata.get("sample_id")) if bundle.metadata else None,
        "train_cameras": list(bundle.train_camera_names),
        "heldout_cameras": list(bundle.heldout_camera_names or []),
        "anchor_camera": str(bundle.metadata.get("anchor_camera")) if bundle.metadata else None,
        "pose_source": bundle.pose_source,
        "coordinate_frame": "model",
        "frame_indices": frame_indices,
        "target_size": render_size,
        "method_requested": str(args.method),
        "ratio": float(args.ratio),
        "reproj_error_px": float(args.reproj_error_px),
        "min_angle_deg": float(args.min_angle_deg),
        "xy_extent": xy_extent,
        "z_min": z_min,
        "z_max": z_max,
        "raw_pair_count": len(pair_stats),
        "raw_valid_points": int(sum(int(item.get("kept", 0)) for item in pair_stats)),
        "point_count": int(points.shape[0]),
        "mean_reproj_error_px": float(errors.mean().item()) if errors.numel() else None,
        "median_reproj_error_px": float(errors.median().item()) if errors.numel() else None,
        "p90_reproj_error_px": float(torch.quantile(errors, 0.9).item()) if errors.numel() else None,
        "pairs": pair_stats,
    }
    summary_path = Path(args.output).with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a train-camera-only feature-triangulated PLY init.")
    parser.add_argument("config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-size", type=int, default=None)
    parser.add_argument("--frame-indices", type=str, default="0")
    parser.add_argument("--method", choices=["sift", "orb"], default="sift")
    parser.add_argument("--max-features", type=int, default=4096)
    parser.add_argument("--ratio", type=float, default=0.82)
    parser.add_argument("--reproj-error-px", type=float, default=3.0)
    parser.add_argument("--min-angle-deg", type=float, default=0.25)
    parser.add_argument("--max-points", type=int, default=8192)
    parser.add_argument("--xy-extent", type=float, default=None)
    parser.add_argument("--z-min", type=float, default=None)
    parser.add_argument("--z-max", type=float, default=None)
    summary = build_feature_triangulation_cloud(parser.parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
