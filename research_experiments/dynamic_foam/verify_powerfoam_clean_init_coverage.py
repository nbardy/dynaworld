from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

try:
    from .report_artifacts import DYNAMIC_FOAM_ROOT, PROJECT_ROOT, ensure_sys_path, ensure_train_path, load_report_json
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import DYNAMIC_FOAM_ROOT, PROJECT_ROOT, ensure_sys_path, ensure_train_path, load_report_json

ROOT = PROJECT_ROOT
ensure_train_path()
ensure_sys_path(DYNAMIC_FOAM_ROOT)

from multicam_video_data import (  # noqa: E402
    camera_from_K_w2c,
    deepview_lens_metadata,
    make_aist_multiview_cameras,
    make_deepview_multiview_cameras,
    make_neural_3d_multiview_cameras,
    make_orthogonal_origin_multiview_cameras,
    make_vivo_multiview_cameras,
    requested_camera_frame_count,
    select_multicam_record,
    validate_multicam_camera_split,
)
from renderers.projection import project_points_camera  # noqa: E402
from verify_powerfoam_paper_acceptance import (  # noqa: E402
    clean_candidate_metrics,
    existing_clean_candidates,
    missing_optional_clean_candidates,
)


def load_ascii_ply_points(path: Path) -> torch.Tensor:
    with path.open("r", encoding="ascii") as fh:
        first = fh.readline().strip()
        if first != "ply":
            raise ValueError(f"{path} is not a PLY file.")
        fmt = None
        vertex_count = None
        vertex_properties: list[str] = []
        in_vertex = False
        for line in fh:
            line = line.strip()
            if line == "end_header":
                break
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    vertex_count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    raise ValueError(f"{path} has list vertex properties; unsupported for coverage audit.")
                vertex_properties.append(parts[2])
        if fmt != "ascii":
            raise ValueError(f"{path} must be ASCII PLY for this lightweight verifier; got {fmt!r}.")
        if vertex_count is None:
            raise ValueError(f"{path} does not declare a vertex count.")
        for required in ("x", "y", "z"):
            if required not in vertex_properties:
                raise ValueError(f"{path} vertex properties must include x/y/z.")
        xyz_indices = [vertex_properties.index(axis) for axis in ("x", "y", "z")]
        rows = []
        for _ in range(vertex_count):
            parts = fh.readline().split()
            if len(parts) < len(vertex_properties):
                raise ValueError(f"{path} ended inside a vertex row.")
            rows.append([float(parts[index]) for index in xyz_indices])
    return torch.tensor(rows, dtype=torch.float32)


def camera_lists(data_cfg: dict[str, Any], record: dict[str, Any]) -> tuple[list[str], list[str], str, str]:
    train_raw = data_cfg.get("multicam_train_cameras")
    train_cameras = [str(camera) for camera in train_raw] if train_raw else [str(record["source_camera"])]
    heldout_raw = data_cfg.get("multicam_heldout_cameras")
    heldout_cameras = (
        [str(camera) for camera in heldout_raw]
        if heldout_raw
        else [str(data_cfg.get("multicam_heldout_camera") or record["target_camera"])]
    )
    anchor_camera = str(data_cfg.get("multicam_anchor_camera") or train_cameras[0])
    condition_camera = str(data_cfg.get("multicam_condition_camera") or anchor_camera)
    validate_multicam_camera_split(
        train_cameras=train_cameras,
        heldout_cameras=heldout_cameras,
        anchor_camera=anchor_camera,
        condition_camera=condition_camera,
    )
    return train_cameras, heldout_cameras, anchor_camera, condition_camera


def selected_frame_count(data_cfg: dict[str, Any], record: dict[str, Any]) -> int:
    if data_cfg.get("frame_indices") is not None:
        return len(data_cfg["frame_indices"])
    return requested_camera_frame_count(data_cfg, record)


def multicam_matrices(cfg: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    data_cfg = cfg["data"]
    camera_cfg = cfg.get("camera", {})
    render_size = int(cfg["render"]["render_size"])
    record = select_multicam_record(data_cfg)
    train_cameras, heldout_cameras, anchor_camera, _condition_camera = camera_lists(data_cfg, record)
    common = {
        "record": record,
        "train_cameras": train_cameras,
        "heldout_cameras": heldout_cameras,
        "anchor_camera": anchor_camera,
        "T": selected_frame_count(data_cfg, record),
        "H": render_size,
        "W": render_size,
        "device": torch.device("cpu"),
    }
    rig_init = str(camera_cfg.get("rig_init", "deepview")).lower()
    train_lens_models = None
    train_distortions = None
    heldout_lens_models = None
    heldout_distortions = None
    if rig_init == "deepview":
        train_K, train_w2c, heldout_K, heldout_w2c, pose_source = make_deepview_multiview_cameras(**common)
        train_lens_models, train_distortions = deepview_lens_metadata(record, train_cameras, device=torch.device("cpu"))
        heldout_lens_models, heldout_distortions = deepview_lens_metadata(
            record,
            heldout_cameras,
            device=torch.device("cpu"),
        )
    elif rig_init == "aist":
        train_K, train_w2c, heldout_K, heldout_w2c, pose_source = make_aist_multiview_cameras(
            **common,
            translation_scale=float(camera_cfg.get("aist_translation_scale", 1.0)),
        )
    elif rig_init == "neural_3d_video":
        train_K, train_w2c, heldout_K, heldout_w2c, pose_source = make_neural_3d_multiview_cameras(
            **common,
            translation_scale=float(camera_cfg.get("n3d_translation_scale", 1.0)),
        )
    elif rig_init == "vivo":
        train_K, train_w2c, heldout_K, heldout_w2c, pose_source = make_vivo_multiview_cameras(
            **common,
            translation_scale=float(camera_cfg.get("vivo_translation_scale", 1.0)),
        )
    elif rig_init == "orthogonal_origin":
        train_K, train_w2c, heldout_K, heldout_w2c, pose_source = make_orthogonal_origin_multiview_cameras(
            view_count=len(train_cameras),
            heldout_count=len(heldout_cameras),
            T=common["T"],
            H=render_size,
            W=render_size,
            radius=float(camera_cfg.get("rig_radius", camera_cfg.get("base_radius", 3.0))),
            fov_degrees=float(camera_cfg.get("base_fov_degrees", 60.0)),
            device=torch.device("cpu"),
        )
    else:
        raise ValueError(
            "camera.rig_init must be one of: deepview, aist, neural_3d_video, vivo, orthogonal_origin"
        )
    meta = {
        "sample_id": record.get("sample_id"),
        "dataset": record.get("dataset"),
        "pose_source": pose_source,
        "train_cameras": train_cameras,
        "heldout_cameras": heldout_cameras,
        "anchor_camera": anchor_camera,
        "frame_count": common["T"],
        "train_lens_models": train_lens_models,
        "heldout_lens_models": heldout_lens_models,
        "train_distortions": None
        if train_distortions is None
        else [[float(value) for value in row] for row in train_distortions.tolist()],
        "heldout_distortions": None
        if heldout_distortions is None
        else [[float(value) for value in row] for row in heldout_distortions.tolist()],
    }
    return train_K, train_w2c, heldout_K, heldout_w2c, meta


def projection_stats(
    points: torch.Tensor,
    *,
    K: torch.Tensor,
    w2c: torch.Tensor,
    render_size: int,
    lens_models: list[str] | None = None,
    distortions: torch.Tensor | None = None,
) -> dict[str, Any]:
    if w2c.ndim == 4:
        w2c = w2c[:, 0]
    if K.ndim != 3 or w2c.ndim != 3:
        raise ValueError(f"Expected K [V,3,3] and w2c [V,4,4], got {tuple(K.shape)} and {tuple(w2c.shape)}.")
    if lens_models is not None and len(lens_models) != int(K.shape[0]):
        raise ValueError(f"Expected {int(K.shape[0])} lens models, got {len(lens_models)}.")
    if distortions is not None and int(distortions.shape[0]) != int(K.shape[0]):
        raise ValueError(f"Expected distortions first dim {int(K.shape[0])}, got {tuple(distortions.shape)}.")
    points_h = torch.cat([points, torch.ones((points.shape[0], 1), dtype=points.dtype)], dim=-1)
    votes = []
    pixel_counts = []
    identity_w2c = torch.eye(4, dtype=points.dtype, device=points.device)
    for view in range(int(K.shape[0])):
        points_camera = (points_h @ w2c[view].T)[:, :3]
        camera = camera_from_K_w2c(
            K[view],
            identity_w2c,
            lens_model="pinhole" if lens_models is None else lens_models[view],
            distortion=None if distortions is None else distortions[view],
        )
        pixels, _depths, _pixel_jacobian, front_mask = project_points_camera(points_camera, camera)
        inside = (
            front_mask
            & (pixels[:, 0] >= 0.0)
            & (pixels[:, 0] < float(render_size))
            & (pixels[:, 1] >= 0.0)
            & (pixels[:, 1] < float(render_size))
        )
        votes.append(inside)
        pixels = torch.stack(
            [
                pixels[inside, 1].floor().long().clamp(0, render_size - 1),
                pixels[inside, 0].floor().long().clamp(0, render_size - 1),
            ],
            dim=-1,
        )
        pixel_counts.append(int(torch.unique(pixels, dim=0).shape[0]) if pixels.numel() > 0 else 0)
    vote_tensor = torch.stack(votes, dim=1)
    views_per_point = vote_tensor.sum(dim=1).to(dtype=torch.float32)
    visible_any = vote_tensor.any(dim=1)
    pixel_count_tensor = torch.tensor(pixel_counts, dtype=torch.float32)
    pixel_area = float(render_size * render_size)
    return {
        "visible_point_count": int(visible_any.sum().item()),
        "visible_point_ratio": float(visible_any.to(dtype=torch.float32).mean().item()),
        "views_per_point_mean": float(views_per_point.mean().item()),
        "views_per_point_p50": float(torch.quantile(views_per_point, 0.5).item()),
        "views_per_point_p90": float(torch.quantile(views_per_point, 0.9).item()),
        "views_per_point_max": int(views_per_point.max().item()),
        "center_pixel_count_per_view": pixel_counts,
        "center_pixel_coverage_mean": float((pixel_count_tensor / pixel_area).mean().item()),
        "center_pixel_coverage_max": float((pixel_count_tensor / pixel_area).max().item()),
        "visible_mask": visible_any,
        "view_vote_counts": vote_tensor.sum(dim=1),
    }


def model_box_mask(points: torch.Tensor, model_cfg: dict[str, Any]) -> torch.Tensor:
    xy_extent = float(model_cfg["xy_extent"])
    z_min = float(model_cfg["z_min"])
    z_max = float(model_cfg["z_max"])
    return (
        torch.isfinite(points).all(dim=-1)
        & (points[:, 0].abs() <= xy_extent)
        & (points[:, 1].abs() <= xy_extent)
        & (points[:, 2] >= z_min)
        & (points[:, 2] <= z_max)
    )


def box_stats(points: torch.Tensor, model_cfg: dict[str, Any]) -> dict[str, Any]:
    in_box = model_box_mask(points, model_cfg)
    return {
        "in_model_box_count": int(in_box.sum().item()),
        "in_model_box_ratio": float(in_box.to(dtype=torch.float32).mean().item()),
        "xyz_min": [float(value) for value in points.min(dim=0).values.tolist()],
        "xyz_max": [float(value) for value in points.max(dim=0).values.tolist()],
    }


def preview_alpha_stats(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        from PIL import Image
    except Exception:
        return {"path": str(path.relative_to(ROOT)), "error": "PIL unavailable"}
    image = Image.open(path).convert("RGB")
    tensor = torch.tensor(list(image.getdata()), dtype=torch.float32).view(image.height, image.width, 3) / 255.0
    alpha = tensor[:, (2 * image.width) // 3 :, 0]
    return {
        "path": str(path.relative_to(ROOT)),
        "alpha_mean": float(alpha.mean().item()),
        "alpha_fraction_gt_0_01": float((alpha > 0.01).to(dtype=torch.float32).mean().item()),
        "alpha_fraction_gt_0_5": float((alpha > 0.5).to(dtype=torch.float32).mean().item()),
        "alpha_max": float(alpha.max().item()),
    }


def coverage_for_candidate(output_dir: Path, artifact_meta: Path) -> dict[str, Any]:
    clean_metrics, artifact_metrics = clean_candidate_metrics(output_dir, artifact_meta)
    cfg = load_report_json(output_dir / "resolved_config.json")
    point_cloud_path = ROOT / artifact_metrics["output"]
    points = load_ascii_ply_points(point_cloud_path)
    train_K, train_w2c, heldout_K, heldout_w2c, camera_meta = multicam_matrices(cfg)
    render_size = int(cfg["render"]["render_size"])
    train_distortions = (
        None if camera_meta["train_distortions"] is None else torch.tensor(camera_meta["train_distortions"])
    )
    heldout_distortions = (
        None if camera_meta["heldout_distortions"] is None else torch.tensor(camera_meta["heldout_distortions"])
    )
    train_projection = projection_stats(
        points,
        K=train_K,
        w2c=train_w2c,
        render_size=render_size,
        lens_models=camera_meta["train_lens_models"],
        distortions=train_distortions,
    )
    heldout_projection = projection_stats(
        points,
        K=heldout_K,
        w2c=heldout_w2c,
        render_size=render_size,
        lens_models=camera_meta["heldout_lens_models"],
        distortions=heldout_distortions,
    )
    train_visible = train_projection.pop("visible_mask")
    heldout_visible = heldout_projection.pop("visible_mask")
    heldout_train_votes = train_projection["view_vote_counts"][heldout_visible].to(dtype=torch.float32)
    train_projection.pop("view_vote_counts")
    heldout_projection.pop("view_vote_counts")
    in_box = model_box_mask(points, cfg["model"])
    filtered_count = int((train_visible & in_box).sum().item())
    cell_count = int(clean_metrics["cells"])
    duplicate_backfill_count = max(0, cell_count - int(filtered_count))
    downsample_drop_count = max(0, int(filtered_count) - cell_count)
    sampled_count = min(cell_count, int(filtered_count))
    preview_stats = [
        stats
        for stats in (
            preview_alpha_stats(path)
            for path in sorted(output_dir.glob("heldout_preview_step_*.png"))
        )
        if stats is not None
    ]
    return {
        "run": clean_metrics,
        "point_cloud": artifact_metrics,
        "camera": camera_meta,
        "source_point_count": int(points.shape[0]),
        "box": box_stats(points, cfg["model"]),
        "train_projection": train_projection,
        "heldout_projection": heldout_projection,
        "heldout_visible_point_train_view_votes_mean": None
        if heldout_train_votes.numel() == 0
        else float(heldout_train_votes.mean().item()),
        "init_sampling": {
            "cell_count": cell_count,
            "train_visible_box_filtered_count": int(filtered_count),
            "sampled_count": int(sampled_count),
            "sampled_fraction_of_filtered": 0.0 if filtered_count == 0 else float(sampled_count) / float(filtered_count),
            "downsample_drop_count": int(downsample_drop_count),
            "downsample_drop_ratio_of_filtered": 0.0
            if filtered_count == 0
            else float(downsample_drop_count) / float(filtered_count),
            "duplicate_backfill_count": int(duplicate_backfill_count),
            "duplicate_backfill_ratio": float(duplicate_backfill_count) / float(cell_count),
        },
        "checkpoint_selection": {
            "best_step": int(clean_metrics["step"]),
            "best_is_initial_step": int(clean_metrics["step"]) == 0,
            "wandb_enabled": bool(clean_metrics["wandb_enabled"]),
        },
        "heldout_preview_alpha": preview_stats,
    }


def audit() -> dict[str, Any]:
    candidates = [
        coverage_for_candidate(output_dir, artifact_meta)
        for output_dir, artifact_meta in existing_clean_candidates(require_point_cloud=True)
    ]
    selected = max(candidates, key=lambda item: item["run"]["heldout_eval_psnr"])
    checks = [
        {
            "name": "selected_train_visible_fraction",
            "passed": selected["train_projection"]["visible_point_ratio"] >= 0.9,
            "evidence": {
                "actual": selected["train_projection"]["visible_point_ratio"],
                "required": 0.9,
            },
        },
        {
            "name": "selected_heldout_center_coverage",
            "passed": selected["heldout_projection"]["center_pixel_coverage_mean"] >= 0.05,
            "evidence": {
                "actual": selected["heldout_projection"]["center_pixel_coverage_mean"],
                "required": 0.05,
            },
        },
        {
            "name": "selected_no_duplicate_backfill",
            "passed": selected["init_sampling"]["duplicate_backfill_count"] == 0,
            "evidence": selected["init_sampling"],
        },
        {
            "name": "selected_keeps_most_filtered_points",
            "passed": selected["init_sampling"]["sampled_fraction_of_filtered"] >= 0.8,
            "evidence": {
                "actual": selected["init_sampling"]["sampled_fraction_of_filtered"],
                "required": 0.8,
                "downsample_drop_count": selected["init_sampling"]["downsample_drop_count"],
            },
        },
        {
            "name": "selected_best_checkpoint_not_initial",
            "passed": not selected["checkpoint_selection"]["best_is_initial_step"],
            "evidence": selected["checkpoint_selection"],
        },
    ]
    return {
        "ok": all(check["passed"] for check in checks),
        "checks": checks,
        "selected_candidate": selected,
        "candidates": candidates,
        "missing_optional_clean_deepview_candidates": missing_optional_clean_candidates(require_point_cloud=True),
        "interpretation": (
            "This verifier explains clean-init failure modes. It is not a paper acceptance gate by itself; "
            "the paper verifier remains authoritative for official CUDA fixture, W&B, PSNR, and SSIM gates."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit clean PowerFoam init coverage for saved DeepView candidates.")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    report = audit()
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"] and not args.allow_incomplete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
