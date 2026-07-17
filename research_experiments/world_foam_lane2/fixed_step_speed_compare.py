#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any


WORLD_FOAM_DIR = Path(__file__).resolve().parent
DYNAWORLD = WORLD_FOAM_DIR.parents[1]
RESULTS_DIR = WORLD_FOAM_DIR / "results"
STAR_VARIANT = DYNAWORLD / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
STAR_BENCH = STAR_VARIANT / "research_project" / "benchmarks"
WORLD_FOAM_VARIANT = DYNAWORLD / "third_party" / "fast-mac-gsplat" / "variants" / "world_foam_lane2_v0"
WORLD_FOAM_TOOLS = WORLD_FOAM_VARIANT / "tools"
DEFAULT_BASELINE_CONFIG = (
    DYNAWORLD
    / "src"
    / "train_configs"
    / "local_mac_multicam_deepview_3cam_train2_test1_vjepa_full_relpose_features_F32_256_16f_8192splats_goodset_train0006_0014_holdout0005_alphaab_alpha1_128.jsonc"
)


def _prepend_sys_path(path: Path) -> None:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)


for _path in (STAR_BENCH, STAR_VARIANT):
    _prepend_sys_path(_path)

import multicam_train_step_timing_probe as star_probe  # noqa: E402

for _path in (WORLD_FOAM_TOOLS, WORLD_FOAM_VARIANT, WORLD_FOAM_DIR, DYNAWORLD / "src" / "train"):
    _prepend_sys_path(_path)

from train_eval_shared_realray_csr_mps import run_train_eval as run_world_foam_train_eval  # noqa: E402


def parse_cases(raw: str) -> list[dict[str, int]]:
    cases: list[dict[str, int]] = []
    for item in raw.split(","):
        text = item.strip().lower().replace("px", "").replace("f", "")
        if not text:
            continue
        if "x" in text:
            size_text, frame_text = text.split("x", 1)
        elif ":" in text:
            size_text, frame_text = text.split(":", 1)
        else:
            raise ValueError(f"Case {item!r} must look like 128x16 or 128:16.")
        size = int(size_text)
        frames = int(frame_text)
        if size <= 0 or frames <= 0:
            raise ValueError(f"Case {item!r} must use positive integers.")
        cases.append({"target_size": size, "frames": frames})
    if not cases:
        raise ValueError("At least one case is required.")
    return cases


def read_manifest_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if text:
                record = json.loads(text)
                if not isinstance(record, dict):
                    raise ValueError(f"Expected object on {path}:{line_number}.")
                records.append(record)
    if not records:
        raise ValueError(f"No records found in {path}.")
    return records


def select_base_record(config: dict[str, Any]) -> tuple[dict[str, Any], Path]:
    data_cfg = config["data"]
    raw_manifest = Path(data_cfg["multicam_manifest"])
    manifest = raw_manifest if raw_manifest.is_absolute() else DYNAWORLD / raw_manifest
    records = [
        record
        for record in read_manifest_records(manifest)
        if str(record.get("split", "val")) == str(data_cfg.get("multicam_split", "val"))
    ]
    if not records:
        raise ValueError(f"No records match split={data_cfg.get('multicam_split', 'val')!r} in {manifest}.")
    sample_id = data_cfg.get("multicam_sample_id")
    if sample_id:
        for record in records:
            if str(record.get("sample_id")) == str(sample_id):
                return record, manifest
        raise ValueError(f"No manifest record with sample_id={sample_id!r} in {manifest}.")
    index = int(data_cfg.get("multicam_sample_index", 0))
    if index < 0 or index >= len(records):
        raise IndexError(f"multicam_sample_index={index} out of range for {len(records)} split records.")
    return records[index], manifest


def _video_duration(record: dict[str, Any], prefix: str) -> float | None:
    payload = record.get(f"{prefix}_video")
    if isinstance(payload, dict) and payload.get("duration_seconds") is not None:
        return float(payload["duration_seconds"])
    return None


def validate_case_timing(record: dict[str, Any], *, frame_count: int, fps: float) -> None:
    last_offset = float(frame_count - 1) / float(fps)
    for prefix in ("source", "target"):
        duration = _video_duration(record, prefix)
        if duration is None:
            continue
        start = float(record.get(f"{prefix}_start_seconds", 0.0))
        if start + last_offset > duration:
            raise ValueError(
                f"{prefix} video is too short for {frame_count} frames at {fps:.6f} fps: "
                f"last timestamp {start + last_offset:.3f}s exceeds duration {duration:.3f}s."
            )


def write_case_inputs(
    *,
    baseline_config: dict[str, Any],
    base_record: dict[str, Any],
    input_dir: Path,
    target_size: int,
    frames: int,
    duration_seconds: float,
) -> dict[str, Any]:
    case_id = f"{target_size}px_{frames}f"
    input_dir.mkdir(parents=True, exist_ok=True)
    fps = float(frames) / float(duration_seconds)
    validate_case_timing(base_record, frame_count=frames, fps=fps)

    record = copy.deepcopy(base_record)
    record["frame_count"] = frames
    record["fps"] = fps
    record["duration_seconds"] = float(duration_seconds)
    record["target_size"] = target_size
    record["dataset_name"] = f"{record.get('dataset_name', 'multicam')}_fixed_step_{case_id}"

    manifest_path = input_dir / f"{case_id}_manifest.jsonl"
    manifest_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")

    config = copy.deepcopy(baseline_config)
    config["data"]["multicam_manifest"] = str(manifest_path)
    config["data"]["max_frames"] = frames
    config["data"]["frame_indices"] = None
    if "model" in config:
        config["model"]["train_frame_count"] = frames
        config["model"]["size"] = target_size
    if "render" in config:
        config["render"]["render_size"] = target_size

    config_path = input_dir / f"{case_id}_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "case_id": case_id,
        "config_path": config_path,
        "manifest_path": manifest_path,
        "fps": fps,
        "duration_seconds": float(duration_seconds),
    }


def star_dynamic_args(args: argparse.Namespace, *, config_path: Path, target_size: int, frames: int) -> argparse.Namespace:
    return argparse.Namespace(
        baseline_config=config_path,
        target_size=target_size,
        max_frames=frames,
        device=args.device,
        seed=args.seed,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        frame_stride_for_probe=args.frame_stride_for_probe,
        uvt_tubes=args.uvt_tubes,
        uvt_lr=args.uvt_lr,
        uvt_render_backend=args.uvt_render_backend,
        uvt_reduction_mode=args.uvt_reduction_mode,
        uvt_sample_emission_mode=args.uvt_sample_emission_mode,
        uvt_camera_sequence_mode=args.uvt_camera_sequence_mode,
        uvt_segment_frames=args.uvt_segment_frames,
        uvt_synthetic_pan_x=args.uvt_synthetic_pan_x,
        uvt_synthetic_pan_y=args.uvt_synthetic_pan_y,
        uvt_synthetic_dolly_z=args.uvt_synthetic_dolly_z,
        uvt_synthetic_zoom=args.uvt_synthetic_zoom,
        uvt_synthetic_principal_x=args.uvt_synthetic_principal_x,
        uvt_synthetic_principal_y=args.uvt_synthetic_principal_y,
        uvt_loss_scope=args.uvt_loss_scope,
        uvt_window_frames=args.uvt_window_frames,
        uvt_init_precision_xy=args.uvt_init_precision_xy,
        uvt_init_lambda_t=args.uvt_init_lambda_t,
        uvt_init_opacity=args.uvt_init_opacity,
        uvt_min_precision_xy=args.uvt_min_precision_xy,
        uvt_min_lambda_t=args.uvt_min_lambda_t,
        uvt_velocity_reg=args.uvt_velocity_reg,
        uvt_depth_velocity_reg=args.uvt_depth_velocity_reg,
        uvt_position_reg=args.uvt_position_reg,
        uvt_tile_load_reg=args.uvt_tile_load_reg,
        uvt_tile_load_target=args.uvt_tile_load_target,
        uvt_depth_slope_reg=args.uvt_depth_slope_reg,
        uvt_depth_margin_reg=args.uvt_depth_margin_reg,
        uvt_depth_margin=args.uvt_depth_margin,
        uvt_tile_x=args.uvt_tile_x,
        uvt_tile_y=args.uvt_tile_y,
        uvt_tile_t=args.uvt_tile_t,
        uvt_tile_capacity=args.uvt_tile_capacity,
        uvt_init_views=args.uvt_init_views,
        uvt_init_sampling=args.uvt_init_sampling,
        uvt_init_frames=args.uvt_init_frames,
        skip_backward_microbreakdown=not args.star_backward_microbreakdown,
        skip_star=args.skip_star,
        skip_dynamic=args.skip_dynamic,
        splat_count=args.splat_count,
        splat_lr=args.splat_lr,
        splat_renderer=args.splat_renderer,
        splat_loss_scope=args.splat_loss_scope,
        splat_init_scale=args.splat_init_scale,
        init_depth=args.init_depth,
    )


def timing_total(summary: dict[str, Any], key: str = "total") -> dict[str, float | int]:
    value = summary.get(key)
    if not isinstance(value, dict):
        return {"count": 0, "mean_s": math.nan, "min_s": math.nan, "max_s": math.nan, "total_s": math.nan}
    return value


def compact_star_dynamic(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    total = timing_total(payload["summary"])
    render = timing_total(payload["summary"], "render")
    return {
        "name": name,
        "ok": True,
        "mean_step_s": float(total["mean_s"]),
        "measured_total_s": float(total["total_s"]),
        "mean_render_s": float(render["mean_s"]),
        "rows": payload["rows"],
        "phase_summary": payload["summary"],
        "loss_value_summary": payload["loss_value_summary"],
        "config": {
            key: payload.get(key)
            for key in (
                "tube_count",
                "splat_count",
                "render_backend",
                "reduction_mode",
                "sample_emission_mode",
                "camera_sequence_mode",
                "renderer",
                "loss_scope",
                "window_frames",
                "tile_x",
                "tile_y",
                "tile_t",
                "tile_capacity",
            )
            if key in payload
        },
    }


def compact_world_foam(payload: dict[str, Any]) -> dict[str, Any]:
    total = timing_total(payload["step_summary"])
    render = timing_total(payload["step_summary"], "render")
    return {
        "name": "world_foam",
        "ok": payload.get("status") == "ok",
        "mean_step_s": float(total["mean_s"]),
        "measured_total_s": float(total["total_s"]),
        "mean_render_s": float(render["mean_s"]),
        "frame_count": int(payload["frame_count"]),
        "render_size": int(payload["render_size"]),
        "rows": payload["step_rows"],
        "phase_summary": payload["step_summary"],
        "train_loop_elapsed_s": float(payload["train_loop_elapsed_s"]),
        "total_elapsed_s": float(payload["total_elapsed_s"]),
        "full_trainer_claim": bool(payload["full_trainer_claim"]),
        "gradient_scope": payload["gradient_scope"],
        "train_shared_forward_boundary_scan_ratio": float(payload["train"]["shared_forward_boundary_scan_ratio"]),
        "heldout_shared_forward_boundary_scan_ratio": float(payload["heldout"]["shared_forward_boundary_scan_ratio"]),
        "acceptance": payload["acceptance"],
    }


def add_summary_rows(summary_rows: list[dict[str, Any]], *, case: dict[str, Any], renderer: str, compact: dict[str, Any]) -> None:
    summary_rows.append(
        {
            "case_id": case["case_id"],
            "target_size": case["target_size"],
            "requested_frames": case["frames"],
            "loaded_frames": compact.get("frame_count", case.get("loaded_frame_count")),
            "renderer": renderer,
            "steps": case["steps"],
            "warmup_steps": case["warmup_steps"],
            "mean_step_s": compact["mean_step_s"],
            "measured_total_s": compact["measured_total_s"],
            "mean_render_s": compact["mean_render_s"],
        }
    )


def run_case(args: argparse.Namespace, *, baseline_config: dict[str, Any], base_record: dict[str, Any], case: dict[str, int]) -> dict[str, Any]:
    target_size = int(case["target_size"])
    frames = int(case["frames"])
    case_inputs = write_case_inputs(
        baseline_config=baseline_config,
        base_record=base_record,
        input_dir=args.input_dir,
        target_size=target_size,
        frames=frames,
        duration_seconds=args.case_duration_seconds,
    )
    case_report: dict[str, Any] = {
        "case_id": case_inputs["case_id"],
        "target_size": target_size,
        "frames": frames,
        "steps": args.steps,
        "warmup_steps": args.warmup_steps,
        "case_duration_seconds": case_inputs["duration_seconds"],
        "case_fps": case_inputs["fps"],
        "config_path": str(case_inputs["config_path"]),
        "manifest_path": str(case_inputs["manifest_path"]),
        "status": "ok",
    }
    started = time.perf_counter()

    if not args.skip_star_dynamic:
        probe = star_probe.run_probe(
            star_dynamic_args(args, config_path=case_inputs["config_path"], target_size=target_size, frames=frames)
        )
        loaded = int(probe["meta"]["loaded_frame_count"])
        case_report["loaded_frame_count"] = loaded
        if args.strict_frame_count and loaded != frames:
            raise ValueError(f"STAR/dynamic loader returned {loaded} frames for requested {frames}.")
        if probe["star_uvt"].get("rows"):
            case_report["star_uvt"] = compact_star_dynamic("star_uvt", probe["star_uvt"])
        if probe["free_dynamic_splats"].get("rows"):
            case_report["free_dynamic_splats"] = compact_star_dynamic("free_dynamic_splats", probe["free_dynamic_splats"])
        case_report["star_dynamic_meta"] = probe["meta"]

    if not args.skip_world_foam:
        world = run_world_foam_train_eval(
            config_path=case_inputs["config_path"],
            max_frames=frames,
            render_size=target_size,
            site_count=args.world_foam_site_count,
            time_slabs=args.world_foam_time_slabs,
            tile_h=args.world_foam_tile_h,
            tile_w=args.world_foam_tile_w,
            near=args.world_foam_near,
            far=args.world_foam_far,
            density=args.world_foam_density,
            invalid_epsilon=args.world_foam_invalid_epsilon,
            transmittance_threshold=args.world_foam_transmittance_threshold,
            steps=args.steps,
            warmup_steps=args.warmup_steps,
            lr=args.world_foam_lr,
            train_ppm_out=None,
            heldout_ppm_out=None,
        )
        loaded = int(world["frame_count"])
        case_report.setdefault("loaded_frame_count", loaded)
        if args.strict_frame_count and loaded != frames:
            raise ValueError(f"World Foam loader returned {loaded} frames for requested {frames}.")
        case_report["world_foam"] = compact_world_foam(world)

    case_report["elapsed_s"] = float(time.perf_counter() - started)
    return case_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fixed-step speed comparison for STAR-UVT, dynamic GSplats, and World Foam.")
    parser.add_argument("--baseline-config", type=Path, default=DEFAULT_BASELINE_CONFIG)
    parser.add_argument("--cases", default="128x8,128x16,128x32,256x32")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--case-duration-seconds", type=float, default=4.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--input-dir", type=Path, default=RESULTS_DIR / "fixed_step_speed_compare_inputs")
    parser.add_argument("--out-json", type=Path, default=RESULTS_DIR / "fixed_step_speed_compare_default.json")
    parser.add_argument("--allow-short-frame-count", dest="strict_frame_count", action="store_false")
    parser.set_defaults(strict_frame_count=True)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--skip-star-dynamic", action="store_true")
    parser.add_argument("--skip-world-foam", action="store_true")
    parser.add_argument("--skip-star", action="store_true")
    parser.add_argument("--skip-dynamic", action="store_true")

    parser.add_argument("--frame-stride-for-probe", type=int, default=3)
    parser.add_argument("--uvt-tubes", type=int, default=256)
    parser.add_argument("--uvt-lr", type=float, default=0.03)
    parser.add_argument("--uvt-render-backend", choices=("dense", "metal_tile"), default="metal_tile")
    parser.add_argument(
        "--uvt-camera-sequence-mode",
        choices=("static_view", "dynamic_first_order", "projective_first_order", "segmented", "per_frame_loop"),
        default="static_view",
    )
    parser.add_argument("--uvt-segment-frames", type=int, default=4)
    parser.add_argument("--uvt-synthetic-pan-x", type=float, default=0.0)
    parser.add_argument("--uvt-synthetic-pan-y", type=float, default=0.0)
    parser.add_argument("--uvt-synthetic-dolly-z", type=float, default=0.0)
    parser.add_argument("--uvt-synthetic-zoom", type=float, default=0.0)
    parser.add_argument("--uvt-synthetic-principal-x", type=float, default=0.0)
    parser.add_argument("--uvt-synthetic-principal-y", type=float, default=0.0)
    parser.add_argument(
        "--uvt-reduction-mode",
        choices=(
            "index_add",
            "sorted_cpu",
            "key_sort_scan_metal",
            "key_sort_compensated_scan_metal",
            "key_sort_segmented_metal",
        ),
        default="index_add",
    )
    parser.add_argument(
        "--uvt-sample-emission-mode",
        choices=(
            "atomic_append",
            "with_keys",
            "tile_pair",
            "tile_pair_compensated",
            "tile_pair_grouped",
            "tile_pair_parallel",
            "tile_pair_scanline",
            "tile_pair_target_bounds",
            "tile_pair_suffix",
            "direct_atomic",
            "direct_fixedpoint",
            "direct_split_fixedpoint",
            "direct_serial",
            "tile_pair_reduced",
            "tile_pair_suffix_reduced",
        ),
        default="direct_atomic",
    )
    parser.add_argument("--uvt-loss-scope", choices=("sampled_frame", "view_sequence", "temporal_window"), default="view_sequence")
    parser.add_argument("--uvt-window-frames", type=int, default=4)
    parser.add_argument("--uvt-init-precision-xy", type=float, default=30.0)
    parser.add_argument("--uvt-init-lambda-t", type=float, default=0.35)
    parser.add_argument("--uvt-init-opacity", type=float, default=0.35)
    parser.add_argument("--uvt-min-precision-xy", type=float, default=1.0e-5)
    parser.add_argument("--uvt-min-lambda-t", type=float, default=1.0e-5)
    parser.add_argument("--uvt-velocity-reg", type=float, default=1.0e-4)
    parser.add_argument("--uvt-depth-velocity-reg", type=float, default=0.0)
    parser.add_argument("--uvt-position-reg", type=float, default=1.0e-6)
    parser.add_argument("--uvt-tile-load-reg", type=float, default=0.001)
    parser.add_argument("--uvt-tile-load-target", type=float, default=7000.0)
    parser.add_argument("--uvt-depth-slope-reg", type=float, default=0.05)
    parser.add_argument("--uvt-depth-margin-reg", type=float, default=0.0)
    parser.add_argument("--uvt-depth-margin", type=float, default=0.05)
    parser.add_argument("--uvt-tile-x", type=int, default=8)
    parser.add_argument("--uvt-tile-y", type=int, default=8)
    parser.add_argument("--uvt-tile-t", type=int, default=1)
    parser.add_argument("--uvt-tile-capacity", type=int, default=256)
    parser.add_argument("--uvt-init-views", choices=("first", "all_train"), default="first")
    parser.add_argument("--uvt-init-sampling", choices=("random", "grid"), default="random")
    parser.add_argument("--uvt-init-frames", choices=("first", "all", "fit"), default="first")
    parser.add_argument("--star-backward-microbreakdown", action="store_true")

    parser.add_argument("--splat-count", type=int, default=2048)
    parser.add_argument("--splat-lr", type=float, default=0.002)
    parser.add_argument("--splat-renderer", choices=("dense", "fast_mac"), default="fast_mac")
    parser.add_argument("--splat-loss-scope", choices=("sampled_frame", "view_sequence"), default="view_sequence")
    parser.add_argument("--splat-init-scale", type=float, default=0.035)
    parser.add_argument("--init-depth", type=float, default=2.0)

    parser.add_argument("--world-foam-site-count", type=int, default=12)
    parser.add_argument("--world-foam-time-slabs", type=int, default=1)
    parser.add_argument("--world-foam-tile-h", type=int, default=8)
    parser.add_argument("--world-foam-tile-w", type=int, default=8)
    parser.add_argument("--world-foam-near", type=float, default=0.05)
    parser.add_argument("--world-foam-far", type=float, default=3.25)
    parser.add_argument("--world-foam-density", type=float, default=2.0)
    parser.add_argument("--world-foam-invalid-epsilon", type=float, default=1.0e-7)
    parser.add_argument("--world-foam-transmittance-threshold", type=float, default=1.0e-4)
    parser.add_argument("--world-foam-lr", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps must be nonnegative")
    if args.case_duration_seconds <= 0.0:
        raise ValueError("--case-duration-seconds must be positive")

    baseline_config_path = star_probe.resolve_dynaworld_path(args.baseline_config)
    baseline_config = star_probe.load_config_file(baseline_config_path)
    base_record, base_manifest = select_base_record(baseline_config)
    cases = parse_cases(args.cases)
    results = []
    summary_rows: list[dict[str, Any]] = []
    failures = []

    for case in cases:
        case_id = f"{case['target_size']}px_{case['frames']}f"
        print(f"Running {case_id}: steps={args.steps} warmup={args.warmup_steps}")
        try:
            payload = run_case(args, baseline_config=baseline_config, base_record=base_record, case=case)
            results.append(payload)
            for renderer in ("star_uvt", "free_dynamic_splats", "world_foam"):
                compact = payload.get(renderer)
                if isinstance(compact, dict):
                    add_summary_rows(summary_rows, case=payload, renderer=renderer, compact=compact)
        except Exception as exc:
            failure = {
                "case_id": case_id,
                "target_size": case["target_size"],
                "frames": case["frames"],
                "status": "failed",
                "error": repr(exc),
            }
            failures.append(failure)
            results.append(failure)
            print(f"FAILED {case_id}: {exc!r}")
            if args.fail_fast:
                raise

    report = {
        "benchmark": "world_foam_lane2_fixed_step_speed_compare",
        "status": "ok" if not failures else "failed",
        "meta": {
            "baseline_config": str(baseline_config_path),
            "base_manifest": str(base_manifest),
            "base_sample_id": base_record.get("sample_id"),
            "cases": cases,
            "steps": args.steps,
            "warmup_steps": args.warmup_steps,
            "case_duration_seconds": args.case_duration_seconds,
            "strict_frame_count": args.strict_frame_count,
            "device": args.device,
            "notes": [
                "Measured totals include fixed optimizer steps only; warmup steps are excluded from mean_step_s.",
                "Generated per-case manifests resample the same raw clip duration so 8f/16f/32f rows really load the requested frame count.",
                "STAR-UVT and dynamic GSplats default to one train camera full-sequence loss per step in this harness.",
                "STAR-UVT defaults to direct_atomic in this fixed-step harness; atomic_append is the historical sample-emission path.",
                "World Foam still renders all train-camera rays per step and is fixed-geometry/site-RGBA only, not a full trainer parity claim.",
            ],
        },
        "summary_table": summary_rows,
        "cases": results,
        "failures": failures,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "out_json": str(args.out_json), "rows": summary_rows}, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
