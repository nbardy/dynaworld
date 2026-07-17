#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DYNAWORLD = Path(__file__).resolve().parents[2]
RESULTS_DIR = DYNAWORLD / "research_experiments" / "world_foam_lane2" / "results"
DEFAULT_WORLDFOAM_ARTIFACT = (
    RESULTS_DIR / "2026-05-20_native_cutwalk_worldfoam_star_starretry.attempt1.worldfoam.json"
)
DEFAULT_STAR_COMPARISON_ARTIFACT = (
    RESULTS_DIR / "2026-05-20_native_cutwalk_worldfoam_star_starretry.star_attempt1.star_compare.json"
)
DEFAULT_STAR_SOURCE_RGB_PSNR = 29.823
DEFAULT_SOLID_SOURCE_RGB_PSNR = 21.36


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return payload


def _finite_float(value: Any) -> float | None:
    if not isinstance(value, (float, int)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _rows_by_frame(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("WorldFoam artifact rows must be a list")
    by_frame: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("WorldFoam artifact rows must contain objects")
        frame_count = row.get("frame_count")
        if not isinstance(frame_count, int):
            raise ValueError("WorldFoam row missing integer frame_count")
        if frame_count in by_frame:
            raise ValueError(f"duplicate WorldFoam frame_count {frame_count}")
        by_frame[frame_count] = row
    if not by_frame:
        raise ValueError("WorldFoam artifact must contain at least one row")
    return by_frame


def _psnr_by_frame(rows_by_frame: dict[int, dict[str, Any]], key: str) -> dict[str, float]:
    values = {}
    for frame_count, row in sorted(rows_by_frame.items()):
        value = _finite_float(row.get(key))
        if value is None:
            raise ValueError(f"WorldFoam {frame_count}f row missing finite {key}")
        values[str(frame_count)] = value
    return values


def _bool_by_frame(rows_by_frame: dict[int, dict[str, Any]], key: str) -> dict[str, bool]:
    return {str(frame_count): bool(row.get(key, False)) for frame_count, row in sorted(rows_by_frame.items())}


def _int_by_frame(rows_by_frame: dict[int, dict[str, Any]], key: str) -> dict[str, int]:
    values = {}
    for frame_count, row in sorted(rows_by_frame.items()):
        value = row.get(key, frame_count)
        if not isinstance(value, int):
            raise ValueError(f"WorldFoam {frame_count}f row missing integer {key}")
        values[str(frame_count)] = value
    return values


def summarize_worldfoam_artifact(
    worldfoam_artifact: Path,
    *,
    star_source_rgb_psnr: float,
    solid_source_rgb_psnr: float,
) -> dict[str, Any]:
    worldfoam_payload = _load_json(worldfoam_artifact)
    rows_by_frame = _rows_by_frame(worldfoam_payload)
    train_psnr_by_frame = _psnr_by_frame(rows_by_frame, "final_train_psnr")
    heldout_psnr_by_frame = _psnr_by_frame(rows_by_frame, "final_heldout_psnr")
    best_train_psnr = max(train_psnr_by_frame.values())
    best_heldout_psnr = max(heldout_psnr_by_frame.values())
    return {
        "artifact": str(worldfoam_artifact),
        "status": worldfoam_payload.get("status"),
        "quality_claim": bool(worldfoam_payload.get("quality_claim", False)),
        "frame_counts": sorted(rows_by_frame),
        "render_size": worldfoam_payload.get("render_size"),
        "site_count": worldfoam_payload.get("site_count"),
        "tape_mode": worldfoam_payload.get("tape_mode"),
        "loaded_frame_count_by_frame": _int_by_frame(rows_by_frame, "loaded_frame_count"),
        "repeat_loaded_frames_by_frame": _bool_by_frame(rows_by_frame, "repeat_loaded_frames"),
        "train_psnr_by_frame": train_psnr_by_frame,
        "heldout_psnr_by_frame": heldout_psnr_by_frame,
        "best_train_psnr": best_train_psnr,
        "best_heldout_psnr": best_heldout_psnr,
        "quality_gaps": {
            "train_psnr_gap_to_star_uvt_source": float(star_source_rgb_psnr) - best_train_psnr,
            "train_psnr_gap_to_solid_same_source": float(solid_source_rgb_psnr) - best_train_psnr,
        },
    }


def _ratio_map(payload: dict[str, Any], key: str) -> dict[str, float]:
    comparison = payload.get("comparison")
    if not isinstance(comparison, dict):
        return {}
    ratios = comparison.get(key)
    if not isinstance(ratios, dict):
        return {}
    parsed = {}
    for frame, value in ratios.items():
        finite = _finite_float(value)
        if finite is not None:
            parsed[str(frame)] = finite
    return parsed


def summarize_speed_bridge(star_comparison: dict[str, Any] | None) -> dict[str, Any]:
    if star_comparison is None:
        return {
            "available": False,
            "speed_competitive_micro_gate": False,
            "reason": "no STAR comparison artifact provided",
        }
    total_ratios = _ratio_map(star_comparison, "total_median_ms_ratio_star_over_worldfoam_by_frame")
    backward_ratios = _ratio_map(star_comparison, "backward_median_ms_ratio_star_over_worldfoam_by_frame")
    common_frames = sorted(set(total_ratios) & set(backward_ratios), key=lambda item: int(item))
    worldfoam_faster_total = bool(common_frames) and all(total_ratios[frame] > 1.0 for frame in common_frames)
    worldfoam_faster_backward = bool(common_frames) and all(backward_ratios[frame] > 1.0 for frame in common_frames)
    return {
        "available": bool(common_frames),
        "source_artifact_status": star_comparison.get("status"),
        "frame_counts": [int(frame) for frame in common_frames],
        "star_over_worldfoam_total_ratio_by_frame": {frame: total_ratios[frame] for frame in common_frames},
        "star_over_worldfoam_backward_ratio_by_frame": {frame: backward_ratios[frame] for frame in common_frames},
        "worldfoam_faster_total_all_frames": worldfoam_faster_total,
        "worldfoam_faster_backward_all_frames": worldfoam_faster_backward,
        "speed_competitive_micro_gate": (
            star_comparison.get("status") == "ok"
            and worldfoam_faster_total
            and worldfoam_faster_backward
        ),
    }


def _compare_candidate_to_primary(
    candidate: dict[str, Any],
    *,
    primary: dict[str, Any],
) -> dict[str, Any]:
    candidate_frames = {int(frame) for frame in candidate["frame_counts"]}
    primary_frames = {int(frame) for frame in primary["frame_counts"]}
    common_frames = sorted(candidate_frames & primary_frames)
    missing_primary_frames = sorted(primary_frames - candidate_frames)
    extra_frames = sorted(candidate_frames - primary_frames)
    candidate_train = candidate["train_psnr_by_frame"]
    primary_train = primary["train_psnr_by_frame"]
    train_delta_by_frame = {
        str(frame): float(candidate_train[str(frame)]) - float(primary_train[str(frame)])
        for frame in common_frames
    }
    candidate_heldout = candidate["heldout_psnr_by_frame"]
    primary_heldout = primary["heldout_psnr_by_frame"]
    heldout_delta_by_frame = {
        str(frame): float(candidate_heldout[str(frame)]) - float(primary_heldout[str(frame)])
        for frame in common_frames
    }
    return {
        "same_frame_set_as_primary": candidate_frames == primary_frames,
        "common_frame_counts_with_primary": common_frames,
        "missing_primary_frame_counts": missing_primary_frames,
        "extra_frame_counts_vs_primary": extra_frames,
        "train_psnr_delta_vs_primary_by_common_frame": train_delta_by_frame,
        "heldout_psnr_delta_vs_primary_by_common_frame": heldout_delta_by_frame,
        "improves_train_psnr_on_any_common_frame": any(delta > 0.0 for delta in train_delta_by_frame.values()),
        "improves_train_psnr_on_all_common_frames": bool(train_delta_by_frame)
        and all(delta > 0.0 for delta in train_delta_by_frame.values()),
    }


def build_report(
    *,
    worldfoam_artifact: Path,
    star_comparison_artifact: Path | None,
    star_source_rgb_psnr: float,
    solid_source_rgb_psnr: float,
    quality_gap_tolerance: float,
    extra_worldfoam_artifacts: tuple[Path, ...] = (),
) -> dict[str, Any]:
    failures: list[str] = []
    try:
        worldfoam_summary = summarize_worldfoam_artifact(
            worldfoam_artifact,
            star_source_rgb_psnr=star_source_rgb_psnr,
            solid_source_rgb_psnr=solid_source_rgb_psnr,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "failed",
            "failures": [str(exc)],
            "worldfoam_artifact": str(worldfoam_artifact),
            "star_comparison_artifact": str(star_comparison_artifact) if star_comparison_artifact else None,
        }

    capacity_candidates = []
    for extra_artifact in extra_worldfoam_artifacts:
        try:
            candidate_summary = summarize_worldfoam_artifact(
                extra_artifact,
                star_source_rgb_psnr=star_source_rgb_psnr,
                solid_source_rgb_psnr=solid_source_rgb_psnr,
            )
            candidate_summary["primary_frame_comparison"] = _compare_candidate_to_primary(
                candidate_summary,
                primary=worldfoam_summary,
            )
            capacity_candidates.append(candidate_summary)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"could not load capacity candidate {extra_artifact}: {exc}")

    star_payload = None
    if star_comparison_artifact is not None:
        try:
            star_payload = _load_json(star_comparison_artifact)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"could not load STAR comparison artifact: {exc}")

    best_train_psnr = float(worldfoam_summary["best_train_psnr"])
    quality_competitive_with_star_source = (
        best_train_psnr + float(quality_gap_tolerance) >= float(star_source_rgb_psnr)
    )
    quality_competitive_with_solid_source = (
        best_train_psnr + float(quality_gap_tolerance) >= float(solid_source_rgb_psnr)
    )
    speed_bridge = summarize_speed_bridge(star_payload)
    star_uvt_competitive_claim = (
        speed_bridge.get("speed_competitive_micro_gate") is True
        and quality_competitive_with_star_source
    )
    if worldfoam_summary["quality_claim"] is True:
        failures.append("WorldFoam artifact unexpectedly claims quality parity")
    claimed_candidates = [
        str(candidate["artifact"]) for candidate in capacity_candidates if candidate["quality_claim"] is True
    ]
    if claimed_candidates:
        failures.append(f"capacity candidates unexpectedly claim quality parity: {','.join(claimed_candidates)}")
    best_quality_summary = max(
        [worldfoam_summary, *capacity_candidates],
        key=lambda item: float(item["best_train_psnr"]),
    )
    best_quality_train_psnr = float(best_quality_summary["best_train_psnr"])
    best_quality_competitive_with_star_source = (
        best_quality_train_psnr + float(quality_gap_tolerance) >= float(star_source_rgb_psnr)
    )
    best_quality_competitive_with_solid_source = (
        best_quality_train_psnr + float(quality_gap_tolerance) >= float(solid_source_rgb_psnr)
    )
    best_quality_is_primary_speed_artifact = best_quality_summary["artifact"] == str(worldfoam_artifact)
    return {
        "status": "failed" if failures else "ok",
        "failures": failures,
        "scope": (
            "WorldFoam fused-MSE quality bridge. This report compares WorldFoam RGB PSNR "
            "from the frozen-geometry train/eval artifact against source-overfit RGB baselines; "
            "it is not a novel-view or full-system promotion by itself."
        ),
        "worldfoam_artifact": str(worldfoam_artifact),
        "star_comparison_artifact": str(star_comparison_artifact) if star_comparison_artifact else None,
        "worldfoam": {key: value for key, value in worldfoam_summary.items() if key != "artifact"},
        "capacity_candidates": capacity_candidates,
        "capacity_candidate_count": len(capacity_candidates),
        "capacity_candidates_improve_train_psnr": any(
            float(candidate["best_train_psnr"]) > best_train_psnr for candidate in capacity_candidates
        ),
        "capacity_candidates_improve_train_psnr_on_any_common_frame": any(
            candidate["primary_frame_comparison"]["improves_train_psnr_on_any_common_frame"]
            for candidate in capacity_candidates
        ),
        "capacity_candidates_improve_train_psnr_on_all_common_frames": any(
            candidate["primary_frame_comparison"]["improves_train_psnr_on_all_common_frames"]
            for candidate in capacity_candidates
        ),
        "capacity_candidate_artifacts_missing_primary_frames": [
            candidate["artifact"]
            for candidate in capacity_candidates
            if candidate["primary_frame_comparison"]["missing_primary_frame_counts"]
        ],
        "best_worldfoam_quality_artifact": best_quality_summary["artifact"],
        "best_worldfoam_quality": {
            "best_train_psnr": best_quality_summary["best_train_psnr"],
            "best_heldout_psnr": best_quality_summary["best_heldout_psnr"],
            "render_size": best_quality_summary["render_size"],
            "site_count": best_quality_summary["site_count"],
            "frame_counts": best_quality_summary["frame_counts"],
            "quality_gaps": best_quality_summary["quality_gaps"],
        },
        "best_worldfoam_quality_is_primary_speed_artifact": best_quality_is_primary_speed_artifact,
        "best_worldfoam_quality_primary_frame_comparison": (
            {
                "same_frame_set_as_primary": True,
                "common_frame_counts_with_primary": worldfoam_summary["frame_counts"],
                "missing_primary_frame_counts": [],
                "extra_frame_counts_vs_primary": [],
                "train_psnr_delta_vs_primary_by_common_frame": {
                    str(frame): 0.0 for frame in worldfoam_summary["frame_counts"]
                },
                "heldout_psnr_delta_vs_primary_by_common_frame": {
                    str(frame): 0.0 for frame in worldfoam_summary["frame_counts"]
                },
                "improves_train_psnr_on_any_common_frame": False,
                "improves_train_psnr_on_all_common_frames": False,
            }
            if best_quality_is_primary_speed_artifact
            else best_quality_summary["primary_frame_comparison"]
        ),
        "best_worldfoam_quality_competitive_with_star_source": best_quality_competitive_with_star_source,
        "best_worldfoam_quality_competitive_with_solid_same_source": best_quality_competitive_with_solid_source,
        "best_worldfoam_quality_needs_matched_speed_gate": (
            speed_bridge.get("speed_competitive_micro_gate") is True
            and best_quality_competitive_with_star_source
            and not best_quality_is_primary_speed_artifact
        ),
        "baselines": {
            "star_uvt_highmotion_source_rgb_psnr": float(star_source_rgb_psnr),
            "solid_same_source_rgb_psnr": float(solid_source_rgb_psnr),
            "quality_gap_tolerance": float(quality_gap_tolerance),
        },
        "quality_gaps": {
            "train_psnr_gap_to_star_uvt_source": float(star_source_rgb_psnr) - best_train_psnr,
            "train_psnr_gap_to_solid_same_source": float(solid_source_rgb_psnr) - best_train_psnr,
        },
        "quality_competitive_with_star_source": quality_competitive_with_star_source,
        "quality_competitive_with_solid_same_source": quality_competitive_with_solid_source,
        "speed_bridge": speed_bridge,
        "star_uvt_competitive_claim": star_uvt_competitive_claim,
        "next_gate": (
            "Do not promote broad STAR competitiveness until a WorldFoam quality/capacity "
            "run closes the RGB PSNR gap while preserving the clean speed gate."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report WorldFoam speed evidence against STAR RGB-quality baselines.")
    parser.add_argument("--worldfoam-artifact", type=Path, default=DEFAULT_WORLDFOAM_ARTIFACT)
    parser.add_argument(
        "--extra-worldfoam-artifact",
        action="append",
        type=Path,
        default=[],
        help="Additional WorldFoam quality/capacity artifact to summarize without promoting broad parity.",
    )
    parser.add_argument("--star-comparison-artifact", type=Path, default=DEFAULT_STAR_COMPARISON_ARTIFACT)
    parser.add_argument("--no-star-comparison", action="store_true")
    parser.add_argument("--star-source-rgb-psnr", type=float, default=DEFAULT_STAR_SOURCE_RGB_PSNR)
    parser.add_argument("--solid-source-rgb-psnr", type=float, default=DEFAULT_SOLID_SOURCE_RGB_PSNR)
    parser.add_argument("--quality-gap-tolerance", type=float, default=1.0)
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    star_comparison_artifact = None if args.no_star_comparison else args.star_comparison_artifact
    report = build_report(
        worldfoam_artifact=args.worldfoam_artifact,
        extra_worldfoam_artifacts=tuple(args.extra_worldfoam_artifact),
        star_comparison_artifact=star_comparison_artifact,
        star_source_rgb_psnr=float(args.star_source_rgb_psnr),
        solid_source_rgb_psnr=float(args.solid_source_rgb_psnr),
        quality_gap_tolerance=float(args.quality_gap_tolerance),
    )
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
