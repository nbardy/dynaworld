from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_experiments.star_uvt_feature_tubes.projective_trained_high_motion_trace_scaling_benchmark import (  # noqa: E402
    verify_trained_high_motion_trace_scaling_report,
)


DEFAULT_OUT_DIR = ROOT / "outputs" / "benchmarks" / "2026-05-25_star_uvt_projective_real_active_set_distribution"
DEFAULT_INPUT_REPORTS = (
    ROOT / "outputs" / "benchmarks" / "2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling" / "summary.json",
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_64px_128t"
    / "summary.json",
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_96px_256t_cap256"
    / "summary.json",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_float(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def _artifact_label(path: Path) -> str:
    return path.parent.name.replace("2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling", "trained")


def _trained_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [row for row in report.get("rows", []) if isinstance(row, dict) and row.get("label") == "trained_checkpoint"]
    return sorted(rows, key=lambda row: int(row.get("frames", 0)))


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    rows = report["rows"]
    artifact_rows = report["artifacts"]
    final_rows = [row for row in rows if int(row["frames"]) == int(report["final_frame_count"])]
    return {
        "artifact_count": int(report["artifact_count"]),
        "row_count": len(rows),
        "final_frame_count": int(report["final_frame_count"]),
        "all_underlying_verifiers_pass": all(not artifact["verifier_errors"] for artifact in artifact_rows),
        "all_source_videos_exist": all(bool(artifact["source_video_exists"]) for artifact in artifact_rows),
        "all_fallback_free": all(int(row["fallback_cells"]) == 0 and float(row["fallback_fraction"]) == 0.0 for row in rows),
        "min_active_set_groups": min(int(row["tile_active_set_groups"]) for row in rows),
        "max_active_set_groups": max(int(row["tile_active_set_groups"]) for row in rows),
        "max_cells_per_active_set_group": max(int(row["max_cells_per_active_set_group"]) for row in rows),
        "max_active_set_group_to_dense_tile_pair_ratio": max(
            float(row["active_set_group_to_dense_tile_pair_ratio"]) for row in rows
        ),
        "max_final_active_set_group_to_dense_tile_pair_ratio": max(
            float(row["active_set_group_to_dense_tile_pair_ratio"]) for row in final_rows
        ),
        "max_cell_to_active_set_group_ratio": max(float(row["cell_to_active_set_group_ratio"]) for row in rows),
        "min_interval_to_dense_tile_pair_ratio": min(float(row["interval_to_dense_tile_pair_ratio"]) for row in rows),
        "max_interval_to_dense_tile_pair_ratio": max(float(row["interval_to_dense_tile_pair_ratio"]) for row in rows),
    }


def run_report(input_reports: tuple[Path, ...] = DEFAULT_INPUT_REPORTS) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    final_frame_count = 0
    for path in input_reports:
        source_report = _load_json(path)
        verifier_errors = verify_trained_high_motion_trace_scaling_report(source_report)
        artifact_rows = _trained_rows(source_report)
        if artifact_rows:
            final_frame_count = max(final_frame_count, max(int(row["frames"]) for row in artifact_rows))
        artifacts.append(
            {
                "path": str(path),
                "label": _artifact_label(path),
                "status": source_report.get("status"),
                "source_video": source_report.get("source_video"),
                "source_video_exists": bool(source_report.get("source_video_exists")),
                "frame_counts": list(source_report.get("frame_counts", [])),
                "size": int(source_report.get("size") or 0),
                "tube_count": int(source_report.get("tube_count") or 0),
                "tile_capacity": int(source_report.get("tile_capacity") or 0),
                "verifier_errors": verifier_errors,
            }
        )
        for row in artifact_rows:
            active_set_groups = int(row["tile_active_set_groups"])
            dense_tile_pairs = int(row["dense_per_frame_tile_pairs"])
            cell_count = int(row["cell_count"])
            rows.append(
                {
                    "artifact_label": _artifact_label(path),
                    "frames": int(row["frames"]),
                    "size": int(source_report.get("size") or 0),
                    "tube_count": int(source_report.get("tube_count") or 0),
                    "tile_capacity": int(source_report.get("tile_capacity") or 0),
                    "trace_count": int(row["trace_count"]),
                    "cell_count": cell_count,
                    "tile_active_set_groups": active_set_groups,
                    "max_cells_per_active_set_group": int(row["max_cells_per_active_set_group"]),
                    "dense_per_frame_tile_pairs": dense_tile_pairs,
                    "active_set_group_to_dense_tile_pair_ratio": float(active_set_groups) / float(max(1, dense_tile_pairs)),
                    "cell_to_active_set_group_ratio": float(cell_count) / float(max(1, active_set_groups)),
                    "interval_trace_entries": int(row["interval_trace_entries"]),
                    "interval_to_dense_tile_pair_ratio": float(row["interval_to_dense_tile_pair_ratio"]),
                    "fallback_cells": int(row["fallback_cells"]),
                    "fallback_fraction": float(row["fallback_fraction"]),
                    "fallback_reasons": list(row["fallback_reasons"]),
                    "velocity_mean_px_per_frame": float(row["velocity_mean_px_per_frame"]),
                    "velocity_max_px_per_frame": float(row["velocity_max_px_per_frame"]),
                }
            )

    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_active_set_distribution",
        "base_domain": "checked-in high-motion real-video projective interval atlases",
        "theory_contract": (
            "This report does not prove broad real-scene quality acceptance. It verifies that checked-in "
            "high-motion real-video projective interval atlases expose bounded, fallback-free active-set "
            "topology distributions, so active-set splitting is measured on real compiled traces rather "
            "than only synthetic q-family strata."
        ),
        "artifact_count": len(artifacts),
        "final_frame_count": int(final_frame_count),
        "artifacts": artifacts,
        "rows": rows,
    }
    report["summary"] = summarize(report)
    errors = verify_real_active_set_distribution_report(report)
    if errors:
        report["status"] = "failed"
        report["errors"] = errors
    return report


def _assert_summary_close(actual: Any, expected: Any, key: str, errors: list[str]) -> None:
    if isinstance(expected, float):
        if not _finite_float(actual) or abs(float(actual) - expected) > 1.0e-8:
            errors.append(f"summary {key} mismatch: expected {expected!r}, got {actual!r}")
    elif actual != expected:
        errors.append(f"summary {key} mismatch: expected {expected!r}, got {actual!r}")


def verify_real_active_set_distribution_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_real_active_set_distribution":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    if report.get("base_domain") != "checked-in high-motion real-video projective interval atlases":
        errors.append(f"base_domain must name the real-video atlas distribution, got {report.get('base_domain')!r}")
    theory_contract = report.get("theory_contract")
    if (
        not isinstance(theory_contract, str)
        or "does not prove broad real-scene quality acceptance" not in theory_contract
        or "active-set topology distributions" not in theory_contract
        or "real compiled traces" not in theory_contract
    ):
        errors.append("theory_contract must preserve the real active-set distribution scope")

    artifacts = report.get("artifacts")
    rows = report.get("rows")
    if not isinstance(artifacts, list) or len(artifacts) < 3:
        errors.append("artifacts must contain at least three checked-in high-motion reports")
    if not isinstance(rows, list) or len(rows) < 9:
        errors.append("rows must contain trained_checkpoint rows from all input artifacts")
        return errors

    artifact_labels: set[str] = set()
    if isinstance(artifacts, list):
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                errors.append("artifact row must be an object")
                continue
            artifact_labels.add(str(artifact.get("label")))
            if artifact.get("status") != "ok":
                errors.append(f"input artifact {artifact.get('path')} must have ok status")
            if artifact.get("source_video_exists") is not True:
                errors.append(f"input artifact {artifact.get('path')} must have an existing source video")
            if artifact.get("verifier_errors"):
                errors.append(f"input artifact {artifact.get('path')} verifier failed: {artifact.get('verifier_errors')}")
            if int(artifact.get("tube_count") or 0) <= 0 or int(artifact.get("tile_capacity") or 0) <= 0:
                errors.append("input artifact tube_count and tile_capacity must be positive")
    if len(artifact_labels) < 3:
        errors.append("real active-set distribution must cover at least three distinct artifact scales")

    row_labels: dict[str, set[int]] = {}
    for row in rows:
        if not isinstance(row, dict):
            errors.append("distribution row must be an object")
            continue
        label = str(row.get("artifact_label"))
        row_labels.setdefault(label, set()).add(int(row.get("frames") or 0))
        active_set_groups = int(row.get("tile_active_set_groups") or 0)
        dense_tile_pairs = int(row.get("dense_per_frame_tile_pairs") or 0)
        cell_count = int(row.get("cell_count") or 0)
        max_cells_per_group = int(row.get("max_cells_per_active_set_group") or 0)
        if active_set_groups <= 0 or dense_tile_pairs <= 0 or cell_count <= 0:
            errors.append("distribution rows must have positive active-set groups, dense tile pairs, and cells")
        if max_cells_per_group <= 0 or max_cells_per_group > 3:
            errors.append("real active-set distribution max_cells_per_active_set_group must stay <= 3")
        if active_set_groups > cell_count:
            errors.append("active-set groups cannot exceed cell count")
        ratio = row.get("active_set_group_to_dense_tile_pair_ratio")
        if not _finite_float(ratio) or float(ratio) >= 0.05:
            errors.append("active-set group/dense-tile-pair ratio must stay below 0.05")
        cell_ratio = row.get("cell_to_active_set_group_ratio")
        if not _finite_float(cell_ratio) or float(cell_ratio) > 1.5:
            errors.append("cell/active-set-group ratio must stay bounded")
        if int(row.get("fallback_cells") or 0) != 0 or float(row.get("fallback_fraction") or 0.0) != 0.0:
            errors.append("real active-set rows must be fallback-free")
        if row.get("fallback_reasons"):
            errors.append("real active-set rows must not report fallback reasons")
        if float(row.get("velocity_mean_px_per_frame") or 0.0) <= 0.0:
            errors.append("real active-set rows must retain nonzero projected motion")

    expected_frames = {4, 8, 16}
    for label, frames in row_labels.items():
        if frames != expected_frames:
            errors.append(f"artifact {label} must cover frames {sorted(expected_frames)}, got {sorted(frames)}")

    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        return errors
    try:
        expected = summarize(report)
    except (KeyError, ZeroDivisionError, TypeError, ValueError) as exc:
        errors.append(f"summary could not be recomputed: {exc}")
        return errors
    for key, expected_value in expected.items():
        _assert_summary_close(summary.get(key), expected_value, key, errors)
    if summary.get("all_underlying_verifiers_pass") is not True:
        errors.append("all input artifact verifiers must pass")
    if summary.get("all_source_videos_exist") is not True:
        errors.append("all input source videos must exist")
    if summary.get("all_fallback_free") is not True:
        errors.append("all real active-set rows must be fallback-free")
    if int(summary.get("max_cells_per_active_set_group") or 0) > 3:
        errors.append("summary max_cells_per_active_set_group must stay <= 3")
    if float(summary.get("max_active_set_group_to_dense_tile_pair_ratio") or 1.0) >= 0.05:
        errors.append("summary active-set group/dense-tile-pair ratio must stay below 0.05")
    return errors


def assert_real_active_set_distribution_report(report: dict[str, Any]) -> None:
    errors = verify_real_active_set_distribution_report(report)
    if errors:
        raise AssertionError("real active-set distribution report failed:\n- " + "\n- ".join(errors))


def write_report(report: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--input-report", type=Path, action="append", dest="input_reports")
    parser.add_argument("--verify-report", type=Path)
    args = parser.parse_args()

    if args.verify_report is not None:
        report = _load_json(args.verify_report)
        assert_real_active_set_distribution_report(report)
        print(f"verified {args.verify_report}")
        return

    input_reports = tuple(args.input_reports) if args.input_reports else DEFAULT_INPUT_REPORTS
    report = run_report(input_reports=input_reports)
    assert_real_active_set_distribution_report(report)
    write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
