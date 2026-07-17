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

from research_experiments.star_uvt_feature_tubes.projective_real_video_trainer_frame_scaling_benchmark import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_REAL_VIDEO_TRAINER_OUT_DIR,
    verify_guarded_real_video_trainer_support_report,
    verify_real_video_trainer_frame_scaling_report,
)


DEFAULT_OUT_DIR = ROOT / "outputs" / "benchmarks" / "2026-05-25_star_uvt_projective_real_video_guarded_support_matrix"
DEFAULT_UNGUARDED_REPORT = DEFAULT_REAL_VIDEO_TRAINER_OUT_DIR / "summary.json"
DEFAULT_GUARDED_REPORTS = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_trainer_frame_scaling_guard025_tail001"
    / "summary.json",
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_trainer_frame_scaling_guard05_tail001"
    / "summary.json",
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_trainer_frame_scaling_guard10_tail001"
    / "summary.json",
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_trainer_frame_scaling_guard20_tail001"
    / "summary.json",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _finite_float(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def _artifact_label(report: dict[str, Any], *, is_guarded: bool) -> str:
    if not is_guarded:
        return "unguarded"
    padding = float(report.get("support_guard_padding") or 0.0)
    return f"guard{padding:g}"


def _rows_for_policy(report: dict[str, Any], policy: str) -> list[dict[str, Any]]:
    return sorted(
        [row for row in report.get("rows", []) if isinstance(row, dict) and row.get("policy") == policy],
        key=lambda row: int(row.get("frames") or 0),
    )


def _measured_rows(report: dict[str, Any], *, label: str, is_guarded: bool) -> list[dict[str, Any]]:
    cadence = {int(row.get("frames") or 0): row for row in _rows_for_policy(report, "cadence")}
    rows: list[dict[str, Any]] = []
    for measured in _rows_for_policy(report, "measured"):
        frames = int(measured.get("frames") or 0)
        cadence_row = cadence.get(frames, {})
        cadence_no_first = float(cadence_row.get("no_first_step_ms") or 0.0)
        cadence_rebuilds = int(cadence_row.get("projective_interval_cache_rebuilds") or 0)
        rows.append(
            {
                "artifact_label": label,
                "is_guarded": bool(is_guarded),
                "frames": frames,
                "pass": bool(measured.get("pass")),
                "loss_decreased": bool(measured.get("loss_decreased")),
                "end_loss": float(measured.get("end_loss") or 0.0),
                "cadence_end_loss": float(cadence_row.get("end_loss") or 0.0),
                "end_loss_delta_vs_cadence": abs(
                    float(measured.get("end_loss") or 0.0) - float(cadence_row.get("end_loss") or 0.0)
                ),
                "no_first_step_ms": float(measured.get("no_first_step_ms") or 0.0),
                "cadence_no_first_step_ms": cadence_no_first,
                "no_first_ratio_vs_cadence": float(measured.get("no_first_step_ms") or 0.0)
                / float(max(cadence_no_first, 1.0e-12)),
                "mean_backward_ms": float(measured.get("mean_backward_ms") or 0.0),
                "mean_render_forward_ms": float(measured.get("mean_render_forward_ms") or 0.0),
                "cache_rebuilds": int(measured.get("projective_interval_cache_rebuilds") or 0),
                "cadence_cache_rebuilds": cadence_rebuilds,
                "cache_rebuild_ratio_vs_cadence": float(measured.get("projective_interval_cache_rebuilds") or 0)
                / float(max(cadence_rebuilds, 1)),
                "cache_live_updates": int(measured.get("projective_interval_cache_live_updates") or 0),
                "cache_staleness_checks": int(measured.get("projective_interval_cache_staleness_checks") or 0),
                "cache_stale_refreshes": int(measured.get("projective_interval_cache_stale_refreshes") or 0),
                "cache_support_rebins": int(measured.get("projective_interval_cache_support_rebins") or 0),
                "cache_visibility_stratifications": int(
                    measured.get("projective_interval_cache_visibility_stratifications") or 0
                ),
                "cache_fallback_marks": int(measured.get("projective_interval_cache_fallback_marks") or 0),
                "support_max_overshoot_px": float(
                    measured.get("projective_interval_cache_max_support_max_overshoot_px") or 0.0
                ),
                "support_tail_alpha_bound": float(
                    measured.get("projective_interval_cache_max_support_tail_alpha_bound") or 0.0
                ),
                "effective_support_uv_padding": float(
                    measured.get("projective_interval_effective_support_uv_padding") or 0.0
                ),
                "tile_overflow_sum": int(measured.get("tile_overflow_sum") or 0),
                "max_tile_count": int(measured.get("max_tile_count") or 0),
            }
        )
    return rows


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    artifacts = report["artifacts"]
    rows = report["rows"]
    guarded_artifacts = [artifact for artifact in artifacts if artifact["is_guarded"]]
    guarded_rows = [row for row in rows if row["is_guarded"]]
    unguarded_rows = [row for row in rows if not row["is_guarded"]]
    return {
        "artifact_count": len(artifacts),
        "guarded_artifact_count": len(guarded_artifacts),
        "measured_row_count": len(rows),
        "guarded_measured_row_count": len(guarded_rows),
        "all_underlying_verifiers_pass": all(not artifact["verifier_errors"] for artifact in artifacts),
        "all_guarded_support_verifiers_pass": all(
            not artifact["guarded_verifier_errors"] for artifact in guarded_artifacts
        ),
        "all_source_videos_exist": all(bool(artifact["source_video_exists"]) for artifact in artifacts),
        "min_guard_padding": min(float(artifact["support_guard_padding"]) for artifact in guarded_artifacts),
        "max_guard_padding": max(float(artifact["support_guard_padding"]) for artifact in guarded_artifacts),
        "all_guarded_loss_matches_cadence": all(float(row["end_loss_delta_vs_cadence"]) < 1.0e-5 for row in guarded_rows),
        "all_guarded_no_overflow": all(int(row["tile_overflow_sum"]) == 0 for row in guarded_rows),
        "all_guarded_fallback_free": all(int(row["cache_fallback_marks"]) == 0 for row in guarded_rows),
        "default_measured_support_rebins": sum(int(row["cache_support_rebins"]) for row in unguarded_rows),
        "guarded_measured_support_rebins": sum(int(row["cache_support_rebins"]) for row in guarded_rows),
        "guarded_measured_stale_refreshes": sum(int(row["cache_stale_refreshes"]) for row in guarded_rows),
        "guarded_measured_fallback_marks": sum(int(row["cache_fallback_marks"]) for row in guarded_rows),
        "max_guarded_measured_tail_alpha_bound": max(float(row["support_tail_alpha_bound"]) for row in guarded_rows),
        "max_guarded_measured_overshoot_px": max(float(row["support_max_overshoot_px"]) for row in guarded_rows),
        "max_guarded_measured_no_first_ratio": max(float(row["no_first_ratio_vs_cadence"]) for row in guarded_rows),
        "max_guarded_measured_rebuild_ratio": max(float(row["cache_rebuild_ratio_vs_cadence"]) for row in guarded_rows),
        "max_guarded_measured_loss_delta": max(float(row["end_loss_delta_vs_cadence"]) for row in guarded_rows),
        "max_guarded_tile_count": max(int(row["max_tile_count"]) for row in guarded_rows),
        "max_guarded_effective_support_uv_padding": max(float(row["effective_support_uv_padding"]) for row in guarded_rows),
    }


def run_report(
    *,
    unguarded_report: Path = DEFAULT_UNGUARDED_REPORT,
    guarded_reports: tuple[Path, ...] = DEFAULT_GUARDED_REPORTS,
) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for path, is_guarded in ((unguarded_report, False), *((path, True) for path in guarded_reports)):
        source_report = _load_json(path)
        label = _artifact_label(source_report, is_guarded=is_guarded)
        verifier_errors = verify_real_video_trainer_frame_scaling_report(source_report)
        guarded_errors = verify_guarded_real_video_trainer_support_report(source_report) if is_guarded else []
        artifacts.append(
            {
                "path": str(path),
                "label": label,
                "is_guarded": bool(is_guarded),
                "status": source_report.get("status"),
                "source_video": source_report.get("source_video"),
                "source_video_exists": bool(source_report.get("source_video_exists")),
                "frame_counts": list(source_report.get("frame_counts", [])),
                "steps": int(source_report.get("steps") or 0),
                "tile_capacity": int(source_report.get("tile_capacity") or 0),
                "support_guard_padding": float(source_report.get("support_guard_padding") or 0.0),
                "support_guard_policy": str(source_report.get("support_guard_policy") or ""),
                "support_stale_tail_alpha_epsilon": float(
                    source_report.get("support_stale_tail_alpha_epsilon") or 0.0
                ),
                "verifier_errors": verifier_errors,
                "guarded_verifier_errors": guarded_errors,
            }
        )
        rows.extend(_measured_rows(source_report, label=label, is_guarded=is_guarded))

    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_guarded_support_matrix",
        "base_domain": "checked-in high-motion real-video trainer guarded support matrix",
        "theory_contract": (
            "This report does not prove broad real-scene quality acceptance. It verifies that guarded support "
            "certificates on the real-video trainer route remove support rebins/stale refreshes while preserving "
            "the measured live-cache reuse contract."
        ),
        "artifacts": artifacts,
        "rows": rows,
    }
    report["summary"] = summarize(report)
    errors = verify_real_video_guarded_support_matrix_report(report)
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


def verify_real_video_guarded_support_matrix_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_real_video_guarded_support_matrix":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    if report.get("base_domain") != "checked-in high-motion real-video trainer guarded support matrix":
        errors.append(f"base_domain must name the real-video guarded support matrix, got {report.get('base_domain')!r}")
    theory_contract = report.get("theory_contract")
    if (
        not isinstance(theory_contract, str)
        or "does not prove broad real-scene quality acceptance" not in theory_contract
        or "guarded support certificates" not in theory_contract
        or "measured live-cache reuse" not in theory_contract
    ):
        errors.append("theory_contract must preserve the guarded real-video support scope")

    artifacts = report.get("artifacts")
    rows = report.get("rows")
    if not isinstance(artifacts, list) or len(artifacts) < 5:
        errors.append("artifacts must contain one unguarded report plus at least four guarded reports")
    if not isinstance(rows, list) or len(rows) < 15:
        errors.append("rows must contain measured rows from the unguarded and guarded reports")
        return errors

    guarded_labels: set[str] = set()
    unguarded_labels: set[str] = set()
    if isinstance(artifacts, list):
        for artifact in artifacts:
            if not isinstance(artifact, dict):
                errors.append("artifact row must be an object")
                continue
            label = str(artifact.get("label"))
            is_guarded = bool(artifact.get("is_guarded"))
            if is_guarded:
                guarded_labels.add(label)
            else:
                unguarded_labels.add(label)
            if artifact.get("status") != "ok":
                errors.append(f"input artifact {artifact.get('path')} must have ok status")
            if artifact.get("source_video_exists") is not True:
                errors.append(f"input artifact {artifact.get('path')} must have an existing source video")
            if artifact.get("verifier_errors"):
                errors.append(f"input artifact {artifact.get('path')} verifier failed: {artifact.get('verifier_errors')}")
            if is_guarded:
                if artifact.get("guarded_verifier_errors"):
                    errors.append(
                        f"guarded input artifact {artifact.get('path')} verifier failed: "
                        f"{artifact.get('guarded_verifier_errors')}"
                    )
                if float(artifact.get("support_guard_padding") or 0.0) <= 0.0:
                    errors.append("guarded artifacts must have positive support_guard_padding")
                if artifact.get("support_guard_policy") != "slack_budgeted":
                    errors.append("guarded artifacts must use slack_budgeted support guards")
                if float(artifact.get("support_stale_tail_alpha_epsilon") or 0.0) <= 0.0:
                    errors.append("guarded artifacts must have positive tail alpha epsilon")
    if len(guarded_labels) < 4:
        errors.append("guarded support matrix must cover at least four distinct guard paddings")
    if len(unguarded_labels) != 1:
        errors.append("guarded support matrix must include exactly one unguarded baseline label")

    row_labels: dict[str, set[int]] = {}
    for row in rows:
        if not isinstance(row, dict):
            errors.append("measured row must be an object")
            continue
        label = str(row.get("artifact_label"))
        row_labels.setdefault(label, set()).add(int(row.get("frames") or 0))
        if row.get("pass") is not True or row.get("loss_decreased") is not True:
            errors.append(f"{label} {row.get('frames')}f measured row must pass and decrease loss")
        if int(row.get("tile_overflow_sum") or 0) != 0:
            errors.append(f"{label} {row.get('frames')}f measured row must have zero tile overflow")
        if int(row.get("cache_visibility_stratifications") or 0) != 0:
            errors.append(f"{label} {row.get('frames')}f measured row must have zero visibility stratifications")
        if int(row.get("cache_fallback_marks") or 0) != 0:
            errors.append(f"{label} {row.get('frames')}f measured row must be fallback-free")
        if not _finite_float(row.get("end_loss_delta_vs_cadence")) or float(row["end_loss_delta_vs_cadence"]) >= 1.0e-5:
            errors.append(f"{label} {row.get('frames')}f measured loss must match cadence")
        if not _finite_float(row.get("no_first_ratio_vs_cadence")) or float(row["no_first_ratio_vs_cadence"]) >= 1.0:
            errors.append(f"{label} {row.get('frames')}f measured no-first timing must beat cadence")
        if not _finite_float(row.get("cache_rebuild_ratio_vs_cadence")) or float(row["cache_rebuild_ratio_vs_cadence"]) >= 1.0:
            errors.append(f"{label} {row.get('frames')}f measured rebuild ratio must stay below cadence")
        if row.get("is_guarded") is True:
            if int(row.get("cache_support_rebins") or 0) != 0:
                errors.append(f"{label} {row.get('frames')}f guarded measured support rebins must be zero")
            if int(row.get("cache_stale_refreshes") or 0) != 0:
                errors.append(f"{label} {row.get('frames')}f guarded measured stale refreshes must be zero")
            if float(row.get("support_tail_alpha_bound") or 0.0) > 1.0e-3:
                errors.append(f"{label} {row.get('frames')}f guarded tail alpha bound must stay <= 1e-3")

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
    if summary.get("all_guarded_support_verifiers_pass") is not True:
        errors.append("all guarded support artifact verifiers must pass")
    if summary.get("all_source_videos_exist") is not True:
        errors.append("all input source videos must exist")
    if int(summary.get("default_measured_support_rebins") or 0) <= 0:
        errors.append("unguarded baseline must expose support rebins for this guarded-support matrix")
    if int(summary.get("guarded_measured_support_rebins") or 0) != 0:
        errors.append("guarded support matrix must eliminate measured support rebins")
    if int(summary.get("guarded_measured_stale_refreshes") or 0) != 0:
        errors.append("guarded support matrix must eliminate measured stale refreshes")
    max_tail_bound = summary.get("max_guarded_measured_tail_alpha_bound")
    if not _finite_float(max_tail_bound) or float(max_tail_bound) > 1.0e-3:
        errors.append("guarded support tail alpha bound must stay <= 1e-3")
    max_no_first_ratio = summary.get("max_guarded_measured_no_first_ratio")
    if not _finite_float(max_no_first_ratio) or float(max_no_first_ratio) >= 1.0:
        errors.append("guarded measured no-first timings must beat cadence")
    max_rebuild_ratio = summary.get("max_guarded_measured_rebuild_ratio")
    if not _finite_float(max_rebuild_ratio) or float(max_rebuild_ratio) >= 1.0:
        errors.append("guarded measured rebuild ratios must stay below cadence")
    return errors


def assert_real_video_guarded_support_matrix_report(report: dict[str, Any]) -> None:
    errors = verify_real_video_guarded_support_matrix_report(report)
    if errors:
        raise AssertionError("real-video guarded support matrix report failed:\n- " + "\n- ".join(errors))


def write_report(report: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--unguarded-report", type=Path, default=DEFAULT_UNGUARDED_REPORT)
    parser.add_argument("--guarded-report", type=Path, action="append", dest="guarded_reports")
    parser.add_argument("--verify-report", type=Path)
    args = parser.parse_args()

    if args.verify_report is not None:
        report = _load_json(args.verify_report)
        assert_real_video_guarded_support_matrix_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report(
        unguarded_report=args.unguarded_report,
        guarded_reports=tuple(args.guarded_reports) if args.guarded_reports else DEFAULT_GUARDED_REPORTS,
    )
    path = write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
