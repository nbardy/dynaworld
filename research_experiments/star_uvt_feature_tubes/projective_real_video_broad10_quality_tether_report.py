from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from projective_real_video_multiscene_quality_tether_report import (  # noqa: E402
    REQUIRED_GRADIENT_FLAGS,
    _max_abs_delta,
)
from projective_real_video_multiscene_trainer_matrix import (  # noqa: E402
    verify_real_video_multiscene_trainer_matrix_report,
)


DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_broad10_quality_tether"
)
DEFAULT_TRAINER_REPORT = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix_broad10"
    / "summary.json"
)
LOSS_CURVE_TOLERANCE = 2.0e-8


def _finite_float(value: Any, label: str, errors: list[str]) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        errors.append(f"{label} must be finite, got {value!r}")
        return 0.0
    return float(value)


def _case_path(case_dir: Path, scene_id: str, policy: str) -> Path:
    return case_dir / f"{scene_id}_{policy}.json"


def _load_case_pair(case_dir: Path, scene_id: str) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    cadence_path = _case_path(case_dir, scene_id, "cadence")
    measured_path = _case_path(case_dir, scene_id, "measured")
    cadence = json.loads(cadence_path.read_text(encoding="utf-8"))
    measured = json.loads(measured_path.read_text(encoding="utf-8"))
    return cadence, measured, cadence_path, measured_path


def _row_from_pair(case_dir: Path, scene: dict[str, Any], frames: int) -> dict[str, Any]:
    scene_id = str(scene["scene_id"])
    errors: list[str] = []
    cadence, measured, cadence_path, measured_path = _load_case_pair(case_dir, scene_id)
    cadence_losses = list(cadence.get("losses") or [])
    measured_losses = list(measured.get("losses") or [])
    cadence_rgb_losses = list(cadence.get("rgb_losses") or [])
    measured_rgb_losses = list(measured.get("rgb_losses") or [])
    max_loss_delta = _max_abs_delta(cadence_losses, measured_losses, "loss", errors)
    max_rgb_loss_delta = _max_abs_delta(cadence_rgb_losses, measured_rgb_losses, "rgb_loss", errors)
    cadence_start_loss = _finite_float(cadence.get("start_loss"), "cadence start_loss", errors)
    cadence_end_loss = _finite_float(cadence.get("end_loss"), "cadence end_loss", errors)
    measured_start_loss = _finite_float(measured.get("start_loss"), "measured start_loss", errors)
    measured_end_loss = _finite_float(measured.get("end_loss"), "measured end_loss", errors)
    cadence_start_psnr = _finite_float(cadence.get("start_psnr"), "cadence start_psnr", errors)
    cadence_end_psnr = _finite_float(cadence.get("end_psnr"), "cadence end_psnr", errors)
    measured_start_psnr = _finite_float(measured.get("start_psnr"), "measured start_psnr", errors)
    measured_end_psnr = _finite_float(measured.get("end_psnr"), "measured end_psnr", errors)
    missing_grad_flags = [
        flag
        for flag in REQUIRED_GRADIENT_FLAGS
        if cadence.get(flag) is not True or measured.get(flag) is not True
    ]
    return {
        "scene_id": scene_id,
        "youtube_id": str(scene.get("youtube_id", "")),
        "frames": int(frames),
        "cadence_case_path": str(cadence_path),
        "measured_case_path": str(measured_path),
        "case_files_exist": cadence_path.exists() and measured_path.exists(),
        "curve_length": len(cadence_losses),
        "max_abs_loss_curve_delta": max_loss_delta,
        "max_abs_rgb_loss_curve_delta": max_rgb_loss_delta,
        "start_loss_abs_delta": abs(measured_start_loss - cadence_start_loss),
        "end_loss_abs_delta": abs(measured_end_loss - cadence_end_loss),
        "start_psnr_abs_delta": abs(measured_start_psnr - cadence_start_psnr),
        "end_psnr_abs_delta": abs(measured_end_psnr - cadence_end_psnr),
        "measured_loss_decrease": measured_start_loss - measured_end_loss,
        "cadence_loss_decrease": cadence_start_loss - cadence_end_loss,
        "measured_psnr_gain": measured_end_psnr - measured_start_psnr,
        "cadence_psnr_gain": cadence_end_psnr - cadence_start_psnr,
        "measured_end_psnr": measured_end_psnr,
        "cadence_end_psnr": cadence_end_psnr,
        "measured_end_loss": measured_end_loss,
        "cadence_end_loss": cadence_end_loss,
        "measured_pass": measured.get("pass") is True,
        "cadence_pass": cadence.get("pass") is True,
        "missing_gradient_flags": missing_grad_flags,
        "row_errors": errors,
    }


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    rows = report["rows"]
    return {
        "scene_count": len({row["scene_id"] for row in rows}),
        "distinct_youtube_id_count": len({row["youtube_id"] for row in rows}),
        "frame_count_count": len({int(row["frames"]) for row in rows}),
        "pair_count": len(rows),
        "all_case_files_exist": all(bool(row["case_files_exist"]) for row in rows),
        "all_rows_pass": all(bool(row["measured_pass"]) and bool(row["cadence_pass"]) for row in rows),
        "all_rows_error_free": all(not row["row_errors"] for row in rows),
        "all_gradient_flags_present": all(not row["missing_gradient_flags"] for row in rows),
        "all_measured_loss_curves_match_cadence": all(
            float(row["max_abs_loss_curve_delta"]) <= LOSS_CURVE_TOLERANCE for row in rows
        ),
        "all_measured_rgb_loss_curves_match_cadence": all(
            float(row["max_abs_rgb_loss_curve_delta"]) <= LOSS_CURVE_TOLERANCE for row in rows
        ),
        "all_measured_end_psnr_matches_cadence": all(float(row["end_psnr_abs_delta"]) <= 1.0e-8 for row in rows),
        "all_measured_psnr_improves": all(float(row["measured_psnr_gain"]) > 0.0 for row in rows),
        "all_measured_loss_decreases": all(float(row["measured_loss_decrease"]) > 0.0 for row in rows),
        "max_abs_loss_curve_delta": max(float(row["max_abs_loss_curve_delta"]) for row in rows),
        "max_abs_rgb_loss_curve_delta": max(float(row["max_abs_rgb_loss_curve_delta"]) for row in rows),
        "max_end_loss_abs_delta": max(float(row["end_loss_abs_delta"]) for row in rows),
        "max_end_psnr_abs_delta": max(float(row["end_psnr_abs_delta"]) for row in rows),
        "min_measured_psnr_gain": min(float(row["measured_psnr_gain"]) for row in rows),
        "min_measured_loss_decrease": min(float(row["measured_loss_decrease"]) for row in rows),
        "min_measured_end_psnr": min(float(row["measured_end_psnr"]) for row in rows),
        "max_measured_end_loss": max(float(row["measured_end_loss"]) for row in rows),
    }


def verify_real_video_broad10_quality_tether_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_real_video_broad10_quality_tether":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    theory_contract = report.get("theory_contract")
    if (
        not isinstance(theory_contract, str)
        or "does not prove broad media acceptance" not in theory_contract
        or "does not prove timing acceptance" not in theory_contract
        or "broad10" not in theory_contract
        or "cadence full-rebuild reference" not in theory_contract
    ):
        errors.append("theory_contract must preserve broad10 quality-only scope")
    source_errors = report.get("source_trainer_verifier_errors")
    if source_errors != []:
        errors.append(f"source broad10 trainer report must verify first, got {source_errors!r}")
    source_summary = report.get("source_trainer_summary")
    if not isinstance(source_summary, dict):
        errors.append("source_trainer_summary must be an object")
    else:
        if int(source_summary.get("distinct_youtube_id_count") or 0) < 10:
            errors.append("source broad10 trainer report must cover at least 10 distinct videos")
        if int(source_summary.get("row_count") or 0) < 20:
            errors.append("source broad10 trainer report must cover at least 20 cadence/measured rows")
        if source_summary.get("all_measured_loss_matches_cadence") is not True:
            errors.append("source broad10 trainer report must match measured/cadence losses")
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("rows must be a nonempty list")
        return errors
    if len({row.get("youtube_id") for row in rows if isinstance(row, dict)}) < 10:
        errors.append("rows must cover at least 10 distinct YouTube ids")
    for row in rows:
        if not isinstance(row, dict):
            errors.append("row must be an object")
            continue
        prefix = f"{row.get('scene_id')} {row.get('frames')}f"
        if row.get("case_files_exist") is not True:
            errors.append(f"{prefix} case files must exist")
        if row.get("measured_pass") is not True or row.get("cadence_pass") is not True:
            errors.append(f"{prefix} measured and cadence cases must pass")
        if row.get("row_errors"):
            errors.append(f"{prefix} row errors must be empty: {row.get('row_errors')}")
        if row.get("missing_gradient_flags"):
            errors.append(f"{prefix} must preserve all required gradient flags: {row.get('missing_gradient_flags')}")
        if int(row.get("curve_length") or 0) < 2:
            errors.append(f"{prefix} loss curve must include multiple steps")
        if (
            _finite_float(row.get("max_abs_loss_curve_delta"), f"{prefix} loss curve delta", errors)
            > LOSS_CURVE_TOLERANCE
        ):
            errors.append(f"{prefix} measured loss curve must match cadence")
        if (
            _finite_float(row.get("max_abs_rgb_loss_curve_delta"), f"{prefix} rgb loss curve delta", errors)
            > LOSS_CURVE_TOLERANCE
        ):
            errors.append(f"{prefix} measured rgb-loss curve must match cadence")
        if _finite_float(row.get("end_psnr_abs_delta"), f"{prefix} end psnr delta", errors) > 1.0e-8:
            errors.append(f"{prefix} measured end PSNR must match cadence")
        if _finite_float(row.get("measured_loss_decrease"), f"{prefix} measured loss decrease", errors) <= 0.0:
            errors.append(f"{prefix} measured loss must decrease")
        if _finite_float(row.get("measured_psnr_gain"), f"{prefix} measured psnr gain", errors) <= 0.0:
            errors.append(f"{prefix} measured PSNR must improve")
    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        return errors
    try:
        expected = summarize(report)
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"summary could not be recomputed: {exc}")
        return errors
    for key, expected_value in expected.items():
        actual = summary.get(key)
        if isinstance(expected_value, float):
            if not isinstance(actual, int | float) or abs(float(actual) - expected_value) > 1.0e-9:
                errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual!r}")
        elif actual != expected_value:
            errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual!r}")
    if int(summary.get("distinct_youtube_id_count") or 0) < 10:
        errors.append("broad10 quality tether must cover at least 10 distinct YouTube ids")
    if summary.get("all_measured_loss_curves_match_cadence") is not True:
        errors.append("broad10 quality tether must match measured and cadence loss curves")
    if summary.get("all_measured_psnr_improves") is not True:
        errors.append("broad10 quality tether must improve measured PSNR on every pair")
    return errors


def assert_real_video_broad10_quality_tether_report(report: dict[str, Any]) -> None:
    errors = verify_real_video_broad10_quality_tether_report(report)
    if errors:
        raise AssertionError("real-video broad10 quality tether failed:\n- " + "\n- ".join(errors))


def run_report(trainer_report: Path = DEFAULT_TRAINER_REPORT, case_dir: Path | None = None) -> dict[str, Any]:
    source = json.loads(trainer_report.read_text(encoding="utf-8"))
    case_dir = case_dir or trainer_report.parent / "cases"
    frames = int(source.get("frames") or 0)
    scenes = [scene for scene in source.get("scenes", []) if isinstance(scene, dict)]
    rows = [_row_from_pair(case_dir, scene, frames) for scene in scenes]
    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_broad10_quality_tether",
        "base_domain": "saved broad10 trainer case payloads",
        "theory_contract": (
            "This broad10 report does not prove broad media acceptance and does not prove timing acceptance. "
            "It tethers the measured live-cache projective-interval path to the cadence full-rebuild reference "
            "by comparing saved loss curves, end losses, PSNR, and gradient-flow flags across 10 source videos."
        ),
        "source_trainer_report": str(trainer_report),
        "source_trainer_verifier_errors": verify_real_video_multiscene_trainer_matrix_report(source),
        "source_trainer_summary": source.get("summary", {}),
        "case_dir": str(case_dir),
        "required_gradient_flags": list(REQUIRED_GRADIENT_FLAGS),
        "loss_curve_tolerance": LOSS_CURVE_TOLERANCE,
        "rows": rows,
    }
    report["summary"] = summarize(report)
    errors = verify_real_video_broad10_quality_tether_report(report)
    if errors:
        report["status"] = "failed"
        report["errors"] = errors
    return report


def write_report(report: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "summary.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--trainer-report", type=Path, default=DEFAULT_TRAINER_REPORT)
    parser.add_argument("--case-dir", type=Path)
    parser.add_argument("--verify-report", type=Path)
    args = parser.parse_args()
    if args.verify_report is not None:
        report = json.loads(args.verify_report.read_text(encoding="utf-8"))
        assert_real_video_broad10_quality_tether_report(report)
        print(f"verified {args.verify_report}")
        return
    report = run_report(trainer_report=args.trainer_report, case_dir=args.case_dir)
    if report.get("status") == "ok":
        assert_real_video_broad10_quality_tether_report(report)
    path = write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
