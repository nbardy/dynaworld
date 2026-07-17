from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from projective_goal_completion_gap_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_COMPLETION_GAP_OUT_DIR,
    verify_projective_goal_completion_gap_current_acceptance,
    verify_projective_goal_completion_gap_report,
)
from projective_goal_progress_audit import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_GOAL_PROGRESS_OUT_DIR,
    OPEN_REQUIREMENT_ID as PROGRESS_OPEN_REQUIREMENT_ID,
    PROVEN_REQUIREMENT_IDS,
    verify_projective_goal_progress_current_acceptance,
    verify_projective_goal_progress_audit,
)
from report_artifacts import write_report_json, write_report_text  # noqa: E402


DEFAULT_OUT_DIR = ROOT / "outputs" / "benchmarks" / "2026-05-25_star_uvt_projective_goal_final_completion_audit"
DEFAULT_GOAL_PROGRESS_REPORT = DEFAULT_GOAL_PROGRESS_OUT_DIR / "summary.json"
DEFAULT_COMPLETION_GAP_REPORT = DEFAULT_COMPLETION_GAP_OUT_DIR / "summary.json"
THEORY_DIR = ROOT / "research_notes" / "gauged_uvt_trace_atlas"
GOAL_META_KEY_MATH = THEORY_DIR / "GOAL_META_KEY_MATH.md"

REQUIRED_THEORY_SUBFOLDERS = (
    "00_bundle_foundations",
    "01_camera_gauge_choices",
    "02_gaussian_fiber_pushforward",
    "03_projective_rational_traces",
    "04_revolving_camera_atlas",
    "05_visibility_strata",
    "06_exposure_and_rolling",
    "07_adjoint_training",
    "08_worldfoam_bridge",
    "09_metal_acceptance_plan",
)

EVIDENCE_ORDER = (
    "theory_plan",
    "goal_progress",
    "completion_gap",
)

MATH_REQUIREMENTS = (
    "formal_goal_contract",
    "fiber_gauge_trace_invariant",
    "clean_fiber_derivatives",
    "local_camera_family_bundle_math",
    "local_camera_family_2d_bundle_math",
)

CAMERA_PROGRAM_REQUIREMENTS = (
    "local_camera_family_shared_metadata",
    "local_camera_family_2d_shared_metadata",
    "local_camera_family_2d_tile_order_reuse",
    "local_camera_family_2d_tile_order_strata",
    "local_camera_family_2d_active_set_strata",
    "real_video_active_set_distribution",
)

METAL_REQUIREMENTS = (
    "local_camera_family_2d_metal_slice_lowering",
    "local_camera_family_2d_metal_shared_backward",
    "local_camera_family_2d_metal_single_launch_materialized",
    "local_camera_family_2d_metal_native_family_eval",
    "local_camera_family_2d_metal_native_interval_forward",
    "local_camera_family_2d_metal_native_interval_backward",
    "metal_time_shared_forward_backward",
)

VISIBILITY_EXPOSURE_REQUIREMENTS = (
    "finite_exposure_rolling_fallback",
    "real_video_guarded_support_matrix",
    "real_video_timing_variance_envelope",
)

REAL_VIDEO_REQUIREMENTS = (
    "real_video_trainer_smoke",
    "real_video_multiscene_trainer_matrix",
    "real_video_multiscene_extended_trainer_matrix",
    "real_video_multiscene_frame_scaling_matrix",
    "real_video_multiscene_extended_frame_scaling_diagnostic",
    "real_video_multiscene_quality_tether",
    "real_video_multiscene_extended_quality_tether",
    "real_video_multiscene_media_tether",
    "real_video_multiscene_extended_media_tether",
    "real_video_acceptance_envelope",
)

COMPLETION_GAP_PROVED_ROWS = (
    "formal_goal_memory_and_audit",
    "sublinear_world_side_work_proxy",
    "broad_real_scene_quality_acceptance",
    "full_compiled_adjoint_trainer_replacement",
    "timing_acceptance_protocol",
)

CONCRETE_GAP_KEYS = (
    "broad_quality_source_gap",
    "broad_media_source_gap",
    "broad_quality_frame_count_gap",
    "strict_timing_failure_gap",
    "timing_acceptance_gap",
    "compiled_trainer_source_gap",
    "compiled_trainer_replacement_gap",
)

Verifier = Callable[[dict[str, Any]], list[str]]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _requirement_statuses(report: dict[str, Any]) -> dict[str, str]:
    rows = report.get("requirements", [])
    if not isinstance(rows, list):
        return {}
    statuses: dict[str, str] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_id = row.get("id")
        status = row.get("status")
        if isinstance(row_id, str) and isinstance(status, str):
            statuses[row_id] = status
    return statuses


def _artifact(
    path: Path,
    verifier: Verifier,
    current_verifier: Verifier,
) -> dict[str, Any]:
    report = _load_json(path)
    return {
        "path": str(path),
        "benchmark": report.get("benchmark"),
        "status": report.get("status"),
        "verifier_errors": verifier(report),
        "current_input_errors": current_verifier(report),
        "summary": report.get("summary", {}),
        "requirement_statuses": _requirement_statuses(report),
    }


def _theory_plan_artifact(theory_dir: Path = THEORY_DIR) -> dict[str, Any]:
    present = sorted(path.name for path in theory_dir.iterdir() if path.is_dir()) if theory_dir.exists() else []
    missing = [name for name in REQUIRED_THEORY_SUBFOLDERS if name not in present]
    extra_numbered = [
        name
        for name in present
        if len(name) >= 3 and name[:2].isdigit() and name not in REQUIRED_THEORY_SUBFOLDERS
    ]
    summary = {
        "theory_dir": str(theory_dir),
        "goal_meta_key_math_path": str(GOAL_META_KEY_MATH),
        "goal_meta_key_math_exists": GOAL_META_KEY_MATH.exists(),
        "required_subfolder_count": len(REQUIRED_THEORY_SUBFOLDERS),
        "present_required_subfolder_count": len(REQUIRED_THEORY_SUBFOLDERS) - len(missing),
        "required_subfolders_present": len(missing) == 0,
        "missing_required_subfolders": missing,
        "extra_numbered_subfolders": extra_numbered,
        "at_most_ten_numbered_theory_subfolders": len(
            [name for name in present if len(name) >= 3 and name[:2].isdigit()]
        )
        <= 10,
    }
    errors = []
    if not summary["goal_meta_key_math_exists"]:
        errors.append("GOAL_META_KEY_MATH.md must exist")
    if not summary["required_subfolders_present"]:
        errors.append(f"missing theory subfolders: {missing}")
    if not summary["at_most_ten_numbered_theory_subfolders"]:
        errors.append("numbered theory subfolders must stay at or below 10")
    if extra_numbered:
        errors.append(f"unexpected numbered theory subfolders: {extra_numbered}")
    return {
        "path": str(theory_dir),
        "benchmark": "gauged_uvt_trace_atlas_theory_plan",
        "status": "ok" if not errors else "incomplete",
        "verifier_errors": errors,
        "current_input_errors": [],
        "summary": summary,
        "requirement_statuses": {},
    }


def _all_status(statuses: dict[str, str], ids: tuple[str, ...], status: str = "proved") -> bool:
    return all(statuses.get(row_id) == status for row_id in ids)


def _missing_statuses(statuses: dict[str, str], ids: tuple[str, ...], status: str = "proved") -> list[str]:
    return [row_id for row_id in ids if statuses.get(row_id) != status]


def _row(
    row_id: str,
    status: str,
    statement: str,
    evidence: list[str],
    missing: list[str],
    current: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": row_id,
        "status": status,
        "statement": statement,
        "evidence": evidence,
        "missing": missing,
    }
    if current is not None:
        row["current"] = current
    return row


def objective_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    progress_statuses = summary["progress_requirement_statuses"]
    gap_statuses = summary["completion_gap_requirement_statuses"]
    concrete_gap_values = summary["concrete_gap_values"]
    theory = summary["theory_plan"]
    inputs_current = summary["all_evidence_current_and_valid"]

    theory_ok = (
        theory["goal_meta_key_math_exists"]
        and theory["required_subfolders_present"]
        and theory["at_most_ten_numbered_theory_subfolders"]
    )
    math_ok = _all_status(progress_statuses, MATH_REQUIREMENTS)
    camera_ok = _all_status(progress_statuses, CAMERA_PROGRAM_REQUIREMENTS)
    metal_ok = _all_status(progress_statuses, METAL_REQUIREMENTS)
    visibility_ok = _all_status(progress_statuses, VISIBILITY_EXPOSURE_REQUIREMENTS)
    real_video_ok = _all_status(progress_statuses, REAL_VIDEO_REQUIREMENTS)
    shared_ok = progress_statuses.get("sublinear_world_side_work_proxy") == "proved"
    gap_rows_ok = _all_status(gap_statuses, COMPLETION_GAP_PROVED_ROWS)
    final_gap_shape_ok = (
        summary["completion_gap_open_gap_ids"] == ["full_goal_completion"]
        and all(value == 0 for value in concrete_gap_values.values())
        and summary["completion_gap_completion_ready"] is False
        and summary["completion_gap_does_not_prove_completion"] is True
    )
    compiled_ok = (
        progress_statuses.get("real_video_compiled_adjoint_replacement") == "proved"
        and gap_statuses.get("full_compiled_adjoint_trainer_replacement") == "proved"
    )

    return [
        _row(
            "theory_plan_memory_contract",
            "proved" if theory_ok else "missing",
            "The theory/plans folder preserves the goal/meta/key-math note and exactly the ten numbered theory tracks requested for the camera-ray bundle atlas.",
            ["theory_plan"],
            []
            if theory_ok
            else [
                "restore GOAL_META_KEY_MATH.md and the 00..09 gauged UVT theory subfolders without adding extra numbered tracks"
            ],
            current=theory,
        ),
        _row(
            "fiber_bundle_trace_math_and_derivatives",
            "proved" if math_ok else "missing",
            "The core UVT trace is formalized as pi_* Gamma^* world_primitive with gauge-invariant values, clean primitive derivatives, and camera-family derivatives.",
            ["goal_progress"],
            _missing_statuses(progress_statuses, MATH_REQUIREMENTS),
        ),
        _row(
            "revolving_camera_family_and_visibility_atlas",
            "proved" if camera_ok else "missing",
            "Complex/revolving camera trajectories are handled as camera-family bundle charts with shared metadata, tile/order strata, and active-set strata rather than one brittle global UVT chart.",
            ["goal_progress"],
            _missing_statuses(progress_statuses, CAMERA_PROGRAM_REQUIREMENTS),
        ),
        _row(
            "metal_forward_backward_renderer_path",
            "proved" if metal_ok else "missing",
            "The projective/gauged atlas reaches Metal forward, compositing, batching, and backward/VJP paths instead of remaining a paper-only theory.",
            ["goal_progress"],
            _missing_statuses(progress_statuses, METAL_REQUIREMENTS),
        ),
        _row(
            "visibility_exposure_rolling_fallback_contract",
            "proved" if visibility_ok else "missing",
            "Finite exposure, rolling shutter, visibility ambiguity, guarded support, and timing-variance caveats are represented as audited renderer contracts.",
            ["goal_progress"],
            _missing_statuses(progress_statuses, VISIBILITY_EXPOSURE_REQUIREMENTS),
        ),
        _row(
            "sublinear_world_side_work_and_bandwidth",
            "proved" if shared_ok and gap_rows_ok else "missing",
            "World-side projection/support/binning/visibility/backward work is shown sublinear versus per-frame replay by the shared-work proxy and completion-gap rows.",
            ["goal_progress", "completion_gap"],
            []
            if shared_ok and gap_rows_ok
            else _missing_statuses(progress_statuses, ("sublinear_world_side_work_proxy",))
            + _missing_statuses(gap_statuses, COMPLETION_GAP_PROVED_ROWS),
            current=summary["shared_work_proxy"],
        ),
        _row(
            "broad_real_video_renderer_acceptance",
            "proved" if real_video_ok and gap_statuses.get("broad_real_scene_quality_acceptance") == "proved" else "missing",
            "The renderer has broad real-video functional, quality, media, frame-count, and accepted timing evidence beyond narrow synthetic cases.",
            ["goal_progress", "completion_gap"],
            []
            if real_video_ok and gap_statuses.get("broad_real_scene_quality_acceptance") == "proved"
            else _missing_statuses(progress_statuses, REAL_VIDEO_REQUIREMENTS)
            + _missing_statuses(gap_statuses, ("broad_real_scene_quality_acceptance",)),
        ),
        _row(
            "compiled_adjoint_training_replacement",
            "proved" if compiled_ok else "missing",
            "The practical real-video trainer path uses the compiled projective interval atlas with Metal forward and direct interval VJP while preserving gradients and cache reuse.",
            ["goal_progress", "completion_gap"],
            []
            if compiled_ok
            else _missing_statuses(progress_statuses, ("real_video_compiled_adjoint_replacement",))
            + _missing_statuses(gap_statuses, ("full_compiled_adjoint_trainer_replacement",)),
        ),
        _row(
            "final_completion_promotion",
            "proved" if inputs_current and final_gap_shape_ok else "missing",
            "The final audit consumes current, verified non-completion artifacts and promotes their only remaining full_goal_completion row into an authoritative completion claim.",
            ["goal_progress", "completion_gap"],
            []
            if inputs_current and final_gap_shape_ok
            else [
                "verify progress/gap artifacts against current inputs and reduce concrete gaps to zero with only full_goal_completion left open"
            ],
            current={
                "all_evidence_current_and_valid": inputs_current,
                "completion_gap_open_gap_ids": summary["completion_gap_open_gap_ids"],
                "concrete_gap_values": concrete_gap_values,
                "goal_progress_is_goal_complete_before_final_audit": summary[
                    "goal_progress_is_goal_complete_before_final_audit"
                ],
            },
        ),
    ]


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    evidence = report["evidence"]
    progress = evidence["goal_progress"]["summary"]
    gap = evidence["completion_gap"]["summary"]
    all_valid = all(
        isinstance(evidence.get(key), dict)
        and evidence[key].get("verifier_errors") == []
        and evidence[key].get("current_input_errors") == []
        for key in EVIDENCE_ORDER
    )
    concrete_gap_values = {key: int(gap[key]) for key in CONCRETE_GAP_KEYS}
    summary = {
        "all_evidence_current_and_valid": all_valid,
        "goal": report["goal"],
        "meta_goal": report["meta_goal"],
        "key_math": report["key_math"],
        "theory": report["theory"],
        "theory_plan": evidence["theory_plan"]["summary"],
        "progress_requirement_count": len(PROVEN_REQUIREMENT_IDS) + 1,
        "progress_proved_requirement_count": int(progress["proved_requirement_count"]),
        "progress_open_requirement_count": int(progress["open_requirement_count"]),
        "progress_failed_requirement_count": int(progress["failed_requirement_count"]),
        "goal_progress_is_goal_complete_before_final_audit": bool(progress["is_goal_complete"]),
        "progress_open_requirement_id": PROGRESS_OPEN_REQUIREMENT_ID,
        "progress_requirement_statuses": dict(evidence["goal_progress"]["requirement_statuses"]),
        "completion_gap_requirement_statuses": dict(evidence["completion_gap"]["requirement_statuses"]),
        "completion_gap_proved_requirement_count": int(gap["proved_requirement_count"]),
        "completion_gap_partial_requirement_count": int(gap["partial_requirement_count"]),
        "completion_gap_open_gap_ids": list(gap["open_gap_ids"]),
        "completion_gap_completion_ready": bool(gap["completion_ready"]),
        "completion_gap_does_not_prove_completion": bool(gap["does_not_prove_completion"]),
        "concrete_gap_values": concrete_gap_values,
        "shared_work_proxy": dict(gap["shared_work_proxy"]),
        "compiled_replacement": dict(gap["compiled_replacement"]),
        "timing_protocol": dict(gap["timing_protocol"]),
    }
    rows = objective_rows(summary)
    accepted = all(row["status"] == "proved" for row in rows)
    summary.update(
        {
            "objective_requirement_count": len(rows),
            "proved_objective_requirement_count": sum(1 for row in rows if row["status"] == "proved"),
            "missing_objective_requirement_count": sum(1 for row in rows if row["status"] != "proved"),
            "open_objective_requirement_ids": [row["id"] for row in rows if row["status"] != "proved"],
            "completion_ready": accepted,
            "does_not_prove_completion": not accepted,
            "final_goal_completion_accepted": accepted,
        }
    )
    return summary


def run_report(
    *,
    goal_progress_report: Path = DEFAULT_GOAL_PROGRESS_REPORT,
    completion_gap_report: Path = DEFAULT_COMPLETION_GAP_REPORT,
) -> dict[str, Any]:
    evidence = {
        "theory_plan": _theory_plan_artifact(),
        "goal_progress": _artifact(
            goal_progress_report,
            verify_projective_goal_progress_audit,
            verify_projective_goal_progress_current_acceptance,
        ),
        "completion_gap": _artifact(
            completion_gap_report,
            verify_projective_goal_completion_gap_report,
            verify_projective_goal_completion_gap_current_acceptance,
        ),
    }
    report = {
        "status": "in_progress",
        "benchmark": "star_uvt_projective_goal_final_completion_audit",
        "goal": "fast 2D rasters across time from 4D spacetime primitives",
        "meta_goal": "share projection/support/binning/visibility/backward work over time",
        "key_math": "UVT trace = pi_* Gamma^* world_primitive",
        "theory": "STAR UVT is one local gauge expression of a camera-ray bundle atlas",
        "evidence": evidence,
        "summary": {},
        "requirements": [],
    }
    report["summary"] = summarize(report)
    report["requirements"] = objective_rows(report["summary"])
    report["status"] = "complete" if report["summary"]["final_goal_completion_accepted"] else "in_progress"
    return report


def verify_projective_goal_final_completion_audit(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("benchmark") != "star_uvt_projective_goal_final_completion_audit":
        errors.append("benchmark must be star_uvt_projective_goal_final_completion_audit")
    for key, phrase in (
        ("goal", "fast 2D rasters across time from 4D spacetime primitives"),
        ("meta_goal", "share projection/support/binning/visibility/backward work over time"),
        ("key_math", "UVT trace = pi_* Gamma^* world_primitive"),
        ("theory", "camera-ray bundle atlas"),
    ):
        value = report.get(key)
        if not isinstance(value, str) or phrase not in value:
            errors.append(f"{key} must preserve phrase {phrase!r}")
    evidence = report.get("evidence")
    requirements = report.get("requirements")
    summary = report.get("summary")
    if not isinstance(evidence, dict):
        errors.append("evidence must be an object")
        return errors
    if not isinstance(requirements, list):
        errors.append("requirements must be a list")
        return errors
    if not isinstance(summary, dict):
        errors.append("summary must be an object")
        return errors
    for key in EVIDENCE_ORDER:
        row = evidence.get(key)
        if not isinstance(row, dict):
            errors.append(f"evidence {key} must be an object")
            continue
        if row.get("verifier_errors"):
            errors.append(f"evidence {key} verifier failed: {row.get('verifier_errors')}")
        if row.get("current_input_errors"):
            errors.append(f"evidence {key} current-input acceptance failed: {row.get('current_input_errors')}")
        if not isinstance(row.get("summary"), dict):
            errors.append(f"evidence {key} summary must be an object")
    try:
        expected_summary = summarize(report)
        for key, value in expected_summary.items():
            if summary.get(key) != value:
                errors.append(f"summary {key} drifted: expected {value!r}, got {summary.get(key)!r}")
        expected_requirements = objective_rows(summary)
        if requirements != expected_requirements:
            errors.append("requirements drifted from recomputed objective rows")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"summary/requirements could not be recomputed: {exc}")
    row_ids = {row.get("id") for row in requirements if isinstance(row, dict)}
    for required_id in (
        "theory_plan_memory_contract",
        "fiber_bundle_trace_math_and_derivatives",
        "revolving_camera_family_and_visibility_atlas",
        "metal_forward_backward_renderer_path",
        "visibility_exposure_rolling_fallback_contract",
        "sublinear_world_side_work_and_bandwidth",
        "broad_real_video_renderer_acceptance",
        "compiled_adjoint_training_replacement",
        "final_completion_promotion",
    ):
        if required_id not in row_ids:
            errors.append(f"missing objective row {required_id}")
    if report.get("status") != "complete":
        errors.append("status must be complete for the final completion audit")
    if summary.get("final_goal_completion_accepted") is not True:
        errors.append("final_goal_completion_accepted must be true")
    if summary.get("completion_ready") is not True:
        errors.append("completion_ready must be true")
    if summary.get("does_not_prove_completion") is not False:
        errors.append("does_not_prove_completion must be false")
    if summary.get("missing_objective_requirement_count") != 0:
        errors.append("missing_objective_requirement_count must be zero")
    if summary.get("open_objective_requirement_ids") != []:
        errors.append("open_objective_requirement_ids must be empty")
    if summary.get("goal_progress_is_goal_complete_before_final_audit") is not False:
        errors.append("goal-progress input must remain pre-final and non-complete")
    if summary.get("completion_gap_open_gap_ids") != ["full_goal_completion"]:
        errors.append("completion gap must have only full_goal_completion open before final promotion")
    concrete = summary.get("concrete_gap_values")
    if not isinstance(concrete, dict) or any(value != 0 for value in concrete.values()):
        errors.append("all concrete completion-gap counters must be zero")
    theory = summary.get("theory_plan")
    if not isinstance(theory, dict):
        errors.append("theory_plan summary must be an object")
    else:
        if theory.get("required_subfolder_count") != 10:
            errors.append("theory plan must keep ten required subfolders")
        if theory.get("required_subfolders_present") is not True:
            errors.append("all required theory subfolders must be present")
        if theory.get("at_most_ten_numbered_theory_subfolders") is not True:
            errors.append("theory plan must not grow beyond ten numbered subfolders")
        if theory.get("goal_meta_key_math_exists") is not True:
            errors.append("GOAL_META_KEY_MATH.md must exist")
    return errors


def assert_projective_goal_final_completion_audit(report: dict[str, Any]) -> None:
    errors = verify_projective_goal_final_completion_audit(report)
    if errors:
        raise AssertionError("projective goal final completion audit failed:\n- " + "\n- ".join(errors))


def _compare_current_value(
    saved: Any,
    current: Any,
    label: str,
    errors: list[str],
    *,
    atol: float = 1.0e-9,
) -> None:
    if isinstance(current, dict):
        if not isinstance(saved, dict):
            errors.append(f"saved final audit differs from current inputs at {label}: expected object")
            return
        for key, current_value in current.items():
            _compare_current_value(saved.get(key), current_value, f"{label}.{key}" if label else key, errors, atol=atol)
        return
    if isinstance(current, list):
        if not isinstance(saved, list) or len(saved) != len(current):
            errors.append(
                f"saved final audit differs from current inputs at {label}: expected list length {len(current)}"
            )
            return
        for idx, (saved_value, current_value) in enumerate(zip(saved, current, strict=True)):
            _compare_current_value(saved_value, current_value, f"{label}[{idx}]", errors, atol=atol)
        return
    if isinstance(current, float):
        if not isinstance(saved, int | float) or abs(float(saved) - current) > atol:
            errors.append(
                f"saved final audit differs from current inputs at {label}: expected {current!r}, got {saved!r}"
            )
        return
    if saved != current:
        errors.append(f"saved final audit differs from current inputs at {label}: expected {current!r}, got {saved!r}")


def verify_projective_goal_final_completion_current_acceptance(
    saved_report: dict[str, Any],
    *,
    current_report: dict[str, Any] | None = None,
) -> list[str]:
    errors = [f"saved report: {error}" for error in verify_projective_goal_final_completion_audit(saved_report)]
    current = run_report() if current_report is None else current_report
    current_errors = verify_projective_goal_final_completion_audit(current)
    if current_errors:
        errors.extend(f"current inputs: {error}" for error in current_errors)
        return errors
    for key in (
        "status",
        "benchmark",
        "goal",
        "meta_goal",
        "key_math",
        "theory",
        "evidence",
        "summary",
        "requirements",
    ):
        _compare_current_value(saved_report.get(key), current.get(key), key, errors)
    return errors


def assert_projective_goal_final_completion_current_acceptance(
    saved_report: dict[str, Any],
    *,
    current_report: dict[str, Any] | None = None,
) -> None:
    errors = verify_projective_goal_final_completion_current_acceptance(saved_report, current_report=current_report)
    if errors:
        raise AssertionError(
            "projective goal final completion current-input acceptance failed:\n- " + "\n- ".join(errors)
        )


def write_report(report: dict[str, Any], out_dir: Path = DEFAULT_OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_report_json(out_dir / "summary.json", report)
    lines = [
        "# STAR UVT Projective Goal Final Completion Audit",
        "",
        "This is the final requirement-level audit for the active 4D spacetime primitive / camera-ray bundle objective.",
        "",
        "## Summary",
        "",
        f"- final goal completion accepted: {report['summary']['final_goal_completion_accepted']}",
        f"- objective requirements: {report['summary']['objective_requirement_count']}",
        f"- proved objective requirements: {report['summary']['proved_objective_requirement_count']}",
        f"- open objective requirement ids: {', '.join(report['summary']['open_objective_requirement_ids']) or '(none)'}",
        f"- completion ready: {report['summary']['completion_ready']}",
        f"- concrete completion-gap counters: {report['summary']['concrete_gap_values']}",
        "",
        "## Requirements",
        "",
    ]
    for row in report["requirements"]:
        lines.append(f"- `{row['id']}`: {row['status']} - {row['statement']}")
    write_report_text(out_dir / "summary.md", "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--goal-progress-report", type=Path, default=DEFAULT_GOAL_PROGRESS_REPORT)
    parser.add_argument("--completion-gap-report", type=Path, default=DEFAULT_COMPLETION_GAP_REPORT)
    parser.add_argument("--verify-report", type=Path)
    parser.add_argument(
        "--verify-current-inputs",
        action="store_true",
        help="also require the saved final completion audit to match a fresh report from current default inputs",
    )
    args = parser.parse_args()
    if args.verify_report is not None:
        report = _load_json(args.verify_report)
        if args.verify_current_inputs:
            assert_projective_goal_final_completion_current_acceptance(report)
            print(f"verified {args.verify_report} against current inputs")
        else:
            assert_projective_goal_final_completion_audit(report)
            print(f"verified {args.verify_report}")
        return
    if args.verify_current_inputs:
        report = _load_json(args.out_dir / "summary.json")
        assert_projective_goal_final_completion_current_acceptance(report)
        print(f"verified {args.out_dir / 'summary.json'} against current inputs")
        return
    report = run_report(
        goal_progress_report=args.goal_progress_report,
        completion_gap_report=args.completion_gap_report,
    )
    assert_projective_goal_final_completion_audit(report)
    write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
