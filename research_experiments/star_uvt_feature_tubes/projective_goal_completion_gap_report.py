from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for path in (ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from projective_goal_progress_audit import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_GOAL_PROGRESS_OUT_DIR,
    verify_projective_goal_progress_current_acceptance,
    verify_projective_goal_progress_audit,
)
from projective_real_video_acceptance_envelope_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_ACCEPTANCE_ENVELOPE_OUT_DIR,
    verify_real_video_acceptance_envelope_report,
)
from projective_real_video_broad10_quality_tether_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_BROAD10_QUALITY_OUT_DIR,
    verify_real_video_broad10_quality_tether_report,
)
from projective_real_video_multiscene_media_tether_report import (  # noqa: E402
    verify_real_video_multiscene_media_tether_report,
)
from projective_real_video_multiscene_trainer_matrix import (  # noqa: E402
    verify_real_video_multiscene_trainer_matrix_report,
)
from projective_real_video_compiled_adjoint_replacement_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_COMPILED_ADJOINT_REPLACEMENT_OUT_DIR,
    verify_real_video_compiled_adjoint_replacement_report,
)
from projective_real_video_timing_variance_envelope_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_TIMING_VARIANCE_ENVELOPE_OUT_DIR,
    verify_real_video_timing_variance_envelope_report,
)
from projective_real_video_timing_protocol_acceptance_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_TIMING_PROTOCOL_ACCEPTANCE_OUT_DIR,
    verify_real_video_timing_protocol_acceptance_report,
)
from projective_shared_work_goal_audit import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_SHARED_WORK_OUT_DIR,
    verify_shared_work_goal_audit,
)
from report_artifacts import write_report_json, write_report_text  # noqa: E402


DEFAULT_OUT_DIR = ROOT / "outputs" / "benchmarks" / "2026-05-25_star_uvt_projective_goal_completion_gap"
DEFAULT_GOAL_PROGRESS_REPORT = DEFAULT_GOAL_PROGRESS_OUT_DIR / "summary.json"
DEFAULT_ACCEPTANCE_ENVELOPE_REPORT = DEFAULT_ACCEPTANCE_ENVELOPE_OUT_DIR / "summary.json"
DEFAULT_BROAD10_TRAINER_REPORT = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix_broad10"
    / "summary.json"
)
DEFAULT_BROAD10_QUALITY_REPORT = DEFAULT_BROAD10_QUALITY_OUT_DIR / "summary.json"
DEFAULT_BROAD10_MEDIA_REPORT = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_broad10_media_tether"
    / "summary.json"
)
DEFAULT_TIMING_VARIANCE_ENVELOPE_REPORT = DEFAULT_TIMING_VARIANCE_ENVELOPE_OUT_DIR / "summary.json"
DEFAULT_TIMING_PROTOCOL_ACCEPTANCE_REPORT = DEFAULT_TIMING_PROTOCOL_ACCEPTANCE_OUT_DIR / "summary.json"
DEFAULT_COMPILED_ADJOINT_REPLACEMENT_REPORT = DEFAULT_COMPILED_ADJOINT_REPLACEMENT_OUT_DIR / "summary.json"
DEFAULT_SHARED_WORK_REPORT = DEFAULT_SHARED_WORK_OUT_DIR / "summary.json"

EVIDENCE_ORDER = (
    "goal_progress",
    "real_video_acceptance_envelope",
    "real_video_broad10_trainer_matrix",
    "real_video_broad10_quality_tether",
    "real_video_broad10_media_tether",
    "real_video_timing_variance_envelope",
    "real_video_timing_protocol_acceptance",
    "real_video_compiled_adjoint_replacement",
    "shared_work",
)

COMPLETION_TARGETS = {
    "broad_quality_min_distinct_sources": 10,
    "broad_media_min_distinct_sources": 10,
    "broad_quality_min_frame_count_count": 4,
    "broad_quality_required_strict_timing_failures": 0,
    "compiled_trainer_min_distinct_sources": 10,
    "compiled_trainer_min_frame_count_count": 4,
    "fresh_process_required_median_ratio_max": 1.0,
    "shared_work_required_orbit_payload_growth_ratio_max": 0.20,
    "shared_work_required_trained_interval_growth_ratio_max": 0.25,
    "shared_work_required_backward_ratio_max": 0.25,
}

Verifier = Callable[[dict[str, Any]], list[str]]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact(path: Path, verifier: Verifier) -> dict[str, Any]:
    report = _load_json(path)
    return {
        "path": str(path),
        "benchmark": report.get("benchmark"),
        "status": report.get("status"),
        "verifier_errors": verifier(report),
        "summary": report.get("summary", {}),
    }


def _goal_progress_artifact(path: Path) -> dict[str, Any]:
    report = _load_json(path)
    return {
        "path": str(path),
        "benchmark": report.get("benchmark"),
        "status": report.get("status"),
        "verifier_errors": (
            verify_projective_goal_progress_audit(report)
            + verify_projective_goal_progress_current_acceptance(report)
        ),
        "summary": report.get("summary", {}),
    }


def _summary(report: dict[str, Any], key: str) -> dict[str, Any]:
    return report["evidence"][key]["summary"]


def _finite_float(value: Any, label: str, errors: list[str]) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        errors.append(f"{label} must be finite, got {value!r}")
        return 0.0
    return float(value)


def _finite_int(value: Any, label: str, errors: list[str]) -> int:
    if not isinstance(value, int):
        errors.append(f"{label} must be an integer, got {value!r}")
        return 0
    return int(value)


def _passes_ratio(summary: dict[str, Any], key: str, threshold: float) -> bool:
    value = summary.get(key)
    return isinstance(value, int | float) and math.isfinite(float(value)) and float(value) <= threshold


def completion_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    targets = summary["completion_targets"]
    current = summary["current_evidence"]
    shared = summary["shared_work_proxy"]
    timing = summary["timing_variance"]
    timing_protocol = summary["timing_protocol"]
    compiled_replacement = summary["compiled_replacement"]
    broad_quality_missing = []
    if current["broad_media_distinct_sources"] < targets["broad_media_min_distinct_sources"]:
        broad_quality_missing.append("scale media acceptance beyond five distinct source videos")
    if current["real_video_frame_count_count"] < targets["broad_quality_min_frame_count_count"]:
        broad_quality_missing.append("include at least one broader real-video frame-scaling point beyond the current three frame counts")
    if not timing_protocol["final_timing_protocol_accepted"]:
        broad_quality_missing.append(
            "turn strict timing failures into a clean timing pass or explicitly replace the strict gate with a stronger accepted timing protocol"
        )
    broad_quality_proved = (
        current["broad_quality_distinct_sources"] >= targets["broad_quality_min_distinct_sources"]
        and current["broad_media_distinct_sources"] >= targets["broad_media_min_distinct_sources"]
        and current["real_video_frame_count_count"] >= targets["broad_quality_min_frame_count_count"]
        and timing_protocol["final_timing_protocol_accepted"]
    )
    return [
        {
            "id": "formal_goal_memory_and_audit",
            "status": "proved",
            "statement": "The active goal/meta/key-math/theory anchors are preserved, and the current goal-progress audit is explicitly non-complete.",
            "evidence": ["goal_progress"],
            "missing": [],
        },
        {
            "id": "sublinear_world_side_work_proxy",
            "status": "proved" if shared["passes_proxy_thresholds"] else "partial",
            "statement": "Current orbit/trained artifacts show sublinear world-side payload, interval-entry, forward, and backward proxy ratios versus per-frame replay.",
            "evidence": ["shared_work"],
            "missing": [] if shared["passes_proxy_thresholds"] else ["refresh shared-work proxy artifacts below ratio thresholds"],
        },
        {
            "id": "broad_real_scene_quality_acceptance",
            "status": "proved" if broad_quality_proved else "partial",
            "statement": "The renderer needs broad real-scene quality acceptance, not only focused source-distinct and five-source tethers.",
            "evidence": [
                "real_video_acceptance_envelope",
                "real_video_timing_variance_envelope",
                "real_video_timing_protocol_acceptance",
            ],
            "current": {
                "distinct_sources": current["broad_quality_distinct_sources"],
                "broad10_quality_distinct_sources": current["broad10_quality_distinct_sources"],
                "frame_count_count": current["real_video_frame_count_count"],
                "strict_timing_failures": timing["strict_failure_count"],
                "media_scene_count": current["media_scene_count"],
                "broad10_media_distinct_sources": current["broad10_media_distinct_sources"],
                "broad_media_distinct_sources": current["broad_media_distinct_sources"],
                "min_quality_psnr_gain": current["min_quality_psnr_gain"],
            },
            "target": {
                "distinct_sources": targets["broad_quality_min_distinct_sources"],
                "media_distinct_sources": targets["broad_media_min_distinct_sources"],
                "frame_count_count": targets["broad_quality_min_frame_count_count"],
                "strict_timing_failures": targets["broad_quality_required_strict_timing_failures"],
            },
            "missing": broad_quality_missing,
        },
        {
            "id": "full_compiled_adjoint_trainer_replacement",
            "status": "proved" if compiled_replacement["final_compiled_adjoint_replacement_accepted"] else "partial",
            "statement": "The trainer needs broad compiled-adjoint replacement evidence, not only local VJPs and focused real-video matrices tethered to cadence.",
            "evidence": [
                "real_video_compiled_adjoint_replacement",
                "real_video_acceptance_envelope",
                "shared_work",
            ],
            "current": {
                "distinct_sources": current["compiled_trainer_distinct_sources"],
                "broad10_trainer_distinct_sources": current["broad10_trainer_distinct_sources"],
                "broad10_trainer_row_count": current["broad10_trainer_row_count"],
                "frame_count_count": current["real_video_frame_count_count"],
                "compiled_replacement_case_count": compiled_replacement["case_payload_count"],
                "compiled_replacement_source_contract_checks_pass": compiled_replacement[
                    "source_contract_checks_pass"
                ],
                "compiled_replacement_gap": compiled_replacement["compiled_trainer_replacement_gap"],
                "goal_progress_proved_requirements": current["proved_requirement_count"],
            },
            "target": {
                "distinct_sources": targets["compiled_trainer_min_distinct_sources"],
                "frame_count_count": targets["compiled_trainer_min_frame_count_count"],
                "replacement_benchmark_required": True,
            },
            "missing": []
            if compiled_replacement["final_compiled_adjoint_replacement_accepted"]
            else [
                "run a broad full-trainer replacement benchmark where compiled adjoints are the main path",
                "show optimizer-step quality and media outputs without relying on narrow cadence tethers as the only evidence",
                "extend the broad10 trainer source coverage into the same frame-count, quality, and media envelope as the final acceptance gate",
            ],
        },
        {
            "id": "timing_acceptance_protocol",
            "status": "proved" if timing_protocol["final_timing_protocol_accepted"] else "partial",
            "statement": "Fresh-process median timing with warmup discard is the accepted timing protocol; strict warm-state misses remain diagnostic caveats when cache/support/workload invariants are clean.",
            "evidence": ["real_video_timing_variance_envelope", "real_video_timing_protocol_acceptance"],
            "current": {
                "fresh_process_status": timing["fresh_process_status"],
                "fresh_process_median_no_first_ratio": timing["fresh_process_median_no_first_ratio"],
                "strict_failure_count": timing["strict_failure_count"],
                "final_timing_protocol_accepted": timing_protocol["final_timing_protocol_accepted"],
                "timing_acceptance_gap": timing_protocol["timing_acceptance_gap"],
                "workload_explains_render_forward_miss_count": timing[
                    "workload_explains_render_forward_miss_count"
                ],
            },
            "missing": []
            if timing_protocol["final_timing_protocol_accepted"]
            else [
                "either make the strict timing gate pass on the broad acceptance set or promote the fresh-process median protocol with an explicit final benchmark contract",
            ],
        },
        {
            "id": "full_goal_completion",
            "status": "partial",
            "statement": "The top-level research goal still needs one final completion audit that promotes the non-completion evidence stack into an authoritative done state.",
            "evidence": [
                "goal_progress",
                "real_video_compiled_adjoint_replacement",
                "real_video_acceptance_envelope",
                "real_video_timing_protocol_acceptance",
                "shared_work",
            ],
            "current": {
                "goal_progress_is_goal_complete": current["is_goal_complete"],
                "goal_progress_open_requirement_count": current["open_requirement_count"],
                "compiled_replacement_does_not_prove_completion": compiled_replacement[
                    "does_not_prove_completion"
                ],
                "acceptance_does_not_prove_completion": current["acceptance_does_not_prove_completion"],
            },
            "target": {
                "goal_progress_is_goal_complete": True,
                "completion_ready": True,
                "does_not_prove_completion": False,
            },
            "missing": [
                "run a final current-state completion audit that allows goal-progress to set is_goal_complete=true",
                "replace the non-completion top-level artifacts with one verified completion artifact once the full objective is satisfied",
            ],
        },
    ]


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    goal = _summary(report, "goal_progress")
    acceptance = _summary(report, "real_video_acceptance_envelope")
    broad10_trainer = _summary(report, "real_video_broad10_trainer_matrix")
    broad10_quality = _summary(report, "real_video_broad10_quality_tether")
    broad10_media = _summary(report, "real_video_broad10_media_tether")
    timing = _summary(report, "real_video_timing_variance_envelope")
    timing_protocol_report = _summary(report, "real_video_timing_protocol_acceptance")
    compiled_replacement_report = _summary(report, "real_video_compiled_adjoint_replacement")
    shared = _summary(report, "shared_work")
    all_underlying = all(
        isinstance(report["evidence"].get(key), dict)
        and report["evidence"][key].get("verifier_errors") == []
        and isinstance(report["evidence"][key].get("summary"), dict)
        for key in EVIDENCE_ORDER
    )
    targets = dict(COMPLETION_TARGETS)
    current_evidence = {
        "proved_requirement_count": int(goal["proved_requirement_count"]),
        "open_requirement_count": int(goal["open_requirement_count"]),
        "is_goal_complete": bool(goal["is_goal_complete"]),
        "broad10_quality_distinct_sources": int(broad10_quality["distinct_youtube_id_count"]),
        "broad10_media_distinct_sources": int(broad10_media["distinct_youtube_id_count"]),
        "broad10_media_pair_count": int(broad10_media["pair_count"]),
        "broad_quality_distinct_sources": max(
            int(acceptance["quality_scene_count"]),
            int(broad10_quality["distinct_youtube_id_count"]),
        ),
        "quality_scene_count": int(acceptance["quality_scene_count"]),
        "media_scene_count": int(acceptance["media_scene_count"]),
        "broad_media_distinct_sources": max(
            int(acceptance.get("broad_media_distinct_youtube_id_count", acceptance["media_scene_count"])),
            int(broad10_media["distinct_youtube_id_count"]),
        ),
        "real_video_frame_count_count": int(
            acceptance.get("broad_frame_count_count", acceptance["frame_scaling_frame_count_count"])
        ),
        "frame_count_breadth_frame_count_count": int(
            acceptance.get("frame_count_breadth_frame_count_count", acceptance["frame_scaling_frame_count_count"])
        ),
        "broad10_trainer_distinct_sources": int(broad10_trainer["distinct_youtube_id_count"]),
        "broad10_trainer_row_count": int(broad10_trainer["row_count"]),
        "broad10_trainer_max_support_rebins": int(broad10_trainer["max_measured_support_rebins"]),
        "broad10_trainer_max_stale_refreshes": int(broad10_trainer["max_measured_stale_refreshes"]),
        "compiled_trainer_distinct_sources": max(
            int(acceptance["functional_distinct_youtube_id_count"]),
            int(broad10_trainer["distinct_youtube_id_count"]),
            int(compiled_replacement_report["broad10_trainer_distinct_youtube_id_count"]),
        ),
        "min_quality_psnr_gain": float(acceptance["min_quality_psnr_gain"]),
        "max_support_rebins": int(acceptance["max_support_rebins"]),
        "max_stale_refreshes": int(acceptance["max_stale_refreshes"]),
        "acceptance_does_not_prove_completion": bool(acceptance["does_not_prove_completion"]),
    }
    timing_variance = {
        "strict_failure_count": int(timing["strict_failure_count"]),
        "strict_timing_win_claimed": bool(timing["strict_timing_win_claimed"]),
        "fresh_process_status": str(timing["fresh_process_timing_acceptance_status"]),
        "fresh_process_median_no_first_ratio": float(timing["fresh_process_median_no_first_ratio"]),
        "fresh_process_median_projective_total_ratio": float(
            timing["fresh_process_median_projective_total_ratio"]
        ),
        "fresh_process_median_feature_state_update_ratio": float(
            timing["fresh_process_median_feature_state_update_ratio"]
        ),
        "workload_explains_render_forward_miss_count": int(
            timing["workload_explains_render_forward_miss_count"]
        ),
        "does_not_prove_completion": bool(timing["does_not_prove_completion"]),
    }
    timing_protocol = {
        "final_timing_protocol_accepted": bool(timing_protocol_report["final_timing_protocol_accepted"]),
        "protocol_name": str(timing_protocol_report["protocol_name"]),
        "timing_acceptance_gap": int(timing_protocol_report["timing_acceptance_gap"]),
        "fresh_process_median_no_first_ratio": float(
            timing_protocol_report["fresh_process_median_no_first_ratio"]
        ),
        "fresh_process_median_projective_total_ratio": float(
            timing_protocol_report["fresh_process_median_projective_total_ratio"]
        ),
        "fresh_process_median_feature_state_update_ratio": float(
            timing_protocol_report["fresh_process_median_feature_state_update_ratio"]
        ),
        "strict_warm_state_failure_count": int(timing_protocol_report["strict_warm_state_failure_count"]),
        "strict_warm_state_failures_demoted_to_caveat": bool(
            timing_protocol_report["strict_warm_state_failures_demoted_to_caveat"]
        ),
        "broad_real_context_passes": bool(timing_protocol_report["broad_real_context_passes"]),
        "frame_count_breadth_passes": bool(timing_protocol_report["frame_count_breadth_passes"]),
    }
    compiled_replacement = {
        "final_compiled_adjoint_replacement_accepted": bool(
            compiled_replacement_report["final_compiled_adjoint_replacement_accepted"]
        ),
        "compiled_trainer_replacement_gap": int(compiled_replacement_report["compiled_trainer_replacement_gap"]),
        "source_contract_checks_pass": bool(compiled_replacement_report["source_contract_checks_pass"]),
        "broad_context_passes": bool(compiled_replacement_report["broad_context_passes"]),
        "clean_cache_and_support": bool(compiled_replacement_report["clean_cache_and_support"]),
        "all_cases_projective_interval_main_path": bool(
            compiled_replacement_report["all_cases_projective_interval_main_path"]
        ),
        "all_cases_gradient_flags_present": bool(
            compiled_replacement_report["all_cases_gradient_flags_present"]
        ),
        "measured_cache_reuse_ok": bool(compiled_replacement_report["measured_cache_reuse_ok"]),
        "case_payload_count": int(compiled_replacement_report["case_payload_count"]),
        "does_not_prove_completion": bool(compiled_replacement_report["does_not_prove_completion"]),
    }
    shared_work_proxy = {
        "orbit_payload_growth_ratio": float(shared["orbit_payload_growth_ratio"]),
        "trained_shared_to_replay_interval_growth_ratio": float(
            shared["trained_shared_to_replay_interval_growth_ratio"]
        ),
        "max_trained_final_backward_ms_ratio": float(shared["max_trained_final_backward_ms_ratio"]),
    }
    shared_work_proxy["passes_proxy_thresholds"] = (
        shared_work_proxy["orbit_payload_growth_ratio"]
        <= targets["shared_work_required_orbit_payload_growth_ratio_max"]
        and shared_work_proxy["trained_shared_to_replay_interval_growth_ratio"]
        <= targets["shared_work_required_trained_interval_growth_ratio_max"]
        and shared_work_proxy["max_trained_final_backward_ms_ratio"]
        <= targets["shared_work_required_backward_ratio_max"]
    )
    summary = {
        "all_underlying_verifiers_pass": all_underlying,
        "completion_targets": targets,
        "current_evidence": current_evidence,
        "timing_variance": timing_variance,
        "timing_protocol": timing_protocol,
        "compiled_replacement": compiled_replacement,
        "shared_work_proxy": shared_work_proxy,
    }
    rows = completion_rows(summary)
    summary.update(
        {
            "completion_ready": False,
            "does_not_prove_completion": True,
            "requirement_count": len(rows),
            "proved_requirement_count": sum(1 for row in rows if row["status"] == "proved"),
            "partial_requirement_count": sum(1 for row in rows if row["status"] == "partial"),
            "missing_requirement_count": sum(1 for row in rows if row["status"] == "missing"),
            "open_gap_ids": [row["id"] for row in rows if row["status"] != "proved"],
            "broad_quality_source_gap": max(
                0,
                targets["broad_quality_min_distinct_sources"]
                - current_evidence["broad_quality_distinct_sources"],
            ),
            "broad_quality_frame_count_gap": max(
                0,
                targets["broad_quality_min_frame_count_count"]
                - current_evidence["real_video_frame_count_count"],
            ),
            "broad_media_source_gap": max(
                0,
                targets["broad_media_min_distinct_sources"] - current_evidence["broad_media_distinct_sources"],
            ),
            "strict_timing_failure_gap": max(
                0,
                0
                if timing_protocol["final_timing_protocol_accepted"]
                else timing_variance["strict_failure_count"]
                - targets["broad_quality_required_strict_timing_failures"],
            ),
            "timing_acceptance_gap": timing_protocol["timing_acceptance_gap"],
            "compiled_trainer_source_gap": max(
                0,
                targets["compiled_trainer_min_distinct_sources"]
                - current_evidence["compiled_trainer_distinct_sources"],
            ),
            "compiled_trainer_replacement_gap": compiled_replacement["compiled_trainer_replacement_gap"],
        }
    )
    return summary


def run_report(
    *,
    goal_progress_report: Path = DEFAULT_GOAL_PROGRESS_REPORT,
    acceptance_envelope_report: Path = DEFAULT_ACCEPTANCE_ENVELOPE_REPORT,
    broad10_trainer_report: Path = DEFAULT_BROAD10_TRAINER_REPORT,
    broad10_quality_report: Path = DEFAULT_BROAD10_QUALITY_REPORT,
    broad10_media_report: Path = DEFAULT_BROAD10_MEDIA_REPORT,
    timing_variance_envelope_report: Path = DEFAULT_TIMING_VARIANCE_ENVELOPE_REPORT,
    timing_protocol_acceptance_report: Path = DEFAULT_TIMING_PROTOCOL_ACCEPTANCE_REPORT,
    compiled_adjoint_replacement_report: Path = DEFAULT_COMPILED_ADJOINT_REPLACEMENT_REPORT,
    shared_work_report: Path = DEFAULT_SHARED_WORK_REPORT,
) -> dict[str, Any]:
    evidence = {
        "goal_progress": _goal_progress_artifact(goal_progress_report),
        "real_video_acceptance_envelope": _artifact(
            acceptance_envelope_report,
            verify_real_video_acceptance_envelope_report,
        ),
        "real_video_broad10_trainer_matrix": _artifact(
            broad10_trainer_report,
            verify_real_video_multiscene_trainer_matrix_report,
        ),
        "real_video_broad10_quality_tether": _artifact(
            broad10_quality_report,
            verify_real_video_broad10_quality_tether_report,
        ),
        "real_video_broad10_media_tether": _artifact(
            broad10_media_report,
            verify_real_video_multiscene_media_tether_report,
        ),
        "real_video_timing_variance_envelope": _artifact(
            timing_variance_envelope_report,
            verify_real_video_timing_variance_envelope_report,
        ),
        "real_video_timing_protocol_acceptance": _artifact(
            timing_protocol_acceptance_report,
            verify_real_video_timing_protocol_acceptance_report,
        ),
        "real_video_compiled_adjoint_replacement": _artifact(
            compiled_adjoint_replacement_report,
            verify_real_video_compiled_adjoint_replacement_report,
        ),
        "shared_work": _artifact(shared_work_report, verify_shared_work_goal_audit),
    }
    report = {
        "status": "in_progress",
        "benchmark": "star_uvt_projective_goal_completion_gap",
        "goal": "fast 2D rasters across time from 4D spacetime primitives",
        "meta_goal": "share projection/support/binning/visibility/backward work over time",
        "key_math": "UVT trace = pi_* Gamma^* world_primitive",
        "theory": "STAR UVT is one local gauge expression of a camera-ray bundle atlas",
        "evidence": evidence,
        "summary": {},
        "requirements": [],
    }
    report["summary"] = summarize(report)
    report["requirements"] = completion_rows(report["summary"])
    return report


def verify_projective_goal_completion_gap_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("benchmark") != "star_uvt_projective_goal_completion_gap":
        errors.append("benchmark must be star_uvt_projective_goal_completion_gap")
    if report.get("status") != "in_progress":
        errors.append("status must be in_progress")
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
        if not isinstance(row.get("summary"), dict):
            errors.append(f"evidence {key} summary must be an object")
    try:
        expected = summarize(report)
        for key, value in expected.items():
            if summary.get(key) != value:
                errors.append(f"summary {key} drifted: expected {value!r}, got {summary.get(key)!r}")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"summary could not be recomputed: {exc}")
    by_id = {row.get("id"): row for row in requirements if isinstance(row, dict)}
    for required_id in (
        "formal_goal_memory_and_audit",
        "sublinear_world_side_work_proxy",
        "broad_real_scene_quality_acceptance",
        "full_compiled_adjoint_trainer_replacement",
        "timing_acceptance_protocol",
        "full_goal_completion",
    ):
        if required_id not in by_id:
            errors.append(f"missing completion row {required_id}")
    if by_id.get("full_compiled_adjoint_trainer_replacement", {}).get("status") == "proved":
        compiled = summary.get("compiled_replacement", {})
        if not isinstance(compiled, dict) or compiled.get("final_compiled_adjoint_replacement_accepted") is not True:
            errors.append("full compiled-adjoint trainer replacement can only be proved by accepted replacement evidence")
    if by_id.get("full_goal_completion", {}).get("status") != "partial":
        errors.append("full goal completion must remain partial until a final completion audit exists")
    try:
        expected_requirements = completion_rows(summary)
        if requirements != expected_requirements:
            errors.append("requirements drifted from recomputed completion rows")
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"requirements could not be recomputed: {exc}")
    if summary.get("completion_ready") is not False:
        errors.append("completion_ready must be false")
    if summary.get("does_not_prove_completion") is not True:
        errors.append("does_not_prove_completion must be true")
    targets = summary.get("completion_targets", {})
    current = summary.get("current_evidence", {})
    timing = summary.get("timing_variance", {})
    timing_protocol = summary.get("timing_protocol", {})
    compiled_replacement = summary.get("compiled_replacement", {})
    shared = summary.get("shared_work_proxy", {})
    if isinstance(targets, dict) and isinstance(current, dict):
        if _finite_int(targets.get("broad_quality_min_distinct_sources"), "target broad sources", errors) < 10:
            errors.append("broad quality source target must stay at least 10")
        if _finite_int(targets.get("broad_media_min_distinct_sources"), "target broad media sources", errors) < 10:
            errors.append("broad media source target must stay at least 10")
        if _finite_int(targets.get("compiled_trainer_min_distinct_sources"), "target trainer sources", errors) < 10:
            errors.append("compiled trainer source target must stay at least 10")
        if _finite_int(targets.get("broad_quality_min_frame_count_count"), "target frame-count count", errors) < 4:
            errors.append("broad quality frame-count target must stay at least 4")
        if current.get("is_goal_complete") is not False:
            errors.append("current goal-progress summary must keep is_goal_complete false")
        if _finite_int(current.get("broad10_trainer_distinct_sources"), "broad10 trainer sources", errors) < 10:
            errors.append("broad10 trainer evidence must cover at least 10 distinct sources")
        if _finite_int(current.get("broad10_trainer_row_count"), "broad10 trainer row count", errors) < 20:
            errors.append("broad10 trainer evidence must cover at least 20 cadence/measured rows")
        if _finite_int(current.get("broad10_trainer_max_support_rebins"), "broad10 support rebins", errors) != 0:
            errors.append("broad10 trainer evidence must keep support rebins at zero")
        if _finite_int(current.get("broad10_trainer_max_stale_refreshes"), "broad10 stale refreshes", errors) != 0:
            errors.append("broad10 trainer evidence must keep stale refreshes at zero")
        if _finite_int(current.get("broad10_quality_distinct_sources"), "broad10 quality sources", errors) < 10:
            errors.append("broad10 quality evidence must cover at least 10 distinct sources")
        if _finite_int(current.get("broad10_media_distinct_sources"), "broad10 media sources", errors) < 10:
            errors.append("broad10 media evidence must cover at least 10 distinct sources")
        if _finite_int(current.get("broad10_media_pair_count"), "broad10 media pair count", errors) < 10:
            errors.append("broad10 media evidence must cover at least 10 media pairs")
    if isinstance(timing, dict):
        if timing.get("fresh_process_status") != "pass":
            errors.append("fresh-process timing status must pass")
        if _finite_float(timing.get("fresh_process_median_no_first_ratio"), "fresh median no-first", errors) > 1.0:
            errors.append("fresh-process no-first median ratio must stay at or below 1")
        if _finite_int(timing.get("strict_failure_count"), "strict failure count", errors) <= 0:
            errors.append("gap report must preserve current strict timing failures as unresolved")
    if isinstance(timing_protocol, dict):
        if timing_protocol.get("final_timing_protocol_accepted") is not True:
            errors.append("final timing protocol must be accepted")
        if _finite_int(timing_protocol.get("timing_acceptance_gap"), "timing acceptance gap", errors) != 0:
            errors.append("timing acceptance gap must be zero")
        if timing_protocol.get("strict_warm_state_failures_demoted_to_caveat") is not True:
            errors.append("strict warm-state timing misses must be demoted to caveats by protocol")
    if isinstance(compiled_replacement, dict):
        if compiled_replacement.get("final_compiled_adjoint_replacement_accepted") is not True:
            errors.append("final compiled-adjoint replacement must be accepted")
        if _finite_int(
            compiled_replacement.get("compiled_trainer_replacement_gap"),
            "compiled replacement gap",
            errors,
        ) != 0:
            errors.append("compiled replacement gap must be zero")
        for key in (
            "source_contract_checks_pass",
            "broad_context_passes",
            "clean_cache_and_support",
            "all_cases_projective_interval_main_path",
            "all_cases_gradient_flags_present",
            "measured_cache_reuse_ok",
        ):
            if compiled_replacement.get(key) is not True:
                errors.append(f"compiled replacement {key} must be true")
        if _finite_int(compiled_replacement.get("case_payload_count"), "compiled case payload count", errors) < 20:
            errors.append("compiled replacement must include at least twenty case payloads")
    if isinstance(shared, dict) and shared.get("passes_proxy_thresholds") is not True:
        errors.append("shared-work proxy thresholds must pass")
    return errors


def assert_projective_goal_completion_gap_report(report: dict[str, Any]) -> None:
    errors = verify_projective_goal_completion_gap_report(report)
    if errors:
        raise AssertionError("projective goal completion gap report failed:\n- " + "\n- ".join(errors))


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
            errors.append(f"saved completion-gap report differs from current inputs at {label}: expected object")
            return
        for key, current_value in current.items():
            child_label = f"{label}.{key}" if label else str(key)
            _compare_current_value(saved.get(key), current_value, child_label, errors, atol=atol)
        return
    if isinstance(current, list):
        if not isinstance(saved, list) or len(saved) != len(current):
            errors.append(
                f"saved completion-gap report differs from current inputs at {label}: "
                f"expected list length {len(current)}, got {len(saved) if isinstance(saved, list) else type(saved).__name__}"
            )
            return
        for idx, (saved_value, current_value) in enumerate(zip(saved, current, strict=True)):
            _compare_current_value(saved_value, current_value, f"{label}[{idx}]", errors, atol=atol)
        return
    if isinstance(current, float):
        if not isinstance(saved, int | float) or abs(float(saved) - current) > atol:
            errors.append(
                f"saved completion-gap report differs from current inputs at {label}: "
                f"expected {current!r}, got {saved!r}"
            )
        return
    if saved != current:
        errors.append(
            f"saved completion-gap report differs from current inputs at {label}: "
            f"expected {current!r}, got {saved!r}"
        )


def verify_projective_goal_completion_gap_current_acceptance(
    saved_report: dict[str, Any],
    *,
    current_report: dict[str, Any] | None = None,
) -> list[str]:
    errors = [f"saved report: {error}" for error in verify_projective_goal_completion_gap_report(saved_report)]
    current = run_report() if current_report is None else current_report
    current_errors = verify_projective_goal_completion_gap_report(current)
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


def assert_projective_goal_completion_gap_current_acceptance(
    saved_report: dict[str, Any],
    *,
    current_report: dict[str, Any] | None = None,
) -> None:
    errors = verify_projective_goal_completion_gap_current_acceptance(
        saved_report,
        current_report=current_report,
    )
    if errors:
        raise AssertionError(
            "projective goal-completion gap current-input acceptance failed:\n- "
            + "\n- ".join(errors)
        )


def write_report(report: dict[str, Any], out_dir: Path = DEFAULT_OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_report_json(out_dir / "summary.json", report)
    lines = [
        "# STAR UVT Projective Goal Completion Gap",
        "",
        "This is not a completion claim. It keeps the active goal open while making the remaining evidence gaps concrete.",
        "",
        "## Summary",
        "",
        f"- proved rows: {report['summary']['proved_requirement_count']}",
        f"- partial rows: {report['summary']['partial_requirement_count']}",
        f"- open gap ids: {', '.join(report['summary']['open_gap_ids'])}",
        f"- broad quality source gap: {report['summary']['broad_quality_source_gap']}",
        f"- broad media source gap: {report['summary']['broad_media_source_gap']}",
        f"- broad quality frame-count gap: {report['summary']['broad_quality_frame_count_gap']}",
        f"- strict timing failure gap: {report['summary']['strict_timing_failure_gap']}",
        f"- timing acceptance gap: {report['summary']['timing_acceptance_gap']}",
        f"- compiled trainer source gap: {report['summary']['compiled_trainer_source_gap']}",
        f"- compiled trainer replacement gap: {report['summary']['compiled_trainer_replacement_gap']}",
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
    parser.add_argument("--acceptance-envelope-report", type=Path, default=DEFAULT_ACCEPTANCE_ENVELOPE_REPORT)
    parser.add_argument("--broad10-trainer-report", type=Path, default=DEFAULT_BROAD10_TRAINER_REPORT)
    parser.add_argument("--broad10-quality-report", type=Path, default=DEFAULT_BROAD10_QUALITY_REPORT)
    parser.add_argument("--broad10-media-report", type=Path, default=DEFAULT_BROAD10_MEDIA_REPORT)
    parser.add_argument(
        "--timing-variance-envelope-report",
        type=Path,
        default=DEFAULT_TIMING_VARIANCE_ENVELOPE_REPORT,
    )
    parser.add_argument(
        "--timing-protocol-acceptance-report",
        type=Path,
        default=DEFAULT_TIMING_PROTOCOL_ACCEPTANCE_REPORT,
    )
    parser.add_argument(
        "--compiled-adjoint-replacement-report",
        type=Path,
        default=DEFAULT_COMPILED_ADJOINT_REPLACEMENT_REPORT,
    )
    parser.add_argument("--shared-work-report", type=Path, default=DEFAULT_SHARED_WORK_REPORT)
    parser.add_argument("--verify-report", type=Path)
    parser.add_argument(
        "--verify-current-inputs",
        action="store_true",
        help="also require the saved completion-gap report to match a fresh report from current default inputs",
    )
    args = parser.parse_args()
    if args.verify_report is not None:
        report = _load_json(args.verify_report)
        if args.verify_current_inputs:
            assert_projective_goal_completion_gap_current_acceptance(report)
            print(f"verified {args.verify_report} against current inputs")
        else:
            assert_projective_goal_completion_gap_report(report)
            print(f"verified {args.verify_report}")
        return
    if args.verify_current_inputs:
        report = _load_json(args.out_dir / "summary.json")
        assert_projective_goal_completion_gap_current_acceptance(report)
        print(f"verified {args.out_dir / 'summary.json'} against current inputs")
        return
    report = run_report(
        goal_progress_report=args.goal_progress_report,
        acceptance_envelope_report=args.acceptance_envelope_report,
        broad10_trainer_report=args.broad10_trainer_report,
        broad10_quality_report=args.broad10_quality_report,
        broad10_media_report=args.broad10_media_report,
        timing_variance_envelope_report=args.timing_variance_envelope_report,
        timing_protocol_acceptance_report=args.timing_protocol_acceptance_report,
        compiled_adjoint_replacement_report=args.compiled_adjoint_replacement_report,
        shared_work_report=args.shared_work_report,
    )
    assert_projective_goal_completion_gap_report(report)
    write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
