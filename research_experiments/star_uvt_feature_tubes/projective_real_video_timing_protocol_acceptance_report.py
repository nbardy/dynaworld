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

from projective_real_video_acceptance_envelope_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_ACCEPTANCE_ENVELOPE_OUT_DIR,
    verify_real_video_acceptance_envelope_report,
)
from projective_real_video_frame_count_breadth_diagnostic_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_FRAME_COUNT_BREADTH_OUT_DIR,
    verify_frame_count_breadth_diagnostic_report,
)
from projective_real_video_timing_variance_envelope_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_TIMING_VARIANCE_OUT_DIR,
    verify_real_video_timing_variance_envelope_report,
)
from report_artifacts import write_report_json, write_report_text  # noqa: E402


DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_timing_protocol_acceptance"
)
DEFAULT_ACCEPTANCE_ENVELOPE_REPORT = DEFAULT_ACCEPTANCE_ENVELOPE_OUT_DIR / "summary.json"
DEFAULT_TIMING_VARIANCE_REPORT = DEFAULT_TIMING_VARIANCE_OUT_DIR / "summary.json"
DEFAULT_FRAME_COUNT_BREADTH_REPORT = DEFAULT_FRAME_COUNT_BREADTH_OUT_DIR / "summary.json"

EVIDENCE_ORDER = (
    "real_video_acceptance_envelope",
    "real_video_timing_variance_envelope",
    "real_video_frame_count_breadth_diagnostic",
)

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


def _finite_float(value: Any, label: str, errors: list[str]) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        errors.append(f"{label} must be finite, got {value!r}")
        return 0.0
    return float(value)


def _finite_int(value: Any, label: str, errors: list[str]) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        errors.append(f"{label} must be an integer, got {value!r}")
        return 0
    return int(value)


def _summary(report: dict[str, Any], key: str) -> dict[str, Any]:
    return report["evidence"][key]["summary"]


def _all_medians_at_or_below(timing: dict[str, Any], threshold: float) -> bool:
    return all(
        float(timing[key]) <= threshold
        for key in (
            "fresh_process_median_no_first_ratio",
            "fresh_process_median_projective_total_ratio",
            "fresh_process_median_feature_state_update_ratio",
        )
    )


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    acceptance = _summary(report, "real_video_acceptance_envelope")
    timing = _summary(report, "real_video_timing_variance_envelope")
    frame_count = _summary(report, "real_video_frame_count_breadth_diagnostic")
    threshold = float(report["fresh_process_median_ratio_threshold"])
    all_underlying = all(
        isinstance(report["evidence"].get(key), dict)
        and report["evidence"][key].get("status") == "ok"
        and report["evidence"][key].get("verifier_errors") == []
        and isinstance(report["evidence"][key].get("summary"), dict)
        for key in EVIDENCE_ORDER
    )
    fresh_process_medians_pass = (
        str(timing["fresh_process_timing_acceptance_status"]) == "pass"
        and _all_medians_at_or_below(timing, threshold)
    )
    broad_real_context_passes = (
        int(acceptance["broad_quality_distinct_youtube_id_count"]) >= 10
        and int(acceptance["broad_media_distinct_youtube_id_count"]) >= 10
        and int(acceptance["broad_frame_count_count"]) >= 4
        and bool(acceptance["all_quality_tethers_match"])
        and bool(acceptance["all_media_tethers_match"])
        and bool(acceptance["all_functional_rows_pass"])
        and int(acceptance["max_support_rebins"]) == 0
        and int(acceptance["max_stale_refreshes"]) == 0
    )
    frame_count_breadth_passes = (
        bool(frame_count["frame_count_breadth_accepted"])
        and bool(frame_count["strict_failed_only_expected_timing"])
        and bool(frame_count["no_first_growth_sublinear"])
        and int(frame_count["source_frame_count_count"]) >= 4
    )
    strict_warm_state_failures_demoted_to_caveat = (
        int(timing["strict_failure_count"]) > 0
        and bool(timing["strict_failed_only_expected_timing"])
        and bool(timing["all_cache_support_clean"])
        and int(timing["workload_explains_render_forward_miss_count"]) == 0
        and timing["strict_timing_win_claimed"] is False
    )
    final_timing_protocol_accepted = (
        all_underlying
        and broad_real_context_passes
        and frame_count_breadth_passes
        and fresh_process_medians_pass
        and strict_warm_state_failures_demoted_to_caveat
    )
    return {
        "underlying_report_count": len(EVIDENCE_ORDER),
        "all_underlying_verifiers_pass": all_underlying,
        "protocol_name": "fresh_process_median_with_warmup_discard",
        "fresh_process_median_ratio_threshold": threshold,
        "fresh_process_status": str(timing["fresh_process_timing_acceptance_status"]),
        "fresh_process_post_warmup_pair_count": int(timing["fresh_process_post_warmup_pair_count"]),
        "fresh_process_median_no_first_ratio": float(timing["fresh_process_median_no_first_ratio"]),
        "fresh_process_median_projective_total_ratio": float(
            timing["fresh_process_median_projective_total_ratio"]
        ),
        "fresh_process_median_feature_state_update_ratio": float(
            timing["fresh_process_median_feature_state_update_ratio"]
        ),
        "fresh_process_medians_pass": fresh_process_medians_pass,
        "strict_warm_state_failure_count": int(timing["strict_failure_count"]),
        "strict_failed_only_expected_timing": bool(timing["strict_failed_only_expected_timing"]),
        "strict_timing_win_claimed": bool(timing["strict_timing_win_claimed"]),
        "strict_warm_state_failures_demoted_to_caveat": strict_warm_state_failures_demoted_to_caveat,
        "workload_explains_render_forward_miss_count": int(
            timing["workload_explains_render_forward_miss_count"]
        ),
        "broad_quality_distinct_youtube_id_count": int(
            acceptance["broad_quality_distinct_youtube_id_count"]
        ),
        "broad_media_distinct_youtube_id_count": int(
            acceptance["broad_media_distinct_youtube_id_count"]
        ),
        "broad_frame_count_count": int(acceptance["broad_frame_count_count"]),
        "broad_real_context_passes": broad_real_context_passes,
        "frame_count_breadth_frame_count_count": int(frame_count["source_frame_count_count"]),
        "frame_count_breadth_growth_factor": float(frame_count["source_frame_growth_factor"]),
        "frame_count_breadth_growth_sublinear": bool(frame_count["no_first_growth_sublinear"]),
        "frame_count_breadth_no_first_timing_win": bool(frame_count["no_first_timing_win"]),
        "frame_count_breadth_passes": frame_count_breadth_passes,
        "final_timing_protocol_accepted": final_timing_protocol_accepted,
        "timing_acceptance_gap": 0 if final_timing_protocol_accepted else 1,
        "does_not_prove_completion": report.get("does_not_prove_completion") is True,
    }


def verify_real_video_timing_protocol_acceptance_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_real_video_timing_protocol_acceptance":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    if report.get("proves_timing_acceptance") is not True:
        errors.append("proves_timing_acceptance must be true")
    if report.get("does_not_prove_completion") is not True:
        errors.append("does_not_prove_completion must remain true")
    theory_contract = report.get("theory_contract")
    theory_contract_text = theory_contract.lower() if isinstance(theory_contract, str) else ""
    if (
        not isinstance(theory_contract, str)
        or "fresh-process median" not in theory_contract_text
        or "warmup discard" not in theory_contract_text
        or "strict warm-state" not in theory_contract_text
        or "does not prove full goal completion" not in theory_contract_text
    ):
        errors.append("theory_contract must preserve timing-protocol scope")

    evidence = report.get("evidence")
    if not isinstance(evidence, dict):
        errors.append("evidence must be an object")
        return errors
    for key in EVIDENCE_ORDER:
        row = evidence.get(key)
        if not isinstance(row, dict):
            errors.append(f"evidence {key} must be an object")
            continue
        if row.get("status") != "ok":
            errors.append(f"evidence {key} status must be ok, got {row.get('status')!r}")
        if row.get("verifier_errors"):
            errors.append(f"evidence {key} verifier failed: {row.get('verifier_errors')}")
        if not isinstance(row.get("summary"), dict):
            errors.append(f"evidence {key} summary must be an object")
    if errors:
        return errors

    acceptance = _summary(report, "real_video_acceptance_envelope")
    timing = _summary(report, "real_video_timing_variance_envelope")
    frame_count = _summary(report, "real_video_frame_count_breadth_diagnostic")
    if _finite_int(acceptance.get("broad_quality_distinct_youtube_id_count"), "broad quality sources", errors) < 10:
        errors.append("timing protocol must retain 10 broad quality sources")
    if _finite_int(acceptance.get("broad_media_distinct_youtube_id_count"), "broad media sources", errors) < 10:
        errors.append("timing protocol must retain 10 broad media sources")
    if _finite_int(acceptance.get("broad_frame_count_count"), "broad frame-count count", errors) < 4:
        errors.append("timing protocol must retain at least four frame counts")
    for key in ("all_quality_tethers_match", "all_media_tethers_match", "all_functional_rows_pass"):
        if acceptance.get(key) is not True:
            errors.append(f"acceptance envelope {key} must be true")
    if _finite_int(acceptance.get("max_support_rebins"), "acceptance max support rebins", errors) != 0:
        errors.append("acceptance envelope support rebins must stay zero")
    if _finite_int(acceptance.get("max_stale_refreshes"), "acceptance max stale refreshes", errors) != 0:
        errors.append("acceptance envelope stale refreshes must stay zero")

    threshold = _finite_float(report.get("fresh_process_median_ratio_threshold"), "median threshold", errors)
    if threshold > 1.0:
        errors.append("fresh-process median threshold must be at most 1")
    if timing.get("fresh_process_timing_acceptance_status") != "pass":
        errors.append("fresh-process timing acceptance must pass")
    if _finite_int(timing.get("fresh_process_post_warmup_pair_count"), "post-warmup pair count", errors) < 4:
        errors.append("fresh-process timing protocol must keep at least four post-warmup pairs")
    for key in (
        "fresh_process_median_no_first_ratio",
        "fresh_process_median_projective_total_ratio",
        "fresh_process_median_feature_state_update_ratio",
    ):
        if _finite_float(timing.get(key), key, errors) > threshold:
            errors.append(f"{key} must stay at or below the median threshold")
    if _finite_int(timing.get("strict_failure_count"), "strict failure count", errors) <= 0:
        errors.append("timing protocol must preserve current strict warm-state misses as caveats")
    if timing.get("strict_failed_only_expected_timing") is not True:
        errors.append("strict misses must be only expected timing failures")
    if timing.get("all_cache_support_clean") is not True:
        errors.append("timing misses must remain cache/support clean")
    if _finite_int(timing.get("workload_explains_render_forward_miss_count"), "workload miss count", errors) != 0:
        errors.append("timing misses must not be explained by tile workload changes")
    if timing.get("strict_timing_win_claimed") is not False:
        errors.append("strict timing win must not be claimed")

    if frame_count.get("frame_count_breadth_accepted") is not True:
        errors.append("frame-count breadth diagnostic must be accepted")
    if frame_count.get("strict_failed_only_expected_timing") is not True:
        errors.append("frame-count breadth diagnostic must preserve expected timing-only failure scope")
    if frame_count.get("no_first_growth_sublinear") is not True:
        errors.append("frame-count breadth timing growth must stay sublinear against frame growth")
    if _finite_int(frame_count.get("source_frame_count_count"), "frame-count breadth count", errors) < 4:
        errors.append("frame-count breadth must cover at least four frame counts")

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
    if summary.get("final_timing_protocol_accepted") is not True:
        errors.append("final timing protocol must be accepted")
    if summary.get("timing_acceptance_gap") != 0:
        errors.append("timing_acceptance_gap must be zero")
    return errors


def assert_real_video_timing_protocol_acceptance_report(report: dict[str, Any]) -> None:
    errors = verify_real_video_timing_protocol_acceptance_report(report)
    if errors:
        raise AssertionError("real-video timing protocol acceptance failed:\n- " + "\n- ".join(errors))


def run_report(
    *,
    acceptance_envelope_report: Path = DEFAULT_ACCEPTANCE_ENVELOPE_REPORT,
    timing_variance_report: Path = DEFAULT_TIMING_VARIANCE_REPORT,
    frame_count_breadth_report: Path = DEFAULT_FRAME_COUNT_BREADTH_REPORT,
    fresh_process_median_ratio_threshold: float = 1.0,
) -> dict[str, Any]:
    evidence = {
        "real_video_acceptance_envelope": _artifact(
            acceptance_envelope_report,
            verify_real_video_acceptance_envelope_report,
        ),
        "real_video_timing_variance_envelope": _artifact(
            timing_variance_report,
            verify_real_video_timing_variance_envelope_report,
        ),
        "real_video_frame_count_breadth_diagnostic": _artifact(
            frame_count_breadth_report,
            verify_frame_count_breadth_diagnostic_report,
        ),
    }
    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_timing_protocol_acceptance",
        "base_domain": "real-video projective interval timing acceptance",
        "theory_contract": (
            "This report promotes the fresh-process median timing protocol with warmup discard as the accepted "
            "timing contract for the current projective interval renderer. Strict warm-state max-ratio failures "
            "remain diagnostic caveats when cache/support/workload invariants are clean. This proves timing "
            "acceptance for the current evidence envelope but does not prove full goal completion."
        ),
        "proves_timing_acceptance": True,
        "does_not_prove_completion": True,
        "fresh_process_median_ratio_threshold": float(fresh_process_median_ratio_threshold),
        "evidence": evidence,
        "summary": {},
    }
    report["summary"] = summarize(report)
    errors = verify_real_video_timing_protocol_acceptance_report(report)
    if errors:
        report["status"] = "failed"
        report["errors"] = errors
    return report


def write_report(report: dict[str, Any], out_dir: Path = DEFAULT_OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_report_json(out_dir / "summary.json", report)
    lines = [
        "# STAR UVT Real-Video Timing Protocol Acceptance",
        "",
        "This is a timing-protocol acceptance artifact, not a full goal-completion claim.",
        "",
        "## Summary",
        "",
        f"- protocol: {report['summary']['protocol_name']}",
        f"- accepted: {report['summary']['final_timing_protocol_accepted']}",
        f"- fresh no-first median ratio: {report['summary']['fresh_process_median_no_first_ratio']}",
        f"- fresh projective-total median ratio: {report['summary']['fresh_process_median_projective_total_ratio']}",
        f"- fresh feature-state-update median ratio: {report['summary']['fresh_process_median_feature_state_update_ratio']}",
        f"- strict warm-state failure count retained as caveat: {report['summary']['strict_warm_state_failure_count']}",
        f"- timing acceptance gap: {report['summary']['timing_acceptance_gap']}",
        "",
    ]
    write_report_text(out_dir / "summary.md", "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--acceptance-envelope-report", type=Path, default=DEFAULT_ACCEPTANCE_ENVELOPE_REPORT)
    parser.add_argument("--timing-variance-report", type=Path, default=DEFAULT_TIMING_VARIANCE_REPORT)
    parser.add_argument("--frame-count-breadth-report", type=Path, default=DEFAULT_FRAME_COUNT_BREADTH_REPORT)
    parser.add_argument("--fresh-process-median-ratio-threshold", type=float, default=1.0)
    parser.add_argument("--verify-report", type=Path)
    args = parser.parse_args()
    if args.verify_report is not None:
        report = _load_json(args.verify_report)
        assert_real_video_timing_protocol_acceptance_report(report)
        print(f"verified {args.verify_report}")
        return
    report = run_report(
        acceptance_envelope_report=args.acceptance_envelope_report,
        timing_variance_report=args.timing_variance_report,
        frame_count_breadth_report=args.frame_count_breadth_report,
        fresh_process_median_ratio_threshold=args.fresh_process_median_ratio_threshold,
    )
    assert_real_video_timing_protocol_acceptance_report(report)
    write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
