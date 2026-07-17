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

from projective_real_video_multiscene_bq4_trace_fresh_process_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_BQ4_FRESH_PROCESS_OUT_DIR,
    verify_bq4_trace_fresh_process_report,
)
from projective_real_video_multiscene_bq4_trace_policy_order_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_BQ4_POLICY_ORDER_OUT_DIR,
    verify_bq4_trace_policy_order_report,
)
from projective_real_video_multiscene_bq4_trace_repeat_stability_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_BQ4_REPEAT_STABILITY_OUT_DIR,
    verify_bq4_trace_repeat_stability_report,
)
from projective_real_video_multiscene_bq4_trace_rerun_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_BQ4_RERUN_OUT_DIR,
    verify_bq4_trace_rerun_report,
)
from projective_real_video_multiscene_bq4_trace_sequence_order_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_BQ4_SEQUENCE_ORDER_OUT_DIR,
    verify_bq4_trace_sequence_order_report,
)
from projective_real_video_multiscene_extended_phase_profile_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_PHASE_PROFILE_OUT_DIR,
    verify_extended_phase_profile_report,
)
from projective_real_video_multiscene_extended_render_forward_residual_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_RENDER_FORWARD_RESIDUAL_OUT_DIR,
    verify_extended_render_forward_residual_report,
)
from projective_real_video_multiscene_extended_render_forward_shape_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_RENDER_FORWARD_SHAPE_OUT_DIR,
    verify_extended_render_forward_shape_report,
)
from projective_real_video_multiscene_extended_timing_breakdown_report import (  # noqa: E402
    DEFAULT_OUT_DIR as DEFAULT_TIMING_BREAKDOWN_OUT_DIR,
    verify_extended_timing_breakdown_report,
)


DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_timing_variance_envelope"
)
DEFAULT_TIMING_BREAKDOWN_REPORT = DEFAULT_TIMING_BREAKDOWN_OUT_DIR / "summary.json"
DEFAULT_PHASE_PROFILE_REPORT = DEFAULT_PHASE_PROFILE_OUT_DIR / "summary.json"
DEFAULT_RENDER_FORWARD_RESIDUAL_REPORT = DEFAULT_RENDER_FORWARD_RESIDUAL_OUT_DIR / "summary.json"
DEFAULT_RENDER_FORWARD_SHAPE_REPORT = DEFAULT_RENDER_FORWARD_SHAPE_OUT_DIR / "summary.json"
DEFAULT_BQ4_RERUN_REPORT = DEFAULT_BQ4_RERUN_OUT_DIR / "summary.json"
DEFAULT_BQ4_REPEAT_STABILITY_REPORT = DEFAULT_BQ4_REPEAT_STABILITY_OUT_DIR / "summary.json"
DEFAULT_BQ4_SEQUENCE_ORDER_REPORT = DEFAULT_BQ4_SEQUENCE_ORDER_OUT_DIR / "summary.json"
DEFAULT_BQ4_POLICY_ORDER_REPORT = DEFAULT_BQ4_POLICY_ORDER_OUT_DIR / "summary.json"
DEFAULT_BQ4_FRESH_PROCESS_REPORT = DEFAULT_BQ4_FRESH_PROCESS_OUT_DIR / "summary.json"

EVIDENCE_ORDER = (
    "timing_breakdown",
    "phase_profile",
    "render_forward_residual",
    "render_forward_shape",
    "bq4_trace_rerun",
    "bq4_repeat_stability",
    "bq4_sequence_order",
    "bq4_policy_order",
    "bq4_fresh_process",
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


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    evidence = report["evidence"]
    timing = _summary(report, "timing_breakdown")
    phase = _summary(report, "phase_profile")
    residual = _summary(report, "render_forward_residual")
    shape = _summary(report, "render_forward_shape")
    rerun = _summary(report, "bq4_trace_rerun")
    repeat = _summary(report, "bq4_repeat_stability")
    sequence = _summary(report, "bq4_sequence_order")
    policy = _summary(report, "bq4_policy_order")
    fresh = _summary(report, "bq4_fresh_process")
    fresh_acceptance = fresh["timing_acceptance"]
    fresh_post = fresh_acceptance["post_warmup_summary"]
    all_underlying = all(
        isinstance(evidence.get(key), dict)
        and evidence[key].get("status") == "ok"
        and evidence[key].get("verifier_errors") == []
        and isinstance(evidence[key].get("summary"), dict)
        for key in EVIDENCE_ORDER
    )
    return {
        "underlying_report_count": len(EVIDENCE_ORDER),
        "all_underlying_verifiers_pass": all_underlying,
        "source_scene_count": int(timing["source_scene_count"]),
        "source_distinct_youtube_id_count": int(timing["source_distinct_youtube_id_count"]),
        "source_row_count": int(timing["source_row_count"]),
        "strict_failure_count": int(timing["strict_failure_count"]),
        "strict_failed_only_expected_timing": bool(timing["strict_failed_only_expected_timing"]),
        "no_first_ratio_gt1_count": int(timing["no_first_ratio_gt1_count"]),
        "no_first_ratio_gt1_fraction": float(timing["no_first_ratio_gt1_fraction"]),
        "growth_ratio_gt1_count": int(timing["growth_ratio_gt1_count"]),
        "growth_ratio_gt1_fraction": float(timing["growth_ratio_gt1_fraction"]),
        "max_no_first_ratio_overage": float(timing["max_no_first_ratio_overage"]),
        "max_growth_ratio_overage": float(timing["max_growth_ratio_overage"]),
        "all_timing_miss_pairs_cache_clean": bool(timing["all_failing_pairs_cache_clean"]),
        "all_timing_miss_pairs_support_clean": bool(timing["all_pair_support_clean"]),
        "max_loss_delta": max(
            float(timing["max_end_loss_abs_delta"]),
            float(phase["max_profile_loss_delta"]),
            float(residual["max_loss_delta"]),
            float(shape["max_loss_delta"]),
        ),
        "max_rebuild_ratio": max(
            float(timing["max_measured_vs_cadence_rebuild_ratio"]),
            float(phase["max_profile_rebuild_ratio"]),
        ),
        "dominant_no_first_render_forward_count": int(
            phase["dominant_positive_phase_counts_for_no_first_misses"].get("render_forward_ms", 0)
        ),
        "dominant_no_first_colorize_count": int(
            phase["dominant_positive_phase_counts_for_no_first_misses"].get("colorize_loss_ms", 0)
        ),
        "workload_explains_render_forward_miss_count": int(
            residual["workload_explains_render_forward_miss_count"]
        ),
        "all_no_first_misses_tile_stats_identical": bool(
            residual["all_no_first_misses_tile_stats_identical"]
        ),
        "all_render_forward_misses_tile_stats_identical": bool(
            residual["all_render_forward_misses_tile_stats_identical"]
        ),
        "all_no_first_misses_single_spike_driven": bool(
            shape["all_no_first_misses_render_single_spike_driven"]
            and shape["all_no_first_misses_step_single_spike_driven"]
        ),
        "drop_spike_render_forward_ratio": float(
            shape["max_no_first_miss_render_forward_drop_spike_ratio"]
        ),
        "bq4_traced_spike_reproduced": bool(rerun["traced_bq4_spike_reproduced"]),
        "bq4_rerun_max_no_first_ratio": float(
            rerun["max_traced_measured_vs_cadence_no_first_step_ms_ratio"]
        ),
        "bq4_repeat_no_first_spike_count": int(repeat["no_first_spike_reproduced_count"]),
        "bq4_repeat_projective_bump_count": int(repeat["projective_total_bump_count"]),
        "bq4_repeat_max_no_first_ratio": float(repeat["max_no_first_ratio"]),
        "bq4_sequence_no_first_bump_count": int(sequence["all_16f_summary"]["no_first_bump_count"]),
        "bq4_sequence_projective_bump_count": int(sequence["all_16f_summary"]["projective_total_bump_count"]),
        "bq4_policy_no_first_bump_count": int(policy["all_target_summary"]["no_first_bump_count"]),
        "bq4_policy_projective_bump_count": int(policy["all_target_summary"]["projective_total_bump_count"]),
        "fresh_process_timing_acceptance_status": str(fresh_acceptance["status"]),
        "fresh_process_post_warmup_pair_count": int(fresh_acceptance["post_warmup_pair_count"]),
        "fresh_process_median_no_first_ratio": float(fresh_post["median_no_first_ratio"]),
        "fresh_process_median_projective_total_ratio": float(fresh_post["median_projective_total_ratio"]),
        "fresh_process_median_feature_state_update_ratio": float(
            fresh_post["median_feature_state_update_ratio"]
        ),
        "fresh_process_max_no_first_ratio": float(fresh["all_target_summary"]["max_no_first_ratio"]),
        "all_cache_support_clean": all(
            bool(_summary(report, key).get("all_rows_cache_support_clean", True))
            for key in (
                "bq4_trace_rerun",
                "bq4_repeat_stability",
                "bq4_sequence_order",
                "bq4_policy_order",
                "bq4_fresh_process",
            )
        )
        and bool(timing["all_failing_pairs_cache_clean"])
        and bool(timing["all_pair_support_clean"])
        and bool(residual["all_pairs_cache_support_clean"])
        and bool(shape["all_pairs_cache_support_clean"]),
        "strict_timing_win_claimed": False,
        "does_not_prove_completion": report.get("does_not_prove_completion") is True,
    }


def verify_real_video_timing_variance_envelope_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_real_video_timing_variance_envelope":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    theory_contract = report.get("theory_contract")
    if (
        not isinstance(theory_contract, str)
        or "timing-variance envelope" not in theory_contract
        or "does not prove a broad timing win" not in theory_contract
        or "MPS process variance" not in theory_contract
        or "does not prove full goal completion" not in theory_contract
    ):
        errors.append("theory_contract must preserve timing-variance non-completion scope")
    if report.get("does_not_prove_completion") is not True:
        errors.append("does_not_prove_completion must remain true")

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

    timing = _summary(report, "timing_breakdown")
    phase = _summary(report, "phase_profile")
    residual = _summary(report, "render_forward_residual")
    shape = _summary(report, "render_forward_shape")
    rerun = _summary(report, "bq4_trace_rerun")
    repeat = _summary(report, "bq4_repeat_stability")
    sequence = _summary(report, "bq4_sequence_order")
    policy = _summary(report, "bq4_policy_order")
    fresh = _summary(report, "bq4_fresh_process")

    if _finite_int(timing.get("source_scene_count"), "timing source_scene_count", errors) < 5:
        errors.append("timing breakdown must cover at least five scenes")
    if _finite_int(timing.get("source_distinct_youtube_id_count"), "timing distinct source count", errors) < 5:
        errors.append("timing breakdown must cover at least five source-distinct videos")
    if timing.get("strict_failed_only_expected_timing") is not True:
        errors.append("timing breakdown must fail only expected timing gates")
    if _finite_int(timing.get("strict_failure_count"), "timing strict failure count", errors) != 2:
        errors.append("timing breakdown must preserve the two expected timing failures")
    for key in (
        "all_failing_pairs_cache_clean",
        "all_pair_support_clean",
        "all_pair_loss_matches_cadence",
        "all_pair_rebuild_ratio_below_cadence",
        "all_scene_rebuild_growth_flat",
    ):
        if timing.get(key) is not True:
            errors.append(f"timing breakdown {key} must be true")
    if _finite_float(timing.get("max_end_loss_abs_delta"), "timing max loss delta", errors) > 1.0e-8:
        errors.append("timing breakdown loss delta must stay below 1e-8")

    if phase.get("all_profile_pairs_cache_support_clean") is not True:
        errors.append("phase profile pairs must remain cache/support clean")
    if phase.get("all_profile_losses_match_cadence") is not True:
        errors.append("phase profile losses must match cadence")
    phase_counts = phase.get("dominant_positive_phase_counts_for_no_first_misses")
    if not isinstance(phase_counts, dict) or int(phase_counts.get("render_forward_ms") or 0) < 1:
        errors.append("phase profile must identify render_forward as a no-first miss phase at least once")

    for label, summary in (("residual", residual), ("shape", shape)):
        if summary.get("all_pairs_cache_support_clean") is not True:
            errors.append(f"render-forward {label} pairs must be cache/support clean")
        if summary.get("all_pairs_losses_match_cadence") is not True:
            errors.append(f"render-forward {label} losses must match cadence")
    if residual.get("all_no_first_misses_tile_stats_identical") is not True:
        errors.append("render-forward residual must preserve identical tile stats for no-first misses")
    if residual.get("all_render_forward_misses_tile_stats_identical") is not True:
        errors.append("render-forward residual must preserve identical tile stats for render-forward misses")
    if _finite_int(residual.get("workload_explains_render_forward_miss_count"), "workload miss count", errors) != 0:
        errors.append("render-forward residual must not attribute misses to tile workload changes")
    if shape.get("all_no_first_misses_render_single_spike_driven") is not True:
        errors.append("render-forward shape must mark render misses as single-spike driven")
    if shape.get("all_no_first_misses_step_single_spike_driven") is not True:
        errors.append("render-forward shape must mark step misses as single-spike driven")
    if (
        _finite_float(
            shape.get("max_no_first_miss_render_forward_drop_spike_ratio"),
            "drop-spike render ratio",
            errors,
        )
        >= 1.0
    ):
        errors.append("dropping the largest render-forward spike must remove the no-first miss")

    for key, summary in (
        ("bq4_trace_rerun", rerun),
        ("bq4_repeat_stability", repeat),
        ("bq4_sequence_order", sequence),
        ("bq4_policy_order", policy),
        ("bq4_fresh_process", fresh),
    ):
        if summary.get("all_rows_cache_support_clean") is not True:
            errors.append(f"{key} rows must remain cache/support clean")
        if summary.get("all_expected_global_steps_traced") is not True:
            errors.append(f"{key} must trace all expected global steps")
        if summary.get("all_projective_interval_timing_present") is not True:
            errors.append(f"{key} must include projective interval substep timing")

    if rerun.get("traced_bq4_spike_reproduced") is not False:
        errors.append("Bq4 traced rerun must record the original spike as not reproduced")
    if _finite_float(rerun.get("max_traced_measured_vs_cadence_no_first_step_ms_ratio"), "Bq4 rerun no-first ratio", errors) >= 1.0:
        errors.append("Bq4 traced rerun no-first ratio must stay below 1")
    if _finite_int(repeat.get("no_first_spike_reproduced_count"), "Bq4 repeat no-first spike count", errors) != 0:
        errors.append("Bq4 repeat stability must not reproduce no-first spikes")
    if _finite_float(repeat.get("max_no_first_ratio"), "Bq4 repeat max no-first ratio", errors) >= 1.0:
        errors.append("Bq4 repeat max no-first ratio must stay below 1")
    if _finite_int(sequence.get("all_16f_summary", {}).get("no_first_bump_count"), "Bq4 sequence no-first bump count", errors) != 0:
        errors.append("Bq4 sequence-order 16f no-first bump count must stay zero")
    if _finite_int(policy.get("all_target_summary", {}).get("projective_total_bump_count"), "Bq4 policy projective bump count", errors) < 1:
        errors.append("Bq4 policy-order artifact must preserve warm-state projective bump caveat")

    if fresh.get("all_rows_fresh_process") is not True:
        errors.append("fresh-process trace must run in fresh processes")
    timing_acceptance = fresh.get("timing_acceptance")
    if not isinstance(timing_acceptance, dict):
        errors.append("fresh-process timing_acceptance must be an object")
    else:
        if timing_acceptance.get("status") != "pass":
            errors.append("fresh-process timing acceptance must pass")
        if timing_acceptance.get("median_ratios_within_threshold") is not True:
            errors.append("fresh-process median timing ratios must stay within threshold")
        post = timing_acceptance.get("post_warmup_summary")
        if not isinstance(post, dict):
            errors.append("fresh-process post_warmup_summary must be an object")
        else:
            for key in (
                "median_no_first_ratio",
                "median_projective_total_ratio",
                "median_feature_state_update_ratio",
            ):
                if _finite_float(post.get(key), f"fresh-process {key}", errors) > 1.0:
                    errors.append(f"fresh-process {key} must stay at or below 1")
    if _finite_float(fresh.get("all_target_summary", {}).get("max_no_first_ratio"), "fresh max no-first ratio", errors) >= 1.0:
        errors.append("fresh-process max no-first ratio must stay below 1")

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
    if summary.get("strict_timing_win_claimed") is not False:
        errors.append("summary strict_timing_win_claimed must remain false")
    if summary.get("does_not_prove_completion") is not True:
        errors.append("summary does_not_prove_completion must remain true")
    return errors


def assert_real_video_timing_variance_envelope_report(report: dict[str, Any]) -> None:
    errors = verify_real_video_timing_variance_envelope_report(report)
    if errors:
        raise AssertionError("real-video timing variance envelope failed:\n- " + "\n- ".join(errors))


def run_report(
    *,
    timing_breakdown_report: Path = DEFAULT_TIMING_BREAKDOWN_REPORT,
    phase_profile_report: Path = DEFAULT_PHASE_PROFILE_REPORT,
    render_forward_residual_report: Path = DEFAULT_RENDER_FORWARD_RESIDUAL_REPORT,
    render_forward_shape_report: Path = DEFAULT_RENDER_FORWARD_SHAPE_REPORT,
    bq4_rerun_report: Path = DEFAULT_BQ4_RERUN_REPORT,
    bq4_repeat_stability_report: Path = DEFAULT_BQ4_REPEAT_STABILITY_REPORT,
    bq4_sequence_order_report: Path = DEFAULT_BQ4_SEQUENCE_ORDER_REPORT,
    bq4_policy_order_report: Path = DEFAULT_BQ4_POLICY_ORDER_REPORT,
    bq4_fresh_process_report: Path = DEFAULT_BQ4_FRESH_PROCESS_REPORT,
) -> dict[str, Any]:
    evidence = {
        "timing_breakdown": _artifact(timing_breakdown_report, verify_extended_timing_breakdown_report),
        "phase_profile": _artifact(phase_profile_report, verify_extended_phase_profile_report),
        "render_forward_residual": _artifact(
            render_forward_residual_report,
            verify_extended_render_forward_residual_report,
        ),
        "render_forward_shape": _artifact(
            render_forward_shape_report,
            verify_extended_render_forward_shape_report,
        ),
        "bq4_trace_rerun": _artifact(bq4_rerun_report, verify_bq4_trace_rerun_report),
        "bq4_repeat_stability": _artifact(
            bq4_repeat_stability_report,
            verify_bq4_trace_repeat_stability_report,
        ),
        "bq4_sequence_order": _artifact(bq4_sequence_order_report, verify_bq4_trace_sequence_order_report),
        "bq4_policy_order": _artifact(bq4_policy_order_report, verify_bq4_trace_policy_order_report),
        "bq4_fresh_process": _artifact(bq4_fresh_process_report, verify_bq4_trace_fresh_process_report),
    }
    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_timing_variance_envelope",
        "goal": "fast 2D rasters across time from 4D spacetime primitives",
        "meta_goal": "share projection/support/binning/visibility/backward work over time",
        "theory_contract": (
            "This timing-variance envelope consolidates five-source timing misses, render-forward "
            "phase diagnostics, Bq4 traced reruns, sequence-order/warm-state caveats, and fresh-process "
            "median acceptance. It does not prove a broad timing win and does not prove full goal "
            "completion; it separates cache/support/atlas correctness from MPS process variance."
        ),
        "does_not_prove_completion": True,
        "evidence": evidence,
    }
    report["summary"] = summarize(report)
    errors = verify_real_video_timing_variance_envelope_report(report)
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
    parser.add_argument("--timing-breakdown-report", type=Path, default=DEFAULT_TIMING_BREAKDOWN_REPORT)
    parser.add_argument("--phase-profile-report", type=Path, default=DEFAULT_PHASE_PROFILE_REPORT)
    parser.add_argument("--render-forward-residual-report", type=Path, default=DEFAULT_RENDER_FORWARD_RESIDUAL_REPORT)
    parser.add_argument("--render-forward-shape-report", type=Path, default=DEFAULT_RENDER_FORWARD_SHAPE_REPORT)
    parser.add_argument("--bq4-rerun-report", type=Path, default=DEFAULT_BQ4_RERUN_REPORT)
    parser.add_argument("--bq4-repeat-stability-report", type=Path, default=DEFAULT_BQ4_REPEAT_STABILITY_REPORT)
    parser.add_argument("--bq4-sequence-order-report", type=Path, default=DEFAULT_BQ4_SEQUENCE_ORDER_REPORT)
    parser.add_argument("--bq4-policy-order-report", type=Path, default=DEFAULT_BQ4_POLICY_ORDER_REPORT)
    parser.add_argument("--bq4-fresh-process-report", type=Path, default=DEFAULT_BQ4_FRESH_PROCESS_REPORT)
    parser.add_argument("--verify-report", type=Path)
    args = parser.parse_args()

    if args.verify_report is not None:
        report = _load_json(args.verify_report)
        assert_real_video_timing_variance_envelope_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report(
        timing_breakdown_report=args.timing_breakdown_report,
        phase_profile_report=args.phase_profile_report,
        render_forward_residual_report=args.render_forward_residual_report,
        render_forward_shape_report=args.render_forward_shape_report,
        bq4_rerun_report=args.bq4_rerun_report,
        bq4_repeat_stability_report=args.bq4_repeat_stability_report,
        bq4_sequence_order_report=args.bq4_sequence_order_report,
        bq4_policy_order_report=args.bq4_policy_order_report,
        bq4_fresh_process_report=args.bq4_fresh_process_report,
    )
    if report.get("status") == "ok":
        assert_real_video_timing_variance_envelope_report(report)
    path = write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
