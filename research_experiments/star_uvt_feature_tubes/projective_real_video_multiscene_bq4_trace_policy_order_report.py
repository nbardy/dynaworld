from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
STAR_UVT_ROOT = ROOT / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
for path in (ROOT, ROOT / "src" / "train", STAR_UVT_ROOT, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch  # noqa: E402

from projective_real_video_multiscene_bq4_trace_rerun_report import (  # noqa: E402
    BQ4_SCENE_ID,
    DEFAULT_SHAPE_REPORT,
    TIMING_KEYS,
    _finite_float,
    _ratio,
    _trace_profile,
    _trace_specs_from_shape,
)
from projective_real_video_multiscene_trainer_matrix import (  # noqa: E402
    DEFAULT_SEGMENTS_MANIFEST,
    _load_segments,
    run_case,
)
from torch_gsplat_bridge_star_uvt import (  # noqa: E402
    has_projective_trace_cell_interval_backward_metal,
    has_projective_trace_cell_interval_metal,
)


DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_policy_order"
)
DEFAULT_REPEATS = 2
DEFAULT_TARGET_FRAMES = 16
DEFAULT_POLICY_ORDERS = (
    {"name": "cadence_then_measured", "policies": ["cadence", "measured"]},
    {"name": "measured_then_cadence", "policies": ["measured", "cadence"]},
)
DEFAULT_WARMUP_CASES = (
    {"frames": 4, "policy": "cadence"},
    {"frames": 4, "policy": "measured"},
    {"frames": 16, "policy": "cadence"},
    {"frames": 16, "policy": "measured"},
)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _profile_timing(profile: dict[str, Any], key: str) -> float:
    chunks = profile.get("chunk_profiles") or []
    if not chunks:
        return 0.0
    timing = chunks[0].get("projective_interval_timing_ms") or {}
    return float(timing.get(key) or 0.0)


def _ratio_summary(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    no_first = [float(pair["no_first_ratio"]) for pair in pairs]
    projective = [float(pair["projective_total_ratio"]) for pair in pairs]
    feature_state = [float(pair["feature_state_update_ratio"]) for pair in pairs]
    measured_slot = [int(pair["measured_slot"]) for pair in pairs]
    return {
        "pair_count": len(pairs),
        "mean_no_first_ratio": _mean(no_first),
        "max_no_first_ratio": max(no_first) if no_first else 0.0,
        "no_first_bump_count": sum(1 for ratio in no_first if ratio >= 1.0),
        "mean_projective_total_ratio": _mean(projective),
        "max_projective_total_ratio": max(projective) if projective else 0.0,
        "projective_total_bump_count": sum(1 for ratio in projective if ratio > 1.0),
        "mean_feature_state_update_ratio": _mean(feature_state),
        "max_feature_state_update_ratio": max(feature_state) if feature_state else 0.0,
        "feature_state_update_bump_count": sum(1 for ratio in feature_state if ratio > 1.0),
        "measured_first_count": sum(1 for slot in measured_slot if slot == 0),
        "measured_second_count": sum(1 for slot in measured_slot if slot == 1),
    }


def build_trace_profiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profiles = []
    for row in rows:
        profile = _trace_profile(Path(row["case_json"]), int(row["expected_trace_global_step"]), row)
        profile["phase"] = str(row["phase"])
        profile["repeat_index"] = int(row["repeat_index"])
        profile["policy_order_name"] = str(row.get("policy_order_name", ""))
        profile["policy_slot"] = int(row.get("policy_slot", -1))
        profiles.append(profile)
    return profiles


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    profiles = [profile for profile in report.get("trace_profiles", []) if isinstance(profile, dict)]
    target_profiles = [profile for profile in profiles if profile.get("phase") == "target"]
    by_key = {
        (
            str(profile.get("policy_order_name")),
            int(profile.get("repeat_index") or 0),
            str(profile.get("policy")),
        ): profile
        for profile in target_profiles
    }
    pairs: list[dict[str, Any]] = []
    policy_order_names = sorted({key[0] for key in by_key})
    repeat_indices = sorted({key[1] for key in by_key})
    for policy_order_name in policy_order_names:
        for repeat_index in repeat_indices:
            cadence = by_key.get((policy_order_name, repeat_index, "cadence"))
            measured = by_key.get((policy_order_name, repeat_index, "measured"))
            if cadence is None or measured is None:
                continue
            cadence_projective = _profile_timing(cadence, "projective_interval_render_ms")
            measured_projective = _profile_timing(measured, "projective_interval_render_ms")
            cadence_feature_state = _profile_timing(cadence, "feature_state_update_ms")
            measured_feature_state = _profile_timing(measured, "feature_state_update_ms")
            pairs.append(
                {
                    "policy_order_name": policy_order_name,
                    "repeat_index": repeat_index,
                    "measured_slot": int(measured.get("policy_slot") or 0),
                    "cadence_slot": int(cadence.get("policy_slot") or 0),
                    "no_first_ratio": _ratio(
                        float(measured.get("no_first_step_ms") or 0.0),
                        float(cadence.get("no_first_step_ms") or 0.0),
                    ),
                    "projective_total_ratio": _ratio(measured_projective, cadence_projective),
                    "feature_state_update_ratio": _ratio(measured_feature_state, cadence_feature_state),
                    "cadence_projective_interval_render_ms": cadence_projective,
                    "measured_projective_interval_render_ms": measured_projective,
                    "cadence_feature_state_update_ms": cadence_feature_state,
                    "measured_feature_state_update_ms": measured_feature_state,
                }
            )
    by_order = {
        policy_order_name: _ratio_summary([pair for pair in pairs if pair["policy_order_name"] == policy_order_name])
        for policy_order_name in policy_order_names
    }
    chunk_profiles = [
        chunk
        for profile in profiles
        for chunk in profile.get("chunk_profiles", [])
        if isinstance(chunk, dict)
    ]
    timing_values = [
        float(chunk["projective_interval_timing_ms"][key])
        for chunk in chunk_profiles
        for key in TIMING_KEYS
    ]
    return {
        "source_shape_status": report.get("source_shape_status"),
        "target_frames": int(report.get("target_frames") or 0),
        "requested_repeat_count": int(report.get("requested_repeat_count") or 0),
        "policy_order_count": len(report.get("policy_orders") or []),
        "warmup_profile_count": sum(1 for profile in profiles if profile.get("phase") == "warmup"),
        "target_profile_count": len(target_profiles),
        "trace_profile_count": len(profiles),
        "chunk_profile_count": len(chunk_profiles),
        "paired_ratio_count": len(pairs),
        "all_expected_global_steps_traced": all(
            int(profile.get("matching_trace_count") or 0) == 1 for profile in profiles
        ),
        "all_projective_interval_timing_present": all(
            all(key in chunk.get("projective_interval_timing_ms", {}) for key in TIMING_KEYS)
            for chunk in chunk_profiles
        ),
        "all_rows_cache_support_clean": all(
            int(profile.get("projective_interval_cache_support_rebins") or 0) == 0
            and int(profile.get("projective_interval_cache_stale_refreshes") or 0) == 0
            and int(profile.get("projective_interval_cache_fallback_marks") or 0) == 0
            and int(profile.get("tile_overflow_sum") or 0) == 0
            for profile in profiles
        ),
        "min_projective_interval_subtiming_ms": min(timing_values) if timing_values else 0.0,
        "max_projective_interval_subtiming_ms": max(timing_values) if timing_values else 0.0,
        "paired_ratios": pairs,
        "all_target_summary": _ratio_summary(pairs),
        "policy_order_summaries": by_order,
    }


def verify_bq4_trace_policy_order_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_real_video_multiscene_bq4_trace_policy_order":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    target_frames = int(report.get("target_frames") or 0)
    if target_frames <= 0:
        errors.append("target_frames must be positive")
    requested = int(report.get("requested_repeat_count") or 0)
    if requested < 1:
        errors.append("requested_repeat_count must be at least 1")
    policy_orders = report.get("policy_orders")
    if not isinstance(policy_orders, list) or len(policy_orders) < 2:
        errors.append("policy_orders must include at least two policy schedules")
        return errors
    order_specs: dict[str, list[str]] = {}
    for order in policy_orders:
        if not isinstance(order, dict):
            errors.append("policy order specs must be objects")
            continue
        name = str(order.get("name"))
        policies = [str(policy) for policy in order.get("policies", [])]
        if set(policies) != {"cadence", "measured"} or len(policies) != 2:
            errors.append(f"policy order {name} must contain cadence and measured exactly once")
        order_specs[name] = policies
    profiles = report.get("trace_profiles")
    if not isinstance(profiles, list) or not profiles:
        errors.append("trace_profiles must be non-empty")
        return errors
    seen_target: set[tuple[str, int, str]] = set()
    warmup_count = 0
    for profile in profiles:
        if not isinstance(profile, dict):
            errors.append("trace profiles must be objects")
            continue
        phase = str(profile.get("phase"))
        repeat_index = int(profile.get("repeat_index") or 0)
        policy = str(profile.get("policy"))
        frames = int(profile.get("frames") or 0)
        order_name = str(profile.get("policy_order_name", ""))
        label = f"{phase} {order_name} repeat {repeat_index} {frames}f {policy}"
        if profile.get("scene_id") != BQ4_SCENE_ID:
            errors.append(f"{label} must be the Bq4 scene")
        if policy not in {"cadence", "measured"}:
            errors.append(f"{label} policy must be cadence or measured")
        if phase == "warmup":
            warmup_count += 1
        elif phase == "target":
            if frames != target_frames:
                errors.append(f"{label} must target {target_frames} frames")
            if order_name not in order_specs:
                errors.append(f"{label} policy order must be declared")
            else:
                slot = int(profile.get("policy_slot", -1))
                if slot < 0 or slot >= len(order_specs[order_name]) or order_specs[order_name][slot] != policy:
                    errors.append(f"{label} policy slot does not match declared order")
            seen_target.add((order_name, repeat_index, policy))
        else:
            errors.append(f"{label} phase must be warmup or target")
        if int(profile.get("matching_trace_count") or 0) != 1:
            errors.append(f"{label} must have exactly one matching traced step")
        chunks = profile.get("chunk_profiles")
        if not isinstance(chunks, list) or not chunks:
            errors.append(f"{label} must have chunk profiles")
            continue
        for chunk in chunks:
            if not isinstance(chunk, dict):
                errors.append(f"{label} chunk profiles must be objects")
                continue
            timing = chunk.get("projective_interval_timing_ms")
            if not isinstance(timing, dict):
                errors.append(f"{label} chunk must include projective interval timing")
                continue
            for key in TIMING_KEYS:
                if _finite_float(timing.get(key), f"{label} {key}", errors) <= 0.0:
                    errors.append(f"{label} {key} must be positive")
            if _finite_float(chunk.get("subtiming_total_abs_delta_ms"), f"{label} timing total delta", errors) > 1e-5:
                errors.append(f"{label} subtiming total must match sum of substeps")
    if warmup_count <= 0:
        errors.append("policy-order report must include warmup profiles")
    for order_name in order_specs:
        for repeat_index in range(requested):
            for policy in ("cadence", "measured"):
                if (order_name, repeat_index, policy) not in seen_target:
                    errors.append(f"missing {order_name} repeat {repeat_index} {policy} target profile")
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
            if not isinstance(actual, int | float) or abs(float(actual) - expected_value) > 1e-9:
                errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual!r}")
        elif actual != expected_value:
            errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {actual!r}")
    if summary.get("all_expected_global_steps_traced") is not True:
        errors.append("all expected Bq4 policy-order steps must be traced")
    if summary.get("all_projective_interval_timing_present") is not True:
        errors.append("all traced policy-order chunks must include projective interval timing")
    if summary.get("all_rows_cache_support_clean") is not True:
        errors.append("Bq4 policy-order rows must remain cache/support clean")
    expected_pairs = requested * len(order_specs)
    if int(summary.get("paired_ratio_count") or 0) != expected_pairs:
        errors.append(f"expected {expected_pairs} target paired ratios")
    return errors


def assert_bq4_trace_policy_order_report(report: dict[str, Any]) -> None:
    errors = verify_bq4_trace_policy_order_report(report)
    if errors:
        raise AssertionError("Bq4 trace policy order failed:\n- " + "\n- ".join(errors))


def run_report(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    shape_report: Path = DEFAULT_SHAPE_REPORT,
    repeats: int = DEFAULT_REPEATS,
    verbose_trainer_output: bool = False,
) -> dict[str, Any]:
    if not DEFAULT_SEGMENTS_MANIFEST.exists():
        return {"status": "skipped", "reason": f"missing segment manifest: {DEFAULT_SEGMENTS_MANIFEST}", "rows": []}
    if not torch.backends.mps.is_available():
        return {"status": "skipped", "reason": "MPS unavailable", "rows": []}
    if not has_projective_trace_cell_interval_metal() or not has_projective_trace_cell_interval_backward_metal():
        return {"status": "skipped", "reason": "projective interval Metal ops unavailable", "rows": []}
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    shape = json.loads(shape_report.read_text(encoding="utf-8"))
    specs = {int(spec["frames"]): spec for spec in _trace_specs_from_shape(shape)}
    required_frames = {int(case["frames"]) for case in DEFAULT_WARMUP_CASES}
    required_frames.add(DEFAULT_TARGET_FRAMES)
    missing = sorted(required_frames.difference(specs))
    if missing:
        raise ValueError(f"missing Bq4 trace specs for frames: {missing}")
    segments = _load_segments(DEFAULT_SEGMENTS_MANIFEST)
    scene = segments[BQ4_SCENE_ID]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for warmup_index, warmup in enumerate(DEFAULT_WARMUP_CASES):
        frames = int(warmup["frames"])
        policy = str(warmup["policy"])
        spec = specs[frames]
        case_json = out_dir / "cases" / f"{BQ4_SCENE_ID}_warmup{warmup_index}_{frames}f_{policy}.json"
        row = run_case(
            scene=scene,
            frames=frames,
            policy=policy,
            size=64,
            steps=4,
            refresh_every=2,
            tile_capacity=128,
            tube_count=128,
            support_guard_padding=1.0,
            support_guard_policy="slack_budgeted",
            support_guard_bisect_steps=8,
            support_stale_overshoot_epsilon=0.0,
            support_stale_tail_alpha_epsilon=0.001,
            out_json=case_json,
            verbose_trainer_output=verbose_trainer_output,
            trace_global_steps=(int(spec["trace_global_step"]),),
        )
        row["case_json"] = str(case_json)
        row["expected_trace_global_step"] = int(spec["trace_global_step"])
        row["phase"] = "warmup"
        row["repeat_index"] = -1
        row["policy_order_name"] = ""
        row["policy_slot"] = -1
        rows.append(row)

    spec = specs[DEFAULT_TARGET_FRAMES]
    for order in DEFAULT_POLICY_ORDERS:
        order_name = str(order["name"])
        policies = [str(policy) for policy in order["policies"]]
        for repeat_index in range(int(repeats)):
            for policy_slot, policy in enumerate(policies):
                case_json = (
                    out_dir
                    / "cases"
                    / f"{BQ4_SCENE_ID}_{order_name}_repeat{repeat_index}_slot{policy_slot}_{DEFAULT_TARGET_FRAMES}f_{policy}.json"
                )
                row = run_case(
                    scene=scene,
                    frames=DEFAULT_TARGET_FRAMES,
                    policy=policy,
                    size=64,
                    steps=4,
                    refresh_every=2,
                    tile_capacity=128,
                    tube_count=128,
                    support_guard_padding=1.0,
                    support_guard_policy="slack_budgeted",
                    support_guard_bisect_steps=8,
                    support_stale_overshoot_epsilon=0.0,
                    support_stale_tail_alpha_epsilon=0.001,
                    out_json=case_json,
                    verbose_trainer_output=verbose_trainer_output,
                    trace_global_steps=(int(spec["trace_global_step"]),),
                )
                row["case_json"] = str(case_json)
                row["expected_trace_global_step"] = int(spec["trace_global_step"])
                row["phase"] = "target"
                row["repeat_index"] = int(repeat_index)
                row["policy_order_name"] = order_name
                row["policy_slot"] = int(policy_slot)
                rows.append(row)

    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_multiscene_bq4_trace_policy_order",
        "base_domain": "Bq4 warmed 16f policy-order timing isolation",
        "theory_contract": (
            "This report warms the process with traced Bq4 cases, then alternates the "
            "16f cadence/measured order to test whether substep bumps follow policy or execution slot."
        ),
        "source_shape_report": str(shape_report),
        "source_shape_status": shape.get("status"),
        "target_frames": DEFAULT_TARGET_FRAMES,
        "requested_repeat_count": int(repeats),
        "warmup_cases": [
            {"frames": int(case["frames"]), "policy": str(case["policy"])} for case in DEFAULT_WARMUP_CASES
        ],
        "policy_orders": [
            {"name": str(order["name"]), "policies": [str(policy) for policy in order["policies"]]}
            for order in DEFAULT_POLICY_ORDERS
        ],
        "rows": rows,
    }
    report["trace_profiles"] = build_trace_profiles(rows)
    report["summary"] = summarize(report)
    errors = verify_bq4_trace_policy_order_report(report)
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
    parser.add_argument("--shape-report", type=Path, default=DEFAULT_SHAPE_REPORT)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--verify-report", type=Path)
    parser.add_argument("--verbose-trainer-output", action="store_true")
    args = parser.parse_args()

    if args.verify_report is not None:
        report = json.loads(args.verify_report.read_text(encoding="utf-8"))
        assert_bq4_trace_policy_order_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report(
        out_dir=args.out_dir,
        shape_report=args.shape_report,
        repeats=int(args.repeats),
        verbose_trainer_output=bool(args.verbose_trainer_output),
    )
    if report.get("status") == "ok":
        assert_bq4_trace_policy_order_report(report)
    path = write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
