from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
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
    / "2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_fresh_process"
)
DEFAULT_REPEATS = 2
DEFAULT_TARGET_FRAMES = 16
DEFAULT_WARMUP_DISCARD_REPEATS = 0
DEFAULT_ACCEPTANCE_RATIO_THRESHOLD = 1.0
DEFAULT_POLICY_ORDERS = (
    {"name": "cadence_then_measured", "policies": ["cadence", "measured"]},
    {"name": "measured_then_cadence", "policies": ["measured", "cadence"]},
)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


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
        "median_no_first_ratio": _median(no_first),
        "max_no_first_ratio": max(no_first) if no_first else 0.0,
        "no_first_bump_count": sum(1 for ratio in no_first if ratio >= 1.0),
        "mean_projective_total_ratio": _mean(projective),
        "median_projective_total_ratio": _median(projective),
        "max_projective_total_ratio": max(projective) if projective else 0.0,
        "projective_total_bump_count": sum(1 for ratio in projective if ratio > 1.0),
        "mean_feature_state_update_ratio": _mean(feature_state),
        "median_feature_state_update_ratio": _median(feature_state),
        "max_feature_state_update_ratio": max(feature_state) if feature_state else 0.0,
        "feature_state_update_bump_count": sum(1 for ratio in feature_state if ratio > 1.0),
        "measured_first_count": sum(1 for slot in measured_slot if slot == 0),
        "measured_second_count": sum(1 for slot in measured_slot if slot == 1),
    }


def _timing_acceptance(
    *,
    pairs: list[dict[str, Any]],
    requested_repeat_count: int,
    policy_order_count: int,
    warmup_discard_repeats: int,
    ratio_threshold: float,
) -> dict[str, Any]:
    post_warmup_pairs = [
        pair for pair in pairs if int(pair.get("repeat_index") or 0) >= warmup_discard_repeats
    ]
    expected_post_warmup_pairs = max(0, requested_repeat_count - warmup_discard_repeats) * policy_order_count
    post_warmup_summary = _ratio_summary(post_warmup_pairs)
    sufficient_repeats = warmup_discard_repeats >= 0 and requested_repeat_count > warmup_discard_repeats
    sufficient_pairs = int(post_warmup_summary["pair_count"]) == expected_post_warmup_pairs and expected_post_warmup_pairs > 0
    median_fields = (
        "median_no_first_ratio",
        "median_projective_total_ratio",
        "median_feature_state_update_ratio",
    )
    median_ratios_within_threshold = all(
        float(post_warmup_summary[key]) <= ratio_threshold for key in median_fields
    )
    status = "insufficient"
    if sufficient_repeats and sufficient_pairs:
        status = "pass" if median_ratios_within_threshold else "fail"
    return {
        "status": status,
        "ratio_threshold": float(ratio_threshold),
        "warmup_discard_repeats": int(warmup_discard_repeats),
        "requested_repeat_count": int(requested_repeat_count),
        "policy_order_count": int(policy_order_count),
        "expected_post_warmup_pair_count": int(expected_post_warmup_pairs),
        "post_warmup_pair_count": int(post_warmup_summary["pair_count"]),
        "sufficient_repeats": bool(sufficient_repeats),
        "sufficient_pairs": bool(sufficient_pairs),
        "median_ratios_within_threshold": bool(median_ratios_within_threshold),
        "post_warmup_summary": post_warmup_summary,
        "post_warmup_paired_ratios": post_warmup_pairs,
    }


def _worker_run(worker_case: Path) -> None:
    case = json.loads(worker_case.read_text(encoding="utf-8"))
    segments = _load_segments(DEFAULT_SEGMENTS_MANIFEST)
    scene = segments[str(case["scene_id"])]
    row = run_case(
        scene=scene,
        frames=int(case["frames"]),
        policy=str(case["policy"]),
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
        out_json=Path(case["case_json"]),
        verbose_trainer_output=False,
        trace_global_steps=(int(case["trace_global_step"]),),
    )
    row.update(
        {
            "case_json": str(case["case_json"]),
            "expected_trace_global_step": int(case["trace_global_step"]),
            "policy_order_name": str(case["policy_order_name"]),
            "repeat_index": int(case["repeat_index"]),
            "policy_slot": int(case["policy_slot"]),
            "fresh_process": True,
        }
    )
    Path(case["row_json"]).write_text(json.dumps(row, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fresh_run_case(case: dict[str, Any], *, env: dict[str, str]) -> dict[str, Any]:
    case_path = Path(case["worker_case_json"])
    row_path = Path(case["row_json"])
    case_path.parent.mkdir(parents=True, exist_ok=True)
    row_path.parent.mkdir(parents=True, exist_ok=True)
    case_path.write_text(json.dumps(case, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker-case", str(case_path)],
        cwd=str(ROOT),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "fresh Bq4 worker failed\n"
            f"case={case_path}\n"
            f"stdout={completed.stdout[-4000:]}\n"
            f"stderr={completed.stderr[-4000:]}"
        )
    row = json.loads(row_path.read_text(encoding="utf-8"))
    row["fresh_process_elapsed_sec"] = float(time.perf_counter() - started)
    row["worker_stdout_tail"] = completed.stdout[-1000:]
    row["worker_stderr_tail"] = completed.stderr[-1000:]
    return row


def build_trace_profiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profiles = []
    for row in rows:
        profile = _trace_profile(Path(row["case_json"]), int(row["expected_trace_global_step"]), row)
        profile["policy_order_name"] = str(row["policy_order_name"])
        profile["repeat_index"] = int(row["repeat_index"])
        profile["policy_slot"] = int(row["policy_slot"])
        profile["fresh_process"] = bool(row.get("fresh_process"))
        profiles.append(profile)
    return profiles


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    profiles = [profile for profile in report.get("trace_profiles", []) if isinstance(profile, dict)]
    by_key = {
        (
            str(profile.get("policy_order_name")),
            int(profile.get("repeat_index") or 0),
            str(profile.get("policy")),
        ): profile
        for profile in profiles
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
    requested_repeat_count = int(report.get("requested_repeat_count") or 0)
    policy_order_count = len(report.get("policy_orders") or [])
    warmup_discard_repeats = int(report.get("warmup_discard_repeats") or 0)
    acceptance_ratio_threshold = float(
        report.get("acceptance_ratio_threshold") or DEFAULT_ACCEPTANCE_RATIO_THRESHOLD
    )
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
        "requested_repeat_count": requested_repeat_count,
        "warmup_discard_repeats": warmup_discard_repeats,
        "acceptance_ratio_threshold": acceptance_ratio_threshold,
        "policy_order_count": policy_order_count,
        "row_count": len(report.get("rows") or []),
        "trace_profile_count": len(profiles),
        "chunk_profile_count": len(chunk_profiles),
        "paired_ratio_count": len(pairs),
        "all_rows_fresh_process": all(bool(profile.get("fresh_process")) for profile in profiles),
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
        "timing_acceptance": _timing_acceptance(
            pairs=pairs,
            requested_repeat_count=requested_repeat_count,
            policy_order_count=policy_order_count,
            warmup_discard_repeats=warmup_discard_repeats,
            ratio_threshold=acceptance_ratio_threshold,
        ),
        "policy_order_summaries": by_order,
    }


def verify_bq4_trace_fresh_process_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_real_video_multiscene_bq4_trace_fresh_process":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    target_frames = int(report.get("target_frames") or 0)
    if target_frames <= 0:
        errors.append("target_frames must be positive")
    requested = int(report.get("requested_repeat_count") or 0)
    if requested < 1:
        errors.append("requested_repeat_count must be at least 1")
    warmup_discard_repeats = int(report.get("warmup_discard_repeats") or 0)
    if warmup_discard_repeats < 0:
        errors.append("warmup_discard_repeats must be non-negative")
    if warmup_discard_repeats >= requested:
        errors.append("warmup_discard_repeats must leave at least one measured repeat")
    if float(report.get("acceptance_ratio_threshold") or 0.0) <= 0.0:
        errors.append("acceptance_ratio_threshold must be positive")
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
    seen: set[tuple[str, int, str]] = set()
    for profile in profiles:
        if not isinstance(profile, dict):
            errors.append("trace profiles must be objects")
            continue
        repeat_index = int(profile.get("repeat_index") or 0)
        policy = str(profile.get("policy"))
        frames = int(profile.get("frames") or 0)
        order_name = str(profile.get("policy_order_name", ""))
        label = f"{order_name} repeat {repeat_index} {frames}f {policy}"
        if profile.get("scene_id") != BQ4_SCENE_ID:
            errors.append(f"{label} must be the Bq4 scene")
        if frames != target_frames:
            errors.append(f"{label} must target {target_frames} frames")
        if policy not in {"cadence", "measured"}:
            errors.append(f"{label} policy must be cadence or measured")
        if order_name not in order_specs:
            errors.append(f"{label} policy order must be declared")
        else:
            slot = int(profile.get("policy_slot", -1))
            if slot < 0 or slot >= len(order_specs[order_name]) or order_specs[order_name][slot] != policy:
                errors.append(f"{label} policy slot does not match declared order")
        if profile.get("fresh_process") is not True:
            errors.append(f"{label} must be marked fresh_process")
        seen.add((order_name, repeat_index, policy))
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
    for order_name in order_specs:
        for repeat_index in range(requested):
            for policy in ("cadence", "measured"):
                if (order_name, repeat_index, policy) not in seen:
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
    if summary.get("all_rows_fresh_process") is not True:
        errors.append("all target rows must be fresh-process workers")
    if summary.get("all_expected_global_steps_traced") is not True:
        errors.append("all expected Bq4 fresh-process steps must be traced")
    if summary.get("all_projective_interval_timing_present") is not True:
        errors.append("all traced fresh-process chunks must include projective interval timing")
    if summary.get("all_rows_cache_support_clean") is not True:
        errors.append("Bq4 fresh-process rows must remain cache/support clean")
    expected_pairs = requested * len(order_specs)
    if int(summary.get("paired_ratio_count") or 0) != expected_pairs:
        errors.append(f"expected {expected_pairs} target paired ratios")
    return errors


def assert_bq4_trace_fresh_process_report(report: dict[str, Any]) -> None:
    errors = verify_bq4_trace_fresh_process_report(report)
    if errors:
        raise AssertionError("Bq4 trace fresh-process failed:\n- " + "\n- ".join(errors))


def run_report(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    shape_report: Path = DEFAULT_SHAPE_REPORT,
    repeats: int = DEFAULT_REPEATS,
    warmup_discard_repeats: int = DEFAULT_WARMUP_DISCARD_REPEATS,
) -> dict[str, Any]:
    if not DEFAULT_SEGMENTS_MANIFEST.exists():
        return {"status": "skipped", "reason": f"missing segment manifest: {DEFAULT_SEGMENTS_MANIFEST}", "rows": []}
    if not torch.backends.mps.is_available():
        return {"status": "skipped", "reason": "MPS unavailable", "rows": []}
    if not has_projective_trace_cell_interval_metal() or not has_projective_trace_cell_interval_backward_metal():
        return {"status": "skipped", "reason": "projective interval Metal ops unavailable", "rows": []}
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if warmup_discard_repeats < 0:
        raise ValueError("warmup_discard_repeats must be non-negative")
    if warmup_discard_repeats >= repeats:
        raise ValueError("warmup_discard_repeats must leave at least one measured repeat")
    shape = json.loads(shape_report.read_text(encoding="utf-8"))
    specs = {int(spec["frames"]): spec for spec in _trace_specs_from_shape(shape)}
    if DEFAULT_TARGET_FRAMES not in specs:
        raise ValueError(f"missing Bq4 trace spec for frames: {DEFAULT_TARGET_FRAMES}")
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = ":".join(
        [
            str(ROOT / "src" / "train"),
            str(STAR_UVT_ROOT),
            str(SCRIPT_DIR),
            env.get("PYTHONPATH", ""),
        ]
    )
    rows: list[dict[str, Any]] = []
    spec = specs[DEFAULT_TARGET_FRAMES]
    with tempfile.TemporaryDirectory(prefix="bq4_fresh_process_", dir=str(out_dir)) as tmp_dir:
        tmp_path = Path(tmp_dir)
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
                    row_json = tmp_path / f"{order_name}_repeat{repeat_index}_slot{policy_slot}_{policy}_row.json"
                    worker_case = tmp_path / f"{order_name}_repeat{repeat_index}_slot{policy_slot}_{policy}_case.json"
                    rows.append(
                        _fresh_run_case(
                            {
                                "scene_id": BQ4_SCENE_ID,
                                "frames": DEFAULT_TARGET_FRAMES,
                                "policy": policy,
                                "trace_global_step": int(spec["trace_global_step"]),
                                "case_json": str(case_json),
                                "row_json": str(row_json),
                                "worker_case_json": str(worker_case),
                                "policy_order_name": order_name,
                                "repeat_index": int(repeat_index),
                                "policy_slot": int(policy_slot),
                            },
                            env=env,
                        )
                    )
    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_multiscene_bq4_trace_fresh_process",
        "base_domain": "Bq4 fresh-process 16f policy-order timing isolation",
        "theory_contract": (
            "This report runs each Bq4 16f target case in a fresh Python process, "
            "separating process/runtime warm-state effects from policy-order effects."
        ),
        "source_shape_report": str(shape_report),
        "source_shape_status": shape.get("status"),
        "target_frames": DEFAULT_TARGET_FRAMES,
        "requested_repeat_count": int(repeats),
        "warmup_discard_repeats": int(warmup_discard_repeats),
        "acceptance_ratio_threshold": DEFAULT_ACCEPTANCE_RATIO_THRESHOLD,
        "policy_orders": [
            {"name": str(order["name"]), "policies": [str(policy) for policy in order["policies"]]}
            for order in DEFAULT_POLICY_ORDERS
        ],
        "rows": rows,
    }
    report["trace_profiles"] = build_trace_profiles(rows)
    report["summary"] = summarize(report)
    errors = verify_bq4_trace_fresh_process_report(report)
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
    parser.add_argument("--warmup-discard-repeats", type=int, default=DEFAULT_WARMUP_DISCARD_REPEATS)
    parser.add_argument("--verify-report", type=Path)
    parser.add_argument("--worker-case", type=Path)
    args = parser.parse_args()

    if args.worker_case is not None:
        _worker_run(args.worker_case)
        return
    if args.verify_report is not None:
        report = json.loads(args.verify_report.read_text(encoding="utf-8"))
        assert_bq4_trace_fresh_process_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report(
        out_dir=args.out_dir,
        shape_report=args.shape_report,
        repeats=int(args.repeats),
        warmup_discard_repeats=int(args.warmup_discard_repeats),
    )
    if report.get("status") == "ok":
        assert_bq4_trace_fresh_process_report(report)
    path = write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
