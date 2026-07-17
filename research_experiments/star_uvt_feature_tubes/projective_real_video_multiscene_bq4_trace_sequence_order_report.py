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
    / "2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_sequence_order"
)
DEFAULT_REPEATS = 2
DEFAULT_SEQUENCES = (
    {"name": "mixed_4_to_16", "frames": [4, 16]},
    {"name": "reverse_16_to_4", "frames": [16, 4]},
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
    }


def build_trace_profiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profiles = []
    for row in rows:
        profile = _trace_profile(Path(row["case_json"]), int(row["expected_trace_global_step"]), row)
        profile["sequence_name"] = str(row["sequence_name"])
        profile["repeat_index"] = int(row["repeat_index"])
        profile["sequence_frame_index"] = int(row["sequence_frame_index"])
        profiles.append(profile)
    return profiles


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    profiles = [profile for profile in report.get("trace_profiles", []) if isinstance(profile, dict)]
    by_key = {
        (
            str(profile.get("sequence_name")),
            int(profile.get("repeat_index") or 0),
            int(profile.get("frames") or 0),
            str(profile.get("policy")),
        ): profile
        for profile in profiles
    }
    pairs: list[dict[str, Any]] = []
    sequence_names = sorted({key[0] for key in by_key})
    repeat_indices = sorted({key[1] for key in by_key})
    frames_values = sorted({key[2] for key in by_key})
    for sequence_name in sequence_names:
        for repeat_index in repeat_indices:
            for frames in frames_values:
                cadence = by_key.get((sequence_name, repeat_index, frames, "cadence"))
                measured = by_key.get((sequence_name, repeat_index, frames, "measured"))
                if cadence is None or measured is None:
                    continue
                cadence_projective = _profile_timing(cadence, "projective_interval_render_ms")
                measured_projective = _profile_timing(measured, "projective_interval_render_ms")
                cadence_feature_state = _profile_timing(cadence, "feature_state_update_ms")
                measured_feature_state = _profile_timing(measured, "feature_state_update_ms")
                pairs.append(
                    {
                        "sequence_name": sequence_name,
                        "repeat_index": repeat_index,
                        "frames": frames,
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
    pairs_16f = [pair for pair in pairs if int(pair["frames"]) == 16]
    by_sequence_16f = {
        sequence_name: _ratio_summary([pair for pair in pairs_16f if pair["sequence_name"] == sequence_name])
        for sequence_name in sequence_names
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
        "requested_repeat_count": int(report.get("requested_repeat_count") or 0),
        "sequence_count": len(report.get("sequences") or []),
        "row_count": len(report.get("rows") or []),
        "trace_profile_count": len(profiles),
        "chunk_profile_count": len(chunk_profiles),
        "paired_ratio_count": len(pairs),
        "paired_16f_ratio_count": len(pairs_16f),
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
        "paired_16f_ratios": pairs_16f,
        "all_16f_summary": _ratio_summary(pairs_16f),
        "sequence_16f_summaries": by_sequence_16f,
    }


def verify_bq4_trace_sequence_order_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_real_video_multiscene_bq4_trace_sequence_order":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    requested = int(report.get("requested_repeat_count") or 0)
    if requested < 1:
        errors.append("requested_repeat_count must be at least 1")
    sequences = report.get("sequences")
    if not isinstance(sequences, list) or len(sequences) < 2:
        errors.append("sequences must include at least two frame-order schedules")
        return errors
    sequence_specs: dict[str, list[int]] = {}
    for sequence in sequences:
        if not isinstance(sequence, dict):
            errors.append("sequence specs must be objects")
            continue
        name = str(sequence.get("name"))
        frames = [int(frame) for frame in sequence.get("frames", [])]
        if not name or not frames:
            errors.append(f"invalid sequence spec {sequence!r}")
            continue
        if 16 not in frames:
            errors.append(f"sequence {name} must include 16f target")
        sequence_specs[name] = frames
    profiles = report.get("trace_profiles")
    if not isinstance(profiles, list) or not profiles:
        errors.append("trace_profiles must be non-empty")
        return errors
    seen: set[tuple[str, int, int, str]] = set()
    for profile in profiles:
        if not isinstance(profile, dict):
            errors.append("trace profiles must be objects")
            continue
        sequence_name = str(profile.get("sequence_name"))
        repeat_index = int(profile.get("repeat_index") or 0)
        frames = int(profile.get("frames") or 0)
        policy = str(profile.get("policy"))
        label = f"{sequence_name} repeat {repeat_index} {frames}f {policy}"
        if profile.get("scene_id") != BQ4_SCENE_ID:
            errors.append(f"{label} must be the Bq4 scene")
        if sequence_name not in sequence_specs:
            errors.append(f"{label} sequence must be declared")
        elif frames not in sequence_specs[sequence_name]:
            errors.append(f"{label} frames must be in declared sequence")
        if policy not in {"cadence", "measured"}:
            errors.append(f"{label} policy must be cadence or measured")
        seen.add((sequence_name, repeat_index, frames, policy))
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
    for sequence_name, frames_list in sequence_specs.items():
        for repeat_index in range(requested):
            for frames in frames_list:
                for policy in ("cadence", "measured"):
                    if (sequence_name, repeat_index, frames, policy) not in seen:
                        errors.append(f"missing {sequence_name} repeat {repeat_index} {frames}f {policy} profile")
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
        errors.append("all expected Bq4 sequence steps must be traced")
    if summary.get("all_projective_interval_timing_present") is not True:
        errors.append("all traced sequence chunks must include projective interval timing")
    if summary.get("all_rows_cache_support_clean") is not True:
        errors.append("Bq4 sequence rows must remain cache/support clean")
    expected_16f_pairs = requested * len(sequence_specs)
    if int(summary.get("paired_16f_ratio_count") or 0) != expected_16f_pairs:
        errors.append(f"expected {expected_16f_pairs} paired 16f ratios")
    return errors


def assert_bq4_trace_sequence_order_report(report: dict[str, Any]) -> None:
    errors = verify_bq4_trace_sequence_order_report(report)
    if errors:
        raise AssertionError("Bq4 trace sequence order failed:\n- " + "\n- ".join(errors))


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
    required_frames = {int(frame) for sequence in DEFAULT_SEQUENCES for frame in sequence["frames"]}
    missing = sorted(required_frames.difference(specs))
    if missing:
        raise ValueError(f"missing Bq4 trace specs for frames: {missing}")
    segments = _load_segments(DEFAULT_SEGMENTS_MANIFEST)
    scene = segments[BQ4_SCENE_ID]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for sequence in DEFAULT_SEQUENCES:
        sequence_name = str(sequence["name"])
        frames_list = [int(frame) for frame in sequence["frames"]]
        for repeat_index in range(int(repeats)):
            for frame_index, frames in enumerate(frames_list):
                spec = specs[int(frames)]
                for policy in ("cadence", "measured"):
                    case_json = (
                        out_dir
                        / "cases"
                        / f"{BQ4_SCENE_ID}_{sequence_name}_repeat{repeat_index}_{frame_index}_{frames}f_{policy}.json"
                    )
                    row = run_case(
                        scene=scene,
                        frames=int(frames),
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
                    row["sequence_name"] = sequence_name
                    row["repeat_index"] = int(repeat_index)
                    row["sequence_frame_index"] = int(frame_index)
                    rows.append(row)
    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_multiscene_bq4_trace_sequence_order",
        "base_domain": "Bq4 traced mixed-frame sequence-order stability",
        "theory_contract": (
            "This report compares Bq4 mixed 4f/16f sequence orders to test whether "
            "the traced 16f timing caveat is a warm-state or launch-order effect."
        ),
        "source_shape_report": str(shape_report),
        "source_shape_status": shape.get("status"),
        "requested_repeat_count": int(repeats),
        "sequences": [{"name": str(item["name"]), "frames": [int(frame) for frame in item["frames"]]} for item in DEFAULT_SEQUENCES],
        "rows": rows,
    }
    report["trace_profiles"] = build_trace_profiles(rows)
    report["summary"] = summarize(report)
    errors = verify_bq4_trace_sequence_order_report(report)
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
        assert_bq4_trace_sequence_order_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report(
        out_dir=args.out_dir,
        shape_report=args.shape_report,
        repeats=int(args.repeats),
        verbose_trainer_output=bool(args.verbose_trainer_output),
    )
    if report.get("status") == "ok":
        assert_bq4_trace_sequence_order_report(report)
    path = write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
