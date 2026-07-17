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
    / "2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_rerun"
)
DEFAULT_SHAPE_REPORT = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-05-25_star_uvt_projective_real_video_multiscene_extended_render_forward_shape"
    / "summary.json"
)
BQ4_SCENE_ID = "Bq4rmeIvJbs_seg_000"
TIMING_KEYS = (
    "feature_state_update_ms",
    "feature_render_ms",
    "alpha_state_update_ms",
    "alpha_render_ms",
    "projective_interval_render_ms",
)


def _finite_float(value: Any, label: str, errors: list[str]) -> float:
    if not isinstance(value, int | float) or not math.isfinite(float(value)):
        errors.append(f"{label} must be finite, got {value!r}")
        return 0.0
    return float(value)


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return math.inf
    return float(numerator) / float(denominator)


def _trace_specs_from_shape(shape: dict[str, Any]) -> list[dict[str, Any]]:
    specs = []
    for row in shape.get("pair_profiles", []):
        if not isinstance(row, dict):
            continue
        if row.get("scene_id") != BQ4_SCENE_ID:
            continue
        if not bool(row.get("no_first_timing_miss")) or not bool(row.get("render_forward_timing_miss")):
            continue
        specs.append(
            {
                "scene_id": BQ4_SCENE_ID,
                "frames": int(row["frames"]),
                "trace_global_step": int(row["render_forward_spike_step_index"]) + 1,
                "source_render_forward_ratio": float(row["render_forward_ratio"]),
                "source_render_forward_spike_delta_ms": float(row["render_forward_spike_delta_ms"]),
                "source_render_forward_drop_spike_ratio": float(
                    row["render_forward_drop_largest_positive_delta_ratio"]
                ),
            }
        )
    return sorted(specs, key=lambda item: int(item["frames"]))


def _trace_profile(case_path: Path, expected_global_step: int, row: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(case_path.read_text(encoding="utf-8"))
    traces = payload.get("chunk_traces") or []
    matching = [
        trace
        for trace in traces
        if isinstance(trace, dict) and int(trace.get("global_step") or -1) == int(expected_global_step)
    ]
    trace = matching[0] if matching else {}
    chunks = trace.get("chunks") if isinstance(trace, dict) else None
    chunk_profiles = []
    if isinstance(chunks, list):
        for chunk in chunks:
            if not isinstance(chunk, dict):
                continue
            timing = chunk.get("projective_interval_timing_ms")
            if not isinstance(timing, dict):
                continue
            timing_sum = sum(float(timing.get(key) or 0.0) for key in TIMING_KEYS[:-1])
            total = float(timing.get("projective_interval_render_ms") or 0.0)
            render_forward_ms = float(chunk.get("render_forward_ms") or 0.0)
            chunk_profiles.append(
                {
                    "frame_start": int(chunk.get("frame_start") or 0),
                    "chunk_frames": int(chunk.get("chunk_frames") or 0),
                    "render_forward_ms": render_forward_ms,
                    "projective_interval_timing_ms": {key: float(timing.get(key) or 0.0) for key in TIMING_KEYS},
                    "subtiming_sum_ms": timing_sum,
                    "subtiming_total_abs_delta_ms": abs(timing_sum - total),
                    "projective_total_to_chunk_render_ratio": _ratio(total, render_forward_ms),
                }
            )
    return {
        "scene_id": str(row["scene_id"]),
        "frames": int(row["frames"]),
        "policy": str(row["policy"]),
        "case_json": str(case_path),
        "expected_trace_global_step": int(expected_global_step),
        "chunk_trace_global_steps": [int(step) for step in payload.get("chunk_trace_global_steps") or []],
        "matching_trace_count": len(matching),
        "chunk_profile_count": len(chunk_profiles),
        "chunk_profiles": chunk_profiles,
        "projective_interval_cache_rebuilds": int(row.get("projective_interval_cache_rebuilds") or 0),
        "projective_interval_cache_live_updates": int(row.get("projective_interval_cache_live_updates") or 0),
        "projective_interval_cache_support_rebins": int(row.get("projective_interval_cache_support_rebins") or 0),
        "projective_interval_cache_stale_refreshes": int(row.get("projective_interval_cache_stale_refreshes") or 0),
        "projective_interval_cache_fallback_marks": int(row.get("projective_interval_cache_fallback_marks") or 0),
        "tile_overflow_sum": int(row.get("tile_overflow_sum") or 0),
        "no_first_step_ms": float(row.get("no_first_step_ms") or 0.0),
        "mean_render_forward_ms": float(row.get("mean_render_forward_ms") or 0.0),
    }


def build_trace_profiles(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _trace_profile(Path(row["case_json"]), int(row["expected_trace_global_step"]), row)
        for row in rows
    ]


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    profiles = report.get("trace_profiles") or []
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
    profile_by_key = {
        (int(profile["frames"]), str(profile["policy"])): profile
        for profile in profiles
        if isinstance(profile, dict)
    }
    measured_vs_cadence_no_first_ratios: list[float] = []
    measured_vs_cadence_projective_total_ratios: list[float] = []
    measured_vs_cadence_feature_state_update_ratios: list[float] = []
    for frames in sorted({key[0] for key in profile_by_key}):
        cadence = profile_by_key.get((frames, "cadence"))
        measured = profile_by_key.get((frames, "measured"))
        if cadence is None or measured is None:
            continue
        cadence_chunks = cadence.get("chunk_profiles") or []
        measured_chunks = measured.get("chunk_profiles") or []
        if not cadence_chunks or not measured_chunks:
            continue
        cadence_timing = cadence_chunks[0]["projective_interval_timing_ms"]
        measured_timing = measured_chunks[0]["projective_interval_timing_ms"]
        measured_vs_cadence_no_first_ratios.append(
            _ratio(float(measured.get("no_first_step_ms") or 0.0), float(cadence.get("no_first_step_ms") or 0.0))
        )
        measured_vs_cadence_projective_total_ratios.append(
            _ratio(
                float(measured_timing.get("projective_interval_render_ms") or 0.0),
                float(cadence_timing.get("projective_interval_render_ms") or 0.0),
            )
        )
        measured_vs_cadence_feature_state_update_ratios.append(
            _ratio(
                float(measured_timing.get("feature_state_update_ms") or 0.0),
                float(cadence_timing.get("feature_state_update_ms") or 0.0),
            )
        )
    return {
        "source_shape_status": report.get("source_shape_status"),
        "source_shape_pair_count": int((report.get("source_shape_summary") or {}).get("pair_count") or 0),
        "trace_spec_count": len(report.get("trace_specs") or []),
        "row_count": len(report.get("rows") or []),
        "trace_profile_count": len(profiles),
        "chunk_profile_count": len(chunk_profiles),
        "all_expected_global_steps_traced": all(
            int(profile.get("matching_trace_count") or 0) == 1 for profile in profiles
        ),
        "all_chunk_profiles_present": all(
            int(profile.get("chunk_profile_count") or 0) >= 1 for profile in profiles
        ),
        "all_projective_interval_timing_present": all(
            all(key in chunk.get("projective_interval_timing_ms", {}) for key in TIMING_KEYS)
            for chunk in chunk_profiles
        ),
        "min_projective_interval_subtiming_ms": min(timing_values) if timing_values else 0.0,
        "max_projective_interval_subtiming_ms": max(timing_values) if timing_values else 0.0,
        "max_subtiming_total_abs_delta_ms": max(
            float(chunk["subtiming_total_abs_delta_ms"]) for chunk in chunk_profiles
        )
        if chunk_profiles
        else 0.0,
        "max_projective_total_to_chunk_render_ratio": max(
            float(chunk["projective_total_to_chunk_render_ratio"]) for chunk in chunk_profiles
        )
        if chunk_profiles
        else 0.0,
        "traced_measured_vs_cadence_no_first_step_ms_ratios": measured_vs_cadence_no_first_ratios,
        "max_traced_measured_vs_cadence_no_first_step_ms_ratio": max(measured_vs_cadence_no_first_ratios)
        if measured_vs_cadence_no_first_ratios
        else 0.0,
        "traced_measured_vs_cadence_projective_total_ratios": measured_vs_cadence_projective_total_ratios,
        "max_traced_measured_vs_cadence_projective_total_ratio": max(
            measured_vs_cadence_projective_total_ratios
        )
        if measured_vs_cadence_projective_total_ratios
        else 0.0,
        "traced_measured_vs_cadence_feature_state_update_ratios": measured_vs_cadence_feature_state_update_ratios,
        "max_traced_measured_vs_cadence_feature_state_update_ratio": max(
            measured_vs_cadence_feature_state_update_ratios
        )
        if measured_vs_cadence_feature_state_update_ratios
        else 0.0,
        "traced_bq4_spike_reproduced": max(measured_vs_cadence_no_first_ratios)
        >= 1.0
        if measured_vs_cadence_no_first_ratios
        else False,
        "all_rows_cache_support_clean": all(
            int(profile.get("projective_interval_cache_support_rebins") or 0) == 0
            and int(profile.get("projective_interval_cache_stale_refreshes") or 0) == 0
            and int(profile.get("projective_interval_cache_fallback_marks") or 0) == 0
            and int(profile.get("tile_overflow_sum") or 0) == 0
            for profile in profiles
        ),
    }


def verify_bq4_trace_rerun_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("status") != "ok":
        errors.append(f"status must be ok, got {report.get('status')!r}")
    if report.get("benchmark") != "star_uvt_projective_real_video_multiscene_bq4_trace_rerun":
        errors.append(f"unexpected benchmark {report.get('benchmark')!r}")
    theory_contract = report.get("theory_contract")
    if (
        not isinstance(theory_contract, str)
        or "Bq4" not in theory_contract
        or "trace_global_steps" not in theory_contract
        or "substep attribution" not in theory_contract
    ):
        errors.append("theory_contract must preserve Bq4 traced-rerun scope")
    trace_specs = report.get("trace_specs")
    profiles = report.get("trace_profiles")
    if not isinstance(trace_specs, list) or len(trace_specs) < 2:
        errors.append("trace_specs must include the Bq4 4f and 16f miss specs")
    if not isinstance(profiles, list) or not profiles:
        errors.append("trace_profiles must be non-empty")
        return errors
    for profile in profiles:
        if not isinstance(profile, dict):
            errors.append("trace profiles must be objects")
            continue
        label = f"{profile.get('scene_id')} {profile.get('frames')}f {profile.get('policy')}"
        if profile.get("scene_id") != BQ4_SCENE_ID:
            errors.append(f"{label} must be the Bq4 scene")
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
            if _finite_float(chunk.get("subtiming_total_abs_delta_ms"), f"{label} timing total delta", errors) > 1.0e-5:
                errors.append(f"{label} subtiming total must match sum of substeps")
            _finite_float(
                chunk.get("projective_total_to_chunk_render_ratio"),
                f"{label} projective/chunk render ratio",
                errors,
            )
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
    if summary.get("all_expected_global_steps_traced") is not True:
        errors.append("all expected Bq4 spike steps must be traced")
    if summary.get("all_projective_interval_timing_present") is not True:
        errors.append("all traced chunks must include projective interval timing")
    if summary.get("all_rows_cache_support_clean") is not True:
        errors.append("Bq4 trace rerun rows must remain cache/support clean")
    if summary.get("traced_bq4_spike_reproduced") is not False:
        errors.append("Bq4 traced rerun should record that the saved spike did not reproduce")
    return errors


def assert_bq4_trace_rerun_report(report: dict[str, Any]) -> None:
    errors = verify_bq4_trace_rerun_report(report)
    if errors:
        raise AssertionError("Bq4 trace rerun failed:\n- " + "\n- ".join(errors))


def run_report(
    *,
    out_dir: Path = DEFAULT_OUT_DIR,
    shape_report: Path = DEFAULT_SHAPE_REPORT,
    verbose_trainer_output: bool = False,
) -> dict[str, Any]:
    if not DEFAULT_SEGMENTS_MANIFEST.exists():
        return {"status": "skipped", "reason": f"missing segment manifest: {DEFAULT_SEGMENTS_MANIFEST}", "rows": []}
    if not torch.backends.mps.is_available():
        return {"status": "skipped", "reason": "MPS unavailable", "rows": []}
    if not has_projective_trace_cell_interval_metal() or not has_projective_trace_cell_interval_backward_metal():
        return {"status": "skipped", "reason": "projective interval Metal ops unavailable", "rows": []}
    shape = json.loads(shape_report.read_text(encoding="utf-8"))
    trace_specs = _trace_specs_from_shape(shape)
    segments = _load_segments(DEFAULT_SEGMENTS_MANIFEST)
    scene = segments[BQ4_SCENE_ID]
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for spec in trace_specs:
        for policy in ("cadence", "measured"):
            case_json = out_dir / "cases" / f"{BQ4_SCENE_ID}_{int(spec['frames'])}f_{policy}_trace.json"
            row = run_case(
                scene=scene,
                frames=int(spec["frames"]),
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
            rows.append(row)
    report: dict[str, Any] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_multiscene_bq4_trace_rerun",
        "base_domain": "Bq4 spike-step traced rerun selected from saved render-forward shape artifact",
        "theory_contract": (
            "This Bq4 trace_global_steps rerun adds substep attribution for the saved single-spike "
            "render-forward misses. It is a diagnostic for render-forward substeps, not broad timing acceptance."
        ),
        "source_shape_report": str(shape_report),
        "source_shape_status": shape.get("status"),
        "source_shape_summary": shape.get("summary", {}),
        "trace_specs": trace_specs,
        "rows": rows,
    }
    report["trace_profiles"] = build_trace_profiles(rows)
    report["summary"] = summarize(report)
    errors = verify_bq4_trace_rerun_report(report)
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
    parser.add_argument("--verify-report", type=Path)
    parser.add_argument("--verbose-trainer-output", action="store_true")
    args = parser.parse_args()

    if args.verify_report is not None:
        report = json.loads(args.verify_report.read_text(encoding="utf-8"))
        assert_bq4_trace_rerun_report(report)
        print(f"verified {args.verify_report}")
        return

    report = run_report(
        out_dir=args.out_dir,
        shape_report=args.shape_report,
        verbose_trainer_output=bool(args.verbose_trainer_output),
    )
    if report.get("status") == "ok":
        assert_bq4_trace_rerun_report(report)
    path = write_report(report, args.out_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
