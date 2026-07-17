#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "research_experiments" / "world_foam_lane2" / "results"
GATE4_AFFINE_CANDIDATE_NUM32_DEN16_FUSED_MSE_MODE = "gate4-affine-candidate-num32-den16-fused-mse"
GATE4_AFFINE_CANDIDATE_NUM32_DEN16_TRACK_MSE_FUSED_MSE_MODE = (
    "gate4-affine-candidate-num32-den16-trackmse-fused-mse"
)
GATE4_AFFINE_CANDIDATE_COEFF16_FUSED_MSE_MODE = "gate4-affine-candidate-coeff16-fused-mse"
GATE4_AFFINE_CANDIDATE_COEFF16_CAP224_FUSED_MSE_MODE = (
    "gate4-affine-candidate-coeff16-cap224-fused-mse"
)
GATE4_AFFINE_CANDIDATE_COEFF16_DENSITYMASK_FUSED_MSE_MODE = (
    "gate4-affine-candidate-coeff16-densitymask-fused-mse"
)
GATE4_AFFINE_CANDIDATE_COEFF16_SAMPLE_REDUCE_FUSED_MSE_MODE = (
    "gate4-affine-candidate-coeff16-samplereduce-fused-mse"
)
GATE4_AFFINE_CANDIDATE_COEFF16_SORTNET_FUSED_MSE_MODE = (
    "gate4-affine-candidate-coeff16-sortnet-fused-mse"
)
GATE4_AFFINE_CANDIDATE_COEFF16_FRAMEGROUP16_CACHED_FUSED_MSE_MODE = (
    "gate4-affine-candidate-coeff16-framegroup16cached-fused-mse"
)
GATE4_AFFINE_CANDIDATE_COEFF16_SITECACHE_FUSED_MSE_MODE = (
    "gate4-affine-candidate-coeff16-sitecache-fused-mse"
)
GATE4_AFFINE_CANDIDATE_COEFF16_OWNERUPDATE_FUSED_MSE_MODE = (
    "gate4-affine-candidate-coeff16-ownerupdate-fused-mse"
)
GATE4_AFFINE_CANDIDATE_COEFF16_OWNERUPDATE_I16_FUSED_MSE_MODE = (
    "gate4-affine-candidate-coeff16-ownerupdate-i16-fused-mse"
)
GATE4_AFFINE_CANDIDATE_COEFF16_OWNERKEEP_FUSED_MSE_MODE = (
    "gate4-affine-candidate-coeff16-ownerkeep-fused-mse"
)
GATE4_AFFINE_CANDIDATE_COEFF16_OWNERKEEP_I16_FUSED_MSE_MODE = (
    "gate4-affine-candidate-coeff16-ownerkeep-i16-fused-mse"
)
GATE4_AFFINE_CANDIDATE_COEFF16_TRACK_MSE_FUSED_MSE_MODE = (
    "gate4-affine-candidate-coeff16-trackmse-fused-mse"
)
GATE4_AFFINE_CANDIDATE_FUSED_MSE_MODES = {
    GATE4_AFFINE_CANDIDATE_NUM32_DEN16_FUSED_MSE_MODE,
    GATE4_AFFINE_CANDIDATE_NUM32_DEN16_TRACK_MSE_FUSED_MSE_MODE,
    GATE4_AFFINE_CANDIDATE_COEFF16_FUSED_MSE_MODE,
    GATE4_AFFINE_CANDIDATE_COEFF16_CAP224_FUSED_MSE_MODE,
    GATE4_AFFINE_CANDIDATE_COEFF16_DENSITYMASK_FUSED_MSE_MODE,
    GATE4_AFFINE_CANDIDATE_COEFF16_SAMPLE_REDUCE_FUSED_MSE_MODE,
    GATE4_AFFINE_CANDIDATE_COEFF16_SORTNET_FUSED_MSE_MODE,
    GATE4_AFFINE_CANDIDATE_COEFF16_FRAMEGROUP16_CACHED_FUSED_MSE_MODE,
    GATE4_AFFINE_CANDIDATE_COEFF16_SITECACHE_FUSED_MSE_MODE,
    GATE4_AFFINE_CANDIDATE_COEFF16_OWNERUPDATE_FUSED_MSE_MODE,
    GATE4_AFFINE_CANDIDATE_COEFF16_OWNERUPDATE_I16_FUSED_MSE_MODE,
    GATE4_AFFINE_CANDIDATE_COEFF16_OWNERKEEP_FUSED_MSE_MODE,
    GATE4_AFFINE_CANDIDATE_COEFF16_OWNERKEEP_I16_FUSED_MSE_MODE,
    GATE4_AFFINE_CANDIDATE_COEFF16_TRACK_MSE_FUSED_MSE_MODE,
}
TAPE_MODE = GATE4_AFFINE_CANDIDATE_COEFF16_FUSED_MSE_MODE
DEFAULT_ARTIFACT = (
    RESULTS_DIR
    / "2026-05-19_gate4_affine_candidate_coeff16_samplemse_scale_2_4_8_16_render16_site24_warm3.json"
)


def _parse_int_list(value: str) -> tuple[int, ...]:
    out = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not out:
        raise ValueError("expected at least one frame count")
    return out


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _finite(value: Any) -> bool:
    return isinstance(value, (float, int)) and math.isfinite(float(value))


def _positive_finite(value: Any) -> bool:
    return _finite(value) and float(value) > 0.0


def _ratio(last: float, first: float) -> float:
    if abs(first) <= 1.0e-12:
        return 0.0 if abs(last) <= 1.0e-12 else float("inf")
    return last / first


def _step_stat(row: dict[str, Any], phase: str, stat: str) -> float | None:
    summary = row.get("step_summary")
    if not isinstance(summary, dict):
        return None
    phase_summary = summary.get(phase)
    if not isinstance(phase_summary, dict):
        return None
    value = phase_summary.get(stat)
    return float(value) if _finite(value) else None


def _benchmark_status(payload: dict[str, Any]) -> str:
    environment = payload.get("benchmark_environment")
    if not isinstance(environment, dict):
        return "missing"
    status = environment.get("status")
    return str(status) if isinstance(status, str) else "missing"


def _row_float(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    return float(value) if _finite(value) else None


def _is_coeff16_mode(tape_mode: str) -> bool:
    return "coeff16" in tape_mode


def _is_trackmse_mode(tape_mode: str) -> bool:
    return "trackmse" in tape_mode


def _is_sample_reduce_mode(tape_mode: str) -> bool:
    return "samplereduce" in tape_mode


def _is_cap224_mode(tape_mode: str) -> bool:
    return "cap224" in tape_mode


def _is_densitymask_mode(tape_mode: str) -> bool:
    return "densitymask" in tape_mode


def _is_sortnet_mode(tape_mode: str) -> bool:
    return "sortnet" in tape_mode


def _is_framegroup16_cached_mode(tape_mode: str) -> bool:
    return "framegroup16cached" in tape_mode


def _is_sitecache_mode(tape_mode: str) -> bool:
    return "sitecache" in tape_mode


def _is_ownerupdate_mode(tape_mode: str) -> bool:
    return "ownerupdate" in tape_mode and "ownerupdate-i16" not in tape_mode


def _is_ownerupdate_i16_mode(tape_mode: str) -> bool:
    return "ownerupdate-i16" in tape_mode


def _is_ownerkeep_mode(tape_mode: str) -> bool:
    return "ownerkeep" in tape_mode and "ownerkeep-i16" not in tape_mode


def _is_ownerkeep_i16_mode(tape_mode: str) -> bool:
    return "ownerkeep-i16" in tape_mode


def _check_mode_flags(
    *,
    failures: list[str],
    scope: str,
    obj: dict[str, Any],
    expected_tape_mode: str,
) -> None:
    expected_coeff16 = _is_coeff16_mode(expected_tape_mode)
    expected_trackmse = _is_trackmse_mode(expected_tape_mode)
    expected_cap224 = _is_cap224_mode(expected_tape_mode)
    expected_densitymask = _is_densitymask_mode(expected_tape_mode)
    expected_sample_reduce = _is_sample_reduce_mode(expected_tape_mode)
    expected_sortnet = _is_sortnet_mode(expected_tape_mode)
    expected_framegroup16_cached = _is_framegroup16_cached_mode(expected_tape_mode)
    expected_sitecache = _is_sitecache_mode(expected_tape_mode)
    expected_ownerupdate = _is_ownerupdate_mode(expected_tape_mode)
    expected_ownerupdate_i16 = _is_ownerupdate_i16_mode(expected_tape_mode)
    expected_ownerkeep = _is_ownerkeep_mode(expected_tape_mode)
    expected_ownerkeep_i16 = _is_ownerkeep_i16_mode(expected_tape_mode)
    if bool(obj.get("gate4_affine_candidate_csr_coeff16_fused_mse", False)) != expected_coeff16:
        failures.append(f"{scope}: coeff16 fused-MSE flag did not match tape_mode {expected_tape_mode!r}")
    if bool(obj.get("gate4_affine_candidate_csr_trackmse_fused_mse", False)) != expected_trackmse:
        failures.append(f"{scope}: trackmse fused-MSE flag did not match tape_mode {expected_tape_mode!r}")
    if bool(obj.get("gate4_affine_candidate_csr_cap224_fused_mse", False)) != expected_cap224:
        failures.append(f"{scope}: cap224 fused-MSE flag did not match tape_mode {expected_tape_mode!r}")
    if bool(obj.get("gate4_affine_candidate_csr_densitymask_fused_mse", False)) != expected_densitymask:
        failures.append(f"{scope}: densitymask fused-MSE flag did not match tape_mode {expected_tape_mode!r}")
    if bool(obj.get("gate4_affine_candidate_csr_sample_reduce_fused_mse", False)) != expected_sample_reduce:
        failures.append(f"{scope}: sample-reduce fused-MSE flag did not match tape_mode {expected_tape_mode!r}")
    if bool(obj.get("gate4_affine_candidate_csr_sortnet_fused_mse", False)) != expected_sortnet:
        failures.append(f"{scope}: sortnet fused-MSE flag did not match tape_mode {expected_tape_mode!r}")
    if (
        bool(obj.get("gate4_affine_candidate_csr_framegroup16_cached_fused_mse", False))
        != expected_framegroup16_cached
    ):
        failures.append(
            f"{scope}: framegroup16 cached fused-MSE flag did not match tape_mode {expected_tape_mode!r}"
        )
    if bool(obj.get("gate4_affine_candidate_csr_sitecache_fused_mse", False)) != expected_sitecache:
        failures.append(f"{scope}: sitecache fused-MSE flag did not match tape_mode {expected_tape_mode!r}")
    if bool(obj.get("gate4_affine_candidate_csr_ownerupdate_fused_mse", False)) != expected_ownerupdate:
        failures.append(f"{scope}: ownerupdate fused-MSE flag did not match tape_mode {expected_tape_mode!r}")
    if bool(obj.get("gate4_affine_candidate_csr_ownerupdate_i16_fused_mse", False)) != expected_ownerupdate_i16:
        failures.append(f"{scope}: ownerupdate-i16 fused-MSE flag did not match tape_mode {expected_tape_mode!r}")
    if bool(obj.get("gate4_affine_candidate_csr_ownerkeep_fused_mse", False)) != expected_ownerkeep:
        failures.append(f"{scope}: ownerkeep fused-MSE flag did not match tape_mode {expected_tape_mode!r}")
    if bool(obj.get("gate4_affine_candidate_csr_ownerkeep_i16_fused_mse", False)) != expected_ownerkeep_i16:
        failures.append(f"{scope}: ownerkeep-i16 fused-MSE flag did not match tape_mode {expected_tape_mode!r}")


def verify(args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    contamination: list[str] = []
    artifact = Path(args.artifact)
    expected_tape_mode = str(args.tape_mode)
    if expected_tape_mode not in GATE4_AFFINE_CANDIDATE_FUSED_MSE_MODES:
        return {
            "status": "failed",
            "artifact": str(artifact),
            "failures": [f"unsupported tape_mode {expected_tape_mode!r}"],
            "contamination": [],
        }
    try:
        payload = _load_json(artifact)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "failed",
            "artifact": str(artifact),
            "failures": [f"could not load artifact: {exc}"],
            "contamination": [],
        }

    expected_frames = _parse_int_list(str(args.frame_counts))
    if payload.get("benchmark") != "world_foam_lane2_segment_tape_train_eval_mps":
        failures.append(f"unexpected benchmark {payload.get('benchmark')!r}")
    if payload.get("status") != "ok":
        failures.append(f"artifact status is {payload.get('status')!r}")
    if payload.get("tape_mode") != expected_tape_mode:
        failures.append(f"unexpected tape_mode {payload.get('tape_mode')!r}")
    if payload.get("gate4_affine_candidate_csr_fused_mse") is not True:
        failures.append("gate4_affine_candidate_csr_fused_mse must be true")
    _check_mode_flags(failures=failures, scope="artifact", obj=payload, expected_tape_mode=expected_tape_mode)
    if payload.get("endpoint_record_source") != "gate4-affine":
        failures.append("endpoint_record_source must be gate4-affine")
    if payload.get("optimizer_mode") != "manual-vjp":
        failures.append("candidate CSR fused-MSE gate must use optimizer_mode manual-vjp")
    if payload.get("full_trainer_claim") is not False:
        failures.append("full_trainer_claim must be false")
    if payload.get("full_geometry_gradient_claim") is not False:
        failures.append("full_geometry_gradient_claim must be false")
    if payload.get("quality_claim") is not False:
        failures.append("quality_claim must be false")
    if payload.get("render_size") != args.render_size:
        failures.append(f"render_size {payload.get('render_size')} did not match {args.render_size}")
    if payload.get("site_count") != args.site_count:
        failures.append(f"site_count {payload.get('site_count')} did not match {args.site_count}")
    if tuple(payload.get("frame_counts", ())) != expected_frames:
        failures.append(f"frame_counts {payload.get('frame_counts')} did not match {list(expected_frames)}")

    benchmark_status = _benchmark_status(payload)
    if benchmark_status != "background":
        contamination.append(f"benchmark_environment status is {benchmark_status!r}")
        if not args.allow_contended:
            failures.append(f"benchmark_environment status is {benchmark_status!r}")

    rows = payload.get("rows")
    rows_by_frame: dict[int, dict[str, Any]] = {}
    if not isinstance(rows, list):
        failures.append("rows must be a list")
    else:
        for row in rows:
            if not isinstance(row, dict):
                failures.append("row is not an object")
                continue
            frame = row.get("frame_count")
            if not isinstance(frame, int):
                failures.append("row missing integer frame_count")
                continue
            rows_by_frame[frame] = row
    found_frames = tuple(sorted(rows_by_frame))
    if found_frames != expected_frames:
        failures.append(f"row frames {found_frames} did not match required {expected_frames}")

    total_means: list[float] = []
    backward_means: list[float] = []
    total_medians: list[float] = []
    backward_medians: list[float] = []
    storage_values: list[float] = []
    noncoeff_storage_values: list[float] = []
    candidate_counts: list[float] = []
    row_summaries: dict[str, dict[str, Any]] = {}

    for frame in expected_frames:
        row = rows_by_frame.get(frame)
        if row is None:
            continue
        if row.get("status") != "ok":
            failures.append(f"frame {frame}: status is {row.get('status')!r}")
        if row.get("tape_mode") != expected_tape_mode:
            failures.append(f"frame {frame}: unexpected tape_mode {row.get('tape_mode')!r}")
        if row.get("gate4_affine_candidate_csr_fused_mse") is not True:
            failures.append(f"frame {frame}: gate4 affine candidate flag is not true")
        _check_mode_flags(
            failures=failures,
            scope=f"frame {frame}",
            obj=row,
            expected_tape_mode=expected_tape_mode,
        )
        if row.get("endpoint_record_source") != "gate4-affine":
            failures.append(f"frame {frame}: endpoint_record_source must be gate4-affine")
        if row.get("render_size") != args.render_size:
            failures.append(f"frame {frame}: render_size {row.get('render_size')} did not match {args.render_size}")
        if row.get("site_count") != args.site_count:
            failures.append(f"frame {frame}: site_count {row.get('site_count')} did not match {args.site_count}")

        acceptance = row.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append(f"frame {frame}: missing acceptance map")
        else:
            for key, value in sorted(acceptance.items()):
                if value is not True:
                    failures.append(f"frame {frame}: acceptance {key} is not true")

        for key in ("final_train_psnr", "final_heldout_psnr", "first_grad_abs_sum", "parameter_update_abs_max"):
            if not _positive_finite(row.get(key)):
                failures.append(f"frame {frame}: {key} must be positive finite")
        if _row_float(row, "final_train_psnr") is not None and float(row["final_train_psnr"]) < args.min_train_psnr:
            failures.append(f"frame {frame}: final_train_psnr below {args.min_train_psnr}")
        if _row_float(row, "final_heldout_psnr") is not None and float(row["final_heldout_psnr"]) < args.min_heldout_psnr:
            failures.append(f"frame {frame}: final_heldout_psnr below {args.min_heldout_psnr}")

        for phase, means, medians in (
            ("total", total_means, total_medians),
            ("backward", backward_means, backward_medians),
        ):
            mean_s = _step_stat(row, phase, "mean_s")
            median_s = _step_stat(row, phase, "median_s")
            max_s = _step_stat(row, phase, "max_s")
            if mean_s is None or mean_s <= 0.0:
                failures.append(f"frame {frame}: step_summary.{phase}.mean_s must be positive finite")
            else:
                means.append(mean_s)
            if median_s is None or median_s <= 0.0:
                failures.append(f"frame {frame}: step_summary.{phase}.median_s must be positive finite")
            else:
                medians.append(median_s)
            if mean_s is not None and median_s is not None and median_s > 0.0:
                mean_to_median = mean_s / median_s
                if mean_to_median > args.max_row_mean_to_median:
                    failures.append(
                        f"frame {frame}: {phase} mean/median {mean_to_median:.3f} "
                        f"exceeds {args.max_row_mean_to_median:.3f}"
                    )
            if max_s is not None and median_s is not None and median_s > 0.0:
                max_to_median = max_s / median_s
                if max_to_median > args.max_row_max_to_median:
                    failures.append(
                        f"frame {frame}: {phase} max/median {max_to_median:.3f} "
                        f"exceeds {args.max_row_max_to_median:.3f}"
                    )

        storage = row.get("train_selected_tape_mps_resident_storage_bytes")
        noncoeff = row.get("train_selected_tape_mps_resident_noncoeff_storage_bytes")
        if not isinstance(storage, int) or storage <= 0:
            failures.append(f"frame {frame}: train selected resident storage must be positive")
        else:
            storage_values.append(float(storage))
        if not isinstance(noncoeff, int) or noncoeff <= 0:
            failures.append(f"frame {frame}: train selected resident noncoeff storage must be positive")
        else:
            noncoeff_storage_values.append(float(noncoeff))
        metadata = row.get("gate4_endpoint_train_metadata")
        candidate_count = metadata.get("candidate_count") if isinstance(metadata, dict) else None
        max_candidates = metadata.get("max_candidates_per_row") if isinstance(metadata, dict) else None
        if not isinstance(candidate_count, int) or candidate_count <= 0:
            failures.append(f"frame {frame}: candidate_count must be positive")
        else:
            candidate_counts.append(float(candidate_count))
        if not isinstance(max_candidates, int) or max_candidates <= 0:
            failures.append(f"frame {frame}: max_candidates_per_row must be positive")
        elif max_candidates > args.max_candidates_per_row:
            failures.append(
                f"frame {frame}: max_candidates_per_row {max_candidates} exceeds {args.max_candidates_per_row}"
            )

        row_summaries[str(frame)] = {
            "status": row.get("status"),
            "heldout_psnr": row.get("final_heldout_psnr"),
            "train_psnr": row.get("final_train_psnr"),
            "candidate_count": candidate_count,
            "max_candidates_per_row": max_candidates,
            "storage_bytes": storage,
            "noncoeff_storage_bytes": noncoeff,
            "total_mean_ms": mean_s * 1000.0 if (mean_s := _step_stat(row, "total", "mean_s")) else None,
            "total_median_ms": median_s * 1000.0 if (median_s := _step_stat(row, "total", "median_s")) else None,
            "backward_mean_ms": mean_s * 1000.0 if (mean_s := _step_stat(row, "backward", "mean_s")) else None,
            "backward_median_ms": median_s * 1000.0 if (median_s := _step_stat(row, "backward", "median_s")) else None,
        }

    frame_scale = expected_frames[-1] / expected_frames[0]
    total_scale = _ratio(total_means[-1], total_means[0]) if len(total_means) == len(expected_frames) else float("inf")
    backward_scale = (
        _ratio(backward_means[-1], backward_means[0])
        if len(backward_means) == len(expected_frames)
        else float("inf")
    )
    total_median_scale = (
        _ratio(total_medians[-1], total_medians[0])
        if len(total_medians) == len(expected_frames)
        else float("inf")
    )
    backward_median_scale = (
        _ratio(backward_medians[-1], backward_medians[0])
        if len(backward_medians) == len(expected_frames)
        else float("inf")
    )
    storage_scale = (
        _ratio(storage_values[-1], storage_values[0])
        if len(storage_values) == len(expected_frames)
        else float("inf")
    )
    noncoeff_storage_scale = (
        _ratio(noncoeff_storage_values[-1], noncoeff_storage_values[0])
        if len(noncoeff_storage_values) == len(expected_frames)
        else float("inf")
    )
    candidate_scale = (
        _ratio(candidate_counts[-1], candidate_counts[0])
        if len(candidate_counts) == len(expected_frames)
        else float("inf")
    )

    scale_checks = (
        ("total_step_scale", total_scale, args.max_total_scale),
        ("backward_scale", backward_scale, args.max_backward_scale),
        ("total_median_scale", total_median_scale, args.max_total_median_scale),
        ("backward_median_scale", backward_median_scale, args.max_backward_median_scale),
        ("resident_storage_scale", storage_scale, args.max_storage_scale),
        ("resident_noncoeff_storage_scale", noncoeff_storage_scale, args.max_noncoeff_storage_scale),
        ("candidate_count_scale", candidate_scale, args.max_candidate_scale),
    )
    for name, value, threshold in scale_checks:
        if not math.isfinite(value) or value > threshold:
            failures.append(f"{name} {value:.3f} exceeds {threshold:.3f}")
    scale_gate_required = expected_frames[-1] > expected_frames[0]
    if scale_gate_required and total_scale >= frame_scale:
        failures.append(f"total scale {total_scale:.3f} is not sublinear versus frame scale {frame_scale:.3f}")
    if scale_gate_required and backward_scale >= frame_scale:
        failures.append(f"backward scale {backward_scale:.3f} is not sublinear versus frame scale {frame_scale:.3f}")

    result = {
        "status": "ok" if not failures else "failed",
        "artifact": str(artifact),
        "tape_mode": expected_tape_mode,
        "failures": failures,
        "contamination": contamination,
        "benchmark_environment_status": benchmark_status,
        "frame_scale": frame_scale,
        "scale_gate_required": scale_gate_required,
        "total_step_scale": total_scale,
        "backward_scale": backward_scale,
        "total_median_scale": total_median_scale,
        "backward_median_scale": backward_median_scale,
        "resident_storage_scale": storage_scale,
        "resident_noncoeff_storage_scale": noncoeff_storage_scale,
        "candidate_count_scale": candidate_scale,
        "rows": row_summaries,
    }
    if args.out_json is not None:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Gate4 affine candidate CSR fused-MSE train/eval artifacts.")
    parser.add_argument("artifact", nargs="?", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--tape-mode", default=TAPE_MODE, choices=sorted(GATE4_AFFINE_CANDIDATE_FUSED_MSE_MODES))
    parser.add_argument("--frame-counts", default="2,4,8,16")
    parser.add_argument("--render-size", type=int, default=16)
    parser.add_argument("--site-count", type=int, default=24)
    parser.add_argument("--min-train-psnr", type=float, default=8.0)
    parser.add_argument("--min-heldout-psnr", type=float, default=8.0)
    parser.add_argument("--max-total-scale", type=float, default=1.25)
    parser.add_argument("--max-backward-scale", type=float, default=1.25)
    parser.add_argument("--max-total-median-scale", type=float, default=1.25)
    parser.add_argument("--max-backward-median-scale", type=float, default=1.25)
    parser.add_argument("--max-storage-scale", type=float, default=1.10)
    parser.add_argument("--max-noncoeff-storage-scale", type=float, default=1.10)
    parser.add_argument("--max-candidate-scale", type=float, default=1.10)
    parser.add_argument("--max-candidates-per-row", type=int, default=256)
    parser.add_argument("--max-row-mean-to-median", type=float, default=2.0)
    parser.add_argument("--max-row-max-to-median", type=float, default=4.0)
    parser.add_argument(
        "--allow-contended",
        action="store_true",
        help="Do not fail the artifact only because benchmark_environment.status is contended.",
    )
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def main() -> int:
    result = verify(parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
