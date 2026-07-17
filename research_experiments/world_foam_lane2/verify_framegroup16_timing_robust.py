#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PROMOTED_TAPE_MODE = "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse"
DEFAULT_EXPECTED_FRAMES = (16, 32, 64, 128)


def _parse_frame_counts(value: str) -> tuple[int, ...]:
    frames = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if len(frames) < 1:
        raise argparse.ArgumentTypeError("expected at least one frame count")
    if tuple(sorted(frames)) != frames:
        raise argparse.ArgumentTypeError("frame counts must be sorted ascending")
    return frames


def _parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def _parse_bool_expectation(value: str) -> tuple[str, bool]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected KEY=BOOL")
    key, raw_bool = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("expected non-empty KEY in KEY=BOOL")
    return key, _parse_bool(raw_bool)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0


def _step_ms(row: dict[str, Any], group: str, stat: str) -> float:
    try:
        value = row["step_summary"][group][stat]
    except KeyError as exc:
        raise KeyError(f"frame {row.get('frame_count')}: missing step_summary.{group}.{stat}") from exc
    if not _finite_positive(value):
        raise ValueError(f"frame {row.get('frame_count')}: step_summary.{group}.{stat} must be positive finite")
    return float(value) * 1000.0


def _ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator <= 0.0:
        return float("inf")
    return numerator / denominator


def _optional_storage_scale(rows: list[dict[str, Any]], key: str) -> float | None:
    if not all(row.get(key) is not None for row in rows):
        return None
    first = float(rows[0][key])
    last = float(rows[-1][key])
    if first == 0.0 and last == 0.0:
        return 0.0
    return _ratio(last, first)


def _optional_nonnegative_int(row: dict[str, Any], key: str, frame_count: int) -> int | None:
    value = row.get(key)
    if value is not None and (not isinstance(value, int) or value < 0):
        raise ValueError(f"frame {frame_count}: {key} must be nonnegative int")
    return value


def _summarize_group(row: dict[str, Any], group: str) -> dict[str, float]:
    mean_ms = _step_ms(row, group, "mean_s")
    median_ms = _step_ms(row, group, "median_s")
    max_ms = _step_ms(row, group, "max_s")
    summary = {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "max_ms": max_ms,
        "mean_to_median": _ratio(mean_ms, median_ms),
        "max_to_median": _ratio(max_ms, median_ms),
    }
    p90_s = row.get("step_summary", {}).get(group, {}).get("p90_s")
    if _finite_positive(p90_s):
        summary["p90_ms"] = float(p90_s) * 1000.0
    return summary


def _summarize_row(row: dict[str, Any]) -> dict[str, Any]:
    frame_count = row.get("frame_count")
    if not isinstance(frame_count, int):
        raise ValueError("row missing integer frame_count")
    storage = row.get("train_selected_tape_storage_bytes")
    if not isinstance(storage, int) or storage <= 0:
        raise ValueError(f"frame {frame_count}: train_selected_tape_storage_bytes must be positive int")
    topology_storage = _optional_nonnegative_int(row, "train_selected_tape_topology_storage_bytes", frame_count)
    coeff_storage = _optional_nonnegative_int(row, "train_endpoint_record_coeff_storage_bytes", frame_count)
    mps_resident_storage = _optional_nonnegative_int(
        row,
        "train_selected_tape_mps_resident_storage_bytes",
        frame_count,
    )
    mps_resident_noncoeff_storage = _optional_nonnegative_int(
        row,
        "train_selected_tape_mps_resident_noncoeff_storage_bytes",
        frame_count,
    )
    mps_resident_coeff_storage = _optional_nonnegative_int(
        row,
        "train_endpoint_record_coeff_mps_resident_storage_bytes",
        frame_count,
    )
    psnr = row.get("final_heldout_psnr")
    if psnr is not None and not isinstance(psnr, (int, float)):
        raise ValueError(f"frame {frame_count}: final_heldout_psnr must be numeric when present")
    return {
        "frame_count": frame_count,
        "row_status": row.get("status"),
        "total": _summarize_group(row, "total"),
        "backward": _summarize_group(row, "backward"),
        "storage_bytes": storage,
        "topology_storage_bytes": topology_storage,
        "coeff_storage_bytes": coeff_storage,
        "mps_resident_storage_bytes": mps_resident_storage,
        "mps_resident_noncoeff_storage_bytes": mps_resident_noncoeff_storage,
        "mps_resident_coeff_storage_bytes": mps_resident_coeff_storage,
        "heldout_psnr": float(psnr) if isinstance(psnr, (int, float)) and math.isfinite(float(psnr)) else None,
        "repeat_loaded_frames": bool(row.get("repeat_loaded_frames")),
        "repeat_loaded_frames_scope": str(row.get("repeat_loaded_frames_scope", "")),
    }


def _rows_by_frame(payload: dict[str, Any], failures: list[str]) -> dict[int, dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        failures.append("rows must be a list")
        return {}
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            failures.append("rows must contain only objects")
            continue
        frame_count = row.get("frame_count")
        if not isinstance(frame_count, int):
            failures.append("row missing integer frame_count")
            continue
        if frame_count in out:
            failures.append(f"duplicate frame_count {frame_count}")
            continue
        try:
            out[frame_count] = _summarize_row(row)
        except (KeyError, ValueError) as exc:
            failures.append(str(exc))
    return out


def _reference_rows_by_frame(path: Path | None, failures: list[str]) -> dict[int, dict[str, Any]]:
    if path is None:
        return {}
    try:
        payload = _load_json(path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        failures.append(f"{path}: could not load reference artifact: {exc}")
        return {}
    rows = payload.get("rows")
    if isinstance(rows, list):
        reference_failures: list[str] = []
        parsed = _rows_by_frame(payload, reference_failures)
        failures.extend(f"{path}: {failure}" for failure in reference_failures)
        return parsed
    if not isinstance(rows, dict):
        failures.append(f"{path}: reference rows must be a list or object keyed by frame count")
        return {}
    out: dict[int, dict[str, Any]] = {}
    for key, row in rows.items():
        try:
            frame_count = int(key)
        except (TypeError, ValueError):
            failures.append(f"{path}: reference row key {key!r} is not an integer frame count")
            continue
        if not isinstance(row, dict):
            failures.append(f"{path}: reference row {key!r} is not an object")
            continue
        total = row.get("total")
        backward = row.get("backward")
        if not isinstance(total, dict) or not isinstance(backward, dict):
            failures.append(f"{path}: reference row {frame_count} missing total/backward summaries")
            continue
        for group_name, group in (("total", total), ("backward", backward)):
            median_ms = group.get("median_ms")
            if not _finite_positive(median_ms):
                failures.append(f"{path}: reference row {frame_count} {group_name}.median_ms must be positive")
        if failures and failures[-1].startswith(f"{path}: reference row {frame_count}"):
            continue
        out[frame_count] = {
            "frame_count": frame_count,
            "total": total,
            "backward": backward,
            "storage_bytes": row.get("storage_bytes"),
            "topology_storage_bytes": row.get("topology_storage_bytes"),
            "coeff_storage_bytes": row.get("coeff_storage_bytes"),
            "mps_resident_storage_bytes": row.get("mps_resident_storage_bytes"),
            "mps_resident_noncoeff_storage_bytes": row.get("mps_resident_noncoeff_storage_bytes"),
            "mps_resident_coeff_storage_bytes": row.get("mps_resident_coeff_storage_bytes"),
            "heldout_psnr": row.get("heldout_psnr"),
        }
    return out


def _scale(rows: list[dict[str, Any]], group: str, stat: str) -> float:
    return _ratio(rows[-1][group][stat], rows[0][group][stat])


def _substituted_last_scale(
    rows: list[dict[str, Any]], confirm_row: dict[str, Any] | None, group: str, stat: str
) -> float | None:
    if confirm_row is None:
        return None
    return _ratio(confirm_row[group][stat], rows[0][group][stat])


def _load_confirm_row(path: Path | None, expected_frame: int, failures: list[str]) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        payload = _load_json(path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        failures.append(f"{path}: could not load confirmation artifact: {exc}")
        return None
    if payload.get("status") != "ok":
        failures.append(f"{path}: confirmation status is {payload.get('status')!r}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        failures.append(f"{path}: confirmation artifact must contain exactly one row")
        return None
    if rows[0].get("frame_count") != expected_frame:
        failures.append(f"{path}: confirmation frame_count must be {expected_frame}")
    try:
        return _summarize_row(rows[0])
    except (KeyError, ValueError) as exc:
        failures.append(f"{path}: {exc}")
        return None


def verify(args: argparse.Namespace) -> dict[str, Any]:
    payload_failures: list[str] = []
    confirm_failures: list[str] = []
    threshold_failures: list[str] = []
    contamination: list[str] = []
    reference_failures: list[str] = []

    try:
        payload = _load_json(args.artifact)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "failed",
            "artifact": str(args.artifact),
            "failures": [f"could not load artifact: {exc}"],
        }

    if payload.get("tape_mode") != args.expected_tape_mode:
        payload_failures.append(f"unexpected tape_mode {payload.get('tape_mode')!r}")
    if payload.get("status") != "ok":
        contamination.append(f"top-level status is {payload.get('status')!r}")
    benchmark_environment = payload.get("benchmark_environment")
    if isinstance(benchmark_environment, dict) and benchmark_environment.get("status") == "contended":
        contamination.append("benchmark_environment status is 'contended'")

    expected_payload_bools = dict(args.expect_payload_bool)
    for key, expected in expected_payload_bools.items():
        value = payload.get(key)
        if value is not expected:
            payload_failures.append(f"expected top-level {key}={expected}, got {value!r}")
    raw_rows = payload.get("rows")
    if isinstance(raw_rows, list):
        for row in raw_rows:
            if not isinstance(row, dict):
                continue
            frame_count = row.get("frame_count")
            for key, expected in expected_payload_bools.items():
                value = row.get(key)
                if value is not expected:
                    payload_failures.append(f"frame {frame_count}: expected {key}={expected}, got {value!r}")

    rows_by_frame = _rows_by_frame(payload, payload_failures)
    found_frames = tuple(sorted(rows_by_frame))
    if found_frames != args.expected_frames:
        payload_failures.append(f"frame counts {found_frames} did not match expected {args.expected_frames}")
    ordered_rows = [rows_by_frame[frame] for frame in args.expected_frames if frame in rows_by_frame]

    confirm_row = _load_confirm_row(args.confirm_artifact, args.expected_frames[-1], confirm_failures)
    reference_rows = _reference_rows_by_frame(args.reference_artifact, reference_failures)
    frame_scale = args.expected_frames[-1] / args.expected_frames[0]
    scales: dict[str, float] = {}
    substituted_scales: dict[str, float | None] = {}
    storage_scale = float("inf")
    topology_storage_scale: float | None = None
    coeff_storage_scale: float | None = None
    mps_resident_storage_scale: float | None = None
    mps_resident_noncoeff_storage_scale: float | None = None
    mps_resident_coeff_storage_scale: float | None = None

    has_scale = len(args.expected_frames) > 1

    if len(ordered_rows) == len(args.expected_frames):
        storage_scale = _ratio(float(ordered_rows[-1]["storage_bytes"]), float(ordered_rows[0]["storage_bytes"]))
        topology_storage_scale = _optional_storage_scale(ordered_rows, "topology_storage_bytes")
        coeff_storage_scale = _optional_storage_scale(ordered_rows, "coeff_storage_bytes")
        mps_resident_storage_scale = _optional_storage_scale(ordered_rows, "mps_resident_storage_bytes")
        mps_resident_noncoeff_storage_scale = _optional_storage_scale(
            ordered_rows,
            "mps_resident_noncoeff_storage_bytes",
        )
        mps_resident_coeff_storage_scale = _optional_storage_scale(
            ordered_rows,
            "mps_resident_coeff_storage_bytes",
        )
        if has_scale:
            for group in ("total", "backward"):
                for stat in ("mean_ms", "median_ms"):
                    key = f"{group}_{stat.removesuffix('_ms')}_scale"
                    scales[key] = _scale(ordered_rows, group, stat)
                    substituted_scales[key] = _substituted_last_scale(ordered_rows, confirm_row, group, stat)
            if storage_scale > args.max_storage_scale:
                threshold_failures.append(
                    f"storage scale {storage_scale:.3f} exceeds {args.max_storage_scale:.3f}"
                )
            if topology_storage_scale is not None and topology_storage_scale > args.max_topology_storage_scale:
                threshold_failures.append(
                    f"topology storage scale {topology_storage_scale:.3f} exceeds "
                    f"{args.max_topology_storage_scale:.3f}"
                )
            if coeff_storage_scale is not None and coeff_storage_scale > args.max_coeff_storage_scale:
                threshold_failures.append(
                    f"coefficient storage scale {coeff_storage_scale:.3f} exceeds "
                    f"{args.max_coeff_storage_scale:.3f}"
                )
            if (
                mps_resident_storage_scale is not None
                and mps_resident_storage_scale > args.max_mps_resident_storage_scale
            ):
                threshold_failures.append(
                    f"MPS resident storage scale {mps_resident_storage_scale:.3f} exceeds "
                    f"{args.max_mps_resident_storage_scale:.3f}"
                )
            if (
                mps_resident_noncoeff_storage_scale is not None
                and mps_resident_noncoeff_storage_scale > args.max_mps_resident_noncoeff_storage_scale
            ):
                threshold_failures.append(
                    f"MPS resident non-coefficient storage scale {mps_resident_noncoeff_storage_scale:.3f} exceeds "
                    f"{args.max_mps_resident_noncoeff_storage_scale:.3f}"
                )
            if (
                mps_resident_coeff_storage_scale is not None
                and mps_resident_coeff_storage_scale > args.max_mps_resident_coeff_storage_scale
            ):
                threshold_failures.append(
                    f"MPS resident coefficient storage scale {mps_resident_coeff_storage_scale:.3f} exceeds "
                    f"{args.max_mps_resident_coeff_storage_scale:.3f}"
                )
            for key, value in scales.items():
                max_scale = args.max_total_scale if key.startswith("total_") else args.max_backward_scale
                if value >= frame_scale:
                    threshold_failures.append(
                        f"{key} {value:.3f} is not sublinear versus frame scale {frame_scale:.3f}"
                    )
                if value > max_scale:
                    threshold_failures.append(f"{key} {value:.3f} exceeds {max_scale:.3f}")
        for row in ordered_rows:
            for group in ("total", "backward"):
                group_summary = row[group]
                mean_to_median = float(group_summary["mean_to_median"])
                max_to_median = float(group_summary["max_to_median"])
                if mean_to_median > args.max_row_mean_to_median:
                    contamination.append(
                        f"{row['frame_count']}f {group} mean/median {mean_to_median:.3f} "
                        f"exceeds {args.max_row_mean_to_median:.3f}"
                    )
                if max_to_median > args.max_row_max_to_median:
                    contamination.append(
                        f"{row['frame_count']}f {group} max/median {max_to_median:.3f} "
                        f"exceeds {args.max_row_max_to_median:.3f}"
                    )
                if row["row_status"] != "ok":
                    payload_failures.append(f"{row['frame_count']}f row status is {row['row_status']!r}")
        if args.reference_artifact is not None:
            found_reference_frames = tuple(sorted(frame for frame in reference_rows if frame in args.expected_frames))
            if found_reference_frames != args.expected_frames:
                reference_failures.append(
                    f"reference frame counts {found_reference_frames} did not match expected {args.expected_frames}"
                )
            for row in ordered_rows:
                reference_row = reference_rows.get(row["frame_count"])
                if reference_row is None:
                    continue
                for group in ("total", "backward"):
                    candidate_ms = float(row[group]["median_ms"])
                    reference_ms = float(reference_row[group]["median_ms"])
                    ratio = _ratio(candidate_ms, reference_ms)
                    max_ratio = (
                        args.max_reference_total_median_ratio
                        if group == "total"
                        else args.max_reference_backward_median_ratio
                    )
                    if ratio > max_ratio:
                        threshold_failures.append(
                            f"{row['frame_count']}f {group} median {candidate_ms:.3f} ms is "
                            f"{ratio:.3f}x reference {reference_ms:.3f} ms, exceeds {max_ratio:.3f}"
                        )

    confirm_clean = confirm_row is not None and not confirm_failures
    confirm_metrics: dict[str, Any] | None = None
    if confirm_row is not None:
        confirm_metrics = {
            "frame_count": confirm_row["frame_count"],
            "total": confirm_row["total"],
            "backward": confirm_row["backward"],
            "storage_bytes": confirm_row["storage_bytes"],
            "topology_storage_bytes": confirm_row.get("topology_storage_bytes"),
            "coeff_storage_bytes": confirm_row.get("coeff_storage_bytes"),
            "mps_resident_storage_bytes": confirm_row.get("mps_resident_storage_bytes"),
            "mps_resident_noncoeff_storage_bytes": confirm_row.get("mps_resident_noncoeff_storage_bytes"),
            "mps_resident_coeff_storage_bytes": confirm_row.get("mps_resident_coeff_storage_bytes"),
            "heldout_psnr": confirm_row["heldout_psnr"],
        }
        if confirm_row["total"]["median_ms"] > args.max_confirm_total_median_ms:
            confirm_failures.append(
                f"confirmation total median {confirm_row['total']['median_ms']:.3f} ms exceeds "
                f"{args.max_confirm_total_median_ms:.3f} ms"
            )
        if confirm_row["backward"]["median_ms"] > args.max_confirm_backward_median_ms:
            confirm_failures.append(
                f"confirmation backward median {confirm_row['backward']['median_ms']:.3f} ms exceeds "
                f"{args.max_confirm_backward_median_ms:.3f} ms"
            )
        if confirm_row["total"]["max_ms"] > args.max_confirm_total_max_ms:
            confirm_failures.append(
                f"confirmation total max {confirm_row['total']['max_ms']:.3f} ms exceeds "
                f"{args.max_confirm_total_max_ms:.3f} ms"
            )
        confirm_clean = not confirm_failures

    substituted_passes = confirm_clean and has_scale and len(ordered_rows) == len(args.expected_frames)
    if substituted_passes:
        for key, value in substituted_scales.items():
            max_scale = args.max_total_scale if key.startswith("total_") else args.max_backward_scale
            if value is None or value >= frame_scale or value > max_scale:
                substituted_passes = False
                break
        if storage_scale > args.max_storage_scale:
            substituted_passes = False
        if topology_storage_scale is not None and topology_storage_scale > args.max_topology_storage_scale:
            substituted_passes = False
        if coeff_storage_scale is not None and coeff_storage_scale > args.max_coeff_storage_scale:
            substituted_passes = False
        if (
            mps_resident_storage_scale is not None
            and mps_resident_storage_scale > args.max_mps_resident_storage_scale
        ):
            substituted_passes = False
        if (
            mps_resident_noncoeff_storage_scale is not None
            and mps_resident_noncoeff_storage_scale > args.max_mps_resident_noncoeff_storage_scale
        ):
            substituted_passes = False
        if (
            mps_resident_coeff_storage_scale is not None
            and mps_resident_coeff_storage_scale > args.max_mps_resident_coeff_storage_scale
        ):
            substituted_passes = False

    clean_speedscale = (
        not payload_failures
        and not threshold_failures
        and not confirm_failures
        and not reference_failures
        and not contamination
    )
    promoted_path_not_regressed = clean_speedscale or (args.allow_confirmed_outliers and substituted_passes)
    if clean_speedscale:
        status = "ok"
    elif promoted_path_not_regressed:
        status = "confirmed_outlier"
    else:
        status = "failed"

    row_summaries = {str(frame): rows_by_frame[frame] for frame in sorted(rows_by_frame)}
    return {
        "status": status,
        "artifact": str(args.artifact),
        "confirm_artifact": str(args.confirm_artifact) if args.confirm_artifact is not None else None,
        "expected_frames": list(args.expected_frames),
        "frame_scale": frame_scale,
        "payload_status": payload.get("status"),
        "tape_mode": payload.get("tape_mode"),
        "expected_payload_bools": expected_payload_bools,
        "clean_speedscale_artifact": clean_speedscale,
        "promoted_path_not_regressed": promoted_path_not_regressed,
        "confirmed_outlier_allowed": bool(args.allow_confirmed_outliers),
        "scales": scales,
        "substituted_last_frame_scales": substituted_scales,
        "storage_scale": storage_scale,
        "topology_storage_scale": topology_storage_scale,
        "coeff_storage_scale": coeff_storage_scale,
        "mps_resident_storage_scale": mps_resident_storage_scale,
        "mps_resident_noncoeff_storage_scale": mps_resident_noncoeff_storage_scale,
        "mps_resident_coeff_storage_scale": mps_resident_coeff_storage_scale,
        "rows": row_summaries,
        "reference_artifact": str(args.reference_artifact) if args.reference_artifact is not None else None,
        "reference_rows": {str(frame): reference_rows[frame] for frame in sorted(reference_rows)},
        "confirm_row": confirm_metrics,
        "contamination": contamination,
        "failures": payload_failures + threshold_failures + confirm_failures + reference_failures,
        "thresholds": {
            "max_total_scale": args.max_total_scale,
            "max_backward_scale": args.max_backward_scale,
            "max_storage_scale": args.max_storage_scale,
            "max_topology_storage_scale": args.max_topology_storage_scale,
            "max_coeff_storage_scale": args.max_coeff_storage_scale,
            "max_mps_resident_storage_scale": args.max_mps_resident_storage_scale,
            "max_mps_resident_noncoeff_storage_scale": args.max_mps_resident_noncoeff_storage_scale,
            "max_mps_resident_coeff_storage_scale": args.max_mps_resident_coeff_storage_scale,
            "max_row_mean_to_median": args.max_row_mean_to_median,
            "max_row_max_to_median": args.max_row_max_to_median,
            "max_confirm_total_median_ms": args.max_confirm_total_median_ms,
            "max_confirm_backward_median_ms": args.max_confirm_backward_median_ms,
            "max_confirm_total_max_ms": args.max_confirm_total_max_ms,
            "max_reference_total_median_ratio": args.max_reference_total_median_ratio,
            "max_reference_backward_median_ratio": args.max_reference_backward_median_ratio,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Robustly classify WorldFoam framegroup16 timing artifacts with median/outlier checks."
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--confirm-artifact", type=Path, default=None)
    parser.add_argument("--reference-artifact", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    parser.add_argument("--expected-frames", type=_parse_frame_counts, default=DEFAULT_EXPECTED_FRAMES)
    parser.add_argument("--expected-tape-mode", default=PROMOTED_TAPE_MODE)
    parser.add_argument(
        "--expect-payload-bool",
        action="append",
        type=_parse_bool_expectation,
        default=[],
        metavar="KEY=BOOL",
        help="Require a top-level boolean artifact field and every row's matching field to equal BOOL.",
    )
    parser.add_argument("--max-total-scale", type=float, default=2.0)
    parser.add_argument("--max-backward-scale", type=float, default=2.0)
    parser.add_argument("--max-storage-scale", type=float, default=1.10)
    parser.add_argument("--max-topology-storage-scale", type=float, default=1.10)
    parser.add_argument("--max-coeff-storage-scale", type=float, default=1.10)
    parser.add_argument("--max-mps-resident-storage-scale", type=float, default=1.10)
    parser.add_argument("--max-mps-resident-noncoeff-storage-scale", type=float, default=1.10)
    parser.add_argument("--max-mps-resident-coeff-storage-scale", type=float, default=1.10)
    parser.add_argument("--max-row-mean-to-median", type=float, default=2.5)
    parser.add_argument("--max-row-max-to-median", type=float, default=8.0)
    parser.add_argument("--max-confirm-total-median-ms", type=float, default=8.0)
    parser.add_argument("--max-confirm-backward-median-ms", type=float, default=8.0)
    parser.add_argument("--max-confirm-total-max-ms", type=float, default=12.0)
    parser.add_argument("--max-reference-total-median-ratio", type=float, default=1.20)
    parser.add_argument("--max-reference-backward-median-ratio", type=float, default=1.20)
    parser.add_argument(
        "--allow-confirmed-outliers",
        action="store_true",
        help=(
            "Exit successfully when the full sweep is contaminated but a separate clean max-frame "
            "confirmation keeps the promoted path from being classified as regressed."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = verify(args)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(encoded, encoding="utf-8")
    print(encoded, end="")
    return 0 if result["status"] != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
