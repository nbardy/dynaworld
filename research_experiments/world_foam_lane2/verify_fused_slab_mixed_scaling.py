#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "research_experiments" / "world_foam_lane2" / "results"

DEFAULT_TRAIN_EVAL_ARTIFACTS = (
    RESULTS_DIR / "2026-05-15_fused_slab_mixed_train_eval_forwardfix_reduce_chunk16_render32_2_4_8_16.json",
    RESULTS_DIR / "2026-05-15_fused_slab_mixed_train_eval_forwardfix_direct_atomic_render32_2_4_8_16.json",
    RESULTS_DIR / "2026-05-15_fused_slab_mixed_train_eval_gradonly_direct_atomic_render32_2_4_8_16.json",
    RESULTS_DIR / "2026-05-15_fused_slab_mixed_train_eval_rgbonly_direct_atomic_render32_2_4_8_16.json",
    RESULTS_DIR / "2026-05-15_fused_slab_mixed_train_eval_track_direct_atomic_render32_2_4_8_16.json",
)
DEFAULT_SMOKE_ARTIFACTS = (
    RESULTS_DIR / "2026-05-15_fused_slab_mixed_direct_atomic_track_smoke_2f_render16_pertrack.json",
    RESULTS_DIR / "2026-05-15_fused_slab_mixed_vjp_rgba_depth_smoke_2f_render16_pertrack.json",
)
DEFAULT_FRAMEGROUP_LOSSREDUCE_ARTIFACT = (
    RESULTS_DIR
    / "2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_lossreduce_fused_mse_repeat_loaded_warm5_steps12_render32_site12_16_32_64_128.json"
)
DEFAULT_FRAMEGROUP_128ONLY_ARTIFACT = (
    RESULTS_DIR
    / "2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_lossreduce_128only_rerun2_warm10_steps20_render32_site12.json"
)
DEFAULT_FRAMEGROUP_COMPARE_SMOKE_ARTIFACT = (
    RESULTS_DIR
    / "2026-05-16_endpoint_run_vs_delta_framegroup16_compare_speedscale_render32_site12_warm3_steps8_16_32_64_128.json"
)
DEFAULT_FRAMEGROUP_REAL32_COMPARE_ARTIFACT = (
    RESULTS_DIR
    / "2026-05-16_endpoint_run_vs_delta_framegroup16_compare_real32_render32_site12_warm3_steps8_16_32.json"
)
DEFAULT_FRAMEGROUP_I16X4_COMPARE_ARTIFACT = (
    RESULTS_DIR
    / "2026-05-16_delta_framegroup_i16x3_i16x4_train_eval_compare_repeat32_warm1_steps3_render32_site12_16_32.json"
)
DEFAULT_REQUIRED_MODES = (
    "reduce",
    "direct_atomic",
    "direct_atomic_grad_only",
    "direct_atomic_rgb_only",
    "direct_atomic_track",
)
DEFAULT_MAX_REALRAY_BOUNDARIES = 128
VJP_DIAGNOSTIC_KEYS = {
    "direct_atomic": "mixed_vjp_direct_diagnostics",
    "direct_atomic_grad_only": "mixed_vjp_direct_grad_only_diagnostics",
    "direct_atomic_rgb_only": "mixed_vjp_direct_rgb_only_diagnostics",
    "direct_atomic_track": "mixed_vjp_direct_track_diagnostics",
}


def _parse_int_list(value: str) -> tuple[int, ...]:
    out = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not out:
        raise ValueError("expected at least one frame count")
    return out


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _mean_s(row: dict[str, Any], key: str) -> float:
    try:
        return float(row["step_summary"][key]["mean_s"])
    except KeyError as exc:
        raise KeyError(f"missing step_summary.{key}.mean_s in frame {row.get('frame_count')}") from exc


def _geo_mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot compute geometric mean of an empty list")
    return math.exp(sum(math.log(max(value, 1.0e-12)) for value in values) / len(values))


def _step_stat_ms(row: dict[str, Any], key: str, stat: str) -> float:
    try:
        return float(row["step_summary"][key][stat]) * 1000.0
    except KeyError as exc:
        raise KeyError(f"missing step_summary.{key}.{stat} in frame {row.get('frame_count')}") from exc


def _row_scale(rows: list[dict[str, Any]], key: str, stat: str = "mean_s") -> float:
    first = _step_stat_ms(rows[0], key, stat)
    last = _step_stat_ms(rows[-1], key, stat)
    return last / first


def _validate_framegroup_lossreduce_payloads(
    *,
    train_eval_path: Path,
    confirm_path: Path,
    frame_counts: tuple[int, ...],
    max_total_scale: float,
    max_backward_scale: float,
    max_storage_scale: float,
    max_mixed_128_total_max_ms: float,
    max_128only_total_median_ms: float,
    max_128only_total_max_ms: float,
    max_128only_backward_median_ms: float,
    failures: list[str],
) -> dict[str, Any]:
    try:
        payload = _load_json(train_eval_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        failures.append(f"{train_eval_path}: could not load framegroup loss-reduce artifact: {exc}")
        return {}
    try:
        confirm = _load_json(confirm_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        failures.append(f"{confirm_path}: could not load framegroup 128-only artifact: {exc}")
        return {}

    if payload.get("status") != "ok":
        failures.append(f"{train_eval_path}: framegroup loss-reduce status is {payload.get('status')!r}")
    if payload.get("tape_mode") != "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse":
        failures.append(f"{train_eval_path}: unexpected tape_mode {payload.get('tape_mode')!r}")
    if payload.get("render_size") != 32:
        failures.append(f"{train_eval_path}: render_size must remain 32")
    if payload.get("site_count") != 12:
        failures.append(f"{train_eval_path}: site_count must remain 12")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        failures.append(f"{train_eval_path}: rows must be a list")
        return {}
    rows_by_frame: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            failures.append(f"{train_eval_path}: row is not an object")
            continue
        frame_count = row.get("frame_count")
        if not isinstance(frame_count, int):
            failures.append(f"{train_eval_path}: row missing integer frame_count")
            continue
        rows_by_frame[frame_count] = row
        if row.get("status") != "ok":
            failures.append(f"{train_eval_path}: frame {frame_count} status is {row.get('status')!r}")
        acceptance = row.get("acceptance")
        if not isinstance(acceptance, dict) or any(value is not True for value in acceptance.values()):
            failures.append(f"{train_eval_path}: frame {frame_count} acceptance has non-true values")
        for key in ("total", "backward"):
            for stat in ("mean_s", "median_s", "max_s"):
                try:
                    value = _step_stat_ms(row, key, stat)
                except KeyError as exc:
                    failures.append(f"{train_eval_path}: {exc}")
                    continue
                if not math.isfinite(value) or value <= 0.0:
                    failures.append(f"{train_eval_path}: frame {frame_count} {key}.{stat} is not positive finite")
        storage = row.get("train_selected_tape_storage_bytes")
        if not isinstance(storage, int) or storage <= 0:
            failures.append(f"{train_eval_path}: frame {frame_count} train_selected_tape_storage_bytes invalid")
        psnr = row.get("final_heldout_psnr")
        if not isinstance(psnr, (float, int)) or not math.isfinite(float(psnr)):
            failures.append(f"{train_eval_path}: frame {frame_count} final_heldout_psnr is not finite")

    found_frames = tuple(sorted(rows_by_frame))
    if found_frames != frame_counts:
        failures.append(f"{train_eval_path}: frame counts {found_frames} did not match required {frame_counts}")
    ordered_rows = [rows_by_frame[frame] for frame in frame_counts if frame in rows_by_frame]
    if len(ordered_rows) != len(frame_counts):
        return {}

    total_scale = _row_scale(ordered_rows, "total")
    backward_scale = _row_scale(ordered_rows, "backward")
    storage_scale = (
        float(ordered_rows[-1]["train_selected_tape_storage_bytes"])
        / float(ordered_rows[0]["train_selected_tape_storage_bytes"])
    )
    mixed_128_total_max_ms = _step_stat_ms(ordered_rows[-1], "total", "max_s")
    mixed_128_backward_max_ms = _step_stat_ms(ordered_rows[-1], "backward", "max_s")
    frame_scale = frame_counts[-1] / frame_counts[0]
    if total_scale >= frame_scale:
        failures.append(f"{train_eval_path}: total scale {total_scale:.3f} is not sublinear versus {frame_scale:.3f}")
    if total_scale > max_total_scale:
        failures.append(f"{train_eval_path}: total scale {total_scale:.3f} exceeds {max_total_scale:.3f}")
    if backward_scale > max_backward_scale:
        failures.append(f"{train_eval_path}: backward scale {backward_scale:.3f} exceeds {max_backward_scale:.3f}")
    if storage_scale > max_storage_scale:
        failures.append(f"{train_eval_path}: storage scale {storage_scale:.3f} exceeds {max_storage_scale:.3f}")
    if mixed_128_total_max_ms > max_mixed_128_total_max_ms:
        failures.append(
            f"{train_eval_path}: 128f mixed-sweep total max {mixed_128_total_max_ms:.3f} ms "
            f"exceeds {max_mixed_128_total_max_ms:.3f} ms"
        )

    confirm_rows = confirm.get("rows")
    if confirm.get("status") != "ok":
        failures.append(f"{confirm_path}: framegroup 128-only status is {confirm.get('status')!r}")
    if confirm.get("render_size") != 32:
        failures.append(f"{confirm_path}: render_size must remain 32")
    if confirm.get("site_count") != 12:
        failures.append(f"{confirm_path}: site_count must remain 12")
    if not isinstance(confirm_rows, list) or len(confirm_rows) != 1 or not isinstance(confirm_rows[0], dict):
        failures.append(f"{confirm_path}: expected exactly one 128-only row")
        confirm_row: dict[str, Any] | None = None
    else:
        confirm_row = confirm_rows[0]
        if confirm_row.get("frame_count") != frame_counts[-1]:
            failures.append(f"{confirm_path}: confirmation frame_count must be {frame_counts[-1]}")
        if confirm_row.get("status") != "ok":
            failures.append(f"{confirm_path}: confirmation row status is {confirm_row.get('status')!r}")

    confirm_metrics: dict[str, float] = {}
    if confirm_row is not None:
        total_median_ms = _step_stat_ms(confirm_row, "total", "median_s")
        total_max_ms = _step_stat_ms(confirm_row, "total", "max_s")
        backward_median_ms = _step_stat_ms(confirm_row, "backward", "median_s")
        confirm_metrics = {
            "total_mean_ms": _step_stat_ms(confirm_row, "total", "mean_s"),
            "total_median_ms": total_median_ms,
            "total_max_ms": total_max_ms,
            "backward_mean_ms": _step_stat_ms(confirm_row, "backward", "mean_s"),
            "backward_median_ms": backward_median_ms,
            "backward_max_ms": _step_stat_ms(confirm_row, "backward", "max_s"),
        }
        if total_median_ms > max_128only_total_median_ms:
            failures.append(
                f"{confirm_path}: 128-only total median {total_median_ms:.3f} ms "
                f"exceeds {max_128only_total_median_ms:.3f} ms"
            )
        if total_max_ms > max_128only_total_max_ms:
            failures.append(
                f"{confirm_path}: 128-only total max {total_max_ms:.3f} ms "
                f"exceeds {max_128only_total_max_ms:.3f} ms"
            )
        if backward_median_ms > max_128only_backward_median_ms:
            failures.append(
                f"{confirm_path}: 128-only backward median {backward_median_ms:.3f} ms "
                f"exceeds {max_128only_backward_median_ms:.3f} ms"
            )

    return {
        "train_eval_path": str(train_eval_path),
        "confirm_path": str(confirm_path),
        "frame_counts": list(frame_counts),
        "total_scale_first_to_last": total_scale,
        "backward_scale_first_to_last": backward_scale,
        "storage_scale_first_to_last": storage_scale,
        "mixed_128_total_max_ms": mixed_128_total_max_ms,
        "mixed_128_backward_max_ms": mixed_128_backward_max_ms,
        "mixed_rows": {
            str(row["frame_count"]): {
                "total_mean_ms": _step_stat_ms(row, "total", "mean_s"),
                "total_median_ms": _step_stat_ms(row, "total", "median_s"),
                "total_max_ms": _step_stat_ms(row, "total", "max_s"),
                "backward_mean_ms": _step_stat_ms(row, "backward", "mean_s"),
                "backward_median_ms": _step_stat_ms(row, "backward", "median_s"),
                "backward_max_ms": _step_stat_ms(row, "backward", "max_s"),
                "storage_bytes": int(row["train_selected_tape_storage_bytes"]),
                "heldout_psnr": float(row["final_heldout_psnr"]),
            }
            for row in ordered_rows
        },
        "confirm_128only": confirm_metrics,
    }


def _summary_metric(summary_by_mode: dict[str, Any], *, mode: str, key: str) -> float:
    mode_summary = summary_by_mode.get(mode)
    if not isinstance(mode_summary, dict):
        raise KeyError(f"missing summary for mode {mode}")
    value = mode_summary.get(key)
    if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        raise KeyError(f"missing finite {key} for mode {mode}")
    return float(value)


def _validate_framegroup_compare_smoke(
    *,
    path: Path,
    frame_counts: tuple[int, ...],
    max_total_ratio_16f: float,
    max_backward_ratio_16f: float,
    max_psnr_delta: float,
    max_storage_vs_full_16f: float,
    max_total_scale: float,
    max_backward_scale: float,
    max_storage_scale: float,
    max_total_ratio_all_frames: float,
    max_psnr_delta_all_frames: float,
    expected_render_size: int,
    expected_site_count: int,
    failures: list[str],
) -> dict[str, Any]:
    try:
        payload = _load_json(path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        failures.append(f"{path}: could not load framegroup compare smoke artifact: {exc}")
        return {}

    framegroup_mode = "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse"
    required_modes = ("endpoint-run", "endpoint-record-edit", framegroup_mode)
    if payload.get("status") != "ok":
        failures.append(f"{path}: compare smoke status is {payload.get('status')!r}")
    if payload.get("allow_repeat_loaded_frames") is not True or payload.get("repeat_loaded_frames") is not True:
        failures.append(f"{path}: compare smoke must preserve repeated-loaded-frame scope")
    scope = str(payload.get("scope", ""))
    if "not a stable benchmark" not in scope or "repeated loaded frames" not in scope:
        failures.append(f"{path}: compare smoke scope must keep benchmark and repeated-fixture caveats")
    conclusion = str(payload.get("conclusion", ""))
    if "Delta-framegroup16 fused-MSE is faster than endpoint-run" not in conclusion:
        failures.append(f"{path}: compare smoke conclusion must record framegroup speed win")

    acceptance = payload.get("acceptance")
    required_acceptance = (
        "endpoint_run_ok",
        "endpoint_record_edit_ok",
        "endpoint_record_delta_framegroup16_fused_mse_ok",
        "psnr_matches",
        "delta_framegroup16_psnr_matches",
        "delta_framegroup16_storage_below_endpoint",
        "delta_framegroup16_total_ratio_positive",
        "delta_framegroup16_total_not_slower_than_endpoint",
    )
    if not isinstance(acceptance, dict):
        failures.append(f"{path}: compare smoke missing acceptance map")
    else:
        for key in required_acceptance:
            if acceptance.get(key) is not True:
                failures.append(f"{path}: compare smoke acceptance {key} must be true")

    ratios = payload.get("ratios")
    if not isinstance(ratios, dict):
        failures.append(f"{path}: compare smoke missing ratios")
        ratios = {}
    ratio_total = float(ratios.get("delta_framegroup16_to_endpoint_total_16f", float("inf")))
    ratio_backward = float(ratios.get("delta_framegroup16_to_endpoint_backward_16f", float("inf")))
    if not math.isfinite(ratio_total) or ratio_total > max_total_ratio_16f:
        failures.append(
            f"{path}: framegroup compare total ratio {ratio_total:.3f} exceeds {max_total_ratio_16f:.3f}"
        )
    if not math.isfinite(ratio_backward) or ratio_backward > max_backward_ratio_16f:
        failures.append(
            f"{path}: framegroup compare backward ratio {ratio_backward:.3f} exceeds {max_backward_ratio_16f:.3f}"
        )

    summary_16f = payload.get("summary_16f")
    summary_by_frame = payload.get("summary_by_frame")
    if not isinstance(summary_16f, dict):
        failures.append(f"{path}: compare smoke missing summary_16f")
        summary_16f = {}
    if not isinstance(summary_by_frame, dict):
        failures.append(f"{path}: compare smoke missing summary_by_frame")
        summary_by_frame = {}

    try:
        endpoint_total = _summary_metric(summary_16f, mode="endpoint-run", key="total_ms")
        endpoint_backward = _summary_metric(summary_16f, mode="endpoint-run", key="backward_ms")
        endpoint_psnr = _summary_metric(summary_16f, mode="endpoint-run", key="heldout_psnr")
        framegroup_total = _summary_metric(summary_16f, mode=framegroup_mode, key="total_ms")
        framegroup_backward = _summary_metric(summary_16f, mode=framegroup_mode, key="backward_ms")
        framegroup_psnr = _summary_metric(summary_16f, mode=framegroup_mode, key="heldout_psnr")
        framegroup_storage = _summary_metric(summary_16f, mode=framegroup_mode, key="storage_vs_full")
    except KeyError as exc:
        failures.append(f"{path}: {exc}")
        return {}

    recomputed_total_ratio = framegroup_total / endpoint_total
    recomputed_backward_ratio = framegroup_backward / endpoint_backward
    if abs(recomputed_total_ratio - ratio_total) > 1.0e-6:
        failures.append(f"{path}: stored total ratio does not match summary_16f")
    if abs(recomputed_backward_ratio - ratio_backward) > 1.0e-6:
        failures.append(f"{path}: stored backward ratio does not match summary_16f")
    if abs(framegroup_psnr - endpoint_psnr) > max_psnr_delta:
        failures.append(
            f"{path}: framegroup PSNR delta {abs(framegroup_psnr - endpoint_psnr):.6g} exceeds {max_psnr_delta}"
        )
    if framegroup_storage > max_storage_vs_full_16f:
        failures.append(
            f"{path}: framegroup 16f storage_vs_full {framegroup_storage:.3f} "
            f"exceeds {max_storage_vs_full_16f:.3f}"
        )

    found_frames = tuple(sorted(int(frame) for frame in summary_by_frame if str(frame).isdigit()))
    if found_frames != frame_counts:
        failures.append(f"{path}: compare smoke frames {found_frames} did not match required {frame_counts}")
    ratios_by_frame: dict[str, dict[str, float]] = {}
    psnr_delta_by_frame: dict[str, float] = {}
    for frame_count in frame_counts:
        frame_summary = summary_by_frame.get(str(frame_count))
        if not isinstance(frame_summary, dict):
            failures.append(f"{path}: compare smoke missing frame {frame_count} summary")
            continue
        for mode in required_modes:
            if mode not in frame_summary:
                failures.append(f"{path}: compare smoke missing {mode} at frame {frame_count}")
        try:
            frame_endpoint_total = _summary_metric(frame_summary, mode="endpoint-run", key="total_ms")
            frame_framegroup_total = _summary_metric(frame_summary, mode=framegroup_mode, key="total_ms")
            frame_endpoint_backward = _summary_metric(frame_summary, mode="endpoint-run", key="backward_ms")
            frame_framegroup_backward = _summary_metric(frame_summary, mode=framegroup_mode, key="backward_ms")
            frame_endpoint_psnr = _summary_metric(frame_summary, mode="endpoint-run", key="heldout_psnr")
            frame_framegroup_psnr = _summary_metric(frame_summary, mode=framegroup_mode, key="heldout_psnr")
        except KeyError as exc:
            failures.append(f"{path}: frame {frame_count}: {exc}")
            continue
        frame_total_ratio = frame_framegroup_total / frame_endpoint_total
        frame_backward_ratio = frame_framegroup_backward / frame_endpoint_backward
        frame_psnr_delta = abs(frame_framegroup_psnr - frame_endpoint_psnr)
        ratios_by_frame[str(frame_count)] = {
            "total": frame_total_ratio,
            "backward": frame_backward_ratio,
        }
        psnr_delta_by_frame[str(frame_count)] = frame_psnr_delta
        if frame_total_ratio > max_total_ratio_all_frames:
            failures.append(
                f"{path}: frame {frame_count} framegroup total ratio {frame_total_ratio:.3f} "
                f"exceeds {max_total_ratio_all_frames:.3f}"
            )
        if frame_psnr_delta > max_psnr_delta_all_frames:
            failures.append(
                f"{path}: frame {frame_count} framegroup PSNR delta {frame_psnr_delta:.6g} "
                f"exceeds {max_psnr_delta_all_frames}"
            )

    results = payload.get("results")
    fg_result = results.get(framegroup_mode) if isinstance(results, dict) else None
    frame_scale = total_scale = backward_scale = storage_scale = float("nan")
    loaded_frame_count: int | None = None
    real_loaded_frame_counts: list[int] = []
    repeated_frame_counts: list[int] = []
    repeat_scope_by_frame: dict[str, str] = {}
    if not isinstance(fg_result, dict):
        failures.append(f"{path}: compare smoke missing framegroup result payload")
    else:
        if fg_result.get("status") != "ok":
            failures.append(f"{path}: framegroup result status is {fg_result.get('status')!r}")
        if fg_result.get("full_trainer_claim") is not False or fg_result.get("quality_claim") is not False:
            failures.append(f"{path}: framegroup result must not claim full trainer or quality")
        fg_acceptance = fg_result.get("acceptance")
        if not isinstance(fg_acceptance, dict) or any(value is not True for value in fg_acceptance.values()):
            failures.append(f"{path}: framegroup result acceptance has non-true values")
        if tuple(fg_result.get("frame_counts", ())) != frame_counts:
            failures.append(f"{path}: framegroup result frame_counts must be {frame_counts}")
        if fg_result.get("render_size") != expected_render_size:
            failures.append(f"{path}: framegroup result render_size must be {expected_render_size}")
        if fg_result.get("site_count") != expected_site_count:
            failures.append(f"{path}: framegroup result site_count must be {expected_site_count}")
        fg_rows = fg_result.get("rows")
        if not isinstance(fg_rows, list):
            failures.append(f"{path}: framegroup result rows must be a list")
        else:
            rows_by_frame = {
                int(row["frame_count"]): row
                for row in fg_rows
                if isinstance(row, dict) and isinstance(row.get("frame_count"), int)
            }
            if tuple(sorted(rows_by_frame)) != frame_counts:
                failures.append(f"{path}: framegroup row frame counts must be {frame_counts}")
            loaded_counts = {
                int(row.get("loaded_frame_count"))
                for row in rows_by_frame.values()
                if isinstance(row.get("loaded_frame_count"), int)
            }
            if len(loaded_counts) != 1:
                failures.append(f"{path}: framegroup rows must share one loaded_frame_count")
            elif loaded_counts:
                loaded_frame_count = loaded_counts.pop()
                if loaded_frame_count != frame_counts[0]:
                    failures.append(
                        f"{path}: framegroup loaded_frame_count {loaded_frame_count} must equal first frame count "
                        f"{frame_counts[0]}"
                    )
            for frame_count, row in sorted(rows_by_frame.items()):
                repeated = bool(row.get("repeat_loaded_frames"))
                repeat_scope_by_frame[str(frame_count)] = str(row.get("repeat_loaded_frames_scope", ""))
                if loaded_frame_count is None:
                    continue
                if frame_count <= loaded_frame_count:
                    if repeated:
                        failures.append(f"{path}: frame {frame_count} must remain a real-loaded row")
                    real_loaded_frame_counts.append(frame_count)
                else:
                    if not repeated:
                        failures.append(f"{path}: frame {frame_count} must be marked as repeated-fixture")
                    if "synthetic repeated-fixture speed-scaling smoke" not in repeat_scope_by_frame[str(frame_count)]:
                        failures.append(f"{path}: frame {frame_count} missing repeated-fixture row scope")
                    repeated_frame_counts.append(frame_count)
        frame_scale = float(fg_result.get("frame_scale_first_to_last", float("inf")))
        total_scale = float(fg_result.get("total_step_scale_first_to_last", float("inf")))
        backward_scale = float(fg_result.get("backward_scale_first_to_last", float("inf")))
        storage_scale = float(fg_result.get("selected_tape_storage_scale_first_to_last", float("inf")))
        if not math.isfinite(frame_scale) or frame_scale <= 0.0:
            failures.append(f"{path}: framegroup frame scale is not positive finite")
        if not math.isfinite(total_scale) or total_scale >= frame_scale:
            failures.append(f"{path}: framegroup total scale {total_scale:.3f} is not sublinear versus frames")
        if not math.isfinite(backward_scale) or backward_scale >= frame_scale:
            failures.append(f"{path}: framegroup backward scale {backward_scale:.3f} is not sublinear versus frames")
        if not math.isfinite(total_scale) or total_scale > max_total_scale:
            failures.append(f"{path}: framegroup total scale {total_scale:.3f} exceeds {max_total_scale:.3f}")
        if not math.isfinite(backward_scale) or backward_scale > max_backward_scale:
            failures.append(f"{path}: framegroup backward scale {backward_scale:.3f} exceeds {max_backward_scale:.3f}")
        if not math.isfinite(storage_scale) or storage_scale > max_storage_scale:
            failures.append(f"{path}: framegroup storage scale {storage_scale:.3f} exceeds {max_storage_scale:.3f}")

    return {
        "path": str(path),
        "frame_counts": list(frame_counts),
        "total_ratio_16f": ratio_total,
        "backward_ratio_16f": ratio_backward,
        "ratios_by_frame": ratios_by_frame,
        "psnr_delta_by_frame": psnr_delta_by_frame,
        "total_scale_first_to_last": total_scale,
        "backward_scale_first_to_last": backward_scale,
        "storage_scale_first_to_last": storage_scale,
        "render_size": expected_render_size,
        "site_count": expected_site_count,
        "loaded_frame_count": loaded_frame_count,
        "real_loaded_frame_counts": real_loaded_frame_counts,
        "repeated_frame_counts": repeated_frame_counts,
        "repeat_scope_by_frame": repeat_scope_by_frame,
        "framegroup_total_ms_16f": framegroup_total,
        "endpoint_total_ms_16f": endpoint_total,
        "framegroup_storage_vs_full_16f": framegroup_storage,
        "framegroup_psnr_delta_16f": abs(framegroup_psnr - endpoint_psnr),
        "scope": scope,
    }


def _validate_framegroup_real32_compare(
    *,
    path: Path,
    frame_counts: tuple[int, ...],
    max_total_ratio_all_frames: float,
    max_backward_ratio_all_frames: float,
    max_psnr_delta_all_frames: float,
    max_total_scale: float,
    max_backward_scale: float,
    max_storage_scale: float,
    expected_render_size: int,
    expected_site_count: int,
    failures: list[str],
) -> dict[str, Any]:
    try:
        payload = _load_json(path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        failures.append(f"{path}: could not load real32 framegroup compare artifact: {exc}")
        return {}

    framegroup_mode = "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse"
    required_modes = ("endpoint-run", "endpoint-record-edit", framegroup_mode)
    if payload.get("allow_repeat_loaded_frames") is not False or payload.get("repeat_loaded_frames") is not False:
        failures.append(f"{path}: real32 compare must not use repeated loaded frames")
    scope = str(payload.get("scope", ""))
    if "not a stable benchmark" not in scope:
        failures.append(f"{path}: real32 compare scope must keep benchmark caveat")

    summary_by_frame = payload.get("summary_by_frame")
    results = payload.get("results")
    if not isinstance(summary_by_frame, dict):
        failures.append(f"{path}: real32 compare missing summary_by_frame")
        summary_by_frame = {}
    if not isinstance(results, dict):
        failures.append(f"{path}: real32 compare missing results")
        results = {}

    found_frames = tuple(sorted(int(frame) for frame in summary_by_frame if str(frame).isdigit()))
    if found_frames != frame_counts:
        failures.append(f"{path}: real32 compare frames {found_frames} did not match required {frame_counts}")

    ratios_by_frame: dict[str, dict[str, float]] = {}
    psnr_delta_by_frame: dict[str, float] = {}
    storage_vs_endpoint_by_frame: dict[str, float] = {}
    for frame_count in frame_counts:
        frame_summary = summary_by_frame.get(str(frame_count))
        if not isinstance(frame_summary, dict):
            failures.append(f"{path}: real32 compare missing frame {frame_count} summary")
            continue
        for mode in required_modes:
            if mode not in frame_summary:
                failures.append(f"{path}: real32 compare missing {mode} at frame {frame_count}")
        try:
            endpoint_total = _summary_metric(frame_summary, mode="endpoint-run", key="total_ms")
            endpoint_backward = _summary_metric(frame_summary, mode="endpoint-run", key="backward_ms")
            endpoint_psnr = _summary_metric(frame_summary, mode="endpoint-run", key="heldout_psnr")
            endpoint_storage = _summary_metric(frame_summary, mode="endpoint-run", key="storage_vs_full")
            framegroup_total = _summary_metric(frame_summary, mode=framegroup_mode, key="total_ms")
            framegroup_backward = _summary_metric(frame_summary, mode=framegroup_mode, key="backward_ms")
            framegroup_psnr = _summary_metric(frame_summary, mode=framegroup_mode, key="heldout_psnr")
            framegroup_storage = _summary_metric(frame_summary, mode=framegroup_mode, key="storage_vs_full")
        except KeyError as exc:
            failures.append(f"{path}: frame {frame_count}: {exc}")
            continue
        total_ratio = framegroup_total / endpoint_total
        backward_ratio = framegroup_backward / endpoint_backward
        psnr_delta = abs(framegroup_psnr - endpoint_psnr)
        storage_ratio = framegroup_storage / endpoint_storage
        ratios_by_frame[str(frame_count)] = {"total": total_ratio, "backward": backward_ratio}
        psnr_delta_by_frame[str(frame_count)] = psnr_delta
        storage_vs_endpoint_by_frame[str(frame_count)] = storage_ratio
        if total_ratio > max_total_ratio_all_frames:
            failures.append(
                f"{path}: real32 frame {frame_count} framegroup total ratio {total_ratio:.3f} "
                f"exceeds {max_total_ratio_all_frames:.3f}"
            )
        if backward_ratio > max_backward_ratio_all_frames:
            failures.append(
                f"{path}: real32 frame {frame_count} framegroup backward ratio {backward_ratio:.3f} "
                f"exceeds {max_backward_ratio_all_frames:.3f}"
            )
        if psnr_delta > max_psnr_delta_all_frames:
            failures.append(
                f"{path}: real32 frame {frame_count} framegroup PSNR delta {psnr_delta:.6g} "
                f"exceeds {max_psnr_delta_all_frames}"
            )
        if storage_ratio >= 1.0:
            failures.append(f"{path}: real32 frame {frame_count} framegroup storage must stay below endpoint-run")

    fg_result = results.get(framegroup_mode)
    frame_scale = total_scale = backward_scale = storage_scale = float("nan")
    real_loaded_frame_counts: list[int] = []
    repeat_scope_by_frame: dict[str, str] = {}
    total_sublinear_real_frames = False
    backward_sublinear_real_frames = False
    if not isinstance(fg_result, dict):
        failures.append(f"{path}: real32 compare missing framegroup result payload")
    else:
        if fg_result.get("full_trainer_claim") is not False or fg_result.get("quality_claim") is not False:
            failures.append(f"{path}: real32 framegroup result must not claim full trainer or quality")
        if tuple(fg_result.get("frame_counts", ())) != frame_counts:
            failures.append(f"{path}: real32 framegroup result frame_counts must be {frame_counts}")
        if fg_result.get("render_size") != expected_render_size:
            failures.append(f"{path}: real32 framegroup result render_size must be {expected_render_size}")
        if fg_result.get("site_count") != expected_site_count:
            failures.append(f"{path}: real32 framegroup result site_count must be {expected_site_count}")
        fg_acceptance = fg_result.get("acceptance")
        if not isinstance(fg_acceptance, dict):
            failures.append(f"{path}: real32 framegroup result missing acceptance")
        elif fg_acceptance.get("all_rows_ok") is not True:
            failures.append(f"{path}: real32 framegroup rows must all be ok")
        fg_rows = fg_result.get("rows")
        if not isinstance(fg_rows, list):
            failures.append(f"{path}: real32 framegroup result rows must be a list")
        else:
            rows_by_frame = {
                int(row["frame_count"]): row
                for row in fg_rows
                if isinstance(row, dict) and isinstance(row.get("frame_count"), int)
            }
            if tuple(sorted(rows_by_frame)) != frame_counts:
                failures.append(f"{path}: real32 framegroup row frame counts must be {frame_counts}")
            for frame_count, row in sorted(rows_by_frame.items()):
                if row.get("status") != "ok":
                    failures.append(f"{path}: real32 frame {frame_count} row status is {row.get('status')!r}")
                if row.get("loaded_frame_count") != frame_count:
                    failures.append(f"{path}: real32 frame {frame_count} must be loaded as itself")
                if row.get("repeat_loaded_frames") is not False:
                    failures.append(f"{path}: real32 frame {frame_count} must not be repeated")
                repeat_scope_by_frame[str(frame_count)] = str(row.get("repeat_loaded_frames_scope", ""))
                if repeat_scope_by_frame[str(frame_count)] != "real loaded frame count":
                    failures.append(f"{path}: real32 frame {frame_count} must keep real-loaded scope")
                real_loaded_frame_counts.append(frame_count)
        frame_scale = float(fg_result.get("frame_scale_first_to_last", float("inf")))
        total_scale = float(fg_result.get("total_step_scale_first_to_last", float("inf")))
        backward_scale = float(fg_result.get("backward_scale_first_to_last", float("inf")))
        storage_scale = float(fg_result.get("selected_tape_storage_scale_first_to_last", float("inf")))
        total_sublinear_real_frames = math.isfinite(total_scale) and total_scale < frame_scale
        backward_sublinear_real_frames = math.isfinite(backward_scale) and backward_scale < frame_scale
        if not math.isfinite(frame_scale) or frame_scale <= 0.0:
            failures.append(f"{path}: real32 framegroup frame scale is not positive finite")
        if math.isfinite(frame_scale) and frame_scale > 0.0 and not total_sublinear_real_frames:
            failures.append(
                f"{path}: real32 framegroup total scale {total_scale:.3f} must stay sublinear "
                f"versus frame scale {frame_scale:.3f}"
            )
        if math.isfinite(frame_scale) and frame_scale > 0.0 and not backward_sublinear_real_frames:
            failures.append(
                f"{path}: real32 framegroup backward scale {backward_scale:.3f} must stay sublinear "
                f"versus frame scale {frame_scale:.3f}"
            )
        if not math.isfinite(total_scale) or total_scale > max_total_scale:
            failures.append(f"{path}: real32 framegroup total scale {total_scale:.3f} exceeds {max_total_scale:.3f}")
        if not math.isfinite(backward_scale) or backward_scale > max_backward_scale:
            failures.append(
                f"{path}: real32 framegroup backward scale {backward_scale:.3f} exceeds {max_backward_scale:.3f}"
            )
        if not math.isfinite(storage_scale) or storage_scale > max_storage_scale:
            failures.append(f"{path}: real32 framegroup storage scale {storage_scale:.3f} exceeds {max_storage_scale:.3f}")

    return {
        "path": str(path),
        "status": payload.get("status"),
        "frame_counts": list(frame_counts),
        "render_size": expected_render_size,
        "site_count": expected_site_count,
        "real_loaded_frame_counts": real_loaded_frame_counts,
        "repeated_frame_counts": [],
        "repeat_scope_by_frame": repeat_scope_by_frame,
        "ratios_by_frame": ratios_by_frame,
        "psnr_delta_by_frame": psnr_delta_by_frame,
        "storage_vs_endpoint_by_frame": storage_vs_endpoint_by_frame,
        "frame_scale_first_to_last": frame_scale,
        "total_scale_first_to_last": total_scale,
        "backward_scale_first_to_last": backward_scale,
        "storage_scale_first_to_last": storage_scale,
        "total_sublinear_real_frames": total_sublinear_real_frames,
        "backward_sublinear_real_frames": backward_sublinear_real_frames,
        "real_frame_sublinear_claim": total_sublinear_real_frames and backward_sublinear_real_frames,
        "scope": scope,
    }


def _validate_framegroup_i16x4_compare(
    *,
    path: Path,
    frame_counts: tuple[int, ...],
    max_total_ratio: float,
    max_backward_ratio: float,
    max_storage_ratio: float,
    max_psnr_delta: float,
    failures: list[str],
) -> dict[str, Any]:
    try:
        payload = _load_json(path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        failures.append(f"{path}: could not load i16x4 framegroup compare artifact: {exc}")
        return {}

    i16x3_mode = "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse"
    i16x4_mode = "endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse"
    if payload.get("status") != "ok":
        failures.append(f"{path}: i16x4 compare status is {payload.get('status')!r}")
    if tuple(payload.get("frame_counts", ())) != frame_counts:
        failures.append(f"{path}: i16x4 compare frame_counts must be {frame_counts}")
    if payload.get("repeat_loaded_frames") is not True:
        failures.append(f"{path}: i16x4 compare must be marked as repeated-frame synthetic evidence")
    scope = str(payload.get("scope", ""))
    if "not a STAR-UVT competitiveness artifact" not in scope:
        failures.append(f"{path}: i16x4 compare scope must keep STAR-UVT caveat")
    if payload.get("optimizer_mode") != "manual-vjp":
        failures.append(f"{path}: i16x4 compare must remain manual-vjp for same shader path comparison")

    mode_statuses = payload.get("mode_statuses")
    if not isinstance(mode_statuses, dict):
        failures.append(f"{path}: i16x4 compare missing mode_statuses")
        mode_statuses = {}
    if mode_statuses.get(i16x3_mode) != "ok":
        failures.append(f"{path}: i16x3 mode status must be ok")
    if mode_statuses.get(i16x4_mode) != "failed":
        failures.append(f"{path}: i16x4 mode status must stay failed until it is an explicit promotion")

    modes = payload.get("modes")
    if not isinstance(modes, dict):
        failures.append(f"{path}: i16x4 compare missing mode payloads")
        modes = {}
    for mode in (i16x3_mode, i16x4_mode):
        mode_payload = modes.get(mode)
        if not isinstance(mode_payload, dict):
            failures.append(f"{path}: i16x4 compare missing payload for {mode}")
            continue
        rows = mode_payload.get("rows")
        if not isinstance(rows, list):
            failures.append(f"{path}: {mode} rows must be a list")
            continue
        rows_by_frame = {
            int(row["frame_count"]): row
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("frame_count"), int)
        }
        if tuple(sorted(rows_by_frame)) != frame_counts:
            failures.append(f"{path}: {mode} row frame counts must be {frame_counts}")
        for frame_count, row in rows_by_frame.items():
            if row.get("status") != "ok":
                failures.append(f"{path}: {mode} frame {frame_count} row status is {row.get('status')!r}")
            if frame_count > frame_counts[0]:
                if row.get("repeat_loaded_frames") is not True:
                    failures.append(f"{path}: {mode} frame {frame_count} must be marked repeated")
            elif row.get("repeat_loaded_frames") is not False:
                failures.append(f"{path}: {mode} first frame must remain real-loaded")

    summary = payload.get("summary")
    if not isinstance(summary, dict):
        failures.append(f"{path}: i16x4 compare missing summary")
        return {}
    if summary.get("i16x4_speed_promotion_candidate") is not False:
        failures.append(f"{path}: i16x4 must not be marked as a speed promotion candidate")
    if summary.get("i16x4_total_sublinear") is not False:
        failures.append(f"{path}: i16x4 total sublinear flag must remain false for this negative artifact")
    if summary.get("i16x4_backward_sublinear") is not False:
        failures.append(f"{path}: i16x4 backward sublinear flag must remain false for this negative artifact")

    max_total = float(summary.get("max_i16x4_over_i16x3_total_mean_ratio", float("inf")))
    max_backward = float(summary.get("max_i16x4_over_i16x3_backward_mean_ratio", float("inf")))
    max_psnr = float(summary.get("max_psnr_delta", float("inf")))
    if not math.isfinite(max_total) or max_total > max_total_ratio:
        failures.append(f"{path}: i16x4 max total ratio {max_total:.3f} exceeds {max_total_ratio:.3f}")
    if not math.isfinite(max_backward) or max_backward > max_backward_ratio:
        failures.append(f"{path}: i16x4 max backward ratio {max_backward:.3f} exceeds {max_backward_ratio:.3f}")
    if not math.isfinite(max_psnr) or max_psnr > max_psnr_delta:
        failures.append(f"{path}: i16x4 max PSNR delta {max_psnr:.6g} exceeds {max_psnr_delta}")

    ratios_by_frame = summary.get("ratios_by_frame")
    if not isinstance(ratios_by_frame, dict):
        failures.append(f"{path}: i16x4 compare missing ratios_by_frame")
        ratios_by_frame = {}
    found_frames = tuple(sorted(int(frame) for frame in ratios_by_frame if str(frame).isdigit()))
    if found_frames != frame_counts:
        failures.append(f"{path}: i16x4 ratio frames {found_frames} did not match {frame_counts}")
    max_storage = 0.0
    for frame_count in frame_counts:
        frame_ratios = ratios_by_frame.get(str(frame_count))
        if not isinstance(frame_ratios, dict):
            failures.append(f"{path}: i16x4 compare missing ratios for frame {frame_count}")
            continue
        storage = float(frame_ratios.get("i16x4_over_i16x3_storage", float("inf")))
        max_storage = max(max_storage, storage)
        if not math.isfinite(storage) or storage > max_storage_ratio:
            failures.append(f"{path}: frame {frame_count} i16x4 storage ratio {storage:.3f} exceeds {max_storage_ratio:.3f}")

    return {
        "path": str(path),
        "frame_counts": list(frame_counts),
        "repeat_loaded_frames": bool(payload.get("repeat_loaded_frames")),
        "mode_statuses": dict(mode_statuses),
        "i16x4_speed_promotion_candidate": bool(summary.get("i16x4_speed_promotion_candidate")),
        "i16x4_total_scale_first_to_last": float(summary.get("i16x4_total_scale_first_to_last", float("nan"))),
        "i16x4_backward_scale_first_to_last": float(summary.get("i16x4_backward_scale_first_to_last", float("nan"))),
        "max_i16x4_over_i16x3_total_mean_ratio": max_total,
        "max_i16x4_over_i16x3_backward_mean_ratio": max_backward,
        "max_i16x4_over_i16x3_storage_ratio": max_storage,
        "max_psnr_delta": max_psnr,
        "ratios_by_frame": ratios_by_frame,
        "scope": scope,
    }


def _validate_train_eval_payload(
    *,
    path: Path,
    payload: dict[str, Any],
    frame_counts: tuple[int, ...],
    max_realray_boundaries: int,
    failures: list[str],
) -> dict[str, dict[str, Any]]:
    mode = payload.get("vjp_mode")
    if not isinstance(mode, str):
        failures.append(f"{path}: missing string vjp_mode")
        return {}
    if payload.get("status") != "ok":
        failures.append(f"{path}: payload status is {payload.get('status')!r}, expected 'ok'")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        failures.append(f"{path}: rows must be a list")
        return {}
    rows_by_frame: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            failures.append(f"{path}: row is not an object")
            continue
        frame_count = row.get("frame_count")
        if not isinstance(frame_count, int):
            failures.append(f"{path}: row missing integer frame_count")
            continue
        rows_by_frame[str(frame_count)] = row
        if row.get("status") != "ok":
            failures.append(f"{path}: frame {frame_count} status is {row.get('status')!r}")
        acceptance = row.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append(f"{path}: frame {frame_count} missing acceptance map")
        else:
            bad_acceptance = sorted(key for key, value in acceptance.items() if value is not True)
            if bad_acceptance:
                failures.append(f"{path}: frame {frame_count} failed acceptance keys {bad_acceptance}")
        for key in ("train_max_candidates_per_row", "heldout_max_candidates_per_row"):
            if key not in row:
                continue
            value = row[key]
            if not isinstance(value, int):
                failures.append(f"{path}: frame {frame_count} {key} is not an integer")
            elif value > max_realray_boundaries:
                failures.append(
                    f"{path}: frame {frame_count} {key}={value} exceeds Metal cap {max_realray_boundaries}"
                )
        for key in ("total", "render", "backward"):
            try:
                value = _mean_s(row, key)
            except KeyError as exc:
                failures.append(f"{path}: {exc}")
                continue
            if not math.isfinite(value) or value <= 0.0:
                failures.append(f"{path}: frame {frame_count} {key} mean_s is not positive finite: {value}")
        for key in ("final_train_psnr", "final_heldout_psnr"):
            value = row.get(key)
            if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
                failures.append(f"{path}: frame {frame_count} {key} is not finite")

    found_frames = tuple(sorted(int(frame) for frame in rows_by_frame))
    if found_frames != frame_counts:
        failures.append(f"{path}: frame counts {found_frames} did not match required {frame_counts}")
    return rows_by_frame


def _validate_smoke_payloads(
    *,
    paths: tuple[Path, ...],
    required_modes: tuple[str, ...],
    max_realray_boundaries: int,
    max_vjp_grad_rel_error: float,
    failures: list[str],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in paths:
        try:
            payload = _load_json(path)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"{path}: could not load smoke artifact: {exc}")
            continue
        if payload.get("status") != "ok":
            failures.append(f"{path}: smoke status is {payload.get('status')!r}, expected 'ok'")
        seed_mode = str(payload.get("vjp_seed_mode", "rgb"))
        acceptance = payload.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append(f"{path}: smoke missing acceptance map")
        else:
            bad_acceptance = sorted(key for key, value in acceptance.items() if value is not True)
            if bad_acceptance:
                failures.append(f"{path}: smoke failed acceptance keys {bad_acceptance}")
        rows = payload.get("rows")
        if not isinstance(rows, list):
            failures.append(f"{path}: smoke rows must be a list")
        else:
            for row in rows:
                if not isinstance(row, dict):
                    failures.append(f"{path}: smoke row is not an object")
                    continue
                frame_count = row.get("frames", row.get("frame_count", "?"))
                value = row.get("max_candidates_per_row")
                if not isinstance(value, int):
                    failures.append(f"{path}: smoke frame {frame_count} missing integer max_candidates_per_row")
                elif value > max_realray_boundaries:
                    failures.append(
                        f"{path}: smoke frame {frame_count} max_candidates_per_row={value} "
                        f"exceeds Metal cap {max_realray_boundaries}"
                    )
        diag_summary: dict[str, dict[str, float | bool]] = {}
        for mode in required_modes:
            diag_key = VJP_DIAGNOSTIC_KEYS.get(mode)
            if diag_key is None:
                continue
            diag = payload.get(diag_key)
            if not isinstance(diag, dict):
                failures.append(f"{path}: missing smoke diagnostic {diag_key}")
                continue
            rel = float(diag.get("max_grad_rel_delta_vs_reduce", float("inf")))
            within = bool(diag.get("within_grad_tolerance", False))
            expected_to_match = not (mode == "direct_atomic_rgb_only" and seed_mode != "rgb")
            if not expected_to_match:
                if not bool(diag.get("has_expected_seed_behavior", False)):
                    failures.append(f"{path}: {diag_key} did not show expected non-RGB seed divergence")
                diag_summary[mode] = {
                    "max_grad_rel_delta_vs_reduce": rel,
                    "within_grad_tolerance": within,
                    "expected_to_match_reduce": False,
                }
                continue
            if not within:
                failures.append(f"{path}: {diag_key}.within_grad_tolerance is false")
            if not math.isfinite(rel) or rel > max_vjp_grad_rel_error:
                failures.append(
                    f"{path}: {diag_key}.max_grad_rel_delta_vs_reduce={rel} "
                    f"exceeds {max_vjp_grad_rel_error}"
                )
            diag_summary[mode] = {
                "max_grad_rel_delta_vs_reduce": rel,
                "within_grad_tolerance": within,
                "expected_to_match_reduce": True,
            }
        autograd_diag = payload.get("autograd_vjp_diagnostics")
        if not isinstance(autograd_diag, dict):
            failures.append(f"{path}: missing smoke diagnostic autograd_vjp_diagnostics")
        else:
            if not bool(autograd_diag.get("general_modes_match_reduce", False)):
                failures.append(f"{path}: autograd_vjp_diagnostics.general_modes_match_reduce is false")
            if not bool(autograd_diag.get("rgb_only_has_expected_seed_behavior", False)):
                failures.append(f"{path}: autograd_vjp_diagnostics.rgb_only_has_expected_seed_behavior is false")
            reduce_rel = float(autograd_diag.get("max_reduce_rel_delta_vs_raw_reduce", float("inf")))
            if not math.isfinite(reduce_rel) or reduce_rel > max_vjp_grad_rel_error:
                failures.append(
                    f"{path}: autograd_vjp_diagnostics.max_reduce_rel_delta_vs_raw_reduce={reduce_rel} "
                    f"exceeds {max_vjp_grad_rel_error}"
                )
            mode_rels = autograd_diag.get("max_grad_rel_delta_by_mode_vs_autograd_reduce")
            if not isinstance(mode_rels, dict):
                failures.append(f"{path}: missing autograd mode relative deltas")
            else:
                collected_rels: list[float] = []
                for mode in ("direct_atomic", "direct_atomic_grad_only", "direct_atomic_rgb_only", "direct_atomic_track"):
                    rel = float(mode_rels.get(mode, float("inf")))
                    collected_rels.append(rel)
                    if not math.isfinite(rel) or rel > max_vjp_grad_rel_error:
                        failures.append(f"{path}: autograd {mode} rel delta {rel} exceeds {max_vjp_grad_rel_error}")
                diag_summary["autograd"] = {
                    "max_reduce_rel_delta_vs_raw_reduce": reduce_rel,
                    "max_mode_rel_delta_vs_autograd_reduce": max(collected_rels),
                }
        summaries.append({"path": str(path), "vjp_seed_mode": seed_mode, "vjp_diagnostics": diag_summary})
    return summaries


def verify(args: argparse.Namespace) -> dict[str, Any]:
    frame_counts = _parse_int_list(args.frame_counts)
    framegroup_frame_counts = _parse_int_list(args.framegroup_frame_counts)
    compare_frame_counts = _parse_int_list(args.compare_frame_counts)
    real32_frame_counts = _parse_int_list(args.real32_frame_counts)
    i16x4_frame_counts = _parse_int_list(args.i16x4_compare_frame_counts)
    required_modes = tuple(args.required_modes.split(","))
    train_eval_paths = tuple(args.train_eval_json or DEFAULT_TRAIN_EVAL_ARTIFACTS)
    smoke_paths = tuple(args.smoke_json or DEFAULT_SMOKE_ARTIFACTS)
    failures: list[str] = []
    mode_rows: dict[str, dict[str, dict[str, Any]]] = {}
    mode_paths: dict[str, str] = {}

    for path in train_eval_paths:
        try:
            payload = _load_json(path)
        except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
            failures.append(f"{path}: could not load train/eval artifact: {exc}")
            continue
        mode = payload.get("vjp_mode")
        if isinstance(mode, str) and mode in mode_rows:
            failures.append(f"{path}: duplicate train/eval artifact for vjp_mode={mode!r}")
            continue
        rows_by_frame = _validate_train_eval_payload(
            path=path,
            payload=payload,
            frame_counts=frame_counts,
            max_realray_boundaries=args.max_realray_boundaries,
            failures=failures,
        )
        if isinstance(mode, str):
            mode_rows[mode] = rows_by_frame
            mode_paths[mode] = str(path)

    missing_modes = sorted(set(required_modes) - set(mode_rows))
    if missing_modes:
        failures.append(f"missing required train/eval modes: {missing_modes}")
    if args.best_mode not in mode_rows:
        failures.append(f"best mode {args.best_mode!r} is missing")

    mode_metrics: dict[str, dict[str, Any]] = {}
    for mode, rows_by_frame in mode_rows.items():
        ordered_rows = [rows_by_frame[str(frame_count)] for frame_count in frame_counts if str(frame_count) in rows_by_frame]
        if len(ordered_rows) != len(frame_counts):
            continue
        totals = [_mean_s(row, "total") for row in ordered_rows]
        renders = [_mean_s(row, "render") for row in ordered_rows]
        backwards = [_mean_s(row, "backward") for row in ordered_rows]
        mode_metrics[mode] = {
            "path": mode_paths[mode],
            "total_mean_s": dict(zip((str(frame) for frame in frame_counts), totals, strict=True)),
            "render_mean_s": dict(zip((str(frame) for frame in frame_counts), renders, strict=True)),
            "backward_mean_s": dict(zip((str(frame) for frame in frame_counts), backwards, strict=True)),
            "total_scale_max_over_min": totals[-1] / totals[0],
            "render_scale_max_over_min": renders[-1] / renders[0],
            "backward_scale_max_over_min": backwards[-1] / backwards[0],
            "total_geomean_s": _geo_mean(totals),
            "heldout_psnr": {
                str(frame_count): float(row["final_heldout_psnr"])
                for frame_count, row in zip(frame_counts, ordered_rows, strict=True)
            },
            "train_psnr": {
                str(frame_count): float(row["final_train_psnr"])
                for frame_count, row in zip(frame_counts, ordered_rows, strict=True)
            },
        }

    best_metrics = mode_metrics.get(args.best_mode)
    if best_metrics is not None:
        if best_metrics["total_scale_max_over_min"] > args.max_total_scale:
            failures.append(
                f"{args.best_mode}: total scale {best_metrics['total_scale_max_over_min']:.3f} "
                f"exceeds {args.max_total_scale:.3f}"
            )
        if best_metrics["render_scale_max_over_min"] > args.max_render_scale:
            failures.append(
                f"{args.best_mode}: render scale {best_metrics['render_scale_max_over_min']:.3f} "
                f"exceeds {args.max_render_scale:.3f}"
            )
        if best_metrics["backward_scale_max_over_min"] > args.max_backward_scale:
            failures.append(
                f"{args.best_mode}: backward scale {best_metrics['backward_scale_max_over_min']:.3f} "
                f"exceeds {args.max_backward_scale:.3f}"
            )
        linear_scale = frame_counts[-1] / frame_counts[0]
        if best_metrics["total_scale_max_over_min"] >= linear_scale:
            failures.append(
                f"{args.best_mode}: total scale {best_metrics['total_scale_max_over_min']:.3f} "
                f"is not sublinear versus frame-count scale {linear_scale:.3f}"
            )

    if mode_metrics:
        max_frame = str(frame_counts[-1])
        fastest_at_max = min(mode_metrics, key=lambda mode: mode_metrics[mode]["total_mean_s"][max_frame])
        fastest_geomean = min(mode_metrics, key=lambda mode: mode_metrics[mode]["total_geomean_s"])
        if fastest_at_max != args.best_mode:
            failures.append(
                f"{args.best_mode} is not fastest at {max_frame}f; "
                f"{fastest_at_max} has lower total step time"
            )
        if fastest_geomean != args.best_mode:
            failures.append(
                f"{args.best_mode} is not fastest by total-step geometric mean; "
                f"{fastest_geomean} has lower geometric mean"
            )
        for frame_count in frame_counts:
            frame_key = str(frame_count)
            for psnr_key in ("heldout_psnr", "train_psnr"):
                values = [metrics[psnr_key][frame_key] for metrics in mode_metrics.values()]
                if max(values) - min(values) > args.max_psnr_spread:
                    failures.append(
                        f"{psnr_key} spread at {frame_key}f is {max(values) - min(values):.6g}, "
                        f"exceeds {args.max_psnr_spread}"
                    )

    smoke_summaries = _validate_smoke_payloads(
        paths=smoke_paths,
        required_modes=required_modes,
        max_realray_boundaries=args.max_realray_boundaries,
        max_vjp_grad_rel_error=args.max_vjp_grad_rel_error,
        failures=failures,
    )
    framegroup_lossreduce = _validate_framegroup_lossreduce_payloads(
        train_eval_path=args.framegroup_lossreduce_json,
        confirm_path=args.framegroup_128only_json,
        frame_counts=framegroup_frame_counts,
        max_total_scale=args.framegroup_max_total_scale,
        max_backward_scale=args.framegroup_max_backward_scale,
        max_storage_scale=args.framegroup_max_storage_scale,
        max_mixed_128_total_max_ms=args.framegroup_max_mixed_128_total_max_ms,
        max_128only_total_median_ms=args.framegroup_max_128only_total_median_ms,
        max_128only_total_max_ms=args.framegroup_max_128only_total_max_ms,
        max_128only_backward_median_ms=args.framegroup_max_128only_backward_median_ms,
        failures=failures,
    )
    framegroup_compare_smoke = _validate_framegroup_compare_smoke(
        path=args.compare_smoke_json,
        frame_counts=compare_frame_counts,
        max_total_ratio_16f=args.compare_max_framegroup_to_endpoint_total_16f,
        max_backward_ratio_16f=args.compare_max_framegroup_to_endpoint_backward_16f,
        max_psnr_delta=args.compare_max_psnr_delta,
        max_storage_vs_full_16f=args.compare_max_framegroup_storage_vs_full_16f,
        max_total_scale=args.compare_max_framegroup_total_scale,
        max_backward_scale=args.compare_max_framegroup_backward_scale,
        max_storage_scale=args.compare_max_framegroup_storage_scale,
        max_total_ratio_all_frames=args.compare_max_framegroup_to_endpoint_total_all_frames,
        max_psnr_delta_all_frames=args.compare_max_psnr_delta_all_frames,
        expected_render_size=args.compare_render_size,
        expected_site_count=args.compare_site_count,
        failures=failures,
    )
    framegroup_real32_compare = _validate_framegroup_real32_compare(
        path=args.real32_compare_json,
        frame_counts=real32_frame_counts,
        max_total_ratio_all_frames=args.real32_max_framegroup_to_endpoint_total_all_frames,
        max_backward_ratio_all_frames=args.real32_max_framegroup_to_endpoint_backward_all_frames,
        max_psnr_delta_all_frames=args.real32_max_psnr_delta_all_frames,
        max_total_scale=args.real32_max_framegroup_total_scale,
        max_backward_scale=args.real32_max_framegroup_backward_scale,
        max_storage_scale=args.real32_max_framegroup_storage_scale,
        expected_render_size=args.real32_render_size,
        expected_site_count=args.real32_site_count,
        failures=failures,
    )
    framegroup_i16x4_compare = _validate_framegroup_i16x4_compare(
        path=args.i16x4_compare_json,
        frame_counts=i16x4_frame_counts,
        max_total_ratio=args.i16x4_max_over_i16x3_total_mean_ratio,
        max_backward_ratio=args.i16x4_max_over_i16x3_backward_mean_ratio,
        max_storage_ratio=args.i16x4_max_over_i16x3_storage_ratio,
        max_psnr_delta=args.i16x4_max_psnr_delta,
        failures=failures,
    )

    return {
        "benchmark": "world_foam_lane2_fused_slab_mixed_scaling_verifier",
        "status": "ok" if not failures else "failed",
        "best_mode": args.best_mode,
        "required_modes": list(required_modes),
        "frame_counts": list(frame_counts),
        "thresholds": {
            "max_total_scale": args.max_total_scale,
            "max_render_scale": args.max_render_scale,
            "max_backward_scale": args.max_backward_scale,
            "max_psnr_spread": args.max_psnr_spread,
            "max_realray_boundaries": args.max_realray_boundaries,
            "max_vjp_grad_rel_error": args.max_vjp_grad_rel_error,
            "framegroup_max_total_scale": args.framegroup_max_total_scale,
            "framegroup_max_backward_scale": args.framegroup_max_backward_scale,
            "framegroup_max_storage_scale": args.framegroup_max_storage_scale,
            "framegroup_max_mixed_128_total_max_ms": args.framegroup_max_mixed_128_total_max_ms,
            "framegroup_max_128only_total_median_ms": args.framegroup_max_128only_total_median_ms,
            "framegroup_max_128only_total_max_ms": args.framegroup_max_128only_total_max_ms,
            "framegroup_max_128only_backward_median_ms": args.framegroup_max_128only_backward_median_ms,
            "compare_max_framegroup_to_endpoint_total_16f": args.compare_max_framegroup_to_endpoint_total_16f,
            "compare_max_framegroup_to_endpoint_backward_16f": args.compare_max_framegroup_to_endpoint_backward_16f,
            "compare_max_psnr_delta": args.compare_max_psnr_delta,
            "compare_max_framegroup_storage_vs_full_16f": args.compare_max_framegroup_storage_vs_full_16f,
            "compare_max_framegroup_total_scale": args.compare_max_framegroup_total_scale,
            "compare_max_framegroup_backward_scale": args.compare_max_framegroup_backward_scale,
            "compare_max_framegroup_storage_scale": args.compare_max_framegroup_storage_scale,
            "compare_max_framegroup_to_endpoint_total_all_frames": (
                args.compare_max_framegroup_to_endpoint_total_all_frames
            ),
            "compare_max_psnr_delta_all_frames": args.compare_max_psnr_delta_all_frames,
            "compare_render_size": args.compare_render_size,
            "compare_site_count": args.compare_site_count,
            "real32_max_framegroup_to_endpoint_total_all_frames": (
                args.real32_max_framegroup_to_endpoint_total_all_frames
            ),
            "real32_max_framegroup_to_endpoint_backward_all_frames": (
                args.real32_max_framegroup_to_endpoint_backward_all_frames
            ),
            "real32_max_psnr_delta_all_frames": args.real32_max_psnr_delta_all_frames,
            "real32_max_framegroup_total_scale": args.real32_max_framegroup_total_scale,
            "real32_max_framegroup_backward_scale": args.real32_max_framegroup_backward_scale,
            "real32_max_framegroup_storage_scale": args.real32_max_framegroup_storage_scale,
            "real32_render_size": args.real32_render_size,
            "real32_site_count": args.real32_site_count,
            "i16x4_max_over_i16x3_total_mean_ratio": args.i16x4_max_over_i16x3_total_mean_ratio,
            "i16x4_max_over_i16x3_backward_mean_ratio": args.i16x4_max_over_i16x3_backward_mean_ratio,
            "i16x4_max_over_i16x3_storage_ratio": args.i16x4_max_over_i16x3_storage_ratio,
            "i16x4_max_psnr_delta": args.i16x4_max_psnr_delta,
        },
        "mode_metrics": mode_metrics,
        "smoke_summaries": smoke_summaries,
        "framegroup_lossreduce": framegroup_lossreduce,
        "framegroup_compare_smoke": framegroup_compare_smoke,
        "framegroup_real32_compare": framegroup_real32_compare,
        "framegroup_i16x4_compare": framegroup_i16x4_compare,
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify fused-slab mixed World Foam VJP mode scaling and saved correctness artifacts."
    )
    parser.add_argument("--frame-counts", default="2,4,8,16")
    parser.add_argument("--best-mode", default="direct_atomic_grad_only")
    parser.add_argument("--required-modes", default=",".join(DEFAULT_REQUIRED_MODES))
    parser.add_argument("--max-total-scale", type=float, default=1.6)
    parser.add_argument("--max-render-scale", type=float, default=1.35)
    parser.add_argument("--max-backward-scale", type=float, default=2.2)
    parser.add_argument("--max-psnr-spread", type=float, default=1.0e-3)
    parser.add_argument("--max-realray-boundaries", type=int, default=DEFAULT_MAX_REALRAY_BOUNDARIES)
    parser.add_argument("--max-vjp-grad-rel-error", type=float, default=2.0e-6)
    parser.add_argument("--train-eval-json", type=Path, nargs="*")
    parser.add_argument("--smoke-json", type=Path, nargs="*")
    parser.add_argument("--framegroup-frame-counts", default="16,32,64,128")
    parser.add_argument("--framegroup-lossreduce-json", type=Path, default=DEFAULT_FRAMEGROUP_LOSSREDUCE_ARTIFACT)
    parser.add_argument("--framegroup-128only-json", type=Path, default=DEFAULT_FRAMEGROUP_128ONLY_ARTIFACT)
    parser.add_argument("--framegroup-max-total-scale", type=float, default=1.5)
    parser.add_argument("--framegroup-max-backward-scale", type=float, default=1.65)
    parser.add_argument("--framegroup-max-storage-scale", type=float, default=1.10)
    parser.add_argument("--framegroup-max-mixed-128-total-max-ms", type=float, default=7.5)
    parser.add_argument("--framegroup-max-128only-total-median-ms", type=float, default=4.5)
    parser.add_argument("--framegroup-max-128only-total-max-ms", type=float, default=8.5)
    parser.add_argument("--framegroup-max-128only-backward-median-ms", type=float, default=3.75)
    parser.add_argument("--compare-frame-counts", default="16,32,64,128")
    parser.add_argument("--compare-smoke-json", type=Path, default=DEFAULT_FRAMEGROUP_COMPARE_SMOKE_ARTIFACT)
    parser.add_argument("--compare-max-framegroup-to-endpoint-total-16f", type=float, default=0.75)
    parser.add_argument("--compare-max-framegroup-to-endpoint-backward-16f", type=float, default=0.95)
    parser.add_argument("--compare-max-psnr-delta", type=float, default=1.0e-3)
    parser.add_argument("--compare-max-framegroup-storage-vs-full-16f", type=float, default=0.15)
    parser.add_argument("--compare-max-framegroup-total-scale", type=float, default=3.25)
    parser.add_argument("--compare-max-framegroup-backward-scale", type=float, default=3.75)
    parser.add_argument("--compare-max-framegroup-storage-scale", type=float, default=1.10)
    parser.add_argument("--compare-max-framegroup-to-endpoint-total-all-frames", type=float, default=0.75)
    parser.add_argument("--compare-max-psnr-delta-all-frames", type=float, default=5.0e-3)
    parser.add_argument("--compare-render-size", type=int, default=32)
    parser.add_argument("--compare-site-count", type=int, default=12)
    parser.add_argument("--real32-frame-counts", default="16,32")
    parser.add_argument("--real32-compare-json", type=Path, default=DEFAULT_FRAMEGROUP_REAL32_COMPARE_ARTIFACT)
    parser.add_argument("--real32-max-framegroup-to-endpoint-total-all-frames", type=float, default=0.75)
    parser.add_argument("--real32-max-framegroup-to-endpoint-backward-all-frames", type=float, default=0.95)
    parser.add_argument("--real32-max-psnr-delta-all-frames", type=float, default=1.0e-3)
    parser.add_argument("--real32-max-framegroup-total-scale", type=float, default=2.25)
    parser.add_argument("--real32-max-framegroup-backward-scale", type=float, default=2.35)
    parser.add_argument("--real32-max-framegroup-storage-scale", type=float, default=1.10)
    parser.add_argument("--real32-render-size", type=int, default=32)
    parser.add_argument("--real32-site-count", type=int, default=12)
    parser.add_argument("--i16x4-compare-frame-counts", default="16,32")
    parser.add_argument("--i16x4-compare-json", type=Path, default=DEFAULT_FRAMEGROUP_I16X4_COMPARE_ARTIFACT)
    parser.add_argument("--i16x4-max-over-i16x3-total-mean-ratio", type=float, default=1.05)
    parser.add_argument("--i16x4-max-over-i16x3-backward-mean-ratio", type=float, default=1.05)
    parser.add_argument("--i16x4-max-over-i16x3-storage-ratio", type=float, default=1.08)
    parser.add_argument("--i16x4-max-psnr-delta", type=float, default=1.0e-4)
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = verify(args)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    if summary["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
