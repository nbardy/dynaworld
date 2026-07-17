#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "research_experiments" / "world_foam_lane2" / "results"
DEFAULT_VERIFIER = RESULTS_DIR / "2026-05-16_fused_slab_mixed_scaling_verifier_with_framegroup_lossreduce.json"
DEFAULT_DEPTH_ORDER = RESULTS_DIR / "2026-05-15_fused_slab_mixed_depth_order_probe_render32_pertrack_2_4_8_16.json"
DEFAULT_OWNERUPDATE = RESULTS_DIR / "2026-05-15_fused_slab_mixed_ownerupdate_gradonly_vjp_render32_pertrack_2_4_8_16.json"
DEFAULT_STAR_SPEED = (
    RESULTS_DIR / "2026-05-15_fixed_step_speed_compare_star_directatomic_20step_32px_2_4_8_16.json"
)
DEFAULT_SEGMENT_TAPE = RESULTS_DIR / "2026-05-15_fused_slab_segment_tape_probe_render32_pertrack_2_4_8_16.json"
DEFAULT_TOPOLOGY_SHARING = RESULTS_DIR / "2026-05-15_segment_topology_sharing_probe_render32_pertrack_2_4_8_16.json"
DEFAULT_DELTA_TAPE = RESULTS_DIR / "2026-05-15_segment_delta_tape_probe_render32_2_4_8_16.json"
DEFAULT_BOUNDARY_DELTA_TAPE = RESULTS_DIR / "2026-05-15_segment_boundary_delta_tape_probe_render32_2_4_8_16.json"
DEFAULT_RECORD_DELTA_TAPE = RESULTS_DIR / "2026-05-15_segment_record_delta_tape_probe_render32_2_4_8_16.json"
DEFAULT_OWNER_RUN_TAPE = RESULTS_DIR / "2026-05-15_segment_owner_run_tape_probe_render32_2_4_8_16.json"
DEFAULT_OWNER_RUN_BOUNDARY_TAPE = RESULTS_DIR / "2026-05-15_owner_run_boundary_tape_probe_render32_2_4_8_16.json"
DEFAULT_OWNER_RUN_INTERNAL_TAPE = RESULTS_DIR / "2026-05-15_owner_run_internal_tape_probe_render32_2_4_8_16.json"
DEFAULT_ENDPOINT_RUN_TAPE = RESULTS_DIR / "2026-05-15_endpoint_run_tape_probe_render32_2_4_8_16.json"
DEFAULT_ENDPOINT_RECORD_DELTA_TAPE = (
    RESULTS_DIR / "2026-05-15_endpoint_record_delta_tape_probe_render32_2_4_8_16.json"
)
DEFAULT_ENDPOINT_RECORD_DELTA_REPLAY = (
    RESULTS_DIR / "2026-05-15_endpoint_record_delta_replay_render32_2_4_8_16.json"
)
DEFAULT_ENDPOINT_RECORD_EDIT_REPLAY = (
    RESULTS_DIR / "2026-05-15_endpoint_record_edit_replay_cutcache_render32_2_4_8_16.json"
)
DEFAULT_ENDPOINT_RECORD_EDIT_RGB_ONLY_REPLAY = (
    RESULTS_DIR / "2026-05-15_endpoint_record_edit_replay_rgbonly_render32_2_4_8_16.json"
)
DEFAULT_ENDPOINT_RECORD_EDIT_TRACKLOOP_REPLAY = (
    RESULTS_DIR / "2026-05-15_endpoint_record_edit_replay_trackloop_render32_2_4_8_16.json"
)
DEFAULT_ENDPOINT_RECORD_EDIT_BLOCK4_REPLAY = (
    RESULTS_DIR / "2026-05-15_endpoint_record_edit_replay_block4_vjp_render32_2_4_8_16.json"
)
DEFAULT_ENDPOINT_RECORD_EDIT_BLOCK_COEFF_REPLAY = (
    RESULTS_DIR / "2026-05-15_endpoint_record_edit_replay_block_coeff_render16_16f.json"
)
DEFAULT_ENDPOINT_RECORD_EDIT_BLOCK_COEFF_SWEEP = (
    RESULTS_DIR / "2026-05-15_endpoint_record_edit_replay_block_coeff_smoke_render16_2_4_8.json"
)
DEFAULT_ENDPOINT_RECORD_EDIT_BLOCK_COEFF_TRAIN_EVAL = (
    RESULTS_DIR
    / "2026-05-15_endpoint_record_edit_block_coeff_rgb_train_eval_autograd_repeat20_render32_2_4_8_16.json"
)
DEFAULT_OWNER_RUN_TRAIN_EVAL = (
    RESULTS_DIR / "2026-05-15_owner_run_rgb_train_eval_autograd_fusedparams_render32_2_4_8_16.json"
)
DEFAULT_ACTIVE_INTERNAL_TRAIN_EVAL = (
    RESULTS_DIR / "2026-05-15_active_internal_rgb_train_eval_autograd_fusedparams_render32_2_4_8_16.json"
)
DEFAULT_FULL_TAPE_TRAIN_EVAL = (
    RESULTS_DIR / "2026-05-15_full_segment_rgb_train_eval_autograd_fusedparams_render32_2_4_8_16.json"
)
DEFAULT_ENDPOINT_RUN_TRAIN_EVAL = (
    RESULTS_DIR / "2026-05-15_endpoint_run_rgb_train_eval_autograd_fusedparams_render32_2_4_8_16.json"
)
DEFAULT_ENDPOINT_RECORD_EDIT_TRAIN_EVAL = (
    RESULTS_DIR / "2026-05-15_endpoint_record_edit_rgb_train_eval_autograd_fusedparams_render32_2_4_8_16.json"
)
DEFAULT_ENDPOINT_RECORD_EDIT_BLOCK4_TRAIN_EVAL = (
    RESULTS_DIR / "2026-05-15_endpoint_record_edit_block4_rgb_train_eval_autograd_block4vjp_repeat12_render32_2_4_8_16.json"
)
DEFAULT_ENDPOINT_RECORD_EDIT_PAIRED_TRAIN_EVAL = (
    RESULTS_DIR / "2026-05-15_endpoint_run_vs_record_edit_current_process_train_eval_cutcache_render32_2_4_8_16.json"
)
DEFAULT_ENDPOINT_RECORD_EDIT_BLOCK4_PAIRED_TRAIN_EVAL = (
    RESULTS_DIR
    / "2026-05-15_endpoint_run_vs_record_edit_block4_current_process_train_eval_repeat12_render32_2_4_8_16.json"
)
DEFAULT_ENDPOINT_RECORD_EDIT_BLOCK_COEFF_PAIRED_TRAIN_EVAL = (
    RESULTS_DIR
    / "2026-05-15_endpoint_run_vs_record_edit_block4_blockcoeff_current_process_train_eval_warm5_render32_2_4_8_16.json"
)
DEFAULT_ENDPOINT_RECORD_EDIT_BLOCK_COEFF_REPEAT20_16F = (
    RESULTS_DIR
    / "2026-05-15_endpoint_run_vs_record_edit_block4_blockcoeff_current_process_train_eval_repeat20_render32_16f.json"
)
DEFAULT_ENDPOINT_RECORD_EDIT_BLOCK_COEFF_REPEAT20_2_4_8_16 = (
    RESULTS_DIR
    / "2026-05-15_endpoint_run_vs_record_edit_block4_blockcoeff_current_process_train_eval_repeat20_render32_2_4_8_16.json"
)
DEFAULT_ENDPOINT_RECORD_EDIT_BLOCK_COEFF16_MANUAL_VJP_PAIRED_TRAIN_EVAL = (
    RESULTS_DIR
    / "2026-05-15_endpoint_run_vs_record_edit_block4_blockcoeff16_manualvjp_smoke_render32_16f.json"
)
DEFAULT_ENDPOINT_RECORD_EDIT_BLOCK_COEFF16_STORAGEFIX_SMOKE = (
    RESULTS_DIR
    / "2026-05-15_endpoint_record_edit_block_coeff16_manualvjp_storagefix_smoke_render16_16f.json"
)
DEFAULT_ENDPOINT_RECORD_EDIT_RGB_ONLY_PAIRED_TRAIN_EVAL = (
    RESULTS_DIR
    / "2026-05-15_endpoint_run_vs_record_edit_current_process_train_eval_rgbonly_repeat12_render32_2_4_8_16.json"
)
DEFAULT_ENDPOINT_RECORD_EDIT_MANUAL_VJP_PAIRED_TRAIN_EVAL = (
    RESULTS_DIR
    / "2026-05-15_endpoint_run_vs_record_edit_current_process_train_eval_manualvjp_repeat12_render32_2_4_8_16.json"
)
DEFAULT_SEGMENT_TAPE_AUTOGRAD_SMOKE = RESULTS_DIR / "2026-05-15_segment_tape_autograd_smoke_render16_2f.json"
DEFAULT_FRAMEGROUP_AUTOGRAD_SMOKE = (
    RESULTS_DIR
    / "2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_autograd_smoke_render16_site4_2f.json"
)
DEFAULT_FRAMEGROUP_AUTOGRAD_SPEEDSCALE = (
    RESULTS_DIR
    / "2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_autograd_speedscale_warm3_steps8_render32_site12_16_32_64_128.json"
)
DEFAULT_FRAMEGROUP_I16X4_PREWARM_COMPARE = (
    RESULTS_DIR
    / "2026-05-16_delta_framegroup_i16x3_i16x4_train_eval_compare_repeat32_prewarm_warm3_steps5_render32_site12_16_32.json"
)
DEFAULT_FRAMEGROUP_PACKED_PREWARM_COMPARE = (
    RESULTS_DIR
    / "2026-05-16_delta_framegroup_i16x3_packed_train_eval_compare_repeat32_prewarm_warm3_steps5_render32_site12_16_32.json"
)
DEFAULT_FRAMEGROUP_PACKED_BROAD_COMPARE = (
    RESULTS_DIR
    / "2026-05-16_delta_framegroup_i16x3_packed_train_eval_compare_repeat64_128_interleaved_prewarm_warm1_steps3_render32_site12_64_128.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _finite_positive(value: Any) -> bool:
    return _finite_number(value) and float(value) > 0.0


def _framegroup_objective_adapter_available(adapter: Any) -> bool:
    return (
        isinstance(adapter, dict)
        and adapter.get("name") == "WorldFoamFrozenRGBMSEObjective"
        and adapter.get("module") == "objective.world_foam_frozen_rgb_mse"
        and adapter.get("construction_scope") == "once_per_frame_count_run"
        and adapter.get("loss_call_scope") == "per_optimizer_step"
        and adapter.get("backend_loss_fn") == "promoted_framegroup16_loss_fn"
        and adapter.get("tape_mode") == "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse"
        and adapter.get("renderer_backend_claim") is False
        and adapter.get("full_trainer_claim") is False
        and adapter.get("full_geometry_gradient_claim") is False
        and adapter.get("quality_claim") is False
        and adapter.get("supports_rgb_mse_only") is True
        and adapter.get("supports_background_composition") is False
        and adapter.get("supports_colorizer") is False
        and adapter.get("supports_vjepa_feature_loss") is False
    )


def _mode_table(verifier: dict[str, Any]) -> list[dict[str, Any]]:
    mode_metrics = verifier.get("mode_metrics")
    if not isinstance(mode_metrics, dict):
        return []
    rows = []
    for mode, metrics in mode_metrics.items():
        if not isinstance(metrics, dict):
            continue
        total_by_frame = metrics.get("total_mean_s", {})
        render_by_frame = metrics.get("render_mean_s", {})
        backward_by_frame = metrics.get("backward_mean_s", {})
        heldout_psnr = metrics.get("heldout_psnr", {})
        rows.append(
            {
                "mode": mode,
                "total_geomean_ms": float(metrics.get("total_geomean_s", 0.0)) * 1000.0,
                "total_2f_ms": float(total_by_frame.get("2", 0.0)) * 1000.0,
                "total_16f_ms": float(total_by_frame.get("16", 0.0)) * 1000.0,
                "total_scale_2_to_16": metrics.get("total_scale_max_over_min"),
                "render_scale_2_to_16": metrics.get("render_scale_max_over_min"),
                "backward_scale_2_to_16": metrics.get("backward_scale_max_over_min"),
                "render_16f_ms": float(render_by_frame.get("16", 0.0)) * 1000.0,
                "backward_16f_ms": float(backward_by_frame.get("16", 0.0)) * 1000.0,
                "heldout_psnr_16f": heldout_psnr.get("16"),
                "artifact": metrics.get("path"),
            }
        )
    return sorted(rows, key=lambda row: float(row["total_geomean_ms"]))


def _psnr_spread_by_frame(verifier: dict[str, Any]) -> dict[str, Any] | None:
    mode_metrics = verifier.get("mode_metrics")
    frame_counts = verifier.get("frame_counts")
    if not isinstance(mode_metrics, dict) or not isinstance(frame_counts, list):
        return None
    by_metric: dict[str, dict[str, float]] = {}
    for metric_name in ("heldout_psnr", "train_psnr"):
        by_frame: dict[str, float] = {}
        for frame_count in frame_counts:
            frame_key = str(frame_count)
            values = []
            for metrics in mode_metrics.values():
                if not isinstance(metrics, dict):
                    continue
                value = metrics.get(metric_name, {}).get(frame_key)
                if _finite_number(value):
                    values.append(float(value))
            if values:
                by_frame[frame_key] = max(values) - min(values)
        by_metric[metric_name] = by_frame
    all_spreads = [spread for by_frame in by_metric.values() for spread in by_frame.values()]
    if not all_spreads:
        return None
    return {
        "max_spread": max(all_spreads),
        "by_metric": by_metric,
    }


def _smoke_coverage(verifier: dict[str, Any]) -> dict[str, Any]:
    summaries = verifier.get("smoke_summaries")
    if not isinstance(summaries, list):
        return {"has_rgb_seed": False, "has_rgba_depth_seed": False, "autograd_checked": False}
    seeds = {str(summary.get("vjp_seed_mode")) for summary in summaries if isinstance(summary, dict)}
    autograd_checked = all(
        isinstance(summary, dict)
        and isinstance(summary.get("vjp_diagnostics"), dict)
        and isinstance(summary["vjp_diagnostics"].get("autograd"), dict)
        for summary in summaries
    )
    return {
        "has_rgb_seed": "rgb" in seeds,
        "has_rgba_depth_seed": "rgba-depth" in seeds,
        "autograd_checked": autograd_checked,
        "smoke_count": len(summaries),
        "seeds": sorted(seeds),
    }


def _framegroup_lossreduce_summary(verifier: dict[str, Any]) -> dict[str, Any]:
    framegroup = verifier.get("framegroup_lossreduce")
    if not isinstance(framegroup, dict):
        return {"available": False}
    confirm = framegroup.get("confirm_128only")
    mixed_rows = framegroup.get("mixed_rows")
    thresholds = verifier.get("thresholds")
    available = (
        verifier.get("status") == "ok"
        and isinstance(confirm, dict)
        and isinstance(mixed_rows, dict)
        and _finite_number(framegroup.get("total_scale_first_to_last"))
        and _finite_number(framegroup.get("backward_scale_first_to_last"))
        and _finite_number(framegroup.get("storage_scale_first_to_last"))
        and _finite_number(framegroup.get("mixed_128_total_max_ms"))
        and _finite_number(confirm.get("total_median_ms"))
        and _finite_number(confirm.get("total_max_ms"))
    )
    return {
        "available": available,
        "artifact": framegroup.get("train_eval_path"),
        "confirm_artifact": framegroup.get("confirm_path"),
        "verifier_artifact": verifier.get("source_artifact", str(DEFAULT_VERIFIER)),
        "frame_counts": framegroup.get("frame_counts"),
        "completion_claim": False,
        "full_trainer_claim": False,
        "quality_claim": False,
        "total_scale_first_to_last": framegroup.get("total_scale_first_to_last"),
        "backward_scale_first_to_last": framegroup.get("backward_scale_first_to_last"),
        "storage_scale_first_to_last": framegroup.get("storage_scale_first_to_last"),
        "mixed_128_total_max_ms": framegroup.get("mixed_128_total_max_ms"),
        "mixed_128_backward_max_ms": framegroup.get("mixed_128_backward_max_ms"),
        "confirm_128only": confirm,
        "thresholds": {
            "max_total_scale": thresholds.get("framegroup_max_total_scale") if isinstance(thresholds, dict) else None,
            "max_backward_scale": thresholds.get("framegroup_max_backward_scale")
            if isinstance(thresholds, dict)
            else None,
            "max_storage_scale": thresholds.get("framegroup_max_storage_scale")
            if isinstance(thresholds, dict)
            else None,
            "max_mixed_128_total_max_ms": thresholds.get("framegroup_max_mixed_128_total_max_ms")
            if isinstance(thresholds, dict)
            else None,
            "max_128only_total_median_ms": thresholds.get("framegroup_max_128only_total_median_ms")
            if isinstance(thresholds, dict)
            else None,
            "max_128only_total_max_ms": thresholds.get("framegroup_max_128only_total_max_ms")
            if isinstance(thresholds, dict)
            else None,
            "max_128only_backward_median_ms": thresholds.get("framegroup_max_128only_backward_median_ms")
            if isinstance(thresholds, dict)
            else None,
        },
        "conclusion": (
            "The selected row-reference framegroup16 fused-MSE kernel now reduces per-frame loss atomics inside "
            "each 16-frame threadgroup and is guarded by saved render32/site12 16/32/64/128 plus 128-only artifacts. "
            "The saved verifier shows sublinear frame scaling and removes the previous mixed-sweep 128f timing "
            "outlier, but this remains a fixed-geometry RGB-only site-RGBA microbench, not a full-trainer, quality, "
            "or STAR-UVT competitiveness claim."
        ),
    }


def _framegroup_compare_summary(verifier: dict[str, Any]) -> dict[str, Any]:
    compare = verifier.get("framegroup_compare_smoke")
    thresholds = verifier.get("thresholds")
    if not isinstance(compare, dict):
        return {"available": False}
    ratios_by_frame = compare.get("ratios_by_frame")
    psnr_delta_by_frame = compare.get("psnr_delta_by_frame")
    available = (
        verifier.get("status") == "ok"
        and isinstance(ratios_by_frame, dict)
        and isinstance(psnr_delta_by_frame, dict)
        and _finite_number(compare.get("total_ratio_16f"))
        and _finite_number(compare.get("backward_ratio_16f"))
        and _finite_number(compare.get("total_scale_first_to_last"))
        and _finite_number(compare.get("backward_scale_first_to_last"))
        and _finite_number(compare.get("storage_scale_first_to_last"))
        and _finite_number(compare.get("framegroup_storage_vs_full_16f"))
    )
    return {
        "available": available,
        "artifact": compare.get("path"),
        "verifier_artifact": verifier.get("source_artifact", str(DEFAULT_VERIFIER)),
        "frame_counts": compare.get("frame_counts"),
        "render_size": compare.get("render_size"),
        "site_count": compare.get("site_count"),
        "loaded_frame_count": compare.get("loaded_frame_count"),
        "real_loaded_frame_counts": compare.get("real_loaded_frame_counts"),
        "repeated_frame_counts": compare.get("repeated_frame_counts"),
        "repeat_scope_by_frame": compare.get("repeat_scope_by_frame"),
        "completion_claim": False,
        "full_trainer_claim": False,
        "quality_claim": False,
        "star_uvt_competitive_claim": False,
        "total_ratio_16f": compare.get("total_ratio_16f"),
        "backward_ratio_16f": compare.get("backward_ratio_16f"),
        "ratios_by_frame": ratios_by_frame,
        "psnr_delta_by_frame": psnr_delta_by_frame,
        "total_scale_first_to_last": compare.get("total_scale_first_to_last"),
        "backward_scale_first_to_last": compare.get("backward_scale_first_to_last"),
        "storage_scale_first_to_last": compare.get("storage_scale_first_to_last"),
        "framegroup_total_ms_16f": compare.get("framegroup_total_ms_16f"),
        "endpoint_total_ms_16f": compare.get("endpoint_total_ms_16f"),
        "framegroup_storage_vs_full_16f": compare.get("framegroup_storage_vs_full_16f"),
        "scope": compare.get("scope"),
        "thresholds": {
            "max_total_ratio_16f": thresholds.get("compare_max_framegroup_to_endpoint_total_16f")
            if isinstance(thresholds, dict)
            else None,
            "max_backward_ratio_16f": thresholds.get("compare_max_framegroup_to_endpoint_backward_16f")
            if isinstance(thresholds, dict)
            else None,
            "max_total_ratio_all_frames": thresholds.get("compare_max_framegroup_to_endpoint_total_all_frames")
            if isinstance(thresholds, dict)
            else None,
            "max_psnr_delta_16f": thresholds.get("compare_max_psnr_delta") if isinstance(thresholds, dict) else None,
            "max_psnr_delta_all_frames": thresholds.get("compare_max_psnr_delta_all_frames")
            if isinstance(thresholds, dict)
            else None,
            "max_storage_vs_full_16f": thresholds.get("compare_max_framegroup_storage_vs_full_16f")
            if isinstance(thresholds, dict)
            else None,
            "max_total_scale": thresholds.get("compare_max_framegroup_total_scale")
            if isinstance(thresholds, dict)
            else None,
            "max_backward_scale": thresholds.get("compare_max_framegroup_backward_scale")
            if isinstance(thresholds, dict)
            else None,
            "max_storage_scale": thresholds.get("compare_max_framegroup_storage_scale")
            if isinstance(thresholds, dict)
            else None,
        },
        "conclusion": (
            "The selected framegroup16 fused-MSE shader is now guarded by a paired render32/site12 "
            "16/32/64/128 compare against endpoint-run. It is faster than endpoint-run at every checked "
            "frame count and remains sublinear in total/backward time and storage, but the paired run keeps "
            "the explicit loaded-frame boundary: only 16f is a real loaded row, while 32/64/128f are "
            "synthetic repeated-fixture rows. It is not a stable benchmark, full-trainer, quality, or STAR-UVT "
            "competitiveness claim."
        ),
    }


def _framegroup_real32_compare_summary(verifier: dict[str, Any]) -> dict[str, Any]:
    compare = verifier.get("framegroup_real32_compare")
    thresholds = verifier.get("thresholds")
    if not isinstance(compare, dict):
        return {"available": False}
    ratios_by_frame = compare.get("ratios_by_frame")
    psnr_delta_by_frame = compare.get("psnr_delta_by_frame")
    storage_vs_endpoint_by_frame = compare.get("storage_vs_endpoint_by_frame")
    available = (
        verifier.get("status") == "ok"
        and isinstance(ratios_by_frame, dict)
        and isinstance(psnr_delta_by_frame, dict)
        and isinstance(storage_vs_endpoint_by_frame, dict)
        and _finite_number(compare.get("total_scale_first_to_last"))
        and _finite_number(compare.get("backward_scale_first_to_last"))
        and _finite_number(compare.get("storage_scale_first_to_last"))
    )
    return {
        "available": available,
        "artifact": compare.get("path"),
        "verifier_artifact": verifier.get("source_artifact", str(DEFAULT_VERIFIER)),
        "status": compare.get("status"),
        "frame_counts": compare.get("frame_counts"),
        "render_size": compare.get("render_size"),
        "site_count": compare.get("site_count"),
        "real_loaded_frame_counts": compare.get("real_loaded_frame_counts"),
        "repeated_frame_counts": compare.get("repeated_frame_counts"),
        "repeat_scope_by_frame": compare.get("repeat_scope_by_frame"),
        "completion_claim": False,
        "full_trainer_claim": False,
        "quality_claim": False,
        "star_uvt_competitive_claim": False,
        "real_frame_sublinear_claim": compare.get("real_frame_sublinear_claim"),
        "total_sublinear_real_frames": compare.get("total_sublinear_real_frames"),
        "backward_sublinear_real_frames": compare.get("backward_sublinear_real_frames"),
        "ratios_by_frame": ratios_by_frame,
        "psnr_delta_by_frame": psnr_delta_by_frame,
        "storage_vs_endpoint_by_frame": storage_vs_endpoint_by_frame,
        "frame_scale_first_to_last": compare.get("frame_scale_first_to_last"),
        "total_scale_first_to_last": compare.get("total_scale_first_to_last"),
        "backward_scale_first_to_last": compare.get("backward_scale_first_to_last"),
        "storage_scale_first_to_last": compare.get("storage_scale_first_to_last"),
        "scope": compare.get("scope"),
        "thresholds": {
            "max_total_ratio_all_frames": thresholds.get("real32_max_framegroup_to_endpoint_total_all_frames")
            if isinstance(thresholds, dict)
            else None,
            "max_backward_ratio_all_frames": thresholds.get("real32_max_framegroup_to_endpoint_backward_all_frames")
            if isinstance(thresholds, dict)
            else None,
            "max_psnr_delta_all_frames": thresholds.get("real32_max_psnr_delta_all_frames")
            if isinstance(thresholds, dict)
            else None,
            "max_total_scale": thresholds.get("real32_max_framegroup_total_scale")
            if isinstance(thresholds, dict)
            else None,
            "max_backward_scale": thresholds.get("real32_max_framegroup_backward_scale")
            if isinstance(thresholds, dict)
            else None,
            "max_storage_scale": thresholds.get("real32_max_framegroup_storage_scale")
            if isinstance(thresholds, dict)
            else None,
        },
        "conclusion": (
            "The selected framegroup16 fused-MSE shader now has a real-loaded render32/site12 16/32 compare "
            "with no repeated frames. It beats endpoint-run on total and backward time at both real frame counts "
            "and keeps compact storage. With the 32-frame chunk patch, measured total/backward scaling is "
            "sublinear on these real rows, but this is still a narrow shader-level result rather than a "
            "full-trainer, quality, or STAR-UVT competitiveness claim."
        ),
    }


def _framegroup_i16x4_compare_summary(verifier: dict[str, Any]) -> dict[str, Any]:
    compare = verifier.get("framegroup_i16x4_compare")
    thresholds = verifier.get("thresholds")
    if not isinstance(compare, dict):
        return {"available": False}
    frame_counts = compare.get("frame_counts")
    mode_statuses = compare.get("mode_statuses")
    ratios_by_frame = compare.get("ratios_by_frame")
    frame_scale = None
    if (
        isinstance(frame_counts, list)
        and len(frame_counts) >= 2
        and _finite_positive(frame_counts[0])
        and _finite_positive(frame_counts[-1])
    ):
        frame_scale = float(frame_counts[-1]) / float(frame_counts[0])
    total_scale = compare.get("i16x4_total_scale_first_to_last")
    backward_scale = compare.get("i16x4_backward_scale_first_to_last")
    i16x4_total_sublinear = (
        _finite_number(total_scale)
        and _finite_number(frame_scale)
        and float(total_scale) < float(frame_scale)
    )
    i16x4_backward_sublinear = (
        _finite_number(backward_scale)
        and _finite_number(frame_scale)
        and float(backward_scale) < float(frame_scale)
    )
    available = (
        verifier.get("status") == "ok"
        and isinstance(mode_statuses, dict)
        and isinstance(ratios_by_frame, dict)
        and compare.get("i16x4_speed_promotion_candidate") is False
        and i16x4_total_sublinear is False
        and i16x4_backward_sublinear is False
        and _finite_positive(total_scale)
        and _finite_positive(backward_scale)
        and _finite_positive(compare.get("max_i16x4_over_i16x3_total_mean_ratio"))
        and _finite_positive(compare.get("max_i16x4_over_i16x3_backward_mean_ratio"))
        and _finite_positive(compare.get("max_i16x4_over_i16x3_storage_ratio"))
        and _finite_number(compare.get("max_psnr_delta"))
    )
    return {
        "available": available,
        "artifact": compare.get("path"),
        "verifier_artifact": verifier.get("source_artifact", str(DEFAULT_VERIFIER)),
        "frame_counts": frame_counts,
        "frame_scale_first_to_last": frame_scale,
        "repeat_loaded_frames": compare.get("repeat_loaded_frames"),
        "mode_statuses": mode_statuses,
        "completion_claim": False,
        "full_trainer_claim": False,
        "quality_claim": False,
        "star_uvt_competitive_claim": False,
        "i16x4_speed_promotion_candidate": compare.get("i16x4_speed_promotion_candidate"),
        "i16x4_total_sublinear_claim": i16x4_total_sublinear,
        "i16x4_backward_sublinear_claim": i16x4_backward_sublinear,
        "i16x4_total_scale_first_to_last": total_scale,
        "i16x4_backward_scale_first_to_last": backward_scale,
        "max_i16x4_over_i16x3_total_mean_ratio": compare.get("max_i16x4_over_i16x3_total_mean_ratio"),
        "max_i16x4_over_i16x3_backward_mean_ratio": compare.get("max_i16x4_over_i16x3_backward_mean_ratio"),
        "max_i16x4_over_i16x3_storage_ratio": compare.get("max_i16x4_over_i16x3_storage_ratio"),
        "max_psnr_delta": compare.get("max_psnr_delta"),
        "ratios_by_frame": ratios_by_frame,
        "scope": compare.get("scope"),
        "thresholds": {
            "max_over_i16x3_total_mean_ratio": thresholds.get("i16x4_max_over_i16x3_total_mean_ratio")
            if isinstance(thresholds, dict)
            else None,
            "max_over_i16x3_backward_mean_ratio": thresholds.get("i16x4_max_over_i16x3_backward_mean_ratio")
            if isinstance(thresholds, dict)
            else None,
            "max_over_i16x3_storage_ratio": thresholds.get("i16x4_max_over_i16x3_storage_ratio")
            if isinstance(thresholds, dict)
            else None,
            "max_psnr_delta": thresholds.get("i16x4_max_psnr_delta") if isinstance(thresholds, dict) else None,
        },
        "nonpromotion_reason": (
            "The i16x4 fork is numerically correct and stays within the i16x3 mean timing/storage guard, but "
            "its own repeated-frame 16f->32f total/backward scale is far above the frame-count scale."
        ),
        "conclusion": (
            "The i16x4 framegroup16 fused-MSE fork is recorded as a negative/non-promotion artifact: PSNR matches "
            "i16x3 and storage overhead stays small, but the 16f->32f repeated-frame train/eval timing is not "
            "sublinear. It is not promoted, not full-trainer evidence, and not a STAR-UVT competitiveness claim."
        ),
    }


def _framegroup_i16x4_prewarm_compare_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path)}
    payload = _load_json(path)
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return {"available": False, "artifact": str(path), "status": payload.get("status")}
    ratios_by_frame = summary.get("ratios_by_frame")
    mode_statuses = payload.get("mode_statuses")
    max_total = summary.get("max_i16x4_over_i16x3_total_mean_ratio")
    max_backward = summary.get("max_i16x4_over_i16x3_backward_mean_ratio")
    max_psnr_delta = summary.get("max_psnr_delta")
    speed_rejected_by_ratio = (
        (_finite_positive(max_total) and float(max_total) > 1.05)
        or (_finite_positive(max_backward) and float(max_backward) > 1.05)
    )
    max_storage = None
    if isinstance(ratios_by_frame, dict):
        storages = [
            float(row["i16x4_over_i16x3_storage"])
            for row in ratios_by_frame.values()
            if isinstance(row, dict) and _finite_positive(row.get("i16x4_over_i16x3_storage"))
        ]
        max_storage = max(storages) if storages else None
    available = (
        payload.get("status") == "ok"
        and payload.get("prewarm_sweep") is True
        and payload.get("repeat_loaded_frames") is True
        and payload.get("frame_counts") == [16, 32]
        and isinstance(mode_statuses, dict)
        and isinstance(ratios_by_frame, dict)
        and summary.get("i16x4_speed_promotion_candidate") is False
        and summary.get("i16x4_total_sublinear") is True
        and summary.get("i16x4_backward_sublinear") is True
        and speed_rejected_by_ratio
        and _finite_number(max_psnr_delta)
        and _finite_positive(max_storage)
        and float(max_storage) <= 1.08
    )
    return {
        "available": available,
        "artifact": str(path),
        "status": payload.get("status"),
        "frame_counts": payload.get("frame_counts"),
        "render_size": payload.get("render_size"),
        "site_count": payload.get("site_count"),
        "steps": payload.get("steps"),
        "warmup_steps": payload.get("warmup_steps"),
        "prewarm_sweep": payload.get("prewarm_sweep"),
        "repeat_loaded_frames": payload.get("repeat_loaded_frames"),
        "mode_statuses": mode_statuses,
        "completion_claim": False,
        "full_trainer_claim": False,
        "quality_claim": False,
        "star_uvt_competitive_claim": False,
        "i16x4_speed_promotion_candidate": summary.get("i16x4_speed_promotion_candidate"),
        "i16x4_total_sublinear_claim": summary.get("i16x4_total_sublinear"),
        "i16x4_backward_sublinear_claim": summary.get("i16x4_backward_sublinear"),
        "i16x4_total_scale_first_to_last": summary.get("i16x4_total_scale_first_to_last"),
        "i16x4_backward_scale_first_to_last": summary.get("i16x4_backward_scale_first_to_last"),
        "max_i16x4_over_i16x3_total_mean_ratio": max_total,
        "max_i16x4_over_i16x3_backward_mean_ratio": max_backward,
        "max_i16x4_over_i16x3_storage_ratio": max_storage,
        "max_psnr_delta": max_psnr_delta,
        "ratios_by_frame": ratios_by_frame,
        "scope": payload.get("scope"),
        "speed_rejected_by_ratio": speed_rejected_by_ratio,
        "nonpromotion_reason": (
            "With prewarm-sweep enabled, the i16x4 fork is sublinear but still loses the 32f mean total/backward "
            "ratio to i16x3, so it is not a speed promotion."
        ),
        "conclusion": (
            "The prewarmed i16x4 framegroup16 fused-MSE compare is a second non-promotion artifact. It removes "
            "the earlier not-sublinear failure mode, but i16x4 is still slower than i16x3 at 32f mean total/backward "
            "time. Treat i16x4 as cadence-sensitive and not promoted; this is not full-trainer evidence and not a "
            "STAR-UVT competitiveness claim."
        ),
    }


def _framegroup_packed_prewarm_compare_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path)}
    payload = _load_json(path)
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return {"available": False, "artifact": str(path), "status": payload.get("status")}
    ratios_by_frame = summary.get("ratios_by_frame")
    mode_statuses = payload.get("mode_statuses")
    max_total = summary.get("max_packed_over_i16x3_total_mean_ratio")
    max_backward = summary.get("max_packed_over_i16x3_backward_mean_ratio")
    max_storage = summary.get("max_packed_over_i16x3_storage_ratio")
    max_psnr_delta = summary.get("max_psnr_delta")
    available = (
        payload.get("status") == "ok"
        and payload.get("prewarm_sweep") is True
        and payload.get("repeat_loaded_frames") is True
        and payload.get("frame_counts") == [16, 32]
        and isinstance(mode_statuses, dict)
        and isinstance(ratios_by_frame, dict)
        and summary.get("packed_speed_promotion_candidate") is True
        and summary.get("packed_storage_below_i16x3") is True
        and summary.get("packed_total_sublinear") is True
        and summary.get("packed_backward_sublinear") is True
        and _finite_positive(max_total)
        and float(max_total) <= 0.85
        and _finite_positive(max_backward)
        and float(max_backward) <= 0.90
        and _finite_positive(max_storage)
        and float(max_storage) < 1.0
        and _finite_number(max_psnr_delta)
        and float(max_psnr_delta) <= 1.0e-4
    )
    return {
        "available": available,
        "artifact": str(path),
        "status": payload.get("status"),
        "frame_counts": payload.get("frame_counts"),
        "render_size": payload.get("render_size"),
        "site_count": payload.get("site_count"),
        "steps": payload.get("steps"),
        "warmup_steps": payload.get("warmup_steps"),
        "prewarm_sweep": payload.get("prewarm_sweep"),
        "repeat_loaded_frames": payload.get("repeat_loaded_frames"),
        "mode_statuses": mode_statuses,
        "completion_claim": False,
        "full_trainer_claim": False,
        "quality_claim": False,
        "star_uvt_competitive_claim": False,
        "packed_speed_promotion_candidate": summary.get("packed_speed_promotion_candidate"),
        "packed_storage_below_i16x3": summary.get("packed_storage_below_i16x3"),
        "packed_total_sublinear_claim": summary.get("packed_total_sublinear"),
        "packed_backward_sublinear_claim": summary.get("packed_backward_sublinear"),
        "packed_total_scale_first_to_last": summary.get("packed_total_scale_first_to_last"),
        "packed_backward_scale_first_to_last": summary.get("packed_backward_scale_first_to_last"),
        "packed_storage_scale_first_to_last": summary.get("packed_storage_scale_first_to_last"),
        "max_packed_over_i16x3_total_mean_ratio": max_total,
        "max_packed_over_i16x3_backward_mean_ratio": max_backward,
        "max_packed_over_i16x3_storage_ratio": max_storage,
        "max_psnr_delta": max_psnr_delta,
        "ratios_by_frame": ratios_by_frame,
        "scope": payload.get("scope"),
        "candidate_reason": (
            "In the paired prewarmed 16/32 train/eval smoke, packed records beat selected i16x3 mean total/backward "
            "time at both rows, match PSNR, and use less selected tape storage."
        ),
        "conclusion": (
            "The packed-record framegroup16 fused-MSE fork is a speed-promotion candidate for the narrow paired "
            "prewarmed fixed-geometry smoke, but it is not full-trainer evidence and not a STAR-UVT competitiveness "
            "claim. Earlier standalone/interleaved probes were cadence-sensitive, so this needs a broader 64/128 or "
            "real-loaded guard before replacing i16x3 as the default speed path."
        ),
    }


def _framegroup_packed_broad_compare_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path)}
    payload = _load_json(path)
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        return {"available": False, "artifact": str(path), "status": payload.get("status")}
    ratios_by_frame = summary.get("ratios_by_frame")
    mode_statuses = payload.get("mode_statuses")
    max_total = summary.get("max_packed_over_i16x3_total_mean_ratio")
    max_backward = summary.get("max_packed_over_i16x3_backward_mean_ratio")
    max_storage = summary.get("max_packed_over_i16x3_storage_ratio")
    max_psnr_delta = summary.get("max_psnr_delta")
    ratio64 = ratios_by_frame.get("64") if isinstance(ratios_by_frame, dict) else None
    ratio128 = ratios_by_frame.get("128") if isinstance(ratios_by_frame, dict) else None
    ratio64_total = ratio64.get("packed_over_i16x3_total_mean") if isinstance(ratio64, dict) else None
    ratio64_backward = ratio64.get("packed_over_i16x3_backward_mean") if isinstance(ratio64, dict) else None
    ratio128_total = ratio128.get("packed_over_i16x3_total_mean") if isinstance(ratio128, dict) else None
    ratio128_backward = ratio128.get("packed_over_i16x3_backward_mean") if isinstance(ratio128, dict) else None
    speed_rejected_by_128 = (
        (_finite_positive(ratio128_total) and float(ratio128_total) > 1.05)
        or (_finite_positive(ratio128_backward) and float(ratio128_backward) > 1.05)
    )
    available = (
        payload.get("status") == "ok"
        and payload.get("prewarm_sweep") is True
        and payload.get("interleave_modes") is True
        and payload.get("repeat_loaded_frames") is True
        and payload.get("frame_counts") == [64, 128]
        and isinstance(mode_statuses, dict)
        and isinstance(ratios_by_frame, dict)
        and summary.get("packed_speed_promotion_candidate") is False
        and summary.get("packed_storage_below_i16x3") is True
        and summary.get("packed_total_sublinear") is True
        and summary.get("packed_backward_sublinear") is True
        and _finite_positive(ratio64_total)
        and float(ratio64_total) < 0.85
        and _finite_positive(ratio64_backward)
        and float(ratio64_backward) < 0.90
        and speed_rejected_by_128
        and _finite_positive(max_storage)
        and float(max_storage) < 1.0
        and _finite_number(max_psnr_delta)
        and float(max_psnr_delta) <= 1.0e-4
    )
    return {
        "available": available,
        "artifact": str(path),
        "status": payload.get("status"),
        "frame_counts": payload.get("frame_counts"),
        "render_size": payload.get("render_size"),
        "site_count": payload.get("site_count"),
        "steps": payload.get("steps"),
        "warmup_steps": payload.get("warmup_steps"),
        "prewarm_sweep": payload.get("prewarm_sweep"),
        "interleave_modes": payload.get("interleave_modes"),
        "repeat_loaded_frames": payload.get("repeat_loaded_frames"),
        "mode_statuses": mode_statuses,
        "completion_claim": False,
        "full_trainer_claim": False,
        "quality_claim": False,
        "star_uvt_competitive_claim": False,
        "packed_speed_promotion_candidate": summary.get("packed_speed_promotion_candidate"),
        "packed_storage_below_i16x3": summary.get("packed_storage_below_i16x3"),
        "packed_total_sublinear_claim": summary.get("packed_total_sublinear"),
        "packed_backward_sublinear_claim": summary.get("packed_backward_sublinear"),
        "packed_total_scale_first_to_last": summary.get("packed_total_scale_first_to_last"),
        "packed_backward_scale_first_to_last": summary.get("packed_backward_scale_first_to_last"),
        "packed_storage_scale_first_to_last": summary.get("packed_storage_scale_first_to_last"),
        "max_packed_over_i16x3_total_mean_ratio": max_total,
        "max_packed_over_i16x3_backward_mean_ratio": max_backward,
        "max_packed_over_i16x3_storage_ratio": max_storage,
        "max_psnr_delta": max_psnr_delta,
        "ratios_by_frame": ratios_by_frame,
        "scope": payload.get("scope"),
        "speed_rejected_by_128": speed_rejected_by_128,
        "candidate_scope_rejected": True,
        "nonpromotion_reason": (
            "Packed wins the interleaved measured 64f row, but loses mean total/backward time at 128f; keep the "
            "16/32 result as narrow evidence only."
        ),
        "conclusion": (
            "The packed-record framegroup16 fused-MSE fork is not broadly promoted by the 64/128 interleaved "
            "guard: storage remains below i16x3 and PSNR matches, but packed is slower than i16x3 at 128f mean "
            "total/backward time. This is still fixed-geometry smoke evidence, not full-trainer evidence and not "
            "a STAR-UVT competitiveness claim."
        ),
    }


def _star_speed_reference(
    path: Path,
    best: dict[str, Any] | None,
    block_coeff_train_eval: dict[str, Any] | None,
) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path)}
    payload = _load_json(path)
    rows = payload.get("summary_table")
    if not isinstance(rows, list):
        return {"available": False, "artifact": str(path), "status": payload.get("status")}
    star_rows = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("renderer") == "star_uvt" and int(row.get("target_size", 0)) == 32
    ]
    by_frame = {
        str(int(row["requested_frames"])): {
            "mean_step_ms": float(row["mean_step_s"]) * 1000.0,
            "mean_render_ms": float(row["mean_render_s"]) * 1000.0,
            "steps": int(row["steps"]),
            "warmup_steps": int(row["warmup_steps"]),
        }
        for row in star_rows
        if _finite_number(row.get("mean_step_s")) and _finite_number(row.get("mean_render_s"))
    }
    comparison: dict[str, Any] = {}
    scaling: dict[str, Any] = {}
    frame_keys = sorted((int(key) for key in by_frame), key=int)
    if len(frame_keys) >= 2:
        first_key = str(frame_keys[0])
        last_key = str(frame_keys[-1])
        frame_scale = float(frame_keys[-1]) / float(max(frame_keys[0], 1))
        step_scale = float(by_frame[last_key]["mean_step_ms"]) / max(float(by_frame[first_key]["mean_step_ms"]), 1.0e-9)
        render_scale = float(by_frame[last_key]["mean_render_ms"]) / max(
            float(by_frame[first_key]["mean_render_ms"]),
            1.0e-9,
        )
        scaling = {
            "first_frame_count": frame_keys[0],
            "last_frame_count": frame_keys[-1],
            "frame_scale_first_to_last": frame_scale,
            "mean_step_scale_first_to_last": step_scale,
            "mean_render_scale_first_to_last": render_scale,
            "step_runtime_sublinear_vs_frames": step_scale < frame_scale,
            "render_runtime_sublinear_vs_frames": render_scale < frame_scale,
            "scope_note": (
                "This is measured runtime scaling for the tiny fixed-step STAR reference. It is not a "
                "matched quality/capacity comparison and does not by itself prove representation quality."
            ),
        }
    if best is not None and "16" in by_frame:
        star_16_ms = float(by_frame["16"]["mean_step_ms"])
        world_16_ms = float(best["total_16f_ms"])
        comparison = {
            "world_foam_16f_total_ms": world_16_ms,
            "star_uvt_16f_mean_step_ms": star_16_ms,
            "world_foam_to_star_16f_step_ratio": world_16_ms / max(star_16_ms, 1.0e-9),
            "scope_note": (
                "Tiny 32px speed reference only: current World Foam is fixed-geometry/site-RGBA with 12 sites, "
                "while STAR UVT uses its world-tube model and is not a matched quality/capacity comparison."
            ),
        }
    block_coeff_comparison: dict[str, Any] = {}
    if block_coeff_train_eval is not None and "16" in by_frame:
        last_row = block_coeff_train_eval.get("last_row")
        if isinstance(last_row, dict) and _finite_number(last_row.get("total_ms")):
            star_16_ms = float(by_frame["16"]["mean_step_ms"])
            block_coeff_16_ms = float(last_row["total_ms"])
            block_coeff_comparison = {
                "block_coeff_16f_total_ms": block_coeff_16_ms,
                "star_uvt_16f_mean_step_ms": star_16_ms,
                "block_coeff_to_star_16f_step_ratio": block_coeff_16_ms / max(star_16_ms, 1.0e-9),
                "scope_note": (
                    "Tiny 32px speed reference only: the coefficient-cached World Foam sidecar is "
                    "fixed-geometry/site-RGBA and STAR UVT uses its world-tube model. This is not a matched "
                    "quality/capacity comparison and does not prove STAR-UVT competitiveness."
                ),
            }
    return {
        "available": payload.get("status") == "ok" and bool(by_frame),
        "artifact": str(path),
        "status": payload.get("status"),
        "renderer": "star_uvt",
        "target_size": 32,
        "by_frame": by_frame,
        "scaling": scaling,
        "comparison_to_current_world_foam": comparison,
        "comparison_to_block_coeff_sidecar": block_coeff_comparison,
    }


def _segment_tape_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path)}
    payload = _load_json(path)
    rows = payload.get("rows")
    last_row = rows[-1] if isinstance(rows, list) and rows and isinstance(rows[-1], dict) else {}
    tape = last_row.get("segment_tape", {}) if isinstance(last_row, dict) else {}
    return {
        "available": payload.get("status") == "ok",
        "artifact": str(path),
        "status": payload.get("status"),
        "claim": payload.get("claim"),
        "completion_claim": payload.get("completion_claim"),
        "max_forward_error_vs_current_mixed": payload.get("max_forward_error_vs_current_mixed"),
        "max_grad_rel_error_vs_current_reduce": payload.get("max_grad_rel_error_vs_current_reduce"),
        "max_grad_rel_error_vs_current_winner_grad_only": payload.get(
            "max_grad_rel_error_vs_current_winner_grad_only"
        ),
        "max_metal_forward_error_vs_current_mixed": payload.get("max_metal_forward_error_vs_current_mixed"),
        "max_metal_grad_rel_error_vs_current_reduce": payload.get("max_metal_grad_rel_error_vs_current_reduce"),
        "max_metal_grad_rel_error_vs_current_winner_grad_only": payload.get(
            "max_metal_grad_rel_error_vs_current_winner_grad_only"
        ),
        "max_metal_track_grad_rel_error_vs_current_winner_grad_only": payload.get(
            "max_metal_track_grad_rel_error_vs_current_winner_grad_only"
        ),
        "max_metal_track_grad_rel_error_vs_sample_atomic": payload.get(
            "max_metal_track_grad_rel_error_vs_sample_atomic"
        ),
        "frame_scale_first_to_last": payload.get("frame_scale_first_to_last"),
        "segment_scale_first_to_last": payload.get("segment_scale_first_to_last"),
        "active_segment_scale_first_to_last": payload.get("active_segment_scale_first_to_last"),
        "torch_tape_forward_scale_first_to_last": payload.get("torch_tape_forward_scale_first_to_last"),
        "metal_kernel_implemented": payload.get("structural_read", {}).get("metal_kernel_implemented")
        if isinstance(payload.get("structural_read"), dict)
        else None,
        "last_row": {
            "frames": last_row.get("frames"),
            "total_segments": tape.get("total_segments") if isinstance(tape, dict) else None,
            "avg_segments_per_sample": tape.get("avg_segments_per_sample") if isinstance(tape, dict) else None,
            "max_segments_per_sample": tape.get("max_segments_per_sample") if isinstance(tape, dict) else None,
            "compact_csr_storage_bytes": tape.get("compact_csr_storage_bytes") if isinstance(tape, dict) else None,
            "compact_csr_storage_vs_current_mixed_csr_plus_affine_ray": tape.get(
                "compact_csr_storage_vs_current_mixed_csr_plus_affine_ray"
            )
            if isinstance(tape, dict)
            else None,
            "metal_segment_tape_forward_ms": last_row.get("timing_ms", {}).get("metal_segment_tape_forward")
            if isinstance(last_row.get("timing_ms"), dict)
            else None,
            "metal_segment_tape_grad_only_ms": last_row.get("timing_ms", {}).get("metal_segment_tape_grad_only")
            if isinstance(last_row.get("timing_ms"), dict)
            else None,
            "metal_segment_tape_track_grad_only_ms": last_row.get("timing_ms", {}).get(
                "metal_segment_tape_track_grad_only"
            )
            if isinstance(last_row.get("timing_ms"), dict)
            else None,
        },
        "conclusion": (
            "Geometry-only compact segment-tape Metal replay matches the current fused forward/VJP contract, "
            "but the naive per-sample tape scales roughly with frame count."
        ),
    }


def _framegroup_autograd_smoke_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path)}
    payload = _load_json(path)
    rows = payload.get("rows")
    row = rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], dict) else {}
    step_summary = row.get("step_summary", {}) if isinstance(row, dict) else {}
    total_summary = step_summary.get("total", {}) if isinstance(step_summary, dict) else {}
    backward_summary = step_summary.get("backward", {}) if isinstance(step_summary, dict) else {}
    fused_summary = step_summary.get("fused_loss_vjp", {}) if isinstance(step_summary, dict) else {}
    render_summary = step_summary.get("render", {}) if isinstance(step_summary, dict) else {}
    acceptance = row.get("acceptance") if isinstance(row, dict) else None
    objective_adapter = payload.get("world_foam_objective_adapter")
    objective_adapter_rows_all_match = payload.get("world_foam_objective_adapter_rows_all_match")
    available = (
        payload.get("status") == "ok"
        and payload.get("optimizer_mode") == "autograd"
        and payload.get("tape_mode") == "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse"
        and _framegroup_objective_adapter_available(objective_adapter)
        and objective_adapter_rows_all_match is True
        and isinstance(acceptance, dict)
        and acceptance.get("gradients_nonzero") is True
        and acceptance.get("parameters_updated") is True
        and _finite_positive(row.get("first_grad_abs_sum"))
        and _finite_positive(row.get("parameter_update_abs_max"))
        and _finite_positive(fused_summary.get("mean_s"))
    )
    return {
        "available": available,
        "artifact": str(path),
        "status": payload.get("status"),
        "tape_mode": payload.get("tape_mode"),
        "optimizer_mode": payload.get("optimizer_mode"),
        "frame_counts": payload.get("frame_counts"),
        "render_size": payload.get("render_size"),
        "site_count": payload.get("site_count"),
        "completion_claim": False,
        "full_trainer_claim": payload.get("full_trainer_claim"),
        "quality_claim": payload.get("quality_claim"),
        "gradient_scope": payload.get("gradient_scope"),
        "world_foam_objective_adapter": objective_adapter,
        "world_foam_objective_adapter_rows_all_match": objective_adapter_rows_all_match,
        "row": {
            "frame_count": row.get("frame_count"),
            "loaded_frame_count": row.get("loaded_frame_count"),
            "steps": row.get("steps"),
            "warmup_steps": row.get("warmup_steps"),
            "total_ms": float(total_summary.get("mean_s", 0.0)) * 1000.0,
            "backward_ms": float(backward_summary.get("mean_s", 0.0)) * 1000.0,
            "fused_loss_vjp_ms": float(fused_summary.get("mean_s", 0.0)) * 1000.0,
            "render_ms": float(render_summary.get("mean_s", 0.0)) * 1000.0,
            "first_grad_abs_sum": row.get("first_grad_abs_sum"),
            "parameter_update_abs_max": row.get("parameter_update_abs_max"),
            "final_train_psnr": row.get("final_train_psnr"),
            "final_heldout_psnr": row.get("final_heldout_psnr"),
        },
        "acceptance": acceptance,
        "conclusion": (
            "The promoted framegroup16 fused-MSE shader now has a narrow autograd-facing train/eval smoke: "
            "the fused loss runs through WorldFoamFrozenRGBMSEObjective, participates in `.backward()`, produces "
            "nonzero site-RGBA gradients, and updates parameters through the existing harness. This removes a "
            "manual-VJP-only interface gap for the selected shader, but it is still a fixed-geometry smoke and "
            "not full trainer or quality parity."
        ),
    }


def _framegroup_autograd_speedscale_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path)}
    payload = _load_json(path)
    rows = payload.get("rows")
    rows = rows if isinstance(rows, list) else []
    by_frame: dict[str, dict[str, Any]] = {}
    real_loaded: list[int] = []
    repeated: list[int] = []
    repeat_scope_by_frame: dict[str, str] = {}
    all_row_acceptance = True
    for raw_row in rows:
        if not isinstance(raw_row, dict):
            continue
        frame_count = int(raw_row.get("frame_count", 0))
        frame_key = str(frame_count)
        repeated_row = raw_row.get("repeat_loaded_frames") is True
        if repeated_row:
            repeated.append(frame_count)
        else:
            real_loaded.append(frame_count)
        repeat_scope_by_frame[frame_key] = str(raw_row.get("repeat_loaded_frames_scope", ""))
        step_summary = raw_row.get("step_summary", {})
        step_summary = step_summary if isinstance(step_summary, dict) else {}
        total_summary = step_summary.get("total", {})
        backward_summary = step_summary.get("backward", {})
        fused_summary = step_summary.get("fused_loss_vjp", {})
        render_summary = step_summary.get("render", {})
        total_summary = total_summary if isinstance(total_summary, dict) else {}
        backward_summary = backward_summary if isinstance(backward_summary, dict) else {}
        fused_summary = fused_summary if isinstance(fused_summary, dict) else {}
        render_summary = render_summary if isinstance(render_summary, dict) else {}
        acceptance = raw_row.get("acceptance")
        row_acceptance = (
            isinstance(acceptance, dict)
            and acceptance.get("gradients_nonzero") is True
            and acceptance.get("parameters_updated") is True
            and acceptance.get("outputs_are_finite") is True
            and _framegroup_objective_adapter_available(raw_row.get("world_foam_objective_adapter"))
        )
        all_row_acceptance = all_row_acceptance and row_acceptance
        by_frame[frame_key] = {
            "loaded_frame_count": raw_row.get("loaded_frame_count"),
            "repeat_loaded_frames": repeated_row,
            "repeat_loaded_frames_scope": repeat_scope_by_frame[frame_key],
            "steps": raw_row.get("steps"),
            "warmup_steps": raw_row.get("warmup_steps"),
            "total_ms": float(total_summary.get("mean_s", 0.0)) * 1000.0,
            "backward_ms": float(backward_summary.get("mean_s", 0.0)) * 1000.0,
            "fused_loss_vjp_ms": float(fused_summary.get("mean_s", 0.0)) * 1000.0,
            "render_ms": float(render_summary.get("mean_s", 0.0)) * 1000.0,
            "first_grad_abs_sum": raw_row.get("first_grad_abs_sum"),
            "parameter_update_abs_max": raw_row.get("parameter_update_abs_max"),
            "final_train_psnr": raw_row.get("final_train_psnr"),
            "final_heldout_psnr": raw_row.get("final_heldout_psnr"),
            "world_foam_objective_adapter": raw_row.get("world_foam_objective_adapter"),
        }
    expected_frames = [16, 32, 64, 128]
    acceptance = payload.get("acceptance")
    total_scale = payload.get("total_step_scale_first_to_last")
    backward_scale = payload.get("backward_scale_first_to_last")
    storage_scale = payload.get("selected_tape_storage_scale_first_to_last")
    objective_adapter = payload.get("world_foam_objective_adapter")
    objective_adapter_rows_all_match = payload.get("world_foam_objective_adapter_rows_all_match")
    available = (
        payload.get("status") == "ok"
        and payload.get("optimizer_mode") == "autograd"
        and payload.get("tape_mode") == "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse"
        and _framegroup_objective_adapter_available(objective_adapter)
        and objective_adapter_rows_all_match is True
        and payload.get("frame_counts") == expected_frames
        and payload.get("render_size") == 32
        and payload.get("site_count") == 12
        and isinstance(acceptance, dict)
        and acceptance.get("all_rows_ok") is True
        and acceptance.get("total_step_sublinear_vs_frames") is True
        and acceptance.get("backward_sublinear_vs_frames") is True
        and all_row_acceptance
        and sorted(real_loaded) == [16, 32]
        and sorted(repeated) == [64, 128]
        and _finite_positive(total_scale)
        and _finite_positive(backward_scale)
        and _finite_positive(storage_scale)
        and all(
            _finite_positive(by_frame.get(str(frame), {}).get(key))
            for frame in expected_frames
            for key in ("total_ms", "backward_ms", "fused_loss_vjp_ms", "first_grad_abs_sum", "parameter_update_abs_max")
        )
    )
    return {
        "available": available,
        "artifact": str(path),
        "status": payload.get("status"),
        "tape_mode": payload.get("tape_mode"),
        "optimizer_mode": payload.get("optimizer_mode"),
        "frame_counts": payload.get("frame_counts"),
        "real_loaded_frame_counts": sorted(real_loaded),
        "repeated_frame_counts": sorted(repeated),
        "repeat_scope_by_frame": repeat_scope_by_frame,
        "render_size": payload.get("render_size"),
        "site_count": payload.get("site_count"),
        "steps": rows[0].get("steps") if rows and isinstance(rows[0], dict) else None,
        "warmup_steps": rows[0].get("warmup_steps") if rows and isinstance(rows[0], dict) else None,
        "completion_claim": False,
        "full_trainer_claim": payload.get("full_trainer_claim"),
        "full_geometry_gradient_claim": payload.get("full_geometry_gradient_claim"),
        "quality_claim": payload.get("quality_claim"),
        "gradient_scope": payload.get("gradient_scope"),
        "world_foam_objective_adapter": objective_adapter,
        "world_foam_objective_adapter_rows_all_match": objective_adapter_rows_all_match,
        "total_scale_first_to_last": total_scale,
        "backward_scale_first_to_last": backward_scale,
        "selected_tape_storage_scale_first_to_last": storage_scale,
        "acceptance": acceptance,
        "by_frame": by_frame,
        "conclusion": (
            "The promoted framegroup16 fused-MSE shader now has a warmed multi-frame autograd speedscale: "
            "16f and 32f are real-loaded rows, 64f and 128f are repeated-fixture rows, and "
            "WorldFoamFrozenRGBMSEObjective keeps `.backward()` total/backward/storage scaling sublinear "
            "through the selected fused-loss path. This is still fixed-geometry site-RGBA training, not "
            "full trainer, geometry-gradient, or STAR-UVT quality parity."
        ),
    }


def _topology_sharing_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path)}
    payload = _load_json(path)
    rows = payload.get("rows")
    last_row = rows[-1] if isinstance(rows, list) and rows and isinstance(rows[-1], dict) else {}
    topology = last_row.get("topology", {}) if isinstance(last_row, dict) else {}
    acceptance = payload.get("acceptance", {})
    return {
        "available": payload.get("status") == "informational" and isinstance(acceptance, dict),
        "artifact": str(path),
        "status": payload.get("status"),
        "completion_claim": payload.get("completion_claim"),
        "star_uvt_competitive_claim": payload.get("star_uvt_competitive_claim"),
        "frame_scale_first_to_last": payload.get("frame_scale_first_to_last"),
        "segment_scale_first_to_last": payload.get("segment_scale_first_to_last"),
        "track_unique_topology_row_scale_first_to_last": payload.get(
            "track_unique_topology_row_scale_first_to_last"
        ),
        "acceptance": acceptance,
        "last_row": {
            "frames": last_row.get("frames"),
            "same_topology_all_frames_tracks": topology.get("same_topology_all_frames_tracks")
            if isinstance(topology, dict)
            else None,
            "track_unique_topology_rows": topology.get("track_unique_topology_rows")
            if isinstance(topology, dict)
            else None,
            "track_unique_topology_rows_vs_samples": topology.get("track_unique_topology_rows_vs_samples")
            if isinstance(topology, dict)
            else None,
            "global_unique_owner_sequences_vs_samples": topology.get("global_unique_owner_sequences_vs_samples")
            if isinstance(topology, dict)
            else None,
            "frame_to_frame_topology_transition_rate": topology.get("frame_to_frame_topology_transition_rate")
            if isinstance(topology, dict)
            else None,
        },
        "conclusion": (
            "Simple owner-topology sharing is weak on the moving-camera probe: no tracks keep one owner "
            "sequence across all frames, and 16f per-track unique topology rows remain close to the full "
            "sample count."
        ),
    }


def _delta_tape_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path)}
    payload = _load_json(path)
    rows = payload.get("rows")
    last_row = rows[-1] if isinstance(rows, list) and rows and isinstance(rows[-1], dict) else {}
    delta = last_row.get("delta_tape", {}) if isinstance(last_row, dict) else {}
    storage = delta.get("storage_estimates", {}) if isinstance(delta, dict) else {}
    acceptance = payload.get("acceptance", {})
    return {
        "available": payload.get("status") in {"ok", "informational"} and isinstance(acceptance, dict),
        "artifact": str(path),
        "status": payload.get("status"),
        "completion_claim": payload.get("completion_claim"),
        "star_uvt_competitive_claim": payload.get("star_uvt_competitive_claim"),
        "acceptance": acceptance,
        "frame_scale_first_to_last": payload.get("frame_scale_first_to_last"),
        "full_segment_scale_first_to_last": payload.get("full_segment_scale_first_to_last"),
        "change_event_scale_first_to_last": payload.get("change_event_scale_first_to_last"),
        "edit_op_scale_first_to_last": payload.get("edit_op_scale_first_to_last"),
        "delta_owner_storage_scale_first_to_last": payload.get("delta_owner_storage_scale_first_to_last"),
        "full_compact_csr_storage_scale_first_to_last": payload.get(
            "full_compact_csr_storage_scale_first_to_last"
        ),
        "last_row": {
            "frames": last_row.get("frames"),
            "total_segments": delta.get("total_segments") if isinstance(delta, dict) else None,
            "change_events": delta.get("change_events") if isinstance(delta, dict) else None,
            "edit_ops_total": delta.get("edit_ops_total") if isinstance(delta, dict) else None,
            "edit_ops_per_transition": delta.get("edit_ops_per_transition") if isinstance(delta, dict) else None,
            "delta_replace_owner_sequence_bytes": storage.get("delta_replace_owner_sequence_bytes")
            if isinstance(storage, dict)
            else None,
            "delta_replace_owner_sequence_vs_full_compact_csr": storage.get(
                "delta_replace_owner_sequence_vs_full_compact_csr"
            )
            if isinstance(storage, dict)
            else None,
            "delta_edit_op_stream_owner_only_vs_full_compact_csr": storage.get(
                "delta_edit_op_stream_owner_only_vs_full_compact_csr"
            )
            if isinstance(storage, dict)
            else None,
            "delta_replace_geometry_rows_vs_full_compact_csr": storage.get(
                "delta_replace_geometry_rows_vs_full_compact_csr"
            )
            if isinstance(storage, dict)
            else None,
        },
        "conclusion": (
            "Frame-to-frame owner edit operations are a promising sublinear topology signal, but coarse "
            "changed-row events are not sublinear and exact replay still needs a compact length/mid model."
        ),
    }


def _boundary_delta_tape_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path)}
    payload = _load_json(path)
    rows = payload.get("rows")
    last_row = rows[-1] if isinstance(rows, list) and rows and isinstance(rows[-1], dict) else {}
    delta = last_row.get("boundary_delta_tape", {}) if isinstance(last_row, dict) else {}
    storage = delta.get("storage_estimates", {}) if isinstance(delta, dict) else {}
    acceptance = payload.get("acceptance", {})
    return {
        "available": payload.get("status") == "informational" and isinstance(acceptance, dict),
        "artifact": str(path),
        "status": payload.get("status"),
        "completion_claim": payload.get("completion_claim"),
        "star_uvt_competitive_claim": payload.get("star_uvt_competitive_claim"),
        "acceptance": acceptance,
        "frame_scale_first_to_last": payload.get("frame_scale_first_to_last"),
        "full_boundary_count_scale_first_to_last": payload.get("full_boundary_count_scale_first_to_last"),
        "boundary_edit_op_scale_first_to_last": payload.get("boundary_edit_op_scale_first_to_last"),
        "delta_replace_boundary_storage_scale_first_to_last": payload.get(
            "delta_replace_boundary_storage_scale_first_to_last"
        ),
        "last_row": {
            "frames": last_row.get("frames"),
            "total_boundaries": delta.get("total_boundaries") if isinstance(delta, dict) else None,
            "total_segments_implied": delta.get("total_segments_implied") if isinstance(delta, dict) else None,
            "change_events": delta.get("change_events") if isinstance(delta, dict) else None,
            "edit_ops_total": delta.get("edit_ops_total") if isinstance(delta, dict) else None,
            "edit_ops_per_transition": delta.get("edit_ops_per_transition") if isinstance(delta, dict) else None,
            "delta_replace_boundary_order_vs_full_segment_csr": storage.get(
                "delta_replace_boundary_order_vs_full_segment_csr"
            )
            if isinstance(storage, dict)
            else None,
            "delta_edit_op_stream_boundary_only_vs_full_segment_csr": storage.get(
                "delta_edit_op_stream_boundary_only_vs_full_segment_csr"
            )
            if isinstance(storage, dict)
            else None,
        },
        "conclusion": (
            "Boundary-order deltas are closer to exact length/mid replay because boundary ids plus rational "
            "depth coefficients recover segment geometry, but raw all-boundary order is still noisy and owner "
            "assignment remains unresolved."
        ),
    }


def _record_delta_tape_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path)}
    payload = _load_json(path)
    rows = payload.get("rows")
    last_row = rows[-1] if isinstance(rows, list) and rows and isinstance(rows[-1], dict) else {}
    record = last_row.get("record_delta_tape", {}) if isinstance(last_row, dict) else {}
    storage = record.get("storage_estimates", {}) if isinstance(record, dict) else {}
    verification = last_row.get("segment_tape_verification", {}) if isinstance(last_row, dict) else {}
    acceptance = payload.get("acceptance", {})
    return {
        "available": payload.get("status") == "informational" and isinstance(acceptance, dict),
        "artifact": str(path),
        "status": payload.get("status"),
        "completion_claim": payload.get("completion_claim"),
        "star_uvt_competitive_claim": payload.get("star_uvt_competitive_claim"),
        "acceptance": acceptance,
        "frame_scale_first_to_last": payload.get("frame_scale_first_to_last"),
        "full_record_count_scale_first_to_last": payload.get("full_record_count_scale_first_to_last"),
        "record_edit_op_scale_first_to_last": payload.get("record_edit_op_scale_first_to_last"),
        "delta_replace_record_storage_scale_first_to_last": payload.get(
            "delta_replace_record_storage_scale_first_to_last"
        ),
        "last_row": {
            "frames": last_row.get("frames"),
            "total_records": record.get("total_records") if isinstance(record, dict) else None,
            "change_events": record.get("change_events") if isinstance(record, dict) else None,
            "change_event_rate": record.get("change_event_rate") if isinstance(record, dict) else None,
            "edit_ops_total": record.get("edit_ops_total") if isinstance(record, dict) else None,
            "edit_ops_per_transition": record.get("edit_ops_per_transition") if isinstance(record, dict) else None,
            "record_counts_match_segment_tape": verification.get("matches_segment_tape_counts_and_owners")
            if isinstance(verification, dict)
            else None,
            "delta_replace_record_vs_full_segment_csr": storage.get(
                "delta_replace_record_vs_full_segment_csr"
            )
            if isinstance(storage, dict)
            else None,
            "delta_edit_op_record_stream_vs_full_segment_csr": storage.get(
                "delta_edit_op_record_stream_vs_full_segment_csr"
            )
            if isinstance(storage, dict)
            else None,
        },
        "conclusion": (
            "Exact owner+boundary-cut record deltas preserve segment-tape counts/owners and recover "
            "length/mid from boundary ids, but the replacement record stream is about full-CSR sized and "
            "edit-op replay is still not a compact STAR-like exact tape."
        ),
    }


def _owner_run_tape_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path)}
    payload = _load_json(path)
    rows = payload.get("rows")
    last_row = rows[-1] if isinstance(rows, list) and rows and isinstance(rows[-1], dict) else {}
    timing = last_row.get("timing_ms", {}) if isinstance(last_row, dict) else {}
    return {
        "available": payload.get("status") in {"ok", "informational"},
        "artifact": str(path),
        "status": payload.get("status"),
        "completion_claim": payload.get("completion_claim"),
        "star_uvt_competitive_claim": payload.get("star_uvt_competitive_claim"),
        "acceptance": payload.get("acceptance"),
        "frame_scale_first_to_last": payload.get("frame_scale_first_to_last"),
        "full_segment_scale_first_to_last": payload.get("full_segment_scale_first_to_last"),
        "owner_run_segment_scale_first_to_last": payload.get("owner_run_segment_scale_first_to_last"),
        "max_forward_rgb_abs_error": payload.get("max_forward_rgb_abs_error"),
        "max_forward_alpha_abs_error": payload.get("max_forward_alpha_abs_error"),
        "max_forward_depth_abs_error": payload.get("max_forward_depth_abs_error"),
        "max_rgb_only_vjp_rel_error": payload.get("max_rgb_only_vjp_rel_error"),
        "last_row": {
            "frames": last_row.get("frames"),
            "full_segments": last_row.get("full_segments"),
            "owner_run_segments": last_row.get("owner_run_segments"),
            "owner_run_segments_vs_full_segments": last_row.get("owner_run_segments_vs_full_segments"),
            "owner_run_storage_vs_full": last_row.get("owner_run_storage_vs_full"),
            "full_rgb_only_grad_ms": timing.get("full_rgb_only_grad") if isinstance(timing, dict) else None,
            "owner_run_rgb_only_grad_ms": timing.get("owner_run_rgb_only_grad") if isinstance(timing, dict) else None,
            "full_forward_ms": timing.get("full_forward") if isinstance(timing, dict) else None,
            "owner_run_forward_ms": timing.get("owner_run_forward") if isinstance(timing, dict) else None,
        },
        "conclusion": (
            "Same-owner run compression reuses the compact segment-tape Metal kernels and is a strong "
            "RGB-training candidate: it preserves RGB/alpha and RGB-only VJP while cutting the 16f tape "
            "to about ten percent of full segments. Depth uses a current-density effective mid, so this is "
            "not yet a final density-independent geometry tape."
        ),
    }


def _owner_run_boundary_tape_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path)}
    payload = _load_json(path)
    rows = payload.get("rows")
    last_row = rows[-1] if isinstance(rows, list) and rows and isinstance(rows[-1], dict) else {}
    tape = last_row.get("owner_run_boundary_tape", {}) if isinstance(last_row, dict) else {}
    endpoint = last_row.get("endpoint_density_replay", {}) if isinstance(last_row, dict) else {}
    storage = tape.get("storage_estimates", {}) if isinstance(tape, dict) else {}
    return {
        "available": payload.get("status") in {"ok", "informational"},
        "artifact": str(path),
        "status": payload.get("status"),
        "completion_claim": payload.get("completion_claim"),
        "star_uvt_competitive_claim": payload.get("star_uvt_competitive_claim"),
        "acceptance": payload.get("acceptance"),
        "frame_scale_first_to_last": payload.get("frame_scale_first_to_last"),
        "owner_run_boundary_run_scale_first_to_last": payload.get("owner_run_boundary_run_scale_first_to_last"),
        "owner_run_boundary_storage_scale_first_to_last": payload.get(
            "owner_run_boundary_storage_scale_first_to_last"
        ),
        "max_endpoint_length_abs_error": payload.get("max_endpoint_length_abs_error"),
        "max_endpoint_density_depth_abs_error_vs_current_owner_run": payload.get(
            "max_endpoint_density_depth_abs_error_vs_current_owner_run"
        ),
        "last_row": {
            "frames": last_row.get("frames"),
            "total_runs": tape.get("total_runs") if isinstance(tape, dict) else None,
            "full_original_segments": tape.get("full_original_segments") if isinstance(tape, dict) else None,
            "runs_vs_full_original_segments": tape.get("runs_vs_full_original_segments")
            if isinstance(tape, dict)
            else None,
            "max_segments_per_run": tape.get("max_segments_per_run") if isinstance(tape, dict) else None,
            "owner_run_boundary_id_vs_full_segment_csr": storage.get(
                "owner_run_boundary_id_vs_full_segment_csr"
            )
            if isinstance(storage, dict)
            else None,
            "owner_run_boundary_id_vs_active_segment_csr": storage.get(
                "owner_run_boundary_id_vs_active_segment_csr"
            )
            if isinstance(storage, dict)
            else None,
            "owner_run_boundary_id_vs_current_owner_run_length_mid_csr": storage.get(
                "owner_run_boundary_id_vs_current_owner_run_length_mid_csr"
            )
            if isinstance(storage, dict)
            else None,
            "max_endpoint_alpha_abs_error_vs_current_owner_run": endpoint.get(
                "max_endpoint_alpha_abs_error_vs_current_owner_run"
            )
            if isinstance(endpoint, dict)
            else None,
            "max_endpoint_density_mid_abs_error_vs_current_owner_run": endpoint.get(
                "max_endpoint_density_mid_abs_error_vs_current_owner_run"
            )
            if isinstance(endpoint, dict)
            else None,
            "mean_endpoint_density_mid_abs_error_vs_current_owner_run": endpoint.get(
                "mean_endpoint_density_mid_abs_error_vs_current_owner_run"
            )
            if isinstance(endpoint, dict)
            else None,
            "max_endpoint_density_depth_abs_error_vs_current_owner_run": endpoint.get(
                "max_endpoint_density_depth_abs_error_vs_current_owner_run"
            )
            if isinstance(endpoint, dict)
            else None,
            "mean_endpoint_density_depth_abs_error_vs_current_owner_run": endpoint.get(
                "mean_endpoint_density_depth_abs_error_vs_current_owner_run"
            )
            if isinstance(endpoint, dict)
            else None,
        },
        "conclusion": (
            "Owner-run boundary endpoint records match the current owner-run tape counts/owners and recover "
            "run lengths from boundary ids plus ray coefficients. Endpoint-only continuous density depth does "
            "not match the current segment-mid depth after same-owner internal cuts are discarded, and run "
            "count still scales worse than frame count."
        ),
    }


def _owner_run_internal_tape_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path)}
    payload = _load_json(path)
    rows = payload.get("rows")
    last_row = rows[-1] if isinstance(rows, list) and rows and isinstance(rows[-1], dict) else {}
    active = last_row.get("active_internal_owner_run_tape", {}) if isinstance(last_row, dict) else {}
    all_internal = last_row.get("all_internal_owner_run_tape", {}) if isinstance(last_row, dict) else {}
    density_errors = last_row.get("density_scale_errors", {}) if isinstance(last_row, dict) else {}
    density_05 = density_errors.get("0.5", {}) if isinstance(density_errors, dict) else {}
    active_05 = density_05.get("active_internal_vs_full", {}) if isinstance(density_05, dict) else {}
    density_1 = density_errors.get("1", {}) if isinstance(density_errors, dict) else {}
    active_1 = density_1.get("active_internal_vs_full", {}) if isinstance(density_1, dict) else {}
    return {
        "available": payload.get("status") == "informational",
        "artifact": str(path),
        "status": payload.get("status"),
        "completion_claim": payload.get("completion_claim"),
        "star_uvt_competitive_claim": payload.get("star_uvt_competitive_claim"),
        "acceptance": payload.get("acceptance"),
        "frame_scale_first_to_last": payload.get("frame_scale_first_to_last"),
        "active_internal_segment_count_scale_first_to_last": payload.get(
            "active_internal_segment_count_scale_first_to_last"
        ),
        "all_internal_segment_count_scale_first_to_last": payload.get(
            "all_internal_segment_count_scale_first_to_last"
        ),
        "last_row": {
            "frames": last_row.get("frames"),
            "full_segment_count": last_row.get("full_segment_count"),
            "active_segment_count": last_row.get("active_segment_count"),
            "active_segment_csr_vs_full_segment_csr": last_row.get("active_segment_csr_vs_full_segment_csr"),
            "active_internal_run_count": active.get("run_count") if isinstance(active, dict) else None,
            "active_internal_segment_count": active.get("internal_segment_count")
            if isinstance(active, dict)
            else None,
            "active_internal_endpoint_run_csr_vs_full_segment_csr": active.get(
                "endpoint_run_csr_vs_full_segment_csr"
            )
            if isinstance(active, dict)
            else None,
            "active_internal_nested_csr_vs_full_segment_csr": active.get("nested_csr_vs_full_segment_csr")
            if isinstance(active, dict)
            else None,
            "all_internal_run_count": all_internal.get("run_count") if isinstance(all_internal, dict) else None,
            "all_internal_segment_count": all_internal.get("internal_segment_count")
            if isinstance(all_internal, dict)
            else None,
            "all_internal_endpoint_run_csr_vs_full_segment_csr": all_internal.get(
                "endpoint_run_csr_vs_full_segment_csr"
            )
            if isinstance(all_internal, dict)
            else None,
            "all_internal_nested_csr_vs_full_segment_csr": all_internal.get("nested_csr_vs_full_segment_csr")
            if isinstance(all_internal, dict)
            else None,
            "active_current_density_depth_max_abs": active_1.get("depth", {}).get("max_abs")
            if isinstance(active_1.get("depth"), dict)
            else None,
            "active_half_density_alpha_max_abs": active_05.get("alpha", {}).get("max_abs")
            if isinstance(active_05.get("alpha"), dict)
            else None,
            "active_half_density_depth_max_abs": active_05.get("depth", {}).get("max_abs")
            if isinstance(active_05.get("depth"), dict)
            else None,
        },
        "conclusion": (
            "Internal cuts close exact current-depth replay at the reference density, but active-only cuts are "
            "not density independent because lower density can reactivate threshold-truncated segments. Keeping "
            "all internal cuts preserves density-independent replay but moves storage back toward the full "
            "segment tape. All-owner-run endpoints are compact only if depth semantics change to continuous "
            "absorption within a same-owner run."
        ),
    }


def _endpoint_run_tape_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path)}
    payload = _load_json(path)
    rows = payload.get("rows")
    last_row = rows[-1] if isinstance(rows, list) and rows and isinstance(rows[-1], dict) else {}
    acceptance = payload.get("acceptance")
    return {
        "available": isinstance(acceptance, dict)
        and acceptance.get("metal_forward_matches_torch_continuous_endpoint_replay") is True
        and acceptance.get("metal_vjp_matches_torch_autograd") is True,
        "artifact": str(path),
        "status": payload.get("status"),
        "completion_claim": payload.get("completion_claim"),
        "star_uvt_competitive_claim": payload.get("star_uvt_competitive_claim"),
        "acceptance": acceptance,
        "frame_scale_first_to_last": payload.get("frame_scale_first_to_last"),
        "endpoint_run_scale_first_to_last": payload.get("endpoint_run_scale_first_to_last"),
        "max_forward_abs_error_vs_torch": payload.get("max_forward_abs_error_vs_torch"),
        "max_vjp_rel_error_vs_torch_autograd": payload.get("max_vjp_rel_error_vs_torch_autograd"),
        "last_row": {
            "frames": last_row.get("frames"),
            "full_segments": last_row.get("full_segments"),
            "endpoint_runs": last_row.get("endpoint_runs"),
            "endpoint_runs_vs_full_segments": last_row.get("endpoint_runs_vs_full_segments"),
            "endpoint_storage_vs_full_segment_csr": last_row.get("endpoint_storage_vs_full_segment_csr"),
            "max_endpoint_runs_per_sample": last_row.get("max_endpoint_runs_per_sample"),
            "endpoint_forward_ms": last_row.get("timing_ms", {}).get("endpoint_forward")
            if isinstance(last_row.get("timing_ms"), dict)
            else None,
            "endpoint_vjp_ms": last_row.get("timing_ms", {}).get("endpoint_vjp")
            if isinstance(last_row.get("timing_ms"), dict)
            else None,
        },
        "conclusion": (
            "Endpoint-run Metal replay and VJP are correct against torch autograd for the continuous-absorption "
            "depth semantic and keep 16f storage near 0.111x full segment CSR, but endpoint run count still "
            "grows slightly worse than frame count and this is an explicit semantic change from segment-mid depth."
        ),
    }


def _endpoint_record_delta_tape_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path)}
    payload = _load_json(path)
    rows = payload.get("rows")
    last_row = rows[-1] if isinstance(rows, list) and rows and isinstance(rows[-1], dict) else {}
    record = last_row.get("endpoint_record_delta_tape", {}) if isinstance(last_row, dict) else {}
    active_record = last_row.get("active_endpoint_record_delta_tape", {}) if isinstance(last_row, dict) else {}
    storage = record.get("storage_estimates", {}) if isinstance(record, dict) else {}
    active_storage = active_record.get("storage_estimates", {}) if isinstance(active_record, dict) else {}
    acceptance = payload.get("acceptance", {})
    return {
        "available": payload.get("status") == "informational" and isinstance(acceptance, dict),
        "artifact": str(path),
        "status": payload.get("status"),
        "completion_claim": payload.get("completion_claim"),
        "star_uvt_competitive_claim": payload.get("star_uvt_competitive_claim"),
        "acceptance": acceptance,
        "frame_scale_first_to_last": payload.get("frame_scale_first_to_last"),
        "endpoint_record_count_scale_first_to_last": payload.get("endpoint_record_count_scale_first_to_last"),
        "endpoint_record_edit_op_scale_first_to_last": payload.get("endpoint_record_edit_op_scale_first_to_last"),
        "delta_replace_endpoint_record_storage_scale_first_to_last": payload.get(
            "delta_replace_endpoint_record_storage_scale_first_to_last"
        ),
        "delta_edit_op_endpoint_record_storage_scale_first_to_last": payload.get(
            "delta_edit_op_endpoint_record_storage_scale_first_to_last"
        ),
        "last_row": {
            "frames": last_row.get("frames"),
            "total_endpoint_records": record.get("total_endpoint_records") if isinstance(record, dict) else None,
            "change_events": record.get("change_events") if isinstance(record, dict) else None,
            "change_event_rate": record.get("change_event_rate") if isinstance(record, dict) else None,
            "edit_ops_total": record.get("edit_ops_total") if isinstance(record, dict) else None,
            "edit_ops_per_transition": record.get("edit_ops_per_transition") if isinstance(record, dict) else None,
            "track_unique_endpoint_record_rows_vs_samples": record.get(
                "track_unique_endpoint_record_rows_vs_samples"
            )
            if isinstance(record, dict)
            else None,
            "full_endpoint_record_csr_vs_full_segment_csr": storage.get(
                "full_endpoint_record_csr_vs_full_segment_csr"
            )
            if isinstance(storage, dict)
            else None,
            "delta_replace_endpoint_record_vs_full_segment_csr": storage.get(
                "delta_replace_endpoint_record_vs_full_segment_csr"
            )
            if isinstance(storage, dict)
            else None,
            "delta_edit_op_endpoint_record_stream_vs_full_segment_csr": storage.get(
                "delta_edit_op_endpoint_record_stream_vs_full_segment_csr"
            )
            if isinstance(storage, dict)
            else None,
            "delta_edit_op_endpoint_record_stream_vs_full_endpoint_record_csr": storage.get(
                "delta_edit_op_endpoint_record_stream_vs_full_endpoint_record_csr"
            )
            if isinstance(storage, dict)
            else None,
            "active_delta_edit_op_endpoint_record_stream_vs_full_segment_csr": active_storage.get(
                "delta_edit_op_endpoint_record_stream_vs_full_segment_csr"
            )
            if isinstance(active_storage, dict)
            else None,
        },
        "conclusion": (
            "Endpoint owner+boundary records match the continuous endpoint-run tape and expose a promising "
            "STAR-port-shaped delta signal: all-run record count is not sublinear, but edit-op and delta "
            "storage scales are strongly sublinear. The replay shader now exists as a sidecar probe, but it "
            "is not main-trainer integrated or STAR-UVT competitive yet."
        ),
    }


def _endpoint_record_delta_replay_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path)}
    payload = _load_json(path)
    rows = payload.get("rows")
    last_row = rows[-1] if isinstance(rows, list) and rows and isinstance(rows[-1], dict) else {}
    mps = last_row.get("mps", {}) if isinstance(last_row, dict) else {}
    timing = mps.get("timing_ms", {}) if isinstance(mps, dict) else {}
    acceptance = payload.get("acceptance", {})
    return {
        "available": payload.get("status") == "ok"
        and isinstance(acceptance, dict)
        and acceptance.get("metal_forward_matches_endpoint_run") is True
        and acceptance.get("metal_vjp_matches_endpoint_run") is True,
        "artifact": str(path),
        "status": payload.get("status"),
        "completion_claim": payload.get("completion_claim"),
        "star_uvt_competitive_claim": payload.get("star_uvt_competitive_claim"),
        "acceptance": acceptance,
        "frame_scale_first_to_last": payload.get("frame_scale_first_to_last"),
        "endpoint_run_scale_first_to_last": payload.get("endpoint_run_scale_first_to_last"),
        "record_delta_storage_scale_first_to_last": payload.get("record_delta_storage_scale_first_to_last"),
        "max_forward_abs_error_vs_endpoint_run": payload.get("max_forward_abs_error_vs_endpoint_run"),
        "max_vjp_rel_error_vs_endpoint_run": payload.get("max_vjp_rel_error_vs_endpoint_run"),
        "last_row": {
            "frames": last_row.get("frames"),
            "endpoint_runs": last_row.get("endpoint_runs"),
            "change_events": last_row.get("change_events"),
            "changed_records": last_row.get("changed_records"),
            "endpoint_storage_bytes": last_row.get("endpoint_storage_bytes"),
            "record_delta_storage_bytes": last_row.get("record_delta_storage_bytes"),
            "record_delta_storage_vs_endpoint_csr": last_row.get("record_delta_storage_vs_endpoint_csr"),
            "record_delta_storage_vs_full_segment_csr": last_row.get("record_delta_storage_vs_full_segment_csr"),
            "endpoint_forward_ms": timing.get("endpoint_forward") if isinstance(timing, dict) else None,
            "record_delta_forward_ms": timing.get("record_delta_forward") if isinstance(timing, dict) else None,
            "endpoint_vjp_ms": timing.get("endpoint_vjp") if isinstance(timing, dict) else None,
            "record_delta_vjp_ms": timing.get("record_delta_vjp") if isinstance(timing, dict) else None,
        },
        "conclusion": (
            "Endpoint owner+cut-id replacement-row replay is now a real Metal shader path: it recovers endpoint "
            "depths from boundary ids and rays, matches endpoint-run forward/VJP, and keeps replacement-row "
            "storage sublinear across frame count. It is still the replacement-row sidecar, not the newer "
            "edit-op stream or main-trainer integration."
        ),
    }


def _endpoint_record_edit_replay_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path)}
    payload = _load_json(path)
    rows = payload.get("rows")
    last_row = rows[-1] if isinstance(rows, list) and rows and isinstance(rows[-1], dict) else {}
    mps = last_row.get("mps", {}) if isinstance(last_row, dict) else {}
    timing = mps.get("timing_ms", {}) if isinstance(mps, dict) else {}
    acceptance = payload.get("acceptance", {})
    return {
        "available": payload.get("status") == "ok"
        and isinstance(acceptance, dict)
        and acceptance.get("metal_forward_matches_endpoint_run") is True
        and acceptance.get("metal_vjp_matches_endpoint_run") is True,
        "artifact": str(path),
        "status": payload.get("status"),
        "completion_claim": payload.get("completion_claim"),
        "star_uvt_competitive_claim": payload.get("star_uvt_competitive_claim"),
        "acceptance": acceptance,
        "frame_scale_first_to_last": payload.get("frame_scale_first_to_last"),
        "endpoint_run_scale_first_to_last": payload.get("endpoint_run_scale_first_to_last"),
        "edit_op_scale_first_to_last": payload.get("edit_op_scale_first_to_last"),
        "edit_storage_scale_first_to_last": payload.get("edit_storage_scale_first_to_last"),
        "max_forward_abs_error_vs_endpoint_run": payload.get("max_forward_abs_error_vs_endpoint_run"),
        "max_vjp_rel_error_vs_endpoint_run": payload.get("max_vjp_rel_error_vs_endpoint_run"),
        "max_block4_forward_abs_error_vs_endpoint_run": payload.get(
            "max_block4_forward_abs_error_vs_endpoint_run"
        ),
        "max_trackloop_forward_abs_error_vs_endpoint_run": payload.get(
            "max_trackloop_forward_abs_error_vs_endpoint_run"
        ),
        "max_framegroup16_forward_abs_error_vs_endpoint_run": payload.get(
            "max_framegroup16_forward_abs_error_vs_endpoint_run"
        ),
        "max_rgb_only_vjp_rel_error_vs_full_zero_alpha_depth": payload.get(
            "max_rgb_only_vjp_rel_error_vs_full_zero_alpha_depth"
        ),
        "max_block4_rgb_only_vjp_rel_error_vs_full_zero_alpha_depth": payload.get(
            "max_block4_rgb_only_vjp_rel_error_vs_full_zero_alpha_depth"
        ),
        "max_block4_rgb_only_vjp_rel_error_vs_edit_rgb_only": payload.get(
            "max_block4_rgb_only_vjp_rel_error_vs_edit_rgb_only"
        ),
        "last_row": {
            "frames": last_row.get("frames"),
            "endpoint_runs": last_row.get("endpoint_runs"),
            "change_events": last_row.get("change_events"),
            "changed_records": last_row.get("changed_records"),
            "edit_ops": last_row.get("edit_ops"),
            "block4_edit_ops": last_row.get("block4_edit_ops"),
            "endpoint_storage_bytes": last_row.get("endpoint_storage_bytes"),
            "edit_storage_bytes": last_row.get("edit_storage_bytes"),
            "block4_storage_bytes": last_row.get("block4_storage_bytes"),
            "edit_storage_vs_endpoint_csr": last_row.get("edit_storage_vs_endpoint_csr"),
            "edit_storage_vs_full_segment_csr": last_row.get("edit_storage_vs_full_segment_csr"),
            "block4_storage_vs_endpoint_csr": last_row.get("block4_storage_vs_endpoint_csr"),
            "block4_storage_vs_full_segment_csr": last_row.get("block4_storage_vs_full_segment_csr"),
            "endpoint_forward_ms": timing.get("endpoint_forward") if isinstance(timing, dict) else None,
            "edit_forward_ms": timing.get("edit_forward") if isinstance(timing, dict) else None,
            "edit_block4_forward_ms": timing.get("edit_block4_forward") if isinstance(timing, dict) else None,
            "endpoint_vjp_ms": timing.get("endpoint_vjp") if isinstance(timing, dict) else None,
            "edit_vjp_ms": timing.get("edit_vjp") if isinstance(timing, dict) else None,
            "edit_rgb_full_vjp_ms": timing.get("edit_rgb_full_vjp") if isinstance(timing, dict) else None,
            "edit_rgb_only_vjp_ms": timing.get("edit_rgb_only_vjp") if isinstance(timing, dict) else None,
            "edit_block4_rgb_only_vjp_ms": (
                timing.get("edit_block4_rgb_only_vjp") if isinstance(timing, dict) else None
            ),
            "edit_trackloop_forward_ms": timing.get("edit_trackloop_forward") if isinstance(timing, dict) else None,
            "edit_framegroup16_forward_ms": timing.get("edit_framegroup16_forward") if isinstance(timing, dict) else None,
        },
        "conclusion": (
            "Endpoint owner+cut-id edit-op stream replay is now a real Metal shader path: it reconstructs rows "
            "from base records plus insert/delete/replace ops, recovers endpoint depths from boundary ids and "
            "moving rays, and matches endpoint-run forward/VJP. Storage is sublinear and compact by 16 frames, "
            "but the current shader is slower than endpoint-run replay and is not main-trainer integrated."
        ),
    }


def _endpoint_record_edit_rgb_only_replay_summary(path: Path) -> dict[str, Any]:
    summary = _endpoint_record_edit_replay_summary(path)
    acceptance = summary.get("acceptance")
    last_row = summary.get("last_row")
    rgb_only_rel = summary.get("max_rgb_only_vjp_rel_error_vs_full_zero_alpha_depth")
    summary["available"] = (
        summary.get("available") is True
        and isinstance(acceptance, dict)
        and acceptance.get("metal_rgb_only_vjp_matches_full_zero_alpha_depth") is True
        and _finite_number(rgb_only_rel)
    )
    summary["conclusion"] = (
        "Endpoint-record edit RGB-only VJP is numerically correct against the full edit VJP with zero "
        "alpha/depth adjoints and keeps the same sublinear edit storage. The isolated 16f RGB-only kernel "
        "timing is not a stable speed win over the full edit VJP or endpoint-run replay, so this remains a "
        "correctness/scope sidecar rather than a STAR-UVT competitive result."
    )
    if isinstance(last_row, dict):
        summary["rgb_only_timing_read"] = {
            "endpoint_vjp_ms_16f": last_row.get("endpoint_vjp_ms"),
            "edit_full_vjp_ms_16f": last_row.get("edit_vjp_ms"),
            "edit_rgb_full_vjp_ms_16f": last_row.get("edit_rgb_full_vjp_ms"),
            "edit_rgb_only_vjp_ms_16f": last_row.get("edit_rgb_only_vjp_ms"),
        }
    return summary


def _endpoint_record_edit_trackloop_replay_summary(path: Path) -> dict[str, Any]:
    summary = _endpoint_record_edit_replay_summary(path)
    acceptance = summary.get("acceptance")
    last_row = summary.get("last_row")
    trackloop_error = summary.get("max_trackloop_forward_abs_error_vs_endpoint_run")
    summary["available"] = (
        summary.get("available") is True
        and isinstance(acceptance, dict)
        and acceptance.get("metal_trackloop_forward_matches_endpoint_run") is True
        and _finite_number(trackloop_error)
    )
    summary["conclusion"] = (
        "Endpoint-record edit track-loop forward replay is numerically correct against endpoint-run and keeps "
        "the same sublinear edit storage, but it is not a speed win: the 16f track-loop forward remains slower "
        "than endpoint-run and does not resolve the row-replay bottleneck. Treat this as a rejected forward "
        "optimization sidecar, not a STAR-UVT competitive result."
    )
    if isinstance(last_row, dict):
        summary["trackloop_timing_read"] = {
            "endpoint_forward_ms_16f": last_row.get("endpoint_forward_ms"),
            "edit_forward_ms_16f": last_row.get("edit_forward_ms"),
            "edit_trackloop_forward_ms_16f": last_row.get("edit_trackloop_forward_ms"),
        }
    return summary


def _endpoint_record_edit_block4_replay_summary(path: Path) -> dict[str, Any]:
    summary = _endpoint_record_edit_replay_summary(path)
    acceptance = summary.get("acceptance")
    last_row = summary.get("last_row")
    block4_error = summary.get("max_block4_forward_abs_error_vs_endpoint_run")
    summary["available"] = (
        summary.get("available") is True
        and isinstance(acceptance, dict)
        and acceptance.get("metal_block4_forward_matches_endpoint_run") is True
        and acceptance.get("metal_block4_rgb_only_vjp_matches_edit_rgb_only") is True
        and acceptance.get("metal_block4_rgb_only_vjp_matches_full_zero_alpha_depth") is True
        and _finite_number(block4_error)
    )
    summary["conclusion"] = (
        "Endpoint-record edit block4 forward replay is numerically correct and is the first isolated forward "
        "variant in this lane that beats the original edit replay at 16f while keeping compact storage; in the "
        "refreshed raw probe it is near but slightly slower than endpoint-run forward. It anchors rows every four "
        "frames and replays only in-block edits, so it preserves sample-level parallelism while bounding history "
        "replay. The sidecar now also has a dedicated block4 RGB-only VJP that matches the old RGB-only and full "
        "zero-alpha/depth VJP checks, but the fixed-geometry train/eval rerun is still noisy and slower than the "
        "earlier borrowed-VJP path. This is not a main-trainer integration or STAR-UVT competitive claim."
    )
    if isinstance(last_row, dict):
        summary["block4_timing_read"] = {
            "endpoint_forward_ms_16f": last_row.get("endpoint_forward_ms"),
            "edit_forward_ms_16f": last_row.get("edit_forward_ms"),
            "edit_block4_forward_ms_16f": last_row.get("edit_block4_forward_ms"),
            "endpoint_vjp_ms_16f": last_row.get("endpoint_vjp_ms"),
            "edit_rgb_only_vjp_ms_16f": last_row.get("edit_rgb_only_vjp_ms"),
            "edit_block4_rgb_only_vjp_ms_16f": last_row.get("edit_block4_rgb_only_vjp_ms"),
            "edit_trackloop_forward_ms_16f": last_row.get("edit_trackloop_forward_ms"),
            "edit_framegroup16_forward_ms_16f": last_row.get("edit_framegroup16_forward_ms"),
        }
        summary["block4_storage_read"] = {
            "edit_storage_vs_full_segment_csr_16f": last_row.get("edit_storage_vs_full_segment_csr"),
            "block4_storage_vs_full_segment_csr_16f": last_row.get("block4_storage_vs_full_segment_csr"),
            "edit_storage_vs_endpoint_csr_16f": last_row.get("edit_storage_vs_endpoint_csr"),
            "block4_storage_vs_endpoint_csr_16f": last_row.get("block4_storage_vs_endpoint_csr"),
        }
    return summary


def _endpoint_record_edit_block_coeff_replay_summary(path: Path, sweep_path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path), "sweep_artifact": str(sweep_path)}
    payload = _load_json(path)
    rows = payload.get("rows")
    last_row = rows[-1] if isinstance(rows, list) and rows and isinstance(rows[-1], dict) else {}
    mps = last_row.get("mps", {}) if isinstance(last_row, dict) else {}
    timing = mps.get("timing_ms", {}) if isinstance(mps, dict) else {}
    acceptance = payload.get("acceptance", {})
    sweep_summary: dict[str, Any] = {"available": False, "artifact": str(sweep_path)}
    if sweep_path.exists():
        sweep = _load_json(sweep_path)
        sweep_rows = sweep.get("rows")
        compact_rows = []
        if isinstance(sweep_rows, list):
            for row in sweep_rows:
                if not isinstance(row, dict):
                    continue
                row_timing = row.get("mps", {}).get("timing_ms", {}) if isinstance(row.get("mps"), dict) else {}
                compact_rows.append(
                    {
                        "frames": row.get("frames"),
                        "endpoint_forward_ms": row_timing.get("endpoint_forward")
                        if isinstance(row_timing, dict)
                        else None,
                        "edit_block4_forward_ms": row_timing.get("edit_block4_forward")
                        if isinstance(row_timing, dict)
                        else None,
                        "edit_block_coeff_forward_ms": row_timing.get("edit_block_coeff_forward")
                        if isinstance(row_timing, dict)
                        else None,
                        "edit_forward_ms": row_timing.get("edit_forward") if isinstance(row_timing, dict) else None,
                        "block_coeff_storage_vs_endpoint_csr": row.get("block_coeff_storage_vs_endpoint_csr"),
                        "block_coeff_storage_vs_full_segment_csr": row.get(
                            "block_coeff_storage_vs_full_segment_csr"
                        ),
                    }
                )
        sweep_acceptance = sweep.get("acceptance", {})
        sweep_summary = {
            "available": sweep.get("status") == "ok"
            and isinstance(sweep_acceptance, dict)
            and sweep_acceptance.get("metal_block_coeff_forward_matches_endpoint_run") is True,
            "artifact": str(sweep_path),
            "status": sweep.get("status"),
            "frame_counts": sweep.get("frame_counts"),
            "acceptance": sweep_acceptance,
            "frame_scale_first_to_last": sweep.get("frame_scale_first_to_last"),
            "edit_op_scale_first_to_last": sweep.get("edit_op_scale_first_to_last"),
            "edit_storage_scale_first_to_last": sweep.get("edit_storage_scale_first_to_last"),
            "block_edit_storage_scale_first_to_last": sweep.get("block_edit_storage_scale_first_to_last"),
            "endpoint_run_scale_first_to_last": sweep.get("endpoint_run_scale_first_to_last"),
            "max_block_coeff_forward_abs_error_vs_endpoint_run": sweep.get(
                "max_block_coeff_forward_abs_error_vs_endpoint_run"
            ),
            "max_block_coeff_rgb_only_vjp_rel_error_vs_full_zero_alpha_depth": sweep.get(
                "max_block_coeff_rgb_only_vjp_rel_error_vs_full_zero_alpha_depth"
            ),
            "max_block_coeff_rgb_only_vjp_rel_error_vs_edit_rgb_only": sweep.get(
                "max_block_coeff_rgb_only_vjp_rel_error_vs_edit_rgb_only"
            ),
            "max_block_coeff_rgb_only_vjp_rel_error_vs_block4_rgb_only": sweep.get(
                "max_block_coeff_rgb_only_vjp_rel_error_vs_block4_rgb_only"
            ),
            "rows": compact_rows,
        }
    endpoint_ms = timing.get("endpoint_forward") if isinstance(timing, dict) else None
    block4_ms = timing.get("edit_block4_forward") if isinstance(timing, dict) else None
    coeff_ms = timing.get("edit_block_coeff_forward") if isinstance(timing, dict) else None
    edit_ms = timing.get("edit_forward") if isinstance(timing, dict) else None
    speed_read = "not_faster_or_not_measured"
    if _finite_number(endpoint_ms) and _finite_number(edit_ms) and _finite_number(coeff_ms):
        speed_read = (
            "faster_than_endpoint_and_edit_forward"
            if float(coeff_ms) < float(endpoint_ms) and float(coeff_ms) < float(edit_ms)
            else "not_faster_or_not_measured"
        )
        if speed_read == "faster_than_endpoint_and_edit_forward" and _finite_number(block4_ms):
            if float(coeff_ms) < float(block4_ms):
                speed_read = "faster_than_endpoint_edit_and_block4_forward"
    return {
        "available": isinstance(acceptance, dict)
        and acceptance.get("metal_block_coeff_forward_matches_endpoint_run") is True
        and _finite_number(payload.get("max_block_coeff_forward_abs_error_vs_endpoint_run")),
        "artifact": str(path),
        "sweep_artifact": str(sweep_path),
        "status": payload.get("status"),
        "completion_claim": payload.get("completion_claim"),
        "star_uvt_competitive_claim": payload.get("star_uvt_competitive_claim"),
        "acceptance": acceptance,
        "structural_read": payload.get("structural_read"),
        "frame_counts": payload.get("frame_counts"),
        "max_block_coeff_forward_abs_error_vs_endpoint_run": payload.get(
            "max_block_coeff_forward_abs_error_vs_endpoint_run"
        ),
        "max_block_coeff_rgb_only_vjp_rel_error_vs_full_zero_alpha_depth": payload.get(
            "max_block_coeff_rgb_only_vjp_rel_error_vs_full_zero_alpha_depth"
        ),
        "max_block_coeff_rgb_only_vjp_rel_error_vs_edit_rgb_only": payload.get(
            "max_block_coeff_rgb_only_vjp_rel_error_vs_edit_rgb_only"
        ),
        "max_block_coeff_rgb_only_vjp_rel_error_vs_block4_rgb_only": payload.get(
            "max_block_coeff_rgb_only_vjp_rel_error_vs_block4_rgb_only"
        ),
        "sweep": sweep_summary,
        "last_row": {
            "frames": last_row.get("frames"),
            "endpoint_runs": last_row.get("endpoint_runs"),
            "edit_ops": last_row.get("edit_ops"),
            "block4_edit_ops": last_row.get("block4_edit_ops"),
            "endpoint_storage_bytes": last_row.get("endpoint_storage_bytes"),
            "block4_storage_bytes": last_row.get("block4_storage_bytes"),
            "block_coeff_storage_bytes": last_row.get("block_coeff_storage_bytes"),
            "block_coeff_storage_vs_endpoint_csr": last_row.get("block_coeff_storage_vs_endpoint_csr"),
            "block_coeff_storage_vs_full_segment_csr": last_row.get("block_coeff_storage_vs_full_segment_csr"),
            "endpoint_forward_ms": endpoint_ms,
            "edit_block4_forward_ms": block4_ms,
            "edit_block_coeff_forward_ms": coeff_ms,
            "edit_forward_ms": edit_ms,
            "edit_trackloop_forward_ms": timing.get("edit_trackloop_forward") if isinstance(timing, dict) else None,
            "edit_framegroup16_forward_ms": timing.get("edit_framegroup16_forward")
            if isinstance(timing, dict)
            else None,
            "edit_block_coeff_rgb_only_vjp_ms": timing.get("edit_block_coeff_rgb_only_vjp")
            if isinstance(timing, dict)
            else None,
        },
        "speed_read": speed_read,
        "conclusion": (
            "Coefficient-cached block edit forward replay is numerically correct and speed-positive in the "
            "16f render16 sidecar: it precomputes ray/time boundary depth coefficients and avoids recomputing "
            "cut depths during replay, beating endpoint-run and original edit forward in the saved 16f probe, "
            "though it does not consistently beat block4. The RGB-only coeff VJP is also numerically correct "
            "against the edit/block4 VJP checks. The train path now has a warmed 2/4/8/16 sidecar smoke, but "
            "there is no main-trainer integration and the coefficient table is still above endpoint CSR storage, "
            "so this is not a STAR-UVT competitive claim."
        ),
    }


def _endpoint_record_edit_block_coeff_train_eval_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path)}
    payload = _load_json(path)
    rows = payload.get("rows")
    first_row = rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], dict) else {}
    last_row = rows[-1] if isinstance(rows, list) and rows and isinstance(rows[-1], dict) else {}

    def row_summary(row: dict[str, Any]) -> dict[str, Any]:
        step_summary = row.get("step_summary", {}) if isinstance(row, dict) else {}
        total_summary = step_summary.get("total", {}) if isinstance(step_summary, dict) else {}
        render_summary = step_summary.get("render", {}) if isinstance(step_summary, dict) else {}
        backward_summary = step_summary.get("backward", {}) if isinstance(step_summary, dict) else {}
        return {
            "frames": row.get("frame_count"),
            "steps": row.get("steps"),
            "warmup_steps": row.get("warmup_steps"),
            "total_ms": float(total_summary.get("mean_s", 0.0)) * 1000.0,
            "render_ms": float(render_summary.get("mean_s", 0.0)) * 1000.0,
            "backward_ms": float(backward_summary.get("mean_s", 0.0)) * 1000.0,
            "final_train_psnr": row.get("final_train_psnr"),
            "final_heldout_psnr": row.get("final_heldout_psnr"),
            "first_grad_abs_sum": row.get("first_grad_abs_sum"),
            "parameter_update_abs_max": row.get("parameter_update_abs_max"),
            "train_selected_tape_storage_vs_full": row.get("train_selected_tape_storage_vs_full"),
            "train_endpoint_record_block4_storage_vs_full": row.get(
                "train_endpoint_record_block4_storage_vs_full"
            ),
            "train_endpoint_record_block4_storage_vs_endpoint_run": row.get(
                "train_endpoint_record_block4_storage_vs_endpoint_run"
            ),
        }

    return {
        "available": payload.get("status") == "ok",
        "artifact": str(path),
        "status": payload.get("status"),
        "tape_mode": payload.get("tape_mode"),
        "acceptance": payload.get("acceptance"),
        "frame_counts": payload.get("frame_counts"),
        "frame_scale_first_to_last": payload.get("frame_scale_first_to_last"),
        "total_step_scale_first_to_last": payload.get("total_step_scale_first_to_last"),
        "render_scale_first_to_last": payload.get("render_scale_first_to_last"),
        "backward_scale_first_to_last": payload.get("backward_scale_first_to_last"),
        "selected_tape_segment_scale_first_to_last": payload.get("selected_tape_segment_scale_first_to_last"),
        "selected_tape_storage_scale_first_to_last": payload.get("selected_tape_storage_scale_first_to_last"),
        "endpoint_record_edit_op_scale_first_to_last": payload.get("endpoint_record_edit_op_scale_first_to_last"),
        "full_trainer_claim": payload.get("full_trainer_claim"),
        "full_geometry_gradient_claim": payload.get("full_geometry_gradient_claim"),
        "density_independent_depth_claim": payload.get("density_independent_depth_claim"),
        "continuous_absorption_depth_semantic": payload.get("continuous_absorption_depth_semantic"),
        "optimizer_mode": payload.get("optimizer_mode"),
        "segment_tape_vjp_mode": payload.get("segment_tape_vjp_mode"),
        "first_row": row_summary(first_row),
        "last_row": row_summary(last_row),
        "conclusion": (
            "Coefficient-cached block edit RGB train/eval now has a green 20-step render32 2/4/8/16 "
            "autograd sweep using the coeff-cached forward and RGB-only VJP. It proves gradients flow "
            "through the coeff path across frame counts, with sublinear saved total/render/backward scaling. "
            "The fixed coefficient table is storage-heavy at tiny frame counts but falls below full CSR at 16f. "
            "The timings are still MPS-noisy and not a full stable speed benchmark, main-trainer integration, "
            "or STAR-UVT competitive result."
        ),
    }


def _owner_run_train_eval_summary(path: Path, best: dict[str, Any] | None) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path)}
    payload = _load_json(path)
    rows = payload.get("rows")
    last_row = rows[-1] if isinstance(rows, list) and rows and isinstance(rows[-1], dict) else {}
    step_summary = last_row.get("step_summary", {}) if isinstance(last_row, dict) else {}
    total_summary = step_summary.get("total", {}) if isinstance(step_summary, dict) else {}
    render_summary = step_summary.get("render", {}) if isinstance(step_summary, dict) else {}
    backward_summary = step_summary.get("backward", {}) if isinstance(step_summary, dict) else {}
    total_16f_ms = float(total_summary.get("mean_s", 0.0)) * 1000.0
    comparison: dict[str, Any] = {}
    if best is not None and _finite_number(best.get("total_16f_ms")) and total_16f_ms > 0.0:
        comparison = {
            "fused_winner_mode": best.get("mode"),
            "fused_winner_16f_total_ms": best.get("total_16f_ms"),
            "owner_run_16f_total_ms": total_16f_ms,
            "owner_run_to_fused_winner_16f_total_ratio": total_16f_ms / float(best["total_16f_ms"]),
            "scope_note": (
                "Matched fused train/eval parameters and RGB/site-RGBA objective, using the segment-tape "
                "autograd wrapper in an isolated owner-run script rather than integration into the main "
                "fused-slab trainer."
            ),
        }
    return {
        "available": payload.get("status") == "ok",
        "artifact": str(path),
        "status": payload.get("status"),
        "acceptance": payload.get("acceptance"),
        "frame_scale_first_to_last": payload.get("frame_scale_first_to_last"),
        "total_step_scale_first_to_last": payload.get("total_step_scale_first_to_last"),
        "render_scale_first_to_last": payload.get("render_scale_first_to_last"),
        "backward_scale_first_to_last": payload.get("backward_scale_first_to_last"),
        "owner_run_segment_scale_first_to_last": payload.get("owner_run_segment_scale_first_to_last"),
        "full_trainer_claim": payload.get("full_trainer_claim"),
        "full_geometry_gradient_claim": payload.get("full_geometry_gradient_claim"),
        "density_independent_depth_claim": payload.get("density_independent_depth_claim"),
        "optimizer_mode": payload.get("optimizer_mode"),
        "segment_tape_vjp_mode": payload.get("segment_tape_vjp_mode"),
        "last_row": {
            "frames": last_row.get("frame_count"),
            "total_ms": total_16f_ms,
            "render_ms": float(render_summary.get("mean_s", 0.0)) * 1000.0,
            "backward_ms": float(backward_summary.get("mean_s", 0.0)) * 1000.0,
            "final_train_psnr": last_row.get("final_train_psnr"),
            "final_heldout_psnr": last_row.get("final_heldout_psnr"),
            "train_full_segments": last_row.get("train_full_segments"),
            "train_owner_run_segments": last_row.get("train_owner_run_segments"),
            "train_owner_run_segments_vs_full": last_row.get("train_owner_run_segments_vs_full"),
            "train_owner_run_storage_vs_full": last_row.get("train_owner_run_storage_vs_full"),
            "max_owner_run_segments_per_sample": last_row.get("max_owner_run_segments_per_sample"),
        },
        "comparison_to_fused_winner": comparison,
        "conclusion": (
            "Matched-parameter owner-run RGB train/eval is green through the segment-tape autograd wrapper "
            "and faster than the current fused-slab winner on the saved 16f smoke-scale run, while preserving "
            "the no-full-trainer, no-geometry-gradient, and no-density-independent-depth scope boundaries."
        ),
    }


def _active_internal_train_eval_summary(
    path: Path,
    best: dict[str, Any] | None,
    owner_run_train_eval: dict[str, Any],
) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path)}
    payload = _load_json(path)
    rows = payload.get("rows")
    last_row = rows[-1] if isinstance(rows, list) and rows and isinstance(rows[-1], dict) else {}
    step_summary = last_row.get("step_summary", {}) if isinstance(last_row, dict) else {}
    total_summary = step_summary.get("total", {}) if isinstance(step_summary, dict) else {}
    render_summary = step_summary.get("render", {}) if isinstance(step_summary, dict) else {}
    backward_summary = step_summary.get("backward", {}) if isinstance(step_summary, dict) else {}
    total_16f_ms = float(total_summary.get("mean_s", 0.0)) * 1000.0
    comparison: dict[str, Any] = {}
    if best is not None and _finite_number(best.get("total_16f_ms")) and total_16f_ms > 0.0:
        comparison["active_internal_to_fused_winner_16f_total_ratio"] = total_16f_ms / float(best["total_16f_ms"])
        comparison["fused_winner_16f_total_ms"] = best.get("total_16f_ms")
    owner_last = owner_run_train_eval.get("last_row") if isinstance(owner_run_train_eval, dict) else None
    if isinstance(owner_last, dict) and _finite_number(owner_last.get("total_ms")) and total_16f_ms > 0.0:
        comparison["active_internal_to_owner_run_16f_total_ratio"] = total_16f_ms / float(owner_last["total_ms"])
        comparison["owner_run_16f_total_ms"] = owner_last.get("total_ms")
    comparison["scope_note"] = (
        "Active-internal train/eval reuses the segment-tape autograd wrapper on threshold-active internal "
        "segments. It measures exact current-density depth/RGB tape cost, not density-independent replay or "
        "main-trainer integration."
    )
    return {
        "available": payload.get("status") == "ok",
        "artifact": str(path),
        "status": payload.get("status"),
        "tape_mode": payload.get("tape_mode"),
        "acceptance": payload.get("acceptance"),
        "frame_scale_first_to_last": payload.get("frame_scale_first_to_last"),
        "total_step_scale_first_to_last": payload.get("total_step_scale_first_to_last"),
        "render_scale_first_to_last": payload.get("render_scale_first_to_last"),
        "backward_scale_first_to_last": payload.get("backward_scale_first_to_last"),
        "selected_tape_segment_scale_first_to_last": payload.get("selected_tape_segment_scale_first_to_last"),
        "full_trainer_claim": payload.get("full_trainer_claim"),
        "full_geometry_gradient_claim": payload.get("full_geometry_gradient_claim"),
        "density_independent_depth_claim": payload.get("density_independent_depth_claim"),
        "optimizer_mode": payload.get("optimizer_mode"),
        "segment_tape_vjp_mode": payload.get("segment_tape_vjp_mode"),
        "last_row": {
            "frames": last_row.get("frame_count"),
            "total_ms": total_16f_ms,
            "render_ms": float(render_summary.get("mean_s", 0.0)) * 1000.0,
            "backward_ms": float(backward_summary.get("mean_s", 0.0)) * 1000.0,
            "final_train_psnr": last_row.get("final_train_psnr"),
            "final_heldout_psnr": last_row.get("final_heldout_psnr"),
            "train_full_segments": last_row.get("train_full_segments"),
            "train_selected_tape_segments": last_row.get("train_selected_tape_segments"),
            "train_selected_tape_segments_vs_full": last_row.get("train_selected_tape_segments_vs_full"),
            "train_selected_tape_storage_vs_full": last_row.get("train_selected_tape_storage_vs_full"),
            "max_selected_tape_segments_per_sample": last_row.get("max_selected_tape_segments_per_sample"),
        },
        "comparison": comparison,
        "conclusion": (
            "Active-internal train/eval is the measured exact-current-depth tradeoff: it stays faster than the "
            "current fused winner at 16f and matches PSNR, but is slower than owner-run and its selected segment "
            "count still scales worse than frame count."
        ),
    }


def _full_tape_train_eval_summary(
    path: Path,
    best: dict[str, Any] | None,
    owner_run_train_eval: dict[str, Any],
    active_internal_train_eval: dict[str, Any],
) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path)}
    payload = _load_json(path)
    rows = payload.get("rows")
    last_row = rows[-1] if isinstance(rows, list) and rows and isinstance(rows[-1], dict) else {}
    step_summary = last_row.get("step_summary", {}) if isinstance(last_row, dict) else {}
    total_summary = step_summary.get("total", {}) if isinstance(step_summary, dict) else {}
    render_summary = step_summary.get("render", {}) if isinstance(step_summary, dict) else {}
    backward_summary = step_summary.get("backward", {}) if isinstance(step_summary, dict) else {}
    total_16f_ms = float(total_summary.get("mean_s", 0.0)) * 1000.0
    comparison: dict[str, Any] = {}
    if best is not None and _finite_number(best.get("total_16f_ms")) and total_16f_ms > 0.0:
        comparison["full_tape_to_fused_winner_16f_total_ratio"] = total_16f_ms / float(best["total_16f_ms"])
        comparison["fused_winner_16f_total_ms"] = best.get("total_16f_ms")
    owner_last = owner_run_train_eval.get("last_row") if isinstance(owner_run_train_eval, dict) else None
    if isinstance(owner_last, dict) and _finite_number(owner_last.get("total_ms")) and total_16f_ms > 0.0:
        comparison["full_tape_to_owner_run_16f_total_ratio"] = total_16f_ms / float(owner_last["total_ms"])
        comparison["owner_run_16f_total_ms"] = owner_last.get("total_ms")
    active_last = active_internal_train_eval.get("last_row") if isinstance(active_internal_train_eval, dict) else None
    if isinstance(active_last, dict) and _finite_number(active_last.get("total_ms")) and total_16f_ms > 0.0:
        comparison["full_tape_to_active_internal_16f_total_ratio"] = total_16f_ms / float(active_last["total_ms"])
        comparison["active_internal_16f_total_ms"] = active_last.get("total_ms")
    comparison["scope_note"] = (
        "Full-tape train/eval reuses the segment-tape autograd wrapper on every stored segment. It is the "
        "exact fixed-geometry density-independent replay cost baseline, not a compact STAR-like structure or "
        "main-trainer integration."
    )
    return {
        "available": payload.get("status") == "ok",
        "artifact": str(path),
        "status": payload.get("status"),
        "tape_mode": payload.get("tape_mode"),
        "acceptance": payload.get("acceptance"),
        "frame_scale_first_to_last": payload.get("frame_scale_first_to_last"),
        "total_step_scale_first_to_last": payload.get("total_step_scale_first_to_last"),
        "render_scale_first_to_last": payload.get("render_scale_first_to_last"),
        "backward_scale_first_to_last": payload.get("backward_scale_first_to_last"),
        "selected_tape_segment_scale_first_to_last": payload.get("selected_tape_segment_scale_first_to_last"),
        "full_trainer_claim": payload.get("full_trainer_claim"),
        "full_geometry_gradient_claim": payload.get("full_geometry_gradient_claim"),
        "density_independent_depth_claim": payload.get("density_independent_depth_claim"),
        "optimizer_mode": payload.get("optimizer_mode"),
        "segment_tape_vjp_mode": payload.get("segment_tape_vjp_mode"),
        "last_row": {
            "frames": last_row.get("frame_count"),
            "total_ms": total_16f_ms,
            "render_ms": float(render_summary.get("mean_s", 0.0)) * 1000.0,
            "backward_ms": float(backward_summary.get("mean_s", 0.0)) * 1000.0,
            "final_train_psnr": last_row.get("final_train_psnr"),
            "final_heldout_psnr": last_row.get("final_heldout_psnr"),
            "train_full_segments": last_row.get("train_full_segments"),
            "train_selected_tape_segments": last_row.get("train_selected_tape_segments"),
            "train_selected_tape_segments_vs_full": last_row.get("train_selected_tape_segments_vs_full"),
            "train_selected_tape_storage_vs_full": last_row.get("train_selected_tape_storage_vs_full"),
            "max_selected_tape_segments_per_sample": last_row.get("max_selected_tape_segments_per_sample"),
        },
        "comparison": comparison,
        "conclusion": (
            "Full-tape train/eval measures exact density-independent fixed-geometry replay cost. It matches PSNR "
            "but is slower than owner-run and active-internal, and its selected segment/storage ratio is 1.0 by "
            "definition."
        ),
    }


def _endpoint_run_train_eval_summary(
    path: Path,
    best: dict[str, Any] | None,
    owner_run_train_eval: dict[str, Any],
    active_internal_train_eval: dict[str, Any],
) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path)}
    payload = _load_json(path)
    rows = payload.get("rows")
    last_row = rows[-1] if isinstance(rows, list) and rows and isinstance(rows[-1], dict) else {}
    step_summary = last_row.get("step_summary", {}) if isinstance(last_row, dict) else {}
    total_summary = step_summary.get("total", {}) if isinstance(step_summary, dict) else {}
    render_summary = step_summary.get("render", {}) if isinstance(step_summary, dict) else {}
    backward_summary = step_summary.get("backward", {}) if isinstance(step_summary, dict) else {}
    total_16f_ms = float(total_summary.get("mean_s", 0.0)) * 1000.0
    comparison: dict[str, Any] = {}
    if best is not None and _finite_number(best.get("total_16f_ms")) and total_16f_ms > 0.0:
        comparison["endpoint_run_to_fused_winner_16f_total_ratio"] = total_16f_ms / float(best["total_16f_ms"])
        comparison["fused_winner_16f_total_ms"] = best.get("total_16f_ms")
    owner_last = owner_run_train_eval.get("last_row") if isinstance(owner_run_train_eval, dict) else None
    if isinstance(owner_last, dict) and _finite_number(owner_last.get("total_ms")) and total_16f_ms > 0.0:
        comparison["endpoint_run_to_owner_run_16f_total_ratio"] = total_16f_ms / float(owner_last["total_ms"])
        comparison["owner_run_16f_total_ms"] = owner_last.get("total_ms")
    active_last = active_internal_train_eval.get("last_row") if isinstance(active_internal_train_eval, dict) else None
    if isinstance(active_last, dict) and _finite_number(active_last.get("total_ms")) and total_16f_ms > 0.0:
        comparison["endpoint_run_to_active_internal_16f_total_ratio"] = total_16f_ms / float(active_last["total_ms"])
        comparison["active_internal_16f_total_ms"] = active_last.get("total_ms")
    comparison["scope_note"] = (
        "Endpoint-run train/eval uses the new continuous-absorption endpoint shader and fixed-geometry "
        "site-RGBA autograd path. It is compact and density-independent under that semantic, not a drop-in "
        "replacement for current segment-mid depth."
    )
    return {
        "available": payload.get("status") == "ok",
        "artifact": str(path),
        "status": payload.get("status"),
        "tape_mode": payload.get("tape_mode"),
        "acceptance": payload.get("acceptance"),
        "frame_scale_first_to_last": payload.get("frame_scale_first_to_last"),
        "total_step_scale_first_to_last": payload.get("total_step_scale_first_to_last"),
        "render_scale_first_to_last": payload.get("render_scale_first_to_last"),
        "backward_scale_first_to_last": payload.get("backward_scale_first_to_last"),
        "selected_tape_segment_scale_first_to_last": payload.get("selected_tape_segment_scale_first_to_last"),
        "full_trainer_claim": payload.get("full_trainer_claim"),
        "full_geometry_gradient_claim": payload.get("full_geometry_gradient_claim"),
        "density_independent_depth_claim": payload.get("density_independent_depth_claim"),
        "continuous_absorption_depth_semantic": payload.get("continuous_absorption_depth_semantic"),
        "optimizer_mode": payload.get("optimizer_mode"),
        "segment_tape_vjp_mode": payload.get("segment_tape_vjp_mode"),
        "last_row": {
            "frames": last_row.get("frame_count"),
            "total_ms": total_16f_ms,
            "render_ms": float(render_summary.get("mean_s", 0.0)) * 1000.0,
            "backward_ms": float(backward_summary.get("mean_s", 0.0)) * 1000.0,
            "final_train_psnr": last_row.get("final_train_psnr"),
            "final_heldout_psnr": last_row.get("final_heldout_psnr"),
            "train_full_segments": last_row.get("train_full_segments"),
            "train_selected_tape_segments": last_row.get("train_selected_tape_segments"),
            "train_selected_tape_segments_vs_full": last_row.get("train_selected_tape_segments_vs_full"),
            "train_selected_tape_storage_vs_full": last_row.get("train_selected_tape_storage_vs_full"),
            "max_selected_tape_segments_per_sample": last_row.get("max_selected_tape_segments_per_sample"),
        },
        "comparison": comparison,
        "conclusion": (
            "Endpoint-run train/eval is the compact density-independent semantic-change path: 16f RGB training "
            "is faster than the current fused winner and active-internal path, slightly slower than the "
            "current-density owner-run shortcut, and run count is still not STAR-like sublinear."
        ),
    }


def _endpoint_record_edit_train_eval_summary(
    path: Path,
    best: dict[str, Any] | None,
    owner_run_train_eval: dict[str, Any],
    endpoint_run_train_eval: dict[str, Any],
) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path)}
    payload = _load_json(path)
    rows = payload.get("rows")
    last_row = rows[-1] if isinstance(rows, list) and rows and isinstance(rows[-1], dict) else {}
    step_summary = last_row.get("step_summary", {}) if isinstance(last_row, dict) else {}
    total_summary = step_summary.get("total", {}) if isinstance(step_summary, dict) else {}
    render_summary = step_summary.get("render", {}) if isinstance(step_summary, dict) else {}
    backward_summary = step_summary.get("backward", {}) if isinstance(step_summary, dict) else {}
    total_16f_ms = float(total_summary.get("mean_s", 0.0)) * 1000.0
    comparison: dict[str, Any] = {}
    if best is not None and _finite_number(best.get("total_16f_ms")) and total_16f_ms > 0.0:
        comparison["endpoint_record_edit_to_fused_winner_16f_total_ratio"] = total_16f_ms / float(best["total_16f_ms"])
        comparison["fused_winner_16f_total_ms"] = best.get("total_16f_ms")
    owner_last = owner_run_train_eval.get("last_row") if isinstance(owner_run_train_eval, dict) else None
    if isinstance(owner_last, dict) and _finite_number(owner_last.get("total_ms")) and total_16f_ms > 0.0:
        comparison["endpoint_record_edit_to_owner_run_16f_total_ratio"] = total_16f_ms / float(owner_last["total_ms"])
        comparison["owner_run_16f_total_ms"] = owner_last.get("total_ms")
    endpoint_last = endpoint_run_train_eval.get("last_row") if isinstance(endpoint_run_train_eval, dict) else None
    if isinstance(endpoint_last, dict) and _finite_number(endpoint_last.get("total_ms")) and total_16f_ms > 0.0:
        comparison["endpoint_record_edit_to_endpoint_run_16f_total_ratio"] = total_16f_ms / float(
            endpoint_last["total_ms"]
        )
        comparison["endpoint_run_16f_total_ms"] = endpoint_last.get("total_ms")
    comparison["scope_note"] = (
        "Endpoint-record edit train/eval uses the owner+cut-id edit stream shader through a PyTorch autograd "
        "wrapper on frozen-geometry site-RGBA. It is compact and density-independent under the continuous "
        "endpoint semantic, not a full trainer, geometry-gradient path, or matched STAR UVT quality/capacity claim."
    )
    return {
        "available": payload.get("status") == "ok",
        "artifact": str(path),
        "status": payload.get("status"),
        "tape_mode": payload.get("tape_mode"),
        "acceptance": payload.get("acceptance"),
        "frame_scale_first_to_last": payload.get("frame_scale_first_to_last"),
        "total_step_scale_first_to_last": payload.get("total_step_scale_first_to_last"),
        "render_scale_first_to_last": payload.get("render_scale_first_to_last"),
        "backward_scale_first_to_last": payload.get("backward_scale_first_to_last"),
        "selected_tape_segment_scale_first_to_last": payload.get("selected_tape_segment_scale_first_to_last"),
        "selected_tape_storage_scale_first_to_last": payload.get("selected_tape_storage_scale_first_to_last"),
        "endpoint_record_edit_op_scale_first_to_last": payload.get("endpoint_record_edit_op_scale_first_to_last"),
        "full_trainer_claim": payload.get("full_trainer_claim"),
        "full_geometry_gradient_claim": payload.get("full_geometry_gradient_claim"),
        "density_independent_depth_claim": payload.get("density_independent_depth_claim"),
        "continuous_absorption_depth_semantic": payload.get("continuous_absorption_depth_semantic"),
        "optimizer_mode": payload.get("optimizer_mode"),
        "segment_tape_vjp_mode": payload.get("segment_tape_vjp_mode"),
        "last_row": {
            "frames": last_row.get("frame_count"),
            "total_ms": total_16f_ms,
            "render_ms": float(render_summary.get("mean_s", 0.0)) * 1000.0,
            "backward_ms": float(backward_summary.get("mean_s", 0.0)) * 1000.0,
            "final_train_psnr": last_row.get("final_train_psnr"),
            "final_heldout_psnr": last_row.get("final_heldout_psnr"),
            "train_full_segments": last_row.get("train_full_segments"),
            "train_endpoint_run_segments": last_row.get("train_endpoint_run_segments"),
            "train_endpoint_record_edit_ops": last_row.get("train_endpoint_record_edit_ops"),
            "train_selected_tape_segments": last_row.get("train_selected_tape_segments"),
            "train_selected_tape_segments_vs_full": last_row.get("train_selected_tape_segments_vs_full"),
            "train_selected_tape_storage_vs_full": last_row.get("train_selected_tape_storage_vs_full"),
            "train_endpoint_record_edit_storage_vs_endpoint_run": last_row.get(
                "train_endpoint_record_edit_storage_vs_endpoint_run"
            ),
            "max_selected_tape_segments_per_sample": last_row.get("max_selected_tape_segments_per_sample"),
        },
        "comparison": comparison,
        "conclusion": (
            "Endpoint-record edit train/eval is now a measured compact endpoint semantic path: it matches PSNR, "
            "keeps edit storage sublinear and about 0.026x full segment storage at 16f, and runs through the "
            "edit-op Metal shader plus autograd wrapper. Standalone timing is smoke-scale/noisy; the paired "
            "current-process comparison is the speed sanity check, and this is still not a main-trainer or "
            "STAR-UVT competitive claim."
        ),
    }


def _endpoint_record_edit_block4_train_eval_summary(
    path: Path,
    endpoint_run_train_eval: dict[str, Any],
    endpoint_record_edit_train_eval: dict[str, Any],
) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path)}
    payload = _load_json(path)
    rows = payload.get("rows")
    last_row = rows[-1] if isinstance(rows, list) and rows and isinstance(rows[-1], dict) else {}
    step_summary = last_row.get("step_summary", {}) if isinstance(last_row, dict) else {}
    total_summary = step_summary.get("total", {}) if isinstance(step_summary, dict) else {}
    render_summary = step_summary.get("render", {}) if isinstance(step_summary, dict) else {}
    backward_summary = step_summary.get("backward", {}) if isinstance(step_summary, dict) else {}
    total_16f_ms = float(total_summary.get("mean_s", 0.0)) * 1000.0
    comparison: dict[str, Any] = {}
    endpoint_last = endpoint_run_train_eval.get("last_row") if isinstance(endpoint_run_train_eval, dict) else None
    if isinstance(endpoint_last, dict) and _finite_number(endpoint_last.get("total_ms")) and total_16f_ms > 0.0:
        comparison["endpoint_record_edit_block4_to_endpoint_run_16f_total_ratio"] = total_16f_ms / float(
            endpoint_last["total_ms"]
        )
        comparison["endpoint_run_16f_total_ms"] = endpoint_last.get("total_ms")
    edit_last = (
        endpoint_record_edit_train_eval.get("last_row")
        if isinstance(endpoint_record_edit_train_eval, dict)
        else None
    )
    if isinstance(edit_last, dict) and _finite_number(edit_last.get("total_ms")) and total_16f_ms > 0.0:
        comparison["endpoint_record_edit_block4_to_edit_16f_total_ratio"] = total_16f_ms / float(
            edit_last["total_ms"]
        )
        comparison["endpoint_record_edit_16f_total_ms"] = edit_last.get("total_ms")
    comparison["scope_note"] = (
        "Block4 endpoint-record edit train/eval uses block-anchored forward replay plus the dedicated block4 "
        "RGB-only VJP for frozen-geometry site-RGBA gradients. It is an isolated experiment path, not a "
        "main-trainer integration or STAR-UVT quality/capacity claim."
    )
    return {
        "available": payload.get("status") == "ok",
        "artifact": str(path),
        "status": payload.get("status"),
        "tape_mode": payload.get("tape_mode"),
        "acceptance": payload.get("acceptance"),
        "frame_scale_first_to_last": payload.get("frame_scale_first_to_last"),
        "total_step_scale_first_to_last": payload.get("total_step_scale_first_to_last"),
        "render_scale_first_to_last": payload.get("render_scale_first_to_last"),
        "backward_scale_first_to_last": payload.get("backward_scale_first_to_last"),
        "selected_tape_segment_scale_first_to_last": payload.get("selected_tape_segment_scale_first_to_last"),
        "selected_tape_storage_scale_first_to_last": payload.get("selected_tape_storage_scale_first_to_last"),
        "endpoint_record_edit_op_scale_first_to_last": payload.get("endpoint_record_edit_op_scale_first_to_last"),
        "full_trainer_claim": payload.get("full_trainer_claim"),
        "full_geometry_gradient_claim": payload.get("full_geometry_gradient_claim"),
        "density_independent_depth_claim": payload.get("density_independent_depth_claim"),
        "continuous_absorption_depth_semantic": payload.get("continuous_absorption_depth_semantic"),
        "optimizer_mode": payload.get("optimizer_mode"),
        "segment_tape_vjp_mode": payload.get("segment_tape_vjp_mode"),
        "last_row": {
            "frames": last_row.get("frame_count"),
            "total_ms": total_16f_ms,
            "render_ms": float(render_summary.get("mean_s", 0.0)) * 1000.0,
            "backward_ms": float(backward_summary.get("mean_s", 0.0)) * 1000.0,
            "final_train_psnr": last_row.get("final_train_psnr"),
            "final_heldout_psnr": last_row.get("final_heldout_psnr"),
            "train_full_segments": last_row.get("train_full_segments"),
            "train_endpoint_run_segments": last_row.get("train_endpoint_run_segments"),
            "train_endpoint_record_edit_ops": last_row.get("train_endpoint_record_edit_ops"),
            "train_endpoint_record_block4_ops": last_row.get("train_endpoint_record_block4_ops"),
            "train_endpoint_record_block4_change_events": last_row.get(
                "train_endpoint_record_block4_change_events"
            ),
            "train_endpoint_record_block4_changed_records": last_row.get(
                "train_endpoint_record_block4_changed_records"
            ),
            "train_selected_tape_segments": last_row.get("train_selected_tape_segments"),
            "train_selected_tape_segments_vs_full": last_row.get("train_selected_tape_segments_vs_full"),
            "train_selected_tape_storage_vs_full": last_row.get("train_selected_tape_storage_vs_full"),
            "train_endpoint_record_block4_storage_vs_endpoint_run": last_row.get(
                "train_endpoint_record_block4_storage_vs_endpoint_run"
            ),
            "train_endpoint_record_block4_storage_vs_full": last_row.get(
                "train_endpoint_record_block4_storage_vs_full"
            ),
            "train_endpoint_record_edit_storage_vs_endpoint_run": last_row.get(
                "train_endpoint_record_edit_storage_vs_endpoint_run"
            ),
            "max_selected_tape_segments_per_sample": last_row.get("max_selected_tape_segments_per_sample"),
        },
        "comparison": comparison,
        "conclusion": (
            "Endpoint-record block4 train/eval is now measured with the dedicated block4 RGB-only VJP: PSNR "
            "matches endpoint-run/edit and storage remains compact at about 0.044x full segment storage at 16f. "
            "The end-to-end train/eval timing is still noisy and not speed-competitive in this rerun, despite "
            "the raw block4 VJP sidecar validating. This remains isolated fixed-geometry work, not main-trainer "
            "integration or a STAR-UVT competitive claim."
        ),
    }


def _endpoint_record_edit_paired_train_eval_summary(path: Path, *, accept_failed: bool = False) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path)}
    payload = _load_json(path)
    summary = payload.get("summary_16f", {})
    endpoint = summary.get("endpoint-run", {}) if isinstance(summary, dict) else {}
    edit = summary.get("endpoint-record-edit", {}) if isinstance(summary, dict) else {}
    block4 = summary.get("endpoint-record-edit-block4", {}) if isinstance(summary, dict) else {}
    block_coeff = summary.get("endpoint-record-edit-block-coeff", {}) if isinstance(summary, dict) else {}
    block_coeff16 = summary.get("endpoint-record-edit-block-coeff16", {}) if isinstance(summary, dict) else {}
    ratios = payload.get("ratios", {})
    if not isinstance(ratios, dict):
        ratios = summary.get("ratios", {}) if isinstance(summary, dict) else {}
    results = payload.get("results", {})
    result_modes = {
        name: result.get("optimizer_mode")
        for name, result in results.items()
        if isinstance(name, str) and isinstance(result, dict)
    } if isinstance(results, dict) else {}
    total_ratio = ratios.get("edit_to_endpoint_total_16f") if isinstance(ratios, dict) else None
    block4_ratio = ratios.get("block4_to_endpoint_total_16f") if isinstance(ratios, dict) else None
    block_coeff_ratio = ratios.get("block_coeff_to_endpoint_total_16f") if isinstance(ratios, dict) else None
    block_coeff_to_block4_ratio = (
        ratios.get("block_coeff_to_block4_total_16f") if isinstance(ratios, dict) else None
    )
    block_coeff16_ratio = ratios.get("block_coeff16_to_endpoint_total_16f") if isinstance(ratios, dict) else None
    block_coeff16_to_block_coeff_ratio = (
        ratios.get("block_coeff16_to_block_coeff_total_16f") if isinstance(ratios, dict) else None
    )
    speed_read = (
        "slower_than_endpoint_run"
        if _finite_number(total_ratio) and float(total_ratio) > 1.0
        else "not_slower_in_this_smoke_run_but_not_stable"
    )
    block4_speed_read = (
        "faster_than_endpoint_run"
        if _finite_number(block4_ratio) and float(block4_ratio) < 1.0
        else "not_faster_or_not_measured"
    )
    block_coeff_speed_read = (
        "faster_than_endpoint_run"
        if _finite_number(block_coeff_ratio) and float(block_coeff_ratio) < 1.0
        else "not_faster_or_not_measured"
    )
    block_coeff16_speed_read = (
        "faster_than_endpoint_run"
        if _finite_number(block_coeff16_ratio) and float(block_coeff16_ratio) < 1.0
        else "slower_than_endpoint_run"
        if _finite_number(block_coeff16_ratio) and float(block_coeff16_ratio) > 1.0
        else "not_measured"
    )
    edit_clause = (
        "Endpoint-record edit is faster than endpoint-run"
        if _finite_number(total_ratio) and float(total_ratio) < 1.0
        else "Endpoint-record edit is slower than endpoint-run"
    )
    block4_clause = (
        "Block4 is faster than endpoint-run"
        if _finite_number(block4_ratio) and float(block4_ratio) < 1.0
        else "Block4 is not faster than endpoint-run"
    )
    block_coeff_clause = (
        "Block-coeff is faster than endpoint-run"
        if _finite_number(block_coeff_ratio) and float(block_coeff_ratio) < 1.0
        else "Block-coeff is slower than endpoint-run"
        if _finite_number(block_coeff_ratio) and float(block_coeff_ratio) > 1.0
        else "Block-coeff is not measured against endpoint-run"
    )
    block_coeff_vs_block4_clause = (
        "block-coeff is faster than block4"
        if _finite_number(block_coeff_to_block4_ratio) and float(block_coeff_to_block4_ratio) < 1.0
        else "block-coeff remains slower than block4"
    )
    block_coeff16_clause = ""
    if block_coeff16:
        block_coeff16_vs_endpoint_clause = (
            "Block-coeff16 is faster than endpoint-run"
            if _finite_number(block_coeff16_ratio) and float(block_coeff16_ratio) < 1.0
            else "Block-coeff16 is slower than endpoint-run"
        )
        block_coeff16_vs_coeff_clause = (
            "and slower than f32 block-coeff"
            if _finite_number(block_coeff16_to_block_coeff_ratio)
            and float(block_coeff16_to_block_coeff_ratio) > 1.0
            else "and not slower than f32 block-coeff"
        )
        block_coeff16_clause = f" {block_coeff16_vs_endpoint_clause} {block_coeff16_vs_coeff_clause}; keep it negative."
    if block_coeff:
        conclusion = (
            "In the promoted same-process smoke-scale train/eval rerun, the endpoint-record variants keep matched "
            f"PSNR. {block4_clause} and {block_coeff_clause} at 16f; {block_coeff_vs_block4_clause}. Raw edit "
            "speed sign remains noisy across paired repeats, so treat raw edit as storage-first. The edit and "
            "block4 paths stay compact, while block-coeff trades heavier storage for less hot-loop math. These are "
            "block-anchored variants compared in the same current process, not a stable benchmark or a STAR-UVT claim."
            f"{block_coeff16_clause}"
        )
    else:
        conclusion = (
            "In a same-process smoke-scale train/eval rerun, endpoint-record edit keeps the same PSNR and much "
            "lower storage. The saved paired runs have noisy speed sign and the latest longer RGB-only repeat is "
            "slower than endpoint-run at 16f, so treat edit-op as a compact storage path that still needs replay "
            "optimization before any speed-competitive claim. When block4 or block-coeff is present in the "
            "artifact, those are block-anchored variants compared in the same current process; this is still "
            "not a STAR-UVT claim."
        )
    return {
        "available": (payload.get("status") == "ok" or accept_failed)
        and isinstance(endpoint, dict)
        and isinstance(edit, dict),
        "artifact": str(path),
        "status": payload.get("status"),
        "acceptance": payload.get("acceptance"),
        "scope": payload.get("scope"),
        "endpoint_run_16f": endpoint,
        "endpoint_record_edit_16f": edit,
        "endpoint_record_edit_block4_16f": block4,
        "endpoint_record_edit_block_coeff_16f": block_coeff,
        "endpoint_record_edit_block_coeff16_16f": block_coeff16,
        "ratios": ratios,
        "optimizer_modes": result_modes,
        "speed_read": speed_read,
        "block4_speed_read": block4_speed_read,
        "block_coeff_speed_read": block_coeff_speed_read,
        "block_coeff16_speed_read": block_coeff16_speed_read,
        "conclusion": conclusion,
    }


def _endpoint_record_edit_block_coeff16_storagefix_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path)}
    payload = _load_json(path)
    rows = payload.get("rows")
    row = rows[-1] if isinstance(rows, list) and rows and isinstance(rows[-1], dict) else {}
    step_summary = row.get("step_summary", {}) if isinstance(row, dict) else {}
    total_summary = step_summary.get("total", {}) if isinstance(step_summary, dict) else {}
    render_summary = step_summary.get("render", {}) if isinstance(step_summary, dict) else {}
    backward_summary = step_summary.get("backward", {}) if isinstance(step_summary, dict) else {}
    selected_bytes = row.get("train_selected_tape_storage_bytes")
    endpoint_bytes = row.get("train_endpoint_run_storage_bytes")
    block4_bytes = row.get("train_endpoint_record_block4_storage_bytes")
    selected_vs_full = row.get("train_selected_tape_storage_vs_full")
    endpoint_vs_full = row.get("train_endpoint_run_storage_vs_full")
    block4_vs_full = row.get("train_endpoint_record_block4_storage_vs_full")
    return {
        "available": payload.get("status") == "ok" and row.get("status") == "ok",
        "artifact": str(path),
        "status": payload.get("status"),
        "tape_mode": payload.get("tape_mode"),
        "optimizer_mode": payload.get("optimizer_mode"),
        "render_size": payload.get("render_size"),
        "frame_counts": payload.get("frame_counts"),
        "last_row": {
            "frames": row.get("frame_count"),
            "total_ms": float(total_summary.get("mean_s", 0.0)) * 1000.0,
            "render_ms": float(render_summary.get("mean_s", 0.0)) * 1000.0,
            "backward_ms": float(backward_summary.get("mean_s", 0.0)) * 1000.0,
            "final_heldout_psnr": row.get("final_heldout_psnr"),
            "train_selected_tape_storage_bytes": selected_bytes,
            "train_endpoint_run_storage_bytes": endpoint_bytes,
            "train_endpoint_record_block4_storage_bytes": block4_bytes,
            "train_selected_tape_storage_vs_full": selected_vs_full,
            "train_endpoint_run_storage_vs_full": endpoint_vs_full,
            "train_endpoint_record_block4_storage_vs_full": block4_vs_full,
        },
        "storage_accounting": {
            "selected_storage_not_endpoint_run": selected_bytes != endpoint_bytes,
            "selected_storage_above_block4": (
                _finite_number(selected_bytes)
                and _finite_number(block4_bytes)
                and float(selected_bytes) > float(block4_bytes)
            ),
            "selected_storage_above_endpoint_run": (
                _finite_number(selected_vs_full)
                and _finite_number(endpoint_vs_full)
                and float(selected_vs_full) > float(endpoint_vs_full)
            ),
            "selected_storage_below_f32_coeff_reference": (
                _finite_number(selected_vs_full)
                and float(selected_vs_full) < 0.1812
            ),
        },
        "conclusion": (
            "The coeff16 storage-accounting smoke exercises the real manual-VJP train/eval path after fixing the "
            "unreachable coeff16 branch. The selected tape is now counted as block-edit storage plus a half-precision "
            "coefficient sidecar, not endpoint-run storage. It remains a storage-accounting and correctness smoke, "
            "not a speed promotion."
        ),
    }


def _segment_tape_autograd_smoke_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"available": False, "artifact": str(path)}
    payload = _load_json(path)
    rows = payload.get("rows")
    rels = [
        float(row.get("grad_error_vs_manual_direct_atomic_grad_only", {}).get("rel_to_manual_abs_max"))
        for row in rows
        if isinstance(row, dict)
        and _finite_number(row.get("grad_error_vs_manual_direct_atomic_grad_only", {}).get("rel_to_manual_abs_max"))
    ] if isinstance(rows, list) else []
    return {
        "available": payload.get("status") == "ok",
        "artifact": str(path),
        "status": payload.get("status"),
        "acceptance": payload.get("acceptance"),
        "full_trainer_claim": payload.get("full_trainer_claim"),
        "full_geometry_gradient_claim": payload.get("full_geometry_gradient_claim"),
        "density_independent_depth_claim": payload.get("density_independent_depth_claim"),
        "owner_run_segments_vs_full": payload.get("owner_run_segments_vs_full"),
        "max_owner_run_segments_per_sample": payload.get("max_owner_run_segments_per_sample"),
        "max_grad_rel_error_vs_manual_vjp": max(rels) if rels else None,
        "modes": [row.get("mode") for row in rows if isinstance(row, dict)] if isinstance(rows, list) else [],
        "conclusion": (
            "Segment-tape replay now has a PyTorch autograd wrapper for frozen-geometry site-RGBA training; "
            "the smoke compares wrapper gradients against the explicit Metal VJP for both direct and track modes."
        ),
    }


def summarize(args: argparse.Namespace) -> dict[str, Any]:
    verifier = _load_json(args.verifier_json)
    depth_order = _load_json(args.depth_order_json)
    ownerupdate = _load_json(args.ownerupdate_json)
    segment_tape = _segment_tape_summary(args.segment_tape_json)
    topology_sharing = _topology_sharing_summary(args.topology_sharing_json)
    delta_tape = _delta_tape_summary(args.delta_tape_json)
    boundary_delta_tape = _boundary_delta_tape_summary(args.boundary_delta_tape_json)
    record_delta_tape = _record_delta_tape_summary(args.record_delta_tape_json)
    owner_run_tape = _owner_run_tape_summary(args.owner_run_tape_json)
    owner_run_boundary_tape = _owner_run_boundary_tape_summary(args.owner_run_boundary_tape_json)
    owner_run_internal_tape = _owner_run_internal_tape_summary(args.owner_run_internal_tape_json)
    endpoint_run_tape = _endpoint_run_tape_summary(args.endpoint_run_tape_json)
    endpoint_record_delta_tape = _endpoint_record_delta_tape_summary(args.endpoint_record_delta_tape_json)
    endpoint_record_delta_replay = _endpoint_record_delta_replay_summary(args.endpoint_record_delta_replay_json)
    endpoint_record_edit_replay = _endpoint_record_edit_replay_summary(args.endpoint_record_edit_replay_json)
    endpoint_record_edit_rgb_only_replay = _endpoint_record_edit_rgb_only_replay_summary(
        args.endpoint_record_edit_rgb_only_replay_json
    )
    endpoint_record_edit_trackloop_replay = _endpoint_record_edit_trackloop_replay_summary(
        args.endpoint_record_edit_trackloop_replay_json
    )
    endpoint_record_edit_block4_replay = _endpoint_record_edit_block4_replay_summary(
        args.endpoint_record_edit_block4_replay_json
    )
    endpoint_record_edit_block_coeff_replay = _endpoint_record_edit_block_coeff_replay_summary(
        args.endpoint_record_edit_block_coeff_replay_json,
        args.endpoint_record_edit_block_coeff_sweep_json,
    )
    endpoint_record_edit_block_coeff_train_eval = _endpoint_record_edit_block_coeff_train_eval_summary(
        args.endpoint_record_edit_block_coeff_train_eval_json
    )
    modes = _mode_table(verifier)
    best_mode = verifier.get("best_mode")
    best = next((row for row in modes if row["mode"] == best_mode), None)
    owner_run_train_eval = _owner_run_train_eval_summary(args.owner_run_train_eval_json, best)
    active_internal_train_eval = _active_internal_train_eval_summary(
        args.active_internal_train_eval_json,
        best,
        owner_run_train_eval,
    )
    full_tape_train_eval = _full_tape_train_eval_summary(
        args.full_tape_train_eval_json,
        best,
        owner_run_train_eval,
        active_internal_train_eval,
    )
    endpoint_run_train_eval = _endpoint_run_train_eval_summary(
        args.endpoint_run_train_eval_json,
        best,
        owner_run_train_eval,
        active_internal_train_eval,
    )
    endpoint_record_edit_train_eval = _endpoint_record_edit_train_eval_summary(
        args.endpoint_record_edit_train_eval_json,
        best,
        owner_run_train_eval,
        endpoint_run_train_eval,
    )
    endpoint_record_edit_block4_train_eval = _endpoint_record_edit_block4_train_eval_summary(
        args.endpoint_record_edit_block4_train_eval_json,
        endpoint_run_train_eval,
        endpoint_record_edit_train_eval,
    )
    endpoint_record_edit_paired_train_eval = _endpoint_record_edit_paired_train_eval_summary(
        args.endpoint_record_edit_paired_train_eval_json
    )
    endpoint_record_edit_block4_paired_train_eval = _endpoint_record_edit_paired_train_eval_summary(
        args.endpoint_record_edit_block4_paired_train_eval_json
    )
    endpoint_record_edit_block_coeff_paired_train_eval = _endpoint_record_edit_paired_train_eval_summary(
        args.endpoint_record_edit_block_coeff_paired_train_eval_json
    )
    endpoint_record_edit_block_coeff_repeat20_16f = _endpoint_record_edit_paired_train_eval_summary(
        args.endpoint_record_edit_block_coeff_repeat20_16f_json
    )
    endpoint_record_edit_block_coeff_repeat20_2_4_8_16 = _endpoint_record_edit_paired_train_eval_summary(
        args.endpoint_record_edit_block_coeff_repeat20_2_4_8_16_json,
        accept_failed=True,
    )
    endpoint_record_edit_block_coeff16_manual_vjp_paired_train_eval = _endpoint_record_edit_paired_train_eval_summary(
        args.endpoint_record_edit_block_coeff16_manual_vjp_paired_train_eval_json
    )
    endpoint_record_edit_block_coeff16_storagefix_smoke = _endpoint_record_edit_block_coeff16_storagefix_summary(
        args.endpoint_record_edit_block_coeff16_storagefix_smoke_json
    )
    endpoint_record_edit_rgb_only_paired_train_eval = _endpoint_record_edit_paired_train_eval_summary(
        args.endpoint_record_edit_rgb_only_paired_train_eval_json
    )
    endpoint_record_edit_manual_vjp_paired_train_eval = _endpoint_record_edit_paired_train_eval_summary(
        args.endpoint_record_edit_manual_vjp_paired_train_eval_json
    )
    segment_tape_autograd = _segment_tape_autograd_smoke_summary(args.segment_tape_autograd_smoke_json)
    depth_diag = depth_order.get("candidate_depth_order_diagnostics", {})
    owner_diag = ownerupdate.get("ownerupdate_diagnostics", {})
    owner_vjp_diag = ownerupdate.get("mixed_vjp_direct_grad_only_ownerupdate_diagnostics", {})
    smoke = _smoke_coverage(verifier)
    psnr_spread = _psnr_spread_by_frame(verifier)
    framegroup_lossreduce = _framegroup_lossreduce_summary(verifier)
    framegroup_compare = _framegroup_compare_summary(verifier)
    framegroup_real32_compare = _framegroup_real32_compare_summary(verifier)
    framegroup_i16x4_compare = _framegroup_i16x4_compare_summary(verifier)
    framegroup_i16x4_prewarm_compare = _framegroup_i16x4_prewarm_compare_summary(
        args.framegroup_i16x4_prewarm_compare_json
    )
    framegroup_packed_prewarm_compare = _framegroup_packed_prewarm_compare_summary(
        args.framegroup_packed_prewarm_compare_json
    )
    framegroup_packed_broad_compare = _framegroup_packed_broad_compare_summary(
        args.framegroup_packed_broad_compare_json
    )
    framegroup_autograd_smoke = _framegroup_autograd_smoke_summary(args.framegroup_autograd_smoke_json)
    framegroup_autograd_speedscale = _framegroup_autograd_speedscale_summary(
        args.framegroup_autograd_speedscale_json
    )
    explicit_checklist = {
        "forked_shader_variant_present": (ROOT / "third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0").exists(),
        "three_or_more_vjp_variants_tested": len(modes) >= 3,
        "speed_scaled_across_2_4_8_16_frames": tuple(verifier.get("frame_counts", ())) == (2, 4, 8, 16),
        "psnr_recorded_for_train_eval": psnr_spread is not None,
        "aggregate_verifier_green": verifier.get("status") == "ok" and not verifier.get("failures"),
        "rgb_and_rgba_depth_smokes_present": smoke["has_rgb_seed"] and smoke["has_rgba_depth_seed"],
        "autograd_wrapper_vjp_checked": smoke["autograd_checked"],
        "ordered_append_rejected_by_depth_probe": bool(depth_diag) and not bool(depth_diag.get("ordered_append_safe", True)),
        "ownerupdate_rejected_by_failed_artifact": ownerupdate.get("status") == "failed"
        and bool(owner_diag)
        and bool(owner_vjp_diag),
        "segment_tape_math_probe_green": segment_tape.get("available") is True
        and segment_tape.get("completion_claim") is False
        and segment_tape.get("metal_kernel_implemented") is True,
        "topology_sharing_probe_recorded": topology_sharing.get("available") is True
        and topology_sharing.get("completion_claim") is False,
        "delta_tape_probe_recorded": delta_tape.get("available") is True and delta_tape.get("completion_claim") is False,
        "boundary_delta_tape_probe_recorded": boundary_delta_tape.get("available") is True
        and boundary_delta_tape.get("completion_claim") is False,
        "record_delta_tape_probe_recorded": record_delta_tape.get("available") is True
        and record_delta_tape.get("completion_claim") is False,
        "owner_run_tape_probe_recorded": owner_run_tape.get("available") is True
        and owner_run_tape.get("completion_claim") is False,
        "owner_run_boundary_tape_probe_recorded": owner_run_boundary_tape.get("available") is True
        and owner_run_boundary_tape.get("completion_claim") is False,
        "owner_run_internal_tape_probe_recorded": owner_run_internal_tape.get("available") is True
        and owner_run_internal_tape.get("completion_claim") is False,
        "endpoint_run_tape_probe_recorded": endpoint_run_tape.get("available") is True
        and endpoint_run_tape.get("completion_claim") is False,
        "endpoint_record_delta_tape_probe_recorded": endpoint_record_delta_tape.get("available") is True
        and endpoint_record_delta_tape.get("completion_claim") is False,
        "endpoint_record_delta_replay_shader_green": endpoint_record_delta_replay.get("available") is True
        and endpoint_record_delta_replay.get("completion_claim") is False,
        "endpoint_record_edit_replay_shader_green": endpoint_record_edit_replay.get("available") is True
        and endpoint_record_edit_replay.get("completion_claim") is False,
        "endpoint_record_edit_rgb_only_replay_shader_green": endpoint_record_edit_rgb_only_replay.get("available")
        is True
        and endpoint_record_edit_rgb_only_replay.get("completion_claim") is False,
        "endpoint_record_edit_trackloop_replay_sidecar_recorded": endpoint_record_edit_trackloop_replay.get(
            "available"
        )
        is True
        and endpoint_record_edit_trackloop_replay.get("completion_claim") is False,
        "endpoint_record_edit_block4_replay_sidecar_recorded": endpoint_record_edit_block4_replay.get("available")
        is True
        and endpoint_record_edit_block4_replay.get("completion_claim") is False,
        "endpoint_record_edit_block_coeff_forward_sidecar_recorded": endpoint_record_edit_block_coeff_replay.get(
            "available"
        )
        is True
        and endpoint_record_edit_block_coeff_replay.get("completion_claim") is False,
        "endpoint_record_edit_block_coeff_rgb_train_eval_smoke_recorded": endpoint_record_edit_block_coeff_train_eval.get(
            "available"
        )
        is True
        and endpoint_record_edit_block_coeff_train_eval.get("full_trainer_claim") is False
        and endpoint_record_edit_block_coeff_train_eval.get("continuous_absorption_depth_semantic") is True,
        "owner_run_rgb_train_eval_recorded": owner_run_train_eval.get("available") is True
        and owner_run_train_eval.get("full_trainer_claim") is False
        and owner_run_train_eval.get("full_geometry_gradient_claim") is False,
        "active_internal_rgb_train_eval_recorded": active_internal_train_eval.get("available") is True
        and active_internal_train_eval.get("full_trainer_claim") is False
        and active_internal_train_eval.get("density_independent_depth_claim") is False,
        "full_tape_rgb_train_eval_recorded": full_tape_train_eval.get("available") is True
        and full_tape_train_eval.get("full_trainer_claim") is False,
        "endpoint_run_rgb_train_eval_recorded": endpoint_run_train_eval.get("available") is True
        and endpoint_run_train_eval.get("full_trainer_claim") is False
        and endpoint_run_train_eval.get("continuous_absorption_depth_semantic") is True,
        "endpoint_record_edit_rgb_train_eval_recorded": endpoint_record_edit_train_eval.get("available") is True
        and endpoint_record_edit_train_eval.get("full_trainer_claim") is False
        and endpoint_record_edit_train_eval.get("continuous_absorption_depth_semantic") is True,
        "endpoint_record_edit_block4_rgb_train_eval_recorded": endpoint_record_edit_block4_train_eval.get(
            "available"
        )
        is True
        and endpoint_record_edit_block4_train_eval.get("full_trainer_claim") is False
        and endpoint_record_edit_block4_train_eval.get("continuous_absorption_depth_semantic") is True,
        "endpoint_record_edit_paired_train_eval_recorded": endpoint_record_edit_paired_train_eval.get("available")
        is True,
        "endpoint_record_edit_block4_paired_train_eval_recorded": endpoint_record_edit_block4_paired_train_eval.get(
            "available"
        )
        is True,
        "endpoint_record_edit_block_coeff_paired_train_eval_recorded": endpoint_record_edit_block_coeff_paired_train_eval.get(
            "available"
        )
        is True,
        "endpoint_record_edit_block_coeff_repeat20_16f_recorded": endpoint_record_edit_block_coeff_repeat20_16f.get(
            "available"
        )
        is True,
        "endpoint_record_edit_block_coeff_repeat20_2_4_8_16_negative_recorded": endpoint_record_edit_block_coeff_repeat20_2_4_8_16.get(
            "available"
        )
        is True
        and endpoint_record_edit_block_coeff_repeat20_2_4_8_16.get("status") == "failed"
        and endpoint_record_edit_block_coeff_repeat20_2_4_8_16.get("block_coeff_speed_read")
        != "faster_than_endpoint_run",
        "endpoint_record_edit_block_coeff16_manual_vjp_negative_recorded": endpoint_record_edit_block_coeff16_manual_vjp_paired_train_eval.get(
            "available"
        )
        is True
        and endpoint_record_edit_block_coeff16_manual_vjp_paired_train_eval.get("block_coeff16_speed_read")
        == "slower_than_endpoint_run",
        "endpoint_record_edit_block_coeff16_storagefix_smoke_recorded": endpoint_record_edit_block_coeff16_storagefix_smoke.get(
            "available"
        )
        is True
        and endpoint_record_edit_block_coeff16_storagefix_smoke.get("storage_accounting", {}).get(
            "selected_storage_not_endpoint_run"
        )
        is True,
        "endpoint_record_edit_rgb_only_paired_train_eval_recorded": endpoint_record_edit_rgb_only_paired_train_eval.get(
            "available"
        )
        is True,
        "endpoint_record_edit_manual_vjp_paired_train_eval_recorded": endpoint_record_edit_manual_vjp_paired_train_eval.get(
            "available"
        )
        is True,
        "segment_tape_autograd_smoke_green": segment_tape_autograd.get("available") is True
        and segment_tape_autograd.get("full_trainer_claim") is False
        and segment_tape_autograd.get("full_geometry_gradient_claim") is False,
        "framegroup16_lossreduce_render32_guardrail_green": framegroup_lossreduce.get("available") is True
        and framegroup_lossreduce.get("completion_claim") is False
        and framegroup_lossreduce.get("full_trainer_claim") is False,
        "framegroup16_compare_render32_speedscale_guardrail_green": framegroup_compare.get("available") is True
        and framegroup_compare.get("completion_claim") is False
        and framegroup_compare.get("full_trainer_claim") is False,
        "framegroup16_real32_render32_sublinear_recorded": framegroup_real32_compare.get("available") is True
        and framegroup_real32_compare.get("completion_claim") is False
        and framegroup_real32_compare.get("full_trainer_claim") is False
        and framegroup_real32_compare.get("real_frame_sublinear_claim") is True,
        "framegroup16_i16x4_nonpromotion_guardrail_recorded": framegroup_i16x4_compare.get("available") is True
        and framegroup_i16x4_compare.get("completion_claim") is False
        and framegroup_i16x4_compare.get("full_trainer_claim") is False
        and framegroup_i16x4_compare.get("i16x4_speed_promotion_candidate") is False
        and framegroup_i16x4_compare.get("i16x4_total_sublinear_claim") is False
        and framegroup_i16x4_compare.get("i16x4_backward_sublinear_claim") is False,
        "framegroup16_i16x4_prewarm_nonpromotion_recorded": framegroup_i16x4_prewarm_compare.get("available")
        is True
        and framegroup_i16x4_prewarm_compare.get("completion_claim") is False
        and framegroup_i16x4_prewarm_compare.get("full_trainer_claim") is False
        and framegroup_i16x4_prewarm_compare.get("i16x4_speed_promotion_candidate") is False
        and framegroup_i16x4_prewarm_compare.get("speed_rejected_by_ratio") is True,
        "framegroup16_packed_prewarm_candidate_recorded": framegroup_packed_prewarm_compare.get("available")
        is True
        and framegroup_packed_prewarm_compare.get("completion_claim") is False
        and framegroup_packed_prewarm_compare.get("full_trainer_claim") is False
        and framegroup_packed_prewarm_compare.get("packed_speed_promotion_candidate") is True
        and framegroup_packed_prewarm_compare.get("packed_storage_below_i16x3") is True,
        "framegroup16_packed_broad_nonpromotion_recorded": framegroup_packed_broad_compare.get("available")
        is True
        and framegroup_packed_broad_compare.get("completion_claim") is False
        and framegroup_packed_broad_compare.get("full_trainer_claim") is False
        and framegroup_packed_broad_compare.get("packed_speed_promotion_candidate") is False
        and framegroup_packed_broad_compare.get("speed_rejected_by_128") is True,
        "framegroup16_fused_mse_autograd_smoke_green": framegroup_autograd_smoke.get("available") is True
        and framegroup_autograd_smoke.get("completion_claim") is False
        and framegroup_autograd_smoke.get("full_trainer_claim") is False,
        "framegroup16_fused_mse_autograd_speedscale_green": framegroup_autograd_speedscale.get("available") is True
        and framegroup_autograd_speedscale.get("completion_claim") is False
        and framegroup_autograd_speedscale.get("full_trainer_claim") is False,
    }
    missing = sorted(key for key, ok in explicit_checklist.items() if not ok)
    return {
        "summary": "world_foam_fused_slab_mixed_status",
        "status": "ok_current_shader_gate_with_structural_gap" if not missing else "incomplete_evidence",
        "completion_claim": False,
        "star_uvt_competitive_claim": False,
        "open_items_before_completion": [
            "Thread the segment-tape autograd wrapper into the main trainer if fixed-geometry/site-RGBA World Foam is promoted beyond isolated experiment scripts.",
            "Thread the endpoint-record block4 path into the main trainer if the target is more than isolated fixed-geometry/site-RGBA train/eval.",
            "Keep improving the endpoint-record delta/block4 structural replay if the target is a better-than-per-frame runtime path at larger quality/capacity settings.",
            "Reduce or stabilize the dedicated block4 VJP train-loop overhead; the raw VJP sidecar is correct, but the full autograd rerun is still not speed-competitive.",
            "Investigate why the longer matched paired 2/4/8/16 repeat makes block-coeff fail at 16f before treating the sidecar as a practical training result.",
            "Do not promote the f16 coefficient-cache sidecar unless a future run fixes the current manual-VJP speed regression.",
            "Rerun a matched STAR-UVT versus World Foam quality/capacity comparison before making a competitive claim.",
            "Stabilize and broaden real-frame scaling for the selected framegroup16 path; the current real-loaded 16/32 shader row is sublinear, but it is still a narrow render32/site12 fixed-geometry result.",
            "Do not promote the i16x4 framegroup fork until a fresh artifact fixes the current 16f->32f timing cliff and updates the non-promotion guard explicitly.",
            "Resolve i16x4 cadence sensitivity before using its sublinear prewarm-sweep result; the prewarmed 32f row is still slower than i16x3.",
            "Broaden the packed framegroup candidate beyond the current 16/32 paired prewarm smoke before making it the default speed path.",
            "Fix or reject the packed 128f timing loss before treating packed as a broad speed promotion; the 64/128 interleaved guard currently rejects it.",
            "Find a compact exact replay layout for owner+boundary records; the current exact record-delta probe is accurate but about full-CSR sized.",
            "Decide whether World Foam can adopt continuous-absorption endpoint depth; it gives compact density-independent replay, but is a semantic change from current segment-mid depth.",
            "Promote beyond fixed-geometry/site-RGBA only if full trainer or geometry-gradient parity is required.",
        ],
        "reason_not_complete": (
            "Current World Foam fused path is measured and verified. A compact Metal segment-tape shader now "
            "verifies the replay/VJP math, and same-owner runs now pass an isolated RGB train/eval probe, but "
            "the naive exact segment count still scales about with frame count, simple owner-topology sharing "
            "is weak on the moving-camera probe, and endpoint-only continuous density depth does not reproduce "
            "the current segment-mid depth tape. Internal cuts recover exact current-depth replay, but the "
            "density-independent all-cut version is much closer to full segment-tape storage; exact owner+cut-id "
            "record deltas also replay topology accurately but are about full-CSR sized. Endpoint-run continuous "
            "absorption now provides a compact density-independent semantic-change path, and endpoint owner+cut-id "
            "record deltas expose a strongly sublinear edit stream. Real owner+cut-id replacement-row and edit-op "
            "replay shaders now validate forward/VJP and sublinear storage, and the edit-op shader now has a "
            "matched fixed-geometry RGB train/eval autograd path. The edit-op path is compact in practice, but "
            "the raw-edit speed sign remains noisy: the promoted paired render32 2/4/8/16 smoke shows raw edit "
            "faster at 16f, while a 20-step 16f repeat and older RGB-only/manual-VJP comparisons show it slower. "
            "Block4 and block-coeff remain faster than endpoint-run in the promoted smoke and the 20-step repeat. "
            "The track-loop forward variant validates numerically but is also slower than endpoint-run at 16f. "
            "A block4 anchored forward shader now validates, beats "
            "the original edit replay in the isolated 16f replay probe, and now has a dedicated RGB-only block4 "
            "VJP that matches the old edit VJP checks. In the corrected fixed-geometry RGB train/eval rerun, however, "
            "the block4 VJP path is still noisy and not speed-competitive despite sublinear frame scaling. No "
            "endpoint-record shader is integrated into the main trainer. A coefficient-cached block edit "
            "sidecar now validates forward replay, RGB-only VJP, and a green 20-step render32 2/4/8/16 "
            "autograd sweep with sublinear measured total/render/backward scaling; however, the refreshed clean "
            "16f replay is speed-positive against endpoint-run/original edit forward but not block4, the promoted "
            "render32 paired smoke is still not a long stable benchmark, and its coefficient table is storage-heavy "
            "at tiny frame counts and still above endpoint CSR storage. A longer matched paired 2/4/8/16 repeat "
            "completed as negative/informational evidence: block4 was fastest at 16f, raw edit was faster than "
            "endpoint-run, and block-coeff failed the 16f speed gate. A f16 coefficient-cache sidecar validates "
            "PSNR in a manual-VJP smoke but is slower than endpoint-run and f32 block-coeff, so it is recorded as "
            "negative rather than promoted. A follow-up coeff16 storage smoke fixed the selected-storage accounting "
            "so the f16 sidecar is no longer reported as endpoint-run storage. The promoted framegroup16 fused-MSE "
            "shader is fast in the synthetic repeated-frame speed-scale guard and now has a real-loaded 16/32 "
            "compare that beats endpoint-run at both rows with sublinear total/backward scale after the 32-frame "
            "chunk patch. The i16x4 framegroup fork is correctness-green and has small storage overhead, but the "
            "current repeated-frame 16f->32f artifact shows a large timing cliff, so it is recorded as negative "
            "non-promotion evidence. A warmer prewarm-sweep artifact flips the reason: i16x4 becomes sublinear, "
            "but it is still slower than i16x3 at 32f mean total/backward time and remains cadence-sensitive. The "
            "packed-record framegroup fork now has a paired prewarmed 16/32 train/eval artifact where it beats "
            "i16x3 mean total/backward time, matches PSNR, and uses less selected tape storage; this is recorded "
            "as a candidate rather than full promotion because earlier standalone/interleaved timings were "
            "cadence-sensitive. A broader interleaved 64/128 guard now rejects broad promotion: packed wins at "
            "64f but loses mean total/backward time at 128f, while preserving the storage win and PSNR match. The "
            "active-internal train/eval path "
            "is measured and practical but not structurally "
            "STAR-like, and the full-tape exact baseline is slower than the current fused winner. This is not "
            "claimed competitive with STAR UVT."
        ),
        "checklist": explicit_checklist,
        "missing_checklist_items": missing,
        "winner": best,
        "mode_table": modes,
        "psnr_spread_across_modes_by_frame": psnr_spread,
        "smoke_coverage": smoke,
        "star_uvt_speed_reference": _star_speed_reference(
            args.star_speed_json,
            best,
            endpoint_record_edit_block_coeff_train_eval,
        ),
        "segment_tape_probe": segment_tape,
        "framegroup16_lossreduce_render32": framegroup_lossreduce,
        "framegroup16_compare_render32_speedscale": framegroup_compare,
        "framegroup16_real32_render32_compare": framegroup_real32_compare,
        "framegroup16_i16x4_compare": framegroup_i16x4_compare,
        "framegroup16_i16x4_prewarm_compare": framegroup_i16x4_prewarm_compare,
        "framegroup16_packed_prewarm_compare": framegroup_packed_prewarm_compare,
        "framegroup16_packed_broad_compare": framegroup_packed_broad_compare,
        "framegroup16_autograd_smoke": framegroup_autograd_smoke,
        "framegroup16_autograd_speedscale": framegroup_autograd_speedscale,
        "topology_sharing_probe": topology_sharing,
        "delta_tape_probe": delta_tape,
        "boundary_delta_tape_probe": boundary_delta_tape,
        "record_delta_tape_probe": record_delta_tape,
        "owner_run_tape_probe": owner_run_tape,
        "owner_run_boundary_tape_probe": owner_run_boundary_tape,
        "owner_run_internal_tape_probe": owner_run_internal_tape,
        "endpoint_run_tape_probe": endpoint_run_tape,
        "endpoint_record_delta_tape_probe": endpoint_record_delta_tape,
        "endpoint_record_delta_replay": endpoint_record_delta_replay,
        "endpoint_record_edit_replay": endpoint_record_edit_replay,
        "endpoint_record_edit_rgb_only_replay": endpoint_record_edit_rgb_only_replay,
        "endpoint_record_edit_trackloop_replay": endpoint_record_edit_trackloop_replay,
        "endpoint_record_edit_block4_replay": endpoint_record_edit_block4_replay,
        "endpoint_record_edit_block_coeff_replay": endpoint_record_edit_block_coeff_replay,
        "endpoint_record_edit_block_coeff_rgb_train_eval": endpoint_record_edit_block_coeff_train_eval,
        "owner_run_rgb_train_eval": owner_run_train_eval,
        "active_internal_rgb_train_eval": active_internal_train_eval,
        "full_tape_rgb_train_eval": full_tape_train_eval,
        "endpoint_run_rgb_train_eval": endpoint_run_train_eval,
        "endpoint_record_edit_rgb_train_eval": endpoint_record_edit_train_eval,
        "endpoint_record_edit_block4_rgb_train_eval": endpoint_record_edit_block4_train_eval,
        "endpoint_record_edit_paired_train_eval": endpoint_record_edit_paired_train_eval,
        "endpoint_record_edit_block4_paired_train_eval": endpoint_record_edit_block4_paired_train_eval,
        "endpoint_record_edit_block_coeff_paired_train_eval": endpoint_record_edit_block_coeff_paired_train_eval,
        "endpoint_record_edit_block_coeff_repeat20_16f": endpoint_record_edit_block_coeff_repeat20_16f,
        "endpoint_record_edit_block_coeff_repeat20_2_4_8_16": endpoint_record_edit_block_coeff_repeat20_2_4_8_16,
        "endpoint_record_edit_block_coeff16_manual_vjp_paired_train_eval": endpoint_record_edit_block_coeff16_manual_vjp_paired_train_eval,
        "endpoint_record_edit_block_coeff16_storagefix_smoke": endpoint_record_edit_block_coeff16_storagefix_smoke,
        "endpoint_record_edit_rgb_only_paired_train_eval": endpoint_record_edit_rgb_only_paired_train_eval,
        "endpoint_record_edit_manual_vjp_paired_train_eval": endpoint_record_edit_manual_vjp_paired_train_eval,
        "segment_tape_autograd_smoke": segment_tape_autograd,
        "negative_variants": {
            "ownerupdate": {
                "status": ownerupdate.get("status"),
                "max_forward_error": owner_diag.get("max_error"),
                "max_vjp_rel_delta_vs_reduce": owner_vjp_diag.get("max_grad_rel_delta_vs_reduce"),
                "conclusion": "rejected",
                "artifact": str(args.ownerupdate_json),
            },
            "ordered_append": {
                "ordered_append_safe": depth_diag.get("ordered_append_safe"),
                "adjacent_inversions": depth_diag.get("adjacent_inversions"),
                "adjacent_pairs": depth_diag.get("adjacent_pairs"),
                "samples_with_adjacent_inversions": depth_diag.get("samples_with_adjacent_inversions"),
                "checked_samples": depth_diag.get("checked_samples"),
                "conclusion": "rejected",
                "artifact": str(args.depth_order_json),
            },
            "owner_topology_sharing": {
                "status": topology_sharing.get("status"),
                "same_topology_all_frames_tracks_16f": topology_sharing.get("last_row", {}).get(
                    "same_topology_all_frames_tracks"
                )
                if isinstance(topology_sharing.get("last_row"), dict)
                else None,
                "track_unique_topology_rows_vs_samples_16f": topology_sharing.get("last_row", {}).get(
                    "track_unique_topology_rows_vs_samples"
                )
                if isinstance(topology_sharing.get("last_row"), dict)
                else None,
                "conclusion": "weak_not_star_like",
                "artifact": str(args.topology_sharing_json),
            },
            "coarse_delta_change_rows": {
                "status": delta_tape.get("status"),
                "change_event_scale_first_to_last": delta_tape.get("change_event_scale_first_to_last"),
                "edit_op_scale_first_to_last": delta_tape.get("edit_op_scale_first_to_last"),
                "delta_owner_storage_scale_first_to_last": delta_tape.get(
                    "delta_owner_storage_scale_first_to_last"
                ),
                "conclusion": "coarse_changed_rows_rejected_edit_ops_promising",
                "artifact": str(args.delta_tape_json),
            },
            "raw_boundary_order_delta": {
                "status": boundary_delta_tape.get("status"),
                "boundary_edit_op_scale_first_to_last": boundary_delta_tape.get(
                    "boundary_edit_op_scale_first_to_last"
                ),
                "delta_replace_boundary_storage_scale_first_to_last": boundary_delta_tape.get(
                    "delta_replace_boundary_storage_scale_first_to_last"
                ),
                "conclusion": "exact_geometry_signal_promising_raw_order_not_fixed",
                "artifact": str(args.boundary_delta_tape_json),
            },
            "exact_record_delta": {
                "status": record_delta_tape.get("status"),
                "full_record_count_scale_first_to_last": record_delta_tape.get(
                    "full_record_count_scale_first_to_last"
                ),
                "record_edit_op_scale_first_to_last": record_delta_tape.get(
                    "record_edit_op_scale_first_to_last"
                ),
                "delta_replace_record_storage_scale_first_to_last": record_delta_tape.get(
                    "delta_replace_record_storage_scale_first_to_last"
                ),
                "delta_replace_record_vs_full_segment_csr_16f": record_delta_tape.get("last_row", {}).get(
                    "delta_replace_record_vs_full_segment_csr"
                )
                if isinstance(record_delta_tape.get("last_row"), dict)
                else None,
                "delta_edit_op_record_stream_vs_full_segment_csr_16f": record_delta_tape.get("last_row", {}).get(
                    "delta_edit_op_record_stream_vs_full_segment_csr"
                )
                if isinstance(record_delta_tape.get("last_row"), dict)
                else None,
                "conclusion": "exact_owner_cutid_replay_not_compact_star_like",
                "artifact": str(args.record_delta_tape_json),
            },
            "same_owner_run_depth_dependency": {
                "status": owner_run_tape.get("status"),
                "owner_run_segments_vs_full_segments_16f": owner_run_tape.get("last_row", {}).get(
                    "owner_run_segments_vs_full_segments"
                )
                if isinstance(owner_run_tape.get("last_row"), dict)
                else None,
                "max_rgb_only_vjp_rel_error": owner_run_tape.get("max_rgb_only_vjp_rel_error"),
                "conclusion": "rgb_training_candidate_depth_mid_density_dependent",
                "artifact": str(args.owner_run_tape_json),
            },
            "owner_run_boundary_endpoint_depth": {
                "status": owner_run_boundary_tape.get("status"),
                "max_endpoint_length_abs_error": owner_run_boundary_tape.get("max_endpoint_length_abs_error"),
                "max_endpoint_density_depth_abs_error_vs_current_owner_run": owner_run_boundary_tape.get(
                    "max_endpoint_density_depth_abs_error_vs_current_owner_run"
                ),
                "owner_run_boundary_run_scale_first_to_last": owner_run_boundary_tape.get(
                    "owner_run_boundary_run_scale_first_to_last"
                ),
                "owner_run_boundary_id_vs_full_segment_csr_16f": owner_run_boundary_tape.get("last_row", {}).get(
                    "owner_run_boundary_id_vs_full_segment_csr"
                )
                if isinstance(owner_run_boundary_tape.get("last_row"), dict)
                else None,
                "conclusion": "length_coefficients_exact_endpoint_only_depth_rejected_run_count_not_sublinear",
                "artifact": str(args.owner_run_boundary_tape_json),
            },
            "owner_run_internal_cut_depth": {
                "status": owner_run_internal_tape.get("status"),
                "active_internal_nested_csr_vs_full_segment_csr_16f": owner_run_internal_tape.get(
                    "last_row", {}
                ).get("active_internal_nested_csr_vs_full_segment_csr")
                if isinstance(owner_run_internal_tape.get("last_row"), dict)
                else None,
                "all_internal_nested_csr_vs_full_segment_csr_16f": owner_run_internal_tape.get("last_row", {}).get(
                    "all_internal_nested_csr_vs_full_segment_csr"
                )
                if isinstance(owner_run_internal_tape.get("last_row"), dict)
                else None,
                "all_internal_endpoint_run_csr_vs_full_segment_csr_16f": owner_run_internal_tape.get(
                    "last_row", {}
                ).get("all_internal_endpoint_run_csr_vs_full_segment_csr")
                if isinstance(owner_run_internal_tape.get("last_row"), dict)
                else None,
                "active_half_density_depth_max_abs_16f": owner_run_internal_tape.get("last_row", {}).get(
                    "active_half_density_depth_max_abs"
                )
                if isinstance(owner_run_internal_tape.get("last_row"), dict)
                else None,
                "conclusion": "active_internal_exact_current_density_all_internal_exact_endpoint_semantic_change_compact",
                "artifact": str(args.owner_run_internal_tape_json),
            },
            "endpoint_run_continuous_depth_semantic": {
                "status": endpoint_run_tape.get("status"),
                "endpoint_run_scale_first_to_last": endpoint_run_tape.get("endpoint_run_scale_first_to_last"),
                "endpoint_storage_vs_full_segment_csr_16f": endpoint_run_tape.get("last_row", {}).get(
                    "endpoint_storage_vs_full_segment_csr"
                )
                if isinstance(endpoint_run_tape.get("last_row"), dict)
                else None,
                "max_vjp_rel_error_vs_torch_autograd": endpoint_run_tape.get(
                    "max_vjp_rel_error_vs_torch_autograd"
                ),
                "conclusion": "compact_density_independent_if_depth_semantic_changes_not_structurally_sublinear",
                "artifact": str(args.endpoint_run_tape_json),
            },
            "endpoint_record_delta_signal": {
                "status": endpoint_record_delta_tape.get("status"),
                "endpoint_record_count_scale_first_to_last": endpoint_record_delta_tape.get(
                    "endpoint_record_count_scale_first_to_last"
                ),
                "endpoint_record_edit_op_scale_first_to_last": endpoint_record_delta_tape.get(
                    "endpoint_record_edit_op_scale_first_to_last"
                ),
                "delta_edit_op_endpoint_record_storage_scale_first_to_last": endpoint_record_delta_tape.get(
                    "delta_edit_op_endpoint_record_storage_scale_first_to_last"
                ),
                "full_endpoint_record_csr_vs_full_segment_csr_16f": endpoint_record_delta_tape.get(
                    "last_row", {}
                ).get("full_endpoint_record_csr_vs_full_segment_csr")
                if isinstance(endpoint_record_delta_tape.get("last_row"), dict)
                else None,
                "delta_edit_op_endpoint_record_stream_vs_full_segment_csr_16f": endpoint_record_delta_tape.get(
                    "last_row", {}
                ).get("delta_edit_op_endpoint_record_stream_vs_full_segment_csr")
                if isinstance(endpoint_record_delta_tape.get("last_row"), dict)
                else None,
                "conclusion": "promising_endpoint_delta_signal_not_shipped_shader",
                "artifact": str(args.endpoint_record_delta_tape_json),
            },
            "endpoint_record_delta_replay_shader_scope": {
                "status": endpoint_record_delta_replay.get("status"),
                "record_delta_storage_scale_first_to_last": endpoint_record_delta_replay.get(
                    "record_delta_storage_scale_first_to_last"
                ),
                "record_delta_storage_vs_endpoint_csr_16f": endpoint_record_delta_replay.get("last_row", {}).get(
                    "record_delta_storage_vs_endpoint_csr"
                )
                if isinstance(endpoint_record_delta_replay.get("last_row"), dict)
                else None,
                "record_delta_storage_vs_full_segment_csr_16f": endpoint_record_delta_replay.get(
                    "last_row", {}
                ).get("record_delta_storage_vs_full_segment_csr")
                if isinstance(endpoint_record_delta_replay.get("last_row"), dict)
                else None,
                "record_delta_forward_ms_16f": endpoint_record_delta_replay.get("last_row", {}).get(
                    "record_delta_forward_ms"
                )
                if isinstance(endpoint_record_delta_replay.get("last_row"), dict)
                else None,
                "record_delta_vjp_ms_16f": endpoint_record_delta_replay.get("last_row", {}).get(
                    "record_delta_vjp_ms"
                )
                if isinstance(endpoint_record_delta_replay.get("last_row"), dict)
                else None,
                "max_forward_abs_error_vs_endpoint_run": endpoint_record_delta_replay.get(
                    "max_forward_abs_error_vs_endpoint_run"
                ),
                "max_vjp_rel_error_vs_endpoint_run": endpoint_record_delta_replay.get(
                    "max_vjp_rel_error_vs_endpoint_run"
                ),
                "conclusion": "real_owner_cutid_replacement_shader_green_sidecar_not_trainer_integrated",
                "artifact": str(args.endpoint_record_delta_replay_json),
            },
            "endpoint_record_edit_replay_shader_scope": {
                "status": endpoint_record_edit_replay.get("status"),
                "edit_op_scale_first_to_last": endpoint_record_edit_replay.get("edit_op_scale_first_to_last"),
                "edit_storage_scale_first_to_last": endpoint_record_edit_replay.get(
                    "edit_storage_scale_first_to_last"
                ),
                "edit_storage_vs_endpoint_csr_16f": endpoint_record_edit_replay.get("last_row", {}).get(
                    "edit_storage_vs_endpoint_csr"
                )
                if isinstance(endpoint_record_edit_replay.get("last_row"), dict)
                else None,
                "edit_storage_vs_full_segment_csr_16f": endpoint_record_edit_replay.get("last_row", {}).get(
                    "edit_storage_vs_full_segment_csr"
                )
                if isinstance(endpoint_record_edit_replay.get("last_row"), dict)
                else None,
                "endpoint_forward_ms_16f": endpoint_record_edit_replay.get("last_row", {}).get(
                    "endpoint_forward_ms"
                )
                if isinstance(endpoint_record_edit_replay.get("last_row"), dict)
                else None,
                "edit_forward_ms_16f": endpoint_record_edit_replay.get("last_row", {}).get("edit_forward_ms")
                if isinstance(endpoint_record_edit_replay.get("last_row"), dict)
                else None,
                "endpoint_vjp_ms_16f": endpoint_record_edit_replay.get("last_row", {}).get("endpoint_vjp_ms")
                if isinstance(endpoint_record_edit_replay.get("last_row"), dict)
                else None,
                "edit_vjp_ms_16f": endpoint_record_edit_replay.get("last_row", {}).get("edit_vjp_ms")
                if isinstance(endpoint_record_edit_replay.get("last_row"), dict)
                else None,
                "max_forward_abs_error_vs_endpoint_run": endpoint_record_edit_replay.get(
                    "max_forward_abs_error_vs_endpoint_run"
                ),
                "max_vjp_rel_error_vs_endpoint_run": endpoint_record_edit_replay.get(
                    "max_vjp_rel_error_vs_endpoint_run"
                ),
                "conclusion": "real_owner_cutid_editop_shader_green_storage_win_not_speed_or_trainer_integrated",
                "artifact": str(args.endpoint_record_edit_replay_json),
            },
            "endpoint_record_edit_rgb_only_replay_shader_scope": {
                "status": endpoint_record_edit_rgb_only_replay.get("status"),
                "max_rgb_only_vjp_rel_error_vs_full_zero_alpha_depth": endpoint_record_edit_rgb_only_replay.get(
                    "max_rgb_only_vjp_rel_error_vs_full_zero_alpha_depth"
                ),
                "edit_storage_vs_full_segment_csr_16f": endpoint_record_edit_rgb_only_replay.get("last_row", {}).get(
                    "edit_storage_vs_full_segment_csr"
                )
                if isinstance(endpoint_record_edit_rgb_only_replay.get("last_row"), dict)
                else None,
                "edit_rgb_only_vjp_ms_16f": endpoint_record_edit_rgb_only_replay.get("last_row", {}).get(
                    "edit_rgb_only_vjp_ms"
                )
                if isinstance(endpoint_record_edit_rgb_only_replay.get("last_row"), dict)
                else None,
                "conclusion": "rgb_only_vjp_correct_storage_sublinear_not_stable_speed_win",
                "artifact": str(args.endpoint_record_edit_rgb_only_replay_json),
            },
            "endpoint_record_edit_trackloop_replay_scope": {
                "status": endpoint_record_edit_trackloop_replay.get("status"),
                "max_trackloop_forward_abs_error_vs_endpoint_run": endpoint_record_edit_trackloop_replay.get(
                    "max_trackloop_forward_abs_error_vs_endpoint_run"
                ),
                "edit_storage_vs_full_segment_csr_16f": endpoint_record_edit_trackloop_replay.get(
                    "last_row", {}
                ).get("edit_storage_vs_full_segment_csr")
                if isinstance(endpoint_record_edit_trackloop_replay.get("last_row"), dict)
                else None,
                "endpoint_forward_ms_16f": endpoint_record_edit_trackloop_replay.get(
                    "trackloop_timing_read", {}
                ).get("endpoint_forward_ms_16f")
                if isinstance(endpoint_record_edit_trackloop_replay.get("trackloop_timing_read"), dict)
                else None,
                "edit_forward_ms_16f": endpoint_record_edit_trackloop_replay.get("trackloop_timing_read", {}).get(
                    "edit_forward_ms_16f"
                )
                if isinstance(endpoint_record_edit_trackloop_replay.get("trackloop_timing_read"), dict)
                else None,
                "edit_trackloop_forward_ms_16f": endpoint_record_edit_trackloop_replay.get(
                    "trackloop_timing_read", {}
                ).get("edit_trackloop_forward_ms_16f")
                if isinstance(endpoint_record_edit_trackloop_replay.get("trackloop_timing_read"), dict)
                else None,
                "conclusion": "trackloop_forward_correct_storage_sublinear_rejected_as_speed_optimization",
                "artifact": str(args.endpoint_record_edit_trackloop_replay_json),
            },
            "endpoint_record_edit_block4_replay_scope": {
                "status": endpoint_record_edit_block4_replay.get("status"),
                "max_block4_forward_abs_error_vs_endpoint_run": endpoint_record_edit_block4_replay.get(
                    "max_block4_forward_abs_error_vs_endpoint_run"
                ),
                "endpoint_forward_ms_16f": endpoint_record_edit_block4_replay.get("block4_timing_read", {}).get(
                    "endpoint_forward_ms_16f"
                )
                if isinstance(endpoint_record_edit_block4_replay.get("block4_timing_read"), dict)
                else None,
                "edit_forward_ms_16f": endpoint_record_edit_block4_replay.get("block4_timing_read", {}).get(
                    "edit_forward_ms_16f"
                )
                if isinstance(endpoint_record_edit_block4_replay.get("block4_timing_read"), dict)
                else None,
                "edit_block4_forward_ms_16f": endpoint_record_edit_block4_replay.get("block4_timing_read", {}).get(
                    "edit_block4_forward_ms_16f"
                )
                if isinstance(endpoint_record_edit_block4_replay.get("block4_timing_read"), dict)
                else None,
                "block4_storage_vs_full_segment_csr_16f": endpoint_record_edit_block4_replay.get(
                    "block4_storage_read", {}
                ).get("block4_storage_vs_full_segment_csr_16f")
                if isinstance(endpoint_record_edit_block4_replay.get("block4_storage_read"), dict)
                else None,
                "block4_storage_vs_endpoint_csr_16f": endpoint_record_edit_block4_replay.get(
                    "block4_storage_read", {}
                ).get("block4_storage_vs_endpoint_csr_16f")
                if isinstance(endpoint_record_edit_block4_replay.get("block4_storage_read"), dict)
                else None,
                "conclusion": "block4_forward_and_rgb_vjp_correct_train_eval_still_isolated",
                "artifact": str(args.endpoint_record_edit_block4_replay_json),
            },
            "endpoint_record_edit_block_coeff_forward_scope": {
                "status": endpoint_record_edit_block_coeff_replay.get("status"),
                "speed_read": endpoint_record_edit_block_coeff_replay.get("speed_read"),
                "max_block_coeff_forward_abs_error_vs_endpoint_run": endpoint_record_edit_block_coeff_replay.get(
                    "max_block_coeff_forward_abs_error_vs_endpoint_run"
                ),
                "endpoint_forward_ms_16f": endpoint_record_edit_block_coeff_replay.get("last_row", {}).get(
                    "endpoint_forward_ms"
                )
                if isinstance(endpoint_record_edit_block_coeff_replay.get("last_row"), dict)
                else None,
                "edit_block4_forward_ms_16f": endpoint_record_edit_block_coeff_replay.get("last_row", {}).get(
                    "edit_block4_forward_ms"
                )
                if isinstance(endpoint_record_edit_block_coeff_replay.get("last_row"), dict)
                else None,
                "edit_block_coeff_forward_ms_16f": endpoint_record_edit_block_coeff_replay.get("last_row", {}).get(
                    "edit_block_coeff_forward_ms"
                )
                if isinstance(endpoint_record_edit_block_coeff_replay.get("last_row"), dict)
                else None,
                "edit_block_coeff_rgb_only_vjp_ms_16f": endpoint_record_edit_block_coeff_replay.get(
                    "last_row", {}
                ).get("edit_block_coeff_rgb_only_vjp_ms")
                if isinstance(endpoint_record_edit_block_coeff_replay.get("last_row"), dict)
                else None,
                "max_block_coeff_rgb_only_vjp_rel_error_vs_full_zero_alpha_depth": endpoint_record_edit_block_coeff_replay.get(
                    "max_block_coeff_rgb_only_vjp_rel_error_vs_full_zero_alpha_depth"
                ),
                "max_block_coeff_rgb_only_vjp_rel_error_vs_edit_rgb_only": endpoint_record_edit_block_coeff_replay.get(
                    "max_block_coeff_rgb_only_vjp_rel_error_vs_edit_rgb_only"
                ),
                "max_block_coeff_rgb_only_vjp_rel_error_vs_block4_rgb_only": endpoint_record_edit_block_coeff_replay.get(
                    "max_block_coeff_rgb_only_vjp_rel_error_vs_block4_rgb_only"
                ),
                "block_coeff_storage_vs_endpoint_csr_16f": endpoint_record_edit_block_coeff_replay.get(
                    "last_row", {}
                ).get("block_coeff_storage_vs_endpoint_csr")
                if isinstance(endpoint_record_edit_block_coeff_replay.get("last_row"), dict)
                else None,
                "block_coeff_storage_vs_full_segment_csr_16f": endpoint_record_edit_block_coeff_replay.get(
                    "last_row", {}
                ).get("block_coeff_storage_vs_full_segment_csr")
                if isinstance(endpoint_record_edit_block_coeff_replay.get("last_row"), dict)
                else None,
                "conclusion": "coeff_cached_forward_and_rgb_vjp_correct_one_step_autograd_smoke_storage_heavy",
                "artifact": str(args.endpoint_record_edit_block_coeff_replay_json),
                "sweep_artifact": str(args.endpoint_record_edit_block_coeff_sweep_json),
            },
            "endpoint_record_edit_block_coeff_rgb_train_eval_scope": {
                "status": endpoint_record_edit_block_coeff_train_eval.get("status"),
                "tape_mode": endpoint_record_edit_block_coeff_train_eval.get("tape_mode"),
                "optimizer_mode": endpoint_record_edit_block_coeff_train_eval.get("optimizer_mode"),
                "endpoint_record_edit_block_coeff_16f_total_ms": endpoint_record_edit_block_coeff_train_eval.get(
                    "last_row", {}
                ).get("total_ms")
                if isinstance(endpoint_record_edit_block_coeff_train_eval.get("last_row"), dict)
                else None,
                "endpoint_record_edit_block_coeff_16f_render_ms": endpoint_record_edit_block_coeff_train_eval.get(
                    "last_row", {}
                ).get("render_ms")
                if isinstance(endpoint_record_edit_block_coeff_train_eval.get("last_row"), dict)
                else None,
                "endpoint_record_edit_block_coeff_16f_backward_ms": endpoint_record_edit_block_coeff_train_eval.get(
                    "last_row", {}
                ).get("backward_ms")
                if isinstance(endpoint_record_edit_block_coeff_train_eval.get("last_row"), dict)
                else None,
                "final_heldout_psnr_16f": endpoint_record_edit_block_coeff_train_eval.get("last_row", {}).get(
                    "final_heldout_psnr"
                )
                if isinstance(endpoint_record_edit_block_coeff_train_eval.get("last_row"), dict)
                else None,
                "train_selected_tape_storage_vs_full_16f": endpoint_record_edit_block_coeff_train_eval.get(
                    "last_row", {}
                ).get("train_selected_tape_storage_vs_full")
                if isinstance(endpoint_record_edit_block_coeff_train_eval.get("last_row"), dict)
                else None,
                "conclusion": "coeff_cached_20step_render32_autograd_sweep_green_not_main_trainer_or_star",
                "artifact": str(args.endpoint_record_edit_block_coeff_train_eval_json),
            },
            "endpoint_record_edit_block_coeff_paired_train_eval_scope": {
                "status": endpoint_record_edit_block_coeff_paired_train_eval.get("status"),
                "block_coeff_speed_read": endpoint_record_edit_block_coeff_paired_train_eval.get(
                    "block_coeff_speed_read"
                ),
                "block_coeff_to_endpoint_total_16f": endpoint_record_edit_block_coeff_paired_train_eval.get(
                    "ratios", {}
                ).get("block_coeff_to_endpoint_total_16f")
                if isinstance(endpoint_record_edit_block_coeff_paired_train_eval.get("ratios"), dict)
                else None,
                "block_coeff_to_block4_total_16f": endpoint_record_edit_block_coeff_paired_train_eval.get(
                    "ratios", {}
                ).get("block_coeff_to_block4_total_16f")
                if isinstance(endpoint_record_edit_block_coeff_paired_train_eval.get("ratios"), dict)
                else None,
                "block_coeff_16f_total_ms": endpoint_record_edit_block_coeff_paired_train_eval.get(
                    "endpoint_record_edit_block_coeff_16f", {}
                ).get("total_ms")
                if isinstance(
                    endpoint_record_edit_block_coeff_paired_train_eval.get("endpoint_record_edit_block_coeff_16f"),
                    dict,
                )
                else None,
                "block4_16f_total_ms": endpoint_record_edit_block_coeff_paired_train_eval.get(
                    "endpoint_record_edit_block4_16f", {}
                ).get("total_ms")
                if isinstance(
                    endpoint_record_edit_block_coeff_paired_train_eval.get("endpoint_record_edit_block4_16f"),
                    dict,
                )
                else None,
                "endpoint_run_16f_total_ms": endpoint_record_edit_block_coeff_paired_train_eval.get(
                    "endpoint_run_16f", {}
                ).get("total_ms")
                if isinstance(endpoint_record_edit_block_coeff_paired_train_eval.get("endpoint_run_16f"), dict)
                else None,
                "conclusion": "block_coeff_paired_speed_positive_not_star_claim",
                "artifact": str(args.endpoint_record_edit_block_coeff_paired_train_eval_json),
            },
            "endpoint_record_edit_block_coeff_repeat20_16f_scope": {
                "status": endpoint_record_edit_block_coeff_repeat20_16f.get("status"),
                "speed_read": endpoint_record_edit_block_coeff_repeat20_16f.get("speed_read"),
                "block4_speed_read": endpoint_record_edit_block_coeff_repeat20_16f.get("block4_speed_read"),
                "block_coeff_speed_read": endpoint_record_edit_block_coeff_repeat20_16f.get(
                    "block_coeff_speed_read"
                ),
                "edit_to_endpoint_total_16f": endpoint_record_edit_block_coeff_repeat20_16f.get(
                    "ratios", {}
                ).get("edit_to_endpoint_total_16f")
                if isinstance(endpoint_record_edit_block_coeff_repeat20_16f.get("ratios"), dict)
                else None,
                "block4_to_endpoint_total_16f": endpoint_record_edit_block_coeff_repeat20_16f.get(
                    "ratios", {}
                ).get("block4_to_endpoint_total_16f")
                if isinstance(endpoint_record_edit_block_coeff_repeat20_16f.get("ratios"), dict)
                else None,
                "block_coeff_to_endpoint_total_16f": endpoint_record_edit_block_coeff_repeat20_16f.get(
                    "ratios", {}
                ).get("block_coeff_to_endpoint_total_16f")
                if isinstance(endpoint_record_edit_block_coeff_repeat20_16f.get("ratios"), dict)
                else None,
                "block_coeff_to_block4_total_16f": endpoint_record_edit_block_coeff_repeat20_16f.get(
                    "ratios", {}
                ).get("block_coeff_to_block4_total_16f")
                if isinstance(endpoint_record_edit_block_coeff_repeat20_16f.get("ratios"), dict)
                else None,
                "endpoint_run_16f_total_ms": endpoint_record_edit_block_coeff_repeat20_16f.get(
                    "endpoint_run_16f", {}
                ).get("total_ms")
                if isinstance(endpoint_record_edit_block_coeff_repeat20_16f.get("endpoint_run_16f"), dict)
                else None,
                "raw_edit_16f_total_ms": endpoint_record_edit_block_coeff_repeat20_16f.get(
                    "endpoint_record_edit_16f", {}
                ).get("total_ms")
                if isinstance(endpoint_record_edit_block_coeff_repeat20_16f.get("endpoint_record_edit_16f"), dict)
                else None,
                "block4_16f_total_ms": endpoint_record_edit_block_coeff_repeat20_16f.get(
                    "endpoint_record_edit_block4_16f", {}
                ).get("total_ms")
                if isinstance(
                    endpoint_record_edit_block_coeff_repeat20_16f.get("endpoint_record_edit_block4_16f"),
                    dict,
                )
                else None,
                "block_coeff_16f_total_ms": endpoint_record_edit_block_coeff_repeat20_16f.get(
                    "endpoint_record_edit_block_coeff_16f", {}
                ).get("total_ms")
                if isinstance(
                    endpoint_record_edit_block_coeff_repeat20_16f.get("endpoint_record_edit_block_coeff_16f"),
                    dict,
                )
                else None,
                "conclusion": "repeat20_confirms_block_coeff_speed_positive_raw_edit_slower_not_star_claim",
                "artifact": str(args.endpoint_record_edit_block_coeff_repeat20_16f_json),
            },
            "endpoint_record_edit_block_coeff_repeat20_2_4_8_16_scope": {
                "status": endpoint_record_edit_block_coeff_repeat20_2_4_8_16.get("status"),
                "speed_read": endpoint_record_edit_block_coeff_repeat20_2_4_8_16.get("speed_read"),
                "block4_speed_read": endpoint_record_edit_block_coeff_repeat20_2_4_8_16.get("block4_speed_read"),
                "block_coeff_speed_read": endpoint_record_edit_block_coeff_repeat20_2_4_8_16.get(
                    "block_coeff_speed_read"
                ),
                "edit_to_endpoint_total_16f": endpoint_record_edit_block_coeff_repeat20_2_4_8_16.get(
                    "ratios", {}
                ).get("edit_to_endpoint_total_16f")
                if isinstance(endpoint_record_edit_block_coeff_repeat20_2_4_8_16.get("ratios"), dict)
                else None,
                "block4_to_endpoint_total_16f": endpoint_record_edit_block_coeff_repeat20_2_4_8_16.get(
                    "ratios", {}
                ).get("block4_to_endpoint_total_16f")
                if isinstance(endpoint_record_edit_block_coeff_repeat20_2_4_8_16.get("ratios"), dict)
                else None,
                "block_coeff_to_endpoint_total_16f": endpoint_record_edit_block_coeff_repeat20_2_4_8_16.get(
                    "ratios", {}
                ).get("block_coeff_to_endpoint_total_16f")
                if isinstance(endpoint_record_edit_block_coeff_repeat20_2_4_8_16.get("ratios"), dict)
                else None,
                "endpoint_run_16f_total_ms": endpoint_record_edit_block_coeff_repeat20_2_4_8_16.get(
                    "endpoint_run_16f", {}
                ).get("total_ms")
                if isinstance(endpoint_record_edit_block_coeff_repeat20_2_4_8_16.get("endpoint_run_16f"), dict)
                else None,
                "raw_edit_16f_total_ms": endpoint_record_edit_block_coeff_repeat20_2_4_8_16.get(
                    "endpoint_record_edit_16f", {}
                ).get("total_ms")
                if isinstance(
                    endpoint_record_edit_block_coeff_repeat20_2_4_8_16.get("endpoint_record_edit_16f"),
                    dict,
                )
                else None,
                "block4_16f_total_ms": endpoint_record_edit_block_coeff_repeat20_2_4_8_16.get(
                    "endpoint_record_edit_block4_16f", {}
                ).get("total_ms")
                if isinstance(
                    endpoint_record_edit_block_coeff_repeat20_2_4_8_16.get("endpoint_record_edit_block4_16f"),
                    dict,
                )
                else None,
                "block_coeff_16f_total_ms": endpoint_record_edit_block_coeff_repeat20_2_4_8_16.get(
                    "endpoint_record_edit_block_coeff_16f", {}
                ).get("total_ms")
                if isinstance(
                    endpoint_record_edit_block_coeff_repeat20_2_4_8_16.get(
                        "endpoint_record_edit_block_coeff_16f"
                    ),
                    dict,
                )
                else None,
                "conclusion": "longer_2_4_8_16_repeat_negative_block_coeff_failed_16f_speed_gate",
                "artifact": str(args.endpoint_record_edit_block_coeff_repeat20_2_4_8_16_json),
            },
            "endpoint_record_edit_block4_rgb_train_eval_scope": {
                "status": endpoint_record_edit_block4_train_eval.get("status"),
                "total_step_scale_first_to_last": endpoint_record_edit_block4_train_eval.get(
                    "total_step_scale_first_to_last"
                ),
                "selected_tape_storage_scale_first_to_last": endpoint_record_edit_block4_train_eval.get(
                    "selected_tape_storage_scale_first_to_last"
                ),
                "endpoint_record_edit_block4_16f_total_ms": endpoint_record_edit_block4_train_eval.get(
                    "last_row", {}
                ).get("total_ms")
                if isinstance(endpoint_record_edit_block4_train_eval.get("last_row"), dict)
                else None,
                "endpoint_record_edit_block4_storage_vs_full_16f": endpoint_record_edit_block4_train_eval.get(
                    "last_row", {}
                ).get("train_endpoint_record_block4_storage_vs_full")
                if isinstance(endpoint_record_edit_block4_train_eval.get("last_row"), dict)
                else None,
                "endpoint_record_edit_block4_storage_vs_endpoint_16f": endpoint_record_edit_block4_train_eval.get(
                    "last_row", {}
                ).get("train_endpoint_record_block4_storage_vs_endpoint_run")
                if isinstance(endpoint_record_edit_block4_train_eval.get("last_row"), dict)
                else None,
                "conclusion": "block4_train_eval_green_dedicated_rgb_vjp_no_main_trainer_or_star_claim",
                "artifact": str(args.endpoint_record_edit_block4_train_eval_json),
            },
            "endpoint_record_edit_rgb_train_eval_scope": {
                "status": endpoint_record_edit_train_eval.get("status"),
                "total_step_scale_first_to_last": endpoint_record_edit_train_eval.get(
                    "total_step_scale_first_to_last"
                ),
                "selected_tape_storage_scale_first_to_last": endpoint_record_edit_train_eval.get(
                    "selected_tape_storage_scale_first_to_last"
                ),
                "endpoint_record_edit_op_scale_first_to_last": endpoint_record_edit_train_eval.get(
                    "endpoint_record_edit_op_scale_first_to_last"
                ),
                "endpoint_record_edit_16f_total_ms": endpoint_record_edit_train_eval.get("last_row", {}).get(
                    "total_ms"
                )
                if isinstance(endpoint_record_edit_train_eval.get("last_row"), dict)
                else None,
                "final_heldout_psnr_16f": endpoint_record_edit_train_eval.get("last_row", {}).get(
                    "final_heldout_psnr"
                )
                if isinstance(endpoint_record_edit_train_eval.get("last_row"), dict)
                else None,
                "train_selected_tape_storage_vs_full_16f": endpoint_record_edit_train_eval.get("last_row", {}).get(
                    "train_selected_tape_storage_vs_full"
                )
                if isinstance(endpoint_record_edit_train_eval.get("last_row"), dict)
                else None,
                "conclusion": "compact_endpoint_edit_train_eval_green_smoke_scale_not_main_trainer_or_star_claim",
                "artifact": str(args.endpoint_record_edit_train_eval_json),
            },
            "endpoint_record_edit_paired_train_eval_scope": {
                "status": endpoint_record_edit_paired_train_eval.get("status"),
                "edit_to_endpoint_total_16f": endpoint_record_edit_paired_train_eval.get("ratios", {}).get(
                    "edit_to_endpoint_total_16f"
                )
                if isinstance(endpoint_record_edit_paired_train_eval.get("ratios"), dict)
                else None,
                "edit_to_endpoint_render_16f": endpoint_record_edit_paired_train_eval.get("ratios", {}).get(
                    "edit_to_endpoint_render_16f"
                )
                if isinstance(endpoint_record_edit_paired_train_eval.get("ratios"), dict)
                else None,
                "edit_storage_vs_full_16f": endpoint_record_edit_paired_train_eval.get(
                    "endpoint_record_edit_16f", {}
                ).get("storage_vs_full")
                if isinstance(endpoint_record_edit_paired_train_eval.get("endpoint_record_edit_16f"), dict)
                else None,
                "endpoint_storage_vs_full_16f": endpoint_record_edit_paired_train_eval.get(
                    "endpoint_run_16f", {}
                ).get("storage_vs_full")
                if isinstance(endpoint_record_edit_paired_train_eval.get("endpoint_run_16f"), dict)
                else None,
                "conclusion": "paired_smoke_edit_storage_win_but_slower_than_endpoint_run",
                "artifact": str(args.endpoint_record_edit_paired_train_eval_json),
            },
            "endpoint_record_edit_rgb_only_paired_train_eval_scope": {
                "status": endpoint_record_edit_rgb_only_paired_train_eval.get("status"),
                "speed_read": endpoint_record_edit_rgb_only_paired_train_eval.get("speed_read"),
                "edit_to_endpoint_total_16f": endpoint_record_edit_rgb_only_paired_train_eval.get("ratios", {}).get(
                    "edit_to_endpoint_total_16f"
                )
                if isinstance(endpoint_record_edit_rgb_only_paired_train_eval.get("ratios"), dict)
                else None,
                "edit_to_endpoint_render_16f": endpoint_record_edit_rgb_only_paired_train_eval.get("ratios", {}).get(
                    "edit_to_endpoint_render_16f"
                )
                if isinstance(endpoint_record_edit_rgb_only_paired_train_eval.get("ratios"), dict)
                else None,
                "edit_storage_vs_full_16f": endpoint_record_edit_rgb_only_paired_train_eval.get(
                    "endpoint_record_edit_16f", {}
                ).get("storage_vs_full")
                if isinstance(endpoint_record_edit_rgb_only_paired_train_eval.get("endpoint_record_edit_16f"), dict)
                else None,
                "endpoint_storage_vs_full_16f": endpoint_record_edit_rgb_only_paired_train_eval.get(
                    "endpoint_run_16f", {}
                ).get("storage_vs_full")
                if isinstance(endpoint_record_edit_rgb_only_paired_train_eval.get("endpoint_run_16f"), dict)
                else None,
                "conclusion": "rgb_only_paired_repeat_latest_slower_speed_not_stable_storage_win",
                "artifact": str(args.endpoint_record_edit_rgb_only_paired_train_eval_json),
            },
            "endpoint_record_edit_manual_vjp_paired_train_eval_scope": {
                "status": endpoint_record_edit_manual_vjp_paired_train_eval.get("status"),
                "optimizer_modes": endpoint_record_edit_manual_vjp_paired_train_eval.get("optimizer_modes"),
                "speed_read": endpoint_record_edit_manual_vjp_paired_train_eval.get("speed_read"),
                "edit_to_endpoint_total_16f": endpoint_record_edit_manual_vjp_paired_train_eval.get("ratios", {}).get(
                    "edit_to_endpoint_total_16f"
                )
                if isinstance(endpoint_record_edit_manual_vjp_paired_train_eval.get("ratios"), dict)
                else None,
                "edit_to_endpoint_render_16f": endpoint_record_edit_manual_vjp_paired_train_eval.get("ratios", {}).get(
                    "edit_to_endpoint_render_16f"
                )
                if isinstance(endpoint_record_edit_manual_vjp_paired_train_eval.get("ratios"), dict)
                else None,
                "edit_to_endpoint_backward_16f": endpoint_record_edit_manual_vjp_paired_train_eval.get(
                    "ratios", {}
                ).get("edit_to_endpoint_backward_16f")
                if isinstance(endpoint_record_edit_manual_vjp_paired_train_eval.get("ratios"), dict)
                else None,
                "conclusion": "manual_vjp_still_slower_forward_replay_is_next_target",
                "artifact": str(args.endpoint_record_edit_manual_vjp_paired_train_eval_json),
            },
            "owner_run_rgb_train_eval_scope": {
                "status": owner_run_train_eval.get("status"),
                "total_step_scale_first_to_last": owner_run_train_eval.get("total_step_scale_first_to_last"),
                "owner_run_16f_total_ms": owner_run_train_eval.get("last_row", {}).get("total_ms")
                if isinstance(owner_run_train_eval.get("last_row"), dict)
                else None,
                "final_heldout_psnr_16f": owner_run_train_eval.get("last_row", {}).get("final_heldout_psnr")
                if isinstance(owner_run_train_eval.get("last_row"), dict)
                else None,
                "conclusion": "practical_rgb_path_green_not_full_geometry_or_density_independent_depth",
                "artifact": str(args.owner_run_train_eval_json),
            },
            "active_internal_rgb_train_eval_scope": {
                "status": active_internal_train_eval.get("status"),
                "total_step_scale_first_to_last": active_internal_train_eval.get("total_step_scale_first_to_last"),
                "active_internal_16f_total_ms": active_internal_train_eval.get("last_row", {}).get("total_ms")
                if isinstance(active_internal_train_eval.get("last_row"), dict)
                else None,
                "train_selected_tape_storage_vs_full_16f": active_internal_train_eval.get("last_row", {}).get(
                    "train_selected_tape_storage_vs_full"
                )
                if isinstance(active_internal_train_eval.get("last_row"), dict)
                else None,
                "conclusion": "exact_current_depth_practical_but_slower_than_owner_run_not_structurally_sublinear",
                "artifact": str(args.active_internal_train_eval_json),
            },
            "full_tape_rgb_train_eval_scope": {
                "status": full_tape_train_eval.get("status"),
                "total_step_scale_first_to_last": full_tape_train_eval.get("total_step_scale_first_to_last"),
                "full_tape_16f_total_ms": full_tape_train_eval.get("last_row", {}).get("total_ms")
                if isinstance(full_tape_train_eval.get("last_row"), dict)
                else None,
                "train_selected_tape_storage_vs_full_16f": full_tape_train_eval.get("last_row", {}).get(
                    "train_selected_tape_storage_vs_full"
                )
                if isinstance(full_tape_train_eval.get("last_row"), dict)
                else None,
                "conclusion": "exact_density_independent_replay_baseline_not_compact_or_fastest",
                "artifact": str(args.full_tape_train_eval_json),
            },
            "endpoint_run_rgb_train_eval_scope": {
                "status": endpoint_run_train_eval.get("status"),
                "total_step_scale_first_to_last": endpoint_run_train_eval.get("total_step_scale_first_to_last"),
                "endpoint_run_16f_total_ms": endpoint_run_train_eval.get("last_row", {}).get("total_ms")
                if isinstance(endpoint_run_train_eval.get("last_row"), dict)
                else None,
                "train_selected_tape_storage_vs_full_16f": endpoint_run_train_eval.get("last_row", {}).get(
                    "train_selected_tape_storage_vs_full"
                )
                if isinstance(endpoint_run_train_eval.get("last_row"), dict)
                else None,
                "conclusion": "compact_density_independent_semantic_change_practical_not_star_structural",
                "artifact": str(args.endpoint_run_train_eval_json),
            },
            "framegroup_i16x4_nonpromotion": {
                "status": "failed_as_promotion" if framegroup_i16x4_compare.get("available") is True else "missing",
                "i16x4_total_scale_first_to_last": framegroup_i16x4_compare.get(
                    "i16x4_total_scale_first_to_last"
                ),
                "i16x4_backward_scale_first_to_last": framegroup_i16x4_compare.get(
                    "i16x4_backward_scale_first_to_last"
                ),
                "max_i16x4_over_i16x3_total_mean_ratio": framegroup_i16x4_compare.get(
                    "max_i16x4_over_i16x3_total_mean_ratio"
                ),
                "max_i16x4_over_i16x3_storage_ratio": framegroup_i16x4_compare.get(
                    "max_i16x4_over_i16x3_storage_ratio"
                ),
                "conclusion": "correctness_green_not_promoted_due_to_repeated_frame_timing_cliff",
                "artifact": framegroup_i16x4_compare.get("artifact"),
            },
            "framegroup_i16x4_prewarm_nonpromotion": {
                "status": "failed_as_promotion"
                if framegroup_i16x4_prewarm_compare.get("available") is True
                else "missing",
                "i16x4_total_scale_first_to_last": framegroup_i16x4_prewarm_compare.get(
                    "i16x4_total_scale_first_to_last"
                ),
                "i16x4_backward_scale_first_to_last": framegroup_i16x4_prewarm_compare.get(
                    "i16x4_backward_scale_first_to_last"
                ),
                "max_i16x4_over_i16x3_total_mean_ratio": framegroup_i16x4_prewarm_compare.get(
                    "max_i16x4_over_i16x3_total_mean_ratio"
                ),
                "max_i16x4_over_i16x3_backward_mean_ratio": framegroup_i16x4_prewarm_compare.get(
                    "max_i16x4_over_i16x3_backward_mean_ratio"
                ),
                "max_i16x4_over_i16x3_storage_ratio": framegroup_i16x4_prewarm_compare.get(
                    "max_i16x4_over_i16x3_storage_ratio"
                ),
                "conclusion": "sublinear_but_not_promoted_due_to_32f_i16x3_ratio_loss",
                "artifact": framegroup_i16x4_prewarm_compare.get("artifact"),
            },
            "framegroup_packed_prewarm_candidate": {
                "status": "candidate" if framegroup_packed_prewarm_compare.get("available") is True else "missing",
                "packed_total_scale_first_to_last": framegroup_packed_prewarm_compare.get(
                    "packed_total_scale_first_to_last"
                ),
                "packed_backward_scale_first_to_last": framegroup_packed_prewarm_compare.get(
                    "packed_backward_scale_first_to_last"
                ),
                "max_packed_over_i16x3_total_mean_ratio": framegroup_packed_prewarm_compare.get(
                    "max_packed_over_i16x3_total_mean_ratio"
                ),
                "max_packed_over_i16x3_backward_mean_ratio": framegroup_packed_prewarm_compare.get(
                    "max_packed_over_i16x3_backward_mean_ratio"
                ),
                "max_packed_over_i16x3_storage_ratio": framegroup_packed_prewarm_compare.get(
                    "max_packed_over_i16x3_storage_ratio"
                ),
                "conclusion": "candidate_only_until_broader_frame_scale_guard",
                "artifact": framegroup_packed_prewarm_compare.get("artifact"),
            },
            "framegroup_packed_broad_nonpromotion": {
                "status": "failed_as_broad_promotion"
                if framegroup_packed_broad_compare.get("available") is True
                else "missing",
                "packed_total_scale_first_to_last": framegroup_packed_broad_compare.get(
                    "packed_total_scale_first_to_last"
                ),
                "packed_backward_scale_first_to_last": framegroup_packed_broad_compare.get(
                    "packed_backward_scale_first_to_last"
                ),
                "max_packed_over_i16x3_total_mean_ratio": framegroup_packed_broad_compare.get(
                    "max_packed_over_i16x3_total_mean_ratio"
                ),
                "max_packed_over_i16x3_backward_mean_ratio": framegroup_packed_broad_compare.get(
                    "max_packed_over_i16x3_backward_mean_ratio"
                ),
                "max_packed_over_i16x3_storage_ratio": framegroup_packed_broad_compare.get(
                    "max_packed_over_i16x3_storage_ratio"
                ),
                "conclusion": "broad_64_128_guard_rejects_default_promotion_due_to_128f_loss",
                "artifact": framegroup_packed_broad_compare.get("artifact"),
            },
        },
        "source_artifacts": {
            "verifier": str(args.verifier_json),
            "depth_order": str(args.depth_order_json),
            "ownerupdate": str(args.ownerupdate_json),
            "star_speed_reference": str(args.star_speed_json),
            "segment_tape": str(args.segment_tape_json),
            "topology_sharing": str(args.topology_sharing_json),
            "delta_tape": str(args.delta_tape_json),
            "boundary_delta_tape": str(args.boundary_delta_tape_json),
            "record_delta_tape": str(args.record_delta_tape_json),
            "owner_run_tape": str(args.owner_run_tape_json),
            "owner_run_boundary_tape": str(args.owner_run_boundary_tape_json),
            "owner_run_internal_tape": str(args.owner_run_internal_tape_json),
            "endpoint_run_tape": str(args.endpoint_run_tape_json),
            "endpoint_record_delta_tape": str(args.endpoint_record_delta_tape_json),
            "endpoint_record_delta_replay": str(args.endpoint_record_delta_replay_json),
            "endpoint_record_edit_replay": str(args.endpoint_record_edit_replay_json),
            "endpoint_record_edit_rgb_only_replay": str(args.endpoint_record_edit_rgb_only_replay_json),
            "endpoint_record_edit_trackloop_replay": str(args.endpoint_record_edit_trackloop_replay_json),
            "endpoint_record_edit_block4_replay": str(args.endpoint_record_edit_block4_replay_json),
            "endpoint_record_edit_block_coeff_replay": str(args.endpoint_record_edit_block_coeff_replay_json),
            "endpoint_record_edit_block_coeff_sweep": str(args.endpoint_record_edit_block_coeff_sweep_json),
            "endpoint_record_edit_block_coeff_train_eval": str(args.endpoint_record_edit_block_coeff_train_eval_json),
            "owner_run_train_eval": str(args.owner_run_train_eval_json),
            "active_internal_train_eval": str(args.active_internal_train_eval_json),
            "full_tape_train_eval": str(args.full_tape_train_eval_json),
            "endpoint_run_train_eval": str(args.endpoint_run_train_eval_json),
            "endpoint_record_edit_train_eval": str(args.endpoint_record_edit_train_eval_json),
            "endpoint_record_edit_block4_train_eval": str(args.endpoint_record_edit_block4_train_eval_json),
            "endpoint_record_edit_paired_train_eval": str(args.endpoint_record_edit_paired_train_eval_json),
            "endpoint_record_edit_block4_paired_train_eval": str(
                args.endpoint_record_edit_block4_paired_train_eval_json
            ),
            "endpoint_record_edit_block_coeff_paired_train_eval": str(
                args.endpoint_record_edit_block_coeff_paired_train_eval_json
            ),
            "endpoint_record_edit_block_coeff_repeat20_16f": str(
                args.endpoint_record_edit_block_coeff_repeat20_16f_json
            ),
            "endpoint_record_edit_block_coeff_repeat20_2_4_8_16": str(
                args.endpoint_record_edit_block_coeff_repeat20_2_4_8_16_json
            ),
            "endpoint_record_edit_block_coeff16_manual_vjp_paired_train_eval": str(
                args.endpoint_record_edit_block_coeff16_manual_vjp_paired_train_eval_json
            ),
            "endpoint_record_edit_block_coeff16_storagefix_smoke": str(
                args.endpoint_record_edit_block_coeff16_storagefix_smoke_json
            ),
            "endpoint_record_edit_rgb_only_paired_train_eval": str(
                args.endpoint_record_edit_rgb_only_paired_train_eval_json
            ),
            "endpoint_record_edit_manual_vjp_paired_train_eval": str(
                args.endpoint_record_edit_manual_vjp_paired_train_eval_json
            ),
            "segment_tape_autograd_smoke": str(args.segment_tape_autograd_smoke_json),
            "framegroup_autograd_smoke": str(args.framegroup_autograd_smoke_json),
            "framegroup_autograd_speedscale": str(args.framegroup_autograd_speedscale_json),
            "framegroup_i16x4_compare": framegroup_i16x4_compare.get("artifact"),
            "framegroup_i16x4_prewarm_compare": str(args.framegroup_i16x4_prewarm_compare_json),
            "framegroup_packed_prewarm_compare": str(args.framegroup_packed_prewarm_compare_json),
            "framegroup_packed_broad_compare": str(args.framegroup_packed_broad_compare_json),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize current World Foam fused slab mixed results.")
    parser.add_argument("--verifier-json", type=Path, default=DEFAULT_VERIFIER)
    parser.add_argument("--depth-order-json", type=Path, default=DEFAULT_DEPTH_ORDER)
    parser.add_argument("--ownerupdate-json", type=Path, default=DEFAULT_OWNERUPDATE)
    parser.add_argument("--star-speed-json", type=Path, default=DEFAULT_STAR_SPEED)
    parser.add_argument("--segment-tape-json", type=Path, default=DEFAULT_SEGMENT_TAPE)
    parser.add_argument("--topology-sharing-json", type=Path, default=DEFAULT_TOPOLOGY_SHARING)
    parser.add_argument("--delta-tape-json", type=Path, default=DEFAULT_DELTA_TAPE)
    parser.add_argument("--boundary-delta-tape-json", type=Path, default=DEFAULT_BOUNDARY_DELTA_TAPE)
    parser.add_argument("--record-delta-tape-json", type=Path, default=DEFAULT_RECORD_DELTA_TAPE)
    parser.add_argument("--owner-run-tape-json", type=Path, default=DEFAULT_OWNER_RUN_TAPE)
    parser.add_argument("--owner-run-boundary-tape-json", type=Path, default=DEFAULT_OWNER_RUN_BOUNDARY_TAPE)
    parser.add_argument("--owner-run-internal-tape-json", type=Path, default=DEFAULT_OWNER_RUN_INTERNAL_TAPE)
    parser.add_argument("--endpoint-run-tape-json", type=Path, default=DEFAULT_ENDPOINT_RUN_TAPE)
    parser.add_argument("--endpoint-record-delta-tape-json", type=Path, default=DEFAULT_ENDPOINT_RECORD_DELTA_TAPE)
    parser.add_argument("--endpoint-record-delta-replay-json", type=Path, default=DEFAULT_ENDPOINT_RECORD_DELTA_REPLAY)
    parser.add_argument("--endpoint-record-edit-replay-json", type=Path, default=DEFAULT_ENDPOINT_RECORD_EDIT_REPLAY)
    parser.add_argument(
        "--endpoint-record-edit-rgb-only-replay-json",
        type=Path,
        default=DEFAULT_ENDPOINT_RECORD_EDIT_RGB_ONLY_REPLAY,
    )
    parser.add_argument(
        "--endpoint-record-edit-trackloop-replay-json",
        type=Path,
        default=DEFAULT_ENDPOINT_RECORD_EDIT_TRACKLOOP_REPLAY,
    )
    parser.add_argument(
        "--endpoint-record-edit-block4-replay-json",
        type=Path,
        default=DEFAULT_ENDPOINT_RECORD_EDIT_BLOCK4_REPLAY,
    )
    parser.add_argument(
        "--endpoint-record-edit-block-coeff-replay-json",
        type=Path,
        default=DEFAULT_ENDPOINT_RECORD_EDIT_BLOCK_COEFF_REPLAY,
    )
    parser.add_argument(
        "--endpoint-record-edit-block-coeff-sweep-json",
        type=Path,
        default=DEFAULT_ENDPOINT_RECORD_EDIT_BLOCK_COEFF_SWEEP,
    )
    parser.add_argument(
        "--endpoint-record-edit-block-coeff-train-eval-json",
        type=Path,
        default=DEFAULT_ENDPOINT_RECORD_EDIT_BLOCK_COEFF_TRAIN_EVAL,
    )
    parser.add_argument("--owner-run-train-eval-json", type=Path, default=DEFAULT_OWNER_RUN_TRAIN_EVAL)
    parser.add_argument("--active-internal-train-eval-json", type=Path, default=DEFAULT_ACTIVE_INTERNAL_TRAIN_EVAL)
    parser.add_argument("--full-tape-train-eval-json", type=Path, default=DEFAULT_FULL_TAPE_TRAIN_EVAL)
    parser.add_argument("--endpoint-run-train-eval-json", type=Path, default=DEFAULT_ENDPOINT_RUN_TRAIN_EVAL)
    parser.add_argument(
        "--endpoint-record-edit-train-eval-json",
        type=Path,
        default=DEFAULT_ENDPOINT_RECORD_EDIT_TRAIN_EVAL,
    )
    parser.add_argument(
        "--endpoint-record-edit-block4-train-eval-json",
        type=Path,
        default=DEFAULT_ENDPOINT_RECORD_EDIT_BLOCK4_TRAIN_EVAL,
    )
    parser.add_argument(
        "--endpoint-record-edit-paired-train-eval-json",
        type=Path,
        default=DEFAULT_ENDPOINT_RECORD_EDIT_PAIRED_TRAIN_EVAL,
    )
    parser.add_argument(
        "--endpoint-record-edit-block4-paired-train-eval-json",
        type=Path,
        default=DEFAULT_ENDPOINT_RECORD_EDIT_BLOCK4_PAIRED_TRAIN_EVAL,
    )
    parser.add_argument(
        "--endpoint-record-edit-block-coeff-paired-train-eval-json",
        type=Path,
        default=DEFAULT_ENDPOINT_RECORD_EDIT_BLOCK_COEFF_PAIRED_TRAIN_EVAL,
    )
    parser.add_argument(
        "--endpoint-record-edit-block-coeff-repeat20-16f-json",
        type=Path,
        default=DEFAULT_ENDPOINT_RECORD_EDIT_BLOCK_COEFF_REPEAT20_16F,
    )
    parser.add_argument(
        "--endpoint-record-edit-block-coeff-repeat20-2-4-8-16-json",
        type=Path,
        default=DEFAULT_ENDPOINT_RECORD_EDIT_BLOCK_COEFF_REPEAT20_2_4_8_16,
    )
    parser.add_argument(
        "--endpoint-record-edit-block-coeff16-manual-vjp-paired-train-eval-json",
        type=Path,
        default=DEFAULT_ENDPOINT_RECORD_EDIT_BLOCK_COEFF16_MANUAL_VJP_PAIRED_TRAIN_EVAL,
    )
    parser.add_argument(
        "--endpoint-record-edit-block-coeff16-storagefix-smoke-json",
        type=Path,
        default=DEFAULT_ENDPOINT_RECORD_EDIT_BLOCK_COEFF16_STORAGEFIX_SMOKE,
    )
    parser.add_argument(
        "--endpoint-record-edit-rgb-only-paired-train-eval-json",
        type=Path,
        default=DEFAULT_ENDPOINT_RECORD_EDIT_RGB_ONLY_PAIRED_TRAIN_EVAL,
    )
    parser.add_argument(
        "--endpoint-record-edit-manual-vjp-paired-train-eval-json",
        type=Path,
        default=DEFAULT_ENDPOINT_RECORD_EDIT_MANUAL_VJP_PAIRED_TRAIN_EVAL,
    )
    parser.add_argument("--segment-tape-autograd-smoke-json", type=Path, default=DEFAULT_SEGMENT_TAPE_AUTOGRAD_SMOKE)
    parser.add_argument("--framegroup-autograd-smoke-json", type=Path, default=DEFAULT_FRAMEGROUP_AUTOGRAD_SMOKE)
    parser.add_argument(
        "--framegroup-autograd-speedscale-json",
        type=Path,
        default=DEFAULT_FRAMEGROUP_AUTOGRAD_SPEEDSCALE,
    )
    parser.add_argument(
        "--framegroup-i16x4-prewarm-compare-json",
        type=Path,
        default=DEFAULT_FRAMEGROUP_I16X4_PREWARM_COMPARE,
    )
    parser.add_argument(
        "--framegroup-packed-prewarm-compare-json",
        type=Path,
        default=DEFAULT_FRAMEGROUP_PACKED_PREWARM_COMPARE,
    )
    parser.add_argument(
        "--framegroup-packed-broad-compare-json",
        type=Path,
        default=DEFAULT_FRAMEGROUP_PACKED_BROAD_COMPARE,
    )
    parser.add_argument("--out-json", type=Path, default=RESULTS_DIR / "2026-05-15_fused_slab_mixed_status_summary.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = summarize(args)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    if payload["status"] != "ok_current_shader_gate_with_structural_gap":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
