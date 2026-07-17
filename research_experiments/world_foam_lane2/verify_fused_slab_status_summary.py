#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "research_experiments" / "world_foam_lane2" / "results"
DEFAULT_SUMMARY = RESULTS_DIR / "2026-05-15_fused_slab_mixed_status_summary.json"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _finite_positive(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0


def _finite_nonnegative(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= 0.0


def _verify_framegroup_objective_adapter(adapter: Any, *, prefix: str, failures: list[str]) -> None:
    if not isinstance(adapter, dict):
        failures.append(f"{prefix} missing WorldFoamFrozenRGBMSEObjective adapter metadata")
        return
    expected = {
        "name": "WorldFoamFrozenRGBMSEObjective",
        "module": "objective.world_foam_frozen_rgb_mse",
        "construction_scope": "once_per_frame_count_run",
        "loss_call_scope": "per_optimizer_step",
        "backend_loss_fn": "promoted_framegroup16_loss_fn",
        "tape_mode": "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse",
        "renderer_backend_claim": False,
        "full_trainer_claim": False,
        "full_geometry_gradient_claim": False,
        "quality_claim": False,
        "supports_rgb_mse_only": True,
        "supports_background_composition": False,
        "supports_colorizer": False,
        "supports_vjepa_feature_loss": False,
    }
    for key, expected_value in expected.items():
        if adapter.get(key) != expected_value:
            failures.append(f"{prefix} adapter {key} must be {expected_value!r}, got {adapter.get(key)!r}")


def verify(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_json(args.summary_json)
    failures: list[str] = []

    if payload.get("status") != "ok_current_shader_gate_with_structural_gap":
        failures.append(f"unexpected status {payload.get('status')!r}")
    if payload.get("completion_claim") is not False:
        failures.append("completion_claim must remain false for this status summary")
    if payload.get("star_uvt_competitive_claim") is not False:
        failures.append("star_uvt_competitive_claim must remain false without matched quality/capacity evidence")

    checklist = payload.get("checklist")
    if not isinstance(checklist, dict):
        failures.append("missing checklist")
    else:
        bad = sorted(key for key, value in checklist.items() if value is not True)
        if bad:
            failures.append(f"summary checklist has non-true items: {bad}")

    open_items = payload.get("open_items_before_completion")
    if not isinstance(open_items, list) or len(open_items) < 2:
        failures.append("open_items_before_completion must list remaining work")
    else:
        open_text = " ".join(str(item) for item in open_items)
        if not any(phrase in open_text for phrase in ("training path", "training integration", "main trainer")):
            failures.append("open items must keep training integration explicit")
        if "endpoint-record delta" not in open_text and "better-than-per-frame" not in open_text:
            failures.append("open items must keep endpoint-delta structural replay work explicit")
        if (
            "density-independent depth" not in open_text
            and "all internal cuts" not in open_text
            and "continuous-absorption endpoint depth" not in open_text
        ):
            failures.append("open items must keep compact density-independent depth replay gap explicit")
        if "compact exact replay" not in open_text and "owner+boundary records" not in open_text:
            failures.append("open items must keep compact exact record replay gap explicit")

    winner = payload.get("winner")
    if not isinstance(winner, dict):
        failures.append("missing winner")
    else:
        if winner.get("mode") != args.expected_winner:
            failures.append(f"winner mode {winner.get('mode')!r} did not match {args.expected_winner!r}")
        for key in ("total_2f_ms", "total_16f_ms", "total_scale_2_to_16", "render_scale_2_to_16", "backward_scale_2_to_16"):
            if not _finite_positive(winner.get(key)):
                failures.append(f"winner.{key} is not positive finite")
        if _finite_positive(winner.get("total_scale_2_to_16")) and float(winner["total_scale_2_to_16"]) >= 8.0:
            failures.append("winner total scale is not sublinear versus 2f->16f frame-count scale")

    psnr = payload.get("psnr_spread_across_modes_by_frame")
    max_spread = psnr.get("max_spread") if isinstance(psnr, dict) else None
    if not isinstance(max_spread, (int, float)) or not math.isfinite(float(max_spread)):
        failures.append("missing finite max matched-frame PSNR spread")
    elif float(max_spread) > args.max_psnr_spread:
        failures.append(f"max matched-frame PSNR spread {max_spread} exceeds {args.max_psnr_spread}")

    negatives = payload.get("negative_variants")
    if not isinstance(negatives, dict):
        failures.append("missing negative_variants")
    else:
        for name in ("ownerupdate", "ordered_append"):
            variant = negatives.get(name)
            if not isinstance(variant, dict) or variant.get("conclusion") != "rejected":
                failures.append(f"{name} negative variant must be recorded as rejected")

    star = payload.get("star_uvt_speed_reference")
    if not isinstance(star, dict) or star.get("available") is not True:
        failures.append("missing available STAR speed reference")
    else:
        by_frame = star.get("by_frame")
        if not isinstance(by_frame, dict):
            failures.append("missing STAR by_frame block")
        else:
            for frame_key in ("2", "4", "8", "16"):
                row = by_frame.get(frame_key)
                if not isinstance(row, dict):
                    failures.append(f"missing STAR {frame_key}f timing row")
                    continue
                if row.get("steps") != 20 or row.get("warmup_steps") != 5:
                    failures.append(f"STAR {frame_key}f timing should use the 20-step/5-warmup cadence")
        scaling = star.get("scaling")
        if not isinstance(scaling, dict):
            failures.append("missing STAR scaling block")
        else:
            frame_scale = scaling.get("frame_scale_first_to_last")
            step_scale = scaling.get("mean_step_scale_first_to_last")
            render_scale = scaling.get("mean_render_scale_first_to_last")
            for key, value in (
                ("frame_scale_first_to_last", frame_scale),
                ("mean_step_scale_first_to_last", step_scale),
                ("mean_render_scale_first_to_last", render_scale),
            ):
                if not _finite_positive(value):
                    failures.append(f"STAR scaling {key} is not positive finite")
            if scaling.get("step_runtime_sublinear_vs_frames") is not True:
                failures.append("STAR speed reference must preserve measured sublinear step-runtime scaling")
            if _finite_positive(frame_scale) and _finite_positive(step_scale) and float(step_scale) >= float(frame_scale):
                failures.append("STAR mean-step scaling is not sublinear versus frame count")
            note = str(scaling.get("scope_note", ""))
            if "not a matched quality/capacity comparison" not in note:
                failures.append("STAR scaling note must keep quality/capacity scope explicit")
        comparison = star.get("comparison_to_current_world_foam")
        if not isinstance(comparison, dict):
            failures.append("missing STAR comparison block")
        else:
            note = str(comparison.get("scope_note", ""))
            if "not a matched quality/capacity comparison" not in note:
                failures.append("STAR speed reference must explicitly say it is not matched quality/capacity")
            for key in ("star_uvt_16f_mean_step_ms", "world_foam_16f_total_ms", "world_foam_to_star_16f_step_ratio"):
                if not _finite_positive(comparison.get(key)):
                    failures.append(f"STAR comparison {key} is not positive finite")
        block_coeff_comparison = star.get("comparison_to_block_coeff_sidecar")
        if not isinstance(block_coeff_comparison, dict):
            failures.append("missing STAR to block-coeff sidecar comparison block")
        else:
            note = str(block_coeff_comparison.get("scope_note", ""))
            if "not a matched quality/capacity comparison" not in note or "does not prove STAR-UVT competitiveness" not in note:
                failures.append("STAR block-coeff sidecar comparison must keep quality/capacity scope explicit")
            for key in ("star_uvt_16f_mean_step_ms", "block_coeff_16f_total_ms", "block_coeff_to_star_16f_step_ratio"):
                if not _finite_positive(block_coeff_comparison.get(key)):
                    failures.append(f"STAR block-coeff comparison {key} is not positive finite")

    framegroup_lossreduce = payload.get("framegroup16_lossreduce_render32")
    if not isinstance(framegroup_lossreduce, dict) or framegroup_lossreduce.get("available") is not True:
        failures.append("missing render32 framegroup16 loss-reduction guardrail")
    else:
        if framegroup_lossreduce.get("completion_claim") is not False:
            failures.append("framegroup16 loss-reduction guardrail must not claim completion")
        if framegroup_lossreduce.get("full_trainer_claim") is not False:
            failures.append("framegroup16 loss-reduction guardrail must not claim full trainer coverage")
        if framegroup_lossreduce.get("quality_claim") is not False:
            failures.append("framegroup16 loss-reduction guardrail must not claim quality/capacity parity")
        frame_counts = framegroup_lossreduce.get("frame_counts")
        if frame_counts != [16, 32, 64, 128]:
            failures.append("framegroup16 loss-reduction frame counts must remain 16/32/64/128")
        total_scale = framegroup_lossreduce.get("total_scale_first_to_last")
        backward_scale = framegroup_lossreduce.get("backward_scale_first_to_last")
        storage_scale = framegroup_lossreduce.get("storage_scale_first_to_last")
        mixed_128_total_max = framegroup_lossreduce.get("mixed_128_total_max_ms")
        mixed_128_backward_max = framegroup_lossreduce.get("mixed_128_backward_max_ms")
        for key, value in (
            ("total_scale_first_to_last", total_scale),
            ("backward_scale_first_to_last", backward_scale),
            ("storage_scale_first_to_last", storage_scale),
            ("mixed_128_total_max_ms", mixed_128_total_max),
            ("mixed_128_backward_max_ms", mixed_128_backward_max),
        ):
            if not _finite_positive(value):
                failures.append(f"framegroup16 loss-reduction {key} is not positive finite")
        if _finite_positive(total_scale) and float(total_scale) > 1.5:
            failures.append("framegroup16 loss-reduction total scale exceeds guarded threshold")
        if _finite_positive(backward_scale) and float(backward_scale) > 1.65:
            failures.append("framegroup16 loss-reduction backward scale exceeds guarded threshold")
        if _finite_positive(storage_scale) and float(storage_scale) > 1.10:
            failures.append("framegroup16 loss-reduction storage scale exceeds guarded threshold")
        if _finite_positive(mixed_128_total_max) and float(mixed_128_total_max) > 7.5:
            failures.append("framegroup16 loss-reduction mixed 128f total max exceeds outlier guard")
        confirm = framegroup_lossreduce.get("confirm_128only")
        if not isinstance(confirm, dict):
            failures.append("framegroup16 loss-reduction missing 128-only confirmation")
        else:
            total_median = confirm.get("total_median_ms")
            total_max = confirm.get("total_max_ms")
            backward_median = confirm.get("backward_median_ms")
            for key, value in (
                ("confirm total_median_ms", total_median),
                ("confirm total_max_ms", total_max),
                ("confirm backward_median_ms", backward_median),
            ):
                if not _finite_positive(value):
                    failures.append(f"framegroup16 loss-reduction {key} is not positive finite")
            if _finite_positive(total_median) and float(total_median) > 4.5:
                failures.append("framegroup16 loss-reduction 128-only total median exceeds guard")
            if _finite_positive(total_max) and float(total_max) > 8.5:
                failures.append("framegroup16 loss-reduction 128-only total max exceeds guard")
            if _finite_positive(backward_median) and float(backward_median) > 3.75:
                failures.append("framegroup16 loss-reduction 128-only backward median exceeds guard")
        conclusion = str(framegroup_lossreduce.get("conclusion", ""))
        if "sublinear frame scaling" not in conclusion or "not a full-trainer" not in conclusion:
            failures.append("framegroup16 loss-reduction conclusion must keep scaling win and scope boundary explicit")

    framegroup_compare = payload.get("framegroup16_compare_render32_speedscale")
    if not isinstance(framegroup_compare, dict) or framegroup_compare.get("available") is not True:
        failures.append("missing render32 framegroup16 paired compare speedscale guardrail")
    else:
        if framegroup_compare.get("completion_claim") is not False:
            failures.append("framegroup16 compare guardrail must not claim completion")
        if framegroup_compare.get("full_trainer_claim") is not False:
            failures.append("framegroup16 compare guardrail must not claim full trainer coverage")
        if framegroup_compare.get("quality_claim") is not False:
            failures.append("framegroup16 compare guardrail must not claim quality/capacity parity")
        if framegroup_compare.get("star_uvt_competitive_claim") is not False:
            failures.append("framegroup16 compare guardrail must not claim STAR-UVT competitiveness")
        if framegroup_compare.get("frame_counts") != [16, 32, 64, 128]:
            failures.append("framegroup16 compare frame counts must remain 16/32/64/128")
        if framegroup_compare.get("render_size") != 32:
            failures.append("framegroup16 compare render_size must remain 32")
        if framegroup_compare.get("site_count") != 12:
            failures.append("framegroup16 compare site_count must remain 12")
        if framegroup_compare.get("loaded_frame_count") != 16:
            failures.append("framegroup16 compare loaded_frame_count must remain 16")
        if framegroup_compare.get("real_loaded_frame_counts") != [16]:
            failures.append("framegroup16 compare real-loaded rows must remain only 16f")
        if framegroup_compare.get("repeated_frame_counts") != [32, 64, 128]:
            failures.append("framegroup16 compare repeated-fixture rows must remain 32/64/128f")
        repeat_scope_by_frame = framegroup_compare.get("repeat_scope_by_frame")
        if not isinstance(repeat_scope_by_frame, dict):
            failures.append("framegroup16 compare missing repeat_scope_by_frame")
        else:
            if repeat_scope_by_frame.get("16") != "real loaded frame count":
                failures.append("framegroup16 compare 16f row must remain real-loaded")
            for frame in ("32", "64", "128"):
                if "synthetic repeated-fixture speed-scaling smoke" not in str(repeat_scope_by_frame.get(frame, "")):
                    failures.append(f"framegroup16 compare {frame}f row must keep repeated-fixture scope")
        for key, limit in (
            ("total_ratio_16f", 0.75),
            ("backward_ratio_16f", 0.95),
            ("total_scale_first_to_last", 3.25),
            ("backward_scale_first_to_last", 3.75),
            ("storage_scale_first_to_last", 1.10),
            ("framegroup_storage_vs_full_16f", 0.15),
        ):
            value = framegroup_compare.get(key)
            if not _finite_positive(value):
                failures.append(f"framegroup16 compare {key} is not positive finite")
            elif float(value) > limit:
                failures.append(f"framegroup16 compare {key} exceeds guarded threshold")
        ratios_by_frame = framegroup_compare.get("ratios_by_frame")
        if not isinstance(ratios_by_frame, dict):
            failures.append("framegroup16 compare missing ratios_by_frame")
        else:
            for frame in ("16", "32", "64", "128"):
                ratios = ratios_by_frame.get(frame)
                if not isinstance(ratios, dict):
                    failures.append(f"framegroup16 compare missing ratios for {frame}f")
                    continue
                total_ratio = ratios.get("total")
                if not _finite_positive(total_ratio):
                    failures.append(f"framegroup16 compare {frame}f total ratio is not positive finite")
                elif float(total_ratio) > 0.75:
                    failures.append(f"framegroup16 compare {frame}f total ratio exceeds guard")
        psnr_delta_by_frame = framegroup_compare.get("psnr_delta_by_frame")
        if not isinstance(psnr_delta_by_frame, dict):
            failures.append("framegroup16 compare missing psnr_delta_by_frame")
        else:
            for frame in ("16", "32", "64", "128"):
                value = psnr_delta_by_frame.get(frame)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0:
                    failures.append(f"framegroup16 compare {frame}f PSNR delta is not finite")
                    continue
                limit = 0.001 if frame == "16" else 0.005
                if float(value) > limit:
                    failures.append(f"framegroup16 compare {frame}f PSNR delta exceeds guard")
        scope = str(framegroup_compare.get("scope", ""))
        if "not a stable benchmark" not in scope or "repeated loaded frames" not in scope:
            failures.append("framegroup16 compare scope must keep benchmark and repeated-frame caveats")
        conclusion = str(framegroup_compare.get("conclusion", ""))
        if (
            "faster than endpoint-run at every checked" not in conclusion
            or "only 16f is a real loaded row" not in conclusion
            or "not a stable benchmark" not in conclusion
        ):
            failures.append("framegroup16 compare conclusion must keep speed win and scope boundary explicit")

    framegroup_real32 = payload.get("framegroup16_real32_render32_compare")
    if not isinstance(framegroup_real32, dict) or framegroup_real32.get("available") is not True:
        failures.append("missing real-loaded render32 framegroup16 paired compare")
    else:
        if framegroup_real32.get("completion_claim") is not False:
            failures.append("real32 framegroup compare must not claim completion")
        if framegroup_real32.get("full_trainer_claim") is not False:
            failures.append("real32 framegroup compare must not claim full trainer coverage")
        if framegroup_real32.get("quality_claim") is not False:
            failures.append("real32 framegroup compare must not claim quality/capacity parity")
        if framegroup_real32.get("star_uvt_competitive_claim") is not False:
            failures.append("real32 framegroup compare must not claim STAR-UVT competitiveness")
        if framegroup_real32.get("real_frame_sublinear_claim") is not True:
            failures.append("real32 framegroup compare must preserve measured real-frame sublinear scaling")
        if framegroup_real32.get("total_sublinear_real_frames") is not True:
            failures.append("real32 framegroup compare must preserve total-sublinear win")
        if framegroup_real32.get("backward_sublinear_real_frames") is not True:
            failures.append("real32 framegroup compare must preserve backward-sublinear win")
        if framegroup_real32.get("frame_counts") != [16, 32]:
            failures.append("real32 framegroup compare frame counts must remain 16/32")
        if framegroup_real32.get("render_size") != 32:
            failures.append("real32 framegroup compare render_size must remain 32")
        if framegroup_real32.get("site_count") != 12:
            failures.append("real32 framegroup compare site_count must remain 12")
        if framegroup_real32.get("real_loaded_frame_counts") != [16, 32]:
            failures.append("real32 framegroup compare real-loaded rows must remain 16/32f")
        if framegroup_real32.get("repeated_frame_counts") != []:
            failures.append("real32 framegroup compare must not include repeated-fixture rows")
        repeat_scope_by_frame = framegroup_real32.get("repeat_scope_by_frame")
        if not isinstance(repeat_scope_by_frame, dict):
            failures.append("real32 framegroup compare missing repeat_scope_by_frame")
        else:
            for frame in ("16", "32"):
                if repeat_scope_by_frame.get(frame) != "real loaded frame count":
                    failures.append(f"real32 framegroup compare {frame}f row must stay real-loaded")
        ratios_by_frame = framegroup_real32.get("ratios_by_frame")
        if not isinstance(ratios_by_frame, dict):
            failures.append("real32 framegroup compare missing ratios_by_frame")
        else:
            for frame in ("16", "32"):
                ratios = ratios_by_frame.get(frame)
                if not isinstance(ratios, dict):
                    failures.append(f"real32 framegroup compare missing ratios for {frame}f")
                    continue
                total_ratio = ratios.get("total")
                backward_ratio = ratios.get("backward")
                if not _finite_positive(total_ratio):
                    failures.append(f"real32 framegroup compare {frame}f total ratio is not positive finite")
                elif float(total_ratio) > 0.75:
                    failures.append(f"real32 framegroup compare {frame}f total ratio exceeds guard")
                if not _finite_positive(backward_ratio):
                    failures.append(f"real32 framegroup compare {frame}f backward ratio is not positive finite")
                elif float(backward_ratio) > 0.95:
                    failures.append(f"real32 framegroup compare {frame}f backward ratio exceeds guard")
        psnr_delta_by_frame = framegroup_real32.get("psnr_delta_by_frame")
        if not isinstance(psnr_delta_by_frame, dict):
            failures.append("real32 framegroup compare missing psnr_delta_by_frame")
        else:
            for frame in ("16", "32"):
                value = psnr_delta_by_frame.get(frame)
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) < 0.0:
                    failures.append(f"real32 framegroup compare {frame}f PSNR delta is not finite")
                elif float(value) > 0.001:
                    failures.append(f"real32 framegroup compare {frame}f PSNR delta exceeds guard")
        storage_vs_endpoint = framegroup_real32.get("storage_vs_endpoint_by_frame")
        if not isinstance(storage_vs_endpoint, dict):
            failures.append("real32 framegroup compare missing storage_vs_endpoint_by_frame")
        else:
            for frame in ("16", "32"):
                value = storage_vs_endpoint.get(frame)
                if not _finite_positive(value):
                    failures.append(f"real32 framegroup compare {frame}f storage ratio is not positive finite")
                elif float(value) >= 1.0:
                    failures.append(f"real32 framegroup compare {frame}f storage must stay below endpoint-run")
        for key, limit in (
            ("total_scale_first_to_last", 2.25),
            ("backward_scale_first_to_last", 2.35),
            ("storage_scale_first_to_last", 1.10),
        ):
            value = framegroup_real32.get(key)
            if not _finite_positive(value):
                failures.append(f"real32 framegroup compare {key} is not positive finite")
            elif float(value) > limit:
                failures.append(f"real32 framegroup compare {key} exceeds guarded threshold")
        scope = str(framegroup_real32.get("scope", ""))
        if "not a stable benchmark" not in scope:
            failures.append("real32 framegroup compare scope must keep benchmark caveat")
        conclusion = str(framegroup_real32.get("conclusion", ""))
        if (
            "real-loaded" not in conclusion
            or "sublinear" not in conclusion
            or "full-trainer" not in conclusion
            or "STAR-UVT competitiveness" not in conclusion
        ):
            failures.append("real32 framegroup compare conclusion must keep sublinear result and scope caveats")

    framegroup_i16x4 = payload.get("framegroup16_i16x4_compare")
    if not isinstance(framegroup_i16x4, dict) or framegroup_i16x4.get("available") is not True:
        failures.append("missing i16x4 framegroup non-promotion guardrail")
    else:
        if framegroup_i16x4.get("completion_claim") is not False:
            failures.append("i16x4 framegroup guardrail must not claim completion")
        if framegroup_i16x4.get("full_trainer_claim") is not False:
            failures.append("i16x4 framegroup guardrail must not claim full trainer coverage")
        if framegroup_i16x4.get("quality_claim") is not False:
            failures.append("i16x4 framegroup guardrail must not claim quality/capacity parity")
        if framegroup_i16x4.get("star_uvt_competitive_claim") is not False:
            failures.append("i16x4 framegroup guardrail must not claim STAR-UVT competitiveness")
        if framegroup_i16x4.get("i16x4_speed_promotion_candidate") is not False:
            failures.append("i16x4 framegroup must remain a non-promotion candidate")
        if framegroup_i16x4.get("i16x4_total_sublinear_claim") is not False:
            failures.append("i16x4 framegroup total-sublinear claim must remain false")
        if framegroup_i16x4.get("i16x4_backward_sublinear_claim") is not False:
            failures.append("i16x4 framegroup backward-sublinear claim must remain false")
        if framegroup_i16x4.get("frame_counts") != [16, 32]:
            failures.append("i16x4 framegroup compare frame counts must remain 16/32")
        if framegroup_i16x4.get("repeat_loaded_frames") is not True:
            failures.append("i16x4 framegroup compare must remain marked as repeated loaded frames")
        mode_statuses = framegroup_i16x4.get("mode_statuses")
        i16x3_mode = "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse"
        i16x4_mode = "endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse"
        if not isinstance(mode_statuses, dict):
            failures.append("i16x4 framegroup compare missing mode_statuses")
        else:
            if mode_statuses.get(i16x3_mode) != "ok":
                failures.append("i16x4 framegroup i16x3 mode status must remain ok")
            if mode_statuses.get(i16x4_mode) != "failed":
                failures.append("i16x4 framegroup i16x4 mode status must stay failed until explicit promotion")
        frame_scale = framegroup_i16x4.get("frame_scale_first_to_last")
        total_scale = framegroup_i16x4.get("i16x4_total_scale_first_to_last")
        backward_scale = framegroup_i16x4.get("i16x4_backward_scale_first_to_last")
        for key, value in (
            ("frame_scale_first_to_last", frame_scale),
            ("i16x4_total_scale_first_to_last", total_scale),
            ("i16x4_backward_scale_first_to_last", backward_scale),
            ("max_i16x4_over_i16x3_total_mean_ratio", framegroup_i16x4.get("max_i16x4_over_i16x3_total_mean_ratio")),
            (
                "max_i16x4_over_i16x3_backward_mean_ratio",
                framegroup_i16x4.get("max_i16x4_over_i16x3_backward_mean_ratio"),
            ),
            ("max_i16x4_over_i16x3_storage_ratio", framegroup_i16x4.get("max_i16x4_over_i16x3_storage_ratio")),
        ):
            if not _finite_positive(value):
                failures.append(f"i16x4 framegroup {key} is not positive finite")
        if (
            _finite_positive(frame_scale)
            and _finite_positive(total_scale)
            and float(total_scale) <= float(frame_scale)
        ):
            failures.append("i16x4 framegroup total scale now looks sublinear; update promotion guard explicitly")
        if (
            _finite_positive(frame_scale)
            and _finite_positive(backward_scale)
            and float(backward_scale) <= float(frame_scale)
        ):
            failures.append("i16x4 framegroup backward scale now looks sublinear; update promotion guard explicitly")
        for key, limit in (
            ("max_i16x4_over_i16x3_total_mean_ratio", 1.05),
            ("max_i16x4_over_i16x3_backward_mean_ratio", 1.05),
            ("max_i16x4_over_i16x3_storage_ratio", 1.08),
        ):
            value = framegroup_i16x4.get(key)
            if _finite_positive(value) and float(value) > limit:
                failures.append(f"i16x4 framegroup {key} exceeds guard")
        max_psnr_delta = framegroup_i16x4.get("max_psnr_delta")
        if not _finite_nonnegative(max_psnr_delta):
            failures.append("i16x4 framegroup max_psnr_delta is not finite nonnegative")
        elif float(max_psnr_delta) > 1.0e-4:
            failures.append("i16x4 framegroup max_psnr_delta exceeds guard")
        ratios_by_frame = framegroup_i16x4.get("ratios_by_frame")
        if not isinstance(ratios_by_frame, dict):
            failures.append("i16x4 framegroup compare missing ratios_by_frame")
        else:
            for frame in ("16", "32"):
                ratios = ratios_by_frame.get(frame)
                if not isinstance(ratios, dict):
                    failures.append(f"i16x4 framegroup compare missing {frame}f ratios")
                    continue
                for key, limit in (
                    ("i16x4_over_i16x3_total_mean", 1.05),
                    ("i16x4_over_i16x3_backward_mean", 1.05),
                    ("i16x4_over_i16x3_storage", 1.08),
                ):
                    value = ratios.get(key)
                    if not _finite_positive(value):
                        failures.append(f"i16x4 framegroup {frame}f {key} is not positive finite")
                    elif float(value) > limit:
                        failures.append(f"i16x4 framegroup {frame}f {key} exceeds guard")
        scope = str(framegroup_i16x4.get("scope", ""))
        if "not a STAR-UVT competitiveness artifact" not in scope:
            failures.append("i16x4 framegroup scope must keep STAR-UVT caveat")
        conclusion = str(framegroup_i16x4.get("conclusion", ""))
        if "not promoted" not in conclusion or "not full-trainer" not in conclusion or "not a STAR-UVT" not in conclusion:
            failures.append("i16x4 framegroup conclusion must keep non-promotion and scope caveats")

    framegroup_i16x4_prewarm = payload.get("framegroup16_i16x4_prewarm_compare")
    if not isinstance(framegroup_i16x4_prewarm, dict) or framegroup_i16x4_prewarm.get("available") is not True:
        failures.append("missing i16x4 prewarm non-promotion compare")
    else:
        if framegroup_i16x4_prewarm.get("completion_claim") is not False:
            failures.append("i16x4 prewarm compare must not claim completion")
        if framegroup_i16x4_prewarm.get("full_trainer_claim") is not False:
            failures.append("i16x4 prewarm compare must not claim full trainer coverage")
        if framegroup_i16x4_prewarm.get("quality_claim") is not False:
            failures.append("i16x4 prewarm compare must not claim quality/capacity parity")
        if framegroup_i16x4_prewarm.get("star_uvt_competitive_claim") is not False:
            failures.append("i16x4 prewarm compare must not claim STAR-UVT competitiveness")
        if framegroup_i16x4_prewarm.get("i16x4_speed_promotion_candidate") is not False:
            failures.append("i16x4 prewarm compare must remain a non-promotion candidate")
        if framegroup_i16x4_prewarm.get("i16x4_total_sublinear_claim") is not True:
            failures.append("i16x4 prewarm compare must preserve total-sublinear evidence")
        if framegroup_i16x4_prewarm.get("i16x4_backward_sublinear_claim") is not True:
            failures.append("i16x4 prewarm compare must preserve backward-sublinear evidence")
        if framegroup_i16x4_prewarm.get("speed_rejected_by_ratio") is not True:
            failures.append("i16x4 prewarm compare must keep ratio-based non-promotion")
        if framegroup_i16x4_prewarm.get("frame_counts") != [16, 32]:
            failures.append("i16x4 prewarm compare frame counts must remain 16/32")
        if framegroup_i16x4_prewarm.get("prewarm_sweep") is not True:
            failures.append("i16x4 prewarm compare must keep prewarm_sweep=true")
        if framegroup_i16x4_prewarm.get("repeat_loaded_frames") is not True:
            failures.append("i16x4 prewarm compare must remain marked as repeated loaded frames")
        if framegroup_i16x4_prewarm.get("steps") != 5:
            failures.append("i16x4 prewarm compare steps must remain 5")
        if framegroup_i16x4_prewarm.get("warmup_steps") != 3:
            failures.append("i16x4 prewarm compare warmup_steps must remain 3")
        mode_statuses = framegroup_i16x4_prewarm.get("mode_statuses")
        i16x3_mode = "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse"
        i16x4_mode = "endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse"
        if not isinstance(mode_statuses, dict):
            failures.append("i16x4 prewarm compare missing mode_statuses")
        else:
            if mode_statuses.get(i16x3_mode) != "ok":
                failures.append("i16x4 prewarm compare i16x3 mode status must remain ok")
            if mode_statuses.get(i16x4_mode) != "ok":
                failures.append("i16x4 prewarm compare i16x4 mode status must remain ok")
        max_total = framegroup_i16x4_prewarm.get("max_i16x4_over_i16x3_total_mean_ratio")
        max_backward = framegroup_i16x4_prewarm.get("max_i16x4_over_i16x3_backward_mean_ratio")
        max_storage = framegroup_i16x4_prewarm.get("max_i16x4_over_i16x3_storage_ratio")
        for key, value in (
            ("max_i16x4_over_i16x3_total_mean_ratio", max_total),
            ("max_i16x4_over_i16x3_backward_mean_ratio", max_backward),
            ("max_i16x4_over_i16x3_storage_ratio", max_storage),
            ("i16x4_total_scale_first_to_last", framegroup_i16x4_prewarm.get("i16x4_total_scale_first_to_last")),
            (
                "i16x4_backward_scale_first_to_last",
                framegroup_i16x4_prewarm.get("i16x4_backward_scale_first_to_last"),
            ),
        ):
            if not _finite_positive(value):
                failures.append(f"i16x4 prewarm compare {key} is not positive finite")
        if not (
            (_finite_positive(max_total) and float(max_total) > 1.05)
            or (_finite_positive(max_backward) and float(max_backward) > 1.05)
        ):
            failures.append("i16x4 prewarm compare must preserve a total/backward ratio above promotion guard")
        if _finite_positive(max_storage) and float(max_storage) > 1.08:
            failures.append("i16x4 prewarm compare storage ratio exceeds guard")
        max_psnr_delta = framegroup_i16x4_prewarm.get("max_psnr_delta")
        if not _finite_nonnegative(max_psnr_delta):
            failures.append("i16x4 prewarm compare max_psnr_delta is not finite nonnegative")
        elif float(max_psnr_delta) > 1.0e-4:
            failures.append("i16x4 prewarm compare max_psnr_delta exceeds guard")
        ratios_by_frame = framegroup_i16x4_prewarm.get("ratios_by_frame")
        if not isinstance(ratios_by_frame, dict):
            failures.append("i16x4 prewarm compare missing ratios_by_frame")
        else:
            ratios32 = ratios_by_frame.get("32")
            if not isinstance(ratios32, dict):
                failures.append("i16x4 prewarm compare missing 32f ratios")
            else:
                total32 = ratios32.get("i16x4_over_i16x3_total_mean")
                backward32 = ratios32.get("i16x4_over_i16x3_backward_mean")
                if not _finite_positive(total32) or float(total32) <= 1.05:
                    failures.append("i16x4 prewarm compare 32f total ratio must stay above promotion guard")
                if not _finite_positive(backward32) or float(backward32) <= 1.05:
                    failures.append("i16x4 prewarm compare 32f backward ratio must stay above promotion guard")
        scope = str(framegroup_i16x4_prewarm.get("scope", ""))
        if "not a STAR-UVT competitiveness artifact" not in scope:
            failures.append("i16x4 prewarm compare scope must keep STAR-UVT caveat")
        conclusion = str(framegroup_i16x4_prewarm.get("conclusion", ""))
        if "not promoted" not in conclusion or "not full-trainer" not in conclusion or "STAR-UVT" not in conclusion:
            failures.append("i16x4 prewarm compare conclusion must keep non-promotion and scope caveats")

    framegroup_packed = payload.get("framegroup16_packed_prewarm_compare")
    if not isinstance(framegroup_packed, dict) or framegroup_packed.get("available") is not True:
        failures.append("missing packed prewarm framegroup candidate compare")
    else:
        if framegroup_packed.get("completion_claim") is not False:
            failures.append("packed prewarm compare must not claim completion")
        if framegroup_packed.get("full_trainer_claim") is not False:
            failures.append("packed prewarm compare must not claim full trainer coverage")
        if framegroup_packed.get("quality_claim") is not False:
            failures.append("packed prewarm compare must not claim quality/capacity parity")
        if framegroup_packed.get("star_uvt_competitive_claim") is not False:
            failures.append("packed prewarm compare must not claim STAR-UVT competitiveness")
        if framegroup_packed.get("packed_speed_promotion_candidate") is not True:
            failures.append("packed prewarm compare must preserve speed-candidate evidence")
        if framegroup_packed.get("packed_storage_below_i16x3") is not True:
            failures.append("packed prewarm compare must preserve storage-below-i16x3 evidence")
        if framegroup_packed.get("packed_total_sublinear_claim") is not True:
            failures.append("packed prewarm compare must preserve total-sublinear evidence")
        if framegroup_packed.get("packed_backward_sublinear_claim") is not True:
            failures.append("packed prewarm compare must preserve backward-sublinear evidence")
        if framegroup_packed.get("frame_counts") != [16, 32]:
            failures.append("packed prewarm compare frame counts must remain 16/32")
        if framegroup_packed.get("prewarm_sweep") is not True:
            failures.append("packed prewarm compare must keep prewarm_sweep=true")
        if framegroup_packed.get("repeat_loaded_frames") is not True:
            failures.append("packed prewarm compare must remain marked as repeated loaded frames")
        if framegroup_packed.get("steps") != 5:
            failures.append("packed prewarm compare steps must remain 5")
        if framegroup_packed.get("warmup_steps") != 3:
            failures.append("packed prewarm compare warmup_steps must remain 3")
        mode_statuses = framegroup_packed.get("mode_statuses")
        i16x3_mode = "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse"
        packed_mode = "endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse"
        if not isinstance(mode_statuses, dict):
            failures.append("packed prewarm compare missing mode_statuses")
        else:
            if mode_statuses.get(i16x3_mode) != "ok":
                failures.append("packed prewarm compare i16x3 mode status must remain ok")
            if mode_statuses.get(packed_mode) != "ok":
                failures.append("packed prewarm compare packed mode status must remain ok")
        max_total = framegroup_packed.get("max_packed_over_i16x3_total_mean_ratio")
        max_backward = framegroup_packed.get("max_packed_over_i16x3_backward_mean_ratio")
        max_storage = framegroup_packed.get("max_packed_over_i16x3_storage_ratio")
        for key, value in (
            ("max_packed_over_i16x3_total_mean_ratio", max_total),
            ("max_packed_over_i16x3_backward_mean_ratio", max_backward),
            ("max_packed_over_i16x3_storage_ratio", max_storage),
            ("packed_total_scale_first_to_last", framegroup_packed.get("packed_total_scale_first_to_last")),
            ("packed_backward_scale_first_to_last", framegroup_packed.get("packed_backward_scale_first_to_last")),
            ("packed_storage_scale_first_to_last", framegroup_packed.get("packed_storage_scale_first_to_last")),
        ):
            if not _finite_positive(value):
                failures.append(f"packed prewarm compare {key} is not positive finite")
        if _finite_positive(max_total) and float(max_total) > 0.85:
            failures.append("packed prewarm compare max total ratio exceeds candidate guard")
        if _finite_positive(max_backward) and float(max_backward) > 0.90:
            failures.append("packed prewarm compare max backward ratio exceeds candidate guard")
        if _finite_positive(max_storage) and float(max_storage) >= 1.0:
            failures.append("packed prewarm compare storage ratio must stay below i16x3")
        max_psnr_delta = framegroup_packed.get("max_psnr_delta")
        if not _finite_nonnegative(max_psnr_delta):
            failures.append("packed prewarm compare max_psnr_delta is not finite nonnegative")
        elif float(max_psnr_delta) > 1.0e-4:
            failures.append("packed prewarm compare max_psnr_delta exceeds guard")
        ratios_by_frame = framegroup_packed.get("ratios_by_frame")
        if not isinstance(ratios_by_frame, dict):
            failures.append("packed prewarm compare missing ratios_by_frame")
        else:
            for frame in ("16", "32"):
                ratios = ratios_by_frame.get(frame)
                if not isinstance(ratios, dict):
                    failures.append(f"packed prewarm compare missing {frame}f ratios")
                    continue
                for key, limit in (
                    ("packed_over_i16x3_total_mean", 0.85),
                    ("packed_over_i16x3_backward_mean", 0.90),
                ):
                    value = ratios.get(key)
                    if not _finite_positive(value):
                        failures.append(f"packed prewarm compare {frame}f {key} is not positive finite")
                    elif float(value) > limit:
                        failures.append(f"packed prewarm compare {frame}f {key} exceeds candidate guard")
                storage = ratios.get("packed_over_i16x3_storage")
                if not _finite_positive(storage):
                    failures.append(f"packed prewarm compare {frame}f storage ratio is not positive finite")
                elif float(storage) >= 1.0:
                    failures.append(f"packed prewarm compare {frame}f storage ratio must stay below i16x3")
        scope = str(framegroup_packed.get("scope", ""))
        if "not a STAR-UVT competitiveness artifact" not in scope:
            failures.append("packed prewarm compare scope must keep STAR-UVT caveat")
        conclusion = str(framegroup_packed.get("conclusion", ""))
        if "candidate" not in conclusion or "not full-trainer" not in conclusion or "STAR-UVT" not in conclusion:
            failures.append("packed prewarm compare conclusion must keep candidate and scope caveats")

    framegroup_packed_broad = payload.get("framegroup16_packed_broad_compare")
    if not isinstance(framegroup_packed_broad, dict) or framegroup_packed_broad.get("available") is not True:
        failures.append("missing packed broad non-promotion compare")
    else:
        if framegroup_packed_broad.get("completion_claim") is not False:
            failures.append("packed broad compare must not claim completion")
        if framegroup_packed_broad.get("full_trainer_claim") is not False:
            failures.append("packed broad compare must not claim full trainer coverage")
        if framegroup_packed_broad.get("quality_claim") is not False:
            failures.append("packed broad compare must not claim quality/capacity parity")
        if framegroup_packed_broad.get("star_uvt_competitive_claim") is not False:
            failures.append("packed broad compare must not claim STAR-UVT competitiveness")
        if framegroup_packed_broad.get("packed_speed_promotion_candidate") is not False:
            failures.append("packed broad compare must remain a non-promotion candidate")
        if framegroup_packed_broad.get("packed_storage_below_i16x3") is not True:
            failures.append("packed broad compare must preserve storage-below-i16x3 evidence")
        if framegroup_packed_broad.get("speed_rejected_by_128") is not True:
            failures.append("packed broad compare must keep 128f speed rejection")
        if framegroup_packed_broad.get("frame_counts") != [64, 128]:
            failures.append("packed broad compare frame counts must remain 64/128")
        if framegroup_packed_broad.get("prewarm_sweep") is not True:
            failures.append("packed broad compare must keep prewarm_sweep=true")
        if framegroup_packed_broad.get("interleave_modes") is not True:
            failures.append("packed broad compare must keep interleave_modes=true")
        if framegroup_packed_broad.get("repeat_loaded_frames") is not True:
            failures.append("packed broad compare must remain marked as repeated loaded frames")
        if framegroup_packed_broad.get("steps") != 3:
            failures.append("packed broad compare steps must remain 3")
        if framegroup_packed_broad.get("warmup_steps") != 1:
            failures.append("packed broad compare warmup_steps must remain 1")
        mode_statuses = framegroup_packed_broad.get("mode_statuses")
        i16x3_mode = "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse"
        packed_mode = "endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse"
        if not isinstance(mode_statuses, dict):
            failures.append("packed broad compare missing mode_statuses")
        else:
            if mode_statuses.get(i16x3_mode) != "ok":
                failures.append("packed broad compare i16x3 mode status must remain ok")
            if mode_statuses.get(packed_mode) != "ok":
                failures.append("packed broad compare packed mode status must remain ok")
        max_total = framegroup_packed_broad.get("max_packed_over_i16x3_total_mean_ratio")
        max_backward = framegroup_packed_broad.get("max_packed_over_i16x3_backward_mean_ratio")
        max_storage = framegroup_packed_broad.get("max_packed_over_i16x3_storage_ratio")
        for key, value in (
            ("max_packed_over_i16x3_total_mean_ratio", max_total),
            ("max_packed_over_i16x3_backward_mean_ratio", max_backward),
            ("max_packed_over_i16x3_storage_ratio", max_storage),
            ("packed_total_scale_first_to_last", framegroup_packed_broad.get("packed_total_scale_first_to_last")),
            ("packed_backward_scale_first_to_last", framegroup_packed_broad.get("packed_backward_scale_first_to_last")),
            ("packed_storage_scale_first_to_last", framegroup_packed_broad.get("packed_storage_scale_first_to_last")),
        ):
            if not _finite_positive(value):
                failures.append(f"packed broad compare {key} is not positive finite")
        if not (
            (_finite_positive(max_total) and float(max_total) > 1.05)
            or (_finite_positive(max_backward) and float(max_backward) > 1.05)
        ):
            failures.append("packed broad compare must preserve total/backward ratio above promotion guard")
        if _finite_positive(max_storage) and float(max_storage) >= 1.0:
            failures.append("packed broad compare storage ratio must stay below i16x3")
        max_psnr_delta = framegroup_packed_broad.get("max_psnr_delta")
        if not _finite_nonnegative(max_psnr_delta):
            failures.append("packed broad compare max_psnr_delta is not finite nonnegative")
        elif float(max_psnr_delta) > 1.0e-4:
            failures.append("packed broad compare max_psnr_delta exceeds guard")
        ratios_by_frame = framegroup_packed_broad.get("ratios_by_frame")
        if not isinstance(ratios_by_frame, dict):
            failures.append("packed broad compare missing ratios_by_frame")
        else:
            ratios64 = ratios_by_frame.get("64")
            ratios128 = ratios_by_frame.get("128")
            if not isinstance(ratios64, dict):
                failures.append("packed broad compare missing 64f ratios")
            else:
                total64 = ratios64.get("packed_over_i16x3_total_mean")
                backward64 = ratios64.get("packed_over_i16x3_backward_mean")
                if not _finite_positive(total64) or float(total64) >= 0.85:
                    failures.append("packed broad compare 64f total ratio must preserve speed win")
                if not _finite_positive(backward64) or float(backward64) >= 0.90:
                    failures.append("packed broad compare 64f backward ratio must preserve speed win")
            if not isinstance(ratios128, dict):
                failures.append("packed broad compare missing 128f ratios")
            else:
                total128 = ratios128.get("packed_over_i16x3_total_mean")
                backward128 = ratios128.get("packed_over_i16x3_backward_mean")
                if not _finite_positive(total128) or float(total128) <= 1.05:
                    failures.append("packed broad compare 128f total ratio must preserve speed rejection")
                if not _finite_positive(backward128) or float(backward128) <= 1.05:
                    failures.append("packed broad compare 128f backward ratio must preserve speed rejection")
        scope = str(framegroup_packed_broad.get("scope", ""))
        if "not a STAR-UVT competitiveness artifact" not in scope:
            failures.append("packed broad compare scope must keep STAR-UVT caveat")
        conclusion = str(framegroup_packed_broad.get("conclusion", ""))
        if "not broadly promoted" not in conclusion and "not broadly" not in conclusion:
            failures.append("packed broad compare conclusion must keep broad non-promotion")
        if "not full-trainer" not in conclusion or "STAR-UVT" not in conclusion:
            failures.append("packed broad compare conclusion must keep scope caveats")

    framegroup_autograd = payload.get("framegroup16_autograd_smoke")
    if not isinstance(framegroup_autograd, dict) or framegroup_autograd.get("available") is not True:
        failures.append("missing framegroup16 fused-MSE autograd smoke")
    else:
        if framegroup_autograd.get("completion_claim") is not False:
            failures.append("framegroup16 autograd smoke must not claim completion")
        if framegroup_autograd.get("full_trainer_claim") is not False:
            failures.append("framegroup16 autograd smoke must not claim full trainer coverage")
        if framegroup_autograd.get("quality_claim") is not False:
            failures.append("framegroup16 autograd smoke must not claim quality/capacity parity")
        _verify_framegroup_objective_adapter(
            framegroup_autograd.get("world_foam_objective_adapter"),
            prefix="framegroup16 autograd smoke",
            failures=failures,
        )
        if framegroup_autograd.get("world_foam_objective_adapter_rows_all_match") is not True:
            failures.append("framegroup16 autograd smoke adapter metadata must be present on every row")
        if framegroup_autograd.get("optimizer_mode") != "autograd":
            failures.append("framegroup16 autograd smoke must use optimizer_mode=autograd")
        if (
            framegroup_autograd.get("tape_mode")
            != "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse"
        ):
            failures.append("framegroup16 autograd smoke must cover the promoted framegroup fused-MSE mode")
        if framegroup_autograd.get("frame_counts") != [2]:
            failures.append("framegroup16 autograd smoke must remain a narrow 2f runtime smoke")
        if framegroup_autograd.get("render_size") != 16:
            failures.append("framegroup16 autograd smoke render_size must remain 16")
        if framegroup_autograd.get("site_count") != 4:
            failures.append("framegroup16 autograd smoke site_count must remain 4")
        acceptance = framegroup_autograd.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("framegroup16 autograd smoke missing row acceptance")
        else:
            for key in ("gradients_nonzero", "parameters_updated", "outputs_are_finite"):
                if acceptance.get(key) is not True:
                    failures.append(f"framegroup16 autograd smoke must preserve {key}")
        row = framegroup_autograd.get("row")
        if not isinstance(row, dict):
            failures.append("framegroup16 autograd smoke missing row summary")
        else:
            for key in ("total_ms", "backward_ms", "fused_loss_vjp_ms", "first_grad_abs_sum", "parameter_update_abs_max"):
                if not _finite_positive(row.get(key)):
                    failures.append(f"framegroup16 autograd smoke row.{key} is not positive finite")
            if not _finite_nonnegative(row.get("render_ms")):
                failures.append("framegroup16 autograd smoke row.render_ms is not finite nonnegative")
            elif float(row.get("render_ms", 0.0)) != 0.0:
                failures.append("framegroup16 autograd smoke should keep render_ms at zero for fused loss")
        conclusion = str(framegroup_autograd.get("conclusion", ""))
        if (
            ".backward()" not in conclusion
            or "WorldFoamFrozenRGBMSEObjective" not in conclusion
            or "not full trainer" not in conclusion
        ):
            failures.append("framegroup16 autograd smoke conclusion must keep backward coverage and scope boundary")

    framegroup_autograd_speedscale = payload.get("framegroup16_autograd_speedscale")
    if not isinstance(framegroup_autograd_speedscale, dict) or framegroup_autograd_speedscale.get("available") is not True:
        failures.append("missing framegroup16 fused-MSE autograd speedscale")
    else:
        if framegroup_autograd_speedscale.get("completion_claim") is not False:
            failures.append("framegroup16 autograd speedscale must not claim completion")
        if framegroup_autograd_speedscale.get("full_trainer_claim") is not False:
            failures.append("framegroup16 autograd speedscale must not claim full trainer coverage")
        if framegroup_autograd_speedscale.get("full_geometry_gradient_claim") is not False:
            failures.append("framegroup16 autograd speedscale must not claim geometry-gradient coverage")
        if framegroup_autograd_speedscale.get("quality_claim") is not False:
            failures.append("framegroup16 autograd speedscale must not claim quality/capacity parity")
        _verify_framegroup_objective_adapter(
            framegroup_autograd_speedscale.get("world_foam_objective_adapter"),
            prefix="framegroup16 autograd speedscale",
            failures=failures,
        )
        if framegroup_autograd_speedscale.get("world_foam_objective_adapter_rows_all_match") is not True:
            failures.append("framegroup16 autograd speedscale adapter metadata must be present on every row")
        if framegroup_autograd_speedscale.get("optimizer_mode") != "autograd":
            failures.append("framegroup16 autograd speedscale must use optimizer_mode=autograd")
        if (
            framegroup_autograd_speedscale.get("tape_mode")
            != "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse"
        ):
            failures.append("framegroup16 autograd speedscale must cover the promoted framegroup fused-MSE mode")
        if framegroup_autograd_speedscale.get("frame_counts") != [16, 32, 64, 128]:
            failures.append("framegroup16 autograd speedscale frame counts must remain 16/32/64/128")
        if framegroup_autograd_speedscale.get("real_loaded_frame_counts") != [16, 32]:
            failures.append("framegroup16 autograd speedscale real-loaded rows must remain 16/32f")
        if framegroup_autograd_speedscale.get("repeated_frame_counts") != [64, 128]:
            failures.append("framegroup16 autograd speedscale repeated rows must remain 64/128f")
        if framegroup_autograd_speedscale.get("render_size") != 32:
            failures.append("framegroup16 autograd speedscale render_size must remain 32")
        if framegroup_autograd_speedscale.get("site_count") != 12:
            failures.append("framegroup16 autograd speedscale site_count must remain 12")
        if framegroup_autograd_speedscale.get("steps") != 8:
            failures.append("framegroup16 autograd speedscale steps must remain 8")
        if framegroup_autograd_speedscale.get("warmup_steps") != 3:
            failures.append("framegroup16 autograd speedscale warmup_steps must remain 3")
        repeat_scope_by_frame = framegroup_autograd_speedscale.get("repeat_scope_by_frame")
        if not isinstance(repeat_scope_by_frame, dict):
            failures.append("framegroup16 autograd speedscale missing repeat_scope_by_frame")
        else:
            for frame in ("16", "32"):
                if repeat_scope_by_frame.get(frame) != "real loaded frame count":
                    failures.append(f"framegroup16 autograd speedscale {frame}f row must stay real-loaded")
            for frame in ("64", "128"):
                if "synthetic repeated-fixture speed-scaling smoke" not in str(repeat_scope_by_frame.get(frame, "")):
                    failures.append(f"framegroup16 autograd speedscale {frame}f row must keep repeated-fixture scope")
        for key, limit in (
            ("total_scale_first_to_last", 2.50),
            ("backward_scale_first_to_last", 2.50),
            ("selected_tape_storage_scale_first_to_last", 1.10),
        ):
            value = framegroup_autograd_speedscale.get(key)
            if not _finite_positive(value):
                failures.append(f"framegroup16 autograd speedscale {key} is not positive finite")
            elif float(value) > limit:
                failures.append(f"framegroup16 autograd speedscale {key} exceeds guarded threshold")
        acceptance = framegroup_autograd_speedscale.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("framegroup16 autograd speedscale missing acceptance")
        else:
            for key in ("all_rows_ok", "total_step_sublinear_vs_frames", "backward_sublinear_vs_frames"):
                if acceptance.get(key) is not True:
                    failures.append(f"framegroup16 autograd speedscale must preserve acceptance {key}")
        by_frame = framegroup_autograd_speedscale.get("by_frame")
        if not isinstance(by_frame, dict):
            failures.append("framegroup16 autograd speedscale missing by_frame")
        else:
            for frame in ("16", "32", "64", "128"):
                row = by_frame.get(frame)
                if not isinstance(row, dict):
                    failures.append(f"framegroup16 autograd speedscale missing {frame}f row")
                    continue
                for key in (
                    "total_ms",
                    "backward_ms",
                    "fused_loss_vjp_ms",
                    "first_grad_abs_sum",
                    "parameter_update_abs_max",
                    "final_train_psnr",
                    "final_heldout_psnr",
                ):
                    if not _finite_positive(row.get(key)):
                        failures.append(f"framegroup16 autograd speedscale {frame}f {key} is not positive finite")
                if not _finite_nonnegative(row.get("render_ms")):
                    failures.append(f"framegroup16 autograd speedscale {frame}f render_ms is not finite nonnegative")
                elif float(row.get("render_ms", 0.0)) != 0.0:
                    failures.append(f"framegroup16 autograd speedscale {frame}f render_ms should stay zero")
                _verify_framegroup_objective_adapter(
                    row.get("world_foam_objective_adapter"),
                    prefix=f"framegroup16 autograd speedscale {frame}f",
                    failures=failures,
                )
        conclusion = str(framegroup_autograd_speedscale.get("conclusion", ""))
        if (
            ".backward()" not in conclusion
            or "warmed multi-frame" not in conclusion
            or "WorldFoamFrozenRGBMSEObjective" not in conclusion
            or "not full trainer" not in conclusion
        ):
            failures.append("framegroup16 autograd speedscale conclusion must keep warmed backward coverage and scope boundary")

    segment_tape = payload.get("segment_tape_probe")
    if not isinstance(segment_tape, dict) or segment_tape.get("available") is not True:
        failures.append("missing green segment tape probe")
    else:
        if segment_tape.get("completion_claim") is not False:
            failures.append("segment tape probe must not claim completion")
        if segment_tape.get("metal_kernel_implemented") is not True:
            failures.append("segment tape probe must preserve that the compact Metal tape kernel is implemented")
        for key in (
            "max_forward_error_vs_current_mixed",
            "max_grad_rel_error_vs_current_winner_grad_only",
            "max_metal_forward_error_vs_current_mixed",
            "max_metal_grad_rel_error_vs_current_winner_grad_only",
            "max_metal_track_grad_rel_error_vs_current_winner_grad_only",
            "max_metal_track_grad_rel_error_vs_sample_atomic",
            "frame_scale_first_to_last",
            "segment_scale_first_to_last",
        ):
            if not _finite_positive(segment_tape.get(key)):
                failures.append(f"segment_tape_probe.{key} is not positive finite")
        grad_rel = segment_tape.get("max_metal_grad_rel_error_vs_current_winner_grad_only")
        if isinstance(grad_rel, (int, float)) and math.isfinite(float(grad_rel)) and float(grad_rel) > 1.0e-5:
            failures.append(f"Metal segment tape winner VJP relative error {grad_rel} exceeds 1e-5")
        track_rel = segment_tape.get("max_metal_track_grad_rel_error_vs_current_winner_grad_only")
        if isinstance(track_rel, (int, float)) and math.isfinite(float(track_rel)) and float(track_rel) > 2.0e-5:
            failures.append(f"Metal track segment tape winner VJP relative error {track_rel} exceeds 2e-5")
        frame_scale = segment_tape.get("frame_scale_first_to_last")
        segment_scale = segment_tape.get("segment_scale_first_to_last")
        if (
            isinstance(frame_scale, (int, float))
            and isinstance(segment_scale, (int, float))
            and math.isfinite(float(frame_scale))
            and math.isfinite(float(segment_scale))
            and float(segment_scale) < 0.75 * float(frame_scale)
        ):
            failures.append("segment tape summary unexpectedly looks sublinear; audit before using as structural claim")
        conclusion = str(segment_tape.get("conclusion", ""))
        if "compact segment-tape Metal replay" not in conclusion or "scales roughly with frame count" not in conclusion:
            failures.append("segment tape conclusion must capture Metal coverage and remaining scaling gap")

    topology = payload.get("topology_sharing_probe")
    if not isinstance(topology, dict) or topology.get("available") is not True:
        failures.append("missing topology sharing probe")
    else:
        if topology.get("completion_claim") is not False:
            failures.append("topology sharing probe must not claim completion")
        acceptance = topology.get("acceptance")
        if not isinstance(acceptance, dict) or acceptance.get("zero_missing_sample_events") is not True:
            failures.append("topology sharing probe must preserve zero missing sample events")
        last_row = topology.get("last_row")
        if not isinstance(last_row, dict):
            failures.append("topology sharing probe missing last_row")
        else:
            same_tracks = last_row.get("same_topology_all_frames_tracks")
            rows_vs_samples = last_row.get("track_unique_topology_rows_vs_samples")
            if same_tracks != 0:
                failures.append("topology sharing probe should record zero all-frame-identical topology tracks")
            if not _finite_positive(rows_vs_samples) or float(rows_vs_samples) < 0.75:
                failures.append("topology sharing probe no longer supports weak owner-topology sharing conclusion")
        conclusion = str(topology.get("conclusion", ""))
        if "weak" not in conclusion or "moving-camera" not in conclusion:
            failures.append("topology sharing conclusion must capture weak moving-camera sharing")

    delta = payload.get("delta_tape_probe")
    if not isinstance(delta, dict) or delta.get("available") is not True:
        failures.append("missing delta tape probe")
    else:
        if delta.get("completion_claim") is not False:
            failures.append("delta tape probe must not claim completion")
        acceptance = delta.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("delta tape probe missing acceptance")
        else:
            if acceptance.get("edit_ops_scale_sublinear_vs_frames") is not True:
                failures.append("delta tape probe must preserve edit-op sublinear signal")
            if acceptance.get("change_events_scale_sublinear_vs_frames") is not False:
                failures.append("delta tape probe must preserve that coarse changed rows are not sublinear")
            if acceptance.get("last_delta_owner_storage_below_full_compact_csr") is not True:
                failures.append("delta owner storage must remain below full compact CSR in the saved result")
        frame_scale = delta.get("frame_scale_first_to_last")
        edit_scale = delta.get("edit_op_scale_first_to_last")
        change_scale = delta.get("change_event_scale_first_to_last")
        owner_storage_scale = delta.get("delta_owner_storage_scale_first_to_last")
        for key, value in (
            ("frame_scale_first_to_last", frame_scale),
            ("edit_op_scale_first_to_last", edit_scale),
            ("change_event_scale_first_to_last", change_scale),
            ("delta_owner_storage_scale_first_to_last", owner_storage_scale),
        ):
            if not _finite_positive(value):
                failures.append(f"delta_tape_probe.{key} is not positive finite")
        if _finite_positive(frame_scale) and _finite_positive(edit_scale) and float(edit_scale) >= float(frame_scale):
            failures.append("delta edit-op scale is not sublinear versus frame count")
        if _finite_positive(frame_scale) and _finite_positive(change_scale) and float(change_scale) < float(frame_scale):
            failures.append("delta changed-row scale unexpectedly looks sublinear; audit before using")
        conclusion = str(delta.get("conclusion", ""))
        if "promising" not in conclusion or "length/mid" not in conclusion:
            failures.append("delta tape conclusion must keep promising signal and exact replay gap explicit")

    boundary_delta = payload.get("boundary_delta_tape_probe")
    if not isinstance(boundary_delta, dict) or boundary_delta.get("available") is not True:
        failures.append("missing boundary delta tape probe")
    else:
        if boundary_delta.get("completion_claim") is not False:
            failures.append("boundary delta tape probe must not claim completion")
        acceptance = boundary_delta.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("boundary delta tape probe missing acceptance")
        else:
            if acceptance.get("boundary_edit_ops_scale_sublinear_vs_frames") is not True:
                failures.append("boundary delta tape must preserve boundary edit-op sublinear signal")
            if acceptance.get("delta_replace_boundary_storage_scale_sublinear_vs_frames") is not False:
                failures.append("boundary delta tape must preserve raw boundary replacement storage gap")
            if acceptance.get("last_delta_replace_boundary_storage_below_full_segment_csr") is not True:
                failures.append("boundary delta storage must remain below full segment CSR at 16f")
        frame_scale = boundary_delta.get("frame_scale_first_to_last")
        edit_scale = boundary_delta.get("boundary_edit_op_scale_first_to_last")
        replace_storage_scale = boundary_delta.get("delta_replace_boundary_storage_scale_first_to_last")
        for key, value in (
            ("frame_scale_first_to_last", frame_scale),
            ("boundary_edit_op_scale_first_to_last", edit_scale),
            ("delta_replace_boundary_storage_scale_first_to_last", replace_storage_scale),
        ):
            if not _finite_positive(value):
                failures.append(f"boundary_delta_tape_probe.{key} is not positive finite")
        if _finite_positive(frame_scale) and _finite_positive(edit_scale) and float(edit_scale) >= float(frame_scale):
            failures.append("boundary edit-op scale is not sublinear versus frame count")
        conclusion = str(boundary_delta.get("conclusion", ""))
        if "exact length/mid replay" not in conclusion or "owner assignment" not in conclusion:
            failures.append("boundary delta conclusion must keep exact geometry signal and owner gap explicit")

    record_delta = payload.get("record_delta_tape_probe")
    if not isinstance(record_delta, dict) or record_delta.get("available") is not True:
        failures.append("missing exact segment-record delta tape probe")
    else:
        if record_delta.get("completion_claim") is not False:
            failures.append("record delta tape probe must not claim completion")
        if record_delta.get("star_uvt_competitive_claim") is not False:
            failures.append("record delta tape probe must not claim STAR UVT competitiveness")
        acceptance = record_delta.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("record delta tape probe missing acceptance")
        else:
            if acceptance.get("record_counts_match_segment_tape") is not True:
                failures.append("record delta tape must preserve exact segment-tape count/owner match")
            if acceptance.get("record_count_scales_about_with_frames") is not True:
                failures.append("record delta tape must preserve that exact records scale about with frames")
            if acceptance.get("record_edit_ops_scale_sublinear_vs_frames") is not True:
                failures.append("record delta tape must preserve edit-op sublinear signal")
            if acceptance.get("delta_replace_record_storage_scale_sublinear_vs_frames") is not False:
                failures.append("record delta tape must preserve replacement-record storage scaling failure")
            if acceptance.get("last_delta_replace_record_storage_below_full_segment_csr") is not False:
                failures.append("record delta tape must preserve full-CSR-sized 16f replacement storage")
        frame_scale = record_delta.get("frame_scale_first_to_last")
        record_scale = record_delta.get("full_record_count_scale_first_to_last")
        edit_scale = record_delta.get("record_edit_op_scale_first_to_last")
        replace_storage_scale = record_delta.get("delta_replace_record_storage_scale_first_to_last")
        for key, value in (
            ("frame_scale_first_to_last", frame_scale),
            ("full_record_count_scale_first_to_last", record_scale),
            ("record_edit_op_scale_first_to_last", edit_scale),
            ("delta_replace_record_storage_scale_first_to_last", replace_storage_scale),
        ):
            if not _finite_positive(value):
                failures.append(f"record_delta_tape_probe.{key} is not positive finite")
        if _finite_positive(frame_scale) and _finite_positive(record_scale) and float(record_scale) < 0.75 * float(frame_scale):
            failures.append("record delta exact record count unexpectedly looks structurally sublinear")
        if _finite_positive(frame_scale) and _finite_positive(edit_scale) and float(edit_scale) >= float(frame_scale):
            failures.append("record delta edit-op scale is not sublinear versus frame count")
        last_row = record_delta.get("last_row")
        if not isinstance(last_row, dict):
            failures.append("record delta tape probe missing last_row")
        else:
            if last_row.get("record_counts_match_segment_tape") is not True:
                failures.append("record delta last row must match segment-tape counts and owners")
            replace_ratio = last_row.get("delta_replace_record_vs_full_segment_csr")
            edit_ratio = last_row.get("delta_edit_op_record_stream_vs_full_segment_csr")
            change_rate = last_row.get("change_event_rate")
            if not _finite_positive(replace_ratio) or float(replace_ratio) < 0.95:
                failures.append("record delta replacement stream no longer supports full-CSR-sized conclusion")
            if not _finite_positive(edit_ratio) or float(edit_ratio) >= 1.0:
                failures.append("record delta edit-op stream should remain below full CSR but not a compact win")
            if not _finite_positive(change_rate) or float(change_rate) <= 0.90:
                failures.append("record delta change rate no longer supports noisy exact-record conclusion")
        conclusion = str(record_delta.get("conclusion", ""))
        if (
            "Exact owner+boundary-cut record deltas" not in conclusion
            or "about full-CSR sized" not in conclusion
            or "not a compact STAR-like exact tape" not in conclusion
        ):
            failures.append("record delta conclusion must keep exact replay and compactness failure explicit")

    owner_run = payload.get("owner_run_tape_probe")
    if not isinstance(owner_run, dict) or owner_run.get("available") is not True:
        failures.append("missing owner-run tape probe")
    else:
        if owner_run.get("completion_claim") is not False:
            failures.append("owner-run tape probe must not claim completion")
        acceptance = owner_run.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("owner-run tape probe missing acceptance")
        else:
            for key in (
                "owner_run_forward_rgb_matches_full",
                "owner_run_forward_alpha_matches_full",
                "owner_run_forward_depth_matches_current_density_full",
                "owner_run_rgb_only_vjp_matches_full",
                "owner_run_segments_below_full",
            ):
                if acceptance.get(key) is not True:
                    failures.append(f"owner-run tape acceptance {key} must be true")
        for key in (
            "max_forward_rgb_abs_error",
            "max_forward_alpha_abs_error",
            "max_forward_depth_abs_error",
            "max_rgb_only_vjp_rel_error",
        ):
            if not _finite_positive(owner_run.get(key)):
                failures.append(f"owner_run_tape_probe.{key} is not positive finite")
        if (
            _finite_positive(owner_run.get("max_rgb_only_vjp_rel_error"))
            and float(owner_run["max_rgb_only_vjp_rel_error"]) > 2.0e-5
        ):
            failures.append("owner-run RGB-only VJP relative error exceeds 2e-5")
        last_row = owner_run.get("last_row")
        if not isinstance(last_row, dict):
            failures.append("owner-run tape missing last_row")
        else:
            ratio = last_row.get("owner_run_segments_vs_full_segments")
            storage_ratio = last_row.get("owner_run_storage_vs_full")
            if not _finite_positive(ratio) or float(ratio) >= 0.25:
                failures.append("owner-run 16f segment ratio no longer supports compression claim")
            if not _finite_positive(storage_ratio) or float(storage_ratio) >= 0.25:
                failures.append("owner-run 16f storage ratio no longer supports compression claim")
        conclusion = str(owner_run.get("conclusion", ""))
        if "RGB-training candidate" not in conclusion or "density-independent" not in conclusion:
            failures.append("owner-run conclusion must keep practical RGB and depth dependency explicit")

    owner_run_boundary = payload.get("owner_run_boundary_tape_probe")
    if not isinstance(owner_run_boundary, dict) or owner_run_boundary.get("available") is not True:
        failures.append("missing owner-run boundary endpoint tape probe")
    else:
        if owner_run_boundary.get("completion_claim") is not False:
            failures.append("owner-run boundary endpoint tape probe must not claim completion")
        acceptance = owner_run_boundary.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("owner-run boundary endpoint tape probe missing acceptance")
        else:
            for key in (
                "matches_current_owner_run_counts_and_owners",
                "endpoint_ids_recover_run_lengths",
                "owner_run_boundary_storage_below_full_at_max_frame",
            ):
                if acceptance.get(key) is not True:
                    failures.append(f"owner-run boundary acceptance {key} must be true")
            if acceptance.get("owner_run_boundary_run_count_sublinear_vs_frames") is not False:
                failures.append("owner-run boundary tape must preserve that run count is not structurally sublinear")
            if acceptance.get("endpoint_continuous_density_depth_matches_current_segment_mid_depth") is not False:
                failures.append("owner-run boundary tape must preserve endpoint-only depth mismatch")
        rel = owner_run_boundary.get("max_endpoint_length_abs_error")
        if not isinstance(rel, (int, float)) or not math.isfinite(float(rel)) or float(rel) > 5.0e-5:
            failures.append("owner-run boundary endpoint length recovery error exceeds 5e-5")
        endpoint_depth_error = owner_run_boundary.get("max_endpoint_density_depth_abs_error_vs_current_owner_run")
        if not _finite_positive(endpoint_depth_error):
            failures.append("owner-run boundary endpoint density depth error must be positive finite")
        elif float(endpoint_depth_error) <= 5.0e-5:
            failures.append("owner-run boundary endpoint-only depth unexpectedly matches current segment-mid depth")
        frame_scale = owner_run_boundary.get("frame_scale_first_to_last")
        run_scale = owner_run_boundary.get("owner_run_boundary_run_scale_first_to_last")
        if _finite_positive(frame_scale) and _finite_positive(run_scale) and float(run_scale) <= float(frame_scale):
            failures.append("owner-run boundary run scale unexpectedly looks sublinear; audit before claiming")
        last_row = owner_run_boundary.get("last_row")
        if not isinstance(last_row, dict):
            failures.append("owner-run boundary endpoint tape missing last_row")
        else:
            ratio = last_row.get("owner_run_boundary_id_vs_full_segment_csr")
            same_storage = last_row.get("owner_run_boundary_id_vs_current_owner_run_length_mid_csr")
            if not _finite_positive(ratio) or float(ratio) >= 0.10:
                failures.append("owner-run boundary endpoint tape 16f storage ratio no longer supports compression")
            if not _finite_positive(same_storage) or abs(float(same_storage) - 1.0) > 1.0e-6:
                failures.append("owner-run boundary endpoint tape should keep same CSR byte size as length/mid owner-run tape")
            alpha_error = last_row.get("max_endpoint_alpha_abs_error_vs_current_owner_run")
            depth_error = last_row.get("max_endpoint_density_depth_abs_error_vs_current_owner_run")
            if not isinstance(alpha_error, (int, float)) or not math.isfinite(float(alpha_error)) or float(alpha_error) > 1.0e-5:
                failures.append("owner-run boundary endpoint alpha error must remain near zero")
            if not _finite_positive(depth_error) or float(depth_error) <= 5.0e-5:
                failures.append("owner-run boundary endpoint last-row depth error must preserve mismatch")
        conclusion = str(owner_run_boundary.get("conclusion", ""))
        if (
            "recover run lengths" not in conclusion
            or "does not match" not in conclusion
            or "run count still scales" not in conclusion
        ):
            failures.append("owner-run boundary conclusion must keep exact-length, depth-mismatch, and scaling-gap findings explicit")

    owner_run_internal = payload.get("owner_run_internal_tape_probe")
    if not isinstance(owner_run_internal, dict) or owner_run_internal.get("available") is not True:
        failures.append("missing owner-run internal-cut tape probe")
    else:
        if owner_run_internal.get("completion_claim") is not False:
            failures.append("owner-run internal-cut tape probe must not claim completion")
        if owner_run_internal.get("star_uvt_competitive_claim") is not False:
            failures.append("owner-run internal-cut tape probe must not claim STAR UVT competitiveness")
        acceptance = owner_run_internal.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("owner-run internal-cut tape probe missing acceptance")
        else:
            expected_true = (
                "active_internal_matches_current_density_depth",
                "active_internal_not_density_independent_under_lower_density",
                "all_internal_preserves_density_independent_replay_by_construction",
                "active_internal_storage_below_full_at_max_frame",
                "all_internal_storage_not_star_like_at_max_frame",
            )
            for key in expected_true:
                if acceptance.get(key) is not True:
                    failures.append(f"owner-run internal-cut acceptance {key} must be true")
            for key in (
                "all_internal_segment_count_sublinear_vs_frames",
                "active_internal_segment_count_sublinear_vs_frames",
            ):
                if acceptance.get(key) is not False:
                    failures.append(f"owner-run internal-cut acceptance {key} must preserve non-sublinear result")
        frame_scale = owner_run_internal.get("frame_scale_first_to_last")
        active_scale = owner_run_internal.get("active_internal_segment_count_scale_first_to_last")
        all_scale = owner_run_internal.get("all_internal_segment_count_scale_first_to_last")
        for key, value in (
            ("frame_scale_first_to_last", frame_scale),
            ("active_internal_segment_count_scale_first_to_last", active_scale),
            ("all_internal_segment_count_scale_first_to_last", all_scale),
        ):
            if not _finite_positive(value):
                failures.append(f"owner_run_internal_tape_probe.{key} is not positive finite")
        if _finite_positive(frame_scale) and _finite_positive(active_scale) and float(active_scale) <= float(frame_scale):
            failures.append("active internal segment scale unexpectedly looks sublinear")
        if _finite_positive(frame_scale) and _finite_positive(all_scale) and float(all_scale) <= float(frame_scale):
            failures.append("all internal segment scale unexpectedly looks sublinear")
        last_row = owner_run_internal.get("last_row")
        if not isinstance(last_row, dict):
            failures.append("owner-run internal-cut tape missing last_row")
        else:
            active_ratio = last_row.get("active_internal_nested_csr_vs_full_segment_csr")
            all_ratio = last_row.get("all_internal_nested_csr_vs_full_segment_csr")
            all_endpoint_ratio = last_row.get("all_internal_endpoint_run_csr_vs_full_segment_csr")
            active_current_depth = last_row.get("active_current_density_depth_max_abs")
            half_depth = last_row.get("active_half_density_depth_max_abs")
            half_alpha = last_row.get("active_half_density_alpha_max_abs")
            if not _finite_positive(active_ratio) or float(active_ratio) >= 0.25:
                failures.append("active internal-cut 16f storage ratio no longer supports compact current-depth path")
            if not _finite_positive(all_ratio) or float(all_ratio) <= 0.50:
                failures.append("all internal-cut 16f storage ratio no longer supports not-STAR-like conclusion")
            if not _finite_positive(all_endpoint_ratio) or float(all_endpoint_ratio) >= 0.20:
                failures.append("all owner-run endpoint 16f storage ratio no longer supports compact semantic-change option")
            if not isinstance(active_current_depth, (int, float)) or not math.isfinite(float(active_current_depth)) or float(active_current_depth) > 5.0e-5:
                failures.append("active internal-cut current-density depth error must remain near zero")
            if not _finite_positive(half_depth) or float(half_depth) <= 5.0e-5:
                failures.append("active internal-cut half-density depth error must preserve density-dependence")
            if not _finite_positive(half_alpha) or float(half_alpha) <= 5.0e-5:
                failures.append("active internal-cut half-density alpha error must preserve threshold-truncation gap")
        conclusion = str(owner_run_internal.get("conclusion", ""))
        if (
            "exact current-depth replay" not in conclusion
            or "not density independent" not in conclusion
            or "moves storage back toward the full" not in conclusion
            or "depth semantics change" not in conclusion
        ):
            failures.append("owner-run internal-cut conclusion must keep current-depth, density, and storage tradeoff explicit")

    endpoint_run = payload.get("endpoint_run_tape_probe")
    if not isinstance(endpoint_run, dict) or endpoint_run.get("available") is not True:
        failures.append("missing endpoint-run continuous-depth tape probe")
    else:
        if endpoint_run.get("completion_claim") is not False:
            failures.append("endpoint-run tape probe must not claim completion")
        if endpoint_run.get("star_uvt_competitive_claim") is not False:
            failures.append("endpoint-run tape probe must not claim STAR UVT competitiveness")
        acceptance = endpoint_run.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("endpoint-run tape probe missing acceptance")
        else:
            for key in (
                "metal_forward_matches_torch_continuous_endpoint_replay",
                "metal_vjp_matches_torch_autograd",
                "endpoint_storage_below_full_at_max_frame",
                "endpoint_runs_under_vjp_cap",
            ):
                if acceptance.get(key) is not True:
                    failures.append(f"endpoint-run tape acceptance {key} must be true")
            if acceptance.get("endpoint_run_count_sublinear_vs_frames") is not False:
                failures.append("endpoint-run tape must preserve non-STAR-like run-count scaling")
        for key in (
            "frame_scale_first_to_last",
            "endpoint_run_scale_first_to_last",
            "max_forward_abs_error_vs_torch",
            "max_vjp_rel_error_vs_torch_autograd",
        ):
            if not _finite_positive(endpoint_run.get(key)):
                failures.append(f"endpoint_run_tape_probe.{key} is not positive finite")
        frame_scale = endpoint_run.get("frame_scale_first_to_last")
        run_scale = endpoint_run.get("endpoint_run_scale_first_to_last")
        if _finite_positive(frame_scale) and _finite_positive(run_scale) and float(run_scale) <= float(frame_scale):
            failures.append("endpoint-run count unexpectedly looks structurally sublinear; audit before using")
        rel = endpoint_run.get("max_vjp_rel_error_vs_torch_autograd")
        if _finite_positive(rel) and float(rel) > 5.0e-4:
            failures.append("endpoint-run VJP relative error exceeds 5e-4")
        last_row = endpoint_run.get("last_row")
        if not isinstance(last_row, dict):
            failures.append("endpoint-run tape probe missing last_row")
        else:
            storage_ratio = last_row.get("endpoint_storage_vs_full_segment_csr")
            run_ratio = last_row.get("endpoint_runs_vs_full_segments")
            max_runs = last_row.get("max_endpoint_runs_per_sample")
            if not _finite_positive(storage_ratio) or float(storage_ratio) >= 0.20:
                failures.append("endpoint-run 16f storage ratio no longer supports compact semantic-change path")
            if not _finite_positive(run_ratio) or float(run_ratio) >= 0.20:
                failures.append("endpoint-run 16f run ratio no longer supports compact semantic-change path")
            if not _finite_positive(max_runs) or float(max_runs) > 129.0:
                failures.append("endpoint-run max runs per sample exceeds VJP cap")
        conclusion = str(endpoint_run.get("conclusion", ""))
        if (
            "continuous-absorption" not in conclusion
            or "0.111x full segment CSR" not in conclusion
            or "semantic change" not in conclusion
        ):
            failures.append("endpoint-run conclusion must keep continuous-depth compactness and semantic change explicit")

    endpoint_record_delta = payload.get("endpoint_record_delta_tape_probe")
    if not isinstance(endpoint_record_delta, dict) or endpoint_record_delta.get("available") is not True:
        failures.append("missing endpoint-record delta tape probe")
    else:
        if endpoint_record_delta.get("completion_claim") is not False:
            failures.append("endpoint-record delta probe must not claim completion")
        if endpoint_record_delta.get("star_uvt_competitive_claim") is not False:
            failures.append("endpoint-record delta probe must not claim STAR UVT competitiveness")
        acceptance = endpoint_record_delta.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("endpoint-record delta probe missing acceptance")
        else:
            if acceptance.get("endpoint_records_match_endpoint_run_counts_and_owners") is not True:
                failures.append("endpoint-record delta must match endpoint-run counts and owners")
            if acceptance.get("endpoint_record_count_sublinear_vs_frames") is not False:
                failures.append("endpoint-record count must preserve non-sublinear result")
            for key in (
                "endpoint_record_edit_ops_sublinear_vs_frames",
                "delta_replace_endpoint_record_storage_sublinear_vs_frames",
                "delta_edit_op_endpoint_record_storage_sublinear_vs_frames",
                "last_full_endpoint_record_storage_below_full_segment_csr",
                "last_delta_edit_op_endpoint_record_storage_below_full_segment_csr",
            ):
                if acceptance.get(key) is not True:
                    failures.append(f"endpoint-record delta acceptance {key} must be true")
        frame_scale = endpoint_record_delta.get("frame_scale_first_to_last")
        record_scale = endpoint_record_delta.get("endpoint_record_count_scale_first_to_last")
        edit_scale = endpoint_record_delta.get("endpoint_record_edit_op_scale_first_to_last")
        edit_storage_scale = endpoint_record_delta.get("delta_edit_op_endpoint_record_storage_scale_first_to_last")
        for key, value in (
            ("frame_scale_first_to_last", frame_scale),
            ("endpoint_record_count_scale_first_to_last", record_scale),
            ("endpoint_record_edit_op_scale_first_to_last", edit_scale),
            ("delta_edit_op_endpoint_record_storage_scale_first_to_last", edit_storage_scale),
        ):
            if not _finite_positive(value):
                failures.append(f"endpoint_record_delta_tape_probe.{key} is not positive finite")
        if _finite_positive(frame_scale) and _finite_positive(record_scale) and float(record_scale) <= float(frame_scale):
            failures.append("endpoint-record count unexpectedly looks structurally sublinear")
        if _finite_positive(frame_scale) and _finite_positive(edit_scale) and float(edit_scale) >= float(frame_scale):
            failures.append("endpoint-record edit-op scale is not sublinear versus frame count")
        if _finite_positive(frame_scale) and _finite_positive(edit_storage_scale) and float(edit_storage_scale) >= float(frame_scale):
            failures.append("endpoint-record edit-op storage scale is not sublinear versus frame count")
        last_row = endpoint_record_delta.get("last_row")
        if not isinstance(last_row, dict):
            failures.append("endpoint-record delta probe missing last_row")
        else:
            full_endpoint_ratio = last_row.get("full_endpoint_record_csr_vs_full_segment_csr")
            edit_ratio = last_row.get("delta_edit_op_endpoint_record_stream_vs_full_segment_csr")
            edit_vs_endpoint = last_row.get("delta_edit_op_endpoint_record_stream_vs_full_endpoint_record_csr")
            unique_ratio = last_row.get("track_unique_endpoint_record_rows_vs_samples")
            change_rate = last_row.get("change_event_rate")
            if not _finite_positive(full_endpoint_ratio) or float(full_endpoint_ratio) >= 0.20:
                failures.append("endpoint-record full CSR 16f ratio no longer supports compact endpoint path")
            if not _finite_positive(edit_ratio) or float(edit_ratio) >= 0.05:
                failures.append("endpoint-record edit stream 16f ratio no longer supports delta signal")
            if not _finite_positive(edit_vs_endpoint) or float(edit_vs_endpoint) >= 0.50:
                failures.append("endpoint-record edit stream must remain smaller than full endpoint CSR")
            if not _finite_positive(unique_ratio) or float(unique_ratio) >= 0.50:
                failures.append("endpoint-record unique row ratio no longer supports cross-frame sharing signal")
            if not _finite_positive(change_rate) or float(change_rate) >= 0.30:
                failures.append("endpoint-record 16f change rate no longer supports amortized delta signal")
        conclusion = str(endpoint_record_delta.get("conclusion", ""))
        if (
            "promising" not in conclusion
            or "strongly sublinear" not in conclusion
            or "not main-trainer integrated" not in conclusion
        ):
            failures.append("endpoint-record delta conclusion must keep promise and trainer scope explicit")

    endpoint_record_replay = payload.get("endpoint_record_delta_replay")
    if not isinstance(endpoint_record_replay, dict) or endpoint_record_replay.get("available") is not True:
        failures.append("missing endpoint-record delta replay shader artifact")
    else:
        if endpoint_record_replay.get("completion_claim") is not False:
            failures.append("endpoint-record replay shader must not claim completion")
        if endpoint_record_replay.get("star_uvt_competitive_claim") is not False:
            failures.append("endpoint-record replay shader must not claim STAR UVT competitiveness")
        acceptance = endpoint_record_replay.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("endpoint-record replay shader missing acceptance")
        else:
            for key in (
                "metal_forward_matches_endpoint_run",
                "metal_vjp_matches_endpoint_run",
                "record_delta_storage_sublinear_vs_frames",
                "last_record_delta_storage_below_endpoint_csr",
            ):
                if acceptance.get(key) is not True:
                    failures.append(f"endpoint-record replay shader acceptance {key} must be true")
        frame_scale = endpoint_record_replay.get("frame_scale_first_to_last")
        endpoint_scale = endpoint_record_replay.get("endpoint_run_scale_first_to_last")
        storage_scale = endpoint_record_replay.get("record_delta_storage_scale_first_to_last")
        fwd_error = endpoint_record_replay.get("max_forward_abs_error_vs_endpoint_run")
        vjp_error = endpoint_record_replay.get("max_vjp_rel_error_vs_endpoint_run")
        for key, value in (
            ("frame_scale_first_to_last", frame_scale),
            ("endpoint_run_scale_first_to_last", endpoint_scale),
            ("record_delta_storage_scale_first_to_last", storage_scale),
        ):
            if not _finite_positive(value):
                failures.append(f"endpoint_record_delta_replay.{key} is not positive finite")
        if _finite_positive(frame_scale) and _finite_positive(endpoint_scale) and float(endpoint_scale) <= float(frame_scale):
            failures.append("endpoint-record replay must preserve endpoint run count non-sublinearity")
        if _finite_positive(frame_scale) and _finite_positive(storage_scale) and float(storage_scale) >= float(frame_scale):
            failures.append("endpoint-record replay storage scale must be sublinear")
        if not _finite_nonnegative(fwd_error) or float(fwd_error) >= 1.0e-4:
            failures.append("endpoint-record replay forward error no longer matches endpoint run")
        if not _finite_nonnegative(vjp_error) or float(vjp_error) >= 1.0e-4:
            failures.append("endpoint-record replay VJP error no longer matches endpoint run")
        last_row = endpoint_record_replay.get("last_row")
        if not isinstance(last_row, dict):
            failures.append("endpoint-record replay shader missing last_row")
        else:
            endpoint_ratio = last_row.get("record_delta_storage_vs_endpoint_csr")
            full_ratio = last_row.get("record_delta_storage_vs_full_segment_csr")
            fwd_ms = last_row.get("record_delta_forward_ms")
            vjp_ms = last_row.get("record_delta_vjp_ms")
            if not _finite_positive(endpoint_ratio) or float(endpoint_ratio) >= 0.50:
                failures.append("endpoint-record replay 16f storage must stay below half endpoint CSR")
            if not _finite_positive(full_ratio) or float(full_ratio) >= 0.05:
                failures.append("endpoint-record replay 16f storage must stay below 0.05x full segment CSR")
            if not _finite_positive(fwd_ms) or not _finite_positive(vjp_ms):
                failures.append("endpoint-record replay timings must be positive")
        conclusion = str(endpoint_record_replay.get("conclusion", ""))
        if "real Metal shader path" not in conclusion or "not the newer edit-op stream" not in conclusion:
            failures.append("endpoint-record replacement replay conclusion must keep shader and edit-op scope explicit")

    endpoint_record_edit = payload.get("endpoint_record_edit_replay")
    if not isinstance(endpoint_record_edit, dict) or endpoint_record_edit.get("available") is not True:
        failures.append("missing endpoint-record edit-op replay shader artifact")
    else:
        if endpoint_record_edit.get("completion_claim") is not False:
            failures.append("endpoint-record edit-op shader must not claim completion")
        if endpoint_record_edit.get("star_uvt_competitive_claim") is not False:
            failures.append("endpoint-record edit-op shader must not claim STAR UVT competitiveness")
        acceptance = endpoint_record_edit.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("endpoint-record edit-op shader missing acceptance")
        else:
            for key in (
                "metal_forward_matches_endpoint_run",
                "metal_vjp_matches_endpoint_run",
                "edit_ops_sublinear_vs_frames",
                "edit_storage_sublinear_vs_frames",
                "last_edit_storage_below_endpoint_csr",
            ):
                if acceptance.get(key) is not True:
                    failures.append(f"endpoint-record edit-op shader acceptance {key} must be true")
        frame_scale = endpoint_record_edit.get("frame_scale_first_to_last")
        endpoint_scale = endpoint_record_edit.get("endpoint_run_scale_first_to_last")
        edit_scale = endpoint_record_edit.get("edit_op_scale_first_to_last")
        storage_scale = endpoint_record_edit.get("edit_storage_scale_first_to_last")
        fwd_error = endpoint_record_edit.get("max_forward_abs_error_vs_endpoint_run")
        vjp_error = endpoint_record_edit.get("max_vjp_rel_error_vs_endpoint_run")
        for key, value in (
            ("frame_scale_first_to_last", frame_scale),
            ("endpoint_run_scale_first_to_last", endpoint_scale),
            ("edit_op_scale_first_to_last", edit_scale),
            ("edit_storage_scale_first_to_last", storage_scale),
        ):
            if not _finite_positive(value):
                failures.append(f"endpoint_record_edit_replay.{key} is not positive finite")
        if _finite_positive(frame_scale) and _finite_positive(endpoint_scale) and float(endpoint_scale) <= float(frame_scale):
            failures.append("endpoint-record edit-op replay must preserve endpoint run count non-sublinearity")
        if _finite_positive(frame_scale) and _finite_positive(edit_scale) and float(edit_scale) >= float(frame_scale):
            failures.append("endpoint-record edit-op count must be sublinear")
        if _finite_positive(frame_scale) and _finite_positive(storage_scale) and float(storage_scale) >= float(frame_scale):
            failures.append("endpoint-record edit-op storage scale must be sublinear")
        if not _finite_nonnegative(fwd_error) or float(fwd_error) >= 1.0e-4:
            failures.append("endpoint-record edit-op forward error no longer matches endpoint run")
        if not _finite_nonnegative(vjp_error) or float(vjp_error) >= 1.0e-4:
            failures.append("endpoint-record edit-op VJP error no longer matches endpoint run")
        last_row = endpoint_record_edit.get("last_row")
        if not isinstance(last_row, dict):
            failures.append("endpoint-record edit-op shader missing last_row")
        else:
            endpoint_ratio = last_row.get("edit_storage_vs_endpoint_csr")
            full_ratio = last_row.get("edit_storage_vs_full_segment_csr")
            endpoint_fwd_ms = last_row.get("endpoint_forward_ms")
            edit_fwd_ms = last_row.get("edit_forward_ms")
            endpoint_vjp_ms = last_row.get("endpoint_vjp_ms")
            edit_vjp_ms = last_row.get("edit_vjp_ms")
            if not _finite_positive(endpoint_ratio) or float(endpoint_ratio) >= 0.50:
                failures.append("endpoint-record edit-op 16f storage must stay below half endpoint CSR")
            if not _finite_positive(full_ratio) or float(full_ratio) >= 0.05:
                failures.append("endpoint-record edit-op 16f storage must stay below 0.05x full segment CSR")
            for key, value in (
                ("endpoint_forward_ms", endpoint_fwd_ms),
                ("edit_forward_ms", edit_fwd_ms),
                ("endpoint_vjp_ms", endpoint_vjp_ms),
                ("edit_vjp_ms", edit_vjp_ms),
            ):
                if not _finite_positive(value):
                    failures.append(f"endpoint-record edit-op timing {key} must be positive")
            if (
                _finite_positive(endpoint_fwd_ms)
                and _finite_positive(edit_fwd_ms)
                and float(edit_fwd_ms) <= float(endpoint_fwd_ms)
            ):
                failures.append("endpoint-record edit-op forward unexpectedly beat endpoint-run; audit before claiming")
        conclusion = str(endpoint_record_edit.get("conclusion", ""))
        if (
            "real Metal shader path" not in conclusion
            or "Storage is sublinear" not in conclusion
            or "slower than endpoint-run replay" not in conclusion
            or "not main-trainer integrated" not in conclusion
        ):
            failures.append("endpoint-record edit-op conclusion must keep shader, storage, speed, and trainer scope explicit")

    endpoint_record_edit_rgb_only = payload.get("endpoint_record_edit_rgb_only_replay")
    if not isinstance(endpoint_record_edit_rgb_only, dict) or endpoint_record_edit_rgb_only.get("available") is not True:
        failures.append("missing endpoint-record edit RGB-only replay sidecar")
    else:
        if endpoint_record_edit_rgb_only.get("completion_claim") is not False:
            failures.append("endpoint-record edit RGB-only replay must not claim completion")
        if endpoint_record_edit_rgb_only.get("star_uvt_competitive_claim") is not False:
            failures.append("endpoint-record edit RGB-only replay must not claim STAR UVT competitiveness")
        acceptance = endpoint_record_edit_rgb_only.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("endpoint-record edit RGB-only replay missing acceptance")
        else:
            for key in (
                "metal_forward_matches_endpoint_run",
                "metal_vjp_matches_endpoint_run",
                "metal_rgb_only_vjp_matches_full_zero_alpha_depth",
                "edit_ops_sublinear_vs_frames",
                "edit_storage_sublinear_vs_frames",
                "last_edit_storage_below_endpoint_csr",
            ):
                if acceptance.get(key) is not True:
                    failures.append(f"endpoint-record edit RGB-only replay acceptance {key} must be true")
        rgb_rel = endpoint_record_edit_rgb_only.get("max_rgb_only_vjp_rel_error_vs_full_zero_alpha_depth")
        if not _finite_nonnegative(rgb_rel) or float(rgb_rel) >= 1.0e-4:
            failures.append("endpoint-record edit RGB-only VJP relative error exceeds 1e-4")
        last_row = endpoint_record_edit_rgb_only.get("last_row")
        if not isinstance(last_row, dict):
            failures.append("endpoint-record edit RGB-only replay missing last_row")
        else:
            for key in ("edit_rgb_full_vjp_ms", "edit_rgb_only_vjp_ms", "edit_storage_vs_full_segment_csr"):
                if not _finite_positive(last_row.get(key)):
                    failures.append(f"endpoint-record edit RGB-only replay last_row.{key} is not positive finite")
            storage_ratio = last_row.get("edit_storage_vs_full_segment_csr")
            if _finite_positive(storage_ratio) and float(storage_ratio) >= 0.05:
                failures.append("endpoint-record edit RGB-only replay storage ratio no longer supports compact path")
        timing_read = endpoint_record_edit_rgb_only.get("rgb_only_timing_read")
        if not isinstance(timing_read, dict):
            failures.append("endpoint-record edit RGB-only replay missing timing read")
        else:
            for key in ("endpoint_vjp_ms_16f", "edit_full_vjp_ms_16f", "edit_rgb_full_vjp_ms_16f", "edit_rgb_only_vjp_ms_16f"):
                if not _finite_positive(timing_read.get(key)):
                    failures.append(f"endpoint-record edit RGB-only timing {key} is not positive finite")
        conclusion = str(endpoint_record_edit_rgb_only.get("conclusion", ""))
        if (
            "RGB-only VJP is numerically correct" not in conclusion
            or "not a stable speed win" not in conclusion
            or "STAR-UVT competitive result" not in conclusion
        ):
            failures.append("endpoint-record edit RGB-only conclusion must keep correctness and no-speed-claim scope")

    endpoint_record_edit_trackloop = payload.get("endpoint_record_edit_trackloop_replay")
    if not isinstance(endpoint_record_edit_trackloop, dict) or endpoint_record_edit_trackloop.get("available") is not True:
        failures.append("missing endpoint-record edit track-loop replay sidecar")
    else:
        if endpoint_record_edit_trackloop.get("completion_claim") is not False:
            failures.append("endpoint-record edit track-loop replay must not claim completion")
        if endpoint_record_edit_trackloop.get("star_uvt_competitive_claim") is not False:
            failures.append("endpoint-record edit track-loop replay must not claim STAR UVT competitiveness")
        acceptance = endpoint_record_edit_trackloop.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("endpoint-record edit track-loop replay missing acceptance")
        else:
            for key in (
                "metal_forward_matches_endpoint_run",
                "metal_vjp_matches_endpoint_run",
                "metal_trackloop_forward_matches_endpoint_run",
                "edit_ops_sublinear_vs_frames",
                "edit_storage_sublinear_vs_frames",
                "last_edit_storage_below_endpoint_csr",
            ):
                if acceptance.get(key) is not True:
                    failures.append(f"endpoint-record edit track-loop replay acceptance {key} must be true")
        trackloop_error = endpoint_record_edit_trackloop.get("max_trackloop_forward_abs_error_vs_endpoint_run")
        if not _finite_nonnegative(trackloop_error) or float(trackloop_error) >= 1.0e-4:
            failures.append("endpoint-record edit track-loop forward error exceeds 1e-4")
        last_row = endpoint_record_edit_trackloop.get("last_row")
        if not isinstance(last_row, dict):
            failures.append("endpoint-record edit track-loop replay missing last_row")
        else:
            for key in (
                "endpoint_forward_ms",
                "edit_forward_ms",
                "edit_trackloop_forward_ms",
                "edit_storage_vs_full_segment_csr",
            ):
                if not _finite_positive(last_row.get(key)):
                    failures.append(f"endpoint-record edit track-loop replay last_row.{key} is not positive finite")
            endpoint_ms = last_row.get("endpoint_forward_ms")
            trackloop_ms = last_row.get("edit_trackloop_forward_ms")
            if (
                _finite_positive(endpoint_ms)
                and _finite_positive(trackloop_ms)
                and float(trackloop_ms) <= float(endpoint_ms)
            ):
                failures.append("endpoint-record edit track-loop unexpectedly beat endpoint-run; audit before claiming")
        timing_read = endpoint_record_edit_trackloop.get("trackloop_timing_read")
        if not isinstance(timing_read, dict):
            failures.append("endpoint-record edit track-loop replay missing timing read")
        else:
            for key in ("endpoint_forward_ms_16f", "edit_forward_ms_16f", "edit_trackloop_forward_ms_16f"):
                if not _finite_positive(timing_read.get(key)):
                    failures.append(f"endpoint-record edit track-loop timing {key} is not positive finite")
        conclusion = str(endpoint_record_edit_trackloop.get("conclusion", ""))
        if (
            "numerically correct" not in conclusion
            or "not a speed win" not in conclusion
            or "row-replay bottleneck" not in conclusion
            or "STAR-UVT competitive result" not in conclusion
        ):
            failures.append("endpoint-record edit track-loop conclusion must keep correctness and rejected-speed scope")

    endpoint_record_edit_block4 = payload.get("endpoint_record_edit_block4_replay")
    if not isinstance(endpoint_record_edit_block4, dict) or endpoint_record_edit_block4.get("available") is not True:
        failures.append("missing endpoint-record edit block4 replay sidecar")
    else:
        if endpoint_record_edit_block4.get("completion_claim") is not False:
            failures.append("endpoint-record edit block4 replay must not claim completion")
        if endpoint_record_edit_block4.get("star_uvt_competitive_claim") is not False:
            failures.append("endpoint-record edit block4 replay must not claim STAR UVT competitiveness")
        acceptance = endpoint_record_edit_block4.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("endpoint-record edit block4 replay missing acceptance")
        else:
            for key in (
                "metal_forward_matches_endpoint_run",
                "metal_vjp_matches_endpoint_run",
                "metal_block4_forward_matches_endpoint_run",
                "metal_block4_rgb_only_vjp_matches_edit_rgb_only",
                "metal_block4_rgb_only_vjp_matches_full_zero_alpha_depth",
                "edit_ops_sublinear_vs_frames",
                "edit_storage_sublinear_vs_frames",
                "last_edit_storage_below_endpoint_csr",
            ):
                if acceptance.get(key) is not True:
                    failures.append(f"endpoint-record edit block4 replay acceptance {key} must be true")
        block4_error = endpoint_record_edit_block4.get("max_block4_forward_abs_error_vs_endpoint_run")
        if not _finite_nonnegative(block4_error) or float(block4_error) >= 1.0e-4:
            failures.append("endpoint-record edit block4 forward error exceeds 1e-4")
        last_row = endpoint_record_edit_block4.get("last_row")
        if not isinstance(last_row, dict):
            failures.append("endpoint-record edit block4 replay missing last_row")
        else:
            for key in (
                "endpoint_forward_ms",
                "edit_forward_ms",
                "edit_block4_forward_ms",
                "block4_storage_vs_full_segment_csr",
                "block4_storage_vs_endpoint_csr",
            ):
                if not _finite_positive(last_row.get(key)):
                    failures.append(f"endpoint-record edit block4 replay last_row.{key} is not positive finite")
            endpoint_ms = last_row.get("endpoint_forward_ms")
            edit_ms = last_row.get("edit_forward_ms")
            block4_ms = last_row.get("edit_block4_forward_ms")
            if _finite_positive(edit_ms) and _finite_positive(block4_ms) and float(block4_ms) >= float(edit_ms):
                failures.append("endpoint-record edit block4 replay should beat original edit forward in saved 16f sidecar")
            storage_full = last_row.get("block4_storage_vs_full_segment_csr")
            storage_endpoint = last_row.get("block4_storage_vs_endpoint_csr")
            if _finite_positive(storage_full) and float(storage_full) >= 0.05:
                failures.append("endpoint-record edit block4 replay must stay below 0.05x full segment CSR")
            if _finite_positive(storage_endpoint) and float(storage_endpoint) >= 0.50:
                failures.append("endpoint-record edit block4 replay must stay below half endpoint CSR")
        timing_read = endpoint_record_edit_block4.get("block4_timing_read")
        if not isinstance(timing_read, dict):
            failures.append("endpoint-record edit block4 replay missing timing read")
        else:
            for key in (
                "endpoint_forward_ms_16f",
                "edit_forward_ms_16f",
                "edit_block4_forward_ms_16f",
                "edit_block4_rgb_only_vjp_ms_16f",
            ):
                if not _finite_positive(timing_read.get(key)):
                    failures.append(f"endpoint-record edit block4 timing {key} is not positive finite")
        conclusion = str(endpoint_record_edit_block4.get("conclusion", ""))
        if (
            "numerically correct" not in conclusion
            or "beats the original edit replay at 16f" not in conclusion
            or "slightly slower than endpoint-run forward" not in conclusion
            or "dedicated block4 RGB-only VJP" not in conclusion
            or "not a main-trainer integration" not in conclusion
            or "STAR-UVT competitive claim" not in conclusion
        ):
            failures.append("endpoint-record edit block4 conclusion must keep positive-forward and scoped-incomplete status")

    endpoint_record_edit_block_coeff = payload.get("endpoint_record_edit_block_coeff_replay")
    if (
        not isinstance(endpoint_record_edit_block_coeff, dict)
        or endpoint_record_edit_block_coeff.get("available") is not True
    ):
        failures.append("missing endpoint-record edit coefficient-cached forward sidecar")
    else:
        if endpoint_record_edit_block_coeff.get("completion_claim") is not False:
            failures.append("endpoint-record edit coefficient-cached sidecar must not claim completion")
        if endpoint_record_edit_block_coeff.get("star_uvt_competitive_claim") is not False:
            failures.append("endpoint-record edit coefficient-cached sidecar must not claim STAR UVT competitiveness")
        acceptance = endpoint_record_edit_block_coeff.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("endpoint-record edit coefficient-cached sidecar missing acceptance")
        else:
            for key in (
                "metal_block_coeff_forward_matches_endpoint_run",
                "metal_block_coeff_rgb_only_vjp_matches_full_zero_alpha_depth",
                "metal_block_coeff_rgb_only_vjp_matches_edit_rgb_only",
                "metal_block_coeff_rgb_only_vjp_matches_block4_rgb_only",
            ):
                if acceptance.get(key) is not True:
                    failures.append(f"coefficient-cached sidecar acceptance {key} must be true")
        structural = endpoint_record_edit_block_coeff.get("structural_read")
        if not isinstance(structural, dict) or structural.get("not_main_trainer_integration") is not True:
            failures.append("coefficient-cached forward sidecar must preserve not-main-trainer scope")
        coeff_error = endpoint_record_edit_block_coeff.get("max_block_coeff_forward_abs_error_vs_endpoint_run")
        if not _finite_nonnegative(coeff_error) or float(coeff_error) >= 1.0e-4:
            failures.append("coefficient-cached forward error exceeds 1e-4")
        for key in (
            "max_block_coeff_rgb_only_vjp_rel_error_vs_full_zero_alpha_depth",
            "max_block_coeff_rgb_only_vjp_rel_error_vs_edit_rgb_only",
            "max_block_coeff_rgb_only_vjp_rel_error_vs_block4_rgb_only",
        ):
            value = endpoint_record_edit_block_coeff.get(key)
            if not _finite_nonnegative(value) or float(value) >= 1.0e-4:
                failures.append(f"coefficient-cached RGB-only VJP error {key} exceeds 1e-4")
        last_row = endpoint_record_edit_block_coeff.get("last_row")
        if not isinstance(last_row, dict):
            failures.append("coefficient-cached forward sidecar missing last_row")
        else:
            for key in (
                "endpoint_forward_ms",
                "edit_block4_forward_ms",
                "edit_block_coeff_forward_ms",
                "edit_forward_ms",
                "edit_block_coeff_rgb_only_vjp_ms",
                "block_coeff_storage_vs_endpoint_csr",
                "block_coeff_storage_vs_full_segment_csr",
            ):
                if not _finite_positive(last_row.get(key)):
                    failures.append(f"coefficient-cached forward sidecar last_row.{key} is not positive finite")
            endpoint_ms = last_row.get("endpoint_forward_ms")
            block4_ms = last_row.get("edit_block4_forward_ms")
            coeff_ms = last_row.get("edit_block_coeff_forward_ms")
            if _finite_positive(endpoint_ms) and _finite_positive(coeff_ms) and float(coeff_ms) >= float(endpoint_ms):
                failures.append("coefficient-cached 16f forward should beat endpoint-run in saved sidecar")
            edit_ms = last_row.get("edit_forward_ms")
            if _finite_positive(edit_ms) and _finite_positive(coeff_ms) and float(coeff_ms) >= float(edit_ms):
                failures.append("coefficient-cached 16f forward should beat original edit replay in saved sidecar")
            storage_endpoint = last_row.get("block_coeff_storage_vs_endpoint_csr")
            storage_full = last_row.get("block_coeff_storage_vs_full_segment_csr")
            if not _finite_positive(storage_endpoint) or float(storage_endpoint) <= 1.0:
                failures.append("coefficient-cached sidecar must preserve that coefficient storage is above endpoint CSR")
            if not _finite_positive(storage_full) or float(storage_full) >= 0.25:
                failures.append("coefficient-cached 16f storage should stay below 0.25x full segment CSR")
        if endpoint_record_edit_block_coeff.get("speed_read") not in {
            "faster_than_endpoint_and_edit_forward",
            "faster_than_endpoint_edit_and_block4_forward",
        }:
            failures.append("coefficient-cached sidecar must record the 16f forward speed win explicitly")
        sweep = endpoint_record_edit_block_coeff.get("sweep")
        if not isinstance(sweep, dict) or sweep.get("available") is not True:
            failures.append("coefficient-cached sidecar missing 2/4/8 sweep evidence")
        else:
            acceptance = sweep.get("acceptance")
            if not isinstance(acceptance, dict):
                failures.append("coefficient-cached sweep missing acceptance")
            elif acceptance.get("metal_block_coeff_forward_matches_endpoint_run") is not True:
                failures.append("coefficient-cached sweep must preserve forward correctness")
            for key in (
                "frame_scale_first_to_last",
                "edit_op_scale_first_to_last",
                "edit_storage_scale_first_to_last",
                "block_edit_storage_scale_first_to_last",
                "endpoint_run_scale_first_to_last",
                "max_block_coeff_forward_abs_error_vs_endpoint_run",
            ):
                if not _finite_positive(sweep.get(key)):
                    failures.append(f"coefficient-cached sweep {key} is not positive finite")
            for key in (
                "max_block_coeff_rgb_only_vjp_rel_error_vs_full_zero_alpha_depth",
                "max_block_coeff_rgb_only_vjp_rel_error_vs_edit_rgb_only",
                "max_block_coeff_rgb_only_vjp_rel_error_vs_block4_rgb_only",
            ):
                value = sweep.get(key)
                if not _finite_nonnegative(value) or float(value) >= 1.0e-4:
                    failures.append(f"coefficient-cached sweep RGB-only VJP error {key} exceeds 1e-4")
            frame_scale = sweep.get("frame_scale_first_to_last")
            block_storage_scale = sweep.get("block_edit_storage_scale_first_to_last")
            if (
                _finite_positive(frame_scale)
                and _finite_positive(block_storage_scale)
                and float(block_storage_scale) >= float(frame_scale)
            ):
                failures.append("coefficient-cached sweep block edit storage scale must remain sublinear")
            rows = sweep.get("rows")
            if not isinstance(rows, list) or len(rows) < 3:
                failures.append("coefficient-cached sweep should include 2/4/8 row summaries")
        conclusion = str(endpoint_record_edit_block_coeff.get("conclusion", ""))
        if (
            "Coefficient-cached block edit forward replay is numerically correct" not in conclusion
            or "speed-positive" not in conclusion
            or "RGB-only coeff VJP" not in conclusion
            or "warmed 2/4/8/16 sidecar smoke" not in conclusion
            or "above endpoint CSR storage" not in conclusion
            or "not a STAR-UVT competitive claim" not in conclusion
        ):
            failures.append("coefficient-cached conclusion must keep correctness, speed, storage, and scope explicit")

    endpoint_record_edit_block_coeff_train_eval = payload.get("endpoint_record_edit_block_coeff_rgb_train_eval")
    if (
        not isinstance(endpoint_record_edit_block_coeff_train_eval, dict)
        or endpoint_record_edit_block_coeff_train_eval.get("available") is not True
    ):
        failures.append("missing endpoint-record edit coefficient-cached RGB train/eval smoke")
    else:
        if endpoint_record_edit_block_coeff_train_eval.get("status") != "ok":
            failures.append("coefficient-cached train/eval smoke must be green")
        if endpoint_record_edit_block_coeff_train_eval.get("tape_mode") != "endpoint-record-edit-block-coeff":
            failures.append("coefficient-cached train/eval must record tape_mode='endpoint-record-edit-block-coeff'")
        if endpoint_record_edit_block_coeff_train_eval.get("optimizer_mode") != "autograd":
            failures.append("coefficient-cached train/eval must use optimizer_mode='autograd'")
        if endpoint_record_edit_block_coeff_train_eval.get("full_trainer_claim") is not False:
            failures.append("coefficient-cached train/eval must not claim full trainer coverage")
        if endpoint_record_edit_block_coeff_train_eval.get("full_geometry_gradient_claim") is not False:
            failures.append("coefficient-cached train/eval must not claim geometry-gradient coverage")
        if endpoint_record_edit_block_coeff_train_eval.get("density_independent_depth_claim") is not True:
            failures.append("coefficient-cached train/eval should preserve continuous endpoint-depth semantics")
        if endpoint_record_edit_block_coeff_train_eval.get("continuous_absorption_depth_semantic") is not True:
            failures.append("coefficient-cached train/eval should mark continuous absorption endpoint depth")
        if endpoint_record_edit_block_coeff_train_eval.get("frame_counts") != [2, 4, 8, 16]:
            failures.append("coefficient-cached train/eval should preserve the 2/4/8/16 smoke scale")
        acceptance = endpoint_record_edit_block_coeff_train_eval.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("coefficient-cached train/eval missing acceptance")
        else:
            for key in (
                "all_rows_ok",
                "total_step_sublinear_vs_frames",
                "render_sublinear_vs_frames",
                "backward_sublinear_vs_frames",
                "selected_tape_storage_below_full_at_max_frame",
            ):
                if acceptance.get(key) is not True:
                    failures.append(f"coefficient-cached train/eval acceptance {key} must be true")
        for key in (
            "frame_scale_first_to_last",
            "total_step_scale_first_to_last",
            "render_scale_first_to_last",
            "backward_scale_first_to_last",
            "selected_tape_segment_scale_first_to_last",
            "selected_tape_storage_scale_first_to_last",
            "endpoint_record_edit_op_scale_first_to_last",
        ):
            if not _finite_positive(endpoint_record_edit_block_coeff_train_eval.get(key)):
                failures.append(f"coefficient-cached train/eval {key} is not positive finite")
        frame_scale = endpoint_record_edit_block_coeff_train_eval.get("frame_scale_first_to_last")
        for key in ("total_step_scale_first_to_last", "render_scale_first_to_last", "backward_scale_first_to_last"):
            value = endpoint_record_edit_block_coeff_train_eval.get(key)
            if _finite_positive(frame_scale) and _finite_positive(value) and float(value) >= float(frame_scale):
                failures.append(f"coefficient-cached train/eval {key} must stay sublinear versus frame count")
        last_row = endpoint_record_edit_block_coeff_train_eval.get("last_row")
        if not isinstance(last_row, dict):
            failures.append("coefficient-cached train/eval missing last_row")
        else:
            for key in (
                "total_ms",
                "render_ms",
                "backward_ms",
                "final_train_psnr",
                "final_heldout_psnr",
                "first_grad_abs_sum",
                "parameter_update_abs_max",
            ):
                if not _finite_positive(last_row.get(key)):
                    failures.append(f"coefficient-cached train/eval last_row.{key} is not positive finite")
            if last_row.get("steps") != 20 or last_row.get("warmup_steps") != 5:
                failures.append("coefficient-cached train/eval should preserve the 20-step/5-warmup sweep")
            selected_storage = last_row.get("train_selected_tape_storage_vs_full")
            block4_storage_full = last_row.get("train_endpoint_record_block4_storage_vs_full")
            block4_storage_endpoint = last_row.get("train_endpoint_record_block4_storage_vs_endpoint_run")
            if not _finite_positive(selected_storage) or float(selected_storage) >= 0.25:
                failures.append("coefficient-cached train/eval selected storage should stay below 0.25x full CSR")
            if not _finite_positive(block4_storage_full) or float(block4_storage_full) >= 0.05:
                failures.append("coefficient-cached train/eval block4 base storage should stay below 0.05x full CSR")
            if not _finite_positive(block4_storage_endpoint) or float(block4_storage_endpoint) >= 0.50:
                failures.append("coefficient-cached train/eval block4 base storage should stay below half endpoint CSR")
        first_row = endpoint_record_edit_block_coeff_train_eval.get("first_row")
        if not isinstance(first_row, dict):
            failures.append("coefficient-cached train/eval missing first_row")
        else:
            if first_row.get("steps") != 20 or first_row.get("warmup_steps") != 5:
                failures.append("coefficient-cached train/eval first row should preserve 20-step/5-warmup settings")
            tiny_frame_storage = first_row.get("train_selected_tape_storage_vs_full")
            if not _finite_positive(tiny_frame_storage) or float(tiny_frame_storage) <= 1.0:
                failures.append("coefficient-cached train/eval should preserve tiny-frame storage overhead nuance")
        conclusion = str(endpoint_record_edit_block_coeff_train_eval.get("conclusion", ""))
        if (
            "green 20-step render32 2/4/8/16 autograd sweep" not in conclusion
            or "gradients flow" not in conclusion
            or "storage-heavy at tiny frame counts" not in conclusion
            or "not a full stable speed benchmark" not in conclusion
            or "main-trainer" not in conclusion
            or "STAR-UVT" not in conclusion
        ):
            failures.append("coefficient-cached train/eval conclusion must keep repeat20, storage, and noncompetitive scope")

    owner_run_train_eval = payload.get("owner_run_rgb_train_eval")
    if not isinstance(owner_run_train_eval, dict) or owner_run_train_eval.get("available") is not True:
        failures.append("missing owner-run RGB train/eval artifact")
    else:
        acceptance = owner_run_train_eval.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("owner-run RGB train/eval missing acceptance")
        else:
            for key in (
                "all_rows_ok",
                "total_step_sublinear_vs_frames",
                "render_sublinear_vs_frames",
                "backward_sublinear_vs_frames",
                "owner_run_segments_below_full_at_max_frame",
            ):
                if acceptance.get(key) is not True:
                    failures.append(f"owner-run RGB train/eval acceptance {key} must be true")
        for key in (
            "frame_scale_first_to_last",
            "total_step_scale_first_to_last",
            "render_scale_first_to_last",
            "backward_scale_first_to_last",
            "owner_run_segment_scale_first_to_last",
        ):
            if not _finite_positive(owner_run_train_eval.get(key)):
                failures.append(f"owner_run_rgb_train_eval.{key} is not positive finite")
        frame_scale = owner_run_train_eval.get("frame_scale_first_to_last")
        total_scale = owner_run_train_eval.get("total_step_scale_first_to_last")
        if _finite_positive(frame_scale) and _finite_positive(total_scale) and float(total_scale) >= float(frame_scale):
            failures.append("owner-run RGB train/eval total step scale is not sublinear versus frame count")
        if owner_run_train_eval.get("full_trainer_claim") is not False:
            failures.append("owner-run RGB train/eval must not claim full trainer coverage")
        if owner_run_train_eval.get("full_geometry_gradient_claim") is not False:
            failures.append("owner-run RGB train/eval must not claim geometry-gradient coverage")
        if owner_run_train_eval.get("density_independent_depth_claim") is not False:
            failures.append("owner-run RGB train/eval must not claim density-independent depth coverage")
        if owner_run_train_eval.get("optimizer_mode") != "autograd":
            failures.append("owner-run RGB train/eval primary artifact must use optimizer_mode='autograd'")
        if owner_run_train_eval.get("segment_tape_vjp_mode") != "direct_atomic_grad_only":
            failures.append("owner-run RGB train/eval primary artifact must use direct_atomic_grad_only VJP")
        last_row = owner_run_train_eval.get("last_row")
        if not isinstance(last_row, dict):
            failures.append("owner-run RGB train/eval missing last_row")
        else:
            for key in ("total_ms", "render_ms", "backward_ms", "final_train_psnr", "final_heldout_psnr"):
                if not _finite_positive(last_row.get(key)):
                    failures.append(f"owner-run RGB train/eval last_row.{key} is not positive finite")
            segment_ratio = last_row.get("train_owner_run_segments_vs_full")
            storage_ratio = last_row.get("train_owner_run_storage_vs_full")
            if not _finite_positive(segment_ratio) or float(segment_ratio) >= 0.10:
                failures.append("owner-run RGB train/eval 16f segment ratio no longer supports compression claim")
            if not _finite_positive(storage_ratio) or float(storage_ratio) >= 0.10:
                failures.append("owner-run RGB train/eval 16f storage ratio no longer supports compression claim")
            if (
                isinstance(winner, dict)
                and _finite_positive(winner.get("heldout_psnr_16f"))
                and _finite_positive(last_row.get("final_heldout_psnr"))
                and abs(float(last_row["final_heldout_psnr"]) - float(winner["heldout_psnr_16f"])) > 1.0e-3
            ):
                failures.append("owner-run RGB train/eval 16f heldout PSNR no longer matches fused winner")
        comparison = owner_run_train_eval.get("comparison_to_fused_winner")
        if not isinstance(comparison, dict):
            failures.append("owner-run RGB train/eval missing fused-winner comparison")
        else:
            ratio = comparison.get("owner_run_to_fused_winner_16f_total_ratio")
            if not _finite_positive(ratio) or float(ratio) >= 1.0:
                failures.append("owner-run RGB train/eval no longer beats fused winner at 16f in saved artifact")
            note = str(comparison.get("scope_note", ""))
            if "autograd wrapper" not in note or "isolated" not in note or "main fused-slab trainer" not in note:
                failures.append("owner-run RGB train/eval comparison must keep integration scope explicit")
        conclusion = str(owner_run_train_eval.get("conclusion", ""))
        if (
            "segment-tape autograd wrapper" not in conclusion
            or "faster than the current fused-slab winner" not in conclusion
            or "no-density-independent-depth" not in conclusion
        ):
            failures.append("owner-run RGB train/eval conclusion must keep practical speed and scope explicit")

    active_internal_train_eval = payload.get("active_internal_rgb_train_eval")
    if not isinstance(active_internal_train_eval, dict) or active_internal_train_eval.get("available") is not True:
        failures.append("missing active-internal RGB train/eval artifact")
    else:
        acceptance = active_internal_train_eval.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("active-internal RGB train/eval missing acceptance")
        else:
            for key in (
                "all_rows_ok",
                "total_step_sublinear_vs_frames",
                "render_sublinear_vs_frames",
                "backward_sublinear_vs_frames",
                "selected_tape_segments_below_full_at_max_frame",
            ):
                if acceptance.get(key) is not True:
                    failures.append(f"active-internal RGB train/eval acceptance {key} must be true")
        if active_internal_train_eval.get("tape_mode") != "active-internal":
            failures.append("active-internal RGB train/eval must record tape_mode='active-internal'")
        if active_internal_train_eval.get("full_trainer_claim") is not False:
            failures.append("active-internal RGB train/eval must not claim full trainer coverage")
        if active_internal_train_eval.get("full_geometry_gradient_claim") is not False:
            failures.append("active-internal RGB train/eval must not claim geometry-gradient coverage")
        if active_internal_train_eval.get("density_independent_depth_claim") is not False:
            failures.append("active-internal RGB train/eval must not claim density-independent depth")
        for key in (
            "frame_scale_first_to_last",
            "total_step_scale_first_to_last",
            "render_scale_first_to_last",
            "backward_scale_first_to_last",
            "selected_tape_segment_scale_first_to_last",
        ):
            if not _finite_positive(active_internal_train_eval.get(key)):
                failures.append(f"active_internal_rgb_train_eval.{key} is not positive finite")
        frame_scale = active_internal_train_eval.get("frame_scale_first_to_last")
        selected_scale = active_internal_train_eval.get("selected_tape_segment_scale_first_to_last")
        if _finite_positive(frame_scale) and _finite_positive(selected_scale) and float(selected_scale) <= float(frame_scale):
            failures.append("active-internal selected segment scale unexpectedly looks structurally sublinear")
        last_row = active_internal_train_eval.get("last_row")
        if not isinstance(last_row, dict):
            failures.append("active-internal RGB train/eval missing last_row")
        else:
            for key in ("total_ms", "render_ms", "backward_ms", "final_train_psnr", "final_heldout_psnr"):
                if not _finite_positive(last_row.get(key)):
                    failures.append(f"active-internal RGB train/eval last_row.{key} is not positive finite")
            storage_ratio = last_row.get("train_selected_tape_storage_vs_full")
            segment_ratio = last_row.get("train_selected_tape_segments_vs_full")
            if not _finite_positive(storage_ratio) or float(storage_ratio) >= 0.25:
                failures.append("active-internal RGB train/eval 16f storage ratio no longer supports compact path")
            if not _finite_positive(segment_ratio) or float(segment_ratio) >= 0.25:
                failures.append("active-internal RGB train/eval 16f segment ratio no longer supports compact path")
            if (
                isinstance(winner, dict)
                and _finite_positive(winner.get("heldout_psnr_16f"))
                and _finite_positive(last_row.get("final_heldout_psnr"))
                and abs(float(last_row["final_heldout_psnr"]) - float(winner["heldout_psnr_16f"])) > 1.0e-3
            ):
                failures.append("active-internal RGB train/eval 16f heldout PSNR no longer matches fused winner")
        comparison = active_internal_train_eval.get("comparison")
        if not isinstance(comparison, dict):
            failures.append("active-internal RGB train/eval missing comparison block")
        else:
            fused_ratio = comparison.get("active_internal_to_fused_winner_16f_total_ratio")
            owner_ratio = comparison.get("active_internal_to_owner_run_16f_total_ratio")
            if not _finite_positive(fused_ratio) or float(fused_ratio) >= 1.0:
                failures.append("active-internal RGB train/eval no longer beats fused winner at 16f")
            if not _finite_positive(owner_ratio) or float(owner_ratio) <= 1.0:
                failures.append("active-internal RGB train/eval should remain slower than owner-run at 16f")
            note = str(comparison.get("scope_note", ""))
            if (
                "exact current-density" not in note
                or "density-independent replay" not in note
                or "main-trainer integration" not in note
            ):
                failures.append("active-internal RGB train/eval comparison must keep density/trainer scope explicit")
        conclusion = str(active_internal_train_eval.get("conclusion", ""))
        if (
            "exact-current-depth" not in conclusion
            or "slower than owner-run" not in conclusion
            or "scales worse than frame count" not in conclusion
        ):
            failures.append("active-internal RGB train/eval conclusion must keep depth, owner-run, and scaling tradeoffs explicit")

    full_tape_train_eval = payload.get("full_tape_rgb_train_eval")
    if not isinstance(full_tape_train_eval, dict) or full_tape_train_eval.get("available") is not True:
        failures.append("missing full-tape RGB train/eval artifact")
    else:
        acceptance = full_tape_train_eval.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("full-tape RGB train/eval missing acceptance")
        else:
            for key in (
                "all_rows_ok",
                "total_step_sublinear_vs_frames",
                "render_sublinear_vs_frames",
                "backward_sublinear_vs_frames",
                "selected_tape_segments_below_full_at_max_frame",
            ):
                if acceptance.get(key) is not True:
                    failures.append(f"full-tape RGB train/eval acceptance {key} must be true")
        if full_tape_train_eval.get("tape_mode") != "full":
            failures.append("full-tape RGB train/eval must record tape_mode='full'")
        if full_tape_train_eval.get("full_trainer_claim") is not False:
            failures.append("full-tape RGB train/eval must not claim full trainer coverage")
        if full_tape_train_eval.get("full_geometry_gradient_claim") is not False:
            failures.append("full-tape RGB train/eval must not claim geometry-gradient coverage")
        for key in (
            "frame_scale_first_to_last",
            "total_step_scale_first_to_last",
            "render_scale_first_to_last",
            "backward_scale_first_to_last",
            "selected_tape_segment_scale_first_to_last",
        ):
            if not _finite_positive(full_tape_train_eval.get(key)):
                failures.append(f"full_tape_rgb_train_eval.{key} is not positive finite")
        frame_scale = full_tape_train_eval.get("frame_scale_first_to_last")
        selected_scale = full_tape_train_eval.get("selected_tape_segment_scale_first_to_last")
        if _finite_positive(frame_scale) and _finite_positive(selected_scale) and float(selected_scale) <= float(frame_scale):
            failures.append("full-tape selected segment scale unexpectedly looks structurally sublinear")
        last_row = full_tape_train_eval.get("last_row")
        if not isinstance(last_row, dict):
            failures.append("full-tape RGB train/eval missing last_row")
        else:
            for key in ("total_ms", "render_ms", "backward_ms", "final_train_psnr", "final_heldout_psnr"):
                if not _finite_positive(last_row.get(key)):
                    failures.append(f"full-tape RGB train/eval last_row.{key} is not positive finite")
            storage_ratio = last_row.get("train_selected_tape_storage_vs_full")
            segment_ratio = last_row.get("train_selected_tape_segments_vs_full")
            if not _finite_positive(storage_ratio) or abs(float(storage_ratio) - 1.0) > 1.0e-6:
                failures.append("full-tape RGB train/eval selected storage ratio must remain 1.0")
            if not _finite_positive(segment_ratio) or abs(float(segment_ratio) - 1.0) > 1.0e-6:
                failures.append("full-tape RGB train/eval selected segment ratio must remain 1.0")
            if (
                isinstance(winner, dict)
                and _finite_positive(winner.get("heldout_psnr_16f"))
                and _finite_positive(last_row.get("final_heldout_psnr"))
                and abs(float(last_row["final_heldout_psnr"]) - float(winner["heldout_psnr_16f"])) > 1.0e-3
            ):
                failures.append("full-tape RGB train/eval 16f heldout PSNR no longer matches fused winner")
        comparison = full_tape_train_eval.get("comparison")
        if not isinstance(comparison, dict):
            failures.append("full-tape RGB train/eval missing comparison block")
        else:
            fused_ratio = comparison.get("full_tape_to_fused_winner_16f_total_ratio")
            owner_ratio = comparison.get("full_tape_to_owner_run_16f_total_ratio")
            active_ratio = comparison.get("full_tape_to_active_internal_16f_total_ratio")
            if not _finite_positive(fused_ratio) or float(fused_ratio) <= 1.0:
                failures.append("full-tape RGB train/eval should remain slower than fused winner at 16f")
            if not _finite_positive(owner_ratio) or float(owner_ratio) <= 1.0:
                failures.append("full-tape RGB train/eval should remain slower than owner-run at 16f")
            if not _finite_positive(active_ratio) or float(active_ratio) <= 1.0:
                failures.append("full-tape RGB train/eval should remain slower than active-internal at 16f")
            note = str(comparison.get("scope_note", ""))
            if (
                "density-independent replay cost baseline" not in note
                or "not a compact STAR-like structure" not in note
                or "main-trainer integration" not in note
            ):
                failures.append("full-tape RGB train/eval comparison must keep density/trainer/compactness scope explicit")
        conclusion = str(full_tape_train_eval.get("conclusion", ""))
        if (
            "exact density-independent" not in conclusion
            or "slower than owner-run and active-internal" not in conclusion
            or "ratio is 1.0" not in conclusion
        ):
            failures.append("full-tape RGB train/eval conclusion must keep exact replay and storage tradeoff explicit")

    endpoint_run_train_eval = payload.get("endpoint_run_rgb_train_eval")
    if not isinstance(endpoint_run_train_eval, dict) or endpoint_run_train_eval.get("available") is not True:
        failures.append("missing endpoint-run RGB train/eval artifact")
    else:
        acceptance = endpoint_run_train_eval.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("endpoint-run RGB train/eval missing acceptance")
        else:
            for key in (
                "all_rows_ok",
                "total_step_sublinear_vs_frames",
                "render_sublinear_vs_frames",
                "backward_sublinear_vs_frames",
                "selected_tape_segments_below_full_at_max_frame",
            ):
                if acceptance.get(key) is not True:
                    failures.append(f"endpoint-run RGB train/eval acceptance {key} must be true")
        if endpoint_run_train_eval.get("tape_mode") != "endpoint-run":
            failures.append("endpoint-run RGB train/eval must record tape_mode='endpoint-run'")
        if endpoint_run_train_eval.get("full_trainer_claim") is not False:
            failures.append("endpoint-run RGB train/eval must not claim full trainer coverage")
        if endpoint_run_train_eval.get("full_geometry_gradient_claim") is not False:
            failures.append("endpoint-run RGB train/eval must not claim geometry-gradient coverage")
        if endpoint_run_train_eval.get("density_independent_depth_claim") is not True:
            failures.append("endpoint-run RGB train/eval must claim density independence only for endpoint semantic")
        if endpoint_run_train_eval.get("continuous_absorption_depth_semantic") is not True:
            failures.append("endpoint-run RGB train/eval must mark continuous absorption semantic")
        for key in (
            "frame_scale_first_to_last",
            "total_step_scale_first_to_last",
            "render_scale_first_to_last",
            "backward_scale_first_to_last",
            "selected_tape_segment_scale_first_to_last",
        ):
            if not _finite_positive(endpoint_run_train_eval.get(key)):
                failures.append(f"endpoint_run_rgb_train_eval.{key} is not positive finite")
        frame_scale = endpoint_run_train_eval.get("frame_scale_first_to_last")
        selected_scale = endpoint_run_train_eval.get("selected_tape_segment_scale_first_to_last")
        if _finite_positive(frame_scale) and _finite_positive(selected_scale) and float(selected_scale) <= float(frame_scale):
            failures.append("endpoint-run selected run scale unexpectedly looks structurally sublinear")
        last_row = endpoint_run_train_eval.get("last_row")
        if not isinstance(last_row, dict):
            failures.append("endpoint-run RGB train/eval missing last_row")
        else:
            for key in ("total_ms", "render_ms", "backward_ms", "final_train_psnr", "final_heldout_psnr"):
                if not _finite_positive(last_row.get(key)):
                    failures.append(f"endpoint-run RGB train/eval last_row.{key} is not positive finite")
            storage_ratio = last_row.get("train_selected_tape_storage_vs_full")
            segment_ratio = last_row.get("train_selected_tape_segments_vs_full")
            if not _finite_positive(storage_ratio) or float(storage_ratio) >= 0.20:
                failures.append("endpoint-run RGB train/eval 16f storage ratio no longer supports compact path")
            if not _finite_positive(segment_ratio) or float(segment_ratio) >= 0.20:
                failures.append("endpoint-run RGB train/eval 16f segment ratio no longer supports compact path")
            if (
                isinstance(winner, dict)
                and _finite_positive(winner.get("heldout_psnr_16f"))
                and _finite_positive(last_row.get("final_heldout_psnr"))
                and abs(float(last_row["final_heldout_psnr"]) - float(winner["heldout_psnr_16f"])) > 1.0e-3
            ):
                failures.append("endpoint-run RGB train/eval 16f heldout PSNR no longer matches fused winner")
        comparison = endpoint_run_train_eval.get("comparison")
        if not isinstance(comparison, dict):
            failures.append("endpoint-run RGB train/eval missing comparison block")
        else:
            fused_ratio = comparison.get("endpoint_run_to_fused_winner_16f_total_ratio")
            owner_ratio = comparison.get("endpoint_run_to_owner_run_16f_total_ratio")
            active_ratio = comparison.get("endpoint_run_to_active_internal_16f_total_ratio")
            if not _finite_positive(fused_ratio) or float(fused_ratio) >= 1.0:
                failures.append("endpoint-run RGB train/eval should beat fused winner at 16f")
            if not _finite_positive(owner_ratio) or float(owner_ratio) <= 1.0:
                failures.append("endpoint-run RGB train/eval should remain slower than current-density owner-run")
            if not _finite_positive(active_ratio) or float(active_ratio) >= 1.0:
                failures.append("endpoint-run RGB train/eval should beat active-internal at 16f")
            note = str(comparison.get("scope_note", ""))
            if (
                "continuous-absorption endpoint shader" not in note
                or "not a drop-in replacement" not in note
                or "segment-mid depth" not in note
            ):
                failures.append("endpoint-run RGB train/eval comparison must keep semantic-change scope explicit")
        conclusion = str(endpoint_run_train_eval.get("conclusion", ""))
        if (
            "compact density-independent semantic-change path" not in conclusion
            or "slightly slower" not in conclusion
            or "not STAR-like sublinear" not in conclusion
        ):
            failures.append("endpoint-run RGB train/eval conclusion must keep compactness, speed, and scaling scope explicit")

    endpoint_record_edit_train_eval = payload.get("endpoint_record_edit_rgb_train_eval")
    if not isinstance(endpoint_record_edit_train_eval, dict) or endpoint_record_edit_train_eval.get("available") is not True:
        failures.append("missing endpoint-record edit RGB train/eval artifact")
    else:
        acceptance = endpoint_record_edit_train_eval.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("endpoint-record edit RGB train/eval missing acceptance")
        else:
            for key in (
                "all_rows_ok",
                "total_step_sublinear_vs_frames",
                "render_sublinear_vs_frames",
                "backward_sublinear_vs_frames",
                "selected_tape_segments_below_full_at_max_frame",
                "selected_tape_storage_below_full_at_max_frame",
            ):
                if acceptance.get(key) is not True:
                    failures.append(f"endpoint-record edit RGB train/eval acceptance {key} must be true")
        if endpoint_record_edit_train_eval.get("tape_mode") != "endpoint-record-edit":
            failures.append("endpoint-record edit RGB train/eval must record tape_mode='endpoint-record-edit'")
        if endpoint_record_edit_train_eval.get("full_trainer_claim") is not False:
            failures.append("endpoint-record edit RGB train/eval must not claim full trainer coverage")
        if endpoint_record_edit_train_eval.get("full_geometry_gradient_claim") is not False:
            failures.append("endpoint-record edit RGB train/eval must not claim geometry-gradient coverage")
        if endpoint_record_edit_train_eval.get("density_independent_depth_claim") is not True:
            failures.append("endpoint-record edit RGB train/eval must claim density independence only for endpoint semantic")
        if endpoint_record_edit_train_eval.get("continuous_absorption_depth_semantic") is not True:
            failures.append("endpoint-record edit RGB train/eval must mark continuous absorption semantic")
        for key in (
            "frame_scale_first_to_last",
            "total_step_scale_first_to_last",
            "render_scale_first_to_last",
            "backward_scale_first_to_last",
            "selected_tape_segment_scale_first_to_last",
            "selected_tape_storage_scale_first_to_last",
            "endpoint_record_edit_op_scale_first_to_last",
        ):
            if not _finite_positive(endpoint_record_edit_train_eval.get(key)):
                failures.append(f"endpoint_record_edit_rgb_train_eval.{key} is not positive finite")
        frame_scale = endpoint_record_edit_train_eval.get("frame_scale_first_to_last")
        selected_scale = endpoint_record_edit_train_eval.get("selected_tape_segment_scale_first_to_last")
        storage_scale = endpoint_record_edit_train_eval.get("selected_tape_storage_scale_first_to_last")
        edit_op_scale = endpoint_record_edit_train_eval.get("endpoint_record_edit_op_scale_first_to_last")
        if _finite_positive(frame_scale) and _finite_positive(selected_scale) and float(selected_scale) <= float(frame_scale):
            failures.append("endpoint-record edit selected endpoint-run scale unexpectedly looks structurally sublinear")
        if _finite_positive(frame_scale) and _finite_positive(storage_scale) and float(storage_scale) >= float(frame_scale):
            failures.append("endpoint-record edit selected storage scale must stay sublinear")
        if _finite_positive(frame_scale) and _finite_positive(edit_op_scale) and float(edit_op_scale) >= float(frame_scale):
            failures.append("endpoint-record edit op scale must stay sublinear")
        last_row = endpoint_record_edit_train_eval.get("last_row")
        if not isinstance(last_row, dict):
            failures.append("endpoint-record edit RGB train/eval missing last_row")
        else:
            for key in ("total_ms", "render_ms", "backward_ms", "final_train_psnr", "final_heldout_psnr"):
                if not _finite_positive(last_row.get(key)):
                    failures.append(f"endpoint-record edit RGB train/eval last_row.{key} is not positive finite")
            storage_ratio = last_row.get("train_selected_tape_storage_vs_full")
            edit_vs_endpoint = last_row.get("train_endpoint_record_edit_storage_vs_endpoint_run")
            segment_ratio = last_row.get("train_selected_tape_segments_vs_full")
            edit_ops = last_row.get("train_endpoint_record_edit_ops")
            if not _finite_positive(storage_ratio) or float(storage_ratio) >= 0.05:
                failures.append("endpoint-record edit RGB train/eval 16f storage ratio no longer supports compact path")
            if not _finite_positive(edit_vs_endpoint) or float(edit_vs_endpoint) >= 0.50:
                failures.append("endpoint-record edit RGB train/eval 16f storage must stay below half endpoint-run CSR")
            if not _finite_positive(segment_ratio) or float(segment_ratio) >= 0.20:
                failures.append("endpoint-record edit RGB train/eval 16f segment ratio no longer supports endpoint compact path")
            if not _finite_positive(edit_ops):
                failures.append("endpoint-record edit RGB train/eval must record positive edit op count")
            if (
                isinstance(winner, dict)
                and _finite_positive(winner.get("heldout_psnr_16f"))
                and _finite_positive(last_row.get("final_heldout_psnr"))
                and abs(float(last_row["final_heldout_psnr"]) - float(winner["heldout_psnr_16f"])) > 1.0e-3
            ):
                failures.append("endpoint-record edit RGB train/eval 16f heldout PSNR no longer matches fused winner")
        comparison = endpoint_record_edit_train_eval.get("comparison")
        if not isinstance(comparison, dict):
            failures.append("endpoint-record edit RGB train/eval missing comparison block")
        else:
            for key in (
                "endpoint_record_edit_to_fused_winner_16f_total_ratio",
                "endpoint_record_edit_to_owner_run_16f_total_ratio",
                "endpoint_record_edit_to_endpoint_run_16f_total_ratio",
            ):
                if not _finite_positive(comparison.get(key)):
                    failures.append(f"endpoint-record edit RGB train/eval comparison {key} is not positive finite")
            note = str(comparison.get("scope_note", ""))
            if (
                "owner+cut-id edit stream shader" not in note
                or "not a full trainer" not in note
                or "matched STAR UVT quality/capacity claim" not in note
            ):
                failures.append("endpoint-record edit RGB train/eval comparison must keep shader/trainer/STAR scope explicit")
        conclusion = str(endpoint_record_edit_train_eval.get("conclusion", ""))
        if (
            "measured compact endpoint semantic path" not in conclusion
            or "edit storage sublinear" not in conclusion
            or "not a main-trainer or" not in conclusion
        ):
            failures.append("endpoint-record edit RGB train/eval conclusion must keep compactness and scope explicit")

    endpoint_record_edit_block4_train_eval = payload.get("endpoint_record_edit_block4_rgb_train_eval")
    if (
        not isinstance(endpoint_record_edit_block4_train_eval, dict)
        or endpoint_record_edit_block4_train_eval.get("available") is not True
    ):
        failures.append("missing endpoint-record edit block4 RGB train/eval artifact")
    else:
        acceptance = endpoint_record_edit_block4_train_eval.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("endpoint-record edit block4 RGB train/eval missing acceptance")
        else:
            for key in (
                "all_rows_ok",
                "total_step_sublinear_vs_frames",
                "render_sublinear_vs_frames",
                "backward_sublinear_vs_frames",
                "selected_tape_segments_below_full_at_max_frame",
                "selected_tape_storage_below_full_at_max_frame",
            ):
                if acceptance.get(key) is not True:
                    failures.append(f"endpoint-record edit block4 RGB train/eval acceptance {key} must be true")
        if endpoint_record_edit_block4_train_eval.get("tape_mode") != "endpoint-record-edit-block4":
            failures.append("endpoint-record edit block4 RGB train/eval must record tape_mode='endpoint-record-edit-block4'")
        if endpoint_record_edit_block4_train_eval.get("full_trainer_claim") is not False:
            failures.append("endpoint-record edit block4 RGB train/eval must not claim full trainer coverage")
        if endpoint_record_edit_block4_train_eval.get("full_geometry_gradient_claim") is not False:
            failures.append("endpoint-record edit block4 RGB train/eval must not claim geometry-gradient coverage")
        if endpoint_record_edit_block4_train_eval.get("density_independent_depth_claim") is not True:
            failures.append("endpoint-record edit block4 RGB train/eval must keep endpoint-depth semantic scope")
        if endpoint_record_edit_block4_train_eval.get("continuous_absorption_depth_semantic") is not True:
            failures.append("endpoint-record edit block4 RGB train/eval must mark continuous absorption semantic")
        for key in (
            "frame_scale_first_to_last",
            "total_step_scale_first_to_last",
            "render_scale_first_to_last",
            "backward_scale_first_to_last",
            "selected_tape_segment_scale_first_to_last",
            "selected_tape_storage_scale_first_to_last",
            "endpoint_record_edit_op_scale_first_to_last",
        ):
            if not _finite_positive(endpoint_record_edit_block4_train_eval.get(key)):
                failures.append(f"endpoint_record_edit_block4_rgb_train_eval.{key} is not positive finite")
        frame_scale = endpoint_record_edit_block4_train_eval.get("frame_scale_first_to_last")
        storage_scale = endpoint_record_edit_block4_train_eval.get("selected_tape_storage_scale_first_to_last")
        edit_op_scale = endpoint_record_edit_block4_train_eval.get("endpoint_record_edit_op_scale_first_to_last")
        if _finite_positive(frame_scale) and _finite_positive(storage_scale) and float(storage_scale) >= float(frame_scale):
            failures.append("endpoint-record edit block4 selected storage scale must stay sublinear")
        if _finite_positive(frame_scale) and _finite_positive(edit_op_scale) and float(edit_op_scale) >= float(frame_scale):
            failures.append("endpoint-record edit block4 edit op scale must stay sublinear")
        last_row = endpoint_record_edit_block4_train_eval.get("last_row")
        if not isinstance(last_row, dict):
            failures.append("endpoint-record edit block4 RGB train/eval missing last_row")
        else:
            for key in ("total_ms", "render_ms", "backward_ms", "final_train_psnr", "final_heldout_psnr"):
                if not _finite_positive(last_row.get(key)):
                    failures.append(f"endpoint-record edit block4 RGB train/eval last_row.{key} is not positive finite")
            storage_ratio = last_row.get("train_endpoint_record_block4_storage_vs_full")
            block4_vs_endpoint = last_row.get("train_endpoint_record_block4_storage_vs_endpoint_run")
            block4_ops = last_row.get("train_endpoint_record_block4_ops")
            if not _finite_positive(storage_ratio) or float(storage_ratio) >= 0.05:
                failures.append("endpoint-record edit block4 RGB train/eval 16f storage ratio must stay below 0.05x full")
            if not _finite_positive(block4_vs_endpoint) or float(block4_vs_endpoint) >= 0.50:
                failures.append("endpoint-record edit block4 RGB train/eval 16f storage must stay below half endpoint-run CSR")
            if not _finite_positive(block4_ops):
                failures.append("endpoint-record edit block4 RGB train/eval must record positive block4 op count")
        comparison = endpoint_record_edit_block4_train_eval.get("comparison")
        if not isinstance(comparison, dict):
            failures.append("endpoint-record edit block4 RGB train/eval missing comparison block")
        else:
            note = str(comparison.get("scope_note", ""))
            if (
                "Block4 endpoint-record edit train/eval" not in note
                or "dedicated block4" not in note
                or "not a main-trainer integration" not in note
                or "STAR-UVT" not in note
            ):
                failures.append("endpoint-record edit block4 RGB train/eval comparison must keep VJP/trainer/STAR scope")
        conclusion = str(endpoint_record_edit_block4_train_eval.get("conclusion", ""))
        if (
            "dedicated block4 RGB-only VJP" not in conclusion
            or "not speed-competitive" not in conclusion
            or "main-trainer" not in conclusion
            or "STAR-UVT competitive claim" not in conclusion
        ):
            failures.append("endpoint-record edit block4 RGB train/eval conclusion must keep speed and scope explicit")

    paired_endpoint_edit = payload.get("endpoint_record_edit_paired_train_eval")
    if not isinstance(paired_endpoint_edit, dict) or paired_endpoint_edit.get("available") is not True:
        failures.append("missing paired endpoint-run versus endpoint-record-edit train/eval comparison")
    else:
        ratios = paired_endpoint_edit.get("ratios")
        endpoint = paired_endpoint_edit.get("endpoint_run_16f")
        edit = paired_endpoint_edit.get("endpoint_record_edit_16f")
        if not isinstance(ratios, dict) or not isinstance(endpoint, dict) or not isinstance(edit, dict):
            failures.append("paired endpoint/edit comparison missing ratio or 16f blocks")
        else:
            total_ratio = ratios.get("edit_to_endpoint_total_16f")
            render_ratio = ratios.get("edit_to_endpoint_render_16f")
            backward_ratio = ratios.get("edit_to_endpoint_backward_16f")
            for key, value in (
                ("edit_to_endpoint_total_16f", total_ratio),
                ("edit_to_endpoint_render_16f", render_ratio),
                ("edit_to_endpoint_backward_16f", backward_ratio),
            ):
                if not _finite_positive(value):
                    failures.append(f"paired endpoint/edit comparison {key} is not positive finite")
            if _finite_positive(total_ratio) and float(total_ratio) <= 1.0:
                failures.append("paired endpoint/edit comparison must preserve that edit is not yet faster")
            edit_storage = edit.get("storage_vs_full")
            endpoint_storage = endpoint.get("storage_vs_full")
            if not _finite_positive(edit_storage) or float(edit_storage) >= 0.05:
                failures.append("paired endpoint/edit comparison edit storage ratio no longer supports compact path")
            if not _finite_positive(endpoint_storage) or float(endpoint_storage) >= 0.20:
                failures.append("paired endpoint/edit comparison endpoint storage ratio no longer supports compact path")
            if (
                _finite_positive(edit_storage)
                and _finite_positive(endpoint_storage)
                and float(edit_storage) >= float(endpoint_storage)
            ):
                failures.append("paired endpoint/edit comparison must preserve edit storage below endpoint storage")
            edit_psnr = edit.get("heldout_psnr")
            endpoint_psnr = endpoint.get("heldout_psnr")
            if (
                _finite_positive(edit_psnr)
                and _finite_positive(endpoint_psnr)
                and abs(float(edit_psnr) - float(endpoint_psnr)) > 1.0e-3
            ):
                failures.append("paired endpoint/edit comparison PSNR mismatch exceeds tolerance")
        scope = str(paired_endpoint_edit.get("scope", ""))
        if "not a stable benchmark" not in scope or "STAR UVT competitive claim" not in scope:
            failures.append("paired endpoint/edit comparison scope must reject stable speed or STAR claims")
        conclusion = str(paired_endpoint_edit.get("conclusion", ""))
        if (
            "lower storage" not in conclusion
            or "slower than endpoint-run" not in conclusion
            or "needs replay optimization" not in conclusion
        ):
            failures.append("paired endpoint/edit comparison conclusion must keep storage/speed tradeoff explicit")

    paired_endpoint_edit_block4 = payload.get("endpoint_record_edit_block4_paired_train_eval")
    if not isinstance(paired_endpoint_edit_block4, dict) or paired_endpoint_edit_block4.get("available") is not True:
        failures.append("missing paired endpoint-run versus endpoint-record-edit-block4 train/eval comparison")
    else:
        ratios = paired_endpoint_edit_block4.get("ratios")
        endpoint = paired_endpoint_edit_block4.get("endpoint_run_16f")
        edit = paired_endpoint_edit_block4.get("endpoint_record_edit_16f")
        block4 = paired_endpoint_edit_block4.get("endpoint_record_edit_block4_16f")
        if (
            not isinstance(ratios, dict)
            or not isinstance(endpoint, dict)
            or not isinstance(edit, dict)
            or not isinstance(block4, dict)
        ):
            failures.append("paired endpoint/edit/block4 comparison missing ratio or 16f blocks")
        else:
            for key in (
                "block4_to_endpoint_total_16f",
                "block4_to_endpoint_render_16f",
                "block4_to_endpoint_backward_16f",
                "block4_to_edit_total_16f",
                "edit_to_endpoint_total_16f",
            ):
                if not _finite_positive(ratios.get(key)):
                    failures.append(f"paired endpoint/edit/block4 comparison {key} is not positive finite")
            total_ratio = ratios.get("block4_to_endpoint_total_16f")
            render_ratio = ratios.get("block4_to_endpoint_render_16f")
            if not _finite_positive(total_ratio):
                failures.append("paired endpoint/edit/block4 comparison should record a positive block4 total ratio")
            if not _finite_positive(render_ratio):
                failures.append("paired endpoint/edit/block4 comparison should record a positive block4 render ratio")
            block4_storage = block4.get("storage_vs_full")
            endpoint_storage = endpoint.get("storage_vs_full")
            if not _finite_positive(block4_storage) or float(block4_storage) >= 0.05:
                failures.append("paired endpoint/edit/block4 block4 storage ratio no longer supports compact path")
            if (
                _finite_positive(block4_storage)
                and _finite_positive(endpoint_storage)
                and float(block4_storage) >= float(endpoint_storage)
            ):
                failures.append("paired endpoint/edit/block4 comparison must preserve block4 storage below endpoint storage")
            block4_psnr = block4.get("heldout_psnr")
            endpoint_psnr = endpoint.get("heldout_psnr")
            edit_psnr = edit.get("heldout_psnr")
            if (
                _finite_positive(block4_psnr)
                and _finite_positive(endpoint_psnr)
                and abs(float(block4_psnr) - float(endpoint_psnr)) > 1.0e-3
            ):
                failures.append("paired endpoint/edit/block4 comparison block4 PSNR mismatch exceeds tolerance")
            if (
                _finite_positive(block4_psnr)
                and _finite_positive(edit_psnr)
                and abs(float(block4_psnr) - float(edit_psnr)) > 1.0e-3
            ):
                failures.append("paired endpoint/edit/block4 comparison edit/block4 PSNR mismatch exceeds tolerance")
        acceptance = paired_endpoint_edit_block4.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("paired endpoint/edit/block4 comparison missing acceptance")
        else:
            for key in (
                "endpoint_record_edit_block4_ok",
                "block4_psnr_matches",
                "block4_storage_below_endpoint",
                "block4_total_ratio_positive",
            ):
                if acceptance.get(key) is not True:
                    failures.append(f"paired endpoint/edit/block4 acceptance {key} must be true")
        if paired_endpoint_edit_block4.get("block4_speed_read") not in {
            "faster_than_endpoint_run",
            "not_faster_or_not_measured",
        }:
            failures.append("paired endpoint/edit/block4 comparison must record an explicit block4 speed read")
        scope = str(paired_endpoint_edit_block4.get("scope", ""))
        if "not a stable benchmark" not in scope or "STAR UVT competitive claim" not in scope:
            failures.append("paired endpoint/edit/block4 comparison scope must reject stable speed or STAR claims")
        conclusion = str(paired_endpoint_edit_block4.get("conclusion", ""))
        if (
            "block-anchored forward variant" not in conclusion
            and "block-anchored variants" not in conclusion
        ) or "STAR-UVT claim" not in conclusion:
            failures.append("paired endpoint/edit/block4 comparison conclusion must keep block4 and STAR scope explicit")

    paired_endpoint_edit_block_coeff = payload.get("endpoint_record_edit_block_coeff_paired_train_eval")
    if (
        not isinstance(paired_endpoint_edit_block_coeff, dict)
        or paired_endpoint_edit_block_coeff.get("available") is not True
    ):
        failures.append("missing paired endpoint-run versus endpoint-record-edit-block-coeff train/eval comparison")
    else:
        ratios = paired_endpoint_edit_block_coeff.get("ratios")
        endpoint = paired_endpoint_edit_block_coeff.get("endpoint_run_16f")
        edit = paired_endpoint_edit_block_coeff.get("endpoint_record_edit_16f")
        block4 = paired_endpoint_edit_block_coeff.get("endpoint_record_edit_block4_16f")
        block_coeff = paired_endpoint_edit_block_coeff.get("endpoint_record_edit_block_coeff_16f")
        if (
            not isinstance(ratios, dict)
            or not isinstance(endpoint, dict)
            or not isinstance(edit, dict)
            or not isinstance(block4, dict)
            or not isinstance(block_coeff, dict)
        ):
            failures.append("paired endpoint/edit/block4/block-coeff comparison missing ratio or 16f blocks")
        else:
            for key in (
                "block_coeff_to_endpoint_total_16f",
                "block_coeff_to_endpoint_render_16f",
                "block_coeff_to_endpoint_backward_16f",
                "block_coeff_to_block4_total_16f",
                "block_coeff_to_edit_total_16f",
                "block4_to_endpoint_total_16f",
                "edit_to_endpoint_total_16f",
            ):
                if not _finite_positive(ratios.get(key)):
                    failures.append(f"paired block-coeff comparison {key} is not positive finite")
            block_coeff_to_endpoint = ratios.get("block_coeff_to_endpoint_total_16f")
            block_coeff_to_block4 = ratios.get("block_coeff_to_block4_total_16f")
            edit_to_endpoint = ratios.get("edit_to_endpoint_total_16f")
            if _finite_positive(block_coeff_to_endpoint) and float(block_coeff_to_endpoint) >= 1.0:
                failures.append("paired block-coeff comparison should preserve block-coeff faster than endpoint-run")
            if _finite_positive(block_coeff_to_block4) and float(block_coeff_to_block4) >= 1.0:
                failures.append("paired block-coeff comparison should preserve block-coeff faster than block4")
            if not _finite_positive(edit_to_endpoint):
                failures.append("paired block-coeff comparison should record raw edit ratio even though speed sign is noisy")
            block_coeff_storage = block_coeff.get("storage_vs_full")
            endpoint_storage = endpoint.get("storage_vs_full")
            block4_storage = block4.get("storage_vs_full")
            if not _finite_positive(block_coeff_storage) or float(block_coeff_storage) >= 0.25:
                failures.append("paired block-coeff storage should stay below 0.25x full CSR")
            if not _finite_positive(endpoint_storage) or float(endpoint_storage) >= 0.20:
                failures.append("paired block-coeff endpoint storage should stay compact")
            if not _finite_positive(block4_storage) or float(block4_storage) >= 0.05:
                failures.append("paired block-coeff block4 storage should stay below 0.05x full CSR")
            block_coeff_psnr = block_coeff.get("heldout_psnr")
            endpoint_psnr = endpoint.get("heldout_psnr")
            if (
                _finite_positive(block_coeff_psnr)
                and _finite_positive(endpoint_psnr)
                and abs(float(block_coeff_psnr) - float(endpoint_psnr)) > 1.0e-3
            ):
                failures.append("paired block-coeff PSNR mismatch exceeds tolerance")
        acceptance = paired_endpoint_edit_block_coeff.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("paired block-coeff comparison missing acceptance")
        else:
            for key in (
                "endpoint_record_edit_block_coeff_ok",
                "block_coeff_psnr_matches",
                "block_coeff_storage_positive",
                "block_coeff_total_ratio_positive",
                "block_coeff_total_not_slower_than_endpoint",
            ):
                if acceptance.get(key) is not True:
                    failures.append(f"paired block-coeff acceptance {key} must be true")
        if paired_endpoint_edit_block_coeff.get("block_coeff_speed_read") != "faster_than_endpoint_run":
            failures.append("paired block-coeff comparison must record faster-than-endpoint speed read")
        scope = str(paired_endpoint_edit_block_coeff.get("scope", ""))
        if "not a stable benchmark" not in scope or "STAR UVT competitive claim" not in scope:
            failures.append("paired block-coeff comparison scope must reject stable speed or STAR claims")
        conclusion = str(paired_endpoint_edit_block_coeff.get("conclusion", ""))
        if (
            "Block-coeff is faster than endpoint-run" not in conclusion
            or "Block4 is faster than endpoint-run" not in conclusion
            or "Raw edit speed sign remains noisy" not in conclusion
            or "STAR-UVT claim" not in conclusion
        ):
            failures.append("paired block-coeff comparison conclusion must keep speed and STAR scope explicit")

    paired_endpoint_edit_block_coeff_repeat20 = payload.get("endpoint_record_edit_block_coeff_repeat20_16f")
    if (
        not isinstance(paired_endpoint_edit_block_coeff_repeat20, dict)
        or paired_endpoint_edit_block_coeff_repeat20.get("available") is not True
    ):
        failures.append("missing 20-step 16f paired endpoint-run versus block-coeff train/eval comparison")
    else:
        ratios = paired_endpoint_edit_block_coeff_repeat20.get("ratios")
        endpoint = paired_endpoint_edit_block_coeff_repeat20.get("endpoint_run_16f")
        edit = paired_endpoint_edit_block_coeff_repeat20.get("endpoint_record_edit_16f")
        block4 = paired_endpoint_edit_block_coeff_repeat20.get("endpoint_record_edit_block4_16f")
        block_coeff = paired_endpoint_edit_block_coeff_repeat20.get("endpoint_record_edit_block_coeff_16f")
        if (
            not isinstance(ratios, dict)
            or not isinstance(endpoint, dict)
            or not isinstance(edit, dict)
            or not isinstance(block4, dict)
            or not isinstance(block_coeff, dict)
        ):
            failures.append("20-step 16f block-coeff comparison missing ratio or 16f blocks")
        else:
            for key in (
                "edit_to_endpoint_total_16f",
                "block4_to_endpoint_total_16f",
                "block_coeff_to_endpoint_total_16f",
                "block_coeff_to_block4_total_16f",
                "block_coeff_to_edit_total_16f",
            ):
                if not _finite_positive(ratios.get(key)):
                    failures.append(f"20-step 16f block-coeff comparison {key} is not positive finite")
            edit_to_endpoint = ratios.get("edit_to_endpoint_total_16f")
            block4_to_endpoint = ratios.get("block4_to_endpoint_total_16f")
            block_coeff_to_endpoint = ratios.get("block_coeff_to_endpoint_total_16f")
            block_coeff_to_block4 = ratios.get("block_coeff_to_block4_total_16f")
            if _finite_positive(edit_to_endpoint) and float(edit_to_endpoint) <= 1.0:
                failures.append("20-step 16f repeat should preserve that raw edit is slower in this run")
            if _finite_positive(block4_to_endpoint) and float(block4_to_endpoint) >= 1.05:
                failures.append("20-step 16f repeat should preserve block4 near or below endpoint-run total time")
            if _finite_positive(block_coeff_to_endpoint) and float(block_coeff_to_endpoint) >= 1.0:
                failures.append("20-step 16f repeat should preserve block-coeff faster than endpoint-run")
            if _finite_positive(block_coeff_to_block4) and float(block_coeff_to_block4) >= 1.0:
                failures.append("20-step 16f repeat should preserve block-coeff faster than block4")

            endpoint_storage = endpoint.get("storage_vs_full")
            edit_storage = edit.get("storage_vs_full")
            block4_storage = block4.get("storage_vs_full")
            block_coeff_storage = block_coeff.get("storage_vs_full")
            for name, value in (
                ("endpoint storage", endpoint_storage),
                ("edit storage", edit_storage),
                ("block4 storage", block4_storage),
                ("block-coeff storage", block_coeff_storage),
            ):
                if not _finite_positive(value):
                    failures.append(f"20-step 16f block-coeff comparison {name} is not positive finite")
            if (
                _finite_positive(edit_storage)
                and _finite_positive(endpoint_storage)
                and float(edit_storage) >= float(endpoint_storage)
            ):
                failures.append("20-step 16f repeat must preserve raw edit storage below endpoint storage")
            if (
                _finite_positive(block4_storage)
                and _finite_positive(endpoint_storage)
                and float(block4_storage) >= float(endpoint_storage)
            ):
                failures.append("20-step 16f repeat must preserve block4 storage below endpoint storage")
            if _finite_positive(block_coeff_storage) and float(block_coeff_storage) >= 0.25:
                failures.append("20-step 16f repeat block-coeff storage should stay below 0.25x full CSR")

            endpoint_psnr = endpoint.get("heldout_psnr")
            for name, candidate in (
                ("raw edit", edit.get("heldout_psnr")),
                ("block4", block4.get("heldout_psnr")),
                ("block-coeff", block_coeff.get("heldout_psnr")),
            ):
                if (
                    _finite_positive(endpoint_psnr)
                    and _finite_positive(candidate)
                    and abs(float(candidate) - float(endpoint_psnr)) > 1.0e-3
                ):
                    failures.append(f"20-step 16f {name} PSNR mismatch exceeds tolerance")
        acceptance = paired_endpoint_edit_block_coeff_repeat20.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("20-step 16f block-coeff comparison missing acceptance")
        else:
            for key in (
                "endpoint_run_ok",
                "endpoint_record_edit_ok",
                "endpoint_record_edit_block4_ok",
                "endpoint_record_edit_block_coeff_ok",
                "psnr_matches",
                "block4_psnr_matches",
                "block_coeff_psnr_matches",
                "block4_total_not_slower_than_endpoint",
                "block_coeff_total_not_slower_than_endpoint",
                "block_coeff_total_ratio_positive",
            ):
                if acceptance.get(key) is not True:
                    failures.append(f"20-step 16f block-coeff acceptance {key} must be true")
            if acceptance.get("edit_total_not_slower_than_endpoint") is not False:
                failures.append("20-step 16f repeat should preserve raw edit as slower in acceptance")
        if paired_endpoint_edit_block_coeff_repeat20.get("speed_read") != "slower_than_endpoint_run":
            failures.append("20-step 16f repeat must record raw edit slower speed read")
        if paired_endpoint_edit_block_coeff_repeat20.get("block4_speed_read") != "faster_than_endpoint_run":
            failures.append("20-step 16f repeat must record block4 faster-than-endpoint speed read")
        if paired_endpoint_edit_block_coeff_repeat20.get("block_coeff_speed_read") != "faster_than_endpoint_run":
            failures.append("20-step 16f repeat must record block-coeff faster-than-endpoint speed read")
        scope = str(paired_endpoint_edit_block_coeff_repeat20.get("scope", ""))
        if "not a stable benchmark" not in scope or "STAR UVT competitive claim" not in scope:
            failures.append("20-step 16f block-coeff comparison scope must reject stable speed or STAR claims")
        conclusion = str(paired_endpoint_edit_block_coeff_repeat20.get("conclusion", ""))
        if (
            "Block-coeff is faster than endpoint-run" not in conclusion
            or "block-coeff is faster than block4" not in conclusion
            or "Raw edit speed sign remains noisy" not in conclusion
            or "STAR-UVT claim" not in conclusion
        ):
            failures.append("20-step 16f block-coeff conclusion must keep repeat speed and STAR scope explicit")

    paired_repeat20_2_4_8_16 = payload.get("endpoint_record_edit_block_coeff_repeat20_2_4_8_16")
    if not isinstance(paired_repeat20_2_4_8_16, dict) or paired_repeat20_2_4_8_16.get("available") is not True:
        failures.append("missing longer paired 2/4/8/16 block-coeff repeat artifact")
    else:
        if paired_repeat20_2_4_8_16.get("status") != "failed":
            failures.append("longer paired 2/4/8/16 repeat should preserve failed status until block-coeff speed is fixed")
        ratios = paired_repeat20_2_4_8_16.get("ratios")
        endpoint = paired_repeat20_2_4_8_16.get("endpoint_run_16f")
        edit = paired_repeat20_2_4_8_16.get("endpoint_record_edit_16f")
        block4 = paired_repeat20_2_4_8_16.get("endpoint_record_edit_block4_16f")
        block_coeff = paired_repeat20_2_4_8_16.get("endpoint_record_edit_block_coeff_16f")
        if (
            not isinstance(ratios, dict)
            or not isinstance(endpoint, dict)
            or not isinstance(edit, dict)
            or not isinstance(block4, dict)
            or not isinstance(block_coeff, dict)
        ):
            failures.append("longer paired 2/4/8/16 repeat missing ratio or 16f blocks")
        else:
            edit_to_endpoint = ratios.get("edit_to_endpoint_total_16f")
            block4_to_endpoint = ratios.get("block4_to_endpoint_total_16f")
            block_coeff_to_endpoint = ratios.get("block_coeff_to_endpoint_total_16f")
            block_coeff_to_block4 = ratios.get("block_coeff_to_block4_total_16f")
            if not _finite_positive(edit_to_endpoint) or float(edit_to_endpoint) >= 1.0:
                failures.append("longer paired 2/4/8/16 repeat should preserve raw edit faster than endpoint at 16f")
            if not _finite_positive(block4_to_endpoint) or float(block4_to_endpoint) >= 1.0:
                failures.append("longer paired 2/4/8/16 repeat should preserve block4 faster than endpoint at 16f")
            if not _finite_positive(block_coeff_to_endpoint) or float(block_coeff_to_endpoint) <= 1.0:
                failures.append("longer paired 2/4/8/16 repeat should preserve block-coeff slower-than-endpoint negative")
            if not _finite_positive(block_coeff_to_block4) or float(block_coeff_to_block4) <= 1.0:
                failures.append("longer paired 2/4/8/16 repeat should preserve block-coeff slower-than-block4 negative")
            endpoint_psnr = endpoint.get("heldout_psnr")
            for name, candidate in (
                ("raw edit", edit.get("heldout_psnr")),
                ("block4", block4.get("heldout_psnr")),
                ("block-coeff", block_coeff.get("heldout_psnr")),
            ):
                if (
                    _finite_positive(endpoint_psnr)
                    and _finite_positive(candidate)
                    and abs(float(candidate) - float(endpoint_psnr)) > 1.0e-3
                ):
                    failures.append(f"longer paired 2/4/8/16 {name} PSNR mismatch exceeds tolerance")
        acceptance = paired_repeat20_2_4_8_16.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("longer paired 2/4/8/16 repeat missing acceptance")
        else:
            for key in (
                "endpoint_run_ok",
                "endpoint_record_edit_ok",
                "endpoint_record_edit_block4_ok",
                "psnr_matches",
                "block4_psnr_matches",
                "block_coeff_psnr_matches",
                "edit_total_not_slower_than_endpoint",
                "block4_total_not_slower_than_endpoint",
                "block_coeff_total_ratio_positive",
            ):
                if acceptance.get(key) is not True:
                    failures.append(f"longer paired 2/4/8/16 acceptance {key} must be true")
            if acceptance.get("endpoint_record_edit_block_coeff_ok") is not False:
                failures.append("longer paired 2/4/8/16 should preserve block-coeff result status as failed")
            if acceptance.get("block_coeff_total_not_slower_than_endpoint") is not False:
                failures.append("longer paired 2/4/8/16 should preserve block-coeff speed-gate failure")
        if paired_repeat20_2_4_8_16.get("block_coeff_speed_read") == "faster_than_endpoint_run":
            failures.append("longer paired 2/4/8/16 must not record block-coeff as speed-positive")
        conclusion = str(paired_repeat20_2_4_8_16.get("conclusion", ""))
        if "Block-coeff is slower than endpoint-run" not in conclusion or "STAR-UVT claim" not in conclusion:
            failures.append("longer paired 2/4/8/16 conclusion must keep negative block-coeff and STAR scope")

    paired_block_coeff16 = payload.get("endpoint_record_edit_block_coeff16_manual_vjp_paired_train_eval")
    if not isinstance(paired_block_coeff16, dict) or paired_block_coeff16.get("available") is not True:
        failures.append("missing paired block-coeff16 manual-VJP train/eval comparison")
    else:
        ratios = paired_block_coeff16.get("ratios")
        endpoint = paired_block_coeff16.get("endpoint_run_16f")
        block_coeff = paired_block_coeff16.get("endpoint_record_edit_block_coeff_16f")
        block_coeff16 = paired_block_coeff16.get("endpoint_record_edit_block_coeff16_16f")
        if not isinstance(ratios, dict) or not isinstance(endpoint, dict) or not isinstance(block_coeff, dict) or not isinstance(block_coeff16, dict):
            failures.append("block-coeff16 comparison missing ratio or 16f blocks")
        else:
            ratio_endpoint = ratios.get("block_coeff16_to_endpoint_total_16f")
            ratio_coeff = ratios.get("block_coeff16_to_block_coeff_total_16f")
            if not _finite_positive(ratio_endpoint) or float(ratio_endpoint) <= 1.0:
                failures.append("block-coeff16 comparison should preserve slower-than-endpoint negative result")
            if not _finite_positive(ratio_coeff) or float(ratio_coeff) <= 1.0:
                failures.append("block-coeff16 comparison should preserve slower-than-f32-block-coeff negative result")
            storage16 = block_coeff16.get("storage_vs_full")
            storage32 = block_coeff.get("storage_vs_full")
            if not _finite_positive(storage16) or not _finite_positive(storage32) or float(storage16) >= float(storage32):
                failures.append("block-coeff16 comparison should preserve f16 storage below f32 coefficient storage")
            endpoint_psnr = endpoint.get("heldout_psnr")
            coeff16_psnr = block_coeff16.get("heldout_psnr")
            if (
                _finite_positive(endpoint_psnr)
                and _finite_positive(coeff16_psnr)
                and abs(float(coeff16_psnr) - float(endpoint_psnr)) > 1.0e-3
            ):
                failures.append("block-coeff16 PSNR mismatch exceeds f16 tolerance")
        acceptance = paired_block_coeff16.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("block-coeff16 comparison missing acceptance")
        else:
            for key in (
                "endpoint_record_edit_block_coeff16_ok",
                "block_coeff16_psnr_matches",
                "block_coeff16_storage_positive",
                "block_coeff16_total_ratio_positive",
            ):
                if acceptance.get(key) is not True:
                    failures.append(f"block-coeff16 acceptance {key} must be true")
            if acceptance.get("block_coeff16_total_not_slower_than_endpoint") is not False:
                failures.append("block-coeff16 acceptance should preserve speed-regression negative result")
        if paired_block_coeff16.get("block_coeff16_speed_read") != "slower_than_endpoint_run":
            failures.append("block-coeff16 summary must mark slower-than-endpoint speed read")
        conclusion = str(paired_block_coeff16.get("conclusion", ""))
        if "Block-coeff16 is slower than endpoint-run" not in conclusion or "keep it negative" not in conclusion:
            failures.append("block-coeff16 conclusion must keep negative speed result explicit")

    coeff16_storagefix = payload.get("endpoint_record_edit_block_coeff16_storagefix_smoke")
    if not isinstance(coeff16_storagefix, dict) or coeff16_storagefix.get("available") is not True:
        failures.append("missing coeff16 storage-accounting smoke")
    else:
        if coeff16_storagefix.get("status") != "ok":
            failures.append("coeff16 storage-accounting smoke must be ok")
        if coeff16_storagefix.get("tape_mode") != "endpoint-record-edit-block-coeff16":
            failures.append("coeff16 storage-accounting smoke tape mode mismatch")
        if coeff16_storagefix.get("optimizer_mode") != "manual-vjp":
            failures.append("coeff16 storage-accounting smoke must use manual-vjp")
        last_row = coeff16_storagefix.get("last_row")
        storage = coeff16_storagefix.get("storage_accounting")
        if not isinstance(last_row, dict) or not isinstance(storage, dict):
            failures.append("coeff16 storage-accounting smoke missing last_row or storage block")
        else:
            selected = last_row.get("train_selected_tape_storage_vs_full")
            endpoint = last_row.get("train_endpoint_run_storage_vs_full")
            block4 = last_row.get("train_endpoint_record_block4_storage_vs_full")
            for key, value in (
                ("train_selected_tape_storage_vs_full", selected),
                ("train_endpoint_run_storage_vs_full", endpoint),
                ("train_endpoint_record_block4_storage_vs_full", block4),
            ):
                if not _finite_positive(value):
                    failures.append(f"coeff16 storage-accounting {key} is not positive finite")
            if storage.get("selected_storage_not_endpoint_run") is not True:
                failures.append("coeff16 storage accounting must not report endpoint-run storage")
            if storage.get("selected_storage_above_block4") is not True:
                failures.append("coeff16 storage accounting must include a sidecar above block4 storage")
            if storage.get("selected_storage_below_f32_coeff_reference") is not True:
                failures.append("coeff16 storage accounting must remain below the f32 coefficient reference")
        conclusion = str(coeff16_storagefix.get("conclusion", ""))
        if "storage-accounting smoke" not in conclusion or "not endpoint-run storage" not in conclusion:
            failures.append("coeff16 storage-accounting conclusion must explain the corrected accounting")

    paired_endpoint_edit_rgb_only = payload.get("endpoint_record_edit_rgb_only_paired_train_eval")
    if not isinstance(paired_endpoint_edit_rgb_only, dict) or paired_endpoint_edit_rgb_only.get("available") is not True:
        failures.append("missing paired endpoint-run versus endpoint-record-edit RGB-only train/eval comparison")
    else:
        ratios = paired_endpoint_edit_rgb_only.get("ratios")
        endpoint = paired_endpoint_edit_rgb_only.get("endpoint_run_16f")
        edit = paired_endpoint_edit_rgb_only.get("endpoint_record_edit_16f")
        if not isinstance(ratios, dict) or not isinstance(endpoint, dict) or not isinstance(edit, dict):
            failures.append("paired endpoint/edit RGB-only comparison missing ratio or 16f blocks")
        else:
            total_ratio = ratios.get("edit_to_endpoint_total_16f")
            render_ratio = ratios.get("edit_to_endpoint_render_16f")
            backward_ratio = ratios.get("edit_to_endpoint_backward_16f")
            for key, value in (
                ("edit_to_endpoint_total_16f", total_ratio),
                ("edit_to_endpoint_render_16f", render_ratio),
                ("edit_to_endpoint_backward_16f", backward_ratio),
            ):
                if not _finite_positive(value):
                    failures.append(f"paired endpoint/edit RGB-only comparison {key} is not positive finite")
            edit_storage = edit.get("storage_vs_full")
            endpoint_storage = endpoint.get("storage_vs_full")
            if not _finite_positive(edit_storage) or float(edit_storage) >= 0.05:
                failures.append("paired endpoint/edit RGB-only edit storage ratio no longer supports compact path")
            if not _finite_positive(endpoint_storage) or float(endpoint_storage) >= 0.20:
                failures.append("paired endpoint/edit RGB-only endpoint storage ratio no longer supports compact path")
            if (
                _finite_positive(edit_storage)
                and _finite_positive(endpoint_storage)
                and float(edit_storage) >= float(endpoint_storage)
            ):
                failures.append("paired endpoint/edit RGB-only comparison must preserve edit storage below endpoint storage")
            edit_psnr = edit.get("heldout_psnr")
            endpoint_psnr = endpoint.get("heldout_psnr")
            if (
                _finite_positive(edit_psnr)
                and _finite_positive(endpoint_psnr)
                and abs(float(edit_psnr) - float(endpoint_psnr)) > 1.0e-3
            ):
                failures.append("paired endpoint/edit RGB-only PSNR mismatch exceeds tolerance")
        acceptance = paired_endpoint_edit_rgb_only.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("paired endpoint/edit RGB-only comparison missing acceptance")
        else:
            for key in (
                "edit_storage_below_endpoint",
                "endpoint_record_edit_ok",
                "endpoint_run_ok",
                "psnr_matches",
                "edit_total_ratio_positive",
            ):
                if acceptance.get(key) is not True:
                    failures.append(f"paired endpoint/edit RGB-only acceptance {key} must be true")
        if paired_endpoint_edit_rgb_only.get("speed_read") != "slower_than_endpoint_run":
            failures.append("latest paired endpoint/edit RGB-only repeat should be recorded as slower than endpoint-run")
        scope = str(paired_endpoint_edit_rgb_only.get("scope", ""))
        if "not a stable benchmark" not in scope or "STAR UVT competitive claim" not in scope:
            failures.append("paired endpoint/edit RGB-only scope must reject stable speed or STAR claims")
        conclusion = str(paired_endpoint_edit_rgb_only.get("conclusion", ""))
        if (
            "noisy speed sign" not in conclusion
            or "latest longer RGB-only repeat is slower" not in conclusion
            or "needs replay optimization" not in conclusion
        ):
            failures.append("paired endpoint/edit RGB-only conclusion must keep noisy-speed scope explicit")

    paired_endpoint_edit_manual_vjp = payload.get("endpoint_record_edit_manual_vjp_paired_train_eval")
    if not isinstance(paired_endpoint_edit_manual_vjp, dict) or paired_endpoint_edit_manual_vjp.get("available") is not True:
        failures.append("missing paired endpoint-run versus endpoint-record-edit manual-VJP train/eval comparison")
    else:
        ratios = paired_endpoint_edit_manual_vjp.get("ratios")
        endpoint = paired_endpoint_edit_manual_vjp.get("endpoint_run_16f")
        edit = paired_endpoint_edit_manual_vjp.get("endpoint_record_edit_16f")
        if not isinstance(ratios, dict) or not isinstance(endpoint, dict) or not isinstance(edit, dict):
            failures.append("paired endpoint/edit manual-VJP comparison missing ratio or 16f blocks")
        else:
            total_ratio = ratios.get("edit_to_endpoint_total_16f")
            render_ratio = ratios.get("edit_to_endpoint_render_16f")
            backward_ratio = ratios.get("edit_to_endpoint_backward_16f")
            for key, value in (
                ("edit_to_endpoint_total_16f", total_ratio),
                ("edit_to_endpoint_render_16f", render_ratio),
                ("edit_to_endpoint_backward_16f", backward_ratio),
            ):
                if not _finite_positive(value):
                    failures.append(f"paired endpoint/edit manual-VJP comparison {key} is not positive finite")
            if _finite_positive(total_ratio) and float(total_ratio) <= 1.0:
                failures.append("paired endpoint/edit manual-VJP comparison unexpectedly beat endpoint-run; audit before claiming")
            if _finite_positive(render_ratio) and float(render_ratio) <= 1.0:
                failures.append("paired endpoint/edit manual-VJP render ratio should preserve forward replay as the bottleneck")
            edit_storage = edit.get("storage_vs_full")
            endpoint_storage = endpoint.get("storage_vs_full")
            if not _finite_positive(edit_storage) or float(edit_storage) >= 0.05:
                failures.append("paired endpoint/edit manual-VJP edit storage ratio no longer supports compact path")
            if not _finite_positive(endpoint_storage) or float(endpoint_storage) >= 0.20:
                failures.append("paired endpoint/edit manual-VJP endpoint storage ratio no longer supports compact path")
            edit_psnr = edit.get("heldout_psnr")
            endpoint_psnr = endpoint.get("heldout_psnr")
            if (
                _finite_positive(edit_psnr)
                and _finite_positive(endpoint_psnr)
                and abs(float(edit_psnr) - float(endpoint_psnr)) > 1.0e-3
            ):
                failures.append("paired endpoint/edit manual-VJP PSNR mismatch exceeds tolerance")
        optimizer_modes = paired_endpoint_edit_manual_vjp.get("optimizer_modes")
        if not isinstance(optimizer_modes, dict) or set(optimizer_modes.values()) != {"manual-vjp"}:
            failures.append("paired endpoint/edit manual-VJP comparison must record manual-vjp optimizer modes")
        if paired_endpoint_edit_manual_vjp.get("speed_read") != "slower_than_endpoint_run":
            failures.append("paired endpoint/edit manual-VJP repeat should be recorded as slower than endpoint-run")
        scope = str(paired_endpoint_edit_manual_vjp.get("scope", ""))
        if "not a stable benchmark" not in scope or "STAR UVT competitive claim" not in scope:
            failures.append("paired endpoint/edit manual-VJP scope must reject stable speed or STAR claims")
        conclusion = str(paired_endpoint_edit_manual_vjp.get("conclusion", ""))
        if (
            "noisy speed sign" not in conclusion
            or "needs replay optimization" not in conclusion
            or "speed-competitive claim" not in conclusion
        ):
            failures.append("paired endpoint/edit manual-VJP conclusion must keep replay-optimization scope explicit")

    segment_tape_autograd = payload.get("segment_tape_autograd_smoke")
    if not isinstance(segment_tape_autograd, dict) or segment_tape_autograd.get("available") is not True:
        failures.append("missing segment-tape autograd smoke")
    else:
        acceptance = segment_tape_autograd.get("acceptance")
        if not isinstance(acceptance, dict):
            failures.append("segment-tape autograd smoke missing acceptance")
        else:
            for key in ("all_modes_ok", "owner_run_segments_below_full", "owner_run_vjp_under_segment_cap"):
                if acceptance.get(key) is not True:
                    failures.append(f"segment-tape autograd smoke acceptance {key} must be true")
        if segment_tape_autograd.get("full_trainer_claim") is not False:
            failures.append("segment-tape autograd smoke must not claim full trainer coverage")
        if segment_tape_autograd.get("full_geometry_gradient_claim") is not False:
            failures.append("segment-tape autograd smoke must not claim geometry-gradient coverage")
        if segment_tape_autograd.get("density_independent_depth_claim") is not False:
            failures.append("segment-tape autograd smoke must not claim density-independent depth coverage")
        rel = segment_tape_autograd.get("max_grad_rel_error_vs_manual_vjp")
        if not _finite_positive(rel) or float(rel) > 2.0e-5:
            failures.append("segment-tape autograd max gradient relative error exceeds 2e-5")
        modes = segment_tape_autograd.get("modes")
        if not isinstance(modes, list) or set(modes) != {"direct_atomic_grad_only", "direct_atomic_track"}:
            failures.append("segment-tape autograd smoke must cover direct and track VJP modes")
        ratio = segment_tape_autograd.get("owner_run_segments_vs_full")
        if not _finite_positive(ratio) or float(ratio) >= 0.25:
            failures.append("segment-tape autograd smoke owner-run ratio no longer supports compression")
        conclusion = str(segment_tape_autograd.get("conclusion", ""))
        if "PyTorch autograd wrapper" not in conclusion or "explicit Metal VJP" not in conclusion:
            failures.append("segment-tape autograd conclusion must mention wrapper and explicit VJP comparison")

    return {
        "benchmark": "world_foam_lane2_fused_slab_status_summary_verifier",
        "status": "ok" if not failures else "failed",
        "summary_json": str(args.summary_json),
        "failures": failures,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the fused slab status summary remains scoped correctly.")
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--expected-winner", default="direct_atomic_grad_only")
    parser.add_argument("--max-psnr-spread", type=float, default=1.0e-3)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_DIR / "2026-05-15_fused_slab_mixed_status_summary_verifier.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = verify(args)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
