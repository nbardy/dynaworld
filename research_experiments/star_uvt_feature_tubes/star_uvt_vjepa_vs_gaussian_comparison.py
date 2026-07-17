from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

try:
    from .report_artifacts import ROOT, load_report_json, write_report_json, write_report_text
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import ROOT, load_report_json, write_report_json, write_report_text

STAR_VJEPA_TARGET_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_chunkedtarget_lr005_5step.json"
)
STAR_VJEPA_TARGET_CACHED_CHUNKS_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_cachedchunks_lr005_5step.json"
)
STAR_VJEPA_TARGET_GRID_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_lr005_5step.json"
)
STAR_VJEPA_TARGET_GRID_MEDIA_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_lr005_20step_media.json"
)
STAR_VJEPA_TARGET_GRID_RGB_AUX_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbaux1_lr005_20step_media.json"
)
STAR_VJEPA_TARGET_GRID_RGB_AUX10_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbaux10_lr005_20step_media.json"
)
STAR_VJEPA_TARGET_GRID_RGB_AUX10_100STEP_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbaux10_lr005_100step_media.json"
)
STAR_VJEPA_TARGET_GRID_RGBWARM20_AUX10_100STEP_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbwarm20_aux10_lr005_100step_media.json"
)
STAR_VJEPA_TARGET_GRID_RGBPROBE10_20STEP_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_20step_media.json"
)
STAR_VJEPA_TARGET_GRID_RGBPROBE10_100STEP_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_100step_media.json"
)
STAR_VJEPA_TARGET_GRID_RGBPROBE10_300STEP_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_300step_media.json"
)
STAR_VJEPA_TARGET_GRID_RGBPROBE10_300STEP_CHECKPOINT_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_300step_checkpoint.json"
)
STAR_VJEPA_TARGET_GRID_RGBPROBE10_RESUME300_FROM300_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_resume300_from300_checkpoint_media.json"
)
STAR_VJEPA_TARGET_GRID_RGBPROBE40_FEATURE025_RESUME200_FROM600_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature025_lr005_resume200_from600_checkpoint_media.json"
)
STAR_VJEPA_TARGET_GRID_RGBPROBE_BALANCE_RESUME200_FROM800_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_balance_resume200_from800_checkpoint_media.json"
)
STAR_VJEPA_TARGET_GRID_RGBPROBE40_FEATURE05_RESUME100_FROM1000_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature05_lr005_resume100_from1000_checkpoint_media.json"
)
STAR_VJEPA_TARGET_GRID_RGBPROBE_RECOVER_RESUME100_FROM1100_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_recover_resume100_from1100_checkpoint_media.json"
)
STAR_VJEPA_TARGET_GRID_RGBPROBE40_FEATURE075_RESUME50_FROM1200_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature075_lr005_resume50_from1200_checkpoint_media.json"
)
STAR_VJEPA_TARGET_GRID_RGBPROBE40_FEATURE1_RESUME50_FROM1250_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume50_from1250_checkpoint_media.json"
)
STAR_VJEPA_TARGET_GRID_RGBPROBE40_FEATURE1_RESUME100_FROM1300_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_checkpoint_media.json"
)
STAR_VJEPA_TARGET_GRID_SPARSEFORWARD_BATCHED_RESUME100_FROM1300_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_64f512_from1300_100step_media.json"
)
STAR_VJEPA_TARGET_GRID_SPARSEFORWARD_BATCHED_LR001_RESUME100_FROM1300_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_lr001_64f512_from1300_100step_media.json"
)
STAR_VJEPA_TARGET_GRID_SPARSEFORWARD_BATCHED_LR001_RESUME50_FROM1400_LR005SPARSE_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_lr001_resume50_from1400_lr005sparse_checkpointselect.json"
)
STAR_VJEPA_TARGET_GRID_SPARSEFORWARD_BATCHED_LR001_RESUME50_FROM1400_LR001SPARSE_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_lr001_resume50_from1400_lr001sparse_checkpointselect.json"
)
STAR_VJEPA_TARGET_GRID_SPARSEFORWARD_BATCHED_LR001_RESUME50_FROM1450_LR005SPARSE_MEDIA_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_lr001_resume50_from1450_lr005sparse_media.json"
)
STAR_VJEPA_TARGET_GRID_AUTOGRAD_RGBAUX1_PROBEINIT_FROM1500_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_autograd_rgbaux1_probeinit_from1500_20step_media.json"
)
STAR_VJEPA_TARGET_GRID_SPARSEVISUAL_MIX_FROM1500_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_mix_from1500_lr001_50step_media.json"
)
STAR_VJEPA_TARGET_GRID_SPARSEVISUAL_PATCH2X2_FROM1500_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_patch2x2_from1500_lr001_50step_media.json"
)
STAR_VJEPA_TARGET_GRID_SPARSEVISUAL_PATCHMEAN64_FROM1500_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_patchmean64_from1500_lr001_50step_media.json"
)
STAR_VJEPA_TARGET_GRID_SPARSEVISUAL_TARGETAREA64_FROM1500_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_from1500_lr001_50step_media.json"
)
STAR_VJEPA_TARGET_GRID_SPARSEVISUAL_TARGETAREA64_PHASE2X2_FROM1500_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_phase2x2_from1500_lr001_50step_media.json"
)
STAR_VJEPA_TARGET_GRID_SPARSEVISUAL_TARGETAREA64_FULLCELL8_FROM1500_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_from1500_lr001_5step_media.json"
)
STAR_VJEPA_TARGET_GRID_SPARSEVISUAL_TARGETAREA64_FULLCELL8_MANUALVJP_FROM1500_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_manualvjp_from1500_lr001_5step_media.json"
)
STAR_VJEPA_TARGET_GRID_SPARSEVISUAL_TARGETAREA64_FULLCELL8_STARONLY_FROM1500_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_staronly_from1500_lr001_5step_media.json"
)
STAR_VJEPA_TARGET_GRID_SPARSEVISUAL_TARGETAREA64_FULLCELL8_FASTGELU_FROM1500_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_fastgelu_from1500_lr001_5step_media.json"
)
STAR_VJEPA_TARGET_GRID_SPARSEVISUAL_TARGETAREA64_FULLCELL8_LINEAR_FROM1500_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_linear_from1500_lr001_5step_media.json"
)
STAR_VJEPA_TARGET_GRID_SPARSEVISUAL_TARGETAREA64_FULLCELL8_HIDDEN32_FROM1500_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsevisual_targetarea64_fullcell8_hidden32_from1500_lr001_5step_media.json"
)
STAR_RENDERED_FEATURE_RGB_PROBE_FROM1500_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_rendered_feature_rgb_probe_from1500_sparsepixels_hidden64_lr01_100step_media.json"
)
STAR_RENDERED_FEATURE_RGB_PROBE_STRATIFIED64_FROM1500_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_rendered_feature_rgb_probe_from1500_stratified64_hidden64_lr01_100step_media.json"
)
STAR_SPARSE_VISUAL_VJP_STRATIFIED64_FROZENPROBE_FROM1500_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_vjp_from1500_stratified64_frozenprobe_lr001_50step_media.json"
)
STAR_SPARSE_VISUAL_VJP_STRATIFIED64_JOINTPROBE_FROM1500_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_vjp_from1500_stratified64_jointprobe_lr001_50step_media.json"
)
STAR_VJEPA_TARGET_GRID_RGB_PROBE_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden64_lr01_1000step.json"
)
STAR_SELECTED_RGB_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_firstclass_testvideo_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_20step_media.json"
)
MULTICAM_VJEPA_DEFAULT = ROOT / "outputs/benchmarks/multicam_vjepa_default_128px_16f_8192splats_2026-05-02.json"
MULTICAM_VJEPA_FAST_CAMERA = ROOT / "outputs/benchmarks/multicam_vjepa_fast_camera_128px_16f_8192splats_2026-05-02.json"
MULTICAM_NO_VJEPA = ROOT / "outputs/benchmarks/multicam_no_vjepa_unconditioned_tokens_128px_16f_8192splats_2026-05-02.json"
GAUSSIAN_300_VJEPA_LOSS_LOG = (
    ROOT
    / "outputs/run_logs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_vjepa_loss_8192splats_token_capacity_run_20260516_202029.log"
)
GAUSSIAN_300_RECON_ONLY_LOG = (
    ROOT
    / "outputs/run_logs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_8192splats_token_capacity_run_20260516_222130.log"
)
GAUSSIAN_300_MULTIRES_LOG = (
    ROOT / "outputs/run_logs/dynaworld_300clips_3k_multires_256to512_framecache_noprofile_20260517_150143.log"
)
RENDERER_MATRIX_MD = ROOT / "outputs/benchmarks/2026-05-19_renderer_scaling_report.md"

TIMING_RE = re.compile(r"Timing step (?P<step>\d+): (?P<body>.*)")
PAIR_RE = re.compile(r"(?P<key>[A-Za-z0-9_/-]+)=(?P<value>-?(?:\d+\.?\d*|\.\d+))s")
MEAN_RE = re.compile(r"mean=(?P<mean>-?(?:\d+\.?\d*|\.\d+))s")
NAN_STEP_RE = re.compile(r"Loss: nan.*?\|\s*(?P<step>\d+)/(?P<total>\d+)")


def _mean_from_summary(text: str | None) -> float | None:
    if not text:
        return None
    match = MEAN_RE.search(text)
    return float(match.group("mean")) if match else None


def _summary(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def _parse_timing_log(path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = TIMING_RE.search(line)
        if not match:
            continue
        values: dict[str, float | int] = {"step": int(match.group("step"))}
        for pair in PAIR_RE.finditer(match.group("body")):
            values[pair.group("key")] = float(pair.group("value"))
        rows.append(values)

    keys = sorted({key for row in rows for key in row if key != "step"})
    summaries = {
        key: _summary([float(row[key]) for row in rows if key in row])
        for key in keys
    }
    return {
        "path": str(path.relative_to(ROOT)),
        "timing_rows": rows,
        "timing_summary": summaries,
        "timing_step_count": len(rows),
        "first_timing_step": rows[0]["step"] if rows else None,
        "last_timing_step": rows[-1]["step"] if rows else None,
    }


def _parse_multires_status(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    switch_seen = "Render size schedule: step 2400 switched 256->512" in text
    nan_match = NAN_STEP_RE.search(text)
    return {
        "path": str(path.relative_to(ROOT)),
        "render_size_switch_seen": switch_seen,
        "first_nan_step": int(nan_match.group("step")) if nan_match else None,
        "first_nan_total_steps": int(nan_match.group("total")) if nan_match else None,
        "status": "blocked_at_512_promotion" if switch_seen and nan_match else "unknown",
    }


def _row(
    *,
    family: str,
    route: str,
    artifact: Path,
    status: str,
    frames: int,
    resolution: int,
    primitives: int,
    feature_dim: int | None,
    feature_source: str,
    target_or_loss: str,
    measured_steps: int | None,
    step_s: float | None,
    backward_s: float | None,
    render_s: float | None,
    target_eval_s: float | None,
    model_forward_s: float | None,
    feature_load_s: float | None,
    sample_s: float | None,
    notes: str,
) -> dict[str, Any]:
    return {
        "family": family,
        "route": route,
        "artifact": str(artifact.relative_to(ROOT)),
        "status": status,
        "frames": frames,
        "resolution": resolution,
        "primitives": primitives,
        "feature_dim": feature_dim,
        "feature_source": feature_source,
        "target_or_loss": target_or_loss,
        "measured_steps": measured_steps,
        "step_s": step_s,
        "backward_s": backward_s,
        "render_s": render_s,
        "target_eval_s": target_eval_s,
        "model_forward_s": model_forward_s,
        "feature_load_s": feature_load_s,
        "sample_s": sample_s,
        "notes": notes,
    }


def _star_vjepa_row(path: Path, route: str) -> dict[str, Any]:
    payload = load_report_json(path)
    timing = payload["mean_timing_ms"]
    target = payload["feature_target"]
    load_ms = payload.get("feature_target_load_ms")
    cached_mib = target.get("cached_target_mib")
    target_grid_mib = target.get("target_grid_mib")
    cache_note = "" if cached_mib is None else f"; cached_target={cached_mib:.1f}MiB"
    target_grid_note = "" if target_grid_mib is None else f"; target_grid={target_grid_mib:.1f}MiB"
    component_note = ""
    if payload.get("start_rgb_loss") is not None:
        component_note = (
            f"; feature_loss {payload['start_feature_target_loss']:.6f}->{payload['end_feature_target_loss']:.6f}; "
            f"rgb_loss {payload['start_rgb_loss']:.6f}->{payload['end_rgb_loss']:.6f}; "
            f"rgb_psnr {payload['start_rgb_psnr']:.3f}->{payload['end_rgb_psnr']:.3f}"
        )
    if payload.get("start_rgb_probe_loss") is not None:
        component_note = (
            f"{component_note}; feature_loss "
            f"{payload['start_feature_target_loss']:.6f}->{payload['end_feature_target_loss']:.6f}; "
            f"rgb_probe_loss {payload['start_rgb_probe_loss']:.6f}->{payload['end_rgb_probe_loss']:.6f}; "
            f"rgb_probe_psnr {payload['start_rgb_probe_psnr']:.3f}->{payload['end_rgb_probe_psnr']:.3f}; "
            f"rgb_probe_weight={payload.get('rgb_probe_loss_weight')}"
        )
    schedule = payload.get("feature_target_weight_schedule") or []
    schedule_note = ""
    if len(schedule) > 1:
        schedule_note = "; schedule=" + ",".join(
            f"{stage['label']}:{stage['start_step']}-{stage['end_step']}="
            f"f{stage['loss_weight']}/rgb{stage['rgb_loss_weight']}"
            for stage in schedule
        )
    global_step_note = ""
    if payload.get("global_step_offset"):
        global_step_note = (
            f"; global_steps={payload.get('start_global_step')}->{payload.get('end_global_step')}"
        )
    image_vjp_note = ""
    if payload.get("feature_target_image_vjp_mode"):
        image_vjp_note = f"; image_vjp={payload['feature_target_image_vjp_mode']}"
    sparse_note = ""
    if payload.get("mean_sparse_pixel_count") is not None:
        sparse_note = (
            f"; sparse_pixels={int(payload['mean_sparse_pixel_count'])}"
            f" ({payload.get('mean_sparse_pixel_fraction', 0.0):.6f})"
        )
    if payload.get("sparse_visual_enabled"):
        source = payload.get("sparse_visual_pixel_source")
        patch_shape = payload.get("sparse_visual_patch_shape")
        patch_phase_shape = payload.get("sparse_visual_patch_phase_shape")
        loss_basis = payload.get("sparse_visual_loss_basis")
        loss_vjp_mode = payload.get("sparse_visual_loss_vjp_mode")
        loss_cells = payload.get("mean_sparse_visual_loss_sample_count")
        support_note = "" if source is None else f"; sparse_visual_source={source}"
        if patch_shape is not None:
            support_note += f"; sparse_visual_patch={patch_shape}"
        if patch_phase_shape is not None:
            support_note += f"; sparse_visual_patch_phase_shape={patch_phase_shape}"
        if loss_basis is not None:
            support_note += f"; sparse_visual_basis={loss_basis}"
        if loss_vjp_mode is not None:
            support_note += f"; sparse_visual_loss_vjp={loss_vjp_mode}"
        if loss_cells is not None:
            support_note += f"; sparse_visual_loss_cells={int(loss_cells)}"
        sparse_note = (
            f"{sparse_note}; sparse_visual_loss {payload['start_sparse_visual_loss']:.6f}->"
            f"{payload['end_sparse_visual_loss']:.6f}; "
            f"sparse_visual_psnr {payload['start_sparse_visual_psnr']:.3f}->"
            f"{payload['end_sparse_visual_psnr']:.3f}; "
            f"full_rgb_psnr={payload.get('final_full_rgb_psnr'):.3f}; "
            f"sparse_visual_pixels={int(payload['mean_sparse_visual_pixel_count'])}"
            f" ({payload.get('mean_sparse_visual_pixel_fraction', 0.0):.6f})"
            f"{support_note}"
        )
    last20_note = ""
    step_timings = payload.get("step_timings_ms") or []
    if len(step_timings) >= 20:
        last20 = step_timings[-20:]
        last20_step_ms = statistics.fmean(float(item["step_ms"]) for item in last20)
        last20_backward_ms = statistics.fmean(float(item["backward_ms"]) for item in last20)
        last20_render_ms = statistics.fmean(float(item["render_forward_ms"]) for item in last20)
        last20_note = (
            f"; last20_step/back/render={last20_step_ms / 1000.0:.3f}/"
            f"{last20_backward_ms / 1000.0:.3f}/{last20_render_ms / 1000.0:.3f}s"
        )
    return _row(
        family="star_uvt_feature",
        route=route,
        artifact=path,
        status="pass" if payload.get("pass") else "nonpassing",
        frames=int(payload["frames"]),
        resolution=int(payload["size"]),
        primitives=int(payload.get("tube_count") or payload["tubes"]),
        feature_dim=int(payload["feature_dim"]),
        feature_source=f"cached {target['extractor']} tokens",
        target_or_loss=f"{target['materialization']} adapted target {target['source_shape']} -> {target['adapted_shape']}",
        measured_steps=len(payload.get("losses") or []),
        step_s=timing["step_ms"] / 1000.0,
        backward_s=timing["backward_ms"] / 1000.0,
        render_s=timing["render_forward_ms"] / 1000.0,
        target_eval_s=(timing.get("feature_target_ms", 0.0) + timing.get("rgb_probe_loss_ms", 0.0)) / 1000.0,
        model_forward_s=None,
        feature_load_s=None if load_ms is None else float(load_ms) / 1000.0,
        sample_s=None,
        notes=(
            f"loss {payload['start_loss']:.6f}->{payload['end_loss']:.6f}; "
            f"tile_overflow={payload.get('tile_overflow_sum')}{cache_note}{target_grid_note}"
            f"{component_note}{schedule_note}{global_step_note}{image_vjp_note}{sparse_note}{last20_note}"
        ),
    )


def _star_target_grid_rgb_probe_row() -> dict[str, Any]:
    payload = load_report_json(STAR_VJEPA_TARGET_GRID_RGB_PROBE_RESULT)
    timing = payload["mean_timing_ms"]
    target = payload["feature_target"]
    load_ms = payload.get("feature_target_load_ms")
    return _row(
        family="star_uvt_feature_probe",
        route="target-grid V-JEPA feature-to-RGB probe, hidden64",
        artifact=STAR_VJEPA_TARGET_GRID_RGB_PROBE_RESULT,
        status="pass" if payload.get("pass") else "nonpassing",
        frames=int(payload["frames"]),
        resolution=int(payload["size"]),
        primitives=0,
        feature_dim=int(payload["feature_dim"]),
        feature_source=f"cached {target['extractor']} target grid",
        target_or_loss="train FeatureToColor directly on cached target_grid -> downsampled RGB",
        measured_steps=len(payload.get("losses") or []),
        step_s=timing["step_ms"] / 1000.0,
        backward_s=timing["backward_ms"] / 1000.0,
        render_s=None,
        target_eval_s=None,
        model_forward_s=timing["forward_loss_ms"] / 1000.0,
        feature_load_s=None if load_ms is None else float(load_ms) / 1000.0,
        sample_s=None,
        notes=(
            f"grid_psnr {payload['start_grid_psnr']:.3f}->{payload['final_grid_psnr']:.3f}; "
            f"full_upsample_psnr={payload['final_full_psnr']:.3f}; "
            f"checkpoint={payload.get('checkpoint')}; "
            f"wandb={payload.get('wandb_run_id')}"
        ),
    )


def _star_rendered_feature_rgb_probe_row(path: Path, route: str) -> dict[str, Any]:
    payload = load_report_json(path)
    timing = payload["mean_timing_ms"]
    return _row(
        family="star_uvt_feature_probe",
        route=route,
        artifact=path,
        status="pass" if payload.get("pass") else "nonpassing",
        frames=int(payload["frames"]),
        resolution=int(payload["size"]),
        primitives=int(payload["tubes"]),
        feature_dim=int(payload["feature_dim"]),
        feature_source="rendered sparse pixels from selected sparse-forward STAR 1500 checkpoint",
        target_or_loss="train FeatureToColor on sparse rendered feature pixels -> RGB",
        measured_steps=len(payload.get("losses") or []),
        step_s=timing["step_ms"] / 1000.0,
        backward_s=timing["backward_ms"] / 1000.0,
        render_s=timing["render_forward_ms"] / 1000.0,
        target_eval_s=timing["colorize_loss_ms"] / 1000.0,
        model_forward_s=None,
        feature_load_s=None,
        sample_s=None,
        notes=(
            f"sparse_sample_psnr {payload['start_sparse_sample_psnr']:.3f}->"
            f"{payload['end_sparse_sample_psnr']:.3f}; "
            f"full_psnr={payload['final_full_psnr']:.3f}; "
            f"{payload['pixel_source']} pixels={int(payload['mean_sample_pixel_count'])} "
            f"({payload['mean_sample_pixel_fraction']:.6f}); "
            f"train_star_model={payload.get('train_star_model')}; "
            f"train_colorizer={payload.get('train_colorizer')}; "
            f"model_grad_seen={payload.get('model_grad_seen')}; "
            f"colorizer_grad_seen={payload.get('colorizer_grad_seen')}; "
            f"wandb={payload.get('wandb_run_id')}"
        ),
    )


def _star_rgb_row() -> dict[str, Any]:
    payload = load_report_json(STAR_SELECTED_RGB_RESULT)
    timing = payload["mean_timing_ms"]
    return _row(
        family="star_uvt_feature",
        route="selected RGB feature diagnostic, gradcache_reduce_vec4",
        artifact=STAR_SELECTED_RGB_RESULT,
        status="pass" if payload.get("pass") else "nonpassing",
        frames=int(payload["frames"]),
        resolution=int(payload["size"]),
        primitives=int(payload.get("tube_count") or payload["tubes"]),
        feature_dim=int(payload["feature_dim"]),
        feature_source="none; RGB target through FeatureToColor",
        target_or_loss="RGB reconstruction",
        measured_steps=len(payload.get("losses") or []),
        step_s=timing["step_ms"] / 1000.0,
        backward_s=timing["backward_ms"] / 1000.0,
        render_s=timing["render_forward_ms"] / 1000.0,
        target_eval_s=timing.get("colorize_loss_ms", 0.0) / 1000.0,
        model_forward_s=None,
        feature_load_s=None,
        sample_s=None,
        notes=(
            "current fastest 512px STAR feature diagnostic; not a V-JEPA target route; "
            f"PSNR={payload.get('end_psnr'):.3f}"
        ),
    )


def _multicam_row(path: Path, route: str, feature_source: str) -> dict[str, Any]:
    payload = load_report_json(path)
    summary = payload["summary"]
    return _row(
        family="gaussian_token_multicam",
        route=route,
        artifact=path,
        status="timed",
        frames=16,
        resolution=128,
        primitives=8192,
        feature_dim=768 if "vjepa" in feature_source.lower() else None,
        feature_source=feature_source,
        target_or_loss="RGB recon with cached conditioning",
        measured_steps=int(payload["steps"]),
        step_s=_mean_from_summary(payload.get("measured_step_total_summary")),
        backward_s=_mean_from_summary(summary.get("backward")),
        render_s=_mean_from_summary(summary.get("render_views_total")),
        target_eval_s=_mean_from_summary(summary.get("recon_losses_total")),
        model_forward_s=_mean_from_summary(summary.get("model_forward_decode")),
        feature_load_s=_mean_from_summary(summary.get("feature_load_or_memory_hit")),
        sample_s=_mean_from_summary(summary.get("sample_clip")),
        notes=f"device={payload.get('device')}; variant={payload.get('model_variant')}",
    )


def _gaussian_log_row(path: Path, route: str, feature_source: str, target_or_loss: str) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed = _parse_timing_log(path)
    timing = parsed["timing_summary"]

    def mean(key: str) -> float | None:
        summary = timing.get(key)
        return None if summary is None else float(summary["mean"])

    row = _row(
        family="gaussian_token_300clip",
        route=route,
        artifact=path,
        status="timed",
        frames=64,
        resolution=512,
        primitives=8192,
        feature_dim=768,
        feature_source=feature_source,
        target_or_loss=target_or_loss,
        measured_steps=int(parsed["timing_step_count"]),
        step_s=mean("step_total"),
        backward_s=mean("backward"),
        render_s=mean("render_view_total"),
        target_eval_s=mean("vjepa_feature_loss"),
        model_forward_s=mean("forward_decode"),
        feature_load_s=None,
        sample_s=mean("sample_clip"),
        notes=f"profiled timing lines {parsed['first_timing_step']}..{parsed['last_timing_step']}",
    )
    return row, parsed


def _build_report() -> dict[str, Any]:
    vjepa_loss_row, vjepa_loss_log = _gaussian_log_row(
        GAUSSIAN_300_VJEPA_LOSS_LOG,
        route="300-clip cached conditioning + differentiable prediction-side V-JEPA loss",
        feature_source="cached V-JEPA conditioning plus frozen V-JEPA on predicted video",
        target_or_loss="RGB recon + differentiable V-JEPA feature loss",
    )
    recon_only_row, recon_only_log = _gaussian_log_row(
        GAUSSIAN_300_RECON_ONLY_LOG,
        route="300-clip cached conditioning, recon-only",
        feature_source="cached V-JEPA conditioning",
        target_or_loss="RGB recon only",
    )
    rows = [
        _star_vjepa_row(STAR_VJEPA_TARGET_RESULT, "chunked V-JEPA target, gradcache_reduce_vec4"),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_CACHED_CHUNKS_RESULT,
            "cached-chunks V-JEPA target, gradcache_reduce_vec4",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_RESULT,
            "target-grid V-JEPA loss, gradcache_reduce_vec4",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_MEDIA_RESULT,
            "target-grid V-JEPA loss 20-step media, gradcache_reduce_vec4",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_RGB_AUX_RESULT,
            "target-grid V-JEPA + RGB aux1 20-step media, gradcache_reduce_vec4",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_RGB_AUX10_RESULT,
            "target-grid V-JEPA + RGB aux10 20-step media, gradcache_reduce_vec4",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_RGB_AUX10_100STEP_RESULT,
            "target-grid V-JEPA + RGB aux10 100-step media, gradcache_reduce_vec4",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_RGBWARM20_AUX10_100STEP_RESULT,
            "target-grid V-JEPA + RGB warm20->aux10 100-step media, gradcache_reduce_vec4",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_RGBPROBE10_20STEP_RESULT,
            "target-grid V-JEPA + frozen RGB-probe10 20-step media, gradcache_reduce_vec4",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_RGBPROBE10_100STEP_RESULT,
            "target-grid V-JEPA + frozen RGB-probe10 100-step media, gradcache_reduce_vec4",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_RGBPROBE10_300STEP_RESULT,
            "target-grid V-JEPA + frozen RGB-probe10 300-step media, gradcache_reduce_vec4",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_RGBPROBE10_300STEP_CHECKPOINT_RESULT,
            "target-grid V-JEPA + frozen RGB-probe10 300-step checkpoint/no-media, gradcache_reduce_vec4",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_RGBPROBE10_RESUME300_FROM300_RESULT,
            "target-grid V-JEPA + frozen RGB-probe10 resume300-from300 media, gradcache_reduce_vec4",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_RGBPROBE40_FEATURE025_RESUME200_FROM600_RESULT,
            "target-grid V-JEPA + frozen RGB-probe40 feature025 resume200-from600 media, gradcache_reduce_vec4",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_RGBPROBE_BALANCE_RESUME200_FROM800_RESULT,
            "target-grid V-JEPA + frozen RGB-probe scheduled balance resume200-from800 media, gradcache_reduce_vec4",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_RGBPROBE40_FEATURE05_RESUME100_FROM1000_RESULT,
            "target-grid V-JEPA + frozen RGB-probe40 feature05 resume100-from1000 media, gradcache_reduce_vec4",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_RGBPROBE_RECOVER_RESUME100_FROM1100_RESULT,
            "target-grid V-JEPA + frozen RGB-probe recover schedule resume100-from1100 media, gradcache_reduce_vec4",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_RGBPROBE40_FEATURE075_RESUME50_FROM1200_RESULT,
            "target-grid V-JEPA + frozen RGB-probe40 feature075 resume50-from1200 media, gradcache_reduce_vec4",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_RGBPROBE40_FEATURE1_RESUME50_FROM1250_RESULT,
            "target-grid V-JEPA + frozen RGB-probe40 feature1 resume50-from1250 media, gradcache_reduce_vec4",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_RGBPROBE40_FEATURE1_RESUME100_FROM1300_RESULT,
            "target-grid V-JEPA + frozen RGB-probe40 feature1 resume100-from1300 media, gradcache_reduce_vec4",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_SPARSEFORWARD_BATCHED_RESUME100_FROM1300_RESULT,
            "target-grid V-JEPA + frozen RGB-probe40 feature1 sparse-forward batched VJP resume100-from1300 media, gradcache_reduce_vec4",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_SPARSEFORWARD_BATCHED_LR001_RESUME100_FROM1300_RESULT,
            "target-grid V-JEPA + frozen RGB-probe40 feature1 lr001 sparse-forward batched VJP resume100-from1300 media, gradcache_reduce_vec4",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_SPARSEFORWARD_BATCHED_LR001_RESUME50_FROM1400_LR005SPARSE_RESULT,
            "target-grid V-JEPA + frozen RGB-probe40 feature1 checkpoint-select lr001 resume50-from-lr005sparse-1400, sparse-forward batched VJP",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_SPARSEFORWARD_BATCHED_LR001_RESUME50_FROM1400_LR001SPARSE_RESULT,
            "target-grid V-JEPA + frozen RGB-probe40 feature1 checkpoint-select lr001 resume50-from-lr001sparse-1400, sparse-forward batched VJP",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_SPARSEFORWARD_BATCHED_LR001_RESUME50_FROM1450_LR005SPARSE_MEDIA_RESULT,
            "target-grid V-JEPA + frozen RGB-probe40 feature1 lr001 resume50-from-lr005sparse-1450 media, sparse-forward batched VJP",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_AUTOGRAD_RGBAUX1_PROBEINIT_FROM1500_RESULT,
            "target-grid V-JEPA + RGB aux1 probe-init resume20-from-sparse1500 media, autograd",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_SPARSEVISUAL_MIX_FROM1500_RESULT,
            "target-grid V-JEPA + frozen RGB-probe40 + sparse visual VJP mix from sparse1500, sparse-forward batched VJP",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_SPARSEVISUAL_PATCH2X2_FROM1500_RESULT,
            "target-grid V-JEPA + frozen RGB-probe40 + patch2x2 sparse visual VJP from sparse1500, sparse-forward batched VJP",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_SPARSEVISUAL_PATCHMEAN64_FROM1500_RESULT,
            "target-grid V-JEPA + frozen RGB-probe40 + patch-mean64 sparse visual VJP from sparse1500, sparse-forward batched VJP",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_SPARSEVISUAL_TARGETAREA64_FROM1500_RESULT,
            "target-grid V-JEPA + frozen RGB-probe40 + target-area64 sparse visual VJP from sparse1500, sparse-forward batched VJP",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_SPARSEVISUAL_TARGETAREA64_PHASE2X2_FROM1500_RESULT,
            "target-grid V-JEPA + frozen RGB-probe40 + target-area64 phased 2x2 sparse visual VJP from sparse1500, sparse-forward batched VJP",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_SPARSEVISUAL_TARGETAREA64_FULLCELL8_FROM1500_RESULT,
            "target-grid V-JEPA + frozen RGB-probe40 + target-area64 full-cell8 visual VJP from sparse1500, sparse-forward batched VJP",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_SPARSEVISUAL_TARGETAREA64_FULLCELL8_MANUALVJP_FROM1500_RESULT,
            "target-grid V-JEPA + frozen RGB-probe40 + target-area64 full-cell8 manual hidden64 visual VJP from sparse1500, sparse-forward batched VJP",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_SPARSEVISUAL_TARGETAREA64_FULLCELL8_STARONLY_FROM1500_RESULT,
            "target-grid V-JEPA + frozen RGB-probe40 + target-area64 full-cell8 star-only manual hidden64 visual VJP from sparse1500, sparse-forward batched VJP",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_SPARSEVISUAL_TARGETAREA64_FULLCELL8_FASTGELU_FROM1500_RESULT,
            "target-grid V-JEPA + frozen RGB-probe40 + target-area64 full-cell8 fast-GELU manual hidden64 visual VJP from sparse1500, sparse-forward batched VJP",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_SPARSEVISUAL_TARGETAREA64_FULLCELL8_LINEAR_FROM1500_RESULT,
            "target-grid V-JEPA + frozen RGB-probe40 + target-area64 full-cell8 manual linear visual VJP from sparse1500, sparse-forward batched VJP",
        ),
        _star_vjepa_row(
            STAR_VJEPA_TARGET_GRID_SPARSEVISUAL_TARGETAREA64_FULLCELL8_HIDDEN32_FROM1500_RESULT,
            "target-grid V-JEPA + frozen RGB-probe40 + target-area64 full-cell8 manual hidden32 visual VJP from sparse1500, sparse-forward batched VJP",
        ),
        _star_rendered_feature_rgb_probe_row(
            STAR_RENDERED_FEATURE_RGB_PROBE_FROM1500_RESULT,
            "rendered sparse-pixel feature-to-RGB probe from sparse1500, hidden64",
        ),
        _star_rendered_feature_rgb_probe_row(
            STAR_RENDERED_FEATURE_RGB_PROBE_STRATIFIED64_FROM1500_RESULT,
            "rendered stratified64 feature-to-RGB probe from sparse1500, hidden64",
        ),
        _star_rendered_feature_rgb_probe_row(
            STAR_SPARSE_VISUAL_VJP_STRATIFIED64_FROZENPROBE_FROM1500_RESULT,
            "sparse visual VJP from sparse1500, stratified64 frozen target-grid probe",
        ),
        _star_rendered_feature_rgb_probe_row(
            STAR_SPARSE_VISUAL_VJP_STRATIFIED64_JOINTPROBE_FROM1500_RESULT,
            "sparse visual VJP from sparse1500, stratified64 joint STAR+colorizer",
        ),
        _star_target_grid_rgb_probe_row(),
        _star_rgb_row(),
        _multicam_row(
            MULTICAM_VJEPA_FAST_CAMERA,
            route="multicam cached V-JEPA fast camera",
            feature_source="cached V-JEPA conditioning",
        ),
        _multicam_row(
            MULTICAM_VJEPA_DEFAULT,
            route="multicam cached V-JEPA default camera",
            feature_source="cached V-JEPA conditioning",
        ),
        _multicam_row(
            MULTICAM_NO_VJEPA,
            route="multicam unconditioned token baseline",
            feature_source="none",
        ),
        recon_only_row,
        vjepa_loss_row,
    ]
    multires_status = _parse_multires_status(GAUSSIAN_300_MULTIRES_LOG)
    return {
        "gate": "star_uvt_vjepa_vs_gaussian_comparison",
        "report_date": "2026-05-19",
        "rows": rows,
        "gaussian_300_vjepa_loss_log": vjepa_loss_log,
        "gaussian_300_recon_only_log": recon_only_log,
        "gaussian_300_multires_status": multires_status,
        "renderer_matrix": {
            "path": str(RENDERER_MATRIX_MD.relative_to(ROOT)),
            "exists": RENDERER_MATRIX_MD.exists(),
            "note": "Full renderer/kernel matrix is kept separate because many rows are synthetic kernel probes, not trainer steps.",
        },
        "conclusion": {
            "star_vjepa_target_is_real": True,
            "star_feature_fast_helper_is_vjepa_target_route": True,
            "star_feature_rgbfast_helper_preserves_old_rgb_route": True,
            "matched_64f_512px_8192_rows_available": [
                "star_uvt_chunked_vjepa_target",
                "star_uvt_cached_chunks_vjepa_target",
                "star_uvt_target_grid_vjepa_loss",
                "star_uvt_target_grid_vjepa_loss_20step_media",
                "star_uvt_target_grid_vjepa_rgb_aux1_20step_media",
                "star_uvt_target_grid_vjepa_rgb_aux10_20step_media",
                "star_uvt_target_grid_vjepa_rgb_aux10_100step_media",
                "star_uvt_target_grid_vjepa_rgbwarm20_aux10_100step_media",
                "star_uvt_target_grid_vjepa_rgbprobe10_20step_media",
                "star_uvt_target_grid_vjepa_rgbprobe10_100step_media",
                "star_uvt_target_grid_vjepa_rgbprobe10_300step_media",
                "star_uvt_target_grid_vjepa_rgbprobe10_300step_checkpoint",
                "star_uvt_target_grid_vjepa_rgbprobe10_resume300_from300_media",
                "star_uvt_target_grid_vjepa_rgbprobe40_feature025_resume200_from600_media",
                "star_uvt_target_grid_vjepa_rgbprobe_schedule_balance_resume200_from800_media",
                "star_uvt_target_grid_vjepa_rgbprobe40_feature05_resume100_from1000_media",
                "star_uvt_target_grid_vjepa_rgbprobe_recover_schedule_resume100_from1100_media",
                "star_uvt_target_grid_vjepa_rgbprobe40_feature075_resume50_from1200_media",
                "star_uvt_target_grid_vjepa_rgbprobe40_feature1_resume50_from1250_media",
                "star_uvt_target_grid_vjepa_rgbprobe40_feature1_resume100_from1300_media",
                "star_uvt_target_grid_sparseforward_batched_vjp_resume100_from1300_media",
                "star_uvt_target_grid_sparseforward_batched_vjp_lr001_resume100_from1300_media",
                "star_uvt_target_grid_sparseforward_batched_vjp_lr001_resume50_from_lr005sparse_1400",
                "star_uvt_target_grid_sparseforward_batched_vjp_lr001_resume50_from_lr001sparse_1400",
                "star_uvt_target_grid_sparseforward_batched_vjp_lr001_resume50_from_lr005sparse_1450_media",
                "star_uvt_target_grid_autograd_rgbaux1_probeinit_from_sparse1500_media",
                "star_uvt_target_grid_sparsevisual_mix_from_sparse1500_media",
                "star_uvt_target_grid_sparsevisual_patch2x2_from_sparse1500_media",
                "star_uvt_target_grid_sparsevisual_patchmean64_from_sparse1500_media",
                "star_uvt_target_grid_sparsevisual_targetarea64_from_sparse1500_media",
                "star_uvt_target_grid_sparsevisual_targetarea64_phase2x2_from_sparse1500_media",
                "star_uvt_target_grid_sparsevisual_targetarea64_fullcell8_from_sparse1500_media",
                "star_uvt_target_grid_sparsevisual_targetarea64_fullcell8_manualvjp_from_sparse1500_media",
                "star_uvt_target_grid_sparsevisual_targetarea64_fullcell8_staronly_from_sparse1500_media",
                "star_uvt_target_grid_sparsevisual_targetarea64_fullcell8_fastgelu_from_sparse1500_media",
                "star_uvt_target_grid_sparsevisual_targetarea64_fullcell8_linear_from_sparse1500_media",
                "star_uvt_target_grid_sparsevisual_targetarea64_fullcell8_hidden32_from_sparse1500_media",
                "star_uvt_rendered_feature_rgb_probe_from_sparse1500",
                "star_uvt_rendered_feature_rgb_probe_stratified64_from_sparse1500",
                "star_uvt_sparse_visual_vjp_stratified64_frozenprobe_from_sparse1500",
                "star_uvt_sparse_visual_vjp_stratified64_jointprobe_from_sparse1500",
                "star_uvt_target_grid_feature_to_rgb_probe_hidden64",
                "star_uvt_selected_rgb_feature",
                "gaussian_300_recon_only",
                "gaussian_300_prediction_side_vjepa_loss",
            ],
            "not_apples_to_apples_rows": [
                "multicam 16f/128px rows are useful cached-feature timing references but not a 64f/512px match.",
                "renderer matrix rows are kernel/probe rows and should not be promoted as training baselines by themselves.",
            ],
            "bottleneck_read": (
                "STAR V-JEPA target at 64f/512px was not rasterizer-only when "
                "the adapted target was rebuilt every step. cached_chunks removes "
                "most of that repeated target interpolation/loss bucket at the cost "
                "of a 2 GiB resident adapted target cache. target_grid avoids the "
                "resident 2 GiB cache by downsampling rendered features into the "
                "V-JEPA token grid before loss. The 20-step target_grid media run "
                "keeps the feature loss monotonic, but RGB PSNR/media are not a "
                "promotion signal because RGB loss is disabled and the colorizer is "
                "not trained in that route. The RGB-aux1 probe trains the colorizer "
                "and decreases both feature and RGB losses, but its RGB PSNR barely "
                "moves at 20 steps and the step time rises to about 2.0s. RGB-aux10 "
                "is only marginally better on RGB PSNR and slightly worse on feature "
                "loss at 20 steps, so weight alone is not the quality lever. The "
                "100-step aux10 run is meaningfully better in both RGB and feature "
                "space, while the matched rgb-warm20 schedule is faster but worse "
                "on final RGB PSNR and feature loss because it spends 20 steps off "
                "the V-JEPA target. A standalone hidden64 FeatureToColor probe "
                "trained directly on the cached target_grid reaches 23.4 dB on "
                "the token grid and 20.1 dB when upsampled to full video, so the "
                "target-grid features themselves are visually decodable. Wiring "
                "that frozen probe into the STAR target-grid trainer passes and "
                "stays cheap at 1.22s/step, but the 20-step probe PSNR only moves "
                "13.99->14.06. The matched 100-step frozen-probe row moves more "
                "clearly to 14.64 dB at 1.27s/step and keeps feature loss near the "
                "constant aux10 feature curve. The 300-step extension reaches "
                "16.56 dB at 1.36s/step and feature loss 0.812, so the probe "
                "objective keeps working. A checkpointed 300-step rerun matches "
                "that curve at 1.27s/step without media, and a resumed local "
                "300-step continuation reaches 19.88 dB probe PSNR and feature "
                "loss 0.655 at 1.44s/step. That nearly reaches the standalone "
                "full-video upsample PSNR number. A probe-emphasis continuation "
                "from that 600-step state reaches 21.42 dB in 200 more local "
                "steps at 1.51s/step, but it does so by letting feature-grid loss "
                "drift upward. The scheduled 800->1000 balance row recovers "
                "feature loss to 0.644 at 1.31s/step, but gives back a small amount "
                "of probe PSNR to 21.38 and records nonpassing because probe loss "
                "does not decrease end-to-end. A constant feature0.5/probe40 "
                "Pareto continuation from that 1000-step state passes the combined "
                "loss gate and moves probe PSNR to 21.79 at 1.46s/step, but "
                "feature loss drifts back up to 0.657. A 1100->1200 recover "
                "schedule then drives feature loss down to 0.635 at 1.52s/step, "
                "but gives back a small amount of probe PSNR to 21.74 and is "
                "nonpassing on the probe-loss-decrease gate. A short "
                "feature0.75/probe40 1200->1250 continuation passes and restores "
                "probe PSNR to 21.93 at 1.52s/step, but pushes feature loss back "
                "up to 0.639. A feature1/probe40 1250->1300 continuation is the "
                "first current both-improving balance row: feature loss falls to "
                "0.632 and probe PSNR nudges to 21.96 at 1.28s/step. A "
                "1300->1400 extension keeps both improving to feature loss 0.627 "
                "and probe PSNR 21.98, but slows to 1.69s/step on the older dense "
                "target-grid path. The sparse-forward batched target/probe VJP route "
                "preserves the same lr005 objective movement and cuts the 100-step mean "
                "to 0.400s/step, with a 0.263s/step last-20 window and valid media. "
                "The lr001 sparse-forward rerun preserves the dense lr001 quality endpoint "
                "(22.03 dB probe PSNR, 0.631 feature loss) at 0.372s/step mean, but its "
                "last-20 timing is noisy at 0.539s/step because the final step spikes. "
                "The 1400-checkpoint selector then shows the lr005-sparse 1400 state is "
                "the safer continuation point under effective lr001: it passes 50 more "
                "steps to feature loss 0.626 and probe PSNR 22.01 at 0.263s/step mean, "
                "while the lr001-sparse 1400 state fails the same gate after a 1444->1445 "
                "objective jump and ends worse at feature loss 0.632 and probe PSNR 21.84. "
                "A media-bearing 1450->1500 continuation from the selected lr005-sparse "
                "lineage passes again and keeps improving to feature loss 0.625 and probe "
                "PSNR 22.03 at 0.316s/step mean, with a 0.254s/step last-20 window. "
                "The probe contact sheet is valid but still blurry. A full-resolution "
                "autograd RGB-aux diagnostic from that 1500 checkpoint, with the trainable "
                "hidden64 colorizer initialized from the target-grid RGB probe, is a negative "
                "bridge: RGB loss falls, but feature loss worsens to 0.627, probe PSNR drops "
                "to 21.88, the trainable-colorizer media becomes high-frequency artifacts, "
                "and the step is 16.5x slower than the sparse 1500 row. The mixed "
                "target-grid/probe plus sparse visual VJP row from sparse 1500 is also "
                "only a mechanics pass: feature/probe and sparse visual losses improve, "
                "but full RGB PSNR stays at 6.02 while step time rises to 0.964s. "
                "The same-pixel-count patch2x2 support gate is faster at 0.620s/step "
                "and improves sparse visual sample PSNR to 6.18, but feature-target "
                "loss worsens and full RGB PSNR drops to 6.00. The denser "
                "patch-mean64 visual-basis gate samples 1.05M sparse visual pixels "
                "per step and pools them into 262k local-mean cells; it restores "
                "feature/probe movement and full RGB PSNR to 6.02, but costs "
                "1.12s/step and the media remains sparse/high-frequency. "
                "The target-area64 variant swaps selected-patch targets for true "
                "area-downsampled RGB cells; it is slightly faster at 1.10s/step "
                "and improves sparse visual PSNR to 6.06, but dense RGB/media are unchanged. "
                "The phased 2x2 target-area variant cycles the compact support across "
                "a 4x4 subcell schedule; sparse visual PSNR rises to 6.08, but dense "
                "RGB slips to 6.02 at 1.17s/step, so phase coverage alone is not a promotion. "
                "The full-cell8 feasibility row sends gradients through every pixel in each "
                "8x8 area cell; it is nonpassing, drops dense RGB to 5.72 in 5 steps, and "
                "costs 7.53s/step with 5.70s in sparse visual loss construction. "
                "The manual hidden64 VJP version proves the loss-side math can be made "
                "cheaper without changing gradients, cutting the row to 6.41s/step and "
                "3.80s sparse visual loss construction, but the endpoint is still the same "
                "nonpassing 5.72 dense RGB result. The star-only manual variant skips "
                "colorizer parameter gradients and reaches 5.80s/step, but dense RGB falls "
                "again to 5.65; split profiling says exact GELU backward and the first "
                "hidden-layer matmul dominate more than the target-area reduction or "
                "colorizer parameter accumulation. Fast-GELU derivative substitution is "
                "also rejected: it is only a tiny end-to-end timing change at 6.25s/step, "
                "has a worse profiled loss-side total, and keeps the same nonpassing "
                "5.72 dense RGB endpoint. The manual-linear variant makes full-cell8 "
                "mechanically affordable at 2.06s/step and 0.38s sparse visual loss "
                "construction, but the weak linear decoder reaches only 16.98 dB full "
                "oracle PSNR and leaves dense RGB at 5.67, so it is diagnostic only. "
                "Hidden32 keeps most hidden64 oracle capacity at 19.70 dB full PSNR, "
                "but the trainer still costs 4.30s/step and leaves dense RGB at 5.68, "
                "so hidden-size trimming in Python is not the route either. "
                "The rendered-feature "
                "sparse-pixel probe then trains on the actual sparse 1500 rendered feature "
                "distribution at 0.241s/step and improves sampled RGB PSNR 7.74->10.04, "
                "but dense full-video PSNR is only 6.10 and media remains sparse-streaked. "
                "A 4x denser full-resolution stratified64 rendered-pixel probe samples "
                "262k pixels/step at 0.332s/step and still only reaches 6.13 full-video "
                "PSNR, so the sparse-pixel failure is not just target-grid sampling bias. "
                "The first native sparse visual VJP gate then proves sparse RGB loss can "
                "update STAR parameters (`model_grad_seen=true`) at 0.337s/step, but "
                "with a frozen target-grid colorizer it worsens full-video PSNR to 5.74. "
                "Joint STAR+colorizer sparse visual VJP proves both gradients and recovers "
                "full-video PSNR to 6.02, but it is slower at 0.729s/step and still trails "
                "the 0.332s/step colorizer-only stratified rendered-pixel diagnostic at 6.13. "
                "Adding target-grid feature/probe supervision to that sparse visual path "
                "preserves the token-grid objective but remains full-RGB tied at 6.02 and "
                "costs 0.964s/step; switching to patch2x2 contiguous visual support at "
                "the same sampled-pixel count is faster but still quality-negative; "
                "switching to patch-mean64 restores the token/probe movement and full "
                "RGB PSNR but is slower and still not visually promoted; switching "
                "to target-area64 shows selected-patch target bias was not the cause; "
                "cycling target-area support across phases also fails to lift dense RGB; "
                "full-cell dense support through the current Python/torch RGB loss path is both "
                "too slow and quality-negative even after manual hidden64 VJP removes part of "
                "the autograd overhead, and freezing the colorizer gradients only makes the "
                "row faster while worsening dense RGB. Fast scalar GELU-gradient "
                "substitution is also rejected; manual-linear makes the row much faster but "
                "the weak decoder is still quality-negative, and manual-hidden32 keeps more "
                "decoder capacity but still costs 4.30s/step with no dense-RGB promotion. "
                "The remaining issue is "
                "objective balance/visual quality against the same-grid 23.4 dB oracle, "
                "not basic target-route speed. "
                "Gaussian 300-clip V-JEPA loss remains backward-dominated by the "
                "prediction-side frozen V-JEPA path, not cached feature loading."
            ),
            "next_actions": [
                "Treat sparse-forward batched target/probe VJP as the current V-JEPA target speed path; target_grid remains the memory representation because it is faster than cached_chunks and keeps only the 1 MiB token-grid target.",
                "Do not treat the 20-step target_grid media row as RGB quality promotion; it has rgb_loss_weight=0 and only proves monotonic target-feature overfit plus media plumbing.",
                "RGB-aux10 only marginally improves RGB PSNR over aux1 while slightly hurting feature loss; after the rgb-warm20 negative gate, the next visual probe should use a trained/frozen feature-to-RGB probe rather than simply delaying feature loss.",
                "Use the passing target-grid FeatureToColor probe as the next STAR objective bridge: freeze/load the decoder and train STAR features against its RGB projection or log it as the canonical visual oracle.",
                "The first frozen-probe STAR trainer row passes at 20 steps but only nudges probe PSNR; treat it as plumbing and speed proof, not final visual quality.",
                "The 100-step frozen-probe row is the better next visual diagnostic than RGB-aux10: it is cheaper per step and improves probe PSNR more, but still needs a longer/scheduled run before quality promotion.",
                "The 300+300 checkpoint/resume frozen-probe row reaches 19.88 dB probe PSNR; the probe-emphasis 600->800 continuation reaches 21.42 dB while drifting feature loss upward; the scheduled 800->1000 row recovers feature loss but gives back a little probe quality; the feature0.5/probe40 1000->1100 Pareto row passes while moving probe PSNR to 21.79 and feature loss back to 0.657; the 1100->1200 recover schedule pulls feature loss down to 0.635 but gives back probe PSNR to 21.74; the 1200->1250 feature0.75/probe40 continuation restores probe PSNR to 21.93 while pushing feature loss back to 0.639; the 1250->1300 feature1/probe40 row finally improves both to feature loss 0.632 and probe PSNR 21.96; the old dense 1300->1400 extension keeps both improving to 0.627/21.98 while slowing to 1.69s/step; sparse-forward batched VJP preserves that lr005 movement at 0.400s/step mean; lr001 sparse-forward reaches the better probe PSNR endpoint at 0.372s/step mean but with worse feature loss and noisy late timing; the 1400 checkpoint selector rejects the lr001-sparse 1400 state while keeping the lr005-sparse 1400 state alive; the selected 1450->1500 media gate passes but remains visually blurry; naive probe-init full-RGB autograd is a nonpromotion because it shows trainable-colorizer artifacts, worsens feature/probe losses, and costs 5.21s/step; the rendered-feature sparse-pixel probe is only a diagnostic because it reaches 10.04 sparse-sample PSNR / 6.10 full-video PSNR at 0.241s/step while dense media remains streaked; the 4x denser stratified64 rendered-pixel probe still reaches only 6.13 full-video PSNR at 0.332s/step, so target-grid sparse sampling bias is not the explanation; native sparse visual VJP updates STAR features at 0.337s/step but is quality-negative with a frozen target-grid colorizer (`5.739` full-video PSNR); joint STAR+colorizer sparse visual VJP proves both gradient paths and improves full-video PSNR to `6.025`, but it still trails colorizer-only stratified at `6.132` and slows to `0.729s/step`; the mixed target-grid/probe+sparse visual VJP gate preserves feature/probe movement and improves sparse visual sample PSNR to `6.036`, but full RGB is still `6.024` at `0.964s/step`; the patch2x2 same-pixel support gate improves sparse visual sample PSNR to `6.179` and runs faster at `0.620s/step`, but feature loss worsens and full RGB falls to `6.000`; the patch-mean64 gate samples 1.05M pixels/step, restores feature/probe movement and `6.023` full RGB PSNR, but costs `1.125s/step` and still shows sparse high-frequency media; target-area64 is slightly faster at `1.103s/step` with sparse visual PSNR `6.064` but unchanged dense RGB/media; phased target-area64 cycles compact support across a 4x4 schedule, raising sparse visual PSNR to `6.077` but dropping dense RGB to `6.019` at `1.169s/step`; full-cell8 dense support is nonpassing, quality-negative (`5.722` dense RGB), and slow (`7.527s/step`, `5.703s` sparse visual loss); manual hidden64 VJP cuts that to `6.414s/step` and `3.804s` sparse visual loss with matching endpoint; star-only manual hidden64 cuts further to `5.802s/step` but dense RGB drops to `5.648`; fast-GELU derivative substitution is rejected at `6.252s/step` with the same nonpassing `5.722` dense RGB and a worse profiled loss-side total; manual-linear cuts full-cell8 to `2.064s/step` and `0.383s` sparse visual loss construction but leaves dense RGB at only `5.668`, so the next test should be a real fused/simplified visibility or loss VJP path, not Python-side dense support, weak linear decoders, scalar derivative swaps, or another sparse RGB support shuffle.",
                "The 100-step RGB-aux10 run improves more clearly, but it is still far below RGB STAR quality; treat it as evidence for schedule length, not final promotion.",
                "The matched rgb-warm20 schedule is a negative visual-control gate: cheaper per step, but worse final RGB PSNR and feature loss than constant aux10 at the same 100 steps.",
                "Keep cached_chunks as the render-grid reference when exact dense-target loss is needed, but use the cache-budget report before larger runs.",
                "Run a true matched Gaussian/token 64f/512px/8192 5-step profile after 512px promotion guardrails are in place.",
                "Keep V-JEPA feature loss off in the 300-clip Gaussian trainer unless explicitly measuring that backward path.",
                "Use star-feature-512-fast for the selected cached V-JEPA target route; use star-feature-512-rgbfast for the older RGB-target diagnostic.",
            ],
        },
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    columns = [
        ("family", "family"),
        ("route", "route"),
        ("status", "status"),
        ("frames", "frames"),
        ("res", "resolution"),
        ("G/tubes", "primitives"),
        ("step s", "step_s"),
        ("bwd s", "backward_s"),
        ("render s", "render_s"),
        ("target/loss s", "target_eval_s"),
        ("model s", "model_forward_s"),
        ("feature load s", "feature_load_s"),
    ]
    lines = [
        "| " + " | ".join(label for label, _key in columns) + " |",
        "| " + " | ".join("---" for _label, _key in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(key)) for _label, key in columns) + " |")
    return lines


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    conclusion = report["conclusion"]
    multires = report["gaussian_300_multires_status"]
    lines = [
        "# STAR UVT V-JEPA vs Gaussian/Token Comparison",
        "",
        f"Date: {report['report_date']}",
        "",
        "## Answer",
        "",
        "The STAR UVT V-JEPA target route exists and is now the selected `star-feature-512-fast` "
        "helper path. The older RGB reconstruction speed diagnostic is preserved separately as "
        "`star-feature-512-rgbfast`. The original chunked row proved the target route and exposed repeated target interpolation/loss "
        "as the largest bucket; cached chunks removed most of that bucket but cost a 2 GiB "
        "resident target cache; target-grid loss is now the fastest V-JEPA target diagnostic "
        "and keeps only a 1 MiB token-grid target. The 20-step target-grid media run confirms "
        "monotonic feature-target overfit, but it is not RGB quality evidence because the route "
        "sets `rgb_loss_weight=0`. The RGB-aux1 probe trains the colorizer and decreases RGB loss, "
        "but the 20-step PSNR gain is tiny; RGB-aux10 is only marginally better at 20 steps. "
        "The 100-step aux10 run moves more clearly, while the matched RGB-warm20 schedule is faster "
        "but worse on final RGB and feature loss. Neither is RGB STAR quality. A standalone "
        "hidden64 feature-to-RGB probe trained directly on the cached target grid reaches "
        "23.4 dB on the grid and 20.1 dB when upsampled to full video, which says the "
        "target-grid features are decodable. The first STAR integration of that frozen "
        "probe passes at 1.22s/step, but only nudges probe PSNR 13.99->14.06 in 20 steps. "
        "The 100-step frozen-probe row moves more clearly to 14.64 dB at 1.27s/step, "
        "and the 300-step extension reaches 16.56 dB at 1.36s/step. The checkpointed "
        "resume continuation reaches 19.88 dB after another 300 local steps. A "
        "probe-emphasis 600->800 continuation reaches 21.42 dB at 1.51s/step, "
        "but feature loss drifts upward under that objective. The probe objective "
        "has now passed the standalone full-video upsample number. The scheduled "
        "800->1000 balance row recovers feature alignment but gives back a little "
        "probe quality. A feature0.5/probe40 1000->1100 continuation passes the "
        "combined gate and moves probe PSNR to 21.79 dB at 1.46s/step, but feature "
        "loss drifts back to 0.657. A 1100->1200 recover schedule lowers feature "
        "loss to 0.635 at 1.52s/step, but gives back a little probe quality to "
        "21.74 dB. A short feature0.75/probe40 1200->1250 continuation restores "
        "probe PSNR to 21.93 dB at 1.52s/step, but feature loss rises to 0.639, "
        "and a feature1/probe40 1250->1300 row improves both to 21.96 dB and "
        "0.632 feature loss at 1.28s/step. The old dense 1300->1400 extension keeps both "
        "moving to 21.98 dB and 0.627 feature loss, but slows to 1.69s/step. "
        "The sparse-forward batched VJP version preserves that lr005 objective movement while cutting "
        "the 100-step helper/media run to 0.400s/step mean and 0.263s/step over the last 20 "
        "steps. The lr001 sparse-forward rerun preserves the dense lr001 quality endpoint at "
        "0.372s/step mean and 22.03 dB probe PSNR, but feature loss is worse than lr005 and "
        "late timing is noisy. The 1400-checkpoint selector then rejects that lr001-sparse "
        "state for continuation: it improves until global step 1444, jumps at 1444->1445, "
        "and fails the 50-step loss-decrease gate, while the lr005-sparse 1400 checkpoint "
        "keeps improving both feature and probe losses under effective lr001. The selected "
        "1450->1500 media continuation passes again and nudges probe PSNR to 22.03 dB, "
        "but the contact sheet remains blurry. A full-resolution autograd RGB-aux bridge "
        "from that sparse 1500 checkpoint is negative even with the trainable colorizer "
        "initialized from the target-grid probe: RGB loss falls, but feature loss worsens, "
        "probe PSNR drops to 21.88 dB, trainable-colorizer media artifacts appear, and "
        "step time rises to 5.21s. The mixed target-grid/probe plus sparse visual "
        "VJP row preserves feature/probe movement and sparse visual sample improvement, "
        "but full RGB PSNR is still 6.02 at 0.964s/step. The patch2x2 same-pixel "
        "support variant is faster at 0.620s/step and improves sparse sample PSNR to "
        "6.18, but feature-target loss worsens and dense RGB drops to 6.00. "
        "The denser patch-mean64 visual-basis gate samples 1.05M sparse visual pixels "
        "per step and pools them into 262k local-mean cells; it restores feature loss "
        "movement and dense RGB PSNR to 6.02, but costs 1.12s/step and media still "
        "looks sparse/high-frequency, so the gate is informative but not a quality promotion. "
        "The target-area64 variant uses true area-downsampled RGB cells instead of selected-patch "
        "target means; it is slightly faster at 1.10s/step and raises sparse visual PSNR to 6.06, "
        "but dense RGB/media are unchanged. The phased 2x2 target-area variant cycles support across "
        "a 4x4 subcell schedule; sparse visual PSNR rises to 6.08, but dense RGB slips to 6.02 at 1.17s/step. "
        "Full-cell8 dense support is nonpassing and slow at 7.53s/step; manual hidden64 VJP cuts it "
        "to 6.41s without changing the bad 5.72 dense RGB endpoint, star-only cuts to 5.80s but "
        "drops dense RGB to 5.65, and fast-GELU is rejected. Manual-linear is the first affordable "
        "full-cell8 mechanics gate at 2.06s/step, but the weak linear decoder still leaves dense RGB "
        "at 5.67, so it is not a quality route. Manual-hidden32 keeps most hidden64 probe capacity "
        "but still costs 4.30s/step and leaves dense RGB at 5.68, so shrinking the Python hidden "
        "decoder is also not a quality route. "
        "A rendered-feature "
        "sparse-pixel probe trained directly "
        "on that sparse 1500 feature distribution is fast at 0.241s/step and improves "
        "sampled RGB PSNR 7.74->10.04, but dense full-video PSNR is still only 6.10 "
        "and media remains sparse-streaked. A 4x denser stratified64 rendered-pixel "
        "probe samples 262k pixels/step at 0.332s/step and still reaches only 6.13 "
        "full-video PSNR, so target-grid sampling bias is not the explanation. The native sparse "
        "visual VJP gate updates STAR parameters at 0.337s/step, but with a frozen target-grid "
        "probe colorizer it reaches only 5.74 full-video PSNR. Joint STAR+colorizer sparse "
        "visual VJP proves both gradients and recovers to 6.02 full-video PSNR, but costs "
        "0.729s/step and still trails the colorizer-only stratified diagnostic at 6.13. The route still "
        "trails the same-grid 23.4 dB oracle, so it needs better objective balance "
        "for quality rather than another dense-VJP speed pass. "
        "The older Gaussian/token V-JEPA-feature-loss run is still much "
        "slower because the frozen V-JEPA model is in the prediction backward path.",
        "",
        "## Timing Table",
        "",
    ]
    lines.extend(_markdown_table(report["rows"]))
    lines.extend(
        [
            "",
            "## Matched 64f/512px Read",
            "",
            "- STAR V-JEPA target: streaming chunks proved the cached-token route but repeatedly rebuilt the `[64, 32, 512, 512]` target; cached chunks remove most of that target/loss cost at the price of a resident 2 GiB adapted target; target-grid loss avoids that cache by downsampling rendered features to `[1,32,16,16]` per chunk before loss. The 20-step target-grid media row decreases feature loss, but RGB PSNR/media are not quality signals because the colorizer is not trained. RGB-aux1 and RGB-aux10 both train the colorizer and decrease RGB loss; 20-step RGB PSNR barely moves, the 100-step aux10 row shows schedule length starts to matter, and the matched rgb-warm20 schedule is a negative control rather than a promotion. The standalone target-grid feature-to-RGB probe is the visual oracle, the 20-step frozen-probe STAR row proves it can be put in the loop cheaply, the 100/300-step rows keep improving, the checkpointed 300+300 row nearly reaches the standalone full-video number, the probe-emphasis 600->800 row passes it while exposing feature-loss drift, the scheduled 800->1000 row recovers feature loss while giving back a little probe quality, the feature0.5/probe40 1000->1100 row passes while raising probe PSNR to 21.79 but drifting feature loss back to 0.657, the 1100->1200 recover row pulls feature loss to 0.635 while giving back probe PSNR to 21.74, the 1200->1250 feature0.75/probe40 row restores probe PSNR to 21.93 while pushing feature loss to 0.639, the 1250->1300 feature1/probe40 row improves both to 21.96 probe PSNR and 0.632 feature loss, the old dense 1300->1400 feature1/probe40 extension keeps both improving to 21.98 probe PSNR and 0.627 feature loss but slows to 1.69s/step, the lr005 sparse-forward batched VJP helper row preserves that movement at 0.400s/step mean with valid but blurry media, the lr001 sparse-forward row reaches the better 22.03 dB probe PSNR endpoint at 0.372s/step mean while accepting worse feature loss and noisy late timing, the 1400 checkpoint-selection gate says to continue from the lr005-sparse state rather than the lr001-sparse state, the selected 1450->1500 media gate passes with 22.03 dB probe PSNR while staying visually blurry, the full-res autograd RGB-aux probe-init bridge from sparse 1500 is a nonpromotion because it worsens feature/probe losses, shows trainable-colorizer media artifacts, and costs 5.21s/step, the rendered-feature sparse-pixel probe trains on the right distribution at 0.241s/step but only reaches 6.10 full-video PSNR with streaked media, the 4x denser stratified64 rendered-pixel probe still only reaches 6.13 full-video PSNR at 0.332s/step, native sparse visual VJP updates STAR features at 0.337s/step but is quality-negative with the frozen target-grid colorizer at 5.74 full-video PSNR, and joint STAR+colorizer sparse visual VJP improves that to 6.02 full-video PSNR but still trails the colorizer-only stratified diagnostic at 6.13 while slowing to 0.729s/step.",
            "- Mixed target-grid/probe plus sparse visual VJP from sparse 1500 preserves feature/probe movement and sparse visual sample improvement, but full RGB PSNR is still 6.02 at 0.964s/step; the next quality gate should change visual support/basis, not remix sparse RGB with the same target-grid objective.",
            "- Patch2x2 same-pixel-count sparse visual support from sparse 1500 is a negative support-basis gate: sparse sample PSNR rises to 6.18 and mean step is 0.620s, but feature-target loss worsens and dense RGB PSNR is 6.00.",
            "- Patch-mean64 sparse visual basis from sparse 1500 samples 1.05M visual pixels/step and pools them into 262k local-mean cells: it passes, restores feature/probe movement, and lands at 6.02 dense RGB PSNR, but it costs 1.12s/step and still has sparse high-frequency media.",
            "- Target-area64 sparse visual basis from sparse 1500 keeps the same 1.05M visual pixels/step and 262k-cell loss shape but compares against true area-downsampled RGB targets: it is slightly faster at 1.10s/step and reaches 6.06 sparse visual PSNR, but dense RGB/media stay unchanged.",
            "- Phased target-area64 sparse visual basis from sparse 1500 cycles the same 2x2 support through a 4x4 subcell schedule: it passes and reaches 6.08 sparse visual PSNR, but dense RGB slips to 6.02 at 1.17s/step, so temporal support coverage alone is not the visual-quality lever.",
            "- Full-cell8 target-area support from sparse 1500 is nonpassing and quality-negative at 5.72 dense RGB PSNR; manual hidden64 VJP reduces step time from 7.53s to 6.41s and sparse visual loss construction from 5.70s to 3.80s without changing the endpoint, star-only is faster but drops dense RGB, fast-GELU is rejected, manual-linear cuts the row to 2.06s/step while leaving dense RGB at 5.67 because the linear decoder is too weak, and manual-hidden32 keeps most hidden64 probe capacity but still costs 4.30s/step with dense RGB at 5.68. Keep these as parity/diagnostic scaffolds for native fused loss/visibility work, not promoted routes.",
            "- STAR RGB feature diagnostic: preserved as `star-feature-512-rgbfast`; it is RGB-target only and should not answer cached-feature training questions.",
            "- Gaussian/token recon-only: uses cached V-JEPA conditioning and avoids prediction-side V-JEPA loss; this is the closest 300-set trainer timing reference.",
            "- Gaussian/token V-JEPA feature loss: keeps frozen V-JEPA in the prediction path and spends almost the whole step in backward.",
            "",
            "## Multires Gaussian State",
            "",
            f"- log: `{multires['path']}`",
            f"- status: `{multires['status']}`",
            f"- 256->512 switch seen: `{multires['render_size_switch_seen']}`",
            f"- first NaN step: `{multires['first_nan_step']}`",
            "",
            "## Bottleneck Read",
            "",
            conclusion["bottleneck_read"],
            "",
            "## Next Actions",
            "",
        ]
    )
    for action in conclusion["next_actions"]:
        lines.append(f"- {action}")
    lines.extend(
        [
            "",
            "## Artifact Notes",
            "",
            f"- full renderer matrix: `{report['renderer_matrix']['path']}` (exists: `{report['renderer_matrix']['exists']}`)",
        ]
    )
    for row in report["rows"]:
        lines.append(f"- `{row['artifact']}` -> {row['route']}")
    lines.append("")
    write_report_text(path, "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=ROOT / "outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.md",
    )
    args = parser.parse_args()
    report = _build_report()
    write_report_json(args.out_json, report)
    _write_markdown(report, args.out_md)
    print(json.dumps({"rows": len(report["rows"]), "out_md": str(args.out_md)}, sort_keys=True))


if __name__ == "__main__":
    main()
