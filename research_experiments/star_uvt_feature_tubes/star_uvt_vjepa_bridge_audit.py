from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


try:
    from .report_artifacts import (
        ROOT as DYNAWORLD_ROOT,
        load_optional_report_json_or_error,
        write_report_json,
        write_report_text,
    )
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import (
        ROOT as DYNAWORLD_ROOT,
        load_optional_report_json_or_error,
        write_report_json,
        write_report_text,
    )
from config_utils import load_config_file


SELECTED_FAST_CONFIG = (
    "src/train_configs/"
    "star_uvt_feature_testvideo_64f_512_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_20step_media.jsonc"
)
SCALED_VJEPA_TARGET_CONFIG = (
    "src/train_configs/"
    "star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_chunkedtarget_lr005_5step.jsonc"
)
SCALED_VJEPA_TARGET_RESULT = (
    "outputs/benchmarks/"
    "2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_chunkedtarget_lr005_5step.json"
)
CACHED_CHUNKS_VJEPA_TARGET_CONFIG = (
    "src/train_configs/"
    "star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_cachedchunks_lr005_5step.jsonc"
)
CACHED_CHUNKS_VJEPA_TARGET_RESULT = (
    "outputs/benchmarks/"
    "2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_cachedchunks_lr005_5step.json"
)
TARGET_GRID_VJEPA_TARGET_CONFIG = (
    "src/train_configs/"
    "star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_lr005_5step.jsonc"
)
TARGET_GRID_VJEPA_TARGET_RESULT = (
    "outputs/benchmarks/"
    "2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_lr005_5step.json"
)
TARGET_GRID_MEDIA_VJEPA_TARGET_CONFIG = (
    "src/train_configs/"
    "star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_lr005_20step_media.jsonc"
)
TARGET_GRID_MEDIA_VJEPA_TARGET_RESULT = (
    "outputs/benchmarks/"
    "2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_lr005_20step_media.json"
)
TARGET_GRID_RGB_AUX_VJEPA_TARGET_CONFIG = (
    "src/train_configs/"
    "star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbaux1_lr005_20step_media.jsonc"
)
TARGET_GRID_RGB_AUX_VJEPA_TARGET_RESULT = (
    "outputs/benchmarks/"
    "2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbaux1_lr005_20step_media.json"
)
TARGET_GRID_RGB_AUX10_VJEPA_TARGET_CONFIG = (
    "src/train_configs/"
    "star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbaux10_lr005_20step_media.jsonc"
)
TARGET_GRID_RGB_AUX10_VJEPA_TARGET_RESULT = (
    "outputs/benchmarks/"
    "2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbaux10_lr005_20step_media.json"
)
TARGET_GRID_RGB_AUX10_100STEP_VJEPA_TARGET_CONFIG = (
    "src/train_configs/"
    "star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbaux10_lr005_100step_media.jsonc"
)
TARGET_GRID_RGB_AUX10_100STEP_VJEPA_TARGET_RESULT = (
    "outputs/benchmarks/"
    "2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbaux10_lr005_100step_media.json"
)
TARGET_GRID_RGBWARM20_AUX10_100STEP_VJEPA_TARGET_CONFIG = (
    "src/train_configs/"
    "star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbwarm20_aux10_lr005_100step_media.jsonc"
)
TARGET_GRID_RGBWARM20_AUX10_100STEP_VJEPA_TARGET_RESULT = (
    "outputs/benchmarks/"
    "2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbwarm20_aux10_lr005_100step_media.json"
)
TARGET_GRID_FEATURE_RGB_PROBE_CONFIG = (
    "src/train_configs/"
    "star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden64_lr01_1000step.jsonc"
)
TARGET_GRID_FEATURE_RGB_PROBE_RESULT = (
    "outputs/benchmarks/"
    "2026-05-19_star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden64_lr01_1000step.json"
)
TARGET_GRID_RGBPROBE10_VJEPA_TARGET_CONFIG = (
    "src/train_configs/"
    "star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe10_lr005_20step_media.jsonc"
)
TARGET_GRID_RGBPROBE10_VJEPA_TARGET_RESULT = (
    "outputs/benchmarks/"
    "2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_20step_media.json"
)
TARGET_GRID_RGBPROBE10_100STEP_VJEPA_TARGET_CONFIG = (
    "src/train_configs/"
    "star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe10_lr005_100step_media.jsonc"
)
TARGET_GRID_RGBPROBE10_100STEP_VJEPA_TARGET_RESULT = (
    "outputs/benchmarks/"
    "2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_100step_media.json"
)
TARGET_GRID_RGBPROBE10_300STEP_VJEPA_TARGET_CONFIG = (
    "src/train_configs/"
    "star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe10_lr005_300step_media.jsonc"
)
TARGET_GRID_RGBPROBE10_300STEP_VJEPA_TARGET_RESULT = (
    "outputs/benchmarks/"
    "2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_300step_media.json"
)
TARGET_GRID_RGBPROBE10_300STEP_CHECKPOINT_VJEPA_TARGET_CONFIG = (
    "src/train_configs/"
    "star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe10_lr005_300step_checkpoint.jsonc"
)
TARGET_GRID_RGBPROBE10_300STEP_CHECKPOINT_VJEPA_TARGET_RESULT = (
    "outputs/benchmarks/"
    "2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_300step_checkpoint.json"
)
TARGET_GRID_RGBPROBE10_RESUME300_FROM300_VJEPA_TARGET_CONFIG = (
    "src/train_configs/"
    "star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe10_lr005_resume300_from300_checkpoint_media.jsonc"
)
TARGET_GRID_RGBPROBE10_RESUME300_FROM300_VJEPA_TARGET_RESULT = (
    "outputs/benchmarks/"
    "2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe10_lr005_resume300_from300_checkpoint_media.json"
)
TARGET_GRID_RGBPROBE40_FEATURE025_RESUME200_FROM600_VJEPA_TARGET_CONFIG = (
    "src/train_configs/"
    "star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature025_lr005_resume200_from600_checkpoint_media.jsonc"
)
TARGET_GRID_RGBPROBE40_FEATURE025_RESUME200_FROM600_VJEPA_TARGET_RESULT = (
    "outputs/benchmarks/"
    "2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature025_lr005_resume200_from600_checkpoint_media.json"
)
TARGET_GRID_RGBPROBE_BALANCE_RESUME200_FROM800_VJEPA_TARGET_CONFIG = (
    "src/train_configs/"
    "star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe_schedule_balance_resume200_from800_checkpoint_media.jsonc"
)
TARGET_GRID_RGBPROBE_BALANCE_RESUME200_FROM800_VJEPA_TARGET_RESULT = (
    "outputs/benchmarks/"
    "2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_balance_resume200_from800_checkpoint_media.json"
)
TARGET_GRID_RGBPROBE40_FEATURE05_RESUME100_FROM1000_VJEPA_TARGET_CONFIG = (
    "src/train_configs/"
    "star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature05_lr005_resume100_from1000_checkpoint_media.jsonc"
)
TARGET_GRID_RGBPROBE40_FEATURE05_RESUME100_FROM1000_VJEPA_TARGET_RESULT = (
    "outputs/benchmarks/"
    "2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature05_lr005_resume100_from1000_checkpoint_media.json"
)
TARGET_GRID_RGBPROBE_RECOVER_RESUME100_FROM1100_VJEPA_TARGET_CONFIG = (
    "src/train_configs/"
    "star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe_schedule_recover_resume100_from1100_checkpoint_media.jsonc"
)
TARGET_GRID_RGBPROBE_RECOVER_RESUME100_FROM1100_VJEPA_TARGET_RESULT = (
    "outputs/benchmarks/"
    "2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe_schedule_recover_resume100_from1100_checkpoint_media.json"
)
TARGET_GRID_RGBPROBE40_FEATURE075_RESUME50_FROM1200_VJEPA_TARGET_CONFIG = (
    "src/train_configs/"
    "star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature075_lr005_resume50_from1200_checkpoint_media.jsonc"
)
TARGET_GRID_RGBPROBE40_FEATURE075_RESUME50_FROM1200_VJEPA_TARGET_RESULT = (
    "outputs/benchmarks/"
    "2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature075_lr005_resume50_from1200_checkpoint_media.json"
)
TARGET_GRID_RGBPROBE40_FEATURE1_RESUME50_FROM1250_VJEPA_TARGET_CONFIG = (
    "src/train_configs/"
    "star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume50_from1250_checkpoint_media.jsonc"
)
TARGET_GRID_RGBPROBE40_FEATURE1_RESUME50_FROM1250_VJEPA_TARGET_RESULT = (
    "outputs/benchmarks/"
    "2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume50_from1250_checkpoint_media.json"
)
TARGET_GRID_RGBPROBE40_FEATURE1_RESUME100_FROM1300_VJEPA_TARGET_CONFIG = (
    "src/train_configs/"
    "star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_checkpoint_media.jsonc"
)
TARGET_GRID_RGBPROBE40_FEATURE1_RESUME100_FROM1300_VJEPA_TARGET_RESULT = (
    "outputs/benchmarks/"
    "2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_checkpoint_media.json"
)
STAR_FEATURE_TRAINER = "src/train/star_uvt_feature_overfit_trainer.py"
DISPATCHER = "src/train/train.py"
TRAINER_REGISTRY = "src/train/trainer_registry.py"


def _load_config(path: Path) -> dict[str, Any]:
    try:
        return load_config_file(path)
    except Exception as exc:  # pragma: no cover - report builder, not library API
        return {"_load_error": str(exc)}


def _load_json(path: Path) -> dict[str, Any] | None:
    return load_optional_report_json_or_error(path)


def _has_vjepa_feature_section(config: dict[str, Any]) -> bool:
    features = config.get("features")
    if isinstance(features, dict):
        extractor = str(features.get("extractor", "")).lower()
        if "vjepa" in extractor:
            return True
    model = config.get("model")
    if isinstance(model, dict):
        backend = str(model.get("video_encoder_backend", "")).lower()
        if "vjepa" in backend or backend.startswith("precomputed"):
            return True
    return "vjepa" in json.dumps(config, sort_keys=True).lower()


def _find_configs() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    star_configs: list[dict[str, Any]] = []
    precomputed_vjepa_configs: list[dict[str, Any]] = []
    for path in sorted((DYNAWORLD_ROOT / "src" / "train_configs").glob("*.jsonc")):
        rel = str(path.relative_to(DYNAWORLD_ROOT))
        config = _load_config(path)
        arch = str(config.get("arch", ""))
        features = config.get("features") if isinstance(config.get("features"), dict) else {}
        data = config.get("data") if isinstance(config.get("data"), dict) else {}
        feature_uvt = config.get("feature_uvt") if isinstance(config.get("feature_uvt"), dict) else {}
        feature_target = config.get("feature_target") if isinstance(config.get("feature_target"), dict) else {}
        row = {
            "path": rel,
            "arch": arch,
            "frames": data.get("max_frames"),
            "target_size": data.get("target_size"),
            "tube_count": feature_uvt.get("tube_count"),
            "render_mode": feature_uvt.get("render_mode"),
            "has_features_section": bool(features),
            "feature_target_enabled": bool(feature_target.get("enabled", False)),
            "feature_target_materialization": feature_target.get("materialization"),
            "feature_extractor": features.get("extractor"),
            "feature_cache_dir": features.get("cache_dir"),
            "sample_cache_key": features.get("sample_cache_key"),
        }
        if arch == "star_uvt_feature_overfit":
            row["has_vjepa_or_precomputed_feature_config"] = _has_vjepa_feature_section(config)
            star_configs.append(row)
        if arch in {"precomputed_feature_implicit_camera", "multicam_precomputed_feature_implicit_camera"}:
            extractor = str(features.get("extractor", "")).lower()
            if "vjepa" in extractor:
                precomputed_vjepa_configs.append(row)
    return star_configs, precomputed_vjepa_configs


def _build_report() -> dict[str, Any]:
    selected_path = DYNAWORLD_ROOT / SELECTED_FAST_CONFIG
    scaled_vjepa_path = DYNAWORLD_ROOT / SCALED_VJEPA_TARGET_CONFIG
    cached_chunks_path = DYNAWORLD_ROOT / CACHED_CHUNKS_VJEPA_TARGET_CONFIG
    target_grid_path = DYNAWORLD_ROOT / TARGET_GRID_VJEPA_TARGET_CONFIG
    target_grid_media_path = DYNAWORLD_ROOT / TARGET_GRID_MEDIA_VJEPA_TARGET_CONFIG
    target_grid_rgb_aux_path = DYNAWORLD_ROOT / TARGET_GRID_RGB_AUX_VJEPA_TARGET_CONFIG
    target_grid_rgb_aux10_path = DYNAWORLD_ROOT / TARGET_GRID_RGB_AUX10_VJEPA_TARGET_CONFIG
    target_grid_rgb_aux10_100step_path = DYNAWORLD_ROOT / TARGET_GRID_RGB_AUX10_100STEP_VJEPA_TARGET_CONFIG
    target_grid_rgbwarm20_aux10_100step_path = (
        DYNAWORLD_ROOT / TARGET_GRID_RGBWARM20_AUX10_100STEP_VJEPA_TARGET_CONFIG
    )
    target_grid_feature_rgb_probe_path = DYNAWORLD_ROOT / TARGET_GRID_FEATURE_RGB_PROBE_CONFIG
    target_grid_rgbprobe10_path = DYNAWORLD_ROOT / TARGET_GRID_RGBPROBE10_VJEPA_TARGET_CONFIG
    target_grid_rgbprobe10_100step_path = DYNAWORLD_ROOT / TARGET_GRID_RGBPROBE10_100STEP_VJEPA_TARGET_CONFIG
    target_grid_rgbprobe10_300step_path = DYNAWORLD_ROOT / TARGET_GRID_RGBPROBE10_300STEP_VJEPA_TARGET_CONFIG
    target_grid_rgbprobe10_300step_checkpoint_path = (
        DYNAWORLD_ROOT / TARGET_GRID_RGBPROBE10_300STEP_CHECKPOINT_VJEPA_TARGET_CONFIG
    )
    target_grid_rgbprobe10_resume300_from300_path = (
        DYNAWORLD_ROOT / TARGET_GRID_RGBPROBE10_RESUME300_FROM300_VJEPA_TARGET_CONFIG
    )
    target_grid_rgbprobe40_feature025_resume200_from600_path = (
        DYNAWORLD_ROOT / TARGET_GRID_RGBPROBE40_FEATURE025_RESUME200_FROM600_VJEPA_TARGET_CONFIG
    )
    target_grid_rgbprobe_balance_resume200_from800_path = (
        DYNAWORLD_ROOT / TARGET_GRID_RGBPROBE_BALANCE_RESUME200_FROM800_VJEPA_TARGET_CONFIG
    )
    target_grid_rgbprobe40_feature05_resume100_from1000_path = (
        DYNAWORLD_ROOT / TARGET_GRID_RGBPROBE40_FEATURE05_RESUME100_FROM1000_VJEPA_TARGET_CONFIG
    )
    target_grid_rgbprobe_recover_resume100_from1100_path = (
        DYNAWORLD_ROOT / TARGET_GRID_RGBPROBE_RECOVER_RESUME100_FROM1100_VJEPA_TARGET_CONFIG
    )
    target_grid_rgbprobe40_feature075_resume50_from1200_path = (
        DYNAWORLD_ROOT / TARGET_GRID_RGBPROBE40_FEATURE075_RESUME50_FROM1200_VJEPA_TARGET_CONFIG
    )
    target_grid_rgbprobe40_feature1_resume50_from1250_path = (
        DYNAWORLD_ROOT / TARGET_GRID_RGBPROBE40_FEATURE1_RESUME50_FROM1250_VJEPA_TARGET_CONFIG
    )
    target_grid_rgbprobe40_feature1_resume100_from1300_path = (
        DYNAWORLD_ROOT / TARGET_GRID_RGBPROBE40_FEATURE1_RESUME100_FROM1300_VJEPA_TARGET_CONFIG
    )
    selected_config = _load_config(selected_path)
    scaled_vjepa_config = _load_config(scaled_vjepa_path)
    cached_chunks_config = _load_config(cached_chunks_path)
    target_grid_config = _load_config(target_grid_path)
    target_grid_media_config = _load_config(target_grid_media_path)
    target_grid_rgb_aux_config = _load_config(target_grid_rgb_aux_path)
    target_grid_rgb_aux10_config = _load_config(target_grid_rgb_aux10_path)
    target_grid_rgb_aux10_100step_config = _load_config(target_grid_rgb_aux10_100step_path)
    target_grid_rgbwarm20_aux10_100step_config = _load_config(target_grid_rgbwarm20_aux10_100step_path)
    target_grid_feature_rgb_probe_config = _load_config(target_grid_feature_rgb_probe_path)
    target_grid_rgbprobe10_config = _load_config(target_grid_rgbprobe10_path)
    target_grid_rgbprobe10_100step_config = _load_config(target_grid_rgbprobe10_100step_path)
    target_grid_rgbprobe10_300step_config = _load_config(target_grid_rgbprobe10_300step_path)
    target_grid_rgbprobe10_300step_checkpoint_config = _load_config(target_grid_rgbprobe10_300step_checkpoint_path)
    target_grid_rgbprobe10_resume300_from300_config = _load_config(target_grid_rgbprobe10_resume300_from300_path)
    target_grid_rgbprobe40_feature025_resume200_from600_config = _load_config(
        target_grid_rgbprobe40_feature025_resume200_from600_path
    )
    target_grid_rgbprobe_balance_resume200_from800_config = _load_config(
        target_grid_rgbprobe_balance_resume200_from800_path
    )
    target_grid_rgbprobe40_feature05_resume100_from1000_config = _load_config(
        target_grid_rgbprobe40_feature05_resume100_from1000_path
    )
    target_grid_rgbprobe_recover_resume100_from1100_config = _load_config(
        target_grid_rgbprobe_recover_resume100_from1100_path
    )
    target_grid_rgbprobe40_feature075_resume50_from1200_config = _load_config(
        target_grid_rgbprobe40_feature075_resume50_from1200_path
    )
    target_grid_rgbprobe40_feature1_resume50_from1250_config = _load_config(
        target_grid_rgbprobe40_feature1_resume50_from1250_path
    )
    target_grid_rgbprobe40_feature1_resume100_from1300_config = _load_config(
        target_grid_rgbprobe40_feature1_resume100_from1300_path
    )
    scaled_vjepa_result = _load_json(DYNAWORLD_ROOT / SCALED_VJEPA_TARGET_RESULT)
    cached_chunks_result = _load_json(DYNAWORLD_ROOT / CACHED_CHUNKS_VJEPA_TARGET_RESULT)
    target_grid_result = _load_json(DYNAWORLD_ROOT / TARGET_GRID_VJEPA_TARGET_RESULT)
    target_grid_media_result = _load_json(DYNAWORLD_ROOT / TARGET_GRID_MEDIA_VJEPA_TARGET_RESULT)
    target_grid_rgb_aux_result = _load_json(DYNAWORLD_ROOT / TARGET_GRID_RGB_AUX_VJEPA_TARGET_RESULT)
    target_grid_rgb_aux10_result = _load_json(DYNAWORLD_ROOT / TARGET_GRID_RGB_AUX10_VJEPA_TARGET_RESULT)
    target_grid_rgb_aux10_100step_result = _load_json(DYNAWORLD_ROOT / TARGET_GRID_RGB_AUX10_100STEP_VJEPA_TARGET_RESULT)
    target_grid_rgbwarm20_aux10_100step_result = _load_json(
        DYNAWORLD_ROOT / TARGET_GRID_RGBWARM20_AUX10_100STEP_VJEPA_TARGET_RESULT
    )
    target_grid_feature_rgb_probe_result = _load_json(DYNAWORLD_ROOT / TARGET_GRID_FEATURE_RGB_PROBE_RESULT)
    target_grid_rgbprobe10_result = _load_json(DYNAWORLD_ROOT / TARGET_GRID_RGBPROBE10_VJEPA_TARGET_RESULT)
    target_grid_rgbprobe10_100step_result = _load_json(
        DYNAWORLD_ROOT / TARGET_GRID_RGBPROBE10_100STEP_VJEPA_TARGET_RESULT
    )
    target_grid_rgbprobe10_300step_result = _load_json(
        DYNAWORLD_ROOT / TARGET_GRID_RGBPROBE10_300STEP_VJEPA_TARGET_RESULT
    )
    target_grid_rgbprobe10_300step_checkpoint_result = _load_json(
        DYNAWORLD_ROOT / TARGET_GRID_RGBPROBE10_300STEP_CHECKPOINT_VJEPA_TARGET_RESULT
    )
    target_grid_rgbprobe10_resume300_from300_result = _load_json(
        DYNAWORLD_ROOT / TARGET_GRID_RGBPROBE10_RESUME300_FROM300_VJEPA_TARGET_RESULT
    )
    target_grid_rgbprobe40_feature025_resume200_from600_result = _load_json(
        DYNAWORLD_ROOT / TARGET_GRID_RGBPROBE40_FEATURE025_RESUME200_FROM600_VJEPA_TARGET_RESULT
    )
    target_grid_rgbprobe_balance_resume200_from800_result = _load_json(
        DYNAWORLD_ROOT / TARGET_GRID_RGBPROBE_BALANCE_RESUME200_FROM800_VJEPA_TARGET_RESULT
    )
    target_grid_rgbprobe40_feature05_resume100_from1000_result = _load_json(
        DYNAWORLD_ROOT / TARGET_GRID_RGBPROBE40_FEATURE05_RESUME100_FROM1000_VJEPA_TARGET_RESULT
    )
    target_grid_rgbprobe_recover_resume100_from1100_result = _load_json(
        DYNAWORLD_ROOT / TARGET_GRID_RGBPROBE_RECOVER_RESUME100_FROM1100_VJEPA_TARGET_RESULT
    )
    target_grid_rgbprobe40_feature075_resume50_from1200_result = _load_json(
        DYNAWORLD_ROOT / TARGET_GRID_RGBPROBE40_FEATURE075_RESUME50_FROM1200_VJEPA_TARGET_RESULT
    )
    target_grid_rgbprobe40_feature1_resume50_from1250_result = _load_json(
        DYNAWORLD_ROOT / TARGET_GRID_RGBPROBE40_FEATURE1_RESUME50_FROM1250_VJEPA_TARGET_RESULT
    )
    target_grid_rgbprobe40_feature1_resume100_from1300_result = _load_json(
        DYNAWORLD_ROOT / TARGET_GRID_RGBPROBE40_FEATURE1_RESUME100_FROM1300_VJEPA_TARGET_RESULT
    )
    trainer_text = (DYNAWORLD_ROOT / STAR_FEATURE_TRAINER).read_text(encoding="utf-8")
    dispatcher_text = (DYNAWORLD_ROOT / DISPATCHER).read_text(encoding="utf-8")
    registry_text = (DYNAWORLD_ROOT / TRAINER_REGISTRY).read_text(encoding="utf-8")
    star_configs, precomputed_vjepa_configs = _find_configs()
    star_vjepa_configs = [row for row in star_configs if row["has_vjepa_or_precomputed_feature_config"]]
    star_cached_target_configs = [row for row in star_configs if row["feature_target_enabled"]]
    star_scaled_vjepa_target_configs = [
        row
        for row in star_vjepa_configs
        if row["feature_target_enabled"] and row["target_size"] == 512 and row["frames"] == 64
    ]

    selected_features = selected_config.get("features") if isinstance(selected_config.get("features"), dict) else None
    selected_model = selected_config.get("model") if isinstance(selected_config.get("model"), dict) else None
    selected_feature_uvt = selected_config.get("feature_uvt", {})
    selected_colorize = selected_config.get("colorize", {})

    star_trainer_uses_rgb_target = "target_rgb" in trainer_text and "rgb_loss_weight" in trainer_text
    star_trainer_uses_feature_cache = "VideoFeatureCache" in trainer_text
    star_trainer_has_cached_target_adapter = "_load_cached_feature_target" in trainer_text
    dispatcher_has_star_feature = (
        "run_config" in dispatcher_text
        and '"star_uvt_feature_overfit"' in registry_text
        and '"star_uvt_feature_overfit_trainer"' in registry_text
    )

    if star_scaled_vjepa_target_configs:
        missing_bridge = (
            "The selected fastest STAR UVT feature route still trains from RGB video "
            "targets through FeatureToColor and has no features section, so the "
            "answer for star-feature-512-fast is still no. The separate real-V-JEPA "
            "STAR target route now exists at 64f/512px/8192t/F32 and has a persisted "
            "scale-gate result. cached_chunks now removes most repeated target "
            "interpolation for short runs, and target_grid now avoids the resident "
            "2 GiB adapted cache by moving the loss to the token grid. The 20-step "
            "target_grid media row decreases feature loss, but it is not RGB quality "
            "evidence because rgb_loss_weight=0 and the colorizer is not trained. "
            "The RGB-aux1 target_grid probe trains the colorizer and decreases both "
            "component losses, but the 20-step RGB PSNR gain is tiny; RGB-aux10 is "
            "only marginally better on RGB and slightly worse on feature loss at "
            "20 steps. The 100-step aux10 run improves more clearly but is still "
            "far below RGB STAR quality. The matched rgb-warm20 schedule is faster "
            "but worse than constant aux10 on final RGB PSNR and feature loss. "
            "A standalone hidden64 feature-to-RGB probe trained directly on the "
            "cached target_grid now passes at 23.4 dB grid PSNR and 20.1 dB "
            "full-video upsampled PSNR, so the target features are decodable. "
            "That frozen decoder is now wired into the STAR target-grid trainer "
            "and passes a 20-step media gate at 1.22s/step, but probe PSNR only "
            "moves 13.99->14.06 in that short run. The 100-step sibling moves "
            "probe PSNR to 14.64 at 1.27s/step, and the 300-step extension reaches "
            "16.56 at 1.36s/step with feature loss 0.812. The probe objective keeps "
            "closing the gap. The checkpointed 300-step rerun produces a reusable "
            "state, and a resumed 300-step continuation reaches 19.88 dB probe PSNR "
            "with feature loss 0.655. That nearly reaches the standalone full-video "
            "upsample PSNR number. A probe-emphasis 600->800 continuation reaches "
            "21.42 dB probe PSNR with zero overflow, but feature-grid loss drifts "
            "upward from 0.655 to 0.704. It passes the standalone full-video "
            "upsample number. The scheduled 800->1000 balance continuation recovers "
            "feature loss from 0.704 to 0.644 at 1.31s/step, but gives back a small "
            "amount of probe PSNR from 21.43 to 21.38 and is nonpassing on the "
            "probe-loss-decrease gate. The feature0.5/probe40 1000->1100 Pareto "
            "continuation passes the combined gate and moves probe PSNR to 21.79 "
            "at 1.46s/step, but feature loss drifts back to 0.657. The "
            "1100->1200 recover schedule lowers feature loss to 0.635 at "
            "1.52s/step, but gives back a little probe PSNR to 21.74 and is "
            "nonpassing on probe-loss decrease. A short feature0.75/probe40 "
            "1200->1250 continuation restores probe PSNR to 21.93 at 1.52s/step, "
            "but feature loss rises to 0.639. A feature1/probe40 1250->1300 "
            "continuation is the first current both-improving row: feature loss "
            "falls to 0.632 and probe PSNR nudges to 21.96 at 1.28s/step. A "
            "1300->1400 extension keeps both improving to feature loss 0.627 "
            "and probe PSNR 21.98, but slows to 1.69s/step. "
            "These rows still trail the same-grid "
            "23.4 dB oracle. "
            "The bridge is no longer config plumbing, simple target caching, or aux "
            "RGB schedules; it is balancing frozen-probe visual gain against V-JEPA "
            "target alignment, plus native-VJP or dataset-scale work."
        )
    elif star_vjepa_configs:
        missing_bridge = (
            "The selected fastest STAR UVT feature route still trains from RGB video "
            "targets through FeatureToColor and has no features section. The STAR "
            "feature trainer now has an opt-in cached-target adapter. A STAR UVT "
            "real-V-JEPA target config exists, but it is a separate smoke gate, not "
            "the selected 512px fast diagnostic. The next bridge is to scale that "
            "V-JEPA target config to the selected 512px renderer and then compare "
            "against the Gaussian/token precomputed-feature rows."
        )
    else:
        missing_bridge = (
            "The selected fastest STAR UVT feature route still trains from RGB video "
            "targets through FeatureToColor and has no features section. The STAR "
            "feature trainer now has an opt-in cached-target adapter and an "
            "rgb_pyramid smoke config, but no STAR UVT feature config uses real "
            "precomputed V-JEPA targets yet. Cached V-JEPA features still live in "
            "the separate precomputed_feature_implicit_camera / "
            "multicam_precomputed_feature_implicit_camera Gaussian-token trainer "
            "family until a V-JEPA STAR target-grid config is added."
        )

    return {
        "gate": "star_uvt_precomputed_vjepa_bridge_audit",
        "audit_pass": True,
        "selected_fast_config": {
            "path": SELECTED_FAST_CONFIG,
            "arch": selected_config.get("arch"),
            "render_mode": selected_feature_uvt.get("render_mode"),
            "feature_dim": selected_feature_uvt.get("feature_dim"),
            "target_size": selected_config.get("data", {}).get("target_size"),
            "frames": selected_config.get("data", {}).get("max_frames"),
            "tube_count": selected_feature_uvt.get("tube_count"),
            "colorize_pre_norm": selected_colorize.get("pre_norm"),
            "has_features_section": selected_features is not None,
            "has_model_section": selected_model is not None,
            "uses_vjepa_or_precomputed_features": _has_vjepa_feature_section(selected_config),
        },
        "code_contract": {
            "dispatcher_has_star_uvt_feature_route": dispatcher_has_star_feature,
            "star_feature_trainer_uses_rgb_video_target": star_trainer_uses_rgb_target,
            "star_feature_trainer_uses_video_feature_cache": star_trainer_uses_feature_cache,
            "star_feature_trainer_has_cached_target_adapter": star_trainer_has_cached_target_adapter,
            "star_feature_trainer_path": STAR_FEATURE_TRAINER,
        },
        "config_inventory": {
            "star_uvt_feature_config_count": len(star_configs),
            "star_uvt_feature_configs_with_vjepa_or_precomputed_features": star_vjepa_configs,
            "star_uvt_feature_configs_with_cached_target": star_cached_target_configs,
            "star_uvt_feature_scaled_vjepa_target_configs": star_scaled_vjepa_target_configs,
            "precomputed_vjepa_config_count": len(precomputed_vjepa_configs),
            "precomputed_vjepa_config_examples": precomputed_vjepa_configs[:12],
        },
        "scaled_vjepa_target_config": {
            "path": SCALED_VJEPA_TARGET_CONFIG,
            "exists": scaled_vjepa_path.exists(),
            "arch": scaled_vjepa_config.get("arch"),
            "render_mode": scaled_vjepa_config.get("feature_uvt", {}).get("render_mode")
            if isinstance(scaled_vjepa_config.get("feature_uvt"), dict)
            else None,
            "target_size": scaled_vjepa_config.get("data", {}).get("target_size")
            if isinstance(scaled_vjepa_config.get("data"), dict)
            else None,
            "frames": scaled_vjepa_config.get("data", {}).get("max_frames")
            if isinstance(scaled_vjepa_config.get("data"), dict)
            else None,
            "tube_count": scaled_vjepa_config.get("feature_uvt", {}).get("tube_count")
            if isinstance(scaled_vjepa_config.get("feature_uvt"), dict)
            else None,
            "feature_target_enabled": bool(scaled_vjepa_config.get("feature_target", {}).get("enabled", False))
            if isinstance(scaled_vjepa_config.get("feature_target"), dict)
            else False,
            "feature_target_materialization": scaled_vjepa_config.get("feature_target", {}).get("materialization")
            if isinstance(scaled_vjepa_config.get("feature_target"), dict)
            else None,
        },
        "scaled_vjepa_target_result": None
        if scaled_vjepa_result is None
        else {
            "path": SCALED_VJEPA_TARGET_RESULT,
            "pass": scaled_vjepa_result.get("pass"),
            "start_loss": scaled_vjepa_result.get("start_loss"),
            "end_loss": scaled_vjepa_result.get("end_loss"),
            "mean_timing_ms": scaled_vjepa_result.get("mean_timing_ms"),
            "tile_overflow_sum": scaled_vjepa_result.get("tile_overflow_sum"),
            "feature_target": scaled_vjepa_result.get("feature_target"),
        },
        "cached_chunks_vjepa_target_config": {
            "path": CACHED_CHUNKS_VJEPA_TARGET_CONFIG,
            "exists": cached_chunks_path.exists(),
            "feature_target_materialization": cached_chunks_config.get("feature_target", {}).get("materialization")
            if isinstance(cached_chunks_config.get("feature_target"), dict)
            else None,
        },
        "cached_chunks_vjepa_target_result": None
        if cached_chunks_result is None
        else {
            "path": CACHED_CHUNKS_VJEPA_TARGET_RESULT,
            "pass": cached_chunks_result.get("pass"),
            "start_loss": cached_chunks_result.get("start_loss"),
            "end_loss": cached_chunks_result.get("end_loss"),
            "feature_target_load_ms": cached_chunks_result.get("feature_target_load_ms"),
            "mean_timing_ms": cached_chunks_result.get("mean_timing_ms"),
            "tile_overflow_sum": cached_chunks_result.get("tile_overflow_sum"),
            "feature_target": cached_chunks_result.get("feature_target"),
        },
        "target_grid_vjepa_target_config": {
            "path": TARGET_GRID_VJEPA_TARGET_CONFIG,
            "exists": target_grid_path.exists(),
            "feature_target_materialization": target_grid_config.get("feature_target", {}).get("materialization")
            if isinstance(target_grid_config.get("feature_target"), dict)
            else None,
        },
        "target_grid_vjepa_target_result": None
        if target_grid_result is None
        else {
            "path": TARGET_GRID_VJEPA_TARGET_RESULT,
            "pass": target_grid_result.get("pass"),
            "start_loss": target_grid_result.get("start_loss"),
            "end_loss": target_grid_result.get("end_loss"),
            "feature_target_load_ms": target_grid_result.get("feature_target_load_ms"),
            "mean_timing_ms": target_grid_result.get("mean_timing_ms"),
            "tile_overflow_sum": target_grid_result.get("tile_overflow_sum"),
            "feature_target": target_grid_result.get("feature_target"),
        },
        "target_grid_media_vjepa_target_config": {
            "path": TARGET_GRID_MEDIA_VJEPA_TARGET_CONFIG,
            "exists": target_grid_media_path.exists(),
            "feature_target_materialization": target_grid_media_config.get("feature_target", {}).get("materialization")
            if isinstance(target_grid_media_config.get("feature_target"), dict)
            else None,
            "rgb_loss_weight": target_grid_media_config.get("feature_target", {}).get("rgb_loss_weight")
            if isinstance(target_grid_media_config.get("feature_target"), dict)
            else None,
        },
        "target_grid_media_vjepa_target_result": None
        if target_grid_media_result is None
        else {
            "path": TARGET_GRID_MEDIA_VJEPA_TARGET_RESULT,
            "pass": target_grid_media_result.get("pass"),
            "start_loss": target_grid_media_result.get("start_loss"),
            "end_loss": target_grid_media_result.get("end_loss"),
            "start_psnr": target_grid_media_result.get("start_psnr"),
            "end_psnr": target_grid_media_result.get("end_psnr"),
            "feature_target_load_ms": target_grid_media_result.get("feature_target_load_ms"),
            "mean_timing_ms": target_grid_media_result.get("mean_timing_ms"),
            "tile_overflow_sum": target_grid_media_result.get("tile_overflow_sum"),
            "feature_target": target_grid_media_result.get("feature_target"),
            "contact_sheet": target_grid_media_result.get("contact_sheet"),
            "side_by_side_video": target_grid_media_result.get("side_by_side_video"),
        },
        "target_grid_rgb_aux_vjepa_target_config": {
            "path": TARGET_GRID_RGB_AUX_VJEPA_TARGET_CONFIG,
            "exists": target_grid_rgb_aux_path.exists(),
            "feature_target_materialization": target_grid_rgb_aux_config.get("feature_target", {}).get("materialization")
            if isinstance(target_grid_rgb_aux_config.get("feature_target"), dict)
            else None,
            "rgb_loss_weight": target_grid_rgb_aux_config.get("feature_target", {}).get("rgb_loss_weight")
            if isinstance(target_grid_rgb_aux_config.get("feature_target"), dict)
            else None,
        },
        "target_grid_rgb_aux_vjepa_target_result": None
        if target_grid_rgb_aux_result is None
        else {
            "path": TARGET_GRID_RGB_AUX_VJEPA_TARGET_RESULT,
            "pass": target_grid_rgb_aux_result.get("pass"),
            "start_loss": target_grid_rgb_aux_result.get("start_loss"),
            "end_loss": target_grid_rgb_aux_result.get("end_loss"),
            "start_feature_target_loss": target_grid_rgb_aux_result.get("start_feature_target_loss"),
            "end_feature_target_loss": target_grid_rgb_aux_result.get("end_feature_target_loss"),
            "start_rgb_loss": target_grid_rgb_aux_result.get("start_rgb_loss"),
            "end_rgb_loss": target_grid_rgb_aux_result.get("end_rgb_loss"),
            "start_rgb_psnr": target_grid_rgb_aux_result.get("start_rgb_psnr"),
            "end_rgb_psnr": target_grid_rgb_aux_result.get("end_rgb_psnr"),
            "feature_target_load_ms": target_grid_rgb_aux_result.get("feature_target_load_ms"),
            "mean_timing_ms": target_grid_rgb_aux_result.get("mean_timing_ms"),
            "tile_overflow_sum": target_grid_rgb_aux_result.get("tile_overflow_sum"),
            "feature_target": target_grid_rgb_aux_result.get("feature_target"),
            "colorizer_grad_seen": target_grid_rgb_aux_result.get("colorizer_grad_seen"),
            "contact_sheet": target_grid_rgb_aux_result.get("contact_sheet"),
            "side_by_side_video": target_grid_rgb_aux_result.get("side_by_side_video"),
        },
        "target_grid_rgb_aux10_vjepa_target_config": {
            "path": TARGET_GRID_RGB_AUX10_VJEPA_TARGET_CONFIG,
            "exists": target_grid_rgb_aux10_path.exists(),
            "feature_target_materialization": target_grid_rgb_aux10_config.get("feature_target", {}).get("materialization")
            if isinstance(target_grid_rgb_aux10_config.get("feature_target"), dict)
            else None,
            "rgb_loss_weight": target_grid_rgb_aux10_config.get("feature_target", {}).get("rgb_loss_weight")
            if isinstance(target_grid_rgb_aux10_config.get("feature_target"), dict)
            else None,
        },
        "target_grid_rgb_aux10_vjepa_target_result": None
        if target_grid_rgb_aux10_result is None
        else {
            "path": TARGET_GRID_RGB_AUX10_VJEPA_TARGET_RESULT,
            "pass": target_grid_rgb_aux10_result.get("pass"),
            "start_loss": target_grid_rgb_aux10_result.get("start_loss"),
            "end_loss": target_grid_rgb_aux10_result.get("end_loss"),
            "start_feature_target_loss": target_grid_rgb_aux10_result.get("start_feature_target_loss"),
            "end_feature_target_loss": target_grid_rgb_aux10_result.get("end_feature_target_loss"),
            "start_rgb_loss": target_grid_rgb_aux10_result.get("start_rgb_loss"),
            "end_rgb_loss": target_grid_rgb_aux10_result.get("end_rgb_loss"),
            "start_rgb_psnr": target_grid_rgb_aux10_result.get("start_rgb_psnr"),
            "end_rgb_psnr": target_grid_rgb_aux10_result.get("end_rgb_psnr"),
            "feature_target_load_ms": target_grid_rgb_aux10_result.get("feature_target_load_ms"),
            "mean_timing_ms": target_grid_rgb_aux10_result.get("mean_timing_ms"),
            "tile_overflow_sum": target_grid_rgb_aux10_result.get("tile_overflow_sum"),
            "feature_target": target_grid_rgb_aux10_result.get("feature_target"),
            "colorizer_grad_seen": target_grid_rgb_aux10_result.get("colorizer_grad_seen"),
            "contact_sheet": target_grid_rgb_aux10_result.get("contact_sheet"),
            "side_by_side_video": target_grid_rgb_aux10_result.get("side_by_side_video"),
        },
        "target_grid_rgb_aux10_100step_vjepa_target_config": {
            "path": TARGET_GRID_RGB_AUX10_100STEP_VJEPA_TARGET_CONFIG,
            "exists": target_grid_rgb_aux10_100step_path.exists(),
            "feature_target_materialization": target_grid_rgb_aux10_100step_config.get("feature_target", {}).get("materialization")
            if isinstance(target_grid_rgb_aux10_100step_config.get("feature_target"), dict)
            else None,
            "rgb_loss_weight": target_grid_rgb_aux10_100step_config.get("feature_target", {}).get("rgb_loss_weight")
            if isinstance(target_grid_rgb_aux10_100step_config.get("feature_target"), dict)
            else None,
        },
        "target_grid_rgb_aux10_100step_vjepa_target_result": None
        if target_grid_rgb_aux10_100step_result is None
        else {
            "path": TARGET_GRID_RGB_AUX10_100STEP_VJEPA_TARGET_RESULT,
            "pass": target_grid_rgb_aux10_100step_result.get("pass"),
            "start_loss": target_grid_rgb_aux10_100step_result.get("start_loss"),
            "end_loss": target_grid_rgb_aux10_100step_result.get("end_loss"),
            "start_feature_target_loss": target_grid_rgb_aux10_100step_result.get("start_feature_target_loss"),
            "end_feature_target_loss": target_grid_rgb_aux10_100step_result.get("end_feature_target_loss"),
            "start_rgb_loss": target_grid_rgb_aux10_100step_result.get("start_rgb_loss"),
            "end_rgb_loss": target_grid_rgb_aux10_100step_result.get("end_rgb_loss"),
            "start_rgb_psnr": target_grid_rgb_aux10_100step_result.get("start_rgb_psnr"),
            "end_rgb_psnr": target_grid_rgb_aux10_100step_result.get("end_rgb_psnr"),
            "feature_target_load_ms": target_grid_rgb_aux10_100step_result.get("feature_target_load_ms"),
            "mean_timing_ms": target_grid_rgb_aux10_100step_result.get("mean_timing_ms"),
            "tile_overflow_sum": target_grid_rgb_aux10_100step_result.get("tile_overflow_sum"),
            "feature_target": target_grid_rgb_aux10_100step_result.get("feature_target"),
            "colorizer_grad_seen": target_grid_rgb_aux10_100step_result.get("colorizer_grad_seen"),
            "contact_sheet": target_grid_rgb_aux10_100step_result.get("contact_sheet"),
            "side_by_side_video": target_grid_rgb_aux10_100step_result.get("side_by_side_video"),
        },
        "target_grid_rgbwarm20_aux10_100step_vjepa_target_config": {
            "path": TARGET_GRID_RGBWARM20_AUX10_100STEP_VJEPA_TARGET_CONFIG,
            "exists": target_grid_rgbwarm20_aux10_100step_path.exists(),
            "feature_target_materialization": target_grid_rgbwarm20_aux10_100step_config.get(
                "feature_target", {}
            ).get("materialization")
            if isinstance(target_grid_rgbwarm20_aux10_100step_config.get("feature_target"), dict)
            else None,
            "rgb_loss_weight": target_grid_rgbwarm20_aux10_100step_config.get("feature_target", {}).get(
                "rgb_loss_weight"
            )
            if isinstance(target_grid_rgbwarm20_aux10_100step_config.get("feature_target"), dict)
            else None,
            "weight_schedule": target_grid_rgbwarm20_aux10_100step_config.get("feature_target", {}).get(
                "weight_schedule"
            )
            if isinstance(target_grid_rgbwarm20_aux10_100step_config.get("feature_target"), dict)
            else None,
        },
        "target_grid_rgbwarm20_aux10_100step_vjepa_target_result": None
        if target_grid_rgbwarm20_aux10_100step_result is None
        else {
            "path": TARGET_GRID_RGBWARM20_AUX10_100STEP_VJEPA_TARGET_RESULT,
            "pass": target_grid_rgbwarm20_aux10_100step_result.get("pass"),
            "start_loss": target_grid_rgbwarm20_aux10_100step_result.get("start_loss"),
            "end_loss": target_grid_rgbwarm20_aux10_100step_result.get("end_loss"),
            "start_feature_target_loss": target_grid_rgbwarm20_aux10_100step_result.get(
                "start_feature_target_loss"
            ),
            "end_feature_target_loss": target_grid_rgbwarm20_aux10_100step_result.get(
                "end_feature_target_loss"
            ),
            "start_rgb_loss": target_grid_rgbwarm20_aux10_100step_result.get("start_rgb_loss"),
            "end_rgb_loss": target_grid_rgbwarm20_aux10_100step_result.get("end_rgb_loss"),
            "start_rgb_psnr": target_grid_rgbwarm20_aux10_100step_result.get("start_rgb_psnr"),
            "end_rgb_psnr": target_grid_rgbwarm20_aux10_100step_result.get("end_rgb_psnr"),
            "feature_target_load_ms": target_grid_rgbwarm20_aux10_100step_result.get("feature_target_load_ms"),
            "mean_timing_ms": target_grid_rgbwarm20_aux10_100step_result.get("mean_timing_ms"),
            "tile_overflow_sum": target_grid_rgbwarm20_aux10_100step_result.get("tile_overflow_sum"),
            "feature_target": target_grid_rgbwarm20_aux10_100step_result.get("feature_target"),
            "feature_target_weight_schedule": target_grid_rgbwarm20_aux10_100step_result.get(
                "feature_target_weight_schedule"
            ),
            "colorizer_grad_seen": target_grid_rgbwarm20_aux10_100step_result.get("colorizer_grad_seen"),
            "contact_sheet": target_grid_rgbwarm20_aux10_100step_result.get("contact_sheet"),
            "side_by_side_video": target_grid_rgbwarm20_aux10_100step_result.get("side_by_side_video"),
        },
        "target_grid_feature_rgb_probe_config": {
            "path": TARGET_GRID_FEATURE_RGB_PROBE_CONFIG,
            "exists": target_grid_feature_rgb_probe_path.exists(),
            "arch": target_grid_feature_rgb_probe_config.get("arch"),
            "target_size": target_grid_feature_rgb_probe_config.get("data", {}).get("target_size")
            if isinstance(target_grid_feature_rgb_probe_config.get("data"), dict)
            else None,
            "frames": target_grid_feature_rgb_probe_config.get("data", {}).get("max_frames")
            if isinstance(target_grid_feature_rgb_probe_config.get("data"), dict)
            else None,
            "feature_dim": target_grid_feature_rgb_probe_config.get("feature_uvt", {}).get("feature_dim")
            if isinstance(target_grid_feature_rgb_probe_config.get("feature_uvt"), dict)
            else None,
            "hidden_dim": target_grid_feature_rgb_probe_config.get("colorize", {}).get("hidden_dim")
            if isinstance(target_grid_feature_rgb_probe_config.get("colorize"), dict)
            else None,
            "steps": target_grid_feature_rgb_probe_config.get("probe", {}).get("steps")
            if isinstance(target_grid_feature_rgb_probe_config.get("probe"), dict)
            else None,
            "feature_target_materialization": target_grid_feature_rgb_probe_config.get("feature_target", {}).get(
                "materialization"
            )
            if isinstance(target_grid_feature_rgb_probe_config.get("feature_target"), dict)
            else None,
        },
        "target_grid_feature_rgb_probe_result": None
        if target_grid_feature_rgb_probe_result is None
        else {
            "path": TARGET_GRID_FEATURE_RGB_PROBE_RESULT,
            "pass": target_grid_feature_rgb_probe_result.get("pass"),
            "start_grid_loss": target_grid_feature_rgb_probe_result.get("start_grid_loss"),
            "end_grid_loss": target_grid_feature_rgb_probe_result.get("end_grid_loss"),
            "final_grid_loss": target_grid_feature_rgb_probe_result.get("final_grid_loss"),
            "start_grid_psnr": target_grid_feature_rgb_probe_result.get("start_grid_psnr"),
            "end_grid_psnr": target_grid_feature_rgb_probe_result.get("end_grid_psnr"),
            "final_grid_psnr": target_grid_feature_rgb_probe_result.get("final_grid_psnr"),
            "final_full_psnr": target_grid_feature_rgb_probe_result.get("final_full_psnr"),
            "feature_target_load_ms": target_grid_feature_rgb_probe_result.get("feature_target_load_ms"),
            "mean_timing_ms": target_grid_feature_rgb_probe_result.get("mean_timing_ms"),
            "target_grid_shape": target_grid_feature_rgb_probe_result.get("target_grid_shape"),
            "target_grid_rgb_shape": target_grid_feature_rgb_probe_result.get("target_grid_rgb_shape"),
            "feature_target": target_grid_feature_rgb_probe_result.get("feature_target"),
            "checkpoint": target_grid_feature_rgb_probe_result.get("checkpoint"),
            "contact_sheet": target_grid_feature_rgb_probe_result.get("contact_sheet"),
            "side_by_side_video": target_grid_feature_rgb_probe_result.get("side_by_side_video"),
            "wandb_run_id": target_grid_feature_rgb_probe_result.get("wandb_run_id"),
        },
        "target_grid_rgbprobe10_vjepa_target_config": {
            "path": TARGET_GRID_RGBPROBE10_VJEPA_TARGET_CONFIG,
            "exists": target_grid_rgbprobe10_path.exists(),
            "feature_target_materialization": target_grid_rgbprobe10_config.get("feature_target", {}).get(
                "materialization"
            )
            if isinstance(target_grid_rgbprobe10_config.get("feature_target"), dict)
            else None,
            "rgb_probe_checkpoint": target_grid_rgbprobe10_config.get("feature_target", {}).get(
                "rgb_probe_checkpoint"
            )
            if isinstance(target_grid_rgbprobe10_config.get("feature_target"), dict)
            else None,
            "rgb_probe_loss_weight": target_grid_rgbprobe10_config.get("feature_target", {}).get(
                "rgb_probe_loss_weight"
            )
            if isinstance(target_grid_rgbprobe10_config.get("feature_target"), dict)
            else None,
        },
        "target_grid_rgbprobe10_vjepa_target_result": None
        if target_grid_rgbprobe10_result is None
        else {
            "path": TARGET_GRID_RGBPROBE10_VJEPA_TARGET_RESULT,
            "pass": target_grid_rgbprobe10_result.get("pass"),
            "start_loss": target_grid_rgbprobe10_result.get("start_loss"),
            "end_loss": target_grid_rgbprobe10_result.get("end_loss"),
            "start_feature_target_loss": target_grid_rgbprobe10_result.get("start_feature_target_loss"),
            "end_feature_target_loss": target_grid_rgbprobe10_result.get("end_feature_target_loss"),
            "start_rgb_probe_loss": target_grid_rgbprobe10_result.get("start_rgb_probe_loss"),
            "end_rgb_probe_loss": target_grid_rgbprobe10_result.get("end_rgb_probe_loss"),
            "start_rgb_probe_psnr": target_grid_rgbprobe10_result.get("start_rgb_probe_psnr"),
            "end_rgb_probe_psnr": target_grid_rgbprobe10_result.get("end_rgb_probe_psnr"),
            "feature_target_load_ms": target_grid_rgbprobe10_result.get("feature_target_load_ms"),
            "mean_timing_ms": target_grid_rgbprobe10_result.get("mean_timing_ms"),
            "tile_overflow_sum": target_grid_rgbprobe10_result.get("tile_overflow_sum"),
            "feature_target": target_grid_rgbprobe10_result.get("feature_target"),
            "rgb_probe": target_grid_rgbprobe10_result.get("rgb_probe"),
            "rgb_probe_media_render_ms": target_grid_rgbprobe10_result.get("rgb_probe_media_render_ms"),
            "rgb_probe_contact_sheet": target_grid_rgbprobe10_result.get("rgb_probe_contact_sheet"),
            "rgb_probe_side_by_side_video": target_grid_rgbprobe10_result.get("rgb_probe_side_by_side_video"),
        },
        "target_grid_rgbprobe10_100step_vjepa_target_config": {
            "path": TARGET_GRID_RGBPROBE10_100STEP_VJEPA_TARGET_CONFIG,
            "exists": target_grid_rgbprobe10_100step_path.exists(),
            "feature_target_materialization": target_grid_rgbprobe10_100step_config.get("feature_target", {}).get(
                "materialization"
            )
            if isinstance(target_grid_rgbprobe10_100step_config.get("feature_target"), dict)
            else None,
            "rgb_probe_checkpoint": target_grid_rgbprobe10_100step_config.get("feature_target", {}).get(
                "rgb_probe_checkpoint"
            )
            if isinstance(target_grid_rgbprobe10_100step_config.get("feature_target"), dict)
            else None,
            "rgb_probe_loss_weight": target_grid_rgbprobe10_100step_config.get("feature_target", {}).get(
                "rgb_probe_loss_weight"
            )
            if isinstance(target_grid_rgbprobe10_100step_config.get("feature_target"), dict)
            else None,
        },
        "target_grid_rgbprobe10_100step_vjepa_target_result": None
        if target_grid_rgbprobe10_100step_result is None
        else {
            "path": TARGET_GRID_RGBPROBE10_100STEP_VJEPA_TARGET_RESULT,
            "pass": target_grid_rgbprobe10_100step_result.get("pass"),
            "start_loss": target_grid_rgbprobe10_100step_result.get("start_loss"),
            "end_loss": target_grid_rgbprobe10_100step_result.get("end_loss"),
            "start_feature_target_loss": target_grid_rgbprobe10_100step_result.get("start_feature_target_loss"),
            "end_feature_target_loss": target_grid_rgbprobe10_100step_result.get("end_feature_target_loss"),
            "start_rgb_probe_loss": target_grid_rgbprobe10_100step_result.get("start_rgb_probe_loss"),
            "end_rgb_probe_loss": target_grid_rgbprobe10_100step_result.get("end_rgb_probe_loss"),
            "start_rgb_probe_psnr": target_grid_rgbprobe10_100step_result.get("start_rgb_probe_psnr"),
            "end_rgb_probe_psnr": target_grid_rgbprobe10_100step_result.get("end_rgb_probe_psnr"),
            "feature_target_load_ms": target_grid_rgbprobe10_100step_result.get("feature_target_load_ms"),
            "mean_timing_ms": target_grid_rgbprobe10_100step_result.get("mean_timing_ms"),
            "tile_overflow_sum": target_grid_rgbprobe10_100step_result.get("tile_overflow_sum"),
            "feature_target": target_grid_rgbprobe10_100step_result.get("feature_target"),
            "rgb_probe": target_grid_rgbprobe10_100step_result.get("rgb_probe"),
            "rgb_probe_media_render_ms": target_grid_rgbprobe10_100step_result.get("rgb_probe_media_render_ms"),
            "rgb_probe_contact_sheet": target_grid_rgbprobe10_100step_result.get("rgb_probe_contact_sheet"),
            "rgb_probe_side_by_side_video": target_grid_rgbprobe10_100step_result.get(
                "rgb_probe_side_by_side_video"
            ),
        },
        "target_grid_rgbprobe10_300step_vjepa_target_config": {
            "path": TARGET_GRID_RGBPROBE10_300STEP_VJEPA_TARGET_CONFIG,
            "exists": target_grid_rgbprobe10_300step_path.exists(),
            "feature_target_materialization": target_grid_rgbprobe10_300step_config.get("feature_target", {}).get(
                "materialization"
            )
            if isinstance(target_grid_rgbprobe10_300step_config.get("feature_target"), dict)
            else None,
            "rgb_probe_checkpoint": target_grid_rgbprobe10_300step_config.get("feature_target", {}).get(
                "rgb_probe_checkpoint"
            )
            if isinstance(target_grid_rgbprobe10_300step_config.get("feature_target"), dict)
            else None,
            "rgb_probe_loss_weight": target_grid_rgbprobe10_300step_config.get("feature_target", {}).get(
                "rgb_probe_loss_weight"
            )
            if isinstance(target_grid_rgbprobe10_300step_config.get("feature_target"), dict)
            else None,
        },
        "target_grid_rgbprobe10_300step_vjepa_target_result": None
        if target_grid_rgbprobe10_300step_result is None
        else {
            "path": TARGET_GRID_RGBPROBE10_300STEP_VJEPA_TARGET_RESULT,
            "pass": target_grid_rgbprobe10_300step_result.get("pass"),
            "start_loss": target_grid_rgbprobe10_300step_result.get("start_loss"),
            "end_loss": target_grid_rgbprobe10_300step_result.get("end_loss"),
            "start_feature_target_loss": target_grid_rgbprobe10_300step_result.get("start_feature_target_loss"),
            "end_feature_target_loss": target_grid_rgbprobe10_300step_result.get("end_feature_target_loss"),
            "start_rgb_probe_loss": target_grid_rgbprobe10_300step_result.get("start_rgb_probe_loss"),
            "end_rgb_probe_loss": target_grid_rgbprobe10_300step_result.get("end_rgb_probe_loss"),
            "start_rgb_probe_psnr": target_grid_rgbprobe10_300step_result.get("start_rgb_probe_psnr"),
            "end_rgb_probe_psnr": target_grid_rgbprobe10_300step_result.get("end_rgb_probe_psnr"),
            "feature_target_load_ms": target_grid_rgbprobe10_300step_result.get("feature_target_load_ms"),
            "mean_timing_ms": target_grid_rgbprobe10_300step_result.get("mean_timing_ms"),
            "tile_overflow_sum": target_grid_rgbprobe10_300step_result.get("tile_overflow_sum"),
            "feature_target": target_grid_rgbprobe10_300step_result.get("feature_target"),
            "rgb_probe": target_grid_rgbprobe10_300step_result.get("rgb_probe"),
            "rgb_probe_media_render_ms": target_grid_rgbprobe10_300step_result.get("rgb_probe_media_render_ms"),
            "rgb_probe_contact_sheet": target_grid_rgbprobe10_300step_result.get("rgb_probe_contact_sheet"),
            "rgb_probe_side_by_side_video": target_grid_rgbprobe10_300step_result.get(
                "rgb_probe_side_by_side_video"
            ),
        },
        "target_grid_rgbprobe10_300step_checkpoint_vjepa_target_config": {
            "path": TARGET_GRID_RGBPROBE10_300STEP_CHECKPOINT_VJEPA_TARGET_CONFIG,
            "exists": target_grid_rgbprobe10_300step_checkpoint_path.exists(),
            "checkpoint": target_grid_rgbprobe10_300step_checkpoint_config.get("output", {}).get("checkpoint")
            if isinstance(target_grid_rgbprobe10_300step_checkpoint_config.get("output"), dict)
            else None,
            "steps": target_grid_rgbprobe10_300step_checkpoint_config.get("train", {}).get("steps")
            if isinstance(target_grid_rgbprobe10_300step_checkpoint_config.get("train"), dict)
            else None,
            "rgb_probe_loss_weight": target_grid_rgbprobe10_300step_checkpoint_config.get("feature_target", {}).get(
                "rgb_probe_loss_weight"
            )
            if isinstance(target_grid_rgbprobe10_300step_checkpoint_config.get("feature_target"), dict)
            else None,
        },
        "target_grid_rgbprobe10_300step_checkpoint_vjepa_target_result": None
        if target_grid_rgbprobe10_300step_checkpoint_result is None
        else {
            "path": TARGET_GRID_RGBPROBE10_300STEP_CHECKPOINT_VJEPA_TARGET_RESULT,
            "pass": target_grid_rgbprobe10_300step_checkpoint_result.get("pass"),
            "checkpoint": target_grid_rgbprobe10_300step_checkpoint_result.get("checkpoint"),
            "start_feature_target_loss": target_grid_rgbprobe10_300step_checkpoint_result.get(
                "start_feature_target_loss"
            ),
            "end_feature_target_loss": target_grid_rgbprobe10_300step_checkpoint_result.get(
                "end_feature_target_loss"
            ),
            "start_rgb_probe_loss": target_grid_rgbprobe10_300step_checkpoint_result.get("start_rgb_probe_loss"),
            "end_rgb_probe_loss": target_grid_rgbprobe10_300step_checkpoint_result.get("end_rgb_probe_loss"),
            "start_rgb_probe_psnr": target_grid_rgbprobe10_300step_checkpoint_result.get("start_rgb_probe_psnr"),
            "end_rgb_probe_psnr": target_grid_rgbprobe10_300step_checkpoint_result.get("end_rgb_probe_psnr"),
            "mean_timing_ms": target_grid_rgbprobe10_300step_checkpoint_result.get("mean_timing_ms"),
            "tile_overflow_sum": target_grid_rgbprobe10_300step_checkpoint_result.get("tile_overflow_sum"),
        },
        "target_grid_rgbprobe10_resume300_from300_vjepa_target_config": {
            "path": TARGET_GRID_RGBPROBE10_RESUME300_FROM300_VJEPA_TARGET_CONFIG,
            "exists": target_grid_rgbprobe10_resume300_from300_path.exists(),
            "resume_checkpoint": target_grid_rgbprobe10_resume300_from300_config.get("train", {}).get(
                "resume_checkpoint"
            )
            if isinstance(target_grid_rgbprobe10_resume300_from300_config.get("train"), dict)
            else None,
            "checkpoint": target_grid_rgbprobe10_resume300_from300_config.get("output", {}).get("checkpoint")
            if isinstance(target_grid_rgbprobe10_resume300_from300_config.get("output"), dict)
            else None,
            "steps": target_grid_rgbprobe10_resume300_from300_config.get("train", {}).get("steps")
            if isinstance(target_grid_rgbprobe10_resume300_from300_config.get("train"), dict)
            else None,
            "rgb_probe_loss_weight": target_grid_rgbprobe10_resume300_from300_config.get("feature_target", {}).get(
                "rgb_probe_loss_weight"
            )
            if isinstance(target_grid_rgbprobe10_resume300_from300_config.get("feature_target"), dict)
            else None,
        },
        "target_grid_rgbprobe10_resume300_from300_vjepa_target_result": None
        if target_grid_rgbprobe10_resume300_from300_result is None
        else {
            "path": TARGET_GRID_RGBPROBE10_RESUME300_FROM300_VJEPA_TARGET_RESULT,
            "pass": target_grid_rgbprobe10_resume300_from300_result.get("pass"),
            "resume_loaded": target_grid_rgbprobe10_resume300_from300_result.get("resume_loaded"),
            "resume_optimizer_loaded": target_grid_rgbprobe10_resume300_from300_result.get("resume_optimizer_loaded"),
            "resume_checkpoint_steps": target_grid_rgbprobe10_resume300_from300_result.get(
                "resume_checkpoint_steps"
            ),
            "checkpoint": target_grid_rgbprobe10_resume300_from300_result.get("checkpoint"),
            "start_feature_target_loss": target_grid_rgbprobe10_resume300_from300_result.get(
                "start_feature_target_loss"
            ),
            "end_feature_target_loss": target_grid_rgbprobe10_resume300_from300_result.get("end_feature_target_loss"),
            "start_rgb_probe_loss": target_grid_rgbprobe10_resume300_from300_result.get("start_rgb_probe_loss"),
            "end_rgb_probe_loss": target_grid_rgbprobe10_resume300_from300_result.get("end_rgb_probe_loss"),
            "start_rgb_probe_psnr": target_grid_rgbprobe10_resume300_from300_result.get("start_rgb_probe_psnr"),
            "end_rgb_probe_psnr": target_grid_rgbprobe10_resume300_from300_result.get("end_rgb_probe_psnr"),
            "mean_timing_ms": target_grid_rgbprobe10_resume300_from300_result.get("mean_timing_ms"),
            "tile_overflow_sum": target_grid_rgbprobe10_resume300_from300_result.get("tile_overflow_sum"),
            "rgb_probe_media_render_ms": target_grid_rgbprobe10_resume300_from300_result.get(
                "rgb_probe_media_render_ms"
            ),
            "rgb_probe_contact_sheet": target_grid_rgbprobe10_resume300_from300_result.get(
                "rgb_probe_contact_sheet"
            ),
            "rgb_probe_side_by_side_video": target_grid_rgbprobe10_resume300_from300_result.get(
                "rgb_probe_side_by_side_video"
            ),
        },
        "target_grid_rgbprobe40_feature025_resume200_from600_vjepa_target_config": {
            "path": TARGET_GRID_RGBPROBE40_FEATURE025_RESUME200_FROM600_VJEPA_TARGET_CONFIG,
            "exists": target_grid_rgbprobe40_feature025_resume200_from600_path.exists(),
            "resume_checkpoint": target_grid_rgbprobe40_feature025_resume200_from600_config.get("train", {}).get(
                "resume_checkpoint"
            )
            if isinstance(target_grid_rgbprobe40_feature025_resume200_from600_config.get("train"), dict)
            else None,
            "global_step_offset": target_grid_rgbprobe40_feature025_resume200_from600_config.get("train", {}).get(
                "global_step_offset"
            )
            if isinstance(target_grid_rgbprobe40_feature025_resume200_from600_config.get("train"), dict)
            else None,
            "checkpoint": target_grid_rgbprobe40_feature025_resume200_from600_config.get("output", {}).get(
                "checkpoint"
            )
            if isinstance(target_grid_rgbprobe40_feature025_resume200_from600_config.get("output"), dict)
            else None,
            "steps": target_grid_rgbprobe40_feature025_resume200_from600_config.get("train", {}).get("steps")
            if isinstance(target_grid_rgbprobe40_feature025_resume200_from600_config.get("train"), dict)
            else None,
            "feature_loss_weight": target_grid_rgbprobe40_feature025_resume200_from600_config.get(
                "feature_target", {}
            ).get("loss_weight")
            if isinstance(target_grid_rgbprobe40_feature025_resume200_from600_config.get("feature_target"), dict)
            else None,
            "rgb_probe_loss_weight": target_grid_rgbprobe40_feature025_resume200_from600_config.get(
                "feature_target", {}
            ).get("rgb_probe_loss_weight")
            if isinstance(target_grid_rgbprobe40_feature025_resume200_from600_config.get("feature_target"), dict)
            else None,
        },
        "target_grid_rgbprobe40_feature025_resume200_from600_vjepa_target_result": None
        if target_grid_rgbprobe40_feature025_resume200_from600_result is None
        else {
            "path": TARGET_GRID_RGBPROBE40_FEATURE025_RESUME200_FROM600_VJEPA_TARGET_RESULT,
            "pass": target_grid_rgbprobe40_feature025_resume200_from600_result.get("pass"),
            "resume_loaded": target_grid_rgbprobe40_feature025_resume200_from600_result.get("resume_loaded"),
            "resume_optimizer_loaded": target_grid_rgbprobe40_feature025_resume200_from600_result.get(
                "resume_optimizer_loaded"
            ),
            "resume_checkpoint_steps": target_grid_rgbprobe40_feature025_resume200_from600_result.get(
                "resume_checkpoint_steps"
            ),
            "global_step_offset": target_grid_rgbprobe40_feature025_resume200_from600_result.get(
                "global_step_offset"
            ),
            "start_global_step": target_grid_rgbprobe40_feature025_resume200_from600_result.get(
                "start_global_step"
            ),
            "end_global_step": target_grid_rgbprobe40_feature025_resume200_from600_result.get("end_global_step"),
            "checkpoint": target_grid_rgbprobe40_feature025_resume200_from600_result.get("checkpoint"),
            "start_feature_target_loss": target_grid_rgbprobe40_feature025_resume200_from600_result.get(
                "start_feature_target_loss"
            ),
            "end_feature_target_loss": target_grid_rgbprobe40_feature025_resume200_from600_result.get(
                "end_feature_target_loss"
            ),
            "start_rgb_probe_loss": target_grid_rgbprobe40_feature025_resume200_from600_result.get(
                "start_rgb_probe_loss"
            ),
            "end_rgb_probe_loss": target_grid_rgbprobe40_feature025_resume200_from600_result.get(
                "end_rgb_probe_loss"
            ),
            "start_rgb_probe_psnr": target_grid_rgbprobe40_feature025_resume200_from600_result.get(
                "start_rgb_probe_psnr"
            ),
            "end_rgb_probe_psnr": target_grid_rgbprobe40_feature025_resume200_from600_result.get(
                "end_rgb_probe_psnr"
            ),
            "mean_timing_ms": target_grid_rgbprobe40_feature025_resume200_from600_result.get("mean_timing_ms"),
            "tile_overflow_sum": target_grid_rgbprobe40_feature025_resume200_from600_result.get(
                "tile_overflow_sum"
            ),
            "rgb_probe_media_render_ms": target_grid_rgbprobe40_feature025_resume200_from600_result.get(
                "rgb_probe_media_render_ms"
            ),
            "rgb_probe_contact_sheet": target_grid_rgbprobe40_feature025_resume200_from600_result.get(
                "rgb_probe_contact_sheet"
            ),
            "rgb_probe_side_by_side_video": target_grid_rgbprobe40_feature025_resume200_from600_result.get(
                "rgb_probe_side_by_side_video"
            ),
        },
        "target_grid_rgbprobe_balance_resume200_from800_vjepa_target_config": {
            "path": TARGET_GRID_RGBPROBE_BALANCE_RESUME200_FROM800_VJEPA_TARGET_CONFIG,
            "exists": target_grid_rgbprobe_balance_resume200_from800_path.exists(),
            "resume_checkpoint": target_grid_rgbprobe_balance_resume200_from800_config.get("train", {}).get(
                "resume_checkpoint"
            )
            if isinstance(target_grid_rgbprobe_balance_resume200_from800_config.get("train"), dict)
            else None,
            "global_step_offset": target_grid_rgbprobe_balance_resume200_from800_config.get("train", {}).get(
                "global_step_offset"
            )
            if isinstance(target_grid_rgbprobe_balance_resume200_from800_config.get("train"), dict)
            else None,
            "checkpoint": target_grid_rgbprobe_balance_resume200_from800_config.get("output", {}).get("checkpoint")
            if isinstance(target_grid_rgbprobe_balance_resume200_from800_config.get("output"), dict)
            else None,
            "steps": target_grid_rgbprobe_balance_resume200_from800_config.get("train", {}).get("steps")
            if isinstance(target_grid_rgbprobe_balance_resume200_from800_config.get("train"), dict)
            else None,
            "weight_schedule": target_grid_rgbprobe_balance_resume200_from800_config.get("feature_target", {}).get(
                "weight_schedule"
            )
            if isinstance(target_grid_rgbprobe_balance_resume200_from800_config.get("feature_target"), dict)
            else None,
        },
        "target_grid_rgbprobe_balance_resume200_from800_vjepa_target_result": None
        if target_grid_rgbprobe_balance_resume200_from800_result is None
        else {
            "path": TARGET_GRID_RGBPROBE_BALANCE_RESUME200_FROM800_VJEPA_TARGET_RESULT,
            "pass": target_grid_rgbprobe_balance_resume200_from800_result.get("pass"),
            "resume_loaded": target_grid_rgbprobe_balance_resume200_from800_result.get("resume_loaded"),
            "resume_optimizer_loaded": target_grid_rgbprobe_balance_resume200_from800_result.get(
                "resume_optimizer_loaded"
            ),
            "resume_checkpoint_steps": target_grid_rgbprobe_balance_resume200_from800_result.get(
                "resume_checkpoint_steps"
            ),
            "global_step_offset": target_grid_rgbprobe_balance_resume200_from800_result.get("global_step_offset"),
            "start_global_step": target_grid_rgbprobe_balance_resume200_from800_result.get("start_global_step"),
            "end_global_step": target_grid_rgbprobe_balance_resume200_from800_result.get("end_global_step"),
            "checkpoint": target_grid_rgbprobe_balance_resume200_from800_result.get("checkpoint"),
            "start_feature_target_loss": target_grid_rgbprobe_balance_resume200_from800_result.get(
                "start_feature_target_loss"
            ),
            "end_feature_target_loss": target_grid_rgbprobe_balance_resume200_from800_result.get(
                "end_feature_target_loss"
            ),
            "start_rgb_probe_loss": target_grid_rgbprobe_balance_resume200_from800_result.get(
                "start_rgb_probe_loss"
            ),
            "end_rgb_probe_loss": target_grid_rgbprobe_balance_resume200_from800_result.get("end_rgb_probe_loss"),
            "start_rgb_probe_psnr": target_grid_rgbprobe_balance_resume200_from800_result.get(
                "start_rgb_probe_psnr"
            ),
            "end_rgb_probe_psnr": target_grid_rgbprobe_balance_resume200_from800_result.get("end_rgb_probe_psnr"),
            "mean_timing_ms": target_grid_rgbprobe_balance_resume200_from800_result.get("mean_timing_ms"),
            "tile_overflow_sum": target_grid_rgbprobe_balance_resume200_from800_result.get("tile_overflow_sum"),
            "feature_target_weight_schedule": target_grid_rgbprobe_balance_resume200_from800_result.get(
                "feature_target_weight_schedule"
            ),
            "rgb_probe_media_render_ms": target_grid_rgbprobe_balance_resume200_from800_result.get(
                "rgb_probe_media_render_ms"
            ),
            "rgb_probe_contact_sheet": target_grid_rgbprobe_balance_resume200_from800_result.get(
                "rgb_probe_contact_sheet"
            ),
            "rgb_probe_side_by_side_video": target_grid_rgbprobe_balance_resume200_from800_result.get(
                "rgb_probe_side_by_side_video"
            ),
        },
        "target_grid_rgbprobe40_feature05_resume100_from1000_vjepa_target_config": {
            "path": TARGET_GRID_RGBPROBE40_FEATURE05_RESUME100_FROM1000_VJEPA_TARGET_CONFIG,
            "exists": target_grid_rgbprobe40_feature05_resume100_from1000_path.exists(),
            "resume_checkpoint": target_grid_rgbprobe40_feature05_resume100_from1000_config.get("train", {}).get(
                "resume_checkpoint"
            )
            if isinstance(target_grid_rgbprobe40_feature05_resume100_from1000_config.get("train"), dict)
            else None,
            "global_step_offset": target_grid_rgbprobe40_feature05_resume100_from1000_config.get("train", {}).get(
                "global_step_offset"
            )
            if isinstance(target_grid_rgbprobe40_feature05_resume100_from1000_config.get("train"), dict)
            else None,
            "checkpoint": target_grid_rgbprobe40_feature05_resume100_from1000_config.get("output", {}).get(
                "checkpoint"
            )
            if isinstance(target_grid_rgbprobe40_feature05_resume100_from1000_config.get("output"), dict)
            else None,
            "steps": target_grid_rgbprobe40_feature05_resume100_from1000_config.get("train", {}).get("steps")
            if isinstance(target_grid_rgbprobe40_feature05_resume100_from1000_config.get("train"), dict)
            else None,
            "feature_loss_weight": target_grid_rgbprobe40_feature05_resume100_from1000_config.get(
                "feature_target", {}
            ).get("loss_weight")
            if isinstance(target_grid_rgbprobe40_feature05_resume100_from1000_config.get("feature_target"), dict)
            else None,
            "rgb_probe_loss_weight": target_grid_rgbprobe40_feature05_resume100_from1000_config.get(
                "feature_target", {}
            ).get("rgb_probe_loss_weight")
            if isinstance(target_grid_rgbprobe40_feature05_resume100_from1000_config.get("feature_target"), dict)
            else None,
        },
        "target_grid_rgbprobe40_feature05_resume100_from1000_vjepa_target_result": None
        if target_grid_rgbprobe40_feature05_resume100_from1000_result is None
        else {
            "path": TARGET_GRID_RGBPROBE40_FEATURE05_RESUME100_FROM1000_VJEPA_TARGET_RESULT,
            "pass": target_grid_rgbprobe40_feature05_resume100_from1000_result.get("pass"),
            "resume_loaded": target_grid_rgbprobe40_feature05_resume100_from1000_result.get("resume_loaded"),
            "resume_optimizer_loaded": target_grid_rgbprobe40_feature05_resume100_from1000_result.get(
                "resume_optimizer_loaded"
            ),
            "resume_checkpoint_steps": target_grid_rgbprobe40_feature05_resume100_from1000_result.get(
                "resume_checkpoint_steps"
            ),
            "global_step_offset": target_grid_rgbprobe40_feature05_resume100_from1000_result.get(
                "global_step_offset"
            ),
            "start_global_step": target_grid_rgbprobe40_feature05_resume100_from1000_result.get(
                "start_global_step"
            ),
            "end_global_step": target_grid_rgbprobe40_feature05_resume100_from1000_result.get("end_global_step"),
            "checkpoint": target_grid_rgbprobe40_feature05_resume100_from1000_result.get("checkpoint"),
            "start_feature_target_loss": target_grid_rgbprobe40_feature05_resume100_from1000_result.get(
                "start_feature_target_loss"
            ),
            "end_feature_target_loss": target_grid_rgbprobe40_feature05_resume100_from1000_result.get(
                "end_feature_target_loss"
            ),
            "start_rgb_probe_loss": target_grid_rgbprobe40_feature05_resume100_from1000_result.get(
                "start_rgb_probe_loss"
            ),
            "end_rgb_probe_loss": target_grid_rgbprobe40_feature05_resume100_from1000_result.get(
                "end_rgb_probe_loss"
            ),
            "start_rgb_probe_psnr": target_grid_rgbprobe40_feature05_resume100_from1000_result.get(
                "start_rgb_probe_psnr"
            ),
            "end_rgb_probe_psnr": target_grid_rgbprobe40_feature05_resume100_from1000_result.get(
                "end_rgb_probe_psnr"
            ),
            "mean_timing_ms": target_grid_rgbprobe40_feature05_resume100_from1000_result.get("mean_timing_ms"),
            "tile_overflow_sum": target_grid_rgbprobe40_feature05_resume100_from1000_result.get(
                "tile_overflow_sum"
            ),
            "rgb_probe_media_render_ms": target_grid_rgbprobe40_feature05_resume100_from1000_result.get(
                "rgb_probe_media_render_ms"
            ),
            "rgb_probe_contact_sheet": target_grid_rgbprobe40_feature05_resume100_from1000_result.get(
                "rgb_probe_contact_sheet"
            ),
            "rgb_probe_side_by_side_video": target_grid_rgbprobe40_feature05_resume100_from1000_result.get(
                "rgb_probe_side_by_side_video"
            ),
        },
        "target_grid_rgbprobe_recover_resume100_from1100_vjepa_target_config": {
            "path": TARGET_GRID_RGBPROBE_RECOVER_RESUME100_FROM1100_VJEPA_TARGET_CONFIG,
            "exists": target_grid_rgbprobe_recover_resume100_from1100_path.exists(),
            "resume_checkpoint": target_grid_rgbprobe_recover_resume100_from1100_config.get("train", {}).get(
                "resume_checkpoint"
            )
            if isinstance(target_grid_rgbprobe_recover_resume100_from1100_config.get("train"), dict)
            else None,
            "global_step_offset": target_grid_rgbprobe_recover_resume100_from1100_config.get("train", {}).get(
                "global_step_offset"
            )
            if isinstance(target_grid_rgbprobe_recover_resume100_from1100_config.get("train"), dict)
            else None,
            "checkpoint": target_grid_rgbprobe_recover_resume100_from1100_config.get("output", {}).get(
                "checkpoint"
            )
            if isinstance(target_grid_rgbprobe_recover_resume100_from1100_config.get("output"), dict)
            else None,
            "steps": target_grid_rgbprobe_recover_resume100_from1100_config.get("train", {}).get("steps")
            if isinstance(target_grid_rgbprobe_recover_resume100_from1100_config.get("train"), dict)
            else None,
            "weight_schedule": target_grid_rgbprobe_recover_resume100_from1100_config.get("feature_target", {}).get(
                "weight_schedule"
            )
            if isinstance(target_grid_rgbprobe_recover_resume100_from1100_config.get("feature_target"), dict)
            else None,
        },
        "target_grid_rgbprobe_recover_resume100_from1100_vjepa_target_result": None
        if target_grid_rgbprobe_recover_resume100_from1100_result is None
        else {
            "path": TARGET_GRID_RGBPROBE_RECOVER_RESUME100_FROM1100_VJEPA_TARGET_RESULT,
            "pass": target_grid_rgbprobe_recover_resume100_from1100_result.get("pass"),
            "resume_loaded": target_grid_rgbprobe_recover_resume100_from1100_result.get("resume_loaded"),
            "resume_optimizer_loaded": target_grid_rgbprobe_recover_resume100_from1100_result.get(
                "resume_optimizer_loaded"
            ),
            "resume_checkpoint_steps": target_grid_rgbprobe_recover_resume100_from1100_result.get(
                "resume_checkpoint_steps"
            ),
            "global_step_offset": target_grid_rgbprobe_recover_resume100_from1100_result.get(
                "global_step_offset"
            ),
            "start_global_step": target_grid_rgbprobe_recover_resume100_from1100_result.get("start_global_step"),
            "end_global_step": target_grid_rgbprobe_recover_resume100_from1100_result.get("end_global_step"),
            "checkpoint": target_grid_rgbprobe_recover_resume100_from1100_result.get("checkpoint"),
            "start_feature_target_loss": target_grid_rgbprobe_recover_resume100_from1100_result.get(
                "start_feature_target_loss"
            ),
            "end_feature_target_loss": target_grid_rgbprobe_recover_resume100_from1100_result.get(
                "end_feature_target_loss"
            ),
            "start_rgb_probe_loss": target_grid_rgbprobe_recover_resume100_from1100_result.get(
                "start_rgb_probe_loss"
            ),
            "end_rgb_probe_loss": target_grid_rgbprobe_recover_resume100_from1100_result.get("end_rgb_probe_loss"),
            "start_rgb_probe_psnr": target_grid_rgbprobe_recover_resume100_from1100_result.get(
                "start_rgb_probe_psnr"
            ),
            "end_rgb_probe_psnr": target_grid_rgbprobe_recover_resume100_from1100_result.get("end_rgb_probe_psnr"),
            "mean_timing_ms": target_grid_rgbprobe_recover_resume100_from1100_result.get("mean_timing_ms"),
            "tile_overflow_sum": target_grid_rgbprobe_recover_resume100_from1100_result.get("tile_overflow_sum"),
            "feature_target_weight_schedule": target_grid_rgbprobe_recover_resume100_from1100_result.get(
                "feature_target_weight_schedule"
            ),
            "rgb_probe_media_render_ms": target_grid_rgbprobe_recover_resume100_from1100_result.get(
                "rgb_probe_media_render_ms"
            ),
            "rgb_probe_contact_sheet": target_grid_rgbprobe_recover_resume100_from1100_result.get(
                "rgb_probe_contact_sheet"
            ),
            "rgb_probe_side_by_side_video": target_grid_rgbprobe_recover_resume100_from1100_result.get(
                "rgb_probe_side_by_side_video"
            ),
        },
        "target_grid_rgbprobe40_feature075_resume50_from1200_vjepa_target_config": {
            "path": TARGET_GRID_RGBPROBE40_FEATURE075_RESUME50_FROM1200_VJEPA_TARGET_CONFIG,
            "exists": target_grid_rgbprobe40_feature075_resume50_from1200_path.exists(),
            "resume_checkpoint": target_grid_rgbprobe40_feature075_resume50_from1200_config.get("train", {}).get(
                "resume_checkpoint"
            )
            if isinstance(target_grid_rgbprobe40_feature075_resume50_from1200_config.get("train"), dict)
            else None,
            "global_step_offset": target_grid_rgbprobe40_feature075_resume50_from1200_config.get("train", {}).get(
                "global_step_offset"
            )
            if isinstance(target_grid_rgbprobe40_feature075_resume50_from1200_config.get("train"), dict)
            else None,
            "checkpoint": target_grid_rgbprobe40_feature075_resume50_from1200_config.get("output", {}).get(
                "checkpoint"
            )
            if isinstance(target_grid_rgbprobe40_feature075_resume50_from1200_config.get("output"), dict)
            else None,
            "steps": target_grid_rgbprobe40_feature075_resume50_from1200_config.get("train", {}).get("steps")
            if isinstance(target_grid_rgbprobe40_feature075_resume50_from1200_config.get("train"), dict)
            else None,
            "feature_loss_weight": target_grid_rgbprobe40_feature075_resume50_from1200_config.get(
                "feature_target", {}
            ).get("loss_weight")
            if isinstance(target_grid_rgbprobe40_feature075_resume50_from1200_config.get("feature_target"), dict)
            else None,
            "rgb_probe_loss_weight": target_grid_rgbprobe40_feature075_resume50_from1200_config.get(
                "feature_target", {}
            ).get("rgb_probe_loss_weight")
            if isinstance(target_grid_rgbprobe40_feature075_resume50_from1200_config.get("feature_target"), dict)
            else None,
        },
        "target_grid_rgbprobe40_feature075_resume50_from1200_vjepa_target_result": None
        if target_grid_rgbprobe40_feature075_resume50_from1200_result is None
        else {
            "path": TARGET_GRID_RGBPROBE40_FEATURE075_RESUME50_FROM1200_VJEPA_TARGET_RESULT,
            "pass": target_grid_rgbprobe40_feature075_resume50_from1200_result.get("pass"),
            "resume_loaded": target_grid_rgbprobe40_feature075_resume50_from1200_result.get("resume_loaded"),
            "resume_optimizer_loaded": target_grid_rgbprobe40_feature075_resume50_from1200_result.get(
                "resume_optimizer_loaded"
            ),
            "resume_checkpoint_steps": target_grid_rgbprobe40_feature075_resume50_from1200_result.get(
                "resume_checkpoint_steps"
            ),
            "global_step_offset": target_grid_rgbprobe40_feature075_resume50_from1200_result.get(
                "global_step_offset"
            ),
            "start_global_step": target_grid_rgbprobe40_feature075_resume50_from1200_result.get(
                "start_global_step"
            ),
            "end_global_step": target_grid_rgbprobe40_feature075_resume50_from1200_result.get("end_global_step"),
            "checkpoint": target_grid_rgbprobe40_feature075_resume50_from1200_result.get("checkpoint"),
            "start_feature_target_loss": target_grid_rgbprobe40_feature075_resume50_from1200_result.get(
                "start_feature_target_loss"
            ),
            "end_feature_target_loss": target_grid_rgbprobe40_feature075_resume50_from1200_result.get(
                "end_feature_target_loss"
            ),
            "start_rgb_probe_loss": target_grid_rgbprobe40_feature075_resume50_from1200_result.get(
                "start_rgb_probe_loss"
            ),
            "end_rgb_probe_loss": target_grid_rgbprobe40_feature075_resume50_from1200_result.get(
                "end_rgb_probe_loss"
            ),
            "start_rgb_probe_psnr": target_grid_rgbprobe40_feature075_resume50_from1200_result.get(
                "start_rgb_probe_psnr"
            ),
            "end_rgb_probe_psnr": target_grid_rgbprobe40_feature075_resume50_from1200_result.get(
                "end_rgb_probe_psnr"
            ),
            "mean_timing_ms": target_grid_rgbprobe40_feature075_resume50_from1200_result.get("mean_timing_ms"),
            "tile_overflow_sum": target_grid_rgbprobe40_feature075_resume50_from1200_result.get(
                "tile_overflow_sum"
            ),
            "rgb_probe_media_render_ms": target_grid_rgbprobe40_feature075_resume50_from1200_result.get(
                "rgb_probe_media_render_ms"
            ),
            "rgb_probe_contact_sheet": target_grid_rgbprobe40_feature075_resume50_from1200_result.get(
                "rgb_probe_contact_sheet"
            ),
            "rgb_probe_side_by_side_video": target_grid_rgbprobe40_feature075_resume50_from1200_result.get(
                "rgb_probe_side_by_side_video"
            ),
        },
        "target_grid_rgbprobe40_feature1_resume50_from1250_vjepa_target_config": {
            "path": TARGET_GRID_RGBPROBE40_FEATURE1_RESUME50_FROM1250_VJEPA_TARGET_CONFIG,
            "exists": target_grid_rgbprobe40_feature1_resume50_from1250_path.exists(),
            "resume_checkpoint": target_grid_rgbprobe40_feature1_resume50_from1250_config.get("train", {}).get(
                "resume_checkpoint"
            )
            if isinstance(target_grid_rgbprobe40_feature1_resume50_from1250_config.get("train"), dict)
            else None,
            "global_step_offset": target_grid_rgbprobe40_feature1_resume50_from1250_config.get("train", {}).get(
                "global_step_offset"
            )
            if isinstance(target_grid_rgbprobe40_feature1_resume50_from1250_config.get("train"), dict)
            else None,
            "checkpoint": target_grid_rgbprobe40_feature1_resume50_from1250_config.get("output", {}).get(
                "checkpoint"
            )
            if isinstance(target_grid_rgbprobe40_feature1_resume50_from1250_config.get("output"), dict)
            else None,
            "steps": target_grid_rgbprobe40_feature1_resume50_from1250_config.get("train", {}).get("steps")
            if isinstance(target_grid_rgbprobe40_feature1_resume50_from1250_config.get("train"), dict)
            else None,
            "feature_loss_weight": target_grid_rgbprobe40_feature1_resume50_from1250_config.get(
                "feature_target", {}
            ).get("loss_weight")
            if isinstance(target_grid_rgbprobe40_feature1_resume50_from1250_config.get("feature_target"), dict)
            else None,
            "rgb_probe_loss_weight": target_grid_rgbprobe40_feature1_resume50_from1250_config.get(
                "feature_target", {}
            ).get("rgb_probe_loss_weight")
            if isinstance(target_grid_rgbprobe40_feature1_resume50_from1250_config.get("feature_target"), dict)
            else None,
        },
        "target_grid_rgbprobe40_feature1_resume50_from1250_vjepa_target_result": None
        if target_grid_rgbprobe40_feature1_resume50_from1250_result is None
        else {
            "path": TARGET_GRID_RGBPROBE40_FEATURE1_RESUME50_FROM1250_VJEPA_TARGET_RESULT,
            "pass": target_grid_rgbprobe40_feature1_resume50_from1250_result.get("pass"),
            "resume_loaded": target_grid_rgbprobe40_feature1_resume50_from1250_result.get("resume_loaded"),
            "resume_optimizer_loaded": target_grid_rgbprobe40_feature1_resume50_from1250_result.get(
                "resume_optimizer_loaded"
            ),
            "resume_checkpoint_steps": target_grid_rgbprobe40_feature1_resume50_from1250_result.get(
                "resume_checkpoint_steps"
            ),
            "global_step_offset": target_grid_rgbprobe40_feature1_resume50_from1250_result.get(
                "global_step_offset"
            ),
            "start_global_step": target_grid_rgbprobe40_feature1_resume50_from1250_result.get(
                "start_global_step"
            ),
            "end_global_step": target_grid_rgbprobe40_feature1_resume50_from1250_result.get("end_global_step"),
            "checkpoint": target_grid_rgbprobe40_feature1_resume50_from1250_result.get("checkpoint"),
            "start_feature_target_loss": target_grid_rgbprobe40_feature1_resume50_from1250_result.get(
                "start_feature_target_loss"
            ),
            "end_feature_target_loss": target_grid_rgbprobe40_feature1_resume50_from1250_result.get(
                "end_feature_target_loss"
            ),
            "start_rgb_probe_loss": target_grid_rgbprobe40_feature1_resume50_from1250_result.get(
                "start_rgb_probe_loss"
            ),
            "end_rgb_probe_loss": target_grid_rgbprobe40_feature1_resume50_from1250_result.get(
                "end_rgb_probe_loss"
            ),
            "start_rgb_probe_psnr": target_grid_rgbprobe40_feature1_resume50_from1250_result.get(
                "start_rgb_probe_psnr"
            ),
            "end_rgb_probe_psnr": target_grid_rgbprobe40_feature1_resume50_from1250_result.get(
                "end_rgb_probe_psnr"
            ),
            "mean_timing_ms": target_grid_rgbprobe40_feature1_resume50_from1250_result.get("mean_timing_ms"),
            "tile_overflow_sum": target_grid_rgbprobe40_feature1_resume50_from1250_result.get(
                "tile_overflow_sum"
            ),
            "rgb_probe_media_render_ms": target_grid_rgbprobe40_feature1_resume50_from1250_result.get(
                "rgb_probe_media_render_ms"
            ),
            "rgb_probe_contact_sheet": target_grid_rgbprobe40_feature1_resume50_from1250_result.get(
                "rgb_probe_contact_sheet"
            ),
            "rgb_probe_side_by_side_video": target_grid_rgbprobe40_feature1_resume50_from1250_result.get(
                "rgb_probe_side_by_side_video"
            ),
        },
        "target_grid_rgbprobe40_feature1_resume100_from1300_vjepa_target_config": {
            "path": TARGET_GRID_RGBPROBE40_FEATURE1_RESUME100_FROM1300_VJEPA_TARGET_CONFIG,
            "exists": target_grid_rgbprobe40_feature1_resume100_from1300_path.exists(),
            "resume_checkpoint": target_grid_rgbprobe40_feature1_resume100_from1300_config.get("train", {}).get(
                "resume_checkpoint"
            )
            if isinstance(target_grid_rgbprobe40_feature1_resume100_from1300_config.get("train"), dict)
            else None,
            "global_step_offset": target_grid_rgbprobe40_feature1_resume100_from1300_config.get("train", {}).get(
                "global_step_offset"
            )
            if isinstance(target_grid_rgbprobe40_feature1_resume100_from1300_config.get("train"), dict)
            else None,
            "checkpoint": target_grid_rgbprobe40_feature1_resume100_from1300_config.get("output", {}).get(
                "checkpoint"
            )
            if isinstance(target_grid_rgbprobe40_feature1_resume100_from1300_config.get("output"), dict)
            else None,
            "steps": target_grid_rgbprobe40_feature1_resume100_from1300_config.get("train", {}).get("steps")
            if isinstance(target_grid_rgbprobe40_feature1_resume100_from1300_config.get("train"), dict)
            else None,
            "feature_loss_weight": target_grid_rgbprobe40_feature1_resume100_from1300_config.get(
                "feature_target", {}
            ).get("loss_weight")
            if isinstance(target_grid_rgbprobe40_feature1_resume100_from1300_config.get("feature_target"), dict)
            else None,
            "rgb_probe_loss_weight": target_grid_rgbprobe40_feature1_resume100_from1300_config.get(
                "feature_target", {}
            ).get("rgb_probe_loss_weight")
            if isinstance(target_grid_rgbprobe40_feature1_resume100_from1300_config.get("feature_target"), dict)
            else None,
        },
        "target_grid_rgbprobe40_feature1_resume100_from1300_vjepa_target_result": None
        if target_grid_rgbprobe40_feature1_resume100_from1300_result is None
        else {
            "path": TARGET_GRID_RGBPROBE40_FEATURE1_RESUME100_FROM1300_VJEPA_TARGET_RESULT,
            "pass": target_grid_rgbprobe40_feature1_resume100_from1300_result.get("pass"),
            "resume_loaded": target_grid_rgbprobe40_feature1_resume100_from1300_result.get("resume_loaded"),
            "resume_optimizer_loaded": target_grid_rgbprobe40_feature1_resume100_from1300_result.get(
                "resume_optimizer_loaded"
            ),
            "resume_checkpoint_steps": target_grid_rgbprobe40_feature1_resume100_from1300_result.get(
                "resume_checkpoint_steps"
            ),
            "global_step_offset": target_grid_rgbprobe40_feature1_resume100_from1300_result.get(
                "global_step_offset"
            ),
            "start_global_step": target_grid_rgbprobe40_feature1_resume100_from1300_result.get(
                "start_global_step"
            ),
            "end_global_step": target_grid_rgbprobe40_feature1_resume100_from1300_result.get("end_global_step"),
            "checkpoint": target_grid_rgbprobe40_feature1_resume100_from1300_result.get("checkpoint"),
            "start_feature_target_loss": target_grid_rgbprobe40_feature1_resume100_from1300_result.get(
                "start_feature_target_loss"
            ),
            "end_feature_target_loss": target_grid_rgbprobe40_feature1_resume100_from1300_result.get(
                "end_feature_target_loss"
            ),
            "start_rgb_probe_loss": target_grid_rgbprobe40_feature1_resume100_from1300_result.get(
                "start_rgb_probe_loss"
            ),
            "end_rgb_probe_loss": target_grid_rgbprobe40_feature1_resume100_from1300_result.get(
                "end_rgb_probe_loss"
            ),
            "start_rgb_probe_psnr": target_grid_rgbprobe40_feature1_resume100_from1300_result.get(
                "start_rgb_probe_psnr"
            ),
            "end_rgb_probe_psnr": target_grid_rgbprobe40_feature1_resume100_from1300_result.get(
                "end_rgb_probe_psnr"
            ),
            "mean_timing_ms": target_grid_rgbprobe40_feature1_resume100_from1300_result.get("mean_timing_ms"),
            "tile_overflow_sum": target_grid_rgbprobe40_feature1_resume100_from1300_result.get(
                "tile_overflow_sum"
            ),
            "rgb_probe_media_render_ms": target_grid_rgbprobe40_feature1_resume100_from1300_result.get(
                "rgb_probe_media_render_ms"
            ),
            "rgb_probe_contact_sheet": target_grid_rgbprobe40_feature1_resume100_from1300_result.get(
                "rgb_probe_contact_sheet"
            ),
            "rgb_probe_side_by_side_video": target_grid_rgbprobe40_feature1_resume100_from1300_result.get(
                "rgb_probe_side_by_side_video"
            ),
        },
        "conclusion": {
            "fastest_star_uvt_feature_route_uses_precomputed_vjepa": False,
            "precomputed_vjepa_exists_elsewhere": bool(precomputed_vjepa_configs),
            "star_cached_feature_target_adapter_exists": star_trainer_has_cached_target_adapter,
            "star_real_vjepa_feature_config_exists": bool(star_vjepa_configs),
            "star_scaled_vjepa_target_config_exists": bool(star_scaled_vjepa_target_configs),
            "star_scaled_vjepa_target_result_passes": bool(
                isinstance(scaled_vjepa_result, dict) and scaled_vjepa_result.get("pass")
            ),
            "star_cached_chunks_vjepa_target_result_passes": bool(
                isinstance(cached_chunks_result, dict) and cached_chunks_result.get("pass")
            ),
            "star_target_grid_vjepa_target_result_passes": bool(
                isinstance(target_grid_result, dict) and target_grid_result.get("pass")
            ),
            "star_target_grid_media_vjepa_target_result_passes": bool(
                isinstance(target_grid_media_result, dict) and target_grid_media_result.get("pass")
            ),
            "star_target_grid_rgb_aux_vjepa_target_result_passes": bool(
                isinstance(target_grid_rgb_aux_result, dict) and target_grid_rgb_aux_result.get("pass")
            ),
            "star_target_grid_rgb_aux10_vjepa_target_result_passes": bool(
                isinstance(target_grid_rgb_aux10_result, dict) and target_grid_rgb_aux10_result.get("pass")
            ),
            "star_target_grid_rgb_aux10_100step_vjepa_target_result_passes": bool(
                isinstance(target_grid_rgb_aux10_100step_result, dict) and target_grid_rgb_aux10_100step_result.get("pass")
            ),
            "star_target_grid_rgbwarm20_aux10_100step_vjepa_target_result_passes": bool(
                isinstance(target_grid_rgbwarm20_aux10_100step_result, dict)
                and target_grid_rgbwarm20_aux10_100step_result.get("pass")
            ),
            "star_target_grid_feature_rgb_probe_result_passes": bool(
                isinstance(target_grid_feature_rgb_probe_result, dict)
                and target_grid_feature_rgb_probe_result.get("pass")
            ),
            "star_target_grid_rgbprobe10_vjepa_target_result_passes": bool(
                isinstance(target_grid_rgbprobe10_result, dict) and target_grid_rgbprobe10_result.get("pass")
            ),
            "star_target_grid_rgbprobe10_100step_vjepa_target_result_passes": bool(
                isinstance(target_grid_rgbprobe10_100step_result, dict)
                and target_grid_rgbprobe10_100step_result.get("pass")
            ),
            "star_target_grid_rgbprobe10_300step_vjepa_target_result_passes": bool(
                isinstance(target_grid_rgbprobe10_300step_result, dict)
                and target_grid_rgbprobe10_300step_result.get("pass")
            ),
            "star_target_grid_rgbprobe10_300step_checkpoint_vjepa_target_result_passes": bool(
                isinstance(target_grid_rgbprobe10_300step_checkpoint_result, dict)
                and target_grid_rgbprobe10_300step_checkpoint_result.get("pass")
            ),
            "star_target_grid_rgbprobe10_resume300_from300_vjepa_target_result_passes": bool(
                isinstance(target_grid_rgbprobe10_resume300_from300_result, dict)
                and target_grid_rgbprobe10_resume300_from300_result.get("pass")
            ),
            "star_target_grid_rgbprobe40_feature025_resume200_from600_vjepa_target_result_passes": bool(
                isinstance(target_grid_rgbprobe40_feature025_resume200_from600_result, dict)
                and target_grid_rgbprobe40_feature025_resume200_from600_result.get("pass")
            ),
            "star_target_grid_rgbprobe_balance_resume200_from800_vjepa_target_result_passes": bool(
                isinstance(target_grid_rgbprobe_balance_resume200_from800_result, dict)
                and target_grid_rgbprobe_balance_resume200_from800_result.get("pass")
            ),
            "star_target_grid_rgbprobe40_feature05_resume100_from1000_vjepa_target_result_passes": bool(
                isinstance(target_grid_rgbprobe40_feature05_resume100_from1000_result, dict)
                and target_grid_rgbprobe40_feature05_resume100_from1000_result.get("pass")
            ),
            "star_target_grid_rgbprobe_recover_resume100_from1100_vjepa_target_result_passes": bool(
                isinstance(target_grid_rgbprobe_recover_resume100_from1100_result, dict)
                and target_grid_rgbprobe_recover_resume100_from1100_result.get("pass")
            ),
            "star_target_grid_rgbprobe40_feature075_resume50_from1200_vjepa_target_result_passes": bool(
                isinstance(target_grid_rgbprobe40_feature075_resume50_from1200_result, dict)
                and target_grid_rgbprobe40_feature075_resume50_from1200_result.get("pass")
            ),
            "star_target_grid_rgbprobe40_feature1_resume50_from1250_vjepa_target_result_passes": bool(
                isinstance(target_grid_rgbprobe40_feature1_resume50_from1250_result, dict)
                and target_grid_rgbprobe40_feature1_resume50_from1250_result.get("pass")
            ),
            "star_target_grid_rgbprobe40_feature1_resume100_from1300_vjepa_target_result_passes": bool(
                isinstance(target_grid_rgbprobe40_feature1_resume100_from1300_result, dict)
                and target_grid_rgbprobe40_feature1_resume100_from1300_result.get("pass")
            ),
            "missing_bridge": missing_bridge,
        },
        "next_contract": [
            "Keep the rgb_pyramid cached-target smoke as the cheap bridge regression.",
            "Keep the V-JEPA target smoke as the real cached-feature regression.",
            "Keep the chunked 64f/512px V-JEPA-target scale gate as the current STAR cached-feature scale regression.",
            "Compare the STAR V-JEPA-target route against the Gaussian/token precomputed V-JEPA rows.",
            "Use target_grid for short-run V-JEPA target speed and memory probes; the 20-step media row proves feature-loss overfit/media plumbing, not RGB quality.",
            "The RGB-aux1 target_grid probe trains the colorizer and decreases RGB loss, but the 20-step RGB PSNR gain is tiny.",
            "RGB-aux10 only marginally improves RGB PSNR over aux1 and slightly hurts feature loss; after the rgb-warm20 negative gate, the next visual probe should use a trained/frozen feature-to-RGB probe rather than simply skipping feature loss early.",
            "The 100-step aux10 run improves more clearly, so schedule length matters; still do not promote it as quality because it remains far below RGB STAR.",
            "The matched rgb-warm20 schedule is a negative visual-control gate: it is faster, but final RGB PSNR and feature loss are both worse than constant aux10 at the same step count.",
            "Use the passing hidden64 target-grid feature-to-RGB probe as the next STAR bridge: load/freeze the decoder for RGB-probe loss or canonical visual logging.",
            "Keep the first frozen-probe STAR target-grid row as the integration/speed proof; it is not a visual-quality promotion because 20-step probe PSNR barely moves.",
            "Use the 100-step frozen-probe STAR row as the stronger visual diagnostic; it moves probe PSNR more clearly but still needs a longer or scheduled objective before promotion.",
            "Use the 300+300 checkpoint/resume frozen-probe STAR row as evidence that the objective almost reaches the standalone full-video PSNR number.",
            "Use the probe-emphasis 600->800 continuation as evidence that visual probe PSNR can pass the standalone full-video number, but the feature-loss drift means the next objective must preserve V-JEPA target alignment while closing the same-grid oracle gap.",
            "Use the scheduled 800->1000 balance continuation as a negative/partial gate: it restores feature alignment but gives back probe PSNR, so simple two-stage alternation is not enough.",
            "Use the feature0.5/probe40 1000->1100 Pareto continuation as a passing combined-loss row: it raises probe PSNR to 21.79 but drifts feature loss back to 0.657.",
            "Use the 1100->1200 recover schedule as a nonpassing partial row: it recovers feature loss to 0.635 while giving back probe PSNR to 21.74.",
            "Use the short 1200->1250 feature0.75/probe40 continuation as a passing probe-recovery row: it restores probe PSNR to 21.93 while pushing feature loss back to 0.639.",
            "Use the 1250->1300 feature1/probe40 continuation as the current both-improving objective-balance row: feature loss drops to 0.632 and probe PSNR nudges to 21.96.",
            "Use the 1300->1400 feature1/probe40 continuation as the first extended balance row: both metrics keep improving to feature loss 0.627 and probe PSNR 21.98, but the run slows to 1.69s/step.",
            "Keep cached_chunks as the exact render-grid-loss reference when dense V-JEPA target loss is needed.",
            "Prototype native-VJP losses for dataset-scale feature objectives.",
        ],
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    selected = report["selected_fast_config"]
    contract = report["code_contract"]
    inventory = report["config_inventory"]
    conclusion = report["conclusion"]
    scaled = report["scaled_vjepa_target_config"]
    scaled_result = report["scaled_vjepa_target_result"]
    cached = report["cached_chunks_vjepa_target_config"]
    cached_result = report["cached_chunks_vjepa_target_result"]
    target_grid = report["target_grid_vjepa_target_config"]
    target_grid_result = report["target_grid_vjepa_target_result"]
    target_grid_media = report["target_grid_media_vjepa_target_config"]
    target_grid_media_result = report["target_grid_media_vjepa_target_result"]
    target_grid_rgb_aux = report["target_grid_rgb_aux_vjepa_target_config"]
    target_grid_rgb_aux_result = report["target_grid_rgb_aux_vjepa_target_result"]
    target_grid_rgb_aux10 = report["target_grid_rgb_aux10_vjepa_target_config"]
    target_grid_rgb_aux10_result = report["target_grid_rgb_aux10_vjepa_target_result"]
    target_grid_rgb_aux10_100step = report["target_grid_rgb_aux10_100step_vjepa_target_config"]
    target_grid_rgb_aux10_100step_result = report["target_grid_rgb_aux10_100step_vjepa_target_result"]
    target_grid_rgbwarm20_aux10_100step = report["target_grid_rgbwarm20_aux10_100step_vjepa_target_config"]
    target_grid_rgbwarm20_aux10_100step_result = report[
        "target_grid_rgbwarm20_aux10_100step_vjepa_target_result"
    ]
    target_grid_feature_rgb_probe = report["target_grid_feature_rgb_probe_config"]
    target_grid_feature_rgb_probe_result = report["target_grid_feature_rgb_probe_result"]
    target_grid_rgbprobe10 = report["target_grid_rgbprobe10_vjepa_target_config"]
    target_grid_rgbprobe10_result = report["target_grid_rgbprobe10_vjepa_target_result"]
    target_grid_rgbprobe10_100step = report["target_grid_rgbprobe10_100step_vjepa_target_config"]
    target_grid_rgbprobe10_100step_result = report["target_grid_rgbprobe10_100step_vjepa_target_result"]
    target_grid_rgbprobe10_300step = report["target_grid_rgbprobe10_300step_vjepa_target_config"]
    target_grid_rgbprobe10_300step_result = report["target_grid_rgbprobe10_300step_vjepa_target_result"]
    target_grid_rgbprobe10_300step_checkpoint = report[
        "target_grid_rgbprobe10_300step_checkpoint_vjepa_target_config"
    ]
    target_grid_rgbprobe10_300step_checkpoint_result = report[
        "target_grid_rgbprobe10_300step_checkpoint_vjepa_target_result"
    ]
    target_grid_rgbprobe10_resume300_from300 = report[
        "target_grid_rgbprobe10_resume300_from300_vjepa_target_config"
    ]
    target_grid_rgbprobe10_resume300_from300_result = report[
        "target_grid_rgbprobe10_resume300_from300_vjepa_target_result"
    ]
    target_grid_rgbprobe40_feature025_resume200_from600 = report[
        "target_grid_rgbprobe40_feature025_resume200_from600_vjepa_target_config"
    ]
    target_grid_rgbprobe40_feature025_resume200_from600_result = report[
        "target_grid_rgbprobe40_feature025_resume200_from600_vjepa_target_result"
    ]
    target_grid_rgbprobe_balance_resume200_from800 = report[
        "target_grid_rgbprobe_balance_resume200_from800_vjepa_target_config"
    ]
    target_grid_rgbprobe_balance_resume200_from800_result = report[
        "target_grid_rgbprobe_balance_resume200_from800_vjepa_target_result"
    ]
    target_grid_rgbprobe40_feature05_resume100_from1000 = report[
        "target_grid_rgbprobe40_feature05_resume100_from1000_vjepa_target_config"
    ]
    target_grid_rgbprobe40_feature05_resume100_from1000_result = report[
        "target_grid_rgbprobe40_feature05_resume100_from1000_vjepa_target_result"
    ]
    target_grid_rgbprobe_recover_resume100_from1100 = report[
        "target_grid_rgbprobe_recover_resume100_from1100_vjepa_target_config"
    ]
    target_grid_rgbprobe_recover_resume100_from1100_result = report[
        "target_grid_rgbprobe_recover_resume100_from1100_vjepa_target_result"
    ]
    target_grid_rgbprobe40_feature075_resume50_from1200 = report[
        "target_grid_rgbprobe40_feature075_resume50_from1200_vjepa_target_config"
    ]
    target_grid_rgbprobe40_feature075_resume50_from1200_result = report[
        "target_grid_rgbprobe40_feature075_resume50_from1200_vjepa_target_result"
    ]
    target_grid_rgbprobe40_feature1_resume50_from1250 = report[
        "target_grid_rgbprobe40_feature1_resume50_from1250_vjepa_target_config"
    ]
    target_grid_rgbprobe40_feature1_resume50_from1250_result = report[
        "target_grid_rgbprobe40_feature1_resume50_from1250_vjepa_target_result"
    ]
    target_grid_rgbprobe40_feature1_resume100_from1300 = report[
        "target_grid_rgbprobe40_feature1_resume100_from1300_vjepa_target_config"
    ]
    target_grid_rgbprobe40_feature1_resume100_from1300_result = report[
        "target_grid_rgbprobe40_feature1_resume100_from1300_vjepa_target_result"
    ]
    lines = [
        "# STAR UVT Precomputed V-JEPA Bridge Audit",
        "",
        "## Answer",
        "",
        "No. The current fastest STAR UVT feature-tube route does not use "
        "precomputed V-JEPA targets. It is a first-class STAR feature overfit "
        "trainer that renders F32 tube features, decodes them with "
        "`FeatureToColor`, and optimizes RGB reconstruction.",
        "",
        "There is now a separate STAR UVT real-V-JEPA target route. It is not the "
        "selected `star-feature-512-fast` route, but it does pass a chunked "
        "64f/512px scale gate and should be treated as the cached-feature scale "
        "regression.",
        "",
        "The cached target grid itself is not the visual blocker: the standalone "
        "hidden64 FeatureToColor probe now decodes it to `23.4 dB` grid PSNR and "
        "`20.1 dB` full-video upsampled PSNR. The missing bridge is using that "
        "trained/frozen decoder inside STAR training or canonical probe logging. "
        "That bridge now exists and passes 20-step, 100-step, 300-step, and "
        "checkpointed 300+300 gates. The 20-step row is integration proof because "
        "probe PSNR barely moves; the 100-step row reaches `14.64 dB` at "
        "`1.27s/step`; the 300-step row reaches `16.56 dB`; the resumed 300-step "
        "continuation reaches `19.88 dB` at `1.44s/step`; and the probe-emphasis "
        "600->800 continuation reaches `21.42 dB` at `1.51s/step`. That last row "
        "passes the standalone full-video upsample number, but feature loss drifts "
        "upward. The scheduled 800->1000 balance row recovers feature loss but gives "
        "back a little probe quality. The feature0.5/probe40 1000->1100 row passes "
        "the combined gate and reaches `21.79 dB` at `1.46s/step`, but feature "
        "loss drifts back to `0.657`. The 1100->1200 recover schedule lowers "
        "feature loss to `0.635`, but probe PSNR slips to `21.74 dB`. A short "
        "feature0.75/probe40 1200->1250 continuation restores probe PSNR to "
        "`21.93 dB`, but feature loss rises to `0.639`. The feature1/probe40 "
        "1250->1300 continuation improves both signals, reaching `21.96 dB` "
        "probe PSNR and `0.632` feature loss at `1.28s/step`. The 1300->1400 "
        "extension keeps both signals moving, reaching `21.98 dB` probe PSNR "
        "and `0.627` feature loss, but slows to `1.69s/step`, so the remaining gate is objective balance "
        "against the same-grid "
        "`23.4 dB` oracle.",
        "",
        "## Selected Fast STAR Feature Config",
        "",
        f"- config: `{selected['path']}`",
        f"- arch: `{selected['arch']}`",
        f"- render mode: `{selected['render_mode']}`",
        f"- target: `{selected['frames']}f/{selected['target_size']}px/{selected['tube_count']}t/F{selected['feature_dim']}`",
        f"- colorize pre-norm: `{selected['colorize_pre_norm']}`",
        f"- has `features` section: `{selected['has_features_section']}`",
        f"- uses V-JEPA/precomputed feature config: `{selected['uses_vjepa_or_precomputed_features']}`",
        "",
        "## Code Contract",
        "",
        f"- dispatcher has `star_uvt_feature_overfit`: `{contract['dispatcher_has_star_uvt_feature_route']}`",
        f"- STAR feature trainer uses RGB video targets: `{contract['star_feature_trainer_uses_rgb_video_target']}`",
        f"- STAR feature trainer uses `VideoFeatureCache`: `{contract['star_feature_trainer_uses_video_feature_cache']}`",
        f"- STAR feature trainer has cached-target adapter: `{contract['star_feature_trainer_has_cached_target_adapter']}`",
        f"- trainer path: `{contract['star_feature_trainer_path']}`",
        "",
        "## Config Inventory",
        "",
        f"- STAR UVT feature configs scanned: `{inventory['star_uvt_feature_config_count']}`",
        f"- STAR UVT feature configs with V-JEPA/precomputed sections: `{len(inventory['star_uvt_feature_configs_with_vjepa_or_precomputed_features'])}`",
        f"- STAR UVT feature configs with cached-target smoke enabled: `{len(inventory['star_uvt_feature_configs_with_cached_target'])}`",
        f"- STAR UVT feature 64f/512px V-JEPA target configs: `{len(inventory['star_uvt_feature_scaled_vjepa_target_configs'])}`",
        f"- precomputed V-JEPA Gaussian/token configs found: `{inventory['precomputed_vjepa_config_count']}`",
        "",
        "STAR UVT V-JEPA target configs:",
        "",
    ]
    for row in inventory["star_uvt_feature_configs_with_vjepa_or_precomputed_features"]:
        lines.append(f"- `{row['path']}` -> `{row['feature_extractor']}` / `{row['sample_cache_key']}`")
    lines.extend(
        [
            "",
            "Example precomputed V-JEPA configs:",
            "",
        ]
    )
    for row in inventory["precomputed_vjepa_config_examples"]:
        lines.append(f"- `{row['path']}` -> `{row['feature_extractor']}` / `{row['sample_cache_key']}`")
    lines.extend(
        [
            "",
            "## 512px STAR V-JEPA Target Gate",
            "",
            f"- config: `{scaled['path']}`",
            f"- exists: `{scaled['exists']}`",
            f"- target: `{scaled['frames']}f/{scaled['target_size']}px/{scaled['tube_count']}t`",
            f"- render mode: `{scaled['render_mode']}`",
            f"- feature target enabled: `{scaled['feature_target_enabled']}`",
            f"- feature target materialization: `{scaled['feature_target_materialization']}`",
            f"- result: `{None if scaled_result is None else scaled_result['path']}`",
            f"- pass: `{None if scaled_result is None else scaled_result['pass']}`",
        ]
    )
    if scaled_result is not None:
        timing = scaled_result["mean_timing_ms"]
        target_meta = scaled_result["feature_target"]
        lines.extend(
            [
                f"- loss: `{scaled_result['start_loss']:.6f} -> {scaled_result['end_loss']:.6f}`",
                f"- mean step/render/target/backward: `{timing['step_ms']:.1f}ms / {timing['render_forward_ms']:.1f}ms / {timing.get('feature_target_ms', 0.0):.1f}ms / {timing['backward_ms']:.1f}ms`",
                f"- tile overflow: `{scaled_result['tile_overflow_sum']}`",
                f"- V-JEPA source/adapted shape: `{target_meta['source_shape']} -> {target_meta['adapted_shape']}`",
                f"- channel-before-grid adapter: `{target_meta.get('channel_adapter_applied_before_grid')}`",
                f"- materialization: `{target_meta.get('materialization')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Cached-Chunks Target Follow-Up",
            "",
            f"- config: `{cached['path']}`",
            f"- exists: `{cached['exists']}`",
            f"- feature target materialization: `{cached['feature_target_materialization']}`",
            f"- result: `{None if cached_result is None else cached_result['path']}`",
            f"- pass: `{None if cached_result is None else cached_result['pass']}`",
        ]
    )
    if cached_result is not None:
        timing = cached_result["mean_timing_ms"]
        target_meta = cached_result["feature_target"]
        lines.extend(
            [
                f"- loss: `{cached_result['start_loss']:.6f} -> {cached_result['end_loss']:.6f}`",
                f"- target load/prep: `{cached_result.get('feature_target_load_ms', 0.0):.1f}ms`",
                f"- mean step/render/target/backward: `{timing['step_ms']:.1f}ms / {timing['render_forward_ms']:.1f}ms / {timing.get('feature_target_ms', 0.0):.1f}ms / {timing['backward_ms']:.1f}ms`",
                f"- cached chunks / MiB: `{target_meta.get('cached_chunk_count')}` / `{target_meta.get('cached_target_mib')}`",
                f"- tile overflow: `{cached_result['tile_overflow_sum']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Target-Grid Loss Follow-Up",
            "",
            f"- config: `{target_grid['path']}`",
            f"- exists: `{target_grid['exists']}`",
            f"- feature target materialization: `{target_grid['feature_target_materialization']}`",
            f"- result: `{None if target_grid_result is None else target_grid_result['path']}`",
            f"- pass: `{None if target_grid_result is None else target_grid_result['pass']}`",
        ]
    )
    if target_grid_result is not None:
        timing = target_grid_result["mean_timing_ms"]
        target_meta = target_grid_result["feature_target"]
        lines.extend(
            [
                f"- loss: `{target_grid_result['start_loss']:.6f} -> {target_grid_result['end_loss']:.6f}`",
                f"- target load/prep: `{target_grid_result.get('feature_target_load_ms', 0.0):.1f}ms`",
                f"- mean step/render/target/backward: `{timing['step_ms']:.1f}ms / {timing['render_forward_ms']:.1f}ms / {timing.get('feature_target_ms', 0.0):.1f}ms / {timing['backward_ms']:.1f}ms`",
                f"- target grid shape / MiB: `{target_meta.get('target_grid_shape')}` / `{target_meta.get('target_grid_mib')}`",
                f"- tile overflow: `{target_grid_result['tile_overflow_sum']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Target-Grid 20-Step Media Follow-Up",
            "",
            f"- config: `{target_grid_media['path']}`",
            f"- exists: `{target_grid_media['exists']}`",
            f"- feature target materialization: `{target_grid_media['feature_target_materialization']}`",
            f"- rgb loss weight: `{target_grid_media['rgb_loss_weight']}`",
            f"- result: `{None if target_grid_media_result is None else target_grid_media_result['path']}`",
            f"- pass: `{None if target_grid_media_result is None else target_grid_media_result['pass']}`",
        ]
    )
    if target_grid_media_result is not None:
        timing = target_grid_media_result["mean_timing_ms"]
        target_meta = target_grid_media_result["feature_target"]
        lines.extend(
            [
                f"- loss: `{target_grid_media_result['start_loss']:.6f} -> {target_grid_media_result['end_loss']:.6f}`",
                f"- PSNR: `{target_grid_media_result['start_psnr']:.6f} -> {target_grid_media_result['end_psnr']:.6f}` (not RGB quality evidence because RGB loss is disabled)",
                f"- target load/prep: `{target_grid_media_result.get('feature_target_load_ms', 0.0):.1f}ms`",
                f"- mean step/render/target/backward: `{timing['step_ms']:.1f}ms / {timing['render_forward_ms']:.1f}ms / {timing.get('feature_target_ms', 0.0):.1f}ms / {timing['backward_ms']:.1f}ms`",
                f"- target grid shape / MiB: `{target_meta.get('target_grid_shape')}` / `{target_meta.get('target_grid_mib')}`",
                f"- media: `{target_grid_media_result.get('contact_sheet')}`, `{target_grid_media_result.get('side_by_side_video')}`",
                f"- tile overflow: `{target_grid_media_result['tile_overflow_sum']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Target-Grid RGB-Aux 20-Step Probe",
            "",
            f"- config: `{target_grid_rgb_aux['path']}`",
            f"- exists: `{target_grid_rgb_aux['exists']}`",
            f"- feature target materialization: `{target_grid_rgb_aux['feature_target_materialization']}`",
            f"- rgb loss weight: `{target_grid_rgb_aux['rgb_loss_weight']}`",
            f"- result: `{None if target_grid_rgb_aux_result is None else target_grid_rgb_aux_result['path']}`",
            f"- pass: `{None if target_grid_rgb_aux_result is None else target_grid_rgb_aux_result['pass']}`",
        ]
    )
    if target_grid_rgb_aux_result is not None:
        timing = target_grid_rgb_aux_result["mean_timing_ms"]
        target_meta = target_grid_rgb_aux_result["feature_target"]
        lines.extend(
            [
                f"- total loss: `{target_grid_rgb_aux_result['start_loss']:.6f} -> {target_grid_rgb_aux_result['end_loss']:.6f}`",
                f"- feature loss: `{target_grid_rgb_aux_result['start_feature_target_loss']:.6f} -> {target_grid_rgb_aux_result['end_feature_target_loss']:.6f}`",
                f"- RGB loss / PSNR: `{target_grid_rgb_aux_result['start_rgb_loss']:.6f} -> {target_grid_rgb_aux_result['end_rgb_loss']:.6f}` / `{target_grid_rgb_aux_result['start_rgb_psnr']:.3f} -> {target_grid_rgb_aux_result['end_rgb_psnr']:.3f}`",
                f"- colorizer grad seen: `{target_grid_rgb_aux_result['colorizer_grad_seen']}`",
                f"- target load/prep: `{target_grid_rgb_aux_result.get('feature_target_load_ms', 0.0):.1f}ms`",
                f"- mean step/render/target/backward: `{timing['step_ms']:.1f}ms / {timing['render_forward_ms']:.1f}ms / {timing.get('feature_target_ms', 0.0):.1f}ms / {timing['backward_ms']:.1f}ms`",
                f"- target grid shape / MiB: `{target_meta.get('target_grid_shape')}` / `{target_meta.get('target_grid_mib')}`",
                f"- media: `{target_grid_rgb_aux_result.get('contact_sheet')}`, `{target_grid_rgb_aux_result.get('side_by_side_video')}`",
                f"- tile overflow: `{target_grid_rgb_aux_result['tile_overflow_sum']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Target-Grid RGB-Aux10 20-Step Probe",
            "",
            f"- config: `{target_grid_rgb_aux10['path']}`",
            f"- exists: `{target_grid_rgb_aux10['exists']}`",
            f"- feature target materialization: `{target_grid_rgb_aux10['feature_target_materialization']}`",
            f"- rgb loss weight: `{target_grid_rgb_aux10['rgb_loss_weight']}`",
            f"- result: `{None if target_grid_rgb_aux10_result is None else target_grid_rgb_aux10_result['path']}`",
            f"- pass: `{None if target_grid_rgb_aux10_result is None else target_grid_rgb_aux10_result['pass']}`",
        ]
    )
    if target_grid_rgb_aux10_result is not None:
        timing = target_grid_rgb_aux10_result["mean_timing_ms"]
        target_meta = target_grid_rgb_aux10_result["feature_target"]
        lines.extend(
            [
                f"- total loss: `{target_grid_rgb_aux10_result['start_loss']:.6f} -> {target_grid_rgb_aux10_result['end_loss']:.6f}`",
                f"- feature loss: `{target_grid_rgb_aux10_result['start_feature_target_loss']:.6f} -> {target_grid_rgb_aux10_result['end_feature_target_loss']:.6f}`",
                f"- RGB loss / PSNR: `{target_grid_rgb_aux10_result['start_rgb_loss']:.6f} -> {target_grid_rgb_aux10_result['end_rgb_loss']:.6f}` / `{target_grid_rgb_aux10_result['start_rgb_psnr']:.3f} -> {target_grid_rgb_aux10_result['end_rgb_psnr']:.3f}`",
                f"- colorizer grad seen: `{target_grid_rgb_aux10_result['colorizer_grad_seen']}`",
                f"- target load/prep: `{target_grid_rgb_aux10_result.get('feature_target_load_ms', 0.0):.1f}ms`",
                f"- mean step/render/target/backward: `{timing['step_ms']:.1f}ms / {timing['render_forward_ms']:.1f}ms / {timing.get('feature_target_ms', 0.0):.1f}ms / {timing['backward_ms']:.1f}ms`",
                f"- target grid shape / MiB: `{target_meta.get('target_grid_shape')}` / `{target_meta.get('target_grid_mib')}`",
                f"- media: `{target_grid_rgb_aux10_result.get('contact_sheet')}`, `{target_grid_rgb_aux10_result.get('side_by_side_video')}`",
                f"- tile overflow: `{target_grid_rgb_aux10_result['tile_overflow_sum']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Target-Grid RGB-Aux10 100-Step Probe",
            "",
            f"- config: `{target_grid_rgb_aux10_100step['path']}`",
            f"- exists: `{target_grid_rgb_aux10_100step['exists']}`",
            f"- feature target materialization: `{target_grid_rgb_aux10_100step['feature_target_materialization']}`",
            f"- rgb loss weight: `{target_grid_rgb_aux10_100step['rgb_loss_weight']}`",
            f"- result: `{None if target_grid_rgb_aux10_100step_result is None else target_grid_rgb_aux10_100step_result['path']}`",
            f"- pass: `{None if target_grid_rgb_aux10_100step_result is None else target_grid_rgb_aux10_100step_result['pass']}`",
        ]
    )
    if target_grid_rgb_aux10_100step_result is not None:
        timing = target_grid_rgb_aux10_100step_result["mean_timing_ms"]
        target_meta = target_grid_rgb_aux10_100step_result["feature_target"]
        lines.extend(
            [
                f"- total loss: `{target_grid_rgb_aux10_100step_result['start_loss']:.6f} -> {target_grid_rgb_aux10_100step_result['end_loss']:.6f}`",
                f"- feature loss: `{target_grid_rgb_aux10_100step_result['start_feature_target_loss']:.6f} -> {target_grid_rgb_aux10_100step_result['end_feature_target_loss']:.6f}`",
                f"- RGB loss / PSNR: `{target_grid_rgb_aux10_100step_result['start_rgb_loss']:.6f} -> {target_grid_rgb_aux10_100step_result['end_rgb_loss']:.6f}` / `{target_grid_rgb_aux10_100step_result['start_rgb_psnr']:.3f} -> {target_grid_rgb_aux10_100step_result['end_rgb_psnr']:.3f}`",
                f"- colorizer grad seen: `{target_grid_rgb_aux10_100step_result['colorizer_grad_seen']}`",
                f"- target load/prep: `{target_grid_rgb_aux10_100step_result.get('feature_target_load_ms', 0.0):.1f}ms`",
                f"- mean step/render/target/backward: `{timing['step_ms']:.1f}ms / {timing['render_forward_ms']:.1f}ms / {timing.get('feature_target_ms', 0.0):.1f}ms / {timing['backward_ms']:.1f}ms`",
                f"- target grid shape / MiB: `{target_meta.get('target_grid_shape')}` / `{target_meta.get('target_grid_mib')}`",
                f"- media: `{target_grid_rgb_aux10_100step_result.get('contact_sheet')}`, `{target_grid_rgb_aux10_100step_result.get('side_by_side_video')}`",
                f"- tile overflow: `{target_grid_rgb_aux10_100step_result['tile_overflow_sum']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Target-Grid RGB-Warm20 Aux10 100-Step Probe",
            "",
            f"- config: `{target_grid_rgbwarm20_aux10_100step['path']}`",
            f"- exists: `{target_grid_rgbwarm20_aux10_100step['exists']}`",
            f"- feature target materialization: `{target_grid_rgbwarm20_aux10_100step['feature_target_materialization']}`",
            f"- rgb loss weight: `{target_grid_rgbwarm20_aux10_100step['rgb_loss_weight']}`",
            f"- schedule: `{target_grid_rgbwarm20_aux10_100step['weight_schedule']}`",
            f"- result: `{None if target_grid_rgbwarm20_aux10_100step_result is None else target_grid_rgbwarm20_aux10_100step_result['path']}`",
            f"- pass: `{None if target_grid_rgbwarm20_aux10_100step_result is None else target_grid_rgbwarm20_aux10_100step_result['pass']}`",
        ]
    )
    if target_grid_rgbwarm20_aux10_100step_result is not None:
        timing = target_grid_rgbwarm20_aux10_100step_result["mean_timing_ms"]
        target_meta = target_grid_rgbwarm20_aux10_100step_result["feature_target"]
        lines.extend(
            [
                f"- total loss: `{target_grid_rgbwarm20_aux10_100step_result['start_loss']:.6f} -> {target_grid_rgbwarm20_aux10_100step_result['end_loss']:.6f}`",
                f"- feature loss: `{target_grid_rgbwarm20_aux10_100step_result['start_feature_target_loss']:.6f} -> {target_grid_rgbwarm20_aux10_100step_result['end_feature_target_loss']:.6f}`",
                f"- RGB loss / PSNR: `{target_grid_rgbwarm20_aux10_100step_result['start_rgb_loss']:.6f} -> {target_grid_rgbwarm20_aux10_100step_result['end_rgb_loss']:.6f}` / `{target_grid_rgbwarm20_aux10_100step_result['start_rgb_psnr']:.3f} -> {target_grid_rgbwarm20_aux10_100step_result['end_rgb_psnr']:.3f}`",
                f"- colorizer grad seen: `{target_grid_rgbwarm20_aux10_100step_result['colorizer_grad_seen']}`",
                f"- target load/prep: `{target_grid_rgbwarm20_aux10_100step_result.get('feature_target_load_ms', 0.0):.1f}ms`",
                f"- mean step/render/target/backward: `{timing['step_ms']:.1f}ms / {timing['render_forward_ms']:.1f}ms / {timing.get('feature_target_ms', 0.0):.1f}ms / {timing['backward_ms']:.1f}ms`",
                f"- target grid shape / MiB: `{target_meta.get('target_grid_shape')}` / `{target_meta.get('target_grid_mib')}`",
                f"- recorded schedule: `{target_grid_rgbwarm20_aux10_100step_result.get('feature_target_weight_schedule')}`",
                f"- media: `{target_grid_rgbwarm20_aux10_100step_result.get('contact_sheet')}`, `{target_grid_rgbwarm20_aux10_100step_result.get('side_by_side_video')}`",
                f"- tile overflow: `{target_grid_rgbwarm20_aux10_100step_result['tile_overflow_sum']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Target-Grid Feature-to-RGB Probe",
            "",
            f"- config: `{target_grid_feature_rgb_probe['path']}`",
            f"- exists: `{target_grid_feature_rgb_probe['exists']}`",
            f"- arch: `{target_grid_feature_rgb_probe['arch']}`",
            f"- target: `{target_grid_feature_rgb_probe['frames']}f/{target_grid_feature_rgb_probe['target_size']}px/F{target_grid_feature_rgb_probe['feature_dim']}`",
            f"- hidden dim / steps: `{target_grid_feature_rgb_probe['hidden_dim']}` / `{target_grid_feature_rgb_probe['steps']}`",
            f"- feature target materialization: `{target_grid_feature_rgb_probe['feature_target_materialization']}`",
            f"- result: `{None if target_grid_feature_rgb_probe_result is None else target_grid_feature_rgb_probe_result['path']}`",
            f"- pass: `{None if target_grid_feature_rgb_probe_result is None else target_grid_feature_rgb_probe_result['pass']}`",
        ]
    )
    if target_grid_feature_rgb_probe_result is not None:
        timing = target_grid_feature_rgb_probe_result["mean_timing_ms"]
        lines.extend(
            [
                f"- grid loss / PSNR: `{target_grid_feature_rgb_probe_result['start_grid_loss']:.6f} -> {target_grid_feature_rgb_probe_result['final_grid_loss']:.6f}` / `{target_grid_feature_rgb_probe_result['start_grid_psnr']:.3f} -> {target_grid_feature_rgb_probe_result['final_grid_psnr']:.3f}`",
                f"- full upsampled PSNR: `{target_grid_feature_rgb_probe_result['final_full_psnr']:.3f}`",
                f"- target grid / RGB grid: `{target_grid_feature_rgb_probe_result['target_grid_shape']}` / `{target_grid_feature_rgb_probe_result['target_grid_rgb_shape']}`",
                f"- target load/prep: `{target_grid_feature_rgb_probe_result.get('feature_target_load_ms', 0.0):.1f}ms`",
                f"- mean step/forward/backward/optimizer: `{timing['step_ms']:.3f}ms / {timing['forward_loss_ms']:.3f}ms / {timing['backward_ms']:.3f}ms / {timing['optimizer_ms']:.3f}ms`",
                f"- checkpoint: `{target_grid_feature_rgb_probe_result.get('checkpoint')}`",
                f"- media: `{target_grid_feature_rgb_probe_result.get('contact_sheet')}`, `{target_grid_feature_rgb_probe_result.get('side_by_side_video')}`",
                f"- W&B: `{target_grid_feature_rgb_probe_result.get('wandb_run_id')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Target-Grid Frozen RGB-Probe STAR Gate",
            "",
            f"- config: `{target_grid_rgbprobe10['path']}`",
            f"- exists: `{target_grid_rgbprobe10['exists']}`",
            f"- feature target materialization: `{target_grid_rgbprobe10['feature_target_materialization']}`",
            f"- rgb probe checkpoint: `{target_grid_rgbprobe10['rgb_probe_checkpoint']}`",
            f"- rgb probe loss weight: `{target_grid_rgbprobe10['rgb_probe_loss_weight']}`",
            f"- result: `{None if target_grid_rgbprobe10_result is None else target_grid_rgbprobe10_result['path']}`",
            f"- pass: `{None if target_grid_rgbprobe10_result is None else target_grid_rgbprobe10_result['pass']}`",
            f"- 100-step config: `{target_grid_rgbprobe10_100step['path']}`",
            f"- 100-step exists: `{target_grid_rgbprobe10_100step['exists']}`",
            f"- 100-step rgb probe checkpoint: `{target_grid_rgbprobe10_100step['rgb_probe_checkpoint']}`",
            f"- 100-step rgb probe loss weight: `{target_grid_rgbprobe10_100step['rgb_probe_loss_weight']}`",
            f"- 100-step result: `{None if target_grid_rgbprobe10_100step_result is None else target_grid_rgbprobe10_100step_result['path']}`",
            f"- 100-step pass: `{None if target_grid_rgbprobe10_100step_result is None else target_grid_rgbprobe10_100step_result['pass']}`",
            f"- 300-step config: `{target_grid_rgbprobe10_300step['path']}`",
            f"- 300-step exists: `{target_grid_rgbprobe10_300step['exists']}`",
            f"- 300-step rgb probe checkpoint: `{target_grid_rgbprobe10_300step['rgb_probe_checkpoint']}`",
            f"- 300-step rgb probe loss weight: `{target_grid_rgbprobe10_300step['rgb_probe_loss_weight']}`",
            f"- 300-step result: `{None if target_grid_rgbprobe10_300step_result is None else target_grid_rgbprobe10_300step_result['path']}`",
            f"- 300-step pass: `{None if target_grid_rgbprobe10_300step_result is None else target_grid_rgbprobe10_300step_result['pass']}`",
            f"- 300-step checkpoint config: `{target_grid_rgbprobe10_300step_checkpoint['path']}`",
            f"- 300-step checkpoint result: `{None if target_grid_rgbprobe10_300step_checkpoint_result is None else target_grid_rgbprobe10_300step_checkpoint_result['path']}`",
            f"- 300-step checkpoint pass: `{None if target_grid_rgbprobe10_300step_checkpoint_result is None else target_grid_rgbprobe10_300step_checkpoint_result['pass']}`",
            f"- resume300-from300 config: `{target_grid_rgbprobe10_resume300_from300['path']}`",
            f"- resume300-from300 result: `{None if target_grid_rgbprobe10_resume300_from300_result is None else target_grid_rgbprobe10_resume300_from300_result['path']}`",
            f"- resume300-from300 pass: `{None if target_grid_rgbprobe10_resume300_from300_result is None else target_grid_rgbprobe10_resume300_from300_result['pass']}`",
            f"- probe-emphasis resume200-from600 config: `{target_grid_rgbprobe40_feature025_resume200_from600['path']}`",
            f"- probe-emphasis resume200-from600 result: `{None if target_grid_rgbprobe40_feature025_resume200_from600_result is None else target_grid_rgbprobe40_feature025_resume200_from600_result['path']}`",
            f"- probe-emphasis resume200-from600 pass: `{None if target_grid_rgbprobe40_feature025_resume200_from600_result is None else target_grid_rgbprobe40_feature025_resume200_from600_result['pass']}`",
            f"- scheduled balance resume200-from800 config: `{target_grid_rgbprobe_balance_resume200_from800['path']}`",
            f"- scheduled balance resume200-from800 result: `{None if target_grid_rgbprobe_balance_resume200_from800_result is None else target_grid_rgbprobe_balance_resume200_from800_result['path']}`",
            f"- scheduled balance resume200-from800 pass: `{None if target_grid_rgbprobe_balance_resume200_from800_result is None else target_grid_rgbprobe_balance_resume200_from800_result['pass']}`",
            f"- feature0.5/probe40 resume100-from1000 config: `{target_grid_rgbprobe40_feature05_resume100_from1000['path']}`",
            f"- feature0.5/probe40 resume100-from1000 result: `{None if target_grid_rgbprobe40_feature05_resume100_from1000_result is None else target_grid_rgbprobe40_feature05_resume100_from1000_result['path']}`",
            f"- feature0.5/probe40 resume100-from1000 pass: `{None if target_grid_rgbprobe40_feature05_resume100_from1000_result is None else target_grid_rgbprobe40_feature05_resume100_from1000_result['pass']}`",
            f"- recover schedule resume100-from1100 config: `{target_grid_rgbprobe_recover_resume100_from1100['path']}`",
            f"- recover schedule resume100-from1100 result: `{None if target_grid_rgbprobe_recover_resume100_from1100_result is None else target_grid_rgbprobe_recover_resume100_from1100_result['path']}`",
            f"- recover schedule resume100-from1100 pass: `{None if target_grid_rgbprobe_recover_resume100_from1100_result is None else target_grid_rgbprobe_recover_resume100_from1100_result['pass']}`",
            f"- feature0.75/probe40 resume50-from1200 config: `{target_grid_rgbprobe40_feature075_resume50_from1200['path']}`",
            f"- feature0.75/probe40 resume50-from1200 result: `{None if target_grid_rgbprobe40_feature075_resume50_from1200_result is None else target_grid_rgbprobe40_feature075_resume50_from1200_result['path']}`",
            f"- feature0.75/probe40 resume50-from1200 pass: `{None if target_grid_rgbprobe40_feature075_resume50_from1200_result is None else target_grid_rgbprobe40_feature075_resume50_from1200_result['pass']}`",
            f"- feature1/probe40 resume50-from1250 config: `{target_grid_rgbprobe40_feature1_resume50_from1250['path']}`",
            f"- feature1/probe40 resume50-from1250 result: `{None if target_grid_rgbprobe40_feature1_resume50_from1250_result is None else target_grid_rgbprobe40_feature1_resume50_from1250_result['path']}`",
            f"- feature1/probe40 resume50-from1250 pass: `{None if target_grid_rgbprobe40_feature1_resume50_from1250_result is None else target_grid_rgbprobe40_feature1_resume50_from1250_result['pass']}`",
            f"- feature1/probe40 resume100-from1300 config: `{target_grid_rgbprobe40_feature1_resume100_from1300['path']}`",
            f"- feature1/probe40 resume100-from1300 result: `{None if target_grid_rgbprobe40_feature1_resume100_from1300_result is None else target_grid_rgbprobe40_feature1_resume100_from1300_result['path']}`",
            f"- feature1/probe40 resume100-from1300 pass: `{None if target_grid_rgbprobe40_feature1_resume100_from1300_result is None else target_grid_rgbprobe40_feature1_resume100_from1300_result['pass']}`",
        ]
    )
    if target_grid_rgbprobe10_result is not None:
        timing = target_grid_rgbprobe10_result["mean_timing_ms"]
        lines.extend(
            [
                f"- total loss: `{target_grid_rgbprobe10_result['start_loss']:.6f} -> {target_grid_rgbprobe10_result['end_loss']:.6f}`",
                f"- feature loss: `{target_grid_rgbprobe10_result['start_feature_target_loss']:.6f} -> {target_grid_rgbprobe10_result['end_feature_target_loss']:.6f}`",
                f"- RGB-probe loss / PSNR: `{target_grid_rgbprobe10_result['start_rgb_probe_loss']:.6f} -> {target_grid_rgbprobe10_result['end_rgb_probe_loss']:.6f}` / `{target_grid_rgbprobe10_result['start_rgb_probe_psnr']:.3f} -> {target_grid_rgbprobe10_result['end_rgb_probe_psnr']:.3f}`",
                f"- mean step/render/feature-target/probe/backward: `{timing['step_ms']:.1f}ms / {timing['render_forward_ms']:.1f}ms / {timing.get('feature_target_ms', 0.0):.1f}ms / {timing.get('rgb_probe_loss_ms', 0.0):.1f}ms / {timing['backward_ms']:.1f}ms`",
                f"- media render: `{target_grid_rgbprobe10_result.get('rgb_probe_media_render_ms')}`",
                f"- media: `{target_grid_rgbprobe10_result.get('rgb_probe_contact_sheet')}`, `{target_grid_rgbprobe10_result.get('rgb_probe_side_by_side_video')}`",
                f"- tile overflow: `{target_grid_rgbprobe10_result['tile_overflow_sum']}`",
            ]
        )
    if target_grid_rgbprobe10_100step_result is not None:
        timing = target_grid_rgbprobe10_100step_result["mean_timing_ms"]
        lines.extend(
            [
                "- 100-step row:",
                f"- 100-step total loss: `{target_grid_rgbprobe10_100step_result['start_loss']:.6f} -> {target_grid_rgbprobe10_100step_result['end_loss']:.6f}`",
                f"- 100-step feature loss: `{target_grid_rgbprobe10_100step_result['start_feature_target_loss']:.6f} -> {target_grid_rgbprobe10_100step_result['end_feature_target_loss']:.6f}`",
                f"- 100-step RGB-probe loss / PSNR: `{target_grid_rgbprobe10_100step_result['start_rgb_probe_loss']:.6f} -> {target_grid_rgbprobe10_100step_result['end_rgb_probe_loss']:.6f}` / `{target_grid_rgbprobe10_100step_result['start_rgb_probe_psnr']:.3f} -> {target_grid_rgbprobe10_100step_result['end_rgb_probe_psnr']:.3f}`",
                f"- 100-step mean step/render/feature-target/probe/backward: `{timing['step_ms']:.1f}ms / {timing['render_forward_ms']:.1f}ms / {timing.get('feature_target_ms', 0.0):.1f}ms / {timing.get('rgb_probe_loss_ms', 0.0):.1f}ms / {timing['backward_ms']:.1f}ms`",
                f"- 100-step media render: `{target_grid_rgbprobe10_100step_result.get('rgb_probe_media_render_ms')}`",
                f"- 100-step media: `{target_grid_rgbprobe10_100step_result.get('rgb_probe_contact_sheet')}`, `{target_grid_rgbprobe10_100step_result.get('rgb_probe_side_by_side_video')}`",
                f"- 100-step tile overflow: `{target_grid_rgbprobe10_100step_result['tile_overflow_sum']}`",
            ]
        )
    if target_grid_rgbprobe10_300step_result is not None:
        timing = target_grid_rgbprobe10_300step_result["mean_timing_ms"]
        lines.extend(
            [
                "- 300-step row:",
                f"- 300-step total loss: `{target_grid_rgbprobe10_300step_result['start_loss']:.6f} -> {target_grid_rgbprobe10_300step_result['end_loss']:.6f}`",
                f"- 300-step feature loss: `{target_grid_rgbprobe10_300step_result['start_feature_target_loss']:.6f} -> {target_grid_rgbprobe10_300step_result['end_feature_target_loss']:.6f}`",
                f"- 300-step RGB-probe loss / PSNR: `{target_grid_rgbprobe10_300step_result['start_rgb_probe_loss']:.6f} -> {target_grid_rgbprobe10_300step_result['end_rgb_probe_loss']:.6f}` / `{target_grid_rgbprobe10_300step_result['start_rgb_probe_psnr']:.3f} -> {target_grid_rgbprobe10_300step_result['end_rgb_probe_psnr']:.3f}`",
                f"- 300-step mean step/render/feature-target/probe/backward: `{timing['step_ms']:.1f}ms / {timing['render_forward_ms']:.1f}ms / {timing.get('feature_target_ms', 0.0):.1f}ms / {timing.get('rgb_probe_loss_ms', 0.0):.1f}ms / {timing['backward_ms']:.1f}ms`",
                f"- 300-step media render: `{target_grid_rgbprobe10_300step_result.get('rgb_probe_media_render_ms')}`",
                f"- 300-step media: `{target_grid_rgbprobe10_300step_result.get('rgb_probe_contact_sheet')}`, `{target_grid_rgbprobe10_300step_result.get('rgb_probe_side_by_side_video')}`",
                f"- 300-step tile overflow: `{target_grid_rgbprobe10_300step_result['tile_overflow_sum']}`",
            ]
        )
    if target_grid_rgbprobe10_300step_checkpoint_result is not None:
        timing = target_grid_rgbprobe10_300step_checkpoint_result["mean_timing_ms"]
        lines.extend(
            [
                "- 300-step checkpoint/no-media row:",
                f"- 300-step checkpoint feature loss: `{target_grid_rgbprobe10_300step_checkpoint_result['start_feature_target_loss']:.6f} -> {target_grid_rgbprobe10_300step_checkpoint_result['end_feature_target_loss']:.6f}`",
                f"- 300-step checkpoint RGB-probe loss / PSNR: `{target_grid_rgbprobe10_300step_checkpoint_result['start_rgb_probe_loss']:.6f} -> {target_grid_rgbprobe10_300step_checkpoint_result['end_rgb_probe_loss']:.6f}` / `{target_grid_rgbprobe10_300step_checkpoint_result['start_rgb_probe_psnr']:.3f} -> {target_grid_rgbprobe10_300step_checkpoint_result['end_rgb_probe_psnr']:.3f}`",
                f"- 300-step checkpoint mean step/render/feature-target/probe/backward: `{timing['step_ms']:.1f}ms / {timing['render_forward_ms']:.1f}ms / {timing.get('feature_target_ms', 0.0):.1f}ms / {timing.get('rgb_probe_loss_ms', 0.0):.1f}ms / {timing['backward_ms']:.1f}ms`",
                f"- 300-step checkpoint: `{target_grid_rgbprobe10_300step_checkpoint_result.get('checkpoint')}`",
                f"- 300-step checkpoint tile overflow: `{target_grid_rgbprobe10_300step_checkpoint_result['tile_overflow_sum']}`",
            ]
        )
    if target_grid_rgbprobe10_resume300_from300_result is not None:
        timing = target_grid_rgbprobe10_resume300_from300_result["mean_timing_ms"]
        lines.extend(
            [
                "- resume300-from300 row:",
                f"- resume loaded / optimizer loaded / source steps: `{target_grid_rgbprobe10_resume300_from300_result['resume_loaded']}` / `{target_grid_rgbprobe10_resume300_from300_result['resume_optimizer_loaded']}` / `{target_grid_rgbprobe10_resume300_from300_result['resume_checkpoint_steps']}`",
                f"- resume feature loss: `{target_grid_rgbprobe10_resume300_from300_result['start_feature_target_loss']:.6f} -> {target_grid_rgbprobe10_resume300_from300_result['end_feature_target_loss']:.6f}`",
                f"- resume RGB-probe loss / PSNR: `{target_grid_rgbprobe10_resume300_from300_result['start_rgb_probe_loss']:.6f} -> {target_grid_rgbprobe10_resume300_from300_result['end_rgb_probe_loss']:.6f}` / `{target_grid_rgbprobe10_resume300_from300_result['start_rgb_probe_psnr']:.3f} -> {target_grid_rgbprobe10_resume300_from300_result['end_rgb_probe_psnr']:.3f}`",
                f"- resume mean step/render/feature-target/probe/backward: `{timing['step_ms']:.1f}ms / {timing['render_forward_ms']:.1f}ms / {timing.get('feature_target_ms', 0.0):.1f}ms / {timing.get('rgb_probe_loss_ms', 0.0):.1f}ms / {timing['backward_ms']:.1f}ms`",
                f"- resume checkpoint: `{target_grid_rgbprobe10_resume300_from300_result.get('checkpoint')}`",
                f"- resume media: `{target_grid_rgbprobe10_resume300_from300_result.get('rgb_probe_contact_sheet')}`, `{target_grid_rgbprobe10_resume300_from300_result.get('rgb_probe_side_by_side_video')}`",
                f"- resume tile overflow: `{target_grid_rgbprobe10_resume300_from300_result['tile_overflow_sum']}`",
            ]
        )
    if target_grid_rgbprobe40_feature025_resume200_from600_result is not None:
        timing = target_grid_rgbprobe40_feature025_resume200_from600_result["mean_timing_ms"]
        lines.extend(
            [
                "- probe-emphasis resume200-from600 row:",
                f"- probe-emphasis objective: `feature={target_grid_rgbprobe40_feature025_resume200_from600['feature_loss_weight']}` / `rgb_probe={target_grid_rgbprobe40_feature025_resume200_from600['rgb_probe_loss_weight']}`",
                f"- probe-emphasis global steps: `{target_grid_rgbprobe40_feature025_resume200_from600_result['start_global_step']} -> {target_grid_rgbprobe40_feature025_resume200_from600_result['end_global_step']}`",
                f"- probe-emphasis resume loaded / optimizer loaded / source steps: `{target_grid_rgbprobe40_feature025_resume200_from600_result['resume_loaded']}` / `{target_grid_rgbprobe40_feature025_resume200_from600_result['resume_optimizer_loaded']}` / `{target_grid_rgbprobe40_feature025_resume200_from600_result['resume_checkpoint_steps']}`",
                f"- probe-emphasis feature loss: `{target_grid_rgbprobe40_feature025_resume200_from600_result['start_feature_target_loss']:.6f} -> {target_grid_rgbprobe40_feature025_resume200_from600_result['end_feature_target_loss']:.6f}`",
                f"- probe-emphasis RGB-probe loss / PSNR: `{target_grid_rgbprobe40_feature025_resume200_from600_result['start_rgb_probe_loss']:.6f} -> {target_grid_rgbprobe40_feature025_resume200_from600_result['end_rgb_probe_loss']:.6f}` / `{target_grid_rgbprobe40_feature025_resume200_from600_result['start_rgb_probe_psnr']:.3f} -> {target_grid_rgbprobe40_feature025_resume200_from600_result['end_rgb_probe_psnr']:.3f}`",
                f"- probe-emphasis mean step/render/feature-target/probe/backward: `{timing['step_ms']:.1f}ms / {timing['render_forward_ms']:.1f}ms / {timing.get('feature_target_ms', 0.0):.1f}ms / {timing.get('rgb_probe_loss_ms', 0.0):.1f}ms / {timing['backward_ms']:.1f}ms`",
                f"- probe-emphasis checkpoint: `{target_grid_rgbprobe40_feature025_resume200_from600_result.get('checkpoint')}`",
                f"- probe-emphasis media: `{target_grid_rgbprobe40_feature025_resume200_from600_result.get('rgb_probe_contact_sheet')}`, `{target_grid_rgbprobe40_feature025_resume200_from600_result.get('rgb_probe_side_by_side_video')}`",
                f"- probe-emphasis tile overflow: `{target_grid_rgbprobe40_feature025_resume200_from600_result['tile_overflow_sum']}`",
            ]
        )
    if target_grid_rgbprobe_balance_resume200_from800_result is not None:
        timing = target_grid_rgbprobe_balance_resume200_from800_result["mean_timing_ms"]
        lines.extend(
            [
                "- scheduled balance resume200-from800 row:",
                f"- scheduled balance schedule: `{target_grid_rgbprobe_balance_resume200_from800_result.get('feature_target_weight_schedule')}`",
                f"- scheduled balance global steps: `{target_grid_rgbprobe_balance_resume200_from800_result['start_global_step']} -> {target_grid_rgbprobe_balance_resume200_from800_result['end_global_step']}`",
                f"- scheduled balance resume loaded / optimizer loaded / source steps: `{target_grid_rgbprobe_balance_resume200_from800_result['resume_loaded']}` / `{target_grid_rgbprobe_balance_resume200_from800_result['resume_optimizer_loaded']}` / `{target_grid_rgbprobe_balance_resume200_from800_result['resume_checkpoint_steps']}`",
                f"- scheduled balance feature loss: `{target_grid_rgbprobe_balance_resume200_from800_result['start_feature_target_loss']:.6f} -> {target_grid_rgbprobe_balance_resume200_from800_result['end_feature_target_loss']:.6f}`",
                f"- scheduled balance RGB-probe loss / PSNR: `{target_grid_rgbprobe_balance_resume200_from800_result['start_rgb_probe_loss']:.6f} -> {target_grid_rgbprobe_balance_resume200_from800_result['end_rgb_probe_loss']:.6f}` / `{target_grid_rgbprobe_balance_resume200_from800_result['start_rgb_probe_psnr']:.3f} -> {target_grid_rgbprobe_balance_resume200_from800_result['end_rgb_probe_psnr']:.3f}`",
                f"- scheduled balance mean step/render/feature-target/probe/backward: `{timing['step_ms']:.1f}ms / {timing['render_forward_ms']:.1f}ms / {timing.get('feature_target_ms', 0.0):.1f}ms / {timing.get('rgb_probe_loss_ms', 0.0):.1f}ms / {timing['backward_ms']:.1f}ms`",
                f"- scheduled balance checkpoint: `{target_grid_rgbprobe_balance_resume200_from800_result.get('checkpoint')}`",
                f"- scheduled balance media: `{target_grid_rgbprobe_balance_resume200_from800_result.get('rgb_probe_contact_sheet')}`, `{target_grid_rgbprobe_balance_resume200_from800_result.get('rgb_probe_side_by_side_video')}`",
                f"- scheduled balance tile overflow: `{target_grid_rgbprobe_balance_resume200_from800_result['tile_overflow_sum']}`",
            ]
        )
    if target_grid_rgbprobe40_feature05_resume100_from1000_result is not None:
        timing = target_grid_rgbprobe40_feature05_resume100_from1000_result["mean_timing_ms"]
        lines.extend(
            [
                "- feature0.5/probe40 resume100-from1000 row:",
                f"- feature0.5/probe40 objective: `feature={target_grid_rgbprobe40_feature05_resume100_from1000['feature_loss_weight']}` / `rgb_probe={target_grid_rgbprobe40_feature05_resume100_from1000['rgb_probe_loss_weight']}`",
                f"- feature0.5/probe40 global steps: `{target_grid_rgbprobe40_feature05_resume100_from1000_result['start_global_step']} -> {target_grid_rgbprobe40_feature05_resume100_from1000_result['end_global_step']}`",
                f"- feature0.5/probe40 resume loaded / optimizer loaded / source steps: `{target_grid_rgbprobe40_feature05_resume100_from1000_result['resume_loaded']}` / `{target_grid_rgbprobe40_feature05_resume100_from1000_result['resume_optimizer_loaded']}` / `{target_grid_rgbprobe40_feature05_resume100_from1000_result['resume_checkpoint_steps']}`",
                f"- feature0.5/probe40 feature loss: `{target_grid_rgbprobe40_feature05_resume100_from1000_result['start_feature_target_loss']:.6f} -> {target_grid_rgbprobe40_feature05_resume100_from1000_result['end_feature_target_loss']:.6f}`",
                f"- feature0.5/probe40 RGB-probe loss / PSNR: `{target_grid_rgbprobe40_feature05_resume100_from1000_result['start_rgb_probe_loss']:.6f} -> {target_grid_rgbprobe40_feature05_resume100_from1000_result['end_rgb_probe_loss']:.6f}` / `{target_grid_rgbprobe40_feature05_resume100_from1000_result['start_rgb_probe_psnr']:.3f} -> {target_grid_rgbprobe40_feature05_resume100_from1000_result['end_rgb_probe_psnr']:.3f}`",
                f"- feature0.5/probe40 mean step/render/feature-target/probe/backward: `{timing['step_ms']:.1f}ms / {timing['render_forward_ms']:.1f}ms / {timing.get('feature_target_ms', 0.0):.1f}ms / {timing.get('rgb_probe_loss_ms', 0.0):.1f}ms / {timing['backward_ms']:.1f}ms`",
                f"- feature0.5/probe40 checkpoint: `{target_grid_rgbprobe40_feature05_resume100_from1000_result.get('checkpoint')}`",
                f"- feature0.5/probe40 media: `{target_grid_rgbprobe40_feature05_resume100_from1000_result.get('rgb_probe_contact_sheet')}`, `{target_grid_rgbprobe40_feature05_resume100_from1000_result.get('rgb_probe_side_by_side_video')}`",
                f"- feature0.5/probe40 tile overflow: `{target_grid_rgbprobe40_feature05_resume100_from1000_result['tile_overflow_sum']}`",
            ]
        )
    if target_grid_rgbprobe_recover_resume100_from1100_result is not None:
        timing = target_grid_rgbprobe_recover_resume100_from1100_result["mean_timing_ms"]
        lines.extend(
            [
                "- recover schedule resume100-from1100 row:",
                f"- recover schedule schedule: `{target_grid_rgbprobe_recover_resume100_from1100_result.get('feature_target_weight_schedule')}`",
                f"- recover schedule global steps: `{target_grid_rgbprobe_recover_resume100_from1100_result['start_global_step']} -> {target_grid_rgbprobe_recover_resume100_from1100_result['end_global_step']}`",
                f"- recover schedule resume loaded / optimizer loaded / source steps: `{target_grid_rgbprobe_recover_resume100_from1100_result['resume_loaded']}` / `{target_grid_rgbprobe_recover_resume100_from1100_result['resume_optimizer_loaded']}` / `{target_grid_rgbprobe_recover_resume100_from1100_result['resume_checkpoint_steps']}`",
                f"- recover schedule feature loss: `{target_grid_rgbprobe_recover_resume100_from1100_result['start_feature_target_loss']:.6f} -> {target_grid_rgbprobe_recover_resume100_from1100_result['end_feature_target_loss']:.6f}`",
                f"- recover schedule RGB-probe loss / PSNR: `{target_grid_rgbprobe_recover_resume100_from1100_result['start_rgb_probe_loss']:.6f} -> {target_grid_rgbprobe_recover_resume100_from1100_result['end_rgb_probe_loss']:.6f}` / `{target_grid_rgbprobe_recover_resume100_from1100_result['start_rgb_probe_psnr']:.3f} -> {target_grid_rgbprobe_recover_resume100_from1100_result['end_rgb_probe_psnr']:.3f}`",
                f"- recover schedule mean step/render/feature-target/probe/backward: `{timing['step_ms']:.1f}ms / {timing['render_forward_ms']:.1f}ms / {timing.get('feature_target_ms', 0.0):.1f}ms / {timing.get('rgb_probe_loss_ms', 0.0):.1f}ms / {timing['backward_ms']:.1f}ms`",
                f"- recover schedule checkpoint: `{target_grid_rgbprobe_recover_resume100_from1100_result.get('checkpoint')}`",
                f"- recover schedule media: `{target_grid_rgbprobe_recover_resume100_from1100_result.get('rgb_probe_contact_sheet')}`, `{target_grid_rgbprobe_recover_resume100_from1100_result.get('rgb_probe_side_by_side_video')}`",
                f"- recover schedule tile overflow: `{target_grid_rgbprobe_recover_resume100_from1100_result['tile_overflow_sum']}`",
            ]
        )
    if target_grid_rgbprobe40_feature075_resume50_from1200_result is not None:
        timing = target_grid_rgbprobe40_feature075_resume50_from1200_result["mean_timing_ms"]
        lines.extend(
            [
                "- feature0.75/probe40 resume50-from1200 row:",
                f"- feature0.75/probe40 objective: `feature={target_grid_rgbprobe40_feature075_resume50_from1200['feature_loss_weight']}` / `rgb_probe={target_grid_rgbprobe40_feature075_resume50_from1200['rgb_probe_loss_weight']}`",
                f"- feature0.75/probe40 global steps: `{target_grid_rgbprobe40_feature075_resume50_from1200_result['start_global_step']} -> {target_grid_rgbprobe40_feature075_resume50_from1200_result['end_global_step']}`",
                f"- feature0.75/probe40 resume loaded / optimizer loaded / source steps: `{target_grid_rgbprobe40_feature075_resume50_from1200_result['resume_loaded']}` / `{target_grid_rgbprobe40_feature075_resume50_from1200_result['resume_optimizer_loaded']}` / `{target_grid_rgbprobe40_feature075_resume50_from1200_result['resume_checkpoint_steps']}`",
                f"- feature0.75/probe40 feature loss: `{target_grid_rgbprobe40_feature075_resume50_from1200_result['start_feature_target_loss']:.6f} -> {target_grid_rgbprobe40_feature075_resume50_from1200_result['end_feature_target_loss']:.6f}`",
                f"- feature0.75/probe40 RGB-probe loss / PSNR: `{target_grid_rgbprobe40_feature075_resume50_from1200_result['start_rgb_probe_loss']:.6f} -> {target_grid_rgbprobe40_feature075_resume50_from1200_result['end_rgb_probe_loss']:.6f}` / `{target_grid_rgbprobe40_feature075_resume50_from1200_result['start_rgb_probe_psnr']:.3f} -> {target_grid_rgbprobe40_feature075_resume50_from1200_result['end_rgb_probe_psnr']:.3f}`",
                f"- feature0.75/probe40 mean step/render/feature-target/probe/backward: `{timing['step_ms']:.1f}ms / {timing['render_forward_ms']:.1f}ms / {timing.get('feature_target_ms', 0.0):.1f}ms / {timing.get('rgb_probe_loss_ms', 0.0):.1f}ms / {timing['backward_ms']:.1f}ms`",
                f"- feature0.75/probe40 checkpoint: `{target_grid_rgbprobe40_feature075_resume50_from1200_result.get('checkpoint')}`",
                f"- feature0.75/probe40 media: `{target_grid_rgbprobe40_feature075_resume50_from1200_result.get('rgb_probe_contact_sheet')}`, `{target_grid_rgbprobe40_feature075_resume50_from1200_result.get('rgb_probe_side_by_side_video')}`",
                f"- feature0.75/probe40 tile overflow: `{target_grid_rgbprobe40_feature075_resume50_from1200_result['tile_overflow_sum']}`",
            ]
        )
    if target_grid_rgbprobe40_feature1_resume50_from1250_result is not None:
        timing = target_grid_rgbprobe40_feature1_resume50_from1250_result["mean_timing_ms"]
        lines.extend(
            [
                "- feature1/probe40 resume50-from1250 row:",
                f"- feature1/probe40 objective: `feature={target_grid_rgbprobe40_feature1_resume50_from1250['feature_loss_weight']}` / `rgb_probe={target_grid_rgbprobe40_feature1_resume50_from1250['rgb_probe_loss_weight']}`",
                f"- feature1/probe40 global steps: `{target_grid_rgbprobe40_feature1_resume50_from1250_result['start_global_step']} -> {target_grid_rgbprobe40_feature1_resume50_from1250_result['end_global_step']}`",
                f"- feature1/probe40 resume loaded / optimizer loaded / source steps: `{target_grid_rgbprobe40_feature1_resume50_from1250_result['resume_loaded']}` / `{target_grid_rgbprobe40_feature1_resume50_from1250_result['resume_optimizer_loaded']}` / `{target_grid_rgbprobe40_feature1_resume50_from1250_result['resume_checkpoint_steps']}`",
                f"- feature1/probe40 feature loss: `{target_grid_rgbprobe40_feature1_resume50_from1250_result['start_feature_target_loss']:.6f} -> {target_grid_rgbprobe40_feature1_resume50_from1250_result['end_feature_target_loss']:.6f}`",
                f"- feature1/probe40 RGB-probe loss / PSNR: `{target_grid_rgbprobe40_feature1_resume50_from1250_result['start_rgb_probe_loss']:.6f} -> {target_grid_rgbprobe40_feature1_resume50_from1250_result['end_rgb_probe_loss']:.6f}` / `{target_grid_rgbprobe40_feature1_resume50_from1250_result['start_rgb_probe_psnr']:.3f} -> {target_grid_rgbprobe40_feature1_resume50_from1250_result['end_rgb_probe_psnr']:.3f}`",
                f"- feature1/probe40 mean step/render/feature-target/probe/backward: `{timing['step_ms']:.1f}ms / {timing['render_forward_ms']:.1f}ms / {timing.get('feature_target_ms', 0.0):.1f}ms / {timing.get('rgb_probe_loss_ms', 0.0):.1f}ms / {timing['backward_ms']:.1f}ms`",
                f"- feature1/probe40 checkpoint: `{target_grid_rgbprobe40_feature1_resume50_from1250_result.get('checkpoint')}`",
                f"- feature1/probe40 media: `{target_grid_rgbprobe40_feature1_resume50_from1250_result.get('rgb_probe_contact_sheet')}`, `{target_grid_rgbprobe40_feature1_resume50_from1250_result.get('rgb_probe_side_by_side_video')}`",
                f"- feature1/probe40 tile overflow: `{target_grid_rgbprobe40_feature1_resume50_from1250_result['tile_overflow_sum']}`",
            ]
        )
    if target_grid_rgbprobe40_feature1_resume100_from1300_result is not None:
        timing = target_grid_rgbprobe40_feature1_resume100_from1300_result["mean_timing_ms"]
        lines.extend(
            [
                "- feature1/probe40 resume100-from1300 row:",
                f"- feature1/probe40 resume100 objective: `feature={target_grid_rgbprobe40_feature1_resume100_from1300['feature_loss_weight']}` / `rgb_probe={target_grid_rgbprobe40_feature1_resume100_from1300['rgb_probe_loss_weight']}`",
                f"- feature1/probe40 resume100 global steps: `{target_grid_rgbprobe40_feature1_resume100_from1300_result['start_global_step']} -> {target_grid_rgbprobe40_feature1_resume100_from1300_result['end_global_step']}`",
                f"- feature1/probe40 resume100 resume loaded / optimizer loaded / source steps: `{target_grid_rgbprobe40_feature1_resume100_from1300_result['resume_loaded']}` / `{target_grid_rgbprobe40_feature1_resume100_from1300_result['resume_optimizer_loaded']}` / `{target_grid_rgbprobe40_feature1_resume100_from1300_result['resume_checkpoint_steps']}`",
                f"- feature1/probe40 resume100 feature loss: `{target_grid_rgbprobe40_feature1_resume100_from1300_result['start_feature_target_loss']:.6f} -> {target_grid_rgbprobe40_feature1_resume100_from1300_result['end_feature_target_loss']:.6f}`",
                f"- feature1/probe40 resume100 RGB-probe loss / PSNR: `{target_grid_rgbprobe40_feature1_resume100_from1300_result['start_rgb_probe_loss']:.6f} -> {target_grid_rgbprobe40_feature1_resume100_from1300_result['end_rgb_probe_loss']:.6f}` / `{target_grid_rgbprobe40_feature1_resume100_from1300_result['start_rgb_probe_psnr']:.3f} -> {target_grid_rgbprobe40_feature1_resume100_from1300_result['end_rgb_probe_psnr']:.3f}`",
                f"- feature1/probe40 resume100 mean step/render/feature-target/probe/backward: `{timing['step_ms']:.1f}ms / {timing['render_forward_ms']:.1f}ms / {timing.get('feature_target_ms', 0.0):.1f}ms / {timing.get('rgb_probe_loss_ms', 0.0):.1f}ms / {timing['backward_ms']:.1f}ms`",
                f"- feature1/probe40 resume100 checkpoint: `{target_grid_rgbprobe40_feature1_resume100_from1300_result.get('checkpoint')}`",
                f"- feature1/probe40 resume100 media: `{target_grid_rgbprobe40_feature1_resume100_from1300_result.get('rgb_probe_contact_sheet')}`, `{target_grid_rgbprobe40_feature1_resume100_from1300_result.get('rgb_probe_side_by_side_video')}`",
                f"- feature1/probe40 resume100 tile overflow: `{target_grid_rgbprobe40_feature1_resume100_from1300_result['tile_overflow_sum']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            conclusion["missing_bridge"],
            "",
            "## Next Contract",
            "",
        ]
    )
    for item in report["next_contract"]:
        lines.append(f"- {item}")
    lines.append("")
    write_report_text(path, "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    report = _build_report()
    write_report_json(args.out_json, report)
    _write_markdown(report, args.out_md)
    print(json.dumps(report["conclusion"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
