from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from config_utils import load_config_file
from dynamicTokenGS import fast_attn_context, pick_device
from sequence_data import load_camera_sequence, load_uncalibrated_sequence, resolve_frames_dir
from train_video_token_implicit_dynamic import (
    build_model_from_config,
    load_manifest_sequences,
    prepare_clip,
    resolve_config,
)


EXPORT_BUNDLE_VERSION = "dynaworld_token_head_bundle/v2"


def _load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        state_dict = payload["state_dict"]
    elif isinstance(payload, dict) and payload and all(torch.is_tensor(value) for value in payload.values()):
        state_dict = payload
    else:
        raise ValueError(
            f"Could not find a plain state_dict in {checkpoint_path}. "
            "Expected either a tensor mapping or a mapping with a 'state_dict' entry."
        )
    return {str(name): value for name, value in state_dict.items()}


def export_id_from_config(config: dict[str, Any], *, suffix: str | None = None) -> str:
    export_cfg = config.get("export", {})
    explicit_id = export_cfg.get("id") if isinstance(export_cfg, dict) else None
    if explicit_id:
        return str(explicit_id)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_name = str(config.get("logging", {}).get("wandb_run_name") or "dynaworld")
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", run_name).strip("-")[:96] or "dynaworld"
    if suffix:
        slug = f"{slug}-{suffix}"
    return f"{timestamp}_{slug}"


def _load_sequence_from_config(
    resolved: dict[str, Any],
    *,
    device: torch.device,
    sequence_index: int,
) -> Any:
    data_cfg = resolved["data"]
    model_cfg = resolved["model"]
    if data_cfg["manifest_path"] is not None:
        sequences = load_manifest_sequences(
            data_cfg["manifest_path"],
            split=data_cfg["split"],
            data_cfg=data_cfg,
            model_cfg=model_cfg,
            device=device,
        )
        if sequence_index < 0 or sequence_index >= len(sequences):
            raise IndexError(
                f"sequence_index={sequence_index} is out of range for {len(sequences)} loaded manifest sequences."
            )
        return sequences[sequence_index]

    if sequence_index != 0:
        raise ValueError("sequence_index is only valid when data.manifest_path is set.")
    if data_cfg["sequence_dir"] is None:
        raise ValueError("config['data']['sequence_dir'] is required when manifest_path is not set.")
    if data_cfg["frame_source"] == "camera_json":
        camera_json_path = data_cfg["camera_json"] or (data_cfg["sequence_dir"] / "per_frame_cameras.json")
        return load_camera_sequence(
            camera_json_path=camera_json_path,
            target_size=model_cfg["size"],
            camera_image_size=data_cfg["camera_image_size"],
            max_frames=data_cfg["max_frames"],
            focal_mode=data_cfg["camera_focal_mode"],
            device=device,
        )
    if data_cfg["frame_source"] == "explicit_video" and data_cfg["video_path"] is None:
        raise ValueError("config['data']['video_path'] is required when frame_source='explicit_video'.")
    frames_dir = resolve_frames_dir(data_cfg["sequence_dir"], data_cfg["frames_dir"])
    return load_uncalibrated_sequence(
        sequence_dir=data_cfg["sequence_dir"],
        frames_dir=frames_dir,
        video_path=data_cfg["video_path"],
        target_size=model_cfg["size"],
        max_frames=data_cfg["max_frames"],
        frame_source=data_cfg["frame_source"],
        device=device,
    )


def _clip_indices(
    *,
    frame_count: int,
    train_frame_count: int,
    window_start: int,
    device: torch.device,
) -> torch.Tensor:
    window = min(int(train_frame_count), int(frame_count))
    if window < 1:
        raise ValueError("window must be at least 1 frame.")
    if window >= frame_count:
        return torch.arange(frame_count, device=device)
    start = max(0, min(int(window_start), frame_count - window))
    return torch.arange(start, start + window, device=device)


def _write_tensor(output_dir: Path, relative_path: str, tensor: torch.Tensor) -> dict[str, Any]:
    path = output_dir / relative_path
    value = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
    path.write_bytes(value.numpy().tobytes())
    return {
        "path": relative_path,
        "dtype": "float32",
        "shape": list(value.shape),
        "count": int(value.numel()),
    }


def _write_module_tensors(
    output_dir: Path,
    tensors: dict[str, dict[str, Any]],
    prefix: str,
    module: torch.nn.Module,
) -> None:
    for name, value in module.state_dict().items():
        key = f"{prefix}.{name}"
        filename = f"{key.replace('.', '_')}.f32"
        tensors[key] = _write_tensor(output_dir, filename, value)


def _gaussian_head_meta(head: Any) -> dict[str, Any]:
    return {
        "gaussians_per_token": int(head.gaussians_per_token),
        "xy_extent": float(head.xy_extent),
        "z_min": float(head.z_min),
        "z_max": float(head.z_min + head.z_extent),
        "z_extent": float(head.z_extent),
        "scale_init": float(head.scale_init),
    }


def _bounds_from_head_meta(
    *,
    static_meta: dict[str, Any],
    dynamic_meta: dict[str, Any],
    dynamic_motion_extent: float,
    dynamic_time_basis_count: int,
) -> dict[str, list[float]]:
    motion_bound = float(dynamic_motion_extent) * float(dynamic_time_basis_count)
    min_xyz = torch.tensor(
        [
            min(-float(static_meta["xy_extent"]), -float(dynamic_meta["xy_extent"]) - motion_bound),
            min(-float(static_meta["xy_extent"]), -float(dynamic_meta["xy_extent"]) - motion_bound),
            min(float(static_meta["z_min"]), float(dynamic_meta["z_min"]) - motion_bound),
        ],
        dtype=torch.float32,
    )
    max_xyz = torch.tensor(
        [
            max(float(static_meta["xy_extent"]), float(dynamic_meta["xy_extent"]) + motion_bound),
            max(float(static_meta["xy_extent"]), float(dynamic_meta["xy_extent"]) + motion_bound),
            max(float(static_meta["z_max"]), float(dynamic_meta["z_max"]) + motion_bound),
        ],
        dtype=torch.float32,
    )
    center = 0.5 * (min_xyz + max_xyz)
    return {
        "min": [float(value) for value in min_xyz.tolist()],
        "max": [float(value) for value in max_xyz.tolist()],
        "center": [float(value) for value in center.tolist()],
    }


def _load_model_input(
    resolved: dict[str, Any],
    sequence_data: Any,
    clip_frames: torch.Tensor,
    clip_times: torch.Tensor,
    *,
    device: torch.device,
) -> Any:
    backend = str(resolved["model"]["video_encoder_backend"]).lower()
    if backend in {"precomputed", "precomputed_ltx"}:
        if "features" not in resolved:
            raise ValueError(
                "This config uses a precomputed feature backend but has no 'features' section. "
                "Use the precomputed-feature trainer config or add the features block."
            )
        from train_precomputed_feature_implicit_dynamic import PrecomputedFeatureImplicitTrainer
        from video_feature_cache import VideoFeatureCache

        feature_resolved = PrecomputedFeatureImplicitTrainer.resolve_config(resolved)
        feature_cache = VideoFeatureCache(feature_resolved["features"], device)
        return feature_cache.load_or_bake(sequence_data)
    del clip_times
    return clip_frames


def export_browser_bundle_from_model(
    *,
    model: torch.nn.Module,
    resolved: dict[str, Any],
    sequence_data: Any,
    clip_indices: torch.Tensor,
    clip_times: torch.Tensor,
    model_input: Any,
    output_dir: Path,
    config_path: Path | None = None,
    state_dict_path: Path | None,
    export_id: str | None = None,
) -> Path:
    if not bool(getattr(model, "use_static_dynamic_split", False)):
        raise ValueError(
            "The browser bundle exporter currently only supports models with "
            "model.static_tokens + model.dynamic_tokens enabled."
        )
    if not hasattr(model, "static_gaussian_heads") or not hasattr(model, "dynamic_gaussian_heads"):
        raise ValueError("Model is marked as static/dynamic split but missing split Gaussian head modules.")

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), fast_attn_context(clip_times.device):
            video_tokens = model.video_encoder(model_input, frame_times=clip_times)
            fixed_queries = model.refine_queries(video_tokens, decode_time=None).squeeze(0)
            static_query_tokens = fixed_queries[2 : 2 + int(model.static_tokens)]
            dynamic_query_tokens = fixed_queries[2 + int(model.static_tokens) :]
    finally:
        if was_training:
            model.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, dict[str, Any]] = {}
    tensors["refined_queries"] = _write_tensor(output_dir, "refined_queries.f32", fixed_queries)
    tensors["static_query_tokens"] = _write_tensor(output_dir, "static_query_tokens.f32", static_query_tokens)
    tensors["dynamic_query_tokens"] = _write_tensor(output_dir, "dynamic_query_tokens.f32", dynamic_query_tokens)
    _write_module_tensors(output_dir, tensors, "static_gaussian_heads", model.static_gaussian_heads)
    _write_module_tensors(output_dir, tensors, "dynamic_gaussian_heads.base_heads", model.dynamic_gaussian_heads.base_heads)
    _write_module_tensors(output_dir, tensors, "dynamic_gaussian_heads.motion_head", model.dynamic_gaussian_heads.motion_head)
    _write_module_tensors(output_dir, tensors, "dynamic_gaussian_heads.rotation_head", model.dynamic_gaussian_heads.rotation_head)
    _write_module_tensors(output_dir, tensors, "dynamic_gaussian_heads.alpha_head", model.dynamic_gaussian_heads.alpha_head)
    _write_module_tensors(output_dir, tensors, "time_proj", model.time_proj)
    _write_module_tensors(output_dir, tensors, "head_time_proj", model.head_time_proj)

    static_meta = _gaussian_head_meta(model.static_gaussian_heads)
    dynamic_base_meta = _gaussian_head_meta(model.dynamic_gaussian_heads.base_heads)
    bounds = _bounds_from_head_meta(
        static_meta=static_meta,
        dynamic_meta=dynamic_base_meta,
        dynamic_motion_extent=float(model.dynamic_gaussian_heads.motion_extent),
        dynamic_time_basis_count=int(model.dynamic_time_basis_count),
    )
    clip_index_values = clip_indices.detach().cpu().tolist()
    clip_time_values = clip_times.squeeze(0).detach().cpu().tolist()
    manifest = {
        "version": EXPORT_BUNDLE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "export_id": export_id,
        "config_path": None if config_path is None else str(config_path),
        "state_dict_path": None if state_dict_path is None else str(state_dict_path),
        "source": {
            "sequence_index": int(resolved.get("export", {}).get("sequence_index", 0)),
            "sequence_path": str(sequence_data.source_path),
            "frame_count": int(sequence_data.frame_count),
            "frame_source": str(sequence_data.frame_source),
            "clip_indices": [int(value) for value in clip_index_values],
            "clip_times": [float(value) for value in clip_time_values],
            "window_start": int(clip_index_values[0]) if clip_index_values else 0,
        },
        "model": {
            "variant": str(resolved["model"]["variant"]),
            "video_encoder_backend": str(resolved["model"]["video_encoder_backend"]),
            "train_frame_count": int(resolved["model"]["train_frame_count"]),
            "feat_dim": int(model.feat_dim),
            "num_tokens": int(model.num_tokens),
            "gaussians_per_token": int(model.gaussians_per_token),
            "static_tokens": int(model.static_tokens),
            "dynamic_tokens": int(model.dynamic_tokens),
            "dynamic_time_basis_count": int(model.dynamic_time_basis_count),
            "dynamic_time_max_frequency": float(model.dynamic_time_max_frequency),
            "image_size": int(model.image_size),
            "bundle_contract": "refined_tokens_plus_decoder_heads",
            "notes": [
                "The viewer decodes static and dynamic splats from saved refined query tokens plus Gaussian head MLP weights.",
                "No decoded Gaussian arrays are saved in this bundle.",
            ],
        },
        "decoder": {
            "static_gaussian_heads": static_meta,
            "dynamic_gaussian_heads": {
                "base_heads": dynamic_base_meta,
                "time_basis_count": int(model.dynamic_gaussian_heads.time_basis_count),
                "motion_extent": float(model.dynamic_gaussian_heads.motion_extent),
                "rotation_radians": float(model.dynamic_gaussian_heads.rotation_radians),
                "alpha_logit_extent": float(model.dynamic_gaussian_heads.alpha_logit_extent),
            },
            "time_proj": {"type": model.time_proj.__class__.__name__},
            "head_time_proj": {"type": model.head_time_proj.__class__.__name__},
        },
        "viewer_defaults": {
            "fov_degrees": 60.0,
            "near": 0.01,
            "far": 100.0,
            "time_domain": [0.0, 1.0],
        },
        "bounds": bounds,
        "counts": {
            "static_gaussians": int(model.static_tokens * model.gaussians_per_token),
            "dynamic_gaussians": int(model.dynamic_tokens * model.gaussians_per_token),
            "total_gaussians": int(model.num_tokens * model.gaussians_per_token),
        },
        "tensors": tensors,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote browser bundle manifest to {manifest_path}")
    return manifest_path


def export_browser_bundle(
    *,
    config_path: Path,
    output_dir: Path,
    state_dict_path: Path | None,
    sequence_index: int,
    window_start: int,
) -> Path:
    resolved = resolve_config(load_config_file(config_path))
    device = pick_device()
    print(f"Using device: {device}")

    model = build_model_from_config(resolved).to(device)
    if state_dict_path is not None:
        state_dict = _load_state_dict(state_dict_path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise ValueError(
                "Checkpoint did not match model strictly.\n"
                f"Missing keys: {sorted(missing)}\n"
                f"Unexpected keys: {sorted(unexpected)}"
            )

    sequence_data = _load_sequence_from_config(resolved, device=device, sequence_index=sequence_index)
    clip_indices = _clip_indices(
        frame_count=sequence_data.frame_count,
        train_frame_count=resolved["model"]["train_frame_count"],
        window_start=window_start,
        device=device,
    )
    clip_frames, clip_times = prepare_clip(sequence_data, clip_indices)
    model_input = _load_model_input(resolved, sequence_data, clip_frames, clip_times, device=device)
    export_id = export_id_from_config(resolved)
    return export_browser_bundle_from_model(
        model=model,
        resolved=resolved,
        sequence_data=sequence_data,
        clip_indices=clip_indices,
        clip_times=clip_times,
        model_input=model_input,
        output_dir=output_dir,
        config_path=config_path,
        state_dict_path=state_dict_path,
        export_id=export_id,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a browser-loadable Dynaworld static/dynamic split bundle. "
            "This saves refined token arrays plus the Gaussian decoder head MLP "
            "weights needed to decode splats in the browser."
        )
    )
    parser.add_argument("config", type=Path, help="Path to the Dynaworld JSONC train config.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write the browser bundle.")
    parser.add_argument(
        "--state-dict",
        type=Path,
        default=None,
        help="Optional model state_dict/checkpoint path. If omitted, exports the random-init model state.",
    )
    parser.add_argument(
        "--sequence-index",
        type=int,
        default=0,
        help="Sequence index to export when data.manifest_path is set. Default: 0.",
    )
    parser.add_argument(
        "--window-start",
        type=int,
        default=0,
        help="Deterministic start index for the exported clip window. Default: 0.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_browser_bundle(
        config_path=args.config,
        output_dir=args.output_dir,
        state_dict_path=args.state_dict,
        sequence_index=args.sequence_index,
        window_start=args.window_start,
    )


if __name__ == "__main__":
    main()
