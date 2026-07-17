from __future__ import annotations

from typing import Any

from rendering import pick_renderer_mode
from runtime_types import ResolvedRendererMode


def token_layout_detail_levels(model_cfg: dict[str, Any]) -> int:
    token_layout = model_cfg.get("token_layout")
    if token_layout is None:
        return 0
    return max(
        len(token_layout.get("static_detail_tokens") or []),
        len(token_layout.get("dynamic_detail_tokens") or []),
    )


def decoded_token_count_from_model_config(
    model_cfg: dict[str, Any],
    *,
    active_detail_level: int | None = None,
) -> int:
    token_layout = model_cfg.get("token_layout")
    if token_layout is None:
        return int(model_cfg["tokens"])

    static_detail_tokens = list(token_layout.get("static_detail_tokens") or [])
    dynamic_detail_tokens = list(token_layout.get("dynamic_detail_tokens") or [])
    detail_levels = token_layout_detail_levels(model_cfg)
    if active_detail_level is None:
        active_detail_level = token_layout.get("active_detail_level", detail_levels)
    if active_detail_level is None:
        active_detail_level = detail_levels
    active_detail_level = int(active_detail_level)
    if not 0 <= active_detail_level <= detail_levels:
        raise ValueError(
            f"model.token_layout active_detail_level must be between 0 and {detail_levels}, "
            f"got {active_detail_level}."
        )

    static_count = int(token_layout["static_core_tokens"]) + sum(
        int(value) for value in static_detail_tokens[:active_detail_level]
    )
    dynamic_count = int(token_layout["dynamic_core_tokens"]) + sum(
        int(value) for value in dynamic_detail_tokens[:active_detail_level]
    )
    return static_count + dynamic_count


def token_summary_from_model_config(model_cfg: dict[str, Any]) -> str:
    token_layout = model_cfg.get("token_layout")
    if token_layout is not None:
        active_level = token_layout.get("active_detail_level")
        decoded_tokens = decoded_token_count_from_model_config(model_cfg, active_detail_level=active_level)
        return (
            f"{decoded_tokens} active decoded 3DGS tokens "
            f"inside {int(model_cfg['tokens'])} total non-camera query tokens"
        )
    if not model_cfg["use_static_dynamic_split"]:
        return f"{model_cfg['tokens']} 3DGS tokens"
    return f"{model_cfg['static_tokens']} static + {model_cfg['dynamic_tokens']} dynamic 3DGS tokens"


def pick_renderer_mode_from_config(
    config: dict[str, Any],
    *,
    active_detail_level: int | None = None,
) -> tuple[ResolvedRendererMode, int]:
    model_cfg = config["model"]
    render_cfg = config["render"]
    decoded_token_count = decoded_token_count_from_model_config(
        model_cfg,
        active_detail_level=active_detail_level,
    )
    effective_gaussians = decoded_token_count * int(model_cfg["gaussians_per_token"])
    renderer_mode = pick_renderer_mode(
        renderer=str(render_cfg["renderer"]),
        gaussian_count=effective_gaussians,
        height=int(render_cfg["render_size"]),
        width=int(render_cfg["render_size"]),
        auto_dense_limit=int(render_cfg["auto_dense_limit"]),
    )
    return renderer_mode, effective_gaussians


__all__ = [
    "decoded_token_count_from_model_config",
    "pick_renderer_mode_from_config",
    "token_layout_detail_levels",
    "token_summary_from_model_config",
]
