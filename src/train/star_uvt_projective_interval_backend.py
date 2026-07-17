from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from star_uvt_render_configs import feature_tube_render_config_from_cfg, uvt_render_config_from_cfg
from star_uvt_runtime import ensure_star_uvt_on_path


PROJECTIVE_INTERVAL_BACKEND_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "sigma_px": 1.0,
    "image_width": None,
    "image_height": None,
    "tile_size": 16,
    "uv_padding": 0.0,
    "support_guard_padding": 0.0,
    "support_guard_policy": "fixed",
    "support_guard_bisect_steps": 8,
    "support_stale_overshoot_epsilon": 0.0,
    "support_stale_tail_alpha_epsilon": 0.0,
    "depth_padding": 0.0,
    "depth_epsilon": 1.0e-6,
    "refresh_policy": "cadence",
    "refresh_every": 1,
    "check_visibility": True,
    "allow_ambiguous_fallback": False,
    "allow_anisotropic_spatial_precision": False,
    "allow_depth_affine_uv": False,
    "fallback_render_mode": "error",
    "enforce_complexity_budget": False,
    "max_interval_to_dense_trace_sample_ratio": 1.0,
    "max_fallback_fraction": 0.20,
    "max_cells_per_active_set_group": 16,
}


@dataclass(frozen=True)
class ProjectiveCellIntervalBackendConfig:
    enabled: bool
    sigma_px: float
    image_width: int
    image_height: int
    tile_size: int
    uv_padding: float
    support_guard_padding: float
    support_guard_policy: str
    support_guard_bisect_steps: int
    support_stale_overshoot_epsilon: float
    support_stale_tail_alpha_epsilon: float
    depth_padding: float
    depth_epsilon: float
    refresh_policy: str
    refresh_every: int
    check_visibility: bool
    allow_ambiguous_fallback: bool
    allow_anisotropic_spatial_precision: bool
    allow_depth_affine_uv: bool
    fallback_render_mode: str
    enforce_complexity_budget: bool
    max_interval_to_dense_trace_sample_ratio: float
    max_fallback_fraction: float
    max_cells_per_active_set_group: int

    def __post_init__(self) -> None:
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("projective_interval image dimensions must be positive")
        if self.tile_size <= 0:
            raise ValueError("projective_interval.tile_size must be positive")
        if self.sigma_px <= 0.0:
            raise ValueError("projective_interval.sigma_px must be positive")
        if self.uv_padding < 0.0:
            raise ValueError("projective_interval.uv_padding must be non-negative")
        if self.support_guard_padding < 0.0:
            raise ValueError("projective_interval.support_guard_padding must be non-negative")
        if self.support_guard_policy not in {
            "fixed",
            "budgeted",
            "local_budgeted",
            "trace_budgeted",
            "slack_budgeted",
        }:
            raise ValueError(
                "projective_interval.support_guard_policy must be one of: fixed, budgeted, local_budgeted, "
                "trace_budgeted, slack_budgeted"
            )
        if self.support_guard_bisect_steps < 0:
            raise ValueError("projective_interval.support_guard_bisect_steps must be non-negative")
        if self.support_stale_overshoot_epsilon < 0.0:
            raise ValueError("projective_interval.support_stale_overshoot_epsilon must be non-negative")
        if self.support_stale_tail_alpha_epsilon < 0.0:
            raise ValueError("projective_interval.support_stale_tail_alpha_epsilon must be non-negative")
        if self.depth_padding < 0.0:
            raise ValueError("projective_interval.depth_padding must be non-negative")
        if self.depth_epsilon < 0.0:
            raise ValueError("projective_interval.depth_epsilon must be non-negative")
        if self.refresh_policy not in {"cadence", "measured"}:
            raise ValueError("projective_interval.refresh_policy must be one of: cadence, measured")
        if self.refresh_every < 1:
            raise ValueError("projective_interval.refresh_every must be positive")
        if self.fallback_render_mode not in {"error", "mixed", "reference"}:
            raise ValueError("projective_interval.fallback_render_mode must be one of: error, mixed, reference")
        if self.max_interval_to_dense_trace_sample_ratio < 0.0:
            raise ValueError("projective_interval.max_interval_to_dense_trace_sample_ratio must be non-negative")
        if self.max_fallback_fraction < 0.0:
            raise ValueError("projective_interval.max_fallback_fraction must be non-negative")
        if self.max_cells_per_active_set_group < 1:
            raise ValueError("projective_interval.max_cells_per_active_set_group must be positive")

    @property
    def support_uv_padding(self) -> float:
        return float(self.uv_padding + self.support_guard_padding)


def _projective_interval_tile_capacity(cfg: dict[str, Any]) -> int:
    return int(cfg["feature_uvt"]["tile_capacity"])


def resolve_projective_interval_backend_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    section = cfg["feature_uvt"].setdefault("projective_interval", {})
    for key, value in PROJECTIVE_INTERVAL_BACKEND_DEFAULTS.items():
        section.setdefault(key, value)
    if section["image_width"] is not None and int(section["image_width"]) <= 0:
        raise ValueError("feature_uvt.projective_interval.image_width must be positive")
    if section["image_height"] is not None and int(section["image_height"]) <= 0:
        raise ValueError("feature_uvt.projective_interval.image_height must be positive")
    ProjectiveCellIntervalBackendConfig(
        enabled=bool(section["enabled"]),
        sigma_px=float(section["sigma_px"]),
        image_width=int(section["image_width"] or cfg["data"]["target_size"]),
        image_height=int(section["image_height"] or cfg["data"]["target_size"]),
        tile_size=int(section["tile_size"]),
        uv_padding=float(section["uv_padding"]),
        support_guard_padding=float(section["support_guard_padding"]),
        support_guard_policy=str(section["support_guard_policy"]),
        support_guard_bisect_steps=int(section["support_guard_bisect_steps"]),
        support_stale_overshoot_epsilon=float(section["support_stale_overshoot_epsilon"]),
        support_stale_tail_alpha_epsilon=float(section["support_stale_tail_alpha_epsilon"]),
        depth_padding=float(section["depth_padding"]),
        depth_epsilon=float(section["depth_epsilon"]),
        refresh_policy=str(section["refresh_policy"]),
        refresh_every=int(section["refresh_every"]),
        check_visibility=bool(section["check_visibility"]),
        allow_ambiguous_fallback=bool(section["allow_ambiguous_fallback"]),
        allow_anisotropic_spatial_precision=bool(section["allow_anisotropic_spatial_precision"]),
        allow_depth_affine_uv=bool(section["allow_depth_affine_uv"]),
        fallback_render_mode=str(section["fallback_render_mode"]),
        enforce_complexity_budget=bool(section["enforce_complexity_budget"]),
        max_interval_to_dense_trace_sample_ratio=float(section["max_interval_to_dense_trace_sample_ratio"]),
        max_fallback_fraction=float(section["max_fallback_fraction"]),
        max_cells_per_active_set_group=int(section["max_cells_per_active_set_group"]),
    )
    return section


def projective_cell_interval_backend_config_from_cfg(
    cfg: dict[str, Any],
    feature_config: Any | None = None,
) -> ProjectiveCellIntervalBackendConfig:
    section = resolve_projective_interval_backend_settings(cfg)
    if feature_config is None:
        feature_config = feature_tube_render_config_from_cfg(cfg)
    return ProjectiveCellIntervalBackendConfig(
        enabled=bool(section["enabled"]),
        sigma_px=float(section["sigma_px"]),
        image_width=int(section["image_width"] or feature_config.width),
        image_height=int(section["image_height"] or feature_config.height),
        tile_size=int(section["tile_size"]),
        uv_padding=float(section["uv_padding"]),
        support_guard_padding=float(section["support_guard_padding"]),
        support_guard_policy=str(section["support_guard_policy"]),
        support_guard_bisect_steps=int(section["support_guard_bisect_steps"]),
        support_stale_overshoot_epsilon=float(section["support_stale_overshoot_epsilon"]),
        support_stale_tail_alpha_epsilon=float(section["support_stale_tail_alpha_epsilon"]),
        depth_padding=float(section["depth_padding"]),
        depth_epsilon=float(section["depth_epsilon"]),
        refresh_policy=str(section["refresh_policy"]),
        refresh_every=int(section["refresh_every"]),
        check_visibility=bool(section["check_visibility"]),
        allow_ambiguous_fallback=bool(section["allow_ambiguous_fallback"]),
        allow_anisotropic_spatial_precision=bool(section["allow_anisotropic_spatial_precision"]),
        allow_depth_affine_uv=bool(section["allow_depth_affine_uv"]),
        fallback_render_mode=str(section["fallback_render_mode"]),
        enforce_complexity_budget=bool(section["enforce_complexity_budget"]),
        max_interval_to_dense_trace_sample_ratio=float(section["max_interval_to_dense_trace_sample_ratio"]),
        max_fallback_fraction=float(section["max_fallback_fraction"]),
        max_cells_per_active_set_group=int(section["max_cells_per_active_set_group"]),
    )


def require_projective_interval_atlas_producer(
    cfg: dict[str, Any],
    feature_config: Any | None = None,
    *,
    owner: str,
    producer_available: bool,
) -> ProjectiveCellIntervalBackendConfig:
    """Return projective interval config and guard enabled routes.

    The production feature-tube trainer currently emits affine UVT feature tubes,
    not a ``ProjectiveTraceCellTraceAtlas``. Enabling the projective interval
    backend without such a producer would silently run the wrong renderer, so
    production callers must make the producer contract explicit.
    """

    backend_config = projective_cell_interval_backend_config_from_cfg(cfg, feature_config)
    if backend_config.enabled and not producer_available:
        raise RuntimeError(
            f"{owner} cannot enable feature_uvt.projective_interval yet: "
            "a ProjectiveTraceCellTraceAtlas producer is required before the "
            "projective interval renderer can replace the current UVT feature-tube renderer"
        )
    return backend_config


def _projective_interval_atlas_overflows_tile_capacity(
    atlas: Any,
    times: torch.Tensor,
    cfg: dict[str, Any],
    backend_config: ProjectiveCellIntervalBackendConfig,
) -> bool:
    ensure_star_uvt_on_path(include_dynaworld_root=False)
    from torch_gsplat_bridge_star_uvt import pack_projective_trace_tile_time_bins

    bins = pack_projective_trace_tile_time_bins(
        atlas.cells,
        image_width=backend_config.image_width,
        image_height=backend_config.image_height,
        frames=int(times.numel()),
        tile_x=backend_config.tile_size,
        tile_y=backend_config.tile_size,
        tile_t=int(times.numel()),
        tile_capacity=_projective_interval_tile_capacity(cfg),
        allow_fallback_cells=True,
    )
    return bool(torch.any(bins.tile_overflow > 0).item())


def _projective_interval_overflow_tile_coords(
    atlas: Any,
    times: torch.Tensor,
    cfg: dict[str, Any],
    backend_config: ProjectiveCellIntervalBackendConfig,
) -> set[tuple[int, int]]:
    ensure_star_uvt_on_path(include_dynaworld_root=False)
    from torch_gsplat_bridge_star_uvt import pack_projective_trace_tile_time_bins

    bins = pack_projective_trace_tile_time_bins(
        atlas.cells,
        image_width=backend_config.image_width,
        image_height=backend_config.image_height,
        frames=int(times.numel()),
        tile_x=backend_config.tile_size,
        tile_y=backend_config.tile_size,
        tile_t=int(times.numel()),
        tile_capacity=_projective_interval_tile_capacity(cfg),
        allow_fallback_cells=True,
    )
    tiles_x = (int(backend_config.image_width) + int(backend_config.tile_size) - 1) // int(backend_config.tile_size)
    tiles_y = (int(backend_config.image_height) + int(backend_config.tile_size) - 1) // int(backend_config.tile_size)
    coords: set[tuple[int, int]] = set()
    for flat_tile_id in bins.tile_overflow.detach().cpu().nonzero(as_tuple=False).reshape(-1).tolist():
        tile_id = int(flat_tile_id) % int(tiles_x * tiles_y)
        coords.add((tile_id % tiles_x, tile_id // tiles_x))
    return coords


def _projective_interval_atlas_with_cells(template: Any, cells: list[Any]) -> Any:
    return type(template)(
        coeffs=template.coeffs,
        opacity=template.opacity,
        color=template.color,
        cells=sorted(cells, key=lambda cell: (cell.start, cell.stop, cell.tile_v, cell.tile_u)),
        source_window_indices=template.source_window_indices,
        source_primitive_ids=template.source_primitive_ids,
        active_start=template.active_start,
        active_stop=template.active_stop,
        opacity_time_coeffs=template.opacity_time_coeffs,
        spatial_precision_uv=template.spatial_precision_uv,
        depth_affine_uv=template.depth_affine_uv,
    )


def _mix_projective_interval_target_and_base_tiles(
    *,
    target_atlas: Any,
    base_atlas: Any,
    base_tile_coords: set[tuple[int, int]],
) -> Any:
    if not base_tile_coords:
        return target_atlas
    mixed_cells = [
        cell
        for cell in target_atlas.cells
        if (int(cell.tile_u), int(cell.tile_v)) not in base_tile_coords
    ]
    mixed_cells.extend(
        cell
        for cell in base_atlas.cells
        if (int(cell.tile_u), int(cell.tile_v)) in base_tile_coords
    )
    return _projective_interval_atlas_with_cells(target_atlas, mixed_cells)


def _filter_projective_interval_cell_by_primitive_ids(cell: Any, selected_ids: set[int]) -> Any | None:
    kept: list[tuple[int, tuple[float, float]]] = [
        (int(primitive_id), depth_interval)
        for primitive_id, depth_interval in zip(cell.ordered_primitive_ids, cell.depth_intervals)
        if int(primitive_id) in selected_ids
    ]
    if not kept:
        return None
    ordered_ids = tuple(primitive_id for primitive_id, _depth_interval in kept)
    return type(cell)(
        tile_u=cell.tile_u,
        tile_v=cell.tile_v,
        start=cell.start,
        stop=cell.stop,
        primitive_ids=tuple(sorted(set(ordered_ids))),
        ordered_primitive_ids=ordered_ids,
        depth_intervals=tuple(depth_interval for _primitive_id, depth_interval in kept),
        fallback=cell.fallback,
        fallback_reasons=cell.fallback_reasons,
    )


def _projective_interval_tile_primitive_ids(atlas: Any, tile_coord: tuple[int, int]) -> set[int]:
    return {
        int(primitive_id)
        for cell in atlas.cells
        if (int(cell.tile_u), int(cell.tile_v)) == tile_coord
        for primitive_id in cell.ordered_primitive_ids
    }


def _projective_interval_tile_support_event_distances(
    atlas: Any,
    times: torch.Tensor,
    *,
    tile_coord: tuple[int, int],
    uv_padding: float,
    tile_size: int,
) -> dict[int, float]:
    ensure_star_uvt_on_path(include_dynaworld_root=False)
    from torch_gsplat_bridge_star_uvt import eval_projective_trace_cell_torch

    coeffs_cpu = atlas.coeffs.detach().cpu().contiguous()
    times_cpu = times.detach().cpu().contiguous()
    samples = eval_projective_trace_cell_torch(coeffs_cpu, times_cpu)
    tile_u, tile_v = (int(tile_coord[0]), int(tile_coord[1]))
    tile_left = float(tile_u * int(tile_size))
    tile_right = float((tile_u + 1) * int(tile_size))
    tile_top = float(tile_v * int(tile_size))
    tile_bottom = float((tile_v + 1) * int(tile_size))
    distances: dict[int, float] = {}
    for cell in atlas.cells:
        if (int(cell.tile_u), int(cell.tile_v)) != (tile_u, tile_v):
            continue
        start = max(0, int(cell.start))
        stop = min(int(times_cpu.numel()), int(cell.stop))
        if start >= stop:
            continue
        for primitive_id in cell.primitive_ids:
            primitive_id = int(primitive_id)
            if primitive_id < 0 or primitive_id >= int(samples.shape[0]):
                continue
            best = distances.get(primitive_id, float("inf"))
            for sample_index in range(start, stop):
                u, v, _depth, valid_sign = samples[primitive_id, sample_index]
                if float(valid_sign.item()) == 0.0:
                    continue
                u_min = float(u.item()) - float(uv_padding)
                u_max = float(u.item()) + float(uv_padding)
                v_min = float(v.item()) - float(uv_padding)
                v_max = float(v.item()) + float(uv_padding)
                distance = max(
                    tile_left - u_max,
                    u_min - tile_right,
                    tile_top - v_max,
                    v_min - tile_bottom,
                    0.0,
                )
                best = min(best, float(distance))
            distances[primitive_id] = best
    return distances


def _mix_projective_interval_target_and_base_traces(
    *,
    target_atlas: Any,
    base_atlas: Any,
    base_tile_coords: set[tuple[int, int]],
    tile_capacity: int,
    times: torch.Tensor | None = None,
    base_uv_padding: float = 0.0,
    tile_size: int = 1,
    rank_by_support_event_distance: bool = False,
) -> Any:
    if not base_tile_coords:
        return target_atlas
    mixed_cells = [
        cell
        for cell in target_atlas.cells
        if (int(cell.tile_u), int(cell.tile_v)) not in base_tile_coords
    ]
    for tile_coord in sorted(base_tile_coords):
        selected_ids = _projective_interval_tile_primitive_ids(base_atlas, tile_coord)
        if len(selected_ids) > int(tile_capacity):
            mixed_cells.extend(
                cell
                for cell in base_atlas.cells
                if (int(cell.tile_u), int(cell.tile_v)) == tile_coord
            )
            continue
        target_ids = _projective_interval_tile_primitive_ids(target_atlas, tile_coord)
        extra_ids = list(target_ids - selected_ids)
        if rank_by_support_event_distance:
            if times is None:
                raise ValueError("times are required for support-event-distance trace budgeting")
            distances = _projective_interval_tile_support_event_distances(
                target_atlas,
                times,
                tile_coord=tile_coord,
                uv_padding=float(base_uv_padding),
                tile_size=int(tile_size),
            )
            extra_ids.sort(key=lambda primitive_id: (distances.get(int(primitive_id), float("inf")), int(primitive_id)))
        else:
            extra_ids.sort()
        for primitive_id in extra_ids:
            if len(selected_ids) >= int(tile_capacity):
                break
            selected_ids.add(int(primitive_id))
        for cell in target_atlas.cells:
            if (int(cell.tile_u), int(cell.tile_v)) != tile_coord:
                continue
            filtered = _filter_projective_interval_cell_by_primitive_ids(cell, selected_ids)
            if filtered is not None:
                mixed_cells.append(filtered)
    return _projective_interval_atlas_with_cells(target_atlas, mixed_cells)


def _projective_interval_uvt_alpha_support_padding(
    ma: torch.Tensor,
    q_uvt: torch.Tensor,
    opacity: torch.Tensor,
    times: torch.Tensor,
    *,
    alpha_threshold: float,
) -> float:
    if alpha_threshold <= 0.0 or int(q_uvt.shape[0]) == 0:
        return 0.0

    q_uu = q_uvt[:, 0]
    q_uv = q_uvt[:, 1]
    q_ut = q_uvt[:, 2]
    q_vv = q_uvt[:, 3]
    q_vt = q_uvt[:, 4]
    q_tt = q_uvt[:, 5]
    det = q_uu * q_vv - q_uv.square()
    if bool(torch.any(det <= 0.0).detach().cpu().item()):
        return 0.0

    inv00 = q_vv / det
    inv01 = -q_uv / det
    inv11 = q_uu / det
    a_inv_b_u = inv00 * q_ut + inv01 * q_vt
    a_inv_b_v = inv01 * q_ut + inv11 * q_vt
    temporal_precision = (q_tt - (q_ut * a_inv_b_u + q_vt * a_inv_b_v)).clamp_min(0.0)
    temporal_envelope = torch.exp(
        -0.5 * temporal_precision.reshape(-1, 1) * (times.reshape(1, -1) - ma[:, 2:3]).square()
    )
    max_effective_opacity = (opacity.reshape(-1, 1) * temporal_envelope).detach().amax(dim=1)
    radius2 = (2.0 * torch.log((max_effective_opacity / float(alpha_threshold)).clamp_min(1.0))).clamp_min(0.0)
    support_u = torch.sqrt((radius2 * inv00.detach().clamp_min(0.0)).clamp_min(0.0))
    support_v = torch.sqrt((radius2 * inv11.detach().clamp_min(0.0)).clamp_min(0.0))
    return float(torch.stack((support_u, support_v), dim=0).amax().detach().cpu().item())


def _compile_projective_cell_interval_atlas_with_support_padding(
    ma: torch.Tensor,
    q_uvt: torch.Tensor,
    depth0: torch.Tensor,
    depth_beta: torch.Tensor,
    opacity: torch.Tensor,
    color: torch.Tensor,
    times: torch.Tensor,
    cfg: dict[str, Any],
    *,
    feature_config: Any,
    backend_config: ProjectiveCellIntervalBackendConfig,
    support_uv_padding: float,
    primitive_ids: torch.Tensor | list[int] | tuple[int, ...] | None,
    temporal_mode: str,
) -> Any:
    ensure_star_uvt_on_path(include_dynaworld_root=False)
    from torch_gsplat_bridge_star_uvt import uvt_tubes_to_projective_trace_cell_atlas

    return uvt_tubes_to_projective_trace_cell_atlas(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        sigma_px=backend_config.sigma_px,
        image_width=backend_config.image_width,
        image_height=backend_config.image_height,
        tile_size=backend_config.tile_size,
        primitive_ids=primitive_ids,
        uv_padding=float(support_uv_padding),
        depth_padding=backend_config.depth_padding,
        alpha_threshold=float(feature_config.alpha_threshold),
        temporal_mode=temporal_mode,
        require_isotropic_spatial=not backend_config.allow_anisotropic_spatial_precision,
        auto_support_padding_from_alpha=backend_config.allow_anisotropic_spatial_precision,
        allow_depth_affine_uv=backend_config.allow_depth_affine_uv,
        stratify_visibility=backend_config.check_visibility,
        mark_visibility_fallback=backend_config.allow_ambiguous_fallback,
        depth_epsilon=backend_config.depth_epsilon,
    )


def _compile_projective_cell_interval_atlas_with_budgeted_support(
    ma: torch.Tensor,
    q_uvt: torch.Tensor,
    depth0: torch.Tensor,
    depth_beta: torch.Tensor,
    opacity: torch.Tensor,
    color: torch.Tensor,
    times: torch.Tensor,
    cfg: dict[str, Any],
    *,
    feature_config: Any,
    backend_config: ProjectiveCellIntervalBackendConfig,
    primitive_ids: torch.Tensor | list[int] | tuple[int, ...] | None,
    temporal_mode: str,
) -> tuple[Any, float]:
    def build(support_uv_padding: float) -> Any:
        return _compile_projective_cell_interval_atlas_with_support_padding(
            ma,
            q_uvt,
            depth0,
            depth_beta,
            opacity,
            color,
            times,
            cfg,
            feature_config=feature_config,
            backend_config=backend_config,
            support_uv_padding=float(support_uv_padding),
            primitive_ids=primitive_ids,
            temporal_mode=temporal_mode,
        )

    minimum_support_padding = 0.0
    if backend_config.allow_anisotropic_spatial_precision:
        minimum_support_padding = _projective_interval_uvt_alpha_support_padding(
            ma,
            q_uvt,
            opacity,
            times,
            alpha_threshold=float(feature_config.alpha_threshold),
        )
    target_padding = max(float(backend_config.support_uv_padding), float(minimum_support_padding))
    target_atlas = build(target_padding)
    if backend_config.support_guard_policy == "fixed" or backend_config.support_guard_padding <= 0.0:
        return target_atlas, target_padding

    target_overflows = _projective_interval_atlas_overflows_tile_capacity(target_atlas, times, cfg, backend_config)
    if not target_overflows:
        return target_atlas, target_padding

    base_padding = max(float(backend_config.uv_padding), float(minimum_support_padding))
    best_padding = base_padding
    best_atlas = build(base_padding)
    if _projective_interval_atlas_overflows_tile_capacity(best_atlas, times, cfg, backend_config):
        return best_atlas, best_padding

    overflow_tile_coords = _projective_interval_overflow_tile_coords(target_atlas, times, cfg, backend_config)
    if backend_config.support_guard_policy in {"trace_budgeted", "slack_budgeted"}:
        trace_mixed_atlas = _mix_projective_interval_target_and_base_traces(
            target_atlas=target_atlas,
            base_atlas=best_atlas,
            base_tile_coords=overflow_tile_coords,
            tile_capacity=_projective_interval_tile_capacity(cfg),
            times=times,
            base_uv_padding=base_padding,
            tile_size=backend_config.tile_size,
            rank_by_support_event_distance=backend_config.support_guard_policy == "slack_budgeted",
        )
        if not _projective_interval_atlas_overflows_tile_capacity(trace_mixed_atlas, times, cfg, backend_config):
            return trace_mixed_atlas, target_padding

    if backend_config.support_guard_policy in {"local_budgeted", "trace_budgeted", "slack_budgeted"}:
        mixed_atlas = _mix_projective_interval_target_and_base_tiles(
            target_atlas=target_atlas,
            base_atlas=best_atlas,
            base_tile_coords=overflow_tile_coords,
        )
        if not _projective_interval_atlas_overflows_tile_capacity(mixed_atlas, times, cfg, backend_config):
            return mixed_atlas, target_padding

    low_guard = 0.0
    high_guard = float(backend_config.support_guard_padding)
    for _ in range(int(backend_config.support_guard_bisect_steps)):
        mid_guard = 0.5 * (low_guard + high_guard)
        if mid_guard <= low_guard + 1.0e-6:
            break
        candidate_padding = base_padding + mid_guard
        candidate_atlas = build(candidate_padding)
        if _projective_interval_atlas_overflows_tile_capacity(candidate_atlas, times, cfg, backend_config):
            high_guard = mid_guard
            continue
        low_guard = mid_guard
        best_padding = candidate_padding
        best_atlas = candidate_atlas

    return best_atlas, best_padding


def make_projective_cell_interval_atlas_from_uvt_tubes(
    ma: torch.Tensor,
    q_uvt: torch.Tensor,
    depth0: torch.Tensor,
    depth_beta: torch.Tensor,
    opacity: torch.Tensor,
    color: torch.Tensor,
    times: torch.Tensor,
    cfg: dict[str, Any],
    *,
    feature_config: Any | None = None,
    backend_config: ProjectiveCellIntervalBackendConfig | None = None,
    primitive_ids: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    temporal_mode: str = "trace",
) -> Any:
    """Compile compatible STAR UVT tubes into the projective interval atlas.

    This is the first production-facing atlas producer. By default it keeps the
    legacy exactness checks: compatible tubes have isotropic spatial precision
    matching ``sigma_px`` and no pixel-varying depth slope. Setting
    ``allow_anisotropic_spatial_precision`` in the backend config opts into
    carrying the full UV precision block through the projective trace atlas.
    Residual temporal opacity is preserved as a trace envelope in the atlas by
    default.
    """

    if feature_config is None:
        feature_config = feature_tube_render_config_from_cfg(cfg)
    if backend_config is None:
        backend_config = projective_cell_interval_backend_config_from_cfg(cfg, feature_config)
    atlas, _support_uv_padding = _compile_projective_cell_interval_atlas_with_budgeted_support(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        cfg,
        feature_config=feature_config,
        backend_config=backend_config,
        primitive_ids=primitive_ids,
        temporal_mode=temporal_mode,
    )
    return atlas


def make_projective_cell_interval_trainer_state_from_uvt_tubes(
    ma: torch.Tensor,
    q_uvt: torch.Tensor,
    depth0: torch.Tensor,
    depth_beta: torch.Tensor,
    opacity: torch.Tensor,
    color: torch.Tensor,
    times: torch.Tensor,
    cfg: dict[str, Any],
    *,
    feature_config: Any | None = None,
    uvt_config: Any | None = None,
    primitive_ids: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    temporal_mode: str = "trace",
) -> Any:
    backend_config = projective_cell_interval_backend_config_from_cfg(cfg, feature_config)
    if feature_config is None:
        feature_config = feature_tube_render_config_from_cfg(cfg)
    atlas, support_uv_padding = _compile_projective_cell_interval_atlas_with_budgeted_support(
        ma,
        q_uvt,
        depth0,
        depth_beta,
        opacity,
        color,
        times,
        cfg,
        feature_config=feature_config,
        backend_config=backend_config,
        primitive_ids=primitive_ids,
        temporal_mode=temporal_mode,
    )
    return make_projective_cell_interval_trainer_state(
        atlas,
        times,
        cfg,
        feature_config=feature_config,
        uvt_config=uvt_config,
        support_uv_padding=support_uv_padding,
    )


def make_projective_cell_interval_live_atlas_from_uvt_tubes(
    ma: torch.Tensor,
    q_uvt: torch.Tensor,
    depth0: torch.Tensor,
    depth_beta: torch.Tensor,
    opacity: torch.Tensor,
    color: torch.Tensor,
    cfg: dict[str, Any],
    *,
    reference_atlas: Any,
    feature_config: Any | None = None,
    backend_config: ProjectiveCellIntervalBackendConfig | None = None,
    temporal_mode: str = "trace",
    spatial_precision_rtol: float = 1.0e-4,
    spatial_precision_atol: float = 1.0e-5,
    depth_spatial_atol: float = 1.0e-6,
    temporal_precision_atol: float = 1.0e-6,
) -> Any:
    """Update differentiable atlas tensors while reusing compiled cell metadata.

    This helper is intentionally narrower than
    ``make_projective_cell_interval_atlas_from_uvt_tubes``. It is for trainer
    cache reuse after a compatible atlas has already been compiled. The
    reference atlas must use source primitive ids that are direct row indices
    into the supplied UVT tensors.
    """

    del feature_config
    if backend_config is None:
        backend_config = projective_cell_interval_backend_config_from_cfg(cfg)
    if temporal_mode not in {"trace", "require_zero"}:
        raise ValueError("live atlas updates support temporal_mode='trace' or 'require_zero'")
    if ma.ndim != 2 or ma.shape[1] != 3:
        raise ValueError("ma must have shape [N,3]")
    tube_count = int(ma.shape[0])
    if q_uvt.shape != (tube_count, 6):
        raise ValueError("q_uvt must have shape [N,6]")
    if depth0.shape != (tube_count,) or depth_beta.shape != (tube_count, 3):
        raise ValueError("depth0/depth_beta must have shapes [N] and [N,3]")
    if opacity.shape != (tube_count,):
        raise ValueError("opacity must have shape [N]")
    if color.ndim != 2 or color.shape[0] != tube_count:
        raise ValueError("color must have shape [N,C]")
    if ma.dtype != torch.float32:
        raise ValueError("ma must be float32")
    for name, tensor in (
        ("q_uvt", q_uvt),
        ("depth0", depth0),
        ("depth_beta", depth_beta),
        ("opacity", opacity),
        ("color", color),
    ):
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must be float32")
        if tensor.device != ma.device:
            raise ValueError(f"{name} must be on the same device as ma")

    source_ids = tuple(int(item) for item in reference_atlas.source_primitive_ids)
    if any(source_id < 0 or source_id >= tube_count for source_id in source_ids):
        raise ValueError("reference_atlas.source_primitive_ids must be direct row indices for live UVT updates")
    ensure_star_uvt_on_path(include_dynaworld_root=False)
    from torch_gsplat_bridge_star_uvt import ProjectiveTraceCellTraceAtlas

    if not source_ids:
        empty_coeffs = torch.empty((0, 9), dtype=torch.float32, device=ma.device)
        empty_opacity = torch.empty((0,), dtype=torch.float32, device=ma.device)
        empty_color = torch.empty((0, int(color.shape[1])), dtype=torch.float32, device=ma.device)
        empty_opacity_time = torch.empty((0, 3), dtype=torch.float32, device=ma.device)
        empty_spatial_precision = torch.empty((0, 3), dtype=torch.float32, device=ma.device)
        empty_depth_affine = (
            torch.empty((0, 6), dtype=torch.float32, device=ma.device)
            if reference_atlas.depth_affine_uv is not None
            else None
        )
        return ProjectiveTraceCellTraceAtlas(
            coeffs=empty_coeffs,
            opacity=empty_opacity,
            color=empty_color,
            cells=reference_atlas.cells,
            source_window_indices=reference_atlas.source_window_indices,
            source_primitive_ids=reference_atlas.source_primitive_ids,
            active_start=reference_atlas.active_start,
            active_stop=reference_atlas.active_stop,
            opacity_time_coeffs=empty_opacity_time,
            spatial_precision_uv=empty_spatial_precision,
            depth_affine_uv=empty_depth_affine,
        )

    row_ids = torch.tensor(source_ids, dtype=torch.long, device=ma.device)
    ma_sel = ma.index_select(0, row_ids)
    q_sel = q_uvt.index_select(0, row_ids)
    depth0_sel = depth0.index_select(0, row_ids)
    depth_beta_sel = depth_beta.index_select(0, row_ids)
    opacity_sel = opacity.index_select(0, row_ids)
    color_sel = color.index_select(0, row_ids)

    q_uu = q_sel[:, 0]
    q_uv = q_sel[:, 1]
    q_ut = q_sel[:, 2]
    q_vv = q_sel[:, 3]
    q_vt = q_sel[:, 4]
    q_tt = q_sel[:, 5]
    det = q_uu * q_vv - q_uv.square()
    if bool(torch.any(det <= 0.0).detach().cpu().item()):
        raise ValueError("q_uvt spatial UV precision block must be positive definite")
    inv00 = q_vv / det
    inv01 = -q_uv / det
    inv11 = q_uu / det
    a_inv_b_u = inv00 * q_ut + inv01 * q_vt
    a_inv_b_v = inv01 * q_ut + inv11 * q_vt
    velocity_u = -a_inv_b_u
    velocity_v = -a_inv_b_v
    temporal_precision = q_tt - (q_ut * a_inv_b_u + q_vt * a_inv_b_v)
    if bool(torch.any(temporal_precision < -float(temporal_precision_atol)).detach().cpu().item()):
        raise ValueError("q_uvt temporal Schur precision must be non-negative")
    temporal_precision = temporal_precision.clamp_min(0.0)
    if temporal_mode == "require_zero" and bool(
        torch.any(temporal_precision > float(temporal_precision_atol)).detach().cpu().item()
    ):
        raise ValueError("temporal_mode='require_zero' requires zero residual temporal precision")

    if not backend_config.allow_anisotropic_spatial_precision:
        target_precision = torch.full_like(q_uu, 1.0 / float(backend_config.sigma_px * backend_config.sigma_px))
        spatial_ok = (
            torch.isclose(q_uu, target_precision, rtol=float(spatial_precision_rtol), atol=float(spatial_precision_atol))
            & torch.isclose(q_vv, target_precision, rtol=float(spatial_precision_rtol), atol=float(spatial_precision_atol))
            & (q_uv.abs() <= float(spatial_precision_atol))
        )
        if not bool(torch.all(spatial_ok).detach().cpu().item()):
            raise ValueError("q_uvt spatial precision must match the atlas isotropic sigma_px")
    allow_depth_affine_uv = reference_atlas.depth_affine_uv is not None
    has_depth_spatial = bool(torch.any(depth_beta_sel[:, :2].abs() > float(depth_spatial_atol)).detach().cpu().item())
    if has_depth_spatial and not allow_depth_affine_uv:
        raise ValueError("projective cell atlas lowering requires depth_beta[:,0:2] near zero")

    t_center = ma_sel[:, 2]
    depth_slope = depth_beta_sel[:, 2] + depth_beta_sel[:, 0] * velocity_u + depth_beta_sel[:, 1] * velocity_v
    zeros = torch.zeros_like(t_center)
    coeffs = torch.stack(
        (
            ma_sel[:, 0] - velocity_u * t_center,
            velocity_u,
            zeros,
            ma_sel[:, 1] - velocity_v * t_center,
            velocity_v,
            zeros,
            depth0_sel - depth_slope * t_center,
            depth_slope,
            zeros,
        ),
        dim=-1,
    ).contiguous()
    opacity_time_coeffs = torch.stack(
        (
            temporal_precision * t_center.square(),
            -2.0 * temporal_precision * t_center,
            temporal_precision,
        ),
        dim=-1,
    ).contiguous()
    spatial_precision_uv = torch.stack((q_uu, q_uv, q_vv), dim=-1).contiguous()
    depth_affine_uv = (
        torch.stack(
            (
                depth_beta_sel[:, 0],
                zeros,
                zeros,
                depth_beta_sel[:, 1],
                zeros,
                zeros,
            ),
            dim=-1,
        ).contiguous()
        if allow_depth_affine_uv
        else None
    )

    return ProjectiveTraceCellTraceAtlas(
        coeffs=coeffs,
        opacity=opacity_sel.contiguous(),
        color=color_sel.contiguous(),
        cells=reference_atlas.cells,
        source_window_indices=reference_atlas.source_window_indices,
        source_primitive_ids=reference_atlas.source_primitive_ids,
        active_start=reference_atlas.active_start,
        active_stop=reference_atlas.active_stop,
        opacity_time_coeffs=opacity_time_coeffs,
        spatial_precision_uv=spatial_precision_uv,
        depth_affine_uv=depth_affine_uv,
    )


def make_projective_cell_interval_trainer_state(
    atlas: Any,
    times: torch.Tensor,
    cfg: dict[str, Any],
    *,
    feature_config: Any | None = None,
    uvt_config: Any | None = None,
    support_uv_padding: float | None = None,
) -> Any:
    backend_config = projective_cell_interval_backend_config_from_cfg(cfg, feature_config)
    ensure_star_uvt_on_path(include_dynaworld_root=False)
    from research_project.trainer_harness import ProjectiveCellIntervalTrainerState

    if uvt_config is None:
        uvt_config = uvt_render_config_from_cfg(cfg, feature_config)
    return ProjectiveCellIntervalTrainerState(
        atlas=atlas,
        times=times,
        config=uvt_config,
        sigma_px=backend_config.sigma_px,
        image_width=backend_config.image_width,
        image_height=backend_config.image_height,
        tile_size=backend_config.tile_size,
        uv_padding=backend_config.uv_padding,
        support_uv_padding=backend_config.support_uv_padding if support_uv_padding is None else float(support_uv_padding),
        depth_padding=backend_config.depth_padding,
        depth_epsilon=backend_config.depth_epsilon,
        refresh_every=backend_config.refresh_every,
        budget_support_guard=backend_config.support_guard_policy in {
            "budgeted",
            "local_budgeted",
            "trace_budgeted",
            "slack_budgeted",
        },
        support_guard_policy=backend_config.support_guard_policy,
        support_guard_bisect_steps=backend_config.support_guard_bisect_steps,
        support_stale_overshoot_epsilon=backend_config.support_stale_overshoot_epsilon,
        support_stale_tail_alpha_epsilon=backend_config.support_stale_tail_alpha_epsilon,
        check_visibility=backend_config.check_visibility,
        allow_ambiguous_fallback=backend_config.allow_ambiguous_fallback,
        fallback_render_mode=backend_config.fallback_render_mode,
        enforce_complexity_budget=backend_config.enforce_complexity_budget,
        max_interval_to_dense_trace_sample_ratio=backend_config.max_interval_to_dense_trace_sample_ratio,
        max_fallback_fraction=backend_config.max_fallback_fraction,
        max_cells_per_active_set_group=backend_config.max_cells_per_active_set_group,
    )


__all__ = [
    "PROJECTIVE_INTERVAL_BACKEND_DEFAULTS",
    "ProjectiveCellIntervalBackendConfig",
    "make_projective_cell_interval_atlas_from_uvt_tubes",
    "make_projective_cell_interval_live_atlas_from_uvt_tubes",
    "make_projective_cell_interval_trainer_state",
    "make_projective_cell_interval_trainer_state_from_uvt_tubes",
    "projective_cell_interval_backend_config_from_cfg",
    "require_projective_interval_atlas_producer",
    "resolve_projective_interval_backend_settings",
]
