"""Deterministic CPU exactness and visibility-stress ablation for WorldFoam.

This suite is deliberately representation-level evidence.  It evaluates the
ordered Beer--Lambert transfer integral on a two-dimensional ray section of
the eight synthetic scenes and seven camera programs named in the WorldFoam
paper plan.  The dense reference, depth-layer approximation, adaptive route,
representative-depth ordering baseline, and depth-marginal baseline all see
the same physical density and color fields.

The report does *not* claim native-kernel speed, allocator behavior, or public
data quality.  Those are separate fail-closed paper gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


SCHEMA_VERSION = 1
SUITE_ID = "worldfoam-synthetic-visibility-cpu-v1"
SCENES = (
    "S1_constant_density_sphere",
    "S2_crossing_translucent_slabs",
    "S3_crossing_gaussian_density_sheets",
    "S4_thin_foreground_occluder",
    "S5_dense_semitransparent_cloud",
    "S6_moving_cell_complex",
    "S7_near_camera_large_cell",
    "S8_fast_object_fast_orbit",
)
CAMERAS = (
    "C1_static",
    "C2_linear_dolly",
    "C3_orbit",
    "C4_fast_orbit",
    "C5_orbit_finite_exposure",
    "C6_rolling_shutter",
    "C7_revolving_near_plane_crossing",
)
DEFAULT_LAYER_COUNTS = (16, 32, 64, 128)
DEFAULT_OUTPUT = Path(
    "outputs/benchmarks/2026-08-15_worldfoam_synthetic_visibility_cpu/summary.json"
)
BACKGROUND = np.asarray((0.018, 0.024, 0.032), dtype=np.float64)
NEAR = 0.05
FAR = 5.50
ADAPTIVE_COARSE_LAYERS = 32
ADAPTIVE_ESTIMATOR_LAYERS = 64
ADAPTIVE_FALLBACK_LAYERS = 128
ADAPTIVE_ERROR_TOLERANCE = 3.0e-3


@dataclass(frozen=True)
class CameraSpec:
    speed: float
    exposure_offsets: tuple[float, ...]
    rolling_shutter_scale: float = 0.0


@dataclass(frozen=True)
class TransferResult:
    rgb: np.ndarray
    transmittance: np.ndarray
    component_tau: np.ndarray
    component_mean_distance: np.ndarray
    colors: np.ndarray


CAMERA_SPECS = {
    "C1_static": CameraSpec(0.0, (0.0,)),
    "C2_linear_dolly": CameraSpec(0.35, (0.0,)),
    "C3_orbit": CameraSpec(0.65, (0.0,)),
    "C4_fast_orbit": CameraSpec(1.80, (0.0,)),
    "C5_orbit_finite_exposure": CameraSpec(0.80, (-0.06, -0.03, 0.0, 0.03, 0.06)),
    "C6_rolling_shutter": CameraSpec(0.60, (0.0,), rolling_shutter_scale=0.10),
    "C7_revolving_near_plane_crossing": CameraSpec(2.50, (0.0,)),
}


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_sha256() -> str:
    return _sha256_bytes(Path(__file__).read_bytes())


def _clip_time(value: np.ndarray) -> np.ndarray:
    return np.clip(value, -1.0, 1.0)


def _camera_rays(
    camera: str,
    times: np.ndarray,
    pixels: np.ndarray,
    exposure_offset: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec = CAMERA_SPECS[camera]
    physical_time = _clip_time(
        times[:, None]
        + float(exposure_offset)
        + float(spec.rolling_shutter_scale) * pixels[None, :]
    )
    pose_time = physical_time
    focus_x = np.zeros_like(pose_time)
    focus_z = np.full_like(pose_time, 2.45)

    if camera == "C1_static":
        origin_x = np.zeros_like(pose_time)
        origin_z = np.full_like(pose_time, -0.60)
    elif camera == "C2_linear_dolly":
        origin_x = 0.30 * pose_time
        origin_z = -0.60 + 0.30 * (pose_time + 1.0) * 0.5
    else:
        if camera == "C3_orbit":
            angle = 0.65 * pose_time
            radius = 3.05
        elif camera == "C4_fast_orbit":
            angle = 1.80 * pose_time
            radius = 3.00
        elif camera == "C5_orbit_finite_exposure":
            angle = 0.80 * pose_time
            radius = 3.00
        elif camera == "C6_rolling_shutter":
            angle = 0.60 * pose_time
            radius = 3.00
        elif camera == "C7_revolving_near_plane_crossing":
            angle = 2.50 * pose_time
            radius = 1.12
        else:  # pragma: no cover - validated by the public entry point.
            raise ValueError(f"unknown camera {camera!r}")
        origin_x = radius * np.sin(angle)
        origin_z = 2.45 - radius * np.cos(angle)

    central_x = focus_x - origin_x
    central_z = focus_z - origin_z
    central_norm = np.sqrt(central_x * central_x + central_z * central_z)
    central_x = central_x / central_norm
    central_z = central_z / central_norm
    perpendicular_x = -central_z
    perpendicular_z = central_x
    direction_x = central_x + 0.34 * pixels[None, :] * perpendicular_x
    direction_z = central_z + 0.34 * pixels[None, :] * perpendicular_z
    direction_norm = np.sqrt(direction_x * direction_x + direction_z * direction_z)
    direction = np.stack((direction_x / direction_norm, direction_z / direction_norm), axis=-1)
    origin = np.stack((origin_x, origin_z), axis=-1)
    return origin, direction, physical_time


def _component_colors(scene: str) -> np.ndarray:
    colors = {
        "S1_constant_density_sphere": ((0.88, 0.26, 0.10),),
        "S2_crossing_translucent_slabs": ((0.92, 0.18, 0.08), (0.06, 0.38, 0.96)),
        "S3_crossing_gaussian_density_sheets": ((0.94, 0.16, 0.06), (0.05, 0.55, 0.96)),
        "S4_thin_foreground_occluder": ((0.04, 0.06, 0.08), (0.82, 0.66, 0.18)),
        "S5_dense_semitransparent_cloud": (
            (0.86, 0.18, 0.12),
            (0.12, 0.58, 0.94),
            (0.24, 0.82, 0.32),
            (0.82, 0.32, 0.74),
            (0.94, 0.68, 0.16),
            (0.18, 0.78, 0.76),
            (0.68, 0.30, 0.12),
            (0.52, 0.70, 0.94),
        ),
        "S6_moving_cell_complex": (
            (0.84, 0.20, 0.12),
            (0.16, 0.62, 0.92),
            (0.24, 0.82, 0.36),
            (0.84, 0.42, 0.74),
            (0.92, 0.72, 0.18),
            (0.18, 0.74, 0.78),
        ),
        "S7_near_camera_large_cell": ((0.80, 0.28, 0.10), (0.08, 0.50, 0.88)),
        "S8_fast_object_fast_orbit": ((0.94, 0.20, 0.08), (0.08, 0.60, 0.90), (0.32, 0.78, 0.30)),
    }
    return np.asarray(colors[scene], dtype=np.float64)


def _scene_components(
    scene: str,
    x: np.ndarray,
    z: np.ndarray,
    time_field: np.ndarray,
) -> Iterator[np.ndarray]:
    """Yield one extinction field per constant-color material component."""

    if scene == "S1_constant_density_sphere":
        yield 1.40 * (((x / 0.78) ** 2 + ((z - 2.45) / 0.78) ** 2) <= 1.0)
        return

    if scene == "S2_crossing_translucent_slabs":
        center_a = 2.42 + 0.72 * time_field
        center_b = 2.42 - 0.72 * time_field
        lateral = np.abs(x) <= 1.38
        yield 0.92 * lateral * (np.abs(z - center_a) <= 0.31)
        yield 1.06 * lateral * (np.abs(z - center_b) <= 0.31)
        return

    if scene == "S3_crossing_gaussian_density_sheets":
        center_a = 2.42 + 0.72 * time_field
        center_b = 2.42 - 0.72 * time_field
        lateral = np.exp(-0.5 * (x / 1.25) ** 8)
        yield 1.18 * lateral * np.exp(-0.5 * ((z - center_a) / 0.27) ** 2)
        yield 1.02 * lateral * np.exp(-0.5 * ((z - center_b) / 0.31) ** 2)
        return

    if scene == "S4_thin_foreground_occluder":
        foreground_z = 1.55 + 0.12 * np.sin(math.pi * time_field)
        foreground_x = 0.42 * time_field
        yield 14.0 * (np.abs(z - foreground_z) <= 0.045) * (np.abs(x - foreground_x) <= 0.58)
        yield 0.82 * (((x / 0.95) ** 2 + ((z - 2.70) / 0.66) ** 2) <= 1.0)
        return

    if scene == "S5_dense_semitransparent_cloud":
        centers = (
            (-0.72, 1.55, 0.03, 0.00, 0.42, 0.32, 0.56),
            (-0.22, 1.92, -0.02, 0.05, 0.50, 0.36, 0.48),
            (0.44, 1.68, 0.04, -0.03, 0.45, 0.30, 0.52),
            (0.76, 2.24, -0.04, 0.01, 0.55, 0.40, 0.44),
            (-0.58, 2.48, 0.01, -0.04, 0.48, 0.34, 0.50),
            (0.02, 2.70, -0.03, 0.03, 0.62, 0.42, 0.46),
            (0.62, 3.00, 0.02, -0.02, 0.52, 0.38, 0.42),
            (-0.26, 3.18, 0.04, 0.02, 0.58, 0.44, 0.40),
        )
        for cx, cz, vx, vz, sx, sz, amplitude in centers:
            dx = (x - (cx + vx * time_field)) / sx
            dz = (z - (cz + vz * time_field)) / sz
            yield amplitude * np.exp(-0.5 * (dx * dx + dz * dz))
        return

    if scene == "S6_moving_cell_complex":
        cells = (
            (-0.78, 1.62, 0.10, 0.00, 0.34, 0.28, 0.62),
            (-0.20, 1.98, -0.08, 0.06, 0.42, 0.30, 0.72),
            (0.48, 1.72, 0.06, -0.04, 0.38, 0.34, 0.66),
            (0.72, 2.48, -0.07, 0.02, 0.48, 0.36, 0.58),
            (-0.52, 2.62, 0.04, -0.06, 0.44, 0.40, 0.64),
            (0.04, 3.05, -0.05, 0.04, 0.56, 0.34, 0.54),
        )
        for cx, cz, vx, vz, hx, hz, density in cells:
            inside = (np.abs(x - (cx + vx * time_field)) <= hx) & (
                np.abs(z - (cz + vz * time_field)) <= hz
            )
            yield density * inside
        return

    if scene == "S7_near_camera_large_cell":
        near_center_z = 0.30 + 0.28 * time_field
        yield 1.10 * (((x / 0.92) ** 2 + ((z - near_center_z) / 0.78) ** 2) <= 1.0)
        yield 0.64 * (((x / 1.15) ** 2 + ((z - 2.62) / 0.76) ** 2) <= 1.0)
        return

    if scene == "S8_fast_object_fast_orbit":
        moving_x = 0.92 * np.sin(1.35 * math.pi * time_field)
        moving_z = 2.30 + 0.52 * np.sin(1.10 * math.pi * time_field + 0.4)
        yield 1.55 * np.exp(
            -0.5 * (((x - moving_x) / 0.24) ** 2 + ((z - moving_z) / 0.26) ** 2)
        )
        yield 0.70 * np.exp(-0.5 * (((x + 0.45) / 0.62) ** 2 + ((z - 2.75) / 0.48) ** 2))
        yield 0.54 * (np.abs(x - 0.58) <= 0.36) * (np.abs(z - 1.72) <= 0.26)
        return

    raise ValueError(f"unknown scene {scene!r}")


def _sample_distances(
    sample_count: int,
    gauge: str,
    *,
    include_jacobian: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_count < 2:
        raise ValueError("sample_count must be at least two")
    if gauge == "ordinary_depth":
        edges = np.linspace(NEAR, FAR, sample_count + 1, dtype=np.float64)
        return 0.5 * (edges[:-1] + edges[1:]), np.diff(edges)
    if gauge == "log_depth":
        q_edges = np.linspace(math.log(NEAR), math.log(FAR), sample_count + 1, dtype=np.float64)
        q = 0.5 * (q_edges[:-1] + q_edges[1:])
        distance = np.exp(q)
        dq = np.diff(q_edges)
        if include_jacobian:
            return distance, distance * dq
        return distance, dq
    raise ValueError(f"unknown gauge {gauge!r}")


def _integrate_single_exposure(
    scene: str,
    camera: str,
    times: np.ndarray,
    pixels: np.ndarray,
    *,
    exposure_offset: float,
    sample_count: int,
    gauge: str = "ordinary_depth",
    include_jacobian: bool = True,
) -> TransferResult:
    origin, direction, physical_time = _camera_rays(camera, times, pixels, exposure_offset)
    distance, ds = _sample_distances(sample_count, gauge, include_jacobian=include_jacobian)
    x = origin[..., 0, None] + direction[..., 0, None] * distance[None, None, :]
    z = origin[..., 1, None] + direction[..., 1, None] * distance[None, None, :]
    time_field = np.broadcast_to(physical_time[..., None], x.shape)
    colors = _component_colors(scene)
    total_sigma = np.zeros_like(x)
    emission = np.zeros(x.shape + (3,), dtype=np.float64)
    component_tau: list[np.ndarray] = []
    component_distance_numerator: list[np.ndarray] = []
    for sigma, color in zip(_scene_components(scene, x, z, time_field), colors, strict=True):
        sigma = np.asarray(sigma, dtype=np.float64)
        weighted_sigma = sigma * ds[None, None, :]
        total_sigma += sigma
        emission += sigma[..., None] * color
        component_tau.append(np.sum(weighted_sigma, axis=-1))
        component_distance_numerator.append(
            np.sum(weighted_sigma * distance[None, None, :], axis=-1)
        )
    delta_tau = total_sigma * ds[None, None, :]
    transmittance_before = np.exp(
        -np.concatenate(
            (
                np.zeros(delta_tau.shape[:-1] + (1,), dtype=np.float64),
                np.cumsum(delta_tau[..., :-1], axis=-1),
            ),
            axis=-1,
        )
    )
    alpha = -np.expm1(-delta_tau)
    source = np.divide(
        emission,
        total_sigma[..., None],
        out=np.zeros_like(emission),
        where=total_sigma[..., None] > 0.0,
    )
    terminal_transmittance = np.exp(-np.sum(delta_tau, axis=-1))
    rgb = np.sum(transmittance_before[..., None] * alpha[..., None] * source, axis=-2)
    rgb += terminal_transmittance[..., None] * BACKGROUND
    tau = np.stack(component_tau, axis=-1)
    numerator = np.stack(component_distance_numerator, axis=-1)
    mean_distance = np.divide(
        numerator,
        tau,
        out=np.full_like(numerator, np.inf),
        where=tau > 1.0e-14,
    )
    return TransferResult(rgb, terminal_transmittance, tau, mean_distance, colors)


def _average_results(results: Sequence[TransferResult]) -> TransferResult:
    return TransferResult(
        rgb=np.mean(np.stack([result.rgb for result in results]), axis=0),
        transmittance=np.mean(np.stack([result.transmittance for result in results]), axis=0),
        component_tau=np.mean(np.stack([result.component_tau for result in results]), axis=0),
        component_mean_distance=np.mean(
            np.stack([result.component_mean_distance for result in results]), axis=0
        ),
        colors=results[0].colors,
    )


def integrate_program(
    scene: str,
    camera: str,
    times: np.ndarray,
    pixels: np.ndarray,
    sample_count: int,
    *,
    gauge: str = "ordinary_depth",
    include_jacobian: bool = True,
) -> TransferResult:
    return _average_results(
        [
            _integrate_single_exposure(
                scene,
                camera,
                times,
                pixels,
                exposure_offset=offset,
                sample_count=sample_count,
                gauge=gauge,
                include_jacobian=include_jacobian,
            )
            for offset in CAMERA_SPECS[camera].exposure_offsets
        ]
    )


def _render_sorted_components(result: TransferResult) -> np.ndarray:
    order = np.argsort(result.component_mean_distance, axis=-1)
    tau = np.take_along_axis(result.component_tau, order, axis=-1)
    colors = result.colors[order]
    transmittance = np.ones(tau.shape[:-1], dtype=np.float64)
    rgb = np.zeros(tau.shape[:-1] + (3,), dtype=np.float64)
    for rank in range(tau.shape[-1]):
        alpha = -np.expm1(-tau[..., rank])
        rgb += transmittance[..., None] * alpha[..., None] * colors[..., rank, :]
        transmittance *= np.exp(-tau[..., rank])
    return rgb + transmittance[..., None] * BACKGROUND


def _render_depth_marginal(result: TransferResult) -> np.ndarray:
    total_tau = np.sum(result.component_tau, axis=-1)
    color_numerator = np.einsum("...k,kc->...c", result.component_tau, result.colors)
    mean_color = np.divide(
        color_numerator,
        total_tau[..., None],
        out=np.zeros_like(color_numerator),
        where=total_tau[..., None] > 1.0e-14,
    )
    transmittance = np.exp(-total_tau)
    return (1.0 - transmittance)[..., None] * mean_color + transmittance[..., None] * BACKGROUND


def _render_program_baselines(
    scene: str,
    camera: str,
    times: np.ndarray,
    pixels: np.ndarray,
    sample_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Render each shutter sample before exposure averaging.

    Averaging optical depths or representative depths before compositing would
    change finite-exposure semantics.  The baselines therefore use the same
    shutter quadrature as the ordered-transfer reference.
    """

    exposure_results = [
        _integrate_single_exposure(
            scene,
            camera,
            times,
            pixels,
            exposure_offset=offset,
            sample_count=sample_count,
        )
        for offset in CAMERA_SPECS[camera].exposure_offsets
    ]
    return (
        np.mean(np.stack([_render_sorted_components(result) for result in exposure_results]), axis=0),
        np.mean(np.stack([_render_depth_marginal(result) for result in exposure_results]), axis=0),
    )


def _metrics(
    predicted_rgb: np.ndarray,
    predicted_transmittance: np.ndarray,
    oracle: TransferResult,
) -> dict[str, float]:
    rgb_error = predicted_rgb - oracle.rgb
    mse = float(np.mean(rgb_error * rgb_error))
    psnr = float(-10.0 * math.log10(max(mse, np.finfo(np.float64).tiny)))
    transmittance_error = predicted_transmittance - oracle.transmittance
    if predicted_rgb.shape[0] > 1:
        temporal_error = np.diff(predicted_rgb, axis=0) - np.diff(oracle.rgb, axis=0)
        flicker_error = float(np.mean(np.abs(temporal_error)))
        gradient_variance = float(np.var(temporal_error))
    else:
        flicker_error = 0.0
        gradient_variance = 0.0
    return {
        "rgb_mse": mse,
        "rgb_psnr_db": psnr,
        "rgb_mean_absolute_error": float(np.mean(np.abs(rgb_error))),
        "rgb_max_absolute_error": float(np.max(np.abs(rgb_error))),
        "transmittance_mean_absolute_error": float(np.mean(np.abs(transmittance_error))),
        "transmittance_max_absolute_error": float(np.max(np.abs(transmittance_error))),
        "temporal_flicker_error": flicker_error,
        "temporal_gradient_error_variance": gradient_variance,
    }


def _representative_order_flip_count(result: TransferResult) -> int:
    if result.component_mean_distance.shape[-1] < 2:
        return 0
    first = np.nanmedian(result.component_mean_distance[..., 0], axis=1)
    second = np.nanmedian(result.component_mean_distance[..., 1], axis=1)
    finite = np.isfinite(first) & np.isfinite(second)
    signs = np.sign(first[finite] - second[finite])
    signs = signs[signs != 0.0]
    return int(np.count_nonzero(signs[1:] != signs[:-1])) if signs.size > 1 else 0


def _analytic_sphere_static(
    times: np.ndarray,
    pixels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    origin, direction, _ = _camera_rays("C1_static", times, pixels, 0.0)
    center = np.asarray((0.0, 2.45), dtype=np.float64)
    offset = origin - center
    b = 2.0 * np.sum(offset * direction, axis=-1)
    c = np.sum(offset * offset, axis=-1) - 0.78**2
    discriminant = np.maximum(b * b - 4.0 * c, 0.0)
    root = np.sqrt(discriminant)
    entry = np.maximum((-b - root) * 0.5, NEAR)
    exit_ = np.minimum((-b + root) * 0.5, FAR)
    length = np.maximum(exit_ - entry, 0.0)
    tau = 1.40 * length
    transmittance = np.exp(-tau)
    color = _component_colors("S1_constant_density_sphere")[0]
    rgb = (1.0 - transmittance)[..., None] * color + transmittance[..., None] * BACKGROUND
    return rgb, transmittance


def _aggregate(rows: Sequence[dict[str, object]], key: str) -> dict[str, float]:
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _svg_escape(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _line_chart_svg(
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: Sequence[tuple[str, Sequence[float], Sequence[float], str]],
    width: int = 900,
    height: int = 540,
    log_y: bool = False,
    x_tick_labels: dict[float, str] | None = None,
) -> str:
    left, right, top, bottom = 92.0, 232.0, 66.0, 76.0
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_values = [float(value) for _, xs, _, _ in series for value in xs]
    y_values = [max(float(value), 1.0e-16) for _, _, ys, _ in series for value in ys]
    x_min, x_max = min(x_values), max(x_values)
    y_plot = [math.log10(value) if log_y else value for value in y_values]
    y_min, y_max = min(y_plot), max(y_plot)
    if math.isclose(y_min, y_max):
        y_min -= 0.5
        y_max += 0.5

    def px(value: float) -> float:
        return left + (float(value) - x_min) / max(x_max - x_min, 1.0e-12) * plot_w

    def py(value: float) -> float:
        transformed = math.log10(max(float(value), 1.0e-16)) if log_y else float(value)
        return top + (y_max - transformed) / (y_max - y_min) * plot_h

    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f"<title>{_svg_escape(title)}</title>",
        f"<desc>{_svg_escape(y_label)} plotted against {_svg_escape(x_label)}.</desc>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" font-family="Helvetica,Arial,sans-serif" font-size="22" font-weight="700">{_svg_escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222" stroke-width="1.5"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222" stroke-width="1.5"/>',
    ]
    for tick in range(6):
        fraction = tick / 5.0
        y_value = y_min + fraction * (y_max - y_min)
        y_position = top + (1.0 - fraction) * plot_h
        label = f"{10.0**y_value:.1e}" if log_y else f"{y_value:.3g}"
        body.extend(
            (
                f'<line x1="{left}" y1="{y_position:.2f}" x2="{left + plot_w}" y2="{y_position:.2f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 10}" y="{y_position + 4:.2f}" text-anchor="end" font-family="Helvetica,Arial,sans-serif" font-size="13">{label}</text>',
            )
        )
    unique_x = sorted(set(x_values))
    for value in unique_x:
        tick_label = x_tick_labels.get(value, f"{value:g}") if x_tick_labels else f"{value:g}"
        body.append(
            f'<text x="{px(value):.2f}" y="{top + plot_h + 24:.2f}" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="12">{_svg_escape(tick_label)}</text>'
        )
    for index, (label, xs, ys, color) in enumerate(series):
        points = [(px(x), py(y)) for x, y in zip(xs, ys, strict=True)]
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:], strict=True):
            dx, dy = x1 - x0, y1 - y0
            length = max(math.hypot(dx, dy), 1.0e-12)
            nx, ny = -1.5 * dy / length, 1.5 * dx / length
            polygon = " ".join(
                f"{x:.2f},{y:.2f}"
                for x, y in (
                    (x0 + nx, y0 + ny),
                    (x1 + nx, y1 + ny),
                    (x1 - nx, y1 - ny),
                    (x0 - nx, y0 - ny),
                )
            )
            body.append(f'<polygon points="{polygon}" fill="{color}"/>')
        for x, y in zip(xs, ys, strict=True):
            body.append(f'<circle cx="{px(x):.2f}" cy="{py(y):.2f}" r="4" fill="{color}"/>')
        legend_y = 54 + index * 22
        body.append(f'<circle cx="{width - 194}" cy="{legend_y}" r="5" fill="{color}"/>')
        body.append(f'<text x="{width - 180}" y="{legend_y + 4}" font-family="Helvetica,Arial,sans-serif" font-size="13">{_svg_escape(label)}</text>')
    body.extend(
        (
            f'<text x="{left + plot_w / 2:.2f}" y="{height - 22}" text-anchor="middle" font-family="Helvetica,Arial,sans-serif" font-size="15">{_svg_escape(x_label)}</text>',
            f'<text x="20" y="{top + plot_h / 2:.2f}" text-anchor="middle" transform="rotate(-90 20 {top + plot_h / 2:.2f})" font-family="Helvetica,Arial,sans-serif" font-size="15">{_svg_escape(y_label)}</text>',
            "</svg>",
        )
    )
    return "\n".join(body) + "\n"


def _write_figures(report: dict[str, object], output_dir: Path) -> list[dict[str, object]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    layer_rows = report["layer_rows"]
    assert isinstance(layer_rows, list)
    convergence_series = []
    for scene, color in (
        ("S2_crossing_translucent_slabs", "#d94841"),
        ("S3_crossing_gaussian_density_sheets", "#3366cc"),
        ("S4_thin_foreground_occluder", "#7a4db3"),
        ("S5_dense_semitransparent_cloud", "#198754"),
    ):
        xs, ys = [], []
        for layer_count in report["settings"]["layer_counts"]:
            selected = [
                row
                for row in layer_rows
                if row["scene"] == scene and row["layer_count"] == layer_count
            ]
            xs.append(float(layer_count))
            ys.append(float(np.median([row["rgb_mean_absolute_error"] for row in selected])))
        convergence_series.append((scene.split("_", 1)[0], xs, ys, color))
    convergence = _line_chart_svg(
        title="WorldFoam synthetic depth-layer convergence",
        x_label="Depth layers",
        y_label="Median RGB mean absolute error",
        series=convergence_series,
        log_y=True,
    )

    adaptive_rows = report["adaptive_rows"]
    assert isinstance(adaptive_rows, list)
    camera_order = sorted(CAMERAS, key=lambda camera: CAMERA_SPECS[camera].speed)
    fallback_y = []
    for camera in camera_order:
        selected = [row for row in adaptive_rows if row["camera"] == camera]
        fallback_y.append(float(np.mean([row["fallback_fraction"] for row in selected])))
    camera_x = [float(index + 1) for index in range(len(camera_order))]
    camera_ticks = {
        float(index + 1): f"{camera.split('_', 1)[0]}:{CAMERA_SPECS[camera].speed:.2f}"
        for index, camera in enumerate(camera_order)
    }
    fallback = _line_chart_svg(
        title="Adaptive retained-depth fallback versus camera speed",
        x_label="Camera program : nominal speed",
        y_label="Fallback fraction",
        series=(
            (
                "all scenes",
                camera_x,
                fallback_y,
                "#c23b22",
            ),
        ),
        x_tick_labels=camera_ticks,
    )

    baseline_rows = report["baseline_rows"]
    assert isinstance(baseline_rows, list)
    methods = (
        ("depth_layer_128", "#2a6fbb"),
        ("representative_depth_sorted", "#d94841"),
        ("depth_marginal", "#7a4db3"),
    )
    crossing_series = []
    for method, color in methods:
        selected = [
            row
            for row in baseline_rows
            if row["method"] == method
            and row["scene"] in {"S2_crossing_translucent_slabs", "S3_crossing_gaussian_density_sheets"}
        ]
        xs = camera_x
        ys = []
        for camera in camera_order:
            camera_rows = [row for row in selected if row["camera"] == camera]
            ys.append(float(np.mean([row["temporal_flicker_error"] for row in camera_rows])))
        crossing_series.append((method, xs, ys, color))
    crossing = _line_chart_svg(
        title="Crossing-scene temporal error",
        x_label="Camera program : nominal speed",
        y_label="Temporal flicker error",
        series=crossing_series,
        log_y=True,
        x_tick_labels=camera_ticks,
    )

    figures = {
        "worldfoam_synthetic_depth_convergence.svg": convergence,
        "worldfoam_synthetic_adaptive_fallback.svg": fallback,
        "worldfoam_synthetic_crossing_flicker.svg": crossing,
    }
    manifest = []
    for name, source in figures.items():
        path = output_dir / name
        encoded = source.encode("utf-8")
        path.write_bytes(encoded)
        manifest.append({"name": name, "sha256": _sha256_bytes(encoded), "bytes": len(encoded)})
    return manifest


def run_suite(
    *,
    frame_count: int = 13,
    pixel_count: int = 17,
    oracle_samples: int = 2048,
    layer_counts: Sequence[int] = DEFAULT_LAYER_COUNTS,
) -> dict[str, object]:
    if frame_count < 5 or pixel_count < 5:
        raise ValueError("frame_count and pixel_count must each be at least five")
    if oracle_samples < max(layer_counts) * 4:
        raise ValueError("oracle_samples must be at least four times the largest layer count")
    if tuple(layer_counts) != tuple(sorted(set(int(value) for value in layer_counts))):
        raise ValueError("layer_counts must be unique and strictly increasing")
    required_adaptive = {
        ADAPTIVE_COARSE_LAYERS,
        ADAPTIVE_ESTIMATOR_LAYERS,
        ADAPTIVE_FALLBACK_LAYERS,
    }
    if not required_adaptive.issubset(set(layer_counts)):
        raise ValueError(f"layer_counts must contain {sorted(required_adaptive)}")

    started = time.perf_counter()
    times = np.linspace(-1.0, 1.0, frame_count, dtype=np.float64)
    pixels = np.linspace(-0.78, 0.78, pixel_count, dtype=np.float64)
    layer_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    adaptive_rows: list[dict[str, object]] = []
    analytic_receipt: dict[str, float] | None = None

    for scene in SCENES:
        for camera in CAMERAS:
            oracle = integrate_program(scene, camera, times, pixels, oracle_samples)
            if scene == "S1_constant_density_sphere" and camera == "C1_static":
                analytic_rgb, analytic_t = _analytic_sphere_static(times, pixels)
                analytic_receipt = {
                    "rgb_max_absolute_error": float(np.max(np.abs(oracle.rgb - analytic_rgb))),
                    "transmittance_max_absolute_error": float(
                        np.max(np.abs(oracle.transmittance - analytic_t))
                    ),
                }

            layer_results: dict[int, TransferResult] = {}
            for layer_count in layer_counts:
                prediction = integrate_program(scene, camera, times, pixels, int(layer_count))
                layer_results[int(layer_count)] = prediction
                metrics = _metrics(prediction.rgb, prediction.transmittance, oracle)
                layer_rows.append(
                    {
                        "scene": scene,
                        "camera": camera,
                        "layer_count": int(layer_count),
                        **metrics,
                    }
                )

            sorted_rgb, marginal_rgb = _render_program_baselines(
                scene, camera, times, pixels, oracle_samples
            )
            for method, predicted_rgb, predicted_t in (
                (
                    f"depth_layer_{max(layer_counts)}",
                    layer_results[max(layer_counts)].rgb,
                    layer_results[max(layer_counts)].transmittance,
                ),
                ("representative_depth_sorted", sorted_rgb, oracle.transmittance),
                ("depth_marginal", marginal_rgb, oracle.transmittance),
            ):
                baseline_rows.append(
                    {
                        "scene": scene,
                        "camera": camera,
                        "method": method,
                        "representative_order_flip_count": _representative_order_flip_count(oracle),
                        **_metrics(predicted_rgb, predicted_t, oracle),
                    }
                )

            coarse = layer_results[ADAPTIVE_COARSE_LAYERS]
            estimate = layer_results[ADAPTIVE_ESTIMATOR_LAYERS]
            fallback = layer_results[ADAPTIVE_FALLBACK_LAYERS]
            estimator_error = np.max(np.abs(coarse.rgb - estimate.rgb), axis=-1)
            estimator_error = np.maximum(
                estimator_error, np.abs(coarse.transmittance - estimate.transmittance)
            )
            fallback_mask = estimator_error > ADAPTIVE_ERROR_TOLERANCE
            adaptive_rgb = np.where(fallback_mask[..., None], fallback.rgb, coarse.rgb)
            adaptive_t = np.where(fallback_mask, fallback.transmittance, coarse.transmittance)
            adaptive_rows.append(
                {
                    "scene": scene,
                    "camera": camera,
                    "fallback_fraction": float(np.mean(fallback_mask)),
                    "estimator_max_error": float(np.max(estimator_error)),
                    **_metrics(adaptive_rgb, adaptive_t, oracle),
                }
            )

    gauge_times = np.linspace(-0.9, 0.9, 9, dtype=np.float64)
    gauge_pixels = np.linspace(-0.65, 0.65, 11, dtype=np.float64)
    gauge_linear = integrate_program(
        "S3_crossing_gaussian_density_sheets",
        "C3_orbit",
        gauge_times,
        gauge_pixels,
        4096,
    )
    gauge_log = integrate_program(
        "S3_crossing_gaussian_density_sheets",
        "C3_orbit",
        gauge_times,
        gauge_pixels,
        4096,
        gauge="log_depth",
    )
    gauge_bad = integrate_program(
        "S3_crossing_gaussian_density_sheets",
        "C3_orbit",
        gauge_times,
        gauge_pixels,
        4096,
        gauge="log_depth",
        include_jacobian=False,
    )
    with_jacobian_error = float(np.max(np.abs(gauge_linear.rgb - gauge_log.rgb)))
    without_jacobian_error = float(np.max(np.abs(gauge_linear.rgb - gauge_bad.rgb)))
    gauge_receipt = {
        "fixture": "S3_crossing_gaussian_density_sheets/C3_orbit",
        "sample_count_per_gauge": 4096,
        "with_physical_jacobian_rgb_max_absolute_error": with_jacobian_error,
        "without_physical_jacobian_rgb_max_absolute_error": without_jacobian_error,
        "error_ratio_without_over_with": float(
            without_jacobian_error / max(with_jacobian_error, np.finfo(np.float64).tiny)
        ),
    }

    deepest = [row for row in layer_rows if row["layer_count"] == max(layer_counts)]
    crossing_worldfoam = [
        row
        for row in baseline_rows
        if row["method"] == f"depth_layer_{max(layer_counts)}"
        and row["scene"] in {"S2_crossing_translucent_slabs", "S3_crossing_gaussian_density_sheets"}
    ]
    crossing_sorted = [
        row
        for row in baseline_rows
        if row["method"] == "representative_depth_sorted"
        and row["scene"] in {"S2_crossing_translucent_slabs", "S3_crossing_gaussian_density_sheets"}
    ]
    crossing_marginal = [
        row
        for row in baseline_rows
        if row["method"] == "depth_marginal"
        and row["scene"] in {"S2_crossing_translucent_slabs", "S3_crossing_gaussian_density_sheets"}
    ]
    deepest_p05_psnr = float(np.quantile([row["rgb_psnr_db"] for row in deepest], 0.05))
    crossing_worldfoam_mse = float(np.mean([row["rgb_mse"] for row in crossing_worldfoam]))
    crossing_sorted_mse = float(np.mean([row["rgb_mse"] for row in crossing_sorted]))
    crossing_marginal_mse = float(np.mean([row["rgb_mse"] for row in crossing_marginal]))
    assert analytic_receipt is not None
    gates = {
        "analytic_constant_sphere": bool(
            analytic_receipt["rgb_max_absolute_error"] <= 6.0e-3
            and analytic_receipt["transmittance_max_absolute_error"] <= 7.0e-3
        ),
        "gauge_jacobian": bool(
            with_jacobian_error <= 8.0e-4
            and without_jacobian_error >= 20.0 * max(with_jacobian_error, 1.0e-12)
        ),
        "deepest_layer_floor": bool(deepest_p05_psnr >= 30.0),
        "crossing_beats_representative_sort": bool(
            crossing_worldfoam_mse <= 0.50 * crossing_sorted_mse
        ),
        "crossing_beats_depth_marginal": bool(
            crossing_worldfoam_mse <= 0.50 * crossing_marginal_mse
        ),
        "finite_all_rows": bool(
            all(
                math.isfinite(float(value))
                for row in (*layer_rows, *baseline_rows, *adaptive_rows)
                for key, value in row.items()
                if key
                not in {
                    "scene",
                    "camera",
                    "method",
                    "layer_count",
                    "representative_order_flip_count",
                }
            )
        ),
    }
    settings = {
        "scenes": list(SCENES),
        "cameras": list(CAMERAS),
        "frame_count": frame_count,
        "pixel_count": pixel_count,
        "oracle_samples": oracle_samples,
        "layer_counts": list(layer_counts),
        "near": NEAR,
        "far": FAR,
        "background": BACKGROUND.tolist(),
        "adaptive_coarse_layers": ADAPTIVE_COARSE_LAYERS,
        "adaptive_estimator_layers": ADAPTIVE_ESTIMATOR_LAYERS,
        "adaptive_fallback_layers": ADAPTIVE_FALLBACK_LAYERS,
        "adaptive_error_tolerance": ADAPTIVE_ERROR_TOLERANCE,
        "dtype": "float64",
        "device": "cpu",
    }
    report: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "suite_id": SUITE_ID,
        "claim_scope": {
            "supports": [
                "CPU representation-level ordered-transfer exactness",
                "depth-layer convergence",
                "adaptive fallback behavior",
                "gauge-Jacobian necessity",
                "crossing visibility comparison",
            ],
            "does_not_support": [
                "native-kernel speed",
                "native allocator or peak-memory scaling",
                "public-data trained quality",
                "end-to-end kinetic compiler acceptance",
            ],
        },
        "settings": settings,
        "protocol_sha256": _sha256_bytes(_canonical_json(settings).encode("utf-8")),
        "source_sha256": _source_sha256(),
        "analytic_constant_sphere": analytic_receipt,
        "gauge_jacobian": gauge_receipt,
        "layer_rows": layer_rows,
        "baseline_rows": baseline_rows,
        "adaptive_rows": adaptive_rows,
        "aggregates": {
            "deepest_layer_psnr_db": _aggregate(deepest, "rgb_psnr_db"),
            "deepest_layer_rgb_max_absolute_error": _aggregate(
                deepest, "rgb_max_absolute_error"
            ),
            "adaptive_fallback_fraction": _aggregate(adaptive_rows, "fallback_fraction"),
            "crossing_worldfoam_rgb_mse_mean": crossing_worldfoam_mse,
            "crossing_sorted_rgb_mse_mean": crossing_sorted_mse,
            "crossing_depth_marginal_rgb_mse_mean": crossing_marginal_mse,
        },
        "acceptance_gates": gates,
        "accepted": bool(all(gates.values())),
        "diagnostic_cpu_wall_seconds": float(time.perf_counter() - started),
        "timing_is_paper_evidence": False,
    }
    return report


def write_suite(report: dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    figure_dir = output.parent / "figures"
    report["figure_manifest"] = _write_figures(report, figure_dir)
    encoded = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    output.write_bytes(encoded)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--frames", type=int, default=13)
    parser.add_argument("--pixels", type=int, default=17)
    parser.add_argument("--oracle-samples", type=int, default=2048)
    parser.add_argument(
        "--layer-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_LAYER_COUNTS),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_suite(
        frame_count=args.frames,
        pixel_count=args.pixels,
        oracle_samples=args.oracle_samples,
        layer_counts=tuple(args.layer_counts),
    )
    write_suite(report, args.output)
    print(
        _canonical_json(
            {
                "accepted": report["accepted"],
                "output": str(args.output),
                "layer_rows": len(report["layer_rows"]),
                "baseline_rows": len(report["baseline_rows"]),
                "adaptive_rows": len(report["adaptive_rows"]),
                "gates": report["acceptance_gates"],
            }
        )
    )


if __name__ == "__main__":
    main()
