from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from gate0_beam_toy import (
    EPS,
    Boundary,
    Site,
    ToyConfig,
    crossing_depth,
    default_sites,
    linspace,
    make_boundaries,
    slab_events,
    slab_ranges,
)


DEFAULT_SITE_SIGNALS = (0.25, -0.10, 0.35, 0.05, 0.16)


@dataclass(frozen=True)
class Segment:
    site_id: int
    depth0: float
    depth1: float

    @property
    def length(self) -> float:
        return self.depth1 - self.depth0


@dataclass(frozen=True)
class RayTape:
    u: float
    t: float
    slab_index: int
    output: float
    grad_output: float
    candidate_event_count: int
    segments: tuple[Segment, ...]


def power_distance(site: Site, *, x: float, z: float, t: float) -> float:
    return (x - site.x) ** 2 + (z - site.z) ** 2 + (t - site.t) ** 2 - site.weight


def owner_at(sites: tuple[Site, ...], *, x: float, z: float, t: float) -> int:
    return min(range(len(sites)), key=lambda idx: power_distance(sites[idx], x=x, z=z, t=t))


def make_boundary_lookup(boundaries: Iterable[Boundary]) -> dict[tuple[int, int], Boundary]:
    return {(boundary.left, boundary.right): boundary for boundary in boundaries}


def dedupe_sorted_depths(depths: list[float], *, epsilon: float = 1.0e-7) -> list[float]:
    depths.sort()
    unique: list[float] = []
    for depth in depths:
        if not unique or abs(depth - unique[-1]) > epsilon:
            unique.append(depth)
    return unique


def candidate_depths(
    candidate_events: Iterable[tuple[int, int]],
    *,
    boundary_lookup: dict[tuple[int, int], Boundary],
    u: float,
    t: float,
    near: float,
    far: float,
    camera_velocity_x: float,
) -> list[float]:
    depths: list[float] = []
    for key in candidate_events:
        depth = crossing_depth(boundary_lookup[key], u=u, t=t, camera_velocity_x=camera_velocity_x)
        if depth is not None and near <= depth <= far:
            depths.append(depth)
    return dedupe_sorted_depths(depths)


def render_ray_from_candidates(
    *,
    sites: tuple[Site, ...],
    boundary_lookup: dict[tuple[int, int], Boundary],
    candidate_events: Iterable[tuple[int, int]],
    site_signals: tuple[float, ...],
    u: float,
    t: float,
    near: float,
    far: float,
    camera_velocity_x: float,
    slab_index: int,
    grad_output: float,
) -> RayTape:
    candidate_event_tuple = tuple(candidate_events)
    depths = candidate_depths(
        candidate_event_tuple,
        boundary_lookup=boundary_lookup,
        u=u,
        t=t,
        near=near,
        far=far,
        camera_velocity_x=camera_velocity_x,
    )
    cuts = [near, *depths, far]
    x = u + camera_velocity_x * t
    output = 0.0
    segments: list[Segment] = []
    for depth0, depth1 in zip(cuts, cuts[1:]):
        if depth1 - depth0 <= EPS:
            continue
        mid_depth = 0.5 * (depth0 + depth1)
        site_id = owner_at(sites, x=x, z=mid_depth, t=t)
        segment = Segment(site_id=site_id, depth0=depth0, depth1=depth1)
        segments.append(segment)
        output += site_signals[site_id] * segment.length
    return RayTape(
        u=u,
        t=t,
        slab_index=slab_index,
        output=output,
        grad_output=grad_output,
        candidate_event_count=len(candidate_event_tuple),
        segments=tuple(segments),
    )


def backward_signal_gradients(tapes: Iterable[RayTape], *, site_count: int) -> tuple[float, ...]:
    grads = [0.0 for _ in range(site_count)]
    for tape in tapes:
        for segment in tape.segments:
            grads[segment.site_id] += tape.grad_output * segment.length
    return tuple(grads)


def loss_from_tapes(tapes: Iterable[RayTape]) -> float:
    return sum(tape.output * tape.grad_output for tape in tapes)


def gradient_seed(u_index: int, t_index: int) -> float:
    return 0.125 + 0.03125 * float(u_index + 1) + 0.015625 * float(t_index + 1)


def build_shared_slab_cache(
    *,
    u_values: list[float],
    boundaries: tuple[Boundary, ...],
    config: ToyConfig,
) -> tuple[dict[tuple[float, int], set[tuple[int, int]]], int]:
    slabs = slab_ranges(config.time_slabs)
    invalid_denominators = 0
    cache: dict[tuple[float, int], set[tuple[int, int]]] = {}
    for u in u_values:
        for slab_i, (t0, t1) in enumerate(slabs):
            events, invalid = slab_events(
                boundaries,
                u=u,
                t0=t0,
                t1=t1,
                near=config.near,
                far=config.far,
                camera_velocity_x=config.camera_velocity_x,
                invalid_epsilon=config.invalid_epsilon,
            )
            cache[(u, slab_i)] = events
            invalid_denominators += invalid
    return cache, invalid_denominators


def render_frame_count(
    *,
    frames: int,
    sites: tuple[Site, ...],
    boundaries: tuple[Boundary, ...],
    config: ToyConfig,
    site_signals: tuple[float, ...],
) -> dict[str, Any]:
    u_values = linspace(-1.0, 1.0, config.u_samples)
    frame_times = linspace(0.0, 1.0, frames)
    boundary_lookup = make_boundary_lookup(boundaries)
    all_events = set(boundary_lookup.keys())
    slab_cache, invalid_denominators = build_shared_slab_cache(
        u_values=u_values,
        boundaries=boundaries,
        config=config,
    )

    direct_tapes: list[RayTape] = []
    shared_tapes: list[RayTape] = []
    max_output_abs_error = 0.0
    max_segment_count_delta = 0
    for u_index, u in enumerate(u_values):
        for t_index, t in enumerate(frame_times):
            slab_index = min(int(math.floor(t * config.time_slabs)), config.time_slabs - 1)
            grad_output = gradient_seed(u_index, t_index)
            direct = render_ray_from_candidates(
                sites=sites,
                boundary_lookup=boundary_lookup,
                candidate_events=all_events,
                site_signals=site_signals,
                u=u,
                t=t,
                near=config.near,
                far=config.far,
                camera_velocity_x=config.camera_velocity_x,
                slab_index=slab_index,
                grad_output=grad_output,
            )
            shared = render_ray_from_candidates(
                sites=sites,
                boundary_lookup=boundary_lookup,
                candidate_events=slab_cache[(u, slab_index)],
                site_signals=site_signals,
                u=u,
                t=t,
                near=config.near,
                far=config.far,
                camera_velocity_x=config.camera_velocity_x,
                slab_index=slab_index,
                grad_output=grad_output,
            )
            direct_tapes.append(direct)
            shared_tapes.append(shared)
            max_output_abs_error = max(max_output_abs_error, abs(direct.output - shared.output))
            max_segment_count_delta = max(
                max_segment_count_delta,
                abs(len(direct.segments) - len(shared.segments)),
            )

    direct_grads = backward_signal_gradients(direct_tapes, site_count=len(sites))
    shared_grads = backward_signal_gradients(shared_tapes, site_count=len(sites))
    max_grad_abs_error = max(abs(a - b) for a, b in zip(direct_grads, shared_grads))

    direct_forward_boundary_scans = len(boundaries) * len(u_values) * len(frame_times)
    direct_backward_boundary_scans = direct_forward_boundary_scans
    shared_forward_boundary_scans = len(boundaries) * len(u_values) * config.time_slabs
    shared_backward_boundary_scans = 0
    shared_replay_candidate_evals = sum(tape.candidate_event_count for tape in shared_tapes)
    shared_backward_segment_visits = sum(len(tape.segments) for tape in shared_tapes)
    direct_backward_segment_visits = sum(len(tape.segments) for tape in direct_tapes)

    return {
        "frames": frames,
        "u_samples": len(u_values),
        "time_slabs": config.time_slabs,
        "direct_forward_boundary_scans": direct_forward_boundary_scans,
        "direct_backward_boundary_scans": direct_backward_boundary_scans,
        "shared_forward_boundary_scans": shared_forward_boundary_scans,
        "shared_backward_boundary_scans": shared_backward_boundary_scans,
        "shared_replay_candidate_evals": shared_replay_candidate_evals,
        "direct_backward_segment_visits": direct_backward_segment_visits,
        "shared_backward_segment_visits": shared_backward_segment_visits,
        "shared_boundary_scan_ratio": shared_forward_boundary_scans
        / float(max(direct_forward_boundary_scans, 1)),
        "shared_forward_backward_boundary_scan_ratio": (
            shared_forward_boundary_scans + shared_backward_boundary_scans
        )
        / float(max(direct_forward_boundary_scans + direct_backward_boundary_scans, 1)),
        "max_output_abs_error": max_output_abs_error,
        "max_segment_count_delta": max_segment_count_delta,
        "signal_gradient_max_abs_error": max_grad_abs_error,
        "direct_loss": loss_from_tapes(direct_tapes),
        "shared_loss": loss_from_tapes(shared_tapes),
        "direct_signal_gradient": direct_grads,
        "shared_signal_gradient": shared_grads,
        "invalid_denominator_count": invalid_denominators,
    }


def finite_difference_signal_gradients(
    *,
    frames: int,
    sites: tuple[Site, ...],
    boundaries: tuple[Boundary, ...],
    config: ToyConfig,
    site_signals: tuple[float, ...],
    epsilon: float = 1.0e-5,
) -> tuple[float, ...]:
    grads: list[float] = []
    for site_id in range(len(site_signals)):
        plus = list(site_signals)
        minus = list(site_signals)
        plus[site_id] += epsilon
        minus[site_id] -= epsilon
        plus_row = render_frame_count(
            frames=frames,
            sites=sites,
            boundaries=boundaries,
            config=config,
            site_signals=tuple(plus),
        )
        minus_row = render_frame_count(
            frames=frames,
            sites=sites,
            boundaries=boundaries,
            config=config,
            site_signals=tuple(minus),
        )
        grads.append((plus_row["shared_loss"] - minus_row["shared_loss"]) / (2.0 * epsilon))
    return tuple(grads)


def run(config: ToyConfig, *, site_signals: tuple[float, ...] = DEFAULT_SITE_SIGNALS) -> dict[str, Any]:
    sites = default_sites()
    if len(site_signals) != len(sites):
        raise ValueError("site_signals must match default site count")
    boundaries = make_boundaries(sites)
    rows = [
        render_frame_count(
            frames=frames,
            sites=sites,
            boundaries=boundaries,
            config=config,
            site_signals=site_signals,
        )
        for frames in config.frame_counts
    ]
    last = rows[-1]
    finite_difference = finite_difference_signal_gradients(
        frames=int(last["frames"]),
        sites=sites,
        boundaries=boundaries,
        config=config,
        site_signals=site_signals,
    )
    finite_difference_max_abs_error = max(
        abs(a - b) for a, b in zip(last["shared_signal_gradient"], finite_difference)
    )
    return {
        "benchmark": "world_foam_lane2_gate0_shared_forward_backward",
        "status": "cpu_signal_gradient_reference",
        "gradient_scope": "site_signal_only_geometry_gradients_not_implemented",
        "site_count": len(sites),
        "boundary_count": len(boundaries),
        "camera_velocity_x": config.camera_velocity_x,
        "rows": rows,
        "finite_difference_signal_gradient": finite_difference,
        "finite_difference_max_abs_error": finite_difference_max_abs_error,
        "acceptance": {
            "shared_outputs_match_direct": all(row["max_output_abs_error"] <= 1.0e-9 for row in rows),
            "shared_segments_match_direct": all(row["max_segment_count_delta"] == 0 for row in rows),
            "shared_gradients_match_direct": all(row["signal_gradient_max_abs_error"] <= 1.0e-9 for row in rows),
            "finite_difference_matches_shared_gradient": finite_difference_max_abs_error <= 1.0e-8,
            "shared_forward_backward_scans_sublinear": all(
                row["shared_forward_backward_boundary_scan_ratio"] < 1.0 for row in rows
            ),
        },
    }


def parse_frame_counts(value: str) -> tuple[int, ...]:
    counts = tuple(int(part) for part in value.split(",") if part.strip())
    if not counts:
        raise ValueError("provide at least one frame count")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="World Foam Lane 2 Gate 0 shared forward/backward reference.")
    parser.add_argument("--frame-counts", default="2,4,8,16")
    parser.add_argument("--u-samples", type=int, default=17)
    parser.add_argument("--time-slabs", type=int, default=1)
    parser.add_argument("--near", type=float, default=0.25)
    parser.add_argument("--far", type=float, default=3.0)
    parser.add_argument("--camera-velocity-x", type=float, default=0.35)
    parser.add_argument("--invalid-epsilon", type=float, default=1.0e-7)
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ToyConfig(
        frame_counts=parse_frame_counts(args.frame_counts),
        u_samples=args.u_samples,
        time_slabs=args.time_slabs,
        near=args.near,
        far=args.far,
        camera_velocity_x=args.camera_velocity_x,
        invalid_epsilon=args.invalid_epsilon,
    )
    payload = run(config)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
