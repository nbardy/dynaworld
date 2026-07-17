from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass
from typing import Iterable


EPS = 1.0e-8


@dataclass(frozen=True)
class Site:
    x: float
    z: float
    t: float
    weight: float


@dataclass(frozen=True)
class Boundary:
    left: int
    right: int
    nx: float
    nz: float
    nt: float
    b: float


@dataclass(frozen=True)
class ToyConfig:
    frame_counts: tuple[int, ...]
    u_samples: int
    time_slabs: int
    near: float
    far: float
    camera_velocity_x: float
    invalid_epsilon: float


def default_sites() -> tuple[Site, ...]:
    return (
        Site(x=-0.75, z=0.65, t=0.08, weight=0.00),
        Site(x=0.10, z=1.05, t=0.28, weight=0.04),
        Site(x=0.72, z=1.55, t=0.58, weight=-0.03),
        Site(x=-0.18, z=2.15, t=0.88, weight=0.02),
        Site(x=0.95, z=2.65, t=0.42, weight=0.01),
    )


def linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 0:
        raise ValueError("count must be positive")
    if count == 1:
        return [(start + stop) * 0.5]
    step = (stop - start) / float(count - 1)
    return [start + step * i for i in range(count)]


def make_boundaries(sites: tuple[Site, ...]) -> tuple[Boundary, ...]:
    boundaries: list[Boundary] = []
    for left, right in itertools.combinations(range(len(sites)), 2):
        a = sites[left]
        c = sites[right]
        nx = 2.0 * (c.x - a.x)
        nz = 2.0 * (c.z - a.z)
        nt = 2.0 * (c.t - a.t)
        b = (
            a.x * a.x
            + a.z * a.z
            + a.t * a.t
            - c.x * c.x
            - c.z * c.z
            - c.t * c.t
            - a.weight
            + c.weight
        )
        boundaries.append(Boundary(left=left, right=right, nx=nx, nz=nz, nt=nt, b=b))
    return tuple(boundaries)


def crossing_depth(boundary: Boundary, *, u: float, t: float, camera_velocity_x: float) -> float | None:
    if abs(boundary.nz) < EPS:
        return None
    x = u + camera_velocity_x * t
    return -(boundary.nx * x + boundary.nt * t + boundary.b) / boundary.nz


def sample_events(
    boundaries: Iterable[Boundary],
    *,
    u: float,
    t: float,
    near: float,
    far: float,
    camera_velocity_x: float,
) -> set[tuple[int, int]]:
    events: set[tuple[int, int]] = set()
    for boundary in boundaries:
        s = crossing_depth(boundary, u=u, t=t, camera_velocity_x=camera_velocity_x)
        if s is not None and near <= s <= far:
            events.add((boundary.left, boundary.right))
    return events


def slab_events(
    boundaries: Iterable[Boundary],
    *,
    u: float,
    t0: float,
    t1: float,
    near: float,
    far: float,
    camera_velocity_x: float,
    invalid_epsilon: float,
) -> tuple[set[tuple[int, int]], int]:
    events: set[tuple[int, int]] = set()
    invalid_denominators = 0
    for boundary in boundaries:
        if abs(boundary.nz) < invalid_epsilon:
            invalid_denominators += 1
            continue
        s0 = crossing_depth(boundary, u=u, t=t0, camera_velocity_x=camera_velocity_x)
        s1 = crossing_depth(boundary, u=u, t=t1, camera_velocity_x=camera_velocity_x)
        if s0 is None or s1 is None:
            invalid_denominators += 1
            continue
        if max(min(s0, s1), near) <= min(max(s0, s1), far):
            events.add((boundary.left, boundary.right))
    return events, invalid_denominators


def slab_ranges(count: int) -> list[tuple[float, float]]:
    if count <= 0:
        raise ValueError("time slab count must be positive")
    step = 1.0 / float(count)
    return [(i * step, (i + 1) * step) for i in range(count)]


def profile_frame_count(
    *,
    frames: int,
    u_values: list[float],
    boundaries: tuple[Boundary, ...],
    config: ToyConfig,
) -> dict[str, float | int]:
    frame_times = linspace(0.0, 1.0, frames)
    per_frame_events = 0
    missing_events = 0
    extra_candidate_events = 0
    slab_candidate_events = 0
    invalid_denominators = 0
    slabs = slab_ranges(config.time_slabs)

    slab_cache: dict[tuple[float, int], set[tuple[int, int]]] = {}
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
            slab_cache[(u, slab_i)] = events
            slab_candidate_events += len(events)
            invalid_denominators += invalid

    for u in u_values:
        for t in frame_times:
            sample = sample_events(
                boundaries,
                u=u,
                t=t,
                near=config.near,
                far=config.far,
                camera_velocity_x=config.camera_velocity_x,
            )
            slab_i = min(int(math.floor(t * config.time_slabs)), config.time_slabs - 1)
            candidates = slab_cache[(u, slab_i)]
            per_frame_events += len(sample)
            missing_events += len(sample - candidates)
            extra_candidate_events += len(candidates - sample)

    ratio = float(slab_candidate_events) / float(per_frame_events) if per_frame_events else 0.0
    return {
        "frames": int(frames),
        "u_samples": int(len(u_values)),
        "time_slabs": int(config.time_slabs),
        "per_frame_event_sum": int(per_frame_events),
        "beam_slab_event_sum": int(slab_candidate_events),
        "event_sharing_ratio": ratio,
        "missing_sample_events": int(missing_events),
        "extra_candidate_events": int(extra_candidate_events),
        "invalid_denominator_count": int(invalid_denominators),
    }


def run(config: ToyConfig) -> dict[str, object]:
    sites = default_sites()
    boundaries = make_boundaries(sites)
    u_values = linspace(-1.0, 1.0, config.u_samples)
    rows = [
        profile_frame_count(frames=frames, u_values=u_values, boundaries=boundaries, config=config)
        for frames in config.frame_counts
    ]
    first = rows[0]
    last = rows[-1]
    per_frame_growth = float(last["per_frame_event_sum"]) / float(max(int(first["per_frame_event_sum"]), 1))
    beam_growth = float(last["beam_slab_event_sum"]) / float(max(int(first["beam_slab_event_sum"]), 1))
    return {
        "scenario": "orthographic_2d_time_power_cells",
        "site_count": len(sites),
        "boundary_count": len(boundaries),
        "near": config.near,
        "far": config.far,
        "camera_velocity_x": config.camera_velocity_x,
        "rows": rows,
        "growth": {
            "from_frames": int(first["frames"]),
            "to_frames": int(last["frames"]),
            "per_frame_event_growth": per_frame_growth,
            "beam_event_growth": beam_growth,
            "sublinear_event_growth": beam_growth < per_frame_growth,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="World Foam Lane 2 Gate 0 beam-event toy.")
    parser.add_argument("--frame-counts", default="2,4,8,16")
    parser.add_argument("--u-samples", type=int, default=17)
    parser.add_argument("--time-slabs", type=int, default=1)
    parser.add_argument("--near", type=float, default=0.25)
    parser.add_argument("--far", type=float, default=3.0)
    parser.add_argument("--camera-velocity-x", type=float, default=0.35)
    parser.add_argument("--invalid-epsilon", type=float, default=1.0e-7)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_counts = tuple(int(part) for part in args.frame_counts.split(",") if part.strip())
    config = ToyConfig(
        frame_counts=frame_counts,
        u_samples=args.u_samples,
        time_slabs=args.time_slabs,
        near=args.near,
        far=args.far,
        camera_velocity_x=args.camera_velocity_x,
        invalid_epsilon=args.invalid_epsilon,
    )
    payload = run(config)
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for row in payload["rows"]:
            print(
                "T={frames}: per_frame={per_frame_event_sum} beam={beam_slab_event_sum} "
                "ratio={event_sharing_ratio:.4f} missing={missing_sample_events}".format(**row)
            )
        growth = payload["growth"]
        print(
            "growth {from_frames}->{to_frames}: per_frame={per_frame_event_growth:.3f} "
            "beam={beam_event_growth:.3f} sublinear={sublinear_event_growth}".format(**growth)
        )


if __name__ == "__main__":
    main()
