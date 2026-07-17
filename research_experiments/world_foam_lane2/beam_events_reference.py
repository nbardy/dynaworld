#!/usr/bin/env python3
"""Tiny CPU reference for World Foam Lane 2 Gate 0 beam event counts.

This is intentionally a toy oracle:

- 2D world coordinates plus sampled time.
- Persistent moving disk supports stand in for foam cells.
- Beams are finite rays.
- Traversal events are analytic enter/exit intersections sorted by distance.

No numpy, torch, GPU, viewer, or training code is used.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Iterable


EPS = 1e-9


@dataclass(frozen=True)
class Vec2:
    x: float
    y: float

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scale: float) -> "Vec2":
        return Vec2(self.x * scale, self.y * scale)

    def dot(self, other: "Vec2") -> float:
        return self.x * other.x + self.y * other.y

    def length(self) -> float:
        return math.hypot(self.x, self.y)

    def normalized(self) -> "Vec2":
        length = self.length()
        if length <= EPS:
            raise ValueError("beam direction must be non-zero")
        return Vec2(self.x / length, self.y / length)

    def as_list(self) -> list[float]:
        return [round_float(self.x), round_float(self.y)]


@dataclass(frozen=True)
class MovingDisk:
    disk_id: str
    center0: Vec2
    velocity: Vec2
    radius: float

    def center_at(self, time_value: float) -> Vec2:
        if self.radius <= 0.0:
            raise ValueError(f"{self.disk_id} radius must be positive")
        return self.center0 + self.velocity * time_value


@dataclass(frozen=True)
class Beam:
    beam_id: str
    origin: Vec2
    direction: Vec2
    max_distance: float

    def unit_direction(self) -> Vec2:
        if self.max_distance <= 0.0:
            raise ValueError(f"{self.beam_id} max_distance must be positive")
        return self.direction.normalized()


@dataclass(frozen=True)
class Event:
    distance: float
    kind: str
    disk_id: str

    def to_dict(self) -> dict[str, object]:
        return {
            "distance": round_float(self.distance),
            "kind": self.kind,
            "disk_id": self.disk_id,
        }


@dataclass(frozen=True)
class Interval:
    disk_id: str
    enter: float
    exit: float

    def to_dict(self) -> dict[str, object]:
        return {
            "disk_id": self.disk_id,
            "enter": round_float(self.enter),
            "exit": round_float(self.exit),
            "length": round_float(max(0.0, self.exit - self.enter)),
        }


def round_float(value: float) -> float:
    rounded = round(value, 6)
    if rounded == -0.0:
        return 0.0
    return rounded


def default_disks() -> list[MovingDisk]:
    return [
        MovingDisk("A", center0=Vec2(0.0, 0.0), velocity=Vec2(0.20, 0.00), radius=0.34),
        MovingDisk("B", center0=Vec2(0.82, 0.15), velocity=Vec2(-0.12, -0.10), radius=0.28),
        MovingDisk("C", center0=Vec2(0.45, 0.64), velocity=Vec2(0.00, -0.34), radius=0.24),
    ]


def default_beams() -> list[Beam]:
    return [
        Beam("centerline", origin=Vec2(-0.90, 0.00), direction=Vec2(1.0, 0.0), max_distance=2.40),
        Beam("upper", origin=Vec2(-0.90, 0.55), direction=Vec2(1.0, -0.10), max_distance=2.40),
        Beam("diagonal", origin=Vec2(-0.80, -0.45), direction=Vec2(1.0, 0.45), max_distance=2.20),
    ]


def default_times() -> list[float]:
    return [0.0, 0.5, 1.0]


def ray_disk_interval(beam: Beam, disk: MovingDisk, time_value: float) -> Interval | None:
    """Return the finite-beam interval inside a moving disk at one time."""
    origin_to_center = disk.center_at(time_value) - beam.origin
    direction = beam.unit_direction()

    projection = origin_to_center.dot(direction)
    closest_sq = origin_to_center.dot(origin_to_center) - projection * projection
    radius_sq = disk.radius * disk.radius
    if closest_sq > radius_sq + EPS:
        return None

    half_span = math.sqrt(max(0.0, radius_sq - closest_sq))
    raw_enter = projection - half_span
    raw_exit = projection + half_span

    enter = max(0.0, raw_enter)
    exit = min(beam.max_distance, raw_exit)
    if exit < 0.0 or enter > beam.max_distance or exit - enter <= EPS:
        return None

    return Interval(disk_id=disk.disk_id, enter=enter, exit=exit)


def intervals_to_events(intervals: Iterable[Interval]) -> list[Event]:
    events: list[Event] = []
    for interval in intervals:
        enter_kind = "inside_start" if abs(interval.enter) <= EPS else "enter"
        events.append(Event(interval.enter, enter_kind, interval.disk_id))
        events.append(Event(interval.exit, "exit", interval.disk_id))
    return sorted(events, key=lambda event: (round_float(event.distance), event.kind != "exit", event.disk_id))


def max_overlap_depth(events: Iterable[Event]) -> int:
    active = 0
    max_depth = 0
    for event in events:
        if event.kind in {"enter", "inside_start"}:
            active += 1
            max_depth = max(max_depth, active)
        elif event.kind == "exit":
            active -= 1
        else:
            raise ValueError(f"unknown event kind {event.kind!r}")
    if active != 0:
        raise ValueError(f"unbalanced event stream ended with active={active}")
    return max_depth


def trace_beam_at_time(beam: Beam, disks: Iterable[MovingDisk], time_value: float) -> dict[str, object]:
    intervals = [
        interval
        for disk in sorted(disks, key=lambda item: item.disk_id)
        if (interval := ray_disk_interval(beam, disk, time_value)) is not None
    ]
    intervals = sorted(intervals, key=lambda item: (round_float(item.enter), round_float(item.exit), item.disk_id))
    events = intervals_to_events(intervals)
    return {
        "time": round_float(time_value),
        "beam_id": beam.beam_id,
        "event_count": len(events),
        "hit_count": len(intervals),
        "max_depth": max_overlap_depth(events),
        "coverage_length": round_float(sum(max(0.0, item.exit - item.enter) for item in intervals)),
        "intervals": [interval.to_dict() for interval in intervals],
        "events": [event.to_dict() for event in events],
    }


def run_reference(times: Iterable[float], beams: Iterable[Beam], disks: Iterable[MovingDisk]) -> dict[str, object]:
    time_values = list(times)
    beam_values = list(beams)
    disk_values = list(disks)
    traces = [
        trace_beam_at_time(beam, disk_values, time_value)
        for time_value in time_values
        for beam in beam_values
    ]
    return {
        "reference": "world_foam_lane2_gate0_cpu_beam_events",
        "units": "2D world distance, normalized time",
        "times": [round_float(value) for value in time_values],
        "disk_count": len(disk_values),
        "beam_count": len(beam_values),
        "total_event_count": sum(int(trace["event_count"]) for trace in traces),
        "total_hit_count": sum(int(trace["hit_count"]) for trace in traces),
        "global_max_depth": max(int(trace["max_depth"]) for trace in traces) if traces else 0,
        "traces": traces,
    }


def parse_times(value: str) -> list[float]:
    times = [float(chunk.strip()) for chunk in value.split(",") if chunk.strip()]
    if not times:
        raise argparse.ArgumentTypeError("provide at least one time value")
    return times


def assert_self_test(summary: dict[str, object]) -> None:
    """Pin the default scene so this file can be used as a cheap reference test."""
    expected_counts = {
        (0.0, "centerline"): (4, 2, 1),
        (0.0, "upper"): (4, 2, 1),
        (0.0, "diagonal"): (4, 2, 1),
        (0.5, "centerline"): (4, 2, 1),
        (0.5, "upper"): (2, 1, 1),
        (0.5, "diagonal"): (4, 2, 1),
        (1.0, "centerline"): (4, 2, 2),
        (1.0, "upper"): (2, 1, 1),
        (1.0, "diagonal"): (6, 3, 3),
    }
    traces = summary["traces"]
    if not isinstance(traces, list):
        raise AssertionError("summary traces must be a list")
    actual = {
        (trace["time"], trace["beam_id"]): (
            trace["event_count"],
            trace["hit_count"],
            trace["max_depth"],
        )
        for trace in traces
    }
    if actual != expected_counts:
        raise AssertionError(f"default trace counts changed: {actual!r}")
    expected_totals = {
        "total_event_count": 34,
        "total_hit_count": 17,
        "global_max_depth": 3,
    }
    for key, expected in expected_totals.items():
        actual_value = summary[key]
        if actual_value != expected:
            raise AssertionError(f"{key} expected {expected}, got {actual_value}")


def format_table(summary: dict[str, object]) -> str:
    lines = [
        "time   beam        events  hits  max_depth  coverage",
        "-----  ----------  ------  ----  ---------  --------",
    ]
    traces = summary["traces"]
    if not isinstance(traces, list):
        raise ValueError("summary traces must be a list")
    for trace in traces:
        lines.append(
            f"{trace['time']:<5}  "
            f"{trace['beam_id']:<10}  "
            f"{trace['event_count']:>6}  "
            f"{trace['hit_count']:>4}  "
            f"{trace['max_depth']:>9}  "
            f"{trace['coverage_length']:>8}"
        )
    lines.append("")
    lines.append(
        "totals: "
        f"events={summary['total_event_count']} "
        f"hits={summary['total_hit_count']} "
        f"global_max_depth={summary['global_max_depth']}"
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="CPU-only 2D+time moving-disk beam traversal event-count reference."
    )
    parser.add_argument(
        "--times",
        type=parse_times,
        default=default_times(),
        help="comma-separated normalized times to sample; default: 0,0.5,1",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit full deterministic JSON trace instead of the compact table",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="assert the pinned default event counts before printing output",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = run_reference(args.times, default_beams(), default_disks())
    if args.self_test:
        assert_self_test(summary)

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(format_table(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
