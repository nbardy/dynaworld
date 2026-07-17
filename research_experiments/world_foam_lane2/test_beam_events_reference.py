#!/usr/bin/env python3
"""Stdlib tests for the World Foam Lane 2 Gate 0 CPU reference."""

from __future__ import annotations

import unittest

from beam_events_reference import (
    Beam,
    MovingDisk,
    Vec2,
    assert_self_test,
    default_beams,
    default_disks,
    default_times,
    ray_disk_interval,
    run_reference,
)


class BeamEventsReferenceTest(unittest.TestCase):
    def test_default_scene_is_pinned(self) -> None:
        summary = run_reference(default_times(), default_beams(), default_disks())
        assert_self_test(summary)

    def test_miss_returns_no_interval(self) -> None:
        beam = Beam("miss", origin=Vec2(0.0, 2.0), direction=Vec2(1.0, 0.0), max_distance=2.0)
        disk = MovingDisk("disk", center0=Vec2(0.5, 0.0), velocity=Vec2(0.0, 0.0), radius=0.25)
        self.assertIsNone(ray_disk_interval(beam, disk, 0.0))

    def test_origin_inside_disk_starts_at_zero(self) -> None:
        beam = Beam("inside", origin=Vec2(0.0, 0.0), direction=Vec2(1.0, 0.0), max_distance=2.0)
        disk = MovingDisk("disk", center0=Vec2(0.2, 0.0), velocity=Vec2(0.0, 0.0), radius=0.5)
        interval = ray_disk_interval(beam, disk, 0.0)
        self.assertIsNotNone(interval)
        assert interval is not None
        self.assertAlmostEqual(interval.enter, 0.0)
        self.assertAlmostEqual(interval.exit, 0.7)


if __name__ == "__main__":
    unittest.main()
