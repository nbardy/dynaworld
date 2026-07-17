from __future__ import annotations

import unittest

from gate1_realray_per_sample_reference import (
    Site4D,
    crossing_depth_4d,
    make_boundaries_4d,
    render_one_ray,
)


class Gate1RealRayPerSampleReferenceTest(unittest.TestCase):
    def test_4d_power_boundary_crossing_depth(self) -> None:
        sites = (
            Site4D(x=0.0, y=0.0, z=1.0, t=0.0, weight=0.0, rgba=(1.0, 0.0, 0.0, 1.0)),
            Site4D(x=0.0, y=0.0, z=3.0, t=0.0, weight=0.0, rgba=(0.0, 1.0, 0.0, 1.0)),
        )
        (boundary,) = make_boundaries_4d(sites)
        depth = crossing_depth_4d(
            boundary,
            origin=(0.0, 0.0, 0.0),
            direction=(0.0, 0.0, 1.0),
            t=0.0,
            invalid_epsilon=1.0e-7,
        )
        self.assertAlmostEqual(depth, 2.0)

    def test_render_one_ray_visits_both_sites(self) -> None:
        sites = (
            Site4D(x=0.0, y=0.0, z=1.0, t=0.0, weight=0.0, rgba=(1.0, 0.0, 0.0, 1.0)),
            Site4D(x=0.0, y=0.0, z=3.0, t=0.0, weight=0.0, rgba=(0.0, 1.0, 0.0, 1.0)),
        )
        boundaries = make_boundaries_4d(sites)
        rgb, alpha, depth, segment_count, invalid = render_one_ray(
            sites=sites,
            boundaries=boundaries,
            origin=(0.0, 0.0, 0.0),
            direction=(0.0, 0.0, 1.0),
            t=0.0,
            near=0.0,
            far=4.0,
            invalid_epsilon=1.0e-7,
            transmittance_threshold=0.0,
        )
        self.assertEqual(invalid, 0)
        self.assertEqual(segment_count, 2)
        self.assertGreater(rgb[0], rgb[1])
        self.assertGreater(rgb[1], 0.0)
        self.assertGreater(alpha, 0.0)
        self.assertLessEqual(alpha, 1.0)
        self.assertGreater(depth, 0.0)


if __name__ == "__main__":
    unittest.main()
