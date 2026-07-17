from __future__ import annotations

import unittest

import numpy as np

from analyze_gate4_owner_transition_preflight import (
    _cut_groups_from_ordered_depth_ids,
    analyze_cut_owner_transition,
    analyze_cut_owner_transition_groups,
)
from gate4_affine_slab_tape import _boundary_other_by_owner


class AnalyzeGate4OwnerTransitionPreflightTests(unittest.TestCase):
    def test_keep_policy_crosses_unrelated_boundary_without_rescan(self) -> None:
        boundary_left = np.array([0, 1], dtype=np.int64)
        boundary_right = np.array([2, 3], dtype=np.int64)
        stats = analyze_cut_owner_transition(
            cut_depths=np.array([0.0, 0.25, 0.50, 1.0], dtype=np.float64),
            cut_ids=np.array([-1, 1, 0, -2], dtype=np.int64),
            full_owners_by_nonempty_segment=np.array([0, 0, 2], dtype=np.int64),
            boundary_left=boundary_left,
            boundary_right=boundary_right,
            boundary_other_by_owner=_boundary_other_by_owner(
                boundary_left=boundary_left,
                boundary_right=boundary_right,
                site_count=4,
            ),
            site_density=np.zeros((4,), dtype=np.float64),
            transmittance_threshold=0.0,
            unrelated_policy="keep",
        )

        self.assertEqual(stats.mismatches, 0)
        self.assertEqual(stats.active_segments, 3)
        self.assertEqual(stats.transition_owner_scans, 1)
        self.assertEqual(stats.unrelated_boundary_crossings, 1)
        self.assertEqual(stats.owner_switches, 1)

    def test_fallback_policy_rescans_after_unrelated_boundary(self) -> None:
        boundary_left = np.array([0, 1], dtype=np.int64)
        boundary_right = np.array([2, 3], dtype=np.int64)
        stats = analyze_cut_owner_transition(
            cut_depths=np.array([0.0, 0.25, 0.50, 1.0], dtype=np.float64),
            cut_ids=np.array([-1, 1, 0, -2], dtype=np.int64),
            full_owners_by_nonempty_segment=np.array([0, 0, 2], dtype=np.int64),
            boundary_left=boundary_left,
            boundary_right=boundary_right,
            boundary_other_by_owner=_boundary_other_by_owner(
                boundary_left=boundary_left,
                boundary_right=boundary_right,
                site_count=4,
            ),
            site_density=np.zeros((4,), dtype=np.float64),
            transmittance_threshold=0.0,
            unrelated_policy="fallback",
        )

        self.assertEqual(stats.mismatches, 0)
        self.assertEqual(stats.active_segments, 3)
        self.assertEqual(stats.transition_owner_scans, 2)
        self.assertEqual(stats.fallback_resets, 1)

    def test_detects_wrong_transition_owner(self) -> None:
        boundary_left = np.array([0], dtype=np.int64)
        boundary_right = np.array([1], dtype=np.int64)
        stats = analyze_cut_owner_transition(
            cut_depths=np.array([0.0, 0.5, 1.0], dtype=np.float64),
            cut_ids=np.array([-1, 0, -2], dtype=np.int64),
            full_owners_by_nonempty_segment=np.array([0, 0], dtype=np.int64),
            boundary_left=boundary_left,
            boundary_right=boundary_right,
            boundary_other_by_owner=_boundary_other_by_owner(
                boundary_left=boundary_left,
                boundary_right=boundary_right,
                site_count=2,
            ),
            site_density=np.zeros((2,), dtype=np.float64),
            transmittance_threshold=0.0,
            unrelated_policy="keep",
        )

        self.assertEqual(stats.mismatches, 1)

    def test_grouped_policy_switches_on_collapsed_current_owner_boundary(self) -> None:
        boundary_left = np.array([0, 0], dtype=np.int64)
        boundary_right = np.array([2, 1], dtype=np.int64)
        cut_depths, groups = _cut_groups_from_ordered_depth_ids(
            depths=np.array([0.5, 0.5000001], dtype=np.float64),
            boundary_ids=np.array([0, 1], dtype=np.int64),
            near=0.0,
            far=1.0,
        )
        stats = analyze_cut_owner_transition_groups(
            cut_depths=cut_depths,
            cut_boundary_groups=groups,
            full_owners_by_nonempty_segment=np.array([0, 1], dtype=np.int64),
            boundary_left=boundary_left,
            boundary_right=boundary_right,
            boundary_other_by_owner=_boundary_other_by_owner(
                boundary_left=boundary_left,
                boundary_right=boundary_right,
                site_count=3,
            ),
            site_density=np.zeros((3,), dtype=np.float64),
            transmittance_threshold=0.0,
        )

        self.assertEqual(groups[1], (0, 1))
        self.assertEqual(stats.mismatches, 0)
        self.assertEqual(stats.transition_owner_scans, 2)
        self.assertEqual(stats.ambiguous_boundary_groups, 1)


if __name__ == "__main__":
    unittest.main()
