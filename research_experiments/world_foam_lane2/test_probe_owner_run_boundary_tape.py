from __future__ import annotations

import unittest

from probe_owner_run_boundary_tape import OwnerRunRecord, _profile_packed_delta_owner_run_storage


class OwnerRunBoundaryTapeProbeTest(unittest.TestCase):
    def test_packed_delta_storage_counts_only_changed_frame_rows(self) -> None:
        first = (
            OwnerRunRecord(owner=0, left_cut_id=-1, right_cut_id=2, length=1.0, segment_count=1),
            OwnerRunRecord(owner=1, left_cut_id=2, right_cut_id=-2, length=2.0, segment_count=1),
        )
        changed = (
            OwnerRunRecord(owner=0, left_cut_id=-1, right_cut_id=-2, length=3.0, segment_count=2),
        )
        sequences = [[first, first, changed]]

        profile = _profile_packed_delta_owner_run_storage(sequences, frame_count=3)

        self.assertEqual(profile["base_record_count"], 2)
        self.assertEqual(profile["change_event_count"], 1)
        self.assertEqual(profile["change_record_count"], 1)
        self.assertEqual(profile["unchanged_frame_rows"], 2)
        self.assertEqual(profile["changed_frame_rows"], 1)
        self.assertEqual(profile["packed_i32_storage_bytes"], 40)
        self.assertEqual(profile["separate_i32_storage_bytes"], 64)
        self.assertLess(profile["packed_i32_vs_materialized_boundary_csr"], 1.0)

    def test_packed_delta_storage_rejects_bad_frame_count(self) -> None:
        sequences = [[(OwnerRunRecord(owner=0, left_cut_id=-1, right_cut_id=-2, length=1.0, segment_count=1),)]]

        with self.assertRaisesRegex(ValueError, "frame_count"):
            _profile_packed_delta_owner_run_storage(sequences, frame_count=2)


if __name__ == "__main__":
    unittest.main()
