from __future__ import annotations

import unittest
from types import SimpleNamespace

from probe_endpoint_record_delta_tape import _edit_distance, _profile_endpoint_records


class EndpointRecordDeltaTapeTest(unittest.TestCase):
    def test_edit_distance_counts_endpoint_record_changes(self) -> None:
        left = ((0, -1, 3), (1, 3, -2))
        right = ((0, -1, 3), (2, 3, 4), (1, 4, -2))

        self.assertEqual(_edit_distance(left, right), 2)

    def test_profile_endpoint_records_reports_delta_storage(self) -> None:
        sequences = [
            [
                (
                    SimpleNamespace(owner=0, left_cut_id=-1, right_cut_id=2),
                    SimpleNamespace(owner=1, left_cut_id=2, right_cut_id=-2),
                ),
                (
                    SimpleNamespace(owner=0, left_cut_id=-1, right_cut_id=2),
                    SimpleNamespace(owner=2, left_cut_id=2, right_cut_id=-2),
                ),
            ]
        ]

        profile = _profile_endpoint_records(sequences, frame_count=2, full_segment_count=8)

        self.assertEqual(profile["total_endpoint_records"], 4)
        self.assertEqual(profile["change_events"], 1)
        self.assertEqual(profile["edit_ops_total"], 1)
        self.assertGreater(
            profile["storage_estimates"]["delta_edit_op_endpoint_record_stream_vs_full_segment_csr"],
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
