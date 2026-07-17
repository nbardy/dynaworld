from __future__ import annotations

import unittest
from types import SimpleNamespace

from probe_endpoint_record_delta_replay import pack_endpoint_record_delta_replace_tape


def _record(owner: int, left: int, right: int) -> SimpleNamespace:
    return SimpleNamespace(owner=owner, left_cut_id=left, right_cut_id=right)


class EndpointRecordDeltaReplayTests(unittest.TestCase):
    def test_pack_replacement_rows_skips_unchanged_frames(self) -> None:
        sequences = [
            [
                (_record(1, -1, 2), _record(3, 2, -2)),
                (_record(1, -1, 2), _record(3, 2, -2)),
                (_record(4, -1, 5),),
            ],
            [
                (_record(2, -1, -2),),
                (_record(5, -1, 6), _record(7, 6, -2)),
                (_record(5, -1, 6), _record(7, 6, -2)),
            ],
        ]

        tape = pack_endpoint_record_delta_replace_tape(sequences, frame_count=3)

        self.assertEqual(tape.base_offsets_i32.tolist(), [0, 2, 3])
        self.assertEqual(tape.base_owner_i32.tolist(), [1, 3, 2])
        self.assertEqual(tape.base_left_i32.tolist(), [-1, 2, -1])
        self.assertEqual(tape.base_right_i32.tolist(), [2, -2, -2])
        self.assertEqual(tape.track_change_offsets_i32.tolist(), [0, 1, 2])
        self.assertEqual(tape.change_frame_i32.tolist(), [2, 1])
        self.assertEqual(tape.change_offsets_i32.tolist(), [0, 1, 3])
        self.assertEqual(tape.change_owner_i32.tolist(), [4, 5, 7])
        self.assertEqual(tape.change_left_i32.tolist(), [-1, -1, 6])
        self.assertEqual(tape.change_right_i32.tolist(), [5, 6, -2])


if __name__ == "__main__":
    unittest.main()
