from __future__ import annotations

import unittest

from gate0_beam_toy import ToyConfig
from gate0_shared_forward_backward import run


class Gate0SharedForwardBackwardTest(unittest.TestCase):
    def test_shared_replay_matches_direct_forward_and_signal_gradients(self) -> None:
        payload = run(
            ToyConfig(
                frame_counts=(2, 4, 8, 16),
                u_samples=17,
                time_slabs=1,
                near=0.25,
                far=3.0,
                camera_velocity_x=0.35,
                invalid_epsilon=1.0e-7,
            )
        )
        acceptance = payload["acceptance"]
        self.assertTrue(acceptance["shared_outputs_match_direct"])
        self.assertTrue(acceptance["shared_segments_match_direct"])
        self.assertTrue(acceptance["shared_gradients_match_direct"])
        self.assertTrue(acceptance["finite_difference_matches_shared_gradient"])
        self.assertTrue(acceptance["shared_forward_backward_scans_sublinear"])

        final_row = payload["rows"][-1]
        self.assertEqual(final_row["shared_backward_boundary_scans"], 0)
        self.assertLess(final_row["shared_forward_backward_boundary_scan_ratio"], 0.1)
        self.assertEqual(final_row["max_output_abs_error"], 0.0)
        self.assertEqual(final_row["signal_gradient_max_abs_error"], 0.0)

    def test_second_velocity_keeps_shared_gradient_parity(self) -> None:
        payload = run(
            ToyConfig(
                frame_counts=(2, 4, 8, 16),
                u_samples=17,
                time_slabs=1,
                near=0.25,
                far=3.0,
                camera_velocity_x=0.7,
                invalid_epsilon=1.0e-7,
            )
        )
        acceptance = payload["acceptance"]
        self.assertTrue(acceptance["shared_outputs_match_direct"])
        self.assertTrue(acceptance["shared_gradients_match_direct"])
        self.assertTrue(acceptance["finite_difference_matches_shared_gradient"])


if __name__ == "__main__":
    unittest.main()
