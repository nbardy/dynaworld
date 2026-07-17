from __future__ import annotations

import unittest

from gate0_beam_toy import ToyConfig, run


class Gate0BeamToyTest(unittest.TestCase):
    def test_default_sweep_has_no_missing_events_and_sublinear_growth(self) -> None:
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
        self.assertTrue(payload["growth"]["sublinear_event_growth"])
        self.assertEqual(payload["growth"]["beam_event_growth"], 1.0)
        rows = payload["rows"]
        self.assertEqual([row["missing_sample_events"] for row in rows], [0, 0, 0, 0])
        self.assertLess(rows[-1]["event_sharing_ratio"], rows[0]["event_sharing_ratio"])

    def test_second_velocity_sweep_has_no_missing_events_and_sublinear_growth(self) -> None:
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
        self.assertTrue(payload["growth"]["sublinear_event_growth"])
        self.assertEqual([row["missing_sample_events"] for row in payload["rows"]], [0, 0, 0, 0])


if __name__ == "__main__":
    unittest.main()
