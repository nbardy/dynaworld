from __future__ import annotations

import copy
import unittest

from compiled_lie_frame_density_gate import (
    assert_compiled_lie_frame_density_report,
    build_compiled_lie_frame_density_report,
    verify_compiled_lie_frame_density_report,
)


class CompiledLieFrameDensityGateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_compiled_lie_frame_density_report(
            frame_counts=(16, 64, 256),
            frame_block_size=4,
        )

    def test_verified_report_has_world_tubes_shaped_work_and_memory(self) -> None:
        self.assertTrue(self.report["verified"])
        self.assertEqual(verify_compiled_lie_frame_density_report(self.report), [])
        assert_compiled_lie_frame_density_report(self.report)
        self.assertEqual(
            self.report["selection_signature"],
            [[-1.0, -0.5, 16], [-0.5, 0.0, 2], [0.0, 1.0, 2]],
        )
        rows = self.report["rows"]
        self.assertEqual(
            {row["refresh_world_forward_run_interactions"] for row in rows},
            {40},
        )
        self.assertEqual(
            {row["step_world_reverse_run_interactions"] for row in rows},
            {40},
        )
        reverse_state_bytes = {
            row[
                "logical_selected_reverse_state_bytes_excluding_targets_and_predictions"
            ]
            for row in rows
        }
        self.assertEqual(len(reverse_state_bytes), 1)
        self.assertGreater(next(iter(reverse_state_bytes)), 0)
        self.assertEqual({row["world_finalize_calls"] for row in rows}, {1})
        self.assertEqual({row["boundary_finalize_calls"] for row in rows}, {1})
        self.assertEqual({row["retained_target_bytes"] for row in rows}, {0})
        self.assertEqual({row["retained_prediction_bytes"] for row in rows}, {0})
        self.assertEqual([row["exact_replay_reverse_run_interactions"] for row in rows], [64, 256, 1024])
        self.assertEqual(
            [row["compiled_total_interaction_proxy"] for row in rows],
            [520, 1048, 3160],
        )
        self.assertEqual(
            [row["exact_total_interaction_proxy"] for row in rows],
            [96, 384, 1536],
        )

    def test_verifier_rejects_hidden_frame_work_memory_and_gradient_drift(self) -> None:
        mutations = (
            ("world", lambda report: report["rows"][-1].__setitem__("step_world_reverse_run_interactions", 41)),
            (
                "memory",
                lambda report: report["rows"][-1].__setitem__(
                    "logical_selected_reverse_state_bytes_excluding_targets_and_predictions",
                    5000,
                ),
            ),
            (
                "gradient",
                lambda report: report["rows"][-1]["error"].__setitem__(
                    "site_density_grad_max_abs",
                    1.0,
                ),
            ),
            ("rank", lambda report: report.__setitem__("selected_rank_independent_of_frame_count", False)),
            (
                "nan-error",
                lambda report: report["rows"][-1]["error"].__setitem__(
                    "site_density_grad_max_abs",
                    float("nan"),
                ),
            ),
            (
                "forged-chart-count",
                lambda report: report["rows"][-1].__setitem__("chart_count", 99),
            ),
            (
                "forged-node-count",
                lambda report: [
                    row.__setitem__("total_node_count", 21) for row in report["rows"]
                ],
            ),
            (
                "zero-structural-bytes",
                lambda report: report.__setitem__("atlas_structural_bytes", 0),
            ),
            (
                "zero-prepared-bytes",
                lambda report: report.__setitem__("prepared_track_block_bytes", 0),
            ),
            (
                "false-scope",
                lambda report: report["scope"].__setitem__("continuous_jacobian_certificate", True),
            ),
            (
                "nonfinite-signature",
                lambda report: report["selection_signature"][0].__setitem__(0, float("nan")),
            ),
            (
                "repeated-world-finalize",
                lambda report: report["rows"][-1].__setitem__("world_finalize_calls", 2),
            ),
            (
                "repeated-boundary-finalize",
                lambda report: report["rows"][-1].__setitem__("boundary_finalize_calls", 2),
            ),
            (
                "retained-targets",
                lambda report: report["rows"][-1].__setitem__("retained_target_bytes", 24),
            ),
            (
                "forged-sample-block-count",
                lambda report: report["rows"][-1].__setitem__("sample_block_count", 1),
            ),
            (
                "hidden-sample-weight-work",
                lambda report: report["rows"][-1].__setitem__(
                    "sample_weight_dense_fallback_interactions",
                    64,
                ),
            ),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                broken = copy.deepcopy(self.report)
                mutate(broken)
                self.assertTrue(verify_compiled_lie_frame_density_report(broken))

    def test_verifier_treats_run_count_as_already_summed_across_tracks(self) -> None:
        multi_track = copy.deepcopy(self.report)
        for row in multi_track["rows"]:
            row["track_count"] = 2
            row["sample_basis_interactions"] *= 2
            row["coefficient_fit_interactions"] *= 2
            row["target_bytes"] *= 2
            row["prediction_bytes"] *= 2
            row["compiled_total_interaction_proxy"] = (
                row["refresh_world_forward_run_interactions"]
                + row["step_world_reverse_run_interactions"]
                + row["sample_basis_interactions"]
                + row["coefficient_fit_interactions"]
                + row["sample_weight_interactions"]
            )
            row["compiled_to_exact_total_interaction_proxy_ratio"] = (
                row["compiled_total_interaction_proxy"]
                / row["exact_total_interaction_proxy"]
            )
        self.assertEqual(verify_compiled_lie_frame_density_report(multi_track), [])

    def test_piecewise_basis_accounting_accepts_endpoint_rounding(self) -> None:
        uneven = build_compiled_lie_frame_density_report(
            frame_counts=(16, 17),
            frame_block_size=4,
        )
        self.assertEqual(verify_compiled_lie_frame_density_report(uneven), [])
        self.assertNotEqual(
            uneven["rows"][0]["sample_basis_interactions"] / 16,
            uneven["rows"][1]["sample_basis_interactions"] / 17,
        )


if __name__ == "__main__":
    unittest.main()
