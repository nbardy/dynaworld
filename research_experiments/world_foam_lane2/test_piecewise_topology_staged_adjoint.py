from __future__ import annotations

import dataclasses
import unittest

import torch
from compiled_lie_world_adjoint import (
    AdaptiveCompiledLieWorldAtlas,
    AdaptiveLieWorldCompilePolicy,
    compile_lie_world_atlas,
)
from compiled_transfer_adjoint import direct_word_render, make_stable_cell_word, power_boundary_parameters
from piecewise_topology_staged_adjoint import (
    PreparedTopologyLieChart,
    piecewise_topology_staged_lie_mse_vjp,
)
from staged_compiled_lie_adjoint import (
    allocate_compact_spatial_gradient_buffers,
    refresh_staged_lie_world_snapshot,
)

DTYPE = torch.float64


class PiecewiseTopologyStagedAdjointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Sites 0 and 1 exchange the near owner at t=0.  Site 2 owns the
        # far segment on both sides.  The active face therefore changes from
        # (0,2) to (1,2), while both charts retain strictly positive runs.
        cls.geometry = torch.tensor(
            [
                [0.0, 0.0, 0.0, -0.1, 0.0],
                [0.0, 0.0, 0.0, 0.1, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
            ],
            dtype=DTYPE,
        )
        cls.density = torch.tensor([0.45, 0.7, 0.3], dtype=DTYPE)
        cls.color = torch.tensor(
            [
                [0.85, 0.15, 0.1],
                [0.1, 0.75, 0.2],
                [0.15, 0.25, 0.9],
            ],
            dtype=DTYPE,
        )
        cls.rays = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
            dtype=DTYPE,
        )
        cls.background = torch.tensor([0.03, 0.04, 0.05], dtype=DTYPE)
        cls.left_pairs = torch.tensor([[0, 2]], dtype=torch.int64)
        cls.right_pairs = torch.tensor([[1, 2]], dtype=torch.int64)
        cls.left_word = make_stable_cell_word([0, 2], [-1, 0], [0, -2])
        cls.right_word = make_stable_cell_word([1, 2], [-1, 0], [0, -2])
        cls.charts = (
            cls._prepare_chart(
                chart_id="owner-0-to-2",
                t_min=-1.0,
                t_max=0.0,
                pairs=cls.left_pairs,
                word=cls.left_word,
            ),
            cls._prepare_chart(
                chart_id="owner-1-to-2",
                t_min=0.0,
                t_max=1.0,
                pairs=cls.right_pairs,
                word=cls.right_word,
            ),
        )

    @classmethod
    def _prepare_chart(cls, *, chart_id, t_min, t_max, pairs, word):
        boundary = power_boundary_parameters(cls.geometry, pairs)
        compiled = compile_lie_world_atlas(
            boundary=boundary,
            ray_coefficients=cls.rays,
            words=(word,),
            site_density=cls.density,
            site_color=cls.color,
            t_min=t_min,
            t_max=t_max,
            near=0.0,
            far=1.0,
            node_count=16,
        )
        adaptive = AdaptiveCompiledLieWorldAtlas(
            charts=(compiled,),
            selections=(),
            policy=AdaptiveLieWorldCompilePolicy(node_count_schedule=(16,)),
            supplied_word_ordering_check=compiled.supplied_word_ordering_check,
        )
        snapshot = refresh_staged_lie_world_snapshot(
            adaptive,
            assume_fixed_topology=True,
            boundary=boundary,
            ray_coefficients=cls.rays,
            site_density=cls.density,
            site_color=cls.color,
        )
        return PreparedTopologyLieChart(
            chart_id=chart_id,
            t_min=t_min,
            t_max=t_max,
            world_snapshot=snapshot,
            source_site_ids=torch.arange(3, dtype=torch.int64),
            boundary_site_pairs=pairs,
        )

    def _targets(self, times: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            (
                0.18 + 0.025 * times,
                0.24 - 0.015 * times,
                0.31 + 0.01 * times.square(),
            ),
            dim=-1,
        ).unsqueeze(0)

    def _direct_predictions(
        self,
        times: torch.Tensor,
        *,
        geometry: torch.Tensor | None = None,
        density: torch.Tensor | None = None,
        color: torch.Tensor | None = None,
    ) -> torch.Tensor:
        geometry = self.geometry if geometry is None else geometry
        density = self.density if density is None else density
        color = self.color if color is None else color
        predictions = torch.empty((1, times.numel(), 3), dtype=DTYPE)
        for chart_id, (pairs, word) in enumerate(
            (
                (self.left_pairs, self.left_word),
                (self.right_pairs, self.right_word),
            )
        ):
            mask = times < 0.0 if chart_id == 0 else times >= 0.0
            sample_ids = torch.nonzero(mask, as_tuple=False).reshape(-1)
            if not sample_ids.numel():
                continue
            predictions[:, sample_ids] = direct_word_render(
                boundary=power_boundary_parameters(geometry, pairs),
                ray_coefficients=self.rays,
                words=(word,),
                site_density=density,
                site_color=color,
                times=times.index_select(0, sample_ids),
                background=self.background,
                near=0.0,
                far=1.0,
            )
        return predictions

    def _run(self, times: torch.Tensor, *, frame_block_size: int, return_predictions: bool = True):
        gradients = allocate_compact_spatial_gradient_buffers(
            site_geometry=self.geometry,
            site_density=self.density,
            site_color=self.color,
        )
        pointers = tuple(tensor.untyped_storage().data_ptr() for tensor in gradients.tensors)
        result = piecewise_topology_staged_lie_mse_vjp(
            self.charts,
            site_geometry=self.geometry,
            site_density=self.density,
            site_color=self.color,
            gradients=gradients,
            times=times,
            targets=self._targets(times),
            background=self.background,
            frame_block_size=frame_block_size,
            track_block_size=1,
            loss_normalization_id="owner-transition-fixture",
            return_predictions=return_predictions,
        )
        self.assertEqual(
            tuple(tensor.untyped_storage().data_ptr() for tensor in gradients.tensors),
            pointers,
        )
        return result

    def _direct_loss(
        self,
        times: torch.Tensor,
        *,
        geometry: torch.Tensor | None = None,
        density: torch.Tensor | None = None,
        color: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = self._direct_predictions(
            times,
            geometry=geometry,
            density=density,
            color=color,
        ) - self._targets(times)
        return residual.square().mean()

    def test_owner_transition_matches_direct_reference_and_finite_differences_away_from_seam(self) -> None:
        times = torch.tensor([-0.9, -0.55, -0.2, 0.15, 0.45, 0.85], dtype=DTYPE)
        self.assertFalse(
            torch.equal(
                self.charts[0].world_snapshot.atlas.charts[0].words[0].owners,
                self.charts[1].world_snapshot.atlas.charts[0].words[0].owners,
            )
        )
        self.assertFalse(torch.equal(self.left_pairs, self.right_pairs))
        result = self._run(times, frame_block_size=2)
        direct_predictions = self._direct_predictions(times)
        direct_loss = self._direct_loss(times)
        torch.testing.assert_close(result.predictions, direct_predictions, atol=2.0e-13, rtol=2.0e-12)
        torch.testing.assert_close(result.loss, direct_loss, atol=2.0e-14, rtol=2.0e-13)

        epsilon = 1.0e-6
        finite_geometry = torch.zeros_like(self.geometry)
        for site_id in range(self.geometry.shape[0]):
            for component_id in range(self.geometry.shape[1]):
                plus = self.geometry.clone()
                minus = self.geometry.clone()
                plus[site_id, component_id] += epsilon
                minus[site_id, component_id] -= epsilon
                finite_geometry[site_id, component_id] = (
                    self._direct_loss(times, geometry=plus)
                    - self._direct_loss(times, geometry=minus)
                ) / (2.0 * epsilon)
        finite_density = torch.zeros_like(self.density)
        for site_id in range(self.density.numel()):
            plus = self.density.clone()
            minus = self.density.clone()
            plus[site_id] += epsilon
            minus[site_id] -= epsilon
            finite_density[site_id] = (
                self._direct_loss(times, density=plus)
                - self._direct_loss(times, density=minus)
            ) / (2.0 * epsilon)
        finite_color = torch.zeros_like(self.color)
        for site_id in range(self.color.shape[0]):
            for channel_id in range(self.color.shape[1]):
                plus = self.color.clone()
                minus = self.color.clone()
                plus[site_id, channel_id] += epsilon
                minus[site_id, channel_id] -= epsilon
                finite_color[site_id, channel_id] = (
                    self._direct_loss(times, color=plus)
                    - self._direct_loss(times, color=minus)
                ) / (2.0 * epsilon)

        actual_geometry = torch.cat(
            (
                result.gradients.grad_site_geometry,
                result.gradients.grad_site_weight[:, None],
            ),
            dim=1,
        )
        torch.testing.assert_close(actual_geometry, finite_geometry, atol=2.0e-9, rtol=2.0e-7)
        torch.testing.assert_close(
            result.gradients.grad_site_density,
            finite_density,
            atol=2.0e-9,
            rtol=2.0e-7,
        )
        torch.testing.assert_close(
            result.gradients.grad_site_color,
            finite_color,
            atol=2.0e-9,
            rtol=2.0e-7,
        )

    def test_k_partition_is_invariant_under_one_global_denominator(self) -> None:
        times = torch.tensor([-0.8, -0.3, 0.0, 0.2, 0.65, 0.95], dtype=DTYPE)
        results = [self._run(times, frame_block_size=value) for value in (1, 2, 6)]
        for actual in results[1:]:
            torch.testing.assert_close(actual.loss, results[0].loss, atol=2.0e-15, rtol=2.0e-14)
            torch.testing.assert_close(actual.predictions, results[0].predictions, atol=2.0e-15, rtol=2.0e-14)
            for actual_bar, expected_bar in zip(
                actual.gradients.tensors,
                results[0].gradients.tensors,
                strict=True,
            ):
                torch.testing.assert_close(actual_bar, expected_bar, atol=2.0e-13, rtol=2.0e-11)
        self.assertEqual(results[0].accounting["global_loss_element_count"], times.numel() * 3)
        self.assertEqual(results[0].accounting["chart_sample_counts"], (2, 4))
        self.assertEqual(results[0].accounting["frame_run_reverse_state_elements"], 0)
        self.assertEqual(results[0].accounting["per_sample_run_tape_bytes"], 0)
        self.assertEqual(results[0].accounting["retained_target_bytes"], 0)

    def test_world_reverse_state_and_work_do_not_gain_a_frame_by_run_axis(self) -> None:
        sparse = self._run(
            torch.linspace(-0.95, 0.95, 6, dtype=DTYPE),
            frame_block_size=2,
            return_predictions=False,
        )
        dense = self._run(
            torch.linspace(-0.95, 0.95, 60, dtype=DTYPE),
            frame_block_size=2,
            return_predictions=False,
        )
        self.assertIsNone(sparse.predictions)
        self.assertIsNone(dense.predictions)
        self.assertEqual(sparse.accounting["retained_prediction_bytes"], 0)
        self.assertEqual(dense.accounting["retained_prediction_bytes"], 0)
        self.assertEqual(
            sparse.accounting["peak_local_accumulator_bytes_excluding_atlas"],
            dense.accounting["peak_local_accumulator_bytes_excluding_atlas"],
        )
        self.assertEqual(
            sparse.accounting["step_world_reverse_run_interactions"],
            dense.accounting["step_world_reverse_run_interactions"],
        )
        self.assertGreater(
            dense.accounting["sample_basis_interactions"],
            sparse.accounting["sample_basis_interactions"],
        )

    def test_seam_is_right_assigned_and_event_vjp_is_explicitly_unresolved(self) -> None:
        times = torch.tensor([-0.25, 0.0, 0.25], dtype=DTYPE)
        result = self._run(times, frame_block_size=2)
        torch.testing.assert_close(
            result.predictions[:, 1:2],
            self._direct_predictions(times)[:, 1:2],
            atol=2.0e-13,
            rtol=2.0e-12,
        )
        self.assertEqual(result.accounting["chart_sample_counts"], (1, 2))
        self.assertEqual(len(result.event_gradients), 1)
        event = result.event_gradients[0]
        self.assertEqual(event.time, 0.0)
        self.assertEqual(event.seam_sample_assignment, "right_chart")
        self.assertEqual(event.frozen_topology_parameter_vjp, "right_one_sided")
        self.assertEqual(event.event_time_vjp, "not_implemented")
        self.assertEqual(event.algebraic_event_dispatch_vjp, "unresolved")

    def test_chart_partition_rejects_a_gap_instead_of_silently_dropping_samples(self) -> None:
        broken = (
            self.charts[0],
            dataclasses.replace(self.charts[1], t_min=0.1),
        )
        gradients = allocate_compact_spatial_gradient_buffers(
            site_geometry=self.geometry,
            site_density=self.density,
            site_color=self.color,
        )
        times = torch.tensor([-0.2, 0.05, 0.2], dtype=DTYPE)
        with self.assertRaisesRegex(ValueError, "contiguous half-open partition"):
            piecewise_topology_staged_lie_mse_vjp(
                broken,
                site_geometry=self.geometry,
                site_density=self.density,
                site_color=self.color,
                gradients=gradients,
                times=times,
                targets=self._targets(times),
                background=self.background,
                frame_block_size=2,
            )


if __name__ == "__main__":
    unittest.main()
