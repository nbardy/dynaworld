from __future__ import annotations

import unittest

import torch
from compact_lie_schedule import (
    CompactLieChartSpec,
    compact_lie_world_schedule_from_atlas,
    compact_lie_world_schedule_from_specs,
)
from compiled_lie_frame_density_gate import (
    _compile_inputs,
    _hard_dormant_fixture,
    _scaling_policy,
    _targets,
)
from compiled_lie_world_adjoint import (
    compile_adaptive_lie_world_atlas,
    piecewise_compiled_lie_world_mse_vjp,
)
from compiled_transfer_adjoint import power_boundary_parameters_vjp
from prepared_track_block import prepare_worldfoam_track_block
from staged_compiled_lie_adjoint import (
    accumulate_staged_piecewise_lie_mse,
    allocate_compact_spatial_gradient_buffers,
    begin_compact_spatial_step,
    begin_compact_spatial_step_v2,
    begin_staged_piecewise_lie_mse,
    consume_compact_spatial_block_result,
    finalize_compact_spatial_step,
    finalize_compact_staged_lie_world_vjp,
    finalize_staged_piecewise_lie_world_vjp,
    prepare_compact_staged_lie_world_snapshot,
    prepare_compact_staged_lie_world_snapshot_v2,
    refresh_staged_lie_world_snapshot,
    slice_adaptive_lie_world_atlas_tracks,
)

DTYPE = torch.float64


class StagedCompiledLieAdjointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = _hard_dormant_fixture()
        cls.atlas = compile_adaptive_lie_world_atlas(
            **_compile_inputs(cls.fixture),
            policy=_scaling_policy(),
            track_block_size=1,
            frame_block_size=4,
        )
        cls.world_snapshot = refresh_staged_lie_world_snapshot(
            cls.atlas,
            assume_fixed_topology=True,
            boundary=cls.fixture["boundary"],
            ray_coefficients=cls.fixture["ray_coefficients"],
            site_density=cls.fixture["site_density"],
            site_color=cls.fixture["site_color"],
        )
        cls.atlas = cls.world_snapshot.atlas
        cls.times = torch.linspace(-1.0, 1.0, 31, dtype=DTYPE)
        cls.targets = _targets(cls.times).unsqueeze(0)
        cls.reference = piecewise_compiled_lie_world_mse_vjp(
            cls.atlas,
            boundary=cls.fixture["boundary"],
            ray_coefficients=cls.fixture["ray_coefficients"],
            site_density=cls.fixture["site_density"],
            site_color=cls.fixture["site_color"],
            times=cls.times,
            targets=cls.targets,
            background=cls.fixture["background"],
            frame_block_size=5,
            track_block_size=1,
            return_predictions=True,
        )

    def _run_partition(self, block_size: int):
        accumulator = begin_staged_piecewise_lie_mse(
            self.world_snapshot,
            background=self.fixture["background"],
            total_frame_count=int(self.times.numel()),
            frame_block_size=3,
            track_block_size=1,
        )
        predictions = []
        for start in range(0, int(self.times.numel()), block_size):
            stop = min(start + block_size, int(self.times.numel()))
            predictions.append(
                accumulate_staged_piecewise_lie_mse(
                    accumulator,
                    times=self.times[start:stop],
                    targets=self.targets[:, start:stop],
                    return_predictions=True,
                )
            )
        result = finalize_staged_piecewise_lie_world_vjp(accumulator)
        return accumulator, result, torch.cat(predictions, dim=1)

    def test_stream_partitions_preserve_one_global_loss_and_world_vjp(self) -> None:
        for block_size in (1, 4, 31):
            with self.subTest(block_size=block_size):
                accumulator, result, predictions = self._run_partition(block_size)
                torch.testing.assert_close(result.loss, self.reference.loss, atol=2.0e-15, rtol=2.0e-14)
                torch.testing.assert_close(predictions, self.reference.predictions, atol=2.0e-15, rtol=2.0e-14)
                for actual, expected in (
                    (result.grad_site_density, self.reference.grad_site_density),
                    (result.grad_site_color, self.reference.grad_site_color),
                    (result.grad_depth_coefficients, self.reference.grad_depth_coefficients),
                    (result.grad_boundary, self.reference.grad_boundary),
                ):
                    torch.testing.assert_close(actual, expected, atol=2.0e-13, rtol=2.0e-11)
                self.assertEqual(result.accounting["world_finalize_calls"], 1)
                self.assertEqual(result.accounting["boundary_finalize_calls"], 1)
                self.assertEqual(
                    result.accounting["step_world_reverse_run_interactions"],
                    self.reference.accounting["step_world_reverse_run_interactions"],
                )
                self.assertEqual(result.accounting["retained_target_bytes"], 0)
                self.assertEqual(result.accounting["retained_prediction_bytes"], 0)
                self.assertEqual(
                    result.accounting["sample_weight_evaluation"],
                    "verified_fit_derived_second_form_barycentric",
                )
                self.assertEqual(result.accounting["sample_weight_common_path_complexity"], "O(FJ)")
                self.assertEqual(result.accounting["sample_weight_dense_fallback_interactions"], 0)
                self.assertEqual(result.accounting["sample_weight_dense_fallback_rows"], 0)
                self.assertGreater(result.accounting["sample_weight_linear_interactions"], 0)
                self.assertTrue(accumulator.finalized)

    def test_accumulator_storage_is_independent_of_declared_frame_count(self) -> None:
        sizes = []
        for frame_count in (16, 1024):
            accumulator = begin_staged_piecewise_lie_mse(
                self.world_snapshot,
                background=self.fixture["background"],
                total_frame_count=frame_count,
                frame_block_size=4,
                track_block_size=1,
            )
            sizes.append(accumulator.resident_bytes_excluding_atlas)
            self.assertFalse(any("target" in name or "prediction" in name for name in vars(accumulator)))
        self.assertEqual(sizes[0], sizes[1])

    def test_topology_chart_subset_can_share_a_larger_global_frame_denominator(self) -> None:
        local_count = 5

        def run(global_frame_count: int):
            accumulator = begin_staged_piecewise_lie_mse(
                self.world_snapshot,
                background=self.fixture["background"],
                total_frame_count=local_count,
                global_frame_count=global_frame_count,
                frame_block_size=2,
                track_block_size=1,
            )
            accumulate_staged_piecewise_lie_mse(
                accumulator,
                times=self.times[:local_count],
                targets=self.targets[:, :local_count],
            )
            return finalize_staged_piecewise_lie_world_vjp(accumulator)

        isolated = run(local_count)
        shared = run(int(self.times.numel()))
        scale = local_count / int(self.times.numel())
        torch.testing.assert_close(shared.loss, isolated.loss * scale, atol=2.0e-15, rtol=2.0e-14)
        for actual, reference in (
            (shared.grad_site_density, isolated.grad_site_density),
            (shared.grad_site_color, isolated.grad_site_color),
            (shared.grad_depth_coefficients, isolated.grad_depth_coefficients),
            (shared.grad_boundary, isolated.grad_boundary),
        ):
            torch.testing.assert_close(actual, reference * scale, atol=2.0e-13, rtol=2.0e-11)
        self.assertEqual(shared.accounting["frame_count"], local_count)
        self.assertEqual(shared.accounting["global_frame_count"], int(self.times.numel()))

        with self.assertRaisesRegex(ValueError, "global_frame_count cannot be smaller"):
            begin_staged_piecewise_lie_mse(
                self.world_snapshot,
                background=self.fixture["background"],
                total_frame_count=local_count,
                global_frame_count=local_count - 1,
                frame_block_size=2,
            )

    def test_stage_order_and_global_count_fail_closed(self) -> None:
        accumulator = begin_staged_piecewise_lie_mse(
            self.world_snapshot,
            background=self.fixture["background"],
            total_frame_count=2,
            frame_block_size=1,
        )
        accumulate_staged_piecewise_lie_mse(
            accumulator,
            times=self.times[:1],
            targets=self.targets[:, :1],
        )
        with self.assertRaisesRegex(ValueError, "before all declared frames"):
            finalize_staged_piecewise_lie_world_vjp(accumulator)
        with self.assertRaisesRegex(ValueError, "exceed"):
            accumulate_staged_piecewise_lie_mse(
                accumulator,
                times=self.times[:2],
                targets=self.targets[:, :2],
            )

    def test_global_frame_intervals_reject_gaps_and_overlap_in_constant_state(self) -> None:
        accumulator = begin_staged_piecewise_lie_mse(
            self.world_snapshot,
            background=self.fixture["background"],
            total_frame_count=2,
            frame_block_size=1,
        )
        with self.assertRaisesRegex(ValueError, "next global frame slot"):
            accumulate_staged_piecewise_lie_mse(
                accumulator,
                times=self.times[1:2],
                targets=self.targets[:, 1:2],
                global_frame_start=1,
            )
        accumulate_staged_piecewise_lie_mse(
            accumulator,
            times=self.times[:1],
            targets=self.targets[:, :1],
            global_frame_start=0,
        )
        with self.assertRaisesRegex(ValueError, "next global frame slot"):
            accumulate_staged_piecewise_lie_mse(
                accumulator,
                times=self.times[1:2],
                targets=self.targets[:, 1:2],
                global_frame_start=0,
            )
        accumulate_staged_piecewise_lie_mse(
            accumulator,
            times=self.times[1:2],
            targets=self.targets[:, 1:2],
            global_frame_start=1,
        )
        finalize_staged_piecewise_lie_world_vjp(accumulator)

    def test_world_snapshot_fails_closed_if_mutated_after_refresh(self) -> None:
        density = self.fixture["site_density"].clone()
        snapshot = refresh_staged_lie_world_snapshot(
            self.atlas,
            assume_fixed_topology=True,
            boundary=self.fixture["boundary"],
            ray_coefficients=self.fixture["ray_coefficients"],
            site_density=density,
            site_color=self.fixture["site_color"],
        )
        accumulator = begin_staged_piecewise_lie_mse(
            snapshot,
            background=self.fixture["background"],
            total_frame_count=1,
            frame_block_size=1,
        )
        accumulate_staged_piecewise_lie_mse(
            accumulator,
            times=self.times[:1],
            targets=self.targets[:, :1],
        )
        density.add_(1.0)
        with self.assertRaisesRegex(ValueError, "changed after atlas refresh"):
            finalize_staged_piecewise_lie_world_vjp(accumulator)

    def test_spatial_track_blocks_share_global_normalization_and_sum_world_bars(self) -> None:
        fixture = dict(self.fixture)
        fixture["ray_coefficients"] = self.fixture["ray_coefficients"].repeat(2, 1)
        fixture["words"] = self.fixture["words"] * 2
        atlas = compile_adaptive_lie_world_atlas(
            **_compile_inputs(fixture),
            policy=_scaling_policy(),
            track_block_size=1,
            frame_block_size=4,
        )
        world_snapshot = refresh_staged_lie_world_snapshot(
            atlas,
            assume_fixed_topology=True,
            boundary=fixture["boundary"],
            ray_coefficients=fixture["ray_coefficients"],
            site_density=fixture["site_density"],
            site_color=fixture["site_color"],
        )
        atlas = world_snapshot.atlas
        targets = torch.stack((_targets(self.times), _targets(self.times) + 0.03))
        reference = piecewise_compiled_lie_world_mse_vjp(
            atlas,
            boundary=fixture["boundary"],
            ray_coefficients=fixture["ray_coefficients"],
            site_density=fixture["site_density"],
            site_color=fixture["site_color"],
            times=self.times,
            targets=targets,
            background=fixture["background"],
            frame_block_size=4,
            track_block_size=1,
        )
        total_loss = torch.zeros((), dtype=DTYPE)
        total_density = torch.zeros_like(reference.grad_site_density)
        total_color = torch.zeros_like(reference.grad_site_color)
        total_boundary = torch.zeros_like(reference.grad_boundary)
        block_bytes = []
        for track_id in range(2):
            sliced = slice_adaptive_lie_world_atlas_tracks(
                atlas,
                track_start=track_id,
                track_end=track_id + 1,
            )
            block_snapshot = refresh_staged_lie_world_snapshot(
                sliced,
                assume_fixed_topology=True,
                boundary=fixture["boundary"],
                ray_coefficients=fixture["ray_coefficients"][track_id : track_id + 1],
                site_density=fixture["site_density"],
                site_color=fixture["site_color"],
            )
            accumulator = begin_staged_piecewise_lie_mse(
                block_snapshot,
                background=fixture["background"],
                total_frame_count=int(self.times.numel()),
                global_track_count=2,
                frame_block_size=4,
                track_block_size=1,
            )
            for start in range(0, int(self.times.numel()), 5):
                stop = min(start + 5, int(self.times.numel()))
                accumulate_staged_piecewise_lie_mse(
                    accumulator,
                    times=self.times[start:stop],
                    targets=targets[track_id : track_id + 1, start:stop],
                )
            result = finalize_staged_piecewise_lie_world_vjp(accumulator)
            total_loss += result.loss
            total_density += result.grad_site_density
            total_color += result.grad_site_color
            total_boundary += result.grad_boundary
            block_bytes.append(accumulator.resident_bytes_excluding_atlas)
        torch.testing.assert_close(total_loss, reference.loss, atol=2.0e-15, rtol=2.0e-14)
        torch.testing.assert_close(total_density, reference.grad_site_density, atol=2.0e-13, rtol=2.0e-11)
        torch.testing.assert_close(total_color, reference.grad_site_color, atol=2.0e-13, rtol=2.0e-11)
        torch.testing.assert_close(total_boundary, reference.grad_boundary, atol=2.0e-13, rtol=2.0e-11)
        full_accumulator = begin_staged_piecewise_lie_mse(
            world_snapshot,
            background=fixture["background"],
            total_frame_count=int(self.times.numel()),
            frame_block_size=4,
        )
        self.assertTrue(all(value < full_accumulator.resident_bytes_excluding_atlas for value in block_bytes))

    def test_compact_block_derives_boundaries_and_scatters_site_bars(self) -> None:
        sites = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, -0.45, -0.5475],
            ],
            dtype=DTYPE,
        )
        pairs = torch.tensor([[0, 1]], dtype=torch.int64)
        topology = prepare_worldfoam_track_block(
            self.fixture["words"],
            pairs,
            site_count=2,
            track_start=0,
            track_end=1,
        )
        prepared = prepare_compact_staged_lie_world_snapshot(
            self.atlas,
            topology,
            site_geometry=sites,
            ray_coefficients=self.fixture["ray_coefficients"],
            site_density=self.fixture["site_density"],
            site_color=self.fixture["site_color"],
        )
        torch.testing.assert_close(
            prepared.world_snapshot.boundary,
            self.fixture["boundary"],
            atol=0.0,
            rtol=0.0,
        )
        accumulator = begin_staged_piecewise_lie_mse(
            prepared.world_snapshot,
            background=self.fixture["background"],
            total_frame_count=int(self.times.numel()),
            loss_normalization_id="single-block",
            frame_block_size=4,
            track_block_size=1,
        )
        for start in range(0, int(self.times.numel()), 4):
            stop = min(start + 4, int(self.times.numel()))
            accumulate_staged_piecewise_lie_mse(
                accumulator,
                times=self.times[start:stop],
                targets=self.targets[:, start:stop],
            )
        result = finalize_compact_staged_lie_world_vjp(accumulator, prepared)
        for actual, expected in (
            (result.transfer.loss, self.reference.loss),
            (result.transfer.grad_site_density, self.reference.grad_site_density),
            (result.transfer.grad_site_color, self.reference.grad_site_color),
            (result.transfer.grad_boundary, self.reference.grad_boundary),
        ):
            torch.testing.assert_close(actual, expected, atol=2.0e-13, rtol=2.0e-11)
        expected_site_grad = power_boundary_parameters_vjp(
            sites,
            pairs,
            self.reference.grad_boundary,
        )
        torch.testing.assert_close(
            result.grad_site_geometry,
            expected_site_grad,
            atol=2.0e-13,
            rtol=2.0e-11,
        )
        gradients = allocate_compact_spatial_gradient_buffers(
            site_geometry=sites,
            site_density=self.fixture["site_density"],
            site_color=self.fixture["site_color"],
        )
        ledger = begin_compact_spatial_step(
            template=self.atlas,
            site_geometry=sites,
            ray_coefficients=self.fixture["ray_coefficients"],
            site_density=self.fixture["site_density"],
            site_color=self.fixture["site_color"],
            gradients=gradients,
            global_track_count=1,
            global_frame_count=int(self.times.numel()),
            loss_normalization_id="single-block",
            expected_blocks=(("all", 0, 1),),
        )
        consume_compact_spatial_block_result(
            ledger,
            block_id="all",
            prepared=prepared,
            accumulator=accumulator,
            result=result,
        )
        global_result = finalize_compact_spatial_step(ledger)
        combined_geometry = torch.cat(
            (
                global_result.gradients.grad_site_geometry,
                global_result.gradients.grad_site_weight[:, None],
            ),
            dim=1,
        )
        torch.testing.assert_close(combined_geometry, expected_site_grad)
        torch.testing.assert_close(
            global_result.gradients.grad_site_density,
            self.reference.grad_site_density,
        )
        torch.testing.assert_close(
            global_result.gradients.grad_site_color,
            self.reference.grad_site_color,
        )
        topology.boundary_site_pairs_i32[0] = topology.boundary_site_pairs_i32[0].flip(0)
        with self.assertRaisesRegex(ValueError, "topology tensors changed"):
            prepared.assert_current()

    def test_compact_token_rejects_source_mutation_before_finalize(self) -> None:
        sites = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, -0.45, -0.5475],
            ],
            dtype=DTYPE,
        )
        topology = prepare_worldfoam_track_block(
            self.fixture["words"],
            torch.tensor([[0, 1]], dtype=torch.int64),
            site_count=2,
            track_start=0,
            track_end=1,
        )
        prepared = prepare_compact_staged_lie_world_snapshot(
            self.atlas,
            topology,
            site_geometry=sites,
            ray_coefficients=self.fixture["ray_coefficients"],
            site_density=self.fixture["site_density"],
            site_color=self.fixture["site_color"],
        )
        accumulator = begin_staged_piecewise_lie_mse(
            prepared.world_snapshot,
            background=self.fixture["background"],
            total_frame_count=1,
            frame_block_size=1,
        )
        accumulate_staged_piecewise_lie_mse(
            accumulator,
            times=self.times[:1],
            targets=self.targets[:, :1],
        )
        sites[1, 2].add_(0.01)
        with self.assertRaisesRegex(ValueError, "source world tensors changed"):
            finalize_compact_staged_lie_world_vjp(accumulator, prepared)

    def _compact_spatial_case(self, *, track_count: int = 4, frame_count: int = 7):
        fixture = dict(self.fixture)
        fixture["ray_coefficients"] = self.fixture["ray_coefficients"].repeat(track_count, 1)
        fixture["words"] = self.fixture["words"] * track_count
        atlas = compile_adaptive_lie_world_atlas(
            **_compile_inputs(fixture),
            policy=_scaling_policy(),
            track_block_size=1,
            frame_block_size=3,
        )
        sites = torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, -0.45, -0.5475],
            ],
            dtype=DTYPE,
        )
        pairs = torch.tensor([[0, 1]], dtype=torch.int64)
        times = self.times[:frame_count]
        targets = torch.stack(tuple(_targets(times) + 0.01 * track_id for track_id in range(track_count)))
        reference = piecewise_compiled_lie_world_mse_vjp(
            atlas,
            boundary=fixture["boundary"],
            ray_coefficients=fixture["ray_coefficients"],
            site_density=fixture["site_density"],
            site_color=fixture["site_color"],
            times=times,
            targets=targets,
            background=fixture["background"],
            frame_block_size=3,
            track_block_size=2,
        )
        return fixture, atlas, sites, pairs, times, targets, reference

    def _finalized_compact_record(
        self,
        case,
        *,
        track_start: int,
        track_end: int,
        template=None,
        site_geometry=None,
    ):
        fixture, atlas, sites, pairs, times, targets, _ = case
        selected_template = atlas if template is None else template
        selected_sites = sites if site_geometry is None else site_geometry
        topology = prepare_worldfoam_track_block(
            fixture["words"],
            pairs,
            site_count=2,
            track_start=track_start,
            track_end=track_end,
        )
        prepared = prepare_compact_staged_lie_world_snapshot(
            selected_template,
            topology,
            site_geometry=selected_sites,
            ray_coefficients=fixture["ray_coefficients"],
            site_density=fixture["site_density"],
            site_color=fixture["site_color"],
        )
        accumulator = begin_staged_piecewise_lie_mse(
            prepared.world_snapshot,
            background=fixture["background"],
            total_frame_count=int(times.numel()),
            global_track_count=atlas.track_count,
            loss_normalization_id="shared-step",
            frame_block_size=2,
            track_block_size=max(1, track_end - track_start),
        )
        for frame_start in range(0, int(times.numel()), 3):
            frame_end = min(frame_start + 3, int(times.numel()))
            accumulate_staged_piecewise_lie_mse(
                accumulator,
                times=times[frame_start:frame_end],
                targets=targets[track_start:track_end, frame_start:frame_end],
                global_frame_start=frame_start,
            )
        result = finalize_compact_staged_lie_world_vjp(accumulator, prepared)
        return prepared, accumulator, result

    def _run_compact_spatial_partition(self, case, *, block_size: int):
        fixture, atlas, sites, _, times, _, _ = case
        blocks = tuple(
            (
                f"tracks-{start}-{min(start + block_size, atlas.track_count)}",
                start,
                min(start + block_size, atlas.track_count),
            )
            for start in range(0, atlas.track_count, block_size)
        )
        gradients = allocate_compact_spatial_gradient_buffers(
            site_geometry=sites,
            site_density=fixture["site_density"],
            site_color=fixture["site_color"],
        )
        for tensor in gradients.tensors:
            tensor.fill_(7.0)
        pointers = tuple(tensor.untyped_storage().data_ptr() for tensor in gradients.tensors)
        ledger = begin_compact_spatial_step(
            template=atlas,
            site_geometry=sites,
            ray_coefficients=fixture["ray_coefficients"],
            site_density=fixture["site_density"],
            site_color=fixture["site_color"],
            gradients=gradients,
            global_track_count=atlas.track_count,
            global_frame_count=int(times.numel()),
            loss_normalization_id="shared-step",
            expected_blocks=blocks,
        )
        self.assertTrue(all(int(torch.count_nonzero(tensor).item()) == 0 for tensor in gradients.tensors))
        for block_id, track_start, track_end in blocks:
            prepared, accumulator, result = self._finalized_compact_record(
                case,
                track_start=track_start,
                track_end=track_end,
            )
            consume_compact_spatial_block_result(
                ledger,
                block_id=block_id,
                prepared=prepared,
                accumulator=accumulator,
                result=result,
            )
            self.assertEqual(
                tuple(tensor.untyped_storage().data_ptr() for tensor in gradients.tensors),
                pointers,
            )
        return finalize_compact_spatial_step(ledger), ledger, pointers

    def test_caller_owned_global_bars_are_invariant_for_bp_1_intermediate_and_p(self) -> None:
        case = self._compact_spatial_case()
        fixture, _, sites, pairs, times, _, reference = case
        expected_site_geometry = power_boundary_parameters_vjp(
            sites,
            pairs,
            reference.grad_boundary,
        )
        allocation_accounting = []
        for block_size in (1, 2, 4):
            with self.subTest(block_size=block_size):
                result, ledger, pointers = self._run_compact_spatial_partition(
                    case,
                    block_size=block_size,
                )
                gradients = result.gradients
                self.assertIs(gradients, ledger.gradients)
                self.assertEqual(
                    tuple(tensor.untyped_storage().data_ptr() for tensor in gradients.tensors),
                    pointers,
                )
                torch.testing.assert_close(result.loss, reference.loss, atol=2.0e-13, rtol=2.0e-11)
                torch.testing.assert_close(
                    gradients.grad_site_geometry,
                    expected_site_geometry[:, :4],
                    atol=2.0e-13,
                    rtol=2.0e-11,
                )
                torch.testing.assert_close(
                    gradients.grad_site_weight,
                    expected_site_geometry[:, 4],
                    atol=2.0e-13,
                    rtol=2.0e-11,
                )
                torch.testing.assert_close(
                    gradients.grad_site_density,
                    reference.grad_site_density,
                    atol=2.0e-13,
                    rtol=2.0e-11,
                )
                torch.testing.assert_close(
                    gradients.grad_site_color,
                    reference.grad_site_color,
                    atol=2.0e-13,
                    rtol=2.0e-11,
                )
                self.assertEqual(
                    result.accounting["global_loss_element_count"],
                    4 * int(times.numel()) * 3,
                )
                self.assertEqual(result.accounting["loss_normalization_id"], "shared-step")
                allocation_accounting.append(
                    tuple(
                        result.accounting[key]
                        for key in (
                            "global_site_count",
                            "global_loss_element_count",
                            "global_gradient_buffer_allocations",
                            "global_gradient_buffer_bytes",
                            "step_state_tensor_bytes",
                        )
                    )
                )
                self.assertEqual(result.accounting["global_gradient_buffer_allocations"], 4)
                self.assertEqual(result.accounting["global_gradient_buffer_bytes"], gradients.resident_bytes)
                self.assertEqual(
                    result.accounting["expected_spatial_block_count"],
                    (4 + block_size - 1) // block_size,
                )
                self.assertTrue(ledger.finalized)
        self.assertEqual(allocation_accounting[0], allocation_accounting[1])
        self.assertEqual(allocation_accounting[1], allocation_accounting[2])
        self.assertEqual(fixture["site_density"].shape[0], 2)

    def test_template_free_schedule_drives_mixed_bp_blocks_and_exact_global_bars(self) -> None:
        case = self._compact_spatial_case()
        fixture, atlas, sites, pairs, times, targets, reference = case
        schedule = compact_lie_world_schedule_from_atlas(atlas)
        one_track_schedule = compact_lie_world_schedule_from_atlas(
            slice_adaptive_lie_world_atlas_tracks(atlas, track_start=0, track_end=1)
        )
        self.assertEqual(schedule.resident_bytes, one_track_schedule.resident_bytes)
        self.assertEqual(schedule.selection_signature, one_track_schedule.selection_signature)
        self.assertTrue(all(not hasattr(chart, "coefficients") for chart in schedule.charts))

        blocks = (("one", 0, 1), ("two", 1, 3), ("last", 3, 4))
        gradients = allocate_compact_spatial_gradient_buffers(
            site_geometry=sites,
            site_density=fixture["site_density"],
            site_color=fixture["site_color"],
        )
        ledger = begin_compact_spatial_step_v2(
            schedule=schedule,
            site_geometry=sites,
            ray_coefficients=fixture["ray_coefficients"],
            site_density=fixture["site_density"],
            site_color=fixture["site_color"],
            gradients=gradients,
            global_track_count=atlas.track_count,
            global_frame_count=int(times.numel()),
            loss_normalization_id="template-free-step",
            expected_blocks=blocks,
        )
        self.assertIsNone(ledger.template)
        for block_id, track_start, track_end in blocks:
            topology = prepare_worldfoam_track_block(
                fixture["words"],
                pairs,
                site_count=2,
                track_start=track_start,
                track_end=track_end,
            )
            prepared = prepare_compact_staged_lie_world_snapshot_v2(
                schedule,
                topology,
                site_geometry=sites,
                ray_coefficients=fixture["ray_coefficients"],
                site_density=fixture["site_density"],
                site_color=fixture["site_color"],
            )
            self.assertIsNone(prepared.template)
            self.assertIs(prepared.schedule, schedule)
            self.assertEqual(prepared.world_snapshot.atlas.track_count, track_end - track_start)
            accumulator = begin_staged_piecewise_lie_mse(
                prepared.world_snapshot,
                background=fixture["background"],
                total_frame_count=int(times.numel()),
                global_track_count=atlas.track_count,
                loss_normalization_id="template-free-step",
                frame_block_size=3,
                track_block_size=track_end - track_start,
            )
            accumulate_staged_piecewise_lie_mse(
                accumulator,
                times=times,
                targets=targets[track_start:track_end],
            )
            result = finalize_compact_staged_lie_world_vjp(accumulator, prepared)
            consume_compact_spatial_block_result(
                ledger,
                block_id=block_id,
                prepared=prepared,
                accumulator=accumulator,
                result=result,
            )

        actual = finalize_compact_spatial_step(ledger)
        expected_geometry = power_boundary_parameters_vjp(sites, pairs, reference.grad_boundary)
        torch.testing.assert_close(actual.loss, reference.loss, atol=2.0e-13, rtol=2.0e-11)
        torch.testing.assert_close(
            actual.gradients.grad_site_geometry,
            expected_geometry[:, :4],
            atol=2.0e-13,
            rtol=2.0e-11,
        )
        torch.testing.assert_close(
            actual.gradients.grad_site_weight,
            expected_geometry[:, 4],
            atol=2.0e-13,
            rtol=2.0e-11,
        )
        torch.testing.assert_close(
            actual.gradients.grad_site_density,
            reference.grad_site_density,
            atol=2.0e-13,
            rtol=2.0e-11,
        )
        torch.testing.assert_close(
            actual.gradients.grad_site_color,
            reference.grad_site_color,
            atol=2.0e-13,
            rtol=2.0e-11,
        )
        self.assertEqual(actual.accounting["chart_schedule_bytes"], schedule.resident_bytes)
        self.assertEqual(actual.accounting["full_global_atlas_retained"], 0)
        template_node_times = atlas.charts[0].transfer_atlas.node_times.clone()
        one_track_schedule.charts[0].node_times.add_(0.125)
        with self.assertRaisesRegex(ValueError, "schedule tensors changed"):
            one_track_schedule.assert_current()
        torch.testing.assert_close(
            atlas.charts[0].transfer_atlas.node_times,
            template_node_times,
            atol=0.0,
            rtol=0.0,
        )

    def test_production_count_schedule_needs_no_full_p_atlas(self) -> None:
        specs = (
            CompactLieChartSpec(0.0, 0.4, 0.1, 3.0, 2),
            CompactLieChartSpec(0.4, 1.0, 0.1, 3.0, 5),
        )
        tiny = compact_lie_world_schedule_from_specs(
            specs,
            global_track_count=1,
            selection_provenance="unit-test-predeclared-ranks-v1",
        )
        production = compact_lie_world_schedule_from_specs(
            specs,
            global_track_count=12_000_000,
            selection_provenance="unit-test-predeclared-ranks-v1",
        )
        self.assertEqual(tiny.resident_bytes, production.resident_bytes)
        self.assertEqual(production.selection_signature, ((0.0, 0.4, 2), (0.4, 1.0, 5)))
        self.assertEqual(
            production.resident_bytes,
            (2 + 2 * 2 + 2 + 5 + 5 * 5 + 5) * torch.tensor([], dtype=DTYPE).element_size(),
        )
        self.assertTrue(
            all(
                tensor.numel() <= 25
                for chart in production.charts
                for tensor in (chart.node_times, chart.fit_matrix, chart.barycentric_weights)
            )
        )
        self.assertNotEqual(tiny.generation_digest, production.generation_digest)

    def test_spatial_accumulator_rejects_overlap_missing_duplicate_and_mixed_tokens(self) -> None:
        case = self._compact_spatial_case(track_count=4, frame_count=2)
        fixture, atlas, sites, _, times, _, _ = case
        gradients = allocate_compact_spatial_gradient_buffers(
            site_geometry=sites,
            site_density=fixture["site_density"],
            site_color=fixture["site_color"],
        )
        common = {
            "template": atlas,
            "site_geometry": sites,
            "ray_coefficients": fixture["ray_coefficients"],
            "site_density": fixture["site_density"],
            "site_color": fixture["site_color"],
            "gradients": gradients,
            "global_track_count": 4,
            "global_frame_count": int(times.numel()),
            "loss_normalization_id": "shared-step",
        }
        with self.assertRaisesRegex(ValueError, "ordered half-open tiling"):
            begin_compact_spatial_step(
                **common,
                expected_blocks=(("left", 0, 3), ("overlap", 2, 4)),
            )
        ledger = begin_compact_spatial_step(
            **common,
            expected_blocks=(("left", 0, 2), ("right", 2, 4)),
        )
        left = self._finalized_compact_record(case, track_start=0, track_end=2)
        right = self._finalized_compact_record(case, track_start=2, track_end=4)
        with self.assertRaisesRegex(ValueError, "do not match"):
            consume_compact_spatial_block_result(
                ledger,
                block_id="left",
                prepared=right[0],
                accumulator=left[1],
                result=left[2],
            )
        other_template = slice_adaptive_lie_world_atlas_tracks(
            atlas,
            track_start=0,
            track_end=atlas.track_count,
        )
        mixed_template = self._finalized_compact_record(
            case,
            track_start=0,
            track_end=2,
            template=other_template,
        )
        with self.assertRaisesRegex(ValueError, "different global atlas template"):
            consume_compact_spatial_block_result(
                ledger,
                block_id="left",
                prepared=mixed_template[0],
                accumulator=mixed_template[1],
                result=mixed_template[2],
            )
        mixed_source = self._finalized_compact_record(
            case,
            track_start=0,
            track_end=2,
            site_geometry=sites.clone(),
        )
        with self.assertRaisesRegex(ValueError, "source world tensors"):
            consume_compact_spatial_block_result(
                ledger,
                block_id="left",
                prepared=mixed_source[0],
                accumulator=mixed_source[1],
                result=mixed_source[2],
            )
        consume_compact_spatial_block_result(
            ledger,
            block_id="left",
            prepared=left[0],
            accumulator=left[1],
            result=left[2],
        )
        with self.assertRaisesRegex(ValueError, "missing track blocks"):
            finalize_compact_spatial_step(ledger)
        with self.assertRaisesRegex(ValueError, "already consumed"):
            consume_compact_spatial_block_result(
                ledger,
                block_id="left",
                prepared=left[0],
                accumulator=left[1],
                result=left[2],
            )
        right[1].loss_normalization_id = "wrong-step"
        with self.assertRaisesRegex(ValueError, "different global loss normalization"):
            consume_compact_spatial_block_result(
                ledger,
                block_id="right",
                prepared=right[0],
                accumulator=right[1],
                result=right[2],
            )
        right[1].loss_normalization_id = "shared-step"
        consume_compact_spatial_block_result(
            ledger,
            block_id="right",
            prepared=right[0],
            accumulator=right[1],
            result=right[2],
        )
        finalize_compact_spatial_step(ledger)
        self.assertTrue(ledger.finalized)


if __name__ == "__main__":
    unittest.main()
