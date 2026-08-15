from __future__ import annotations

import inspect
from dataclasses import replace

import pytest
import torch
from kinetic_multichart_transfer_program import (
    compile_kinetic_multichart_p0_program,
    refresh_kinetic_multichart_p0_transfer,
)
from kinetic_native_precompiled_length_adapter import (
    FORWARD_OP_NAME,
    RUNTIME_STATUS,
    VJP_OP_NAME,
    execute_kinetic_native_precompiled_length_node_vjp,
    prepare_kinetic_native_precompiled_length_topology_token,
    refresh_kinetic_native_precompiled_length_world_token,
)
from kinetic_native_precompiled_length_oracle import (
    kinetic_native_precompiled_length_node_vjp,
    refresh_kinetic_native_precompiled_length_world,
)
from kinetic_native_topology_lowering import (
    lower_kinetic_multichart_to_native_topology,
    materialize_kinetic_native_topology_chart,
)
from kinetic_owner_chart_compiler import compile_exact_kinetic_owner_charts
from kinetic_power_word_compiler import AffineKineticPowerSites

DTYPE = torch.float64


class _FakeNativeOps:
    """Differentiable CPU double for the two source-level launch functions."""

    def __init__(self) -> None:
        self.forward_calls: list[dict[str, object]] = []
        self.vjp_calls: list[dict[str, object]] = []

    def kinetic_precompiled_length_p0_lie_node_forward_launch_only(
        self,
        word_offsets_i32: torch.Tensor,
        word_owner_i32: torch.Tensor,
        node_physical_length_f32: torch.Tensor,
        site_rgba_f32: torch.Tensor,
        config_i32: torch.Tensor,
        config_f32: torch.Tensor,
        *,
        track_count: int,
        node_count: int,
    ) -> torch.Tensor:
        self._validate_launch(
            word_offsets_i32,
            word_owner_i32,
            node_physical_length_f32,
            site_rgba_f32,
            config_i32,
            config_f32,
            track_count=track_count,
            node_count=node_count,
        )
        self.forward_calls.append(
            {
                "track_count": track_count,
                "node_count": node_count,
                "config_i32": tuple(config_i32.tolist()),
                "config_f32": tuple(config_f32.tolist()),
            }
        )
        return self._node_charts(
            word_owner_i32,
            node_physical_length_f32,
            site_rgba_f32,
        )

    def kinetic_precompiled_length_p0_lie_node_vjp_accumulate_launch_only(
        self,
        word_offsets_i32: torch.Tensor,
        word_owner_i32: torch.Tensor,
        node_physical_length_f32: torch.Tensor,
        site_rgba_f32: torch.Tensor,
        node_chart_f32: torch.Tensor,
        grad_node_chart_f32: torch.Tensor,
        grad_site_rgba_f32: torch.Tensor,
        config_i32: torch.Tensor,
        config_f32: torch.Tensor,
        *,
        track_count: int,
        node_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_launch(
            word_offsets_i32,
            word_owner_i32,
            node_physical_length_f32,
            site_rgba_f32,
            config_i32,
            config_f32,
            track_count=track_count,
            node_count=node_count,
        )
        assert tuple(node_chart_f32.shape) == (track_count, node_count, 4)
        assert tuple(grad_node_chart_f32.shape) == (track_count, node_count, 4)
        assert tuple(grad_site_rgba_f32.shape) == tuple(site_rgba_f32.shape)
        self.vjp_calls.append(
            {
                "track_count": track_count,
                "node_count": node_count,
                "grad_node_chart": grad_node_chart_f32.clone(),
            }
        )
        with torch.enable_grad():
            rgba = site_rgba_f32.detach().clone().requires_grad_(True)
            lengths = node_physical_length_f32.detach().clone().requires_grad_(True)
            charts = self._node_charts(word_owner_i32, lengths, rgba)
            objective = torch.sum(charts * grad_node_chart_f32.detach())
            grad_rgba, grad_lengths = torch.autograd.grad(objective, (rgba, lengths))
        grad_site_rgba_f32.add_(grad_rgba)
        return grad_site_rgba_f32, grad_lengths.contiguous()

    @staticmethod
    def _validate_launch(
        word_offsets_i32: torch.Tensor,
        word_owner_i32: torch.Tensor,
        node_physical_length_f32: torch.Tensor,
        site_rgba_f32: torch.Tensor,
        config_i32: torch.Tensor,
        config_f32: torch.Tensor,
        *,
        track_count: int,
        node_count: int,
    ) -> None:
        assert track_count == 1
        assert tuple(word_offsets_i32.tolist()) == (0, word_owner_i32.numel())
        expected_config = (
            track_count,
            node_count,
            site_rgba_f32.shape[0],
            word_owner_i32.numel(),
        )
        assert tuple(config_i32.tolist()) == expected_config
        assert config_f32.numel() == 1
        assert tuple(node_physical_length_f32.shape) == (
            node_count,
            word_owner_i32.numel(),
        )
        assert all(
            tensor.device.type == "cpu" and tensor.is_contiguous()
            for tensor in (
                word_offsets_i32,
                word_owner_i32,
                node_physical_length_f32,
                site_rgba_f32,
                config_i32,
                config_f32,
            )
        )

    @staticmethod
    def _node_charts(
        word_owner_i32: torch.Tensor,
        node_physical_length_f32: torch.Tensor,
        site_rgba_f32: torch.Tensor,
    ) -> torch.Tensor:
        owners = word_owner_i32.to(dtype=torch.long)
        rows = []
        for lengths in node_physical_length_f32:
            total_kappa = torch.zeros((), dtype=torch.float32)
            total_beta = torch.ones((), dtype=torch.float32)
            total_moment = torch.zeros(3, dtype=torch.float32)
            for run_index, owner in enumerate(owners):
                rgba = site_rgba_f32[owner]
                optical_depth = rgba[3] * lengths[run_index]
                beta = torch.exp(-optical_depth)
                alpha = -torch.expm1(-optical_depth)
                total_kappa = total_kappa + optical_depth
                total_moment = total_moment + total_beta * alpha * rgba[:3]
                total_beta = total_beta * beta
            small = total_kappa.abs() < 1.0e-4
            kappa2 = total_kappa.square()
            series = 1.0 + 0.5 * total_kappa + kappa2 / 12.0 - kappa2.square() / 720.0 + kappa2.pow(3) / 30240.0
            denominator = -torch.expm1(-total_kappa)
            inverse_phi = torch.where(
                small,
                series,
                total_kappa / torch.where(small, torch.ones_like(denominator), denominator),
            )
            rows.append(torch.cat((total_kappa.reshape(1), inverse_phi * total_moment)))
        return torch.stack(rows).unsqueeze(0).contiguous()


class _WrongShapeNativeOps(_FakeNativeOps):
    def kinetic_precompiled_length_p0_lie_node_forward_launch_only(self, *args, **kwargs):
        valid = super().kinetic_precompiled_length_p0_lie_node_forward_launch_only(*args, **kwargs)
        return valid.squeeze(0)


def _fixture(node_count: int = 5):
    # Source order is deliberately (3,1,2); the compact rows are (1,2,3).
    # Global site zero is present but unused, exercising the scatter boundary.
    sites = AffineKineticPowerSites(
        positions0=torch.tensor(
            [
                [100.0, 100.0, 100.0],
                [1.5, 0.0, 0.0],
                [2.5, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ],
            dtype=DTYPE,
        ),
        velocities=torch.zeros((4, 3), dtype=DTYPE),
        weight_coefficients=torch.zeros((4, 1), dtype=DTYPE),
    )
    ray = torch.tensor(
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        dtype=DTYPE,
    )
    owner_program = compile_exact_kinetic_owner_charts(
        sites,
        ray,
        t_min=-1,
        t_max=1,
        near=0,
        far=3,
    )
    program = compile_kinetic_multichart_p0_program(
        owner_program,
        sites,
        ray,
        node_count=node_count,
    )
    assert program.chart_count == 1
    assert program.charts[0].owner_word == (3, 1, 2)
    density = torch.tensor([0.91, 0.37, 0.68, 0.24], dtype=DTYPE)
    color = torch.tensor(
        [
            [0.02, 0.03, 0.04],
            [0.82, 0.16, 0.11],
            [0.12, 0.71, 0.91],
            [0.46, 0.31, 0.77],
        ],
        dtype=DTYPE,
    )
    transfer = refresh_kinetic_multichart_p0_transfer(program, density, color)
    lowering = lower_kinetic_multichart_to_native_topology(program)
    payload = materialize_kinetic_native_topology_chart(lowering, program, 0)
    return transfer, payload, density, color


def test_source_only_adapter_prepares_frame_free_config_and_matches_lie_oracle() -> None:
    transfer, payload, density, color = _fixture()
    fake = _FakeNativeOps()
    topology = prepare_kinetic_native_precompiled_length_topology_token(
        payload,
        device="cpu",
        native_ops=fake,
        physical_length_epsilon=1.0e-8,
    )
    world = refresh_kinetic_native_precompiled_length_world_token(
        topology,
        density,
        color,
    )
    oracle = refresh_kinetic_native_precompiled_length_world(payload, density, color)

    assert tuple(payload.topology.source_site_ids.tolist()) == (1, 2, 3)
    assert tuple(topology.config_i32.tolist()) == (
        1,
        payload.spec.node_count,
        payload.topology.site_count,
        payload.spec.run_count,
    )
    assert tuple(topology.config_f32.tolist()) == pytest.approx((1.0e-8,))
    assert all(tensor.device.type == "cpu" and tensor.is_contiguous() for tensor in topology._persistent_tensors())
    assert topology.source_site_ids_i64.dtype == torch.int64
    assert topology.word_offsets_i32.dtype == torch.int32
    assert topology.word_owner_i32.dtype == torch.int32
    assert topology.node_physical_length_f32.dtype == torch.float32
    assert topology.config_i32.dtype == torch.int32
    assert topology.config_f32.dtype == torch.float32
    assert len(fake.forward_calls) == 1
    torch.testing.assert_close(
        world.node_chart_f32[0].to(dtype=DTYPE),
        oracle.node_charts,
        rtol=2.0e-6,
        atol=2.0e-7,
    )
    assert not torch.allclose(
        world.node_chart_f32[0],
        transfer.chart_node_transfers[0].to(dtype=torch.float32),
    )
    assert topology.runtime_status == world.runtime_status == RUNTIME_STATUS
    assert topology.source_only and world.source_only
    assert not topology.runtime_verified and not world.runtime_verified
    assert not topology.native_execution_ready and not world.native_execution_ready
    assert topology.persistent_sample_time_tensor_bytes == 0
    assert topology.persistent_frame_or_sample_tensor_bytes == 0
    for function in (
        prepare_kinetic_native_precompiled_length_topology_token,
        refresh_kinetic_native_precompiled_length_world_token,
        execute_kinetic_native_precompiled_length_node_vjp,
    ):
        parameters = inspect.signature(function).parameters
        assert "times" not in parameters
        assert "targets" not in parameters
        assert "frame_count" not in parameters
        assert "sample_count" not in parameters


def test_native_lie_vjp_matches_oracle_and_accumulates_compact_bars_globally() -> None:
    _, payload, density, color = _fixture(node_count=4)
    fake = _FakeNativeOps()
    topology = prepare_kinetic_native_precompiled_length_topology_token(
        payload,
        device=torch.device("cpu"),
        native_ops=fake,
    )
    world = refresh_kinetic_native_precompiled_length_world_token(topology, density, color)
    oracle_world = refresh_kinetic_native_precompiled_length_world(payload, density, color)
    grad_node_chart = torch.linspace(
        -0.41,
        0.73,
        world.node_count * 4,
        dtype=DTYPE,
    ).reshape(world.node_count, 4)
    oracle_vjp = kinetic_native_precompiled_length_node_vjp(
        oracle_world,
        grad_node_chart,
    )
    global_accumulator = torch.full(
        (density.numel(), 4),
        0.125,
        dtype=torch.float32,
    )
    initial_global = global_accumulator.clone()
    result = execute_kinetic_native_precompiled_length_node_vjp(
        world,
        grad_node_chart,
        global_grad_site_rgba_f32=global_accumulator,
    )

    assert len(fake.vjp_calls) == 1
    torch.testing.assert_close(
        fake.vjp_calls[0]["grad_node_chart"][0].to(dtype=DTYPE),
        grad_node_chart,
    )
    torch.testing.assert_close(
        result.grad_compact_site_rgba_f32.to(dtype=DTYPE),
        oracle_vjp.grad_compact_site_rgba,
        rtol=4.0e-6,
        atol=4.0e-7,
    )
    torch.testing.assert_close(
        result.grad_node_physical_length_f32.to(dtype=DTYPE),
        oracle_vjp.grad_node_physical_lengths,
        rtol=5.0e-6,
        atol=5.0e-7,
    )
    expected_global = initial_global.clone()
    expected_global.index_add_(
        0,
        payload.topology.source_site_ids,
        oracle_vjp.grad_compact_site_rgba.to(dtype=torch.float32),
    )
    assert result.grad_global_site_rgba_f32 is global_accumulator
    torch.testing.assert_close(result.grad_global_site_rgba_f32, expected_global)
    torch.testing.assert_close(result.grad_global_site_rgba_f32[0], initial_global[0])
    assert result.accounting["requested_frame_count_used"] == 0
    assert result.accounting["persistent_sample_time_tensor_bytes"] == 0
    assert result.accounting["persistent_frame_or_sample_tensor_bytes"] == 0
    assert not result.accounting["frame_by_run_reverse_state_allocated"]
    assert result.accounting["reverse_scaling"] == "O(J * R)"
    assert result.accounting["source_only"]
    assert not result.accounting["runtime_verified"]
    assert not result.accounting["native_execution_ready"]
    assert result.source_only and not result.runtime_verified
    assert not result.native_execution_ready
    assert not result.geometry_vjp_implemented


@pytest.mark.parametrize("referenced_density", (1.0e-18, 1.0e4))
def test_adapter_matches_direct_lie_oracle_at_tiny_and_underflowing_optical_depth(
    referenced_density: float,
) -> None:
    _, payload, density, color = _fixture(node_count=4)
    density = density.clone()
    density[payload.topology.source_site_ids] = referenced_density
    fake = _FakeNativeOps()
    topology = prepare_kinetic_native_precompiled_length_topology_token(
        payload,
        device="cpu",
        native_ops=fake,
    )
    world = refresh_kinetic_native_precompiled_length_world_token(topology, density, color)
    oracle = refresh_kinetic_native_precompiled_length_world(payload, density, color)
    torch.testing.assert_close(
        world.node_chart_f32[0].to(dtype=DTYPE),
        oracle.node_charts,
        rtol=3.0e-6,
        atol=1.0e-23,
    )
    assert bool(torch.all(world.node_chart_f32[0, :, 0] > 0.0).item())
    assert bool(torch.isfinite(world.node_chart_f32).all().item())

    grad_node_chart = torch.linspace(-0.29, 0.53, world.node_count * 4, dtype=DTYPE).reshape(
        world.node_count,
        4,
    )
    result = execute_kinetic_native_precompiled_length_node_vjp(world, grad_node_chart)
    oracle_vjp = kinetic_native_precompiled_length_node_vjp(oracle, grad_node_chart)
    torch.testing.assert_close(
        result.grad_compact_site_rgba_f32.to(dtype=DTYPE),
        oracle_vjp.grad_compact_site_rgba,
        rtol=8.0e-6,
        atol=1.0e-21,
    )
    torch.testing.assert_close(
        result.grad_node_physical_length_f32.to(dtype=DTYPE),
        oracle_vjp.grad_node_physical_lengths,
        rtol=8.0e-6,
        atol=1.0e-21,
    )


def test_adapter_fails_closed_on_stale_payload_config_native_identity_and_bad_output() -> None:
    _, payload, density, color = _fixture()
    fake = _FakeNativeOps()
    topology = prepare_kinetic_native_precompiled_length_topology_token(
        payload,
        device="cpu",
        native_ops=fake,
    )

    wrong_native = replace(topology, native_ops=_FakeNativeOps())
    with pytest.raises(ValueError, match="different native ops"):
        wrong_native.assert_current()

    topology.config_i32[0].add_(1)
    with pytest.raises(ValueError, match="topology/config tensors changed"):
        topology.assert_current()

    _, second_payload, second_density, second_color = _fixture()
    second_topology = prepare_kinetic_native_precompiled_length_topology_token(
        second_payload,
        device="cpu",
        native_ops=_FakeNativeOps(),
    )
    second_payload.node_physical_lengths[0, 0].add_(0.125)
    with pytest.raises(ValueError, match="payload tensors changed"):
        second_topology.assert_current()

    _, third_payload, third_density, third_color = _fixture()
    bad_shape_topology = prepare_kinetic_native_precompiled_length_topology_token(
        third_payload,
        device="cpu",
        native_ops=_WrongShapeNativeOps(),
    )
    with pytest.raises(ValueError, match="native node_chart_f32"):
        refresh_kinetic_native_precompiled_length_world_token(
            bad_shape_topology,
            third_density,
            third_color,
        )

    _, fourth_payload, _, _ = _fixture()
    minimum_f32_length = float(fourth_payload.node_physical_lengths.float().min().item())
    with pytest.raises(ValueError, match="strictly above epsilon"):
        prepare_kinetic_native_precompiled_length_topology_token(
            fourth_payload,
            device="cpu",
            native_ops=_FakeNativeOps(),
            physical_length_epsilon=minimum_f32_length,
        )

    _, fifth_payload, fifth_density, fifth_color = _fixture()
    fifth_topology = prepare_kinetic_native_precompiled_length_topology_token(
        fifth_payload,
        device="cpu",
        native_ops=_FakeNativeOps(),
    )
    world = refresh_kinetic_native_precompiled_length_world_token(
        fifth_topology,
        fifth_density,
        fifth_color,
    )
    world.site_rgba_f32[0, 0].add_(0.125)
    with pytest.raises(ValueError, match="world tensors changed"):
        world.assert_current()


def test_adapter_requires_both_exact_low_level_native_op_names() -> None:
    _, payload, _, _ = _fixture()

    class _IncompleteNative:
        pass

    with pytest.raises(TypeError, match=FORWARD_OP_NAME):
        prepare_kinetic_native_precompiled_length_topology_token(
            payload,
            device="cpu",
            native_ops=_IncompleteNative(),
        )

    incomplete = _IncompleteNative()
    setattr(incomplete, FORWARD_OP_NAME, lambda *args, **kwargs: None)
    with pytest.raises(TypeError, match=VJP_OP_NAME):
        prepare_kinetic_native_precompiled_length_topology_token(
            payload,
            device="cpu",
            native_ops=incomplete,
        )
    with pytest.raises(TypeError, match="compiled kinetic ABI attestation"):
        prepare_kinetic_native_precompiled_length_topology_token(
            payload,
            device="mps",
            native_ops=_FakeNativeOps(),
        )
