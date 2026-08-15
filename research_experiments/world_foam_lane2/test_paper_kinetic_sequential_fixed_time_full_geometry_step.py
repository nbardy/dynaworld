from __future__ import annotations

import hashlib

import torch

from kinetic_power_word_compiler import (
    AffineKineticPowerSites,
    discover_kinetic_power_word_at_time,
)
from paper_kinetic_sequential_fixed_time_full_geometry_step import (
    PaperKineticSequentialFixedTimeMemoryPolicy,
    paper_kinetic_sequential_fixed_time_geometry_generation_id,
    run_paper_kinetic_sequential_fixed_time_full_geometry_step,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _inverse_phi(kappa: torch.Tensor) -> torch.Tensor:
    small = kappa.abs() < 1.0e-4
    k2 = kappa * kappa
    k4 = k2 * k2
    k6 = k4 * k2
    series = 1.0 + 0.5 * kappa + k2 / 12.0 - k4 / 720.0 + k6 / 30240.0
    denominator = -torch.expm1(-kappa)
    direct = kappa / torch.where(small, torch.ones_like(denominator), denominator)
    return torch.where(small, series, direct)


def _node_charts(
    offsets: torch.Tensor,
    owners: torch.Tensor,
    lengths: torch.Tensor,
    rgba: torch.Tensor,
) -> torch.Tensor:
    rows = []
    for row in range(int(offsets.numel()) - 1):
        start = int(offsets[row].item())
        end = int(offsets[row + 1].item())
        node_rows = []
        for node in range(int(lengths.shape[0])):
            beta_prefix = torch.ones((), dtype=rgba.dtype)
            moment = torch.zeros(3, dtype=rgba.dtype)
            kappa = torch.zeros((), dtype=rgba.dtype)
            for word_index in range(start, end):
                owner = int(owners[word_index].item())
                optical_depth = rgba[owner, 3] * lengths[node, word_index]
                beta = torch.exp(-optical_depth)
                alpha = -torch.expm1(-optical_depth)
                moment = moment + beta_prefix * alpha * rgba[owner, :3]
                beta_prefix = beta_prefix * beta
                kappa = kappa + optical_depth
            node_rows.append(
                torch.cat((kappa.reshape(1), _inverse_phi(kappa) * moment))
            )
        rows.append(torch.stack(node_rows))
    return torch.stack(rows).contiguous()


class _FakeNativeP0:
    def __init__(self) -> None:
        self.forward_calls = 0
        self.vjp_calls = 0

    def kinetic_precompiled_length_p0_lie_node_forward_into_launch_only_v1(
        self,
        word_offsets_i32,
        word_owner_i32,
        node_physical_length_f32,
        site_rgba_f32,
        config_i32,
        _config_f32,
        node_chart_out_f32,
        *,
        track_count,
        node_count,
    ):
        assert tuple(config_i32.tolist()) == (
            track_count,
            node_count,
            int(site_rgba_f32.shape[0]),
            int(word_owner_i32.numel()),
        )
        node_chart_out_f32.copy_(
            _node_charts(
                word_offsets_i32,
                word_owner_i32,
                node_physical_length_f32,
                site_rgba_f32,
            )
        )
        self.forward_calls += 1
        return None

    def kinetic_precompiled_length_p0_lie_node_vjp_accumulate_launch_only(
        self,
        word_offsets_i32,
        word_owner_i32,
        node_physical_length_f32,
        site_rgba_f32,
        _node_chart_f32,
        grad_node_chart_f32,
        grad_site_rgba_f32,
        _config_i32,
        _config_f32,
        *,
        track_count,
        node_count,
    ):
        assert tuple(grad_node_chart_f32.shape) == (track_count, node_count, 4)
        with torch.enable_grad():
            rgba = site_rgba_f32.detach().clone().requires_grad_(True)
            lengths = (
                node_physical_length_f32.detach().clone().requires_grad_(True)
            )
            charts = _node_charts(
                word_offsets_i32,
                word_owner_i32,
                lengths,
                rgba,
            )
            grad_rgba, grad_lengths = torch.autograd.grad(
                torch.sum(charts * grad_node_chart_f32.detach()),
                (rgba, lengths),
            )
        grad_site_rgba_f32.add_(grad_rgba)
        self.vjp_calls += 1
        return grad_site_rgba_f32, grad_lengths.contiguous()


def _case():
    sites = AffineKineticPowerSites(
        positions0=torch.tensor(
            [
                [-0.20, 0.00, 0.60],
                [0.02, 0.03, 1.80],
                [0.24, -0.02, 3.00],
            ],
            dtype=torch.float64,
        ),
        velocities=torch.tensor(
            [
                [0.03, 0.00, 0.04],
                [-0.02, 0.01, -0.03],
                [0.01, -0.01, 0.02],
            ],
            dtype=torch.float64,
        ),
        weight_coefficients=torch.tensor(
            [[0.02, 0.03], [-0.01, -0.02], [0.015, 0.01]],
            dtype=torch.float64,
        ),
    )
    rays = torch.zeros((2, 12), dtype=torch.float64)
    rays[:, 0] = torch.tensor([-0.12, 0.14], dtype=torch.float64)
    rays[:, 8] = 1.0
    times = torch.tensor([-0.20, 0.35], dtype=torch.float64)
    rgba = torch.tensor(
        [
            [0.75, 0.20, 0.10, 0.42],
            [0.15, 0.70, 0.25, 0.55],
            [0.20, 0.25, 0.80, 0.48],
        ],
        dtype=torch.float32,
    )
    background = torch.tensor([0.04, 0.05, 0.06], dtype=torch.float32)
    targets = (
        torch.tensor([[0.25, 0.35, 0.45], [0.55, 0.30, 0.20]], dtype=torch.float32),
        torch.tensor([[0.40, 0.25, 0.35], [0.20, 0.55, 0.30]], dtype=torch.float32),
    )
    return sites, rays, times, rgba, background, targets


def _fixed_word_reference(sites, rays, times, rgba, background, targets):
    p0 = sites.positions0.detach().clone().requires_grad_(True)
    velocities = sites.velocities.detach().clone().requires_grad_(True)
    weights = sites.weight_coefficients.detach().clone().requires_grad_(True)
    material = rgba.detach().to(dtype=torch.float64).clone().requires_grad_(True)
    background_f64 = background.to(dtype=torch.float64)
    loss = torch.zeros((), dtype=torch.float64)
    denominator = int(times.numel() * rays.shape[0] * 3)
    for frame_index, time_tensor in enumerate(times):
        time = float(time_tensor.item())
        for track_id, ray in enumerate(rays):
            discovered = discover_kinetic_power_word_at_time(
                sites,
                ray,
                time=time,
                near=0.10,
                far=3.60,
            )
            owners = tuple(int(value) for value in discovered.word.owners.tolist())
            position = p0 + time_tensor * velocities
            time_powers = torch.stack(
                (torch.ones_like(time_tensor), time_tensor, time_tensor.square())
            )[: weights.shape[1]]
            weight = weights @ time_powers
            origin = ray[:3] + time_tensor * ray[3:6]
            direction = ray[6:9] + time_tensor * ray[9:12]
            cuts = [torch.tensor(0.10, dtype=torch.float64)]
            for left, right in zip(owners[:-1], owners[1:], strict=True):
                normal = 2.0 * (position[right] - position[left])
                intercept = (
                    torch.dot(normal, origin)
                    + torch.dot(position[left], position[left])
                    - torch.dot(position[right], position[right])
                    - weight[left]
                    + weight[right]
                )
                cuts.append(-intercept / torch.dot(normal, direction))
            cuts.append(torch.tensor(3.60, dtype=torch.float64))
            lengths = torch.linalg.vector_norm(direction) * (
                torch.stack(cuts[1:]) - torch.stack(cuts[:-1])
            )
            beta_prefix = torch.ones((), dtype=torch.float64)
            moment = torch.zeros(3, dtype=torch.float64)
            for run_index, owner in enumerate(owners):
                optical_depth = material[owner, 3] * lengths[run_index]
                beta = torch.exp(-optical_depth)
                alpha = -torch.expm1(-optical_depth)
                moment = moment + beta_prefix * alpha * material[owner, :3]
                beta_prefix = beta_prefix * beta
            prediction = moment + beta_prefix * background_f64
            loss = loss + (
                prediction - targets[frame_index][track_id].to(dtype=torch.float64)
            ).square().sum() / float(denominator)
    gradients = torch.autograd.grad(loss, (material, p0, velocities, weights))
    return loss.detach(), tuple(gradient.detach() for gradient in gradients)


def test_sequential_fixed_time_native_full_geometry_matches_independent_reference_and_updates_once():
    sites, rays, times, rgba, background, targets = _case()
    reference_loss, reference_gradients = _fixed_word_reference(
        sites,
        rays,
        times,
        rgba,
        background,
        targets,
    )
    native = _FakeNativeP0()
    fences = []
    target_loads = []
    updates = []
    grad_rgba = torch.empty_like(rgba)
    grad_p0 = torch.empty_like(sites.positions0)
    grad_velocity = torch.empty_like(sites.velocities)
    grad_weights = torch.empty_like(sites.weight_coefficients)
    original = tuple(
        tensor.clone()
        for tensor in (
            rgba,
            sites.positions0,
            sites.velocities,
            sites.weight_coefficients,
        )
    )
    learning_rate = 0.025

    def load_target(frame_index: int, physical_time: float) -> torch.Tensor:
        target_loads.append((frame_index, physical_time))
        return targets[frame_index]

    def fence() -> None:
        fences.append("fenced")

    def update(result) -> None:
        result.assert_current()
        updates.append(result.generation_digest)
        rgba.add_(result.grad_global_site_rgba_f32, alpha=-learning_rate)
        sites.positions0.add_(result.grad_positions0_f64_cpu, alpha=-learning_rate)
        sites.velocities.add_(result.grad_velocities_f64_cpu, alpha=-learning_rate)
        sites.weight_coefficients.add_(
            result.grad_weight_coefficients_f64_cpu,
            alpha=-learning_rate,
        )

    result = run_paper_kinetic_sequential_fixed_time_full_geometry_step(
        sites,
        rays,
        times,
        step_index=0,
        geometry_generation_id=(
            paper_kinetic_sequential_fixed_time_geometry_generation_id(
                sites,
                rays,
                near=0.10,
                far=3.60,
            )
        ),
        material_generation_id=_sha("material"),
        background_generation_id=_sha("background"),
        target_generation_id=_sha("target"),
        near=0.10,
        far=3.60,
        global_site_rgba_f32=rgba,
        global_grad_site_rgba_f32=grad_rgba,
        grad_positions0_f64_cpu=grad_p0,
        grad_velocities_f64_cpu=grad_velocity,
        grad_weight_coefficients_f64_cpu=grad_weights,
        background_rgb_f32=background,
        target_frame_loader=load_target,
        native_ops=native,
        device_completion_fence=fence,
        device_completion_fence_provenance="cpu-fake-native-synchronous",
        memory_policy=PaperKineticSequentialFixedTimeMemoryPolicy(
            maximum_target_frame_logical_tensor_bytes=4096,
            maximum_frame_cpu_topology_logical_tensor_bytes=65536,
            maximum_active_device_scratch_logical_tensor_bytes=65536,
            maximum_geometry_d2h_logical_tensor_bytes=4096,
            maximum_tracks_per_native_block=1,
        ),
        optimizer_update=update,
    )
    result.assert_current()
    torch.testing.assert_close(
        result.loss_f32.double(),
        reference_loss.reshape(1),
        rtol=3.0e-5,
        atol=3.0e-6,
    )
    torch.testing.assert_close(
        result.grad_global_site_rgba_f32.double(),
        reference_gradients[0],
        rtol=8.0e-5,
        atol=8.0e-6,
    )
    for actual, expected in zip(
        (
            result.grad_positions0_f64_cpu,
            result.grad_velocities_f64_cpu,
            result.grad_weight_coefficients_f64_cpu,
        ),
        reference_gradients[1:],
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=2.0e-4, atol=2.0e-5)
    assert len(target_loads) == 2
    assert native.forward_calls == native.vjp_calls == 4
    assert len(fences) == 4
    assert len(updates) == 1
    assert result.receipt.streamed_sample_count == 4
    assert result.receipt.selected_time_grid_tensor_bytes == 16
    assert len(result.receipt.native_callable_identity_digest) == 64
    assert result.receipt.fixed_time_lower_envelope_discovery_call_count == 4
    assert result.receipt.candidate_line_evaluation_count == 12
    assert result.receipt.continuous_compiler_invocation_count == 0
    assert result.receipt.target_frame_release_count == 2
    assert result.receipt.frame_scratch_release_count == 2
    assert result.receipt.retained_frame_receipt_count == 0
    assert result.accounting["material_grad_nonzero"] is True
    assert result.accounting["position_grad_nonzero"] is True
    assert result.accounting["velocity_grad_nonzero"] is True
    assert result.accounting["weight_grad_nonzero"] is True
    for updated, before, gradient in zip(
        (rgba, sites.positions0, sites.velocities, sites.weight_coefficients),
        original,
        (
            result.grad_global_site_rgba_f32,
            result.grad_positions0_f64_cpu,
            result.grad_velocities_f64_cpu,
            result.grad_weight_coefficients_f64_cpu,
        ),
        strict=True,
    ):
        torch.testing.assert_close(updated, before - learning_rate * gradient)
