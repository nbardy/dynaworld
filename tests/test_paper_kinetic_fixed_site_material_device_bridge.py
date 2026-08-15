from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
import paper_kinetic_fixed_site_material_device_bridge as bridge_module
from paper_kinetic_fixed_site_material_device_bridge import (
    _seal_paper_kinetic_fixed_site_material_device_gradient_receipt_raw_for_test,
    apply_paper_kinetic_fixed_site_material_device_gradient_receipt,
    snapshot_paper_kinetic_fixed_site_material_to_device,
)
from paper_kinetic_fixed_site_material_state import (
    checkpoint_paper_kinetic_fixed_site_material_state,
    restore_paper_kinetic_fixed_site_material_state,
)
from test_paper_kinetic_fixed_site_material_state import _state


class _Fence:
    def __init__(self, *, failure: BaseException | None = None) -> None:
        self.calls = 0
        self.failure = failure

    def __call__(self) -> None:
        self.calls += 1
        if self.failure is not None:
            raise self.failure


def _snapshot(
    state,
    background: torch.Tensor,
    *,
    background_generation_id: str = "fixed-background-v1",
):
    fence = _Fence()
    snapshot = snapshot_paper_kinetic_fixed_site_material_to_device(
        state,
        background_rgb_f32_cpu=background,
        background_generation_id=background_generation_id,
        device="cpu",
        device_completion_fence=fence,
        device_completion_fence_provenance="cpu-synchronous-test-fence-v1",
    )
    assert fence.calls == 1
    return snapshot


def _monotone_step(state, background: torch.Tensor, *, step_index: int):
    snapshot = _snapshot(state, background)
    target = torch.tensor(
        ((0.35, 0.45, 0.55, 63.5), (0.65, 0.55, 0.25, 64.5)),
        dtype=torch.float32,
    )
    material = snapshot.site_rgba_f32_device
    assert material is not None
    residual = material - target
    loss = residual.square().mean().reshape(1).contiguous()
    gradient = (2.0 * residual / float(residual.numel())).contiguous()
    fence = _Fence()
    receipt = _seal_paper_kinetic_fixed_site_material_device_gradient_receipt_raw_for_test(
        state,
        snapshot,
        grad_site_rgba_f32_device=gradient,
        loss_f32_device=loss,
        step_generation_id=f"device-bridge-step-{step_index}",
        device_completion_fence=fence,
        device_completion_fence_provenance="cpu-synchronous-test-fence-v1",
    )
    assert fence.calls == 1
    state_identity = id(state)
    step_receipt = apply_paper_kinetic_fixed_site_material_device_gradient_receipt(
        state,
        receipt,
    )
    assert id(state) == state_identity
    assert receipt.consumed and receipt.released_after_consumption
    assert receipt.grad_site_rgba_f32_cpu is None
    assert receipt.loss_f32_cpu is None
    assert snapshot.released_after_consumption
    assert snapshot.site_rgba_f32_device is None
    assert snapshot.background_rgb_f32_device is None
    receipt.assert_current(state, require_unconsumed=False)
    step_receipt.assert_current(state)
    return float(loss.item()), receipt, step_receipt


def test_two_device_bridge_steps_decrease_loss_and_restart_exactly(
    tmp_path: Path,
) -> None:
    state, _material, world = _state(tmp_path)
    background = torch.tensor((0.02, 0.03, 0.04), dtype=torch.float32)

    first_loss, first_receipt, first_step = _monotone_step(
        state,
        background,
        step_index=0,
    )
    checkpoint = checkpoint_paper_kinetic_fixed_site_material_state(state)
    restarted = restore_paper_kinetic_fixed_site_material_state(
        checkpoint,
        world=world,
        device="cpu",
        maximum_material_state_logical_tensor_bytes=1_000_000,
    )

    second_loss, second_receipt, second_step = _monotone_step(
        state,
        background,
        step_index=1,
    )
    restarted_loss, restarted_receipt, restarted_step = _monotone_step(
        restarted,
        background,
        step_index=1,
    )

    assert second_loss < first_loss
    assert restarted_loss == second_loss
    assert first_step.loss == first_loss
    assert second_step.loss == restarted_step.loss == second_loss
    assert first_receipt.material_generation_id_after == checkpoint.material_generation_id
    assert second_receipt.generation_digest == restarted_receipt.generation_digest
    assert second_step.generation_digest == restarted_step.generation_digest
    assert state.step_index == restarted.step_index == 2
    assert state.material_generation_id == restarted.material_generation_id
    torch.testing.assert_close(state.raw_color_f32, restarted.raw_color_f32)
    torch.testing.assert_close(state.raw_density_f32, restarted.raw_density_f32)
    torch.testing.assert_close(state.site_rgba_f32, restarted.site_rgba_f32)


def test_bridge_exposes_constant_time_live_byte_accounting(tmp_path: Path) -> None:
    state, _material, _world = _state(tmp_path)
    background = torch.tensor((0.02, 0.03, 0.04), dtype=torch.float32)
    snapshot = _snapshot(state, background)
    snapshot_accounting = snapshot.accounting()
    expected_material_bytes = state.site_count * 4 * 4

    assert snapshot_accounting["live_device_material_tensor_bytes"] == (
        expected_material_bytes
    )
    assert snapshot_accounting["live_device_background_tensor_bytes"] == 3 * 4
    assert snapshot_accounting["live_snapshot_tensor_bytes"] == (
        expected_material_bytes + 3 * 4
    )
    assert snapshot_accounting["live_cpu_receipt_tensor_bytes"] == 0
    assert snapshot_accounting["device_material_copy_count"] == 1
    assert snapshot_accounting["device_material_content_digest_count"] == 0
    assert snapshot_accounting["device_material_readback_count"] == 0
    for key in (
        "persistent_frame_tensor_bytes",
        "persistent_sample_tensor_bytes",
        "persistent_target_tensor_bytes",
        "persistent_prediction_tensor_bytes",
    ):
        assert snapshot_accounting[key] == 0

    receipt = _seal_paper_kinetic_fixed_site_material_device_gradient_receipt_raw_for_test(
        state,
        snapshot,
        grad_site_rgba_f32_device=torch.zeros(
            (state.site_count, 4),
            dtype=torch.float32,
        ),
        loss_f32_device=torch.tensor((0.5,), dtype=torch.float32),
        step_generation_id="accounted-step",
        device_completion_fence=_Fence(),
        device_completion_fence_provenance="cpu-synchronous-test-fence-v1",
    )
    receipt_accounting = receipt.accounting()
    expected_receipt_bytes = expected_material_bytes + 4
    assert receipt_accounting["live_device_snapshot_tensor_bytes"] == (
        expected_material_bytes + 3 * 4
    )
    assert receipt_accounting["live_cpu_gradient_tensor_bytes"] == (
        expected_material_bytes
    )
    assert receipt_accounting["live_cpu_loss_tensor_bytes"] == 4
    assert receipt_accounting["live_cpu_receipt_tensor_bytes"] == (
        expected_receipt_bytes
    )
    assert receipt_accounting["live_bridge_tensor_bytes"] == (
        expected_material_bytes + 3 * 4 + expected_receipt_bytes
    )
    assert receipt_accounting["device_to_host_copy_phase_count"] == 1
    assert receipt_accounting["device_material_content_digest_count"] == 0
    for key in (
        "persistent_frame_tensor_bytes",
        "persistent_sample_tensor_bytes",
        "persistent_target_tensor_bytes",
        "persistent_prediction_tensor_bytes",
    ):
        assert receipt_accounting[key] == 0

    apply_paper_kinetic_fixed_site_material_device_gradient_receipt(state, receipt)
    assert snapshot.accounting()["live_snapshot_tensor_bytes"] == 0
    released_accounting = receipt.accounting()
    assert released_accounting["live_device_snapshot_tensor_bytes"] == 0
    assert released_accounting["live_cpu_receipt_tensor_bytes"] == 0
    assert released_accounting["live_bridge_tensor_bytes"] == 0


def test_gradient_receipt_requires_shape_finiteness_and_successful_fence(
    tmp_path: Path,
) -> None:
    state, _material, _world = _state(tmp_path)
    background = torch.tensor((0.02, 0.03, 0.04), dtype=torch.float32)
    snapshot = _snapshot(state, background)
    loss = torch.tensor((1.0,), dtype=torch.float32)

    wrong_shape_fence = _Fence()
    with pytest.raises(ValueError, match="shape|contiguous float32"):
        _seal_paper_kinetic_fixed_site_material_device_gradient_receipt_raw_for_test(
            state,
            snapshot,
            grad_site_rgba_f32_device=torch.zeros((state.site_count, 3)),
            loss_f32_device=loss,
            step_generation_id="wrong-shape",
            device_completion_fence=wrong_shape_fence,
            device_completion_fence_provenance="cpu-synchronous-test-fence-v1",
        )
    assert wrong_shape_fence.calls == 0
    assert not snapshot.gradient_receipt_issued

    failing_fence = _Fence(failure=RuntimeError("injected fence failure"))
    with pytest.raises(RuntimeError, match="injected fence failure"):
        _seal_paper_kinetic_fixed_site_material_device_gradient_receipt_raw_for_test(
            state,
            snapshot,
            grad_site_rgba_f32_device=torch.zeros((state.site_count, 4)),
            loss_f32_device=loss,
            step_generation_id="failed-fence",
            device_completion_fence=failing_fence,
            device_completion_fence_provenance="cpu-synchronous-test-fence-v1",
        )
    assert failing_fence.calls == 1
    assert not snapshot.gradient_receipt_issued

    nonfinite_fence = _Fence()
    nonfinite = torch.zeros((state.site_count, 4), dtype=torch.float32)
    nonfinite[0, 0] = float("inf")
    with pytest.raises(ValueError, match="nonfinite"):
        _seal_paper_kinetic_fixed_site_material_device_gradient_receipt_raw_for_test(
            state,
            snapshot,
            grad_site_rgba_f32_device=nonfinite,
            loss_f32_device=loss,
            step_generation_id="nonfinite",
            device_completion_fence=nonfinite_fence,
            device_completion_fence_provenance="cpu-synchronous-test-fence-v1",
        )
    assert nonfinite_fence.calls == 1
    assert not snapshot.gradient_receipt_issued

    with pytest.raises(TypeError, match="must be callable"):
        _seal_paper_kinetic_fixed_site_material_device_gradient_receipt_raw_for_test(
            state,
            snapshot,
            grad_site_rgba_f32_device=torch.zeros((state.site_count, 4)),
            loss_f32_device=loss,
            step_generation_id="unfenced",
            device_completion_fence=None,
            device_completion_fence_provenance="cpu-synchronous-test-fence-v1",
        )
    assert not snapshot.gradient_receipt_issued


def test_device_gradient_receipt_rejects_foreign_stale_cloned_and_replayed(
    tmp_path: Path,
) -> None:
    state, _material, world = _state(tmp_path)
    background = torch.tensor((0.02, 0.03, 0.04), dtype=torch.float32)
    initial_checkpoint = checkpoint_paper_kinetic_fixed_site_material_state(state)
    foreign = restore_paper_kinetic_fixed_site_material_state(
        initial_checkpoint,
        world=world,
        device="cpu",
        maximum_material_state_logical_tensor_bytes=1_000_000,
    )
    snapshot = _snapshot(state, background)
    with pytest.raises(ValueError, match="stale or foreign"):
        snapshot.assert_current(foreign)

    receipt = _seal_paper_kinetic_fixed_site_material_device_gradient_receipt_raw_for_test(
        state,
        snapshot,
        grad_site_rgba_f32_device=torch.full(
            (state.site_count, 4),
            0.01,
            dtype=torch.float32,
        ),
        loss_f32_device=torch.tensor((0.5,), dtype=torch.float32),
        step_generation_id="one-shot-step",
        device_completion_fence=_Fence(),
        device_completion_fence_provenance="cpu-synchronous-test-fence-v1",
    )
    cloned = replace(receipt)
    with pytest.raises(ValueError, match="changed or is foreign"):
        cloned.assert_current(state)

    apply_paper_kinetic_fixed_site_material_device_gradient_receipt(state, receipt)
    with pytest.raises(ValueError, match="already consumed"):
        apply_paper_kinetic_fixed_site_material_device_gradient_receipt(state, receipt)
    with pytest.raises(ValueError, match="already consumed"):
        snapshot.assert_current(state)


def test_device_snapshot_binds_material_tensor_identity_and_version(
    tmp_path: Path,
) -> None:
    state, _material, _world = _state(tmp_path)
    background = torch.tensor((0.02, 0.03, 0.04), dtype=torch.float32)
    snapshot = _snapshot(state, background)
    assert snapshot.site_rgba_f32_device is not None
    snapshot.site_rgba_f32_device[0, 0].add_(0.125)
    with pytest.raises(ValueError, match="tensor identity/layout changed"):
        snapshot.assert_current(state)


def test_device_snapshot_never_hashes_or_reads_back_live_material(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, _material, _world = _state(tmp_path)
    background = torch.tensor((0.02, 0.03, 0.04), dtype=torch.float32)
    original_digest = bridge_module._tensor_content_digest
    digest_calls: list[tuple[torch.device, tuple[int, ...]]] = []

    def guarded_digest(tensor: torch.Tensor) -> str:
        # Snapshot sealing may hash the tiny CPU background.  It must bind the
        # O(S) material through the live CPU state identity/version and the
        # fenced device destination identity/version, never through a device
        # readback or a full material content digest.
        assert tensor.device.type == "cpu"
        assert tensor is not state.site_rgba_f32
        assert tuple(tensor.shape) == (3,)
        digest_calls.append((tensor.device, tuple(tensor.shape)))
        return original_digest(tensor)

    def forbidden_device_to_host(*_args, **_kwargs):
        raise AssertionError("snapshot creation attempted a device-to-host readback")

    monkeypatch.setattr(bridge_module, "_tensor_content_digest", guarded_digest)
    monkeypatch.setattr(bridge_module, "_owned_cpu_f32", forbidden_device_to_host)
    snapshot = _snapshot(state, background)
    snapshot.assert_current(state, require_unissued=True)

    assert digest_calls == [(torch.device("cpu"), (3,))]
