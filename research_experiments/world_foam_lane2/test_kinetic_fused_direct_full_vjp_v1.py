from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import kinetic_native_equal_rank_runtime_adapter as runtime_adapter


ROOT = Path(__file__).resolve().parents[2]
VARIANT = (
    ROOT
    / "third_party"
    / "fast-mac-gsplat"
    / "variants"
    / "world_foam_lane2_fused_slab_v0"
)
OPS = VARIANT / "torch_world_foam_lane2_fused_slab" / "ops.py"
METAL = VARIANT / "csrc" / "metal" / "world_foam_lane2_shared_replay_tensor.metal"
HOST = VARIANT / "csrc" / "metal" / "world_foam_lane2_metal.mm"
BINDINGS = VARIANT / "csrc" / "bindings.cpp"
STAGED_ORACLE = (
    ROOT
    / "research_experiments"
    / "world_foam_lane2"
    / "kinetic_dense_cached_native_material_request.py"
)
RUNTIME_ADAPTER = (
    ROOT
    / "research_experiments"
    / "world_foam_lane2"
    / "kinetic_native_equal_rank_runtime_adapter.py"
)
SPARSE_GEOMETRY_ORACLE = (
    ROOT
    / "research_experiments"
    / "world_foam_lane2"
    / "kinetic_native_equal_rank_sparse_geometry_reduction.py"
)


_FUSED_STATUS_INVALID_ROW = 0x02
_FUSED_STATUS_NONFINITE_DYNAMIC_GRAD = 0x10
_FUSED_STATUS_NONFINITE_INTERMEDIATE = 0x80
_FUSED_STATUS_OUTPUT_LEDGER = 0x100

_CpuFakeFusedBlock = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


@dataclass(frozen=True)
class _CpuFakeFusedDirectFullVjpResult:
    """CPU behavioral stand-in for the source-only native transaction."""

    grad_site_rgba_f32: torch.Tensor
    grad_global_positions0_f32: torch.Tensor
    grad_global_velocities_f32: torch.Tensor
    grad_global_weight_coefficients_f32: torch.Tensor
    validation_status_i32: torch.Tensor


def _cpu_fake_inputs(
    row_count: int = 4,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    compact_to_global_i64 = torch.tensor([1, 4, 6], dtype=torch.int64)
    row_owner_i64 = torch.arange(row_count, dtype=torch.int64).remainder(
        compact_to_global_i64.numel()
    )
    row_gain_f32 = torch.linspace(0.25, 1.0, row_count, dtype=torch.float32)
    grad_node_chart_f32 = (
        torch.arange(row_count * 4, dtype=torch.float32).reshape(row_count, 4)
        .remainder(9)
        .sub_(4.0)
        .div_(7.0)
    )
    bars = (
        torch.zeros((3, 4), dtype=torch.float32),
        torch.zeros((8, 3), dtype=torch.float32),
        torch.zeros((8, 3), dtype=torch.float32),
        torch.zeros((8, 2), dtype=torch.float32),
    )
    return (
        row_owner_i64,
        compact_to_global_i64,
        row_gain_f32,
        grad_node_chart_f32,
        bars,
    )


def _cpu_staged_fused_vjp_reference(
    row_owner_i64: torch.Tensor,
    compact_to_global_i64: torch.Tensor,
    row_gain_f32: torch.Tensor,
    grad_node_chart_f32: torch.Tensor,
    bars: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent vectorized staged-VJP oracle for valid CPU fixtures."""

    scaled_chart = row_gain_f32[:, None] * grad_node_chart_f32
    position_rows = row_gain_f32[:, None] * grad_node_chart_f32[:, :3]
    velocity_rows = row_gain_f32[:, None] * grad_node_chart_f32[:, 1:4]
    weight_rows = row_gain_f32[:, None] * torch.stack(
        (
            grad_node_chart_f32[:, 0] + grad_node_chart_f32[:, 3],
            grad_node_chart_f32[:, 1] - grad_node_chart_f32[:, 2],
        ),
        dim=1,
    )
    global_owner_i64 = compact_to_global_i64.index_select(0, row_owner_i64)
    staged = tuple(torch.zeros_like(bar) for bar in bars)
    staged[0].index_add_(0, row_owner_i64, scaled_chart)
    staged[1].index_add_(0, global_owner_i64, position_rows)
    staged[2].index_add_(0, global_owner_i64, velocity_rows)
    staged[3].index_add_(0, global_owner_i64, weight_rows)
    return staged


def _cpu_fake_block_stage(
    row_owner_i64: torch.Tensor,
    compact_to_global_i64: torch.Tensor,
    row_gain_f32: torch.Tensor,
    grad_node_chart_f32: torch.Tensor,
    bars: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[
    int,
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """Evaluate one block for the CPU behavioral oracle.

    This test-only CPU oracle deliberately stages full-sized bar deltas.  It is
    evidence for all-or-nothing behavior only; the source assertions below,
    not this fake, enforce the native four-byte status/no-work-tape contract.
    """

    staged = tuple(torch.zeros_like(bar) for bar in bars)
    status_mask = 0
    compact_site_count = bars[0].shape[0]
    global_site_count = bars[1].shape[0]
    for row_id in range(row_owner_i64.numel()):
        owner = int(row_owner_i64[row_id].item())
        if owner < 0 or owner >= compact_site_count:
            status_mask |= _FUSED_STATUS_INVALID_ROW
            continue
        global_owner = int(compact_to_global_i64[owner].item())
        if global_owner < 0 or global_owner >= global_site_count:
            status_mask |= _FUSED_STATUS_INVALID_ROW
            continue
        grad_chart = grad_node_chart_f32[row_id]
        if not bool(torch.isfinite(grad_chart).all().item()):
            status_mask |= _FUSED_STATUS_NONFINITE_DYNAMIC_GRAD
            continue
        gain = row_gain_f32[row_id]
        row_contributions = (
            gain * grad_chart,
            gain * grad_chart[:3],
            gain * grad_chart[1:4],
            gain
            * torch.stack(
                (grad_chart[0] + grad_chart[3], grad_chart[1] - grad_chart[2])
            ),
        )
        updated_rows = (
            staged[0][owner] + row_contributions[0],
            staged[1][global_owner] + row_contributions[1],
            staged[2][global_owner] + row_contributions[2],
            staged[3][global_owner] + row_contributions[3],
        )
        if any(
            not bool(torch.isfinite(updated).all().item())
            for updated in updated_rows
        ):
            status_mask |= _FUSED_STATUS_NONFINITE_INTERMEDIATE
            continue
        staged[0][owner] = updated_rows[0]
        staged[1][global_owner] = updated_rows[1]
        staged[2][global_owner] = updated_rows[2]
        staged[3][global_owner] = updated_rows[3]
    return status_mask, staged


def _cpu_fake_validate_shared_status(
    block: _CpuFakeFusedBlock,
    bars: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    validation_status_i32: torch.Tensor,
    *,
    validate_shared_global_ledgers: bool,
) -> None:
    ledgers = bars if validate_shared_global_ledgers else bars[:1]
    if any(
        not bool(torch.isfinite(ledger).all().item())
        or bool(torch.any(ledger != 0.0).item())
        for ledger in ledgers
    ):
        validation_status_i32[0] = (
            int(validation_status_i32.item()) | _FUSED_STATUS_OUTPUT_LEDGER
        )
    status_mask, _staged = _cpu_fake_block_stage(*block, bars)
    validation_status_i32[0] = int(validation_status_i32.item()) | status_mask


def _cpu_fake_accumulate_shared_status(
    block: _CpuFakeFusedBlock,
    bars: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    validation_status_i32: torch.Tensor,
) -> None:
    if int(validation_status_i32.item()) != 0:
        return
    status_mask, staged = _cpu_fake_block_stage(*block, bars)
    if status_mask != 0:
        raise AssertionError("validated CPU fake block changed before accumulation")
    for bar, delta in zip(bars, staged, strict=True):
        bar.add_(delta)


def _cpu_fake_finalize_shared_status(
    bars: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    validation_status_i32: torch.Tensor,
    *,
    finalize_shared_global_ledgers: bool,
) -> None:
    ledgers = bars if finalize_shared_global_ledgers else bars[:1]
    if any(not bool(torch.isfinite(ledger).all().item()) for ledger in ledgers):
        validation_status_i32[0] = (
            int(validation_status_i32.item()) | _FUSED_STATUS_OUTPUT_LEDGER
        )


def _cpu_fake_validate_all_blocks_then_accumulate(
    blocks: tuple[_CpuFakeFusedBlock, ...],
    bars: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    protocol_events: list[str] | None = None,
) -> _CpuFakeFusedDirectFullVjpResult:
    validation_status_i32 = torch.zeros(1, dtype=torch.int32)
    for block_index, block in enumerate(blocks):
        if protocol_events is not None:
            protocol_events.append(f"validate:{block_index}")
        _cpu_fake_validate_shared_status(
            block,
            bars,
            validation_status_i32,
            validate_shared_global_ledgers=block_index == 0,
        )
    for block_index, block in enumerate(blocks):
        if protocol_events is not None:
            protocol_events.append(f"accumulate:{block_index}")
        _cpu_fake_accumulate_shared_status(block, bars, validation_status_i32)
    for block_index, _block in enumerate(blocks):
        if protocol_events is not None:
            protocol_events.append(f"finalize:{block_index}")
        _cpu_fake_finalize_shared_status(
            bars,
            validation_status_i32,
            finalize_shared_global_ledgers=block_index == 0,
        )
    if protocol_events is not None:
        protocol_events.append("device_completion_fence")
    return _CpuFakeFusedDirectFullVjpResult(
        grad_site_rgba_f32=bars[0],
        grad_global_positions0_f32=bars[1],
        grad_global_velocities_f32=bars[2],
        grad_global_weight_coefficients_f32=bars[3],
        validation_status_i32=validation_status_i32,
    )


def _cpu_fake_native_validate_before_write(
    row_owner_i64: torch.Tensor,
    compact_to_global_i64: torch.Tensor,
    row_gain_f32: torch.Tensor,
    grad_node_chart_f32: torch.Tensor,
    bars: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> _CpuFakeFusedDirectFullVjpResult:
    block = (
        row_owner_i64,
        compact_to_global_i64,
        row_gain_f32,
        grad_node_chart_f32,
    )
    return _cpu_fake_validate_all_blocks_then_accumulate((block,), bars)


def _result_bars(
    result: _CpuFakeFusedDirectFullVjpResult,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        result.grad_site_rgba_f32,
        result.grad_global_positions0_f32,
        result.grad_global_velocities_f32,
        result.grad_global_weight_coefficients_f32,
    )


def _snapshot_bars(
    bars: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    return tuple((bar.clone(), bar.view(torch.uint8).clone()) for bar in bars)


def _assert_bars_unchanged(
    bars: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    snapshots: tuple[tuple[torch.Tensor, torch.Tensor], ...],
) -> None:
    for bar, (values, storage_bytes) in zip(bars, snapshots, strict=True):
        assert torch.equal(bar, values)
        assert torch.equal(bar.view(torch.uint8), storage_bytes)


def _accept_cpu_fake_result(
    result: _CpuFakeFusedDirectFullVjpResult,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    status = result.validation_status_i32
    if status.dtype != torch.int32 or tuple(status.shape) != (1,):
        raise RuntimeError("fused validation receipt must be one int32 scalar")
    reason_mask = int(status.item())
    if reason_mask != 0:
        raise RuntimeError(f"fused validation rejected reason mask {reason_mask}")
    return _result_bars(result)


def _assert_rejected_without_bar_writes(
    row_owner_i64: torch.Tensor,
    compact_to_global_i64: torch.Tensor,
    row_gain_f32: torch.Tensor,
    grad_node_chart_f32: torch.Tensor,
    bars: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    expected_reason: int,
) -> None:
    snapshots = _snapshot_bars(bars)
    result = _cpu_fake_native_validate_before_write(
        row_owner_i64,
        compact_to_global_i64,
        row_gain_f32,
        grad_node_chart_f32,
        bars,
    )
    assert result.validation_status_i32.dtype == torch.int32
    assert tuple(result.validation_status_i32.shape) == (1,)
    assert int(result.validation_status_i32.item()) & expected_reason
    for returned, caller_owned in zip(_result_bars(result), bars, strict=True):
        assert returned is caller_owned
    _assert_bars_unchanged(bars, snapshots)
    with pytest.raises(RuntimeError, match="rejected reason mask"):
        _accept_cpu_fake_result(result)


def test_cpu_fake_fused_transaction_rejects_late_invalid_row_atomically() -> None:
    owner, compact_to_global, gain, grad_chart, bars = _cpu_fake_inputs()
    owner[-1] = compact_to_global.numel()
    _assert_rejected_without_bar_writes(
        owner,
        compact_to_global,
        gain,
        grad_chart,
        bars,
        expected_reason=_FUSED_STATUS_INVALID_ROW,
    )


def test_cpu_fake_fused_transaction_rejects_nonfinite_dynamic_grad_atomically() -> None:
    for invalid_value in (float("nan"), float("inf"), float("-inf")):
        owner, compact_to_global, gain, grad_chart, bars = _cpu_fake_inputs()
        grad_chart[-1, 2] = invalid_value
        _assert_rejected_without_bar_writes(
            owner,
            compact_to_global,
            gain,
            grad_chart,
            bars,
            expected_reason=_FUSED_STATUS_NONFINITE_DYNAMIC_GRAD,
        )


def test_cpu_fake_fused_transaction_rejects_staging_overflow_atomically() -> None:
    owner, compact_to_global, gain, grad_chart, bars = _cpu_fake_inputs(row_count=2)
    owner.zero_()
    gain.fill_(1.0)
    grad_chart.zero_()
    grad_chart[:, 0] = 0.75 * torch.finfo(torch.float32).max
    _assert_rejected_without_bar_writes(
        owner,
        compact_to_global,
        gain,
        grad_chart,
        bars,
        expected_reason=_FUSED_STATUS_NONFINITE_INTERMEDIATE,
    )


def test_cpu_fake_fused_transaction_rejects_finite_nonzero_scratch_before_write() -> None:
    owner, compact_to_global, gain, grad_chart, bars = _cpu_fake_inputs()
    bars[2][0, 1] = 0.125
    snapshots = _snapshot_bars(bars)
    result = _cpu_fake_native_validate_before_write(
        owner,
        compact_to_global,
        gain,
        grad_chart,
        bars,
    )
    assert int(result.validation_status_i32.item()) & _FUSED_STATUS_OUTPUT_LEDGER
    _assert_bars_unchanged(bars, snapshots)
    with pytest.raises(RuntimeError, match="rejected reason mask"):
        _accept_cpu_fake_result(result)


def test_cpu_fake_fused_transaction_accepts_signed_zero_scratch() -> None:
    owner, compact_to_global, gain, grad_chart, bars = _cpu_fake_inputs()
    bars[0][0, 0] = -0.0
    bars[1][0, 0] = -0.0
    result = _cpu_fake_native_validate_before_write(
        owner,
        compact_to_global,
        gain,
        grad_chart,
        bars,
    )
    assert int(result.validation_status_i32.item()) == 0


def test_cpu_fake_two_block_request_rejects_late_invalid_block_atomically() -> None:
    owner_a, compact_a, gain_a, grad_a, bars = _cpu_fake_inputs(row_count=3)
    owner_b, compact_b, gain_b, grad_b, _unused_bars = _cpu_fake_inputs(row_count=4)
    gain_b.mul_(0.6)
    grad_b.add_(0.125)
    owner_b[-1] = compact_b.numel()
    snapshots = _snapshot_bars(bars)
    result = _cpu_fake_validate_all_blocks_then_accumulate(
        (
            (owner_a, compact_a, gain_a, grad_a),
            (owner_b, compact_b, gain_b, grad_b),
        ),
        bars,
    )
    assert result.validation_status_i32.dtype == torch.int32
    assert tuple(result.validation_status_i32.shape) == (1,)
    assert int(result.validation_status_i32.item()) & _FUSED_STATUS_INVALID_ROW
    _assert_bars_unchanged(bars, snapshots)
    with pytest.raises(RuntimeError, match="rejected reason mask"):
        _accept_cpu_fake_result(result)


def test_cpu_fake_two_block_request_matches_combined_staged_parity() -> None:
    owner_a, compact_a, gain_a, grad_a, bars = _cpu_fake_inputs(row_count=3)
    owner_b, compact_b, gain_b, grad_b, _unused_bars = _cpu_fake_inputs(row_count=4)
    gain_b.mul_(0.6)
    grad_b.add_(0.125)
    initial = tuple(bar.clone() for bar in bars)
    staged_a = _cpu_staged_fused_vjp_reference(
        owner_a,
        compact_a,
        gain_a,
        grad_a,
        bars,
    )
    staged_b = _cpu_staged_fused_vjp_reference(
        owner_b,
        compact_b,
        gain_b,
        grad_b,
        bars,
    )
    result = _cpu_fake_validate_all_blocks_then_accumulate(
        (
            (owner_a, compact_a, gain_a, grad_a),
            (owner_b, compact_b, gain_b, grad_b),
        ),
        bars,
    )
    assert result.validation_status_i32.dtype == torch.int32
    assert tuple(result.validation_status_i32.shape) == (1,)
    assert int(result.validation_status_i32.item()) == 0
    for actual, before, delta_a, delta_b in zip(
        _accept_cpu_fake_result(result),
        initial,
        staged_a,
        staged_b,
        strict=True,
    ):
        torch.testing.assert_close(actual, before + delta_a + delta_b)


def test_cpu_fake_three_phase_transaction_has_one_final_fence_and_read() -> None:
    owner_a, compact_a, gain_a, grad_a, bars = _cpu_fake_inputs(row_count=2)
    owner_b, compact_b, gain_b, grad_b, _unused_bars = _cpu_fake_inputs(row_count=3)
    protocol_events: list[str] = []
    result = _cpu_fake_validate_all_blocks_then_accumulate(
        (
            (owner_a, compact_a, gain_a, grad_a),
            (owner_b, compact_b, gain_b, grad_b),
        ),
        bars,
        protocol_events=protocol_events,
    )
    protocol_events.append("host_status_read")
    assert int(result.validation_status_i32.item()) == 0
    assert protocol_events == [
        "validate:0",
        "validate:1",
        "accumulate:0",
        "accumulate:1",
        "finalize:0",
        "finalize:1",
        "device_completion_fence",
        "host_status_read",
    ]


def test_cpu_fake_postwrite_finalizer_quarantines_atomic_sum_overflow() -> None:
    owner_a, compact_a, gain_a, grad_a, bars = _cpu_fake_inputs(row_count=1)
    owner_b, compact_b, gain_b, grad_b, _unused_bars = _cpu_fake_inputs(row_count=1)
    for bar in bars:
        bar.zero_()
    owner_a.zero_()
    owner_b.zero_()
    gain_a.fill_(1.0)
    gain_b.fill_(1.0)
    grad_a.zero_()
    grad_b.zero_()
    contribution = 0.75 * torch.finfo(torch.float32).max
    grad_a[:, 0] = contribution
    grad_b[:, 0] = contribution
    result = _cpu_fake_validate_all_blocks_then_accumulate(
        (
            (owner_a, compact_a, gain_a, grad_a),
            (owner_b, compact_b, gain_b, grad_b),
        ),
        bars,
    )
    assert int(result.validation_status_i32.item()) & _FUSED_STATUS_OUTPUT_LEDGER
    assert any(not bool(torch.isfinite(bar).all().item()) for bar in bars)
    with pytest.raises(RuntimeError, match="rejected reason mask"):
        _accept_cpu_fake_result(result)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "native preflight checks existing bars and individual contributions, "
        "but not prospective cross-thread destination sums"
    ),
)
def test_native_preflight_closes_atomic_destination_overflow_before_write() -> None:
    metal = METAL.read_text(encoding="utf-8")
    validation = metal.split(
        "static inline uint wf2_kinetic_fused_direct_full_vjp_validation_reason_v1(",
        1,
    )[1].split(
        "kernel void wf2_kinetic_fused_direct_full_vjp_validate_v1_tensor(", 1
    )[0]
    for bar_name in (
        "grad_site_rgba_f32",
        "grad_global_positions0_f32",
        "grad_global_velocities_f32",
        "grad_global_weight_coefficients_f32",
    ):
        assert bar_name in validation
    assert "WF2_KINETIC_FUSED_V1_REASON_ACCUMULATION" in validation


def test_cpu_fake_fused_transaction_success_matches_staged_reference() -> None:
    owner, compact_to_global, gain, grad_chart, bars = _cpu_fake_inputs()
    initial = tuple(bar.clone() for bar in bars)
    staged = _cpu_staged_fused_vjp_reference(
        owner,
        compact_to_global,
        gain,
        grad_chart,
        bars,
    )
    result = _cpu_fake_native_validate_before_write(
        owner,
        compact_to_global,
        gain,
        grad_chart,
        bars,
    )
    assert result.validation_status_i32.dtype == torch.int32
    assert tuple(result.validation_status_i32.shape) == (1,)
    assert int(result.validation_status_i32.item()) == 0
    accepted = _accept_cpu_fake_result(result)
    for actual, before, expected_delta in zip(
        accepted,
        initial,
        staged,
        strict=True,
    ):
        torch.testing.assert_close(actual, before + expected_delta)


def test_cpu_fake_fused_validation_receipt_is_o1_not_a_work_tape() -> None:
    receipt_bytes = []
    for row_count in (1, 257):
        owner, compact_to_global, gain, grad_chart, bars = _cpu_fake_inputs(row_count)
        result = _cpu_fake_native_validate_before_write(
            owner,
            compact_to_global,
            gain,
            grad_chart,
            bars,
        )
        status = result.validation_status_i32
        assert status.dtype == torch.int32
        assert tuple(status.shape) == (1,)
        receipt_bytes.append(status.numel() * status.element_size())
    assert receipt_bytes == [4, 4]


def _physical_lengths(
    positions0: torch.Tensor,
    velocities: torch.Tensor,
    weight_coefficients: torch.Tensor,
    ray_coefficients: torch.Tensor,
    node_times: torch.Tensor,
    near: float,
    far: float,
) -> torch.Tensor:
    rows = []
    for time in node_times:
        powers = torch.stack((torch.ones_like(time), time, time.square()))
        positions = positions0 + time * velocities
        weights = weight_coefficients @ powers
        origin = ray_coefficients[:3] + time * ray_coefficients[3:6]
        direction = ray_coefficients[6:9] + time * ray_coefficients[9:12]
        speed = torch.linalg.vector_norm(direction)
        cuts = [positions0.new_tensor(near)]
        for left in range(positions.shape[0] - 1):
            right = left + 1
            normal = 2.0 * (positions[right] - positions[left])
            denominator = torch.dot(normal, direction)
            intercept = (
                torch.dot(normal, origin)
                + torch.dot(positions[left], positions[left])
                - torch.dot(positions[right], positions[right])
                - weights[left]
                + weights[right]
            )
            cuts.append(-intercept / denominator)
        cuts.append(positions0.new_tensor(far))
        cut_tensor = torch.stack(cuts)
        rows.append(speed * (cut_tensor[1:] - cut_tensor[:-1]))
    return torch.stack(rows)


def _adjacent_bar_reference(
    positions0: torch.Tensor,
    velocities: torch.Tensor,
    weight_coefficients: torch.Tensor,
    ray_coefficients: torch.Tensor,
    node_times: torch.Tensor,
    near: float,
    far: float,
    bar_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_positions0 = torch.zeros_like(positions0)
    grad_velocities = torch.zeros_like(velocities)
    grad_weights = torch.zeros_like(weight_coefficients)
    for node_id, time in enumerate(node_times):
        powers = torch.stack((torch.ones_like(time), time, time.square()))
        positions = positions0 + time * velocities
        weights = weight_coefficients @ powers
        origin = ray_coefficients[:3] + time * ray_coefficients[3:6]
        direction = ray_coefficients[6:9] + time * ray_coefficients[9:12]
        speed = torch.linalg.vector_norm(direction)
        cuts = [positions0.new_tensor(near)]
        denominators = []
        for left in range(positions.shape[0] - 1):
            right = left + 1
            normal = 2.0 * (positions[right] - positions[left])
            denominator = torch.dot(normal, direction)
            intercept = (
                torch.dot(normal, origin)
                + torch.dot(positions[left], positions[left])
                - torch.dot(positions[right], positions[right])
                - weights[left]
                + weights[right]
            )
            cuts.append(-intercept / denominator)
            denominators.append(denominator)
        cuts.append(positions0.new_tensor(far))
        cut_tensor = torch.stack(cuts)
        current_bars = bar_lengths[node_id]
        cut_bars = speed * (current_bars[:-1] - current_bars[1:])
        position_bars = torch.zeros_like(positions)
        weight_bars = torch.zeros_like(weights)
        for cut_id, cut_bar in enumerate(cut_bars):
            left = cut_id
            right = cut_id + 1
            depth = cut_tensor[cut_id + 1]
            point = origin + depth * direction
            implicit_bar = -cut_bar / denominators[cut_id]
            position_bars[left] += (
                implicit_bar * 2.0 * (positions[left] - point)
            )
            position_bars[right] += (
                implicit_bar * 2.0 * (point - positions[right])
            )
            weight_bars[left] -= implicit_bar
            weight_bars[right] += implicit_bar
        grad_positions0 += position_bars
        grad_velocities += time * position_bars
        grad_weights += weight_bars[:, None] * powers[None, :]
    return grad_positions0, grad_velocities, grad_weights


def test_adjacent_bar_recurrence_matches_dense_physical_length_autograd() -> None:
    dtype = torch.float64
    positions0 = torch.tensor(
        [[0.00, 0.00, 1.00], [0.05, -0.01, 2.00], [-0.03, 0.02, 3.00]],
        dtype=dtype,
        requires_grad=True,
    )
    velocities = torch.tensor(
        [[0.01, 0.00, 0.03], [-0.02, 0.01, -0.02], [0.01, -0.02, 0.01]],
        dtype=dtype,
        requires_grad=True,
    )
    weights = torch.tensor(
        [[0.02, -0.01, 0.004], [-0.03, 0.02, -0.003], [0.01, 0.01, 0.002]],
        dtype=dtype,
        requires_grad=True,
    )
    ray = torch.tensor(
        [
            0.05,
            -0.02,
            0.00,
            0.01,
            0.02,
            0.03,
            0.10,
            0.01,
            1.00,
            0.03,
            -0.01,
            -0.02,
        ],
        dtype=dtype,
        requires_grad=False,
    )
    node_times = torch.tensor([0.15, 0.72], dtype=dtype)
    bar_lengths = torch.tensor(
        [[0.31, -0.27, 0.19], [-0.14, 0.22, 0.37]],
        dtype=dtype,
    )
    lengths = _physical_lengths(
        positions0,
        velocities,
        weights,
        ray,
        node_times,
        0.2,
        3.8,
    )
    dense_grads = torch.autograd.grad(
        torch.sum(lengths * bar_lengths),
        (positions0, velocities, weights),
    )
    adjacent_grads = _adjacent_bar_reference(
        positions0.detach(),
        velocities.detach(),
        weights.detach(),
        ray.detach(),
        node_times,
        0.2,
        3.8,
        bar_lengths,
    )
    for dense, adjacent in zip(dense_grads, adjacent_grads, strict=True):
        torch.testing.assert_close(dense, adjacent, rtol=2.0e-11, atol=2.0e-11)


def test_suffixed_native_source_has_no_length_bar_or_work_axis_status() -> None:
    metal = METAL.read_text(encoding="utf-8")
    validation_kernel = metal.split(
        "kernel void wf2_kinetic_fused_direct_full_vjp_validate_v1_tensor(", 1
    )[1].split("kernel void wf2_kinetic_fused_direct_full_vjp_v1_tensor(", 1)[0]
    kernel = metal.split(
        "kernel void wf2_kinetic_fused_direct_full_vjp_v1_tensor(", 1
    )[1].split("kernel void wf2_clear_affine_loss_site_rgba_grad_tensor", 1)[0]
    assert "grad_node_chart_f32" in validation_kernel
    validation_helper = metal.split(
        "uint wf2_kinetic_fused_direct_full_vjp_validation_reason_v1(", 1
    )[1].split(
        "kernel void wf2_kinetic_fused_direct_full_vjp_validate_v1_tensor(", 1
    )[0]
    assert "all(isfinite(grad_chart))" in validation_helper
    assert "atomic_fetch_or_explicit" in validation_kernel
    assert "validation_status_u32, reason" in validation_kernel
    assert kernel.index(
        "atomic_load_explicit(validation_status_u32"
    ) < kernel.index("wf2_atomic_add4(")
    assert "grad_node_physical_length" not in kernel
    assert "previous_bar_ell - current_bar_ell" in kernel
    assert "row_node_time_f32[row_id * node_count + node_id]" in kernel
    assert "row_near_far_f32" in kernel
    assert "source_site_ids_i64" in kernel
    assert "const float optical_depth = rgba.w * physical_length" in kernel
    assert "!isfinite(optical_depth)" in kernel

    host = HOST.read_text(encoding="utf-8")
    host_launch = host.split(
        "metal_kinetic_fused_direct_full_vjp_phase_core_v1(", 1
    )[1].split("torch::Tensor metal_fixed_word_p0_lie_material_node_vjp", 1)[0]
    assert "grad_node_physical_length" not in host_launch
    assert "torch::zeros({1}, config_i32.options())" in host_launch
    assert "torch::zeros({row_count" not in host_launch
    assert "torch::zeros({node_count" not in host_launch
    assert "torch::zeros({row_count, node_count" not in host_launch
    assert "torch::empty" not in host_launch
    assert "torch::clone" not in host_launch
    assert 'check_i32_mps_1d(config_i32, "config_i32", 6)' in host_launch
    assert "config_f32.numel() == 7" in host_launch
    assert "source_site_ids_i64.scalar_type() == torch::kInt64" in host_launch
    assert "all fused kinetic tensors must share one MPS device" in host_launch
    assert "fused kinetic output bars must be storage-distinct" in host_launch
    assert "fused kinetic row-node chart indexing exceeds uint32" in host_launch
    assert "fused kinetic node-word indexing exceeds uint32" in host_launch
    assert "fused kinetic global weight indexing exceeds uint32" in host_launch
    assert "fn.setArg(24, global_site_count_i32)" in host_launch
    assert "fn.setArg(25, validation_status_i32)" in host_launch
    assert host_launch.index(
        "launch(k.kinetic_fused_direct_full_vjp_validate_v1"
    ) < host_launch.index("launch(k.kinetic_fused_direct_full_vjp_v1")
    assert "validation_status_i32);" in host_launch

    bindings = BINDINGS.read_text(encoding="utf-8")
    schema = next(
        line
        for line in bindings.splitlines()
        if "kinetic_fused_direct_full_vjp_accumulate_launch_only_v1(Tensor" in line
    )
    assert "grad_node_physical_length" not in schema
    assert "Tensor source_site_ids_i64" in schema
    assert "Tensor row_node_time_f32" in schema
    assert "Tensor row_near_far_f32" in schema
    assert "Tensor row_ray_coeff_f32" in schema
    assert "grad_row_ray_coeff" not in schema
    assert "optimize_camera_rays" not in schema
    assert schema.endswith(
        "-> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!), Tensor)\");"
    )


def test_source_exposes_shared_status_cross_block_transaction_phases() -> None:
    ops = OPS.read_text(encoding="utf-8")
    status_init = ops.split(
        "def kinetic_fused_direct_full_vjp_validation_status_init_v1(", 1
    )[1].split("def kinetic_fused_direct_full_vjp_accumulate_launch_only_v1(", 1)[0]
    assert "torch.zeros((1,), dtype=torch.int32" in status_init
    assert "coordinator must allocate this once" in status_init
    split_launch = ops.split(
        "def kinetic_fused_direct_full_vjp_accumulate_launch_only_v1(", 1
    )[1].split("def fixed_word_p0_lie_material_node_vjp", 1)[0]
    assert 'launch_phase: str = "combined"' in split_launch
    assert "validation_status_i32: Tensor | None = None" in split_launch
    assert "validate_shared_global_ledgers: bool | None = None" in split_launch
    assert "finalize_shared_global_ledgers: bool | None = None" in split_launch
    assert "on exactly the first" in split_launch
    assert "and ``False`` on the rest" in split_launch
    assert "split validation must explicitly choose" in split_launch
    assert (
        "kinetic_fused_direct_full_vjp_validate_shared_status_launch_only_v1"
        in split_launch
    )
    assert (
        "kinetic_fused_direct_full_vjp_accumulate_shared_status_launch_only_v1"
        in split_launch
    )
    assert (
        "kinetic_fused_direct_full_vjp_finalize_shared_status_launch_only_v1"
        in split_launch
    )
    assert 'accumulation_enqueued=launch_phase in {"combined", "accumulate"}' in split_launch
    assert 'finalization_enqueued=launch_phase in {"combined", "finalize"}' in split_launch
    assert 'shared_status_reused=launch_phase != "combined"' in split_launch

    bindings = BINDINGS.read_text(encoding="utf-8")
    validation_schema = next(
        line
        for line in bindings.splitlines()
        if "fused_direct_full_vjp_validate_shared_status_launch_only_v1(Tensor" in line
    )
    assert "Tensor(e!) validation_status_i32" in validation_schema
    assert "bool validate_shared_global_ledgers" in validation_schema
    assert validation_schema.endswith("-> Tensor(e!)\");")
    accumulation_schema = next(
        line
        for line in bindings.splitlines()
        if "fused_direct_full_vjp_accumulate_shared_status_launch_only_v1(Tensor"
        in line
    )
    assert "Tensor(e) validation_status_i32" in accumulation_schema
    assert accumulation_schema.endswith(
        "-> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!), Tensor(e))\");"
    )
    finalization_schema = next(
        line
        for line in bindings.splitlines()
        if "fused_direct_full_vjp_finalize_shared_status_launch_only_v1(Tensor"
        in line
    )
    assert "bool finalize_shared_global_ledgers" in finalization_schema
    assert finalization_schema.endswith(
        "-> (Tensor(a), Tensor(b), Tensor(c), Tensor(d), Tensor(e!))\");"
    )

    metal = METAL.read_text(encoding="utf-8")
    assert "value != 0.0f" in metal
    assert "with either sign of" in metal
    assert "wf2_kinetic_fused_direct_full_vjp_finalize_v1_tensor" in metal
    assert "does not promise byte-for-byte rollback" in metal


def test_adapter_seals_single_use_zero_scratch_and_fail_stop_transaction() -> None:
    adapter = RUNTIME_ADAPTER.read_text(encoding="utf-8")
    assert "def _assert_fused_process_not_quarantined()" in adapter
    assert adapter.count("_assert_fused_process_not_quarantined()") >= 5
    assert "process restart is required" in adapter
    prepared = adapter.split(
        "def prepare_kinetic_native_equal_rank_fused_direct_full_vjp_transaction_v1(",
        1,
    )[1].split(
        "def execute_kinetic_native_equal_rank_fused_direct_full_vjp_transaction_v1(",
        1,
    )[0]
    assert prepared.index("required_output_bytes =") < prepared.index("torch.zeros(")
    assert "max_output_scratch_tensor_bytes < required_output_bytes" in prepared
    assert "duplicate active generation" in adapter
    assert "exact_zero_output_scratch_allocated: bool = True" in adapter
    assert "active_manifest_coverage_certified: bool = False" in adapter
    assert "single_use_scratch_generation_certified: bool = True" in adapter
    assert "set(node_storage) & raw_storage" in adapter

    execute = adapter.split(
        "def execute_kinetic_native_equal_rank_fused_direct_full_vjp_transaction_v1(",
        1,
    )[1].split("def execute_kinetic_native_equal_rank_node_vjp(", 1)[0]
    assert 'for phase in ("validate", "accumulate", "finalize"):' in execute
    assert execute.index("state.consumed = True") < execute.index(
        "validation_status = init_status("
    )
    assert "init_abort_returned = device_completion_fence()" in execute
    assert execute.index("state.consumed = True") < execute.index(
        'for phase in ("validate", "accumulate", "finalize"):'
    )
    assert "state.launch_attempt_count += 1" in execute
    assert "abort_returned = device_completion_fence()" in execute
    assert "_FUSED_VJP_RESTART_REQUIRED_QUARANTINE.append(state)" in execute
    assert "restart required and scratch remains quarantined" in execute
    assert execute.index("returned = device_completion_fence()") < execute.index(
        "reason_mask = int(validation_status.item())"
    )
    assert "del raw_result" in execute
    assert "retained_validation_status_tensor_bytes: int = 0" in adapter
    receipt = adapter.split(
        "class KineticNativeEqualRankFusedDirectFullVjpV1TransactionResult:",
        1,
    )[1].split(
        "class KineticNativeEqualRankFusedUnionFullVjpV2ConstructionLifetime:",
        1,
    )[0]
    assert "validation_status_i32:" not in receipt
    assert "raw_result" not in receipt
    assert "optimizer_fail_atomicity_certified: bool = False" in receipt
    assert "prospective_atomic_sum_bound_certified: bool = False" in receipt
    assert "postwrite_failure_byte_rollback_certified: bool = False" in receipt

    abi = adapter.split("def _fused_abi_identity(", 1)[1].split(
        "def _fused_generation_id(", 1
    )[0]
    assert "FUSED_STATUS_INIT_OP_NAME" in abi
    exports = adapter.split("__all__ = [", 1)[1]
    for name in (
        "FUSED_STATUS_INIT_OP_NAME",
        "KineticNativeEqualRankFusedDirectFullVjpV1Transaction",
        "KineticNativeEqualRankFusedDirectFullVjpV1TransactionResult",
        "prepare_kinetic_native_equal_rank_fused_direct_full_vjp_transaction_v1",
        "execute_kinetic_native_equal_rank_fused_direct_full_vjp_transaction_v1",
    ):
        assert f'"{name}"' in exports


@pytest.mark.parametrize(
    "mutation_kind",
    (
        "output",
        "status",
        "output_storage",
        "status_storage",
        "lifecycle",
        "structure",
    ),
)
def test_adapter_rejects_completion_callback_transaction_mutation(
    monkeypatch,
    mutation_kind,
) -> None:
    class FakeFusedOps:
        def kinetic_fused_direct_full_vjp_validation_status_init_v1(self, _rgba):
            return torch.zeros((1,), dtype=torch.int32)

        def kinetic_fused_direct_full_vjp_accumulate_launch_only_v1(
            self,
            _raw_prepared,
            _node_bar,
            compact_bar,
            grad_positions,
            grad_velocities,
            grad_weights,
            *,
            validation_status_i32,
            launch_phase,
            **_kwargs,
        ):
            return SimpleNamespace(
                grad_site_rgba_f32=compact_bar,
                grad_global_positions0_f32=grad_positions,
                grad_global_velocities_f32=grad_velocities,
                grad_global_weight_coefficients_f32=grad_weights,
                validation_status_i32=validation_status_i32,
                accumulation_enqueued=launch_phase == "accumulate",
                finalization_enqueued=launch_phase == "finalize",
                shared_status_reused=True,
                runtime_status=(
                    "raw_fixed_camera_source_only_until_native_rebuild_and_sparse_oracle_parity"
                ),
            )

    class FakePreparedBlock:
        def __init__(self):
            self.world = SimpleNamespace(
                runtime=SimpleNamespace(device=torch.device("cpu")),
                compact_site_rgba_f32=torch.zeros((1, 4), dtype=torch.float32),
            )
            self.raw_prepared = object()
            self.fused_ops = FakeFusedOps()

        def assert_cold_current(self) -> None:
            return None

    block = FakePreparedBlock()
    node_bar = torch.zeros((1, 1, 4), dtype=torch.float32)
    compact_bar = torch.zeros((1, 4), dtype=torch.float32)
    grad_positions = torch.zeros((2, 3), dtype=torch.float32)
    grad_velocities = torch.zeros((2, 3), dtype=torch.float32)
    grad_weights = torch.zeros((2, 1), dtype=torch.float32)
    output_bars = (compact_bar, grad_positions, grad_velocities, grad_weights)
    state = runtime_adapter._KineticNativeEqualRankFusedDirectFullVjpV1TransactionState(
        prepared_blocks=(block,),
        grad_node_chart_f32_by_block=(node_bar,),
        grad_compact_site_rgba_f32_by_block=(compact_bar,),
        grad_global_positions0_f32=grad_positions,
        grad_global_velocities_f32=grad_velocities,
        grad_global_weight_coefficients_f32=grad_weights,
    )
    transaction = runtime_adapter.KineticNativeEqualRankFusedDirectFullVjpV1Transaction(
        _state=state,
        active_block_generation_ids=("synthetic-block-v1",),
        prepared_block_generation_ids=("synthetic-prepared-v1",),
        prepared_block_identities=(id(block),),
        node_bar_signatures=(runtime_adapter._warm_tensor_signature(node_bar),),
        output_bar_signatures=tuple(
            runtime_adapter._warm_tensor_signature(tensor)
            for tensor in output_bars
        ),
        compact_output_scratch_tensor_bytes=compact_bar.numel() * 4,
        global_output_scratch_tensor_bytes=sum(
            tensor.numel() * 4 for tensor in output_bars[1:]
        ),
        total_output_scratch_tensor_bytes=sum(
            tensor.numel() * 4 for tensor in output_bars
        ),
        output_scratch_tensor_byte_budget=10_000,
        output_scratch_tensor_count=len(output_bars),
        generation_id="synthetic-fused-transaction-v1",
        _seal=runtime_adapter._FUSED_VJP_TRANSACTION_SEAL,
    )
    monkeypatch.setattr(
        runtime_adapter,
        "_assert_fused_transaction_ready",
        lambda _transaction: None,
    )
    monkeypatch.setattr(
        runtime_adapter,
        "_fused_raw_prepared_tensors",
        lambda _raw: (),
    )

    def completion_fence() -> None:
        if mutation_kind == "output":
            state.grad_global_positions0_f32.add_(1.0)
        elif mutation_kind == "status":
            state.validation_status_i32.fill_(7)
        elif mutation_kind == "output_storage":
            state.grad_global_positions0_f32.data = torch.ones_like(
                state.grad_global_positions0_f32
            )
        elif mutation_kind == "status_storage":
            state.validation_status_i32.data = torch.zeros_like(
                state.validation_status_i32
            )
        elif mutation_kind == "lifecycle":
            state.accepted = True
        else:
            state.prepared_blocks = ()

    with pytest.raises(
        RuntimeError,
        match="completion callback mutated bound transaction state",
    ) as caught:
        runtime_adapter.execute_kinetic_native_equal_rank_fused_direct_full_vjp_transaction_v1(
            transaction,
            device_completion_fence=completion_fence,
            device_completion_fence_provenance="cpu-adversarial-callback-v1",
        )
    assert state.consumed
    assert state.settled
    assert state.quarantined
    assert not state.accepted
    assert not state.completion_unknown
    assert state.completion_fence_call_count == 1
    assert state.validation_status_i32 is None
    assert state.failure is caught.value
    assert state.prepared_blocks == (block,)


def test_fixed_camera_v1_has_no_ray_cotangent_or_alias_surface() -> None:
    metal = METAL.read_text(encoding="utf-8")
    kernel = metal.split(
        "kernel void wf2_kinetic_fused_direct_full_vjp_v1_tensor(", 1
    )[1].split("kernel void wf2_clear_affine_loss_site_rgba_grad_tensor", 1)[0]
    assert "row_ray_coeff_f32" in kernel
    assert "grad_row_ray_coeff" not in kernel
    assert "optimize_camera_rays" not in kernel
    assert "speed_bar" not in kernel
    assert "origin_bar" not in kernel
    assert "direction_bar" not in kernel
    assert "minimum_cut_cosine" in kernel
    assert "cut_cosine > minimum_cut_cosine" in kernel
    assert "coordinate_length > minimum_coordinate_length" in kernel

    ops = OPS.read_text(encoding="utf-8")
    launch = ops.split(
        "def kinetic_fused_direct_full_vjp_accumulate_launch_only_v1(", 1
    )[1].split("def fixed_word_p0_lie_material_node_vjp", 1)[0]
    assert "grad_row_ray_coeff" not in launch
    assert "optimize_camera_rays" not in launch
    assert "ray_bar_or_guarded_alias" not in launch
    assert "output bars must be storage-distinct" in launch
    assert "must not alias prepared primal storage" in launch
    result_fields = ops.split(
        "class KineticFusedDirectFullVjpResultV1:", 1
    )[1].split("def _tensor_mutation_signature", 1)[0]
    assert "grad_node_physical_length" not in result_fields
    assert "length_bar_allocated:" not in result_fields
    assert "length_bar_copy" not in result_fields
    assert "grad_row_ray_coeff" not in result_fields
    assert "validation_status_i32: Tensor" in result_fields
    assert "validation_status_tensor_bytes != 4" in result_fields
    assert "return int(self.validation_status_i32.item())" in result_fields
    accepted_bars = result_fields.split("def accepted_bars", 1)[1]
    assert accepted_bars.index(
        "reason_mask = self.validation_reason_mask()"
    ) < accepted_bars.index("return (")
    assert "if reason_mask != 0:" in accepted_bars
    assert "if not self.accumulation_enqueued:" in accepted_bars
    assert "retained_logical_tensor_bytes" in ops
    assert "preparer_owned_logical_tensor_bytes" in ops
    assert 'tensor.device.type not in {"cpu", "mps"}' in ops
    assert "return tensor, False" in ops


def test_fused_v1_stays_out_of_selected_abi_and_staged_oracle() -> None:
    ops = OPS.read_text(encoding="utf-8")
    selected = ops.split("_KINETIC_MEMORY_LIGHT_SELECTED_KERNELS =", 1)[1].split(
        "_KINETIC_MEMORY_LIGHT_COMPILED_SCHEMAS =", 1
    )[0]
    assert "kinetic_fused_direct_full_vjp" not in selected
    staged_oracle = STAGED_ORACLE.read_text(encoding="utf-8")
    assert "kinetic_fused_direct_full_vjp" not in staged_oracle
    assert "raw_fixed_camera_source_only_until_native_rebuild_and_sparse_oracle_parity" in ops
    raw_prepare = ops.split(
        "def prepare_kinetic_fused_direct_full_vjp_v1(", 1
    )[1].split("def _kinetic_fused_direct_full_vjp_v1_tensors", 1)[0]
    assert "continuous_owner_certificate_digests" not in raw_prepare
    assert "low-level function does not accept certificate-shaped strings" in raw_prepare
    assert "threshold_values_f32" in raw_prepare
    assert "must remain valid after float32 conversion" in raw_prepare

    adapter = RUNTIME_ADAPTER.read_text(encoding="utf-8")
    assert "prepare_kinetic_native_equal_rank_fused_direct_full_vjp_v1" in adapter
    assert "execute_kinetic_native_equal_rank_fused_direct_full_vjp_v1" in adapter
    assert "payload.assert_cold_current(self.lowering, self.sources)" in adapter
    assert "prepared.assert_cold_current()" in adapter
    sealed_execute = adapter.split(
        "def execute_kinetic_native_equal_rank_fused_direct_full_vjp_v1(", 1
    )[1].split("def execute_kinetic_native_equal_rank_node_vjp(", 1)[0]
    assert sealed_execute.index("accepted_bars = accept()") < sealed_execute.index(
        "return raw_result"
    )
    assert "getattr(raw_result, name, None) is not expected" in sealed_execute
    assert 'getattr(raw_result, "accumulation_enqueued", None) is not True' in sealed_execute
    assert 'getattr(raw_result, "finalization_enqueued", None) is not True' in sealed_execute
    assert 'getattr(raw_result, "shared_status_reused", None) is not False' in sealed_execute
    assert "if int(validation_status.item()) != 0:" in sealed_execute
    assert (
        "accepted fused fixed-camera VJP lost its scalar int32 validation receipt"
        in sealed_execute
    )
    assert "continuous_owner_certificate_digests" in adapter
    assert "trainer_promotion_complete: bool = False" in adapter
    assert "world.runtime.word_offsets_i32" in adapter
    assert "world.runtime.word_owner_i32" in adapter
    assert "world.runtime.source_site_ids_i64" in adapter
    assert "owned_row_payload_tensor_bytes" in adapter
    assert "owned_config_tensor_bytes" in adapter
    assert "persistent_frame_tensor_bytes: int = 0" in adapter
    assert "persistent_sample_tensor_bytes: int = 0" in adapter

    sparse_oracle = SPARSE_GEOMETRY_ORACLE.read_text(encoding="utf-8")
    assert "validate_kinetic_native_equal_rank_continuous_owner_certificate" in sparse_oracle
    assert "reduce_kinetic_native_equal_rank_sparse_geometry_vjp" in sparse_oracle
