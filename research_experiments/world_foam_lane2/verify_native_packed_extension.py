#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch


DYNAWORLD = Path(__file__).resolve().parents[2]
VARIANT_ROOT = DYNAWORLD / "third_party" / "fast-mac-gsplat" / "variants" / "world_foam_lane2_fused_slab_v0"


def _ensure_variant_imported() -> None:
    sys.path.insert(0, str(VARIANT_ROOT))
    import torch_world_foam_lane2_fused_slab  # noqa: F401


def _fixture() -> dict[str, Any]:
    return {
        "sorted_depths": torch.tensor([[[0.5, 0.5]]], dtype=torch.float64),
        "sorted_ids": torch.tensor([[[0, 0]]], dtype=torch.int64),
        "valid_counts": torch.tensor([[1, 1]], dtype=torch.int64),
        "row_active": torch.tensor([1], dtype=torch.int64),
        "ray_coeff": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float64),
        "frame_t": torch.tensor([0.0, 1.0], dtype=torch.float64),
        "site_xyz": torch.tensor([[0.0, 0.0, 0.2], [0.0, 0.0, 0.8]], dtype=torch.float64),
        "site_t": torch.tensor([0.0, 0.0], dtype=torch.float64),
        "site_weight": torch.tensor([0.0, 0.0], dtype=torch.float64),
        "boundary_other": torch.tensor([[1], [0]], dtype=torch.int64),
    }


def _changing_sorted_fixture() -> dict[str, Any]:
    fixture = _fixture()
    fixture["site_t"] = torch.tensor([0.0, 1.0], dtype=torch.float64)
    return fixture


def _two_candidate_sorted_fixture() -> dict[str, Any]:
    fixture = _fixture()
    fixture["sorted_depths"] = torch.tensor([[[0.4, 0.4], [0.6, 0.6]]], dtype=torch.float64)
    fixture["sorted_ids"] = torch.tensor([[[0, 0], [0, 0]]], dtype=torch.int64)
    fixture["valid_counts"] = torch.tensor([[2, 2]], dtype=torch.int64)
    return fixture


def _cut_fixture() -> dict[str, Any]:
    return {
        "cut_depths": torch.tensor([0.0, 0.5, 1.0, 0.0, 0.5, 1.0], dtype=torch.float64),
        "cut_ids": torch.tensor([-1, 0, -2, -1, 0, -2], dtype=torch.int64),
        "cut_offsets": torch.tensor([0, 3, 6], dtype=torch.int64),
        "start_segments": torch.tensor([0, 0], dtype=torch.int64),
        "initial_owner": torch.tensor([0, 0], dtype=torch.int64),
        "boundary_other": torch.tensor([[1], [0]], dtype=torch.int64),
    }


def _changing_cut_fixture() -> dict[str, Any]:
    fixture = _cut_fixture()
    fixture["initial_owner"] = torch.tensor([0, 1], dtype=torch.int64)
    return fixture


def _tensor_contract_fields(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    return {
        f"{name}_dtype": str(tensor.dtype).replace("torch.", ""),
        f"{name}_device": tensor.device.type,
        f"{name}_shape": list(tensor.shape),
        f"{name}_contiguous": bool(tensor.is_contiguous()),
    }


def _tensor_result(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    return {
        name: tensor.tolist(),
        **_tensor_contract_fields(name, tensor),
    }


def _assert_packed_matches_unpacked(ops: Any, unpacked: tuple[Any, ...], packed: tuple[Any, ...]) -> None:
    if len(unpacked) != 10:
        raise RuntimeError(f"expected 10 unpacked tensors, got {len(unpacked)}")
    if len(packed) != 12:
        raise RuntimeError(f"expected 12 packed tensors, got {len(packed)}")
    for index, tensor in enumerate(packed):
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(f"packed output {index} is not a tensor: {type(tensor)!r}")
        if tensor.device.type != "cpu":
            raise RuntimeError(f"packed output {index} must be a CPU tensor, got {tensor.device}")
        if tensor.dtype != torch.int32:
            raise RuntimeError(f"packed output {index} must be int32, got {tensor.dtype}")
        if tensor.ndim != 1:
            raise RuntimeError(f"packed output {index} must be 1D, got shape {tuple(tensor.shape)}")
        if not tensor.is_contiguous():
            raise RuntimeError(f"packed output {index} must be contiguous")

    for unpacked_index, packed_index in (
        (0, 0),
        (1, 1),
        (2, 2),
        (3, 3),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 10),
    ):
        torch.testing.assert_close(unpacked[unpacked_index], packed[packed_index])

    expected_base_record = ops.pack_endpoint_records_i32_cpu(unpacked[1], unpacked[2], unpacked[3])
    expected_change_record = ops.pack_endpoint_records_i32_cpu(unpacked[7], unpacked[8], unpacked[9])
    torch.testing.assert_close(packed[4], expected_base_record)
    torch.testing.assert_close(packed[11], expected_change_record)

    expected_base_owner = torch.tensor([0, 1], dtype=torch.int32)
    expected_base_left = torch.tensor([-1, 0], dtype=torch.int32)
    expected_base_right = torch.tensor([0, -2], dtype=torch.int32)
    torch.testing.assert_close(packed[1], expected_base_owner)
    torch.testing.assert_close(packed[2], expected_base_left)
    torch.testing.assert_close(packed[3], expected_base_right)


def _assert_expected_change_records(packed: tuple[Any, ...]) -> None:
    expected_track_change_offsets = torch.tensor([0, 1], dtype=torch.int32)
    expected_change_frame = torch.tensor([1], dtype=torch.int32)
    expected_change_offsets = torch.tensor([0, 2], dtype=torch.int32)
    expected_change_owner = torch.tensor([1, 0], dtype=torch.int32)
    expected_change_left = torch.tensor([-1, 0], dtype=torch.int32)
    expected_change_right = torch.tensor([0, -2], dtype=torch.int32)
    expected_change_record = torch.tensor([2097153, 1049088], dtype=torch.int32)
    torch.testing.assert_close(packed[5], expected_track_change_offsets)
    torch.testing.assert_close(packed[6], expected_change_frame)
    torch.testing.assert_close(packed[7], expected_change_offsets)
    torch.testing.assert_close(packed[8], expected_change_owner)
    torch.testing.assert_close(packed[9], expected_change_left)
    torch.testing.assert_close(packed[10], expected_change_right)
    torch.testing.assert_close(packed[11], expected_change_record)


def _assert_pack_rejects(ops: Any, name: str, owner: torch.Tensor, left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    try:
        ops.pack_endpoint_records_i32_cpu(owner, left, right)
    except RuntimeError as exc:
        return {
            name: True,
            f"{name}_message": str(exc).splitlines()[0],
        }
    raise RuntimeError(f"pack_endpoint_records_i32_cpu accepted invalid fixture {name}")


def _assert_op_rejects(name: str, op: Any, args: tuple[Any, ...]) -> dict[str, Any]:
    try:
        op(*args)
    except RuntimeError as exc:
        return {
            name: True,
            f"{name}_message": str(exc).splitlines()[0],
        }
    raise RuntimeError(f"native op accepted invalid fixture {name}")


def _verify_pack_rejects_bad_contracts(ops: Any) -> dict[str, Any]:
    good = torch.zeros((2,), dtype=torch.int32)
    return {
        **_assert_pack_rejects(
            ops,
            "pack_endpoint_records_i32_rejects_rank2",
            torch.zeros((1, 2), dtype=torch.int32),
            torch.zeros((1, 2), dtype=torch.int32),
            torch.zeros((1, 2), dtype=torch.int32),
        ),
        **_assert_pack_rejects(
            ops,
            "pack_endpoint_records_i32_rejects_owner_out_of_range",
            torch.tensor([256, 0], dtype=torch.int32),
            good,
            good,
        ),
        **_assert_pack_rejects(
            ops,
            "pack_endpoint_records_i32_rejects_cut_out_of_range",
            torch.zeros((2,), dtype=torch.int32),
            torch.tensor([4094, -1], dtype=torch.int32),
            good,
        ),
    }


def _verify_cut_rejects_bad_contracts(ops: Any) -> dict[str, Any]:
    bad_start = _cut_fixture()
    bad_start["start_segments"] = torch.tensor([2, 0], dtype=torch.int64)
    bad_owner = _cut_fixture()
    bad_owner["initial_owner"] = torch.tensor([-1, 0], dtype=torch.int64)
    bad_boundary_other = _cut_fixture()
    bad_boundary_other["boundary_other"] = torch.tensor([[1], [2]], dtype=torch.int64)
    bad_nan_depth = _cut_fixture()
    bad_nan_depth["cut_depths"] = torch.tensor([0.0, float("nan"), 1.0, 0.0, 0.5, 1.0], dtype=torch.float64)
    bad_decreasing_depth = _cut_fixture()
    bad_decreasing_depth["cut_depths"] = torch.tensor([0.0, 0.6, 0.5, 0.0, 0.5, 1.0], dtype=torch.float64)
    bad_first_sentinel = _cut_fixture()
    bad_first_sentinel["cut_ids"] = torch.tensor([0, 0, -2, -1, 0, -2], dtype=torch.int64)
    bad_last_sentinel = _cut_fixture()
    bad_last_sentinel["cut_ids"] = torch.tensor([-1, 0, 0, -1, 0, -2], dtype=torch.int64)
    bad_internal_id = _cut_fixture()
    bad_internal_id["cut_ids"] = torch.tensor([-1, 1, -2, -1, 0, -2], dtype=torch.int64)
    bad_single_cut = _cut_fixture()
    bad_single_cut["cut_offsets"] = torch.tensor([0, 1, 6], dtype=torch.int64)
    bad_single_cut["start_segments"] = torch.tensor([-1, 0], dtype=torch.int64)
    bad_single_cut["initial_owner"] = torch.tensor([-1, 0], dtype=torch.int64)
    return {
        **_assert_op_rejects(
            "gate4_delta_replace_from_cuts_rejects_start_segment_oob",
            ops.gate4_delta_replace_from_cuts_cpu,
            _cut_args(bad_start),
        ),
        **_assert_op_rejects(
            "gate4_delta_replace_packed_from_cuts_rejects_start_segment_oob",
            ops.gate4_delta_replace_packed_from_cuts_cpu,
            _cut_args(bad_start),
        ),
        **_assert_op_rejects(
            "gate4_delta_replace_from_cuts_rejects_active_mismatch",
            ops.gate4_delta_replace_from_cuts_cpu,
            _cut_args(bad_owner),
        ),
        **_assert_op_rejects(
            "gate4_delta_replace_packed_from_cuts_rejects_active_mismatch",
            ops.gate4_delta_replace_packed_from_cuts_cpu,
            _cut_args(bad_owner),
        ),
        **_assert_op_rejects(
            "gate4_delta_replace_from_cuts_rejects_boundary_other_oob",
            ops.gate4_delta_replace_from_cuts_cpu,
            _cut_args(bad_boundary_other),
        ),
        **_assert_op_rejects(
            "gate4_delta_replace_packed_from_cuts_rejects_boundary_other_oob",
            ops.gate4_delta_replace_packed_from_cuts_cpu,
            _cut_args(bad_boundary_other),
        ),
        **_assert_cut_rejects(ops, "nan_depth", bad_nan_depth),
        **_assert_cut_rejects(ops, "decreasing_depth", bad_decreasing_depth),
        **_assert_cut_rejects(ops, "bad_first_sentinel", bad_first_sentinel),
        **_assert_cut_rejects(ops, "bad_last_sentinel", bad_last_sentinel),
        **_assert_cut_rejects(ops, "internal_boundary_id_oob", bad_internal_id),
        **_assert_cut_rejects(ops, "single_cut_row", bad_single_cut),
    }


def _assert_cut_rejects(ops: Any, name: str, fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        **_assert_op_rejects(
            f"gate4_delta_replace_from_cuts_rejects_{name}",
            ops.gate4_delta_replace_from_cuts_cpu,
            _cut_args(fixture),
        ),
        **_assert_op_rejects(
            f"gate4_delta_replace_packed_from_cuts_rejects_{name}",
            ops.gate4_delta_replace_packed_from_cuts_cpu,
            _cut_args(fixture),
        ),
    }


def _assert_sorted_rejects(ops: Any, name: str, fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        **_assert_op_rejects(
            f"gate4_delta_replace_from_sorted_rejects_{name}",
            ops.gate4_delta_replace_from_sorted_cpu,
            _sorted_args(fixture),
        ),
        **_assert_op_rejects(
            f"gate4_delta_replace_packed_from_sorted_rejects_{name}",
            ops.gate4_delta_replace_packed_from_sorted_cpu,
            _sorted_args(fixture),
        ),
    }


def _assert_sorted_cut_array_rejects(ops: Any, name: str, fixture: dict[str, Any]) -> dict[str, Any]:
    return _assert_op_rejects(
        f"gate4_cut_arrays_from_sorted_rejects_{name}",
        ops.gate4_cut_arrays_from_sorted_cpu,
        _sorted_cut_array_args(fixture),
    )


def _verify_sorted_rejects_bad_contracts(ops: Any) -> dict[str, Any]:
    bad_active = _fixture()
    bad_active["row_active"] = torch.tensor([2], dtype=torch.int64)
    bad_count = _fixture()
    bad_count["valid_counts"] = torch.tensor([[2, 1]], dtype=torch.int64)
    bad_negative_id = _fixture()
    bad_negative_id["sorted_ids"] = torch.tensor([[[-1, 0]]], dtype=torch.int64)
    bad_oob_id = _fixture()
    bad_oob_id["sorted_ids"] = torch.tensor([[[1, 0]]], dtype=torch.int64)
    bad_boundary_other = _fixture()
    bad_boundary_other["boundary_other"] = torch.tensor([[1], [2]], dtype=torch.int64)
    bad_nan_depth = _fixture()
    bad_nan_depth["sorted_depths"] = torch.tensor([[[float("nan"), 0.5]]], dtype=torch.float64)
    bad_below_near = _fixture()
    bad_below_near["sorted_depths"] = torch.tensor([[[-0.1, 0.5]]], dtype=torch.float64)
    bad_above_far = _fixture()
    bad_above_far["sorted_depths"] = torch.tensor([[[1.1, 0.5]]], dtype=torch.float64)
    bad_decreasing = _two_candidate_sorted_fixture()
    bad_decreasing["sorted_depths"] = torch.tensor([[[0.6, 0.6], [0.4, 0.4]]], dtype=torch.float64)
    return {
        **_assert_sorted_rejects(ops, "row_active_bad_value", bad_active),
        **_assert_sorted_rejects(ops, "valid_count_oob", bad_count),
        **_assert_sorted_rejects(ops, "negative_boundary_id", bad_negative_id),
        **_assert_sorted_rejects(ops, "boundary_id_oob", bad_oob_id),
        **_assert_sorted_rejects(ops, "boundary_other_oob", bad_boundary_other),
        **_assert_sorted_rejects(ops, "nan_depth", bad_nan_depth),
        **_assert_sorted_rejects(ops, "below_near_depth", bad_below_near),
        **_assert_sorted_rejects(ops, "above_far_depth", bad_above_far),
        **_assert_sorted_rejects(ops, "decreasing_depth", bad_decreasing),
        **_assert_sorted_cut_array_rejects(ops, "row_active_bad_value", bad_active),
        **_assert_sorted_cut_array_rejects(ops, "valid_count_oob", bad_count),
        **_assert_sorted_cut_array_rejects(ops, "negative_boundary_id", bad_negative_id),
        **_assert_sorted_cut_array_rejects(ops, "nan_depth", bad_nan_depth),
        **_assert_sorted_cut_array_rejects(ops, "below_near_depth", bad_below_near),
        **_assert_sorted_cut_array_rejects(ops, "above_far_depth", bad_above_far),
        **_assert_sorted_cut_array_rejects(ops, "decreasing_depth", bad_decreasing),
    }


def _sorted_cut_array_args(fixture: dict[str, Any]) -> tuple[Any, ...]:
    args = (
        fixture["sorted_depths"],
        fixture["sorted_ids"],
        fixture["valid_counts"],
        fixture["row_active"],
        fixture["ray_coeff"],
        fixture["frame_t"],
        fixture["site_xyz"],
        fixture["site_t"],
        fixture["site_weight"],
        2,
        0.0,
        1.0,
        1.0e-6,
        1.0e-8,
    )
    return args


def _sorted_args(fixture: dict[str, Any]) -> tuple[Any, ...]:
    args = (
        fixture["sorted_depths"],
        fixture["sorted_ids"],
        fixture["valid_counts"],
        fixture["row_active"],
        fixture["ray_coeff"],
        fixture["frame_t"],
        fixture["site_xyz"],
        fixture["site_t"],
        fixture["site_weight"],
        fixture["boundary_other"],
        2,
        0.0,
        1.0,
        1.0e-6,
        1.0e-8,
    )
    return args


def _coeff_csr_args(fixture: dict[str, Any]) -> tuple[Any, ...]:
    sorted_depths = fixture["sorted_depths"]
    sorted_ids = fixture["sorted_ids"]
    valid_count = int(fixture["valid_counts"][0, 0].item())
    candidate_ids = sorted_ids[0, :valid_count, 0].contiguous().to(dtype=torch.int64)
    candidate_depth_coeffs = torch.stack(
        (
            sorted_depths[0, :valid_count, 0],
            torch.zeros((valid_count,), dtype=torch.float64),
            torch.ones((valid_count,), dtype=torch.float64),
            torch.zeros((valid_count,), dtype=torch.float64),
        ),
        dim=1,
    ).contiguous()
    args = (
        torch.tensor([0, valid_count], dtype=torch.int64),
        candidate_ids,
        candidate_depth_coeffs,
        torch.tensor([0], dtype=torch.int64),
        fixture["ray_coeff"],
        fixture["frame_t"],
        fixture["site_xyz"],
        fixture["site_t"],
        fixture["site_weight"],
        fixture["boundary_other"],
        2,
        0.0,
        1.0,
        1.0e-6,
        1.0e-6,
        1.0e-8,
    )
    return args


def _verify_sorted_fixture(ops: Any) -> dict[str, Any]:
    cut_arrays = ops.gate4_cut_arrays_from_sorted_cpu(*_sorted_cut_array_args(_fixture()))
    args = _sorted_args(_fixture())
    unpacked = ops.gate4_delta_replace_from_sorted_cpu(*args)
    packed = ops.gate4_delta_replace_packed_from_sorted_cpu(*args)
    _assert_packed_matches_unpacked(ops, unpacked, packed)
    changing_args = _sorted_args(_changing_sorted_fixture())
    changing_unpacked = ops.gate4_delta_replace_from_sorted_cpu(*changing_args)
    changing_packed = ops.gate4_delta_replace_packed_from_sorted_cpu(*changing_args)
    _assert_packed_matches_unpacked(ops, changing_unpacked, changing_packed)
    _assert_expected_change_records(changing_packed)
    return {
        **_tensor_result("base_record_i32", packed[4]),
        **_tensor_result("change_record_i32", packed[11]),
        **_tensor_result("base_offsets_i32", packed[0]),
        **_tensor_result("track_change_offsets_i32", packed[5]),
        **_tensor_result("cut_array_cut_ids_i64", cut_arrays[1]),
        **_tensor_result("cut_array_cut_offsets_i64", cut_arrays[2]),
        **_tensor_result("cut_array_start_segments_i64", cut_arrays[3]),
        **_tensor_result("cut_array_initial_owner_i64", cut_arrays[4]),
        **_tensor_result("changing_sorted_change_frame_i32", changing_packed[6]),
        **_tensor_result("changing_sorted_change_record_i32", changing_packed[11]),
        **_tensor_result("changing_sorted_change_offsets_i32", changing_packed[7]),
        **_tensor_result("changing_sorted_track_change_offsets_i32", changing_packed[5]),
    }


def _verify_coeff_csr_fixture(ops: Any) -> dict[str, Any]:
    packed = ops.gate4_delta_replace_packed_from_coeff_csr_cpu(*_coeff_csr_args(_fixture()))
    sorted_packed = ops.gate4_delta_replace_packed_from_sorted_cpu(*_sorted_args(_fixture()))
    changing_packed = ops.gate4_delta_replace_packed_from_coeff_csr_cpu(
        *_coeff_csr_args(_changing_sorted_fixture())
    )
    changing_sorted_packed = ops.gate4_delta_replace_packed_from_sorted_cpu(
        *_sorted_args(_changing_sorted_fixture())
    )
    for direct_tensor, sorted_tensor in zip(packed, sorted_packed):
        torch.testing.assert_close(direct_tensor, sorted_tensor)
    for direct_tensor, sorted_tensor in zip(changing_packed, changing_sorted_packed):
        torch.testing.assert_close(direct_tensor, sorted_tensor)
    _assert_expected_change_records(changing_packed)
    return {
        **_tensor_result("direct_csr_base_record_i32", packed[4]),
        **_tensor_result("direct_csr_change_record_i32", packed[11]),
        **_tensor_result("direct_csr_base_offsets_i32", packed[0]),
        **_tensor_result("direct_csr_track_change_offsets_i32", packed[5]),
        **_tensor_result("changing_direct_csr_change_frame_i32", changing_packed[6]),
        **_tensor_result("changing_direct_csr_change_record_i32", changing_packed[11]),
        **_tensor_result("changing_direct_csr_change_offsets_i32", changing_packed[7]),
        **_tensor_result("changing_direct_csr_track_change_offsets_i32", changing_packed[5]),
    }


def _cut_args(fixture: dict[str, Any]) -> tuple[Any, ...]:
    args = (
        fixture["cut_depths"],
        fixture["cut_ids"],
        fixture["cut_offsets"],
        fixture["start_segments"],
        fixture["initial_owner"],
        fixture["boundary_other"],
        2,
        1.0e-8,
    )
    return args


def _verify_cut_fixture(ops: Any) -> dict[str, Any]:
    args = _cut_args(_cut_fixture())
    unpacked = ops.gate4_delta_replace_from_cuts_cpu(*args)
    packed = ops.gate4_delta_replace_packed_from_cuts_cpu(*args)
    _assert_packed_matches_unpacked(ops, unpacked, packed)
    changing_args = _cut_args(_changing_cut_fixture())
    changing_unpacked = ops.gate4_delta_replace_from_cuts_cpu(*changing_args)
    changing_packed = ops.gate4_delta_replace_packed_from_cuts_cpu(*changing_args)
    _assert_packed_matches_unpacked(ops, changing_unpacked, changing_packed)
    _assert_expected_change_records(changing_packed)
    return {
        **_tensor_result("cut_base_record_i32", packed[4]),
        **_tensor_result("cut_change_record_i32", packed[11]),
        **_tensor_result("cut_base_offsets_i32", packed[0]),
        **_tensor_result("cut_track_change_offsets_i32", packed[5]),
        **_tensor_result("changing_cut_change_frame_i32", changing_packed[6]),
        **_tensor_result("changing_cut_change_record_i32", changing_packed[11]),
        **_tensor_result("changing_cut_change_offsets_i32", changing_packed[7]),
        **_tensor_result("changing_cut_track_change_offsets_i32", changing_packed[5]),
    }


def verify() -> dict[str, Any]:
    _ensure_variant_imported()
    ops = torch.ops.world_foam_lane2_fused_slab_v0
    required = (
        "gate4_delta_replace_from_cuts_cpu",
        "gate4_delta_replace_packed_from_cuts_cpu",
        "gate4_delta_replace_from_sorted_cpu",
        "gate4_delta_replace_packed_from_sorted_cpu",
        "gate4_delta_replace_packed_from_coeff_csr_cpu",
        "gate4_cut_arrays_from_sorted_cpu",
        "pack_endpoint_records_i32_cpu",
        "endpoint_record_delta_replace_coeff16_packed_framegroup16_launch_only_mse_vjp_direct_atomic_rgb_only",
        "endpoint_record_delta_replace_coeff16_packed_framegroup16_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only",
        "endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_launch_only_mse_vjp_direct_atomic_rgb_only",
        "endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only",
        "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_launch_only_mse_vjp_direct_atomic_rgb_only",
        "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only",
        "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_launch_only_mse_vjp_direct_atomic_rgb_only",
        "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only",
        "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_launch_only_mse_vjp_direct_atomic_rgb_only",
        "endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only",
        "endpoint_record_delta_replace_coeff16_packed_framegroup16_recompute_launch_only_mse_vjp_direct_atomic_rgb_only",
        "endpoint_record_delta_replace_coeff16_packed_framegroup16_smallrun16_launch_only_mse_vjp_direct_atomic_rgb_only",
        "endpoint_record_delta_replace_coeff16_packed_framegroup16_materialized_launch_only_mse_vjp_direct_atomic_rgb_only",
        "fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only",
        "fused_slab_affine_num32_den16_mse_vjp_direct_atomic_track_rgb_only",
    )
    missing = [name for name in required if not hasattr(ops, name)]
    if missing:
        raise RuntimeError(f"missing native WorldFoam ops: {missing}")

    sorted_result = _verify_sorted_fixture(ops)
    coeff_csr_result = _verify_coeff_csr_fixture(ops)
    cut_result = _verify_cut_fixture(ops)
    reject_result = _verify_pack_rejects_bad_contracts(ops)
    cut_reject_result = _verify_cut_rejects_bad_contracts(ops)
    sorted_reject_result = _verify_sorted_rejects_bad_contracts(ops)
    return {
        "status": "ok",
        "variant_root": str(VARIANT_ROOT),
        "has_launch_only_packed_framegroup16_op": True,
        "has_launch_only_packed_framegroup16_unchecked_op": True,
        "has_launch_only_packed_framegroup16_reduce32_op": True,
        "has_launch_only_packed_framegroup16_reduce32_unchecked_op": True,
        "has_launch_only_packed_framegroup16_rowselect32_op": True,
        "has_launch_only_packed_framegroup16_rowselect32_unchecked_op": True,
        "has_launch_only_packed_framegroup16_rowdesc_op": True,
        "has_launch_only_packed_framegroup16_rowdesc_unchecked_op": True,
        "has_launch_only_packed_framegroup16_rowdesc32_op": True,
        "has_launch_only_packed_framegroup16_rowdesc32_unchecked_op": True,
        "has_launch_only_packed_framegroup16_recompute_op": True,
        "has_launch_only_packed_framegroup16_smallrun16_op": True,
        "has_launch_only_packed_framegroup16_materialized_op": True,
        "has_affine_candidate_num32_den16_fused_mse_op": True,
        "has_affine_candidate_num32_den16_track_fused_mse_op": True,
        **sorted_result,
        **coeff_csr_result,
        **cut_result,
        **reject_result,
        **cut_reject_result,
        **sorted_reject_result,
    }


def main() -> int:
    print(json.dumps(verify(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
