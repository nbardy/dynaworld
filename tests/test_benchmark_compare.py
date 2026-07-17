from __future__ import annotations

import torch

from benchmark_compare import grad_diff_stats, max_tensor_diff, tensor_diff_stats


def test_tensor_diff_stats_reports_exact_tensor_difference() -> None:
    diff = tensor_diff_stats(torch.tensor([1.0, 3.0]), torch.tensor([2.0, 1.0]))
    assert diff["both_none"] is False
    assert diff["max_abs"] == 2.0
    assert diff["mean_abs"] == 1.5
    assert diff["shape"] == [2]


def test_tensor_diff_stats_handles_none_and_shape_mismatch() -> None:
    assert tensor_diff_stats(None, None)["both_none"] is True
    one_none = tensor_diff_stats(torch.ones(1), None)
    assert one_none["both_none"] is False
    assert one_none["max_abs"] is None

    mismatch = tensor_diff_stats(torch.ones(1), torch.ones(2))
    assert mismatch["shape_mismatch"] == [[1], [2]]
    assert mismatch["max_abs"] is None


def test_grad_diff_stats_and_max_tensor_diff_share_schema() -> None:
    diff = grad_diff_stats(
        {"a": torch.tensor([1.0]), "b": None},
        {"a": torch.tensor([3.5]), "c": torch.tensor([0.0])},
    )
    assert sorted(diff) == ["a", "b", "c"]
    assert diff["a"]["max_abs"] == 2.5
    assert max_tensor_diff(diff) == 2.5
