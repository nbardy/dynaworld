from __future__ import annotations

import torch

from star_uvt_feature_targets import _adapt_render_to_feature_target
from star_uvt_sparse_grid import (
    _pack_sparse_target_grid_vjp,
    _sparse_feature_values_to_target_grid,
    _sparse_target_grid_pixel_ids,
)


def _gather_sparse_feature_values(rendered: torch.Tensor, pixel_ids: torch.Tensor) -> torch.Tensor:
    return (
        rendered.permute(0, 2, 3, 1)
        .reshape(-1, int(rendered.shape[1]))
        .index_select(0, pixel_ids.to(torch.int64))
        .contiguous()
    )


def test_sparse_target_grid_forward_matches_dense_adapter() -> None:
    torch.manual_seed(71)
    rendered = torch.randn(2, 4, 6, 8)
    target_shape = (3, 4, 3, 5)
    dense = _adapt_render_to_feature_target(rendered, target_shape=target_shape, mode="trilinear")
    pixel_ids = _sparse_target_grid_pixel_ids(
        input_shape=tuple(int(item) for item in rendered.shape),
        target_shape=target_shape,
        mode="trilinear",
        device=rendered.device,
    )
    sparse_values = _gather_sparse_feature_values(rendered, pixel_ids)

    sparse = _sparse_feature_values_to_target_grid(
        sparse_values,
        input_shape=tuple(int(item) for item in rendered.shape),
        target_shape=target_shape,
        mode="trilinear",
    )

    torch.testing.assert_close(sparse, dense, rtol=1.0e-5, atol=1.0e-6)


def test_sparse_target_grid_vjp_matches_dense_adapter_backward() -> None:
    torch.manual_seed(73)
    rendered = torch.randn(2, 4, 6, 8, requires_grad=True)
    target_shape = (3, 4, 3, 5)
    dense = _adapt_render_to_feature_target(rendered, target_shape=target_shape, mode="trilinear")
    grad_target = torch.randn_like(dense)
    dense.backward(grad_target)
    expected_grad = rendered.grad.detach()

    pack = _pack_sparse_target_grid_vjp(
        grad_target,
        input_shape=tuple(int(item) for item in rendered.shape),
        mode="trilinear",
    )
    flat_grad = torch.zeros(
        (int(rendered.shape[0]) * int(rendered.shape[2]) * int(rendered.shape[3]), int(rendered.shape[1])),
        dtype=rendered.dtype,
    )
    flat_grad.index_add_(0, pack.pixel_ids.to(torch.int64), pack.grad_feature_values)
    sparse_grad = flat_grad.reshape(
        int(rendered.shape[0]),
        int(rendered.shape[2]),
        int(rendered.shape[3]),
        int(rendered.shape[1]),
    ).permute(0, 3, 1, 2)

    torch.testing.assert_close(sparse_grad, expected_grad, rtol=1.0e-5, atol=1.0e-6)
    assert pack.grad_alpha_values.shape == (pack.pixel_count,)
    assert float(pack.grad_alpha_values.abs().sum()) == 0.0
