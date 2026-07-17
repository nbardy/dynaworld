from __future__ import annotations

import pytest
import torch

from colorize import FeatureToColor
from star_uvt_checkpoints import (
    load_star_training_checkpoint as _load_training_checkpoint,
    save_star_training_checkpoint as _save_training_checkpoint,
)
from star_uvt_feature_targets import (
    FeatureTargetTensor,
    _adapt_feature_target_grid,
    _adapt_feature_target_grid_chunk,
    _adapt_rgb_to_grid,
    _adapt_render_to_feature_target,
    _feature_target_channel_stats,
    _normalize_feature_target,
    _normalize_feature_target_with_stats,
    _upsample_grid_rgb,
)
from star_uvt_feature_losses import _trainable_colorizer_grid_loss_and_grid_grad
from star_uvt_common import load_colorizer_init_checkpoint as _load_colorizer_init_checkpoint
from star_uvt_schedules import (
    _feature_target_weight_schedule,
    _feature_target_weights_for_step,
)
from star_uvt_sparse_visual_sampling import (
    _sparse_visual_local_frame_ids_for_chunk,
    _sparse_visual_loss_sample_count,
    _sparse_visual_patch_phase_for_step,
    _sparse_visual_pixel_ids_for_chunk,
)
from star_uvt_sparse_visual_losses import (
    _gelu_fast_sigmoid_grad,
    _sparse_visual_alpha_loss_and_grad,
    _sparse_visual_black_hole_loss_and_grad,
    _sparse_visual_rgb_loss_and_grads,
)
from star_uvt_visibility_support import (
    _apply_support_birth_split,
    _support_birth_split_repair_tile_overflow_ids,
    _support_birth_split_sample_target_grid_features,
    _support_birth_split_sampled_tile_load,
    _support_birth_split_set_tube_opacity,
    _support_birth_split_target_patch_pixel_ids_for_chunk,
    _support_birth_split_target_pixel_ids_for_chunk,
    _support_birth_split_target_points,
    _support_birth_split_tube_counts,
    _visibility_proxy_loss,
)
from star_uvt_feature_config import resolve_config
from star_uvt_feature_tube_model import (
    FeatureScreenTimeTubeModel,
    FeatureTubeRenderConfig,
)


def test_chunked_feature_target_grid_matches_dense_interpolate() -> None:
    torch.manual_seed(7)
    source = torch.randn(4, 5, 3, 3)

    for mode in ("trilinear", "nearest"):
        dense = _adapt_feature_target_grid(source, frames=7, height=6, width=5, mode=mode)
        chunked = torch.cat(
            [
                _adapt_feature_target_grid_chunk(
                    source,
                    frames=7,
                    height=6,
                    width=5,
                    frame_start=frame_start,
                    chunk_frames=min(2, 7 - frame_start),
                    mode=mode,
                )
                for frame_start in range(0, 7, 2)
            ],
            dim=0,
        )

        torch.testing.assert_close(chunked, dense, rtol=1.0e-5, atol=1.0e-6)


def test_streaming_feature_target_standardization_matches_dense() -> None:
    torch.manual_seed(11)
    source = torch.randn(4, 5, 3, 3)

    for mode in ("trilinear", "nearest"):
        dense = _adapt_feature_target_grid(source, frames=7, height=6, width=5, mode=mode)
        dense_normalized = _normalize_feature_target(dense, "channel_standardize")
        mean, std = _feature_target_channel_stats(
            source,
            frames=7,
            height=6,
            width=5,
            grid_mode=mode,
            chunk_size=2,
        )
        chunked_normalized = torch.cat(
            [
                _normalize_feature_target_with_stats(
                    _adapt_feature_target_grid_chunk(
                        source,
                        frames=7,
                        height=6,
                        width=5,
                        frame_start=frame_start,
                        chunk_frames=min(2, 7 - frame_start),
                        mode=mode,
                    ),
                    mode="channel_standardize",
                    mean=mean,
                    std=std,
                )
                for frame_start in range(0, 7, 2)
            ],
            dim=0,
        )

        torch.testing.assert_close(chunked_normalized, dense_normalized, rtol=1.0e-4, atol=1.0e-5)


def test_cached_feature_target_chunks_match_streaming_chunks() -> None:
    torch.manual_seed(17)
    source = torch.randn(5, 7, 4, 3)
    frames = 9
    height = 8
    width = 6
    chunk_size = 2
    mean, std = _feature_target_channel_stats(
        source,
        frames=frames,
        height=height,
        width=width,
        grid_mode="trilinear",
        chunk_size=chunk_size,
    )
    chunks = tuple(
        _normalize_feature_target_with_stats(
            _adapt_feature_target_grid_chunk(
                source,
                frames=frames,
                height=height,
                width=width,
                frame_start=frame_start,
                chunk_frames=min(chunk_size, frames - frame_start),
                mode="trilinear",
            ),
            mode="channel_standardize",
            mean=mean,
            std=std,
        ).detach()
        for frame_start in range(0, frames, chunk_size)
    )
    target = FeatureTargetTensor(
        materialization="cached_chunks",
        frames=frames,
        height=height,
        width=width,
        feature_dim=7,
        grid_mode="trilinear",
        normalization="channel_standardize",
        dense=None,
        source=None,
        chunks=chunks,
        chunk_size=chunk_size,
        mean=mean,
        std=std,
        meta={},
    )

    cached = torch.cat(
        [
            target.chunk(frame_start, min(chunk_size, frames - frame_start))
            for frame_start in range(0, frames, chunk_size)
        ],
        dim=0,
    )
    dense = _normalize_feature_target(
        _adapt_feature_target_grid(source, frames=frames, height=height, width=width, mode="trilinear"),
        "channel_standardize",
    )

    torch.testing.assert_close(cached, dense, rtol=1.0e-4, atol=1.0e-5)


def test_target_grid_feature_target_chunks_and_render_adapter() -> None:
    torch.manual_seed(23)
    source = torch.randn(4, 6, 3, 2)
    target = FeatureTargetTensor(
        materialization="target_grid",
        frames=8,
        height=12,
        width=10,
        feature_dim=6,
        grid_mode="trilinear",
        normalization="channel_standardize",
        dense=None,
        source=source,
        chunks=None,
        chunk_size=None,
        mean=None,
        std=None,
        meta={},
    )
    first = target.chunk(0, 2)
    second = target.chunk(2, 2)
    rendered = torch.randn(2, 6, 12, 10, requires_grad=True)
    adapted = _adapt_render_to_feature_target(
        rendered,
        target_shape=tuple(int(item) for item in first.shape),
        mode="trilinear",
    )
    loss = (adapted - first).square().mean()
    loss.backward()

    assert target.numel == source.numel()
    torch.testing.assert_close(first, source[0:1])
    torch.testing.assert_close(second, source[1:2])
    assert list(adapted.shape) == list(first.shape)
    assert rendered.grad is not None
    assert float(rendered.grad.abs().sum()) > 0.0


def test_rgb_probe_grid_adapters_keep_shapes_and_gradients() -> None:
    frames_rgb = torch.rand(8, 3, 12, 10, requires_grad=True)
    grid = _adapt_rgb_to_grid(frames_rgb, target_shape=(4, 3, 2), mode="trilinear")
    restored = _upsample_grid_rgb(grid, target_shape=(8, 12, 10), mode="trilinear")
    loss = restored.square().mean()
    loss.backward()

    assert list(grid.shape) == [4, 3, 3, 2]
    assert list(restored.shape) == [8, 3, 12, 10]
    assert frames_rgb.grad is not None
    assert float(frames_rgb.grad.abs().sum()) > 0.0


def test_trainable_colorizer_grid_loss_matches_autograd() -> None:
    torch.manual_seed(37)
    colorizer = FeatureToColor(feature_dim=4, hidden_dim=5, activation="sigmoid", pre_norm=False)
    grid = torch.randn(3, 4, 6, 5)
    target = torch.rand(3, 3, 6, 5)
    loss_weight = 7.0

    expected_grid = grid.detach().clone().requires_grad_(True)
    expected_loss = (colorizer(expected_grid) - target).square().sum() / float(target.numel())
    (loss_weight * expected_loss).backward()
    expected_grid_grad = expected_grid.grad.detach().clone()
    expected_param_grads = [
        param.grad.detach().clone() if param.grad is not None else None
        for param in colorizer.parameters()
    ]
    for param in colorizer.parameters():
        param.grad = None

    loss, grid_grad = _trainable_colorizer_grid_loss_and_grid_grad(
        colorizer,
        grid,
        target,
        total_rgb_grid_loss_elems=target.numel(),
        loss_weight=loss_weight,
    )

    torch.testing.assert_close(loss, expected_loss.detach())
    torch.testing.assert_close(grid_grad, expected_grid_grad)
    for param, expected_grad in zip(colorizer.parameters(), expected_param_grads, strict=True):
        assert expected_grad is not None
        assert param.grad is not None
        torch.testing.assert_close(param.grad, expected_grad)


def test_sparse_visual_pixel_ids_are_chunk_local_and_empty_int32() -> None:
    ids = _sparse_visual_pixel_ids_for_chunk(
        pixel_source="stratified_grid",
        chunk_frames=2,
        height=8,
        width=8,
        render_frames=4,
        frame_start=2,
        sample_grid_shape=(4, 2, 2),
        device=torch.device("cpu"),
    )
    empty = _sparse_visual_pixel_ids_for_chunk(
        pixel_source="stratified_grid",
        chunk_frames=1,
        height=8,
        width=8,
        render_frames=4,
        frame_start=0,
        sample_grid_shape=(2, 2, 2),
        device=torch.device("cpu"),
    )

    assert ids.tolist() == [18, 22, 50, 54, 82, 86, 114, 118]
    assert empty.dtype == torch.int32
    assert empty.numel() == 0


def test_support_birth_split_target_pixel_ids_are_chunk_local_unique_int32() -> None:
    target_points = torch.tensor(
        [
            [1.2, 2.8, -1.0],
            [5.9, 3.5, 0.0],
            [1.2, 2.8, -1.0],
            [-3.0, 9.0, 0.0],
            [0.5, 0.5, 2.0],
        ],
        dtype=torch.float32,
    )

    ids = _support_birth_split_target_pixel_ids_for_chunk(
        target_points,
        frames=5,
        height=4,
        width=6,
        frame_start=1,
        chunk_frames=2,
        device=torch.device("cpu"),
    )
    empty = _support_birth_split_target_pixel_ids_for_chunk(
        target_points,
        frames=5,
        height=4,
        width=6,
        frame_start=3,
        chunk_frames=1,
        device=torch.device("cpu"),
    )

    assert ids.dtype == torch.int32
    assert ids.tolist() == [13, 42, 47]
    assert empty.dtype == torch.int32
    assert empty.numel() == 0


def test_support_birth_split_target_patch_pixel_ids_keep_patch_groups() -> None:
    target_points = torch.tensor(
        [
            [1.2, 2.8, -1.0],
            [5.9, 3.5, 0.0],
            [0.5, 0.5, 2.0],
        ],
        dtype=torch.float32,
    )

    ids, cell_count = _support_birth_split_target_patch_pixel_ids_for_chunk(
        target_points,
        frames=5,
        height=4,
        width=6,
        frame_start=1,
        chunk_frames=2,
        patch_shape=(2, 3),
        device=torch.device("cpu"),
    )
    empty, empty_cell_count = _support_birth_split_target_patch_pixel_ids_for_chunk(
        target_points,
        frames=5,
        height=4,
        width=6,
        frame_start=3,
        chunk_frames=1,
        patch_shape=(2, 3),
        device=torch.device("cpu"),
    )

    assert ids.dtype == torch.int32
    assert cell_count == 2
    assert ids.tolist() == [
        6,
        7,
        8,
        12,
        13,
        14,
        39,
        40,
        41,
        45,
        46,
        47,
    ]
    assert empty.dtype == torch.int32
    assert empty.numel() == 0
    assert empty_cell_count == 0


def test_sparse_visual_patch_grid_uses_local_contiguous_support() -> None:
    ids = _sparse_visual_pixel_ids_for_chunk(
        pixel_source="stratified_patch_grid",
        chunk_frames=2,
        height=8,
        width=8,
        render_frames=4,
        frame_start=2,
        sample_grid_shape=(4, 2, 2),
        patch_shape=(2, 2),
        device=torch.device("cpu"),
    )

    assert ids.dtype == torch.int32
    assert ids.numel() == 32
    assert ids.unique().numel() == ids.numel()
    assert ids[:16].tolist() == [
        9,
        10,
        13,
        14,
        17,
        18,
        21,
        22,
        41,
        42,
        45,
        46,
        49,
        50,
        53,
        54,
    ]
    assert ids[16:].tolist() == [value + 64 for value in ids[:16].tolist()]


def test_sparse_visual_phase_patch_grid_cycles_within_cells() -> None:
    phase0 = _sparse_visual_pixel_ids_for_chunk(
        pixel_source="stratified_patch_grid_phase",
        chunk_frames=1,
        height=8,
        width=8,
        render_frames=1,
        frame_start=0,
        sample_grid_shape=(1, 2, 2),
        patch_shape=(2, 2),
        patch_phase=(0, 0),
        patch_phase_shape=(2, 2),
        device=torch.device("cpu"),
    )
    phase1 = _sparse_visual_pixel_ids_for_chunk(
        pixel_source="stratified_patch_grid_phase",
        chunk_frames=1,
        height=8,
        width=8,
        render_frames=1,
        frame_start=0,
        sample_grid_shape=(1, 2, 2),
        patch_shape=(2, 2),
        patch_phase=(0, 1),
        patch_phase_shape=(2, 2),
        device=torch.device("cpu"),
    )

    assert phase0.dtype == torch.int32
    assert phase0.numel() == phase1.numel() == 16
    assert phase0.unique().numel() == 16
    assert phase1.unique().numel() == 16
    assert phase0.tolist() == [0, 1, 4, 5, 8, 9, 12, 13, 32, 33, 36, 37, 40, 41, 44, 45]
    assert phase1.tolist() == [2, 3, 6, 7, 10, 11, 14, 15, 34, 35, 38, 39, 42, 43, 46, 47]
    assert set(phase0.tolist()).isdisjoint(set(phase1.tolist()))


def test_sparse_visual_patch_phase_for_step_cycles_row_major() -> None:
    assert _sparse_visual_patch_phase_for_step(
        pixel_source="stratified_patch_grid",
        global_step=7,
        patch_phase_shape=(2, 2),
    ) == (0, 0)
    assert _sparse_visual_patch_phase_for_step(
        pixel_source="stratified_patch_grid_phase",
        global_step=0,
        patch_phase_shape=(2, 2),
    ) == (0, 0)
    assert _sparse_visual_patch_phase_for_step(
        pixel_source="stratified_patch_grid_phase",
        global_step=3,
        patch_phase_shape=(2, 2),
    ) == (1, 1)
    assert _sparse_visual_patch_phase_for_step(
        pixel_source="stratified_patch_grid_phase",
        global_step=4,
        patch_phase_shape=(2, 2),
    ) == (0, 0)


def test_sparse_visual_rgb_loss_scales_local_grads_and_colorizer_grads() -> None:
    colorizer = torch.nn.Conv2d(2, 3, kernel_size=1, bias=False)
    with torch.no_grad():
        colorizer.weight.zero_()
        colorizer.weight[0, 0, 0, 0] = 1.0
        colorizer.weight[1, 1, 0, 0] = 1.0
    feature_values = torch.tensor([[0.25, 0.5], [1.0, 0.0]], dtype=torch.float32)
    alpha_values = torch.tensor([0.5, 0.25], dtype=torch.float32)
    target_values = torch.zeros((2, 3), dtype=torch.float32)

    loss, grad_feature, grad_alpha = _sparse_visual_rgb_loss_and_grads(
        feature_values,
        alpha_values,
        target_values,
        colorizer,  # type: ignore[arg-type]
        total_loss_elems=6,
        loss_weight=3.0,
    )

    assert loss.item() > 0.0
    assert grad_feature.shape == feature_values.shape
    assert grad_alpha.shape == alpha_values.shape
    assert colorizer.weight.grad is not None
    assert float(colorizer.weight.grad.abs().sum()) > 0.0


def test_sparse_visual_alpha_loss_pushes_sampled_alpha_to_target() -> None:
    alpha_values = torch.tensor([0.25, 0.75], dtype=torch.float32)

    loss, grad_alpha = _sparse_visual_alpha_loss_and_grad(
        alpha_values,
        target=1.0,
        total_loss_elems=2,
        loss_weight=4.0,
    )

    assert loss.item() == pytest.approx(0.3125)
    torch.testing.assert_close(
        grad_alpha,
        torch.tensor([-3.0, -1.0], dtype=torch.float32),
    )


def test_sparse_visual_target_background_composition_removes_empty_pixel_penalty() -> None:
    colorizer = torch.nn.Conv2d(2, 3, kernel_size=1, bias=False)
    with torch.no_grad():
        colorizer.weight.zero_()
    feature_values = torch.zeros((2, 2), dtype=torch.float32)
    alpha_values = torch.zeros(2, dtype=torch.float32)
    target_values = torch.ones((2, 3), dtype=torch.float32)

    black_loss, black_grad_feature, black_grad_alpha = _sparse_visual_rgb_loss_and_grads(
        feature_values,
        alpha_values,
        target_values,
        colorizer,  # type: ignore[arg-type]
        total_loss_elems=6,
        loss_weight=1.0,
        composition="black",
    )
    target_bg_loss, target_bg_grad_feature, target_bg_grad_alpha = _sparse_visual_rgb_loss_and_grads(
        feature_values,
        alpha_values,
        target_values,
        colorizer,  # type: ignore[arg-type]
        total_loss_elems=6,
        loss_weight=1.0,
        composition="target_background",
    )

    assert black_loss.item() == pytest.approx(1.0)
    assert target_bg_loss.item() == pytest.approx(0.0)
    assert black_grad_feature.abs().sum().item() == pytest.approx(0.0)
    assert black_grad_alpha.abs().sum().item() == pytest.approx(0.0)
    assert target_bg_grad_feature.abs().sum().item() == pytest.approx(0.0)
    assert target_bg_grad_alpha.abs().sum().item() == pytest.approx(0.0)


def test_sparse_visual_black_hole_loss_weights_empty_alpha_by_target_energy() -> None:
    alpha_values = torch.tensor([0.25, 0.75], dtype=torch.float32)
    target_values = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    loss, grad_alpha = _sparse_visual_black_hole_loss_and_grad(
        alpha_values,
        target_values,
        total_loss_elems=2,
        loss_weight=4.0,
        loss_basis="pixel",
    )

    assert loss.item() == pytest.approx(0.28125)
    torch.testing.assert_close(
        grad_alpha,
        torch.tensor([-3.0, -0.0], dtype=torch.float32),
    )


def test_sparse_visual_patch_mean_loss_pools_local_patch_basis() -> None:
    colorizer = torch.nn.Conv2d(2, 3, kernel_size=1, bias=False)
    with torch.no_grad():
        colorizer.weight.zero_()
        colorizer.weight[0, 0, 0, 0] = 1.0
        colorizer.weight[1, 1, 0, 0] = 1.0
    feature_values = torch.tensor(
        [
            [1.0, 0.0],
            [3.0, 0.0],
            [0.0, 2.0],
            [0.0, 4.0],
        ],
        dtype=torch.float32,
    )
    alpha_values = torch.ones(4, dtype=torch.float32)
    target_values = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 3.0, 0.0],
        ],
        dtype=torch.float32,
    )

    loss, grad_feature, grad_alpha = _sparse_visual_rgb_loss_and_grads(
        feature_values,
        alpha_values,
        target_values,
        colorizer,  # type: ignore[arg-type]
        total_loss_elems=3,
        loss_weight=1.0,
        loss_basis="patch_mean",
        sample_grid_shape=(1, 1, 1),
        patch_shape=(2, 2),
    )

    assert loss.item() == pytest.approx(0.0)
    assert grad_feature.abs().sum().item() == pytest.approx(0.0)
    assert grad_alpha.abs().sum().item() == pytest.approx(0.0)
    assert _sparse_visual_loss_sample_count(4, loss_basis="patch_mean", patch_shape=(2, 2)) == 1


def test_sparse_visual_target_area_mean_uses_dense_downsampled_target() -> None:
    colorizer = torch.nn.Conv2d(2, 3, kernel_size=1, bias=False)
    with torch.no_grad():
        colorizer.weight.zero_()
        colorizer.weight[0, 0, 0, 0] = 1.0
        colorizer.weight[1, 1, 0, 0] = 1.0
    feature_values = torch.tensor(
        [
            [1.0, 0.0],
            [3.0, 0.0],
            [0.0, 2.0],
            [0.0, 4.0],
        ],
        dtype=torch.float32,
    )
    alpha_values = torch.ones(4, dtype=torch.float32)
    target_rgb_chunk = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
    target_rgb_chunk[:, 0] = 1.0
    target_rgb_chunk[:, 1] = 1.5

    loss, grad_feature, grad_alpha = _sparse_visual_rgb_loss_and_grads(
        feature_values,
        alpha_values,
        None,
        colorizer,  # type: ignore[arg-type]
        total_loss_elems=3,
        loss_weight=1.0,
        loss_basis="target_area_mean",
        sample_grid_shape=(1, 1, 1),
        patch_shape=(2, 2),
        target_rgb_chunk=target_rgb_chunk,
        local_frame_ids=torch.tensor([0], dtype=torch.int64),
    )

    assert loss.item() == pytest.approx(0.0)
    assert grad_feature.abs().sum().item() == pytest.approx(0.0)
    assert grad_alpha.abs().sum().item() == pytest.approx(0.0)
    assert _sparse_visual_loss_sample_count(4, loss_basis="target_area_mean", patch_shape=(2, 2)) == 1


def test_sparse_visual_manual_hidden_vjp_matches_autograd_target_area() -> None:
    torch.manual_seed(11)
    feature_values = torch.randn((8, 3), dtype=torch.float32) * 0.2
    alpha_values = torch.linspace(0.2, 0.9, 8, dtype=torch.float32)
    target_rgb_chunk = torch.rand((1, 3, 4, 4), dtype=torch.float32)
    autograd_colorizer = FeatureToColor(
        feature_dim=3,
        hidden_dim=4,
        activation="sigmoid",
        pre_norm=False,
        weight_init="kaiming",
        weight_init_gain=1.0,
    )
    manual_colorizer = FeatureToColor(
        feature_dim=3,
        hidden_dim=4,
        activation="sigmoid",
        pre_norm=False,
        weight_init="kaiming",
        weight_init_gain=1.0,
    )
    manual_colorizer.load_state_dict(autograd_colorizer.state_dict())

    autograd_loss, autograd_feature_grad, autograd_alpha_grad = _sparse_visual_rgb_loss_and_grads(
        feature_values,
        alpha_values,
        None,
        autograd_colorizer,
        total_loss_elems=6,
        loss_weight=2.5,
        loss_basis="target_area_mean",
        sample_grid_shape=(1, 1, 2),
        patch_shape=(2, 2),
        target_rgb_chunk=target_rgb_chunk,
        local_frame_ids=torch.tensor([0], dtype=torch.int64),
    )
    manual_loss, manual_feature_grad, manual_alpha_grad = _sparse_visual_rgb_loss_and_grads(
        feature_values,
        alpha_values,
        None,
        manual_colorizer,
        total_loss_elems=6,
        loss_weight=2.5,
        loss_basis="target_area_mean",
        sample_grid_shape=(1, 1, 2),
        patch_shape=(2, 2),
        target_rgb_chunk=target_rgb_chunk,
        local_frame_ids=torch.tensor([0], dtype=torch.int64),
        vjp_mode="manual_hidden",
    )

    assert manual_loss.item() == pytest.approx(autograd_loss.item(), abs=1.0e-7)
    torch.testing.assert_close(manual_feature_grad, autograd_feature_grad, atol=1.0e-6, rtol=1.0e-5)
    torch.testing.assert_close(manual_alpha_grad, autograd_alpha_grad, atol=1.0e-6, rtol=1.0e-5)
    for (_, autograd_param), (_, manual_param) in zip(
        autograd_colorizer.named_parameters(),
        manual_colorizer.named_parameters(),
        strict=True,
    ):
        assert autograd_param.grad is not None
        assert manual_param.grad is not None
        torch.testing.assert_close(manual_param.grad, autograd_param.grad, atol=1.0e-6, rtol=1.0e-5)


def test_sparse_visual_manual_hidden64_star_only_skips_colorizer_grads() -> None:
    torch.manual_seed(13)
    feature_values = torch.randn((8, 3), dtype=torch.float32) * 0.2
    alpha_values = torch.linspace(0.2, 0.9, 8, dtype=torch.float32)
    target_rgb_chunk = torch.rand((1, 3, 4, 4), dtype=torch.float32)
    joint_colorizer = FeatureToColor(
        feature_dim=3,
        hidden_dim=4,
        activation="sigmoid",
        pre_norm=False,
        weight_init="kaiming",
        weight_init_gain=1.0,
    )
    star_only_colorizer = FeatureToColor(
        feature_dim=3,
        hidden_dim=4,
        activation="sigmoid",
        pre_norm=False,
        weight_init="kaiming",
        weight_init_gain=1.0,
    )
    star_only_colorizer.load_state_dict(joint_colorizer.state_dict())

    joint_loss, joint_feature_grad, joint_alpha_grad = _sparse_visual_rgb_loss_and_grads(
        feature_values,
        alpha_values,
        None,
        joint_colorizer,
        total_loss_elems=6,
        loss_weight=2.5,
        loss_basis="target_area_mean",
        sample_grid_shape=(1, 1, 2),
        patch_shape=(2, 2),
        target_rgb_chunk=target_rgb_chunk,
        local_frame_ids=torch.tensor([0], dtype=torch.int64),
        vjp_mode="manual_hidden64",
    )
    star_only_loss, star_only_feature_grad, star_only_alpha_grad = _sparse_visual_rgb_loss_and_grads(
        feature_values,
        alpha_values,
        None,
        star_only_colorizer,
        total_loss_elems=6,
        loss_weight=2.5,
        loss_basis="target_area_mean",
        sample_grid_shape=(1, 1, 2),
        patch_shape=(2, 2),
        target_rgb_chunk=target_rgb_chunk,
        local_frame_ids=torch.tensor([0], dtype=torch.int64),
        vjp_mode="manual_hidden64_star_only",
    )

    assert star_only_loss.item() == pytest.approx(joint_loss.item(), abs=1.0e-7)
    torch.testing.assert_close(star_only_feature_grad, joint_feature_grad, atol=1.0e-6, rtol=1.0e-5)
    torch.testing.assert_close(star_only_alpha_grad, joint_alpha_grad, atol=1.0e-6, rtol=1.0e-5)
    for param in star_only_colorizer.parameters():
        assert param.grad is None


def test_sparse_visual_manual_hidden64_fastgelu_uses_approx_gelu_vjp() -> None:
    values = torch.tensor([-2.0, -0.5, 0.0, 1.0, 2.0], dtype=torch.float32)
    expected = torch.autograd.functional.jacobian(lambda x: x * torch.sigmoid(1.702 * x), values).diagonal()
    torch.testing.assert_close(_gelu_fast_sigmoid_grad(values), expected, atol=1.0e-6, rtol=1.0e-6)

    torch.manual_seed(17)
    feature_values = torch.randn((8, 3), dtype=torch.float32) * 0.4
    alpha_values = torch.linspace(0.2, 0.9, 8, dtype=torch.float32)
    target_rgb_chunk = torch.rand((1, 3, 4, 4), dtype=torch.float32)
    exact_colorizer = FeatureToColor(
        feature_dim=3,
        hidden_dim=4,
        activation="sigmoid",
        pre_norm=False,
        weight_init="kaiming",
        weight_init_gain=1.0,
    )
    fast_colorizer = FeatureToColor(
        feature_dim=3,
        hidden_dim=4,
        activation="sigmoid",
        pre_norm=False,
        weight_init="kaiming",
        weight_init_gain=1.0,
    )
    fast_colorizer.load_state_dict(exact_colorizer.state_dict())

    exact_loss, exact_feature_grad, exact_alpha_grad = _sparse_visual_rgb_loss_and_grads(
        feature_values,
        alpha_values,
        None,
        exact_colorizer,
        total_loss_elems=6,
        loss_weight=2.5,
        loss_basis="target_area_mean",
        sample_grid_shape=(1, 1, 2),
        patch_shape=(2, 2),
        target_rgb_chunk=target_rgb_chunk,
        local_frame_ids=torch.tensor([0], dtype=torch.int64),
        vjp_mode="manual_hidden64",
    )
    fast_loss, fast_feature_grad, fast_alpha_grad = _sparse_visual_rgb_loss_and_grads(
        feature_values,
        alpha_values,
        None,
        fast_colorizer,
        total_loss_elems=6,
        loss_weight=2.5,
        loss_basis="target_area_mean",
        sample_grid_shape=(1, 1, 2),
        patch_shape=(2, 2),
        target_rgb_chunk=target_rgb_chunk,
        local_frame_ids=torch.tensor([0], dtype=torch.int64),
        vjp_mode="manual_hidden64_fastgelu",
    )

    assert fast_loss.item() == pytest.approx(exact_loss.item(), abs=1.0e-7)
    torch.testing.assert_close(fast_alpha_grad, exact_alpha_grad, atol=1.0e-6, rtol=1.0e-5)
    assert torch.isfinite(fast_feature_grad).all()
    assert not torch.allclose(fast_feature_grad, exact_feature_grad, atol=1.0e-7, rtol=1.0e-5)
    exact_params = dict(exact_colorizer.named_parameters())
    fast_params = dict(fast_colorizer.named_parameters())
    torch.testing.assert_close(fast_params["net.2.weight"].grad, exact_params["net.2.weight"].grad)
    torch.testing.assert_close(fast_params["net.2.bias"].grad, exact_params["net.2.bias"].grad)
    assert not torch.allclose(fast_params["net.0.weight"].grad, exact_params["net.0.weight"].grad)


def test_sparse_visual_manual_linear_vjp_matches_autograd_target_area() -> None:
    torch.manual_seed(19)
    feature_values = torch.randn((8, 3), dtype=torch.float32) * 0.2
    alpha_values = torch.linspace(0.2, 0.9, 8, dtype=torch.float32)
    target_rgb_chunk = torch.rand((1, 3, 4, 4), dtype=torch.float32)
    autograd_colorizer = FeatureToColor(
        feature_dim=3,
        hidden_dim=None,
        activation="sigmoid",
        pre_norm=False,
        weight_init="kaiming",
        weight_init_gain=1.0,
    )
    manual_colorizer = FeatureToColor(
        feature_dim=3,
        hidden_dim=None,
        activation="sigmoid",
        pre_norm=False,
        weight_init="kaiming",
        weight_init_gain=1.0,
    )
    manual_colorizer.load_state_dict(autograd_colorizer.state_dict())

    autograd_loss, autograd_feature_grad, autograd_alpha_grad = _sparse_visual_rgb_loss_and_grads(
        feature_values,
        alpha_values,
        None,
        autograd_colorizer,
        total_loss_elems=6,
        loss_weight=2.5,
        loss_basis="target_area_mean",
        sample_grid_shape=(1, 1, 2),
        patch_shape=(2, 2),
        target_rgb_chunk=target_rgb_chunk,
        local_frame_ids=torch.tensor([0], dtype=torch.int64),
    )
    manual_loss, manual_feature_grad, manual_alpha_grad = _sparse_visual_rgb_loss_and_grads(
        feature_values,
        alpha_values,
        None,
        manual_colorizer,
        total_loss_elems=6,
        loss_weight=2.5,
        loss_basis="target_area_mean",
        sample_grid_shape=(1, 1, 2),
        patch_shape=(2, 2),
        target_rgb_chunk=target_rgb_chunk,
        local_frame_ids=torch.tensor([0], dtype=torch.int64),
        vjp_mode="manual_linear",
    )

    assert manual_loss.item() == pytest.approx(autograd_loss.item(), abs=1.0e-7)
    torch.testing.assert_close(manual_feature_grad, autograd_feature_grad, atol=1.0e-6, rtol=1.0e-5)
    torch.testing.assert_close(manual_alpha_grad, autograd_alpha_grad, atol=1.0e-6, rtol=1.0e-5)
    for (_, autograd_param), (_, manual_param) in zip(
        autograd_colorizer.named_parameters(),
        manual_colorizer.named_parameters(),
        strict=True,
    ):
        assert autograd_param.grad is not None
        assert manual_param.grad is not None
        torch.testing.assert_close(manual_param.grad, autograd_param.grad, atol=1.0e-6, rtol=1.0e-5)


@pytest.mark.parametrize(("hidden_dim", "vjp_mode"), [(4, "manual_hidden"), (None, "manual_linear")])
def test_sparse_visual_manual_vjp_matches_autograd_target_background_target_area(
    hidden_dim: int | None,
    vjp_mode: str,
) -> None:
    torch.manual_seed(23)
    feature_values = torch.randn((8, 3), dtype=torch.float32) * 0.2
    alpha_values = torch.linspace(0.2, 0.9, 8, dtype=torch.float32)
    target_values = torch.rand((8, 3), dtype=torch.float32)
    target_rgb_chunk = torch.rand((1, 3, 4, 4), dtype=torch.float32)
    autograd_colorizer = FeatureToColor(
        feature_dim=3,
        hidden_dim=hidden_dim,
        activation="sigmoid",
        pre_norm=False,
        weight_init="kaiming",
        weight_init_gain=1.0,
    )
    manual_colorizer = FeatureToColor(
        feature_dim=3,
        hidden_dim=hidden_dim,
        activation="sigmoid",
        pre_norm=False,
        weight_init="kaiming",
        weight_init_gain=1.0,
    )
    manual_colorizer.load_state_dict(autograd_colorizer.state_dict())

    autograd_loss, autograd_feature_grad, autograd_alpha_grad = _sparse_visual_rgb_loss_and_grads(
        feature_values,
        alpha_values,
        target_values,
        autograd_colorizer,
        total_loss_elems=6,
        loss_weight=2.5,
        loss_basis="target_area_mean",
        sample_grid_shape=(1, 1, 2),
        patch_shape=(2, 2),
        target_rgb_chunk=target_rgb_chunk,
        local_frame_ids=torch.tensor([0], dtype=torch.int64),
        composition="target_background",
    )
    manual_loss, manual_feature_grad, manual_alpha_grad = _sparse_visual_rgb_loss_and_grads(
        feature_values,
        alpha_values,
        target_values,
        manual_colorizer,
        total_loss_elems=6,
        loss_weight=2.5,
        loss_basis="target_area_mean",
        sample_grid_shape=(1, 1, 2),
        patch_shape=(2, 2),
        target_rgb_chunk=target_rgb_chunk,
        local_frame_ids=torch.tensor([0], dtype=torch.int64),
        vjp_mode=vjp_mode,
        composition="target_background",
    )

    assert manual_loss.item() == pytest.approx(autograd_loss.item(), abs=1.0e-7)
    torch.testing.assert_close(manual_feature_grad, autograd_feature_grad, atol=1.0e-6, rtol=1.0e-5)
    torch.testing.assert_close(manual_alpha_grad, autograd_alpha_grad, atol=1.0e-6, rtol=1.0e-5)
    for (_, autograd_param), (_, manual_param) in zip(
        autograd_colorizer.named_parameters(),
        manual_colorizer.named_parameters(),
        strict=True,
    ):
        assert autograd_param.grad is not None
        assert manual_param.grad is not None
        torch.testing.assert_close(manual_param.grad, autograd_param.grad, atol=1.0e-6, rtol=1.0e-5)


def test_sparse_visual_local_frame_ids_match_chunk_window() -> None:
    ids = _sparse_visual_local_frame_ids_for_chunk(
        render_frames=8,
        frame_start=2,
        chunk_frames=3,
        sample_grid_shape=(8, 2, 2),
        device=torch.device("cpu"),
    )

    assert ids.tolist() == [0, 1, 2]


def test_rgb_probe_config_requires_target_grid_materialization() -> None:
    cfg = {
        "data": {
            "video_path": "unused.mp4",
            "start_seconds": None,
            "fps": None,
            "duration_seconds": None,
            "image_crop_mode": "center_square",
            "target_size": 64,
            "max_frames": 8,
        },
        "train": {
            "steps": 1,
            "lr": 0.01,
            "device": "mps",
            "seed": 1,
            "frame_chunk_size": 2,
            "require_loss_decrease": False,
            "require_gradient_flow": False,
            "require_no_tile_overflow": False,
        },
        "features": {
            "extractor": "rgb_pyramid",
            "layers": ["rgb_x1"],
            "cache_dir": "unused",
            "sample_cache_key": "unused",
        },
        "feature_uvt": {
            "tube_count": 8,
            "feature_dim": 32,
            "tile_t": 8,
            "tile_capacity": 128,
            "alpha_threshold": 0.0,
            "max_alpha": 0.99,
        },
        "feature_target": {
            "enabled": True,
            "layer": "rgb_x1",
            "loss_type": "mse",
            "loss_weight": 1.0,
            "rgb_loss_weight": 0.0,
            "channel_adapter": "repeat_truncate",
            "temporal_spatial_adapter": "trilinear",
            "normalization": "none",
            "materialization": "cached_chunks",
            "rgb_probe_checkpoint": "probe.pt",
            "rgb_probe_loss_weight": 1.0,
        },
        "sparse_visual": {
            "enabled": True,
            "loss_weight": 1.0,
            "pixel_source": "stratified_patch_grid",
            "loss_basis": "target_area_mean",
            "composition": "target_background",
            "sample_grid_shape": [8, 4, 4],
            "patch_shape": [2, 2],
            "alpha_loss_weight": 0.25,
            "alpha_target": 0.8,
            "black_hole_loss_weight": 0.5,
        },
        "dense_alpha": {
            "enabled": True,
            "loss_weight": 0.75,
            "alpha_target": 0.75,
            "backward_mode": "direct_atomic_skip_feature_grad",
            "render_mode": "sparse_f1",
        },
        "visibility_proxy": {
            "enabled": True,
            "loss_weight": 0.01,
            "target_top_fraction": 0.02,
            "max_points": 128,
            "grid_stride": 4,
            "frame_stride": 2,
            "center_weight": 0.25,
            "support_weight": 1.0,
            "support_epsilon": 1.0e-4,
            "scale_px": 16.0,
            "temperature": 0.75,
            "velocity_penalty": 0.001,
        },
        "support_birth_split": {
            "enabled": True,
            "target_top_fraction": 0.02,
            "max_points": 128,
            "grid_stride": 4,
            "frame_stride": 2,
            "reallocate_tubes": 4,
            "support_radius_px": 12.0,
            "temporal_radius_frames": 4.0,
            "opacity": 0.7,
            "tube_selection": "lowest_opacity",
        },
        "colorize": {
            "hidden_dim": 8,
            "activation": "sigmoid",
            "pre_norm": False,
            "weight_init": "kaiming",
            "weight_init_gain": 1.0,
        },
        "output": {
            "out_json": None,
            "contact_sheet": None,
            "contact_sheet_frames": 4,
            "contact_sheet_mode": "linspace",
            "side_by_side_video": None,
            "side_by_side_fps": None,
        },
        "logging": {
            "wandb_enabled": False,
            "wandb_project": "unused",
            "wandb_run_name": "unused",
            "wandb_tags": [],
            "wandb_mode": None,
        },
    }

    with pytest.raises(ValueError, match="materialization=target_grid"):
        resolve_config(cfg)

    cfg["feature_target"]["materialization"] = "target_grid"
    resolved = resolve_config(cfg)
    assert resolved["train"]["resume_checkpoint"] is None
    assert resolved["train"]["resume_optimizer"] is True
    assert resolved["train"]["global_step_offset"] == 0
    assert resolved["output"]["checkpoint"] is None
    assert resolved["sparse_visual"]["alpha_loss_weight"] == 0.25
    assert resolved["sparse_visual"]["alpha_target"] == 0.8
    assert resolved["sparse_visual"]["black_hole_loss_weight"] == 0.5
    assert resolved["sparse_visual"]["composition"] == "target_background"
    assert resolved["dense_alpha"]["loss_weight"] == 0.75
    assert resolved["dense_alpha"]["alpha_target"] == 0.75
    assert resolved["dense_alpha"]["backward_mode"] == "direct_atomic_skip_feature_grad"
    assert resolved["dense_alpha"]["render_mode"] == "sparse_f1"
    assert resolved["visibility_proxy"]["loss_weight"] == 0.01
    assert resolved["visibility_proxy"]["max_points"] == 128
    assert resolved["visibility_proxy"]["grid_stride"] == 4
    assert resolved["visibility_proxy"]["center_weight"] == 0.25
    assert resolved["visibility_proxy"]["support_weight"] == 1.0
    assert resolved["visibility_proxy"]["support_epsilon"] == 1.0e-4
    assert resolved["support_birth_split"]["target_point_source"] == "top_brightness"
    assert resolved["support_birth_split"]["reallocate_tubes"] == 4
    assert resolved["support_birth_split"]["support_shape"] == "isotropic"
    assert resolved["support_birth_split"]["support_radius_along_px"] == 12.0
    assert resolved["support_birth_split"]["support_radius_across_px"] == 12.0
    assert resolved["support_birth_split"]["support_precision_radius_px"] == 12.0
    assert resolved["support_birth_split"]["tube_selection"] == "lowest_opacity"
    assert resolved["support_birth_split"]["center_strategy"] == "global_line"
    assert resolved["support_birth_split"]["center_count"] == 1
    assert resolved["support_birth_split"]["tube_allocation"] == "proportional"
    assert resolved["support_birth_split"]["feature_init_mode"] == "preserve"
    assert resolved["support_birth_split"]["target_alpha_loss_weight"] == 0.0
    assert resolved["support_birth_split"]["target_alpha_target"] == 0.99
    assert resolved["support_birth_split"]["target_alpha_max_points"] == 128
    assert resolved["support_birth_split"]["target_area_loss_weight"] == 0.0
    assert resolved["support_birth_split"]["target_area_patch_shape"] == [2, 2]
    assert resolved["support_birth_split"]["target_area_max_points"] == 128
    assert resolved["support_birth_split"]["target_area_vjp_mode"] == "manual_hidden64_star_only"
    assert resolved["support_birth_split"]["target_area_composition"] == "black"
    assert resolved["support_birth_split"]["tile_overflow_repair_enabled"] is False
    assert resolved["support_birth_split"]["tile_overflow_repair_max_drops"] == 4
    assert resolved["support_birth_split"]["tile_overflow_repair_guard_refs"] == 0
    assert resolved["support_birth_split"]["tile_overflow_repair_opacity"] == 1.0e-5

    cfg["dense_alpha"]["backward_mode"] = "gradcache_reduce_feature_grad_vec4"
    with pytest.raises(ValueError, match="dense_alpha.backward_mode"):
        resolve_config(cfg)
    cfg["dense_alpha"]["backward_mode"] = "direct_atomic_skip_feature_grad"
    cfg["dense_alpha"]["render_mode"] = "feature_dense"
    with pytest.raises(ValueError, match="dense_alpha.render_mode"):
        resolve_config(cfg)
    cfg["dense_alpha"]["render_mode"] = "sparse_f1"
    cfg["visibility_proxy"]["target_top_fraction"] = 0.0
    with pytest.raises(ValueError, match="visibility_proxy.target_top_fraction"):
        resolve_config(cfg)
    cfg["visibility_proxy"]["target_top_fraction"] = 0.02
    cfg["visibility_proxy"]["center_weight"] = 0.0
    cfg["visibility_proxy"]["support_weight"] = 0.0
    with pytest.raises(ValueError, match="center_weight or support_weight"):
        resolve_config(cfg)
    cfg["visibility_proxy"]["support_weight"] = 1.0
    cfg["support_birth_split"]["tube_selection"] = "random"
    with pytest.raises(ValueError, match="support_birth_split.tube_selection"):
        resolve_config(cfg)
    cfg["support_birth_split"]["tube_selection"] = "lowest_opacity"
    cfg["support_birth_split"]["target_point_source"] = "invalid"
    with pytest.raises(ValueError, match="support_birth_split.target_point_source"):
        resolve_config(cfg)
    cfg["support_birth_split"]["target_point_source"] = "top_brightness"
    cfg["support_birth_split"]["support_shape"] = "unknown"
    with pytest.raises(ValueError, match="support_birth_split.support_shape"):
        resolve_config(cfg)
    cfg["support_birth_split"]["support_shape"] = "isotropic"
    cfg["support_birth_split"]["center_strategy"] = "unknown"
    with pytest.raises(ValueError, match="support_birth_split.center_strategy"):
        resolve_config(cfg)
    cfg["support_birth_split"]["center_strategy"] = "global_line"
    cfg["support_birth_split"]["center_count"] = 0
    with pytest.raises(ValueError, match="support_birth_split.center_count"):
        resolve_config(cfg)
    cfg["support_birth_split"]["center_count"] = 1
    cfg["support_birth_split"]["tube_allocation"] = "unknown"
    with pytest.raises(ValueError, match="support_birth_split.tube_allocation"):
        resolve_config(cfg)
    cfg["support_birth_split"]["tube_allocation"] = "proportional"
    cfg["support_birth_split"]["feature_init_mode"] = "random"
    with pytest.raises(ValueError, match="support_birth_split.feature_init_mode"):
        resolve_config(cfg)
    cfg["support_birth_split"]["feature_init_mode"] = "preserve"
    cfg["support_birth_split"]["target_alpha_loss_weight"] = -0.1
    with pytest.raises(ValueError, match="support_birth_split.target_alpha_loss_weight"):
        resolve_config(cfg)
    cfg["support_birth_split"]["target_alpha_loss_weight"] = 0.0
    cfg["support_birth_split"]["target_alpha_target"] = 1.1
    with pytest.raises(ValueError, match="support_birth_split.target_alpha_target"):
        resolve_config(cfg)
    cfg["support_birth_split"]["target_alpha_target"] = 0.75
    cfg["support_birth_split"]["target_alpha_max_points"] = 0
    with pytest.raises(ValueError, match="support_birth_split.target_alpha_max_points"):
        resolve_config(cfg)
    cfg["support_birth_split"]["target_alpha_max_points"] = 128
    cfg["support_birth_split"]["target_area_loss_weight"] = -0.1
    with pytest.raises(ValueError, match="support_birth_split.target_area_loss_weight"):
        resolve_config(cfg)
    cfg["support_birth_split"]["target_area_loss_weight"] = 0.0
    cfg["support_birth_split"]["target_area_patch_shape"] = [2, 0]
    with pytest.raises(ValueError, match="support_birth_split.target_area_patch_shape"):
        resolve_config(cfg)
    cfg["support_birth_split"]["target_area_patch_shape"] = [2, 2]
    cfg["support_birth_split"]["target_area_max_points"] = 0
    with pytest.raises(ValueError, match="support_birth_split.target_area_max_points"):
        resolve_config(cfg)
    cfg["support_birth_split"]["target_area_max_points"] = 128
    cfg["support_birth_split"]["target_area_vjp_mode"] = "native_hidden64_star_only"
    with pytest.raises(ValueError, match="support_birth_split.target_area_vjp_mode"):
        resolve_config(cfg)
    cfg["support_birth_split"]["target_area_vjp_mode"] = "manual_hidden64_star_only"
    cfg["support_birth_split"]["target_area_composition"] = "unknown"
    with pytest.raises(ValueError, match="support_birth_split.target_area_composition"):
        resolve_config(cfg)
    cfg["support_birth_split"]["target_area_composition"] = "black"
    cfg["support_birth_split"]["tile_overflow_repair_max_drops"] = 5
    with pytest.raises(ValueError, match="support_birth_split.tile_overflow_repair_max_drops"):
        resolve_config(cfg)
    cfg["support_birth_split"]["tile_overflow_repair_max_drops"] = 4
    cfg["support_birth_split"]["tile_overflow_repair_guard_refs"] = -1
    with pytest.raises(ValueError, match="support_birth_split.tile_overflow_repair_guard_refs"):
        resolve_config(cfg)
    cfg["support_birth_split"]["tile_overflow_repair_guard_refs"] = 0
    cfg["train"]["resume_checkpoint"] = "checkpoint.pt"
    cfg["train"]["resume_optimizer"] = True
    with pytest.raises(ValueError, match="support_birth_split"):
        resolve_config(cfg)


def test_visibility_support_proxy_sends_opacity_and_precision_gradients() -> None:
    render_cfg = FeatureTubeRenderConfig(frames=3, height=16, width=16, feature_dim=4)
    model = FeatureScreenTimeTubeModel(8, render_cfg, seed=5, device="cpu")
    target_points = torch.tensor([[8.5, 8.5, 0.0], [9.5, 7.5, 1.0]], dtype=torch.float32)

    loss = _visibility_proxy_loss(
        model,
        target_points,
        center_weight=0.0,
        support_weight=1.0,
        support_epsilon=1.0e-4,
        max_alpha=render_cfg.max_alpha,
        scale_px=16.0,
        temperature=0.75,
        velocity_penalty=0.0,
    )
    loss.backward()

    assert model.raw_opacity.grad is not None
    assert model.raw_precision.grad is not None
    assert bool((model.raw_opacity.grad.abs() > 0).any())
    assert bool((model.raw_precision.grad.abs() > 0).any())


def test_support_birth_split_reallocates_low_opacity_tubes_and_preserves_budget() -> None:
    render_cfg = FeatureTubeRenderConfig(frames=3, height=16, width=16, feature_dim=4)
    model = FeatureScreenTimeTubeModel(6, render_cfg, seed=5, device="cpu")
    with torch.no_grad():
        model.raw_opacity.fill_(3.0)
        model.raw_opacity[:2].fill_(-10.0)
    target_points = torch.tensor(
        [
            [2.0, 6.0, -1.0],
            [4.0, 5.0, 0.0],
            [6.0, 4.0, 1.0],
        ],
        dtype=torch.float32,
    )

    state = _apply_support_birth_split(
        model,
        target_points,
        reallocate_tubes=2,
        support_radius_px=4.0,
        support_shape="isotropic",
        support_radius_along_px=4.0,
        support_radius_across_px=4.0,
        support_precision_radius_px=4.0,
        temporal_radius_frames=2.0,
        opacity=0.7,
        max_alpha=render_cfg.max_alpha,
        tube_selection="lowest_opacity",
    )

    assert model.tube_count == 6
    assert state["tube_count_preserved"] is True
    assert state["reallocated_tubes"] == 2
    assert set(state["selected_tube_ids"]) == {0, 1}
    torch.testing.assert_close(torch.tensor(state["fit_center_uv_at_t0"]), torch.tensor([4.0, 5.0]))
    torch.testing.assert_close(torch.tensor(state["fit_velocity_uv"]), torch.tensor([2.0, -1.0]))
    assert state["selected_opacity_mean_after"] > state["selected_opacity_mean_before"]


def test_support_birth_split_samples_target_grid_features_at_points() -> None:
    target_grid = torch.tensor(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[10.0, 20.0], [30.0, 40.0]],
            ],
            [
                [[5.0, 6.0], [7.0, 8.0]],
                [[50.0, 60.0], [70.0, 80.0]],
            ],
        ],
        dtype=torch.float32,
    )
    target_points = torch.tensor(
        [
            [0.5, 0.5, -0.5],
            [1.5, 1.5, 0.5],
        ],
        dtype=torch.float32,
    )

    values = _support_birth_split_sample_target_grid_features(
        target_grid,
        target_points,
        frames=2,
        height=2,
        width=2,
        mode="trilinear",
    )

    torch.testing.assert_close(values, torch.tensor([[1.0, 10.0], [8.0, 80.0]]))


def test_support_birth_split_target_group_mean_initializes_reallocated_features() -> None:
    render_cfg = FeatureTubeRenderConfig(frames=3, height=16, width=16, feature_dim=4)
    model = FeatureScreenTimeTubeModel(6, render_cfg, seed=6, device="cpu")
    with torch.no_grad():
        model.raw_opacity.fill_(3.0)
        model.raw_opacity[:2].fill_(-10.0)
        model.raw_feature.zero_()
    target_points = torch.tensor(
        [
            [2.0, 6.0, -1.0],
            [4.0, 5.0, 0.0],
            [6.0, 4.0, 1.0],
        ],
        dtype=torch.float32,
    )
    target_point_features = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0, 8.0],
        ],
        dtype=torch.float32,
    )

    state = _apply_support_birth_split(
        model,
        target_points,
        reallocate_tubes=2,
        support_radius_px=4.0,
        support_shape="isotropic",
        support_radius_along_px=4.0,
        support_radius_across_px=4.0,
        support_precision_radius_px=4.0,
        temporal_radius_frames=2.0,
        opacity=0.7,
        max_alpha=render_cfg.max_alpha,
        tube_selection="lowest_opacity",
        target_point_features=target_point_features,
        feature_init_mode="target_group_mean",
    )

    selected = torch.tensor(state["selected_tube_ids"], dtype=torch.int64)
    expected = target_point_features.mean(dim=0, keepdim=True).expand(2, -1)
    torch.testing.assert_close(model.raw_feature.detach().index_select(0, selected), expected)
    assert state["feature_init_mode"] == "target_group_mean"
    assert state["feature_init_applied"] is True
    assert state["selected_feature_abs_mean_after"] > state["selected_feature_abs_mean_before"]


def test_support_birth_split_trajectory_ellipse_records_anisotropic_support() -> None:
    render_cfg = FeatureTubeRenderConfig(frames=4, height=32, width=32, feature_dim=4)
    model = FeatureScreenTimeTubeModel(8, render_cfg, seed=7, device="cpu")
    target_points = torch.tensor(
        [
            [6.0, 12.0, -1.0],
            [10.0, 12.0, 0.0],
            [14.0, 12.0, 1.0],
            [18.0, 12.0, 2.0],
        ],
        dtype=torch.float32,
    )

    state = _apply_support_birth_split(
        model,
        target_points,
        reallocate_tubes=4,
        support_radius_px=24.0,
        support_shape="trajectory_ellipse",
        support_radius_along_px=12.0,
        support_radius_across_px=3.0,
        support_precision_radius_px=6.0,
        temporal_radius_frames=3.0,
        opacity=0.6,
        max_alpha=render_cfg.max_alpha,
        tube_selection="first",
    )

    assert state["support_shape"] == "trajectory_ellipse"
    assert state["support_radius_along_px"] == 12.0
    assert state["support_radius_across_px"] == 3.0
    assert state["support_precision_radius_px"] == 6.0
    torch.testing.assert_close(torch.tensor(state["fit_velocity_uv"]), torch.tensor([4.0, 0.0]))


def test_support_birth_split_farthest_xy_uses_multiple_centers() -> None:
    render_cfg = FeatureTubeRenderConfig(frames=4, height=32, width=32, feature_dim=4)
    model = FeatureScreenTimeTubeModel(8, render_cfg, seed=9, device="cpu")
    target_points = torch.tensor(
        [
            [4.0, 4.0, -1.0],
            [5.0, 4.5, 0.0],
            [24.0, 24.0, 1.0],
            [25.0, 23.5, 2.0],
        ],
        dtype=torch.float32,
    )

    state = _apply_support_birth_split(
        model,
        target_points,
        reallocate_tubes=4,
        support_radius_px=4.0,
        support_shape="isotropic",
        support_radius_along_px=4.0,
        support_radius_across_px=4.0,
        support_precision_radius_px=4.0,
        temporal_radius_frames=3.0,
        opacity=0.6,
        max_alpha=render_cfg.max_alpha,
        tube_selection="first",
        center_strategy="farthest_xy",
        center_count=2,
    )

    assert state["center_strategy"] == "farthest_xy"
    assert state["requested_center_count"] == 2
    assert state["actual_center_count"] == 2
    assert sum(state["center_point_counts"]) == 4
    assert sum(state["center_tube_counts"]) == 4
    assert len(state["fit_centers_uv_at_t0"]) == 2


def test_support_birth_split_uniform_tube_allocation_limits_cluster_packing() -> None:
    groups = [
        torch.zeros((10, 3), dtype=torch.float32),
        torch.zeros((1, 3), dtype=torch.float32),
        torch.zeros((1, 3), dtype=torch.float32),
        torch.zeros((1, 3), dtype=torch.float32),
    ]

    assert _support_birth_split_tube_counts(groups, 8, tube_allocation="proportional") == [5, 1, 1, 1]
    assert _support_birth_split_tube_counts(groups, 8, tube_allocation="uniform") == [2, 2, 2, 2]
    uniform_with_remainder = _support_birth_split_tube_counts(groups, 10, tube_allocation="uniform")
    assert uniform_with_remainder[0] == 3
    assert sorted(uniform_with_remainder[1:]) == [2, 2, 3]


def test_support_birth_split_tile_overflow_repair_selects_new_tube_to_drop() -> None:
    tile_counts = torch.tensor([130, 128, 129], dtype=torch.int32)
    tile_tube_ids = torch.full((3, 128), -1, dtype=torch.int32)
    tile_tube_ids[0, :3] = torch.tensor([10, 20, 30], dtype=torch.int32)
    tile_tube_ids[2, :3] = torch.tensor([20, 30, 40], dtype=torch.int32)
    selected_ids = torch.tensor([20, 30, 99], dtype=torch.int64)

    repair = _support_birth_split_repair_tile_overflow_ids(
        tile_counts,
        tile_tube_ids.reshape(-1),
        selected_ids,
        tile_capacity=128,
        max_drops=2,
    )

    assert repair["initial_overflow_tile_count"] == 2
    assert repair["initial_overflow_excess_tube_refs"] == 3
    assert repair["drop_count"] == 2
    assert repair["dropped_tube_ids"] == [20, 30]
    assert repair["estimated_remaining_overflow_tile_count"] == 0


def test_support_birth_split_set_tube_opacity_hides_repaired_tubes() -> None:
    render_cfg = FeatureTubeRenderConfig(frames=3, height=16, width=16, feature_dim=4)
    model = FeatureScreenTimeTubeModel(4, render_cfg, seed=5, device="cpu")

    _support_birth_split_set_tube_opacity(model, torch.tensor([1, 3]), opacity=1.0e-5)
    _ma, _q_uvt, _depth0, _depth_beta, opacity, _feature = model.tensors()

    assert opacity[1].item() < 1.1e-5
    assert opacity[3].item() < 1.1e-5
    assert opacity[0].item() > 0.1


def test_support_birth_split_uncovered_target_points_prefer_low_alpha_bright_pixels() -> None:
    target_rgb = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
    target_rgb[:, :, 2, 2] = 1.0
    target_rgb[:, :, 6, 6] = 0.9
    sampled_alpha = torch.zeros((1, 8, 8), dtype=torch.float32)
    sampled_alpha[:, 2, 2] = 1.0

    top_points, top_meta = _support_birth_split_target_points(
        target_rgb,
        target_point_source="top_brightness",
        target_top_fraction=0.1,
        max_points=1,
        grid_stride=1,
        frame_stride=1,
        device=torch.device("cpu"),
    )
    uncovered_points, uncovered_meta = _support_birth_split_target_points(
        target_rgb,
        target_point_source="uncovered_brightness",
        target_top_fraction=0.1,
        max_points=1,
        grid_stride=1,
        frame_stride=1,
        device=torch.device("cpu"),
        sampled_alpha=sampled_alpha,
    )

    torch.testing.assert_close(top_points[0, :2], torch.tensor([2.5, 2.5]))
    torch.testing.assert_close(uncovered_points[0, :2], torch.tensor([6.5, 6.5]))
    assert top_meta["target_point_source"] == "top_brightness"
    assert uncovered_meta["target_point_source"] == "uncovered_brightness"
    assert uncovered_meta["selected_alpha_mean"] == 0.0


def test_support_birth_split_sampled_tile_load_maps_grid_to_tile_bins() -> None:
    # frames=4, 16x16 image, 8x8x2 tiles -> 2 temporal bins * 2 * 2 spatial bins.
    tile_counts = torch.arange(8, dtype=torch.int32)

    sampled_tile_load = _support_birth_split_sampled_tile_load(
        tile_counts,
        frames=4,
        height=16,
        width=16,
        frame_stride=2,
        grid_stride=8,
        tile_x=8,
        tile_y=8,
        tile_t=2,
    )

    assert sampled_tile_load.shape == (2, 2, 2)
    torch.testing.assert_close(sampled_tile_load[0], torch.tensor([[0.0, 1.0], [2.0, 3.0]]))
    torch.testing.assert_close(sampled_tile_load[1], torch.tensor([[4.0, 5.0], [6.0, 7.0]]))


def test_support_birth_split_cap_slack_target_points_avoid_loaded_tiles() -> None:
    target_rgb = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
    target_rgb[:, :, 2, 2] = 1.0
    target_rgb[:, :, 6, 6] = 0.9
    sampled_alpha = torch.zeros((1, 8, 8), dtype=torch.float32)
    sampled_tile_load = torch.zeros((1, 8, 8), dtype=torch.float32)
    sampled_tile_load[:, 2, 2] = 128.0

    points, meta = _support_birth_split_target_points(
        target_rgb,
        target_point_source="cap_slack_uncovered_brightness",
        target_top_fraction=0.1,
        max_points=1,
        grid_stride=1,
        frame_stride=1,
        device=torch.device("cpu"),
        sampled_alpha=sampled_alpha,
        sampled_tile_load=sampled_tile_load,
        tile_capacity=128,
    )

    torch.testing.assert_close(points[0, :2], torch.tensor([6.5, 6.5]))
    assert meta["target_point_source"] == "cap_slack_uncovered_brightness"
    assert meta["selected_tile_load_mean"] == 0.0
    assert meta["selected_tile_slack_mean"] == 1.0


def test_support_birth_split_residual_cap_slack_target_points_prefer_errorful_uncovered_pixels() -> None:
    target_rgb = torch.ones((1, 3, 8, 8), dtype=torch.float32) * 0.5
    target_rgb[:, :, 2, 2] = 1.0
    target_rgb[:, :, 6, 6] = 0.9
    sampled_alpha = torch.zeros((1, 8, 8), dtype=torch.float32)
    sampled_residual = torch.zeros((1, 8, 8), dtype=torch.float32)
    sampled_residual[:, 2, 2] = 0.1
    sampled_residual[:, 6, 6] = 0.8
    sampled_tile_load = torch.zeros((1, 8, 8), dtype=torch.float32)

    points, meta = _support_birth_split_target_points(
        target_rgb,
        target_point_source="cap_slack_residual_uncovered_brightness",
        target_top_fraction=0.1,
        max_points=1,
        grid_stride=1,
        frame_stride=1,
        device=torch.device("cpu"),
        sampled_alpha=sampled_alpha,
        sampled_residual=sampled_residual,
        sampled_tile_load=sampled_tile_load,
        tile_capacity=128,
    )

    torch.testing.assert_close(points[0, :2], torch.tensor([6.5, 6.5]))
    assert meta["target_point_source"] == "cap_slack_residual_uncovered_brightness"
    assert meta["selected_residual_mean"] == pytest.approx(0.8)
    assert meta["selected_tile_slack_mean"] == 1.0


def test_support_birth_split_footprint_residual_can_choose_neighbor_with_tile_slack() -> None:
    target_rgb = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
    target_rgb[:, :, 2, 2] = 1.0
    sampled_alpha = torch.zeros((1, 8, 8), dtype=torch.float32)
    sampled_residual = torch.zeros((1, 8, 8), dtype=torch.float32)
    sampled_residual[:, 2, 2] = 1.0
    sampled_tile_load = torch.zeros((1, 8, 8), dtype=torch.float32)
    sampled_tile_load[:, 2, 2] = 128.0

    points, meta = _support_birth_split_target_points(
        target_rgb,
        target_point_source="cap_slack_footprint_residual_uncovered_brightness",
        target_top_fraction=0.1,
        max_points=1,
        grid_stride=1,
        frame_stride=1,
        device=torch.device("cpu"),
        sampled_alpha=sampled_alpha,
        sampled_residual=sampled_residual,
        sampled_tile_load=sampled_tile_load,
        tile_capacity=128,
        footprint_radius_px=1.0,
    )

    assert meta["target_point_source"] == "cap_slack_footprint_residual_uncovered_brightness"
    assert meta["footprint_radius_samples"] == 1
    assert meta["selected_footprint_score_mean"] > 0.0
    assert meta["selected_tile_load_mean"] == 0.0
    assert meta["selected_residual_mean"] == 0.0
    assert torch.dist(points[0, :2], torch.tensor([2.5, 2.5])) <= 2.0


def test_feature_target_weight_schedule_selects_stages() -> None:
    schedule = _feature_target_weight_schedule(
        {
            "train": {"steps": 5},
            "feature_target": {
                "enabled": True,
                "loss_weight": 1.0,
                "rgb_loss_weight": 10.0,
                "weight_schedule": [
                    {
                        "label": "rgb_warm",
                        "until_step": 2,
                        "loss_weight": 0.0,
                        "rgb_loss_weight": 20.0,
                    },
                    {
                        "label": "mixed",
                        "until_step": 5,
                        "loss_weight": 1.0,
                        "rgb_loss_weight": 10.0,
                        "rgb_grid_loss_weight": 3.0,
                    },
                ],
            },
        }
    )

    assert [stage.label for stage in schedule] == ["rgb_warm", "mixed"]
    assert _feature_target_weights_for_step(schedule, 0).rgb_loss_weight == 20.0
    assert _feature_target_weights_for_step(schedule, 0).rgb_probe_loss_weight == 0.0
    assert _feature_target_weights_for_step(schedule, 1).loss_weight == 0.0
    assert _feature_target_weights_for_step(schedule, 2).loss_weight == 1.0
    assert _feature_target_weights_for_step(schedule, 4).rgb_loss_weight == 10.0
    assert _feature_target_weights_for_step(schedule, 4).rgb_grid_loss_weight == 3.0


def test_feature_target_weight_schedule_must_cover_all_steps() -> None:
    with pytest.raises(ValueError, match="cover exactly"):
        _feature_target_weight_schedule(
            {
                "train": {"steps": 5},
                "feature_target": {
                    "enabled": True,
                    "loss_weight": 1.0,
                    "rgb_loss_weight": 10.0,
                    "weight_schedule": [
                        {
                            "until_step": 2,
                            "loss_weight": 1.0,
                            "rgb_loss_weight": 10.0,
                        }
                    ],
                },
            }
        )


def test_feature_target_weight_schedule_uses_explicit_global_step_offset() -> None:
    schedule = _feature_target_weight_schedule(
        {
            "train": {"steps": 2, "global_step_offset": 3},
            "feature_target": {
                "enabled": True,
                "loss_weight": 1.0,
                "rgb_loss_weight": 10.0,
                "weight_schedule": [
                    {
                        "label": "previous_segment",
                        "until_step": 3,
                        "loss_weight": 1.0,
                        "rgb_loss_weight": 10.0,
                    },
                    {
                        "label": "resume_segment",
                        "until_step": 5,
                        "loss_weight": 0.5,
                        "rgb_loss_weight": 20.0,
                        "rgb_probe_loss_weight": 5.0,
                    },
                ],
            },
        }
    )

    assert _feature_target_weights_for_step(schedule, 3).label == "resume_segment"
    assert _feature_target_weights_for_step(schedule, 4).rgb_loss_weight == 20.0
    assert _feature_target_weights_for_step(schedule, 4).rgb_probe_loss_weight == 5.0
    with pytest.raises(IndexError):
        _feature_target_weights_for_step(schedule, 5)


def test_training_checkpoint_roundtrips_model_colorizer_optimizer(tmp_path) -> None:
    torch.manual_seed(29)
    model = torch.nn.Linear(2, 3)
    colorizer = torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam([*model.parameters(), *colorizer.parameters()], lr=0.01)
    x = torch.randn(5, 2)
    target = torch.randn(5, 1)
    loss = (colorizer(model(x)) - target).square().mean()
    loss.backward()
    optimizer.step()

    path = tmp_path / "star_feature_overfit.pt"
    row = {
        "steps": 3,
        "losses": [1.0, 0.8, 0.6],
        "rgb_losses": [],
        "feature_target_losses": [1.0, 0.8, 0.6],
        "rgb_probe_losses": [0.5, 0.4, 0.3],
    }
    _save_training_checkpoint(
        path,
        model=model,
        colorizer=colorizer,
        optimizer=optimizer,
        cfg={"train": {"steps": 3}},
        row=row,
    )

    loaded_model = torch.nn.Linear(2, 3)
    loaded_colorizer = torch.nn.Linear(3, 1)
    loaded_optimizer = torch.optim.Adam([*loaded_model.parameters(), *loaded_colorizer.parameters()], lr=0.01)
    state = _load_training_checkpoint(
        path,
        model=loaded_model,
        colorizer=loaded_colorizer,
        optimizer=loaded_optimizer,
        device=torch.device("cpu"),
        resume_optimizer=True,
    )

    assert state == {
        "path": str(path),
        "loaded": True,
        "colorizer_loaded": True,
        "optimizer_loaded": True,
        "optimizer_lrs_loaded": [0.01],
        "steps": 3,
    }
    torch.testing.assert_close(loaded_model(x), model(x))
    torch.testing.assert_close(loaded_colorizer(loaded_model(x)), colorizer(model(x)))
    assert len(loaded_optimizer.state) == len(optimizer.state)


def test_training_checkpoint_can_skip_colorizer_for_probe_init(tmp_path) -> None:
    torch.manual_seed(31)
    model = torch.nn.Linear(2, 3)
    checkpoint_colorizer = torch.nn.Linear(3, 1)
    optimizer = torch.optim.Adam([*model.parameters(), *checkpoint_colorizer.parameters()], lr=0.01)
    row = {
        "steps": 1,
        "losses": [1.0],
        "rgb_losses": [],
        "feature_target_losses": [1.0],
        "rgb_probe_losses": [],
    }
    checkpoint_path = tmp_path / "star_feature_overfit.pt"
    _save_training_checkpoint(
        checkpoint_path,
        model=model,
        colorizer=checkpoint_colorizer,
        optimizer=optimizer,
        cfg={"train": {"steps": 1}},
        row=row,
    )

    loaded_model = torch.nn.Linear(2, 3)
    fresh_colorizer = torch.nn.Linear(3, 1)
    with torch.no_grad():
        fresh_colorizer.weight.fill_(7.0)
        fresh_colorizer.bias.fill_(3.0)
    loaded_optimizer = torch.optim.Adam([*loaded_model.parameters(), *fresh_colorizer.parameters()], lr=0.02)
    state = _load_training_checkpoint(
        checkpoint_path,
        model=loaded_model,
        colorizer=fresh_colorizer,
        optimizer=loaded_optimizer,
        device=torch.device("cpu"),
        resume_optimizer=False,
        resume_colorizer=False,
    )

    assert state["colorizer_loaded"] is False
    assert state["optimizer_loaded"] is False
    torch.testing.assert_close(loaded_model.weight, model.weight)
    assert torch.all(fresh_colorizer.weight == 7.0)


def test_colorizer_init_checkpoint_loads_probe_weights(tmp_path) -> None:
    probe_colorizer = torch.nn.Linear(3, 1)
    with torch.no_grad():
        probe_colorizer.weight.fill_(2.0)
        probe_colorizer.bias.fill_(-1.0)
    probe_path = tmp_path / "probe.pt"
    torch.save({"colorizer": probe_colorizer.state_dict()}, probe_path)

    target_colorizer = torch.nn.Linear(3, 1)
    state = _load_colorizer_init_checkpoint(
        probe_path,
        colorizer=target_colorizer,
        device=torch.device("cpu"),
    )

    assert state == {"path": str(probe_path), "loaded": True}
    torch.testing.assert_close(target_colorizer.weight, probe_colorizer.weight)
    torch.testing.assert_close(target_colorizer.bias, probe_colorizer.bias)
