from __future__ import annotations

from pathlib import Path

import powerfoam_training_data as training_data
import pytest
import torch
from camera import (
    CameraSpec,
    build_camera_rays,
    build_camera_rays_at_pixels,
    build_look_at_camera_to_world,
)
from multicam_video_data import MulticamVideoFrameSource
from powerfoam_track_staging import (
    AffineRayProgramUnavailableError,
    PowerFoamTrackStagingPlan,
)
from powerfoam_training_data import (
    PowerFoamRayProvider,
    PowerFoamTargetProvider,
    VideoSeekPowerFoamTargetSource,
)


def _frames(*, views: int = 2, frames: int = 3, height: int = 6, width: int = 8) -> torch.Tensor:
    values = torch.arange(views * frames * 3 * height * width, dtype=torch.float32)
    return values.reshape(views, frames, 3, height, width).div(float(values.numel()))


def _static_cameras(*, views: int = 2, frames: int = 3) -> tuple[tuple[CameraSpec, ...], ...]:
    return tuple(
        tuple(
            CameraSpec(
                fx=8.0,
                fy=6.0,
                cx=4.0,
                cy=3.0,
                camera_to_world=build_look_at_camera_to_world(
                    torch.tensor([0.25 * view, -0.1 * view, -1.0], dtype=torch.float32)
                ),
                lens_model="radial_tangential" if view else "pinhole",
                distortion=(0.01, -0.02, 0.001, -0.002, 0.003) if view else None,
            )
            for _frame in range(frames)
        )
        for view in range(views)
    )


def _providers() -> tuple[PowerFoamTargetProvider, PowerFoamRayProvider, torch.Tensor]:
    frames = _frames()
    device = torch.device("cpu")
    return (
        PowerFoamTargetProvider.from_resident_frames(frames, device=device),
        PowerFoamRayProvider(_static_cameras(), height=6, width=8, device=device),
        frames,
    )


@pytest.mark.parametrize(
    ("lens_model", "distortion"),
    [
        ("pinhole", None),
        ("radial_tangential", (0.01, -0.02, 0.001, -0.002, 0.003)),
        ("opencv_fisheye", (0.01, -0.02, 0.003, -0.004)),
    ],
)
def test_selected_pixel_camera_math_is_bit_exact_with_full_grid_for_supported_lenses(
    lens_model: str,
    distortion: tuple[float, ...] | None,
) -> None:
    camera = CameraSpec(
        fx=8.0,
        fy=6.0,
        cx=4.0,
        cy=3.0,
        camera_to_world=build_look_at_camera_to_world(torch.tensor([0.2, -0.1, -1.0])),
        lens_model=lens_model,
        distortion=distortion,
    )
    pixels = torch.tensor([47, 0, 13, 21])
    full_origins, full_directions = build_camera_rays(
        camera,
        6,
        8,
        device=torch.device("cpu"),
        pixel_center=0.5,
    )
    origins, directions = build_camera_rays_at_pixels(
        camera,
        pixels,
        height=6,
        width=8,
        device=torch.device("cpu"),
        pixel_center=0.5,
    )

    torch.testing.assert_close(origins, full_origins.reshape(-1, 3)[pixels], rtol=0.0, atol=0.0)
    torch.testing.assert_close(directions, full_directions.reshape(-1, 3)[pixels], rtol=0.0, atol=0.0)


def test_track_stage_is_exact_gather_of_full_target_and_ray_contracts() -> None:
    target_provider, ray_provider, _frames_tensor = _providers()
    pixels = torch.tensor([47, 0, 13, 21], dtype=torch.long)
    samples = torch.tensor([5, 0, 4], dtype=torch.long)
    plan = PowerFoamTrackStagingPlan(
        target_provider,
        ray_provider,
        pixels,
        samples,
    )

    block = plan.stage(require_affine_ray_program=True)
    full_targets = target_provider.select(samples, device=torch.device("cpu"))
    expected_targets = full_targets.reshape(3, 3, -1).index_select(2, pixels).permute(2, 0, 1)
    full_rays = ray_provider.select(samples)
    expected_rays = full_rays.reshape(3, -1, 6).index_select(1, pixels).permute(1, 0, 2)

    torch.testing.assert_close(block.targets, expected_targets, rtol=0.0, atol=0.0)
    torch.testing.assert_close(block.rays, expected_rays, rtol=0.0, atol=0.0)
    assert block.view_indices.tolist() == [1, 0, 1]
    assert block.frame_indices.tolist() == [2, 0, 1]
    assert block.affine_ray_program is not None
    torch.testing.assert_close(block.affine_ray_program.evaluate(), block.rays, rtol=0.0, atol=0.0)
    assert not block.targets.requires_grad
    assert not block.rays.requires_grad


def test_target_only_stage_matches_targets_and_omits_explicit_ray_payload() -> None:
    target_provider, ray_provider, _frames_tensor = _providers()
    pixels = torch.tensor([47, 0, 13, 21], dtype=torch.long)
    samples = torch.tensor([2, 0, 1], dtype=torch.long)
    plan = PowerFoamTrackStagingPlan(
        target_provider,
        ray_provider,
        pixels,
        samples,
    )

    full = plan.stage(require_affine_ray_program=True)
    targets = plan.stage_targets()

    torch.testing.assert_close(targets.targets, full.targets, rtol=0.0, atol=0.0)
    assert not hasattr(targets, "rays")
    assert targets.normalization == full.normalization
    assert targets.accounting["target_bytes"] == 4 * 3 * 3 * 4
    assert targets.accounting["ray_bytes"] == 0
    assert targets.accounting["explicit_rays_staged"] is False
    assert targets.accounting["omitted_explicit_ray_bytes"] == 4 * 3 * 6 * 4
    assert targets.accounting["output_payload_bytes"] == targets.accounting["target_bytes"]

    assert full.affine_ray_program is not None
    coefficients = full.affine_ray_program.coefficients[0]
    plan.assert_fixed_camera_affine_coefficients(coefficients)
    for column in (0, 3):
        mismatched = coefficients.clone()
        mismatched[0, column] += 0.25
        with pytest.raises(ValueError, match="does not match the certified live track rays"):
            plan.assert_fixed_camera_affine_coefficients(mismatched)


def test_rectangular_multiview_samples_move_view_onto_the_native_track_axis() -> None:
    target_provider, ray_provider, _frames_tensor = _providers()
    pixels = torch.tensor([47, 0], dtype=torch.long)
    samples = torch.tensor([5, 0, 4, 1, 3, 2], dtype=torch.long)
    block = PowerFoamTrackStagingPlan(
        target_provider,
        ray_provider,
        pixels,
        samples,
    ).stage(require_affine_ray_program=True)

    view_tracks = block.as_view_tracks()
    assert view_tracks.source_view_indices.tolist() == [0, 0, 1, 1]
    assert view_tracks.source_pixel_indices.tolist() == [47, 0, 47, 0]
    assert view_tracks.frame_indices.tolist() == [0, 1, 2]
    torch.testing.assert_close(
        view_tracks.sample_times,
        torch.tensor([0.0, 0.5, 1.0]),
        rtol=0.0,
        atol=0.0,
    )
    expected_targets = target_provider.select(
        torch.tensor([0, 1, 2, 3, 4, 5]),
        device=torch.device("cpu"),
    ).reshape(2, 3, 3, 6, 8)
    expected_targets = expected_targets.reshape(2, 3, 3, -1).index_select(3, pixels)
    expected_targets = expected_targets.permute(0, 3, 1, 2).reshape(4, 3, 3)
    torch.testing.assert_close(view_tracks.targets, expected_targets, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        view_tracks.rays[..., :3],
        view_tracks.ray_coefficients[:, None, :3].expand_as(view_tracks.rays[..., :3]),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        view_tracks.rays[..., 3:6],
        view_tracks.ray_coefficients[:, None, 6:9].expand_as(view_tracks.rays[..., 3:6]),
        rtol=0.0,
        atol=0.0,
    )
    assert view_tracks.global_track_count == 4
    assert view_tracks.global_sample_count == 3
    assert view_tracks.global_rgb_element_count == 4 * 3 * 3
    assert view_tracks.accounting["denominator_preserved"] is True


def test_unbalanced_view_selection_cannot_claim_a_view_track_factorization() -> None:
    target_provider, ray_provider, _frames_tensor = _providers()
    block = PowerFoamTrackStagingPlan(
        target_provider,
        ray_provider,
        torch.tensor([0, 1]),
        torch.tensor([5, 0, 4]),
    ).stage(require_affine_ray_program=True)

    with pytest.raises(ValueError, match="rectangular view-time grid"):
        block.as_view_tracks()


def test_progressive_track_stage_gathers_after_exact_target_resize_and_scaled_camera_rays() -> None:
    target_provider, ray_provider, _frames_tensor = _providers()
    pixels = torch.tensor([11, 0, 6], dtype=torch.long)
    samples = torch.tensor([4, 0], dtype=torch.long)
    plan = PowerFoamTrackStagingPlan(
        target_provider,
        ray_provider,
        pixels,
        samples,
        height=3,
        width=4,
    )

    block = plan.stage()
    full_targets = target_provider.select(samples, height=3, width=4, device=torch.device("cpu"))
    expected_targets = full_targets.reshape(2, 3, -1).index_select(2, pixels).permute(2, 0, 1)
    full_rays = ray_provider.select(samples, height=3, width=4)
    expected_rays = full_rays.reshape(2, -1, 6).index_select(1, pixels).permute(1, 0, 2)

    torch.testing.assert_close(block.targets, expected_targets, rtol=0.0, atol=0.0)
    torch.testing.assert_close(block.rays, expected_rays, rtol=0.0, atol=0.0)


def test_track_and_sample_partitions_reconstruct_one_stage_and_share_global_normalization() -> None:
    target_provider, ray_provider, _frames_tensor = _providers()
    plan = PowerFoamTrackStagingPlan(
        target_provider,
        ray_provider,
        torch.tensor([47, 0, 13, 21, 7]),
        torch.tensor([5, 0, 4]),
    )
    full = plan.stage()
    partitions = [
        plan.stage(track_start=track_start, track_end=track_end, sample_start=sample_start, sample_end=sample_end)
        for track_start, track_end in ((0, 2), (2, 5))
        for sample_start, sample_end in ((0, 1), (1, 3))
    ]

    reconstructed_targets = torch.cat(
        (
            torch.cat((partitions[0].targets, partitions[1].targets), dim=1),
            torch.cat((partitions[2].targets, partitions[3].targets), dim=1),
        ),
        dim=0,
    )
    reconstructed_rays = torch.cat(
        (
            torch.cat((partitions[0].rays, partitions[1].rays), dim=1),
            torch.cat((partitions[2].rays, partitions[3].rays), dim=1),
        ),
        dim=0,
    )
    torch.testing.assert_close(reconstructed_targets, full.targets, rtol=0.0, atol=0.0)
    torch.testing.assert_close(reconstructed_rays, full.rays, rtol=0.0, atol=0.0)
    assert {part.normalization.global_rgb_element_count for part in partitions} == {5 * 3 * 3}
    assert sum(part.normalization.block_fraction for part in partitions) == pytest.approx(1.0)
    full_normalized_sum = full.targets.square().sum() / full.normalization.global_rgb_element_count
    partitioned_normalized_sum = sum(
        part.targets.square().sum() / part.normalization.global_rgb_element_count for part in partitions
    )
    torch.testing.assert_close(partitioned_normalized_sum, full_normalized_sum, rtol=1.0e-6, atol=1.0e-7)


def test_video_seek_track_stage_preserves_logical_to_native_frame_mapping_and_one_frame_residency(
    tmp_path,
    monkeypatch,
) -> None:
    paths = tuple(tmp_path / f"cam{view}.mp4" for view in range(2))
    for path in paths:
        path.touch()
    frame_sources = tuple(
        MulticamVideoFrameSource(
            camera_name=f"cam{view}",
            video_path=path,
            start_seconds=0.0,
            sample_fps=30.0,
            source_frame_count=10,
            selected_frame_indices=(2, 5, 9),
            height=6,
            width=8,
        )
        for view, path in enumerate(paths)
    )
    calls: list[tuple[str, tuple[int, ...]]] = []

    def decode_selected(**kwargs) -> torch.Tensor:
        camera_name = Path(kwargs["video_path"]).stem
        native_frames = tuple(kwargs["sample_indices"])
        calls.append((camera_name, native_frames))
        view = int(camera_name.removeprefix("cam"))
        return torch.stack(
            [torch.full((3, 6, 8), view * 100 + frame, dtype=torch.float32).div(255.0) for frame in native_frames]
        )

    monkeypatch.setattr(training_data, "load_multicam_val_selected_camera_frames", decode_selected)
    device = torch.device("cpu")
    target_provider = PowerFoamTargetProvider(
        source=VideoSeekPowerFoamTargetSource(frame_sources),
        device=device,
    )
    ray_provider = PowerFoamRayProvider(_static_cameras(), height=6, width=8, device=device)
    block = PowerFoamTrackStagingPlan(
        target_provider,
        ray_provider,
        torch.tensor([47, 1]),
        torch.tensor([5, 1, 3]),
    ).stage()

    assert calls == [("cam1", (9,)), ("cam0", (5,)), ("cam1", (2,))]
    torch.testing.assert_close(
        block.targets[:, :, 0],
        torch.tensor([[109, 5, 102], [109, 5, 102]], dtype=torch.float32).div(255.0),
        rtol=0.0,
        atol=0.0,
    )
    assert block.accounting["peak_decoded_frame_count"] == 1
    assert block.accounting["target_bytes"] == 2 * 3 * 3 * 4
    assert block.accounting["ray_bytes"] == 2 * 3 * 6 * 4
    assert block.accounting["full_image_accelerator_resident_bytes"] == 0
    assert block.accounting["source_residency"]["resident_bytes"] == 0
    assert block.accounting["source_residency"]["source_kind"] == "video_seek_mp4"
    assert block.accounting["bounded_residency_contract"] is True


@pytest.mark.parametrize(
    ("pixels", "samples", "error", "message"),
    [
        (torch.tensor([1, 1]), torch.tensor([0]), ValueError, "pixel_indices must be unique"),
        (torch.tensor([1]), torch.tensor([0, 0]), ValueError, "sample_indices must be unique"),
        (torch.tensor([48]), torch.tensor([0]), IndexError, "pixel_indices value 48"),
        (torch.tensor([1]), torch.tensor([6]), IndexError, "sample_indices value 6"),
        (torch.tensor([1.5]), torch.tensor([0]), ValueError, "pixel_indices must contain integer ids"),
    ],
)
def test_track_plan_rejects_duplicate_and_out_of_range_ids(
    pixels: torch.Tensor,
    samples: torch.Tensor,
    error: type[Exception],
    message: str,
) -> None:
    target_provider, ray_provider, _frames_tensor = _providers()
    with pytest.raises(error, match=message):
        PowerFoamTrackStagingPlan(target_provider, ray_provider, pixels, samples)


def test_track_plan_rejects_invalid_partition_bounds() -> None:
    target_provider, ray_provider, _frames_tensor = _providers()
    plan = PowerFoamTrackStagingPlan(
        target_provider,
        ray_provider,
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
    )

    with pytest.raises(IndexError, match="track partition"):
        plan.stage(track_end=3)
    with pytest.raises(ValueError, match="sample partition"):
        plan.stage(sample_start=1, sample_end=1)


def test_track_stage_rejects_camera_gradients_instead_of_silently_detaching() -> None:
    frames = _frames(views=1, frames=1)
    camera = CameraSpec(
        fx=8.0,
        fy=6.0,
        cx=4.0,
        cy=3.0,
        camera_to_world=torch.eye(4, requires_grad=True),
    )
    device = torch.device("cpu")
    plan = PowerFoamTrackStagingPlan(
        PowerFoamTargetProvider.from_resident_frames(frames, device=device),
        PowerFoamRayProvider(((camera,),), height=6, width=8, device=device),
        torch.tensor([0]),
        torch.tensor([0]),
    )

    with pytest.raises(ValueError, match="rejects camera gradients"):
        plan.stage()
    with pytest.raises(ValueError, match="rejects camera gradients"):
        plan.assert_fixed_camera_affine_coefficients(
            torch.zeros((1, 12), dtype=torch.float32)
        )


def test_moving_camera_exact_rays_remain_available_but_affine_program_fails_closed() -> None:
    frames = _frames(views=1, frames=2)
    first = CameraSpec(8.0, 6.0, 4.0, 3.0, torch.eye(4))
    moved_transform = torch.eye(4)
    moved_transform[0, 3] = 0.25
    second = CameraSpec(8.0, 6.0, 4.0, 3.0, moved_transform)
    device = torch.device("cpu")
    plan = PowerFoamTrackStagingPlan(
        PowerFoamTargetProvider.from_resident_frames(frames, device=device),
        PowerFoamRayProvider(((first, second),), height=6, width=8, device=device),
        torch.tensor([0, 47]),
        torch.tensor([0, 1]),
    )

    block = plan.stage()
    assert block.affine_ray_program is None
    assert "piecewise-affine/projective camera-gauge compiler" in block.affine_ray_program_unavailable_reason
    with pytest.raises(
        AffineRayProgramUnavailableError, match="approximate endpoint fitting is intentionally disabled"
    ):
        plan.stage(require_affine_ray_program=True)
    with pytest.raises(
        AffineRayProgramUnavailableError, match="approximate endpoint fitting is intentionally disabled"
    ):
        plan.assert_fixed_camera_affine_coefficients(torch.zeros((2, 12), dtype=torch.float32))
