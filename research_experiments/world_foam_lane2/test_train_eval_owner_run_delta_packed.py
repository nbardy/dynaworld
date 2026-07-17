from __future__ import annotations

from functools import lru_cache
import unittest
from types import SimpleNamespace

import torch

from train_eval_owner_run_tape import (
    DEFAULT_CONFIG,
    DELTA_I16X3_FRAMEGROUP16_MATERIALIZED_MODE,
    DELTA_PACKED_FRAMEGROUP16_MATERIALIZED_MODE,
    DELTA_PACKED_FRAMEGROUP16_MODE,
    DELTA_PACKED_FRAMEGROUP16_RECOMPUTE_MODE,
    DELTA_PACKED_FRAMEGROUP16_SMALLRUN16_MODE,
    DELTA_PACKED_SCALAR_MODE,
    OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
    OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE,
    OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
    OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
    OWNER_RUN_FUSED_MSE_NOMID_MODE,
    RealRayReplayConfig,
    SyntheticRayMotion,
    _delta_replace_coeff16_fused_mse_loss_vjp,
    _fit_loaded_frame_count,
    _build_delta_frame_bitmask_i32,
    _build_delta_frame_select_i16,
    _build_owner_run_delta_replace_native_cutwalk_tape,
    _load_config,
    _pack_endpoint_records_i32,
    _packed_endpoint_direct_config_validation_marker,
    _prepare_owner_run_tapes,
    _segment_tape_fused_mse_loss_vjp,
    _track_major_rgb_from_image,
    apply_synthetic_ray_motion,
    initialize_sites_from_train_samples,
    load_powerfoam_training_data,
    make_boundaries_4d,
)
from probe_endpoint_record_delta_replay import pack_endpoint_record_delta_replace_tape
from probe_owner_run_boundary_tape import _build_owner_run_sequences


@lru_cache(maxsize=None)
def _cached_loaded_training_frames(
    *,
    frame_count: int,
    render_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg = _load_config(DEFAULT_CONFIG, max_frames=frame_count, render_size=render_size)
    data = load_powerfoam_training_data(cfg, torch.device("cpu"))
    targets = data["targets"].detach().cpu().to(dtype=torch.float32)
    rays = data["sample_rays"].detach().cpu().to(dtype=torch.float32)
    frame_indices = data["sample_frame_indices"].detach().cpu().to(dtype=torch.long)
    targets, rays, frame_indices, _repeated = _fit_loaded_frame_count(
        split_name="train",
        targets=targets,
        rays=rays,
        frame_indices=frame_indices,
        loaded_frame_count=int(data["frame_count"]),
        requested_frame_count=frame_count,
        allow_repeat_loaded_frames=False,
    )
    return targets.contiguous(), rays.contiguous(), frame_indices.contiguous()


def _common_moving_ray_fixture(
    *,
    frame_count: int,
    render_size: int = 16,
    site_count: int = 8,
) -> tuple[dict[str, object], torch.Tensor]:
    near = 0.0
    far = 3.5
    density = 8.0
    invalid_epsilon = 1.0e-7
    transmittance_threshold = 1.0e-4
    targets, rays, frame_indices = (
        tensor.clone()
        for tensor in _cached_loaded_training_frames(
            frame_count=frame_count,
            render_size=render_size,
        )
    )
    rays = apply_synthetic_ray_motion(
        rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
        motion=SyntheticRayMotion(
            origin_velocity=(0.02, 0.0, 0.0),
            direction_velocity=(0.0, 0.01, 0.0),
        ),
    )
    sites = initialize_sites_from_train_samples(
        targets=targets,
        rays=rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
        site_count=site_count,
        near=near,
        far=far,
        density=density,
    )
    site_rgba_cpu = torch.tensor([site.rgba for site in sites], dtype=torch.float32)
    return (
        {
            "sites": sites,
            "rays": rays,
            "frame_indices": frame_indices,
            "frame_count": frame_count,
            "near": near,
            "far": far,
            "invalid_epsilon": invalid_epsilon,
            "transmittance_threshold": transmittance_threshold,
            "site_rgba": site_rgba_cpu,
        },
        targets,
    )


def _duplicate_fixture_with_shifted_second_view(
    *,
    common_kwargs: dict[str, object],
    targets: torch.Tensor,
) -> tuple[dict[str, object], torch.Tensor]:
    rays = common_kwargs["rays"].clone()
    shifted = rays.clone()
    shifted[..., 0] += 0.075
    shifted[..., 1] -= 0.025
    shifted[..., 3] += 0.003
    shifted[..., 4] -= 0.002
    return (
        {
            **common_kwargs,
            "rays": torch.cat((rays, shifted), dim=0).contiguous(),
            "frame_indices": torch.cat(
                (
                    common_kwargs["frame_indices"],
                    common_kwargs["frame_indices"],
                ),
                dim=0,
            ).contiguous(),
        },
        torch.cat((targets, targets), dim=0).contiguous(),
    )


def _synthetic_moving_ray_fixture(
    *,
    frame_count: int,
    render_size: int = 4,
    site_count: int = 6,
) -> tuple[dict[str, object], torch.Tensor]:
    near = 0.0
    far = 3.5
    density = 8.0
    invalid_epsilon = 1.0e-7
    transmittance_threshold = 1.0e-4
    coords = torch.linspace(-0.5, 0.5, render_size, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    directions = torch.stack((xx, yy, torch.ones_like(xx)), dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    origins = torch.zeros_like(directions)
    rays = torch.cat((origins, directions), dim=-1).unsqueeze(0).repeat(frame_count, 1, 1, 1)
    frame_indices = torch.arange(frame_count, dtype=torch.long)
    rays = apply_synthetic_ray_motion(
        rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
        motion=SyntheticRayMotion(
            origin_velocity=(0.018, -0.006, 0.0),
            direction_velocity=(0.002, 0.004, 0.0),
        ),
    )
    frame_t = torch.linspace(0.0, 1.0, frame_count, dtype=torch.float32).view(frame_count, 1, 1)
    targets = torch.stack(
        (
            (xx.unsqueeze(0) + 0.5 + 0.25 * frame_t).fmod(1.0),
            (yy.unsqueeze(0) + 0.5 + 0.15 * frame_t).fmod(1.0),
            torch.full((frame_count, render_size, render_size), 0.35, dtype=torch.float32) + 0.2 * frame_t,
        ),
        dim=1,
    ).contiguous()
    sites = initialize_sites_from_train_samples(
        targets=targets,
        rays=rays,
        frame_indices=frame_indices,
        frame_count=frame_count,
        site_count=site_count,
        near=near,
        far=far,
        density=density,
    )
    site_rgba_cpu = torch.tensor([site.rgba for site in sites], dtype=torch.float32)
    return (
        {
            "sites": sites,
            "rays": rays,
            "frame_indices": frame_indices,
            "frame_count": frame_count,
            "near": near,
            "far": far,
            "invalid_epsilon": invalid_epsilon,
            "transmittance_threshold": transmittance_threshold,
            "site_rgba": site_rgba_cpu,
        },
        targets,
    )


class SiteInitializationTests(unittest.TestCase):
    @staticmethod
    def _pixel_identity_rays(
        *,
        sample_count: int = 4,
        frame_count: int = 4,
        height: int = 9,
        width: int = 9,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        yy, xx = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing="ij",
        )
        origins = torch.stack((xx, yy, torch.zeros_like(xx)), dim=-1)
        directions = torch.zeros_like(origins)
        directions[..., 2] = 1.0
        rays = torch.cat((origins, directions), dim=-1).unsqueeze(0).repeat(sample_count, 1, 1, 1)
        frame_indices = torch.arange(sample_count, dtype=torch.long) % int(frame_count)
        targets = torch.stack(
            (
                xx.unsqueeze(0).repeat(sample_count, 1, 1) / float(max(width - 1, 1)),
                yy.unsqueeze(0).repeat(sample_count, 1, 1) / float(max(height - 1, 1)),
                frame_indices.to(dtype=torch.float32).view(sample_count, 1, 1).expand(sample_count, height, width)
                / float(max(frame_count - 1, 1)),
            ),
            dim=1,
        ).contiguous()
        return targets, rays.contiguous(), frame_indices

    def test_default_site_initialization_preserves_legacy_sparse_layout(self) -> None:
        targets, rays, frame_indices = self._pixel_identity_rays()

        default_sites = initialize_sites_from_train_samples(
            targets=targets,
            rays=rays,
            frame_indices=frame_indices,
            frame_count=4,
            site_count=9,
            near=0.0,
            far=1.0,
            density=2.0,
        )
        explicit_legacy_sites = initialize_sites_from_train_samples(
            targets=targets,
            rays=rays,
            frame_indices=frame_indices,
            frame_count=4,
            site_count=9,
            near=0.0,
            far=1.0,
            density=2.0,
            initialization="legacy_sparse",
        )

        self.assertEqual(default_sites, explicit_legacy_sites)

    def test_stratified_grid_site_initialization_covers_image_cells(self) -> None:
        targets, rays, frame_indices = self._pixel_identity_rays()

        sites = initialize_sites_from_train_samples(
            targets=targets,
            rays=rays,
            frame_indices=frame_indices,
            frame_count=4,
            site_count=9,
            near=0.0,
            far=1.0,
            density=2.0,
            initialization="stratified_grid",
        )

        observed_xy = {(round(site.x), round(site.y)) for site in sites}
        self.assertEqual(
            observed_xy,
            {
                (1, 1),
                (4, 1),
                (7, 1),
                (1, 4),
                (4, 4),
                (7, 4),
                (1, 7),
                (4, 7),
                (7, 7),
            },
        )
        self.assertEqual({round(site.x) for site in sites}, {1, 4, 7})
        self.assertEqual({round(site.y) for site in sites}, {1, 4, 7})

    def test_legacy_pixel_mean_keeps_legacy_geometry_but_averages_color_over_samples(self) -> None:
        targets, rays, frame_indices = self._pixel_identity_rays()

        legacy_sites = initialize_sites_from_train_samples(
            targets=targets,
            rays=rays,
            frame_indices=frame_indices,
            frame_count=4,
            site_count=9,
            near=0.0,
            far=1.0,
            density=2.0,
            initialization="legacy_sparse",
        )
        pixel_mean_sites = initialize_sites_from_train_samples(
            targets=targets,
            rays=rays,
            frame_indices=frame_indices,
            frame_count=4,
            site_count=9,
            near=0.0,
            far=1.0,
            density=2.0,
            initialization="legacy_pixel_mean",
        )

        for site_id, (legacy, pixel_mean) in enumerate(zip(legacy_sites, pixel_mean_sites, strict=True)):
            self.assertEqual((pixel_mean.x, pixel_mean.y, pixel_mean.z, pixel_mean.t), (legacy.x, legacy.y, legacy.z, legacy.t))
            y = (site_id * 7 + site_id // 9) % 9
            x = (site_id * 11 + site_id // 9) % 9
            expected_rgb = targets[:, :, y, x].mean(dim=0)
            self.assertAlmostEqual(pixel_mean.rgba[0], float(expected_rgb[0].item()), places=6)
            self.assertAlmostEqual(pixel_mean.rgba[1], float(expected_rgb[1].item()), places=6)
            self.assertAlmostEqual(pixel_mean.rgba[2], float(expected_rgb[2].item()), places=6)
            self.assertEqual(pixel_mean.rgba[3], legacy.rgba[3])

    def test_legacy_frame_pixel_mean_averages_color_only_within_site_frame(self) -> None:
        targets, rays, frame_indices = self._pixel_identity_rays(sample_count=8, frame_count=4)
        targets[:, 0, :, :] = torch.arange(8, dtype=torch.float32).view(8, 1, 1) / 7.0

        legacy_sites = initialize_sites_from_train_samples(
            targets=targets,
            rays=rays,
            frame_indices=frame_indices,
            frame_count=4,
            site_count=9,
            near=0.0,
            far=1.0,
            density=2.0,
            initialization="legacy_sparse",
        )
        frame_mean_sites = initialize_sites_from_train_samples(
            targets=targets,
            rays=rays,
            frame_indices=frame_indices,
            frame_count=4,
            site_count=9,
            near=0.0,
            far=1.0,
            density=2.0,
            initialization="legacy_frame_pixel_mean",
        )

        for site_id, (legacy, frame_mean) in enumerate(zip(legacy_sites, frame_mean_sites, strict=True)):
            self.assertEqual(
                (frame_mean.x, frame_mean.y, frame_mean.z, frame_mean.t),
                (legacy.x, legacy.y, legacy.z, legacy.t),
            )
            y = (site_id * 7 + site_id // 9) % 9
            x = (site_id * 11 + site_id // 9) % 9
            site_frame = round(frame_mean.t * 3)
            expected_rgb = targets[frame_indices == site_frame, :, y, x].mean(dim=0)
            all_frame_rgb = targets[:, :, y, x].mean(dim=0)
            self.assertAlmostEqual(frame_mean.rgba[0], float(expected_rgb[0].item()), places=6)
            self.assertNotAlmostEqual(frame_mean.rgba[0], float(all_frame_rgb[0].item()), places=6)
            self.assertAlmostEqual(frame_mean.rgba[1], float(expected_rgb[1].item()), places=6)
            self.assertAlmostEqual(frame_mean.rgba[2], float(expected_rgb[2].item()), places=6)
            self.assertEqual(frame_mean.rgba[3], legacy.rgba[3])

    def test_legacy_frame_patch3_mean_averages_same_frame_local_patch(self) -> None:
        targets, rays, frame_indices = self._pixel_identity_rays(sample_count=8, frame_count=4)
        patch_signal = (torch.arange(81, dtype=torch.float32).view(1, 1, 9, 9) / 80.0).square()
        frame_signal = torch.arange(8, dtype=torch.float32).view(8, 1, 1, 1) / 7.0
        targets[:, 0:1, :, :] = (patch_signal + frame_signal).clamp(0.0, 1.0)

        frame_mean_sites = initialize_sites_from_train_samples(
            targets=targets,
            rays=rays,
            frame_indices=frame_indices,
            frame_count=4,
            site_count=9,
            near=0.0,
            far=1.0,
            density=2.0,
            initialization="legacy_frame_pixel_mean",
        )
        patch_mean_sites = initialize_sites_from_train_samples(
            targets=targets,
            rays=rays,
            frame_indices=frame_indices,
            frame_count=4,
            site_count=9,
            near=0.0,
            far=1.0,
            density=2.0,
            initialization="legacy_frame_patch3_mean",
        )

        for site_id, (frame_mean, patch_mean) in enumerate(zip(frame_mean_sites, patch_mean_sites, strict=True)):
            self.assertEqual(
                (patch_mean.x, patch_mean.y, patch_mean.z, patch_mean.t),
                (frame_mean.x, frame_mean.y, frame_mean.z, frame_mean.t),
            )
            y = (site_id * 7 + site_id // 9) % 9
            x = (site_id * 11 + site_id // 9) % 9
            y0 = max(y - 1, 0)
            y1 = min(y + 2, 9)
            x0 = max(x - 1, 0)
            x1 = min(x + 2, 9)
            site_frame = round(patch_mean.t * 3)
            expected_rgb = targets[frame_indices == site_frame, :, y0:y1, x0:x1].mean(dim=(0, 2, 3))
            self.assertAlmostEqual(patch_mean.rgba[0], float(expected_rgb[0].item()), places=6)
            self.assertNotAlmostEqual(patch_mean.rgba[0], frame_mean.rgba[0], places=6)
            self.assertAlmostEqual(patch_mean.rgba[1], float(expected_rgb[1].item()), places=6)
            self.assertAlmostEqual(patch_mean.rgba[2], float(expected_rgb[2].item()), places=6)
            self.assertEqual(patch_mean.rgba[3], frame_mean.rgba[3])

    def test_stratified_pixel_mean_combines_grid_geometry_with_sample_mean_color(self) -> None:
        targets, rays, frame_indices = self._pixel_identity_rays()

        stratified = initialize_sites_from_train_samples(
            targets=targets,
            rays=rays,
            frame_indices=frame_indices,
            frame_count=4,
            site_count=9,
            near=0.0,
            far=1.0,
            density=2.0,
            initialization="stratified_grid",
        )
        stratified_mean = initialize_sites_from_train_samples(
            targets=targets,
            rays=rays,
            frame_indices=frame_indices,
            frame_count=4,
            site_count=9,
            near=0.0,
            far=1.0,
            density=2.0,
            initialization="stratified_pixel_mean",
        )

        for plain, pixel_mean in zip(stratified, stratified_mean, strict=True):
            self.assertEqual(
                (pixel_mean.x, pixel_mean.y, pixel_mean.z, pixel_mean.t),
                (plain.x, plain.y, plain.z, plain.t),
            )
            x = round(pixel_mean.x)
            y = round(pixel_mean.y)
            expected_rgb = targets[:, :, y, x].mean(dim=0)
            self.assertAlmostEqual(pixel_mean.rgba[0], float(expected_rgb[0].item()), places=6)
            self.assertAlmostEqual(pixel_mean.rgba[1], float(expected_rgb[1].item()), places=6)
            self.assertAlmostEqual(pixel_mean.rgba[2], float(expected_rgb[2].item()), places=6)
            self.assertEqual(pixel_mean.rgba[3], plain.rgba[3])

    def test_unknown_site_initialization_rejected(self) -> None:
        targets, rays, frame_indices = self._pixel_identity_rays()

        with self.assertRaisesRegex(ValueError, "initialization must be one of"):
            initialize_sites_from_train_samples(
                targets=targets,
                rays=rays,
                frame_indices=frame_indices,
                frame_count=4,
                site_count=9,
                near=0.0,
                far=1.0,
                density=2.0,
                initialization="diagonal_only",
            )


class NativeOwnerRunCutwalkCpuTests(unittest.TestCase):
    def test_framebitmask_supports_frame31_signed_int32_payload_for_32_frames(self) -> None:
        delta = SimpleNamespace(
            track_change_offsets_i32=torch.tensor([0, 1], dtype=torch.int32),
            change_frame_i32=torch.tensor([31], dtype=torch.int32),
        )

        mask = _build_delta_frame_bitmask_i32(delta, frame_count=32)

        self.assertEqual(mask.dtype, torch.int32)
        self.assertEqual(int(mask.item()), -(1 << 31))
        self.assertEqual(int(mask.to(dtype=torch.int64).item()) & 0xFFFFFFFF, 1 << 31)

    def test_framebitmask_still_rejects_more_than_32_frames(self) -> None:
        delta = SimpleNamespace(
            track_change_offsets_i32=torch.tensor([0, 0], dtype=torch.int32),
            change_frame_i32=torch.empty((0,), dtype=torch.int32),
        )

        with self.assertRaisesRegex(ValueError, "frame_count <= 32"):
            _build_delta_frame_bitmask_i32(delta, frame_count=33)

    def test_framebitmask_rejects_unsorted_change_frames_per_track(self) -> None:
        delta = SimpleNamespace(
            track_change_offsets_i32=torch.tensor([0, 2], dtype=torch.int32),
            change_frame_i32=torch.tensor([3, 2], dtype=torch.int32),
        )

        with self.assertRaisesRegex(ValueError, "strictly ascending"):
            _build_delta_frame_bitmask_i32(delta, frame_count=4)

    def test_framebitmask_rejects_empty_change_offsets(self) -> None:
        delta = SimpleNamespace(
            track_change_offsets_i32=torch.empty((0,), dtype=torch.int32),
            change_frame_i32=torch.empty((0,), dtype=torch.int32),
        )

        with self.assertRaisesRegex(ValueError, "at least one offset"):
            _build_delta_frame_bitmask_i32(delta, frame_count=4)

    def test_framebitmask_rejects_nonzero_first_change_offset(self) -> None:
        delta = SimpleNamespace(
            track_change_offsets_i32=torch.tensor([1, 1], dtype=torch.int32),
            change_frame_i32=torch.tensor([2], dtype=torch.int32),
        )

        with self.assertRaisesRegex(ValueError, "start at 0"):
            _build_delta_frame_bitmask_i32(delta, frame_count=4)

    def test_framebitmask_rejects_nonmonotonic_change_offsets(self) -> None:
        delta = SimpleNamespace(
            track_change_offsets_i32=torch.tensor([0, 2, 1], dtype=torch.int32),
            change_frame_i32=torch.tensor([1, 2], dtype=torch.int32),
        )

        with self.assertRaisesRegex(ValueError, "must be monotonic"):
            _build_delta_frame_bitmask_i32(delta, frame_count=4)

    def test_framebitmask_rejects_change_offset_final_mismatch(self) -> None:
        delta = SimpleNamespace(
            track_change_offsets_i32=torch.tensor([0, 2], dtype=torch.int32),
            change_frame_i32=torch.tensor([1], dtype=torch.int32),
        )

        with self.assertRaisesRegex(ValueError, "final offset"):
            _build_delta_frame_bitmask_i32(delta, frame_count=4)

    def test_frameselect_rejects_unsorted_change_frames_per_track(self) -> None:
        delta = SimpleNamespace(
            track_change_offsets_i32=torch.tensor([0, 2], dtype=torch.int32),
            change_frame_i32=torch.tensor([3, 2], dtype=torch.int32),
        )

        with self.assertRaisesRegex(ValueError, "strictly ascending"):
            _build_delta_frame_select_i16(delta, frame_count=4)

    def test_frameselect_rejects_frame0_change(self) -> None:
        delta = SimpleNamespace(
            track_change_offsets_i32=torch.tensor([0, 1], dtype=torch.int32),
            change_frame_i32=torch.tensor([0], dtype=torch.int32),
        )

        with self.assertRaisesRegex(ValueError, r"expected in \[1, 4\)"):
            _build_delta_frame_select_i16(delta, frame_count=4)

    def test_frameselect_rejects_non1d_change_offsets(self) -> None:
        delta = SimpleNamespace(
            track_change_offsets_i32=torch.tensor([[0, 0]], dtype=torch.int32),
            change_frame_i32=torch.empty((0,), dtype=torch.int32),
        )

        with self.assertRaisesRegex(ValueError, "must be 1D"):
            _build_delta_frame_select_i16(delta, frame_count=4)

    def _assert_native_delta_matches_python(
        self,
        *,
        common_kwargs: dict[str, object],
        frame_count: int,
    ) -> None:
        boundaries = make_boundaries_4d(common_kwargs["sites"])
        sequences, _sample_meta = _build_owner_run_sequences(
            sites=common_kwargs["sites"],
            boundaries=boundaries,
            rays=common_kwargs["rays"],
            frame_indices=common_kwargs["frame_indices"],
            frame_count=frame_count,
            near=common_kwargs["near"],
            far=common_kwargs["far"],
            invalid_epsilon=common_kwargs["invalid_epsilon"],
            transmittance_threshold=common_kwargs["transmittance_threshold"],
            site_rgba=common_kwargs["site_rgba"],
            include_sample_meta=False,
        )
        python_delta = pack_endpoint_record_delta_replace_tape(sequences, frame_count=frame_count)
        native_delta = _build_owner_run_delta_replace_native_cutwalk_tape(
            sites=common_kwargs["sites"],
            boundaries=boundaries,
            rays=common_kwargs["rays"],
            frame_indices=common_kwargs["frame_indices"],
            frame_count=frame_count,
            near=common_kwargs["near"],
            far=common_kwargs["far"],
            invalid_epsilon=common_kwargs["invalid_epsilon"],
            transmittance_threshold=common_kwargs["transmittance_threshold"],
            site_rgba=common_kwargs["site_rgba"],
        )
        for field in (
            "base_offsets_i32",
            "base_owner_i32",
            "base_left_i32",
            "base_right_i32",
            "track_change_offsets_i32",
            "change_frame_i32",
            "change_offsets_i32",
            "change_owner_i32",
            "change_left_i32",
            "change_right_i32",
        ):
            self.assertTrue(
                torch.equal(getattr(native_delta, field), getattr(python_delta, field)),
                msg=f"native owner-run cutwalk differs from Python sequence delta for {field}",
            )

    def test_native_cutwalk_delta_matches_python_owner_run_sequences(self) -> None:
        frame_count = 4
        common_kwargs, _targets = _common_moving_ray_fixture(frame_count=frame_count, render_size=8, site_count=6)
        self._assert_native_delta_matches_python(common_kwargs=common_kwargs, frame_count=frame_count)

    def test_native_cutwalk_delta_matches_python_for_multiview_moving_rays(self) -> None:
        frame_count = 4
        common_kwargs, targets = _common_moving_ray_fixture(frame_count=frame_count, render_size=6, site_count=6)
        common_kwargs, _targets = _duplicate_fixture_with_shifted_second_view(
            common_kwargs=common_kwargs,
            targets=targets,
        )

        self._assert_native_delta_matches_python(common_kwargs=common_kwargs, frame_count=frame_count)

    def test_native_cutwalk_delta_matches_python_at_32_frame_boundary(self) -> None:
        frame_count = 32
        common_kwargs, _targets = _synthetic_moving_ray_fixture(frame_count=frame_count)

        self.assertEqual(int(common_kwargs["frame_indices"].max().item()), 31)
        self.assertEqual(int(common_kwargs["frame_indices"].unique().numel()), frame_count)
        self._assert_native_delta_matches_python(common_kwargs=common_kwargs, frame_count=frame_count)


@unittest.skipUnless(torch.backends.mps.is_available(), "MPS is required for the fused Metal owner-run parity test")
class OwnerRunDeltaPackedTrainEvalTests(unittest.TestCase):
    def _common_moving_ray_fixture(
        self,
        *,
        frame_count: int,
        render_size: int = 16,
        site_count: int = 8,
    ) -> tuple[dict[str, object], torch.Tensor]:
        return _common_moving_ray_fixture(
            frame_count=frame_count,
            render_size=render_size,
            site_count=site_count,
        )

    def _stamp_delta_launch_contract(
        self,
        tape_device: dict[str, object],
        *,
        site_rgba: torch.Tensor,
    ) -> dict[str, object]:
        tape_device["delta_packed_records_validated"] = _packed_endpoint_direct_config_validation_marker(
            tape_device=tape_device,
            site_count=int(site_rgba.shape[0]),
            track_count=int(tape_device["track_ray_coeff_f32"].shape[0]),
            frame_count=int(tape_device["frame_t_f32"].shape[0]),
        )
        return tape_device

    def test_delta_packed_recompute_nomid_matches_owner_run_fused_mse_on_moving_rays(self) -> None:
        frame_count = 2
        render_size = 16
        site_count = 8
        common_kwargs, targets = self._common_moving_ray_fixture(
            frame_count=frame_count,
            render_size=render_size,
            site_count=site_count,
        )
        site_rgba_cpu = common_kwargs["site_rgba"]
        owner_tape = _prepare_owner_run_tapes(
            **common_kwargs,
            tape_mode=OWNER_RUN_FUSED_MSE_NOMID_MODE,
        )
        packed_tape = _prepare_owner_run_tapes(
            **common_kwargs,
            tape_mode=OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            endpoint_record_source="slow-owner-run",
        )
        factorized_tape = _prepare_owner_run_tapes(
            **common_kwargs,
            tape_mode=OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            endpoint_record_source="slow-owner-run",
        )
        frameselect_tape = _prepare_owner_run_tapes(
            **common_kwargs,
            tape_mode=OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            endpoint_record_source="slow-owner-run",
        )
        framebitmask_tape = _prepare_owner_run_tapes(
            **common_kwargs,
            tape_mode=OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            endpoint_record_source="slow-owner-run",
        )

        self.assertEqual(packed_tape["selected_segments"], owner_tape["selected_segments"])
        self.assertEqual(packed_tape["selected_segments"], packed_tape["owner_run_segments"])
        self.assertEqual(factorized_tape["selected_segments"], owner_tape["selected_segments"])
        self.assertEqual(factorized_tape["selected_segments"], factorized_tape["owner_run_segments"])
        self.assertEqual(frameselect_tape["selected_segments"], owner_tape["selected_segments"])
        self.assertEqual(frameselect_tape["selected_segments"], frameselect_tape["owner_run_segments"])
        self.assertEqual(framebitmask_tape["selected_segments"], owner_tape["selected_segments"])
        self.assertEqual(framebitmask_tape["selected_segments"], framebitmask_tape["owner_run_segments"])
        self.assertLess(
            packed_tape["endpoint_record_delta_replace_changed_records"],
            0.6 * float(packed_tape["selected_segments"]),
        )
        self.assertLess(
            packed_tape["selected_schema_topology_storage_bytes"],
            owner_tape["selected_schema_topology_storage_bytes"],
        )
        self.assertLess(
            packed_tape["selected_mps_resident_noncoeff_storage_bytes"],
            owner_tape["selected_schema_topology_storage_bytes"],
        )
        packed_device = packed_tape["selected_device"]
        self.assertIn("delta_packed_framegroup16_recompute_fused_mse", packed_device)
        self.assertTrue(packed_device.get("delta_packed_records_validated"))
        self.assertNotIn("boundary_f32", packed_device)
        self.assertNotIn("rays_f32", packed_device)
        self.assertFalse(bool(packed_tape["gate4_affine_candidate_csr_fused_mse"]))
        factorized_device = factorized_tape["selected_device"]
        self.assertIn("delta_packed_framegroup16_factorized_recompute_fused_mse", factorized_device)
        self.assertTrue(factorized_device.get("delta_packed_records_validated"))
        self.assertNotIn("delta_coeff_f16", factorized_device)
        self.assertIn("boundary_f32", factorized_device)
        self.assertIn("track_ray_coeff_f32", factorized_device)
        frameselect_device = frameselect_tape["selected_device"]
        self.assertIn("delta_packed_frameselect_factorized_recompute_fused_mse", frameselect_device)
        self.assertTrue(frameselect_device.get("delta_packed_records_validated"))
        self.assertNotIn("delta_coeff_f16", frameselect_device)
        self.assertNotIn("track_change_offsets_i16", frameselect_device)
        self.assertNotIn("track_chunk_change_offsets_i16", frameselect_device)
        self.assertNotIn("change_frame_i16", frameselect_device)
        self.assertIn("frame_change_index_i16", frameselect_device)
        self.assertIn("change_offsets_i16", frameselect_device)
        framebitmask_device = framebitmask_tape["selected_device"]
        self.assertIn("delta_packed_framebitmask_factorized_recompute_fused_mse", framebitmask_device)
        self.assertTrue(framebitmask_device.get("delta_packed_records_validated"))
        self.assertNotIn("delta_coeff_f16", framebitmask_device)
        self.assertIn("track_change_offsets_i32", framebitmask_device)
        self.assertIn("track_frame_mask_i32", framebitmask_device)
        self.assertNotIn("track_change_offsets_i16", framebitmask_device)
        self.assertNotIn("frame_change_index_i16", framebitmask_device)
        self.assertNotIn("track_chunk_change_offsets_i16", framebitmask_device)
        self.assertNotIn("change_frame_i16", framebitmask_device)
        self.assertIn("change_offsets_i32", framebitmask_device)
        self.assertLess(
            factorized_tape["endpoint_record_coeff_mps_resident_storage_bytes"],
            packed_tape["endpoint_record_coeff_mps_resident_storage_bytes"],
        )

        device = torch.device("mps")
        site_rgba = site_rgba_cpu.to(device=device).contiguous()
        view_count = int(targets.shape[0] // frame_count)
        _channels, height, width = targets.shape[1:]
        target_track = _track_major_rgb_from_image(
            targets.to(device=device),
            view_count=view_count,
            frame_count=frame_count,
            height=int(height),
            width=int(width),
        )
        op_config = RealRayReplayConfig(
            near=common_kwargs["near"],
            far=common_kwargs["far"],
            invalid_epsilon=common_kwargs["invalid_epsilon"],
            transmittance_threshold=common_kwargs["transmittance_threshold"],
        )
        owner_loss, owner_grad = _segment_tape_fused_mse_loss_vjp(
            tape_device=owner_tape["selected_device"],
            site_rgba=site_rgba,
            target_rgb_track=target_track,
            op_config=op_config,
            track_count=owner_tape["track_count"],
            frame_count=frame_count,
        )
        packed_loss, packed_grad = _delta_replace_coeff16_fused_mse_loss_vjp(
            tape_device=packed_tape["selected_device"],
            site_rgba=site_rgba,
            target_rgb_track=target_track,
            op_config=op_config,
            track_count=packed_tape["track_count"],
            frame_count=frame_count,
        )
        factorized_loss, factorized_grad = _delta_replace_coeff16_fused_mse_loss_vjp(
            tape_device=factorized_tape["selected_device"],
            site_rgba=site_rgba,
            target_rgb_track=target_track,
            op_config=op_config,
            track_count=factorized_tape["track_count"],
            frame_count=frame_count,
        )
        frameselect_loss, frameselect_grad = _delta_replace_coeff16_fused_mse_loss_vjp(
            tape_device=frameselect_tape["selected_device"],
            site_rgba=site_rgba,
            target_rgb_track=target_track,
            op_config=op_config,
            track_count=frameselect_tape["track_count"],
            frame_count=frame_count,
        )
        framebitmask_loss, framebitmask_grad = _delta_replace_coeff16_fused_mse_loss_vjp(
            tape_device=framebitmask_tape["selected_device"],
            site_rgba=site_rgba,
            target_rgb_track=target_track,
            op_config=op_config,
            track_count=framebitmask_tape["track_count"],
            frame_count=frame_count,
        )
        torch.mps.synchronize()

        self.assertLessEqual(float((owner_loss - packed_loss).abs().cpu().item()), 5.0e-7)
        self.assertLessEqual(float((owner_grad - packed_grad).abs().max().cpu().item()), 1.0e-6)
        self.assertLessEqual(float((owner_loss - factorized_loss).abs().cpu().item()), 5.0e-6)
        self.assertLessEqual(float((owner_grad - factorized_grad).abs().max().cpu().item()), 1.0e-5)
        self.assertLessEqual(float((owner_loss - frameselect_loss).abs().cpu().item()), 5.0e-6)
        self.assertLessEqual(float((owner_grad - frameselect_grad).abs().max().cpu().item()), 1.0e-5)
        self.assertLessEqual(float((owner_loss - framebitmask_loss).abs().cpu().item()), 5.0e-6)
        self.assertLessEqual(float((owner_grad - framebitmask_grad).abs().max().cpu().item()), 1.0e-5)
        self.assertGreater(float(owner_grad.abs().sum().cpu().item()), 0.0)
        self.assertGreater(float(packed_grad.abs().sum().cpu().item()), 0.0)
        self.assertGreater(float(factorized_grad.abs().sum().cpu().item()), 0.0)
        self.assertGreater(float(frameselect_grad.abs().sum().cpu().item()), 0.0)
        self.assertGreater(float(framebitmask_grad.abs().sum().cpu().item()), 0.0)

    def _delta_wrapper_validation_inputs(
        self,
        *,
        tape_mode: str,
        frame_count: int = 4,
        **prepare_kwargs: object,
    ) -> tuple[dict[str, object], dict[str, object], torch.Tensor, torch.Tensor, RealRayReplayConfig]:
        common_kwargs, targets = self._common_moving_ray_fixture(
            frame_count=frame_count,
            render_size=6,
            site_count=6,
        )
        tape = _prepare_owner_run_tapes(
            **common_kwargs,
            tape_mode=tape_mode,
            endpoint_record_source="slow-owner-run",
            **prepare_kwargs,
        )
        device = torch.device("mps")
        site_rgba = common_kwargs["site_rgba"].to(device=device).contiguous()
        view_count = int(targets.shape[0] // frame_count)
        _channels, height, width = targets.shape[1:]
        target_track = _track_major_rgb_from_image(
            targets.to(device=device),
            view_count=view_count,
            frame_count=frame_count,
            height=int(height),
            width=int(width),
        )
        op_config = RealRayReplayConfig(
            near=common_kwargs["near"],
            far=common_kwargs["far"],
            invalid_epsilon=common_kwargs["invalid_epsilon"],
            transmittance_threshold=common_kwargs["transmittance_threshold"],
        )
        return tape, dict(tape["selected_device"]), site_rgba, target_track, op_config

    def test_all_delta_direct_config_selectors_require_prevalidated_marker(self) -> None:
        frame_count = 4
        cases: tuple[tuple[str, str, str, dict[str, object]], ...] = (
            (
                "raw_i32",
                "endpoint-record-delta-replace-coeff16-fused-mse",
                "base_owner_i32",
                {},
            ),
            (
                "packed_scalar",
                DELTA_PACKED_SCALAR_MODE,
                "delta_packed_scalar_fused_mse",
                {},
            ),
            (
                "packed_framegroup16",
                DELTA_PACKED_FRAMEGROUP16_MODE,
                "delta_packed_framegroup16_fused_mse",
                {},
            ),
            (
                "packed_materialized",
                DELTA_PACKED_FRAMEGROUP16_MATERIALIZED_MODE,
                "delta_packed_framegroup16_materialized_fused_mse",
                {},
            ),
            (
                "packed_recompute",
                DELTA_PACKED_FRAMEGROUP16_RECOMPUTE_MODE,
                "delta_packed_framegroup16_recompute_fused_mse",
                {},
            ),
            (
                "packed_smallrun16",
                DELTA_PACKED_FRAMEGROUP16_SMALLRUN16_MODE,
                "delta_packed_framegroup16_smallrun16_fused_mse",
                {},
            ),
            (
                "packed_launch_only",
                DELTA_PACKED_FRAMEGROUP16_MODE,
                "delta_packed_framegroup16_launch_only_fused_mse",
                {"experimental_launch_only_packed_delta": True},
            ),
            (
                "packed_unchecked_launch_only",
                DELTA_PACKED_FRAMEGROUP16_MODE,
                "delta_packed_framegroup16_unchecked_launch_only_fused_mse",
                {
                    "experimental_launch_only_packed_delta": True,
                    "experimental_unchecked_launch_only_packed_delta": True,
                },
            ),
            (
                "packed_reduce32_launch_only",
                DELTA_PACKED_FRAMEGROUP16_MODE,
                "delta_packed_framegroup16_reduce32_launch_only_fused_mse",
                {
                    "experimental_launch_only_packed_delta": True,
                    "experimental_reduce32_launch_only_packed_delta": True,
                },
            ),
            (
                "packed_rowselect32_launch_only",
                DELTA_PACKED_FRAMEGROUP16_MODE,
                "delta_packed_framegroup16_rowselect32_launch_only_fused_mse",
                {
                    "experimental_launch_only_packed_delta": True,
                    "experimental_rowselect32_launch_only_packed_delta": True,
                },
            ),
            (
                "packed_rowdesc_launch_only",
                DELTA_PACKED_FRAMEGROUP16_MODE,
                "delta_packed_framegroup16_rowdesc_launch_only_fused_mse",
                {
                    "experimental_launch_only_packed_delta": True,
                    "experimental_rowdesc_launch_only_packed_delta": True,
                },
            ),
            (
                "packed_rowdesc32_launch_only",
                DELTA_PACKED_FRAMEGROUP16_MODE,
                "delta_packed_framegroup16_rowdesc32_launch_only_fused_mse",
                {
                    "experimental_launch_only_packed_delta": True,
                    "experimental_rowdesc_launch_only_packed_delta": True,
                    "experimental_rowdesc32_launch_only_packed_delta": True,
                },
            ),
            (
                "i16x4",
                "endpoint-record-delta-replace-coeff16-i16x4-fused-mse",
                "delta_base_record_i16x4",
                {},
            ),
            (
                "i16x4_framegroup16",
                "endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse",
                "delta_i16x4_framegroup16_fused_mse",
                {},
            ),
            (
                "i16cols",
                "endpoint-record-delta-replace-coeff16-i16cols-framegroup16-fused-mse",
                "delta_i16cols_framegroup16_fused_mse",
                {},
            ),
            (
                "i16x3",
                "endpoint-record-delta-replace-coeff16-i16x3-fused-mse",
                "delta_base_record_i16x3",
                {},
            ),
            (
                "i16x3_framegroup16",
                "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse",
                "delta_i16x3_framegroup16_fused_mse",
                {},
            ),
            (
                "i16x3_materialized",
                DELTA_I16X3_FRAMEGROUP16_MATERIALIZED_MODE,
                "delta_i16x3_framegroup16_materialized_fused_mse",
                {},
            ),
            (
                "i16x3_ownerreduce",
                "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-ownerreduce-fused-mse",
                "delta_i16x3_framegroup16_ownerreduce_fused_mse",
                {},
            ),
            (
                "i16x3_framegroup64",
                "endpoint-record-delta-replace-coeff16-i16x3-framegroup64-fused-mse",
                "delta_i16x3_framegroup64_fused_mse",
                {},
            ),
            (
                "factorized_packed",
                OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
                "delta_packed_framegroup16_factorized_recompute_fused_mse",
                {},
            ),
            (
                "factorized_frameselect",
                OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE,
                "delta_packed_frameselect_factorized_recompute_fused_mse",
                {},
            ),
            (
                "factorized_framebitmask",
                OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
                "delta_packed_framebitmask_factorized_recompute_fused_mse",
                {},
            ),
        )
        for label, tape_mode, expected_key, prepare_kwargs in cases:
            with self.subTest(mode=label):
                tape, tape_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
                    tape_mode=tape_mode,
                    frame_count=frame_count,
                    **prepare_kwargs,
                )
                self.assertIn(expected_key, tape_device)
                self.assertIn("delta_config_i32", tape_device)
                self.assertIn("delta_config_f32", tape_device)
                self.assertTrue(tape_device.pop("delta_packed_records_validated"))

                with self.assertRaisesRegex(ValueError, "prevalidated launch contract"):
                    _delta_replace_coeff16_fused_mse_loss_vjp(
                        tape_device=tape_device,
                        site_rgba=site_rgba,
                        target_rgb_track=target_track,
                        op_config=op_config,
                        track_count=tape["track_count"],
                        frame_count=frame_count,
                    )

    def test_launch_only_direct_config_rejects_mutated_launch_scalar_after_marker(self) -> None:
        frame_count = 4
        tape, tape_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
            tape_mode=DELTA_PACKED_FRAMEGROUP16_MODE,
            frame_count=frame_count,
            experimental_launch_only_packed_delta=True,
        )
        self.assertIn("delta_packed_framegroup16_launch_only_fused_mse", tape_device)
        self.assertIn("delta_launch_site_count", tape_device)
        self.assertTrue(tape_device.get("delta_packed_records_validated"))
        tape_device["delta_launch_site_count"] = int(tape_device["delta_launch_site_count"]) + 1

        with self.assertRaisesRegex(ValueError, "current tensors"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=tape_device,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

    def test_rowdesc_direct_config_rejects_replaced_rowdesc_after_marker(self) -> None:
        frame_count = 4
        tape, tape_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
            tape_mode=DELTA_PACKED_FRAMEGROUP16_MODE,
            frame_count=frame_count,
            experimental_launch_only_packed_delta=True,
            experimental_rowdesc_launch_only_packed_delta=True,
        )
        self.assertIn("delta_packed_framegroup16_rowdesc_launch_only_fused_mse", tape_device)
        self.assertIn("row_begin_i32", tape_device)
        self.assertTrue(tape_device.get("delta_packed_records_validated"))
        tape_device["row_begin_i32"] = tape_device["row_begin_i32"].clone()

        with self.assertRaisesRegex(ValueError, "current tensors"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=tape_device,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

    def test_i16x3_ownerreduce_direct_config_rejects_replaced_owner_chunks_after_marker(self) -> None:
        frame_count = 4
        tape, tape_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
            tape_mode="endpoint-record-delta-replace-coeff16-i16x3-framegroup16-ownerreduce-fused-mse",
            frame_count=frame_count,
        )
        self.assertIn("delta_i16x3_framegroup16_ownerreduce_fused_mse", tape_device)
        self.assertIn("track_chunk_owner_i16", tape_device)
        self.assertTrue(tape_device.get("delta_packed_records_validated"))
        tape_device["track_chunk_owner_i16"] = tape_device["track_chunk_owner_i16"].clone()

        with self.assertRaisesRegex(ValueError, "current tensors"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=tape_device,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

    def test_direct_config_marker_rejects_runtime_count_mismatch_after_marker(self) -> None:
        frame_count = 4
        tape, tape_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
            tape_mode=OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            frame_count=frame_count,
        )
        self.assertIn("delta_packed_framebitmask_factorized_recompute_fused_mse", tape_device)
        self.assertTrue(tape_device.get("delta_packed_records_validated"))

        with self.assertRaisesRegex(ValueError, "current tensors"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=dict(tape_device),
                site_rgba=site_rgba,
                target_rgb_track=torch.zeros(
                    (int(tape["track_count"]), frame_count + 1, 3),
                    dtype=target_track.dtype,
                    device=target_track.device,
                ),
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count + 1,
            )

        with self.assertRaisesRegex(ValueError, "current tensors"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=dict(tape_device),
                site_rgba=site_rgba,
                target_rgb_track=torch.zeros(
                    (int(tape["track_count"]) + 1, frame_count, 3),
                    dtype=target_track.dtype,
                    device=target_track.device,
                ),
                op_config=op_config,
                track_count=tape["track_count"] + 1,
                frame_count=frame_count,
            )

    def test_delta_direct_config_rejects_bad_runtime_tensor_layout_before_native_launch(self) -> None:
        frame_count = 4
        tape, tape_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
            tape_mode=OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            frame_count=frame_count,
        )
        self.assertIn("delta_packed_framebitmask_factorized_recompute_fused_mse", tape_device)
        self.assertTrue(tape_device.get("delta_packed_records_validated"))

        with self.assertRaisesRegex(ValueError, "target_rgb_track must have shape"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=dict(tape_device),
                site_rgba=site_rgba,
                target_rgb_track=target_track[:, :-1, :],
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

        with self.assertRaisesRegex(ValueError, "site_rgba must have shape"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=dict(tape_device),
                site_rgba=site_rgba[:, :3],
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

    def test_delta_direct_config_rejects_bad_runtime_tensor_storage_before_native_launch(self) -> None:
        frame_count = 4
        tape, tape_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
            tape_mode=OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            frame_count=frame_count,
        )
        self.assertIn("delta_packed_framebitmask_factorized_recompute_fused_mse", tape_device)
        self.assertTrue(tape_device.get("delta_packed_records_validated"))

        bad_target_dtype = target_track.to(dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "target_rgb_track must be float32"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=dict(tape_device),
                site_rgba=site_rgba,
                target_rgb_track=bad_target_dtype,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

        bad_site_dtype = site_rgba.to(dtype=torch.float16)
        with self.assertRaisesRegex(ValueError, "site_rgba must be float32"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=dict(tape_device),
                site_rgba=bad_site_dtype,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

        with self.assertRaisesRegex(ValueError, "same device"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=dict(tape_device),
                site_rgba=site_rgba.cpu(),
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

        noncontiguous_target = target_track.transpose(0, 1).contiguous().transpose(0, 1)
        self.assertEqual(tuple(noncontiguous_target.shape), tuple(target_track.shape))
        self.assertFalse(noncontiguous_target.is_contiguous())
        with self.assertRaisesRegex(ValueError, "target_rgb_track must be contiguous"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=dict(tape_device),
                site_rgba=site_rgba,
                target_rgb_track=noncontiguous_target,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

        noncontiguous_site = site_rgba.transpose(0, 1).contiguous().transpose(0, 1)
        self.assertEqual(tuple(noncontiguous_site.shape), tuple(site_rgba.shape))
        self.assertFalse(noncontiguous_site.is_contiguous())
        with self.assertRaisesRegex(ValueError, "site_rgba must be contiguous"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=dict(tape_device),
                site_rgba=noncontiguous_site,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

    def test_delta_direct_config_rejects_bad_tape_tensor_storage_after_marker(self) -> None:
        frame_count = 4
        tape, tape_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
            tape_mode=OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            frame_count=frame_count,
        )
        self.assertIn("delta_packed_framebitmask_factorized_recompute_fused_mse", tape_device)
        self.assertTrue(tape_device.get("delta_packed_records_validated"))

        bad_dtype = dict(tape_device)
        bad_dtype["delta_base_record_i32"] = bad_dtype["delta_base_record_i32"].to(dtype=torch.float32)
        self._stamp_delta_launch_contract(bad_dtype, site_rgba=site_rgba)
        with self.assertRaisesRegex(ValueError, "delta_base_record_i32 must be torch.int32"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=bad_dtype,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

        bad_device = dict(tape_device)
        bad_device["delta_base_record_i32"] = bad_device["delta_base_record_i32"].cpu()
        self._stamp_delta_launch_contract(bad_device, site_rgba=site_rgba)
        with self.assertRaisesRegex(ValueError, "delta_base_record_i32 must be on"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=bad_device,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

        bad_contiguous = dict(tape_device)
        boundary_f32 = bad_contiguous["boundary_f32"]
        bad_boundary = boundary_f32.transpose(0, 1).contiguous().transpose(0, 1)
        self.assertEqual(tuple(bad_boundary.shape), tuple(boundary_f32.shape))
        self.assertFalse(bad_boundary.is_contiguous())
        bad_contiguous["boundary_f32"] = bad_boundary
        self._stamp_delta_launch_contract(bad_contiguous, site_rgba=site_rgba)
        with self.assertRaisesRegex(ValueError, "boundary_f32 must be contiguous"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=bad_contiguous,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

    def test_delta_direct_config_rejects_bad_scalar_launch_contract_after_marker(self) -> None:
        frame_count = 4
        tape, tape_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
            tape_mode=OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            frame_count=frame_count,
        )
        self.assertIn("delta_coeff_boundary_count", tape_device)
        self.assertIn("delta_launch_base_record_count", tape_device)
        self.assertIn("delta_launch_change_count", tape_device)
        self.assertIn("delta_launch_change_record_count", tape_device)
        bad_boundary_count = dict(tape_device)
        bad_boundary_count["delta_coeff_boundary_count"] = int(bad_boundary_count["delta_coeff_boundary_count"]) + 1
        self._stamp_delta_launch_contract(bad_boundary_count, site_rgba=site_rgba)
        with self.assertRaisesRegex(ValueError, "delta_coeff_boundary_count must match boundary_f32"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=bad_boundary_count,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

        bad_change_count = dict(tape_device)
        bad_change_count["delta_launch_change_count"] = int(bad_change_count["delta_launch_change_count"]) + 1
        self._stamp_delta_launch_contract(bad_change_count, site_rgba=site_rgba)
        with self.assertRaisesRegex(ValueError, "delta_launch_change_count must match change_offsets_i32"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=bad_change_count,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

        bad_base_record_count = dict(tape_device)
        bad_base_record_count["delta_launch_base_record_count"] = (
            int(bad_base_record_count["delta_launch_base_record_count"]) + 1
        )
        self._stamp_delta_launch_contract(bad_base_record_count, site_rgba=site_rgba)
        with self.assertRaisesRegex(ValueError, "delta_launch_base_record_count must match"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=bad_base_record_count,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

        bad_change_record_count = dict(tape_device)
        bad_change_record_count["delta_launch_change_record_count"] = (
            int(bad_change_record_count["delta_launch_change_record_count"]) + 1
        )
        self._stamp_delta_launch_contract(bad_change_record_count, site_rgba=site_rgba)
        with self.assertRaisesRegex(ValueError, "delta_launch_change_record_count must match"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=bad_change_record_count,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

        missing_change_record_count = dict(tape_device)
        missing_change_record_count.pop("delta_launch_change_record_count")
        self._stamp_delta_launch_contract(missing_change_record_count, site_rgba=site_rgba)
        with self.assertRaisesRegex(ValueError, "missing scalar contract keys"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=missing_change_record_count,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

        bad_scalar_type = dict(tape_device)
        bad_scalar_type["delta_launch_track_count"] = "not-an-int"
        bad_scalar_type["delta_packed_records_validated"] = _packed_endpoint_direct_config_validation_marker(
            tape_device=bad_scalar_type,
            site_count=int(site_rgba.shape[0]),
            track_count=int(tape["track_count"]),
            frame_count=frame_count,
        )
        with self.assertRaisesRegex(ValueError, "delta_launch_track_count must be a Python integer scalar"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=bad_scalar_type,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

        stale_bool_scalar_type = dict(tape_device)
        stale_bool_scalar_type["delta_packed_records_validated"] = _packed_endpoint_direct_config_validation_marker(
            tape_device=stale_bool_scalar_type,
            site_count=int(site_rgba.shape[0]),
            track_count=int(tape["track_count"]),
            frame_count=frame_count,
        )
        stale_bool_scalar_type["delta_launch_track_count"] = True
        with self.assertRaisesRegex(ValueError, "direct-config path requires a prevalidated launch contract"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=stale_bool_scalar_type,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

        launch_tape, launch_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
            tape_mode=DELTA_PACKED_FRAMEGROUP16_MODE,
            frame_count=frame_count,
            experimental_launch_only_packed_delta=True,
        )
        self.assertIn("delta_launch_track_count", launch_device)
        self.assertIn("delta_launch_base_record_count", launch_device)

        bad_track_count = dict(launch_device)
        bad_track_count["delta_launch_track_count"] = int(bad_track_count["delta_launch_track_count"]) + 1
        bad_track_count["delta_packed_records_validated"] = _packed_endpoint_direct_config_validation_marker(
            tape_device=bad_track_count,
            site_count=int(site_rgba.shape[0]),
            track_count=int(launch_tape["track_count"]),
            frame_count=frame_count,
        )
        with self.assertRaisesRegex(ValueError, "delta_launch_track_count must match runtime track_count"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=bad_track_count,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=launch_tape["track_count"],
                frame_count=frame_count,
            )

        bad_base_count = dict(launch_device)
        bad_base_count["delta_launch_base_record_count"] = int(bad_base_count["delta_launch_base_record_count"]) + 1
        bad_base_count["delta_packed_records_validated"] = _packed_endpoint_direct_config_validation_marker(
            tape_device=bad_base_count,
            site_count=int(site_rgba.shape[0]),
            track_count=int(launch_tape["track_count"]),
            frame_count=frame_count,
        )
        with self.assertRaisesRegex(ValueError, "delta_launch_base_record_count must match"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=bad_base_count,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=launch_tape["track_count"],
                frame_count=frame_count,
            )

    def test_i32_packed_direct_config_selectors_require_scalar_launch_count_keys_after_marker(self) -> None:
        frame_count = 4
        cases: tuple[tuple[str, str, str, dict[str, object]], ...] = (
            ("packed_scalar", DELTA_PACKED_SCALAR_MODE, "delta_packed_scalar_fused_mse", {}),
            ("packed_framegroup16", DELTA_PACKED_FRAMEGROUP16_MODE, "delta_packed_framegroup16_fused_mse", {}),
            (
                "packed_materialized",
                DELTA_PACKED_FRAMEGROUP16_MATERIALIZED_MODE,
                "delta_packed_framegroup16_materialized_fused_mse",
                {},
            ),
            (
                "packed_recompute",
                DELTA_PACKED_FRAMEGROUP16_RECOMPUTE_MODE,
                "delta_packed_framegroup16_recompute_fused_mse",
                {},
            ),
            (
                "packed_smallrun16",
                DELTA_PACKED_FRAMEGROUP16_SMALLRUN16_MODE,
                "delta_packed_framegroup16_smallrun16_fused_mse",
                {},
            ),
            (
                "packed_launch_only",
                DELTA_PACKED_FRAMEGROUP16_MODE,
                "delta_packed_framegroup16_launch_only_fused_mse",
                {"experimental_launch_only_packed_delta": True},
            ),
            (
                "factorized_packed",
                OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
                "delta_packed_framegroup16_factorized_recompute_fused_mse",
                {},
            ),
            (
                "factorized_frameselect",
                OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE,
                "delta_packed_frameselect_factorized_recompute_fused_mse",
                {},
            ),
            (
                "factorized_framebitmask",
                OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
                "delta_packed_framebitmask_factorized_recompute_fused_mse",
                {},
            ),
        )
        required_scalars = (
            "delta_coeff_boundary_count",
            "delta_launch_boundary_count",
            "delta_launch_track_count",
            "delta_launch_frame_count",
            "delta_launch_site_count",
            "delta_launch_base_record_count",
            "delta_launch_change_count",
            "delta_launch_change_record_count",
        )
        for label, tape_mode, expected_key, prepare_kwargs in cases:
            with self.subTest(mode=label):
                tape, tape_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
                    tape_mode=tape_mode,
                    frame_count=frame_count,
                    **prepare_kwargs,
                )
                self.assertIn(expected_key, tape_device)
                for scalar_key in required_scalars:
                    self.assertIn(scalar_key, tape_device)

                missing_change_record_count = dict(tape_device)
                missing_change_record_count.pop("delta_launch_change_record_count")
                missing_change_record_count["delta_packed_records_validated"] = (
                    _packed_endpoint_direct_config_validation_marker(
                        tape_device=missing_change_record_count,
                        site_count=int(site_rgba.shape[0]),
                        track_count=int(tape["track_count"]),
                        frame_count=frame_count,
                    )
                )
                with self.assertRaisesRegex(ValueError, "missing scalar contract keys"):
                    _delta_replace_coeff16_fused_mse_loss_vjp(
                        tape_device=missing_change_record_count,
                        site_rgba=site_rgba,
                        target_rgb_track=target_track,
                        op_config=op_config,
                        track_count=tape["track_count"],
                        frame_count=frame_count,
                    )
                if label == "packed_scalar":
                    missing_coeff_boundary_count = dict(tape_device)
                    missing_coeff_boundary_count.pop("delta_coeff_boundary_count")
                    missing_coeff_boundary_count["delta_packed_records_validated"] = (
                        _packed_endpoint_direct_config_validation_marker(
                            tape_device=missing_coeff_boundary_count,
                            site_count=int(site_rgba.shape[0]),
                            track_count=int(tape["track_count"]),
                            frame_count=frame_count,
                        )
                    )
                    with self.assertRaisesRegex(ValueError, "missing scalar contract keys"):
                        _delta_replace_coeff16_fused_mse_loss_vjp(
                            tape_device=missing_coeff_boundary_count,
                            site_rgba=site_rgba,
                            target_rgb_track=target_track,
                            op_config=op_config,
                            track_count=tape["track_count"],
                            frame_count=frame_count,
                        )

    def test_delta_direct_config_rejects_bad_tape_tensor_shape_after_marker(self) -> None:
        frame_count = 4
        tape, tape_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
            tape_mode=OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            frame_count=frame_count,
        )
        self.assertIn("boundary_f32", tape_device)
        self.assertIn("track_ray_coeff_f32", tape_device)

        bad_boundary_shape = dict(tape_device)
        bad_boundary_shape["boundary_f32"] = bad_boundary_shape["boundary_f32"][:, :4].contiguous()
        self._stamp_delta_launch_contract(bad_boundary_shape, site_rgba=site_rgba)
        with self.assertRaisesRegex(ValueError, "boundary_f32 must have shape"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=bad_boundary_shape,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

        bad_track_coeff_shape = dict(tape_device)
        bad_track_coeff_shape["track_ray_coeff_f32"] = bad_track_coeff_shape["track_ray_coeff_f32"][:, :11].contiguous()
        self._stamp_delta_launch_contract(bad_track_coeff_shape, site_rgba=site_rgba)
        with self.assertRaisesRegex(ValueError, "track_ray_coeff_f32 must have shape"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=bad_track_coeff_shape,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

        ownerreduce_tape, ownerreduce_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
            tape_mode="endpoint-record-delta-replace-coeff16-i16x3-framegroup16-ownerreduce-fused-mse",
            frame_count=frame_count,
        )
        self.assertIn("delta_base_record_i16x3", ownerreduce_device)
        bad_i16x3_shape = dict(ownerreduce_device)
        bad_i16x3_shape["delta_base_record_i16x3"] = bad_i16x3_shape["delta_base_record_i16x3"][:-1].contiguous()
        bad_i16x3_shape["delta_packed_records_validated"] = _packed_endpoint_direct_config_validation_marker(
            tape_device=bad_i16x3_shape,
            site_count=int(site_rgba.shape[0]),
            track_count=int(ownerreduce_tape["track_count"]),
            frame_count=frame_count,
        )
        with self.assertRaisesRegex(ValueError, "delta_base_record_i16x3 length must be a multiple of 3"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=bad_i16x3_shape,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=ownerreduce_tape["track_count"],
                frame_count=frame_count,
            )

        rowdesc_tape, rowdesc_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
            tape_mode=DELTA_PACKED_FRAMEGROUP16_MODE,
            frame_count=frame_count,
            experimental_launch_only_packed_delta=True,
            experimental_rowdesc_launch_only_packed_delta=True,
        )
        self.assertIn("row_begin_i32", rowdesc_device)
        bad_rowdesc_shape = dict(rowdesc_device)
        bad_rowdesc_shape["row_begin_i32"] = bad_rowdesc_shape["row_begin_i32"][:-1].contiguous()
        bad_rowdesc_shape["delta_packed_records_validated"] = _packed_endpoint_direct_config_validation_marker(
            tape_device=bad_rowdesc_shape,
            site_count=int(site_rgba.shape[0]),
            track_count=int(rowdesc_tape["track_count"]),
            frame_count=frame_count,
        )
        with self.assertRaisesRegex(ValueError, "row_begin_i32 must have shape"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=bad_rowdesc_shape,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=rowdesc_tape["track_count"],
                frame_count=frame_count,
            )

    def test_delta_direct_config_rejects_conflicting_selectors_after_marker(self) -> None:
        frame_count = 4
        tape, tape_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
            tape_mode=OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            frame_count=frame_count,
        )
        bad_factorized_selector = dict(tape_device)
        bad_factorized_selector["delta_packed_frameselect_factorized_recompute_fused_mse"] = torch.tensor(
            [1],
            dtype=torch.int32,
            device=site_rgba.device,
        )
        self._stamp_delta_launch_contract(bad_factorized_selector, site_rgba=site_rgba)
        with self.assertRaisesRegex(ValueError, "conflicting primary selectors"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=bad_factorized_selector,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

        scalar_tape, scalar_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
            tape_mode=DELTA_PACKED_SCALAR_MODE,
            frame_count=frame_count,
        )
        bad_scalar_launch = dict(scalar_device)
        bad_scalar_launch["delta_packed_framegroup16_launch_only_fused_mse"] = True
        bad_scalar_launch["delta_packed_records_validated"] = _packed_endpoint_direct_config_validation_marker(
            tape_device=bad_scalar_launch,
            site_count=int(site_rgba.shape[0]),
            track_count=int(scalar_tape["track_count"]),
            frame_count=frame_count,
        )
        with self.assertRaisesRegex(ValueError, "launch-only modifiers require a non-scalar packed framegroup selector"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=bad_scalar_launch,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=scalar_tape["track_count"],
                frame_count=frame_count,
            )

        launch_tape, launch_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
            tape_mode=DELTA_PACKED_FRAMEGROUP16_MODE,
            frame_count=frame_count,
            experimental_launch_only_packed_delta=True,
        )
        bad_launch_row_selector = dict(launch_device)
        bad_launch_row_selector["delta_packed_framegroup16_reduce32_launch_only_fused_mse"] = True
        bad_launch_row_selector["delta_packed_framegroup16_rowselect32_launch_only_fused_mse"] = True
        bad_launch_row_selector["delta_packed_records_validated"] = _packed_endpoint_direct_config_validation_marker(
            tape_device=bad_launch_row_selector,
            site_count=int(site_rgba.shape[0]),
            track_count=int(launch_tape["track_count"]),
            frame_count=frame_count,
        )
        with self.assertRaisesRegex(ValueError, "row selector modifiers are mutually exclusive"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=bad_launch_row_selector,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=launch_tape["track_count"],
                frame_count=frame_count,
            )

    def test_packed_recompute_direct_config_requires_prevalidated_records_marker(self) -> None:
        frame_count = 4
        tape, tape_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
            tape_mode=OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            frame_count=frame_count,
        )
        self.assertIn("delta_config_i32", tape_device)
        self.assertIn("delta_config_f32", tape_device)
        self.assertTrue(tape_device.pop("delta_packed_records_validated"))

        with self.assertRaisesRegex(ValueError, "direct-config path requires a prevalidated launch contract"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=tape_device,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

    def test_packed_recompute_direct_config_rejects_replaced_records_after_marker(self) -> None:
        frame_count = 4
        tape, tape_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
            tape_mode=OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            frame_count=frame_count,
        )
        self.assertIn("delta_config_i32", tape_device)
        self.assertIn("delta_config_f32", tape_device)
        self.assertTrue(tape_device.get("delta_packed_records_validated"))
        bad_base = tape_device["delta_base_record_i32"].clone()
        self.assertGreater(int(bad_base.numel()), 0)
        bad_base[0] = int(site_rgba.shape[0]) | (0 << 8) | (1 << 20)
        tape_device["delta_base_record_i32"] = bad_base

        with self.assertRaisesRegex(ValueError, "current tensors"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=tape_device,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

    def test_packed_recompute_direct_config_rejects_mutated_config_after_marker(self) -> None:
        frame_count = 4
        tape, tape_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
            tape_mode=OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            frame_count=frame_count,
        )
        self.assertIn("delta_config_i32", tape_device)
        self.assertIn("delta_config_f32", tape_device)
        self.assertTrue(tape_device.get("delta_packed_records_validated"))
        tape_device["delta_config_i32"][0] += 1

        with self.assertRaisesRegex(ValueError, "current tensors"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=tape_device,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

    def test_packed_recompute_direct_config_rejects_mutated_topology_after_marker(self) -> None:
        frame_count = 4
        tape, tape_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
            tape_mode=OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            frame_count=frame_count,
        )
        self.assertIn("base_offsets_i32", tape_device)
        self.assertTrue(tape_device.get("delta_packed_records_validated"))
        tape_device["base_offsets_i32"][0] += 1

        with self.assertRaisesRegex(ValueError, "current tensors"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=tape_device,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

    def test_packed_recompute_direct_config_rejects_replaced_topology_after_marker(self) -> None:
        frame_count = 4
        tape, tape_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
            tape_mode=OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            frame_count=frame_count,
        )
        self.assertIn("track_chunk_change_offsets_i16", tape_device)
        self.assertTrue(tape_device.get("delta_packed_records_validated"))
        tape_device["track_chunk_change_offsets_i16"] = tape_device["track_chunk_change_offsets_i16"].clone()

        with self.assertRaisesRegex(ValueError, "current tensors"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=tape_device,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

    def test_packed_recompute_direct_config_rejects_selector_change_after_marker(self) -> None:
        frame_count = 4
        tape, tape_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
            tape_mode=OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            frame_count=frame_count,
        )
        self.assertNotIn("delta_packed_framegroup16_launch_only_fused_mse", tape_device)
        self.assertTrue(tape_device.get("delta_packed_records_validated"))
        tape_device["delta_packed_framegroup16_launch_only_fused_mse"] = True

        with self.assertRaisesRegex(ValueError, "current tensors"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=tape_device,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

    def test_legacy_delta_direct_config_requires_prevalidated_marker(self) -> None:
        frame_count = 4
        cases = (
            (
                "raw_i32",
                "endpoint-record-delta-replace-coeff16-fused-mse",
                "base_owner_i32",
            ),
            (
                "i16x4_framegroup16",
                "endpoint-record-delta-replace-coeff16-i16x4-framegroup16-fused-mse",
                "delta_base_record_i16x4",
            ),
            (
                "i16x3_framegroup16",
                "endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse",
                "delta_base_record_i16x3",
            ),
        )
        for label, tape_mode, expected_key in cases:
            with self.subTest(mode=label):
                tape, tape_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
                    tape_mode=tape_mode,
                    frame_count=frame_count,
                )
                self.assertIn(expected_key, tape_device)
                self.assertIn("delta_config_i32", tape_device)
                self.assertIn("delta_config_f32", tape_device)
                self.assertTrue(tape_device.pop("delta_packed_records_validated"))

                with self.assertRaisesRegex(ValueError, "direct-config path requires a prevalidated launch contract"):
                    _delta_replace_coeff16_fused_mse_loss_vjp(
                        tape_device=tape_device,
                        site_rgba=site_rgba,
                        target_rgb_track=target_track,
                        op_config=op_config,
                        track_count=tape["track_count"],
                        frame_count=frame_count,
                    )

    def test_raw_delta_direct_config_rejects_replaced_owner_after_marker(self) -> None:
        frame_count = 4
        tape, tape_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
            tape_mode="endpoint-record-delta-replace-coeff16-fused-mse",
            frame_count=frame_count,
        )
        self.assertIn("base_owner_i32", tape_device)
        self.assertTrue(tape_device.get("delta_packed_records_validated"))
        tape_device["base_owner_i32"] = tape_device["base_owner_i32"].clone()

        with self.assertRaisesRegex(ValueError, "current tensors"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=tape_device,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

    def test_i16x3_direct_config_rejects_selector_change_after_marker(self) -> None:
        frame_count = 4
        tape, tape_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
            tape_mode="endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse",
            frame_count=frame_count,
        )
        self.assertIn("delta_i16x3_framegroup16_fused_mse", tape_device)
        self.assertNotIn("delta_i16x3_framegroup64_fused_mse", tape_device)
        self.assertTrue(tape_device.get("delta_packed_records_validated"))
        tape_device["delta_i16x3_framegroup64_fused_mse"] = True

        with self.assertRaisesRegex(ValueError, "current tensors"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=tape_device,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

    def test_factorized_prepared_paths_require_prevalidated_marker(self) -> None:
        frame_count = 4
        cases = (
            (
                "factorized_packed",
                OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
                "delta_packed_framegroup16_factorized_recompute_fused_mse",
            ),
            (
                "factorized_frameselect",
                OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE,
                "delta_packed_frameselect_factorized_recompute_fused_mse",
            ),
            (
                "factorized_framebitmask",
                OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
                "delta_packed_framebitmask_factorized_recompute_fused_mse",
            ),
        )
        for label, tape_mode, expected_key in cases:
            with self.subTest(mode=label):
                tape, tape_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
                    tape_mode=tape_mode,
                    frame_count=frame_count,
                )
                self.assertIn(expected_key, tape_device)
                self.assertTrue(tape_device.pop("delta_packed_records_validated"))

                with self.assertRaisesRegex(ValueError, "prevalidated launch contract"):
                    _delta_replace_coeff16_fused_mse_loss_vjp(
                        tape_device=tape_device,
                        site_rgba=site_rgba,
                        target_rgb_track=target_track,
                        op_config=op_config,
                        track_count=tape["track_count"],
                        frame_count=frame_count,
                    )

    def test_factorized_framebitmask_rejects_mutated_mask_after_marker(self) -> None:
        frame_count = 4
        tape, tape_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
            tape_mode=OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            frame_count=frame_count,
        )
        self.assertIn("track_frame_mask_i32", tape_device)
        self.assertTrue(tape_device.get("delta_packed_records_validated"))
        tape_device["track_frame_mask_i32"][0] += 1 << 1

        with self.assertRaisesRegex(ValueError, "current tensors"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=tape_device,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

    def test_factorized_frameselect_rejects_replaced_index_after_marker(self) -> None:
        frame_count = 4
        tape, tape_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
            tape_mode=OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            frame_count=frame_count,
        )
        self.assertIn("frame_change_index_i16", tape_device)
        self.assertTrue(tape_device.get("delta_packed_records_validated"))
        tape_device["frame_change_index_i16"] = tape_device["frame_change_index_i16"].clone()

        with self.assertRaisesRegex(ValueError, "current tensors"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=tape_device,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=tape["track_count"],
                frame_count=frame_count,
            )

    def test_non_framebitmask_packed_wrappers_reject_endpoint_record_bounds(self) -> None:
        modes = (
            (
                "packed_recompute",
                OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
                True,
            ),
            (
                "factorized_packed",
                OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
                False,
            ),
            (
                "factorized_frameselect",
                OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE,
                False,
            ),
        )
        for label, tape_mode, strip_prebaked_config in modes:
            with self.subTest(mode=label, record="base_owner"):
                frame_count = 4
                tape, tape_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
                    tape_mode=tape_mode,
                    frame_count=frame_count,
                )
                if strip_prebaked_config:
                    tape_device.pop("delta_config_i32", None)
                    tape_device.pop("delta_config_f32", None)
                bad_base = tape_device["delta_base_record_i32"].clone()
                self.assertGreater(int(bad_base.numel()), 0)
                bad_base[0] = int(site_rgba.shape[0]) | (0 << 8) | (1 << 20)
                tape_device["delta_base_record_i32"] = bad_base

                expected_error = (
                    "base_record_i32 owner code must be < site_count"
                    if strip_prebaked_config
                    else "current tensors"
                )
                with self.assertRaisesRegex(ValueError, expected_error):
                    _delta_replace_coeff16_fused_mse_loss_vjp(
                        tape_device=tape_device,
                        site_rgba=site_rgba,
                        target_rgb_track=target_track,
                        op_config=op_config,
                        track_count=tape["track_count"],
                        frame_count=frame_count,
                    )

            with self.subTest(mode=label, record="change_left_cut"):
                frame_count = 4
                tape, tape_device, site_rgba, target_track, op_config = self._delta_wrapper_validation_inputs(
                    tape_mode=tape_mode,
                    frame_count=frame_count,
                )
                if strip_prebaked_config:
                    tape_device.pop("delta_config_i32", None)
                    tape_device.pop("delta_config_f32", None)
                bad_change = tape_device["delta_change_record_i32"].clone()
                self.assertGreater(int(bad_change.numel()), 0)
                boundary_count = int(tape_device["delta_coeff_boundary_count"])
                bad_left_code = boundary_count + 2
                self.assertLess(bad_left_code, 4096)
                bad_change[0] = 0 | (bad_left_code << 8) | (1 << 20)
                tape_device["delta_change_record_i32"] = bad_change

                expected_error = (
                    "change_record_i32 left cut id must be < boundary_count"
                    if strip_prebaked_config
                    else "current tensors"
                )
                with self.assertRaisesRegex(ValueError, expected_error):
                    _delta_replace_coeff16_fused_mse_loss_vjp(
                        tape_device=tape_device,
                        site_rgba=site_rgba,
                        target_rgb_track=target_track,
                        op_config=op_config,
                        track_count=tape["track_count"],
                        frame_count=frame_count,
                    )

    def test_factorized_packed_recompute_storage_stays_coeff_constant_across_frames(self) -> None:
        rows = []
        for frame_count in (2, 4, 8):
            with self.subTest(frame_count=frame_count):
                common_kwargs, _targets = self._common_moving_ray_fixture(frame_count=frame_count)
                tape = _prepare_owner_run_tapes(
                    **common_kwargs,
                    tape_mode=OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
                    endpoint_record_source="slow-owner-run",
                )
                selected_device = tape["selected_device"]
                self.assertIn("delta_packed_framegroup16_factorized_recompute_fused_mse", selected_device)
                self.assertNotIn("delta_coeff_f16", selected_device)
                self.assertIn("boundary_f32", selected_device)
                self.assertIn("track_ray_coeff_f32", selected_device)
                rows.append(
                    {
                        "frame_count": frame_count,
                        "selected_storage": int(tape["selected_schema_storage_bytes"]),
                        "topology_storage": int(tape["selected_schema_topology_storage_bytes"]),
                        "coeff_storage": int(tape["endpoint_record_coeff_storage_bytes"]),
                        "resident_coeff_storage": int(tape["endpoint_record_coeff_mps_resident_storage_bytes"]),
                    }
                )

        self.assertEqual(rows[0]["coeff_storage"], rows[-1]["coeff_storage"])
        self.assertEqual(rows[0]["resident_coeff_storage"], rows[-1]["resident_coeff_storage"])
        frame_scale = rows[-1]["frame_count"] / rows[0]["frame_count"]
        self.assertLess(rows[-1]["selected_storage"] / rows[0]["selected_storage"], frame_scale)
        self.assertGreater(rows[-1]["topology_storage"], rows[0]["topology_storage"])

    def test_factorized_highcap_storage_removes_dense_coeff16(self) -> None:
        rows = []
        for frame_count in (2, 8):
            with self.subTest(frame_count=frame_count):
                common_kwargs, _targets = self._common_moving_ray_fixture(
                    frame_count=frame_count,
                    site_count=24,
                )
                tape = _prepare_owner_run_tapes(
                    **common_kwargs,
                    tape_mode=OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
                    endpoint_record_source="slow-owner-run",
                )
                schema_by_key = tape["selected_schema_storage_by_key"]
                projected_by_key = tape["selected_schema_i16_meta_projected_storage_by_key"]
                by_key = tape["selected_mps_resident_storage_by_key"]
                self.assertIn("delta_packed_framegroup16_factorized_recompute_fused_mse", tape["selected_device"])
                self.assertNotIn("delta_coeff_f16", tape["selected_device"])
                self.assertNotIn("delta_coeff_f16", schema_by_key)
                self.assertNotIn("base_offsets_i32", schema_by_key)
                self.assertNotIn("track_change_offsets_i32", schema_by_key)
                self.assertNotIn("change_frame_i32", schema_by_key)
                self.assertNotIn("change_offsets_i32", schema_by_key)
                self.assertIn("base_offsets_i16", schema_by_key)
                self.assertIn("track_change_offsets_i16", schema_by_key)
                self.assertIn("change_frame_i16", schema_by_key)
                self.assertIn("change_offsets_i16", schema_by_key)
                self.assertNotIn("base_offsets_i32", tape["selected_device"])
                self.assertNotIn("track_change_offsets_i32", tape["selected_device"])
                self.assertNotIn("change_frame_i32", tape["selected_device"])
                self.assertNotIn("change_offsets_i32", tape["selected_device"])
                self.assertIn("base_offsets_i16", tape["selected_device"])
                self.assertIn("track_change_offsets_i16", tape["selected_device"])
                self.assertIn("change_frame_i16", tape["selected_device"])
                self.assertIn("change_offsets_i16", tape["selected_device"])
                self.assertNotIn("unattributed_storage", schema_by_key)
                self.assertEqual(
                    sum(int(value) for value in schema_by_key.values()),
                    tape["selected_schema_storage_bytes"],
                )
                self.assertEqual(
                    int(schema_by_key["factorized_coeff_f32"]),
                    int(tape["endpoint_record_coeff_storage_bytes"]),
                )
                self.assertTrue(bool(tape["selected_schema_i16_meta_projection_eligible"]))
                self.assertNotIn("change_frame_i32", projected_by_key)
                self.assertIn("change_frame_i16", projected_by_key)
                self.assertEqual(
                    sum(int(value) for value in projected_by_key.values()),
                    tape["selected_schema_i16_meta_projected_storage_bytes"],
                )
                self.assertEqual(tape["selected_schema_i16_meta_projected_storage_savings_bytes"], 0)
                self.assertEqual(
                    tape["selected_schema_i16_meta_projected_storage_bytes"],
                    tape["selected_schema_storage_bytes"],
                )
                self.assertEqual(
                    int(tape["endpoint_record_coeff_mps_resident_storage_bytes"]),
                    int(by_key["boundary_f32"]) + int(by_key["track_ray_coeff_f32"]),
                )
                rows.append(
                    {
                        "frame_count": frame_count,
                        "selected_storage": int(tape["selected_schema_storage_bytes"]),
                        "change_record_storage": int(schema_by_key["change_record_packed"]),
                        "resident_coeff_storage": int(tape["endpoint_record_coeff_mps_resident_storage_bytes"]),
                    }
                )

        self.assertEqual(rows[0]["resident_coeff_storage"], rows[-1]["resident_coeff_storage"])
        self.assertLess(
            rows[-1]["selected_storage"] / rows[0]["selected_storage"],
            rows[-1]["frame_count"] / rows[0]["frame_count"],
        )
        self.assertGreater(rows[-1]["change_record_storage"], rows[0]["change_record_storage"])

        common_kwargs, _targets = self._common_moving_ray_fixture(frame_count=8, site_count=24)
        packed_tape = _prepare_owner_run_tapes(
            **common_kwargs,
            tape_mode=OWNER_RUN_DELTA_PACKED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            endpoint_record_source="slow-owner-run",
        )
        factorized_tape = _prepare_owner_run_tapes(
            **common_kwargs,
            tape_mode=OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            endpoint_record_source="slow-owner-run",
        )
        self.assertIn("delta_coeff_f16", packed_tape["selected_device"])
        self.assertNotIn("delta_coeff_f16", factorized_tape["selected_device"])
        self.assertLess(
            factorized_tape["endpoint_record_coeff_mps_resident_storage_bytes"],
            0.05 * float(packed_tape["endpoint_record_coeff_mps_resident_storage_bytes"]),
        )
        self.assertLess(
            factorized_tape["selected_schema_storage_bytes"],
            packed_tape["selected_schema_storage_bytes"],
        )

    def test_factorized_frameselect_removes_sparse_frame_scan_metadata(self) -> None:
        frame_count = 8
        common_kwargs, _targets = self._common_moving_ray_fixture(frame_count=frame_count, site_count=24)
        factorized_tape = _prepare_owner_run_tapes(
            **common_kwargs,
            tape_mode=OWNER_RUN_DELTA_PACKED_FACTORIZED_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            endpoint_record_source="slow-owner-run",
        )
        frameselect_tape = _prepare_owner_run_tapes(
            **common_kwargs,
            tape_mode=OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            endpoint_record_source="slow-owner-run",
        )
        schema_by_key = frameselect_tape["selected_schema_storage_by_key"]
        selected_device = frameselect_tape["selected_device"]

        self.assertIn("delta_packed_frameselect_factorized_recompute_fused_mse", selected_device)
        self.assertIn("frame_change_index_i16", selected_device)
        self.assertNotIn("track_change_offsets_i16", selected_device)
        self.assertNotIn("track_chunk_change_offsets_i16", selected_device)
        self.assertNotIn("change_frame_i16", selected_device)
        self.assertIn("frame_select_i16", schema_by_key)
        self.assertNotIn("track_change_offsets_i16", schema_by_key)
        self.assertNotIn("change_frame_i16", schema_by_key)
        self.assertEqual(
            int(schema_by_key["frame_select_i16"]),
            int(frameselect_tape["track_count"]) * (frame_count - 1) * 2,
        )
        self.assertLess(
            frameselect_tape["selected_schema_storage_bytes"],
            factorized_tape["selected_schema_storage_bytes"],
        )

    def test_factorized_framebitmask_removes_dense_frame_table(self) -> None:
        frame_count = 16
        common_kwargs, _targets = self._common_moving_ray_fixture(frame_count=frame_count, site_count=8)
        frameselect_tape = _prepare_owner_run_tapes(
            **common_kwargs,
            tape_mode=OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMESELECT_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            endpoint_record_source="slow-owner-run",
        )
        framebitmask_tape = _prepare_owner_run_tapes(
            **common_kwargs,
            tape_mode=OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            endpoint_record_source="slow-owner-run",
        )
        schema_by_key = framebitmask_tape["selected_schema_storage_by_key"]
        selected_device = framebitmask_tape["selected_device"]

        self.assertIn("delta_packed_framebitmask_factorized_recompute_fused_mse", selected_device)
        self.assertIn("base_offsets_i32", selected_device)
        self.assertIn("track_change_offsets_i32", selected_device)
        self.assertIn("track_frame_mask_i32", selected_device)
        self.assertNotIn("base_offsets_i16", selected_device)
        self.assertNotIn("track_change_offsets_i16", selected_device)
        self.assertNotIn("frame_change_index_i16", selected_device)
        self.assertNotIn("track_chunk_change_offsets_i16", selected_device)
        self.assertNotIn("change_frame_i16", selected_device)
        self.assertIn("base_offsets_i32", schema_by_key)
        self.assertIn("track_frame_mask_i32", schema_by_key)
        self.assertIn("track_change_offsets_i32", schema_by_key)
        self.assertNotIn("base_offsets_i16", schema_by_key)
        self.assertNotIn("frame_select_i16", schema_by_key)
        self.assertNotIn("track_chunk_change_offsets_i16", schema_by_key)
        self.assertNotIn("change_frame_i16", schema_by_key)
        self.assertEqual(int(schema_by_key["track_frame_mask_i32"]), int(framebitmask_tape["track_count"]) * 4)
        self.assertLess(
            framebitmask_tape["selected_schema_storage_bytes"],
            frameselect_tape["selected_schema_storage_bytes"],
        )
        self.assertLess(
            framebitmask_tape["selected_schema_topology_storage_bytes"],
            frameselect_tape["selected_schema_topology_storage_bytes"],
        )

    def test_selected_only_framebitmask_prep_skips_baseline_segment_tape(self) -> None:
        frame_count = 4
        common_kwargs, _targets = self._common_moving_ray_fixture(frame_count=frame_count, site_count=8)
        full_metric_tape = _prepare_owner_run_tapes(
            **common_kwargs,
            tape_mode=OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            endpoint_record_source="slow-owner-run",
        )
        selected_only_tape = _prepare_owner_run_tapes(
            **common_kwargs,
            tape_mode=OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            endpoint_record_source="slow-owner-run",
            experimental_selected_only_owner_run_delta_prep=True,
        )

        self.assertFalse(selected_only_tape["baseline_segment_metrics_built"])
        self.assertTrue(selected_only_tape["experimental_selected_only_owner_run_delta_prep"])
        self.assertNotIn("build_segment_tape_s", selected_only_tape["prepare_timings"])
        self.assertNotIn("compact_baseline_tapes_s", selected_only_tape["prepare_timings"])
        self.assertEqual(selected_only_tape["selected_segments"], full_metric_tape["selected_segments"])
        self.assertEqual(selected_only_tape["owner_run_segments"], selected_only_tape["selected_segments"])
        self.assertEqual(selected_only_tape["endpoint_run_segments"], selected_only_tape["selected_segments"])
        self.assertEqual(selected_only_tape["full_segments"], selected_only_tape["selected_segments"])
        self.assertIn("delta_packed_framebitmask_factorized_recompute_fused_mse", selected_only_tape["selected_device"])

    def test_framebitmask_keeps_i32_base_offsets_when_offsets_exceed_i16(self) -> None:
        common_kwargs, _targets = self._common_moving_ray_fixture(
            frame_count=2,
            render_size=96,
            site_count=48,
        )
        tape = _prepare_owner_run_tapes(
            **common_kwargs,
            tape_mode=OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            endpoint_record_source="slow-owner-run",
            experimental_selected_only_owner_run_delta_prep=True,
            experimental_native_owner_run_cutwalk_delta=True,
        )
        selected_device = tape["selected_device"]
        schema_by_key = tape["selected_schema_storage_by_key"]

        self.assertIn("base_offsets_i32", selected_device)
        self.assertEqual(selected_device["base_offsets_i32"].dtype, torch.int32)
        self.assertNotIn("base_offsets_i16", selected_device)
        self.assertGreater(int(selected_device["base_offsets_i32"].detach().cpu().max().item()), 32767)
        self.assertIn("base_offsets_i32", schema_by_key)
        self.assertNotIn("base_offsets_i16", schema_by_key)
        self.assertNotIn("unattributed_storage", schema_by_key)
        self.assertEqual(
            sum(int(value) for value in schema_by_key.values()),
            tape["selected_schema_storage_bytes"],
        )
        self.assertFalse(
            tape["selected_schema_i16_meta_projection_fields"]["base_offsets_i32"]["eligible"]
        )

    def _assert_native_cutwalk_framebitmask_matches_python_shader_output(
        self,
        *,
        common_kwargs: dict[str, object],
        targets: torch.Tensor,
        frame_count: int,
    ) -> None:
        python_tape = _prepare_owner_run_tapes(
            **common_kwargs,
            tape_mode=OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            endpoint_record_source="slow-owner-run",
            experimental_selected_only_owner_run_delta_prep=True,
        )
        native_tape = _prepare_owner_run_tapes(
            **common_kwargs,
            tape_mode=OWNER_RUN_DELTA_PACKED_FACTORIZED_FRAMEBITMASK_RECOMPUTE_FUSED_MSE_NOMID_MODE,
            endpoint_record_source="slow-owner-run",
            experimental_selected_only_owner_run_delta_prep=True,
            experimental_native_owner_run_cutwalk_delta=True,
        )

        self.assertTrue(native_tape["experimental_native_owner_run_cutwalk_delta"])
        self.assertFalse(native_tape["baseline_segment_metrics_built"])
        self.assertEqual(native_tape["selected_segments"], python_tape["selected_segments"])
        for key in (
            "base_offsets_i32",
            "delta_base_record_i32",
            "track_change_offsets_i32",
            "track_frame_mask_i32",
            "change_offsets_i32",
            "delta_change_record_i32",
        ):
            self.assertTrue(
                torch.equal(
                    native_tape["selected_device"][key].detach().cpu(),
                    python_tape["selected_device"][key].detach().cpu(),
                ),
                msg=f"native cutwalk selected-device tensor differs for {key}",
            )

        device = torch.device("mps")
        site_rgba = common_kwargs["site_rgba"].to(device=device).contiguous()
        view_count = int(targets.shape[0] // frame_count)
        _channels, height, width = targets.shape[1:]
        target_track = _track_major_rgb_from_image(
            targets.to(device=device),
            view_count=view_count,
            frame_count=frame_count,
            height=int(height),
            width=int(width),
        )
        op_config = RealRayReplayConfig(
            near=common_kwargs["near"],
            far=common_kwargs["far"],
            invalid_epsilon=common_kwargs["invalid_epsilon"],
            transmittance_threshold=common_kwargs["transmittance_threshold"],
        )
        python_loss, python_grad = _delta_replace_coeff16_fused_mse_loss_vjp(
            tape_device=python_tape["selected_device"],
            site_rgba=site_rgba,
            target_rgb_track=target_track,
            op_config=op_config,
            track_count=python_tape["track_count"],
            frame_count=frame_count,
        )
        native_loss, native_grad = _delta_replace_coeff16_fused_mse_loss_vjp(
            tape_device=native_tape["selected_device"],
            site_rgba=site_rgba,
            target_rgb_track=target_track,
            op_config=op_config,
            track_count=native_tape["track_count"],
            frame_count=frame_count,
        )
        self.assertLessEqual(float((python_loss - native_loss).abs().cpu().item()), 5.0e-6)
        self.assertLessEqual(float((python_grad - native_grad).abs().max().cpu().item()), 1.0e-5)

    def test_native_cutwalk_framebitmask_matches_python_sequence_shader_output(self) -> None:
        frame_count = 4
        common_kwargs, targets = self._common_moving_ray_fixture(frame_count=frame_count, site_count=8)
        self._assert_native_cutwalk_framebitmask_matches_python_shader_output(
            common_kwargs=common_kwargs,
            targets=targets,
            frame_count=frame_count,
        )

    def test_native_cutwalk_framebitmask_shader_output_matches_python_for_multiview_moving_rays(self) -> None:
        frame_count = 4
        common_kwargs, targets = self._common_moving_ray_fixture(frame_count=frame_count, render_size=6, site_count=6)
        base_view_count = int(targets.shape[0] // frame_count)
        common_kwargs, targets = _duplicate_fixture_with_shifted_second_view(
            common_kwargs=common_kwargs,
            targets=targets,
        )
        self.assertGreaterEqual(base_view_count, 1)
        self.assertEqual(int(targets.shape[0] // frame_count), 2 * base_view_count)

        self._assert_native_cutwalk_framebitmask_matches_python_shader_output(
            common_kwargs=common_kwargs,
            targets=targets,
            frame_count=frame_count,
        )

    def _one_track_frame31_framebitmask_fixture(
        self,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, RealRayReplayConfig]:
        device = torch.device("mps")
        frame_count = 32
        base_record_i32 = _pack_endpoint_records_i32(
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([-1], dtype=torch.int32),
            torch.tensor([-2], dtype=torch.int32),
            site_count=2,
            boundary_count=1,
        ).to(device=device)
        change_record_i32 = _pack_endpoint_records_i32(
            torch.tensor([1], dtype=torch.int32),
            torch.tensor([-1], dtype=torch.int32),
            torch.tensor([-2], dtype=torch.int32),
            site_count=2,
            boundary_count=1,
        ).to(device=device)
        common_tape = {
            "delta_packed_framebitmask_factorized_recompute_fused_mse": torch.tensor(
                [1],
                dtype=torch.int32,
                device=device,
            ),
            "boundary_f32": torch.zeros((1, 5), dtype=torch.float32, device=device),
            "track_ray_coeff_f32": torch.zeros((1, 12), dtype=torch.float32, device=device),
            "frame_t_f32": torch.linspace(0.0, 1.0, frame_count, dtype=torch.float32, device=device),
            "base_offsets_i32": torch.tensor([0, 1], dtype=torch.int32, device=device),
            "delta_base_record_i32": base_record_i32,
            "track_change_offsets_i32": torch.tensor([0, 1], dtype=torch.int32, device=device),
            "change_offsets_i32": torch.tensor([0, 1], dtype=torch.int32, device=device),
            "delta_change_record_i32": change_record_i32,
            "delta_coeff_boundary_count": 1,
            "delta_launch_boundary_count": 1,
            "delta_launch_track_count": 1,
            "delta_launch_frame_count": frame_count,
            "delta_launch_site_count": 2,
            "delta_launch_base_record_count": 1,
            "delta_launch_change_count": 1,
            "delta_launch_change_record_count": 1,
        }
        site_rgba = torch.tensor(
            [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        target_track = torch.zeros((1, frame_count, 3), dtype=torch.float32, device=device)
        op_config = RealRayReplayConfig(
            near=0.0,
            far=1.0,
            invalid_epsilon=1.0e-7,
            transmittance_threshold=0.0,
        )
        return common_tape, site_rgba, target_track, op_config

    def test_framebitmask_shader_uses_signed_frame31_mask_bit(self) -> None:
        frame_count = 32
        common_tape, site_rgba, target_track, op_config = self._one_track_frame31_framebitmask_fixture()
        device = torch.device("mps")
        signbit_tape = {
            **common_tape,
            "track_frame_mask_i32": torch.tensor([-(1 << 31)], dtype=torch.int32, device=device),
        }
        self._stamp_delta_launch_contract(signbit_tape, site_rgba=site_rgba)
        all_base_tape = {
            **common_tape,
            "track_change_offsets_i32": torch.tensor([0, 0], dtype=torch.int32, device=device),
            "track_frame_mask_i32": torch.tensor([0], dtype=torch.int32, device=device),
            "change_offsets_i32": torch.tensor([0], dtype=torch.int32, device=device),
            "delta_change_record_i32": torch.empty((0,), dtype=torch.int32, device=device),
            "delta_launch_change_count": 0,
            "delta_launch_change_record_count": 0,
        }
        self._stamp_delta_launch_contract(all_base_tape, site_rgba=site_rgba)
        signbit_loss, signbit_grad = _delta_replace_coeff16_fused_mse_loss_vjp(
            tape_device=signbit_tape,
            site_rgba=site_rgba,
            target_rgb_track=target_track,
            op_config=op_config,
            track_count=1,
            frame_count=frame_count,
        )
        all_base_loss, all_base_grad = _delta_replace_coeff16_fused_mse_loss_vjp(
            tape_device=all_base_tape,
            site_rgba=site_rgba,
            target_rgb_track=target_track,
            op_config=op_config,
            track_count=1,
            frame_count=frame_count,
        )
        torch.mps.synchronize()

        frame_loss = float((1.0 - torch.exp(torch.tensor(-1.0))).pow(2).item()) / float(frame_count * 3)
        loss_drop = float((all_base_loss - signbit_loss).detach().cpu().item())
        self.assertGreater(loss_drop, 0.0)
        self.assertLessEqual(abs(loss_drop - frame_loss), 2.0e-6)
        self.assertGreater(float((all_base_grad - signbit_grad).abs().sum().detach().cpu().item()), 0.0)

    def test_framebitmask_shader_rejects_mask_change_count_mismatch(self) -> None:
        frame_count = 32
        common_tape, site_rgba, target_track, op_config = self._one_track_frame31_framebitmask_fixture()
        bad_tape = {
            **common_tape,
            "track_frame_mask_i32": torch.tensor([0], dtype=torch.int32, device=torch.device("mps")),
        }
        self._stamp_delta_launch_contract(bad_tape, site_rgba=site_rgba)

        with self.assertRaisesRegex(ValueError, "popcount must match"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=bad_tape,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=1,
                frame_count=frame_count,
            )

    def test_framebitmask_shader_rejects_empty_change_offsets(self) -> None:
        frame_count = 32
        common_tape, site_rgba, target_track, op_config = self._one_track_frame31_framebitmask_fixture()
        device = torch.device("mps")
        bad_tape = {
            **common_tape,
            "track_change_offsets_i32": torch.tensor([0, 0], dtype=torch.int32, device=device),
            "track_frame_mask_i32": torch.tensor([0], dtype=torch.int32, device=device),
            "change_offsets_i32": torch.empty((0,), dtype=torch.int32, device=device),
            "delta_change_record_i32": torch.empty((0,), dtype=torch.int32, device=device),
            "delta_launch_change_count": 0,
            "delta_launch_change_record_count": 0,
        }
        self._stamp_delta_launch_contract(bad_tape, site_rgba=site_rgba)

        with self.assertRaisesRegex(ValueError, "change_offsets_i32 must contain at least one offset"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=bad_tape,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=1,
                frame_count=frame_count,
            )

    def test_framebitmask_shader_rejects_base_record_owner_out_of_range(self) -> None:
        frame_count = 32
        common_tape, site_rgba, target_track, op_config = self._one_track_frame31_framebitmask_fixture()
        device = torch.device("mps")
        bad_base_record_i32 = torch.tensor([2 | (0 << 8) | (1 << 20)], dtype=torch.int32, device=device)
        bad_tape = {
            **common_tape,
            "delta_base_record_i32": bad_base_record_i32,
            "track_frame_mask_i32": torch.tensor([-(1 << 31)], dtype=torch.int32, device=device),
        }
        self._stamp_delta_launch_contract(bad_tape, site_rgba=site_rgba)

        with self.assertRaisesRegex(ValueError, "base_record_i32 owner code must be < site_count"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=bad_tape,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=1,
                frame_count=frame_count,
            )

    def test_framebitmask_shader_rejects_change_record_cut_out_of_range(self) -> None:
        frame_count = 32
        common_tape, site_rgba, target_track, op_config = self._one_track_frame31_framebitmask_fixture()
        device = torch.device("mps")
        bad_change_record_i32 = torch.tensor([1 | (3 << 8) | (1 << 20)], dtype=torch.int32, device=device)
        bad_tape = {
            **common_tape,
            "delta_change_record_i32": bad_change_record_i32,
            "track_frame_mask_i32": torch.tensor([-(1 << 31)], dtype=torch.int32, device=device),
        }
        self._stamp_delta_launch_contract(bad_tape, site_rgba=site_rgba)

        with self.assertRaisesRegex(ValueError, "change_record_i32 left cut id must be < boundary_count"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=bad_tape,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=1,
                frame_count=frame_count,
            )

    def test_framebitmask_shader_rejects_frame0_mask_bit(self) -> None:
        frame_count = 32
        common_tape, site_rgba, target_track, op_config = self._one_track_frame31_framebitmask_fixture()
        bad_tape = {
            **common_tape,
            "track_frame_mask_i32": torch.tensor([1], dtype=torch.int32, device=torch.device("mps")),
        }
        self._stamp_delta_launch_contract(bad_tape, site_rgba=site_rgba)

        with self.assertRaisesRegex(ValueError, "bits outside"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=bad_tape,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=1,
                frame_count=frame_count,
            )

    def test_framebitmask_shader_rejects_mask_bit_at_frame_count_boundary(self) -> None:
        frame_count = 4
        common_tape, site_rgba, _target_track, op_config = self._one_track_frame31_framebitmask_fixture()
        device = torch.device("mps")
        bad_tape = {
            **common_tape,
            "frame_t_f32": torch.linspace(0.0, 1.0, frame_count, dtype=torch.float32, device=device),
            "track_frame_mask_i32": torch.tensor([1 << frame_count], dtype=torch.int32, device=device),
            "delta_launch_frame_count": frame_count,
        }
        target_track = torch.zeros((1, frame_count, 3), dtype=torch.float32, device=device)
        self._stamp_delta_launch_contract(bad_tape, site_rgba=site_rgba)

        with self.assertRaisesRegex(ValueError, "bits outside"):
            _delta_replace_coeff16_fused_mse_loss_vjp(
                tape_device=bad_tape,
                site_rgba=site_rgba,
                target_rgb_track=target_track,
                op_config=op_config,
                track_count=1,
                frame_count=frame_count,
            )

    def test_native_cutwalk_framebitmask_shader_output_matches_python_at_32_frame_boundary(self) -> None:
        frame_count = 32
        common_kwargs, targets = _synthetic_moving_ray_fixture(frame_count=frame_count)

        self.assertEqual(int(common_kwargs["frame_indices"].max().item()), 31)
        self._assert_native_cutwalk_framebitmask_matches_python_shader_output(
            common_kwargs=common_kwargs,
            targets=targets,
            frame_count=frame_count,
        )


if __name__ == "__main__":
    unittest.main()
