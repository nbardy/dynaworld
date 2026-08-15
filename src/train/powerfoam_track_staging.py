"""Memory-bounded ``B_p x K`` target/ray staging for track-native fitters.

The legacy PowerFoam trainer still consumes full images.  This module is the
data-side seam for a track-native fitter: it decodes at most one target frame
at a time on CPU, gathers explicit row-major pixels, generates only those rays,
and transfers only compact ``[B_p,K,*]`` tensors.  It deliberately does not
approximate moving cameras.  Fixed calibrated cameras receive an exact affine
ray program; moving cameras require a future certified piecewise-affine gauge
compiler and fail closed when that program is requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from camera import CameraSpec, build_camera_rays_at_pixels
from paper_training_protocol import normalize_image_size, resize_video_frames
from powerfoam_training_data import PowerFoamRayProvider, PowerFoamTargetProvider


class AffineRayProgramUnavailableError(ValueError):
    """The requested camera path has no certified affine ray program."""


@dataclass(frozen=True)
class PowerFoamTrackLossNormalization:
    """One global denominator shared by every partition of a staging plan."""

    global_track_count: int
    global_sample_count: int
    block_track_count: int
    block_sample_count: int
    rgb_channel_count: int = 3
    global_view_count: int | None = None
    global_temporal_sample_count: int | None = None

    @property
    def global_rgb_element_count(self) -> int:
        return self.global_track_count * self.global_sample_count * self.rgb_channel_count

    @property
    def block_rgb_element_count(self) -> int:
        return self.block_track_count * self.block_sample_count * self.rgb_channel_count

    @property
    def block_fraction(self) -> float:
        return self.block_rgb_element_count / self.global_rgb_element_count

    @property
    def view_track_global_track_count(self) -> int:
        if self.global_view_count is None:
            raise ValueError("selected samples do not form a rectangular view-time grid")
        return self.global_track_count * self.global_view_count

    @property
    def view_track_global_sample_count(self) -> int:
        if self.global_temporal_sample_count is None:
            raise ValueError("selected samples do not form a rectangular view-time grid")
        return self.global_temporal_sample_count

    def as_dict(self) -> dict[str, int | float | None]:
        return {
            "global_track_count": self.global_track_count,
            "global_sample_count": self.global_sample_count,
            "block_track_count": self.block_track_count,
            "block_sample_count": self.block_sample_count,
            "rgb_channel_count": self.rgb_channel_count,
            "global_rgb_element_count": self.global_rgb_element_count,
            "block_rgb_element_count": self.block_rgb_element_count,
            "block_fraction": self.block_fraction,
            "global_view_count": self.global_view_count,
            "global_temporal_sample_count": self.global_temporal_sample_count,
        }


@dataclass(frozen=True)
class FixedCameraAffineRayProgram:
    """Exact ``o(t)=o0+t*o1, d(t)=d0+t*d1`` rows for fixed views.

    ``coefficients`` has shape ``[active_views,B_p,12]`` in
    ``[o0,o1,d0,d1]`` order.  ``sample_program_indices`` maps each of the K
    staged samples to its active-view row, so the representation remains
    ``O(active_views * B_p + K)`` rather than ``O(P * F)``.
    """

    coefficients: torch.Tensor
    view_indices: torch.Tensor
    sample_program_indices: torch.Tensor
    sample_times: torch.Tensor

    def __post_init__(self) -> None:
        if self.coefficients.ndim != 3 or int(self.coefficients.shape[-1]) != 12:
            raise ValueError("affine ray coefficients must have shape [active_views,B_p,12]")
        if self.view_indices.ndim != 1 or int(self.view_indices.numel()) != int(self.coefficients.shape[0]):
            raise ValueError("affine ray view ids must match the coefficient view axis")
        if self.sample_program_indices.ndim != 1 or self.sample_times.ndim != 1:
            raise ValueError("affine ray sample mappings and times must be one-dimensional")
        if self.sample_program_indices.shape != self.sample_times.shape:
            raise ValueError("affine ray sample mappings and times must have equal length")
        if self.sample_program_indices.numel() and (
            int(self.sample_program_indices.min()) < 0
            or int(self.sample_program_indices.max()) >= int(self.coefficients.shape[0])
        ):
            raise ValueError("affine ray sample mapping contains an out-of-range program row")
        if self.coefficients.requires_grad:
            raise ValueError("fixed-camera affine ray coefficients must not require gradients")

    @property
    def resident_bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for value in (
                self.coefficients,
                self.view_indices,
                self.sample_program_indices,
                self.sample_times,
            )
        )

    @torch.no_grad()
    def evaluate(self) -> torch.Tensor:
        """Return rays with the output tensor plus only ``O(B_p)`` scratch."""

        rays = torch.empty(
            (int(self.coefficients.shape[1]), int(self.sample_times.numel()), 6),
            dtype=self.coefficients.dtype,
            device=self.coefficients.device,
        )
        for slot, (program_row, time) in enumerate(
            zip(
                self.sample_program_indices.tolist(),
                self.sample_times.unbind(),
                strict=True,
            )
        ):
            coefficients = self.coefficients[int(program_row)]
            rays[:, slot, 0:3] = coefficients[:, 0:3] + time * coefficients[:, 3:6]
            rays[:, slot, 3:6] = coefficients[:, 6:9] + time * coefficients[:, 9:12]
        return rays


@dataclass(frozen=True)
class PowerFoamViewTrackStageBlock:
    """A rectangular native layout with ``(view,pixel)`` as the track axis."""

    source_view_indices: torch.Tensor
    source_pixel_indices: torch.Tensor
    frame_indices: torch.Tensor
    sample_times: torch.Tensor
    targets: torch.Tensor
    rays: torch.Tensor
    ray_coefficients: torch.Tensor
    global_track_count: int
    global_sample_count: int
    global_rgb_element_count: int
    accounting: dict[str, Any]

    def __post_init__(self) -> None:
        local_track_count = int(self.source_view_indices.numel())
        local_sample_count = int(self.sample_times.numel())
        if self.source_pixel_indices.shape != self.source_view_indices.shape:
            raise ValueError("view-track source view and pixel ids must have equal shape")
        if self.frame_indices.shape != self.sample_times.shape:
            raise ValueError("view-track frame ids and times must have equal shape")
        if tuple(self.targets.shape) != (local_track_count, local_sample_count, 3):
            raise ValueError("view-track targets must have shape [view*B_p,K,3]")
        if tuple(self.rays.shape) != (local_track_count, local_sample_count, 6):
            raise ValueError("view-track rays must have shape [view*B_p,K,6]")
        if tuple(self.ray_coefficients.shape) != (local_track_count, 12):
            raise ValueError("view-track affine rays must have shape [view*B_p,12]")
        if self.global_track_count < local_track_count or self.global_sample_count < local_sample_count:
            raise ValueError("view-track global dimensions must cover the local block")
        if self.global_rgb_element_count != self.global_track_count * self.global_sample_count * 3:
            raise ValueError("view-track dimensions must preserve the global RGB denominator")


@dataclass(frozen=True)
class PowerFoamTrackStageBlock:
    pixel_indices: torch.Tensor
    sample_indices: torch.Tensor
    view_indices: torch.Tensor
    frame_indices: torch.Tensor
    sample_times: torch.Tensor
    targets: torch.Tensor
    rays: torch.Tensor
    normalization: PowerFoamTrackLossNormalization
    accounting: dict[str, Any]
    affine_ray_program: FixedCameraAffineRayProgram | None
    affine_ray_program_unavailable_reason: str | None

    def __post_init__(self) -> None:
        expected_targets = (int(self.pixel_indices.numel()), int(self.sample_indices.numel()), 3)
        expected_rays = (*expected_targets[:2], 6)
        if tuple(self.targets.shape) != expected_targets:
            raise ValueError(f"track targets must have shape {expected_targets}, got {tuple(self.targets.shape)}")
        if tuple(self.rays.shape) != expected_rays:
            raise ValueError(f"track rays must have shape {expected_rays}, got {tuple(self.rays.shape)}")
        if self.targets.requires_grad or self.rays.requires_grad:
            raise ValueError("fixed-camera staged targets and rays must not require gradients")

    @torch.no_grad()
    def as_view_tracks(self) -> PowerFoamViewTrackStageBlock:
        """Move camera view from the sample axis onto the native track axis.

        Native fixed-word replay owns one affine ray program per track. A
        rectangular multi-view observation block is therefore represented as
        ``(view,pixel) x time``. The reshape is exact and preserves
        ``P * (V K) * 3 == (V P) * K * 3``.
        """

        if self.affine_ray_program is None:
            raise AffineRayProgramUnavailableError(
                self.affine_ray_program_unavailable_reason
                or "view-track conversion requires a certified affine ray program"
            )
        normalization = self.normalization
        if (
            normalization.global_view_count is None
            or normalization.global_temporal_sample_count is None
        ):
            raise ValueError("global samples do not form a rectangular view-time grid")
        view_rows = []
        reference_frames = None
        reference_times = None
        target_rows = []
        ray_rows = []
        coefficient_rows = []
        source_views = []
        source_pixels = []
        for program_row, view_tensor in enumerate(self.affine_ray_program.view_indices):
            view = int(view_tensor.item())
            slots = torch.nonzero(self.view_indices == view, as_tuple=False).reshape(-1)
            if slots.numel() == 0:
                continue
            order = torch.argsort(self.sample_times.index_select(0, slots), stable=True)
            slots = slots.index_select(0, order)
            frames = self.frame_indices.index_select(0, slots)
            times = self.sample_times.index_select(0, slots)
            if reference_frames is None:
                reference_frames = frames
                reference_times = times
            elif not torch.equal(reference_frames, frames) or not torch.equal(reference_times, times):
                raise ValueError("each staged view must contain the same ordered frame/time grid")
            view_rows.append(view)
            target_rows.append(self.targets.index_select(1, slots.to(device=self.targets.device)))
            ray_rows.append(self.rays.index_select(1, slots.to(device=self.rays.device)))
            coefficient_rows.append(self.affine_ray_program.coefficients[program_row])
            source_views.append(
                torch.full_like(self.pixel_indices, view, dtype=torch.long)
            )
            source_pixels.append(self.pixel_indices)
        if not view_rows or reference_frames is None or reference_times is None:
            raise ValueError("view-track conversion requires at least one staged view")
        targets = torch.cat(target_rows, dim=0).contiguous()
        rays = torch.cat(ray_rows, dim=0).contiguous()
        ray_coefficients = torch.cat(coefficient_rows, dim=0).contiguous()
        if self.rays.device.type == "cpu":
            if not _affine_program_matches_staged_rays_bounded(
                self.affine_ray_program,
                self.rays,
            ):
                raise ValueError("affine ray program no longer reproduces the staged rays exactly")
        elif any(
            int(tensor._version) != 0
            for tensor in (
                self.rays,
                self.affine_ray_program.coefficients,
                self.affine_ray_program.sample_program_indices,
                self.affine_ray_program.sample_times,
            )
        ):
            raise ValueError("accelerator view-track tensors changed after bounded staging")
        global_track_count = normalization.view_track_global_track_count
        global_sample_count = normalization.view_track_global_sample_count
        local_rgb_elements = int(targets.numel())
        return PowerFoamViewTrackStageBlock(
            source_view_indices=torch.cat(source_views).contiguous(),
            source_pixel_indices=torch.cat(source_pixels).contiguous(),
            frame_indices=reference_frames.contiguous(),
            sample_times=reference_times.contiguous(),
            targets=targets,
            rays=rays,
            ray_coefficients=ray_coefficients,
            global_track_count=global_track_count,
            global_sample_count=global_sample_count,
            global_rgb_element_count=normalization.global_rgb_element_count,
            accounting={
                **self.accounting,
                "layout": "view_pixel_tracks_by_time",
                "local_view_count": len(view_rows),
                "local_track_count": int(targets.shape[0]),
                "local_sample_count": int(targets.shape[1]),
                "local_rgb_element_count": local_rgb_elements,
                "global_track_count": global_track_count,
                "global_sample_count": global_sample_count,
                "global_rgb_element_count": normalization.global_rgb_element_count,
                "denominator_preserved": True,
            },
        )


@dataclass(frozen=True)
class PowerFoamTrackTargetStageBlock:
    """Target-only ``B_p x K`` partition for the material-training hot path.

    This is intentionally a separate type from :class:`PowerFoamTrackStageBlock`.
    Evaluation callers keep the stronger explicit-ray contract, while material
    training cannot accidentally retain or consume a nullable ray payload.
    """

    pixel_indices: torch.Tensor
    sample_indices: torch.Tensor
    view_indices: torch.Tensor
    frame_indices: torch.Tensor
    sample_times: torch.Tensor
    targets: torch.Tensor
    normalization: PowerFoamTrackLossNormalization
    accounting: dict[str, Any]

    def __post_init__(self) -> None:
        expected_targets = (int(self.pixel_indices.numel()), int(self.sample_indices.numel()), 3)
        if tuple(self.targets.shape) != expected_targets:
            raise ValueError(
                f"track targets must have shape {expected_targets}, got {tuple(self.targets.shape)}"
            )
        if self.targets.requires_grad:
            raise ValueError("staged targets must not require gradients")
        if self.accounting.get("ray_bytes") != 0 or self.accounting.get("explicit_rays_staged") is not False:
            raise ValueError("target-only staging accounting must contain no explicit ray payload")


@dataclass(frozen=True)
class PowerFoamTrackStagingPlan:
    """Validated global selection which can be partitioned without renormalizing."""

    target_provider: PowerFoamTargetProvider
    ray_provider: PowerFoamRayProvider
    pixel_indices: torch.Tensor
    sample_indices: torch.Tensor
    height: int | None = None
    width: int | None = None
    sample_times: torch.Tensor | None = None
    device: torch.device | str | None = None

    def __post_init__(self) -> None:
        height = self.target_provider.height if self.height is None else int(self.height)
        width = self.target_provider.width if self.width is None else int(self.width)
        if height < 1 or width < 1:
            raise ValueError("track staging requires positive target dimensions")
        if (self.target_provider.view_count, self.target_provider.frame_count) != (
            self.ray_provider.view_count,
            self.ray_provider.frame_count,
        ):
            raise ValueError("target and ray providers must have the same view/frame grid")
        if (self.target_provider.height, self.target_provider.width) != (
            self.ray_provider.height,
            self.ray_provider.width,
        ):
            raise ValueError("target and ray providers must share source image dimensions")
        if not self.ray_provider.cameras or any(
            len(cameras) != self.ray_provider.frame_count for cameras in self.ray_provider.cameras
        ):
            raise ValueError("ray provider cameras must form a non-empty rectangular view/frame grid")

        pixels = _validated_indices(
            "pixel_indices",
            self.pixel_indices,
            upper_bound=height * width,
        )
        samples = _validated_indices(
            "sample_indices",
            self.sample_indices,
            upper_bound=self.target_provider.sample_count,
        )
        if self.sample_times is None:
            frame_indices = torch.remainder(samples, self.target_provider.frame_count)
            times = (
                torch.zeros_like(frame_indices, dtype=torch.float32)
                if self.target_provider.frame_count == 1
                else frame_indices.to(dtype=torch.float32) / float(self.target_provider.frame_count - 1)
            )
        else:
            raw_times = torch.as_tensor(self.sample_times)
            if raw_times.ndim != 1 or int(raw_times.numel()) != int(samples.numel()):
                raise ValueError("sample_times must be one-dimensional with one value per logical sample")
            times = raw_times.detach().to(device="cpu", dtype=torch.float32)
            if not bool(torch.isfinite(times).all()):
                raise ValueError("sample_times must be finite")

        object.__setattr__(self, "height", height)
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "pixel_indices", pixels)
        object.__setattr__(self, "sample_indices", samples)
        object.__setattr__(self, "sample_times", times.contiguous())
        object.__setattr__(
            self,
            "device",
            torch.device(self.target_provider.device if self.device is None else self.device),
        )

    @property
    def track_count(self) -> int:
        return int(self.pixel_indices.numel())

    @property
    def sample_count(self) -> int:
        return int(self.sample_indices.numel())

    @torch.no_grad()
    def assert_fixed_camera_affine_coefficients(
        self,
        coefficients: torch.Tensor,
        *,
        track_start: int = 0,
        track_end: int | None = None,
    ) -> None:
        """Validate one view-local immutable ``[B_p,12]`` ray program.

        Only one ``[B_p,1,6]`` reference row is generated.  Fixed-camera
        equality across the complete provider frame axis proves that the same
        row applies to every selected time; nonzero affine slopes and moving
        cameras therefore fail closed rather than being endpoint-fit.
        """

        track_start, track_end = _partition_bounds(
            "track",
            track_start,
            track_end,
            self.track_count,
        )
        pixels = self.pixel_indices[track_start:track_end]
        views = torch.div(
            self.sample_indices,
            self.target_provider.frame_count,
            rounding_mode="floor",
        )
        active_views = tuple(sorted({int(value) for value in views.tolist()}))
        if len(active_views) != 1:
            raise AffineRayProgramUnavailableError(
                "fixed-camera affine validation requires one view-local staging plan"
            )
        _reject_camera_gradients(self.ray_provider, self.sample_indices)
        affine_reason = _fixed_camera_program_unavailable_reason(
            self.ray_provider,
            active_views,
        )
        if affine_reason is not None:
            raise AffineRayProgramUnavailableError(affine_reason)

        expected = torch.as_tensor(coefficients)
        expected_shape = (int(pixels.numel()), 12)
        if tuple(expected.shape) != expected_shape:
            raise ValueError(
                f"fixed-camera affine coefficients must have shape {expected_shape}, got {tuple(expected.shape)}"
            )
        if expected.dtype != torch.float32 or expected.requires_grad:
            raise ValueError("fixed-camera affine coefficients must be frozen float32")
        reference_sample = self.sample_indices[:1]
        reference = _stage_selected_rays(
            self.ray_provider,
            pixels,
            reference_sample,
            height=self.height,
            width=self.width,
            device=expected.device,
        )[:, 0]
        zeros = torch.zeros_like(reference[:, :3])
        if not (
            torch.equal(expected[:, 0:3], reference[:, 0:3])
            and torch.equal(expected[:, 3:6], zeros)
            and torch.equal(expected[:, 6:9], reference[:, 3:6])
            and torch.equal(expected[:, 9:12], zeros)
        ):
            raise ValueError("fixed-camera affine program does not match the certified live track rays")

    @torch.no_grad()
    def stage_targets(
        self,
        *,
        track_start: int = 0,
        track_end: int | None = None,
        sample_start: int = 0,
        sample_end: int | None = None,
    ) -> PowerFoamTrackTargetStageBlock:
        """Stage one target-only partition while retaining the plan denominator."""

        track_start, track_end = _partition_bounds(
            "track",
            track_start,
            track_end,
            self.track_count,
        )
        sample_start, sample_end = _partition_bounds(
            "sample",
            sample_start,
            sample_end,
            self.sample_count,
        )
        pixels = self.pixel_indices[track_start:track_end]
        samples = self.sample_indices[sample_start:sample_end]
        times = self.sample_times[sample_start:sample_end]
        views = torch.div(samples, self.target_provider.frame_count, rounding_mode="floor")
        frames = torch.remainder(samples, self.target_provider.frame_count)
        targets = _stage_target_pixels(
            self.target_provider,
            pixels,
            views,
            frames,
            height=self.height,
            width=self.width,
            device=self.device,
        )
        global_view_count, global_temporal_sample_count = _rectangular_view_time_shape(
            self.sample_indices,
            self.sample_times,
            frame_count=self.target_provider.frame_count,
        )
        normalization = PowerFoamTrackLossNormalization(
            global_track_count=self.track_count,
            global_sample_count=self.sample_count,
            block_track_count=int(pixels.numel()),
            block_sample_count=int(samples.numel()),
            global_view_count=global_view_count,
            global_temporal_sample_count=global_temporal_sample_count,
        )
        accounting = _staging_accounting(
            self.target_provider,
            track_count=int(pixels.numel()),
            sample_count=int(samples.numel()),
            height=self.height,
            width=self.width,
            device=self.device,
            affine_program=None,
            include_explicit_rays=False,
        )
        return PowerFoamTrackTargetStageBlock(
            pixel_indices=pixels,
            sample_indices=samples,
            view_indices=views,
            frame_indices=frames,
            sample_times=times,
            targets=targets,
            normalization=normalization,
            accounting=accounting,
        )

    @torch.no_grad()
    def stage(
        self,
        *,
        track_start: int = 0,
        track_end: int | None = None,
        sample_start: int = 0,
        sample_end: int | None = None,
        require_affine_ray_program: bool = False,
    ) -> PowerFoamTrackStageBlock:
        """Stage one rectangular partition while retaining the plan denominator."""

        track_start, track_end = _partition_bounds(
            "track",
            track_start,
            track_end,
            self.track_count,
        )
        sample_start, sample_end = _partition_bounds(
            "sample",
            sample_start,
            sample_end,
            self.sample_count,
        )
        pixels = self.pixel_indices[track_start:track_end]
        samples = self.sample_indices[sample_start:sample_end]
        times = self.sample_times[sample_start:sample_end]
        views = torch.div(samples, self.target_provider.frame_count, rounding_mode="floor")
        frames = torch.remainder(samples, self.target_provider.frame_count)
        active_views = tuple(sorted({int(value) for value in views.tolist()}))

        _reject_camera_gradients(
            self.ray_provider,
            samples,
        )
        affine_reason = _fixed_camera_program_unavailable_reason(
            self.ray_provider,
            active_views,
        )
        if affine_reason is not None and require_affine_ray_program:
            raise AffineRayProgramUnavailableError(affine_reason)

        targets = _stage_target_pixels(
            self.target_provider,
            pixels,
            views,
            frames,
            height=self.height,
            width=self.width,
            device=self.device,
        )
        rays = _stage_selected_rays(
            self.ray_provider,
            pixels,
            samples,
            height=self.height,
            width=self.width,
            device=self.device,
        )
        affine_program = (
            None
            if affine_reason is not None
            else _fixed_camera_affine_program(
                rays,
                views=views,
                times=times,
                active_views=active_views,
            )
        )
        global_view_count, global_temporal_sample_count = _rectangular_view_time_shape(
            self.sample_indices,
            self.sample_times,
            frame_count=self.target_provider.frame_count,
        )
        normalization = PowerFoamTrackLossNormalization(
            global_track_count=self.track_count,
            global_sample_count=self.sample_count,
            block_track_count=int(pixels.numel()),
            block_sample_count=int(samples.numel()),
            global_view_count=global_view_count,
            global_temporal_sample_count=global_temporal_sample_count,
        )
        accounting = _staging_accounting(
            self.target_provider,
            track_count=int(pixels.numel()),
            sample_count=int(samples.numel()),
            height=self.height,
            width=self.width,
            device=self.device,
            affine_program=affine_program,
        )
        return PowerFoamTrackStageBlock(
            pixel_indices=pixels,
            sample_indices=samples,
            view_indices=views,
            frame_indices=frames,
            sample_times=times,
            targets=targets,
            rays=rays,
            normalization=normalization,
            accounting=accounting,
            affine_ray_program=affine_program,
            affine_ray_program_unavailable_reason=affine_reason,
        )


def _validated_indices(name: str, values: torch.Tensor, *, upper_bound: int) -> torch.Tensor:
    raw = torch.as_tensor(values)
    if raw.ndim != 1 or raw.numel() < 1:
        raise ValueError(f"{name} must be a non-empty one-dimensional tensor")
    if raw.dtype == torch.bool or (raw.is_floating_point() and not bool(torch.equal(raw, raw.trunc()))):
        raise ValueError(f"{name} must contain integer ids")
    indices = raw.detach().to(device="cpu", dtype=torch.long).contiguous()
    if int(torch.unique(indices).numel()) != int(indices.numel()):
        raise ValueError(f"{name} must be unique")
    invalid = indices[(indices < 0) | (indices >= int(upper_bound))]
    if invalid.numel():
        raise IndexError(f"{name} value {int(invalid[0])} is outside [0, {int(upper_bound)})")
    return indices


def _rectangular_view_time_shape(
    sample_indices: torch.Tensor,
    sample_times: torch.Tensor,
    *,
    frame_count: int,
) -> tuple[int | None, int | None]:
    """Return ``(V,F)`` only when selected observations form one view-time grid."""

    views = torch.div(sample_indices, frame_count, rounding_mode="floor")
    frames = torch.remainder(sample_indices, frame_count)
    active_views = tuple(sorted({int(value) for value in views.tolist()}))
    reference_frames = None
    reference_times = None
    for view in active_views:
        slots = torch.nonzero(views == view, as_tuple=False).reshape(-1)
        order = torch.argsort(sample_times.index_select(0, slots), stable=True)
        slots = slots.index_select(0, order)
        current_frames = frames.index_select(0, slots)
        current_times = sample_times.index_select(0, slots)
        if reference_frames is None:
            reference_frames = current_frames
            reference_times = current_times
        elif not torch.equal(reference_frames, current_frames) or not torch.equal(
            reference_times,
            current_times,
        ):
            return None, None
    if reference_frames is None:
        return None, None
    return len(active_views), int(reference_frames.numel())


def _partition_bounds(name: str, start: int, end: int | None, count: int) -> tuple[int, int]:
    resolved_start = int(start)
    resolved_end = count if end is None else int(end)
    if resolved_start < 0 or resolved_end > count:
        raise IndexError(f"{name} partition [{resolved_start}, {resolved_end}) is outside [0, {count})")
    if resolved_end <= resolved_start:
        raise ValueError(f"{name} partition must be non-empty and ordered")
    return resolved_start, resolved_end


def _value_requires_grad(value: Any) -> bool:
    if torch.is_tensor(value):
        return bool(value.requires_grad)
    if isinstance(value, (tuple, list)):
        return any(_value_requires_grad(item) for item in value)
    return False


def _reject_camera_gradients(
    provider: PowerFoamRayProvider,
    sample_indices: torch.Tensor,
) -> None:
    for index in sample_indices.tolist():
        view, frame = divmod(int(index), provider.frame_count)
        camera = provider.cameras[view][frame]
        if any(
            _value_requires_grad(value)
            for value in (
                camera.fx,
                camera.fy,
                camera.cx,
                camera.cy,
                camera.camera_to_world,
                camera.distortion,
            )
        ):
            raise ValueError(
                "track staging is fixed-camera only and rejects camera gradients; "
                "use a camera-aware fitter instead of silently detaching calibration"
            )


def _camera_signature(camera: CameraSpec) -> tuple[str, tuple[torch.Tensor, ...]]:
    coefficient_count = {
        "pinhole": 0,
        "radial_tangential": 5,
        "opencv_fisheye": 4,
    }.get(camera.lens_model)
    if coefficient_count is None:
        raise ValueError(f"unsupported camera lens model {camera.lens_model!r}")
    scalars = tuple(
        torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        for value in (camera.fx, camera.fy, camera.cx, camera.cy)
    )
    if any(value.numel() != 1 for value in scalars):
        raise ValueError("fixed-camera ray programs require scalar intrinsics")
    transform = camera.camera_to_world.detach().to(device="cpu", dtype=torch.float64)
    if tuple(transform.shape) != (4, 4):
        raise ValueError("fixed-camera ray programs require camera_to_world [4,4]")
    distortion = torch.zeros(coefficient_count, dtype=torch.float64)
    if coefficient_count and camera.distortion is not None:
        supplied = torch.as_tensor(camera.distortion).detach().to(device="cpu", dtype=torch.float64).reshape(-1)
        if supplied.numel() > coefficient_count:
            raise ValueError(f"{camera.lens_model} expects at most {coefficient_count} distortion coefficients")
        distortion[: supplied.numel()] = supplied
    return camera.lens_model, (*scalars, transform, distortion)


def _cameras_equal(left: CameraSpec, right: CameraSpec) -> bool:
    left_lens, left_values = _camera_signature(left)
    right_lens, right_values = _camera_signature(right)
    return left_lens == right_lens and all(
        torch.equal(left_value, right_value) for left_value, right_value in zip(left_values, right_values, strict=True)
    )


def _fixed_camera_program_unavailable_reason(
    provider: PowerFoamRayProvider,
    active_views: tuple[int, ...],
) -> str | None:
    for view in active_views:
        reference = provider.cameras[view][0]
        if any(not _cameras_equal(reference, camera) for camera in provider.cameras[view][1:]):
            return (
                f"view {view} changes camera parameters over time; an exact affine program is not certified. "
                "The remaining fitter seam is a bounded-residual piecewise-affine/projective camera-gauge "
                "compiler, so approximate endpoint fitting is intentionally disabled."
            )
    return None


def _cpu_scaled_camera(
    camera: CameraSpec,
    *,
    scale_x: float,
    scale_y: float,
) -> CameraSpec:
    def scalar(value: float | torch.Tensor) -> float:
        tensor = torch.as_tensor(value)
        if tensor.numel() != 1:
            raise ValueError("track ray staging requires scalar camera intrinsics")
        return float(tensor.detach().to(device="cpu", dtype=torch.float32))

    distortion = (
        None
        if camera.distortion is None
        else torch.as_tensor(camera.distortion).detach().to(device="cpu", dtype=torch.float32)
    )
    return CameraSpec(
        fx=scalar(camera.fx) * scale_x,
        fy=scalar(camera.fy) * scale_y,
        cx=scalar(camera.cx) * scale_x,
        cy=scalar(camera.cy) * scale_y,
        camera_to_world=camera.camera_to_world.detach().to(device="cpu", dtype=torch.float32),
        lens_model=camera.lens_model,
        distortion=distortion,
    )


def _stage_target_pixels(
    provider: PowerFoamTargetProvider,
    pixels: torch.Tensor,
    views: torch.Tensor,
    frames: torch.Tensor,
    *,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    staged = torch.empty((pixels.numel(), views.numel(), 3), dtype=torch.float32, device="cpu")
    for slot, (view, frame) in enumerate(zip(views.tolist(), frames.tolist(), strict=True)):
        decoded = provider.source.select_view_frames((int(view),), (int(frame),))
        expected_shape = (1, 3, provider.height, provider.width)
        if tuple(decoded.shape) != expected_shape or decoded.dtype != torch.float32 or decoded.device.type != "cpu":
            raise ValueError(
                "PowerFoam target source violated the CPU normalized RGB frame contract: "
                f"expected float32 CPU {expected_shape}, got {decoded.dtype} {decoded.device} "
                f"{tuple(decoded.shape)}"
            )
        if (height, width) != (provider.height, provider.width):
            decoded = resize_video_frames(decoded, normalize_image_size((height, width)))
        gathered = decoded[0].reshape(3, height * width).index_select(1, pixels)
        staged[:, slot] = gathered.transpose(0, 1)
    return staged.to(device=device, dtype=torch.float32).contiguous()


def _stage_selected_rays(
    provider: PowerFoamRayProvider,
    pixels: torch.Tensor,
    samples: torch.Tensor,
    *,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    scale_x = float(width) / float(provider.width)
    scale_y = float(height) / float(provider.height)
    staged = torch.empty((pixels.numel(), samples.numel(), 6), dtype=torch.float32, device="cpu")
    for slot, index in enumerate(samples.tolist()):
        view, frame = divmod(int(index), provider.frame_count)
        camera = _cpu_scaled_camera(
            provider.cameras[view][frame],
            scale_x=scale_x,
            scale_y=scale_y,
        )
        origins, directions = build_camera_rays_at_pixels(
            camera,
            pixels,
            height=height,
            width=width,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        staged[:, slot] = torch.cat((origins, directions), dim=-1)
    return staged.to(device=device, dtype=torch.float32).contiguous()


def _fixed_camera_affine_program(
    rays: torch.Tensor,
    *,
    views: torch.Tensor,
    times: torch.Tensor,
    active_views: tuple[int, ...],
) -> FixedCameraAffineRayProgram:
    view_to_program = {view: index for index, view in enumerate(active_views)}
    coefficients = []
    for view in active_views:
        sample_slot = views.tolist().index(view)
        reference = rays[:, sample_slot]
        zeros = torch.zeros_like(reference[..., :3])
        coefficients.append(torch.cat((reference[..., :3], zeros, reference[..., 3:6], zeros), dim=-1))
    return FixedCameraAffineRayProgram(
        coefficients=torch.stack(coefficients).contiguous(),
        view_indices=torch.tensor(active_views, device=rays.device, dtype=torch.long),
        sample_program_indices=torch.tensor(
            [view_to_program[int(view)] for view in views.tolist()],
            device=rays.device,
            dtype=torch.long,
        ),
        sample_times=times.to(device=rays.device, dtype=rays.dtype),
    )


def _affine_program_matches_staged_rays_bounded(
    program: FixedCameraAffineRayProgram,
    rays: torch.Tensor,
) -> bool:
    """Check exact program parity with only ``O(B_p)`` temporary rows."""

    if tuple(rays.shape) != (int(program.coefficients.shape[1]), int(program.sample_times.numel()), 6):
        return False
    for slot, (program_row, time) in enumerate(
        zip(
            program.sample_program_indices.tolist(),
            program.sample_times.unbind(),
            strict=True,
        )
    ):
        coefficients = program.coefficients[int(program_row)]
        origin = coefficients[:, 0:3] + time * coefficients[:, 3:6]
        direction = coefficients[:, 6:9] + time * coefficients[:, 9:12]
        if not torch.equal(torch.cat((origin, direction), dim=-1), rays[:, slot]):
            return False
    return True


def _staging_accounting(
    provider: PowerFoamTargetProvider,
    *,
    track_count: int,
    sample_count: int,
    height: int,
    width: int,
    device: torch.device,
    affine_program: FixedCameraAffineRayProgram | None,
    include_explicit_rays: bool = True,
) -> dict[str, Any]:
    target_bytes = track_count * sample_count * 3 * torch.tensor([], dtype=torch.float32).element_size()
    full_block_ray_bytes = track_count * sample_count * 6 * torch.tensor([], dtype=torch.float32).element_size()
    ray_bytes = full_block_ray_bytes if include_explicit_rays else 0
    affine_bytes = 0 if affine_program is None else affine_program.resident_bytes
    index_metadata_bytes = track_count * 8 + sample_count * (8 + 8 + 8 + 4)
    decoded_frame_bytes = 3 * provider.height * provider.width * 4
    resized_frame_bytes = 0 if (height, width) == (provider.height, provider.width) else 3 * height * width * 4
    output_payload_bytes = target_bytes + ray_bytes + affine_bytes
    return {
        "selection_mode": "pixel_track_block",
        "track_count": track_count,
        "sample_count": sample_count,
        "target_bytes": target_bytes,
        "ray_bytes": ray_bytes,
        "explicit_rays_staged": include_explicit_rays,
        "omitted_explicit_ray_bytes": 0 if include_explicit_rays else full_block_ray_bytes,
        "affine_program_bytes": affine_bytes,
        "index_metadata_bytes": index_metadata_bytes,
        "output_payload_bytes": output_payload_bytes,
        "accelerator_transfer_bytes": 0 if device.type == "cpu" else output_payload_bytes,
        "full_pk_target_bytes": sample_count * height * width * 3 * 4,
        "full_pk_ray_bytes": sample_count * height * width * 6 * 4,
        "full_image_accelerator_resident_bytes": 0,
        "peak_decoded_frame_count": 1,
        "decoded_rgb_frame_bytes": decoded_frame_bytes,
        "resized_rgb_frame_bytes": resized_frame_bytes,
        "peak_cpu_tensor_staging_bytes_upper_bound": (
            decoded_frame_bytes
            + resized_frame_bytes
            + target_bytes
            + ray_bytes
            + index_metadata_bytes
            + (affine_bytes if device.type == "cpu" else 0)
        ),
        "source_residency": provider.residency(),
        "camera_gradients": False,
        "bounded_residency_contract": True,
    }


__all__ = [
    "AffineRayProgramUnavailableError",
    "FixedCameraAffineRayProgram",
    "PowerFoamTrackLossNormalization",
    "PowerFoamTrackStageBlock",
    "PowerFoamTrackStagingPlan",
    "PowerFoamTrackTargetStageBlock",
    "PowerFoamViewTrackStageBlock",
]
