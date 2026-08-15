"""Deterministic frame-independent initialization for native-4D WorldFoam.

The lazy kinetic provider accepts only an initializer protocol returning
``AffineKineticPowerSites``.  This module supplies the smallest production
implementation of that protocol from an explicit point-cloud asset.  It does
not inspect targets, videos, sampled frames, or cameras:

* one content-addressed PLY/COLMAP asset seeds ``positions0`` and P0 RGB;
* a mandatory power-of-two grid bounds the exact compiler's coordinate and
  weight rational bit complexity;
* velocities start at zero;
* one configured degree-``<=2`` power-weight polynomial is repeated per site;
* one configured positive density completes physical ``RGB+density`` P0
  material rows;
* every returned tensor is CPU resident and has no frame/sample axis.

The request's frame/camera/dataset fields are validation context only.  They
do not enter row selection or tensor construction.  The structural initializer
generation binds source geometry, the explicit coordinate transform,
selection policy, geometry coefficients, and final site contents.  Initial
RGB/density has a separate material-seed generation so material changes do not
invalidate structural program caches.  Raw asset bytes remain diagnostic
provenance for fail-closed source-drift detection.  Parameter storage is
invariant to requested frame density.

This initializer deliberately does not infer a world-to-model transform from
cameras and does not initialize motion from video.  External-world assets must
provide an explicit affine transform.  Richer velocity, per-site weight, and
dynamic material assets require a separate versioned schema rather than a
silent fallback here.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from paper_kinetic_runtime_paths import ensure_worldfoam_lane2_research_path

ensure_worldfoam_lane2_research_path()

from kinetic_power_word_compiler import AffineKineticPowerSites  # noqa: E402
from paper_kinetic_lazy_program_bundles import (  # noqa: E402
    PaperKineticWorldInitializationRequest,
)
from powerfoam_point_cloud import load_point_cloud_xyz_rgb, resolve_point_cloud_path  # noqa: E402


INITIALIZER_PROVENANCE = "paper-kinetic-point-cloud-world-initializer-v1"
P0_MATERIAL_PROVENANCE = "paper-kinetic-physical-p0-material-initialization-v1"
SELECTION_ALGORITHM = "content-addressed-sha256-rank-v1"
POINT_LOADER_SEMANTICS = "powerfoam-point-cloud-f32-load-promoted-to-f64-v1"
P0_MATERIAL_LAYOUT = "rgb_then_density"
MAX_QUANTIZED_INTEGER_MAGNITUDE = (1 << 31) - 1
MIN_EXACT_GRID_EXPONENT = -24
MAX_EXACT_GRID_EXPONENT = 24

_CONFIG_KEYS = frozenset(
    {
        "source_path",
        "source_coordinate_frame",
        "point_transform",
        "maximum_source_asset_bytes",
        "maximum_source_point_count",
        "site_count",
        "sample_mode",
        "sample_seed",
        "coordinate_quantization_step",
        "weight_coefficients",
        "weight_quantization_step",
        "initial_density",
    }
)
_INITIALIZER_SEAL = object()
_MATERIAL_SEAL = object()


@dataclass(frozen=True)
class PaperKineticInitializationStorageReport:
    """Logical persistent parameter bytes for one initialized world.

    ``requested_frame_count`` is carried only to make the invariance test
    explicit.  It never changes any other field.
    """

    requested_frame_count: int
    site_count: int
    geometry_parameter_bytes: int
    p0_material_parameter_bytes: int
    total_parameter_bytes: int
    stored_frame_state_bytes: int = 0
    frame_dependent_parameter_bytes: int = 0
    target_tensor_bytes: int = 0
    video_tensor_bytes: int = 0
    camera_tensor_bytes: int = 0


@dataclass(frozen=True)
class PaperKineticP0MaterialInitialization:
    """Physical time-constant material rows bound to exact site contents.

    Columns are ``[red, green, blue, density]``.  RGB is constrained to
    ``[0,1]`` and density is strictly positive.  These are physical values for
    the dense native executor, not raw sigmoid/softplus optimizer parameters.
    A trainer that owns raw parameters must invert its declared
    parameterization explicitly and record that transform in its checkpoint.
    """

    site_rgba_f32: torch.Tensor = field(repr=False)
    initializer_generation_digest: str
    material_seed_generation_digest: str
    sites_content_digest: str
    material_content_digest: str
    generation_digest: str
    provenance: str = P0_MATERIAL_PROVENANCE
    temporal_basis: str = "P0"
    layout: str = P0_MATERIAL_LAYOUT
    frame_dependent_parameter_bytes: int = 0
    _seal: object = field(default=None, repr=False)

    @property
    def site_count(self) -> int:
        return int(self.site_rgba_f32.shape[0])

    @property
    def parameter_bytes(self) -> int:
        return int(self.site_rgba_f32.numel() * self.site_rgba_f32.element_size())

    def assert_current(self, sites: AffineKineticPowerSites | None = None) -> None:
        if (
            self._seal is not _MATERIAL_SEAL
            or self.provenance != P0_MATERIAL_PROVENANCE
            or self.temporal_basis != "P0"
            or self.layout != P0_MATERIAL_LAYOUT
            or self.frame_dependent_parameter_bytes != 0
        ):
            raise ValueError("paper kinetic P0 material seal/semantics changed")
        _require_sha256(
            self.initializer_generation_digest,
            name="initializer_generation_digest",
        )
        _require_sha256(
            self.material_seed_generation_digest,
            name="material_seed_generation_digest",
        )
        _require_sha256(self.sites_content_digest, name="sites_content_digest")
        _require_sha256(self.material_content_digest, name="material_content_digest")
        _validate_physical_p0_material(self.site_rgba_f32)
        if _tensor_content_digest(self.site_rgba_f32) != self.material_content_digest:
            raise ValueError("paper kinetic P0 material tensor changed")
        if sites is not None:
            if not isinstance(sites, AffineKineticPowerSites):
                raise TypeError("sites must be AffineKineticPowerSites")
            if _sites_content_digest(sites) != self.sites_content_digest:
                raise ValueError("paper kinetic P0 material belongs to different sites")
        expected = _digest_parts(
            P0_MATERIAL_PROVENANCE,
            self.initializer_generation_digest,
            self.material_seed_generation_digest,
            self.sites_content_digest,
            self.material_content_digest,
            self.site_count,
            self.temporal_basis,
            self.layout,
            0,
        )
        if self.generation_digest != expected:
            raise ValueError("paper kinetic P0 material generation changed")


def prepare_paper_kinetic_p0_material_initialization(
    site_rgba_f32: torch.Tensor,
    sites: AffineKineticPowerSites,
    *,
    initializer_generation_digest: str,
    source_material_seed_digest: str,
) -> PaperKineticP0MaterialInitialization:
    """Seal caller-supplied physical P0 material for an exact kinetic world.

    Production point-cloud initialization has its own richer asset provenance,
    but deterministic fixtures and externally constructed worlds also need a
    public route into the same fixed-site optimizer contract.  This function
    owns a CPU clone, binds it to the complete site contents, and derives the
    material-seed generation rather than accepting a caller-invented live-state
    identifier.
    """

    if not isinstance(sites, AffineKineticPowerSites):
        raise TypeError("sites must be AffineKineticPowerSites")
    _require_sha256(
        initializer_generation_digest,
        name="initializer_generation_digest",
    )
    _require_sha256(
        source_material_seed_digest,
        name="source_material_seed_digest",
    )
    _validate_physical_p0_material(site_rgba_f32)
    if int(site_rgba_f32.shape[0]) != sites.site_count:
        raise ValueError("physical P0 material row count must match the kinetic world")

    owned = site_rgba_f32.detach().clone().contiguous()
    sites_content_digest = _sites_content_digest(sites)
    material_content_digest = _tensor_content_digest(owned)
    material_seed_generation_digest = _digest_parts(
        P0_MATERIAL_PROVENANCE,
        "external-physical-material-seed-v1",
        initializer_generation_digest,
        source_material_seed_digest,
        sites_content_digest,
        material_content_digest,
        sites.site_count,
        "P0",
        P0_MATERIAL_LAYOUT,
        0,
    )
    result = PaperKineticP0MaterialInitialization(
        site_rgba_f32=owned,
        initializer_generation_digest=initializer_generation_digest,
        material_seed_generation_digest=material_seed_generation_digest,
        sites_content_digest=sites_content_digest,
        material_content_digest=material_content_digest,
        generation_digest=_digest_parts(
            P0_MATERIAL_PROVENANCE,
            initializer_generation_digest,
            material_seed_generation_digest,
            sites_content_digest,
            material_content_digest,
            sites.site_count,
            "P0",
            P0_MATERIAL_LAYOUT,
            0,
        ),
        _seal=_MATERIAL_SEAL,
    )
    result.assert_current(sites)
    return result


@dataclass(frozen=True)
class PaperKineticPointCloudWorldInitializer:
    """Cold-sealed point-cloud implementation of ``PaperKineticWorldInitializer``.

    Construct instances with
    :func:`prepare_paper_kinetic_point_cloud_world_initializer`.  The private
    templates are immutable initialization sources, not live model parameters.
    Each public initialization call returns fresh tensors owned by its caller.
    """

    source_path: Path
    source_asset_digest: str
    source_geometry_digest: str
    source_material_seed_digest: str
    source_coordinate_frame: str
    point_transform: tuple[tuple[float, float, float, float], ...] | None
    maximum_source_asset_bytes: int
    maximum_source_point_count: int
    sample_mode: str
    sample_seed: int
    coordinate_quantization_step: float
    weight_coefficients: tuple[float, ...]
    weight_quantization_step: float
    initial_density: float
    source_finite_point_count: int
    generation_digest: str
    p0_material_seed_generation_digest: str
    _template_sites: AffineKineticPowerSites = field(repr=False)
    _template_site_rgba_f32: torch.Tensor = field(repr=False)
    _template_sites_content_digest: str = field(repr=False)
    _template_material_content_digest: str = field(repr=False)
    provenance: str = INITIALIZER_PROVENANCE
    frame_dependent_parameter_bytes: int = 0
    target_or_video_decode_allowed: bool = False
    camera_dependent_initialization: bool = False
    _seal: object = field(default=None, repr=False)

    @property
    def site_count(self) -> int:
        return self._template_sites.site_count

    def assert_current(self, *, check_source_asset: bool = True) -> None:
        if (
            self._seal is not _INITIALIZER_SEAL
            or self.provenance != INITIALIZER_PROVENANCE
            or self.frame_dependent_parameter_bytes != 0
            or self.target_or_video_decode_allowed
            or self.camera_dependent_initialization
        ):
            raise ValueError("paper kinetic point-cloud initializer seal/semantics changed")
        _require_sha256(self.source_asset_digest, name="source_asset_digest")
        _require_sha256(self.source_geometry_digest, name="source_geometry_digest")
        _require_sha256(
            self.source_material_seed_digest,
            name="source_material_seed_digest",
        )
        _require_sha256(self.generation_digest, name="generation_digest")
        _require_sha256(
            self.p0_material_seed_generation_digest,
            name="p0_material_seed_generation_digest",
        )
        if check_source_asset:
            if int(self.source_path.stat().st_size) > self.maximum_source_asset_bytes:
                raise MemoryError(
                    "paper kinetic initializer source asset exceeds its sealed byte budget"
                )
            if _sha256_file(self.source_path) != self.source_asset_digest:
                raise ValueError("paper kinetic initializer source asset changed")
        _validate_initializer_options(
            source_coordinate_frame=self.source_coordinate_frame,
            point_transform=self.point_transform,
            maximum_source_asset_bytes=self.maximum_source_asset_bytes,
            maximum_source_point_count=self.maximum_source_point_count,
            site_count=self.site_count,
            sample_mode=self.sample_mode,
            sample_seed=self.sample_seed,
            coordinate_quantization_step=self.coordinate_quantization_step,
            weight_coefficients=self.weight_coefficients,
            weight_quantization_step=self.weight_quantization_step,
            initial_density=self.initial_density,
        )
        if _sites_content_digest(self._template_sites) != self._template_sites_content_digest:
            raise ValueError("paper kinetic initializer geometry template changed")
        _validate_physical_p0_material(self._template_site_rgba_f32)
        if (
            _tensor_content_digest(self._template_site_rgba_f32)
            != self._template_material_content_digest
        ):
            raise ValueError("paper kinetic initializer material template changed")
        expected = _initializer_digest(
            source_geometry_digest=self.source_geometry_digest,
            source_coordinate_frame=self.source_coordinate_frame,
            point_transform=self.point_transform,
            maximum_source_asset_bytes=self.maximum_source_asset_bytes,
            maximum_source_point_count=self.maximum_source_point_count,
            site_count=self.site_count,
            sample_mode=self.sample_mode,
            sample_seed=self.sample_seed,
            coordinate_quantization_step=self.coordinate_quantization_step,
            weight_coefficients=self.weight_coefficients,
            weight_quantization_step=self.weight_quantization_step,
            source_finite_point_count=self.source_finite_point_count,
            sites_content_digest=self._template_sites_content_digest,
        )
        if self.generation_digest != expected:
            raise ValueError("paper kinetic point-cloud initializer generation changed")
        expected_material = _material_seed_digest(
            initializer_generation_digest=self.generation_digest,
            source_material_seed_digest=self.source_material_seed_digest,
            sites_content_digest=self._template_sites_content_digest,
            material_content_digest=self._template_material_content_digest,
            initial_density=self.initial_density,
        )
        if self.p0_material_seed_generation_digest != expected_material:
            raise ValueError("paper kinetic P0 material seed generation changed")

    def initialize_world(
        self,
        request: PaperKineticWorldInitializationRequest,
    ) -> AffineKineticPowerSites:
        """Return fresh CPU float64 site tensors; request ``F`` is ignored."""

        if not isinstance(request, PaperKineticWorldInitializationRequest):
            raise TypeError(
                "paper kinetic point-cloud initializer requires "
                "PaperKineticWorldInitializationRequest"
            )
        request.assert_self_consistent()
        if request.initializer_generation_digest != self.generation_digest:
            raise ValueError("paper kinetic world request names a different initializer")
        self.assert_current()
        return _clone_sites(self._template_sites)

    def initialize_p0_material(
        self,
        sites: AffineKineticPowerSites,
    ) -> PaperKineticP0MaterialInitialization:
        """Return fresh physical P0 material bound to ``sites`` by content."""

        if not isinstance(sites, AffineKineticPowerSites):
            raise TypeError("sites must be AffineKineticPowerSites")
        self.assert_current()
        sites_digest = _sites_content_digest(sites)
        if sites_digest != self._template_sites_content_digest:
            raise ValueError("cannot seed P0 material for non-initializer site contents")
        rgba = self._template_site_rgba_f32.clone().contiguous()
        material_digest = _tensor_content_digest(rgba)
        result = PaperKineticP0MaterialInitialization(
            site_rgba_f32=rgba,
            initializer_generation_digest=self.generation_digest,
            material_seed_generation_digest=(
                self.p0_material_seed_generation_digest
            ),
            sites_content_digest=sites_digest,
            material_content_digest=material_digest,
            generation_digest=_digest_parts(
                P0_MATERIAL_PROVENANCE,
                self.generation_digest,
                self.p0_material_seed_generation_digest,
                sites_digest,
                material_digest,
                int(rgba.shape[0]),
                "P0",
                P0_MATERIAL_LAYOUT,
                0,
            ),
            _seal=_MATERIAL_SEAL,
        )
        result.assert_current(sites)
        return result

    def storage_report(
        self,
        *,
        requested_frame_count: int,
    ) -> PaperKineticInitializationStorageReport:
        """Report the same parameter bytes for every positive requested ``F``."""

        _require_positive_int(requested_frame_count, name="requested_frame_count")
        self.assert_current()
        geometry_bytes = self._template_sites.parameter_bytes
        material_bytes = int(
            self._template_site_rgba_f32.numel()
            * self._template_site_rgba_f32.element_size()
        )
        return PaperKineticInitializationStorageReport(
            requested_frame_count=requested_frame_count,
            site_count=self.site_count,
            geometry_parameter_bytes=geometry_bytes,
            p0_material_parameter_bytes=material_bytes,
            total_parameter_bytes=geometry_bytes + material_bytes,
        )

    def accounting(self, *, requested_frame_count: int) -> dict[str, Any]:
        report = self.storage_report(requested_frame_count=requested_frame_count)
        return {
            "provenance": self.provenance,
            "generation_digest": self.generation_digest,
            "source_asset_digest": self.source_asset_digest,
            "source_geometry_digest": self.source_geometry_digest,
            "source_material_seed_digest": self.source_material_seed_digest,
            "p0_material_seed_generation_digest": (
                self.p0_material_seed_generation_digest
            ),
            "structural_generation_excludes_material": True,
            "source_coordinate_frame": self.source_coordinate_frame,
            "point_loader_semantics": POINT_LOADER_SEMANTICS,
            "maximum_source_asset_bytes": self.maximum_source_asset_bytes,
            "maximum_source_point_count": self.maximum_source_point_count,
            "selection_algorithm": SELECTION_ALGORITHM,
            "selection_scratch_entry_bound": report.site_count,
            "selection_scratch_independent_of_source_point_count": True,
            "sample_mode": self.sample_mode,
            "coordinate_quantization_step": self.coordinate_quantization_step,
            "site_count": report.site_count,
            "weight_polynomial_degree": len(self.weight_coefficients) - 1,
            "weight_quantization_step": self.weight_quantization_step,
            "maximum_quantized_integer_magnitude": (
                MAX_QUANTIZED_INTEGER_MAGNITUDE
            ),
            "minimum_exact_grid_exponent": MIN_EXACT_GRID_EXPONENT,
            "maximum_exact_grid_exponent": MAX_EXACT_GRID_EXPONENT,
            "exact_compiler_inputs_on_bounded_dyadic_grid": True,
            "material_temporal_basis": "P0",
            "material_layout": P0_MATERIAL_LAYOUT,
            "requested_frame_count": report.requested_frame_count,
            "geometry_parameter_bytes": report.geometry_parameter_bytes,
            "p0_material_parameter_bytes": report.p0_material_parameter_bytes,
            "total_parameter_bytes": report.total_parameter_bytes,
            "stored_frame_state_bytes": 0,
            "frame_dependent_parameter_bytes": 0,
            "target_tensor_bytes": 0,
            "video_tensor_bytes": 0,
            "camera_tensor_bytes": 0,
            "target_or_video_decode_used": False,
            "camera_values_used_to_initialize_parameters": False,
            "request_frame_count_used_to_initialize_parameters": False,
            "positions_dtype_device": "torch.float64/cpu",
            "velocities_initialized_to_zero": True,
            "physical_material_dtype_device": "torch.float32/cpu",
            "physical_material_requires_grad": False,
            "raw_optimizer_parameterization_owned_here": False,
        }


def prepare_paper_kinetic_point_cloud_world_initializer(
    config: Mapping[str, Any],
) -> PaperKineticPointCloudWorldInitializer:
    """Load and seal one deterministic frame-independent initial world.

    Required config keys are exact; unknown/missing keys fail closed.  The
    supported schema intentionally has no frame count, camera, target, or video
    field.
    """

    if not isinstance(config, Mapping):
        raise TypeError("paper kinetic point-cloud initializer config must be a mapping")
    keys = frozenset(config)
    if keys != _CONFIG_KEYS:
        missing = sorted(_CONFIG_KEYS - keys)
        unknown = sorted(keys - _CONFIG_KEYS)
        raise ValueError(
            "paper kinetic point-cloud initializer config keys differ; "
            f"missing={missing}, unknown={unknown}"
        )
    maximum_source_asset_bytes = _positive_int(
        config["maximum_source_asset_bytes"],
        name="maximum_source_asset_bytes",
    )
    maximum_source_point_count = _positive_int(
        config["maximum_source_point_count"],
        name="maximum_source_point_count",
    )
    source_path = resolve_point_cloud_path(Path(config["source_path"]).expanduser())
    source_asset_bytes = int(source_path.stat().st_size)
    if source_asset_bytes > maximum_source_asset_bytes:
        raise MemoryError(
            "paper kinetic initializer source asset exceeds its pre-load byte budget"
        )
    source_coordinate_frame = str(config["source_coordinate_frame"])
    point_transform = _canonical_point_transform(config["point_transform"])
    site_count = _positive_int(config["site_count"], name="site_count")
    sample_mode = str(config["sample_mode"])
    sample_seed = _nonnegative_int(config["sample_seed"], name="sample_seed")
    coordinate_quantization_step = _power_of_two_grid_step(
        config["coordinate_quantization_step"],
        name="coordinate_quantization_step",
    )
    requested_weight_coefficients = _finite_tuple(
        config["weight_coefficients"],
        name="weight_coefficients",
        minimum_length=1,
        maximum_length=3,
    )
    weight_quantization_step = _power_of_two_grid_step(
        config["weight_quantization_step"],
        name="weight_quantization_step",
    )
    weight_coefficients = _quantized_scalar_tuple(
        requested_weight_coefficients,
        step=weight_quantization_step,
        name="weight_coefficients",
    )
    initial_density = _finite_float(config["initial_density"], name="initial_density")
    _validate_initializer_options(
        source_coordinate_frame=source_coordinate_frame,
        point_transform=point_transform,
        maximum_source_asset_bytes=maximum_source_asset_bytes,
        maximum_source_point_count=maximum_source_point_count,
        site_count=site_count,
        sample_mode=sample_mode,
        sample_seed=sample_seed,
        coordinate_quantization_step=coordinate_quantization_step,
        weight_coefficients=weight_coefficients,
        weight_quantization_step=weight_quantization_step,
        initial_density=initial_density,
    )

    source_asset_digest = _sha256_file(source_path)
    declared_source_point_count = _declared_point_count(source_path)
    if declared_source_point_count > maximum_source_point_count:
        raise MemoryError(
            "paper kinetic initializer source point count exceeds its pre-load budget"
        )
    source_points_f32, source_colors_f32 = load_point_cloud_xyz_rgb(source_path)
    source_finite_point_count = int(source_points_f32.shape[0])
    if source_finite_point_count > maximum_source_point_count:
        raise MemoryError(
            "paper kinetic initializer loaded point count exceeds its declared budget"
        )
    source_geometry_digest = _tensor_content_digest(source_points_f32)
    source_material_seed_digest = _tensor_content_digest(source_colors_f32)
    if source_finite_point_count < site_count:
        raise ValueError(
            "paper kinetic initializer refuses duplicate padding: "
            f"asset has {source_finite_point_count} finite points for {site_count} sites"
        )
    indices = _selected_source_indices(
        source_count=source_finite_point_count,
        site_count=site_count,
        sample_mode=sample_mode,
        sample_seed=sample_seed,
        source_geometry_digest=source_geometry_digest,
    )
    positions0 = source_points_f32.index_select(0, indices).to(
        device="cpu",
        dtype=torch.float64,
    )
    positions0 = _apply_affine_point_transform(positions0, point_transform)
    positions0 = _quantize_tensor_to_grid(
        positions0,
        step=coordinate_quantization_step,
        name="positions0",
    )
    positions0 = positions0.contiguous()
    if int(torch.unique(positions0, dim=0).shape[0]) != site_count:
        raise ValueError("paper kinetic initializer selected coincident site positions")
    colors = source_colors_f32.index_select(0, indices).to(
        device="cpu",
        dtype=torch.float32,
    ).contiguous()
    sites = AffineKineticPowerSites(
        positions0=positions0,
        velocities=torch.zeros_like(positions0),
        weight_coefficients=torch.tensor(
            weight_coefficients,
            dtype=torch.float64,
            device="cpu",
        ).view(1, -1).repeat(site_count, 1).contiguous(),
    )
    density = torch.full(
        (site_count, 1),
        initial_density,
        dtype=torch.float32,
        device="cpu",
    )
    rgba = torch.cat((colors, density), dim=1).contiguous()
    _validate_physical_p0_material(rgba)
    sites_digest = _sites_content_digest(sites)
    material_digest = _tensor_content_digest(rgba)
    generation_digest = _initializer_digest(
        source_geometry_digest=source_geometry_digest,
        source_coordinate_frame=source_coordinate_frame,
        point_transform=point_transform,
        maximum_source_asset_bytes=maximum_source_asset_bytes,
        maximum_source_point_count=maximum_source_point_count,
        site_count=site_count,
        sample_mode=sample_mode,
        sample_seed=sample_seed,
        coordinate_quantization_step=coordinate_quantization_step,
        weight_coefficients=weight_coefficients,
        weight_quantization_step=weight_quantization_step,
        source_finite_point_count=source_finite_point_count,
        sites_content_digest=sites_digest,
    )
    material_seed_generation_digest = _material_seed_digest(
        initializer_generation_digest=generation_digest,
        source_material_seed_digest=source_material_seed_digest,
        sites_content_digest=sites_digest,
        material_content_digest=material_digest,
        initial_density=initial_density,
    )
    result = PaperKineticPointCloudWorldInitializer(
        source_path=source_path,
        source_asset_digest=source_asset_digest,
        source_geometry_digest=source_geometry_digest,
        source_material_seed_digest=source_material_seed_digest,
        source_coordinate_frame=source_coordinate_frame,
        point_transform=point_transform,
        maximum_source_asset_bytes=maximum_source_asset_bytes,
        maximum_source_point_count=maximum_source_point_count,
        sample_mode=sample_mode,
        sample_seed=sample_seed,
        coordinate_quantization_step=coordinate_quantization_step,
        weight_coefficients=weight_coefficients,
        weight_quantization_step=weight_quantization_step,
        initial_density=initial_density,
        source_finite_point_count=source_finite_point_count,
        generation_digest=generation_digest,
        p0_material_seed_generation_digest=material_seed_generation_digest,
        _template_sites=sites,
        _template_site_rgba_f32=rgba,
        _template_sites_content_digest=sites_digest,
        _template_material_content_digest=material_digest,
        _seal=_INITIALIZER_SEAL,
    )
    result.assert_current()
    return result


def _validate_initializer_options(
    *,
    source_coordinate_frame: str,
    point_transform: tuple[tuple[float, float, float, float], ...] | None,
    maximum_source_asset_bytes: int,
    maximum_source_point_count: int,
    site_count: int,
    sample_mode: str,
    sample_seed: int,
    coordinate_quantization_step: float,
    weight_coefficients: tuple[float, ...],
    weight_quantization_step: float,
    initial_density: float,
) -> None:
    _require_positive_int(site_count, name="site_count")
    _require_positive_int(
        maximum_source_asset_bytes,
        name="maximum_source_asset_bytes",
    )
    _require_positive_int(
        maximum_source_point_count,
        name="maximum_source_point_count",
    )
    if site_count > maximum_source_point_count:
        raise ValueError("site_count cannot exceed maximum_source_point_count")
    _require_nonnegative_int(sample_seed, name="sample_seed")
    if source_coordinate_frame not in {"model", "external_affine"}:
        raise ValueError(
            "source_coordinate_frame must be 'model' or 'external_affine'"
        )
    if source_coordinate_frame == "model" and point_transform is not None:
        raise ValueError("model-frame point clouds must not declare a point transform")
    if source_coordinate_frame == "external_affine" and point_transform is None:
        raise ValueError("external_affine point clouds require an explicit point transform")
    if sample_mode not in {"first", "sha256_rank"}:
        raise ValueError("sample_mode must be 'first' or 'sha256_rank'")
    if sample_mode == "first" and sample_seed != 0:
        raise ValueError("sample_seed must be zero when sample_mode='first'")
    _power_of_two_grid_step(
        coordinate_quantization_step,
        name="coordinate_quantization_step",
    )
    if not 1 <= len(weight_coefficients) <= 3 or not all(
        math.isfinite(value) for value in weight_coefficients
    ):
        raise ValueError("weight_coefficients must contain 1..3 finite scalars")
    _power_of_two_grid_step(
        weight_quantization_step,
        name="weight_quantization_step",
    )
    if _quantized_scalar_tuple(
        weight_coefficients,
        step=weight_quantization_step,
        name="weight_coefficients",
    ) != weight_coefficients:
        raise ValueError("weight_coefficients must already lie on their declared grid")
    if not math.isfinite(initial_density) or initial_density <= 0.0:
        raise ValueError("initial_density must be finite and strictly positive")


def _canonical_point_transform(
    value: Any,
) -> tuple[tuple[float, float, float, float], ...] | None:
    if value is None:
        return None
    tensor = torch.as_tensor(value, dtype=torch.float64, device="cpu")
    if tuple(tensor.shape) != (4, 4) or not bool(torch.isfinite(tensor).all().item()):
        raise ValueError("point_transform must be a finite affine 4x4 matrix")
    expected_bottom = torch.tensor((0.0, 0.0, 0.0, 1.0), dtype=torch.float64)
    if not bool(torch.equal(tensor[3], expected_bottom)):
        raise ValueError("point_transform bottom row must be exactly [0,0,0,1]")
    linear = tensor[:3, :3]
    determinant = float(torch.linalg.det(linear).item())
    if not math.isfinite(determinant) or abs(determinant) <= 1.0e-12:
        raise ValueError("point_transform affine linear part must be nonsingular")
    return tuple(tuple(float(value) for value in row) for row in tensor.tolist())


def _apply_affine_point_transform(
    points: torch.Tensor,
    transform: tuple[tuple[float, float, float, float], ...] | None,
) -> torch.Tensor:
    if transform is None:
        return points
    matrix = torch.tensor(transform, dtype=torch.float64, device="cpu")
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _selected_source_indices(
    *,
    source_count: int,
    site_count: int,
    sample_mode: str,
    sample_seed: int,
    source_geometry_digest: str,
) -> torch.Tensor:
    if sample_mode == "first":
        selected = range(site_count)
    elif sample_mode == "sha256_rank":
        # Keep only the smallest ``site_count`` keys.  Materializing one
        # Python tuple per source point made cold initialization peak at O(N)
        # non-tensor objects even though the retained world has only S sites.
        # ``inverted_rank`` turns heapq's min-heap root into the largest
        # currently retained canonical ``(rank,index)`` key.
        ranked: list[tuple[int, int, int]] = []
        for index in range(source_count):
            rank = int.from_bytes(
                hashlib.sha256(
                    (
                        f"{SELECTION_ALGORITHM}:{source_geometry_digest}:"
                        f"{sample_seed}:{index}"
                    ).encode("utf-8")
                ).digest(),
                byteorder="big",
                signed=False,
            )
            inverted_rank = (-rank, -index, index)
            if len(ranked) < site_count:
                heapq.heappush(ranked, inverted_rank)
            elif inverted_rank > ranked[0]:
                heapq.heapreplace(ranked, inverted_rank)
        selected = (
            index
            for _inverted_rank, _inverted_index, index in sorted(
                ranked,
                key=lambda entry: (-entry[0], -entry[1]),
            )
        )
    else:  # Defensive: public validation should reject before this helper.
        raise ValueError("unsupported point-cloud selection mode")
    return torch.tensor(tuple(selected), dtype=torch.int64, device="cpu")


def _clone_sites(sites: AffineKineticPowerSites) -> AffineKineticPowerSites:
    return AffineKineticPowerSites(
        positions0=sites.positions0.clone().contiguous(),
        velocities=sites.velocities.clone().contiguous(),
        weight_coefficients=sites.weight_coefficients.clone().contiguous(),
    )


def _validate_physical_p0_material(site_rgba_f32: torch.Tensor) -> None:
    if not isinstance(site_rgba_f32, torch.Tensor):
        raise TypeError("site_rgba_f32 must be a tensor")
    if (
        site_rgba_f32.device.type != "cpu"
        or site_rgba_f32.dtype != torch.float32
        or site_rgba_f32.ndim != 2
        or int(site_rgba_f32.shape[0]) < 1
        or int(site_rgba_f32.shape[1]) != 4
        or not site_rgba_f32.is_contiguous()
        or site_rgba_f32.requires_grad
    ):
        raise ValueError(
            "physical P0 material must be contiguous non-autograd CPU float32 [S,4]"
        )
    if not bool(torch.isfinite(site_rgba_f32).all().item()):
        raise ValueError("physical P0 material must be finite")
    if bool(torch.any((site_rgba_f32[:, :3] < 0.0) | (site_rgba_f32[:, :3] > 1.0)).item()):
        raise ValueError("physical P0 RGB must lie in [0,1]")
    if bool(torch.any(site_rgba_f32[:, 3] <= 0.0).item()):
        raise ValueError("physical P0 density must be strictly positive")


def _power_of_two_grid_step(value: Any, *, name: str) -> float:
    step = _finite_float(value, name=name)
    if step <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    mantissa, exponent = math.frexp(step)
    if mantissa != 0.5:
        raise ValueError(f"{name} must be an exact binary power of two")
    grid_exponent = exponent - 1
    if not MIN_EXACT_GRID_EXPONENT <= grid_exponent <= MAX_EXACT_GRID_EXPONENT:
        raise ValueError(
            f"{name} binary exponent must be in "
            f"[{MIN_EXACT_GRID_EXPONENT},{MAX_EXACT_GRID_EXPONENT}]"
        )
    return step


def _quantized_scalar_tuple(
    values: tuple[float, ...],
    *,
    step: float,
    name: str,
) -> tuple[float, ...]:
    grid_step = _power_of_two_grid_step(step, name=f"{name}_quantization_step")
    result = []
    for value in values:
        units = round(float(value) / grid_step)
        if abs(units) > MAX_QUANTIZED_INTEGER_MAGNITUDE:
            raise ValueError(
                f"{name} exceeds the bounded exact-compiler grid magnitude"
            )
        result.append(float(units) * grid_step)
    return tuple(result)


def _quantize_tensor_to_grid(
    value: torch.Tensor,
    *,
    step: float,
    name: str,
) -> torch.Tensor:
    grid_step = _power_of_two_grid_step(step, name=f"{name}_quantization_step")
    units = torch.round(value / grid_step)
    if not bool(torch.isfinite(units).all().item()):
        raise ValueError(f"{name} produced non-finite quantized grid units")
    if bool(torch.any(units.abs() > MAX_QUANTIZED_INTEGER_MAGNITUDE).item()):
        raise ValueError(f"{name} exceeds the bounded exact-compiler grid magnitude")
    return units.mul(grid_step)


def _initializer_digest(
    *,
    source_geometry_digest: str,
    source_coordinate_frame: str,
    point_transform: tuple[tuple[float, float, float, float], ...] | None,
    maximum_source_asset_bytes: int,
    maximum_source_point_count: int,
    site_count: int,
    sample_mode: str,
    sample_seed: int,
    coordinate_quantization_step: float,
    weight_coefficients: tuple[float, ...],
    weight_quantization_step: float,
    source_finite_point_count: int,
    sites_content_digest: str,
) -> str:
    payload = {
        "provenance": INITIALIZER_PROVENANCE,
        "source_geometry_digest": source_geometry_digest,
        "source_coordinate_frame": source_coordinate_frame,
        "point_transform": point_transform,
        "maximum_source_asset_bytes": maximum_source_asset_bytes,
        "maximum_source_point_count": maximum_source_point_count,
        "site_count": site_count,
        "sample_mode": sample_mode,
        "sample_seed": sample_seed,
        "selection_algorithm": SELECTION_ALGORITHM,
        "point_loader_semantics": POINT_LOADER_SEMANTICS,
        "coordinate_quantization_step": coordinate_quantization_step,
        "weight_coefficients": weight_coefficients,
        "weight_quantization_step": weight_quantization_step,
        "maximum_quantized_integer_magnitude": (
            MAX_QUANTIZED_INTEGER_MAGNITUDE
        ),
        "minimum_exact_grid_exponent": MIN_EXACT_GRID_EXPONENT,
        "maximum_exact_grid_exponent": MAX_EXACT_GRID_EXPONENT,
        "source_finite_point_count": source_finite_point_count,
        "sites_content_digest": sites_content_digest,
        "frame_dependent_parameter_bytes": 0,
        "target_or_video_decode_used": False,
        "camera_values_used": False,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _material_seed_digest(
    *,
    initializer_generation_digest: str,
    source_material_seed_digest: str,
    sites_content_digest: str,
    material_content_digest: str,
    initial_density: float,
) -> str:
    return _digest_parts(
        P0_MATERIAL_PROVENANCE,
        "initializer-material-seed",
        initializer_generation_digest,
        source_material_seed_digest,
        sites_content_digest,
        material_content_digest,
        initial_density,
        "P0",
        P0_MATERIAL_LAYOUT,
        0,
    )


def _sites_content_digest(sites: AffineKineticPowerSites) -> str:
    return _digest_parts(
        "paper-kinetic-initializer-sites-v1",
        sites.site_count,
        _tensor_content_digest(sites.positions0),
        _tensor_content_digest(sites.velocities),
        _tensor_content_digest(sites.weight_coefficients),
    )


def _tensor_content_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(repr(tuple(value.shape)).encode("utf-8"))
    digest.update(memoryview(value.numpy()).cast("B"))
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _declared_point_count(path: Path) -> int:
    """Read only structural metadata before the allocating point loader runs."""

    if path.suffix.lower() == ".ply":
        vertex_count: int | None = None
        with path.open("rb") as handle:
            if handle.readline().decode("ascii", errors="strict").strip() != "ply":
                raise ValueError(f"{path} is not a PLY file")
            while True:
                raw = handle.readline()
                if not raw:
                    raise ValueError(f"{path} ended before PLY end_header")
                parts = raw.decode("ascii", errors="strict").strip().split()
                if parts[:2] == ["element", "vertex"]:
                    if len(parts) != 3:
                        raise ValueError(f"{path} has an invalid PLY vertex declaration")
                    vertex_count = int(parts[2])
                if parts == ["end_header"]:
                    break
        if vertex_count is None or vertex_count < 1:
            raise ValueError(f"{path} must declare at least one PLY vertex")
        return vertex_count
    if path.name == "points3D.bin":
        with path.open("rb") as handle:
            payload = handle.read(8)
        if len(payload) != 8:
            raise ValueError(f"{path} is too short for a COLMAP points3D.bin file")
        return int(struct.unpack("<Q", payload)[0])
    if path.name == "points3D.txt":
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line and not line.startswith("#") and len(line.split()) >= 8:
                    count += 1
        if count < 1:
            raise ValueError(f"{path} contains no COLMAP points3D rows")
        return count
    raise ValueError(f"unsupported point-cloud format for {path}")


def _digest_parts(*parts: object) -> str:
    digest = hashlib.sha256()
    for part in parts:
        encoded = repr(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _finite_tuple(
    value: Any,
    *,
    name: str,
    minimum_length: int,
    maximum_length: int,
) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{name} must be a scalar sequence")
    result = tuple(_finite_float(item, name=name) for item in value)
    if not minimum_length <= len(result) <= maximum_length:
        raise ValueError(
            f"{name} must contain {minimum_length}..{maximum_length} scalars"
        )
    return result


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive_int(value: Any, *, name: str) -> int:
    _require_positive_int(value, name=name)
    return int(value)


def _nonnegative_int(value: Any, *, name: str) -> int:
    _require_nonnegative_int(value, name=name)
    return int(value)


def _require_positive_int(value: Any, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require_nonnegative_int(value: Any, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _require_sha256(value: str, *, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    try:
        parsed = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest") from error
    if len(parsed) != 32 or value != value.lower():
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


__all__ = [
    "INITIALIZER_PROVENANCE",
    "MAX_EXACT_GRID_EXPONENT",
    "MAX_QUANTIZED_INTEGER_MAGNITUDE",
    "MIN_EXACT_GRID_EXPONENT",
    "P0_MATERIAL_LAYOUT",
    "P0_MATERIAL_PROVENANCE",
    "POINT_LOADER_SEMANTICS",
    "SELECTION_ALGORITHM",
    "PaperKineticInitializationStorageReport",
    "PaperKineticP0MaterialInitialization",
    "PaperKineticPointCloudWorldInitializer",
    "prepare_paper_kinetic_p0_material_initialization",
    "prepare_paper_kinetic_point_cloud_world_initializer",
]
