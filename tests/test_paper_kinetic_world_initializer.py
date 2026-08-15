from __future__ import annotations

import gc
import hashlib
import weakref
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from camera import CameraSpec
from paper_kinetic_lazy_program_bundles import (
    PROVIDER_PROVENANCE,
    PaperKineticWorldInitializationRequest,
    PaperKineticWorldInitializer,
    _digest_parts,
    prepare_paper_kinetic_lazy_program_bundle_provider,
)
from paper_kinetic_world_initializer import (
    MAX_QUANTIZED_INTEGER_MAGNITUDE,
    P0_MATERIAL_LAYOUT,
    PaperKineticPointCloudWorldInitializer,
    prepare_paper_kinetic_p0_material_initialization,
    prepare_paper_kinetic_point_cloud_world_initializer,
)
from powerfoam_training_data import PowerFoamRayProvider, PowerFoamTargetProvider


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_ascii_ply(path: Path, *, duplicate: bool = False) -> None:
    fourth = "0.10 0.10 0.10 12 34 56" if duplicate else "3.10 1.90 -0.10 12 34 56"
    path.write_text(
        "\n".join(
            (
                "ply",
                "format ascii 1.0",
                "element vertex 4",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "0.10 0.10 0.10 255 0 64",
                "0.90 -0.60 0.40 0 128 255",
                "2.10 1.10 -1.10 10 20 30",
                fourth,
                "",
            )
        ),
        encoding="utf-8",
    )


def _config(path: Path, **updates):
    config = {
        "source_path": path,
        "source_coordinate_frame": "model",
        "point_transform": None,
        "maximum_source_asset_bytes": 1_000_000,
        "maximum_source_point_count": 1_000,
        "site_count": 3,
        "sample_mode": "first",
        "sample_seed": 0,
        "coordinate_quantization_step": 0.25,
        "weight_coefficients": (0.1, -0.2, 0.3),
        "weight_quantization_step": 0.0625,
        "initial_density": 64.0,
    }
    config.update(updates)
    return config


def _request(
    initializer: PaperKineticPointCloudWorldInitializer,
    *,
    frame_count: int,
    initializer_generation_digest: str | None = None,
) -> PaperKineticWorldInitializationRequest:
    initializer_digest = (
        initializer.generation_digest
        if initializer_generation_digest is None
        else initializer_generation_digest
    )
    provisional = PaperKineticWorldInitializationRequest(
        dataset_generation_digest=_sha256(f"dataset-F{frame_count}"),
        camera_grid_digest=_sha256(f"camera-F{frame_count}"),
        view_count=2,
        frame_count=frame_count,
        height=32,
        width=48,
        initializer_generation_digest=initializer_digest,
        generation_digest="",
    )
    return replace(
        provisional,
        generation_digest=_digest_parts(
            PROVIDER_PROVENANCE,
            "world-init-request",
            provisional.dataset_generation_digest,
            provisional.camera_grid_digest,
            provisional.view_count,
            provisional.frame_count,
            provisional.height,
            provisional.width,
            provisional.initializer_generation_digest,
        ),
    )


def test_point_cloud_initializer_is_protocol_compatible_and_frame_independent(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "world.ply"
    _write_ascii_ply(asset)
    first = prepare_paper_kinetic_point_cloud_world_initializer(_config(asset))
    second = prepare_paper_kinetic_point_cloud_world_initializer(_config(asset))

    assert isinstance(first, PaperKineticWorldInitializer)
    assert first.generation_digest == second.generation_digest
    world_f1 = first.initialize_world(_request(first, frame_count=1))
    world_f300 = first.initialize_world(_request(first, frame_count=300))
    for left, right in (
        (world_f1.positions0, world_f300.positions0),
        (world_f1.velocities, world_f300.velocities),
        (world_f1.weight_coefficients, world_f300.weight_coefficients),
    ):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
        assert left.untyped_storage().data_ptr() != right.untyped_storage().data_ptr()

    assert world_f1.positions0.dtype == torch.float64
    assert world_f1.positions0.device.type == "cpu"
    assert tuple(world_f1.positions0.shape) == (3, 3)
    assert bool(torch.equal(world_f1.velocities, torch.zeros_like(world_f1.velocities)))
    assert tuple(world_f1.weight_coefficients.shape) == (3, 3)
    torch.testing.assert_close(
        world_f1.weight_coefficients[0],
        torch.tensor((0.125, -0.1875, 0.3125), dtype=torch.float64),
        rtol=0.0,
        atol=0.0,
    )
    grid_units = world_f1.positions0 / 0.25
    torch.testing.assert_close(grid_units, grid_units.round(), rtol=0.0, atol=0.0)
    assert int(grid_units.abs().max().item()) <= MAX_QUANTIZED_INTEGER_MAGNITUDE

    material = first.initialize_p0_material(world_f1)
    material.assert_current(world_f1)
    assert material.temporal_basis == "P0"
    assert material.layout == P0_MATERIAL_LAYOUT == "rgb_then_density"
    assert material.site_rgba_f32.dtype == torch.float32
    assert material.site_rgba_f32.device.type == "cpu"
    assert tuple(material.site_rgba_f32.shape) == (3, 4)
    torch.testing.assert_close(
        material.site_rgba_f32[:, 3],
        torch.full((3,), 64.0, dtype=torch.float32),
    )
    torch.testing.assert_close(
        material.site_rgba_f32[0, :3],
        torch.tensor((1.0, 0.0, 64.0 / 255.0), dtype=torch.float32),
    )

    report_f1 = first.storage_report(requested_frame_count=1)
    report_f300 = first.storage_report(requested_frame_count=300)
    assert report_f1.geometry_parameter_bytes == report_f300.geometry_parameter_bytes
    assert report_f1.p0_material_parameter_bytes == report_f300.p0_material_parameter_bytes
    assert report_f1.total_parameter_bytes == report_f300.total_parameter_bytes
    assert report_f1.frame_dependent_parameter_bytes == 0
    assert report_f300.stored_frame_state_bytes == 0
    accounting = first.accounting(requested_frame_count=300)
    assert accounting["request_frame_count_used_to_initialize_parameters"] is False
    assert accounting["target_or_video_decode_used"] is False
    assert accounting["camera_values_used_to_initialize_parameters"] is False
    assert accounting["raw_optimizer_parameterization_owned_here"] is False
    assert accounting["exact_compiler_inputs_on_bounded_dyadic_grid"] is True
    assert accounting["selection_scratch_entry_bound"] == first.site_count
    assert accounting["selection_scratch_independent_of_source_point_count"] is True


def test_initializer_plugs_into_lazy_provider_without_decoding_targets(
    tmp_path: Path,
) -> None:
    class NoDecodeTargetSource:
        view_count = 1
        frame_count = 2
        height = 2
        width = 3
        decode_calls = 0

        def select_view_frames(self, view_indices, frame_indices):
            self.decode_calls += 1
            raise AssertionError("provider construction must not decode targets")

        def residency(self):
            return {
                "source_kind": "initializer_contract_no_decode_fixture",
                "source_device": "disk",
                "logical_bytes": 1 * 2 * 3 * 2 * 3 * 4,
                "resident_bytes": 0,
                "full_source_resident": False,
                "disk_lazy_decode": True,
            }

    class NoCompileProgramFactory:
        provenance = "initializer-contract-no-compile-factory-v1"
        generation_digest = _sha256(provenance)

        def compile_track(self, request):
            raise AssertionError("provider construction must not compile tracks")

    asset = tmp_path / "world.ply"
    _write_ascii_ply(asset)
    initializer = prepare_paper_kinetic_point_cloud_world_initializer(_config(asset))
    source = NoDecodeTargetSource()
    camera = CameraSpec(
        fx=torch.tensor(2.0, dtype=torch.float64),
        fy=torch.tensor(2.0, dtype=torch.float64),
        cx=torch.tensor(1.5, dtype=torch.float64),
        cy=torch.tensor(1.0, dtype=torch.float64),
        camera_to_world=torch.eye(4, dtype=torch.float64),
        lens_model="pinhole",
        distortion=None,
    )
    provider = prepare_paper_kinetic_lazy_program_bundle_provider(
        dataset_generation_digest=_sha256("initializer-provider-dataset"),
        target_provider=PowerFoamTargetProvider(
            source=source,
            device=torch.device("cpu"),
        ),
        ray_provider=PowerFoamRayProvider(
            cameras=((camera, camera),),
            height=2,
            width=3,
            device=torch.device("cpu"),
        ),
        frame_times=(0.0, 1.0),
        height=2,
        width=3,
        maximum_tracks_per_bundle=2,
        maximum_observations_per_bundle=2,
        maximum_rows_per_native_block=2,
        world_initializer=initializer,
        program_factory=NoCompileProgramFactory(),
    )
    provider.assert_current()
    material = initializer.initialize_p0_material(provider.world.sites)
    material.assert_current(provider.world.sites)
    initializer_ref = weakref.ref(initializer)
    del initializer
    gc.collect()

    assert initializer_ref() is None
    provider.assert_current()
    assert provider.accounting()["provider_retains_world_initializer"] is False
    assert provider.accounting()["initializer_contract_receipt_only"] is True
    assert (
        provider.accounting()["initializer_template_tensor_bytes_retained_by_provider"]
        == 0
    )
    assert source.decode_calls == 0


def test_material_is_content_bound_but_returned_world_does_not_alias_template(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "world.ply"
    _write_ascii_ply(asset)
    initializer = prepare_paper_kinetic_point_cloud_world_initializer(_config(asset))
    request = _request(initializer, frame_count=8)
    sites = initializer.initialize_world(request)
    material = initializer.initialize_p0_material(sites)

    sites.positions0[0, 0].add_(0.25)
    with pytest.raises(ValueError, match="different sites"):
        material.assert_current(sites)
    with pytest.raises(ValueError, match="non-initializer site contents"):
        initializer.initialize_p0_material(sites)

    fresh = initializer.initialize_world(request)
    assert float(fresh.positions0[0, 0]) == 0.0
    initializer.assert_current()


def test_external_physical_p0_material_factory_owns_and_binds_exact_world(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "world.ply"
    _write_ascii_ply(asset)
    initializer = prepare_paper_kinetic_point_cloud_world_initializer(_config(asset))
    sites = initializer.initialize_world(_request(initializer, frame_count=8))
    physical = torch.tensor(
        (
            (0.15, 0.25, 0.35, 0.5),
            (0.45, 0.55, 0.65, 0.75),
            (0.75, 0.35, 0.20, 1.25),
        ),
        dtype=torch.float32,
    )

    sealed = prepare_paper_kinetic_p0_material_initialization(
        physical,
        sites,
        initializer_generation_digest=initializer.generation_digest,
        source_material_seed_digest=_sha256("configured-physical-material"),
    )

    sealed.assert_current(sites)
    torch.testing.assert_close(sealed.site_rgba_f32, physical)
    assert sealed.site_rgba_f32.data_ptr() != physical.data_ptr()
    physical[0, 0] = 0.95
    sealed.assert_current(sites)
    assert float(sealed.site_rgba_f32[0, 0]) == pytest.approx(0.15)

    foreign_sites = initializer.initialize_world(_request(initializer, frame_count=16))
    foreign_sites.positions0[0, 0] += 0.25
    with pytest.raises(ValueError, match="different sites"):
        sealed.assert_current(foreign_sites)


def test_material_seed_changes_do_not_invalidate_structural_generation(
    tmp_path: Path,
) -> None:
    first_asset = tmp_path / "first.ply"
    second_asset = tmp_path / "second.ply"
    _write_ascii_ply(first_asset)
    _write_ascii_ply(second_asset)
    second_asset.write_text(
        second_asset.read_text(encoding="utf-8").replace("255 0 64", "1 2 3"),
        encoding="utf-8",
    )
    first = prepare_paper_kinetic_point_cloud_world_initializer(_config(first_asset))
    recolored = prepare_paper_kinetic_point_cloud_world_initializer(_config(second_asset))
    denser = prepare_paper_kinetic_point_cloud_world_initializer(
        _config(first_asset, initial_density=32.0)
    )

    assert first.generation_digest == recolored.generation_digest
    assert first.generation_digest == denser.generation_digest
    assert (
        first.p0_material_seed_generation_digest
        != recolored.p0_material_seed_generation_digest
    )
    assert (
        first.p0_material_seed_generation_digest
        != denser.p0_material_seed_generation_digest
    )
    first_sites = first.initialize_world(_request(first, frame_count=4))
    recolored_sites = recolored.initialize_world(_request(recolored, frame_count=9))
    torch.testing.assert_close(
        first_sites.positions0,
        recolored_sites.positions0,
        rtol=0.0,
        atol=0.0,
    )
    assert (
        first.initialize_p0_material(first_sites).generation_digest
        != recolored.initialize_p0_material(recolored_sites).generation_digest
    )


def test_request_and_source_asset_generation_drift_fail_closed(tmp_path: Path) -> None:
    asset = tmp_path / "world.ply"
    _write_ascii_ply(asset)
    initializer = prepare_paper_kinetic_point_cloud_world_initializer(_config(asset))

    foreign = _request(
        initializer,
        frame_count=3,
        initializer_generation_digest=_sha256("foreign-initializer"),
    )
    foreign.assert_self_consistent()
    with pytest.raises(ValueError, match="different initializer"):
        initializer.initialize_world(foreign)

    _write_ascii_ply(asset, duplicate=True)
    with pytest.raises(ValueError, match="source asset changed"):
        initializer.initialize_world(_request(initializer, frame_count=3))


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"frame_count": 300}, "config keys differ"),
        ({"sample_mode": "random"}, "sample_mode"),
        ({"sample_seed": 1}, "sample_seed must be zero"),
        ({"coordinate_quantization_step": 0.3}, "binary power of two"),
        ({"weight_quantization_step": 0.1}, "binary power of two"),
        ({"weight_coefficients": (0.0, 0.0, 0.0, 0.0)}, "1..3"),
        ({"initial_density": 0.0}, "strictly positive"),
        ({"site_count": 5}, "refuses duplicate padding"),
        ({"source_coordinate_frame": "multicam_world"}, "source_coordinate_frame"),
    ),
)
def test_unsupported_or_frame_dependent_config_fails_closed(
    tmp_path: Path,
    updates,
    message: str,
) -> None:
    asset = tmp_path / "world.ply"
    _write_ascii_ply(asset)
    with pytest.raises((TypeError, ValueError), match=message):
        prepare_paper_kinetic_point_cloud_world_initializer(_config(asset, **updates))


def test_source_asset_and_declared_point_budgets_fail_before_point_loading(
    tmp_path: Path,
) -> None:
    asset = tmp_path / "world.ply"
    _write_ascii_ply(asset)

    with pytest.raises(MemoryError, match="pre-load byte budget"):
        prepare_paper_kinetic_point_cloud_world_initializer(
            _config(asset, maximum_source_asset_bytes=1)
        )
    with pytest.raises(MemoryError, match="point count.*pre-load budget"):
        prepare_paper_kinetic_point_cloud_world_initializer(
            _config(asset, maximum_source_point_count=3)
        )


def test_external_coordinates_require_explicit_affine_then_quantize(tmp_path: Path) -> None:
    asset = tmp_path / "world.ply"
    _write_ascii_ply(asset)
    transform = (
        (1.0, 0.0, 0.0, 10.0),
        (0.0, 1.0, 0.0, -2.0),
        (0.0, 0.0, 1.0, 0.5),
        (0.0, 0.0, 0.0, 1.0),
    )
    initializer = prepare_paper_kinetic_point_cloud_world_initializer(
        _config(
            asset,
            source_coordinate_frame="external_affine",
            point_transform=transform,
            sample_mode="sha256_rank",
            sample_seed=17,
        )
    )
    sites = initializer.initialize_world(_request(initializer, frame_count=2))
    units = sites.positions0 / 0.25
    torch.testing.assert_close(units, units.round(), rtol=0.0, atol=0.0)
    assert initializer.accounting(requested_frame_count=1024)[
        "frame_dependent_parameter_bytes"
    ] == 0

    with pytest.raises(ValueError, match="require an explicit point transform"):
        prepare_paper_kinetic_point_cloud_world_initializer(
            _config(asset, source_coordinate_frame="external_affine")
        )
