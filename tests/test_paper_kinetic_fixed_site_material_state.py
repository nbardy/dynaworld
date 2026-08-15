from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from paper_kinetic_fixed_site_material_state import (
    PaperKineticFixedSiteMaterialParameterization,
    PaperKineticFixedSiteMaterialSGDPolicy,
    PaperKineticFixedSiteMaterialStepReceipt,
    apply_paper_kinetic_fixed_site_material_sgd_step,
    checkpoint_paper_kinetic_fixed_site_material_state,
    paper_kinetic_fixed_site_material_checkpoint_from_payload,
    prepare_paper_kinetic_fixed_site_material_state,
    restore_paper_kinetic_fixed_site_material_state,
)
from paper_kinetic_lazy_program_bundles import (
    PROVIDER_PROVENANCE,
    WORLD_SNAPSHOT_PROVENANCE,
    PaperKineticWorldSnapshot,
    PaperKineticWorldInitializationRequest,
    _digest_parts,
    _site_content_digest,
    _site_tensors,
    _tensor_signature,
)
from paper_kinetic_world_initializer import (
    prepare_paper_kinetic_point_cloud_world_initializer,
)
from kinetic_dense_cached_native_material_request import (
    PaperKineticDenseOptimizerAuthorization,
)


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_world(path: Path) -> None:
    path.write_text(
        "\n".join(
            (
                "ply",
                "format ascii 1.0",
                "element vertex 2",
                "property float x",
                "property float y",
                "property float z",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "end_header",
                "0.25 0.50 -0.25 64 96 128",
                "1.00 -0.50 0.75 192 160 32",
                "",
            )
        ),
        encoding="utf-8",
    )


def _request(initializer) -> PaperKineticWorldInitializationRequest:
    provisional = PaperKineticWorldInitializationRequest(
        dataset_generation_digest=_sha256("fixed-site-dataset"),
        camera_grid_digest=_sha256("fixed-site-cameras"),
        view_count=2,
        frame_count=300,
        height=32,
        width=48,
        initializer_generation_digest=initializer.generation_digest,
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


def _state(
    tmp_path: Path,
    *,
    initial_density: float = 64.0,
    parameterization: PaperKineticFixedSiteMaterialParameterization | None = None,
):
    asset = tmp_path / "fixed_site_world.ply"
    _write_world(asset)
    initializer = prepare_paper_kinetic_point_cloud_world_initializer(
        {
            "source_path": asset,
            "source_coordinate_frame": "model",
            "point_transform": None,
            "maximum_source_asset_bytes": 1_000_000,
            "maximum_source_point_count": 100,
            "site_count": 2,
            "sample_mode": "first",
            "sample_seed": 0,
            "coordinate_quantization_step": 0.25,
            "weight_coefficients": (0.0,),
            "weight_quantization_step": 0.0625,
            "initial_density": initial_density,
        }
    )
    sites = initializer.initialize_world(_request(initializer))
    material = initializer.initialize_p0_material(sites)
    dataset_generation_digest = _sha256("fixed-site-dataset")
    sites_content_digest = _site_content_digest(sites)
    world = PaperKineticWorldSnapshot(
        sites=sites,
        dataset_generation_digest=dataset_generation_digest,
        initializer_generation_digest=initializer.generation_digest,
        sites_content_digest=sites_content_digest,
        generation_digest=_digest_parts(
            WORLD_SNAPSHOT_PROVENANCE,
            dataset_generation_digest,
            initializer.generation_digest,
            sites_content_digest,
            sites.site_count,
        ),
        _site_tensor_identities=tuple(id(tensor) for tensor in _site_tensors(sites)),
        _site_tensor_signatures=tuple(
            _tensor_signature(tensor) for tensor in _site_tensors(sites)
        ),
    )
    world.assert_current()
    state = prepare_paper_kinetic_fixed_site_material_state(
        material,
        world,
        parameterization=(
            PaperKineticFixedSiteMaterialParameterization()
            if parameterization is None
            else parameterization
        ),
        optimizer_policy=PaperKineticFixedSiteMaterialSGDPolicy(
            color_learning_rate=0.125,
            density_learning_rate=0.25,
        ),
        device="cpu",
        maximum_material_state_logical_tensor_bytes=1_000_000,
    )
    return state, material, world


def _authorization(
    state,
    *,
    full_geometry: bool = False,
) -> tuple[PaperKineticDenseOptimizerAuthorization, SimpleNamespace]:
    grad = torch.tensor(
        ((2.0, -3.0, 4.0, 0.5), (-1.5, 0.25, 3.0, -0.75)),
        dtype=torch.float32,
    )
    authorization = PaperKineticDenseOptimizerAuthorization(
        source_generation_digest=_sha256("source"),
        compact_manifest_digest=_sha256("manifest"),
        step_generation_id="fixed-site-material-step-0",
        replay_receipt_generation_digest=_sha256("replay"),
        accumulator_generation_digest=_sha256("accumulator"),
        request_count=2,
        observation_count=4,
        full_geometry=full_geometry,
        optimize_camera_rays=False,
        grad_site_rgba_f32=grad,
        loss_f32=torch.tensor((1.25,), dtype=torch.float32),
        grad_positions0_f64=(
            torch.zeros((state.site_count, 3), dtype=torch.float64)
            if full_geometry
            else None
        ),
        grad_velocities_f64=None,
        grad_weight_coefficients_f64=None,
        ray_bar_keys=(),
        grad_track_ray_coefficients_f64=None,
        tensor_signatures=(),
        generation_digest=_sha256("authorization"),
        _accumulator_identity=0,
    )
    accumulator = SimpleNamespace(
        full_geometry=full_geometry,
        optimize_camera_rays=False,
        _material_tensor_ref=state.site_rgba_f32,
        material_generation_id=state.material_generation_id,
        world_generation_digest=state.world_generation_digest,
        world_sites_content_digest=state.sites_content_digest,
    )
    return authorization, accumulator


def _snapshot(state):
    return (
        tuple(tensor.clone() for tensor in state._tensors()),
        state.material_generation_id,
        state.step_index,
    )


def _assert_snapshot_unchanged(state, snapshot) -> None:
    tensors, generation, step_index = snapshot
    for current, previous in zip(state._tensors(), tensors, strict=True):
        torch.testing.assert_close(current, previous, rtol=0.0, atol=0.0)
    assert state.material_generation_id == generation
    assert state.step_index == step_index
    state.assert_current()


def test_fixed_site_state_and_restart_storage_are_frame_independent(tmp_path: Path) -> None:
    state, material, world = _state(tmp_path)
    state.assert_current()
    torch.testing.assert_close(state.site_rgba_f32, material.site_rgba_f32)
    assert tuple(state.raw_color_f32.shape) == (2, 3)
    assert tuple(state.raw_density_f32.shape) == (2,)
    assert bool(torch.equal(state.raw_color_grad_f32, torch.zeros_like(state.raw_color_f32)))
    assert bool(
        torch.equal(state.raw_density_grad_f32, torch.zeros_like(state.raw_density_f32))
    )

    f1 = state.accounting(requested_frame_count=1)
    f300 = state.accounting(requested_frame_count=300)
    assert f1["total_persistent_tensor_bytes"] == f300["total_persistent_tensor_bytes"]
    assert f300["total_persistent_tensor_bytes"] == 2 * 12 * 4
    assert f300["frame_dependent_parameter_bytes"] == 0
    assert f300["optimizer_history_tensor_bytes"] == 0
    assert f300["geometry_trainable"] is False
    assert f300["accelerator_optimizer_update_supported"] is False
    assert "version_chain" in f300["material_generation_semantics"]
    with pytest.raises(NotImplementedError, match="CPU-only"):
        prepare_paper_kinetic_fixed_site_material_state(
            material,
            world,
            parameterization=PaperKineticFixedSiteMaterialParameterization(),
            optimizer_policy=PaperKineticFixedSiteMaterialSGDPolicy(
                color_learning_rate=0.125,
                density_learning_rate=0.25,
            ),
            device="mps",
            maximum_material_state_logical_tensor_bytes=1_000_000,
        )

    with pytest.raises(MemoryError, match="state exceeds"):
        prepare_paper_kinetic_fixed_site_material_state(
            material,
            world,
            parameterization=PaperKineticFixedSiteMaterialParameterization(),
            optimizer_policy=PaperKineticFixedSiteMaterialSGDPolicy(
                color_learning_rate=0.125,
                density_learning_rate=0.25,
            ),
            device="cpu",
            maximum_material_state_logical_tensor_bytes=(2 * 12 * 4 - 1),
        )

    checkpoint = checkpoint_paper_kinetic_fixed_site_material_state(state)
    assert checkpoint.checkpoint_tensor_bytes == 2 * 4 * 4
    payload = checkpoint.payload()
    assert (
        payload["raw_color_f32_cpu"].untyped_storage().data_ptr()
        != checkpoint.raw_color_f32_cpu.untyped_storage().data_ptr()
    )
    parsed = paper_kinetic_fixed_site_material_checkpoint_from_payload(
        payload,
        expected_world_site_count=world.site_count,
        maximum_checkpoint_logical_tensor_bytes=1_000_000,
    )
    with pytest.raises(MemoryError, match="checkpoint exceeds"):
        paper_kinetic_fixed_site_material_checkpoint_from_payload(
            payload,
            expected_world_site_count=world.site_count,
            maximum_checkpoint_logical_tensor_bytes=(2 * 4 * 4 - 1),
        )
    assert (
        parsed.raw_color_f32_cpu.untyped_storage().data_ptr()
        != payload["raw_color_f32_cpu"].untyped_storage().data_ptr()
    )
    restored = restore_paper_kinetic_fixed_site_material_state(
        parsed,
        world=world,
        device="cpu",
        maximum_material_state_logical_tensor_bytes=1_000_000,
    )
    torch.testing.assert_close(restored.raw_color_f32, state.raw_color_f32)
    torch.testing.assert_close(restored.raw_density_f32, state.raw_density_f32)
    torch.testing.assert_close(restored.site_rgba_f32, state.site_rgba_f32)
    assert restored.material_generation_id == state.material_generation_id
    assert restored.restart_checkpoint_generation_digest == checkpoint.generation_digest

    foreign_dataset_digest = _sha256("foreign-fixed-site-dataset")
    foreign_world = replace(
        world,
        dataset_generation_digest=foreign_dataset_digest,
        generation_digest=_digest_parts(
            WORLD_SNAPSHOT_PROVENANCE,
            foreign_dataset_digest,
            world.initializer_generation_digest,
            world.sites_content_digest,
            world.site_count,
        ),
    )
    foreign_world.assert_current()
    with pytest.raises(ValueError, match="different world"):
        restore_paper_kinetic_fixed_site_material_state(
            checkpoint,
            world=foreign_world,
            device="cpu",
            maximum_material_state_logical_tensor_bytes=1_000_000,
        )
    with pytest.raises(MemoryError, match="state exceeds"):
        restore_paper_kinetic_fixed_site_material_state(
            checkpoint,
            world=world,
            device="cpu",
            maximum_material_state_logical_tensor_bytes=(2 * 12 * 4 - 1),
        )


def test_density_seed_round_trips_canonical_nonlinear_and_tiny_branches(
    tmp_path: Path,
) -> None:
    nonlinear = PaperKineticFixedSiteMaterialParameterization(
        density_beta=2.0,
        density_threshold=3.0,
        minimum_density=0.1,
    )
    nonlinear_state, nonlinear_seed, _ = _state(
        tmp_path,
        initial_density=0.35,
        parameterization=nonlinear,
    )
    torch.testing.assert_close(
        nonlinear_state.site_rgba_f32[:, 3],
        nonlinear_seed.site_rgba_f32[:, 3],
        rtol=2.0e-6,
        atol=0.0,
    )
    assert bool(
        torch.all(
            nonlinear_state.raw_density_f32 * nonlinear.density_beta
            <= nonlinear.density_threshold
        ).item()
    )

    # A float32 raw softplus value cannot represent every arbitrary tiny
    # physical seed to the lifecycle's declared 2e-6 relative round-trip
    # bound.  Exercise a genuinely representable tiny branch, then prove that
    # a nearby non-representable seed fails closed instead of being clamped.
    representable_tiny_density = float(
        F.softplus(torch.tensor(-70.0, dtype=torch.float32)).item()
    )
    tiny_state, tiny_seed, _ = _state(
        tmp_path,
        initial_density=representable_tiny_density,
    )
    torch.testing.assert_close(
        tiny_state.site_rgba_f32[:, 3],
        tiny_seed.site_rgba_f32[:, 3],
        rtol=2.0e-6,
        atol=0.0,
    )
    assert bool(torch.all(tiny_state.site_rgba_f32[:, 3] < 1.0e-30).item())

    with pytest.raises(
        ValueError,
        match="initial density does not round-trip",
    ):
        _state(tmp_path, initial_density=1.0e-31)


def test_manual_sgd_uses_exact_parameter_chain_rule_and_preserves_rgba_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, _, _world = _state(tmp_path)
    authorization, accumulator = _authorization(state)
    validation_calls = []

    def validate(self, actual_accumulator, replay_receipt) -> None:
        validation_calls.append((actual_accumulator, replay_receipt))

    monkeypatch.setattr(PaperKineticDenseOptimizerAuthorization, "assert_current", validate)
    raw_color_before = state.raw_color_f32.clone()
    raw_density_before = state.raw_density_f32.clone()
    physical_before = state.site_rgba_f32.clone()
    generation_before = state.material_generation_id
    expected_raw_color = raw_color_before - 0.125 * (
        authorization.grad_site_rgba_f32[:, :3]
        * physical_before[:, :3]
        * (1.0 - physical_before[:, :3])
    )
    # Density 64 is in PyTorch softplus's declared linear branch, whose exact
    # derivative is one rather than an unthresholded sigmoid approximation.
    expected_raw_density = (
        raw_density_before - 0.25 * authorization.grad_site_rgba_f32[:, 3]
    )

    receipt = apply_paper_kinetic_fixed_site_material_sgd_step(
        state,
        authorization,
        accumulator,
        object(),
    )

    assert len(validation_calls) == 1
    torch.testing.assert_close(state.raw_color_f32, expected_raw_color)
    torch.testing.assert_close(state.raw_density_f32, expected_raw_density)
    torch.testing.assert_close(state.site_rgba_f32[:, :3], torch.sigmoid(expected_raw_color))
    torch.testing.assert_close(
        state.site_rgba_f32[:, 3],
        F.softplus(expected_raw_density, beta=1.0, threshold=20.0),
    )
    assert state.material_generation_id != generation_before
    assert state.generation_parent_digest == generation_before
    assert state.step_index == 1
    assert receipt.geometry_updated is False
    assert receipt.persistent_tensor_bytes == 0
    assert not any(isinstance(value, torch.Tensor) for value in vars(receipt).values())
    assert bool(torch.all(state.raw_color_grad_f32 == 0.0))
    assert bool(torch.all(state.raw_density_grad_f32 == 0.0))
    receipt.assert_current(state)


def test_authorization_is_validated_before_mutation_and_geometry_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, _, _world = _state(tmp_path)
    authorization, accumulator = _authorization(state)
    before = _snapshot(state)

    def reject(*args) -> None:
        raise ValueError("injected stale authorization")

    monkeypatch.setattr(PaperKineticDenseOptimizerAuthorization, "assert_current", reject)
    with pytest.raises(ValueError, match="injected stale authorization"):
        apply_paper_kinetic_fixed_site_material_sgd_step(
            state,
            authorization,
            accumulator,
            object(),
        )
    _assert_snapshot_unchanged(state, before)

    validation_calls = []

    def accept(*args) -> None:
        validation_calls.append(True)

    monkeypatch.setattr(PaperKineticDenseOptimizerAuthorization, "assert_current", accept)
    foreign_authorization, foreign_accumulator = _authorization(state)
    foreign_accumulator.world_generation_digest = _sha256("foreign-world")
    with pytest.raises(ValueError, match="different world snapshot"):
        apply_paper_kinetic_fixed_site_material_sgd_step(
            state,
            foreign_authorization,
            foreign_accumulator,
            object(),
        )
    _assert_snapshot_unchanged(state, before)

    geometry_authorization, geometry_accumulator = _authorization(
        state,
        full_geometry=True,
    )
    with pytest.raises(ValueError, match="rejects geometry"):
        apply_paper_kinetic_fixed_site_material_sgd_step(
            state,
            geometry_authorization,
            geometry_accumulator,
            object(),
        )
    assert validation_calls == [True, True]
    _assert_snapshot_unchanged(state, before)


def test_rejected_material_candidate_clears_scratch_without_advancing_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, _, _world = _state(tmp_path)
    authorization, accumulator = _authorization(state)
    authorization = replace(
        authorization,
        grad_site_rgba_f32=torch.full(
            (state.site_count, 4),
            1.0e6,
            dtype=torch.float32,
        ),
        generation_digest=_sha256("oversized-candidate-authorization"),
    )
    before = _snapshot(state)
    monkeypatch.setattr(
        PaperKineticDenseOptimizerAuthorization,
        "assert_current",
        lambda *args: None,
    )

    with pytest.raises(ValueError, match="candidate exceeds"):
        apply_paper_kinetic_fixed_site_material_sgd_step(
            state,
            authorization,
            accumulator,
            object(),
        )

    _assert_snapshot_unchanged(state, before)


def test_checkpoint_binds_numeric_content_and_exact_history_schema(tmp_path: Path) -> None:
    state, _, _world = _state(tmp_path)
    checkpoint = checkpoint_paper_kinetic_fixed_site_material_state(state)

    changed_content = checkpoint.payload()
    changed_content["raw_density_f32_cpu"][0].add_(1.0)
    with pytest.raises(ValueError, match="tensor/generation changed"):
        paper_kinetic_fixed_site_material_checkpoint_from_payload(
            changed_content,
            expected_world_site_count=state.site_count,
            maximum_checkpoint_logical_tensor_bytes=1_000_000,
        )

    changed_history = checkpoint.payload()
    changed_history["step_index"] = True
    with pytest.raises(ValueError, match="nonnegative integer"):
        paper_kinetic_fixed_site_material_checkpoint_from_payload(
            changed_history,
            expected_world_site_count=state.site_count,
            maximum_checkpoint_logical_tensor_bytes=1_000_000,
        )

    foreign_dtype = checkpoint.payload()
    foreign_dtype["raw_color_f32_cpu"] = foreign_dtype["raw_color_f32_cpu"].double()
    with pytest.raises(ValueError, match="CPU float32"):
        paper_kinetic_fixed_site_material_checkpoint_from_payload(
            foreign_dtype,
            expected_world_site_count=state.site_count,
            maximum_checkpoint_logical_tensor_bytes=1_000_000,
        )


def test_post_mutation_receipt_failure_poisons_the_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, _, _world = _state(tmp_path)
    authorization, accumulator = _authorization(state)

    monkeypatch.setattr(
        PaperKineticDenseOptimizerAuthorization,
        "assert_current",
        lambda *args: None,
    )

    def reject_receipt(*args) -> None:
        raise ValueError("injected receipt seal failure")

    monkeypatch.setattr(
        PaperKineticFixedSiteMaterialStepReceipt,
        "assert_current",
        reject_receipt,
    )
    with pytest.raises(ValueError, match="receipt seal failure"):
        apply_paper_kinetic_fixed_site_material_sgd_step(
            state,
            authorization,
            accumulator,
            object(),
        )
    assert state.poisoned
    assert state.step_index == 1
    with pytest.raises(ValueError, match="poisoned"):
        state.assert_current()


def test_fresh_authorization_cannot_reuse_the_previous_step_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state, _, _world = _state(tmp_path)
    monkeypatch.setattr(
        PaperKineticDenseOptimizerAuthorization,
        "assert_current",
        lambda *args: None,
    )
    first, first_accumulator = _authorization(state)
    apply_paper_kinetic_fixed_site_material_sgd_step(
        state,
        first,
        first_accumulator,
        object(),
    )

    repeated_step, repeated_accumulator = _authorization(state)
    repeated_step = replace(
        repeated_step,
        generation_digest=_sha256("fresh-authorization-same-step"),
    )
    before = _snapshot(state)
    with pytest.raises(ValueError, match="previous logical step identity"):
        apply_paper_kinetic_fixed_site_material_sgd_step(
            state,
            repeated_step,
            repeated_accumulator,
            object(),
        )
    _assert_snapshot_unchanged(state, before)
