from __future__ import annotations

import hashlib

import torch
from kinetic_lazy_native_material_step import (
    TARGET_FRAME_STEP_CACHE,
    paper_kinetic_observation_manifest_digest,
    prepare_paper_kinetic_lazy_native_trainer_state,
    run_paper_kinetic_lazy_native_material_step,
)
from paper_kinetic_fixed_site_material_device_bridge import (
    snapshot_paper_kinetic_fixed_site_material_to_device,
)
from paper_kinetic_fixed_site_material_state import (
    PaperKineticFixedSiteMaterialParameterization,
    PaperKineticFixedSiteMaterialSGDPolicy,
    prepare_paper_kinetic_fixed_site_material_state,
)
from paper_kinetic_lazy_material_device_bridge import (
    apply_paper_kinetic_lazy_material_device_gradient_receipt,
    seal_paper_kinetic_lazy_material_device_gradient_receipt,
)
from paper_kinetic_world_initializer import (
    prepare_paper_kinetic_p0_material_initialization,
)
from test_kinetic_lazy_native_material_step import (
    _FakeNativeOps,
    _background,
    _material,
    _memory_policy,
    _observations,
    _provider,
)


def _sha256(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def test_lazy_bar_receipt_updates_cpu_material_and_releases_snapshot() -> None:
    _source, _factory, provider = _provider(
        maximum_tracks_per_bundle=2,
        maximum_observations_per_bundle=6,
    )
    observations = _observations(
        (
            (0, 0, 0),
            (0, 2, 0),
            (0, 1, 1),
            (0, 2, 2),
            (1, 1, 3),
            (1, 2, 4),
        )
    )
    initialization = prepare_paper_kinetic_p0_material_initialization(
        _material(),
        provider.world.sites,
        initializer_generation_digest=_sha256("lazy-bridge-initializer"),
        source_material_seed_digest=_sha256("lazy-bridge-material-seed"),
    )
    material_state = prepare_paper_kinetic_fixed_site_material_state(
        initialization,
        provider.world,
        parameterization=PaperKineticFixedSiteMaterialParameterization(),
        optimizer_policy=PaperKineticFixedSiteMaterialSGDPolicy(
            color_learning_rate=0.01,
            density_learning_rate=0.01,
        ),
        device="cpu",
        maximum_material_state_logical_tensor_bytes=1_000_000,
    )
    background = _background()
    background_generation_id = _sha256("lazy-bridge-background")
    snapshot = snapshot_paper_kinetic_fixed_site_material_to_device(
        material_state,
        background_rgb_f32_cpu=background,
        background_generation_id=background_generation_id,
        device="cpu",
        device_completion_fence=lambda: None,
        device_completion_fence_provenance="cpu-synchronous-test-fence-v1",
    )
    material_before = material_state.site_rgba_f32.clone()
    lazy_state = prepare_paper_kinetic_lazy_native_trainer_state(
        provider,
        device="cpu",
    )
    grad = torch.zeros_like(snapshot.site_rgba_f32_device)
    callback_values = {}

    def optimizer_update(result) -> None:
        receipt = seal_paper_kinetic_lazy_material_device_gradient_receipt(
            material_state,
            snapshot,
            result,
        )
        step_receipt = apply_paper_kinetic_lazy_material_device_gradient_receipt(
            material_state,
            receipt,
        )
        callback_values.update(receipt=receipt, step_receipt=step_receipt)

    result = run_paper_kinetic_lazy_native_material_step(
        lazy_state,
        provider,
        observations,
        step_index=0,
        expected_observation_count=len(observations),
        expected_observation_manifest_digest=(
            paper_kinetic_observation_manifest_digest(observations)
        ),
        loss_normalization_id="lazy-bridge-cpu-e2e-v1",
        material_generation_id=snapshot.material_generation_id,
        background_generation_id=background_generation_id,
        global_site_rgba_f32=snapshot.site_rgba_f32_device,
        global_grad_site_rgba_f32=grad,
        background_rgb_f32=snapshot.background_rgb_f32_device,
        native_ops=_FakeNativeOps(),
        maximum_samples_per_launch=2,
        memory_policy=_memory_policy(
            provider,
            target_frame_access_mode=TARGET_FRAME_STEP_CACHE,
        ),
        optimizer_update=optimizer_update,
    )

    receipt = callback_values["receipt"]
    step_receipt = callback_values["step_receipt"]
    result.assert_current()
    receipt.assert_current(material_state, require_unconsumed=False)
    step_receipt.assert_current(material_state)
    assert material_state.step_index == 1
    assert not torch.equal(material_state.site_rgba_f32, material_before)
    assert receipt.accounting()["global_material_bar_shape"] == [1, 4]
    assert receipt.accounting()["device_to_host_tensor_count"] == 2
    assert receipt.accounting()["live_cpu_receipt_tensor_bytes"] == 0
    assert snapshot.released_after_consumption
    assert snapshot.site_rgba_f32_device is None
    assert snapshot.background_rgb_f32_device is None

