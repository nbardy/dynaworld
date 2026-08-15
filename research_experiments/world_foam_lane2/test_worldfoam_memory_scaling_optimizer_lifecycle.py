from __future__ import annotations

import hashlib

from test_kinetic_ragged_paper_step_cpu_fake_native import _FakeNativeOps
from test_paper_kinetic_fixed_site_material_step import (
    _Fence,
    _batch,
    _case,
    _policy,
)
from worldfoam_memory_scaling_mps_trial_driver import (
    OPTIMIZER_LIFECYCLE_PROTOCOL,
    run_worldfoam_fixed_site_optimizer_lifecycle,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def test_cpu_proxy_two_step_optimizer_lifecycle_is_bounded_and_replay_exact() -> None:
    (
        _target_source,
        _factory,
        provider,
        store,
        _legacy_coordinator_state,
        initial_material,
        background,
    ) = _case(maximum_store_entries=6, maximum_store_bytes=60_000_000)

    report = run_worldfoam_fixed_site_optimizer_lifecycle(
        provider=provider,
        artifact_store=store,
        batch=_batch(),
        step_policy=_policy(),
        initial_site_rgba_f32_cpu=initial_material,
        background_rgb_f32_cpu=background,
        device="cpu",
        native_ops=_FakeNativeOps(),
        backend_provenance="cpu-fake-native/optimizer-lifecycle-proxy-v1",
        device_completion_fence=_Fence(),
        device_completion_fence_provenance="cpu-synchronous-fence-v1",
        initializer_generation_digest=_digest("optimizer-lifecycle-initializer"),
        source_material_seed_digest=_digest("optimizer-lifecycle-material-seed"),
        target_generation_id=_digest("optimizer-lifecycle-targets"),
        maximum_material_state_logical_tensor_bytes=1_000_000,
    )

    site_count = provider.world.site_count
    assert report["protocol"] == OPTIMIZER_LIFECYCLE_PROTOCOL
    assert report["executed_optimizer_step_count"] == 2
    assert report["in_process_checkpoint_replay_optimizer_step_count"] == 1
    assert report["strict_monotone_loss"] is True
    first_loss, second_loss = report["losses_rgb_mean"]
    assert second_loss < first_loss
    assert report["in_process_checkpoint_continuation_exact"] is True
    assert report["fresh_process_restart_verified"] is False
    assert report["device_lifecycle_executed"] is True
    assert report["native_runtime_verified"] is False
    assert report["native_runtime_attestation_required_for_promotion"] is True
    assert report["cpu_proxy_verified"] is True

    material = report["material_state_accounting"]
    assert material["total_persistent_tensor_bytes"] == 48 * site_count
    assert material["optimizer_history_tensor_bytes"] == 0
    assert material["frame_dependent_parameter_bytes"] == 0
    assert report["checkpoint_tensor_bytes"] == 16 * site_count
    assert report["peak_bridge_tensor_bytes"] == 32 * site_count + 16
    assert report["final_material_generation_id"] == (
        report["in_process_checkpoint_final_material_generation_id"]
    )

    records = (
        *report["steps"],
        report["in_process_checkpoint_second_step"],
    )
    for record in records:
        assert record["parameter_mutation_count"] == 1
        assert record["material_state_identity_preserved"] is True
        assert record["material_generation_id_before"] != (
            record["material_generation_id_after"]
        )

        snapshot_before = record["snapshot_accounting"]
        assert snapshot_before["device_material_copy_count"] == 1
        assert snapshot_before["device_material_content_digest_count"] == 0
        assert snapshot_before["device_material_readback_count"] == 0
        assert snapshot_before["live_snapshot_tensor_bytes"] == (
            16 * site_count + 12
        )

        bridge_before = record["bridge_accounting_before_apply"]
        assert bridge_before["production_step_result_bound"] is True
        assert bridge_before["device_to_host_copy_phase_count"] == 1
        assert bridge_before["device_to_host_tensor_count"] == 2
        assert bridge_before["live_cpu_receipt_tensor_bytes"] == (
            16 * site_count + 4
        )
        assert bridge_before["live_bridge_tensor_bytes"] == (
            32 * site_count + 16
        )

        bridge_after = record["bridge_accounting_after_apply"]
        assert bridge_after["consumed"] is True
        assert bridge_after["released_after_consumption"] is True
        assert bridge_after["live_bridge_tensor_bytes"] == 0
        snapshot_after = record["snapshot_accounting_after_apply"]
        assert snapshot_after["released_after_consumption"] is True
        assert snapshot_after["live_snapshot_tensor_bytes"] == 0

        coordinator = record["coordinator_accounting"]
        assert coordinator["optimizer_step_executed"] is False
        assert coordinator["parameter_mutation_count"] == 0
        for key in (
            "persistent_frame_tensor_bytes",
            "persistent_sample_tensor_bytes",
            "persistent_target_tensor_bytes",
            "persistent_prediction_tensor_bytes",
        ):
            assert coordinator[key] == 0
            assert bridge_before[key] == 0
