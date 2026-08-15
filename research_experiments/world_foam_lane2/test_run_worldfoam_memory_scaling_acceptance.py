from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

import run_worldfoam_memory_scaling_acceptance as producer


def _driver_capabilities() -> dict[str, object]:
    return {
        "schema_version": producer.DRIVER_CAPABILITY_SCHEMA_VERSION,
        "driver_protocol": producer.DRIVER_PROTOCOL,
        "supported_backends": ("mps",),
        "selected_pixel_target_access": {
            "implemented": True,
            "access_mode": "direct_pixels",
            "full_frame_materialization_count": 0,
            "preserves_request_order_and_duplicates": True,
            "source_budget_enforced_before_allocation": True,
            "contract": "PowerFoamSelectedPixelRead/v1",
        },
    }


def test_static_driver_capabilities_cover_only_driver_owned_selected_pixels() -> None:
    capabilities = _driver_capabilities()
    producer._validate_driver_capabilities(capabilities)

    blockers = producer._driver_capability_blockers(
        capabilities,
        {"require_selected_pixel_target_access": True},
    )

    assert blockers == ()


def test_static_driver_capability_blocker_rejects_full_frame_target_access() -> None:
    capabilities = _driver_capabilities()
    capabilities["selected_pixel_target_access"][  # type: ignore[index]
        "access_mode"
    ] = "full_frame_fallback"

    blockers = producer._driver_capability_blockers(
        capabilities,
        {"require_selected_pixel_target_access": True},
    )

    assert blockers == ("selected_pixel_target_access",)


def test_static_capability_reader_does_not_import_the_trial_driver(tmp_path) -> None:
    driver_path = tmp_path / "driver.py"
    driver_path.write_text(
        f"{producer.DRIVER_CAPABILITY_CONSTANT} = {_driver_capabilities()!r}\n"
        "raise RuntimeError('driver import must not run during preflight')\n",
        encoding="utf-8",
    )

    capabilities = producer._load_driver_capabilities(driver_path)

    assert capabilities["driver_protocol"] == producer.DRIVER_PROTOCOL


def test_checked_in_mps_driver_satisfies_observable_v3_preflight() -> None:
    driver_path = Path(__file__).with_name(
        "worldfoam_memory_scaling_mps_trial_driver.py"
    )
    capabilities = producer._load_driver_capabilities(driver_path)

    blockers = producer._driver_capability_blockers(
        capabilities,
        producer.verifier.load_json_object(producer.verifier.DEFAULT_CONTRACT),
    )

    assert blockers == ()


def test_checked_in_source_manifest_closes_local_python_imports() -> None:
    driver_path = Path(__file__).with_name(
        "worldfoam_memory_scaling_mps_trial_driver.py"
    )
    config_path = Path(__file__).with_name(
        "worldfoam_memory_scaling_mps_trial_v1.json"
    )
    manifest, _digest = producer.build_source_manifest(
        trial_driver_path=driver_path,
        trial_config_path=config_path,
    )
    manifested = {str(record["path"]) for record in manifest}
    assert {
        "src/train/camera.py",
        "src/train/paper_kinetic_active_track_program_factory.py",
        "research_experiments/world_foam_lane2/kinetic_chart_transfer_bridge.py",
        "research_experiments/world_foam_lane2/rational_polynomial_roots.py",
        (
            "third_party/fast-mac-gsplat/variants/"
            "world_foam_lane2_fused_slab_v0/setup.py"
        ),
    } <= manifested

    for label in tuple(manifested):
        if not label.endswith(".py") or label.startswith("external/"):
            continue
        tree = ast.parse((producer.ROOT / label).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: tuple[str, ...] = ()
            if isinstance(node, ast.Import):
                modules = tuple(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = (node.module,)
            for module_name in modules:
                for dependency in producer._local_module_source_files(module_name):
                    dependency_label = dependency.relative_to(producer.ROOT).as_posix()
                    assert dependency_label in manifested


def test_local_source_closure_includes_relative_imports_and_package_initializers() -> None:
    objective_loss = producer.ROOT / "src" / "train" / "objective" / "loss.py"
    closure = {
        path.relative_to(producer.ROOT).as_posix()
        for path in producer._local_python_source_closure((objective_loss,))
    }

    assert {
        "src/train/objective/__init__.py",
        "src/train/objective/background.py",
        "src/train/objective/choices.py",
        "src/train/objective/metal_dssim.py",
        "src/train/objective/types.py",
    } <= closure


def test_static_driver_capabilities_reject_unbudgeted_selected_pixels() -> None:
    capabilities = _driver_capabilities()
    capabilities["selected_pixel_target_access"][  # type: ignore[index]
        "source_budget_enforced_before_allocation"
    ] = False

    with pytest.raises(ValueError, match="sealed direct selected-pixel contract"):
        producer._validate_driver_capabilities(capabilities)


def test_darwin_resource_parsers_use_reclaimable_pages_and_swap_used() -> None:
    vm_stat = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                               100.
Pages active:                             900.
Pages inactive:                           200.
Pages speculative:                         50.
"""
    swap = "total = 4096.00M  used = 1536.50M  free = 2559.50M  (encrypted)"

    assert producer._darwin_available_memory_bytes(vm_stat) == 350 * 16384
    assert producer._darwin_swap_used_bytes(swap) == int(1536.5 * 1024**2)


def test_parent_watchdog_sums_process_group_rss_and_enforces_sampled_rss_cap() -> None:
    ps_output = "  91  100\n  42  512\n  42  256\n"

    assert producer._parse_process_group_rss_bytes(ps_output, 42) == 768 * 1024
    assert (
        producer._worker_watchdog_violation(
            elapsed_seconds=1.0,
            group_rss_bytes=producer.WORKER_PROCESS_GROUP_RSS_LIMIT_BYTES + 1,
        )
        .startswith("worker process-group sampled RSS exceeded")
    )
    assert producer._worker_watchdog_violation(
        elapsed_seconds=producer.WORKER_TIMEOUT_SECONDS + 1.0,
        group_rss_bytes=0,
    ).startswith("worker exceeded the hard wall-time")


def test_parent_watchdog_is_bound_into_the_normalized_trial() -> None:
    receipt = {
        "normalized_trial": {
            "measurement": {"trial_execution_evidence_sha256": "a" * 64}
        }
    }
    watchdog = {
        "returncode": 0,
        "watchdog_completed": True,
    }

    producer._attach_parent_watchdog(receipt, watchdog)

    measurement = receipt["normalized_trial"]["measurement"]
    assert measurement["parent_watchdog"] == watchdog
    assert receipt["parent_watchdog"] == watchdog
    assert measurement["parent_watchdog_evidence_sha256"] == (
        receipt["parent_watchdog_evidence_sha256"]
    )


def test_resource_guard_rejects_memory_pressure_and_swap_pressure() -> None:
    failures = producer._resource_guard_failures(
        {
            "platform": "darwin",
            "free_disk_bytes": 20 * producer.GIB,
            "available_memory_bytes": producer.GIB,
            "swap_used_bytes": 3 * producer.GIB,
            "load_average_1m": 1.0,
        },
        minimum_free_disk_bytes=8 * producer.GIB,
        minimum_available_memory_bytes=8 * producer.GIB,
        maximum_swap_used_bytes=2 * producer.GIB,
        maximum_load_average=8.0,
    )

    assert failures == ("available_memory_bytes", "swap_used_bytes")


def test_resource_policy_cannot_relax_the_incident_guard() -> None:
    with pytest.raises(ValueError, match="available-memory"):
        producer._validate_resource_policy(
            minimum_free_disk_bytes=producer.DEFAULT_MINIMUM_FREE_DISK_BYTES,
            minimum_available_memory_bytes=(
                producer.DEFAULT_MINIMUM_AVAILABLE_MEMORY_BYTES - 1
            ),
            maximum_swap_used_bytes=producer.DEFAULT_MAXIMUM_SWAP_USED_BYTES,
            maximum_load_average=producer.DEFAULT_MAXIMUM_LOAD_AVERAGE,
        )


def test_saved_tensor_audit_counts_packed_tensor_payload() -> None:
    audit = producer._SavedTensorAudit()
    tensor = SimpleNamespace(numel=lambda: 12, element_size=lambda: 4)

    assert audit.pack(tensor) is tensor
    assert audit.count == 1
    assert audit.packed_tensor_bytes == 48
    assert audit.unpack(tensor) is tensor


def test_driver_cannot_spoof_producer_owned_saved_tensor_measurements() -> None:
    with pytest.raises(ValueError, match="autograd_saved_tensor_count"):
        producer._reject_producer_owned_measurement_claims(
            {"autograd_saved_tensor_count": 0}
        )


def test_mps_memory_fraction_is_capped_and_applied() -> None:
    recorded: list[float] = []
    torch = SimpleNamespace(
        mps=SimpleNamespace(
            set_per_process_memory_fraction=recorded.append,
            recommended_max_memory=lambda: 4 * producer.GIB,
        )
    )

    receipt = producer._configure_mps_memory_limit(torch, 0.35)
    assert recorded == [0.35]
    assert receipt["absolute_working_set_limit_bytes"] == 2 * producer.GIB
    assert receipt["effective_working_set_limit_bytes"] == int(0.35 * 4 * producer.GIB)

    recorded.clear()
    large_host = SimpleNamespace(
        mps=SimpleNamespace(
            set_per_process_memory_fraction=recorded.append,
            recommended_max_memory=lambda: 16 * producer.GIB,
        )
    )
    receipt = producer._configure_mps_memory_limit(large_host, 0.35)
    assert recorded == [0.125]
    assert receipt["effective_working_set_limit_bytes"] == 2 * producer.GIB

    with pytest.raises(ValueError, match="no greater than"):
        producer._configure_mps_memory_limit(torch, 0.75)


def test_mps_sampler_labels_sampled_growth_as_nonexact() -> None:
    current = iter((128, 192, 160))
    driver = iter((256, 384, 320))
    torch = SimpleNamespace(
        mps=SimpleNamespace(
            current_allocated_memory=lambda: next(current),
            driver_allocated_memory=lambda: next(driver),
        )
    )
    sampler = producer._MpsMemorySampler(torch)
    sampler._sample()
    sampler.baseline_current_allocated_bytes = 128
    sampler.baseline_driver_allocated_bytes = 256
    sampler._sample()

    receipt = sampler.receipt()
    assert receipt["current_allocated_sampled_growth_bytes"] == 64
    assert receipt["driver_allocated_sampled_growth_bytes"] == 128
    assert receipt["exact_peak_claimed"] is False


def test_execute_remains_explicitly_opt_in() -> None:
    args = producer._parser().parse_args(
        [
            "--backend",
            "mps",
            "--trial-driver",
            "driver.py",
            "--trial-config",
            "config.json",
        ]
    )

    assert args.execute is False
    assert args.mps_memory_fraction == producer.DEFAULT_MPS_MEMORY_FRACTION
