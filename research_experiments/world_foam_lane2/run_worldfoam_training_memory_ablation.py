#!/usr/bin/env python3
"""Produce the 21-row WorldFoam training-memory ablation.

Every primary/control row runs in a fresh child.  Fused F=8 rows additionally
launch a fresh restart child.  The parent reuses the incident-calibrated v3
Darwin host guard, 4 GiB process-group watchdog, 2 GiB MPS ceiling, and public
MPS counter sampler.  Driver rows cannot claim those producer-owned fields.

Execution remains fail-closed until the production lazy full-geometry core,
combined-state bridge, and compiled ABI export exact capability seals.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import json
import math
import os
import secrets
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import run_worldfoam_memory_scaling_acceptance as shared
import verify_worldfoam_training_memory_ablation as verifier


ROOT = Path(__file__).resolve().parents[2]
PRODUCER_PATH = Path(__file__).resolve()
DEFAULT_CONFIG = verifier.DEFAULT_CONFIG
DEFAULT_CONTRACT = verifier.DEFAULT_CONTRACT
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "worldfoam_training_memory_ablation"
    / "worldfoam_training_memory_ablation.json"
)
DEFAULT_DRIVER = Path(__file__).with_name(
    "worldfoam_training_memory_spatial_native_driver.py"
)
DEFAULT_DRIVER_MODULE = "worldfoam_training_memory_spatial_native_driver"
DEFAULT_NATIVE_OPS_MODULE = shared.DEFAULT_NATIVE_OPS_MODULE
DRIVER_PROTOCOL = "worldfoam-training-memory-spatial-native-driver-v1"
DRIVER_FUNCTION = "run_worldfoam_training_memory_worker"
DRIVER_CAPABILITY_CONSTANT = "WORLDFOAM_TRAINING_MEMORY_DRIVER_CAPABILITIES"
DRIVER_CAPABILITY_SCHEMA_VERSION = 1
WORKER_NONCE_ENV = "DYNAWORLD_WORLDFOAM_TRAINING_MEMORY_WORKER_NONCE"
COMMAND_PROTOCOL = "argv-no-shell+nonce-bound-receipt-v1"
EXECUTION_IMPLEMENTED = True

HOST_GUARD = shared._guard_host
RESOURCE_POLICY_VALIDATOR = shared._validate_resource_policy
MPS_MEMORY_SAMPLER = shared._MpsMemorySampler
PARENT_WATCHDOG = shared._run_guarded_worker

REQUIRED_DRIVER_CAPABILITIES = {
    "real_native_spatial_block_coordinator": True,
    "full_geometry_trainable": True,
    "all_competitor_active_owner_certification": True,
    "post_certification_compact_device_lowering": True,
    "production_device_material_gradient_receipt": True,
    "production_geometry_device_to_host_reduction_receipt": True,
    "geometry_optimizer_authorization_receipt": True,
    "cpu_manual_sgd_mutation": True,
    "checkpoint_restart_lifecycle": True,
    "staged_sparse_mode": True,
    "fused_union_v2_mode": True,
    "fresh_process_measurements": True,
    "direct_selected_pixel_target_stream": True,
    "zero_full_frame_target_materialization": True,
}

# The driver manifest describes the complete worker/producer stack.  The core
# source seal must not claim producer-owned fresh-process or allocator evidence.
REQUIRED_CORE_CAPABILITIES = {
    key: value
    for key, value in REQUIRED_DRIVER_CAPABILITIES.items()
    if key != "fresh_process_measurements"
}

SOURCE_ROOTS = (
    PRODUCER_PATH,
    Path(verifier.__file__).resolve(),
    Path(shared.__file__).resolve(),
    ROOT / "src/train/worldfoam_training_memory_ablation_adapter.py",
    ROOT / "src/train/paper_kinetic_active_track_program_factory.py",
    ROOT / "src/train/paper_kinetic_compiled_framewise_full_geometry_control.py",
    ROOT / "src/train/paper_kinetic_fixed_site_material_device_bridge.py",
    ROOT / "src/train/paper_kinetic_fixed_site_material_state.py",
    ROOT / "src/train/paper_kinetic_fixed_camera_combined_state.py",
    ROOT / "src/train/paper_kinetic_fixed_camera_full_geometry_step.py",
    ROOT / "src/train/paper_kinetic_lazy_full_geometry_device_bridge.py",
    ROOT / "src/train/paper_kinetic_lazy_full_geometry_step.py",
    ROOT / "src/train/paper_kinetic_lazy_program_bundles.py",
    ROOT / "src/train/paper_kinetic_union_local_bar_assembly.py",
    ROOT / "src/train/paper_kinetic_world_initializer.py",
    ROOT / "research_experiments/world_foam_lane2/kinetic_active_owner_chart_compiler.py",
    ROOT / "research_experiments/world_foam_lane2/kinetic_compiled_cpu_artifact_store.py",
    ROOT / "research_experiments/world_foam_lane2/kinetic_lazy_native_material_step.py",
    ROOT / "research_experiments/world_foam_lane2/kinetic_native_material_step_executor.py",
    ROOT / "research_experiments/world_foam_lane2/kinetic_native_equal_rank_runtime_adapter.py",
    ROOT / "research_experiments/world_foam_lane2/kinetic_power_word_compiler.py",
)

PRODUCER_OWNED_MEMORY_KEYS = frozenset(
    {
        "process_rss_baseline_bytes",
        "process_rss_peak_bytes",
        "sampled_mps_current_baseline_bytes",
        "sampled_mps_current_peak_bytes",
        "sampled_mps_driver_baseline_bytes",
        "sampled_mps_driver_peak_bytes",
        "bridge_process_rss_sampled_peak_bytes",
        "bridge_mps_current_sampled_peak_bytes",
    }
)


def _resolved_file(path: Path, *, name: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} does not exist: {resolved}")
    return resolved


def _python_executable(path: Path) -> Path:
    """Validate Python without resolving a virtual-environment symlink.

    Resolving ``.venv/bin/python`` to the Homebrew framework binary drops the
    virtual-environment prefix and therefore its Torch installation.  Worker
    commands must preserve the user-selected executable path verbatim.
    """

    expanded = Path(path).expanduser()
    absolute = expanded if expanded.is_absolute() else Path.cwd() / expanded
    absolute = absolute.absolute()
    if not absolute.is_file() or not os.access(absolute, os.X_OK):
        raise FileNotFoundError(f"Python executable does not exist: {absolute}")
    return absolute


def _repo_file(path: Path, *, name: str) -> Path:
    resolved = _resolved_file(path, name=name)
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"{name} must live under the dynaworld root") from exc
    return resolved


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _sha256(value: Any) -> str:
    return verifier.canonical_sha256(value)


def _driver_path_from_args(args: argparse.Namespace) -> Path:
    value = getattr(args, "trial_driver", None)
    if value is not None:
        return Path(value)
    module_name = getattr(args, "driver_module", DEFAULT_DRIVER_MODULE)
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        return DEFAULT_DRIVER
    return Path(spec.origin)


def _literal_driver_capabilities(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id == DRIVER_CAPABILITY_CONSTANT
            for target in node.targets
        ):
            values.append(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == DRIVER_CAPABILITY_CONSTANT
            and node.value is not None
        ):
            values.append(node.value)
    if len(values) != 1:
        raise ValueError(
            f"driver must define exactly one literal {DRIVER_CAPABILITY_CONSTANT}"
        )
    value = ast.literal_eval(values[0])
    return dict(_require_mapping(value, name="driver capability manifest"))


def _validate_driver_capabilities(capabilities: Mapping[str, Any]) -> None:
    if capabilities.get("schema_version") != DRIVER_CAPABILITY_SCHEMA_VERSION:
        raise ValueError("driver capability schema changed")
    if capabilities.get("driver_protocol") != DRIVER_PROTOCOL:
        raise ValueError("driver protocol changed")
    if capabilities.get("driver_function") != DRIVER_FUNCTION:
        raise ValueError("driver function changed")
    if capabilities.get("supported_backends") != ["mps"]:
        raise ValueError("driver backend list changed")
    if capabilities.get("supported_worker_kinds") != [
        "primary",
        "control",
        "restart",
    ]:
        raise ValueError("driver worker-kind list changed")
    required = capabilities.get("required_runtime_capabilities")
    if required != sorted(REQUIRED_DRIVER_CAPABILITIES):
        raise ValueError("driver runtime-capability request changed")
    for key in (
        "production_core_module",
        "production_core_callable",
        "production_adapter_status",
        "sequential_control_adapter_status",
        "compiled_framewise_control_provenance",
        "required_core_capability_seal",
    ):
        if not isinstance(capabilities.get(key), str) or not capabilities[key]:
            raise ValueError(f"driver capability {key} is missing")


def _driver_blockers(path: Path) -> tuple[str, ...]:
    blockers: list[str] = []
    try:
        driver = _repo_file(path, name="training-memory driver")
        capabilities = _literal_driver_capabilities(driver)
        _validate_driver_capabilities(capabilities)
    except (OSError, SyntaxError, TypeError, ValueError) as exc:
        return (f"invalid_training_memory_driver:{type(exc).__name__}",)
    if capabilities["production_adapter_status"] != "source_complete":
        blockers.append("production_full_geometry_adapter_not_source_complete")
    if capabilities["sequential_control_adapter_status"] != "source_complete":
        blockers.append(
            "compiled_framewise_full_geometry_control_not_source_complete"
        )
    if capabilities["compiled_framewise_control_provenance"] != (
        "paper-kinetic-compiled-framewise-full-geometry-control-v1"
    ):
        blockers.append("compiled_framewise_control_provenance_changed")
    core_path = ROOT / "src/train" / (
        capabilities["production_core_module"].replace(".", "/") + ".py"
    )
    if not core_path.is_file():
        blockers.append("production_lazy_full_geometry_core_missing")
    native_root = shared.NATIVE_SOURCE_ROOT
    native_ops_source = (
        native_root / "torch_world_foam_lane2_fused_slab" / "ops.py"
    )
    if (
        not native_ops_source.is_file()
        or "def assert_kinetic_lazy_full_geometry_compiled_abi_registered"
        not in native_ops_source.read_text(encoding="utf-8")
    ):
        blockers.append("native_lazy_full_geometry_abi_attestation_missing")
    extension_candidates = tuple(
        sorted(
            (native_root / "torch_world_foam_lane2_fused_slab").glob("_C*.so")
        )
    )
    if len(extension_candidates) != 1:
        blockers.append("native_extension_missing_or_ambiguous")
    else:
        newest_source_mtime_ns = max(
            source.stat().st_mtime_ns for source in shared._native_source_files()
        )
        if extension_candidates[0].stat().st_mtime_ns < newest_source_mtime_ns:
            blockers.append("native_extension_older_than_bound_native_sources")
    if not EXECUTION_IMPLEMENTED:
        blockers.append("fresh_process_training_ablation_worker_not_implemented")
    return tuple(blockers)


def build_source_manifest(
    *,
    config_path: Path = DEFAULT_CONFIG,
    contract_path: Path = DEFAULT_CONTRACT,
    driver_path: Path = DEFAULT_DRIVER,
) -> tuple[tuple[dict[str, Any], ...], str, str]:
    config = _repo_file(config_path, name="training-memory config")
    contract = _repo_file(contract_path, name="training-memory contract")
    driver = _repo_file(driver_path, name="training-memory driver")
    capabilities = _literal_driver_capabilities(driver)
    roots = [*SOURCE_ROOTS, driver]
    core_path = ROOT / "src/train" / (
        str(capabilities["production_core_module"]).replace(".", "/") + ".py"
    )
    if core_path.is_file():
        roots.append(core_path)
    resolved_roots = tuple(_resolved_file(path, name="source root") for path in roots)
    python_closure = shared._local_python_source_closure(
        tuple(path for path in resolved_roots if path.suffix == ".py")
    )
    native_sources = shared._native_source_files()
    paths = sorted(
        {*resolved_roots, *python_closure, *native_sources, config, contract}
    )
    records: list[dict[str, Any]] = []
    native_records: list[dict[str, Any]] = []
    for path in paths:
        resolved = _resolved_file(path, name="source-manifest member")
        label = resolved.relative_to(ROOT).as_posix()
        record = {
            "path": label,
            "size_bytes": resolved.stat().st_size,
            "sha256": verifier.file_sha256(resolved),
        }
        records.append(record)
        if resolved in native_sources:
            native_records.append(record)
    frozen = tuple(records)
    return (
        frozen,
        verifier.source_manifest_sha256(frozen),
        verifier.source_manifest_sha256(tuple(native_records)),
    )


def planned_row_keys(config: Mapping[str, Any]) -> tuple[dict[str, int | str], ...]:
    rows = (
        (
            str(config["ablation"]["staged_mode"]),
            tuple(config["ablation"]["staged_frame_counts"]),
        ),
        (
            str(config["ablation"]["fused_mode"]),
            tuple(config["ablation"]["fused_frame_counts"]),
        ),
    )
    return tuple(
        {
            "mode": mode,
            "requested_frame_count": int(frame_count),
            "repeat_index": repeat_index,
        }
        for mode, frame_counts in rows
        for frame_count in frame_counts
        for repeat_index in range(int(config["ablation"]["repeat_count"]))
    )


def planned_control_row_keys(
    config: Mapping[str, Any],
) -> tuple[dict[str, int | str], ...]:
    return tuple(
        {
            "mode": str(config["ablation"]["control_mode"]),
            "requested_frame_count": int(frame_count),
            "repeat_index": repeat_index,
        }
        for frame_count in config["ablation"]["control_frame_counts"]
        for repeat_index in range(int(config["ablation"]["repeat_count"]))
    )


def make_plan(args: argparse.Namespace) -> dict[str, Any]:
    RESOURCE_POLICY_VALIDATOR(
        minimum_free_disk_bytes=args.minimum_free_disk_bytes,
        minimum_available_memory_bytes=args.minimum_available_memory_bytes,
        maximum_swap_used_bytes=args.maximum_swap_used_bytes,
        maximum_load_average=args.maximum_load_average,
    )
    config_path = _repo_file(args.config, name="training-memory config")
    contract_path = _repo_file(args.contract, name="training-memory contract")
    driver_path = _driver_path_from_args(args)
    config = verifier.load_json_object(config_path)
    contract = verifier.load_json_object(contract_path)
    verifier.validate_config(config)
    verifier.validate_contract(contract)
    manifest, manifest_sha256, native_source_sha256 = build_source_manifest(
        config_path=config_path,
        contract_path=contract_path,
        driver_path=driver_path,
    )
    blockers = _driver_blockers(driver_path)
    rows = planned_row_keys(config)
    controls = planned_control_row_keys(config)
    restart_count = int(config["ablation"]["repeat_count"])
    return {
        "status": "blocked" if blockers else "planned",
        "execution_ready": not blockers,
        "blocking_reasons": list(blockers),
        "benchmark": verifier.BENCHMARK,
        "config_id": config["config_id"],
        "backend": config["backend"],
        "execution_scope": config["execution_scope"],
        "driver_protocol": DRIVER_PROTOCOL,
        "driver_path": str(driver_path.resolve()),
        "planned_rows": list(rows),
        "planned_row_count": len(rows),
        "planned_control_rows": list(controls),
        "planned_control_row_count": len(controls),
        "planned_restart_process_count": restart_count,
        "planned_total_process_count": len(rows) + len(controls) + restart_count,
        "evidence_rows_emitted": 0,
        "control_evidence_rows_emitted": 0,
        "artifact_written": False,
        "output": str(Path(args.output).resolve()),
        "config_sha256": verifier.file_sha256(config_path),
        "contract_sha256": verifier.file_sha256(contract_path),
        "source_manifest_sha256": manifest_sha256,
        "native_source_sha256": native_source_sha256,
        "source_manifest_file_count": len(manifest),
        "fresh_process_per_row_required": True,
        "fresh_process_restart_required": True,
        "mechanical_two_site_fixture_reused": False,
        "fixed_track_count": config["track_manifest"]["track_count"],
        "all_competitor_active_owner_certification_required": True,
        "heuristic_spatial_culling_permitted": False,
        "full_frame_target_materialization_permitted": False,
        "sequential_same_representation_controls_measured": [8, 64, 300],
        "control_censorship_permitted": False,
        "dry_run_is_evidence": False,
        "paper_claim_permitted": False,
        "host_safety_implementation": {
            "guard": "run_worldfoam_memory_scaling_acceptance._guard_host",
            "mps_sampler": "run_worldfoam_memory_scaling_acceptance._MpsMemorySampler",
            "parent_watchdog": "run_worldfoam_memory_scaling_acceptance._run_guarded_worker",
            "maximum_mps_working_set_bytes": shared.MAXIMUM_MPS_WORKING_SET_BYTES,
            "worker_process_group_rss_limit_bytes": shared.WORKER_PROCESS_GROUP_RSS_LIMIT_BYTES,
        },
    }


def _attest_core(
    capabilities: Mapping[str, Any],
) -> tuple[Any, Mapping[str, Any], str]:
    module = importlib.import_module(str(capabilities["production_core_module"]))
    core_callable = getattr(module, str(capabilities["production_core_callable"]), None)
    if not callable(core_callable):
        raise TypeError("production lazy full-geometry callable is missing")
    seal = getattr(module, "PAPER_KINETIC_LAZY_FULL_GEOMETRY_CAPABILITY_SEAL", None)
    seal = _require_mapping(seal, name="lazy full-geometry capability seal")
    if seal.get("seal_id") != capabilities["required_core_capability_seal"]:
        raise ValueError("lazy full-geometry capability seal id changed")
    for key in REQUIRED_CORE_CAPABILITIES:
        if seal.get(key) is not True:
            raise ValueError(f"lazy full-geometry core did not seal capability {key}")
    for producer_owned_key in (
        "fresh_process_measurements",
        "native_execution_measured",
        "allocator_peak_measured",
        "process_rss_measured",
    ):
        if producer_owned_key in seal:
            raise ValueError(
                "lazy full-geometry source seal claimed producer-owned capability "
                f"{producer_owned_key}"
            )
    return module, seal, _sha256(seal)


def _install_runtime_paths() -> None:
    for path in (
        ROOT / "src" / "train",
        ROOT / "research_experiments" / "world_foam_lane2",
        shared.NATIVE_SOURCE_ROOT,
    ):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)


def _load_driver(path: Path) -> Any:
    module = shared._load_driver(path)
    driver = getattr(module, DRIVER_FUNCTION, None)
    if not callable(driver):
        raise TypeError(f"driver must expose callable {DRIVER_FUNCTION}")
    return driver


def _worker_bindings(
    *,
    config_path: Path,
    manifest_sha256: str,
    native_source_sha256: str,
    native_extension_sha256: str,
    hardware_sha256: str,
) -> dict[str, str]:
    return {
        "config_sha256": verifier.file_sha256(config_path),
        "source_manifest_sha256": manifest_sha256,
        "native_source_sha256": native_source_sha256,
        "native_extension_sha256": native_extension_sha256,
        "hardware_fingerprint_sha256": hardware_sha256,
    }


def _measured_row_from_driver(
    raw: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    process_generation_id: str,
    process_rss_baseline: int,
    process_rss_peak: int,
    sampled_mps: Mapping[str, Any],
    bindings: Mapping[str, str],
) -> dict[str, Any]:
    row = dict(_require_mapping(raw.get("row"), name="driver row"))
    for section in ("execution", "structure", "work", "memory", "quality"):
        if not isinstance(row.get(section), Mapping):
            raise TypeError(f"driver row.{section} must be a mapping")
        row[section] = dict(row[section])
    claimed = PRODUCER_OWNED_MEMORY_KEYS.intersection(row["memory"])
    if claimed:
        raise ValueError(
            "driver claimed producer-owned memory fields: " + ", ".join(sorted(claimed))
        )
    if "measurement" in row:
        raise ValueError("driver may not claim producer-owned measurement fields")
    row.update(
        {
            "mode": args.mode,
            "requested_frame_count": args.frame_count,
            "repeat_index": args.repeat_index,
        }
    )
    row["memory"].update(
        {
            "process_rss_baseline_bytes": process_rss_baseline,
            "process_rss_peak_bytes": process_rss_peak,
            "sampled_mps_current_baseline_bytes": sampled_mps[
                "baseline_current_allocated_bytes"
            ],
            "sampled_mps_current_peak_bytes": sampled_mps[
                "maximum_current_allocated_bytes"
            ],
            "sampled_mps_driver_baseline_bytes": sampled_mps[
                "baseline_driver_allocated_bytes"
            ],
            "sampled_mps_driver_peak_bytes": sampled_mps[
                "maximum_driver_allocated_bytes"
            ],
        }
    )
    row["measurement"] = {
        "fresh_process": True,
        "measurement_kind": "fresh-process-mps-and-rss-sampled-high-water-v1",
        "completion_fenced_before_final_measurement": True,
        "allocator_exact_peak_claimed": False,
        "mps_memory_limit_bytes": shared.MAXIMUM_MPS_WORKING_SET_BYTES,
        "process_group_rss_limit_bytes": shared.WORKER_PROCESS_GROUP_RSS_LIMIT_BYTES,
        "mps_memory_sample_count": sampled_mps["sample_count"],
        "process_generation_id": process_generation_id,
        "bindings": dict(bindings),
    }
    row["lifecycle"] = None
    return row


def _worker(args: argparse.Namespace) -> int:
    nonce = str(args.worker_nonce)
    if not nonce or os.environ.get(WORKER_NONCE_ENV) != nonce:
        raise PermissionError("worker nonce is missing or foreign")
    driver_path = _repo_file(args.trial_driver, name="training-memory driver")
    config_path = _repo_file(args.config, name="training-memory config")
    contract_path = _repo_file(args.contract, name="training-memory contract")
    receipt_path = Path(args.receipt).resolve()
    capabilities = _literal_driver_capabilities(driver_path)
    _validate_driver_capabilities(capabilities)
    if capabilities["production_adapter_status"] != "source_complete":
        raise RuntimeError("production full-geometry adapter is not source-complete")
    if (
        args.worker_kind == "control"
        and capabilities["sequential_control_adapter_status"] != "source_complete"
    ):
        raise RuntimeError("compiled-framewise control adapter is not source-complete")
    manifest, manifest_sha256, native_source_sha256 = build_source_manifest(
        config_path=config_path,
        contract_path=contract_path,
        driver_path=driver_path,
    )
    if manifest_sha256 != args.expected_source_manifest_sha256:
        raise RuntimeError("child source manifest differs from parent launch")
    command_sha256 = shared._nonce_bound_command_sha256((sys.executable, *sys.argv))
    if command_sha256 != args.expected_command_sha256:
        raise RuntimeError("worker argv differs from nonce-bound parent command")
    host_snapshot = HOST_GUARD(
        minimum_free_disk_bytes=args.minimum_free_disk_bytes,
        minimum_available_memory_bytes=args.minimum_available_memory_bytes,
        maximum_swap_used_bytes=args.maximum_swap_used_bytes,
        maximum_load_average=args.maximum_load_average,
    )
    process_rss_baseline = shared._maximum_rss_bytes()
    hardware = shared._hardware_record(
        args.backend, mps_memory_fraction=args.mps_memory_fraction
    )
    torch = importlib.import_module("torch")
    _install_runtime_paths()
    native_ops, native_extension = shared._attest_native_extension(
        args.native_ops_module
    )
    native_v2_attest = getattr(
        native_ops,
        "assert_kinetic_lazy_full_geometry_compiled_abi_registered",
        None,
    )
    if not callable(native_v2_attest):
        raise TypeError("compiled native module lacks lazy full-geometry ABI attestation")
    native_v2_attest()
    _, core_seal, core_seal_sha256 = _attest_core(capabilities)
    driver = _load_driver(driver_path)
    config = verifier.load_json_object(config_path)
    verifier.validate_config(config)
    native_extension_sha256 = verifier.file_sha256(native_extension)
    hardware_sha256 = _sha256(hardware)
    bindings = _worker_bindings(
        config_path=config_path,
        manifest_sha256=manifest_sha256,
        native_source_sha256=native_source_sha256,
        native_extension_sha256=native_extension_sha256,
        hardware_sha256=hardware_sha256,
    )
    process_generation_id = _sha256(
        {
            "nonce": nonce,
            "pid": os.getpid(),
            "start_time_ns": args.worker_start_time_ns,
            "command_sha256": command_sha256,
        }
    )
    context: dict[str, Any] = {
        "backend": args.backend,
        "worker_kind": args.worker_kind,
        "mode": args.mode,
        "frame_count": args.frame_count,
        "repeat_index": args.repeat_index,
        "config": config,
        "native_ops": native_ops,
        "native_ops_module": args.native_ops_module,
        "native_extension_path": str(native_extension),
        "native_extension_sha256": native_extension_sha256,
        "require_real_native": True,
        "core_capability_seal": core_seal,
        "core_capability_seal_sha256": core_seal_sha256,
        "checkpoint_path": str(args.checkpoint_path) if args.checkpoint_path else None,
        "expected_checkpoint_sha256": args.expected_checkpoint_sha256,
        "worker_output_dir": str(receipt_path.parent),
    }
    saved_audit = shared._SavedTensorAudit()
    hooks = getattr(getattr(torch.autograd, "graph", None), "saved_tensors_hooks", None)
    if not callable(hooks):
        raise RuntimeError("PyTorch saved-tensor audit is unavailable")
    with MPS_MEMORY_SAMPLER(torch) as sampler:
        with hooks(saved_audit.pack, saved_audit.unpack):
            raw = _require_mapping(driver(context), name="driver result")
        torch.mps.synchronize()
        sampler.capture_after_completion_fence()
    sampled_mps = sampler.receipt()
    process_rss_peak = shared._maximum_rss_bytes()
    if raw.get("native_ops_identity_verified") is not True:
        raise ValueError("driver did not verify native_ops identity")
    if saved_audit.count != 0 or saved_audit.packed_tensor_bytes != 0:
        raise RuntimeError("manual full-geometry worker unexpectedly retained autograd tensors")
    receipt: dict[str, Any] = {
        "nonce": nonce,
        "worker_kind": args.worker_kind,
        "mode": args.mode,
        "frame_count": args.frame_count,
        "repeat_index": args.repeat_index,
        "command_sha256": command_sha256,
        "process_generation_id": process_generation_id,
        "source_manifest_sha256": manifest_sha256,
        "native_source_sha256": native_source_sha256,
        "driver_sha256": verifier.file_sha256(driver_path),
        "core_capability_seal_sha256": core_seal_sha256,
        "native_extension_path": str(native_extension),
        "bindings": bindings,
        "hardware": hardware,
        "host_resource_preflight": host_snapshot,
        "mps_sampled_memory": sampled_mps,
    }
    if args.worker_kind == "restart":
        receipt["restart_result"] = dict(
            _require_mapping(raw.get("restart_result"), name="restart result")
        )
    else:
        receipt["row"] = _measured_row_from_driver(
            raw,
            args=args,
            process_generation_id=process_generation_id,
            process_rss_baseline=process_rss_baseline,
            process_rss_peak=process_rss_peak,
            sampled_mps=sampled_mps,
            bindings=bindings,
        )
        for key in ("parity_payload",):
            if key in raw:
                receipt[key] = raw[key]
    shared._write_json_atomic(receipt_path, receipt)
    return 0


def _worker_command(
    args: argparse.Namespace,
    *,
    worker_kind: str,
    mode: str,
    frame_count: int,
    repeat_index: int,
    nonce: str,
    start_time_ns: int,
    receipt_path: Path,
    manifest_sha256: str,
    checkpoint_path: Path | None = None,
    checkpoint_sha256: str = "",
) -> tuple[list[str], str]:
    command = [
        str(_python_executable(args.python)),
        str(PRODUCER_PATH),
        "--worker",
        "--worker-nonce",
        nonce,
        "--worker-start-time-ns",
        str(start_time_ns),
        "--worker-kind",
        worker_kind,
        "--mode",
        mode,
        "--frame-count",
        str(frame_count),
        "--repeat-index",
        str(repeat_index),
        "--backend",
        args.backend,
        "--config",
        str(Path(args.config).resolve()),
        "--contract",
        str(Path(args.contract).resolve()),
        "--trial-driver",
        str(_driver_path_from_args(args).resolve()),
        "--native-ops-module",
        args.native_ops_module,
        "--mps-memory-fraction",
        str(args.mps_memory_fraction),
        "--minimum-free-disk-bytes",
        str(args.minimum_free_disk_bytes),
        "--minimum-available-memory-bytes",
        str(args.minimum_available_memory_bytes),
        "--maximum-swap-used-bytes",
        str(args.maximum_swap_used_bytes),
        "--maximum-load-average",
        str(args.maximum_load_average),
        "--receipt",
        str(receipt_path),
        "--expected-source-manifest-sha256",
        manifest_sha256,
    ]
    if checkpoint_path is not None:
        command.extend(
            [
                "--checkpoint-path",
                str(checkpoint_path),
                "--expected-checkpoint-sha256",
                checkpoint_sha256,
            ]
        )
    command.extend(["--expected-command-sha256", "PLACEHOLDER"])
    digest = shared._nonce_bound_command_sha256(command)
    command[-1] = digest
    return command, digest


def _launch_worker(
    args: argparse.Namespace,
    *,
    trial_root: Path,
    worker_kind: str,
    mode: str,
    frame_count: int,
    repeat_index: int,
    manifest_sha256: str,
    checkpoint_path: Path | None = None,
    checkpoint_sha256: str = "",
) -> dict[str, Any]:
    HOST_GUARD(
        minimum_free_disk_bytes=args.minimum_free_disk_bytes,
        minimum_available_memory_bytes=args.minimum_available_memory_bytes,
        maximum_swap_used_bytes=args.maximum_swap_used_bytes,
        maximum_load_average=args.maximum_load_average,
    )
    nonce = secrets.token_hex(32)
    stem = f"{worker_kind}_{mode}_f{frame_count}_r{repeat_index}"
    receipt_path = trial_root / f"{stem}.json"
    stdout_path = trial_root / f"{stem}.stdout.log"
    stderr_path = trial_root / f"{stem}.stderr.log"
    command, command_sha256 = _worker_command(
        args,
        worker_kind=worker_kind,
        mode=mode,
        frame_count=frame_count,
        repeat_index=repeat_index,
        nonce=nonce,
        start_time_ns=time.time_ns(),
        receipt_path=receipt_path,
        manifest_sha256=manifest_sha256,
        checkpoint_path=checkpoint_path,
        checkpoint_sha256=checkpoint_sha256,
    )
    environment = os.environ.copy()
    environment[WORKER_NONCE_ENV] = nonce
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        watchdog = PARENT_WATCHDOG(
            command,
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
        )
    if watchdog["returncode"] != 0:
        raise RuntimeError(f"{stem} failed; see {stderr_path}")
    receipt = verifier.load_json_object(receipt_path)
    if (
        receipt.get("nonce") != nonce
        or receipt.get("worker_kind") != worker_kind
        or receipt.get("mode") != mode
        or receipt.get("frame_count") != frame_count
        or receipt.get("repeat_index") != repeat_index
        or receipt.get("command_sha256") != command_sha256
        or receipt.get("source_manifest_sha256") != manifest_sha256
    ):
        raise ValueError("fresh worker receipt identity changed")
    watchdog_evidence_sha256 = _sha256(
        {
            "parent_watchdog": watchdog,
            "process_generation_id": receipt["process_generation_id"],
            "worker_command_sha256": command_sha256,
        }
    )
    receipt["parent_watchdog"] = watchdog
    receipt["parent_watchdog_evidence_sha256"] = watchdog_evidence_sha256
    if "row" in receipt:
        receipt["row"]["measurement"].update(
            {
                "worker_command_sha256": command_sha256,
                "parent_watchdog": watchdog,
                "parent_watchdog_evidence_sha256": watchdog_evidence_sha256,
            }
        )
        receipt["row"]["memory"][
            "parent_process_group_rss_sampled_peak_bytes"
        ] = watchdog["sampled_process_group_rss_high_water_bytes"]
    return receipt


def _attach_lifecycle(
    row: dict[str, Any],
    primary: Mapping[str, Any],
    restart: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    auxiliary = _require_mapping(
        restart.get("restart_result"), name="auxiliary lifecycle result"
    )
    auxiliary_step_1 = _require_mapping(
        auxiliary.get("auxiliary_step_1"), name="auxiliary lifecycle step1"
    )
    uninterrupted = _require_mapping(
        auxiliary.get("uninterrupted_step_2"), name="uninterrupted step2"
    )
    restored = _require_mapping(
        auxiliary.get("restored_step_2"), name="restored step2"
    )
    primary_parity = _require_mapping(
        primary.get("parity_payload"), name="primary F8 parity payload"
    )
    primary_step_1 = {
        "loss_pre_update": float(primary_parity["loss"]),
        "gradient_sha256": _sha256(
            {
                "loss": primary_parity["loss"],
                "material_gradient": primary_parity["material_gradient"],
                "geometry_gradient": primary_parity["geometry_gradient"],
            }
        ),
        "parameters_after_step_sha256": _sha256(
            primary_parity["parameters_after_step"]
        ),
        "parameter_delta_l2": {
            key: float(row["execution"]["gradient_update"][key])
            for key in (
                "raw_color_parameter_delta_l2_norm",
                "raw_density_parameter_delta_l2_norm",
                "positions0_parameter_delta_l2_norm",
                "velocities_parameter_delta_l2_norm",
                "weight_coefficients_parameter_delta_l2_norm",
            )
        },
    }
    primary_step_1["update_content_sha256"] = _sha256(
        {
            key: primary_step_1[key]
            for key in (
                "loss_pre_update",
                "gradient_sha256",
                "parameters_after_step_sha256",
                "parameter_delta_l2",
            )
        }
    )
    for key, value in primary_step_1.items():
        if auxiliary_step_1.get(key) != value:
            raise ArithmeticError(
                f"auxiliary lifecycle step1 {key} differs from primary scaling row"
            )
    for key in (
        "loss_pre_update",
        "gradient_sha256",
        "parameters_after_step_sha256",
        "state_sha256",
        "update_content_sha256",
        "parameter_delta_l2",
    ):
        if uninterrupted.get(key) != restored.get(key):
            raise ArithmeticError(
                f"restored step2 {key} differs from uninterrupted step2"
            )
    uninterrupted_loss = float(uninterrupted["loss_pre_update"])
    restored_loss = float(restored["loss_pre_update"])
    row["lifecycle"] = {
        "performed": True,
        "step_count": 2,
        "checkpoint_created_after_step": 1,
        "checkpoint_restore_used": True,
        "restart_fresh_process": True,
        "restart_process_generation_id": restart["process_generation_id"],
        "restart_hardware_fingerprint_sha256": restart["bindings"][
            "hardware_fingerprint_sha256"
        ],
        "restart_source_manifest_sha256": restart["bindings"][
            "source_manifest_sha256"
        ],
        "restart_native_source_sha256": restart["bindings"][
            "native_source_sha256"
        ],
        "restart_native_extension_sha256": restart["bindings"][
            "native_extension_sha256"
        ],
        "restart_worker_command_sha256": restart["command_sha256"],
        "restart_parent_watchdog": restart["parent_watchdog"],
        "restart_parent_watchdog_evidence_sha256": restart[
            "parent_watchdog_evidence_sha256"
        ],
        "primary_scaling_worker_step_count": 1,
        "primary_scaling_worker_checkpoint_count": 0,
        "primary_scaling_worker_measurement_excludes_auxiliary_lifecycle": True,
        "auxiliary_lifecycle_worker": True,
        "auxiliary_step_1_matches_primary_scaling_row": True,
        "loss_step_1_pre_update": primary_step_1["loss_pre_update"],
        "loss_step_1_pre_update_auxiliary": auxiliary_step_1["loss_pre_update"],
        "loss_step_2_uninterrupted_pre_update": uninterrupted_loss,
        "loss_step_2_restored_pre_update": restored_loss,
        "step_1_to_step_2_pre_update_loss_decrease": (
            float(primary_step_1["loss_pre_update"]) - uninterrupted_loss
        ),
        "restart_loss_absolute_error": abs(uninterrupted_loss - restored_loss),
        "step_1_gradient_sha256_primary": primary_step_1["gradient_sha256"],
        "step_1_gradient_sha256_auxiliary": auxiliary_step_1["gradient_sha256"],
        "step_1_parameters_after_step_sha256_primary": primary_step_1[
            "parameters_after_step_sha256"
        ],
        "step_1_parameters_after_step_sha256_auxiliary": auxiliary_step_1[
            "parameters_after_step_sha256"
        ],
        "step_1_parameter_delta_l2_primary": primary_step_1[
            "parameter_delta_l2"
        ],
        "step_1_parameter_delta_l2_auxiliary": auxiliary_step_1[
            "parameter_delta_l2"
        ],
        "step_1_update_content_sha256_primary": primary_step_1[
            "update_content_sha256"
        ],
        "step_1_update_content_sha256_auxiliary": auxiliary_step_1[
            "update_content_sha256"
        ],
        "step_1_update_receipt_generation_sha256_primary": row["execution"][
            "combined_update_receipt_generation_digest"
        ],
        "step_1_update_receipt_generation_sha256_auxiliary": auxiliary_step_1[
            "update_receipt_sha256"
        ],
        "step_2_gradient_sha256_uninterrupted": uninterrupted[
            "gradient_sha256"
        ],
        "step_2_gradient_sha256_restored": restored["gradient_sha256"],
        "step_2_gradient_content_match": True,
        "step_2_parameters_after_step_sha256_uninterrupted": uninterrupted[
            "parameters_after_step_sha256"
        ],
        "step_2_parameters_after_step_sha256_restored": restored[
            "parameters_after_step_sha256"
        ],
        "step_2_parameter_delta_l2_uninterrupted": uninterrupted[
            "parameter_delta_l2"
        ],
        "step_2_parameter_delta_l2_restored": restored["parameter_delta_l2"],
        "step_2_state_sha256_uninterrupted": uninterrupted["state_sha256"],
        "step_2_state_sha256_restored": restored["state_sha256"],
        "step_2_state_content_match": True,
        "step_2_update_content_sha256_uninterrupted": uninterrupted[
            "update_content_sha256"
        ],
        "step_2_update_content_sha256_restored": restored[
            "update_content_sha256"
        ],
        "step_2_update_content_match": True,
        "uninterrupted_process_optimizer_mutation_count": auxiliary[
            "uninterrupted_process_optimizer_mutation_count"
        ],
        "fresh_restart_optimizer_mutation_count": auxiliary[
            "fresh_restart_optimizer_mutation_count"
        ],
        "auxiliary_optimizer_mutation_count": auxiliary[
            "auxiliary_optimizer_mutation_count"
        ],
        "post_step_1_loss_measured_by_step_2_pre_update": True,
        "measurement_includes_checkpoint_and_uninterrupted_second_step": False,
        "maximum_simultaneously_retained_world_count": auxiliary[
            "maximum_simultaneously_retained_world_count"
        ],
        "uninterrupted_world_released_before_restore": auxiliary[
            "uninterrupted_world_released_before_restore"
        ],
        "restore_receipt": auxiliary["restore_receipt"],
        "checkpoint_sha256": auxiliary["checkpoint_sha256"],
        "combined_checkpoint_payload_bytes": contract[
            "combined_checkpoint_payload_bytes"
        ],
        "live_state_logical_tensor_bytes_at_checkpoint": contract[
            "combined_live_state_bytes"
        ],
        "live_state_plus_checkpoint_bytes": contract[
            "live_state_plus_checkpoint_bytes"
        ],
        "live_state_plus_checkpoint_payload_clone_peak_bytes": contract[
            "live_state_plus_checkpoint_payload_clone_peak_bytes"
        ],
        "optimizer_history_tensor_bytes": 0,
    }


def _relative_l2(left: Sequence[Any], right: Sequence[Any]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("parity vectors must be nonempty and equal length")
    numerator = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))
    denominator = max(
        1.0e-30,
        math.sqrt(sum(float(value) ** 2 for value in left)),
        math.sqrt(sum(float(value) ** 2 for value in right)),
    )
    return numerator / denominator


def _parity_record(
    *,
    repeat_index: int,
    staged_receipt: Mapping[str, Any],
    fused_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    staged = _require_mapping(
        staged_receipt.get("parity_payload"), name="staged parity payload"
    )
    fused = _require_mapping(
        fused_receipt.get("parity_payload"), name="fused parity payload"
    )
    staged_row = staged_receipt["row"]
    fused_row = fused_receipt["row"]
    return {
        "repeat_index": repeat_index,
        "staged_row_evidence_sha256": staged_row["evidence_sha256"],
        "fused_row_evidence_sha256": fused_row["evidence_sha256"],
        "loss_absolute_error": abs(float(staged["loss"]) - float(fused["loss"])),
        "material_gradient_relative_l2": _relative_l2(
            staged["material_gradient"], fused["material_gradient"]
        ),
        "geometry_gradient_relative_l2": _relative_l2(
            staged["geometry_gradient"], fused["geometry_gradient"]
        ),
        "parameter_relative_l2": _relative_l2(
            staged["parameters_after_step"], fused["parameters_after_step"]
        ),
    }


def _fused_compiled_framewise_parity_record(
    *,
    repeat_index: int,
    fused_receipt: Mapping[str, Any],
    control_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    fused = _require_mapping(
        fused_receipt.get("parity_payload"), name="fused parity payload"
    )
    control = _require_mapping(
        control_receipt.get("parity_payload"),
        name="compiled-framewise parity payload",
    )
    fused_row = _require_mapping(fused_receipt.get("row"), name="fused row")
    control_row = _require_mapping(
        control_receipt.get("row"), name="compiled-framewise control row"
    )
    return {
        "repeat_index": repeat_index,
        "fused_row_evidence_sha256": fused_row["evidence_sha256"],
        "compiled_framewise_control_row_evidence_sha256": control_row[
            "evidence_sha256"
        ],
        "loss_absolute_error": abs(float(fused["loss"]) - float(control["loss"])),
        "material_gradient_relative_l2": _relative_l2(
            fused["material_gradient"], control["material_gradient"]
        ),
        "geometry_gradient_relative_l2": _relative_l2(
            fused["geometry_gradient"], control["geometry_gradient"]
        ),
        "parameter_relative_l2": _relative_l2(
            fused["parameters_after_step"], control["parameters_after_step"]
        ),
    }


def _orchestrate(args: argparse.Namespace) -> int:
    plan = make_plan(args)
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if plan["blocking_reasons"]:
        raise RuntimeError(
            "training-memory ablation execution is fail-closed: "
            + ", ".join(plan["blocking_reasons"])
        )
    if not 0.0 < args.mps_memory_fraction <= shared.MAXIMUM_MPS_MEMORY_FRACTION:
        raise ValueError("MPS memory fraction exceeds the incident-calibrated limit")
    HOST_GUARD(
        minimum_free_disk_bytes=args.minimum_free_disk_bytes,
        minimum_available_memory_bytes=args.minimum_available_memory_bytes,
        maximum_swap_used_bytes=args.maximum_swap_used_bytes,
        maximum_load_average=args.maximum_load_average,
    )
    output = Path(args.output).resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output}")
    driver_path = _driver_path_from_args(args)
    manifest, manifest_sha256, native_source_sha256 = build_source_manifest(
        config_path=args.config,
        contract_path=args.contract,
        driver_path=driver_path,
    )
    config = verifier.load_json_object(args.config)
    contract = verifier.load_json_object(args.contract)
    trial_root = output.parent / f".{output.stem}_fresh_workers"
    trial_root.mkdir(parents=True, exist_ok=True)
    primary_receipts: dict[tuple[str, int, int], dict[str, Any]] = {}
    restart_receipts: list[dict[str, Any]] = []
    for key in planned_row_keys(config):
        receipt = _launch_worker(
            args,
            trial_root=trial_root,
            worker_kind="primary",
            mode=str(key["mode"]),
            frame_count=int(key["requested_frame_count"]),
            repeat_index=int(key["repeat_index"]),
            manifest_sha256=manifest_sha256,
        )
        primary_receipts[(str(key["mode"]), int(key["requested_frame_count"]), int(key["repeat_index"]))] = receipt
        if (
            key["mode"] == config["ablation"]["fused_mode"]
            and key["requested_frame_count"] == config["optimizer"]["lifecycle_frame_count"]
        ):
            restarted = _launch_worker(
                args,
                trial_root=trial_root,
                worker_kind="restart",
                mode=str(key["mode"]),
                frame_count=int(key["requested_frame_count"]),
                repeat_index=int(key["repeat_index"]),
                manifest_sha256=manifest_sha256,
            )
            restart_receipts.append(restarted)
            _attach_lifecycle(receipt["row"], receipt, restarted, contract)
    control_receipts: dict[tuple[str, int, int], dict[str, Any]] = {}
    for key in planned_control_row_keys(config):
        receipt = _launch_worker(
            args,
            trial_root=trial_root,
            worker_kind="control",
            mode=str(key["mode"]),
            frame_count=int(key["requested_frame_count"]),
            repeat_index=int(key["repeat_index"]),
            manifest_sha256=manifest_sha256,
        )
        row = receipt["row"]
        row["status"] = "measured"
        control_receipts[(str(key["mode"]), int(key["requested_frame_count"]), int(key["repeat_index"]))] = receipt
    all_receipts = [
        *primary_receipts.values(),
        *control_receipts.values(),
        *restart_receipts,
    ]
    first_bindings = all_receipts[0]["bindings"]
    for receipt in all_receipts:
        if receipt["bindings"] != first_bindings:
            raise ValueError("source/native/hardware bindings drifted across fresh workers")
    bindings = {
        "config_sha256": verifier.file_sha256(args.config),
        "contract_sha256": verifier.file_sha256(args.contract),
        "source_manifest_sha256": manifest_sha256,
        "native_source_sha256": native_source_sha256,
        "native_extension_sha256": first_bindings["native_extension_sha256"],
        "hardware_fingerprint_sha256": first_bindings[
            "hardware_fingerprint_sha256"
        ],
        "producer_sha256": verifier.file_sha256(PRODUCER_PATH),
        "driver_sha256": verifier.file_sha256(driver_path),
    }
    for receipt in (*primary_receipts.values(), *control_receipts.values()):
        receipt["row"]["evidence_sha256"] = verifier.row_evidence_sha256(
            receipt["row"]
        )
    parity = [
        _parity_record(
            repeat_index=repeat_index,
            staged_receipt=primary_receipts[(config["ablation"]["staged_mode"], 8, repeat_index)],
            fused_receipt=primary_receipts[(config["ablation"]["fused_mode"], 8, repeat_index)],
        )
        for repeat_index in range(int(config["ablation"]["repeat_count"]))
    ]
    fused_compiled_framewise_parity = [
        _fused_compiled_framewise_parity_record(
            repeat_index=repeat_index,
            fused_receipt=primary_receipts[
                (config["ablation"]["fused_mode"], 8, repeat_index)
            ],
            control_receipt=control_receipts[
                (config["ablation"]["control_mode"], 8, repeat_index)
            ],
        )
        for repeat_index in range(int(config["ablation"]["repeat_count"]))
    ]
    artifact: dict[str, Any] = {
        "schema_version": verifier.SCHEMA_VERSION,
        "artifact_kind": verifier.ARTIFACT_KIND,
        "benchmark": verifier.BENCHMARK,
        "config_id": config["config_id"],
        "backend": config["backend"],
        "execution_scope": config["execution_scope"],
        "status": "measured",
        "evidence_origin": contract["required_evidence_origin"],
        "proxy_or_test_artifact": False,
        "dataset_is_procedural_synthetic": True,
        "native_execution_measured": True,
        "measurement_is_simulated": False,
        "public_quality_evidence": False,
        "claim_scope": "synthetic_systems_memory_trainability_only",
        "producer_execution_implemented": True,
        "all_rows_fresh_process_completed": True,
        "bindings": bindings,
        "hardware": all_receipts[0]["hardware"],
        "source_manifest": list(manifest),
        "rows": [receipt["row"] for receipt in primary_receipts.values()],
        "control_rows": [receipt["row"] for receipt in control_receipts.values()],
        "staged_fused_f8_parity": parity,
        "fused_compiled_framewise_f8_parity": fused_compiled_framewise_parity,
    }
    report = verifier.verify_artifact_payload(
        artifact,
        config,
        contract,
        config_sha256=bindings["config_sha256"],
        contract_sha256=bindings["contract_sha256"],
    )
    artifact["acceptance_report"] = report
    shared._write_json_atomic(output, artifact)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["accepted"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Produce the paper-scale WorldFoam training-memory ablation."
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backend", choices=("mps",), default="mps")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trial-driver", type=Path, default=DEFAULT_DRIVER)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--native-ops-module", default=DEFAULT_NATIVE_OPS_MODULE)
    parser.add_argument(
        "--minimum-free-disk-bytes",
        type=int,
        default=shared.DEFAULT_MINIMUM_FREE_DISK_BYTES,
    )
    parser.add_argument(
        "--minimum-available-memory-bytes",
        type=int,
        default=shared.DEFAULT_MINIMUM_AVAILABLE_MEMORY_BYTES,
    )
    parser.add_argument(
        "--maximum-swap-used-bytes",
        type=int,
        default=shared.DEFAULT_MAXIMUM_SWAP_USED_BYTES,
    )
    parser.add_argument(
        "--maximum-load-average",
        type=float,
        default=shared.DEFAULT_MAXIMUM_LOAD_AVERAGE,
    )
    parser.add_argument(
        "--mps-memory-fraction",
        type=float,
        default=shared.DEFAULT_MPS_MEMORY_FRACTION,
    )
    parser.add_argument("--worker-nonce", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-start-time-ns", type=int, default=0, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--worker-kind",
        choices=("primary", "control", "restart"),
        default="primary",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--mode", default="", help=argparse.SUPPRESS)
    parser.add_argument("--frame-count", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--repeat-index", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--receipt", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--expected-checkpoint-sha256", default="", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--expected-source-manifest-sha256", default="", help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--expected-command-sha256", default="", help=argparse.SUPPRESS
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.worker:
        if (
            args.receipt is None
            or not args.mode
            or args.frame_count < 1
            or args.repeat_index < 0
        ):
            raise SystemExit("worker arguments are incomplete")
        raise SystemExit(_worker(args))
    raise SystemExit(_orchestrate(args))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_CONTRACT",
    "DEFAULT_DRIVER",
    "DEFAULT_DRIVER_MODULE",
    "DEFAULT_OUTPUT",
    "DRIVER_PROTOCOL",
    "EXECUTION_IMPLEMENTED",
    "HOST_GUARD",
    "MPS_MEMORY_SAMPLER",
    "PARENT_WATCHDOG",
    "RESOURCE_POLICY_VALIDATOR",
    "build_source_manifest",
    "make_plan",
    "planned_control_row_keys",
    "planned_row_keys",
]
