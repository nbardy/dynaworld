#!/usr/bin/env python3
"""Plan the matched selected-ray WorldFoam G4-v2 public-quality matrix.

The default command is a pure-CPU, allocation-light preflight.  It binds the
unchanged G4-v1 scenes, seeds, representations, optimizer schedule, and full
heldout evaluator to a new route-independent selected-pixel schedule.  It also
emits exact compiler-work and target-cost receipts for every scene/seed.

The 36-row producer exists, but execution remains fail closed until the
real-native tractability pilot, fresh native extension, host-resource guard,
and independent artifact verifier all pass.  The v1 all-pixel config and
evidence schema are never mutated or silently reused.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "src" / "train"
LANE2 = Path(__file__).resolve().parent
for import_root in (TRAIN, LANE2):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from config_utils import serialize_config_value  # noqa: E402
from verify_worldfoam_public_quality_ablation import validate_contract  # noqa: E402
from worldfoam_g4_selected_ray_contract import (  # noqa: E402
    DEFAULT_CONFIG,
    REQUIRED_ROUTES,
    REQUIRED_SCENES,
    REQUIRED_SEEDS,
    build_matrix_workload_receipts,
    canonical_sha256,
    file_sha256,
)
from worldfoam_g4_v2_capability import required_source_capability  # noqa: E402
from run_worldfoam_g4_v2_pilot import build_plan as build_pilot_plan  # noqa: E402
from verify_worldfoam_public_quality_ablation_v2 import (  # noqa: E402
    collect_and_verify,
)
from verify_worldfoam_g4_v2_pilot import validate_pilot_receipt  # noqa: E402


PLAN_SCHEMA_VERSION = 1
PLAN_KIND = "worldfoam-native4d-g4-v2-selected-ray-execution-plan-v1"
V2_ROW_KIND = "worldfoam-native4d-public-quality-selected-ray-row-v2"
V2_VERIFIER = LANE2 / "verify_worldfoam_public_quality_ablation_v2.py"
PILOT_KIND = "worldfoam-g4-v2-selected-ray-real-native-pilot-v1"
ROW_WATCHDOG_KIND = "worldfoam-g4-v2-row-process-group-watchdog-v1"
ROW_WATCHDOG_FILENAME = "process_group_watchdog.json"
ROW_STDOUT_FILENAME = "row_worker.stdout.log"
ROW_STDERR_FILENAME = "row_worker.stderr.log"
WORKER_TERMINATION_GRACE_SECONDS = 5.0
DEFAULT_WORKER_PYTHON = ROOT / ".venv" / "bin" / "python"


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def expected_row_path(
    output_root: Path, *, scene: str, seed: int, route: str
) -> Path:
    return output_root / scene / f"seed_{seed}" / route / "g4_v2_row.json"


def expected_watchdog_path(row_path: Path) -> Path:
    return Path(row_path).resolve().parent / ROW_WATCHDOG_FILENAME


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    return {
        "path": _display(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _parse_process_group_rss_bytes(output: str, process_group_id: int) -> int:
    total_kib = 0
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            row_group, rss_kib = (int(value) for value in fields)
        except ValueError:
            continue
        if row_group == process_group_id and rss_kib > 0:
            total_kib += rss_kib
    return total_kib * 1024


def _process_group_rss_bytes(process_group_id: int) -> int:
    try:
        output = subprocess.check_output(
            ("ps", "-axo", "pgid=,rss="),
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise RuntimeError("could not sample row-worker process-group RSS") from error
    return _parse_process_group_rss_bytes(output, process_group_id)


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=WORKER_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=WORKER_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("row-worker process group did not terminate") from error


def _run_guarded_row_process(
    argv: Sequence[str],
    *,
    stdout: Any,
    stderr: Any,
    maximum_process_group_rss_bytes: int,
    poll_interval_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Run one row in a fresh process group under a sampled hard RSS cap."""

    process = subprocess.Popen(
        tuple(str(value) for value in argv),
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )
    started = time.monotonic()
    peak_rss = 0
    sample_count = 0
    leader_exit_seen_at: float | None = None
    try:
        while True:
            returncode = process.poll()
            group_rss = _process_group_rss_bytes(process.pid)
            elapsed = time.monotonic() - started
            sample_count += 1
            peak_rss = max(peak_rss, group_rss)
            if group_rss > maximum_process_group_rss_bytes:
                raise MemoryError(
                    "row-worker process-group sampled RSS exceeded the hard cap: "
                    f"{group_rss} > {maximum_process_group_rss_bytes}"
                )
            if elapsed > timeout_seconds:
                raise TimeoutError(
                    f"row worker exceeded {timeout_seconds:.1f} seconds"
                )
            if returncode is not None:
                if group_rss == 0:
                    return {
                        "returncode": int(returncode),
                        "elapsed_seconds": float(elapsed),
                        "rss_measurement_kind": "parent-ps-sampled-process-group-high-water-v1",
                        "rss_sampling_interval_seconds": float(poll_interval_seconds),
                        "sampled_process_group_rss_high_water_bytes": int(peak_rss),
                        "sample_count": int(sample_count),
                        "worker_timeout_seconds": float(timeout_seconds),
                        "worker_process_group_rss_limit_bytes": int(
                            maximum_process_group_rss_bytes
                        ),
                        "watchdog_completed": True,
                        "process_group_empty_after_exit": True,
                        "worker_terminated_by_watchdog": False,
                    }
                now = time.monotonic()
                if leader_exit_seen_at is None:
                    leader_exit_seen_at = now
                elif now - leader_exit_seen_at > WORKER_TERMINATION_GRACE_SECONDS:
                    raise RuntimeError(
                        "row-worker leader exited while child processes remained resident"
                    )
            time.sleep(poll_interval_seconds)
    except BaseException:
        _terminate_process_group(process)
        raise


def _write_row_watchdog_receipt(
    *,
    row: Mapping[str, Any],
    measurement: Mapping[str, Any],
    config_sha256: str,
    source_capability: Mapping[str, Any],
    pre_worker_host_resource_guard: Mapping[str, Any],
    stdout_path: Path,
    stderr_path: Path,
) -> Path:
    row_path = (ROOT / str(row["output"])).resolve()
    if measurement.get("returncode") != 0 or not row_path.is_file():
        raise RuntimeError(f"row worker failed before evidence publication: {row['row_id']}")
    payload = {
        "schema_version": 1,
        "kind": ROW_WATCHDOG_KIND,
        "row_id": str(row["row_id"]),
        "worker_argv": [str(value) for value in row["command"]],
        "worker_command_sha256": canonical_sha256(
            [str(value) for value in row["command"]]
        ),
        "v2_config_sha256": str(config_sha256),
        "source_capability_sha256": source_capability["capability_sha256"],
        "row_file": _file_identity(row_path),
        "stdout_log": _file_identity(stdout_path),
        "stderr_log": _file_identity(stderr_path),
        "measurement": dict(measurement),
        "pre_worker_host_resource_guard": dict(pre_worker_host_resource_guard),
        "parent_only_rusage_is_not_total_host_memory": True,
        "cross_route_host_memory_field": (
            "measurement.sampled_process_group_rss_high_water_bytes"
        ),
    }
    receipt = {**payload, "generation_digest": canonical_sha256(payload)}
    destination = expected_watchdog_path(row_path)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _worker_command(
    *,
    worker: Path,
    config_path: Path,
    protocol_path: Path,
    scene: str,
    seed: int,
    route: str,
    output_path: Path,
    maximum_mps_working_set_bytes: int,
    allow_local_mps_execution: bool,
) -> list[str]:
    command = [
        str(DEFAULT_WORKER_PYTHON),
        str(worker),
        "--execute",
        "--g4-v2-config",
        str(config_path),
        "--protocol",
        str(protocol_path),
        "--scene",
        scene,
        "--seed",
        str(seed),
        "--route",
        route,
        "--output",
        str(output_path),
        "--maximum-mps-working-set-bytes",
        str(maximum_mps_working_set_bytes),
    ]
    if allow_local_mps_execution:
        command.append("--allow-local-mps-execution")
    return command


def _required_capability(config: Mapping[str, Any]) -> dict[str, Any]:
    return required_source_capability(Path(config["_config_path"]))


def _native_binary_blockers(base: Mapping[str, Any]) -> list[str]:
    del base
    variant = (
        ROOT
        / "third_party"
        / "fast-mac-gsplat"
        / "variants"
        / "world_foam_lane2_fused_slab_v0"
    )
    binary = (
        variant
        / "torch_world_foam_lane2_fused_slab"
        / "_C.cpython-311-darwin.so"
    )
    sources = tuple(
        path
        for path in (
            variant / "csrc" / "bindings.cpp",
            *sorted((variant / "csrc" / "metal").glob("*.mm")),
            *sorted((variant / "csrc" / "metal").glob("*.metal")),
        )
        if path.is_file()
    )
    if not binary.is_file():
        return ["worldfoam_native_extension_missing"]
    if sources and binary.stat().st_mtime_ns < max(
        path.stat().st_mtime_ns for path in sources
    ):
        return ["worldfoam_native_extension_older_than_sources"]
    return []


def _pilot_blockers(
    *,
    config: Mapping[str, Any],
    source_capability: Mapping[str, Any] | None,
) -> list[str]:
    path = (ROOT / str(config["execution"]["pilot_receipt"])).resolve()
    if not path.is_file():
        return ["g4_v2_real_native_timing_pilot_missing"]
    try:
        payload = _load_json(path)
    except Exception as error:
        return [f"g4_v2_real_native_timing_pilot_invalid:{type(error).__name__}:{error}"]
    independent_failures = validate_pilot_receipt(
        payload,
        config_path=Path(str(config["_config_path"])),
        artifact_path=path,
        verify_files=True,
    )
    if independent_failures:
        return [
            "g4_v2_real_native_timing_pilot_verification_failed:"
            + ",".join(independent_failures[:4])
        ]
    if source_capability is None:
        return ["g4_v2_real_native_timing_pilot_source_capability_missing"]
    return []


def _enforce_pre_worker_host_resource_guard(
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Sample the host once at execute time and abort before the first row."""

    if policy.get("required") is not True:
        raise RuntimeError("G4-v2 pre-matrix host-resource guard is required")
    import run_worldfoam_memory_scaling_acceptance as shared

    shared._validate_resource_policy(
        minimum_free_disk_bytes=int(policy["minimum_free_disk_bytes"]),
        minimum_available_memory_bytes=int(policy["minimum_available_memory_bytes"]),
        maximum_swap_used_bytes=int(policy["maximum_swap_used_bytes"]),
        maximum_load_average=float(policy["maximum_load_average"]),
    )
    snapshot = shared._host_resource_snapshot()
    failures = shared._resource_guard_failures(
        snapshot,
        minimum_free_disk_bytes=int(policy["minimum_free_disk_bytes"]),
        minimum_available_memory_bytes=int(policy["minimum_available_memory_bytes"]),
        maximum_swap_used_bytes=int(policy["maximum_swap_used_bytes"]),
        maximum_load_average=float(policy["maximum_load_average"]),
    )
    payload = {
        "schema_version": 1,
        "kind": "worldfoam-g4-v2-pre-worker-host-resource-guard-v1",
        "policy": dict(policy),
        "snapshot": snapshot,
        "failures": list(failures),
        "passed_immediately_before_worker": not failures,
    }
    if failures:
        raise RuntimeError(
            "G4-v2 aborted before first row by host-resource guard: "
            + ", ".join(failures)
        )
    return {**payload, "generation_digest": canonical_sha256(payload)}


def runtime_blockers(
    *,
    config: Mapping[str, Any],
    base: Mapping[str, Any],
    base_path: Path,
    receipts: Mapping[tuple[str, int], Any],
) -> list[str]:
    """Return v2-specific blockers without importing Torch or native ops."""

    blockers: list[str] = []
    try:
        validate_contract(base, config_path=base_path)
    except Exception as error:
        blockers.append(f"bound_g4_v1_contract_invalid:{type(error).__name__}:{error}")
    if any(receipt.tractability_preflight_passed is not True for receipt in receipts.values()):
        blockers.append("g4_v2_selected_ray_tractability_preflight_failed")

    worker = ROOT / str(config["execution"]["row_worker"])
    capability = ROOT / str(config["execution"]["source_capability"])
    if not worker.is_file():
        blockers.append("g4_v2_selected_ray_row_worker_missing")
    if not DEFAULT_WORKER_PYTHON.is_file() or not os.access(
        DEFAULT_WORKER_PYTHON, os.X_OK
    ):
        blockers.append("g4_v2_cpython311_worker_missing")
    if not capability.is_file():
        blockers.append("g4_v2_selected_ray_source_capability_missing")
        actual_capability = None
    else:
        try:
            actual = _load_json(capability)
        except Exception as error:
            blockers.append(
                "g4_v2_selected_ray_source_capability_invalid:"
                f"{type(error).__name__}:{error}"
            )
        else:
            expected_config = {**config, "_config_path": str(config["_config_path"])}
            if actual != _required_capability(expected_config):
                blockers.append("g4_v2_selected_ray_source_capability_not_verified")
            actual_capability = actual
            if actual.get(
                "spatial_major_cross_time_worldfoam_heldout_source_complete"
            ) is not True:
                blockers.append("g4_v2_spatial_major_heldout_source_incomplete")
    if not V2_VERIFIER.is_file():
        blockers.append("g4_v2_independent_artifact_verifier_missing")

    for path, blocker in (
        (
            TRAIN / "worldfoam_public_quality_dataset_provider.py",
            "g4_v2_public_dataset_provider_missing",
        ),
        (
            TRAIN / "worldfoam_public_quality_inputs.py",
            "g4_v2_public_input_binding_missing",
        ),
        (
            TRAIN / "worldfoam_native4d_public_quality_executor.py",
            "g4_v2_worldfoam_executor_missing",
        ),
        (
            TRAIN / "world_tubes_public_quality_executor.py",
            "g4_v2_world_tubes_executor_missing",
        ),
        (
            TRAIN / "dynamic_3dgs_public_quality_executor.py",
            "g4_v2_dynamic_3dgs_executor_missing",
        ),
    ):
        if not path.is_file():
            blockers.append(blocker)
    blockers.extend(_native_binary_blockers(base))
    blockers.extend(
        _pilot_blockers(config=config, source_capability=actual_capability)
    )
    return sorted(set(blockers))


def build_plan(
    config_path: Path = DEFAULT_CONFIG,
    *,
    allow_local_mps_execution: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config, base, base_path, receipts = build_matrix_workload_receipts(config_path)
    config = {**config, "_config_path": str(config_path)}
    output_root = (ROOT / str(config["output_root"])).resolve()
    worker = ROOT / str(config["execution"]["row_worker"])
    rows: list[dict[str, Any]] = []
    for scene in REQUIRED_SCENES:
        scene_spec = next(value for value in base["scenes"] if value["scene"] == scene)
        protocol_path = (ROOT / str(scene_spec["protocol"])).resolve()
        for seed in REQUIRED_SEEDS:
            workload = receipts[(scene, seed)]
            for route in REQUIRED_ROUTES:
                output_path = expected_row_path(
                    output_root,
                    scene=scene,
                    seed=seed,
                    route=route,
                )
                rows.append(
                    {
                        "row_id": f"{scene}/seed_{seed}/{route}",
                        "scene": scene,
                        "seed": seed,
                        "route": route,
                        "protocol": _display(protocol_path),
                        "protocol_sha256": workload.protocol_sha256,
                        "workload_receipt_generation_digest": (
                            workload.generation_digest
                        ),
                        "route_schedule_sha256": workload.route_schedule_sha256,
                        "training_target_pixels": workload.selected_target_pixels,
                        "training_loss_scalars": workload.selected_loss_scalar_count,
                        "heldout_target_pixels": workload.heldout_target_pixels,
                        "route_specific_training_rasterized_work_receipt_required": True,
                        "training_rasterized_work_claimed_equal": False,
                        "output": _display(output_path),
                        "process_group_watchdog": _display(
                            expected_watchdog_path(output_path)
                        ),
                        "command": _worker_command(
                            worker=worker,
                            config_path=config_path,
                            protocol_path=protocol_path,
                            scene=scene,
                            seed=seed,
                            route=route,
                            output_path=output_path,
                            maximum_mps_working_set_bytes=int(
                                config["execution"][
                                    "maximum_mps_working_set_bytes_per_worker"
                                ]
                            ),
                            allow_local_mps_execution=allow_local_mps_execution,
                        ),
                    }
                )
    blockers = runtime_blockers(
        config=config,
        base=base,
        base_path=base_path,
        receipts=receipts,
    )
    workload_payload = {
        f"{scene}/seed_{seed}": receipt.as_dict()
        for (scene, seed), receipt in receipts.items()
    }
    pilot_plan = build_pilot_plan(
        config_path=config_path,
        python=DEFAULT_WORKER_PYTHON,
    )
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "kind": PLAN_KIND,
        "status": "blocked" if blockers else "planned",
        "config": _display(config_path),
        "config_sha256": file_sha256(config_path),
        "base_g4_v1_config": _display(base_path),
        "base_g4_v1_sha256": file_sha256(base_path),
        "all_pixel_g4_v1_mutated": False,
        "scientific_scope": (
            "matched selected-ray training across four routes with unchanged "
            "full-pixel/full-300-frame heldout evaluation"
        ),
        "selected_ray_training": True,
        "identical_target_and_loss_budget_all_routes": True,
        "training_rasterized_work_claimed_equal": False,
        "route_specific_training_rasterized_work_receipt_required": True,
        "full_pixel_full_temporal_heldout_evaluation": True,
        "expected_row_count": 36,
        "fresh_process_count": 36,
        "process_group_watchdog_policy": {
            "required": config["execution"]["process_group_watchdog_required"],
            "maximum_worker_process_group_rss_bytes": config["execution"][
                "maximum_worker_process_group_rss_bytes"
            ],
            "poll_interval_seconds": config["execution"][
                "worker_watchdog_poll_interval_seconds"
            ],
            "worker_timeout_seconds": config["execution"][
                "worker_timeout_seconds"
            ],
            "measurement_kind": "parent-ps-sampled-process-group-high-water-v1",
            "includes_child_processes_such_as_ffmpeg": True,
        },
        "pre_matrix_host_resource_policy": {
            "required": config["execution"][
                "pre_matrix_host_resource_guard_required"
            ],
            "minimum_free_disk_bytes": config["execution"][
                "pre_matrix_minimum_free_disk_bytes"
            ],
            "minimum_available_memory_bytes": config["execution"][
                "pre_matrix_minimum_available_memory_bytes"
            ],
            "maximum_swap_used_bytes": config["execution"][
                "pre_matrix_maximum_swap_used_bytes"
            ],
            "maximum_load_average": config["execution"][
                "pre_matrix_maximum_load_average"
            ],
            "default_dry_plan_samples_host_resources": False,
            "rechecked_immediately_before_every_row": True,
        },
        "hard_mps_working_set_limit_bytes_per_worker": config["execution"][
            "maximum_mps_working_set_bytes_per_worker"
        ],
        "default_dry_plan_imports_torch": False,
        "required_real_native_pilot_plan": pilot_plan,
        "runtime_ready": not blockers,
        "abort_before_first_row": bool(blockers),
        "runtime_blockers": blockers,
        "workload_receipts": workload_payload,
        "workload_receipts_sha256": canonical_sha256(workload_payload),
        "rows": rows,
        "evidence_emitted": False,
        "public_quality_claim_permitted": False,
        "native_memory_fit_claim_permitted": False,
        "plan_sha256": "",
    }
    plan["plan_sha256"] = canonical_sha256(
        {key: value for key, value in plan.items() if key != "plan_sha256"}
    )
    return plan


def execute_plan(plan: Mapping[str, Any], *, allow_local_mps_execution: bool) -> None:
    """Run sequential rows under a process-group watchdog, then verify them."""

    if not allow_local_mps_execution:
        raise RuntimeError(
            "G4-v2 local MPS execution requires explicit --allow-local-mps-execution"
        )
    blockers = tuple(str(value) for value in plan.get("runtime_blockers", ()))
    if blockers:
        raise RuntimeError(
            "G4-v2 aborted before first row: " + ", ".join(blockers)
        )
    capability_path = ROOT / "src" / "train" / "worldfoam_g4_v2_source_capability.json"
    source_capability = _load_json(capability_path)
    policy = plan.get("process_group_watchdog_policy")
    if not isinstance(policy, Mapping) or policy.get("required") is not True:
        raise RuntimeError("G4-v2 process-group watchdog policy is missing")
    host_policy = plan.get("pre_matrix_host_resource_policy")
    if not isinstance(host_policy, Mapping):
        raise RuntimeError("G4-v2 pre-matrix host-resource policy is missing")
    # Keep each route in one fresh, sequential process.  The parent watcher is
    # evidence-critical because the child's RUSAGE_SELF peak omits ffmpeg and
    # any other descendants.
    for row in plan["rows"]:
        host_guard = _enforce_pre_worker_host_resource_guard(host_policy)
        row_path = (ROOT / str(row["output"])).resolve()
        row_path.parent.mkdir(parents=True, exist_ok=True)
        stdout_path = row_path.parent / ROW_STDOUT_FILENAME
        stderr_path = row_path.parent / ROW_STDERR_FILENAME
        watchdog_path = expected_watchdog_path(row_path)
        if watchdog_path.exists():
            raise FileExistsError(f"row watchdog already exists: {watchdog_path}")
        with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
            measurement = _run_guarded_row_process(
                tuple(str(value) for value in row["command"]),
                stdout=stdout,
                stderr=stderr,
                maximum_process_group_rss_bytes=int(
                    policy["maximum_worker_process_group_rss_bytes"]
                ),
                poll_interval_seconds=float(policy["poll_interval_seconds"]),
                timeout_seconds=float(policy["worker_timeout_seconds"]),
            )
        _write_row_watchdog_receipt(
            row=row,
            measurement=measurement,
            config_sha256=str(plan["config_sha256"]),
            source_capability=source_capability,
            pre_worker_host_resource_guard=host_guard,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )
    artifact = collect_and_verify(config_path=Path(str(plan["config"])))
    if artifact.get("status") != "measured":
        raise RuntimeError(
            "G4-v2 independent collector rejected completed rows: "
            + "; ".join(str(value) for value in artifact.get("failures", ()))
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan the matched selected-ray WorldFoam G4-v2 matrix."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-local-mps-execution", action="store_true")
    args = parser.parse_args(argv)
    plan = build_plan(
        args.config,
        allow_local_mps_execution=args.allow_local_mps_execution,
    )
    if args.execute:
        execute_plan(
            plan,
            allow_local_mps_execution=args.allow_local_mps_execution,
        )
    print(json.dumps(serialize_config_value(plan), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "PLAN_KIND",
    "PLAN_SCHEMA_VERSION",
    "V2_ROW_KIND",
    "build_plan",
    "execute_plan",
    "expected_row_path",
    "expected_watchdog_path",
    "main",
    "runtime_blockers",
)
