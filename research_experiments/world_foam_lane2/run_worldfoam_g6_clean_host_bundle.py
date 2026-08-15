#!/usr/bin/env python3
"""Rebuild, attest, execute, and verify the real WorldFoam G6 ablation.

The default invocation is a source-only plan.  It starts no subprocess,
imports neither Torch nor the native package, performs no MPS allocation or
dispatch, writes no artifact, and emits zero evidence rows.

``--execute`` is intentionally Mac/MPS-only.  It applies the incident host
guard before compiling, force-rebuilds the exact fused-slab variant with the
same CPython 3.11 interpreter used by every evidence worker, writes and
re-verifies a 133-schema native-build attestation, checks both G6 ABI seals,
requires an unblocked allocation-free producer plan, runs all 21 evidence rows
plus three restart processes, and independently verifies the final artifact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import attest_worldfoam_fused_slab_build as build_attester
import run_worldfoam_memory_scaling_acceptance as shared
import run_worldfoam_training_memory_ablation as producer
import verify_worldfoam_training_memory_ablation as evidence_verifier


ROOT = Path(__file__).resolve().parents[2]
BUNDLE_PATH = Path(__file__).resolve()
VARIANT_DIR = build_attester.DEFAULT_VARIANT_DIR
PACKAGE_DIR = VARIANT_DIR / build_attester.PACKAGE_NAME
BUILD_SETUP = VARIANT_DIR / "setup.py"
ATTEST_SCRIPT = Path(build_attester.__file__).resolve()
IMPORT_VERIFY_SCRIPT = SCRIPT_DIR / "verify_worldfoam_native_variant_imports.py"
PRODUCER_SCRIPT = Path(producer.__file__).resolve()
EVIDENCE_VERIFY_SCRIPT = Path(evidence_verifier.__file__).resolve()

BUNDLE_KIND = "worldfoam_g6_clean_host_execution_bundle"
BUNDLE_SCHEMA_VERSION = 1
EXPECTED_SOURCE_SCHEMA_COUNT = 133
EXPECTED_PRIMARY_ROWS = 12
EXPECTED_CONTROL_ROWS = 9
EXPECTED_EVIDENCE_ROWS = 21
EXPECTED_RESTART_PROCESSES = 3
EXPECTED_TOTAL_PROCESSES = 24
EXPECTED_MPS_LIMIT_BYTES = 2 * 1024**3
EXPECTED_PROCESS_GROUP_RSS_LIMIT_BYTES = 4 * 1024**3
PYTHON_IDENTITY_CODE = (
    "import json,platform,sys,torch;"
    "print(json.dumps({"
    "'implementation':sys.implementation.name,"
    "'version_info':list(sys.version_info[:3]),"
    "'executable':sys.executable,"
    "'platform_system':platform.system(),"
    "'platform_machine':platform.machine(),"
    "'torch_version':torch.__version__,"
    "'mps_built':bool(torch.backends.mps.is_built()),"
    "'mps_available':bool(torch.backends.mps.is_available())"
    "},sort_keys=True))"
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _selected_python(args: argparse.Namespace) -> Path:
    expanded = Path(args.python).expanduser()
    path = expanded if expanded.is_absolute() else Path.cwd() / expanded
    path = path.absolute()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise FileNotFoundError(f"selected Python executable is missing: {path}")
    return path


def _derived_paths(args: argparse.Namespace) -> dict[str, Path]:
    output = Path(args.output).expanduser().resolve()
    parent = output.parent
    return {
        "output": output,
        "attestation": Path(args.attestation_output).expanduser().resolve(),
        "import_report": (
            Path(args.import_report_output).expanduser().resolve()
            if args.import_report_output is not None
            else parent / "worldfoam_g6_native_import_verification.json"
        ),
        "bundle_receipt": (
            Path(args.bundle_receipt_output).expanduser().resolve()
            if args.bundle_receipt_output is not None
            else parent / "worldfoam_g6_clean_host_bundle_receipt.json"
        ),
        "run_dir": parent / f".{output.stem}_clean_host_bundle",
        "worker_dir": parent / f".{output.stem}_fresh_workers",
    }


def _producer_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        backend="mps",
        config=Path(args.config),
        contract=Path(args.contract),
        output=Path(args.output),
        trial_driver=Path(args.trial_driver),
        python=Path(args.python),
        native_ops_module=producer.DEFAULT_NATIVE_OPS_MODULE,
        minimum_free_disk_bytes=args.minimum_free_disk_bytes,
        minimum_available_memory_bytes=args.minimum_available_memory_bytes,
        maximum_swap_used_bytes=args.maximum_swap_used_bytes,
        maximum_load_average=args.maximum_load_average,
        mps_memory_fraction=args.mps_memory_fraction,
        overwrite=args.overwrite,
    )


def _build_command(args: argparse.Namespace) -> list[str]:
    return [
        str(_selected_python(args)),
        str(BUILD_SETUP),
        "build_ext",
        "--inplace",
        "--force",
    ]


def _attest_write_command(args: argparse.Namespace, paths: Mapping[str, Path]) -> list[str]:
    return [
        str(_selected_python(args)),
        str(ATTEST_SCRIPT),
        "--variant-dir",
        str(VARIANT_DIR),
        "--write-receipt",
        str(paths["attestation"]),
    ]


def _attest_verify_command(args: argparse.Namespace, paths: Mapping[str, Path]) -> list[str]:
    return [
        str(_selected_python(args)),
        str(ATTEST_SCRIPT),
        "--variant-dir",
        str(VARIANT_DIR),
        "--verify-receipt",
        str(paths["attestation"]),
    ]


def _import_verify_command(args: argparse.Namespace, paths: Mapping[str, Path]) -> list[str]:
    return [
        str(_selected_python(args)),
        str(IMPORT_VERIFY_SCRIPT),
        "--variant-root",
        str(VARIANT_DIR.parent),
        "--attestation-json",
        str(paths["attestation"]),
        "--out-json",
        str(paths["import_report"]),
    ]


def _producer_command(
    args: argparse.Namespace,
    *,
    execute: bool,
) -> list[str]:
    command = [
        str(_selected_python(args)),
        str(PRODUCER_SCRIPT),
        "--backend",
        "mps",
        "--config",
        str(Path(args.config).expanduser().resolve()),
        "--contract",
        str(Path(args.contract).expanduser().resolve()),
        "--trial-driver",
        str(Path(args.trial_driver).expanduser().resolve()),
        "--output",
        str(Path(args.output).expanduser().resolve()),
        "--python",
        str(_selected_python(args)),
        "--native-ops-module",
        producer.DEFAULT_NATIVE_OPS_MODULE,
        "--minimum-free-disk-bytes",
        str(args.minimum_free_disk_bytes),
        "--minimum-available-memory-bytes",
        str(args.minimum_available_memory_bytes),
        "--maximum-swap-used-bytes",
        str(args.maximum_swap_used_bytes),
        "--maximum-load-average",
        str(args.maximum_load_average),
        "--mps-memory-fraction",
        str(args.mps_memory_fraction),
    ]
    if execute:
        command.append("--execute")
        if args.overwrite:
            command.append("--overwrite")
    return command


def _evidence_verify_command(args: argparse.Namespace, paths: Mapping[str, Path]) -> list[str]:
    return [
        str(_selected_python(args)),
        str(EVIDENCE_VERIFY_SCRIPT),
        str(paths["output"]),
        "--config",
        str(Path(args.config).expanduser().resolve()),
        "--contract",
        str(Path(args.contract).expanduser().resolve()),
    ]


def _static_plan(args: argparse.Namespace) -> dict[str, Any]:
    paths = _derived_paths(args)
    blockers: list[str] = []
    source_snapshot: Mapping[str, Any] = {}
    try:
        source_snapshot = build_attester._source_snapshot(VARIANT_DIR)
    except Exception as exc:
        blockers.append(f"native_source_contract_failed:{type(exc).__name__}:{exc}")

    try:
        config = evidence_verifier.load_json_object(Path(args.config))
        contract = evidence_verifier.load_json_object(Path(args.contract))
        evidence_verifier.validate_config(config)
        evidence_verifier.validate_contract(contract)
    except Exception as exc:
        config = {}
        contract = {}
        blockers.append(f"g6_contract_failed:{type(exc).__name__}:{exc}")

    try:
        producer_plan = producer.make_plan(_producer_namespace(args))
    except Exception as exc:
        producer_plan = {}
        blockers.append(f"g6_producer_plan_failed:{type(exc).__name__}:{exc}")

    candidates = tuple(sorted(PACKAGE_DIR.glob("_C*.so")))
    if len(candidates) > 1:
        blockers.append("multiple_native_extension_candidates_require_manual_cleanup")
    try:
        selected_python = _selected_python(args)
    except FileNotFoundError:
        selected_python = Path(args.python).expanduser().absolute()
        blockers.append("selected_python_missing")
    if sys.platform != "darwin":
        blockers.append("bundle_execute_requires_darwin")
    if source_snapshot and source_snapshot.get("schema_count") != EXPECTED_SOURCE_SCHEMA_COUNT:
        blockers.append("native_source_schema_count_changed")

    counts = {
        "primary_rows": producer_plan.get("planned_row_count"),
        "control_rows": producer_plan.get("planned_control_row_count"),
        "evidence_rows": (
            int(producer_plan.get("planned_row_count", -1))
            + int(producer_plan.get("planned_control_row_count", -1))
        ),
        "restart_processes": producer_plan.get("planned_restart_process_count"),
        "total_processes": producer_plan.get("planned_total_process_count"),
    }
    expected_counts = {
        "primary_rows": EXPECTED_PRIMARY_ROWS,
        "control_rows": EXPECTED_CONTROL_ROWS,
        "evidence_rows": EXPECTED_EVIDENCE_ROWS,
        "restart_processes": EXPECTED_RESTART_PROCESSES,
        "total_processes": EXPECTED_TOTAL_PROCESSES,
    }
    if counts != expected_counts:
        blockers.append("g6_matrix_count_changed")

    if config:
        limits = config.get("memory_limits_bytes", {})
        if limits.get("maximum_mps_working_set") != EXPECTED_MPS_LIMIT_BYTES:
            blockers.append("g6_mps_limit_changed")
        if (
            limits.get("maximum_worker_process_group_rss")
            != EXPECTED_PROCESS_GROUP_RSS_LIMIT_BYTES
        ):
            blockers.append("g6_process_group_rss_limit_changed")
    if contract:
        if contract.get("maximum_mps_working_set_bytes") != EXPECTED_MPS_LIMIT_BYTES:
            blockers.append("g6_contract_mps_limit_changed")
        if (
            contract.get("maximum_worker_process_group_rss_bytes")
            != EXPECTED_PROCESS_GROUP_RSS_LIMIT_BYTES
        ):
            blockers.append("g6_contract_process_group_rss_limit_changed")

    prebuild_blockers = tuple(producer_plan.get("blocking_reasons", ()))
    remediable = {
        "native_extension_missing_or_ambiguous",
        "native_extension_older_than_bound_native_sources",
    }
    unremediable_prebuild = sorted(set(prebuild_blockers) - remediable)
    blockers.extend(
        f"unremediable_prebuild_producer_blocker:{value}"
        for value in unremediable_prebuild
    )

    commands = {
        "build": _build_command(args),
        "write_attestation": _attest_write_command(args, paths),
        "verify_attestation": _attest_verify_command(args, paths),
        "verify_import_and_g6_abis": _import_verify_command(args, paths),
        "postbuild_allocation_free_plan": _producer_command(args, execute=False),
        "execute_ablation": _producer_command(args, execute=True),
        "verify_evidence": _evidence_verify_command(args, paths),
    }
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "status": (
            "static_prebuild_ready_host_unchecked" if not blockers else "blocked"
        ),
        "static_execution_path_ready_after_host_guard_and_rebuild": not blockers,
        "blocking_reasons": blockers,
        "backend": "mps",
        "claim": "worldfoam_g6_native_memory_and_work_ablation",
        "matrix": counts,
        "expected_matrix": expected_counts,
        "memory_gates": {
            "maximum_mps_working_set_bytes": EXPECTED_MPS_LIMIT_BYTES,
            "maximum_sampled_process_group_rss_bytes": (
                EXPECTED_PROCESS_GROUP_RSS_LIMIT_BYTES
            ),
            "mps_gate_kind": "hard_effective_per_process_allocator_limit",
            "rss_gate_kind": "parent_sampled_process_group_watchdog",
        },
        "current_prebuild_producer_status": producer_plan.get("status"),
        "current_prebuild_producer_blockers": list(prebuild_blockers),
        "native_source_schema_count": source_snapshot.get("schema_count"),
        "native_source_schema_inventory_sha256": source_snapshot.get(
            "full_schema_inventory_sha256"
        ),
        "current_native_extension_candidate_count": len(candidates),
        "current_native_extension_candidates": [str(path) for path in candidates],
        "selected_python": str(selected_python),
        "current_wrapper_runtime": {
            "implementation": sys.implementation.name,
            "version": platform.python_version(),
            "platform": sys.platform,
            "machine": platform.machine(),
        },
        "paths": {key: str(value) for key, value in paths.items()},
        "commands": commands,
        "dry_plan_receipt": {
            "subprocess_started": False,
            "torch_imported_by_bundle": False,
            "native_extension_imported": False,
            "native_build_started": False,
            "mps_allocation_or_dispatch_started": False,
            "artifact_written": False,
            "evidence_rows_emitted": 0,
            "dry_plan_is_evidence": False,
            "host_resource_guard_sampled": False,
        },
        "cuda_portability": {
            "b200_can_validate_current_g6_claim": False,
            "reason": (
                "G6 is sealed to Darwin, Apple MPS allocator counters and limits, "
                "Metal/Objective-C++ kernels, and an MPS-only production adapter"
            ),
            "cuda_port_required": True,
            "cuda_result_would_require_separate_backend_contract": True,
        },
    }


def _environment() -> dict[str, str]:
    result = os.environ.copy()
    result["PYTHONDONTWRITEBYTECODE"] = "1"
    return result


def _run_guarded_phase(
    command: Sequence[str],
    *,
    cwd: Path,
    run_dir: Path,
    label: str,
) -> dict[str, Any]:
    stdout_path = run_dir / f"{label}.stdout.log"
    stderr_path = run_dir / f"{label}.stderr.log"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            watchdog = shared._run_guarded_worker(
                command,
                cwd=cwd,
                env=_environment(),
                stdout=stdout,
                stderr=stderr,
            )
        except Exception as exc:
            raise RuntimeError(
                f"{label} violated its guard; see {stderr_path}"
            ) from exc
    if watchdog["returncode"] != 0:
        raise RuntimeError(
            f"{label} failed with return code {watchdog['returncode']}; "
            f"see {stderr_path}"
        )
    return {
        "command": list(command),
        "command_sha256": _sha256(tuple(command)),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "watchdog": watchdog,
    }


def _read_single_json_log(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"{label} did not emit one JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} JSON root is not an object")
    return value


def _require_python_identity(identity: Mapping[str, Any]) -> None:
    if identity.get("implementation") != "cpython":
        raise RuntimeError("G6 native build requires CPython")
    version = identity.get("version_info")
    if not isinstance(version, list) or version[:2] != [3, 11]:
        raise RuntimeError("G6 native build requires Python 3.11")
    if identity.get("platform_system") != "Darwin":
        raise RuntimeError("G6 native build requires Darwin")
    if identity.get("mps_built") is not True or identity.get("mps_available") is not True:
        raise RuntimeError("G6 evidence requires an available PyTorch MPS backend")


def _require_postbuild_plan(plan: Mapping[str, Any]) -> None:
    expected = {
        "planned_row_count": EXPECTED_PRIMARY_ROWS,
        "planned_control_row_count": EXPECTED_CONTROL_ROWS,
        "planned_restart_process_count": EXPECTED_RESTART_PROCESSES,
        "planned_total_process_count": EXPECTED_TOTAL_PROCESSES,
        "evidence_rows_emitted": 0,
        "control_evidence_rows_emitted": 0,
        "artifact_written": False,
        "dry_run_is_evidence": False,
        "paper_claim_permitted": False,
        "execution_ready": True,
        "status": "planned",
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise RuntimeError(
                f"postbuild allocation-free G6 plan changed {key}: "
                f"expected {value!r}, found {plan.get(key)!r}"
            )
    if plan.get("blocking_reasons") != []:
        raise RuntimeError(
            "postbuild allocation-free G6 plan remains blocked: "
            + ", ".join(str(value) for value in plan.get("blocking_reasons", ()))
        )


def _require_import_report(report: Mapping[str, Any]) -> None:
    if report.get("status") != "ok" or report.get("variant_count") != 1:
        raise RuntimeError(f"native import verification failed: {report.get('failures')}")
    variants = report.get("variants")
    if not isinstance(variants, list) or len(variants) != 1:
        raise RuntimeError("native import verification emitted a noncanonical variant list")
    row = variants[0]
    required = {
        "status": "ok",
        "schema_count": EXPECTED_SOURCE_SCHEMA_COUNT,
        "registered_schema_count": EXPECTED_SOURCE_SCHEMA_COUNT,
        "exact_schema_inventory_match": True,
        "kinetic_compiled_abi_attestation_present": True,
        "kinetic_compiled_abi_attestation_error": "",
        "kinetic_lazy_full_geometry_compiled_abi_attestation_present": True,
        "kinetic_lazy_full_geometry_compiled_abi_attestation_error": "",
    }
    if not isinstance(row, Mapping):
        raise RuntimeError("native import verification row is not an object")
    for key, value in required.items():
        if row.get(key) != value:
            raise RuntimeError(
                f"native import verification changed {key}: "
                f"expected {value!r}, found {row.get(key)!r}"
            )
    for key in (
        "missing_registered_schemas",
        "unexpected_registered_schemas",
        "mismatched_registered_schema_signatures",
        "missing_dispatch_kernels",
    ):
        if row.get(key) != []:
            raise RuntimeError(f"native import verification reports {key}")
    attestation = row.get("attestation")
    if not isinstance(attestation, Mapping) or attestation.get("status") != "accepted":
        raise RuntimeError("native import verification did not accept the build attestation")


def _run_long_ablation(
    command: Sequence[str],
    *,
    run_dir: Path,
) -> dict[str, Any]:
    """Run the producer without an outer timeout; every child has its own guard."""

    stdout_path = run_dir / "g6_ablation.stdout.log"
    stderr_path = run_dir / "g6_ablation.stderr.log"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        completed = subprocess.run(
            tuple(command),
            cwd=ROOT,
            env=_environment(),
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return {
        "command": list(command),
        "command_sha256": _sha256(tuple(command)),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
        "returncode": int(completed.returncode),
        "inner_fresh_process_watchdogs_required": True,
        "outer_timeout_intentionally_disabled": True,
    }


def _execute(args: argparse.Namespace, plan: Mapping[str, Any]) -> int:
    if (
        plan.get("static_execution_path_ready_after_host_guard_and_rebuild")
        is not True
    ):
        raise RuntimeError(
            "G6 clean-host bundle is statically blocked: "
            + ", ".join(str(value) for value in plan.get("blocking_reasons", ()))
        )
    if sys.platform != "darwin":
        raise RuntimeError("G6 clean-host execution is Darwin/MPS-only")
    _selected_python(args)
    paths = _derived_paths(args)
    claimed_paths = (
        paths["output"],
        paths["attestation"],
        paths["import_report"],
        paths["bundle_receipt"],
        paths["run_dir"],
        paths["worker_dir"],
    )
    existing = [str(path) for path in claimed_paths if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "G6 output already exists; pass --overwrite only for an intentional rerun: "
            + ", ".join(existing)
        )
    paths["run_dir"].mkdir(parents=True, exist_ok=True)

    host_snapshots = [
        shared._guard_host(
            minimum_free_disk_bytes=args.minimum_free_disk_bytes,
            minimum_available_memory_bytes=args.minimum_available_memory_bytes,
            maximum_swap_used_bytes=args.maximum_swap_used_bytes,
            maximum_load_average=args.maximum_load_average,
        )
    ]
    phases: dict[str, Any] = {}

    identity_phase = _run_guarded_phase(
        [_selected_python(args).as_posix(), "-c", PYTHON_IDENTITY_CODE],
        cwd=ROOT,
        run_dir=paths["run_dir"],
        label="python_identity",
    )
    phases["python_identity"] = identity_phase
    python_identity = _read_single_json_log(
        Path(identity_phase["stdout"]), label="Python identity"
    )
    _require_python_identity(python_identity)

    phases["native_build"] = _run_guarded_phase(
        _build_command(args),
        cwd=VARIANT_DIR,
        run_dir=paths["run_dir"],
        label="native_build",
    )
    host_snapshots.append(
        shared._guard_host(
            minimum_free_disk_bytes=args.minimum_free_disk_bytes,
            minimum_available_memory_bytes=args.minimum_available_memory_bytes,
            maximum_swap_used_bytes=args.maximum_swap_used_bytes,
            maximum_load_average=args.maximum_load_average,
        )
    )

    phases["write_attestation"] = _run_guarded_phase(
        _attest_write_command(args, paths),
        cwd=ROOT,
        run_dir=paths["run_dir"],
        label="write_attestation",
    )
    phases["verify_attestation"] = _run_guarded_phase(
        _attest_verify_command(args, paths),
        cwd=ROOT,
        run_dir=paths["run_dir"],
        label="verify_attestation",
    )
    attestation_report = _read_single_json_log(
        Path(phases["verify_attestation"]["stdout"]),
        label="build attestation verification",
    )
    if (
        attestation_report.get("status") != "accepted"
        or attestation_report.get("schema_count") != EXPECTED_SOURCE_SCHEMA_COUNT
        or attestation_report.get("no_metal_dispatch") is not True
    ):
        raise RuntimeError(
            f"build attestation verification failed: {attestation_report.get('failures')}"
        )

    phases["verify_import_and_g6_abis"] = _run_guarded_phase(
        _import_verify_command(args, paths),
        cwd=ROOT,
        run_dir=paths["run_dir"],
        label="verify_import_and_g6_abis",
    )
    import_report = evidence_verifier.load_json_object(paths["import_report"])
    _require_import_report(import_report)

    phases["postbuild_allocation_free_plan"] = _run_guarded_phase(
        _producer_command(args, execute=False),
        cwd=ROOT,
        run_dir=paths["run_dir"],
        label="postbuild_allocation_free_plan",
    )
    postbuild_plan = _read_single_json_log(
        Path(phases["postbuild_allocation_free_plan"]["stdout"]),
        label="postbuild allocation-free G6 plan",
    )
    _require_postbuild_plan(postbuild_plan)
    host_snapshots.append(
        shared._guard_host(
            minimum_free_disk_bytes=args.minimum_free_disk_bytes,
            minimum_available_memory_bytes=args.minimum_available_memory_bytes,
            maximum_swap_used_bytes=args.maximum_swap_used_bytes,
            maximum_load_average=args.maximum_load_average,
        )
    )

    phases["g6_ablation"] = _run_long_ablation(
        _producer_command(args, execute=True),
        run_dir=paths["run_dir"],
    )
    if not paths["output"].is_file():
        raise RuntimeError(
            "G6 producer did not write an artifact; see "
            + phases["g6_ablation"]["stderr"]
        )

    phases["verify_evidence"] = _run_guarded_phase(
        _evidence_verify_command(args, paths),
        cwd=ROOT,
        run_dir=paths["run_dir"],
        label="verify_evidence",
    )
    verification_report = _read_single_json_log(
        Path(phases["verify_evidence"]["stdout"]),
        label="G6 evidence verification",
    )
    if (
        phases["g6_ablation"]["returncode"] != 0
        or verification_report.get("accepted") is not True
        or verification_report.get("observed_row_count") != EXPECTED_PRIMARY_ROWS
        or verification_report.get("observed_control_row_count")
        != EXPECTED_CONTROL_ROWS
    ):
        raise RuntimeError(
            "G6 artifact is a preserved negative or failed result; "
            f"producer_returncode={phases['g6_ablation']['returncode']}, "
            f"failures={verification_report.get('failures')}"
        )

    receipt = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "kind": BUNDLE_KIND,
        "status": "accepted",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": "mps",
        "claim_scope": "synthetic_systems_memory_trainability_only",
        "matrix": {
            "primary_rows": EXPECTED_PRIMARY_ROWS,
            "control_rows": EXPECTED_CONTROL_ROWS,
            "evidence_rows": EXPECTED_EVIDENCE_ROWS,
            "restart_processes": EXPECTED_RESTART_PROCESSES,
            "total_fresh_processes": EXPECTED_TOTAL_PROCESSES,
        },
        "memory_gates": plan["memory_gates"],
        "python_identity": python_identity,
        "host_resource_preflight_snapshots": host_snapshots,
        "native_build_attestation": {
            "path": str(paths["attestation"]),
            "sha256": _file_sha256(paths["attestation"]),
            "receipt_payload_sha256": attestation_report[
                "receipt_payload_sha256"
            ],
            "schema_count": attestation_report["schema_count"],
            "no_metal_dispatch_during_attestation": True,
        },
        "native_import_verification": {
            "path": str(paths["import_report"]),
            "sha256": _file_sha256(paths["import_report"]),
            "status": import_report["status"],
        },
        "evidence_artifact": {
            "path": str(paths["output"]),
            "sha256": _file_sha256(paths["output"]),
            "verification_report": verification_report,
        },
        "phases": phases,
        "source_bindings": {
            "bundle_sha256": _file_sha256(BUNDLE_PATH),
            "attester_sha256": _file_sha256(ATTEST_SCRIPT),
            "import_verifier_sha256": _file_sha256(IMPORT_VERIFY_SCRIPT),
            "producer_sha256": _file_sha256(PRODUCER_SCRIPT),
            "evidence_verifier_sha256": _file_sha256(EVIDENCE_VERIFY_SCRIPT),
        },
        "cuda_portability": plan["cuda_portability"],
    }
    receipt["receipt_payload_sha256"] = _sha256(receipt)
    _write_json_atomic(paths["bundle_receipt"], receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    default_python = ROOT / ".venv" / "bin" / "python"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--python",
        type=Path,
        default=default_python if default_python.is_file() else Path(sys.executable),
    )
    parser.add_argument("--config", type=Path, default=producer.DEFAULT_CONFIG)
    parser.add_argument("--contract", type=Path, default=producer.DEFAULT_CONTRACT)
    parser.add_argument("--trial-driver", type=Path, default=producer.DEFAULT_DRIVER)
    parser.add_argument("--output", type=Path, default=producer.DEFAULT_OUTPUT)
    parser.add_argument(
        "--attestation-output",
        type=Path,
        default=build_attester.DEFAULT_RECEIPT,
    )
    parser.add_argument("--import-report-output", type=Path)
    parser.add_argument("--bundle-receipt-output", type=Path)
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
    return parser


def main() -> int:
    args = _parser().parse_args()
    shared._validate_resource_policy(
        minimum_free_disk_bytes=args.minimum_free_disk_bytes,
        minimum_available_memory_bytes=args.minimum_available_memory_bytes,
        maximum_swap_used_bytes=args.maximum_swap_used_bytes,
        maximum_load_average=args.maximum_load_average,
    )
    if not 0.0 < args.mps_memory_fraction <= shared.MAXIMUM_MPS_MEMORY_FRACTION:
        raise ValueError("MPS memory fraction exceeds the incident-calibrated limit")
    torch_present_before_plan = "torch" in sys.modules
    plan = _static_plan(args)
    torch_present_after_plan = "torch" in sys.modules
    if torch_present_before_plan or torch_present_after_plan:
        raise RuntimeError(
            "source-only G6 planning imported Torch; allocation-free dry contract failed"
        )
    plan["dry_plan_receipt"].update(
        {
            "torch_module_absent_before_plan": True,
            "torch_module_absent_after_plan": True,
            "source_only_self_check_passed": True,
        }
    )
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    return _execute(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())
