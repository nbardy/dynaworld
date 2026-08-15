#!/usr/bin/env python3
"""Run the real-native, non-evidence tractability pilot for WorldFoam G4-v2.

The default invocation is a source-only dry plan.  It imports neither Torch
nor the native extension, starts no subprocess, samples no host resources, and
writes nothing.  Real execution requires both ``--execute`` and
``--allow-local-mps-execution`` on a clean, resource-safe macOS/MPS host with
the rebuilt and attested WorldFoam extension and public Coffee Martini cache.

Each WorldFoam route runs in a separate watched process.  A route performs
exactly one v2 RGB-MSE optimizer step (four spacetime samples x 1,024 selected
pixels), transitions through ``prepare_heldout_pilot_from_current_state()``,
evaluates one bounded 128-track block across all 300 heldout times, and checks
one complete track bitwise against the old frame-major prediction path.  The
reported full-row duration is a conservative 1.25x projection of 300 measured
training steps plus all 196,608 heldout tracks.  It is not a quality row and
cannot be consumed as G4 or G6 evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import secrets
import signal
import subprocess
import sys
import tempfile
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

from config_utils import load_config_file  # noqa: E402
from worldfoam_g4_selected_ray_contract import (  # noqa: E402
    DEFAULT_CONFIG,
    canonical_sha256,
    file_sha256,
    load_selected_ray_contract,
)
from worldfoam_g4_v2_capability import required_source_capability  # noqa: E402
from verify_worldfoam_g4_v2_pilot import (  # noqa: E402
    PILOT_KIND,
    PILOT_SCHEMA_VERSION,
    RAW_LOG_KIND,
    REQUIRED_ROUTES,
    ROUTE_KIND,
    WORKER_RESULT_PREFIX,
    validate_pilot_receipt,
    verify_pilot_file,
)


SCRIPT_PATH = Path(__file__).resolve()
VERIFIER_PATH = LANE2 / "verify_worldfoam_g4_v2_pilot.py"
EXECUTOR_PATH = TRAIN / "worldfoam_native4d_public_quality_executor.py"
ROW_V2_PATH = TRAIN / "worldfoam_native4d_public_quality_row_v2.py"
SPATIAL_REPLAY_PATH = TRAIN / "worldfoam_spatial_major_heldout_evaluator.py"
CAPABILITY_SOURCE_PATH = TRAIN / "worldfoam_g4_v2_capability.py"
RUNTIME_CAPABILITY_PATH = TRAIN / "worldfoam_native4d_public_quality_capabilities.json"
RUNTIME_EVIDENCE_PATH = TRAIN / "worldfoam_native4d_public_quality_capabilities.evidence.json"
VARIANT_ROOT = (
    ROOT
    / "third_party"
    / "fast-mac-gsplat"
    / "variants"
    / "world_foam_lane2_fused_slab_v0"
)
PACKAGE_ROOT = VARIANT_ROOT / "torch_world_foam_lane2_fused_slab"
PILOT_TRACK_COUNT = 128
PILOT_FRAME_COUNT = 300
PILOT_TARGET_PIXELS = 4096
PILOT_RGB_SCALARS = PILOT_TARGET_PIXELS * 3
FULL_HELDOUT_TRACK_COUNT = 384 * 512
PROJECTION_SAFETY_MULTIPLIER = 1.25
DEFAULT_MINIMUM_FREE_DISK_BYTES = 8 * 1024**3
DEFAULT_MINIMUM_AVAILABLE_MEMORY_BYTES = 8 * 1024**3
DEFAULT_MAXIMUM_SWAP_USED_BYTES = 2 * 1024**3
DEFAULT_MAXIMUM_LOAD_AVERAGE = 8.0
DEFAULT_MPS_MEMORY_FRACTION = 0.35
MPS_WORKING_SET_LIMIT_BYTES = 2 * 1024**3
PROCESS_GROUP_RSS_LIMIT_BYTES = 4 * 1024**3
WATCHDOG_TIMEOUT_SECONDS = 2 * 60 * 60
WATCHDOG_POLL_SECONDS = 0.25
MAXIMUM_RAW_LOG_BYTES = 64 * 1024**2
WATCHDOG_KIND = "worldfoam-g4-v2-pilot-parent-watchdog-v1"
WORKER_STAGE_KIND = "worldfoam-g4-v2-pilot-worker-stage-v1"
PLAN_KIND = "worldfoam-g4-v2-real-native-pilot-plan-v1"


def _repo_display(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(Path(path).resolve())


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any], *, sort_keys: bool) -> None:
    destination = Path(path).resolve()
    try:
        destination.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("pilot output left the repository") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    temporary.unlink(missing_ok=True)
    encoded = (
        json.dumps(dict(payload), indent=2, sort_keys=sort_keys, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _process_peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _native_library_path() -> Path:
    libraries = tuple(sorted(PACKAGE_ROOT.glob("_C*.so")))
    if len(libraries) != 1:
        raise RuntimeError(f"expected one WorldFoam native library, found {len(libraries)}")
    return libraries[0].resolve()


def _native_freshness_blockers() -> list[str]:
    try:
        library = _native_library_path()
    except Exception as error:
        return [f"native_library_invalid:{type(error).__name__}:{error}"]
    sources = tuple(
        path
        for path in (
            VARIANT_ROOT / "csrc" / "bindings.cpp",
            *sorted((VARIANT_ROOT / "csrc" / "metal").glob("*.mm")),
            *sorted((VARIANT_ROOT / "csrc" / "metal").glob("*.metal")),
        )
        if path.is_file()
    )
    if sources and library.stat().st_mtime_ns < max(path.stat().st_mtime_ns for path in sources):
        return ["worldfoam_native_extension_older_than_sources"]
    return []


def _coffee_paths(
    config: Mapping[str, Any],
    base: Mapping[str, Any],
) -> tuple[Path, Path]:
    scene = next(
        item
        for item in base["scenes"]
        if isinstance(item, Mapping) and item.get("scene") == config["execution"]["pilot_scene"]
    )
    protocol_path = (ROOT / str(scene["protocol"])).resolve()
    protocol_raw = load_config_file(protocol_path)
    sample_id = str(protocol_raw["dataset"]["sample_id"])
    dataset_capability = (
        ROOT
        / "outputs"
        / "cache"
        / "worldfoam_public_quality"
        / sample_id
        / "public_train_heldout_capability.json"
    )
    return protocol_path, dataset_capability


def _source_files(config_path: Path, protocol_path: Path) -> tuple[Path, ...]:
    return (
        SCRIPT_PATH,
        VERIFIER_PATH,
        EXECUTOR_PATH,
        ROW_V2_PATH,
        SPATIAL_REPLAY_PATH,
        CAPABILITY_SOURCE_PATH,
        TRAIN / "worldfoam_g4_selected_ray_contract.py",
        TRAIN / "worldfoam_g4_selected_ray_work_plan.py",
        Path(config_path).resolve(),
        Path(protocol_path).resolve(),
    )


def _source_manifest(config_path: Path, protocol_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in _source_files(config_path, protocol_path):
        if not path.is_file():
            raise FileNotFoundError(f"pilot source is missing: {path}")
        result[_repo_display(path)] = {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
    return result


def _git_source_identity() -> dict[str, Any]:
    commit = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=ROOT, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ("git", "status", "--porcelain", "--untracked-files=all"),
            cwd=ROOT,
            text=True,
        ).strip()
    )
    return {"repository_commit": commit, "repository_dirty": dirty}


def _static_blockers(
    *,
    config_path: Path,
    python: Path,
    output_path: Path,
    overwrite: bool,
) -> tuple[list[str], dict[str, Any]]:
    blockers: list[str] = []
    details: dict[str, Any] = {}
    try:
        config, base, _base_path = load_selected_ray_contract(config_path)
        protocol_path, dataset_capability_path = _coffee_paths(config, base)
        details["protocol"] = _repo_display(protocol_path)
        details["dataset_capability"] = _repo_display(dataset_capability_path)
    except Exception as error:
        return [f"pilot_contract_invalid:{type(error).__name__}:{error}"], details
    expected_output = (ROOT / str(config["execution"]["pilot_receipt"])).resolve()
    if output_path != expected_output:
        blockers.append("pilot_output_path_changed")
    if overwrite:
        blockers.append("pilot_receipts_are_immutable_overwrite_forbidden")
    if output_path.exists():
        blockers.append("pilot_receipt_already_exists")
    if any(
        (output_path.parent / f"{route}_pilot_raw_log.json").exists()
        for route in REQUIRED_ROUTES
    ):
        blockers.append("pilot_raw_log_already_exists")
    if not python.is_file() or not os.access(python, os.X_OK):
        blockers.append("python_executable_missing")
    required_seams = {
        "prepare_heldout_pilot_from_current_state": EXECUTOR_PATH,
        "heldout_spatial_major_partial_pilot_receipt": EXECUTOR_PATH,
        "render_heldout_track_block_across_frames": EXECUTOR_PATH,
        "render_heldout_chunk": EXECUTOR_PATH,
    }
    for seam, path in required_seams.items():
        if not path.is_file() or seam not in path.read_text(encoding="utf-8"):
            blockers.append(f"pilot_source_seam_missing:{seam}")
    blockers.extend(_native_freshness_blockers())
    for path, blocker in (
        (RUNTIME_CAPABILITY_PATH, "worldfoam_runtime_capability_missing"),
        (RUNTIME_EVIDENCE_PATH, "worldfoam_runtime_evidence_missing"),
        (dataset_capability_path, "coffee_martini_public_cache_capability_missing"),
    ):
        if not path.is_file():
            blockers.append(blocker)
    capability_path = (ROOT / str(config["execution"]["source_capability"])).resolve()
    details["source_capability"] = _repo_display(capability_path)
    if not capability_path.is_file():
        blockers.append("g4_v2_source_capability_missing")
    else:
        try:
            actual = _load_json(capability_path)
            expected = required_source_capability(config_path)
        except Exception as error:
            blockers.append(f"g4_v2_source_capability_invalid:{type(error).__name__}:{error}")
        else:
            if actual != expected:
                blockers.append("g4_v2_source_capability_stale")
            details["source_capability_sha256"] = actual.get("capability_sha256")
    try:
        manifest = _source_manifest(config_path, protocol_path)
    except Exception as error:
        blockers.append(f"pilot_source_manifest_invalid:{type(error).__name__}:{error}")
    else:
        details["source_manifest_sha256"] = canonical_sha256(manifest)
    return sorted(set(blockers)), details


def build_plan(
    *,
    config_path: Path = DEFAULT_CONFIG,
    python: Path | None = None,
    output_path: Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config, base, _base_path = load_selected_ray_contract(config_path)
    protocol_path, dataset_capability_path = _coffee_paths(config, base)
    selected_python = Path(python or sys.executable).expanduser().absolute()
    output = Path(
        output_path or (ROOT / str(config["execution"]["pilot_receipt"]))
    ).resolve()
    blockers, details = _static_blockers(
        config_path=config_path,
        python=selected_python,
        output_path=output,
        overwrite=overwrite,
    )
    command = [
        str(selected_python),
        str(SCRIPT_PATH),
        "--execute",
        "--allow-local-mps-execution",
        "--config",
        str(config_path),
        "--output",
        str(output),
    ]
    payload = {
        "schema_version": 1,
        "kind": PLAN_KIND,
        "status": "blocked" if blockers else "ready_for_explicit_execute",
        "config": _repo_display(config_path),
        "config_sha256": file_sha256(config_path),
        "scene": config["execution"]["pilot_scene"],
        "seed": config["execution"]["pilot_seed"],
        "routes": list(REQUIRED_ROUTES),
        "protocol": _repo_display(protocol_path),
        "dataset_capability": _repo_display(dataset_capability_path),
        "optimizer_steps_per_route": 1,
        "selected_target_pixels_per_route": PILOT_TARGET_PIXELS,
        "heldout_frame_count_per_route": PILOT_FRAME_COUNT,
        "heldout_spatial_track_count_per_route": PILOT_TRACK_COUNT,
        "frame_major_bitwise_parity_track_count_per_route": 1,
        "projected_full_heldout_spatial_track_count": FULL_HELDOUT_TRACK_COUNT,
        "projection_safety_multiplier": PROJECTION_SAFETY_MULTIPLIER,
        "public_quality_evidence": False,
        "pilot_only": True,
        "default_plan_imports_torch": False,
        "default_plan_starts_subprocess": False,
        "default_plan_samples_host_resources": False,
        "default_plan_writes_files": False,
        "build_or_rebuild_performed": False,
        "runtime_blockers": blockers,
        "details": details,
        "output": _repo_display(output),
        "execute_command": command,
    }
    return {**payload, "plan_sha256": canonical_sha256(payload)}


def _ps_process_group_rss_bytes(process_group_id: int) -> int:
    output = subprocess.check_output(
        ("ps", "-axo", "pid=,pgid=,rss="), text=True
    )
    total_kib = 0
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            _pid, pgid, rss_kib = (int(value) for value in fields)
        except ValueError:
            continue
        if pgid == int(process_group_id):
            total_kib += rss_kib
    return total_kib * 1024


def _watchdog_payload(
    *,
    nonce: str,
    route: str,
    pid: int,
    peak_rss_bytes: int,
    sample_count: int,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "kind": WATCHDOG_KIND,
        "nonce": nonce,
        "route": route,
        "worker_pid": pid,
        "process_group_id": pid,
        "observed_process_group_peak_rss_bytes": peak_rss_bytes,
        "sample_count": sample_count,
        "timeout_seconds": WATCHDOG_TIMEOUT_SECONDS,
        "rss_limit_bytes": PROCESS_GROUP_RSS_LIMIT_BYTES,
    }
    return {**payload, "generation_digest": canonical_sha256(payload)}


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5.0)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=5.0)


def _parse_worker_stdout(stdout: str) -> dict[str, Any]:
    matches = [
        line[len(WORKER_RESULT_PREFIX) :]
        for line in stdout.splitlines()
        if line.startswith(WORKER_RESULT_PREFIX)
    ]
    if len(matches) != 1:
        raise ValueError("worker stdout must contain exactly one pilot result")
    value = json.loads(matches[0])
    if not isinstance(value, dict):
        raise TypeError("worker pilot result must be a mapping")
    return value


def _run_worker(
    *,
    python: Path,
    config_path: Path,
    protocol_path: Path,
    dataset_capability_path: Path,
    route: str,
    temporary_root: Path,
    mps_memory_fraction: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    nonce = secrets.token_hex(32)
    staging = temporary_root / f"{route}.stage.json"
    watchdog = temporary_root / f"{route}.watchdog.json"
    stdout_path = temporary_root / f"{route}.stdout.log"
    stderr_path = temporary_root / f"{route}.stderr.log"
    command = [
        str(python),
        str(SCRIPT_PATH),
        "--_worker",
        "--config",
        str(config_path),
        "--protocol",
        str(protocol_path),
        "--dataset-capability",
        str(dataset_capability_path),
        "--route",
        route,
        "--worker-nonce",
        nonce,
        "--worker-stage",
        str(staging),
        "--watchdog-receipt",
        str(watchdog),
        "--mps-memory-fraction",
        repr(float(mps_memory_fraction)),
    ]
    started = time.monotonic()
    peak_rss = 0
    sample_count = 0
    watchdog_written = False
    with stdout_path.open("wb") as stdout_handle, stderr_path.open("wb") as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
        try:
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if elapsed > WATCHDOG_TIMEOUT_SECONDS:
                    raise TimeoutError(f"pilot route exceeded watchdog timeout: {route}")
                current = _ps_process_group_rss_bytes(process.pid)
                sample_count += 1
                peak_rss = max(peak_rss, current)
                if peak_rss > PROCESS_GROUP_RSS_LIMIT_BYTES:
                    raise MemoryError(f"pilot route exceeded process-group RSS limit: {route}")
                if (
                    stdout_path.stat().st_size > MAXIMUM_RAW_LOG_BYTES
                    or stderr_path.stat().st_size > MAXIMUM_RAW_LOG_BYTES
                ):
                    raise OSError(f"pilot route exceeded bounded raw-log bytes: {route}")
                if staging.is_file() and not watchdog_written:
                    stage = _load_json(staging)
                    without_digest = {
                        key: value for key, value in stage.items() if key != "generation_digest"
                    }
                    if (
                        stage.get("schema_version") != 1
                        or stage.get("kind") != WORKER_STAGE_KIND
                        or stage.get("nonce") != nonce
                        or stage.get("route") != route
                        or stage.get("worker_pid") != process.pid
                        or stage.get("generation_digest") != canonical_sha256(without_digest)
                    ):
                        raise ValueError("pilot worker staging handshake changed")
                    receipt = _watchdog_payload(
                        nonce=nonce,
                        route=route,
                        pid=process.pid,
                        peak_rss_bytes=peak_rss,
                        sample_count=sample_count,
                    )
                    _write_json_atomic(watchdog, receipt, sort_keys=True)
                    watchdog_written = True
                time.sleep(WATCHDOG_POLL_SECONDS)
            returncode = int(process.returncode or 0)
        except BaseException:
            _terminate_process_group(process)
            raise
    if returncode != 0:
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(f"pilot worker failed ({route}, rc={returncode}): {stderr[-4000:]}")
    if not watchdog_written:
        raise RuntimeError(f"pilot worker exited before watchdog handshake: {route}")
    if stdout_path.stat().st_size > MAXIMUM_RAW_LOG_BYTES or stderr_path.stat().st_size > MAXIMUM_RAW_LOG_BYTES:
        raise OSError("pilot worker raw log exceeded its bounded size")
    stdout = stdout_path.read_text(encoding="utf-8", errors="strict")
    stderr = stderr_path.read_text(encoding="utf-8", errors="strict")
    report = _parse_worker_stdout(stdout)
    raw_payload = {
        "schema_version": 1,
        "kind": RAW_LOG_KIND,
        "route": route,
        "command": command,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
    }
    raw_log = {**raw_payload, "generation_digest": canonical_sha256(raw_payload)}
    return report, raw_log


def _configure_mps_limit(torch: Any, requested_fraction: float) -> dict[str, Any]:
    if not 0.0 < requested_fraction <= DEFAULT_MPS_MEMORY_FRACTION:
        raise ValueError("pilot MPS fraction may not exceed its safe default")
    setter = getattr(torch.mps, "set_per_process_memory_fraction", None)
    recommended = getattr(torch.mps, "recommended_max_memory", None)
    if not callable(setter) or not callable(recommended):
        raise RuntimeError("PyTorch MPS memory limit API is unavailable")
    recommended_bytes = int(recommended())
    if recommended_bytes < 1:
        raise RuntimeError("MPS recommended working set is invalid")
    effective_fraction = min(
        float(requested_fraction),
        float(MPS_WORKING_SET_LIMIT_BYTES) / float(recommended_bytes),
    )
    setter(effective_fraction)
    return {
        "requested_fraction": float(requested_fraction),
        "effective_fraction": effective_fraction,
        "recommended_max_memory_bytes": recommended_bytes,
        "effective_limit_bytes": min(
            MPS_WORKING_SET_LIMIT_BYTES,
            int(float(requested_fraction) * recommended_bytes),
        ),
    }


def _worker_stage_handshake(
    *,
    path: Path,
    nonce: str,
    route: str,
    core_sha256: str,
) -> None:
    payload = {
        "schema_version": 1,
        "kind": WORKER_STAGE_KIND,
        "nonce": nonce,
        "route": route,
        "worker_pid": os.getpid(),
        "core_sha256": core_sha256,
    }
    _write_json_atomic(
        path,
        {**payload, "generation_digest": canonical_sha256(payload)},
        sort_keys=True,
    )


def _wait_for_watchdog(
    *,
    path: Path,
    nonce: str,
    route: str,
) -> dict[str, Any]:
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if path.is_file():
            value = _load_json(path)
            without_digest = {
                key: item for key, item in value.items() if key != "generation_digest"
            }
            if (
                value.get("schema_version") != 1
                or value.get("kind") != WATCHDOG_KIND
                or value.get("nonce") != nonce
                or value.get("route") != route
                or value.get("worker_pid") != os.getpid()
                or value.get("process_group_id") != os.getpid()
                or value.get("generation_digest") != canonical_sha256(without_digest)
            ):
                raise ValueError("parent watchdog receipt changed")
            return value
        time.sleep(0.05)
    raise TimeoutError("parent watchdog receipt did not arrive")


def _worker_main(args: argparse.Namespace) -> int:
    if sys.platform != "darwin" or platform.system() != "Darwin":
        raise RuntimeError("real G4-v2 pilot requires macOS/MPS")
    if args.route not in REQUIRED_ROUTES:
        raise ValueError("pilot worker route left the frozen matrix")
    import torch
    from kinetic_dense_cached_native_material_request import (
        synchronize_mps_device_completion_fence,
    )
    import worldfoam_native4d_public_quality_row as v1
    import worldfoam_native4d_public_quality_row_v2 as v2

    if not torch.backends.mps.is_available():
        raise RuntimeError("real G4-v2 pilot requires available MPS")
    mps_limit = _configure_mps_limit(torch, float(args.mps_memory_fraction))
    config_path = Path(args.config).resolve()
    config, base, _base_path = load_selected_ray_contract(config_path)
    protocol_path = Path(args.protocol).resolve()
    output_path = (
        ROOT
        / str(config["output_root"])
        / str(config["execution"]["pilot_scene"])
        / f"seed_{int(config['execution']['pilot_seed'])}"
        / str(args.route)
        / "g4_v2_row.json"
    ).resolve()
    request = v1.RowRequest(
        config_path=config_path,
        protocol_path=protocol_path,
        scene=str(config["execution"]["pilot_scene"]),
        seed=int(config["execution"]["pilot_seed"]),
        route=str(args.route),
        output_path=output_path,
        allow_local_mps_execution=True,
        dataset_capability_path=Path(args.dataset_capability).resolve(),
    )
    config_resolved, base_resolved, _base_path, workload, context = v2.resolve_v2_context(request)
    if config_resolved != config or base_resolved != base:
        raise ValueError("pilot v2 context changed during resolution")
    if (
        dict(context.work_plan.training_loss_contract) != dict(config["training_loss"])
        or context.work_plan.training_loss_contract["identifier"] != "rgb_mse_mean_v1"
    ):
        raise ValueError("pilot did not bind the v2 RGB-MSE contract")
    capability_path = (ROOT / str(config["execution"]["source_capability"])).resolve()
    capability = _load_json(capability_path)
    if capability != required_source_capability(config_path):
        raise ValueError("pilot source capability is missing or stale")

    route_started = time.perf_counter()
    baseline_mps = int(torch.mps.driver_allocated_memory())
    dataset = session = None
    peak_rss = 0
    peak_mps = baseline_mps
    try:
        setup_started = time.perf_counter()
        dataset = v1.load_public_quality_dataset(context)
        executor = v1._load_route_executor(context)
        executor_capability = v1._validate_executor_capability(
            executor.capability(context), context=context
        )
        if executor_capability.get("real_native") is not True:
            raise RuntimeError("pilot executor is not real-native")
        v1.validate_worldfoam_training_inputs(dataset, context=context)
        session = executor.open_session(context, dataset)
        setup_s = time.perf_counter() - setup_started

        work = context.work_plan.steps[0]
        requests = tuple(context.work_plan.iter_step_training_chunks(work))
        if len(requests) != 4 or sum(request.pixel_count for request in requests) != PILOT_TARGET_PIXELS:
            raise ArithmeticError("pilot step changed the selected-ray budget")
        training_started = time.perf_counter()
        session.begin_step(work)
        for selected_request in requests:
            session.accumulate_train_request(selected_request)
        session.finish_step(work)
        synchronize_mps_device_completion_fence()
        training_s = time.perf_counter() - training_started
        if (
            session._optimizer_steps != 1
            or session.state.geometry_update_count != 1
            or session._target_pixels != PILOT_TARGET_PIXELS
            or session._sampled_images != 4
            or session._pixel_chunks != 4
            or session._training_source_read_observation_count != PILOT_TARGET_PIXELS
            or session._training_full_frame_target_materialization_count != 0
            or len(session._promotion_receipts) != 1
        ):
            raise ArithmeticError("pilot did not complete exactly one selected optimizer step")
        peak_mps = max(peak_mps, int(torch.mps.driver_allocated_memory()))

        prepare_started = time.perf_counter()
        transition = dict(session.prepare_heldout_pilot_from_current_state())
        prepare_s = time.perf_counter() - prepare_started
        if (
            transition.get("optimizer_step") != 1
            or transition.get("training_finalized") is not False
            or transition.get("pilot_only") is not True
            or transition.get("frame_count") != PILOT_FRAME_COUNT
        ):
            raise ValueError("pilot transition receipt changed")
        maximum_tracks = int(session.maximum_heldout_tracks_per_cross_time_block())
        if maximum_tracks < PILOT_TRACK_COUNT:
            raise MemoryError("heldout spatial track bound fell below the pilot contract")
        pixel_ids = tuple(range(PILOT_TRACK_COUNT))
        spatial_started = time.perf_counter()
        cross_time = session.render_heldout_track_block_across_frames(
            camera_index=0,
            pixel_ids=pixel_ids,
        )
        target_track, target_receipt = session.read_heldout_target_track_block_across_frames(
            camera_index=0,
            pixel_ids=pixel_ids,
        )
        spatial_s = time.perf_counter() - spatial_started
        if (
            tuple(cross_time.shape) != (PILOT_FRAME_COUNT, PILOT_TRACK_COUNT, 3)
            or tuple(target_track.shape) != tuple(cross_time.shape)
            or not bool(torch.isfinite(cross_time).all().item())
            or not bool(torch.isfinite(target_track).all().item())
            or target_receipt.get("observation_count")
            != PILOT_FRAME_COUNT * PILOT_TRACK_COUNT
        ):
            raise ValueError("pilot spatial-major block changed shape or finiteness")
        peak_mps = max(peak_mps, int(torch.mps.driver_allocated_memory()))

        parity_started = time.perf_counter()
        parity_equal = True
        for frame_index in range(PILOT_FRAME_COUNT):
            old_request = v1.PixelChunkRequest(
                split="heldout",
                step=None,
                sample_slot=None,
                camera_index=0,
                frame_index=frame_index,
                pixel_start=0,
                pixel_count=1,
                image_height=transition["image_height"],
                image_width=transition["image_width"],
            )
            payload = v1._validate_pixel_payload(
                dataset.read_heldout_chunk(old_request), request=old_request
            )
            old_prediction = session.render_heldout_chunk(
                old_request,
                payload.rays_f32_cpu,
            )
            parity_equal = parity_equal and bool(
                torch.equal(old_prediction[0], cross_time[frame_index, 0])
            )
            del payload, old_prediction
        parity_s = time.perf_counter() - parity_started
        if not parity_equal:
            raise ArithmeticError("old frame-major and cross-time predictions differ bitwise")

        fence_started = time.perf_counter()
        synchronize_mps_device_completion_fence()
        fence_s = time.perf_counter() - fence_started
        partial = dict(session.heldout_spatial_major_partial_pilot_receipt())
        if partial.get("full_coverage") is not False:
            raise ValueError("partial heldout pilot receipt claimed full coverage")
        sampled_peak_rss, sampled_peak_mps = session._peak_sampler.stop()
        peak_rss = max(int(sampled_peak_rss), _process_peak_rss_bytes())
        peak_mps = max(peak_mps, int(sampled_peak_mps), int(torch.mps.driver_allocated_memory()))

        samples = tuple(work.batch.samples)
        selected_views = len({int(sample.view_index) for sample in samples})
        tracks_per_bundle = int(session.inputs.maximum_tracks_per_bundle)
        training_spatial_bundles = selected_views * math.ceil(
            context.work_plan.selected_pixels_per_spacetime_sample / tracks_per_bundle
        )
        framewise_calls = len(samples) * math.ceil(
            context.work_plan.selected_pixels_per_spacetime_sample / tracks_per_bundle
        )
        training_native_calls = (
            training_spatial_bundles
            if args.route == "worldfoam_native4d"
            else framewise_calls
        )
        native_counts = {
            "training_native_call_count": training_native_calls,
            "training_native_sample_count": PILOT_TARGET_PIXELS,
            "spatial_major_native_bundle_count": int(
                partial["cross_time_native_bundle_count"]
            ),
            "spatial_major_native_sample_count": int(
                partial["cross_time_native_sample_count"]
            ),
            "frame_major_parity_native_call_count": int(
                partial["old_frame_major_render_call_count"]
            ),
            "frame_major_parity_native_sample_count": int(
                partial["old_frame_major_observation_count"]
            ),
        }
        training_cold_tracks = (
            selected_views * context.work_plan.selected_pixels_per_spacetime_sample
        )
        compiler_counts = {
            "training_cold_track_compile_count": training_cold_tracks,
            "training_complete_camera_record_validation_count": (
                training_cold_tracks * PILOT_FRAME_COUNT
            ),
            "spatial_major_cold_track_compile_count": int(
                partial["cross_time_cold_track_compile_count"]
            ),
            "spatial_major_complete_camera_record_validation_count": int(
                partial["cross_time_complete_camera_record_validation_count"]
            ),
            # The old path deliberately recompiles one track once per frame;
            # each exact track compile validates the complete 300-record camera.
            "frame_major_parity_cold_track_compile_count": PILOT_FRAME_COUNT,
            "frame_major_parity_complete_camera_record_validation_count": (
                PILOT_FRAME_COUNT * PILOT_FRAME_COUNT
            ),
        }
        projected_blocks = math.ceil(FULL_HELDOUT_TRACK_COUNT / PILOT_TRACK_COUNT)
        projected_training = training_s * 300
        projected_heldout = spatial_s * projected_blocks
        projected_fixed = setup_s + prepare_s + parity_s + fence_s
        projected_total = PROJECTION_SAFETY_MULTIPLIER * (
            projected_training + projected_heldout + projected_fixed
        )
        projection = {
            "training_step_multiplier": 300,
            "full_heldout_spatial_track_count": FULL_HELDOUT_TRACK_COUNT,
            "measured_spatial_track_block_count": PILOT_TRACK_COUNT,
            "projected_heldout_block_count": projected_blocks,
            "safety_multiplier": PROJECTION_SAFETY_MULTIPLIER,
            "projected_training_seconds": projected_training,
            "projected_heldout_seconds": projected_heldout,
            "projected_fixed_seconds": projected_fixed,
            "projected_total_seconds": projected_total,
            "projected_full_row_hours": projected_total / 3600.0,
        }
        core = {
            "route": str(args.route),
            "stage_timings_s": {
                "dataset_and_session_setup": setup_s,
                "selected_training_step": training_s,
                "pilot_heldout_prepare": prepare_s,
                "spatial_major_track_block": spatial_s,
                "frame_major_parity": parity_s,
                "device_completion_fence": fence_s,
                "route_total": time.perf_counter() - route_started,
            },
            "native_counts": native_counts,
            "compiler_counts": compiler_counts,
            "projection": projection,
            "worker_process_peak_rss_bytes": peak_rss,
            "worker_mps_baseline_driver_bytes": baseline_mps,
            "worker_mps_peak_driver_bytes": peak_mps,
            "worker_mps_effective_limit_bytes": int(mps_limit["effective_limit_bytes"]),
            "native_library_sha256": str(session.native_library_sha256),
            "source_capability_sha256": str(capability["capability_sha256"]),
            "pilot_transition_receipt": transition,
            "spatial_replay_receipt": partial,
        }
        _worker_stage_handshake(
            path=Path(args.worker_stage),
            nonce=str(args.worker_nonce),
            route=str(args.route),
            core_sha256=canonical_sha256(core),
        )
        watchdog = _wait_for_watchdog(
            path=Path(args.watchdog_receipt),
            nonce=str(args.worker_nonce),
            route=str(args.route),
        )
        runtime_measurements = {
            "worker_process_peak_rss_bytes": peak_rss,
            "worker_mps_baseline_driver_bytes": baseline_mps,
            "worker_mps_peak_driver_bytes": peak_mps,
            "worker_mps_effective_limit_bytes": int(mps_limit["effective_limit_bytes"]),
            "parent_observed_process_group_peak_rss_bytes": int(
                watchdog["observed_process_group_peak_rss_bytes"]
            ),
            "parent_watchdog_sample_count": int(watchdog["sample_count"]),
            "parent_watchdog_timeout_seconds": int(watchdog["timeout_seconds"]),
            "parent_watchdog_rss_limit_bytes": int(watchdog["rss_limit_bytes"]),
            "completion_fenced": True,
        }
        report_payload = {
            "schema_version": 1,
            "kind": ROUTE_KIND,
            "route": str(args.route),
            "real_native": True,
            "backend": "metal",
            "device": "mps",
            "selected_training_optimizer_steps": 1,
            "selected_training_target_pixels": PILOT_TARGET_PIXELS,
            "selected_training_rgb_scalar_count": PILOT_RGB_SCALARS,
            "selected_training_spacetime_sample_count": 4,
            "training_loss_identifier": "rgb_mse_mean_v1",
            "training_loss_contract_sha256": canonical_sha256(config["training_loss"]),
            "heldout_frame_count_exercised": PILOT_FRAME_COUNT,
            "heldout_spatial_track_count_exercised": PILOT_TRACK_COUNT,
            "heldout_prediction_observation_count": PILOT_FRAME_COUNT * PILOT_TRACK_COUNT,
            "frame_major_parity_track_count": 1,
            "frame_major_parity_observation_count": PILOT_FRAME_COUNT,
            "frame_major_cross_time_bitwise_equal": True,
            "stage_timings_s": core["stage_timings_s"],
            "native_counts": native_counts,
            "compiler_counts": compiler_counts,
            "projection": projection,
            "runtime_measurements": runtime_measurements,
            "native_library_sha256": str(session.native_library_sha256),
            "source_capability_sha256": str(capability["capability_sha256"]),
            "worker_source_sha256": file_sha256(SCRIPT_PATH),
            "pilot_transition_receipt": transition,
            "spatial_replay_receipt": partial,
        }
        report = {
            **report_payload,
            "generation_digest": canonical_sha256(report_payload),
        }
        print(
            WORKER_RESULT_PREFIX
            + json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False),
            flush=True,
        )
        return 0
    finally:
        if session is not None:
            session.close()
        if dataset is not None:
            dataset.close()


def _validate_resource_args(args: argparse.Namespace) -> None:
    if args.minimum_free_disk_bytes < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise ValueError("free-disk guard cannot be relaxed")
    if args.minimum_available_memory_bytes < DEFAULT_MINIMUM_AVAILABLE_MEMORY_BYTES:
        raise ValueError("available-memory guard cannot be relaxed")
    if not 0 <= args.maximum_swap_used_bytes <= DEFAULT_MAXIMUM_SWAP_USED_BYTES:
        raise ValueError("swap guard cannot be relaxed")
    if not 0.0 < args.maximum_load_average <= DEFAULT_MAXIMUM_LOAD_AVERAGE:
        raise ValueError("load guard cannot be relaxed")
    if not 0.0 < args.mps_memory_fraction <= DEFAULT_MPS_MEMORY_FRACTION:
        raise ValueError("MPS fraction cannot exceed the pilot safe default")


def _host_guard(args: argparse.Namespace) -> dict[str, Any]:
    import run_worldfoam_memory_scaling_acceptance as shared

    _validate_resource_args(args)
    snapshot = shared._host_resource_snapshot()
    failures = list(
        shared._resource_guard_failures(
            snapshot,
            minimum_free_disk_bytes=args.minimum_free_disk_bytes,
            minimum_available_memory_bytes=args.minimum_available_memory_bytes,
            maximum_swap_used_bytes=args.maximum_swap_used_bytes,
            maximum_load_average=args.maximum_load_average,
        )
    )
    return {
        "policy": {
            "minimum_free_disk_bytes": args.minimum_free_disk_bytes,
            "minimum_available_memory_bytes": args.minimum_available_memory_bytes,
            "maximum_swap_used_bytes": args.maximum_swap_used_bytes,
            "maximum_load_average": args.maximum_load_average,
        },
        "snapshot": snapshot,
        "failures": failures,
        "passed_before_workers": not failures,
    }


def _execute(args: argparse.Namespace, plan: Mapping[str, Any]) -> int:
    if not args.allow_local_mps_execution:
        raise RuntimeError("pilot execution requires --allow-local-mps-execution")
    if sys.platform != "darwin" or platform.system() != "Darwin":
        raise RuntimeError("pilot execution requires a macOS/MPS host")
    if plan["runtime_blockers"]:
        raise RuntimeError(
            "pilot aborted before host/MPS work: "
            + ", ".join(str(value) for value in plan["runtime_blockers"])
        )
    host_guard = _host_guard(args)
    if not host_guard["passed_before_workers"]:
        raise RuntimeError(
            "pilot host resource guard failed: " + ", ".join(host_guard["failures"])
        )

    config_path = Path(args.config).resolve()
    config, base, _base_path = load_selected_ray_contract(config_path)
    protocol_path, dataset_capability_path = _coffee_paths(config, base)
    output_path = Path(args.output).resolve()
    source_before = _source_manifest(config_path, protocol_path)
    git_source = _git_source_identity()
    capability_path = (ROOT / str(config["execution"]["source_capability"])).resolve()
    capability = _load_json(capability_path)
    python = Path(args.python).expanduser().absolute()
    output_parent = output_path.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    raw_reports: dict[str, dict[str, Any]] = {}
    raw_logs: dict[str, dict[str, Any]] = {}
    final_log_paths: dict[str, Path] = {}
    with tempfile.TemporaryDirectory(
        prefix=".worldfoam-g4-v2-pilot-", dir=output_parent
    ) as raw_temp:
        temporary_root = Path(raw_temp)
        for route in REQUIRED_ROUTES:
            report, raw_log = _run_worker(
                python=python,
                config_path=config_path,
                protocol_path=protocol_path,
                dataset_capability_path=dataset_capability_path,
                route=route,
                temporary_root=temporary_root,
                mps_memory_fraction=float(args.mps_memory_fraction),
            )
            raw_reports[route] = report
            hidden_log = temporary_root / f".{route}.pilot_raw.{os.getpid()}.json"
            _write_json_atomic(hidden_log, raw_log, sort_keys=True)
            final_log_paths[route] = output_parent / f"{route}_pilot_raw_log.json"

        if _source_manifest(config_path, protocol_path) != source_before:
            raise RuntimeError("pilot source changed while workers were running")
        native_path = _native_library_path()
        native_digests = [
            raw_reports[route].get("native_library_sha256") for route in REQUIRED_ROUTES
        ]
        if native_digests != [file_sha256(native_path)] * len(REQUIRED_ROUTES):
            raise RuntimeError("pilot routes did not use the bound native library")
        for route in REQUIRED_ROUTES:
            hidden_log = temporary_root / f".{route}.pilot_raw.{os.getpid()}.json"
            final_log = final_log_paths[route]
            if final_log.exists():
                raise FileExistsError(f"pilot raw log already exists: {final_log}")
            os.replace(hidden_log, final_log)
            raw_logs[route] = {
                "path": _repo_display(final_log),
                "bytes": final_log.stat().st_size,
                "sha256": file_sha256(final_log),
                "worker_returncode": 0,
                "worker_report_generation_digest": raw_reports[route]["generation_digest"],
            }

    payload = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "kind": PILOT_KIND,
        "status": "pass",
        "scene": config["execution"]["pilot_scene"],
        "seed": config["execution"]["pilot_seed"],
        "v2_config_path": _repo_display(config_path),
        "v2_config_sha256": file_sha256(config_path),
        "source_capability_path": _repo_display(capability_path),
        "source_capability_sha256": capability["capability_sha256"],
        "training_loss_contract": dict(config["training_loss"]),
        "training_loss_contract_sha256": canonical_sha256(config["training_loss"]),
        "public_quality_evidence": False,
        "pilot_only": True,
        "spatial_major_full_temporal_heldout_exercised": True,
        "host_guard": host_guard,
        "source_binding": {
            **git_source,
            "source_manifest": source_before,
            "source_manifest_sha256": canonical_sha256(source_before),
            "parent_process_peak_rss_bytes": _process_peak_rss_bytes(),
        },
        "native_binding": {
            "module": "torch_world_foam_lane2_fused_slab.ops",
            "library_path": _repo_display(native_path),
            "library_bytes": native_path.stat().st_size,
            "library_sha256": file_sha256(native_path),
            "same_library_both_routes": True,
        },
        # Insertion order is a frozen part of the runner contract.  Do not
        # serialize the final receipt with sort_keys=True.
        "raw_logs": {route: raw_logs[route] for route in REQUIRED_ROUTES},
        "routes": {route: raw_reports[route] for route in REQUIRED_ROUTES},
    }
    receipt = {**payload, "generation_digest": canonical_sha256(payload)}
    failures = validate_pilot_receipt(
        receipt,
        config_path=config_path,
        artifact_path=output_path,
        verify_files=True,
    )
    if failures:
        for path in final_log_paths.values():
            path.unlink(missing_ok=True)
        raise ValueError("independent pilot self-validation failed: " + "; ".join(failures))
    if output_path.exists():
        raise FileExistsError(f"pilot receipt already exists: {output_path}")
    _write_json_atomic(output_path, receipt, sort_keys=False)
    verification = verify_pilot_file(output_path, config_path=config_path)
    if verification["status"] != "pass":
        output_path.unlink(missing_ok=True)
        for path in final_log_paths.values():
            path.unlink(missing_ok=True)
        raise ValueError(
            "serialized pilot failed independent verification: "
            + "; ".join(verification["failures"])
        )
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    default_python = ROOT / ".venv" / "bin" / "python"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-local-mps-execution", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--python",
        type=Path,
        default=default_python if default_python.is_file() else Path(sys.executable),
    )
    parser.add_argument("--minimum-free-disk-bytes", type=int, default=DEFAULT_MINIMUM_FREE_DISK_BYTES)
    parser.add_argument(
        "--minimum-available-memory-bytes",
        type=int,
        default=DEFAULT_MINIMUM_AVAILABLE_MEMORY_BYTES,
    )
    parser.add_argument("--maximum-swap-used-bytes", type=int, default=DEFAULT_MAXIMUM_SWAP_USED_BYTES)
    parser.add_argument("--maximum-load-average", type=float, default=DEFAULT_MAXIMUM_LOAD_AVERAGE)
    parser.add_argument("--mps-memory-fraction", type=float, default=DEFAULT_MPS_MEMORY_FRACTION)

    # Private worker protocol.  The parent supplies every field and binds it
    # through a random nonce plus a process-group watchdog receipt.
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--protocol", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--dataset-capability", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--route", choices=REQUIRED_ROUTES, help=argparse.SUPPRESS)
    parser.add_argument("--worker-nonce", help=argparse.SUPPRESS)
    parser.add_argument("--worker-stage", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--watchdog-receipt", type=Path, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args._worker:
        required = (
            args.protocol,
            args.dataset_capability,
            args.route,
            args.worker_nonce,
            args.worker_stage,
            args.watchdog_receipt,
        )
        if any(value is None for value in required):
            raise ValueError("private pilot worker arguments are incomplete")
        return _worker_main(args)
    torch_before = "torch" in sys.modules
    config_path = Path(args.config).resolve()
    config, _base, _base_path = load_selected_ray_contract(config_path)
    output = Path(
        args.output or (ROOT / str(config["execution"]["pilot_receipt"]))
    ).resolve()
    plan = build_plan(
        config_path=config_path,
        python=args.python,
        output_path=output,
        overwrite=args.overwrite,
    )
    if torch_before or "torch" in sys.modules:
        raise RuntimeError("G4-v2 pilot dry plan imported Torch")
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    return _execute(args, plan)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "PLAN_KIND",
    "PILOT_TRACK_COUNT",
    "build_plan",
    "main",
)
