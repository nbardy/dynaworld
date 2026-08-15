#!/usr/bin/env python3
"""Fresh-process producer for the WorldFoam 8/64/300 memory gate.

This runner is intentionally opt-in.  The parent process launches exactly one
trial at a time.  Each child imports a checked-in trial driver only after a
process-RSS baseline, requires the compiled kinetic ABI attestation, runs one
material-only coordinator step, and writes a nonce-bound receipt.  The parent
then normalizes those real coordinator reports and immediately runs the
fail-closed acceptance verifier.

The driver boundary exists because scene construction and native coordinator
dispatch are deployment concerns. Allocator limits and parent-watchdog evidence
remain producer-owned. A driver must expose exactly:

``run_worldfoam_memory_scaling_trial(context: Mapping[str, Any]) -> Mapping``

and return ``step_accounting``, ``material_state_accounting``,
``runtime_measurements``, ``maximum_node_count``, and
``persistent_world_geometry_tensor_bytes``.  It must also return the exact
``context["native_ops"]`` object as ``native_ops_used`` after passing that
object into the coordinator.  Hand-authored normalized trials are not
accepted by this producer.  The producer, rather than the driver, owns the
fresh-process RSS fields, the final MPS completion fence, and the autograd
saved-tensor hooks.  Execution also fails closed on Darwin memory/swap
pressure and installs a bounded per-process MPS allocation fraction.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import re
import resource
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

import verify_worldfoam_memory_scaling_acceptance as verifier


ROOT = Path(__file__).resolve().parents[2]
PRODUCER_PATH = Path(__file__).resolve()
WORKER_NONCE_ENV = "DYNAWORLD_WORLDFOAM_MEMORY_WORKER_NONCE"
DRIVER_FUNCTION = "run_worldfoam_memory_scaling_trial"
DRIVER_CAPABILITY_CONSTANT = "WORLDFOAM_MEMORY_SCALING_DRIVER_CAPABILITIES"
DRIVER_CAPABILITY_SCHEMA_VERSION = 3
DRIVER_PROTOCOL = "worldfoam-memory-scaling-trial-driver-v1"
COMMAND_PROTOCOL = "argv-no-shell+nonce-bound-receipt-v1"
DEFAULT_NATIVE_OPS_MODULE = "torch_world_foam_lane2_fused_slab.ops"
GIB = 1024**3
DEFAULT_MINIMUM_FREE_DISK_BYTES = 8 * GIB
DEFAULT_MINIMUM_AVAILABLE_MEMORY_BYTES = 8 * GIB
DEFAULT_MAXIMUM_SWAP_USED_BYTES = 2 * GIB
DEFAULT_MAXIMUM_LOAD_AVERAGE = 8.0
DEFAULT_MPS_MEMORY_FRACTION = 0.35
MAXIMUM_MPS_MEMORY_FRACTION = 0.50
MAXIMUM_MPS_WORKING_SET_BYTES = 2 * GIB
MPS_MEMORY_SAMPLE_INTERVAL_SECONDS = 0.005
WORKER_TIMEOUT_SECONDS = 30.0 * 60.0
WORKER_PROCESS_GROUP_RSS_LIMIT_BYTES = 4 * GIB
WORKER_WATCHDOG_POLL_INTERVAL_SECONDS = 0.25
WORKER_TERMINATION_GRACE_SECONDS = 5.0
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "worldfoam_memory_scaling"
    / "worldfoam_fixed_site_material_memory_scaling.json"
)

SOURCE_MANIFEST_FILES = (
    "src/train/paper_kinetic_fixed_site_material_step.py",
    "src/train/paper_kinetic_fixed_site_material_state.py",
    "src/train/paper_kinetic_replayable_observations.py",
    "src/train/paper_kinetic_lazy_program_bundles.py",
    "src/train/powerfoam_training_data.py",
    "research_experiments/world_foam_lane2/kinetic_compiled_cpu_artifact_store.py",
    "research_experiments/world_foam_lane2/kinetic_dense_cached_native_material_request.py",
    "research_experiments/world_foam_lane2/kinetic_native_material_step_executor.py",
    "research_experiments/world_foam_lane2/kinetic_native_equal_rank_lowering.py",
    "research_experiments/world_foam_lane2/kinetic_native_equal_rank_runtime_adapter.py",
    "research_experiments/world_foam_lane2/verify_worldfoam_memory_scaling_acceptance.py",
    "research_experiments/world_foam_lane2/worldfoam_memory_scaling_acceptance_v3.json",
    "research_experiments/world_foam_lane2/run_worldfoam_memory_scaling_acceptance.py",
)

NATIVE_SOURCE_ROOT = (
    ROOT
    / "third_party"
    / "fast-mac-gsplat"
    / "variants"
    / "world_foam_lane2_fused_slab_v0"
)
LOCAL_PYTHON_SOURCE_ROOTS = (
    ROOT / "research_experiments" / "world_foam_lane2",
    ROOT / "src" / "train",
    NATIVE_SOURCE_ROOT,
)

PRODUCER_OWNED_DRIVER_MEASUREMENT_KEYS = frozenset(
    {
        "fresh_process",
        "process_generation_id",
        "process_rss_baseline_bytes",
        "process_rss_peak_bytes",
        "completion_fenced_before_measurement",
        "completion_fence_provenance",
        "autograd_saved_tensor_hooks_enabled",
        "autograd_saved_tensor_measurement_provenance",
        "autograd_saved_tensor_count",
        "autograd_saved_tensor_peak_bytes",
        "source_manifest_sha256",
        "trial_driver_sha256",
        "trial_config_sha256",
        "hardware_fingerprint_sha256",
        "native_extension_sha256",
        "trial_command_sha256",
        "trial_execution_evidence_sha256",
        "mps_sampled_memory",
        "mps_memory_limit",
        "mps_memory_limit_sha256",
        "parent_watchdog",
        "parent_watchdog_evidence_sha256",
    }
)


class _SavedTensorAudit:
    """Fail-safe count/byte audit for tensors packed by autograd.

    The byte total is cumulative packed logical payload rather than an inferred
    allocator high-water.  The material-only gate requires both values to be
    zero, so the distinction cannot turn a failing saved-tensor regression into
    an accepted row.
    """

    def __init__(self) -> None:
        self.count = 0
        self.packed_tensor_bytes = 0

    def pack(self, tensor: Any) -> Any:
        self.count += 1
        self.packed_tensor_bytes += int(tensor.numel()) * int(tensor.element_size())
        return tensor

    @staticmethod
    def unpack(value: Any) -> Any:
        return value


class _MpsMemorySampler:
    """Sample public MPS counters without mislabelling them as exact peaks.

    PyTorch exposes current tensor and driver allocation counters on MPS, but
    no resettable high-water API.  A short-interval producer-owned sampler is
    therefore useful cross-F evidence, while the non-relaxable MPS allocation
    limit remains the hard upper bound.  The sampled maxima are lower bounds
    on an instantaneous peak and must never be promoted to exact allocator
    peaks by the driver or verifier.
    """

    def __init__(self, torch: ModuleType) -> None:
        self._torch = torch
        self._stop = threading.Event()
        self._sample_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self.sample_count = 0
        self.baseline_current_allocated_bytes = 0
        self.baseline_driver_allocated_bytes = 0
        self.maximum_current_allocated_bytes = 0
        self.maximum_driver_allocated_bytes = 0

    def _sample(self) -> None:
        with self._sample_lock:
            current = int(self._torch.mps.current_allocated_memory())
            driver = int(self._torch.mps.driver_allocated_memory())
            if current < 0 or driver < current:
                raise RuntimeError("MPS allocator counters returned an invalid ordering")
            self.sample_count += 1
            self.maximum_current_allocated_bytes = max(
                self.maximum_current_allocated_bytes,
                current,
            )
            self.maximum_driver_allocated_bytes = max(
                self.maximum_driver_allocated_bytes,
                driver,
            )

    def _run(self) -> None:
        try:
            while not self._stop.wait(MPS_MEMORY_SAMPLE_INTERVAL_SECONDS):
                self._sample()
        except BaseException as exc:
            self._error = exc
            self._stop.set()

    def __enter__(self) -> _MpsMemorySampler:
        self._sample()
        self.baseline_current_allocated_bytes = (
            self.maximum_current_allocated_bytes
        )
        self.baseline_driver_allocated_bytes = self.maximum_driver_allocated_bytes
        self._thread = threading.Thread(
            target=self._run,
            name="worldfoam-mps-memory-sampler",
            daemon=True,
        )
        self._thread.start()
        return self

    def capture_after_completion_fence(self) -> None:
        self._sample()

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                raise RuntimeError("MPS memory sampler did not stop")
        if self._error is not None:
            raise RuntimeError("MPS memory sampler failed") from self._error

    def receipt(self) -> dict[str, Any]:
        if self.sample_count < 2:
            raise RuntimeError("MPS memory sampler did not capture a trial interval")
        return {
            "measurement_kind": "producer-thread-sampled-high-water-lower-bound-v1",
            "sampling_interval_ms": MPS_MEMORY_SAMPLE_INTERVAL_SECONDS * 1000.0,
            "sample_count": self.sample_count,
            "baseline_current_allocated_bytes": (
                self.baseline_current_allocated_bytes
            ),
            "maximum_current_allocated_bytes": self.maximum_current_allocated_bytes,
            "current_allocated_sampled_growth_bytes": max(
                0,
                self.maximum_current_allocated_bytes
                - self.baseline_current_allocated_bytes,
            ),
            "baseline_driver_allocated_bytes": (
                self.baseline_driver_allocated_bytes
            ),
            "maximum_driver_allocated_bytes": self.maximum_driver_allocated_bytes,
            "driver_allocated_sampled_growth_bytes": max(
                0,
                self.maximum_driver_allocated_bytes
                - self.baseline_driver_allocated_bytes,
            ),
            "exact_peak_claimed": False,
            "completion_fenced_before_final_sample": True,
        }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_file(path: Path, *, name: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{name} is not a file: {resolved}")
    return resolved


def _repo_file(path: Path, *, name: str) -> Path:
    resolved = _resolved_file(path, name=name)
    try:
        resolved.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError(f"{name} must live under the dynaworld root") from exc
    return resolved


def _literal_driver_capabilities(path: Path) -> Mapping[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    literals: list[ast.AST] = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id == DRIVER_CAPABILITY_CONSTANT
            for target in node.targets
        ):
            literals.append(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == DRIVER_CAPABILITY_CONSTANT
            and node.value is not None
        ):
            literals.append(node.value)
    if len(literals) != 1:
        raise ValueError(
            f"trial driver must define exactly one literal {DRIVER_CAPABILITY_CONSTANT}"
        )
    try:
        value = ast.literal_eval(literals[0])
    except (TypeError, ValueError) as exc:
        raise ValueError("trial driver capability manifest must be an AST literal") from exc
    return _require_mapping(value, "trial driver capability manifest")


def _validate_driver_capabilities(capabilities: Mapping[str, Any]) -> None:
    expected_top_keys = {
        "schema_version",
        "driver_protocol",
        "supported_backends",
        "selected_pixel_target_access",
    }
    if set(capabilities) != expected_top_keys:
        raise ValueError("trial driver capability manifest has noncanonical top-level keys")
    if capabilities.get("schema_version") != DRIVER_CAPABILITY_SCHEMA_VERSION:
        raise ValueError("trial driver capability schema is missing or stale")
    if capabilities.get("driver_protocol") != DRIVER_PROTOCOL:
        raise ValueError("trial driver protocol is missing or wrong")
    if tuple(capabilities.get("supported_backends", ())) != ("mps",):
        raise ValueError("trial driver capabilities must be bound only to MPS")
    selected_pixels = _require_mapping(
        capabilities.get("selected_pixel_target_access"),
        "trial driver selected_pixel_target_access",
    )
    if set(selected_pixels) != {
        "implemented",
        "access_mode",
        "full_frame_materialization_count",
        "preserves_request_order_and_duplicates",
        "source_budget_enforced_before_allocation",
        "contract",
    }:
        raise ValueError("selected-pixel target capability fields are noncanonical")
    if (
        selected_pixels.get("implemented") is not True
        or selected_pixels.get("access_mode") != "direct_pixels"
        or selected_pixels.get("full_frame_materialization_count") != 0
        or selected_pixels.get("preserves_request_order_and_duplicates") is not True
        or selected_pixels.get("source_budget_enforced_before_allocation") is not True
        or selected_pixels.get("contract") != "PowerFoamSelectedPixelRead/v1"
    ):
        raise ValueError("trial driver lacks the sealed direct selected-pixel contract")


def _load_driver_capabilities(path: Path) -> dict[str, Any]:
    capabilities = dict(_literal_driver_capabilities(path))
    _validate_driver_capabilities(capabilities)
    return capabilities


def _driver_capability_blockers(
    capabilities: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[str, ...]:
    blockers: list[str] = []
    if contract.get("require_selected_pixel_target_access") is True:
        selected_pixels = _require_mapping(
            capabilities.get("selected_pixel_target_access"),
            "selected_pixel_target_access",
        )
        if (
            selected_pixels.get("implemented") is not True
            or selected_pixels.get("access_mode") != "direct_pixels"
            or selected_pixels.get("full_frame_materialization_count") != 0
            or selected_pixels.get("source_budget_enforced_before_allocation") is not True
        ):
            blockers.append("selected_pixel_target_access")
    return tuple(blockers)


def _native_source_files() -> tuple[Path, ...]:
    roots = (
        NATIVE_SOURCE_ROOT / "torch_world_foam_lane2_fused_slab",
        NATIVE_SOURCE_ROOT / "csrc",
    )
    suffixes = {".cpp", ".h", ".hpp", ".metal", ".mm", ".py"}
    paths = {
        path.resolve()
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and path.suffix in suffixes
    }
    paths.add(
        _resolved_file(
            NATIVE_SOURCE_ROOT / "setup.py",
            name="native extension build source",
        )
    )
    return tuple(sorted(paths))


def _local_module_source_files(module_name: str) -> tuple[Path, ...]:
    """Resolve a local module and every executed ancestor package initializer."""

    if not module_name or any(
        not part.isidentifier() for part in module_name.split(".")
    ):
        return ()
    relative = Path(*module_name.split("."))
    parts = module_name.split(".")
    candidates: set[Path] = set()
    for root in LOCAL_PYTHON_SOURCE_ROOTS:
        for candidate in (
            root / relative.with_suffix(".py"),
            root / relative / "__init__.py",
        ):
            if candidate.is_file():
                candidates.add(candidate.resolve())
        for depth in range(1, len(parts)):
            initializer = root.joinpath(*parts[:depth], "__init__.py")
            if initializer.is_file():
                candidates.add(initializer.resolve())
    return tuple(sorted(candidates))


def _local_python_source_closure(roots: Sequence[Path]) -> tuple[Path, ...]:
    """AST-walk the local import closure without importing experiment code.

    This binds every local Python module that can influence the producer,
    driver, compiler, target path, or verifier.  Runtime package sources for
    the native extension are covered separately by :func:`_native_source_files`.
    """

    pending = [
        _resolved_file(path, name="source-manifest Python root")
        for path in roots
    ]
    seen: set[Path] = set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeError) as exc:
            raise RuntimeError(f"cannot parse source-manifest member {path}") from exc
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if node.level:
                    relative = next(
                        (
                            path.relative_to(root)
                            for root in LOCAL_PYTHON_SOURCE_ROOTS
                            if path.is_relative_to(root)
                        ),
                        None,
                    )
                    if relative is None:
                        raise ValueError(
                            f"relative import is outside local source roots: {path}"
                        )
                    parents = relative.parent.parts
                    ascent = node.level - 1
                    if ascent > len(parents):
                        raise ValueError(
                            f"relative import escapes its local source root: {path}"
                        )
                    base = parents[: len(parents) - ascent]
                    module_name = ".".join(
                        (*base, *(() if not node.module else node.module.split(".")))
                    )
                if module_name:
                    imported.add(module_name)
                imported.update(
                    ".".join(part for part in (module_name, alias.name) if part)
                    for alias in node.names
                    if alias.name != "*"
                )
        for module_name in imported:
            pending.extend(
                path
                for path in _local_module_source_files(module_name)
                if path not in seen
            )
    return tuple(sorted(seen))


def build_source_manifest(
    *,
    trial_driver_path: Path,
    trial_config_path: Path,
) -> tuple[tuple[dict[str, Any], ...], str]:
    driver = _repo_file(trial_driver_path, name="trial driver")
    config = _repo_file(trial_config_path, name="trial config")
    declared = tuple(ROOT / relative for relative in SOURCE_MANIFEST_FILES)
    native_sources = _native_source_files()
    paths = {
        *declared,
        *_local_python_source_closure(
            tuple(
                path
                for path in (*declared, *native_sources, driver)
                if path.suffix == ".py"
            )
        ),
        *native_sources,
        driver,
        config,
    }
    records: list[dict[str, Any]] = []
    for path in sorted(paths):
        resolved = _resolved_file(path, name="source-manifest member")
        try:
            label = resolved.relative_to(ROOT).as_posix()
        except ValueError:
            label = f"external/{resolved.name}"
        records.append(
            {
                "path": label,
                "size_bytes": resolved.stat().st_size,
                "sha256": _file_sha256(resolved),
            }
        )
    frozen = tuple(records)
    return frozen, _sha256_bytes(_canonical_bytes(frozen))


def _maximum_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _checked_output(
    argv: Sequence[str],
    *,
    timeout_seconds: float = 10.0,
) -> str:
    try:
        return subprocess.check_output(
            tuple(argv),
            text=True,
            timeout=timeout_seconds,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"host resource command failed: {' '.join(argv)}") from exc


def _parse_process_group_rss_bytes(output: str, process_group_id: int) -> int:
    total_kib = 0
    for line in output.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            row_process_group_id, rss_kib = (int(value) for value in fields)
        except ValueError:
            continue
        if row_process_group_id == process_group_id and rss_kib > 0:
            total_kib += rss_kib
    return total_kib * 1024


def _process_group_rss_bytes(process_group_id: int) -> int:
    return _parse_process_group_rss_bytes(
        _checked_output(
            ("ps", "-axo", "pgid=,rss="),
            timeout_seconds=2.0,
        ),
        process_group_id,
    )


def _worker_watchdog_violation(*, elapsed_seconds: float, group_rss_bytes: int) -> str:
    if group_rss_bytes > WORKER_PROCESS_GROUP_RSS_LIMIT_BYTES:
        return (
            "worker process-group sampled RSS exceeded the configured ceiling: "
            f"{group_rss_bytes} > {WORKER_PROCESS_GROUP_RSS_LIMIT_BYTES}"
        )
    if elapsed_seconds > WORKER_TIMEOUT_SECONDS:
        return (
            "worker exceeded the hard wall-time ceiling: "
            f"{elapsed_seconds:.1f}s > {WORKER_TIMEOUT_SECONDS:.1f}s"
        )
    return ""


def _terminate_worker_process_group(process: subprocess.Popen[Any]) -> None:
    process_group_id = process.pid
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=WORKER_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process_group_id, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=WORKER_TERMINATION_GRACE_SECONDS)


def _run_guarded_worker(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    stdout: Any,
    stderr: Any,
) -> dict[str, Any]:
    """Run one fresh child under a hard timeout and sampled RSS watchdog."""

    process = subprocess.Popen(
        tuple(argv),
        cwd=cwd,
        env=dict(env),
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        start_new_session=True,
    )
    start = time.monotonic()
    sampled_rss_high_water = 0
    sample_count = 0
    try:
        while True:
            returncode = process.poll()
            group_rss = _process_group_rss_bytes(process.pid)
            if group_rss <= 0 and returncode is None:
                returncode = process.poll()
                if returncode is None:
                    raise RuntimeError(
                        "parent watchdog could not observe positive live process-group RSS"
                    )
            sample_count += 1
            sampled_rss_high_water = max(sampled_rss_high_water, group_rss)
            elapsed = time.monotonic() - start
            violation = _worker_watchdog_violation(
                elapsed_seconds=elapsed,
                group_rss_bytes=group_rss,
            )
            if violation:
                raise RuntimeError(violation)
            if returncode is not None:
                if group_rss != 0:
                    raise RuntimeError(
                        "worker leader exited while its process group remained resident"
                    )
                return {
                    "returncode": int(returncode),
                    "elapsed_seconds": elapsed,
                    "rss_measurement_kind": (
                        verifier.PARENT_WATCHDOG_RSS_MEASUREMENT_KIND
                    ),
                    "rss_sampling_interval_seconds": (
                        WORKER_WATCHDOG_POLL_INTERVAL_SECONDS
                    ),
                    "sampled_process_group_rss_high_water_bytes": (
                        sampled_rss_high_water
                    ),
                    "sample_count": sample_count,
                    "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
                    "worker_process_group_rss_limit_bytes": (
                        WORKER_PROCESS_GROUP_RSS_LIMIT_BYTES
                    ),
                    "watchdog_completed": True,
                    "process_group_empty_after_exit": True,
                    "worker_terminated_by_watchdog": False,
                }
            time.sleep(WORKER_WATCHDOG_POLL_INTERVAL_SECONDS)
    except BaseException:
        _terminate_worker_process_group(process)
        raise


def _darwin_available_memory_bytes(vm_stat_output: str) -> int:
    page_match = re.search(r"page size of\s+(\d+)\s+bytes", vm_stat_output)
    if page_match is None:
        raise ValueError("could not parse macOS VM page size")
    page_counts: dict[str, int] = {}
    for line in vm_stat_output.splitlines():
        match = re.match(r"([^:]+):\s+([0-9]+)\.?$", line.strip())
        if match is not None:
            page_counts[match.group(1)] = int(match.group(2))
    required_keys = ("Pages free", "Pages inactive", "Pages speculative")
    if not all(key in page_counts for key in required_keys):
        raise ValueError("could not parse conservative macOS available pages")
    return int(page_match.group(1)) * sum(page_counts[key] for key in required_keys)


def _darwin_swap_used_bytes(swap_output: str) -> int:
    match = re.search(r"used\s*=\s*([0-9.]+)([KMGT])", swap_output)
    if match is None:
        raise ValueError("could not parse macOS swap usage")
    multiplier = {
        "K": 1024,
        "M": 1024**2,
        "G": 1024**3,
        "T": 1024**4,
    }[match.group(2)]
    return int(float(match.group(1)) * multiplier)


def _host_resource_snapshot() -> dict[str, Any]:
    if sys.platform != "darwin":
        raise RuntimeError("the MPS memory producer requires a macOS host audit")
    load_1m, load_5m, load_15m = os.getloadavg()
    return {
        "platform": sys.platform,
        "available_memory_bytes": _darwin_available_memory_bytes(
            _checked_output(("vm_stat",))
        ),
        "swap_used_bytes": _darwin_swap_used_bytes(
            _checked_output(("sysctl", "-n", "vm.swapusage"))
        ),
        "free_disk_bytes": int(shutil.disk_usage(ROOT).free),
        "load_average_1m": float(load_1m),
        "load_average_5m": float(load_5m),
        "load_average_15m": float(load_15m),
    }


def _resource_guard_failures(
    snapshot: Mapping[str, Any],
    *,
    minimum_free_disk_bytes: int,
    minimum_available_memory_bytes: int,
    maximum_swap_used_bytes: int,
    maximum_load_average: float,
) -> tuple[str, ...]:
    failures: list[str] = []
    if snapshot.get("platform") != "darwin":
        failures.append("platform")
    if int(snapshot.get("free_disk_bytes", 0)) < minimum_free_disk_bytes:
        failures.append("free_disk_bytes")
    if int(snapshot.get("available_memory_bytes", 0)) < minimum_available_memory_bytes:
        failures.append("available_memory_bytes")
    if int(snapshot.get("swap_used_bytes", sys.maxsize)) > maximum_swap_used_bytes:
        failures.append("swap_used_bytes")
    if float(snapshot.get("load_average_1m", float("inf"))) > maximum_load_average:
        failures.append("load_average_1m")
    return tuple(failures)


def _validate_resource_policy(
    *,
    minimum_free_disk_bytes: int,
    minimum_available_memory_bytes: int,
    maximum_swap_used_bytes: int,
    maximum_load_average: float,
) -> None:
    if minimum_free_disk_bytes < DEFAULT_MINIMUM_FREE_DISK_BYTES:
        raise ValueError("free-disk guard cannot be relaxed below its safe default")
    if minimum_available_memory_bytes < DEFAULT_MINIMUM_AVAILABLE_MEMORY_BYTES:
        raise ValueError("available-memory guard cannot be relaxed below its safe default")
    if not 0 <= maximum_swap_used_bytes <= DEFAULT_MAXIMUM_SWAP_USED_BYTES:
        raise ValueError("swap guard cannot be relaxed above its safe default")
    if not 0.0 < maximum_load_average <= DEFAULT_MAXIMUM_LOAD_AVERAGE:
        raise ValueError("load guard cannot be relaxed above its safe default")


def _configure_mps_memory_limit(
    torch: ModuleType,
    requested_fraction: float,
) -> dict[str, int | float]:
    if not 0.0 < requested_fraction <= MAXIMUM_MPS_MEMORY_FRACTION:
        raise ValueError(
            "MPS memory fraction must be positive and no greater than "
            f"{MAXIMUM_MPS_MEMORY_FRACTION:.2f}"
        )
    setter = getattr(torch.mps, "set_per_process_memory_fraction", None)
    if not callable(setter):
        raise RuntimeError("PyTorch lacks the required MPS per-process memory limiter")
    recommended = getattr(torch.mps, "recommended_max_memory", None)
    if not callable(recommended):
        raise RuntimeError("PyTorch lacks the required MPS working-set query")
    recommended_bytes = int(recommended())
    if recommended_bytes < 1:
        raise RuntimeError("MPS recommended working set must be positive")
    absolute_fraction = float(MAXIMUM_MPS_WORKING_SET_BYTES) / float(
        recommended_bytes
    )
    effective_fraction = min(float(requested_fraction), absolute_fraction)
    if not 0.0 < effective_fraction <= MAXIMUM_MPS_MEMORY_FRACTION:
        raise RuntimeError("effective MPS memory fraction is outside the safe range")
    setter(effective_fraction)
    return {
        "requested_fraction": float(requested_fraction),
        "effective_fraction": effective_fraction,
        "recommended_max_memory_bytes": recommended_bytes,
        "absolute_working_set_limit_bytes": MAXIMUM_MPS_WORKING_SET_BYTES,
        "effective_working_set_limit_bytes": min(
            MAXIMUM_MPS_WORKING_SET_BYTES,
            int(float(requested_fraction) * recommended_bytes),
        ),
    }


def _hardware_record(
    backend: str,
    *,
    mps_memory_fraction: float,
) -> dict[str, Any]:
    if backend != "mps":
        raise ValueError("the current producer is bound only to the Metal/MPS ABI")
    torch = importlib.import_module("torch")
    record: dict[str, Any] = {
        "backend": backend,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": str(torch.__version__),
    }
    if not bool(torch.backends.mps.is_available()):
        raise RuntimeError("MPS backend requested but unavailable")
    memory_limit = _configure_mps_memory_limit(torch, mps_memory_fraction)
    record["device"] = "Apple MPS"
    record["mps_memory_limit"] = memory_limit
    return record


def _load_driver(path: Path) -> ModuleType:
    name = f"_worldfoam_memory_trial_driver_{_file_sha256(path)[:16]}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load trial driver {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _attest_native_extension(module_name: str) -> tuple[ModuleType, Path]:
    native_ops = importlib.import_module(module_name)
    attest = getattr(
        native_ops,
        "assert_kinetic_memory_light_compiled_abi_registered",
        None,
    )
    if not callable(attest):
        raise TypeError("native ops module lacks compiled kinetic ABI attestation")
    attest()
    module_path = _resolved_file(Path(native_ops.__file__), name="native ops module")
    candidates = tuple(sorted(module_path.parent.glob("_C*.so")))
    if len(candidates) != 1:
        raise RuntimeError("compiled native package must contain exactly one _C library")
    extension = candidates[0].resolve()
    try:
        extension.relative_to(NATIVE_SOURCE_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("native extension is outside the bound WorldFoam variant") from exc
    return native_ops, extension


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _reject_producer_owned_measurement_claims(
    runtime_measurements: Mapping[str, Any],
) -> None:
    claims = sorted(
        PRODUCER_OWNED_DRIVER_MEASUREMENT_KEYS.intersection(runtime_measurements)
    )
    if claims:
        raise ValueError(
            "trial driver claimed producer-owned measurements: "
            + ", ".join(claims)
        )


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _command_sha256(argv: Sequence[str]) -> str:
    return _sha256_bytes(_canonical_bytes(tuple(argv)))


def _nonce_bound_command_sha256(argv: Sequence[str]) -> str:
    normalized = list(argv)
    try:
        value_index = normalized.index("--expected-command-sha256") + 1
        normalized[value_index] = "PLACEHOLDER"
    except (ValueError, IndexError) as exc:
        raise ValueError("worker argv lacks its expected-command binding") from exc
    return _command_sha256(normalized)


def _worker(args: argparse.Namespace) -> int:
    nonce = str(args.worker_nonce)
    if not nonce or os.environ.get(WORKER_NONCE_ENV) != nonce:
        raise PermissionError("worker nonce is missing or foreign")
    driver_path = _repo_file(args.trial_driver, name="trial driver")
    config_path = _repo_file(args.trial_config, name="trial config")
    receipt_path = args.receipt.resolve()
    manifest, manifest_sha256 = build_source_manifest(
        trial_driver_path=driver_path,
        trial_config_path=config_path,
    )
    driver_capabilities = _load_driver_capabilities(driver_path)
    driver_capabilities_sha256 = _sha256_bytes(
        _canonical_bytes(driver_capabilities)
    )
    if manifest_sha256 != args.expected_source_manifest_sha256:
        raise RuntimeError("child source manifest differs from the parent launch")
    command_sha256 = _nonce_bound_command_sha256((sys.executable, *sys.argv))
    if command_sha256 != args.expected_command_sha256:
        raise RuntimeError("worker argv differs from the nonce-bound parent command")

    host_resource_preflight = _guard_host(
        minimum_free_disk_bytes=args.minimum_free_disk_bytes,
        minimum_available_memory_bytes=args.minimum_available_memory_bytes,
        maximum_swap_used_bytes=args.maximum_swap_used_bytes,
        maximum_load_average=args.maximum_load_average,
    )
    process_rss_baseline = _maximum_rss_bytes()
    hardware = _hardware_record(
        args.backend,
        mps_memory_fraction=args.mps_memory_fraction,
    )
    torch = importlib.import_module("torch")
    native_ops, native_extension = _attest_native_extension(args.native_ops_module)
    native_extension_sha256 = _file_sha256(native_extension)
    driver_module = _load_driver(driver_path)
    driver = getattr(driver_module, DRIVER_FUNCTION, None)
    if not callable(driver):
        raise TypeError(f"trial driver must expose callable {DRIVER_FUNCTION}")
    config = verifier.load_json_object(config_path)
    context = {
        "backend": args.backend,
        "frame_count": args.frame_count,
        "repeat_index": args.repeat_index,
        "trial_config": config,
        "trial_config_path": str(config_path),
        "driver_capabilities": driver_capabilities,
        "driver_capabilities_sha256": driver_capabilities_sha256,
        "source_manifest_sha256": manifest_sha256,
        "worker_nonce": nonce,
        "material_only_scope": True,
        "require_real_native": True,
        "native_ops": native_ops,
        "native_ops_module": args.native_ops_module,
        "native_extension_path": str(native_extension),
        "native_extension_sha256": native_extension_sha256,
        "hardware": hardware,
        "host_resource_preflight": host_resource_preflight,
        "mps_memory_limit": hardware["mps_memory_limit"],
    }
    saved_tensor_audit = _SavedTensorAudit()
    saved_tensors_hooks = getattr(
        getattr(torch.autograd, "graph", None),
        "saved_tensors_hooks",
        None,
    )
    if not callable(saved_tensors_hooks):
        raise RuntimeError("PyTorch lacks producer-owned saved_tensors_hooks")
    with _MpsMemorySampler(torch) as mps_memory_sampler:
        with saved_tensors_hooks(saved_tensor_audit.pack, saved_tensor_audit.unpack):
            raw_result = _require_mapping(driver(context), "trial driver result")
        torch.mps.synchronize()
        mps_memory_sampler.capture_after_completion_fence()
    mps_sampled_memory = mps_memory_sampler.receipt()
    process_rss_peak = _maximum_rss_bytes()
    if raw_result.get("native_ops_used") is not native_ops:
        raise ValueError("trial driver did not return the bound native_ops identity")
    if process_rss_peak <= process_rss_baseline:
        raise RuntimeError("fresh child did not observe a positive process-RSS peak")

    step_accounting = _require_mapping(
        raw_result.get("step_accounting"), "driver step_accounting"
    )
    material_state_accounting = _require_mapping(
        raw_result.get("material_state_accounting"),
        "driver material_state_accounting",
    )
    runtime_measurements = dict(
        _require_mapping(
            raw_result.get("runtime_measurements"),
            "driver runtime_measurements",
        )
    )
    _reject_producer_owned_measurement_claims(runtime_measurements)
    if runtime_measurements.get("fake_native_backend") is not False:
        raise ValueError("producer refuses fake-native measurements")
    if runtime_measurements.get("native_runtime_verified") is not True:
        raise ValueError("driver did not verify the native runtime")
    if runtime_measurements.get("production_coordinator_integrated") is not True:
        raise ValueError("driver did not execute the production coordinator")

    driver_sha256 = _file_sha256(driver_path)
    config_sha256 = _file_sha256(config_path)
    hardware_sha256 = _sha256_bytes(_canonical_bytes(hardware))
    process_generation_id = _sha256_bytes(
        _canonical_bytes(
            {
                "nonce": nonce,
                "pid": os.getpid(),
                "start_time_ns": args.worker_start_time_ns,
                "command_sha256": command_sha256,
            }
        )
    )
    evidence_payload = {
        "nonce": nonce,
        "pid": os.getpid(),
        "frame_count": args.frame_count,
        "repeat_index": args.repeat_index,
        "command_sha256": command_sha256,
        "source_manifest_sha256": manifest_sha256,
        "trial_driver_sha256": driver_sha256,
        "trial_config_sha256": config_sha256,
        "driver_capabilities_sha256": driver_capabilities_sha256,
        "hardware_fingerprint_sha256": hardware_sha256,
        "native_extension_sha256": native_extension_sha256,
        "process_rss_baseline_bytes": process_rss_baseline,
        "process_rss_peak_bytes": process_rss_peak,
        "step_accounting": step_accounting,
        "material_state_accounting": material_state_accounting,
        "runtime_measurements_from_driver": runtime_measurements,
        "producer_saved_tensor_count": saved_tensor_audit.count,
        "producer_saved_tensor_bytes": saved_tensor_audit.packed_tensor_bytes,
        "mps_sampled_memory": mps_sampled_memory,
        "host_resource_preflight": host_resource_preflight,
        "mps_memory_limit": hardware["mps_memory_limit"],
    }
    execution_evidence_sha256 = _sha256_bytes(_canonical_bytes(evidence_payload))
    runtime_measurements.update(
        {
            "fresh_process": True,
            "process_generation_id": process_generation_id,
            "process_rss_baseline_bytes": process_rss_baseline,
            "process_rss_peak_bytes": process_rss_peak,
            "completion_fenced_before_measurement": True,
            "completion_fence_provenance": "producer-torch.mps.synchronize-v1",
            "autograd_saved_tensor_hooks_enabled": True,
            "autograd_saved_tensor_measurement_provenance": (
                "producer-saved-tensors-hooks-cumulative-logical-bytes-v1"
            ),
            "autograd_saved_tensor_count": saved_tensor_audit.count,
            "autograd_saved_tensor_peak_bytes": (
                saved_tensor_audit.packed_tensor_bytes
            ),
            "source_manifest_sha256": manifest_sha256,
            "trial_driver_sha256": driver_sha256,
            "trial_config_sha256": config_sha256,
            "hardware_fingerprint_sha256": hardware_sha256,
            "native_extension_sha256": native_extension_sha256,
            "trial_command_sha256": command_sha256,
            "trial_execution_evidence_sha256": execution_evidence_sha256,
            "mps_sampled_memory": mps_sampled_memory,
            "mps_memory_limit": dict(hardware["mps_memory_limit"]),
        }
    )
    normalized = verifier.build_trial_from_fixed_site_accounting(
        frame_count=args.frame_count,
        repeat_index=args.repeat_index,
        maximum_node_count=int(raw_result["maximum_node_count"]),
        persistent_world_geometry_tensor_bytes=int(
            raw_result["persistent_world_geometry_tensor_bytes"]
        ),
        step_accounting=step_accounting,
        material_state_accounting=material_state_accounting,
        runtime_measurements=runtime_measurements,
    )
    receipt = {
        "producer_name": verifier.PRODUCER_NAME,
        "producer_schema_version": verifier.PRODUCER_SCHEMA_VERSION,
        "nonce": nonce,
        "pid": os.getpid(),
        "frame_count": args.frame_count,
        "repeat_index": args.repeat_index,
        "command_sha256": command_sha256,
        "execution_evidence_sha256": execution_evidence_sha256,
        "source_manifest": manifest,
        "driver_capabilities": driver_capabilities,
        "driver_capabilities_sha256": driver_capabilities_sha256,
        "hardware": hardware,
        "host_resource_preflight": host_resource_preflight,
        "mps_sampled_memory": mps_sampled_memory,
        "native_extension_path": str(native_extension),
        "normalized_trial": normalized,
    }
    _write_json_atomic(receipt_path, receipt)
    return 0


def _guard_host(
    *,
    minimum_free_disk_bytes: int,
    minimum_available_memory_bytes: int,
    maximum_swap_used_bytes: int,
    maximum_load_average: float,
) -> dict[str, Any]:
    _validate_resource_policy(
        minimum_free_disk_bytes=minimum_free_disk_bytes,
        minimum_available_memory_bytes=minimum_available_memory_bytes,
        maximum_swap_used_bytes=maximum_swap_used_bytes,
        maximum_load_average=maximum_load_average,
    )
    snapshot = _host_resource_snapshot()
    failures = _resource_guard_failures(
        snapshot,
        minimum_free_disk_bytes=minimum_free_disk_bytes,
        minimum_available_memory_bytes=minimum_available_memory_bytes,
        maximum_swap_used_bytes=maximum_swap_used_bytes,
        maximum_load_average=maximum_load_average,
    )
    if failures:
        raise RuntimeError(
            "memory gate host failed its Darwin/MPS resource guard: "
            + ", ".join(failures)
        )
    return snapshot


def _producer_binding(
    *,
    manifest: Sequence[Mapping[str, Any]],
    manifest_sha256: str,
    driver_path: Path,
    config_path: Path,
    first_receipt: Mapping[str, Any],
    python_executable: Path,
) -> dict[str, Any]:
    trial = _require_mapping(first_receipt["normalized_trial"], "normalized trial")
    measurement = _require_mapping(trial["measurement"], "normalized measurement")
    hardware = _require_mapping(first_receipt["hardware"], "hardware")
    return {
        "producer_name": verifier.PRODUCER_NAME,
        "schema_version": verifier.PRODUCER_SCHEMA_VERSION,
        "fresh_process_per_trial": True,
        "material_only_scope": True,
        "real_native_required": True,
        "source_manifest_sha256": manifest_sha256,
        "source_manifest_file_count": len(manifest),
        "trial_driver_path": str(driver_path),
        "trial_driver_sha256": measurement["trial_driver_sha256"],
        "trial_config_path": str(config_path),
        "trial_config_sha256": measurement["trial_config_sha256"],
        "driver_capabilities_sha256": first_receipt[
            "driver_capabilities_sha256"
        ],
        "hardware_fingerprint_sha256": measurement["hardware_fingerprint_sha256"],
        "hardware_summary": json.dumps(hardware, sort_keys=True),
        "native_extension_path": str(first_receipt["native_extension_path"]),
        "native_extension_sha256": measurement["native_extension_sha256"],
        "producer_source_sha256": _file_sha256(PRODUCER_PATH),
        "python_executable": str(python_executable),
        "command_protocol": COMMAND_PROTOCOL,
        "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
        "worker_process_group_rss_limit_bytes": (
            WORKER_PROCESS_GROUP_RSS_LIMIT_BYTES
        ),
        "worker_watchdog_rss_measurement_kind": (
            verifier.PARENT_WATCHDOG_RSS_MEASUREMENT_KIND
        ),
        "worker_watchdog_poll_interval_seconds": (
            WORKER_WATCHDOG_POLL_INTERVAL_SECONDS
        ),
        "maximum_mps_working_set_bytes": MAXIMUM_MPS_WORKING_SET_BYTES,
        "mps_memory_limit": dict(
            _require_mapping(
                measurement["mps_memory_limit"],
                "normalized MPS memory-limit receipt",
            )
        ),
        "mps_memory_limit_sha256": measurement["mps_memory_limit_sha256"],
        "mps_memory_sample_interval_ms": (
            MPS_MEMORY_SAMPLE_INTERVAL_SECONDS * 1000.0
        ),
    }


def _validate_receipt(
    receipt: Mapping[str, Any],
    *,
    nonce: str,
    frame_count: int,
    repeat_index: int,
    command_sha256: str,
    source_manifest_sha256: str,
    driver_capabilities_sha256: str,
) -> None:
    if (
        receipt.get("producer_name") != verifier.PRODUCER_NAME
        or receipt.get("producer_schema_version") != verifier.PRODUCER_SCHEMA_VERSION
        or receipt.get("nonce") != nonce
        or receipt.get("frame_count") != frame_count
        or receipt.get("repeat_index") != repeat_index
        or receipt.get("command_sha256") != command_sha256
    ):
        raise ValueError("fresh child receipt identity changed")
    trial = _require_mapping(receipt.get("normalized_trial"), "normalized trial")
    measurement = _require_mapping(trial.get("measurement"), "normalized measurement")
    if measurement.get("source_manifest_sha256") != source_manifest_sha256:
        raise ValueError("fresh child receipt source manifest changed")
    if receipt.get("driver_capabilities_sha256") != driver_capabilities_sha256:
        raise ValueError("fresh child driver capabilities changed")
    if measurement.get("trial_execution_evidence_sha256") != receipt.get(
        "execution_evidence_sha256"
    ):
        raise ValueError("fresh child execution evidence digest changed")
    if (
        measurement.get("parent_watchdog") != receipt.get("parent_watchdog")
        or measurement.get("parent_watchdog_evidence_sha256")
        != receipt.get("parent_watchdog_evidence_sha256")
    ):
        raise ValueError("parent watchdog binding changed")


def _attach_parent_watchdog(
    receipt: dict[str, Any],
    watchdog: Mapping[str, Any],
) -> None:
    """Bind the parent-only safety receipt into the normalized trial."""

    trial = receipt.get("normalized_trial")
    if not isinstance(trial, dict):
        raise TypeError("normalized trial must be mutable before watchdog binding")
    measurement = trial.get("measurement")
    if not isinstance(measurement, dict):
        raise TypeError("normalized measurement must be mutable before watchdog binding")
    frozen_watchdog = dict(watchdog)
    watchdog_evidence_sha256 = _sha256_bytes(
        _canonical_bytes(
            {
                "parent_watchdog": frozen_watchdog,
                "trial_execution_evidence_sha256": measurement.get(
                    "trial_execution_evidence_sha256"
                ),
            }
        )
    )
    measurement["parent_watchdog"] = frozen_watchdog
    measurement["parent_watchdog_evidence_sha256"] = watchdog_evidence_sha256
    receipt["parent_watchdog"] = frozen_watchdog
    receipt["parent_watchdog_evidence_sha256"] = watchdog_evidence_sha256


def _orchestrate(args: argparse.Namespace) -> int:
    if not 0.0 < args.mps_memory_fraction <= MAXIMUM_MPS_MEMORY_FRACTION:
        raise ValueError(
            "MPS memory fraction must be positive and no greater than "
            f"{MAXIMUM_MPS_MEMORY_FRACTION:.2f}"
        )
    _validate_resource_policy(
        minimum_free_disk_bytes=args.minimum_free_disk_bytes,
        minimum_available_memory_bytes=args.minimum_available_memory_bytes,
        maximum_swap_used_bytes=args.maximum_swap_used_bytes,
        maximum_load_average=args.maximum_load_average,
    )
    driver_path = _repo_file(args.trial_driver, name="trial driver")
    config_path = _repo_file(args.trial_config, name="trial config")
    python_executable = _resolved_file(args.python, name="Python executable")
    contract = verifier.load_json_object(args.contract)
    verifier.validate_contract(contract)
    driver_capabilities = _load_driver_capabilities(driver_path)
    driver_capabilities_sha256 = _sha256_bytes(
        _canonical_bytes(driver_capabilities)
    )
    capability_blockers = _driver_capability_blockers(
        driver_capabilities,
        contract,
    )
    frames = tuple(int(value) for value in contract["required_frame_counts"])
    repeats = int(contract["minimum_repeat_count"])
    manifest, manifest_sha256 = build_source_manifest(
        trial_driver_path=driver_path,
        trial_config_path=config_path,
    )
    plan = {
        "status": "planned" if not args.execute else "executing",
        "backend": args.backend,
        "frame_counts": frames,
        "repeat_count": repeats,
        "trial_count": len(frames) * repeats,
        "source_manifest_sha256": manifest_sha256,
        "driver_capabilities_sha256": driver_capabilities_sha256,
        "driver_capability_blockers": capability_blockers,
        "execution_ready": not capability_blockers,
        "output": str(args.output.resolve()),
        "material_only_scope": True,
        "fresh_process_per_trial": True,
        "minimum_available_memory_bytes": args.minimum_available_memory_bytes,
        "maximum_swap_used_bytes": args.maximum_swap_used_bytes,
        "requested_mps_per_process_memory_fraction": args.mps_memory_fraction,
        "maximum_mps_working_set_bytes": MAXIMUM_MPS_WORKING_SET_BYTES,
        "mps_memory_sample_interval_ms": (
            MPS_MEMORY_SAMPLE_INTERVAL_SECONDS * 1000.0
        ),
        "worker_timeout_seconds": WORKER_TIMEOUT_SECONDS,
        "worker_process_group_rss_limit_bytes": (
            WORKER_PROCESS_GROUP_RSS_LIMIT_BYTES
        ),
    }
    if not args.execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if capability_blockers:
        raise RuntimeError(
            "trial driver lacks contract-required measurement capabilities: "
            + ", ".join(capability_blockers)
        )
    _guard_host(
        minimum_free_disk_bytes=args.minimum_free_disk_bytes,
        minimum_available_memory_bytes=args.minimum_available_memory_bytes,
        maximum_swap_used_bytes=args.maximum_swap_used_bytes,
        maximum_load_average=args.maximum_load_average,
    )
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {output}")
    trial_root = output.parent / f".{output.stem}_fresh_trials"
    trial_root.mkdir(parents=True, exist_ok=True)
    normalized_by_frame: dict[int, list[Mapping[str, Any]]] = {
        frame_count: [] for frame_count in frames
    }
    receipts: list[Mapping[str, Any]] = []
    for frame_count in frames:
        for repeat_index in range(repeats):
            _guard_host(
                minimum_free_disk_bytes=args.minimum_free_disk_bytes,
                minimum_available_memory_bytes=args.minimum_available_memory_bytes,
                maximum_swap_used_bytes=args.maximum_swap_used_bytes,
                maximum_load_average=args.maximum_load_average,
            )
            nonce = secrets.token_hex(32)
            receipt_path = trial_root / f"f{frame_count}_r{repeat_index}.json"
            stdout_path = trial_root / f"f{frame_count}_r{repeat_index}.stdout.log"
            stderr_path = trial_root / f"f{frame_count}_r{repeat_index}.stderr.log"
            worker_start_time_ns = time.time_ns()
            command = [
                str(python_executable),
                str(PRODUCER_PATH),
                "--worker",
                "--worker-nonce",
                nonce,
                "--worker-start-time-ns",
                str(worker_start_time_ns),
                "--backend",
                args.backend,
                "--frame-count",
                str(frame_count),
                "--repeat-index",
                str(repeat_index),
                "--trial-driver",
                str(driver_path),
                "--trial-config",
                str(config_path),
                "--native-ops-module",
                args.native_ops_module,
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
                "--receipt",
                str(receipt_path),
                "--expected-source-manifest-sha256",
                manifest_sha256,
            ]
            command_with_digest = [
                *command,
                "--expected-command-sha256",
                "PLACEHOLDER",
            ]
            command_sha256 = _nonce_bound_command_sha256(command_with_digest)
            command_with_digest[-1] = command_sha256
            env = os.environ.copy()
            env[WORKER_NONCE_ENV] = nonce
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                try:
                    watchdog = _run_guarded_worker(
                        command_with_digest,
                        cwd=ROOT,
                        env=env,
                        stdout=stdout,
                        stderr=stderr,
                    )
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"fresh trial f{frame_count}/r{repeat_index} violated its "
                        f"parent watchdog; see {stderr_path}"
                    ) from exc
            if watchdog["returncode"] != 0:
                raise RuntimeError(
                    f"fresh trial f{frame_count}/r{repeat_index} failed; "
                    f"see {stderr_path}"
                )
            receipt = verifier.load_json_object(receipt_path)
            _attach_parent_watchdog(receipt, watchdog)
            _validate_receipt(
                receipt,
                nonce=nonce,
                frame_count=frame_count,
                repeat_index=repeat_index,
                command_sha256=command_sha256,
                source_manifest_sha256=manifest_sha256,
                driver_capabilities_sha256=driver_capabilities_sha256,
            )
            receipts.append(receipt)
            normalized_by_frame[frame_count].append(
                _require_mapping(receipt["normalized_trial"], "normalized trial")
            )
    first_receipt = receipts[0]
    binding = _producer_binding(
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        driver_path=driver_path,
        config_path=config_path,
        first_receipt=first_receipt,
        python_executable=python_executable,
    )
    for receipt in receipts:
        trial = _require_mapping(receipt["normalized_trial"], "normalized trial")
        measurement = _require_mapping(trial["measurement"], "normalized measurement")
        for measurement_key, binding_key in (
            ("source_manifest_sha256", "source_manifest_sha256"),
            ("trial_driver_sha256", "trial_driver_sha256"),
            ("trial_config_sha256", "trial_config_sha256"),
            ("hardware_fingerprint_sha256", "hardware_fingerprint_sha256"),
            ("native_extension_sha256", "native_extension_sha256"),
            ("mps_memory_limit_sha256", "mps_memory_limit_sha256"),
        ):
            if measurement[measurement_key] != binding[binding_key]:
                raise ValueError("fresh-process producer identity drifted across trials")
    rows = [
        verifier.build_row_from_normalized_trials(normalized_by_frame[frame_count])
        for frame_count in frames
    ]
    artifact = verifier.build_artifact(
        backend=args.backend,
        source_tree_sha256=manifest_sha256,
        source_manifest=manifest,
        contract_path=args.contract,
        producer_binding=binding,
        rows=rows,
    )
    report = verifier.verify_artifact_payload(
        artifact,
        contract,
        contract_sha256=verifier.file_sha256(args.contract),
    )
    artifact["acceptance_report"] = report
    _write_json_atomic(output, artifact)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Produce fresh-process real-native WorldFoam memory evidence."
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--backend", choices=("mps",), required=True)
    parser.add_argument("--trial-driver", type=Path, required=True)
    parser.add_argument("--trial-config", type=Path, required=True)
    parser.add_argument("--contract", type=Path, default=verifier.DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--native-ops-module", default=DEFAULT_NATIVE_OPS_MODULE)
    parser.add_argument(
        "--minimum-free-disk-bytes",
        type=int,
        default=DEFAULT_MINIMUM_FREE_DISK_BYTES,
    )
    parser.add_argument(
        "--minimum-available-memory-bytes",
        type=int,
        default=DEFAULT_MINIMUM_AVAILABLE_MEMORY_BYTES,
    )
    parser.add_argument(
        "--maximum-swap-used-bytes",
        type=int,
        default=DEFAULT_MAXIMUM_SWAP_USED_BYTES,
    )
    parser.add_argument(
        "--mps-memory-fraction",
        type=float,
        default=DEFAULT_MPS_MEMORY_FRACTION,
    )
    parser.add_argument(
        "--maximum-load-average",
        type=float,
        default=DEFAULT_MAXIMUM_LOAD_AVERAGE,
    )
    parser.add_argument("--worker-nonce", default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--worker-start-time-ns", type=int, default=0, help=argparse.SUPPRESS
    )
    parser.add_argument("--frame-count", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--repeat-index", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--receipt", type=Path, help=argparse.SUPPRESS)
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
        if args.receipt is None or args.frame_count < 1 or args.repeat_index < 0:
            raise SystemExit("worker arguments are incomplete")
        raise SystemExit(_worker(args))
    raise SystemExit(_orchestrate(args))


if __name__ == "__main__":
    main()


__all__ = [
    "COMMAND_PROTOCOL",
    "DEFAULT_OUTPUT",
    "DRIVER_FUNCTION",
    "build_source_manifest",
]
