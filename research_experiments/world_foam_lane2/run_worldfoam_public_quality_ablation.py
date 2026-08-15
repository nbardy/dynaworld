#!/usr/bin/env python3
"""Plan, execute, or collect the fail-closed WorldFoam G4 matrix.

The worker capability is intentionally absent until the checked-in runtime
attestor has exercised all four production routes against sealed public data.
``--execute`` performs every capability, source, and host check before
launching the first representation, so an unavailable native4d route cannot
leave a misleading directory of completed Gaussian baselines.  The default
command is allocation-free planning; ``--collect`` only aggregates 36 already
measured raw row receipts and immediately runs the independent G4 verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "src" / "train"
LANE2 = Path(__file__).resolve().parent
for import_root in (ROOT, TRAIN, LANE2):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from config_utils import serialize_config_value  # noqa: E402
from verify_worldfoam_public_quality_ablation import (  # noqa: E402
    ARTIFACT_KIND,
    DEFAULT_ARTIFACT,
    DEFAULT_CONFIG,
    REQUIRED_ROUTES,
    ROW_KEYS,
    ROW_KIND,
    artifact_sha256,
    compute_acceptance,
    file_sha256,
    load_contract,
    validate_contract,
    verify_artifact_file,
)


WORKER_PATH = ROOT / "src" / "train" / "train_worldfoam_native4d_public_quality_row.py"
WORKER_CAPABILITY_PATH = (
    ROOT
    / "src"
    / "train"
    / "worldfoam_native4d_public_quality_capabilities.json"
)
REQUIRED_WORKER_CAPABILITIES = {
    "schema_version": 1,
    "status": "runtime_verified",
    "row_kind": ROW_KIND,
    "supported_routes": list(REQUIRED_ROUTES),
    "real_native_only": True,
    "public_neural3d_targets": True,
    "heldout_camera_evaluation": True,
    "full_temporal_evaluation": True,
    "compiled_shared_adjoint": True,
    "same_representation_framewise_replay": True,
    "final_checkpoint_metrics": True,
    "wandb_run_file": True,
    "proxy_or_fake_native_permitted": False,
    "smoke_as_public_evidence_permitted": False,
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        serialize_config_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return payload


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def source_identity() -> dict[str, Any]:
    commit = _git("rev-parse", "HEAD")
    dirty = bool(_git("status", "--porcelain", "--untracked-files=all"))
    return {"repository_commit": commit, "repository_dirty": dirty}


def expected_row_path(
    output_root: Path, *, scene: str, seed: int, route: str
) -> Path:
    return output_root / scene / f"seed_{seed}" / route / "g4_row.json"


def _worker_command(
    *,
    config_path: Path,
    protocol_path: Path,
    scene: str,
    seed: int,
    route: str,
    output_path: Path,
    allow_local_mps_execution: bool,
) -> list[str]:
    command = [
        sys.executable,
        str(WORKER_PATH),
        "--execute",
        "--g4-config",
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
    ]
    if allow_local_mps_execution:
        command.append("--allow-local-mps-execution")
    return command


def runtime_blockers(
    config: Mapping[str, Any], *, config_path: Path = DEFAULT_CONFIG
) -> list[str]:
    """Return only code/runtime blockers; never import torch or native ops."""

    blockers: list[str] = []
    try:
        validate_contract(config, config_path=config_path)
    except Exception as error:
        blockers.append(f"g4_contract_invalid:{type(error).__name__}:{error}")
        return blockers
    if not WORKER_PATH.is_file():
        blockers.append("public_native4d_row_worker_missing")
    if not WORKER_CAPABILITY_PATH.is_file():
        blockers.append("public_native4d_runtime_capability_receipt_missing")
    else:
        try:
            receipt = _load_json(WORKER_CAPABILITY_PATH)
        except Exception as error:
            blockers.append(
                "public_native4d_runtime_capability_receipt_invalid:"
                f"{type(error).__name__}:{error}"
            )
        else:
            if receipt != REQUIRED_WORKER_CAPABILITIES:
                blockers.append("public_native4d_runtime_capabilities_not_verified")

    target_binding = LANE2 / "worldfoam_target_dataset_binding.py"
    target_source = target_binding.read_text(encoding="utf-8")
    if 'binding.get("target_split") != "train"' in target_source:
        blockers.append("mapped_public_target_binding_is_train_only")
    neural_adapter = LANE2 / "neural3d_mapped_rgb8_adapter.py"
    neural_source = neural_adapter.read_text(encoding="utf-8")
    if '"target_split": "train"' in neural_source:
        blockers.append("neural3d_cache_adapter_has_no_heldout_split")
    # G4 no longer routes through the synthetic G6 memory-ablation adapter.
    # Keep that fixture honest, but judge public execution only by the sealed
    # mapped provider and the dedicated native4d executor actually imported by
    # the row worker.
    provider = ROOT / "src" / "train" / "worldfoam_public_quality_dataset_provider.py"
    executor = ROOT / "src" / "train" / "worldfoam_native4d_public_quality_executor.py"
    if not provider.is_file():
        blockers.append("public_mapped_dataset_provider_missing")
    if not executor.is_file():
        blockers.append("public_native4d_executor_missing")
    else:
        executor_source = executor.read_text(encoding="utf-8")
        if "def run_public_quality_runtime_smoke(" not in executor_source:
            blockers.append("public_native4d_runtime_smoke_missing")
        if "CERTIFIED_SPATIAL_COMPILE_REUSE = False" in executor_source:
            blockers.append("public_native4d_spatial_compile_reuse_unimplemented")

    binary = (
        ROOT
        / "third_party"
        / "fast-mac-gsplat"
        / "variants"
        / "world_foam_lane2_fused_slab_v0"
        / "torch_world_foam_lane2_fused_slab"
        / "_C.cpython-311-darwin.so"
    )
    variant_root = binary.parents[1]
    native_sources = tuple(
        path
        for path in (
            variant_root / "csrc" / "bindings.cpp",
            *sorted((variant_root / "csrc" / "metal").glob("*.mm")),
            *sorted((variant_root / "csrc" / "metal").glob("*.metal")),
        )
        if path.is_file()
    )
    if not binary.is_file():
        blockers.append("worldfoam_native_extension_missing")
    elif native_sources and binary.stat().st_mtime_ns < max(
        path.stat().st_mtime_ns for path in native_sources
    ):
        blockers.append("worldfoam_native_extension_older_than_sources")
    return sorted(set(blockers))


def build_plan(
    config: Mapping[str, Any],
    *,
    config_path: Path = DEFAULT_CONFIG,
    allow_local_mps_execution: bool = False,
) -> dict[str, Any]:
    receipt = validate_contract(config, config_path=config_path)
    output_root = (ROOT / str(config["output_root"])).resolve()
    rows: list[dict[str, Any]] = []
    for scene, scene_receipt in receipt["scenes"].items():
        protocol_path = ROOT / str(scene_receipt["protocol_path"])
        for seed in config["seeds"]:
            for route in REQUIRED_ROUTES:
                output_path = expected_row_path(
                    output_root, scene=scene, seed=seed, route=route
                )
                rows.append(
                    {
                        "row_id": f"{scene}/seed_{seed}/{route}",
                        "scene": scene,
                        "seed": seed,
                        "route": route,
                        "protocol": _display(protocol_path),
                        "protocol_sha256": scene_receipt["protocol_sha256"],
                        "output": _display(output_path),
                        "command": _worker_command(
                            config_path=config_path,
                            protocol_path=protocol_path,
                            scene=scene,
                            seed=seed,
                            route=route,
                            output_path=output_path,
                            allow_local_mps_execution=allow_local_mps_execution,
                        ),
                    }
                )
    blockers = runtime_blockers(config, config_path=config_path)
    return {
        "schema_version": 1,
        "kind": "worldfoam-native4d-g4-execution-plan-v1",
        "config": _display(config_path),
        "config_sha256": file_sha256(config_path),
        "expected_row_count": 36,
        "fresh_process_count": 36,
        "abort_before_first_row": bool(blockers),
        "runtime_ready": not blockers,
        "runtime_blockers": blockers,
        "rows": rows,
        "artifact": _display(DEFAULT_ARTIFACT),
        "plan_sha256": "",
    }


def finalize_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(plan)
    result["plan_sha256"] = _canonical_sha256(
        {key: value for key, value in result.items() if key != "plan_sha256"}
    )
    return result


def _parse_swap_bytes(output: str) -> int:
    match = re.search(r"used\s*=\s*([0-9.]+)([KMG])", output)
    if match is None:
        raise ValueError("could not parse swap usage")
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)]
    return int(float(match.group(1)) * scale)


def live_resource_snapshot() -> dict[str, Any]:
    if sys.platform != "darwin":
        raise RuntimeError("G4 Metal execution requires macOS")
    cpu_count = int(os.cpu_count() or 1)
    load = os.getloadavg()
    vm_output = subprocess.check_output(("vm_stat",), text=True)
    page_match = re.search(r"page size of\s+(\d+)\s+bytes", vm_output)
    if page_match is None:
        raise ValueError("could not parse vm_stat page size")
    page_size = int(page_match.group(1))
    pages: dict[str, int] = {}
    for line in vm_output.splitlines():
        match = re.match(r"([^:]+):\s+([0-9.]+)\.?$", line.strip())
        if match:
            pages[match.group(1)] = int(float(match.group(2)))
    swap = subprocess.check_output(
        ("sysctl", "-n", "vm.swapusage"), text=True, stderr=subprocess.DEVNULL
    )
    return {
        "platform": sys.platform,
        "available_memory_bytes": page_size
        * sum(
            pages.get(key, 0)
            for key in ("Pages free", "Pages inactive", "Pages speculative")
        ),
        "swap_used_bytes": _parse_swap_bytes(swap),
        "disk_free_bytes": int(shutil.disk_usage(ROOT).free),
        "logical_cpu_count": cpu_count,
        "load_average_1m": float(load[0]),
        "load_1m_per_logical_cpu": float(load[0]) / cpu_count,
    }


def require_live_resources(
    snapshot: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    guard = config["execution_guard"]
    failures = []
    if int(snapshot.get("available_memory_bytes", 0)) < guard[
        "minimum_available_memory_bytes"
    ]:
        failures.append("available_memory_bytes")
    if int(snapshot.get("swap_used_bytes", sys.maxsize)) > guard[
        "maximum_swap_used_bytes"
    ]:
        failures.append("swap_used_bytes")
    if int(snapshot.get("disk_free_bytes", 0)) < guard[
        "minimum_disk_free_bytes"
    ]:
        failures.append("disk_free_bytes")
    if float(snapshot.get("load_1m_per_logical_cpu", math.inf)) > guard[
        "maximum_load_1m_per_logical_cpu"
    ]:
        failures.append("load_1m_per_logical_cpu")
    if failures:
        raise RuntimeError("G4 live-resource gate rejected: " + ", ".join(failures))


def _relative_identity(path: Path) -> dict[str, Any]:
    return {
        "path": _display(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }


def collect_artifact(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    receipt = validate_contract(config, config_path=config_path)
    output_root = (ROOT / str(config["output_root"])).resolve()
    rows: list[dict[str, Any]] = []
    source_commits: set[str] = set()
    for scene in receipt["scenes"]:
        for seed in config["seeds"]:
            for route in REQUIRED_ROUTES:
                path = expected_row_path(
                    output_root, scene=scene, seed=seed, route=route
                )
                if not path.is_file():
                    raise FileNotFoundError(f"measured G4 row is missing: {path}")
                raw = _load_json(path)
                if set(raw) != set(ROW_KEYS):
                    raise ValueError(f"G4 raw row keys changed: {path}")
                if raw.get("row_kind") != ROW_KIND:
                    raise ValueError(f"G4 raw row kind changed: {path}")
                source_commits.add(str(raw.get("source_commit")))
                rows.append({**raw, "receipt": _relative_identity(path)})
    if len(source_commits) != 1:
        raise ValueError("G4 rows were not produced from one clean source commit")
    source_commit = next(iter(source_commits))
    acceptance = compute_acceptance(rows, config)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": ARTIFACT_KIND,
        "status": "measured",
        "public_quality_evidence": True,
        "proxy_or_test_artifact": False,
        "measurement_is_simulated": False,
        "matrix_config": _display(config_path),
        "matrix_config_sha256": file_sha256(config_path),
        "source_commit": source_commit,
        "rows": rows,
        "acceptance": acceptance,
        "artifact_sha256": "",
    }
    payload["artifact_sha256"] = artifact_sha256(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(serialize_config_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = verify_artifact_file(output_path, config_path=config_path)
    if report.get("accepted") is not True:
        raise RuntimeError(
            "collected G4 artifact failed independent verification: "
            + "; ".join(report.get("failures", ()))
        )
    return {"artifact": _display(output_path), "verification": report}


def execute_plan(
    plan: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    config_path: Path,
    output_path: Path,
    allow_local_mps_execution: bool,
) -> dict[str, Any]:
    if not allow_local_mps_execution:
        raise RuntimeError(
            "local MPS G4 execution requires explicit --allow-local-mps-execution"
        )
    blockers = list(plan.get("runtime_blockers", ()))
    if blockers:
        raise RuntimeError(
            "G4 aborted before first row; production lane is unavailable: "
            + ", ".join(blockers)
        )
    source = source_identity()
    if source.get("repository_dirty") is not False:
        raise RuntimeError("G4 paper rows require a clean repository before first launch")
    require_live_resources(live_resource_snapshot(), config)
    completed = 0
    for row in plan["rows"]:
        require_live_resources(live_resource_snapshot(), config)
        command = [str(value) for value in row["command"]]
        subprocess.run(command, cwd=ROOT, check=True)
        expected = ROOT / str(row["output"])
        if not expected.is_file() or expected.stat().st_size <= 0:
            raise FileNotFoundError(f"G4 worker emitted no row: {expected}")
        if source_identity() != source:
            raise RuntimeError("source changed during the G4 matrix")
        completed += 1
    if completed != 36:
        raise ArithmeticError("G4 execution did not complete all 36 fresh processes")
    return collect_artifact(
        config=config, config_path=config_path, output_path=output_path
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run or collect the WorldFoam native4d G4 public matrix."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--collect", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--allow-local-mps-execution", action="store_true")
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    config = load_contract(config_path)
    plan = finalize_plan(
        build_plan(
            config,
            config_path=config_path,
            allow_local_mps_execution=args.allow_local_mps_execution,
        )
    )
    if args.collect:
        result = collect_artifact(
            config=config,
            config_path=config_path,
            output_path=args.artifact.resolve(),
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.execute:
        result = execute_plan(
            plan,
            config=config,
            config_path=config_path,
            output_path=args.artifact.resolve(),
            allow_local_mps_execution=args.allow_local_mps_execution,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    print(json.dumps(serialize_config_value(plan), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REQUIRED_WORKER_CAPABILITIES",
    "WORKER_CAPABILITY_PATH",
    "WORKER_PATH",
    "build_plan",
    "collect_artifact",
    "execute_plan",
    "expected_row_path",
    "finalize_plan",
    "live_resource_snapshot",
    "require_live_resources",
    "runtime_blockers",
    "source_identity",
]
