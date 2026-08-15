#!/usr/bin/env python3
"""Attest all four real-native G4 routes before enabling public rows.

The default command is allocation-free and writes nothing.  Execution is an
explicit, clean-tree, local-MPS operation.  Each route runs in a fresh child
process against the same sealed mapped public dataset.  The flat capability
file consumed by the row worker is published last, only after all primitive
smoke receipts and the fused-slab build receipt validate.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "src" / "train"
LANE2 = ROOT / "research_experiments" / "world_foam_lane2"
for _root in (TRAIN, LANE2, ROOT):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from config_utils import load_config_file  # noqa: E402
from public_quality_runtime_smoke import (  # noqa: E402
    canonical_sha256,
    validate_public_quality_runtime_smoke,
)
from verify_worldfoam_public_quality_ablation import (  # noqa: E402
    DEFAULT_CONFIG,
    REQUIRED_ROUTES,
    file_sha256,
)
from worldfoam_native4d_public_quality_row import (  # noqa: E402
    REQUIRED_RUNTIME_CAPABILITIES,
    ROUTE_EXECUTOR_MODULES,
    RUNTIME_CAPABILITY_PATH,
    RowContext,
    RowRequest,
    _initialization_blockers,
    default_dataset_capability_path,
    load_dataset_capability,
    load_public_quality_dataset,
    resolve_row_request,
)


FUSED_BUILD_ATTESTOR = (
    LANE2 / "attest_worldfoam_fused_slab_build.py"
)
FUSED_BUILD_RECEIPT = (
    ROOT
    / "artifacts"
    / "native_build_attestations"
    / "world_foam_lane2_fused_slab_v0.json"
)
EVIDENCE_PATH = RUNTIME_CAPABILITY_PATH.with_name(
    "worldfoam_native4d_public_quality_capabilities.evidence.json"
)
EVIDENCE_KIND = "worldfoam-g4-public-quality-runtime-attestation-v1"


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    partial.unlink(missing_ok=True)
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    try:
        with partial.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def _source_identity() -> dict[str, Any]:
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


def _repo_display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _file_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.stat().st_size < 1:
        raise FileNotFoundError(f"attested source/artifact is missing: {resolved}")
    return {
        "path": _repo_display(resolved),
        "bytes": int(resolved.stat().st_size),
        "sha256": file_sha256(resolved),
    }


def _protocol_for_scene(config_path: Path, scene: str) -> Path:
    config = load_config_file(config_path)
    matches = [row for row in config.get("scenes", ()) if row.get("scene") == scene]
    if len(matches) != 1:
        raise ValueError(f"G4 config has no unique scene {scene!r}")
    candidate = Path(str(matches[0]["protocol"]))
    return candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()


def _expected_output(config_path: Path, *, scene: str, seed: int, route: str) -> Path:
    config = load_config_file(config_path)
    root = Path(str(config["output_root"]))
    root = root.resolve() if root.is_absolute() else (ROOT / root).resolve()
    return root / scene / f"seed_{seed}" / route / "g4_row.json"


def _resolve_context(
    *,
    config_path: Path,
    protocol_path: Path,
    scene: str,
    seed: int,
    route: str,
    dataset_capability_path: Path | None,
) -> RowContext:
    request = RowRequest(
        config_path=config_path,
        protocol_path=protocol_path,
        scene=scene,
        seed=seed,
        route=route,
        output_path=_expected_output(
            config_path,
            scene=scene,
            seed=seed,
            route=route,
        ),
        allow_local_mps_execution=True,
        dataset_capability_path=dataset_capability_path,
    )
    (
        config,
        config_receipt,
        protocol,
        route_spec,
        scene_receipt,
        work_plan,
    ) = resolve_row_request(request)
    initialization_failures, _details = _initialization_blockers(
        scene_receipt["initialization"]
    )
    if initialization_failures:
        raise RuntimeError(
            "runtime smoke initialization is not sealed: "
            + ", ".join(initialization_failures)
        )
    capability_path = dataset_capability_path or default_dataset_capability_path(
        protocol
    )
    capability = load_dataset_capability(
        capability_path,
        request=request,
        protocol=protocol,
        scene_receipt=scene_receipt,
    )
    source = _source_identity()
    if source["repository_dirty"] is not False:
        raise RuntimeError("runtime capability attestation requires a clean source tree")
    return RowContext(
        request=request,
        config=config,
        config_receipt=config_receipt,
        protocol=protocol,
        route_spec=route_spec,
        scene_receipt=scene_receipt,
        work_plan=work_plan,
        source_commit=str(source["repository_commit"]),
        dataset_capability=capability,
    )


def _module_source(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError(module_name)
    path = Path(spec.origin).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _worldfoam_native_library_identity() -> dict[str, Any]:
    from worldfoam_native4d_public_quality_executor import VARIANT_ROOT

    libraries = tuple(
        sorted(
            (VARIANT_ROOT / "torch_world_foam_lane2_fused_slab").glob("_C*.so")
        )
    )
    if len(libraries) != 1:
        raise RuntimeError(
            f"expected one WorldFoam native library, found {len(libraries)}"
        )
    return _file_identity(libraries[0])


def _source_exports_smoke(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "run_public_quality_runtime_smoke"
        for node in tree.body
    )


def _dry_plan(
    *,
    config_path: Path,
    protocol_path: Path,
    scene: str,
    seed: int,
    dataset_capability_path: Path | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    route_sources: dict[str, Any] = {}
    context_summary: dict[str, Any] = {}
    try:
        context = _resolve_context(
            config_path=config_path,
            protocol_path=protocol_path,
            scene=scene,
            seed=seed,
            route=REQUIRED_ROUTES[0],
            dataset_capability_path=dataset_capability_path,
        )
    except Exception as error:
        blockers.append(f"sealed_public_context_unavailable:{type(error).__name__}:{error}")
    else:
        context_summary = {
            "sample_id": context.protocol.dataset.sample_id,
            "frame_count": context.protocol.dataset.frame_count,
            "image_size": context.protocol.final_stage.image_size.as_list(),
            "dataset_capability_sha256": context.dataset_capability[
                "capability_sha256"
            ],
            "source_commit": context.source_commit,
        }
    for route in REQUIRED_ROUTES:
        module_name = ROUTE_EXECUTOR_MODULES[route]
        try:
            source = _module_source(module_name)
        except Exception as error:
            blockers.append(
                f"route_executor_missing:{route}:{type(error).__name__}:{error}"
            )
        else:
            route_sources[route] = _file_identity(source)
            if not _source_exports_smoke(source):
                blockers.append(f"route_runtime_smoke_missing:{route}:{module_name}")
    if platform.system() != "Darwin":
        blockers.append("runtime_attestation_requires_macos")
    if not FUSED_BUILD_RECEIPT.is_file():
        blockers.append("fused_slab_build_attestation_missing")
    if RUNTIME_CAPABILITY_PATH.exists():
        try:
            existing = json.loads(RUNTIME_CAPABILITY_PATH.read_text(encoding="utf-8"))
        except Exception:
            blockers.append("existing_runtime_capability_is_invalid")
        else:
            if existing != REQUIRED_RUNTIME_CAPABILITIES:
                blockers.append("existing_runtime_capability_is_invalid")
    return {
        "schema_version": 1,
        "kind": "worldfoam-g4-public-quality-runtime-attestation-plan-v1",
        "status": "ready" if not blockers else "blocked",
        "allocation_started": False,
        "decode_started": False,
        "write_started": False,
        "scene": scene,
        "seed": seed,
        "config": _repo_display(config_path),
        "protocol": _repo_display(protocol_path),
        "context": context_summary,
        "routes": list(REQUIRED_ROUTES),
        "route_sources": route_sources,
        "fresh_process_per_route": True,
        "runtime_capability_output": _repo_display(RUNTIME_CAPABILITY_PATH),
        "evidence_output": _repo_display(EVIDENCE_PATH),
        "blockers": sorted(set(blockers)),
    }


def _verify_fused_build() -> dict[str, Any]:
    command = [
        sys.executable,
        str(FUSED_BUILD_ATTESTOR),
        "--verify-receipt",
        str(FUSED_BUILD_RECEIPT),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    try:
        result = json.loads(completed.stdout)
    except Exception as error:
        raise RuntimeError(
            "fused-slab build verifier returned invalid JSON: "
            + completed.stderr[-2000:]
        ) from error
    if completed.returncode != 0 or result.get("status") != "accepted":
        raise RuntimeError(
            "fused-slab build attestation failed: "
            + "; ".join(str(item) for item in result.get("failures", ()))
        )
    return result


def _run_child_route(args: argparse.Namespace) -> int:
    if not args.execute or not args.allow_local_mps_execution:
        raise RuntimeError("child runtime smoke lacks explicit MPS authorization")
    context = _resolve_context(
        config_path=args.g4_config,
        protocol_path=args.protocol,
        scene=args.scene,
        seed=args.seed,
        route=args._run_route,
        dataset_capability_path=args.dataset_capability,
    )
    import torch

    if not torch.backends.mps.is_available():
        raise RuntimeError("real MPS is unavailable")
    dataset = load_public_quality_dataset(context)
    try:
        module = importlib.import_module(ROUTE_EXECUTOR_MODULES[args._run_route])
        smoke = getattr(module, "run_public_quality_runtime_smoke", None)
        if not callable(smoke):
            raise AttributeError("production route has no runtime smoke callable")
        receipt = validate_public_quality_runtime_smoke(
            smoke(context=context, dataset=dataset),
            context=context,
        )
    finally:
        dataset.close()
    _atomic_write_json(args._result_path, receipt)
    return 0


def _run_all_routes(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_local_mps_execution:
        raise RuntimeError("--execute requires --allow-local-mps-execution")
    if RUNTIME_CAPABILITY_PATH.exists() or EVIDENCE_PATH.exists():
        raise FileExistsError(
            "runtime capability/evidence already exists; verify or remove it explicitly"
        )
    source_before = _source_identity()
    if source_before["repository_dirty"] is not False:
        raise RuntimeError("runtime capability attestation requires a clean source tree")
    fused_build = _verify_fused_build()
    receipts: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(
        prefix="dynaworld-g4-runtime-attestation-"
    ) as temporary:
        for route in REQUIRED_ROUTES:
            result_path = Path(temporary) / f"{route}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--execute",
                "--allow-local-mps-execution",
                "--g4-config",
                str(args.g4_config),
                "--protocol",
                str(args.protocol),
                "--scene",
                args.scene,
                "--seed",
                str(args.seed),
                "--_run-route",
                route,
                "--_result-path",
                str(result_path),
            ]
            if args.dataset_capability is not None:
                command.extend(
                    ("--dataset-capability", str(args.dataset_capability))
                )
            completed = subprocess.run(
                command,
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0 or not result_path.is_file():
                raise RuntimeError(
                    f"runtime smoke failed for {route}: "
                    + (completed.stderr or completed.stdout)[-4000:]
                )
            context = _resolve_context(
                config_path=args.g4_config,
                protocol_path=args.protocol,
                scene=args.scene,
                seed=args.seed,
                route=route,
                dataset_capability_path=args.dataset_capability,
            )
            primitive = json.loads(result_path.read_text(encoding="utf-8"))
            receipts[route] = validate_public_quality_runtime_smoke(
                primitive,
                context=context,
            )
    if tuple(receipts) != tuple(REQUIRED_ROUTES):
        raise ArithmeticError("runtime attestor did not cover all four routes")
    source_after = _source_identity()
    if source_after != source_before:
        raise RuntimeError("source changed during runtime capability attestation")
    context = _resolve_context(
        config_path=args.g4_config,
        protocol_path=args.protocol,
        scene=args.scene,
        seed=args.seed,
        route=REQUIRED_ROUTES[0],
        dataset_capability_path=args.dataset_capability,
    )
    executor_sources = {
        route: _file_identity(_module_source(ROUTE_EXECUTOR_MODULES[route]))
        for route in REQUIRED_ROUTES
    }
    from worldfoam_native_heldout_prediction import (
        PREDICTION_ABI_SCHEMA_SHA256,
    )

    evidence: dict[str, Any] = {
        "schema_version": 1,
        "kind": EVIDENCE_KIND,
        "status": "runtime_verified",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scene": args.scene,
        "seed": args.seed,
        "sample_id": context.protocol.dataset.sample_id,
        "protocol": _file_identity(args.protocol),
        "dataset_capability_sha256": context.dataset_capability[
            "capability_sha256"
        ],
        "source": source_before,
        "executor_sources": executor_sources,
        "worldfoam_native_library": _worldfoam_native_library_identity(),
        "worldfoam_prediction_abi_schema_sha256": (
            PREDICTION_ABI_SCHEMA_SHA256
        ),
        "fused_slab_build_receipt": _file_identity(FUSED_BUILD_RECEIPT),
        "fused_slab_build_verification": fused_build,
        "route_runtime_smokes": receipts,
        "capabilities": dict(REQUIRED_RUNTIME_CAPABILITIES),
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    _atomic_write_json(EVIDENCE_PATH, evidence)
    try:
        _atomic_write_json(
            RUNTIME_CAPABILITY_PATH,
            dict(REQUIRED_RUNTIME_CAPABILITIES),
        )
    except BaseException:
        EVIDENCE_PATH.unlink(missing_ok=True)
        raise
    return {
        "status": "runtime_verified",
        "runtime_capability": _file_identity(RUNTIME_CAPABILITY_PATH),
        "evidence": _file_identity(EVIDENCE_PATH),
        "routes": list(receipts),
        "paper_evidence_emitted": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-local-mps-execution", action="store_true")
    parser.add_argument("--g4-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--scene", default="coffee_martini")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dataset-capability", type=Path)
    parser.add_argument("--_run-route", choices=REQUIRED_ROUTES, help=argparse.SUPPRESS)
    parser.add_argument("--_result-path", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.g4_config = args.g4_config.resolve()
    args.protocol = (
        args.protocol.resolve()
        if args.protocol is not None
        else _protocol_for_scene(args.g4_config, args.scene)
    )
    if args.dataset_capability is not None:
        args.dataset_capability = args.dataset_capability.resolve()
    return args


def main() -> int:
    args = _parse_args()
    try:
        if args._run_route is not None:
            if args._result_path is None:
                raise ValueError("child route execution requires --_result-path")
            return _run_child_route(args)
        if args._result_path is not None:
            raise ValueError("--_result-path is internal to one child route")
        result = (
            _run_all_routes(args)
            if args.execute
            else _dry_plan(
                config_path=args.g4_config,
                protocol_path=args.protocol,
                scene=args.scene,
                seed=args.seed,
                dataset_capability_path=args.dataset_capability,
            )
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "runtime_capability_written": RUNTIME_CAPABILITY_PATH.is_file(),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
