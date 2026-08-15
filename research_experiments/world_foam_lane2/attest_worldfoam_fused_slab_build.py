#!/usr/bin/env python3
"""Attest one exact fused-slab native build without dispatching Metal.

The attestation binds the active CPython ABI, Darwin architecture, Torch and
compiler identities, every native/runtime/Python ABI source byte, the selected
extension byte stream, and the complete dispatcher schema inventory.  It does
not launch an operator or claim MPS correctness/performance.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import sysconfig
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from verify_worldfoam_native_variant_sources import (
    DEFAULT_VARIANT_ROOT,
    _load_native_build_contract,
    _schema_inventory,
)


DYNAWORLD = Path(__file__).resolve().parents[2]
VARIANT_NAME = "world_foam_lane2_fused_slab_v0"
PACKAGE_NAME = "torch_world_foam_lane2_fused_slab"
DEFAULT_VARIANT_DIR = DEFAULT_VARIANT_ROOT / VARIANT_NAME
DEFAULT_RECEIPT = (
    DYNAWORLD
    / "artifacts"
    / "native_build_attestations"
    / "world_foam_lane2_fused_slab_v0.json"
)
RECEIPT_SCHEMA_PATH = (
    SCRIPT_DIR
    / "worldfoam_fused_slab_build_attestation_v1.schema.json"
)
ATTESTATION_KIND = "worldfoam_fused_slab_native_build_attestation"
ATTESTATION_SCHEMA_VERSION = 1
REQUIRED_PYTHON_IMPLEMENTATION = "cpython"
REQUIRED_PYTHON_VERSION = (3, 11)
REQUIRED_SYSTEM = "Darwin"
DISPATCH_KEY = "CompositeExplicitAutograd"


class BuildAttestationError(RuntimeError):
    """Raised when a native build cannot be attested exactly."""


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _payload_sha256(payload: dict[str, Any]) -> str:
    unsigned = dict(payload)
    unsigned.pop("receipt_payload_sha256", None)
    return _sha256_bytes(_canonical_json_bytes(unsigned))


def _inventory_sha256(values: list[str]) -> str:
    return _sha256_bytes(("\n".join(sorted(values)) + "\n").encode("utf-8"))


def _source_file_role(relative: str, contract: Any) -> str:
    if relative in contract.TRANSLATION_UNITS:
        return "compiled_translation_unit"
    if relative in contract.RUNTIME_METAL_SOURCES:
        return "runtime_compiled_metal_source"
    if relative in contract.PYTHON_ABI_SOURCES:
        return "python_abi_source"
    if relative == "setup.py":
        return "build_recipe"
    if relative.endswith(".h"):
        return "compiled_header"
    return "build_contract"


def _source_snapshot(variant_dir: Path) -> dict[str, Any]:
    contract, summary, failures = _load_native_build_contract(variant_dir)
    if failures or contract is None or summary is None:
        raise BuildAttestationError("; ".join(failures or ["native build contract unavailable"]))
    files = []
    for relative in contract.ATTESTED_SOURCE_FILES:
        path = variant_dir / relative
        files.append(
            {
                "path": relative,
                "role": _source_file_role(relative, contract),
                "bytes": path.stat().st_size,
                "sha256": _sha256_path(path),
            }
        )
    aggregate = _sha256_bytes(
        b"".join(
            (
                f"{row['path']}\0{row['role']}\0{row['bytes']}\0{row['sha256']}\n"
            ).encode("utf-8")
            for row in files
        )
    )
    return {
        "contract_schema_version": summary["contract_schema_version"],
        "contract_path": "native_build_contract.py",
        "contract_sha256": _sha256_path(variant_dir / "native_build_contract.py"),
        "source_aggregate_sha256": aggregate,
        "files": files,
        "translation_units": summary["translation_units"],
        "native_dependencies": summary["native_dependencies"],
        "runtime_metal_sources": summary["runtime_metal_sources"],
        "python_abi_sources": summary["python_abi_sources"],
        "schema_count": summary["schema_count"],
        "schema_names": summary["schema_names"],
        "schema_name_inventory_sha256": summary["schema_name_inventory_sha256"],
        "full_schema_inventory_sha256": summary["full_schema_inventory_sha256"],
        "required_post_103_schema_count": summary["required_post_103_schema_count"],
    }


def _expected_extension_path(variant_dir: Path) -> Path:
    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    if not isinstance(extension_suffix, str) or not extension_suffix:
        raise BuildAttestationError("Python EXT_SUFFIX is unavailable")
    return variant_dir / PACKAGE_NAME / f"_C{extension_suffix}"


def _select_extension(variant_dir: Path) -> Path:
    expected = _expected_extension_path(variant_dir).resolve()
    candidates = sorted((variant_dir / PACKAGE_NAME).glob("_C*.so"))
    if len(candidates) != 1:
        raise BuildAttestationError(
            f"expected exactly one native extension, found {len(candidates)}: {candidates}"
        )
    selected = candidates[0].resolve()
    if selected != expected:
        raise BuildAttestationError(
            f"native extension does not match active CPython ABI: expected {expected}, found {selected}"
        )
    return selected


def _runtime_identity() -> dict[str, Any]:
    if sys.implementation.name != REQUIRED_PYTHON_IMPLEMENTATION:
        raise BuildAttestationError(
            f"requires {REQUIRED_PYTHON_IMPLEMENTATION}, found {sys.implementation.name}"
        )
    if sys.version_info[:2] != REQUIRED_PYTHON_VERSION:
        raise BuildAttestationError(
            f"requires Python {REQUIRED_PYTHON_VERSION[0]}.{REQUIRED_PYTHON_VERSION[1]}, "
            f"found {sys.version_info.major}.{sys.version_info.minor}"
        )
    if platform.system() != REQUIRED_SYSTEM:
        raise BuildAttestationError(f"requires {REQUIRED_SYSTEM}, found {platform.system()}")

    import torch

    return {
        "python_implementation": sys.implementation.name,
        "python_version": platform.python_version(),
        "python_version_info": list(sys.version_info[:3]),
        "python_executable": str(Path(sys.executable).resolve()),
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "python_ext_suffix": sysconfig.get_config_var("EXT_SUFFIX"),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_version": platform.version(),
        "platform_machine": platform.machine(),
        "platform_processor": platform.processor(),
        "mac_ver": list(platform.mac_ver()),
        "torch_version": torch.__version__,
        "torch_cxx11_abi": bool(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", False)),
    }


def _command_identity(command: str) -> dict[str, Any]:
    tokens = shlex.split(command)
    if not tokens:
        raise BuildAttestationError("empty compiler command")
    resolved = shutil.which(tokens[0])
    if resolved is None:
        raise BuildAttestationError(f"toolchain executable is not available: {tokens[0]}")
    executable = Path(resolved).resolve()
    completed = subprocess.run(
        [str(executable), "--version"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    version_output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0 or not version_output:
        raise BuildAttestationError(
            f"toolchain version query failed for {executable}: returncode={completed.returncode}"
        )
    return {
        "configured_command": command,
        "executable": str(executable),
        "executable_sha256": _sha256_path(executable),
        "version_output": version_output,
        "version_output_sha256": _sha256_bytes(version_output.encode("utf-8")),
    }


def _toolchain_identity() -> dict[str, Any]:
    cc = os.environ.get("CC") or str(sysconfig.get_config_var("CC") or "")
    cxx = os.environ.get("CXX") or str(sysconfig.get_config_var("CXX") or "")
    if not cc or not cxx:
        raise BuildAttestationError("Python sysconfig did not provide CC/CXX")
    return {
        "cc": _command_identity(cc),
        "cxx": _command_identity(cxx),
        "sysconfig_cc": sysconfig.get_config_var("CC"),
        "sysconfig_cxx": sysconfig.get_config_var("CXX"),
        "sysconfig_ldshared": sysconfig.get_config_var("LDSHARED"),
        "environment": {
            key: os.environ.get(key)
            for key in ("CC", "CXX", "ARCHFLAGS", "MACOSX_DEPLOYMENT_TARGET")
        },
    }


def _load_package_and_compiled_inventory(variant_dir: Path) -> tuple[dict[str, Any], Path]:
    import torch

    variant_text = str(variant_dir)
    if variant_text not in sys.path:
        sys.path.insert(0, variant_text)
    package = importlib.import_module(PACKAGE_NAME)
    ops_module = importlib.import_module(f"{PACKAGE_NAME}.ops")
    load_error = getattr(ops_module, "_EXTENSION_LOAD_ERROR", None)
    if load_error is not None:
        raise BuildAttestationError(f"native extension load failed: {load_error!r}")

    selected = _select_extension(variant_dir)
    loaded = getattr(ops_module, "_EXTENSION_LIBRARY_PATH", None)
    if loaded is None or Path(loaded).resolve() != selected:
        raise BuildAttestationError(
            f"package loaded a different extension: selected={selected}, loaded={loaded}"
        )
    package_path = Path(getattr(package, "__file__", "")).resolve()
    if package_path.parent != (variant_dir / PACKAGE_NAME).resolve():
        raise BuildAttestationError(f"imported package came from the wrong directory: {package_path}")

    prefix = f"{VARIANT_NAME}::"
    names = sorted(
        op_name[len(prefix) :]
        for op_name in torch._C._dispatch_get_all_op_names()
        if op_name.startswith(prefix)
    )
    schemas: list[str] = []
    missing_dispatch_kernels: list[str] = []
    for name in names:
        qualified = f"{prefix}{name}"
        handle = torch._C._dispatch_find_schema_or_throw(qualified, "")
        actual_schema = str(handle.schema())
        if not actual_schema.startswith(prefix):
            raise BuildAttestationError(f"compiled schema has wrong namespace: {actual_schema}")
        schemas.append(actual_schema[len(prefix) :])
        if not torch._C._dispatch_has_kernel_for_dispatch_key(qualified, DISPATCH_KEY):
            missing_dispatch_kernels.append(name)

    return (
        {
            "dispatch_key": DISPATCH_KEY,
            "schema_count": len(names),
            "schema_names": names,
            "schemas": sorted(schemas),
            "schema_name_inventory_sha256": _inventory_sha256(names),
            "full_schema_inventory_sha256": _inventory_sha256(schemas),
            "missing_dispatch_kernels": missing_dispatch_kernels,
        },
        selected,
    )


def _compare_source_and_compiled(
    source_snapshot: dict[str, Any],
    compiled_snapshot: dict[str, Any],
    bindings_source: str,
) -> list[str]:
    failures: list[str] = []
    source_schemas = _schema_inventory(bindings_source)
    compiled_schemas = {
        schema.split("(", 1)[0].strip(): schema
        for schema in compiled_snapshot["schemas"]
    }
    source_names = set(source_schemas)
    compiled_names = set(compiled_schemas)
    if source_names != compiled_names:
        failures.append(
            "compiled/source schema-name mismatch: "
            f"missing={sorted(source_names - compiled_names)}, "
            f"unexpected={sorted(compiled_names - source_names)}"
        )
    mismatched_signatures = sorted(
        name
        for name in source_names & compiled_names
        if source_schemas[name] != compiled_schemas[name]
    )
    if mismatched_signatures:
        failures.append(f"compiled schema signatures differ from bindings.cpp: {mismatched_signatures}")
    for key in (
        "schema_count",
        "schema_name_inventory_sha256",
        "full_schema_inventory_sha256",
    ):
        if compiled_snapshot.get(key) != source_snapshot.get(key):
            failures.append(
                f"compiled/source {key} mismatch: "
                f"source={source_snapshot.get(key)!r}, compiled={compiled_snapshot.get(key)!r}"
            )
    if compiled_snapshot.get("missing_dispatch_kernels"):
        failures.append(
            f"compiled schemas missing {DISPATCH_KEY} kernels: "
            f"{compiled_snapshot['missing_dispatch_kernels']}"
        )
    return failures


def _extension_snapshot(extension: Path, variant_dir: Path, source_snapshot: dict[str, Any]) -> dict[str, Any]:
    native_inputs = {
        *source_snapshot["translation_units"],
        *source_snapshot["native_dependencies"],
    }
    newest_native_input_mtime_ns = max(
        (variant_dir / relative).stat().st_mtime_ns for relative in native_inputs
    )
    extension_mtime_ns = extension.stat().st_mtime_ns
    if extension_mtime_ns < newest_native_input_mtime_ns:
        raise BuildAttestationError("native extension is older than an attested native build input")
    return {
        "path": str(extension.resolve()),
        "repo_relative_path": str(extension.resolve().relative_to(DYNAWORLD.resolve())),
        "basename": extension.name,
        "bytes": extension.stat().st_size,
        "sha256": _sha256_path(extension),
        "mtime_ns": extension_mtime_ns,
        "newest_native_input_mtime_ns": newest_native_input_mtime_ns,
    }


def build_attestation(variant_dir: Path = DEFAULT_VARIANT_DIR) -> dict[str, Any]:
    variant_dir = variant_dir.resolve()
    source_snapshot = _source_snapshot(variant_dir)
    runtime = _runtime_identity()
    toolchain = _toolchain_identity()
    compiled, extension = _load_package_and_compiled_inventory(variant_dir)
    failures = _compare_source_and_compiled(
        source_snapshot,
        compiled,
        (variant_dir / "csrc" / "bindings.cpp").read_text(encoding="utf-8"),
    )
    if failures:
        raise BuildAttestationError("; ".join(failures))
    extension_snapshot = _extension_snapshot(extension, variant_dir, source_snapshot)
    payload: dict[str, Any] = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "kind": ATTESTATION_KIND,
        "status": "accepted",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": "build_and_dispatcher_registration_only_no_metal_dispatch",
        "variant": VARIANT_NAME,
        "package": PACKAGE_NAME,
        "variant_dir": str(variant_dir),
        "receipt_schema": str(RECEIPT_SCHEMA_PATH.relative_to(DYNAWORLD)),
        "receipt_schema_sha256": _sha256_path(RECEIPT_SCHEMA_PATH),
        "source": source_snapshot,
        "runtime": runtime,
        "toolchain": toolchain,
        "extension": extension_snapshot,
        "compiled_operator_inventory": compiled,
    }
    payload["receipt_payload_sha256"] = _payload_sha256(payload)
    return payload


def _compare_exact(label: str, current: Any, recorded: Any, failures: list[str]) -> None:
    if current != recorded:
        failures.append(f"{label} changed: recorded={recorded!r}, current={current!r}")


def verify_attestation(
    receipt_path: Path,
    variant_dir: Path = DEFAULT_VARIANT_DIR,
) -> dict[str, Any]:
    failures: list[str] = []
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"status": "failed", "failures": [f"could not read receipt: {exc}"]}
    if not isinstance(payload, dict):
        return {"status": "failed", "failures": ["receipt root must be an object"]}

    for key, expected in (
        ("schema_version", ATTESTATION_SCHEMA_VERSION),
        ("kind", ATTESTATION_KIND),
        ("status", "accepted"),
        ("variant", VARIANT_NAME),
        ("package", PACKAGE_NAME),
        ("scope", "build_and_dispatcher_registration_only_no_metal_dispatch"),
    ):
        _compare_exact(key, payload.get(key), expected, failures)
    if payload.get("receipt_payload_sha256") != _payload_sha256(payload):
        failures.append("receipt payload digest mismatch")
    if not RECEIPT_SCHEMA_PATH.is_file():
        failures.append(f"receipt schema is missing: {RECEIPT_SCHEMA_PATH}")
    elif payload.get("receipt_schema_sha256") != _sha256_path(RECEIPT_SCHEMA_PATH):
        failures.append("receipt schema digest mismatch")

    variant_dir = variant_dir.resolve()
    try:
        source = _source_snapshot(variant_dir)
        runtime = _runtime_identity()
        toolchain = _toolchain_identity()
        compiled, extension = _load_package_and_compiled_inventory(variant_dir)
        extension_snapshot = _extension_snapshot(extension, variant_dir, source)
        failures.extend(
            _compare_source_and_compiled(
                source,
                compiled,
                (variant_dir / "csrc" / "bindings.cpp").read_text(encoding="utf-8"),
            )
        )
    except Exception as exc:
        failures.append(f"current build could not be re-attested: {exc}")
    else:
        for label, current, recorded in (
            ("variant_dir", str(variant_dir), payload.get("variant_dir")),
            ("source", source, payload.get("source")),
            ("runtime", runtime, payload.get("runtime")),
            ("toolchain", toolchain, payload.get("toolchain")),
            ("extension", extension_snapshot, payload.get("extension")),
            ("compiled_operator_inventory", compiled, payload.get("compiled_operator_inventory")),
        ):
            _compare_exact(label, current, recorded, failures)

    return {
        "status": "accepted" if not failures else "failed",
        "failures": failures,
        "receipt": str(receipt_path),
        "receipt_payload_sha256": payload.get("receipt_payload_sha256"),
        "schema_count": payload.get("compiled_operator_inventory", {}).get("schema_count"),
        "extension": payload.get("extension", {}).get("path"),
        "no_metal_dispatch": True,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant-dir", type=Path, default=DEFAULT_VARIANT_DIR)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--source-only", action="store_true")
    action.add_argument("--write-receipt", type=Path, nargs="?", const=DEFAULT_RECEIPT)
    action.add_argument("--verify-receipt", type=Path, nargs="?", const=DEFAULT_RECEIPT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.source_only:
        payload = {
            "status": "accepted",
            "scope": "source_only_no_extension_import_no_metal_dispatch",
            "source": _source_snapshot(args.variant_dir.resolve()),
        }
    elif args.write_receipt is not None:
        try:
            payload = build_attestation(args.variant_dir)
        except Exception as exc:
            print(json.dumps({"status": "failed", "failures": [str(exc)]}, indent=2, sort_keys=True))
            return 1
        args.write_receipt.parent.mkdir(parents=True, exist_ok=True)
        args.write_receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        payload = verify_attestation(args.verify_receipt, args.variant_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
