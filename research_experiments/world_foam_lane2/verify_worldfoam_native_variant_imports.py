#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

from attest_worldfoam_fused_slab_build import (
    DEFAULT_RECEIPT,
    PACKAGE_NAME as ATTESTED_PACKAGE_NAME,
    VARIANT_NAME as ATTESTED_VARIANT_NAME,
    _compare_source_and_compiled,
    _load_package_and_compiled_inventory,
    _source_snapshot,
    verify_attestation,
)
from verify_worldfoam_native_variant_sources import (
    DEFAULT_VARIANT_ROOT,
    DEFAULT_VARIANTS,
    _load_native_build_contract,
    _schema_inventory,
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extension_candidates(variant_dir: Path, package_name: str) -> list[Path]:
    return sorted((variant_dir / package_name).glob("_C*.so"))


def _compiled_source_paths(variant_dir: Path) -> list[Path]:
    contract, summary, failures = _load_native_build_contract(variant_dir)
    if failures or contract is None or summary is None:
        return []
    return [
        variant_dir / relative
        for relative in dict.fromkeys(
            (*summary["translation_units"], *summary["native_dependencies"])
        )
    ]


def _variant_import_summary(
    variant_root: Path,
    *,
    variant_name: str,
    package_name: str,
    attestation_path: Path | None = DEFAULT_RECEIPT,
) -> dict[str, Any]:
    variant_dir = variant_root / variant_name
    bindings = variant_dir / "csrc" / "bindings.cpp"
    package_dir = variant_dir / package_name
    failures: list[str] = []
    missing_files = [str(path) for path in (bindings, package_dir) if not path.exists()]
    if missing_files:
        return {
            "variant": variant_name,
            "package": package_name,
            "status": "failed",
            "failures": [f"missing required file or directory: {path}" for path in missing_files],
            "missing_files": missing_files,
        }

    source_schema_inventory = _schema_inventory(_read(bindings))
    schemas = set(source_schema_inventory)
    build_contract, build_contract_summary, build_contract_failures = (
        _load_native_build_contract(variant_dir)
    )
    failures.extend(build_contract_failures)
    extension_candidates = _extension_candidates(variant_dir, package_name)
    if not extension_candidates:
        return {
            "variant": variant_name,
            "package": package_name,
            "status": "failed",
            "failures": [f"missing built extension library under {package_dir}"],
            "schema_count": len(schemas),
            "extension_library": None,
        }

    if len(extension_candidates) != 1:
        failures.append(
            f"expected exactly one built extension library, found {len(extension_candidates)}: "
            f"{extension_candidates}"
        )
    extension_library = extension_candidates[0]
    compiled_sources = _compiled_source_paths(variant_dir)
    missing_compiled_sources = [str(path) for path in compiled_sources if not path.exists()]
    if missing_compiled_sources:
        failures.append(f"compiled source files missing: {missing_compiled_sources}")
    else:
        newest_source_mtime = max(path.stat().st_mtime for path in compiled_sources)
        if extension_library.stat().st_mtime < newest_source_mtime:
            failures.append("built extension library is older than compiled source files")

    if str(variant_dir) not in sys.path:
        sys.path.insert(0, str(variant_dir))
    import_error = ""
    try:
        package = importlib.import_module(package_name)
        ops_module = importlib.import_module(f"{package_name}.ops")
    except Exception as exc:
        package = None
        ops_module = None
        import_error = repr(exc)
        failures.append(f"package import failed: {import_error}")

    extension_load_error = ""
    kinetic_compiled_abi_attestation_error = ""
    kinetic_compiled_abi_attestation_present = False
    kinetic_lazy_full_geometry_compiled_abi_attestation_error = ""
    kinetic_lazy_full_geometry_compiled_abi_attestation_present = False
    if ops_module is not None:
        error = getattr(ops_module, "_EXTENSION_LOAD_ERROR", None)
        if error is not None:
            extension_load_error = repr(error)
            failures.append(f"extension load failed during package import: {extension_load_error}")
        attestation = getattr(
            ops_module,
            "assert_kinetic_memory_light_compiled_abi_registered",
            None,
        )
        kinetic_compiled_abi_attestation_present = callable(attestation)
        if not kinetic_compiled_abi_attestation_present:
            failures.append("kinetic memory-light compiled ABI attestation is missing")
        else:
            try:
                attestation()
            except Exception as exc:
                kinetic_compiled_abi_attestation_error = repr(exc)
                failures.append(
                    "kinetic memory-light compiled ABI attestation failed: "
                    f"{kinetic_compiled_abi_attestation_error}"
                )
        full_geometry_attestation = getattr(
            ops_module,
            "assert_kinetic_lazy_full_geometry_compiled_abi_registered",
            None,
        )
        kinetic_lazy_full_geometry_compiled_abi_attestation_present = callable(
            full_geometry_attestation
        )
        if not kinetic_lazy_full_geometry_compiled_abi_attestation_present:
            failures.append(
                "kinetic lazy full-geometry compiled ABI attestation is missing"
            )
        else:
            try:
                full_geometry_attestation()
            except Exception as exc:
                kinetic_lazy_full_geometry_compiled_abi_attestation_error = repr(
                    exc
                )
                failures.append(
                    "kinetic lazy full-geometry compiled ABI attestation failed: "
                    f"{kinetic_lazy_full_geometry_compiled_abi_attestation_error}"
                )

    registered_schemas: list[str] = []
    missing_registered_schemas: list[str] = []
    unexpected_registered_schemas: list[str] = []
    mismatched_registered_schema_signatures: list[str] = []
    missing_dispatch_kernels: list[str] = []
    compiled_schema_inventory_sha256: str | None = None
    exact_schema_inventory_match = False
    compiled_snapshot: dict[str, Any] | None = None
    if (
        variant_name == ATTESTED_VARIANT_NAME
        and package_name == ATTESTED_PACKAGE_NAME
        and package is not None
        and ops_module is not None
        and not extension_load_error
    ):
        try:
            source_snapshot = _source_snapshot(variant_dir)
            compiled_snapshot, selected_extension = (
                _load_package_and_compiled_inventory(variant_dir)
            )
            if selected_extension != extension_library.resolve():
                failures.append("compiled inventory selected a different extension library")
            inventory_failures = _compare_source_and_compiled(
                source_snapshot,
                compiled_snapshot,
                _read(bindings),
            )
            failures.extend(inventory_failures)
            compiled_names = set(compiled_snapshot["schema_names"])
            registered_schemas = sorted(schemas & compiled_names)
            missing_registered_schemas = sorted(schemas - compiled_names)
            unexpected_registered_schemas = sorted(compiled_names - schemas)
            compiled_schema_map = {
                schema.split("(", 1)[0].strip(): schema
                for schema in compiled_snapshot["schemas"]
            }
            mismatched_registered_schema_signatures = sorted(
                name
                for name in schemas & compiled_names
                if source_schema_inventory[name] != compiled_schema_map[name]
            )
            missing_dispatch_kernels = list(
                compiled_snapshot["missing_dispatch_kernels"]
            )
            compiled_schema_inventory_sha256 = compiled_snapshot[
                "full_schema_inventory_sha256"
            ]
            exact_schema_inventory_match = not inventory_failures
        except Exception as exc:
            failures.append(f"exact compiled schema inventory audit failed: {exc}")
    else:
        ops_namespace = getattr(torch.ops, variant_name)
        for schema_name in sorted(schemas):
            try:
                getattr(ops_namespace, schema_name)
            except AttributeError:
                missing_registered_schemas.append(schema_name)
            else:
                registered_schemas.append(schema_name)
    if missing_registered_schemas:
        failures.append(f"compiled schemas missing after package import: {missing_registered_schemas}")
    if unexpected_registered_schemas:
        failures.append(
            f"compiled schemas absent from bindings.cpp: {unexpected_registered_schemas}"
        )
    if mismatched_registered_schema_signatures:
        failures.append(
            "compiled schema signatures differ from bindings.cpp: "
            f"{mismatched_registered_schema_signatures}"
        )
    if missing_dispatch_kernels:
        failures.append(
            "compiled schemas missing CompositeExplicitAutograd kernels: "
            f"{missing_dispatch_kernels}"
        )

    attestation_summary: dict[str, Any] | None = None
    if variant_name == ATTESTED_VARIANT_NAME and package_name == ATTESTED_PACKAGE_NAME:
        if attestation_path is None:
            failures.append("native build attestation path is required")
        elif not attestation_path.is_file():
            failures.append(f"native build attestation is missing: {attestation_path}")
        else:
            attestation_summary = verify_attestation(attestation_path, variant_dir)
            if attestation_summary["status"] != "accepted":
                failures.extend(
                    f"native build attestation: {failure}"
                    for failure in attestation_summary["failures"]
                )

    rebuild_required = bool(
        missing_registered_schemas
        or unexpected_registered_schemas
        or mismatched_registered_schema_signatures
        or missing_dispatch_kernels
        or any("older than" in failure for failure in failures)
    )

    return {
        "variant": variant_name,
        "package": package_name,
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "schema_count": len(schemas),
        "source_schema_inventory_sha256": (
            build_contract_summary.get("full_schema_inventory_sha256")
            if build_contract_summary is not None
            else None
        ),
        "registered_schema_count": len(registered_schemas),
        "missing_registered_schemas": missing_registered_schemas,
        "unexpected_registered_schemas": unexpected_registered_schemas,
        "mismatched_registered_schema_signatures": (
            mismatched_registered_schema_signatures
        ),
        "missing_dispatch_kernels": missing_dispatch_kernels,
        "compiled_schema_inventory_sha256": compiled_schema_inventory_sha256,
        "exact_schema_inventory_match": exact_schema_inventory_match,
        "rebuild_required": rebuild_required,
        "extension_library": str(extension_library),
        "extension_mtime": extension_library.stat().st_mtime,
        "compiled_source_count": len(compiled_sources),
        "imported_package_file": str(getattr(package, "__file__", "")) if package is not None else "",
        "extension_load_error": extension_load_error,
        "kinetic_compiled_abi_attestation_present": (
            kinetic_compiled_abi_attestation_present
        ),
        "kinetic_compiled_abi_attestation_error": (
            kinetic_compiled_abi_attestation_error
        ),
        "kinetic_lazy_full_geometry_compiled_abi_attestation_present": (
            kinetic_lazy_full_geometry_compiled_abi_attestation_present
        ),
        "kinetic_lazy_full_geometry_compiled_abi_attestation_error": (
            kinetic_lazy_full_geometry_compiled_abi_attestation_error
        ),
        "import_error": import_error,
        "build_contract_loaded": build_contract is not None,
        "build_contract_schema_count": (
            build_contract_summary.get("schema_count")
            if build_contract_summary is not None
            else None
        ),
        "attestation_required": variant_name == ATTESTED_VARIANT_NAME,
        "attestation_path": str(attestation_path) if attestation_path is not None else None,
        "attestation": attestation_summary,
        "compiled_operator_inventory": compiled_snapshot,
    }


def verify(
    variant_root: Path = DEFAULT_VARIANT_ROOT,
    variants: tuple[tuple[str, str], ...] = DEFAULT_VARIANTS,
    attestation_path: Path | None = DEFAULT_RECEIPT,
) -> dict[str, Any]:
    rows = [
        _variant_import_summary(
            variant_root,
            variant_name=variant_name,
            package_name=package_name,
            attestation_path=(
                attestation_path if variant_name == ATTESTED_VARIANT_NAME else None
            ),
        )
        for variant_name, package_name in variants
    ]
    failures = [
        f"{row['variant']}: {failure}"
        for row in rows
        for failure in row.get("failures", [])
    ]
    return {
        "status": "ok" if not failures else "failed",
        "variant_root": str(variant_root),
        "variant_count": len(rows),
        "variants": rows,
        "failures": failures,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate built WorldFoam native variant imports and op registration.")
    parser.add_argument("--variant-root", type=Path, default=DEFAULT_VARIANT_ROOT)
    parser.add_argument("--attestation-json", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = verify(args.variant_root, attestation_path=args.attestation_json)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
