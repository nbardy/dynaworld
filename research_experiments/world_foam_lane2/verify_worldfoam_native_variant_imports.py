#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import re
import sys
from typing import Any

import torch

from verify_worldfoam_native_variant_sources import DEFAULT_VARIANT_ROOT, DEFAULT_VARIANTS, _schema_names


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _extension_candidates(variant_dir: Path, package_name: str) -> list[Path]:
    return sorted((variant_dir / package_name).glob("_C*.so"))


def _compiled_source_paths(variant_dir: Path) -> list[Path]:
    metal_host = variant_dir / "csrc" / "metal" / "world_foam_lane2_metal.mm"
    sources = [variant_dir / "csrc" / "bindings.cpp", metal_host]
    for filename in sorted(set(re.findall(r'stringByAppendingPathComponent:@"([^"]+\.metal)"', _read(metal_host)))):
        sources.append(variant_dir / "csrc" / "metal" / filename)
    return sources


def _variant_import_summary(variant_root: Path, *, variant_name: str, package_name: str) -> dict[str, Any]:
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

    schemas = _schema_names(_read(bindings))
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
    if ops_module is not None:
        error = getattr(ops_module, "_EXTENSION_LOAD_ERROR", None)
        if error is not None:
            extension_load_error = repr(error)
            failures.append(f"extension load failed during package import: {extension_load_error}")

    registered_schemas: list[str] = []
    missing_registered_schemas: list[str] = []
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

    return {
        "variant": variant_name,
        "package": package_name,
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "schema_count": len(schemas),
        "registered_schema_count": len(registered_schemas),
        "missing_registered_schemas": missing_registered_schemas,
        "extension_library": str(extension_library),
        "extension_mtime": extension_library.stat().st_mtime,
        "compiled_source_count": len(compiled_sources),
        "imported_package_file": str(getattr(package, "__file__", "")) if package is not None else "",
        "extension_load_error": extension_load_error,
        "import_error": import_error,
    }


def verify(
    variant_root: Path = DEFAULT_VARIANT_ROOT,
    variants: tuple[tuple[str, str], ...] = DEFAULT_VARIANTS,
) -> dict[str, Any]:
    rows = [
        _variant_import_summary(variant_root, variant_name=variant_name, package_name=package_name)
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
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = verify(args.variant_root)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
