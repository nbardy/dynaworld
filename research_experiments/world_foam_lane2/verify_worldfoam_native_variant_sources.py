#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any


DYNAWORLD = Path(__file__).resolve().parents[2]
DEFAULT_VARIANT_ROOT = DYNAWORLD / "third_party" / "fast-mac-gsplat" / "variants"
DEFAULT_VARIANTS: tuple[tuple[str, str], ...] = (
    ("world_foam_lane2_fused_slab_v0", "torch_world_foam_lane2_fused_slab"),
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _schema_names(bindings_cpp: str) -> set[str]:
    return {
        schema.split("(", 1)[0].strip()
        for schema in re.findall(r'm\.def\(\s*"([^"]+)"', bindings_cpp, flags=re.DOTALL)
    }


def _impl_names(bindings_cpp: str) -> set[str]:
    return set(re.findall(r'm\.impl\(\s*"([^"]+)"', bindings_cpp, flags=re.DOTALL))


def _impl_targets(bindings_cpp: str) -> dict[str, str]:
    raw_targets = re.findall(
        r'm\.impl\(\s*"([^"]+)"\s*,\s*([A-Za-z_][A-Za-z0-9_]*(?:::\s*[A-Za-z_][A-Za-z0-9_]*)*)',
        bindings_cpp,
        flags=re.DOTALL,
    )
    return {name: re.sub(r"\s+", "", target) for name, target in raw_targets}


def _source_without_torch_registrations(bindings_cpp: str) -> str:
    return bindings_cpp.split("TORCH_LIBRARY", 1)[0]


def _missing_impl_target_definitions(bindings_cpp: str, metal_mm: str, impl_targets: dict[str, str]) -> list[str]:
    searchable_source = _source_without_torch_registrations(bindings_cpp) + "\n" + metal_mm
    missing = []
    for op_name, target in sorted(impl_targets.items()):
        symbol = target.rsplit("::", 1)[-1]
        if not re.search(rf"\b{re.escape(symbol)}\s*\(", searchable_source):
            missing.append(f"{op_name} -> {target}")
    return missing


def _torch_ops_refs(ops_py: str) -> set[str]:
    return {
        name
        for name in re.findall(r"\bops\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", ops_py)
        if name != "load_library"
    }


def _host_kernel_refs(metal_mm: str) -> set[str]:
    return set(re.findall(r'getKernelFunction\("([^"]+)"\)', metal_mm))


def _loaded_metal_filenames(metal_mm: str) -> list[str]:
    return sorted(set(re.findall(r'stringByAppendingPathComponent:@"([^"]+\.metal)"', metal_mm)))


def _metal_kernel_fields(metal_mm: str) -> set[str]:
    match = re.search(r"struct\s+MetalKernels\s*\{(.*?)\n\};", metal_mm, flags=re.DOTALL)
    if not match:
        return set()
    return set(
        re.findall(
            r"std::shared_ptr<MetalKernelFunction>\s+([A-Za-z_][A-Za-z0-9_]*)\s*;",
            match.group(1),
        )
    )


def _metal_kernel_initializers(metal_mm: str) -> set[str]:
    return set(re.findall(r"\bout\.([A-Za-z_][A-Za-z0-9_]*)\s*=", metal_mm))


def _metal_kernel_field_uses(metal_mm: str) -> set[str]:
    return set(re.findall(r"\bkernels\(\)\.([A-Za-z_][A-Za-z0-9_]*)\b", metal_mm))


def _metal_kernel_names(variant_dir: Path) -> set[str]:
    kernels: set[str] = set()
    for path in sorted((variant_dir / "csrc" / "metal").glob("*.metal")):
        kernels.update(re.findall(r"\bkernel\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", _read(path)))
    return kernels


def _loaded_metal_kernel_names(variant_dir: Path, loaded_filenames: list[str]) -> tuple[set[str], list[str]]:
    kernels: set[str] = set()
    missing_files: list[str] = []
    metal_dir = variant_dir / "csrc" / "metal"
    for filename in loaded_filenames:
        path = metal_dir / filename
        if not path.exists():
            missing_files.append(str(path))
            continue
        kernels.update(re.findall(r"\bkernel\s+void\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", _read(path)))
    return kernels, missing_files


def _variant_summary(variant_root: Path, *, variant_name: str, package_name: str) -> dict[str, Any]:
    variant_dir = variant_root / variant_name
    bindings = variant_dir / "csrc" / "bindings.cpp"
    metal_host = variant_dir / "csrc" / "metal" / "world_foam_lane2_metal.mm"
    ops_py = variant_dir / package_name / "ops.py"
    failures: list[str] = []
    missing_files = [str(path) for path in (bindings, metal_host, ops_py) if not path.exists()]
    if missing_files:
        return {
            "variant": variant_name,
            "package": package_name,
            "status": "failed",
            "failures": [f"missing required source file: {path}" for path in missing_files],
            "missing_files": missing_files,
        }

    bindings_source = _read(bindings)
    metal_host_source = _read(metal_host)
    schemas = _schema_names(bindings_source)
    impls = _impl_names(bindings_source)
    impl_targets = _impl_targets(bindings_source)
    refs = _torch_ops_refs(_read(ops_py))
    host_kernels = _host_kernel_refs(metal_host_source)
    loaded_metal_filenames = _loaded_metal_filenames(metal_host_source)
    host_kernel_fields = _metal_kernel_fields(metal_host_source)
    initialized_kernel_fields = _metal_kernel_initializers(metal_host_source)
    used_kernel_fields = _metal_kernel_field_uses(metal_host_source)
    metal_kernels = _metal_kernel_names(variant_dir)
    loaded_metal_kernels, missing_loaded_metal_files = _loaded_metal_kernel_names(
        variant_dir, loaded_metal_filenames
    )

    if not schemas:
        failures.append("bindings.cpp registers no TORCH_LIBRARY schemas")
    if not impls:
        failures.append("bindings.cpp registers no TORCH_LIBRARY_IMPL kernels")
    if not host_kernels:
        failures.append("world_foam_lane2_metal.mm loads no Metal kernels")
    if not loaded_metal_filenames:
        failures.append("load_shader_source does not append any .metal source files")
    if not host_kernel_fields:
        failures.append("world_foam_lane2_metal.mm declares no MetalKernels fields")
    if not metal_kernels:
        failures.append("variant has no Metal kernel declarations")

    missing_impls = sorted(impls - schemas)
    missing_ref_schemas = sorted(refs - schemas)
    missing_ref_impls = sorted(refs - impls)
    missing_impl_targets = sorted(impls - set(impl_targets))
    missing_target_definitions = _missing_impl_target_definitions(bindings_source, metal_host_source, impl_targets)
    missing_metal_kernels = sorted(host_kernels - metal_kernels)
    missing_loaded_metal_kernels = sorted(host_kernels - loaded_metal_kernels)
    initialized_without_field = sorted(initialized_kernel_fields - host_kernel_fields)
    fields_without_initializer = sorted(host_kernel_fields - initialized_kernel_fields)
    used_without_field = sorted(used_kernel_fields - host_kernel_fields)
    used_without_initializer = sorted(used_kernel_fields - initialized_kernel_fields)
    if missing_impls:
        failures.append(f"impls without schema definitions: {missing_impls}")
    if missing_ref_schemas:
        failures.append(f"Python ops references without schema definitions: {missing_ref_schemas}")
    if missing_ref_impls:
        failures.append(f"Python ops references without native implementations: {missing_ref_impls}")
    if missing_impl_targets:
        failures.append(f"native implementations without parseable dispatch targets: {missing_impl_targets}")
    if missing_target_definitions:
        failures.append(f"native dispatch targets without source definitions: {missing_target_definitions}")
    if missing_metal_kernels:
        failures.append(f"host loads Metal kernels that are not declared: {missing_metal_kernels}")
    if missing_loaded_metal_files:
        failures.append(f"load_shader_source references missing .metal files: {missing_loaded_metal_files}")
    if missing_loaded_metal_kernels:
        failures.append(
            f"host loads Metal kernels absent from dynamically loaded Metal sources: {missing_loaded_metal_kernels}"
        )
    if initialized_without_field:
        failures.append(f"host initializes Metal kernel fields that are not declared: {initialized_without_field}")
    if fields_without_initializer:
        failures.append(f"declared Metal kernel fields without initializers: {fields_without_initializer}")
    if used_without_field:
        failures.append(f"host uses Metal kernel fields that are not declared: {used_without_field}")
    if used_without_initializer:
        failures.append(f"host uses Metal kernel fields that are not initialized: {used_without_initializer}")

    return {
        "variant": variant_name,
        "package": package_name,
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "schema_count": len(schemas),
        "impl_count": len(impls),
        "impl_target_count": len(impl_targets),
        "python_ops_ref_count": len(refs),
        "loaded_metal_files": loaded_metal_filenames,
        "loaded_metal_file_count": len(loaded_metal_filenames),
        "loaded_metal_kernel_count": len(loaded_metal_kernels),
        "host_kernel_ref_count": len(host_kernels),
        "host_kernel_field_count": len(host_kernel_fields),
        "initialized_kernel_field_count": len(initialized_kernel_fields),
        "used_kernel_field_count": len(used_kernel_fields),
        "metal_kernel_count": len(metal_kernels),
        "metal_kernels_without_host_loader": sorted(metal_kernels - host_kernels),
    }


def verify(variant_root: Path = DEFAULT_VARIANT_ROOT, variants: tuple[tuple[str, str], ...] = DEFAULT_VARIANTS) -> dict[str, Any]:
    rows = [
        _variant_summary(variant_root, variant_name=variant_name, package_name=package_name)
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
    parser = argparse.ArgumentParser(description="Validate WorldFoam native variant source wiring without building.")
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
