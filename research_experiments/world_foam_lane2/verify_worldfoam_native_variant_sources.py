#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any

DYNAWORLD = Path(__file__).resolve().parents[2]
DEFAULT_VARIANT_ROOT = DYNAWORLD / "third_party" / "fast-mac-gsplat" / "variants"
DEFAULT_VARIANTS: tuple[tuple[str, str], ...] = (
    ("world_foam_lane2_fused_slab_v0", "torch_world_foam_lane2_fused_slab"),
)
BUILD_CONTRACT_VARIANT_NAME = "world_foam_lane2_fused_slab_v0"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _schema_inventory(bindings_cpp: str) -> dict[str, str]:
    schemas = re.findall(r'm\.def\(\s*"([^"]+)"', bindings_cpp, flags=re.DOTALL)
    inventory: dict[str, str] = {}
    for schema in schemas:
        name = schema.split("(", 1)[0].strip()
        if name in inventory:
            raise ValueError(f"duplicate schema name: {name}")
        inventory[name] = schema
    return inventory


def _schema_names(bindings_cpp: str) -> set[str]:
    return set(_schema_inventory(bindings_cpp))


def _load_native_build_contract(variant_dir: Path) -> tuple[Any | None, dict[str, Any] | None, list[str]]:
    contract_path = variant_dir / "native_build_contract.py"
    if not contract_path.is_file():
        return None, None, [f"missing native build contract: {contract_path}"]
    module_name = f"_worldfoam_native_build_contract_{hashlib.sha256(str(contract_path).encode()).hexdigest()[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, contract_path)
    if spec is None or spec.loader is None:
        return None, None, [f"could not load native build contract: {contract_path}"]
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        summary = module.validate_source_contract(variant_dir)
    except Exception as exc:
        return module, None, [f"native build contract rejected source: {exc}"]
    return module, summary, []


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
    indirect = {name for name in re.findall(r"\bops\.([A-Za-z_][A-Za-z0-9_]*)\s*\(", ops_py) if name != "load_library"}
    direct = set(
        re.findall(
            r"\btorch\.ops\.[A-Za-z_][A-Za-z0-9_]*\.([A-Za-z_][A-Za-z0-9_]*)\s*\(",
            ops_py,
        )
    )
    return indirect | direct


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


def _brace_delimited_source(source: str, signature: str) -> str:
    """Return one function/kernel body, or an empty string when absent."""

    start = source.find(signature)
    if start < 0:
        return ""
    cursor = source.find("{", start)
    if cursor < 0:
        return ""
    depth = 0
    while cursor < len(source):
        if source[cursor] == "{":
            depth += 1
        elif source[cursor] == "}":
            depth -= 1
            if depth == 0:
                return source[start : cursor + 1]
        cursor += 1
    return ""


def _kinetic_memory_light_source_contract(
    bindings_source: str,
    metal_host_source: str,
    metal_source: str,
) -> dict[str, Any]:
    """Audit allocation shapes in the bounded precompiled-length hot path.

    This deliberately proves only source structure.  Invocation frequency,
    allocator peaks, command-buffer overlap, and release timing require a
    rebuilt runtime plus step-level telemetry.
    """

    failures: list[str] = []
    material_name = "kinetic_precompiled_length_p0_lie_material_node_vjp_accumulate_launch_only"
    ragged_loss_name = "kinetic_ragged_p0_lie_sample_accumulate_loss_only_launch_only"
    forward_oracle = _brace_delimited_source(
        metal_host_source,
        "metal_kinetic_precompiled_length_p0_lie_node_forward_launch_only(",
    )
    forward_into_name = (
        "kinetic_precompiled_length_p0_lie_node_forward_into_launch_only_v1"
    )
    forward_into = _brace_delimited_source(
        metal_host_source,
        f"metal_{forward_into_name}(",
    )
    material_host = _brace_delimited_source(metal_host_source, f"metal_{material_name}(")
    ragged_loss_host = _brace_delimited_source(metal_host_source, f"metal_{ragged_loss_name}(")
    material_kernel = _brace_delimited_source(
        metal_source,
        "kernel void wf2_kinetic_precompiled_length_p0_lie_material_node_vjp_tensor(",
    )
    ragged_loss_kernel = _brace_delimited_source(
        metal_source,
        "kernel void wf2_kinetic_ragged_p0_lie_sample_mse_vjp_accumulate_only_tensor(",
    )
    required_bodies = {
        "kinetic return-allocating node forward oracle host": forward_oracle,
        "kinetic caller-owned-output node forward host": forward_into,
        "kinetic material VJP host": material_host,
        "kinetic ragged loss-only host": ragged_loss_host,
        "kinetic material VJP kernel": material_kernel,
        "kinetic ragged loss-only kernel": ragged_loss_kernel,
    }
    for label, body in required_bodies.items():
        if not body:
            failures.append(f"missing {label}")
    if forward_oracle and "{track_count, node_count, 4}" not in forward_oracle:
        failures.append("kinetic node forward oracle does not allocate only [R,J,4] node state")
    if forward_into:
        for forbidden in ("torch::empty", "torch::zeros"):
            if forbidden in forward_into:
                failures.append(
                    "kinetic caller-owned-output forward allocates forbidden "
                    f"scratch: {forbidden}"
                )
        for required in (
            "fn.setArg(4, node_chart_out_f32);",
            "return node_chart_out_f32;",
        ):
            if required not in forward_into:
                failures.append(
                    "kinetic caller-owned-output forward misses lifetime-safe "
                    f"contract: {required}"
                )
    if material_host:
        for forbidden in ("torch::empty", "torch::zeros", "grad_node_physical_length"):
            if forbidden in material_host:
                failures.append(f"kinetic material VJP host retains forbidden allocation fragment: {forbidden}")
        for required in (
            "fn.setArg(6, grad_site_rgba_f32);",
            "fn.setArg(7, grad_site_rgba_f32);",
            "return grad_site_rgba_f32;",
        ):
            if required not in material_host:
                failures.append(f"kinetic material VJP host missing no-length-bar contract: {required}")
    if material_kernel:
        if "false," not in material_kernel or "unused_length_bar_f32" not in material_kernel:
            failures.append("kinetic material VJP kernel does not disable the [J,W] length-bar write")
        if "grad_node_physical_length_f32[length_index]" in material_kernel:
            failures.append("kinetic material VJP kernel directly writes a [J,W] length bar")
    if ragged_loss_host:
        for forbidden in ("torch::empty", "torch::zeros", "prediction_rgb"):
            if forbidden in ragged_loss_host:
                failures.append(f"ragged loss-only host retains forbidden output allocation: {forbidden}")
    if ragged_loss_kernel and "prediction_rgb_f32" in ragged_loss_kernel:
        failures.append("ragged loss-only kernel writes a prediction tensor")
    if f'"{material_name}(' not in bindings_source:
        failures.append("bindings omit the kinetic material-only VJP schema")
    if f'"{forward_into_name}(' not in bindings_source:
        failures.append("bindings omit the caller-owned-output forward schema")
    return {
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "source_contract_only": True,
        "native_runtime_built_or_executed": False,
        "allocator_peak_measured": False,
        "invocation_frequency_verified": False,
        "proven_live_shapes": {
            "compiled_node_state_and_cotangent": "2 * float32[R,J,4]",
            "node_forward_output": "caller-owned float32[R,J,4]",
            "precompiled_physical_lengths": "float32[J,W]",
            "ragged_sample_rows": "int32[N]",
            "ragged_sample_weights": "float32[N,J]",
            "ragged_targets": "float32[N,3]",
            "material_bar": "caller-owned float32[S_block,4]",
            "material_length_bar": "0 bytes",
            "optional_geometry_length_bar": "float32[J,W]",
        },
        "unproven_lifecycle_requirement": (
            "spatial bundle outer / K chunks inner; node forward and material VJP exactly once "
            "per active native block per optimizer step"
        ),
    }


def _variant_summary(variant_root: Path, *, variant_name: str, package_name: str) -> dict[str, Any]:
    variant_dir = variant_root / variant_name
    bindings = variant_dir / "csrc" / "bindings.cpp"
    metal_host = variant_dir / "csrc" / "metal" / "world_foam_lane2_metal.mm"
    ops_py = variant_dir / package_name / "ops.py"
    setup_py = variant_dir / "setup.py"
    build_contract_path = variant_dir / "native_build_contract.py"
    failures: list[str] = []
    required_files = [bindings, metal_host, ops_py]
    if variant_name == BUILD_CONTRACT_VARIANT_NAME:
        required_files.append(setup_py)
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        return {
            "variant": variant_name,
            "package": package_name,
            "status": "failed",
            "failures": [f"missing required source file: {path}" for path in missing_files],
            "missing_files": missing_files,
        }

    contract_module: Any | None = None
    build_contract: dict[str, Any] | None = None
    if variant_name == BUILD_CONTRACT_VARIANT_NAME:
        contract_module, build_contract, contract_failures = (
            _load_native_build_contract(variant_dir)
        )
        failures.extend(contract_failures)
        setup_source = _read(setup_py)
        for required in (
            "from native_build_contract import",
            "TRANSLATION_UNITS",
            "NATIVE_DEPENDENCIES",
            "Path.cwd().resolve() != this_dir",
            "validate_source_contract(this_dir)",
            "depends=depends",
        ):
            if required not in setup_source:
                failures.append(
                    f"setup.py does not consume native build contract: {required}"
                )

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
    loaded_metal_kernels, missing_loaded_metal_files = _loaded_metal_kernel_names(variant_dir, loaded_metal_filenames)
    kinetic_memory_contract: dict[str, Any] | None = None
    if "kinetic_precompiled_length_p0_lie_node_forward_launch_only" in schemas:
        shared_replay = variant_dir / "csrc" / "metal" / "world_foam_lane2_shared_replay_tensor.metal"
        if not shared_replay.exists():
            failures.append(f"missing kinetic shared-replay Metal source: {shared_replay}")
        else:
            kinetic_memory_contract = _kinetic_memory_light_source_contract(
                bindings_source,
                metal_host_source,
                _read(shared_replay),
            )
            failures.extend(
                f"kinetic memory contract: {failure}"
                for failure in kinetic_memory_contract["failures"]
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
        "kinetic_memory_contract": kinetic_memory_contract,
        "build_contract": build_contract,
        "build_contract_path": (
            str(build_contract_path)
            if variant_name == BUILD_CONTRACT_VARIANT_NAME
            else None
        ),
        "build_contract_sha256": (
            _sha256(build_contract_path)
            if variant_name == BUILD_CONTRACT_VARIANT_NAME
            and build_contract_path.is_file()
            else None
        ),
        "build_contract_module_loaded": contract_module is not None,
    }


def verify(
    variant_root: Path = DEFAULT_VARIANT_ROOT, variants: tuple[tuple[str, str], ...] = DEFAULT_VARIANTS
) -> dict[str, Any]:
    rows = [
        _variant_summary(variant_root, variant_name=variant_name, package_name=package_name)
        for variant_name, package_name in variants
    ]
    failures = [f"{row['variant']}: {failure}" for row in rows for failure in row.get("failures", [])]
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
