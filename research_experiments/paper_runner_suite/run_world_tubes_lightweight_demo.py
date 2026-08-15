from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import math
import platform
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[2]
STAR_UVT_ROOT = (
    ROOT / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
)
STAR_UVT_PACKAGE = STAR_UVT_ROOT / "torch_gsplat_bridge_star_uvt"
for path in (ROOT, ROOT / "src" / "train", STAR_UVT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


BENCHMARK = "world_tubes_lightweight_replay_compiled_demo"
SCHEMA_VERSION = 1
DEFAULT_OUT_DIR = ROOT / "outputs" / "demos" / "world_tubes_lightweight"
DEFAULT_FRAMES = 8
DEFAULT_IMAGE_SIZE = 16
MAX_FRAMES = 32
MAX_IMAGE_SIZE = 64
SIGMA_PX = 1.25
ACCEPTANCE = {
    "image_max_abs_error": 1.0e-5,
    "gradient_global_normalized_l2_error": 1.0e-5,
    "gradient_max_parameter_normalized_l2_error": 1.0e-5,
    "min_world_vjp_l2_norm": 1.0e-12,
}
SOURCE_PATHS = (
    Path(__file__).resolve(),
    ROOT
    / "research_experiments"
    / "star_uvt_feature_tubes"
    / "projective_decisive_demo_report.py",
    STAR_UVT_PACKAGE / "projective_trace.py",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to_root(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _file_identity(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"demo artifact is missing: {resolved}")
    if relative_to is None:
        display_path = _relative_to_root(resolved)
    else:
        display_path = str(resolved.relative_to(relative_to.resolve()))
    return {
        "path": display_path,
        "bytes": int(resolved.stat().st_size),
        "sha256": _sha256_file(resolved),
    }


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _native_build_instruction() -> str:
    return (
        f"( cd {STAR_UVT_ROOT} && "
        f"uv run --project {ROOT} python setup.py build_ext --inplace )"
    )


def require_star_uvt_native_binary(
    package_dir: Path = STAR_UVT_PACKAGE,
    *,
    verify_import: bool = True,
) -> dict[str, Any]:
    """Require a loadable native binary for this Python ABI before demo work."""

    package_dir = package_dir.resolve()
    exact_candidates = [
        package_dir / f"_C{suffix}"
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
        if (package_dir / f"_C{suffix}").is_file()
    ]
    if not exact_candidates:
        available = sorted(path.name for path in package_dir.glob("_C*.so"))
        available_text = ", ".join(available) if available else "none"
        raise RuntimeError(
            "World Tubes demo requires a STAR UVT native binary compatible "
            f"with Python {sys.version_info.major}.{sys.version_info.minor}; "
            f"found {available_text} in {package_dir}. Build it with: "
            f"{_native_build_instruction()}"
        )
    native_path = exact_candidates[0]
    if verify_import:
        try:
            torch.ops.load_library(str(native_path))
        except (ImportError, OSError, RuntimeError) as error:
            raise RuntimeError(
                "World Tubes demo found the STAR UVT native binary but could "
                f"not load it for this Python/Torch environment: {native_path}. "
                f"Rebuild it with: {_native_build_instruction()}"
            ) from error
        if not hasattr(
            torch.ops.star_uvt_v0,
            "render_projective_trace_cell_interval_tiles",
        ):
            raise RuntimeError(
                "World Tubes demo loaded the STAR UVT native binary but its "
                "projective interval operator is missing. Rebuild it with: "
                f"{_native_build_instruction()}"
            )
    return {
        **_file_identity(native_path),
        "python_abi_suffix": native_path.name.removeprefix("_C"),
        "import_verified": bool(verify_import),
        "used_for_demo_computation": False,
        "role": (
            "packaging preflight; this bounded demo intentionally executes the "
            "CPU reference route and never invokes MPS"
        ),
    }


def _git_identity(path: Path) -> dict[str, Any]:
    def _run(*args: str) -> str | None:
        result = subprocess.run(
            ("git", "-C", str(path), *args),
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    status = _run("status", "--porcelain")
    return {
        "path": _relative_to_root(path),
        "commit": _run("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def _tensor_identity(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    metadata = {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
    }
    digest = hashlib.sha256()
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    )
    digest.update(b"\n")
    digest.update(value.numpy().tobytes(order="C"))
    return {**metadata, "sha256": digest.hexdigest()}


def _world_identity(atlas: Any) -> dict[str, Any]:
    payload = {
        "coeffs": _tensor_identity(atlas.coeffs),
        "opacity": _tensor_identity(atlas.opacity),
        "color": _tensor_identity(atlas.color),
        "source_window_indices": list(atlas.source_window_indices),
        "source_primitive_ids": list(atlas.source_primitive_ids),
        "active_start": list(atlas.active_start),
        "active_stop": list(atlas.active_stop),
    }
    return {**payload, "sha256": _canonical_json_sha256(payload)}


def _load_fixture_api() -> dict[str, Any]:
    from torch_gsplat_bridge_star_uvt import (
        projective_trace_cell_atlas_complexity_stats,
        projective_trace_cell_atlas_fallback_stats,
        render_projective_trace_cell_atlas_reference,
    )
    from research_experiments.star_uvt_feature_tubes.projective_decisive_demo_report import (
        build_fixture_atlas,
    )

    return {
        "build_fixture_atlas": build_fixture_atlas,
        "complexity_stats": projective_trace_cell_atlas_complexity_stats,
        "fallback_stats": projective_trace_cell_atlas_fallback_stats,
        "render": render_projective_trace_cell_atlas_reference,
    }


def _render_and_vjp(
    atlas: Any,
    times: torch.Tensor,
    adjoint: torch.Tensor,
    *,
    image_size: int,
    api: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    parameters = {
        "coeffs": atlas.coeffs.detach().clone().requires_grad_(True),
        "opacity": atlas.opacity.detach().clone().requires_grad_(True),
        "color": atlas.color.detach().clone().requires_grad_(True),
    }
    differentiable_atlas = replace(atlas, **parameters)
    image = api["render"](
        differentiable_atlas,
        times,
        image_width=image_size,
        image_height=image_size,
        tile_size=image_size,
        sigma_px=SIGMA_PX,
    )
    gradients = torch.autograd.grad(
        (image * adjoint).sum(),
        tuple(parameters.values()),
    )
    return image.detach(), {
        name: gradient.detach()
        for name, gradient in zip(parameters, gradients, strict=True)
    }


def _gradient_comparison(
    replay: Mapping[str, torch.Tensor],
    compiled: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    if replay.keys() != compiled.keys():
        raise ValueError("replay and compiled VJPs cover different parameters")
    difference_sq = 0.0
    reference_sq = 0.0
    dot = 0.0
    replay_sq = 0.0
    compiled_sq = 0.0
    per_parameter: dict[str, Any] = {}
    for name in replay:
        replay_grad = replay[name].to(dtype=torch.float64)
        compiled_grad = compiled[name].to(dtype=torch.float64)
        difference = replay_grad - compiled_grad
        replay_norm = float(torch.linalg.vector_norm(replay_grad))
        compiled_norm = float(torch.linalg.vector_norm(compiled_grad))
        difference_norm = float(torch.linalg.vector_norm(difference))
        per_parameter[name] = {
            "normalized_l2_error": difference_norm
            / max(replay_norm + compiled_norm, 1.0e-12),
            "max_abs_error": float(difference.abs().max()),
            "replay_l2_norm": replay_norm,
            "compiled_l2_norm": compiled_norm,
        }
        difference_sq += float(torch.sum(difference.square()))
        reference_sq += float(
            torch.sum(replay_grad.square() + compiled_grad.square())
        )
        dot += float(torch.sum(replay_grad * compiled_grad))
        replay_sq += float(torch.sum(replay_grad.square()))
        compiled_sq += float(torch.sum(compiled_grad.square()))
    normalized_errors = [
        float(row["normalized_l2_error"]) for row in per_parameter.values()
    ]
    return {
        "metric_definition": "frozen_world_gradient_comparison_v1",
        "global_normalized_l2_error": math.sqrt(difference_sq)
        / max(math.sqrt(reference_sq), 1.0e-12),
        "cosine_similarity": dot
        / max(math.sqrt(replay_sq) * math.sqrt(compiled_sq), 1.0e-12),
        "replay_l2_norm": math.sqrt(replay_sq),
        "compiled_l2_norm": math.sqrt(compiled_sq),
        "parameter_tensor_count": len(per_parameter),
        "max_parameter_normalized_l2_error": max(
            normalized_errors,
            default=0.0,
        ),
        "per_parameter": per_parameter,
    }


def _structure(atlas: Any, *, api: Mapping[str, Any]) -> dict[str, Any]:
    complexity = api["complexity_stats"](atlas)
    fallback = api["fallback_stats"](atlas)
    return {
        "trace_count": int(atlas.coeffs.shape[0]),
        "tile_cell_count": int(complexity.total_cells),
        "interval_trace_entries": int(complexity.interval_trace_entries),
        "dense_trace_samples": int(complexity.dense_trace_samples),
        "interval_to_dense_trace_sample_ratio": float(
            complexity.interval_to_dense_trace_sample_ratio
        ),
        "tile_active_set_groups": int(complexity.tile_active_set_groups),
        "fallback_cells": int(fallback.fallback_cells),
        "fallback_trace_samples": int(fallback.fallback_trace_samples),
    }


def _psnr(actual: torch.Tensor, expected: torch.Tensor) -> float:
    mse = float((actual - expected).square().mean())
    return 120.0 if mse <= 0.0 else min(120.0, 10.0 * math.log10(1.0 / mse))


def _comparison_pixels(
    replay: torch.Tensor,
    compiled: torch.Tensor,
) -> torch.Tensor:
    frame_index = int(replay.shape[0]) // 2
    replay_frame = replay[frame_index].clamp(0.0, 1.0)
    compiled_frame = compiled[frame_index].clamp(0.0, 1.0)
    amplified_error = (replay_frame - compiled_frame).abs().mul(64.0).clamp(0.0, 1.0)
    gutter = torch.ones(
        (int(replay_frame.shape[0]), 2, 3),
        dtype=replay_frame.dtype,
    )
    return (
        torch.cat(
            (replay_frame, gutter, compiled_frame, gutter, amplified_error),
            dim=1,
        )
        .mul(255.0)
        .round()
        .to(dtype=torch.uint8)
        .cpu()
        .contiguous()
    )


def _write_comparison_image(
    out_dir: Path,
    replay: torch.Tensor,
    compiled: torch.Tensor,
) -> tuple[Path, str]:
    pixels = _comparison_pixels(replay, compiled)
    try:
        from PIL import Image

        path = out_dir / "replay_compiled_error.png"
        Image.fromarray(pixels.numpy()).save(path)
        return path, "image/png"
    except ImportError:
        path = out_dir / "replay_compiled_error.ppm"
        height, width, _channels = pixels.shape
        with path.open("wb") as handle:
            handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
            handle.write(pixels.numpy().tobytes(order="C"))
        return path, "image/x-portable-pixmap"


def build_demo_report(
    *,
    native_binary: Mapping[str, Any],
    frames: int = DEFAULT_FRAMES,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    if frames <= 1 or frames > MAX_FRAMES:
        raise ValueError(f"frames must be in [2, {MAX_FRAMES}]")
    if image_size <= 0 or image_size > MAX_IMAGE_SIZE:
        raise ValueError(f"image_size must be in [1, {MAX_IMAGE_SIZE}]")

    api = _load_fixture_api()
    times = torch.arange(frames, dtype=torch.float32, device="cpu").contiguous()
    replay_atlas = api["build_fixture_atlas"](
        route="per_frame_replay",
        frames=frames,
    )
    compiled_atlas = api["build_fixture_atlas"](
        route="compiled_interval_atlas",
        frames=frames,
    )
    replay_world = _world_identity(replay_atlas)
    compiled_world = _world_identity(compiled_atlas)
    adjoint = torch.linspace(
        -0.35,
        0.42,
        steps=frames * image_size * image_size * 3,
        dtype=torch.float32,
        device="cpu",
    ).reshape(frames, image_size, image_size, 3)
    replay_image, replay_vjp = _render_and_vjp(
        replay_atlas,
        times,
        adjoint,
        image_size=image_size,
        api=api,
    )
    compiled_image, compiled_vjp = _render_and_vjp(
        compiled_atlas,
        times,
        adjoint,
        image_size=image_size,
        api=api,
    )
    difference = (replay_image - compiled_image).abs()
    gradient = _gradient_comparison(replay_vjp, compiled_vjp)
    replay_structure = _structure(replay_atlas, api=api)
    compiled_structure = _structure(compiled_atlas, api=api)
    structure = {
        "per_frame_replay": replay_structure,
        "compiled_interval_atlas": compiled_structure,
        "compiled_to_replay_interval_entry_ratio": float(
            compiled_structure["interval_trace_entries"]
        )
        / float(replay_structure["interval_trace_entries"]),
        "compiled_to_replay_cell_ratio": float(
            compiled_structure["tile_cell_count"]
        )
        / float(replay_structure["tile_cell_count"]),
        "dense_sample_count_matches": (
            compiled_structure["dense_trace_samples"]
            == replay_structure["dense_trace_samples"]
        ),
    }
    forward = {
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "psnr_db": _psnr(compiled_image, replay_image),
        "replay_image": _tensor_identity(replay_image),
        "compiled_image": _tensor_identity(compiled_image),
    }
    same_world = {
        "replay": replay_world,
        "compiled": compiled_world,
        "matches": replay_world["sha256"] == compiled_world["sha256"],
    }
    checks = {
        "same_world": same_world["matches"],
        "forward_matches": forward["max_abs_error"]
        <= ACCEPTANCE["image_max_abs_error"],
        "world_vjp_matches": gradient["global_normalized_l2_error"]
        <= ACCEPTANCE["gradient_global_normalized_l2_error"],
        "world_vjp_per_parameter_matches": gradient[
            "max_parameter_normalized_l2_error"
        ]
        <= ACCEPTANCE["gradient_max_parameter_normalized_l2_error"],
        "world_vjp_nonzero": min(
            gradient["replay_l2_norm"],
            gradient["compiled_l2_norm"],
        )
        > ACCEPTANCE["min_world_vjp_l2_norm"],
        "compiled_reuses_interval_entries": compiled_structure[
            "interval_trace_entries"
        ]
        < replay_structure["interval_trace_entries"],
        "same_dense_sensor_samples": structure["dense_sample_count_matches"],
        "fallback_free": (
            replay_structure["fallback_cells"] == 0
            and compiled_structure["fallback_cells"] == 0
        ),
        "cpu_only": all(
            tensor.device.type == "cpu"
            for tensor in (times, adjoint, replay_image, compiled_image)
        ),
    }
    accepted = all(checks.values())
    report = {
        "benchmark": BENCHMARK,
        "schema_version": SCHEMA_VERSION,
        "status": "accepted" if accepted else "rejected",
        "claim_scope": (
            "bounded synthetic same-world correctness and structural reuse demo; "
            "not publication timing, public-data quality, or a full-orbit claim"
        ),
        "configuration": {
            "frames": frames,
            "image_size": image_size,
            "tile_size": image_size,
            "trace_count": int(replay_atlas.coeffs.shape[0]),
            "sigma_px": SIGMA_PX,
        },
        "execution": {
            "device": "cpu",
            "training_steps": 0,
            "dataset": "deterministic_bounded_decisive_demo_fixture",
            "wandb": "not_imported_or_used",
            "mps_used": False,
            "timing_claim": False,
            "implementation": {
                "fixture_builder": (
                    "projective_decisive_demo_report.build_fixture_atlas"
                ),
                "renderer": (
                    "torch_gsplat_bridge_star_uvt."
                    "render_projective_trace_cell_atlas_reference"
                ),
                "vjp": "torch.autograd.grad_over_actual_route_execution",
                "route_semantics": {
                    "per_frame_replay": "one trace cell per frame",
                    "compiled_interval_atlas": (
                        "one shared trace cell over the full frame interval"
                    ),
                },
            },
        },
        "native_binary": dict(native_binary),
        "same_world": same_world,
        "forward": forward,
        "vjp": gradient,
        "structure": structure,
        "acceptance": dict(ACCEPTANCE),
        "checks": checks,
        "accepted": accepted,
    }
    return report, replay_image, compiled_image


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


def verify_demo_report(report: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("benchmark") != BENCHMARK:
        errors.append(f"benchmark must be {BENCHMARK}")
    if report.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    if report.get("acceptance") != ACCEPTANCE:
        errors.append("acceptance thresholds drifted")
    execution = report.get("execution")
    if not isinstance(execution, Mapping):
        errors.append("execution must be an object")
    elif (
        execution.get("device") != "cpu"
        or execution.get("training_steps") != 0
        or execution.get("mps_used") is not False
        or execution.get("wandb") != "not_imported_or_used"
        or execution.get("timing_claim") is not False
    ):
        errors.append("execution must remain CPU-only, training-free, and W&B-free")
    same_world = report.get("same_world")
    if not isinstance(same_world, Mapping):
        errors.append("same_world must be an object")
    else:
        replay_sha = (same_world.get("replay") or {}).get("sha256")
        compiled_sha = (same_world.get("compiled") or {}).get("sha256")
        if same_world.get("matches") is not True or replay_sha != compiled_sha:
            errors.append("replay and compiled routes must bind the same world")
    forward = report.get("forward")
    vjp = report.get("vjp")
    structure = report.get("structure")
    if not isinstance(forward, Mapping):
        errors.append("forward must be an object")
    if not isinstance(vjp, Mapping):
        errors.append("vjp must be an object")
    if not isinstance(structure, Mapping):
        errors.append("structure must be an object")
    if errors:
        return errors
    assert isinstance(forward, Mapping)
    assert isinstance(vjp, Mapping)
    assert isinstance(structure, Mapping)
    for container, keys, name in (
        (forward, ("max_abs_error", "mean_abs_error", "psnr_db"), "forward"),
        (
            vjp,
            (
                "global_normalized_l2_error",
                "max_parameter_normalized_l2_error",
                "replay_l2_norm",
                "compiled_l2_norm",
            ),
            "vjp",
        ),
    ):
        for key in keys:
            if not _finite_number(container.get(key)):
                errors.append(f"{name}.{key} must be finite")
    replay_structure = structure.get("per_frame_replay")
    compiled_structure = structure.get("compiled_interval_atlas")
    if not isinstance(replay_structure, Mapping) or not isinstance(
        compiled_structure,
        Mapping,
    ):
        errors.append("both structural routes are required")
        return errors
    for route_name, row in (
        ("per_frame_replay", replay_structure),
        ("compiled_interval_atlas", compiled_structure),
    ):
        for key in (
            "trace_count",
            "tile_cell_count",
            "interval_trace_entries",
            "dense_trace_samples",
        ):
            if isinstance(row.get(key), bool) or not isinstance(row.get(key), int):
                errors.append(f"{route_name}.{key} must be an integer")
            elif int(row[key]) <= 0:
                errors.append(f"{route_name}.{key} must be positive")
    expected_interval_ratio = float(compiled_structure["interval_trace_entries"]) / float(
        replay_structure["interval_trace_entries"]
    )
    expected_cell_ratio = float(compiled_structure["tile_cell_count"]) / float(
        replay_structure["tile_cell_count"]
    )
    if not math.isclose(
        float(structure.get("compiled_to_replay_interval_entry_ratio", math.inf)),
        expected_interval_ratio,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        errors.append("compiled/replay interval-entry ratio is inconsistent")
    if not math.isclose(
        float(structure.get("compiled_to_replay_cell_ratio", math.inf)),
        expected_cell_ratio,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        errors.append("compiled/replay cell ratio is inconsistent")
    expected_checks = {
        "same_world": same_world.get("matches") is True,
        "forward_matches": float(forward["max_abs_error"])
        <= ACCEPTANCE["image_max_abs_error"],
        "world_vjp_matches": float(vjp["global_normalized_l2_error"])
        <= ACCEPTANCE["gradient_global_normalized_l2_error"],
        "world_vjp_per_parameter_matches": float(
            vjp["max_parameter_normalized_l2_error"]
        )
        <= ACCEPTANCE["gradient_max_parameter_normalized_l2_error"],
        "world_vjp_nonzero": min(
            float(vjp["replay_l2_norm"]),
            float(vjp["compiled_l2_norm"]),
        )
        > ACCEPTANCE["min_world_vjp_l2_norm"],
        "compiled_reuses_interval_entries": int(
            compiled_structure["interval_trace_entries"]
        )
        < int(replay_structure["interval_trace_entries"]),
        "same_dense_sensor_samples": int(compiled_structure["dense_trace_samples"])
        == int(replay_structure["dense_trace_samples"]),
        "fallback_free": int(replay_structure.get("fallback_cells", -1)) == 0
        and int(compiled_structure.get("fallback_cells", -1)) == 0,
        "cpu_only": True,
    }
    if report.get("checks") != expected_checks:
        errors.append("checks do not match the reported measurements")
    expected_accepted = all(expected_checks.values())
    if report.get("accepted") is not expected_accepted:
        errors.append("accepted does not match the checks")
    expected_status = "accepted" if expected_accepted else "rejected"
    if report.get("status") != expected_status:
        errors.append(f"status must be {expected_status}")
    return errors


def _resolve_output_artifact(out_dir: Path, identity: Mapping[str, Any]) -> Path:
    raw_path = identity.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("artifact identity path must be a non-empty string")
    relative_path = Path(raw_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"artifact path must stay inside the demo directory: {raw_path}")
    return out_dir / relative_path


def _verify_file_identity(
    out_dir: Path,
    identity: Any,
    *,
    label: str,
) -> list[str]:
    if not isinstance(identity, Mapping):
        return [f"{label} identity must be an object"]
    try:
        path = _resolve_output_artifact(out_dir, identity)
    except ValueError as error:
        return [f"{label}: {error}"]
    if not path.is_file():
        return [f"{label} is missing: {path}"]
    actual = _file_identity(path, relative_to=out_dir)
    expected = {key: identity.get(key) for key in ("path", "bytes", "sha256")}
    return [] if expected == actual else [f"{label} byte identity drifted"]


def verify_demo_directory(out_dir: Path) -> list[str]:
    out_dir = out_dir.resolve()
    manifest_path = out_dir / "demo_manifest.json"
    if not manifest_path.is_file():
        return [f"demo manifest is missing: {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        return [f"demo manifest is unreadable: {error}"]
    errors: list[str] = []
    if manifest.get("manifest_schema_version") != SCHEMA_VERSION:
        errors.append(f"manifest_schema_version must be {SCHEMA_VERSION}")
    if manifest.get("demo") != BENCHMARK:
        errors.append(f"manifest demo must be {BENCHMARK}")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return errors + ["manifest artifacts must be an object"]
    for key in ("summary", "comparison_image"):
        errors.extend(
            _verify_file_identity(
                out_dir,
                artifacts.get(key),
                label=key,
            )
        )
    if errors:
        return errors
    summary_path = _resolve_output_artifact(out_dir, artifacts["summary"])
    report = json.loads(summary_path.read_text(encoding="utf-8"))
    errors.extend(verify_demo_report(report))
    if report.get("comparison_image") != artifacts.get("comparison_image"):
        errors.append("summary and manifest comparison-image identities differ")
    source_files = manifest.get("source_files")
    if not isinstance(source_files, list):
        errors.append("manifest source_files must be a list")
    else:
        expected_sources = [_file_identity(path) for path in SOURCE_PATHS]
        if source_files != expected_sources:
            errors.append("demo source identities drifted")
    try:
        current_native = require_star_uvt_native_binary()
    except RuntimeError as error:
        errors.append(str(error))
    else:
        if manifest.get("native_binary") != current_native:
            errors.append("STAR UVT native binary identity drifted")
    if manifest.get("status") != report.get("status"):
        errors.append("manifest status does not match summary status")
    return errors


def run_demo(
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    frames: int = DEFAULT_FRAMES,
    image_size: int = DEFAULT_IMAGE_SIZE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    native_binary = require_star_uvt_native_binary()
    report, replay_image, compiled_image = build_demo_report(
        native_binary=native_binary,
        frames=frames,
        image_size=image_size,
    )
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path, image_media_type = _write_comparison_image(
        out_dir,
        replay_image,
        compiled_image,
    )
    image_identity = {
        **_file_identity(image_path, relative_to=out_dir),
        "media_type": image_media_type,
        "panels": [
            "per_frame_replay",
            "compiled_interval_atlas",
            "absolute_error_x64",
        ],
    }
    report["comparison_image"] = image_identity
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_identity = _file_identity(summary_path, relative_to=out_dir)
    manifest = {
        "manifest_schema_version": SCHEMA_VERSION,
        "demo": BENCHMARK,
        "status": report["status"],
        "scope": report["claim_scope"],
        "configuration": report["configuration"],
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "wandb": "not_used",
            "mps": "not_used",
        },
        "repositories": {
            "superproject": _git_identity(ROOT),
            "star_uvt": _git_identity(STAR_UVT_ROOT),
        },
        "source_files": [_file_identity(path) for path in SOURCE_PATHS],
        "native_binary": native_binary,
        "artifacts": {
            "summary": summary_identity,
            "comparison_image": image_identity,
        },
        "reproduce": {
            "command": [
                ".venv/bin/python",
                _relative_to_root(Path(__file__)),
                "--out-dir",
                _relative_to_root(out_dir),
                "--frames",
                str(frames),
                "--image-size",
                str(image_size),
            ],
            "verify_command": [
                ".venv/bin/python",
                _relative_to_root(Path(__file__)),
                "--verify-dir",
                _relative_to_root(out_dir),
            ],
        },
    }
    (out_dir / "demo_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    errors = verify_demo_directory(out_dir)
    if errors:
        raise RuntimeError("World Tubes demo verification failed:\n- " + "\n- ".join(errors))
    if report["accepted"] is not True:
        raise RuntimeError(
            "World Tubes demo completed but replay/compiled acceptance failed; "
            f"inspect {summary_path}"
        )
    return report, manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the bounded CPU-only World Tubes replay-versus-compiled demo."
        )
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAMES)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--verify-dir", type=Path)
    args = parser.parse_args()
    if args.verify_dir is not None:
        errors = verify_demo_directory(args.verify_dir)
        if errors:
            raise ValueError(
                "World Tubes demo verification failed:\n- " + "\n- ".join(errors)
            )
        print(f"verified {args.verify_dir}")
        return
    report, manifest = run_demo(
        args.out_dir,
        frames=args.frames,
        image_size=args.image_size,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "out_dir": str(args.out_dir),
                "forward_max_abs_error": report["forward"]["max_abs_error"],
                "vjp_global_normalized_l2_error": report["vjp"][
                    "global_normalized_l2_error"
                ],
                "compiled_to_replay_interval_entry_ratio": report["structure"][
                    "compiled_to_replay_interval_entry_ratio"
                ],
                "artifacts": manifest["artifacts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
