from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_NATIVE_GATE = (
    ROOT / "artifacts" / "foundation_gates" / "spd4_retained_fiber_cpu_metal.json"
)
DEFAULT_SMOKE_ROOT = ROOT / "artifacts" / "spd4_retained_hybrid_smoke"
DEFAULT_OUTPUT = (
    ROOT
    / "artifacts"
    / "foundation_gates"
    / "world_tubes_ordered_transfer_ablation_verified.json"
)
REPORT_PATHS = {
    "all_retained_16": Path("retained_16x4f_2step/comparison_report.json"),
    "hybrid_16": Path("hybrid_16x4f_2step/comparison_report.json"),
    "hybrid_199": Path("hybrid_199x4f_2step/comparison_report.json"),
    "physical_fast_199": Path(
        "metal_199x4f_2step_optimized/comparison_report.json"
    ),
}
QUALITY_KEYS = (
    "heldout_eval_psnr",
    "heldout_eval_ssim",
    "heldout_eval_lpips",
    "heldout_eval_l1",
)
THRESHOLDS = {
    "native_forward_max_abs_error": 5.0e-6,
    "native_vjp_normalized_error": 5.0e-6,
    "fixed_quadrature_reference_error": 5.0e-4,
    "hybrid_oracle_metric_max_abs_delta": 1.0e-6,
    "selective_fallback_fraction_max": 0.20,
    "native_driver_memory_bytes": 100_000_000,
}


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"ordered-transfer artifact is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"ordered-transfer artifact must be an object: {path}")
    return value


def file_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    try:
        display_path = str(path.resolve().relative_to(ROOT))
    except ValueError:
        display_path = str(path.resolve())
    return {
        "path": display_path,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def exact_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return int(value)


def star_payload(report: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    star = report.get("star_uvt")
    if not isinstance(star, Mapping):
        raise ValueError(f"{label} is missing star_uvt")
    metrics = star.get("metrics")
    visibility = star.get("physical_visibility")
    if not isinstance(metrics, Mapping) or not isinstance(visibility, Mapping):
        raise ValueError(f"{label} is missing metrics/physical_visibility")
    return star


def validate_report_identity(
    report: Mapping[str, Any],
    *,
    label: str,
    backend: str,
    atom_count: int,
) -> Mapping[str, Any]:
    meta = report.get("meta")
    if not isinstance(meta, Mapping):
        raise ValueError(f"{label} is missing meta")
    expected = {
        "seed": 17,
        "device": "mps",
        "frame_count": 4,
        "uvt_world_representation": "full_spd4",
        "uvt_alpha_mode": "beer_lambert",
        "uvt_render_backend": backend,
        "uvt_amplitude_convention": "fiber_integrated",
    }
    drifted = [
        key for key, value in expected.items() if meta.get(key) != value
    ]
    if drifted:
        raise ValueError(f"{label} identity drifted: {', '.join(drifted)}")
    star = star_payload(report, label)
    if exact_int(star.get("tube_count"), f"{label} tube_count") != atom_count:
        raise ValueError(f"{label} atom count drifted")
    if exact_int(star.get("steps"), f"{label} steps") != 2:
        raise ValueError(f"{label} is not the declared two-step smoke")
    expected_parameters = 18 * atom_count
    cost = star.get("paper_protocol", {}).get("cost", {})
    if (
        not isinstance(cost, Mapping)
        or exact_int(
            cost.get("trainable_parameter_count"),
            f"{label} trainable_parameter_count",
        )
        != expected_parameters
    ):
        raise ValueError(f"{label} parameter count drifted")
    return star


def visibility_counts(
    star: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    visibility = star["physical_visibility"]
    counts = {
        key: exact_int(visibility.get(key), f"{label} {key}")
        for key in (
            "tile_count",
            "fallback_tile_count",
            "ambiguous_tile_count",
            "invalid_tile_count",
            "certificate_overflow_tile_count",
        )
    }
    counts["fallback_fraction"] = finite_float(
        visibility.get("fallback_fraction"),
        f"{label} fallback_fraction",
    )
    if counts["fallback_tile_count"] > counts["tile_count"]:
        raise ValueError(f"{label} fallback count exceeds tile count")
    expected_fraction = counts["fallback_tile_count"] / max(
        1,
        counts["tile_count"],
    )
    if not math.isclose(
        counts["fallback_fraction"],
        expected_fraction,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(f"{label} fallback fraction does not match counts")
    return counts


def quality_metrics(star: Mapping[str, Any], label: str) -> dict[str, float]:
    metrics = star["metrics"]
    return {
        key: finite_float(metrics.get(key), f"{label} {key}")
        for key in QUALITY_KEYS
    }


def verify(
    *,
    native_gate_path: Path = DEFAULT_NATIVE_GATE,
    smoke_root: Path = DEFAULT_SMOKE_ROOT,
) -> dict[str, Any]:
    native = load_json(native_gate_path)
    reports = {
        name: load_json(smoke_root / relative_path)
        for name, relative_path in REPORT_PATHS.items()
    }
    native_checks = native.get("checks")
    native_metrics = native.get("metrics")
    if (
        native.get("schema_version") != 1
        or native.get("status") != "pass"
        or not isinstance(native_checks, Mapping)
        or not native_checks
        or not all(value is True for value in native_checks.values())
        or not isinstance(native_metrics, Mapping)
    ):
        raise ValueError("retained-fiber native gate is not accepted")
    forward_error = finite_float(
        native_metrics.get("forward_max_abs_error"),
        "native forward_max_abs_error",
    )
    quadrature_error = finite_float(
        native_metrics.get("cpu_32_vs_1024_sample_max_abs_error"),
        "native fixed-quadrature reference error",
    )
    driver_memory = exact_int(
        native_metrics.get("mps_driver_allocated_bytes"),
        "native driver memory",
    )
    vjp_errors = native_metrics.get("vjp_normalized_errors")
    if not isinstance(vjp_errors, Mapping) or not vjp_errors:
        raise ValueError("native VJP errors are missing")
    worst_vjp_error = max(
        finite_float(value, f"native {name} VJP error")
        for name, value in vjp_errors.items()
    )

    retained_16 = validate_report_identity(
        reports["all_retained_16"],
        label="all-retained 16-atom row",
        backend="retained_fiber_metal",
        atom_count=16,
    )
    hybrid_16 = validate_report_identity(
        reports["hybrid_16"],
        label="hybrid 16-atom row",
        backend="hybrid_retained_fiber",
        atom_count=16,
    )
    hybrid_199 = validate_report_identity(
        reports["hybrid_199"],
        label="hybrid 199-atom row",
        backend="hybrid_retained_fiber",
        atom_count=199,
    )
    physical_fast_199 = validate_report_identity(
        reports["physical_fast_199"],
        label="physical-fast 199-atom row",
        backend="metal_tile",
        atom_count=199,
    )
    retained_quality = quality_metrics(retained_16, "all-retained 16-atom row")
    hybrid_quality = quality_metrics(hybrid_16, "hybrid 16-atom row")
    quality_deltas = {
        key: abs(retained_quality[key] - hybrid_quality[key])
        for key in QUALITY_KEYS
    }
    max_quality_delta = max(quality_deltas.values())
    selective = visibility_counts(
        hybrid_16,
        label="hybrid 16-atom row",
    )
    dense = visibility_counts(
        hybrid_199,
        label="hybrid 199-atom row",
    )
    physical_fast_quality = quality_metrics(
        physical_fast_199,
        "physical-fast 199-atom row",
    )

    checks = {
        "native_forward_matches_cpu": (
            forward_error <= THRESHOLDS["native_forward_max_abs_error"]
        ),
        "native_all_source_vjps_match_cpu": (
            worst_vjp_error <= THRESHOLDS["native_vjp_normalized_error"]
        ),
        "fixed_quadrature_matches_dense_reference": (
            quadrature_error
            <= THRESHOLDS["fixed_quadrature_reference_error"]
        ),
        "native_driver_memory_below_bound": (
            driver_memory <= THRESHOLDS["native_driver_memory_bytes"]
        ),
        "hybrid_matches_all_retained_metrics": (
            max_quality_delta
            <= THRESHOLDS["hybrid_oracle_metric_max_abs_delta"]
        ),
        "small_fixture_hybrid_is_selective": (
            selective["tile_count"] == 64
            and selective["fallback_tile_count"] == 10
            and selective["ambiguous_tile_count"] == 10
            and selective["fallback_fraction"]
            <= THRESHOLDS["selective_fallback_fraction_max"]
            and selective["invalid_tile_count"] == 0
            and selective["certificate_overflow_tile_count"] == 0
        ),
        "dense_fixture_is_negative_selectivity_control": (
            dense["tile_count"] == 64
            and dense["fallback_tile_count"] == 64
            and dense["ambiguous_tile_count"] == 64
            and dense["fallback_fraction"] == 1.0
            and dense["invalid_tile_count"] == 0
            and dense["certificate_overflow_tile_count"] == 0
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            "ordered-transfer bounded evidence failed: " + ", ".join(failed)
        )

    source_paths = {
        "native_gate": native_gate_path,
        **{
            name: smoke_root / relative_path
            for name, relative_path in REPORT_PATHS.items()
        },
    }
    return {
        "schema_version": 1,
        "status": "pass",
        "scope": "bounded_static_world_tubes_ordered_ray_transfer",
        "paper_label": "World Tubes + Ordered Ray Transfer",
        "checks": checks,
        "thresholds": THRESHOLDS,
        "results": {
            "native": {
                "forward_max_abs_error": forward_error,
                "worst_normalized_vjp_error": worst_vjp_error,
                "fixed_quadrature_reference_max_abs_error": quadrature_error,
                "driver_memory_bytes": driver_memory,
            },
            "selective_16_atom_smoke": {
                **selective,
                "quality_metrics": hybrid_quality,
                "all_retained_quality_metrics": retained_quality,
                "quality_metric_absolute_deltas": quality_deltas,
                "max_quality_metric_absolute_delta": max_quality_delta,
            },
            "dense_199_atom_negative_control": {
                **dense,
                "quality_metrics": quality_metrics(
                    hybrid_199,
                    "hybrid 199-atom row",
                ),
            },
            "physical_fast_199_atom_reference": {
                "quality_metrics": physical_fast_quality,
            },
        },
        "source_artifacts": {
            name: file_identity(path) for name, path in source_paths.items()
        },
        "claim_limits": {
            "public_quality_or_speed_ablation_complete": False,
            "projective_retained_fiber_supported": False,
            "adaptive_error_certified_quadrature": False,
            "dense_scene_hybrid_selective": False,
            "accepted_claim": (
                "bounded native forward/VJP correctness, small-fixture "
                "selectivity with oracle metric parity, and a retained "
                "dense-scene negative selectivity control"
            ),
        },
    }


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-gate", type=Path, default=DEFAULT_NATIVE_GATE)
    parser.add_argument("--smoke-root", type=Path, default=DEFAULT_SMOKE_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = verify(
        native_gate_path=args.native_gate.resolve(),
        smoke_root=args.smoke_root.resolve(),
    )
    output = args.output.resolve()
    write_json(output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "scope": result["scope"],
                "output": str(output),
                "checks": result["checks"],
                "claim_limits": result["claim_limits"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
