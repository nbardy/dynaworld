from __future__ import annotations

"""Build a deterministic, fail-closed WorldFoam Paper-B foundation bundle.

This packager launches no renderer and does not convert old PowerFoam runs into
evidence for the memory-light WorldFoam implementation.  Every consumed JSON is
reopened and checked by an independent verifier.  Rejected inputs remain in the
ledger with their failures, but contribute no numeric table or plot rows.

The bundle remains explicitly incomplete while G4 or G6 is absent.  When both
measured artifacts exist, their independent verifiers must accept the exact
36-row public-quality and 21-row native-memory matrices before this generator
replaces either placeholder with numeric tables and figures.  Consequently the
default command exits non-zero until all required evidence is present;
``--allow-incomplete`` writes an honest partial bundle without promoting a
missing claim.
"""

import argparse
import csv
import hashlib
import html
import io
import json
import math
import sys
import textwrap
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
LANE2 = Path(__file__).resolve().parent
for import_root in (ROOT, LANE2):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from research_experiments.world_foam_lane2.cell_path_optical_transfer_fixture import (  # noqa: E402
    verify_summary as verify_constant_transfer_summary,
)
from research_experiments.world_foam_lane2.compiled_lie_frame_density_gate import (  # noqa: E402
    verify_compiled_lie_frame_density_report,
)
from research_experiments.world_foam_lane2.verify_finite_element_material_fit import (  # noqa: E402
    verify_artifact as verify_material_fit_artifact,
)
from research_experiments.world_foam_lane2.verify_adaptive_material_basis_selection import (  # noqa: E402
    verify_report as verify_adaptive_material_report,
)
from research_experiments.world_foam_lane2.verify_worldfoam_synthetic_visibility_suite import (  # noqa: E402
    EXPECTED_FIGURES as EXPECTED_VISIBILITY_FIGURES,
    verify_report as verify_visibility_report,
)
from research_experiments.world_foam_lane2 import (  # noqa: E402
    generate_worldfoam_public_quality_assets as g4_assets,
    generate_worldfoam_training_memory_assets as g6_assets,
)
from research_experiments.world_foam_lane2.verify_worldfoam_public_quality_ablation_v2 import (  # noqa: E402,E501
    verify_artifact_file as verify_g4_artifact_file,
)
from research_experiments.world_foam_lane2.verify_worldfoam_training_memory_ablation import (  # noqa: E402,E501
    verify_artifact_file as verify_g6_artifact_file,
)


GENERATOR_NAME = "worldfoam_paper_b_foundation_artifacts"
GENERATOR_SCHEMA_VERSION = 3
BUNDLE_SCHEMA_VERSION = 3

DEFAULT_MATERIAL_PARITY = (
    ROOT
    / "artifacts"
    / "foundation_gates"
    / "worldfoam_material_m0_m5_cpu_metal_20260727.json"
)
DEFAULT_MATERIAL_FIT = (
    ROOT
    / "artifacts"
    / "foundation_gates"
    / "worldfoam_material_value_fit_cpu_20260727.json"
)
DEFAULT_ADAPTIVE_MATERIAL = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-08-15_worldfoam_adaptive_material_basis_cpu"
    / "summary.json"
)
EXPECTED_ADAPTIVE_MATERIAL_ASSETS = (
    "adaptive_material_basis_table.md",
    "adaptive_material_basis_table.tex",
    "worldfoam_adaptive_material_basis.svg",
)
DEFAULT_COMPILED_LIE = (
    ROOT
    / "artifacts"
    / "foundation_gates"
    / "worldfoam_compiled_lie_frame_density_cpu_20260803.json"
)
DEFAULT_CONSTANT_TRANSFER = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-07-08_worldfoam_cell_path_optical_transfer_summary.json"
)
DEFAULT_SYNTHETIC_VISIBILITY = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-08-15_worldfoam_synthetic_visibility_cpu"
    / "summary.json"
)
DEFAULT_G4_PUBLIC_QUALITY = g4_assets.DEFAULT_ARTIFACT
DEFAULT_G6_NATIVE_MEMORY = g6_assets.DEFAULT_OUTPUT
DEFAULT_OUT_DIR = (
    ROOT
    / "research_notes"
    / "worldfoam_paper"
    / "generated"
    / "foundation_v1"
)
MATERIAL_METAL_SOURCE = Path(__file__).with_name(
    "finite_element_material_transfer.metal"
)

EXPECTED_MATERIAL_MODES = (
    "M0_P0_CONSTANT",
    "M1_P0_AFFINE_RGB",
    "M2_POSITIVE_BERNSTEIN_P1",
    "M3_POSITIVE_BERNSTEIN_P2",
    "M4_LOG_P1",
    "M5_CONVEX_LOG_P2",
)
EXPECTED_MATERIAL_RECORD_COUNTS = {
    "M0_P0_CONSTANT": 2,
    "M1_P0_AFFINE_RGB": 1,
    "M2_POSITIVE_BERNSTEIN_P1": 1,
    "M3_POSITIVE_BERNSTEIN_P2": 1,
    "M4_LOG_P1": 3,
    "M5_CONVEX_LOG_P2": 4,
}
REQUIRED_MATERIAL_CLAIM_LIMITS = frozenset(
    {
        "local fixed-segment material correctness only",
        "not trained image quality",
        "not renderer throughput",
        "not native-4D parameter or event scaling",
    }
)

PLACEHOLDER_GATES = (
    {
        "gate": "G4",
        "slug": "public_quality",
        "title": "Public heldout quality",
        "required_evidence": (
            "new memory-light WorldFoam on calibrated public dynamic scenes; "
            "matched replay/World Tubes/dynamic-3DGS rows; PSNR, SSIM, LPIPS, "
            "L1, cost, and seeds"
        ),
        "forbidden_claim": "WorldFoam is competitive on public dynamic NVS",
    },
    {
        "gate": "G6",
        "slug": "native_memory",
        "title": "Native training memory and temporal sharing",
        "required_evidence": (
            "verified 21-row F=8/64/300 fresh-process artifact with native "
            "execution, staged/fused/framewise parity, restart parity, MPS/RSS "
            "peaks, and exact work receipts"
        ),
        "forbidden_claim": "WorldFoam fits the declared memory budget",
    },
)


@dataclass(frozen=True)
class EvidenceSpec:
    evidence_id: str
    gate: str
    path: Path
    verifier: str
    scope: str
    validator: Callable[[Path, Mapping[str, Any]], list[str]]
    extractor: Callable[[Mapping[str, Any]], list[dict[str, Any]]]
    asset_paths: (
        Callable[[Path, Mapping[str, Any]], Mapping[str, Path]] | None
    ) = None


@dataclass(frozen=True)
class Bundle:
    complete: bool
    ledger: dict[str, Any]
    gate_status: dict[str, Any]
    foundation_rows: list[dict[str, Any]]
    files: dict[str, bytes]


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("root must be a JSON object")
    return payload


def _finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _require_finite(
    failures: list[str],
    mapping: Any,
    key: str,
    *,
    nonnegative: bool = False,
) -> float | None:
    if not isinstance(mapping, Mapping) or not _finite(mapping.get(key)):
        failures.append(f"{key} must be finite numeric")
        return None
    value = float(mapping[key])
    if nonnegative and value < 0.0:
        failures.append(f"{key} must be nonnegative")
    return value


def _validate_material_parity(
    path: Path,
    payload: Mapping[str, Any],
) -> list[str]:
    """Strictly validate the retained local CPU/Metal segment artifact.

    This is deliberately labelled historical/source-hash-checked.  The saved
    artifact binds the unchanged Metal kernel, not the entire current trainer.
    """

    failures: list[str] = []
    if payload.get("schema_version") != "worldfoam.material_foundation_gate.v1":
        failures.append("unsupported material foundation schema")
    if payload.get("passed") is not True:
        failures.append("artifact passed flag is not true")
    if payload.get("source_sha256") != _file_sha256(MATERIAL_METAL_SOURCE):
        failures.append("saved Metal source hash does not match current source")
    if set(payload.get("material_modes", ())) != set(EXPECTED_MATERIAL_MODES):
        failures.append("material mode set is incomplete")
    if set(payload.get("claim_scope", ())) != REQUIRED_MATERIAL_CLAIM_LIMITS:
        failures.append("material claim limits changed or are incomplete")

    cpu = payload.get("cpu")
    if not isinstance(cpu, Mapping):
        failures.append("cpu section is missing")
    else:
        if cpu.get("device") != "cpu" or cpu.get("dtype") != "float64":
            failures.append("cpu device/dtype contract changed")
        if int(cpu.get("segment_count", -1)) != 12:
            failures.append("cpu segment_count must be 12")
        gate = cpu.get("gate")
        if not isinstance(gate, Mapping) or gate.get("passed") is not True:
            failures.append("cpu gate is not passed")
        else:
            checks = (
                ("max_integral_abs_error", "integral_tolerance"),
                ("max_vjp_normalized_error", "vjp_tolerance"),
                (
                    "max_finite_difference_vjp_normalized_error",
                    "finite_difference_vjp_tolerance",
                ),
            )
            for metric_key, threshold_key in checks:
                metric = _require_finite(
                    failures, cpu, metric_key, nonnegative=True
                )
                threshold = _require_finite(
                    failures, gate, threshold_key, nonnegative=True
                )
                if (
                    metric is not None
                    and threshold is not None
                    and metric > threshold
                ):
                    failures.append(f"{metric_key} exceeds {threshold_key}")
        bounds = _require_finite(
            failures,
            cpu,
            "max_density_bound_violation",
            nonnegative=True,
        )
        if bounds is not None and bounds != 0.0:
            failures.append("density bound violation must be zero")
        records = cpu.get("records")
        if not isinstance(records, list) or len(records) != 12:
            failures.append("cpu records must contain exactly 12 rows")
        else:
            counts = {mode: 0 for mode in EXPECTED_MATERIAL_MODES}
            for index, record in enumerate(records):
                if not isinstance(record, Mapping):
                    failures.append(f"cpu record {index} is not an object")
                    continue
                mode = record.get("mode")
                if mode not in counts:
                    failures.append(f"cpu record {index} has unknown mode")
                else:
                    counts[str(mode)] += 1
                for key in (
                    "tau",
                    "quadrature_max_abs_error",
                    "vjp_normalized_error",
                    "finite_difference_vjp_normalized_error",
                    "density_bound_violation",
                ):
                    _require_finite(failures, record, key, nonnegative=True)
            if counts != EXPECTED_MATERIAL_RECORD_COUNTS:
                failures.append("cpu record mode/branch matrix changed")
        branches = cpu.get("branch_counts")
        if not isinstance(branches, Mapping):
            failures.append("cpu branch counts are missing")
        else:
            if int(branches.get("total", -1)) != 12:
                failures.append("cpu branch total must be 12")
            if int(branches.get("invalid_input", -1)) != 0:
                failures.append("cpu invalid_input count must be zero")
            for key in (
                "small_tau_series",
                "log_linear_series",
                "log_quadratic_series",
                "log_quadratic_erf",
                "log_quadratic_tail",
            ):
                if int(branches.get(key, 0)) <= 0:
                    failures.append(f"cpu branch {key} was not exercised")

    metal = payload.get("metal")
    if not isinstance(metal, Mapping):
        failures.append("saved Metal parity section is missing")
    else:
        if metal.get("device") != "mps" or metal.get("dtype") != "float32":
            failures.append("Metal device/dtype contract changed")
        if int(metal.get("segment_count", -1)) != 12:
            failures.append("Metal segment_count must be 12")
        if metal.get("status_matches_vjp") is not True:
            failures.append("Metal forward/VJP branch status differs")
        gate = metal.get("gate")
        if not isinstance(gate, Mapping) or gate.get("passed") is not True:
            failures.append("Metal gate is not passed")
        else:
            for metric_key, threshold_key in (
                ("max_forward_normalized_error", "forward_tolerance"),
                ("max_vjp_normalized_error", "vjp_tolerance"),
            ):
                metric = _require_finite(
                    failures, metal, metric_key, nonnegative=True
                )
                threshold = _require_finite(
                    failures, gate, threshold_key, nonnegative=True
                )
                if (
                    metric is not None
                    and threshold is not None
                    and metric > threshold
                ):
                    failures.append(f"Metal {metric_key} exceeds tolerance")
        for key in ("current_allocated_bytes", "driver_allocated_bytes"):
            _require_finite(failures, metal, key, nonnegative=True)
        if metal.get("branch_counts") != (
            cpu.get("branch_counts") if isinstance(cpu, Mapping) else None
        ):
            failures.append("CPU and Metal branch counts differ")

    return sorted(set(failures))


def _validate_material_fit(
    path: Path,
    payload: Mapping[str, Any],
) -> list[str]:
    try:
        verify_material_fit_artifact(path)
    except Exception as error:
        return [f"material-fit verifier rejected artifact: {error}"]
    return []


def _validate_adaptive_material(
    path: Path,
    payload: Mapping[str, Any],
) -> list[str]:
    return sorted(
        set(
            verify_adaptive_material_report(
                dict(payload),
                summary_path=path,
                require_current_source=True,
            )
        )
    )


def _validate_compiled_lie(
    _path: Path,
    payload: Mapping[str, Any],
) -> list[str]:
    return sorted(set(verify_compiled_lie_frame_density_report(dict(payload))))


def _validate_constant_transfer(
    _path: Path,
    payload: Mapping[str, Any],
) -> list[str]:
    return sorted(set(verify_constant_transfer_summary(dict(payload))))


def _validate_synthetic_visibility(
    path: Path,
    payload: Mapping[str, Any],
) -> list[str]:
    return sorted(
        set(
            verify_visibility_report(
                dict(payload),
                report_path=path,
                require_accepted=True,
            )
        )
    )


def _extract_material_parity(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    cpu = payload["cpu"]
    metal = payload["metal"]
    return [
        {
            "row_id": "m0_m5_cpu_segment_parity",
            "evidence_id": "m0_m5_segment_parity",
            "category": "local_material_correctness",
            "metric_1": "integral_max_abs_error",
            "value_1": float(cpu["max_integral_abs_error"]),
            "metric_2": "finite_difference_vjp_normalized_error",
            "value_2": float(
                cpu["max_finite_difference_vjp_normalized_error"]
            ),
            "verdict": "accepted_local_cpu",
            "claim_scope": "fixed segments only",
        },
        {
            "row_id": "m0_m5_metal_segment_parity",
            "evidence_id": "m0_m5_segment_parity",
            "category": "local_material_correctness",
            "metric_1": "forward_normalized_error",
            "value_1": float(metal["max_forward_normalized_error"]),
            "metric_2": "vjp_normalized_error",
            "value_2": float(metal["max_vjp_normalized_error"]),
            "verdict": "accepted_historical_source_hash_checked_metal",
            "claim_scope": "fixed segments only; not current trainer runtime",
        },
    ]


def _extract_material_fit(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    medians = payload["medians"]
    direct = medians["positive_p2_hump"]
    logarithmic = medians["convex_log_p2_hump"]
    m3 = "M3_POSITIVE_BERNSTEIN_P2"
    m5 = "M5_CONVEX_LOG_P2"
    return [
        {
            "row_id": "partial_chord_positive_p2",
            "evidence_id": "m3_m5_partial_chord_fit",
            "category": "material_capacity",
            "metric_1": "M3_heldout_loss",
            "value_1": float(direct[m3]["heldout_loss"]),
            "metric_2": "M5_heldout_loss",
            "value_2": float(direct[m5]["heldout_loss"]),
            "verdict": "M3_family_win",
            "claim_scope": "synthetic partial chords; no universal winner",
        },
        {
            "row_id": "partial_chord_convex_log_p2",
            "evidence_id": "m3_m5_partial_chord_fit",
            "category": "material_capacity",
            "metric_1": "M3_heldout_loss",
            "value_1": float(logarithmic[m3]["heldout_loss"]),
            "metric_2": "M5_heldout_loss",
            "value_2": float(logarithmic[m5]["heldout_loss"]),
            "verdict": "M5_family_win",
            "claim_scope": "synthetic partial chords; no universal winner",
        },
    ]


def _extract_adaptive_material(
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    aggregates = payload["aggregates"]
    shared_scope = (
        "verified float64 CPU synthetic per-cell basis selection on disjoint "
        "chords; matched 24-byte M3/M5 payload plus one tag bit; no native, "
        "public-image, runtime, or memory claim"
    )
    return [
        {
            "row_id": "adaptive_m3_m5_mean_loss",
            "evidence_id": "adaptive_m3_m5_basis_selection",
            "category": "material_basis_selection",
            "metric_1": "adaptive_to_best_fixed_ratio",
            "value_1": float(aggregates["adaptive_to_best_fixed_ratio"]),
            "metric_2": "adaptive_to_oracle_ratio",
            "value_2": float(aggregates["adaptive_to_oracle_ratio"]),
            "verdict": "accepted_cpu_adaptive_selection",
            "claim_scope": shared_scope,
        },
        {
            "row_id": "adaptive_m3_m5_selection_accuracy",
            "evidence_id": "adaptive_m3_m5_basis_selection",
            "category": "material_basis_selection",
            "metric_1": "pure_family_selection_accuracy",
            "value_1": float(aggregates["pure_family_selection_accuracy"]),
            "metric_2": "selection_oracle_agreement",
            "value_2": float(aggregates["selection_oracle_agreement"]),
            "verdict": "accepted_cpu_adaptive_selection",
            "claim_scope": shared_scope,
        },
    ]


def _extract_compiled_lie(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = payload["rows"]
    first = rows[0]
    last = rows[-1]
    return [
        {
            "row_id": "compiled_lie_requested_density",
            "evidence_id": "compiled_lie_frame_density",
            "category": "cpu_fixed_surrogate_scaling",
            "metric_1": "reverse_state_scale",
            "value_1": float(
                last[
                    "logical_selected_reverse_state_bytes_excluding_targets_and_predictions"
                ]
            )
            / float(
                first[
                    "logical_selected_reverse_state_bytes_excluding_targets_and_predictions"
                ]
            ),
            "metric_2": "world_reverse_interaction_scale",
            "value_2": float(last["step_world_reverse_run_interactions"])
            / float(first["step_world_reverse_run_interactions"]),
            "verdict": "accepted_cpu_fixed_surrogate",
            "claim_scope": "CPU fixed topology; no native allocator claim",
        }
    ]


def _extract_constant_transfer(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "row_id": "constant_density_ordered_transfer",
            "evidence_id": "constant_density_ordered_transfer",
            "category": "ordered_transfer_algebra",
            "metric_1": "render_max_abs_error",
            "value_1": float(payload["max_errors"]["render"]),
            "metric_2": "vjp_max_abs_error",
            "value_2": float(payload["max_errors"]["grad"]),
            "verdict": "accepted_cpu_constant_density",
            "claim_scope": "constant-density owner word only",
        }
    ]


def _linear_quantile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a quantile of zero values")
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _mean(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        raise ValueError(f"cannot average zero rows for {key}")
    return sum(float(row[key]) for row in rows) / len(rows)


def _extract_synthetic_visibility(
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    layer_rows = payload["layer_rows"]
    baseline_rows = payload["baseline_rows"]
    adaptive = payload["adaptive_rows"]
    deepest = [row for row in layer_rows if row["layer_count"] == 128]
    crossing_scenes = {
        "S2_crossing_translucent_slabs",
        "S3_crossing_gaussian_density_sheets",
    }

    def crossing(method: str) -> list[Mapping[str, Any]]:
        return [
            row
            for row in baseline_rows
            if row["scene"] in crossing_scenes and row["method"] == method
        ]

    worldfoam = crossing("depth_layer_128")
    representative = crossing("representative_depth_sorted")
    marginal = crossing("depth_marginal")
    analytic = payload["analytic_constant_sphere"]
    gauge = payload["gauge_jacobian"]
    fallback = payload["aggregates"]["adaptive_fallback_fraction"]
    shared_scope = (
        "verified float64 CPU synthetic S1-S8/C1-C7 only; no native runtime, "
        "allocator, trained-image, or public-data claim"
    )
    return [
        {
            "row_id": "g0_analytic_constant_sphere",
            "evidence_id": "synthetic_visibility_g0_g3",
            "category": "synthetic_correctness",
            "metric_1": "rgb_max_absolute_error",
            "value_1": float(analytic["rgb_max_absolute_error"]),
            "metric_2": "transmittance_max_absolute_error",
            "value_2": float(analytic["transmittance_max_absolute_error"]),
            "verdict": "accepted_cpu_g0",
            "claim_scope": shared_scope,
        },
        {
            "row_id": "g0_physical_gauge_jacobian",
            "evidence_id": "synthetic_visibility_g0_g3",
            "category": "synthetic_correctness",
            "metric_1": "with_jacobian_rgb_max_error",
            "value_1": float(
                gauge["with_physical_jacobian_rgb_max_absolute_error"]
            ),
            "metric_2": "without_over_with_error_ratio",
            "value_2": float(gauge["error_ratio_without_over_with"]),
            "verdict": "accepted_cpu_g0",
            "claim_scope": shared_scope,
        },
        {
            "row_id": "g3_depth_layer_128_accuracy",
            "evidence_id": "synthetic_visibility_g0_g3",
            "category": "synthetic_visibility",
            "metric_1": "psnr_db_mean",
            "value_1": _mean(deepest, "rgb_psnr_db"),
            "metric_2": "psnr_db_p05",
            "value_2": _linear_quantile(
                [float(row["rgb_psnr_db"]) for row in deepest], 0.05
            ),
            "verdict": "accepted_cpu_g3",
            "claim_scope": shared_scope,
        },
        {
            "row_id": "g3_crossing_vs_representative_sort",
            "evidence_id": "synthetic_visibility_g0_g3",
            "category": "synthetic_visibility",
            "metric_1": "depth_layer_128_rgb_mse_mean",
            "value_1": _mean(worldfoam, "rgb_mse"),
            "metric_2": "representative_sorted_rgb_mse_mean",
            "value_2": _mean(representative, "rgb_mse"),
            "verdict": "accepted_cpu_g3",
            "claim_scope": shared_scope,
        },
        {
            "row_id": "g3_crossing_vs_depth_marginal",
            "evidence_id": "synthetic_visibility_g0_g3",
            "category": "synthetic_visibility",
            "metric_1": "depth_layer_128_rgb_mse_mean",
            "value_1": _mean(worldfoam, "rgb_mse"),
            "metric_2": "depth_marginal_rgb_mse_mean",
            "value_2": _mean(marginal, "rgb_mse"),
            "verdict": "accepted_cpu_g3",
            "claim_scope": shared_scope,
        },
        {
            "row_id": "g3_crossing_flicker_vs_representative_sort",
            "evidence_id": "synthetic_visibility_g0_g3",
            "category": "synthetic_visibility",
            "metric_1": "depth_layer_128_flicker_mean",
            "value_1": _mean(worldfoam, "temporal_flicker_error"),
            "metric_2": "representative_sorted_flicker_mean",
            "value_2": _mean(representative, "temporal_flicker_error"),
            "verdict": "accepted_cpu_g3",
            "claim_scope": shared_scope,
        },
        {
            "row_id": "g3_crossing_gradient_variance_vs_representative_sort",
            "evidence_id": "synthetic_visibility_g0_g3",
            "category": "synthetic_visibility",
            "metric_1": "depth_layer_128_gradient_variance_mean",
            "value_1": _mean(
                worldfoam, "temporal_gradient_error_variance"
            ),
            "metric_2": "representative_sorted_gradient_variance_mean",
            "value_2": _mean(
                representative, "temporal_gradient_error_variance"
            ),
            "verdict": "accepted_cpu_g3",
            "claim_scope": shared_scope,
        },
        {
            "row_id": "g3_adaptive_fallback",
            "evidence_id": "synthetic_visibility_g0_g3",
            "category": "synthetic_visibility",
            "metric_1": "fallback_fraction_mean",
            "value_1": float(fallback["mean"]),
            "metric_2": "fallback_fraction_p95",
            "value_2": float(fallback["p95"]),
            "verdict": "accepted_cpu_g3_diagnostic",
            "claim_scope": shared_scope,
        },
        {
            "row_id": "g3_crossing_order_flips",
            "evidence_id": "synthetic_visibility_g0_g3",
            "category": "synthetic_visibility",
            "metric_1": "representative_sorted_order_flips",
            "value_1": float(
                sum(
                    int(row["representative_order_flip_count"])
                    for row in representative
                )
            ),
            "metric_2": "depth_layer_128_order_flips",
            "value_2": float(
                sum(
                    int(row["representative_order_flip_count"])
                    for row in worldfoam
                )
            ),
            "verdict": "accepted_cpu_g3_diagnostic",
            "claim_scope": shared_scope,
        },
    ]


def _synthetic_visibility_asset_paths(
    path: Path,
    _payload: Mapping[str, Any],
) -> Mapping[str, Path]:
    return {
        name: path.parent / "figures" / name
        for name in EXPECTED_VISIBILITY_FIGURES
    }


def _adaptive_material_asset_paths(
    path: Path,
    payload: Mapping[str, Any],
) -> Mapping[str, Path]:
    names = tuple(sorted(str(name) for name in payload["assets"]))
    return {
        name: path.parent / ("figures" if name.endswith(".svg") else "") / name
        for name in names
    }


def _validate_g4_public_quality(
    path: Path,
    _payload: Mapping[str, Any],
) -> list[str]:
    report = verify_g4_artifact_file(path, config_path=g4_assets.DEFAULT_CONFIG)
    failures = [str(value) for value in report.get("failures", ())]
    if report.get("accepted") is not True:
        failures.append("independent G4-v2 verifier did not accept the artifact")
    if report.get("public_quality_evidence") is not True:
        failures.append("G4-v2 artifact is not public-quality evidence")
    if report.get("row_count") != 36:
        failures.append("G4-v2 artifact does not contain exactly 36 measured rows")
    return sorted(set(failures))


def _extract_g4_public_quality(
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != 36:
        raise ValueError("accepted G4-v2 payload lost its 36-row matrix")
    result: list[dict[str, Any]] = []
    for row in rows:
        metrics = row["metrics"]
        result.append(
            {
                "row_id": str(row["row_id"]),
                "evidence_id": "g4_public_quality",
                "category": "public_heldout_quality",
                "metric_1": "heldout_eval_psnr",
                "value_1": float(metrics["heldout_eval_psnr"]),
                "metric_2": "heldout_eval_ssim",
                "value_2": float(metrics["heldout_eval_ssim"]),
                "verdict": "accepted_public_measured",
                "claim_scope": (
                    "matched selected-ray training; full 300-frame heldout camera"
                ),
            }
        )
    return result


def _validate_g6_native_memory(
    path: Path,
    _payload: Mapping[str, Any],
) -> list[str]:
    report = verify_g6_artifact_file(
        path,
        config_path=g6_assets.DEFAULT_CONFIG,
        contract_path=g6_assets.DEFAULT_CONTRACT,
    )
    failures = [str(value) for value in report.get("failures", ())]
    if report.get("accepted") is not True:
        failures.append("independent G6 verifier did not accept the artifact")
    if report.get("observed_row_count") != 12:
        failures.append("G6 artifact does not contain 12 primary measured rows")
    if report.get("observed_control_row_count") != 9:
        failures.append("G6 artifact does not contain 9 measured control rows")
    return sorted(set(failures))


def _extract_g6_native_memory(
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    primary = payload.get("rows")
    controls = payload.get("control_rows")
    if (
        not isinstance(primary, list)
        or len(primary) != 12
        or not isinstance(controls, list)
        or len(controls) != 9
    ):
        raise ValueError("accepted G6 payload lost its exact 12+9 row matrix")
    result: list[dict[str, Any]] = []
    for row in (*primary, *controls):
        memory = row["memory"]
        result.append(
            {
                "row_id": (
                    f"{row['mode']}/F{row['requested_frame_count']}/"
                    f"repeat_{row['repeat_index']}"
                ),
                "evidence_id": "g6_native_memory",
                "category": "native_training_memory",
                "metric_1": "sampled_mps_driver_peak_bytes",
                "value_1": float(memory["sampled_mps_driver_peak_bytes"]),
                "metric_2": "process_group_rss_peak_bytes",
                "value_2": float(
                    memory["parent_process_group_rss_sampled_peak_bytes"]
                ),
                "verdict": "accepted_native_measured",
                "claim_scope": (
                    "fresh-process synthetic systems memory/trainability only"
                ),
            }
        )
    return result


def default_specs(
    *,
    material_parity: Path = DEFAULT_MATERIAL_PARITY,
    material_fit: Path = DEFAULT_MATERIAL_FIT,
    adaptive_material: Path = DEFAULT_ADAPTIVE_MATERIAL,
    compiled_lie: Path = DEFAULT_COMPILED_LIE,
    constant_transfer: Path = DEFAULT_CONSTANT_TRANSFER,
    synthetic_visibility: Path = DEFAULT_SYNTHETIC_VISIBILITY,
    g4_public_quality: Path = DEFAULT_G4_PUBLIC_QUALITY,
    g6_native_memory: Path = DEFAULT_G6_NATIVE_MEMORY,
) -> tuple[EvidenceSpec, ...]:
    return (
        EvidenceSpec(
            evidence_id="m0_m5_segment_parity",
            gate="G0/G2",
            path=material_parity,
            verifier="strict_m0_m5_foundation_v1",
            scope=(
                "local fixed-segment CPU correctness and historical "
                "source-hash-checked Metal forward/VJP"
            ),
            validator=_validate_material_parity,
            extractor=_extract_material_parity,
        ),
        EvidenceSpec(
            evidence_id="m3_m5_partial_chord_fit",
            gate="material_ablation",
            path=material_fit,
            verifier="verify_finite_element_material_fit.verify_artifact",
            scope=(
                "three-seed synthetic heldout partial-chord capacity; "
                "complementarity, not a universal material winner"
            ),
            validator=_validate_material_fit,
            extractor=_extract_material_fit,
        ),
        EvidenceSpec(
            evidence_id="adaptive_m3_m5_basis_selection",
            gate="material_ablation",
            path=adaptive_material,
            verifier=(
                "verify_adaptive_material_basis_selection.verify_report"
            ),
            scope=(
                "three-seed float64 CPU synthetic adaptive M3/M5 selection "
                "on disjoint train/selection/heldout chords at matched "
                "24-byte payloads plus one basis-tag bit; no native, public "
                "quality, renderer-speed, or memory claim"
            ),
            validator=_validate_adaptive_material,
            extractor=_extract_adaptive_material,
            asset_paths=_adaptive_material_asset_paths,
        ),
        EvidenceSpec(
            evidence_id="compiled_lie_frame_density",
            gate="G1/G6-foundation",
            path=compiled_lie,
            verifier=(
                "compiled_lie_frame_density_gate."
                "verify_compiled_lie_frame_density_report"
            ),
            scope=(
                "CPU fixed-surrogate requested-density accounting only; "
                "never native allocator evidence"
            ),
            validator=_validate_compiled_lie,
            extractor=_extract_compiled_lie,
        ),
        EvidenceSpec(
            evidence_id="constant_density_ordered_transfer",
            gate="G0/G2",
            path=constant_transfer,
            verifier="cell_path_optical_transfer_fixture.verify_summary",
            scope=(
                "constant-density owner-word monoid/replay/finite-difference "
                "VJP only"
            ),
            validator=_validate_constant_transfer,
            extractor=_extract_constant_transfer,
        ),
        EvidenceSpec(
            evidence_id="synthetic_visibility_g0_g3",
            gate="G0/G3",
            path=synthetic_visibility,
            verifier=(
                "verify_worldfoam_synthetic_visibility_suite.verify_report"
            ),
            scope=(
                "accepted full float64 CPU S1-S8/C1-C7 synthetic matrix; "
                "no native runtime, allocator, trained-image, or public-data "
                "claim"
            ),
            validator=_validate_synthetic_visibility,
            extractor=_extract_synthetic_visibility,
            asset_paths=_synthetic_visibility_asset_paths,
        ),
        EvidenceSpec(
            evidence_id="g4_public_quality",
            gate="G4",
            path=g4_public_quality,
            verifier=(
                "verify_worldfoam_public_quality_ablation_v2."
                "verify_artifact_file"
            ),
            scope=(
                "36 real rows: three public calibrated scenes, seeds 17/29/43, "
                "four matched selected-ray training routes, unchanged full "
                "300-frame heldout evaluation"
            ),
            validator=_validate_g4_public_quality,
            extractor=_extract_g4_public_quality,
        ),
        EvidenceSpec(
            evidence_id="g6_native_memory",
            gate="G1/G2/G6",
            path=g6_native_memory,
            verifier=(
                "verify_worldfoam_training_memory_ablation."
                "verify_artifact_file"
            ),
            scope=(
                "21 fresh-process native MPS systems rows with hard allocator "
                "and process-group RSS limits; synthetic trainability only"
            ),
            validator=_validate_g6_native_memory,
            extractor=_extract_g6_native_memory,
        ),
    )


def collect_evidence(
    specs: Sequence[EvidenceSpec],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, bytes],
]:
    ledger: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    assets: dict[str, bytes] = {}
    for spec in specs:
        path = spec.path.expanduser().resolve()
        record: dict[str, Any] = {
            "evidence_id": spec.evidence_id,
            "gate": spec.gate,
            "path": _display_path(path),
            "verifier": spec.verifier,
            "scope": spec.scope,
            "status": "missing",
            "errors": [],
            "sha256": None,
            "bytes": None,
            "numeric_rows_emitted": 0,
            "dependencies": [],
        }
        if not path.is_file():
            record["errors"] = ["input file is missing"]
            ledger.append(record)
            continue
        record["sha256"] = _file_sha256(path)
        record["bytes"] = path.stat().st_size
        try:
            payload = _load_json(path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            record["status"] = "rejected"
            record["errors"] = [f"could not parse input: {error}"]
            ledger.append(record)
            continue
        try:
            failures = spec.validator(path, payload)
        except Exception as error:  # fail closed on verifier defects
            failures = [f"verifier raised {type(error).__name__}: {error}"]
        if failures:
            record["status"] = "rejected"
            record["errors"] = sorted(set(str(value) for value in failures))
            ledger.append(record)
            continue
        pending_assets: dict[str, bytes] = {}
        dependencies: list[dict[str, Any]] = []
        if spec.asset_paths is not None:
            try:
                source_paths = spec.asset_paths(path, payload)
                for bundle_name, source_path in sorted(source_paths.items()):
                    relative = Path(bundle_name)
                    if (
                        relative.is_absolute()
                        or ".." in relative.parts
                        or str(relative) in {"", "."}
                    ):
                        raise ValueError(
                            f"invalid bundle asset path: {bundle_name!r}"
                        )
                    resolved_source = source_path.expanduser().resolve()
                    if not resolved_source.is_file():
                        raise ValueError(
                            f"asset source is missing: {resolved_source}"
                        )
                    payload_bytes = resolved_source.read_bytes()
                    normalized_name = relative.as_posix()
                    if normalized_name in assets or normalized_name in pending_assets:
                        raise ValueError(
                            f"duplicate bundle asset path: {normalized_name}"
                        )
                    pending_assets[normalized_name] = payload_bytes
                    dependencies.append(
                        {
                            "bundle_path": normalized_name,
                            "path": _display_path(resolved_source),
                            "bytes": len(payload_bytes),
                            "sha256": hashlib.sha256(payload_bytes).hexdigest(),
                        }
                    )
            except Exception as error:
                record["status"] = "rejected"
                record["errors"] = [
                    "accepted input assets could not be retained: "
                    f"{type(error).__name__}: {error}"
                ]
                ledger.append(record)
                continue
        try:
            extracted = spec.extractor(payload)
        except Exception as error:
            record["status"] = "rejected"
            record["errors"] = [
                f"accepted input could not be flattened: {type(error).__name__}: {error}"
            ]
            ledger.append(record)
            continue
        record["status"] = "accepted"
        record["numeric_rows_emitted"] = len(extracted)
        record["dependencies"] = dependencies
        ledger.append(record)
        rows.extend(extracted)
        assets.update(pending_assets)
    return ledger, rows, assets


def _gate_status(ledger: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_id = {str(record["evidence_id"]): record for record in ledger}
    local_material = by_id.get("m0_m5_segment_parity", {}).get("status")
    adaptive_material = by_id.get(
        "adaptive_m3_m5_basis_selection", {}
    ).get("status")
    transfer = by_id.get("constant_density_ordered_transfer", {}).get("status")
    compiled = by_id.get("compiled_lie_frame_density", {}).get("status")
    visibility = by_id.get("synthetic_visibility_g0_g3", {}).get("status")
    g4 = by_id.get("g4_public_quality", {}).get("status")
    g6 = by_id.get("g6_native_memory", {}).get("status")
    g0_accepted = visibility == "accepted" and transfer == "accepted"
    g2_accepted = (
        g6 == "accepted"
        and local_material == "accepted"
        and transfer == "accepted"
    )
    complete = (
        g0_accepted
        and g2_accepted
        and visibility == "accepted"
        and g4 == "accepted"
        and g6 == "accepted"
    )
    return {
        "schema_version": 1,
        "paper_ready": complete,
        "iclr_ready": complete,
        "gates": [
            {
                "gate": "material_ablation",
                "status": (
                    "accepted_cpu_synthetic"
                    if adaptive_material == "accepted"
                    else "missing_or_rejected"
                ),
                "reason": (
                    "adaptive M3/M5 selection is accepted only for the "
                    "independently verified float64 CPU synthetic chord gate; "
                    "P0 replacement, native integration, G4, and G6 remain open"
                ),
            },
            {
                "gate": "G0",
                "status": (
                    "accepted_cpu_synthetic"
                    if g0_accepted
                    else "missing_or_rejected"
                ),
                "reason": (
                    "the independently verified S1-S8/C1-C7 float64 CPU "
                    "suite and constant-density transfer fixture establish "
                    "synthetic correctness only"
                ),
            },
            {
                "gate": "G1",
                "status": (
                    "accepted_native_systems"
                    if g6 == "accepted"
                    else (
                        "cpu_foundation_only"
                        if compiled == "accepted"
                        else "missing_or_rejected"
                    )
                ),
                "reason": (
                    "the independently verified G6 matrix contains native "
                    "same-representation work and end-to-end step timings"
                    if g6 == "accepted"
                    else "native same-representation speed and end-to-end "
                    "wall-time evidence are absent"
                ),
            },
            {
                "gate": "G2",
                "status": (
                    "accepted_native_training"
                    if g2_accepted
                    else (
                        "partial_foundation"
                        if local_material == "accepted" and transfer == "accepted"
                        else "missing_or_rejected"
                    )
                ),
                "reason": (
                    "local correctness plus the accepted G6 staged/fused/replay "
                    "parity and real optimizer mutations establish this gate"
                    if g2_accepted
                    else "local material and fixed-word VJPs do not establish "
                    "the rebuilt native full-geometry training path"
                ),
            },
            {
                "gate": "G3",
                "status": (
                    "accepted_cpu_synthetic"
                    if visibility == "accepted"
                    else "missing_or_rejected"
                ),
                "reason": (
                    "accepted only for the verified float64 CPU synthetic "
                    "S1-S8/C1-C7 suite; it is not trained public-data or "
                    "native-runtime evidence"
                ),
            },
            {
                "gate": "G4",
                "status": (
                    "accepted_public_quality"
                    if g4 == "accepted"
                    else "not_measured"
                ),
                "reason": (
                    "independently verified 36-row matched selected-ray public "
                    "quality matrix"
                    if g4 == "accepted"
                    else PLACEHOLDER_GATES[0]["required_evidence"]
                ),
            },
            {
                "gate": "G5",
                "status": "not_claimed",
                "reason": (
                    "official CUDA/Warp parity is absent; this bundle makes "
                    "no upstream PowerFoam parity claim"
                ),
            },
            {
                "gate": "G6",
                "status": (
                    "accepted_native_memory"
                    if g6 == "accepted"
                    else "not_measured"
                ),
                "reason": (
                    "independently verified 21-row fresh-process native "
                    "memory/work matrix under hard allocator and RSS limits"
                    if g6 == "accepted"
                    else PLACEHOLDER_GATES[1]["required_evidence"]
                ),
            },
        ],
        "claims": {
            "adaptive_material_basis_cpu": adaptive_material == "accepted",
            "synthetic_cpu_g0_g3": g0_accepted,
            "native_memory_fit": g6 == "accepted",
            "public_quality": g4 == "accepted",
            "public_or_native_visibility_advantage": False,
            "official_cuda_warp_parity": False,
            "state_of_the_art": False,
        },
    }


FOUNDATION_COLUMNS = (
    "row_id",
    "evidence_id",
    "category",
    "metric_1",
    "value_1",
    "metric_2",
    "value_2",
    "verdict",
    "claim_scope",
)


def _csv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=FOUNDATION_COLUMNS,
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in FOUNDATION_COLUMNS})
    return stream.getvalue().encode("utf-8")


def _markdown_table(rows: Sequence[Mapping[str, Any]]) -> bytes:
    visibility_accepted = any(
        row.get("evidence_id") == "synthetic_visibility_g0_g3"
        for row in rows
    )
    adaptive_material_accepted = any(
        row.get("evidence_id") == "adaptive_m3_m5_basis_selection"
        for row in rows
    )
    g4_accepted = any(row.get("evidence_id") == "g4_public_quality" for row in rows)
    g6_accepted = any(row.get("evidence_id") == "g6_native_memory" for row in rows)
    lines = [
        "# WorldFoam Paper-B verified foundation rows",
        "",
        (
            "Only independently accepted local/foundation evidence appears "
            "below. This table is not native-memory or public-quality evidence."
        ),
        "",
        "| Row | Category | Metric 1 | Value | Metric 2 | Value | Verdict | Scope |",
        "| --- | --- | --- | ---: | --- | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {row_id} | {category} | {metric_1} | {value_1:.6g} | "
            "{metric_2} | {value_2:.6g} | {verdict} | {claim_scope} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            (
                "G3 visibility stress: **ACCEPTED — CPU SYNTHETIC ONLY** "
                "(S1-S8/C1-C7; not native runtime or public-data quality)."
                if visibility_accepted
                else "G3 visibility stress: **MISSING OR REJECTED**."
            ),
            "",
            (
                "Adaptive M3/M5 basis selection: **ACCEPTED — CPU SYNTHETIC "
                "ONLY** (not native material promotion)."
                if adaptive_material_accepted
                else "Adaptive M3/M5 basis selection: **MISSING OR REJECTED**."
            ),
            "",
            (
                "G4 public heldout quality: **ACCEPTED — 36 MEASURED ROWS**."
                if g4_accepted
                else "G4 public heldout quality: **NOT MEASURED**."
            ),
            "",
            (
                "G6 native training memory: **ACCEPTED — 21 MEASURED ROWS**."
                if g6_accepted
                else "G6 native training memory: **NOT MEASURED**."
            ),
            "",
        ]
    )
    return ("\n".join(lines)).encode("utf-8")


def _tex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


TEX_ROW_LABELS = {
    "m0_m5_cpu_segment_parity": "M0--M5 CPU parity",
    "m0_m5_metal_segment_parity": "M0--M5 Metal parity (historical)",
    "partial_chord_positive_p2": "Positive-P2 target",
    "partial_chord_convex_log_p2": "Convex-log-P2 target",
    "adaptive_m3_m5_mean_loss": "Adaptive M3/M5 mean loss",
    "adaptive_m3_m5_selection_accuracy": "Adaptive M3/M5 selection",
    "constant_density_ordered_transfer": "Constant-density transfer",
    "g0_analytic_constant_sphere": "G0 analytic sphere",
    "g0_physical_gauge_jacobian": "G0 physical gauge Jacobian",
    "g3_depth_layer_128_accuracy": "G3 128-layer accuracy",
    "g3_crossing_vs_representative_sort": "G3 crossing vs. repr. sort",
    "g3_crossing_vs_depth_marginal": "G3 crossing vs. depth marginal",
    "g3_crossing_flicker_vs_representative_sort": "G3 crossing flicker",
    "g3_crossing_gradient_variance_vs_representative_sort": (
        "G3 crossing gradient variance"
    ),
    "g3_adaptive_fallback": "G3 adaptive fallback",
    "g3_crossing_order_flips": "G3 crossing order flips",
}
TEX_METRIC_LABELS = {
    "integral_max_abs_error": "integral max abs. err.",
    "finite_difference_vjp_normalized_error": "FD VJP norm. err.",
    "forward_normalized_error": "forward norm. err.",
    "vjp_normalized_error": "VJP norm. err.",
    "M3_heldout_loss": "M3 heldout loss",
    "M5_heldout_loss": "M5 heldout loss",
    "adaptive_to_best_fixed_ratio": "adaptive / best fixed",
    "adaptive_to_oracle_ratio": "adaptive / oracle",
    "pure_family_selection_accuracy": "pure-family accuracy",
    "selection_oracle_agreement": "oracle agreement",
    "render_max_abs_error": "render max abs. err.",
    "vjp_max_abs_error": "VJP max abs. err.",
    "rgb_max_absolute_error": "RGB max abs. err.",
    "transmittance_max_absolute_error": "transmittance max abs. err.",
    "with_jacobian_rgb_max_error": "with-Jac. RGB max err.",
    "without_over_with_error_ratio": "no-Jac./Jac. error ratio",
    "psnr_db_mean": "PSNR mean (dB)",
    "psnr_db_p05": "PSNR p05 (dB)",
    "depth_layer_128_rgb_mse_mean": "128-layer RGB MSE",
    "representative_sorted_rgb_mse_mean": "repr.-sort RGB MSE",
    "depth_marginal_rgb_mse_mean": "depth-marginal RGB MSE",
    "depth_layer_128_flicker_mean": "128-layer flicker",
    "representative_sorted_flicker_mean": "repr.-sort flicker",
    "depth_layer_128_gradient_variance_mean": "128-layer grad. var.",
    "representative_sorted_gradient_variance_mean": "repr.-sort grad. var.",
    "fallback_fraction_mean": "fallback mean",
    "fallback_fraction_p95": "fallback p95",
    "representative_sorted_order_flips": "repr.-sort flips",
    "depth_layer_128_order_flips": "128-layer flips",
}


def _tex_table(rows: Sequence[Mapping[str, Any]]) -> bytes:
    visibility_accepted = any(
        row.get("evidence_id") == "synthetic_visibility_g0_g3"
        for row in rows
    )
    g4_accepted = any(row.get("evidence_id") == "g4_public_quality" for row in rows)
    g6_accepted = any(row.get("evidence_id") == "g6_native_memory" for row in rows)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llrlr}",
        r"\toprule",
        r"Evidence & Metric 1 & Value & Metric 2 & Value \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            "{} & {} & {:.3e} & {} & {:.3e} \\\\".format(
                _tex_escape(
                    TEX_ROW_LABELS.get(row["row_id"], row["row_id"])
                ),
                _tex_escape(
                    TEX_METRIC_LABELS.get(row["metric_1"], row["metric_1"])
                ),
                float(row["value_1"]),
                _tex_escape(
                    TEX_METRIC_LABELS.get(row["metric_2"], row["metric_2"])
                ),
                float(row["value_2"]),
            )
        )
    lines.extend(
        [
            r"\midrule",
            (
                r"G3 & \multicolumn{4}{l}{ACCEPTED: CPU synthetic only; "
                r"not native/public evidence} \\"
                if visibility_accepted
                else r"G3 & \multicolumn{4}{l}{MISSING OR REJECTED} \\"
            ),
            (
                r"G4 & \multicolumn{4}{l}{ACCEPTED: 36 measured public rows} \\"
                if g4_accepted
                else r"G4 & \multicolumn{4}{l}{NOT MEASURED: public quality} \\"
            ),
            (
                r"G6 & \multicolumn{4}{l}{ACCEPTED: 21 measured native rows} \\"
                if g6_accepted
                else r"G6 & \multicolumn{4}{l}{NOT MEASURED: native memory} \\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Verifier-accepted WorldFoam foundation and CPU "
                r"synthetic evidence. G3 covers only the S1--S8/C1--C7 "
                r"float64 CPU suite; it is not native-runtime or public-data "
                r"evidence. The Metal row is historical and source-hash "
                r"checked. Adaptive M3/M5 rows are synthetic chord-selection "
                r"evidence and do not authorize native material promotion. "
                + (
                    r"G4 public quality and G6 native memory are supplied in "
                    r"their dedicated verifier-derived tables.}"
                    if g4_accepted and g6_accepted
                    else r"Missing G4/G6 gates remain explicitly unmeasured.}"
                )
            ),
            r"\label{tab:worldfoam-foundation}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _synthetic_visibility_table_tex(
    rows: Sequence[Mapping[str, Any]],
) -> bytes:
    by_id = {str(row["row_id"]): row for row in rows}
    required = (
        "g0_analytic_constant_sphere",
        "g0_physical_gauge_jacobian",
        "g3_depth_layer_128_accuracy",
        "g3_crossing_vs_representative_sort",
        "g3_crossing_vs_depth_marginal",
    )
    if any(row_id not in by_id for row_id in required):
        return (
            "\\begin{table*}[t]\n"
            "\\centering\n"
            "\\fbox{\\parbox{0.92\\linewidth}{"
            "Synthetic visibility evidence is missing or rejected.}}\n"
            "\\caption{WorldFoam synthetic visibility gate. No numeric rows "
            "are emitted unless the complete verifier-accepted G0/G3 source "
            "artifact is present.}\n"
            "\\label{tab:worldfoam-synthetic-visibility}\n"
            "\\end{table*}\n"
        ).encode("utf-8")

    sphere = by_id[required[0]]
    gauge = by_id[required[1]]
    accuracy = by_id[required[2]]
    representative = by_id[required[3]]
    marginal = by_id[required[4]]
    depth_mse = float(representative["value_1"])
    representative_gain = float(representative["value_2"]) / depth_mse
    marginal_gain = float(marginal["value_2"]) / float(marginal["value_1"])
    with_jacobian = float(gauge["value_1"])
    without_jacobian = with_jacobian * float(gauge["value_2"])
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}p{0.27\linewidth}p{0.43\linewidth}p{0.22\linewidth}@{}}",
        r"\toprule",
        "Check & Metric & Verified result \\\\",
        r"\midrule",
        (
            r"Retained depth, 128 layers & fifth-percentile RGB PSNR & "
            f"{float(accuracy['value_2']):.4f} dB \\\\"
        ),
        (
            r"Crossings vs. representative sort & RGB-MSE reduction & "
            f"{representative_gain:.4f}$\\times$ \\\\"
        ),
        (
            r"Crossings vs. depth marginal & RGB-MSE reduction & "
            f"{marginal_gain:.3f}$\\times$ \\\\"
        ),
        (
            r"Gauge reparameterization & max. RGB error, with / without Jacobian & "
            f"{with_jacobian:.5e} / {without_jacobian:.6f} \\\\"
        ),
        (
            r"Analytic constant sphere & max. RGB / transmittance error & "
            f"{float(sphere['value_1']):.5e} / {float(sphere['value_2']):.5e} \\\\"
        ),
        r"\bottomrule",
        r"\end{tabular}",
        (
            r"\caption{Verifier-derived float64 CPU G0/G3 evidence over all "
            r"eight synthetic scenes and seven camera programs. This table "
            r"does not report native runtime, native peak memory, kinetic "
            r"compiler acceptance, or trained public-data quality.}"
        ),
        r"\label{tab:worldfoam-synthetic-visibility}",
        r"\end{table*}",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _svg_document(width: int, height: int, body: Sequence[str]) -> bytes:
    prefix = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        "<title>WorldFoam Paper-B generated artifact</title>",
        (
            "<desc>Deterministically generated from independently verified "
            "evidence; claim scope is recorded in the bundle ledger.</desc>"
        ),
        '<rect width="100%" height="100%" fill="#fbfbfd"/>',
    ]
    return ("\n".join([*prefix, *body, "</svg>", ""])).encode("utf-8")


def _material_loss_svg(rows: Sequence[Mapping[str, Any]]) -> bytes:
    selected = {
        str(row["row_id"]): row
        for row in rows
        if row.get("category") == "material_capacity"
    }
    required = ("partial_chord_positive_p2", "partial_chord_convex_log_p2")
    if any(key not in selected for key in required):
        return _placeholder_svg(
            "Material family loss",
            "NO ACCEPTED M3/M5 MATERIAL-FIT EVIDENCE",
        )
    width, height = 900, 520
    plot_left, plot_top, plot_width, plot_height = 110, 90, 700, 330
    log_min, log_max = -18.0, -2.0
    colors = ("#3066be", "#d1495b")
    body = [
        '<text x="450" y="40" text-anchor="middle" font-family="sans-serif" '
        'font-size="24" font-weight="700">Heldout partial-chord material fit</text>',
        '<text x="450" y="67" text-anchor="middle" font-family="sans-serif" '
        'font-size="14" fill="#555">matched six-scalar M3/M5; lower is better</text>',
    ]
    for tick in range(-18, -1, 2):
        y = plot_top + (log_max - tick) / (log_max - log_min) * plot_height
        body.append(
            f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_left + plot_width}" '
            f'y2="{y:.2f}" stroke="#d9dce3" stroke-width="1"/>'
        )
        body.append(
            f'<text x="{plot_left - 12}" y="{y + 5:.2f}" text-anchor="end" '
            f'font-family="monospace" font-size="12">1e{tick}</text>'
        )
    groups = (
        ("positive P2 target", selected[required[0]]),
        ("convex log-P2 target", selected[required[1]]),
    )
    for group_index, (label, row) in enumerate(groups):
        center = plot_left + 210 + group_index * 330
        values = (float(row["value_1"]), float(row["value_2"]))
        for series_index, value in enumerate(values):
            clipped_log = min(log_max, max(log_min, math.log10(max(value, 1e-30))))
            y = plot_top + (log_max - clipped_log) / (log_max - log_min) * plot_height
            x = center - 62 + series_index * 80
            bar_height = plot_top + plot_height - y
            body.append(
                f'<rect x="{x}" y="{y:.2f}" width="54" height="{bar_height:.2f}" '
                f'fill="{colors[series_index]}" rx="3"/>'
            )
            body.append(
                f'<text x="{x + 27}" y="{max(84.0, y - 7):.2f}" '
                f'text-anchor="middle" font-family="monospace" font-size="11">'
                f'{value:.2e}</text>'
            )
        body.append(
            f'<text x="{center - 22}" y="455" text-anchor="middle" '
            f'font-family="sans-serif" font-size="14">{html.escape(label)}</text>'
        )
    body.extend(
        [
            '<rect x="296" y="482" width="14" height="14" fill="#3066be"/>',
            '<text x="318" y="494" font-family="sans-serif" font-size="13">M3 positive P2</text>',
            '<rect x="490" y="482" width="14" height="14" fill="#d1495b"/>',
            '<text x="512" y="494" font-family="sans-serif" font-size="13">M5 convex log-P2</text>',
        ]
    )
    return _svg_document(width, height, body)


def _error_summary_svg(rows: Sequence[Mapping[str, Any]]) -> bytes:
    readable_labels = {
        ("m0_m5_cpu_segment_parity", "integral_max_abs_error"):
            "CPU segment integral",
        ("m0_m5_cpu_segment_parity", "finite_difference_vjp_normalized_error"):
            "CPU segment VJP",
        ("m0_m5_metal_segment_parity", "forward_normalized_error"):
            "Metal segment forward",
        ("m0_m5_metal_segment_parity", "vjp_normalized_error"):
            "Metal segment VJP",
        ("constant_density_ordered_transfer", "render_max_abs_error"):
            "Constant-density render",
        ("constant_density_ordered_transfer", "vjp_max_abs_error"):
            "Constant-density VJP",
    }
    metrics: list[tuple[str, float]] = []
    for row in rows:
        if row.get("category") not in {
            "local_material_correctness",
            "ordered_transfer_algebra",
        }:
            continue
        for metric_key, value_key in (("metric_1", "value_1"), ("metric_2", "value_2")):
            value = float(row[value_key])
            identity = (str(row["row_id"]), str(row[metric_key]))
            metrics.append((readable_labels.get(identity, " / ".join(identity)), value))
    if not metrics:
        return _placeholder_svg(
            "Foundation error summary",
            "NO ACCEPTED LOCAL CORRECTNESS EVIDENCE",
        )
    width = 1000
    row_height = 46
    height = 115 + row_height * len(metrics)
    bar_left, bar_width = 330, 610
    log_min, log_max = -18.0, -4.0
    body = [
        '<text x="500" y="38" text-anchor="middle" font-family="sans-serif" '
        'font-size="24" font-weight="700">Accepted local foundation errors</text>',
        '<text x="500" y="63" text-anchor="middle" font-family="sans-serif" '
        'font-size="13" fill="#555">log scale; zero values are drawn at the 1e-18 floor</text>',
    ]
    for index, (label, value) in enumerate(metrics):
        y = 95 + index * row_height
        log_value = min(log_max, max(log_min, math.log10(max(value, 1e-18))))
        fraction = (log_value - log_min) / (log_max - log_min)
        body.append(
            f'<text x="{bar_left - 12}" y="{y + 15}" text-anchor="end" '
            f'font-family="sans-serif" font-size="12">{html.escape(label)}</text>'
        )
        body.append(
            f'<rect x="{bar_left}" y="{y}" width="{bar_width}" height="20" '
            'fill="#e7e9ef" rx="3"/>'
        )
        body.append(
            f'<rect x="{bar_left}" y="{y}" width="{max(2.0, fraction * bar_width):.2f}" '
            'height="20" fill="#3b7ea1" rx="3"/>'
        )
        body.append(
            f'<text x="{bar_left + bar_width + 10}" y="{y + 15}" '
            f'font-family="monospace" font-size="12">{value:.2e}</text>'
        )
    return _svg_document(width, height, body)


def _placeholder_svg(title: str, subtitle: str) -> bytes:
    safe_title = html.escape(title)
    wrapped_subtitle = textwrap.wrap(subtitle, width=88)[:3]
    subtitle_lines = [
        (
            f'<text x="450" y="{205 + 22 * index}" text-anchor="middle" '
            'font-family="sans-serif" font-size="14" fill="#4a4a4a">'
            f"{html.escape(line)}</text>"
        )
        for index, line in enumerate(wrapped_subtitle)
    ]
    return _svg_document(
        900,
        340,
        [
            '<rect x="28" y="28" width="844" height="284" fill="#fff4f4" '
            'stroke="#b3261e" stroke-width="3" stroke-dasharray="10 7" rx="12"/>',
            f'<text x="450" y="105" text-anchor="middle" font-family="sans-serif" '
            f'font-size="25" font-weight="700">{safe_title}</text>',
            '<text x="450" y="155" text-anchor="middle" font-family="sans-serif" '
            'font-size="30" font-weight="800" fill="#b3261e">NOT MEASURED</text>',
            *subtitle_lines,
        ],
    )


def _readme_bytes(
    ledger: Sequence[Mapping[str, Any]],
    gate_status: Mapping[str, Any],
) -> bytes:
    accepted = [record["evidence_id"] for record in ledger if record["status"] == "accepted"]
    rejected = [record["evidence_id"] for record in ledger if record["status"] == "rejected"]
    missing = [record["evidence_id"] for record in ledger if record["status"] == "missing"]
    visibility_accepted = any(
        record["evidence_id"] == "synthetic_visibility_g0_g3"
        and record["status"] == "accepted"
        for record in ledger
    )
    adaptive_material_accepted = any(
        record["evidence_id"] == "adaptive_m3_m5_basis_selection"
        and record["status"] == "accepted"
        for record in ledger
    )
    g4_accepted = any(
        record["evidence_id"] == "g4_public_quality"
        and record["status"] == "accepted"
        for record in ledger
    )
    g6_accepted = any(
        record["evidence_id"] == "g6_native_memory"
        and record["status"] == "accepted"
        for record in ledger
    )
    lines = [
        "# WorldFoam Paper-B foundation artifact bundle",
        "",
        "This is a deterministic, fail-closed evidence bundle. It is not itself a submission package.",
        "",
        f"- Accepted inputs: {', '.join(accepted) if accepted else 'none'}",
        f"- Rejected inputs: {', '.join(rejected) if rejected else 'none'}",
        f"- Missing inputs: {', '.join(missing) if missing else 'none'}",
        f"- Native memory fit: {str(g6_accepted).lower()}",
        f"- Public quality evidence: {str(g4_accepted).lower()}",
        (
            "- Adaptive M3/M5 CPU synthetic basis selection: accepted"
            if adaptive_material_accepted
            else "- Adaptive M3/M5 CPU synthetic basis selection: missing or rejected"
        ),
        (
            "- G0/G3 synthetic CPU visibility: accepted (S1-S8/C1-C7 only)"
            if visibility_accepted
            else "- G0/G3 synthetic CPU visibility: missing or rejected"
        ),
        "- Public/native visibility advantage: false",
        f"- Evidence ready for ICLR packaging: {str(bool(gate_status['iclr_ready'])).lower()}",
        "",
        (
            "G4/G6 placeholders are emitted only for missing or rejected gates. "
            "Accepted gates are replaced by independently rebuilt numeric assets."
        ),
        "",
        f"Gate ledger digest: `{_canonical_json_sha256(gate_status)}`",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def build_bundle(specs: Sequence[EvidenceSpec] | None = None) -> Bundle:
    resolved_specs = tuple(default_specs() if specs is None else specs)
    ledger_rows, foundation_rows, accepted_assets = collect_evidence(
        resolved_specs
    )
    gate_status = _gate_status(ledger_rows)
    claims = gate_status["claims"]
    complete = bool(gate_status["paper_ready"])
    ledger = {
        "schema_version": 1,
        "generator": GENERATOR_NAME,
        "records": ledger_rows,
        "accepted_count": sum(record["status"] == "accepted" for record in ledger_rows),
        "rejected_count": sum(record["status"] == "rejected" for record in ledger_rows),
        "missing_count": sum(record["status"] == "missing" for record in ledger_rows),
        "numeric_row_count": len(foundation_rows),
        "claim_boundary": {
            "native_memory_fit": bool(claims["native_memory_fit"]),
            "public_quality": bool(claims["public_quality"]),
            "public_or_native_visibility_advantage": bool(
                claims["public_or_native_visibility_advantage"]
            ),
            "paper_ready": complete,
        },
    }
    files: dict[str, bytes] = {
        "README.md": _readme_bytes(ledger_rows, gate_status),
        "evidence_ledger.json": _canonical_json_bytes(ledger),
        "foundation_rows.json": _canonical_json_bytes(
            {"schema_version": 1, "rows": foundation_rows}
        ),
        "gate_status.json": _canonical_json_bytes(gate_status),
        "foundation_table.csv": _csv_bytes(foundation_rows),
        "foundation_table.md": _markdown_table(foundation_rows),
        "foundation_table.tex": _tex_table(foundation_rows),
        "synthetic_visibility_table.tex": _synthetic_visibility_table_tex(
            foundation_rows
        ),
        "material_family_loss.svg": _material_loss_svg(foundation_rows),
        "foundation_error_summary.svg": _error_summary_svg(foundation_rows),
    }
    collisions = set(files).intersection(accepted_assets)
    if collisions:
        raise ValueError(
            "accepted evidence assets collide with generated files: "
            + ", ".join(sorted(collisions))
        )
    files.update(accepted_assets)
    records_by_id = {
        str(record["evidence_id"]): record for record in ledger_rows
    }
    specs_by_id = {spec.evidence_id: spec for spec in resolved_specs}
    if records_by_id.get("g4_public_quality", {}).get("status") == "accepted":
        spec = specs_by_id["g4_public_quality"]
        files.update(
            g4_assets.build_assets(spec.path, g4_assets.DEFAULT_CONFIG)
        )
    else:
        placeholder = PLACEHOLDER_GATES[0]
        files["g4_public_quality_placeholder.svg"] = _placeholder_svg(
            str(placeholder["title"]),
            str(placeholder["required_evidence"]),
        )
    if records_by_id.get("g6_native_memory", {}).get("status") == "accepted":
        spec = specs_by_id["g6_native_memory"]
        files.update(
            g6_assets.build_assets(
                spec.path,
                g6_assets.DEFAULT_CONFIG,
                g6_assets.DEFAULT_CONTRACT,
            )
        )
    else:
        placeholder = PLACEHOLDER_GATES[1]
        files["g6_native_memory_placeholder.svg"] = _placeholder_svg(
            str(placeholder["title"]),
            str(placeholder["required_evidence"]),
        )
    return Bundle(
        complete=complete,
        ledger=ledger,
        gate_status=gate_status,
        foundation_rows=foundation_rows,
        files=files,
    )


def _manifest(bundle: Bundle) -> dict[str, Any]:
    file_rows = [
        {
            "path": name,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        for name, data in sorted(bundle.files.items())
    ]
    generator_source = Path(__file__).resolve()
    content_contract = {
        "complete": bundle.complete,
        "ledger_sha256": _canonical_json_sha256(bundle.ledger),
        "gate_status_sha256": _canonical_json_sha256(bundle.gate_status),
        "foundation_rows_sha256": _canonical_json_sha256(bundle.foundation_rows),
        "files": file_rows,
    }
    return {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "generator": GENERATOR_NAME,
        "generator_schema_version": GENERATOR_SCHEMA_VERSION,
        "generator_source": {
            "path": _display_path(generator_source),
            "bytes": generator_source.stat().st_size,
            "sha256": _file_sha256(generator_source),
        },
        "complete": bundle.complete,
        "claims": bundle.gate_status["claims"],
        "files": file_rows,
        "content_sha256": _canonical_json_sha256(content_contract),
    }


def write_bundle(bundle: Bundle, out_dir: Path) -> dict[str, Any]:
    destination = out_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    expected = set(bundle.files) | {"manifest.json"}
    prior_manifest_path = destination / "manifest.json"
    prior_owned_files: dict[str, Mapping[str, Any]] = {}
    if prior_manifest_path.is_file():
        try:
            prior_manifest = _load_json(prior_manifest_path)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                "refusing to reconcile a bundle with an unreadable manifest"
            ) from error
        if prior_manifest.get("generator") != GENERATOR_NAME:
            raise ValueError("refusing to reconcile files owned by another generator")
        prior_rows = prior_manifest.get("files")
        if not isinstance(prior_rows, list):
            raise ValueError("prior bundle manifest has no file inventory")
        prior_owned_files = {
            str(row.get("path")): row
            for row in prior_rows
            if isinstance(row, Mapping) and isinstance(row.get("path"), str)
        }
    for path in destination.iterdir():
        if path.is_file() and path.name not in expected:
            prior = prior_owned_files.get(path.name)
            if (
                not isinstance(prior, Mapping)
                or prior.get("bytes") != path.stat().st_size
                or prior.get("sha256") != _file_sha256(path)
            ):
                raise ValueError(f"refusing to remove an unowned bundle file: {path}")
            path.unlink()
    for name, data in sorted(bundle.files.items()):
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = _manifest(bundle)
    (destination / "manifest.json").write_bytes(_canonical_json_bytes(manifest))
    return manifest


def verify_bundle_dir(path: Path) -> list[str]:
    directory = path.expanduser().resolve()
    failures: list[str] = []
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        return ["manifest.json is missing"]
    try:
        manifest = _load_json(manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        return [f"could not load manifest: {error}"]
    if manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
        failures.append("unsupported bundle schema")
    if manifest.get("generator") != GENERATOR_NAME:
        failures.append("generator identity changed")
    if not isinstance(manifest.get("complete"), bool):
        failures.append("bundle completeness flag must be boolean")
    claims = manifest.get("claims")
    always_prohibited_claims = (
        "public_or_native_visibility_advantage",
        "official_cuda_warp_parity",
        "state_of_the_art",
    )
    if not isinstance(claims, Mapping):
        failures.append("manifest claims are missing")
        claims = {}
    else:
        expected_claim_keys = {
            "adaptive_material_basis_cpu",
            "synthetic_cpu_g0_g3",
            "native_memory_fit",
            "public_quality",
            *always_prohibited_claims,
        }
        if set(claims) != expected_claim_keys:
            failures.append("manifest claim key set changed")
        if any(claims.get(key) is not False for key in always_prohibited_claims):
            failures.append("manifest contains an unsupported promoted claim")
        for key in expected_claim_keys:
            if not isinstance(claims.get(key), bool):
                failures.append(f"manifest claim must be boolean: {key}")
    generator_source = manifest.get("generator_source")
    if not isinstance(generator_source, Mapping):
        failures.append("generator source identity is missing")
    else:
        if generator_source.get("sha256") != _file_sha256(Path(__file__)):
            failures.append("bundle was generated by a different source revision")

    file_rows = manifest.get("files")
    if not isinstance(file_rows, list):
        failures.append("manifest files must be a list")
        file_rows = []
    expected_files: set[str] = {"manifest.json"}
    seen: set[str] = set()
    for index, row in enumerate(file_rows):
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            failures.append(f"manifest file row {index} is invalid")
            continue
        name = str(row["path"])
        if name in seen:
            failures.append(f"duplicate manifest file: {name}")
        seen.add(name)
        expected_files.add(name)
        candidate = directory / name
        if not candidate.is_file():
            failures.append(f"bundle file is missing: {name}")
            continue
        if candidate.stat().st_size != row.get("bytes"):
            failures.append(f"bundle byte size changed: {name}")
        if _file_sha256(candidate) != row.get("sha256"):
            failures.append(f"bundle digest changed: {name}")
    actual_files = {
        str(candidate.relative_to(directory))
        for candidate in directory.rglob("*")
        if candidate.is_file()
    }
    if actual_files != expected_files:
        failures.append("bundle contains missing or unexpected files")

    try:
        ledger = _load_json(directory / "evidence_ledger.json")
        gate_status = _load_json(directory / "gate_status.json")
        foundation_payload = _load_json(directory / "foundation_rows.json")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        failures.append(f"could not load core bundle JSON: {error}")
        return sorted(set(failures))
    records = ledger.get("records")
    if not isinstance(records, list):
        failures.append("ledger records are missing")
        records = []
    for record in records:
        if not isinstance(record, Mapping):
            failures.append("ledger record is invalid")
            continue
        status = record.get("status")
        if status not in {"accepted", "rejected", "missing"}:
            failures.append("ledger status is invalid")
            continue
        input_path = record.get("path")
        if not isinstance(input_path, str):
            failures.append("ledger input path is invalid")
            continue
        resolved = Path(input_path)
        if not resolved.is_absolute():
            resolved = ROOT / resolved
        if status != "missing":
            if not resolved.is_file():
                failures.append(f"retained input disappeared: {input_path}")
            elif _file_sha256(resolved) != record.get("sha256"):
                failures.append(f"retained input changed: {input_path}")
            elif status == "accepted":
                matching = {
                    spec.evidence_id: spec for spec in default_specs()
                }.get(str(record.get("evidence_id")))
                if matching is None:
                    failures.append(
                        "accepted record has no current independent verifier: "
                        f"{record.get('evidence_id')}"
                    )
                else:
                    try:
                        rejections = matching.validator(
                            resolved, _load_json(resolved)
                        )
                    except Exception as error:
                        rejections = [
                            "current verifier raised "
                            f"{type(error).__name__}: {error}"
                        ]
                    if rejections:
                        failures.append(
                            "current independent verifier now rejects "
                            f"{record.get('evidence_id')}: "
                            + "; ".join(sorted(set(rejections)))
                        )
        if status == "accepted" and record.get("errors") != []:
            failures.append(f"accepted record has errors: {record.get('evidence_id')}")
        if status != "accepted" and int(record.get("numeric_rows_emitted", -1)) != 0:
            failures.append(f"rejected/missing record emitted rows: {record.get('evidence_id')}")
        dependencies = record.get("dependencies")
        if not isinstance(dependencies, list):
            failures.append(
                f"record dependencies are invalid: {record.get('evidence_id')}"
            )
            dependencies = []
        for dependency in dependencies:
            if not isinstance(dependency, Mapping):
                failures.append("dependency record is invalid")
                continue
            source_value = dependency.get("path")
            bundle_value = dependency.get("bundle_path")
            if not isinstance(source_value, str) or not isinstance(
                bundle_value, str
            ):
                failures.append("dependency paths are invalid")
                continue
            source = Path(source_value)
            if not source.is_absolute():
                source = ROOT / source
            retained = directory / bundle_value
            for label, candidate in (
                ("source", source),
                ("bundled", retained),
            ):
                if not candidate.is_file():
                    failures.append(
                        f"dependency {label} is missing: {candidate}"
                    )
                elif (
                    _file_sha256(candidate) != dependency.get("sha256")
                    or candidate.stat().st_size != dependency.get("bytes")
                ):
                    failures.append(
                        f"dependency {label} changed: {candidate}"
                    )
    expected_gate_status = _gate_status(records)
    if gate_status != expected_gate_status:
        failures.append("gate status disagrees with independently retained evidence")
    expected_claims = expected_gate_status["claims"]
    if claims != expected_claims:
        failures.append("manifest claims disagree with independently retained evidence")
    expected_complete = bool(expected_gate_status["paper_ready"])
    if manifest.get("complete") is not expected_complete:
        failures.append("manifest completeness disagrees with verified gates")
    expected_boundary = {
        "native_memory_fit": bool(expected_claims["native_memory_fit"]),
        "paper_ready": expected_complete,
        "public_quality": bool(expected_claims["public_quality"]),
        "public_or_native_visibility_advantage": bool(
            expected_claims["public_or_native_visibility_advantage"]
        ),
    }
    if ledger.get("claim_boundary") != expected_boundary:
        failures.append("ledger claim boundary disagrees with verified gates")
    rows = foundation_payload.get("rows")
    if not isinstance(rows, list):
        failures.append("foundation rows are missing")
        rows = []
    accepted_ids = {
        record.get("evidence_id")
        for record in records
        if isinstance(record, Mapping) and record.get("status") == "accepted"
    }
    if any(row.get("evidence_id") not in accepted_ids for row in rows if isinstance(row, Mapping)):
        failures.append("foundation table contains rejected evidence")
    visibility_record = next(
        (
            record
            for record in records
            if isinstance(record, Mapping)
            and record.get("evidence_id") == "synthetic_visibility_g0_g3"
        ),
        None,
    )
    if (
        isinstance(visibility_record, Mapping)
        and visibility_record.get("status") == "accepted"
    ):
        bundled_visibility = {
            dependency.get("bundle_path")
            for dependency in visibility_record.get("dependencies", ())
            if isinstance(dependency, Mapping)
        }
        if bundled_visibility != set(EXPECTED_VISIBILITY_FIGURES):
            failures.append("accepted G3 figure set is incomplete")
    adaptive_material_record = next(
        (
            record
            for record in records
            if isinstance(record, Mapping)
            and record.get("evidence_id")
            == "adaptive_m3_m5_basis_selection"
        ),
        None,
    )
    if (
        isinstance(adaptive_material_record, Mapping)
        and adaptive_material_record.get("status") == "accepted"
    ):
        bundled_adaptive_assets = {
            dependency.get("bundle_path")
            for dependency in adaptive_material_record.get(
                "dependencies", ()
            )
            if isinstance(dependency, Mapping)
        }
        if bundled_adaptive_assets != set(EXPECTED_ADAPTIVE_MATERIAL_ASSETS):
            failures.append(
                "accepted adaptive-material asset set is incomplete"
            )
    measured_asset_specs = (
        (
            "G4",
            "g4_public_quality",
            "g4_public_quality_placeholder.svg",
            lambda source: g4_assets.build_assets(
                source, g4_assets.DEFAULT_CONFIG
            ),
        ),
        (
            "G6",
            "g6_native_memory",
            "g6_native_memory_placeholder.svg",
            lambda source: g6_assets.build_assets(
                source,
                g6_assets.DEFAULT_CONFIG,
                g6_assets.DEFAULT_CONTRACT,
            ),
        ),
    )
    records_by_id = {
        str(record.get("evidence_id")): record
        for record in records
        if isinstance(record, Mapping)
    }
    for gate, evidence_id, placeholder_name, asset_builder in measured_asset_specs:
        record = records_by_id.get(evidence_id, {})
        accepted = record.get("status") == "accepted"
        placeholder = directory / placeholder_name
        if accepted:
            if placeholder.exists():
                failures.append(f"{gate} accepted bundle retains a placeholder")
            source = Path(str(record.get("path", "")))
            if not source.is_absolute():
                source = ROOT / source
            try:
                expected_assets = asset_builder(source)
            except Exception as error:
                failures.append(
                    f"{gate} measured assets could not be independently rebuilt: "
                    f"{type(error).__name__}: {error}"
                )
                expected_assets = {}
            for name, expected_bytes in expected_assets.items():
                candidate = directory / name
                if not candidate.is_file() or candidate.read_bytes() != expected_bytes:
                    failures.append(
                        f"{gate} measured asset is missing or nondeterministic: {name}"
                    )
        else:
            if not placeholder.is_file():
                failures.append(f"{gate} SVG placeholder is missing")
            elif b"NOT MEASURED" not in placeholder.read_bytes():
                failures.append(f"{gate} SVG is not an explicit placeholder")

    content_contract = {
        "complete": expected_complete,
        "ledger_sha256": _canonical_json_sha256(ledger),
        "gate_status_sha256": _canonical_json_sha256(gate_status),
        "foundation_rows_sha256": _canonical_json_sha256(rows),
        "files": file_rows,
    }
    if manifest.get("content_sha256") != _canonical_json_sha256(content_contract):
        failures.append("manifest content digest is inconsistent")

    # A self-consistent manifest is not enough: an attacker could edit a
    # generated table, re-hash it, and rebind the outer content digest.  Reopen
    # every retained evidence path with the current independent validators,
    # rebuild the complete generator output in memory, and require byte-for-
    # byte identity for every generator-owned file and the manifest itself.
    templates = default_specs()
    template_ids = [spec.evidence_id for spec in templates]
    record_ids = [
        str(record.get("evidence_id"))
        for record in records
        if isinstance(record, Mapping)
    ]
    if (
        len(record_ids) != len(set(record_ids))
        or set(record_ids) != set(template_ids)
    ):
        failures.append("ledger evidence identity set changed")
    else:
        records_by_id = {
            str(record["evidence_id"]): record
            for record in records
            if isinstance(record, Mapping)
        }
        rebuilt_specs: list[EvidenceSpec] = []
        for template in templates:
            retained_path = records_by_id[template.evidence_id].get("path")
            if not isinstance(retained_path, str):
                failures.append(
                    f"ledger input path is invalid: {template.evidence_id}"
                )
                continue
            resolved = Path(retained_path)
            if not resolved.is_absolute():
                resolved = ROOT / resolved
            rebuilt_specs.append(replace(template, path=resolved))
        if len(rebuilt_specs) == len(templates):
            try:
                rebuilt = build_bundle(tuple(rebuilt_specs))
                rebuilt_manifest = _manifest(rebuilt)
            except Exception as error:
                failures.append(
                    "bundle could not be independently regenerated: "
                    f"{type(error).__name__}: {error}"
                )
            else:
                if ledger != rebuilt.ledger:
                    failures.append("evidence ledger differs from regenerated evidence")
                if gate_status != rebuilt.gate_status:
                    failures.append("gate status differs from regenerated evidence")
                if rows != rebuilt.foundation_rows:
                    failures.append("foundation rows differ from regenerated evidence")
                for name, expected_bytes in rebuilt.files.items():
                    candidate = directory / name
                    if (
                        not candidate.is_file()
                        or candidate.read_bytes() != expected_bytes
                    ):
                        failures.append(
                            "generator-owned file differs from independent rebuild: "
                            f"{name}"
                        )
                if manifest != rebuilt_manifest:
                    failures.append(
                        "manifest differs from independently regenerated manifest"
                    )
    return sorted(set(failures))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or verify the fail-closed WorldFoam Paper-B foundation bundle."
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--material-parity", type=Path, default=DEFAULT_MATERIAL_PARITY)
    parser.add_argument("--material-fit", type=Path, default=DEFAULT_MATERIAL_FIT)
    parser.add_argument(
        "--adaptive-material",
        type=Path,
        default=DEFAULT_ADAPTIVE_MATERIAL,
    )
    parser.add_argument("--compiled-lie", type=Path, default=DEFAULT_COMPILED_LIE)
    parser.add_argument("--constant-transfer", type=Path, default=DEFAULT_CONSTANT_TRANSFER)
    parser.add_argument(
        "--synthetic-visibility",
        type=Path,
        default=DEFAULT_SYNTHETIC_VISIBILITY,
    )
    parser.add_argument(
        "--g4-public-quality",
        type=Path,
        default=DEFAULT_G4_PUBLIC_QUALITY,
    )
    parser.add_argument(
        "--g6-native-memory",
        type=Path,
        default=DEFAULT_G6_NATIVE_MEMORY,
    )
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--verify-dir", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.verify_dir is not None:
        failures = verify_bundle_dir(args.verify_dir)
        print(
            json.dumps(
                {
                    "status": "ok" if not failures else "failed",
                    "bundle": _display_path(args.verify_dir),
                    "failures": failures,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if not failures else 1
    specs = default_specs(
        material_parity=args.material_parity,
        material_fit=args.material_fit,
        adaptive_material=args.adaptive_material,
        compiled_lie=args.compiled_lie,
        constant_transfer=args.constant_transfer,
        synthetic_visibility=args.synthetic_visibility,
        g4_public_quality=args.g4_public_quality,
        g6_native_memory=args.g6_native_memory,
    )
    bundle = build_bundle(specs)
    manifest = write_bundle(bundle, args.out_dir)
    result = {
        "status": "complete" if bundle.complete else "incomplete",
        "bundle": _display_path(args.out_dir),
        "accepted_input_count": bundle.ledger["accepted_count"],
        "rejected_input_count": bundle.ledger["rejected_count"],
        "missing_input_count": bundle.ledger["missing_count"],
        "foundation_row_count": len(bundle.foundation_rows),
        "manifest_content_sha256": manifest["content_sha256"],
        "native_memory_fit_claimed": bundle.gate_status["claims"][
            "native_memory_fit"
        ],
        "public_quality_claimed": bundle.gate_status["claims"]["public_quality"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if bundle.complete or args.allow_incomplete else 1


if __name__ == "__main__":
    raise SystemExit(main())
