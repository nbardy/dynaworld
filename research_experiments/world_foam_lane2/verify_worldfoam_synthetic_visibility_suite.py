"""Independent verifier for the WorldFoam CPU synthetic visibility suite."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Iterable


EXPECTED_SCHEMA_VERSION = 1
EXPECTED_SUITE_ID = "worldfoam-synthetic-visibility-cpu-v1"
EXPECTED_SCENES = (
    "S1_constant_density_sphere",
    "S2_crossing_translucent_slabs",
    "S3_crossing_gaussian_density_sheets",
    "S4_thin_foreground_occluder",
    "S5_dense_semitransparent_cloud",
    "S6_moving_cell_complex",
    "S7_near_camera_large_cell",
    "S8_fast_object_fast_orbit",
)
EXPECTED_CAMERAS = (
    "C1_static",
    "C2_linear_dolly",
    "C3_orbit",
    "C4_fast_orbit",
    "C5_orbit_finite_exposure",
    "C6_rolling_shutter",
    "C7_revolving_near_plane_crossing",
)
EXPECTED_METHODS = (
    "depth_layer_128",
    "representative_depth_sorted",
    "depth_marginal",
)
EXPECTED_FIGURES = (
    "worldfoam_synthetic_depth_convergence.svg",
    "worldfoam_synthetic_adaptive_fallback.svg",
    "worldfoam_synthetic_crossing_flicker.svg",
)
EXPECTED_EXCLUSIONS = {
    "native-kernel speed",
    "native allocator or peak-memory scaling",
    "public-data trained quality",
    "end-to-end kinetic compiler acceptance",
}
NUMERIC_METRICS = (
    "rgb_mse",
    "rgb_psnr_db",
    "rgb_mean_absolute_error",
    "rgb_max_absolute_error",
    "transmittance_mean_absolute_error",
    "transmittance_max_absolute_error",
    "temporal_flicker_error",
    "temporal_gradient_error_variance",
)
SOURCE = Path(__file__).with_name("worldfoam_synthetic_visibility_suite.py")


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _close(left: float, right: float, *, atol: float = 1.0e-12, rtol: float = 1.0e-9) -> bool:
    return math.isclose(float(left), float(right), abs_tol=atol, rel_tol=rtol)


def _finite_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _validate_metric_row(row: dict[str, Any], label: str, errors: list[str]) -> None:
    for key in NUMERIC_METRICS:
        value = row.get(key)
        if not _finite_number(value):
            errors.append(f"{label}.{key} must be finite")
        elif key != "rgb_psnr_db" and float(value) < 0.0:
            errors.append(f"{label}.{key} must be nonnegative")
    mse = row.get("rgb_mse")
    psnr = row.get("rgb_psnr_db")
    if _finite_number(mse) and _finite_number(psnr):
        expected = -10.0 * math.log10(max(float(mse), float.fromhex("0x1.0000000000000p-1022")))
        if not _close(float(psnr), expected, atol=1.0e-10, rtol=1.0e-10):
            errors.append(f"{label}.rgb_psnr_db does not match rgb_mse")


def _unique_rows(
    rows: object,
    keys: tuple[str, ...],
    expected: set[tuple[object, ...]],
    label: str,
    errors: list[str],
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        errors.append(f"{label} must be a list")
        return []
    actual_rows = [row for row in rows if isinstance(row, dict)]
    if len(actual_rows) != len(rows):
        errors.append(f"{label} must contain objects only")
    identities = [tuple(row.get(key) for key in keys) for row in actual_rows]
    if len(identities) != len(set(identities)):
        errors.append(f"{label} contains duplicate identities")
    if set(identities) != expected:
        missing = sorted(expected - set(identities))
        extra = sorted(set(identities) - expected)
        errors.append(f"{label} coverage mismatch: missing={missing[:4]} extra={extra[:4]}")
    return actual_rows


def _aggregate(rows: Iterable[dict[str, Any]], key: str) -> dict[str, float]:
    values = sorted(float(row[key]) for row in rows)
    count = len(values)
    if count == 0:
        raise ValueError("cannot aggregate zero rows")

    def quantile(q: float) -> float:
        position = (count - 1) * q
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return values[lower]
        weight = position - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight

    middle = quantile(0.5)
    return {
        "mean": sum(values) / count,
        "median": middle,
        "p95": quantile(0.95),
        "max": values[-1],
    }


def _verify_aggregate(
    actual: object,
    expected: dict[str, float],
    label: str,
    errors: list[str],
) -> None:
    if not isinstance(actual, dict):
        errors.append(f"{label} must be an object")
        return
    for key, value in expected.items():
        if not _finite_number(actual.get(key)) or not _close(float(actual[key]), value):
            errors.append(f"{label}.{key} mismatch")


def _verify_figures(report: dict[str, Any], report_path: Path, errors: list[str]) -> None:
    manifest = report.get("figure_manifest")
    if not isinstance(manifest, list):
        errors.append("figure_manifest must be a list")
        return
    names = [entry.get("name") for entry in manifest if isinstance(entry, dict)]
    if tuple(sorted(names)) != tuple(sorted(EXPECTED_FIGURES)):
        errors.append("figure_manifest names mismatch")
        return
    for entry in manifest:
        if not isinstance(entry, dict):
            errors.append("figure_manifest entries must be objects")
            continue
        name = entry["name"]
        path = report_path.parent / "figures" / name
        if not path.is_file():
            errors.append(f"missing figure {name}")
            continue
        payload = path.read_bytes()
        if entry.get("sha256") != _sha256(payload):
            errors.append(f"figure hash mismatch for {name}")
        if entry.get("bytes") != len(payload):
            errors.append(f"figure byte count mismatch for {name}")
        try:
            root = ET.fromstring(payload)
        except ET.ParseError as exc:
            errors.append(f"figure {name} is invalid XML: {exc}")
            continue
        if not root.tag.endswith("svg"):
            errors.append(f"figure {name} root is not SVG")
        children = list(root)
        if not any(child.tag.endswith("title") and (child.text or "").strip() for child in children):
            errors.append(f"figure {name} lacks a nonempty title")
        if not any(child.tag.endswith("desc") and (child.text or "").strip() for child in children):
            errors.append(f"figure {name} lacks a nonempty description")


def verify_report(
    report: dict[str, Any],
    *,
    report_path: Path,
    require_accepted: bool = True,
) -> list[str]:
    errors: list[str] = []
    if report.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        errors.append("schema_version mismatch")
    if report.get("suite_id") != EXPECTED_SUITE_ID:
        errors.append("suite_id mismatch")
    settings = report.get("settings")
    if not isinstance(settings, dict):
        errors.append("settings must be an object")
        return errors
    if tuple(settings.get("scenes", ())) != EXPECTED_SCENES:
        errors.append("settings.scenes mismatch")
    if tuple(settings.get("cameras", ())) != EXPECTED_CAMERAS:
        errors.append("settings.cameras mismatch")
    layer_counts = tuple(settings.get("layer_counts", ()))
    if layer_counts != (16, 32, 64, 128):
        errors.append("settings.layer_counts must be [16,32,64,128]")
    if settings.get("dtype") != "float64" or settings.get("device") != "cpu":
        errors.append("suite must be float64 CPU")
    if settings.get("oracle_samples", 0) < 512:
        errors.append("oracle_samples must be at least 512")
    expected_protocol = _sha256(_canonical_json(settings).encode("utf-8"))
    if report.get("protocol_sha256") != expected_protocol:
        errors.append("protocol_sha256 mismatch")
    if not SOURCE.is_file() or report.get("source_sha256") != _sha256(SOURCE.read_bytes()):
        errors.append("source_sha256 does not match current suite source")

    layer_expected = {
        (scene, camera, layer)
        for scene in EXPECTED_SCENES
        for camera in EXPECTED_CAMERAS
        for layer in layer_counts
    }
    baseline_expected = {
        (scene, camera, method)
        for scene in EXPECTED_SCENES
        for camera in EXPECTED_CAMERAS
        for method in EXPECTED_METHODS
    }
    adaptive_expected = {
        (scene, camera) for scene in EXPECTED_SCENES for camera in EXPECTED_CAMERAS
    }
    layer_rows = _unique_rows(
        report.get("layer_rows"),
        ("scene", "camera", "layer_count"),
        layer_expected,
        "layer_rows",
        errors,
    )
    baseline_rows = _unique_rows(
        report.get("baseline_rows"),
        ("scene", "camera", "method"),
        baseline_expected,
        "baseline_rows",
        errors,
    )
    adaptive_rows = _unique_rows(
        report.get("adaptive_rows"),
        ("scene", "camera"),
        adaptive_expected,
        "adaptive_rows",
        errors,
    )
    for index, row in enumerate(layer_rows):
        _validate_metric_row(row, f"layer_rows[{index}]", errors)
    for index, row in enumerate(baseline_rows):
        _validate_metric_row(row, f"baseline_rows[{index}]", errors)
        if not isinstance(row.get("representative_order_flip_count"), int):
            errors.append(f"baseline_rows[{index}].representative_order_flip_count must be int")
    for index, row in enumerate(adaptive_rows):
        _validate_metric_row(row, f"adaptive_rows[{index}]", errors)
        fraction = row.get("fallback_fraction")
        if not _finite_number(fraction) or not 0.0 <= float(fraction) <= 1.0:
            errors.append(f"adaptive_rows[{index}].fallback_fraction must lie in [0,1]")

    analytic = report.get("analytic_constant_sphere")
    gauge = report.get("gauge_jacobian")
    if not isinstance(analytic, dict) or not all(
        _finite_number(analytic.get(key))
        for key in ("rgb_max_absolute_error", "transmittance_max_absolute_error")
    ):
        errors.append("analytic_constant_sphere receipt is incomplete")
        analytic = {}
    if not isinstance(gauge, dict) or not all(
        _finite_number(gauge.get(key))
        for key in (
            "with_physical_jacobian_rgb_max_absolute_error",
            "without_physical_jacobian_rgb_max_absolute_error",
            "error_ratio_without_over_with",
        )
    ):
        errors.append("gauge_jacobian receipt is incomplete")
        gauge = {}

    deepest = [row for row in layer_rows if row.get("layer_count") == 128]
    crossing_worldfoam = [
        row
        for row in baseline_rows
        if row.get("method") == "depth_layer_128"
        and row.get("scene")
        in {"S2_crossing_translucent_slabs", "S3_crossing_gaussian_density_sheets"}
    ]
    crossing_sorted = [
        row
        for row in baseline_rows
        if row.get("method") == "representative_depth_sorted"
        and row.get("scene")
        in {"S2_crossing_translucent_slabs", "S3_crossing_gaussian_density_sheets"}
    ]
    crossing_marginal = [
        row
        for row in baseline_rows
        if row.get("method") == "depth_marginal"
        and row.get("scene")
        in {"S2_crossing_translucent_slabs", "S3_crossing_gaussian_density_sheets"}
    ]
    if deepest and crossing_worldfoam and crossing_sorted and crossing_marginal and analytic and gauge:
        p05 = sorted(float(row["rgb_psnr_db"]) for row in deepest)
        p05_position = (len(p05) - 1) * 0.05
        lo, hi = math.floor(p05_position), math.ceil(p05_position)
        weight = p05_position - lo
        deepest_p05 = p05[lo] * (1.0 - weight) + p05[hi] * weight
        worldfoam_mse = sum(float(row["rgb_mse"]) for row in crossing_worldfoam) / len(
            crossing_worldfoam
        )
        sorted_mse = sum(float(row["rgb_mse"]) for row in crossing_sorted) / len(crossing_sorted)
        marginal_mse = sum(float(row["rgb_mse"]) for row in crossing_marginal) / len(
            crossing_marginal
        )
        recomputed_gates = {
            "analytic_constant_sphere": bool(
                float(analytic["rgb_max_absolute_error"]) <= 6.0e-3
                and float(analytic["transmittance_max_absolute_error"]) <= 7.0e-3
            ),
            "gauge_jacobian": bool(
                float(gauge["with_physical_jacobian_rgb_max_absolute_error"]) <= 8.0e-4
                and float(gauge["without_physical_jacobian_rgb_max_absolute_error"])
                >= 20.0
                * max(float(gauge["with_physical_jacobian_rgb_max_absolute_error"]), 1.0e-12)
            ),
            "deepest_layer_floor": bool(deepest_p05 >= 30.0),
            "crossing_beats_representative_sort": bool(worldfoam_mse <= 0.50 * sorted_mse),
            "crossing_beats_depth_marginal": bool(worldfoam_mse <= 0.50 * marginal_mse),
            "finite_all_rows": bool(
                all(_finite_number(row.get(key)) for row in (*layer_rows, *baseline_rows, *adaptive_rows) for key in NUMERIC_METRICS)
            ),
        }
        if report.get("acceptance_gates") != recomputed_gates:
            errors.append("acceptance_gates do not match independently recomputed gates")
        if report.get("accepted") is not all(recomputed_gates.values()):
            errors.append("accepted does not equal conjunction of acceptance_gates")
        aggregates = report.get("aggregates")
        if not isinstance(aggregates, dict):
            errors.append("aggregates must be an object")
        else:
            _verify_aggregate(
                aggregates.get("deepest_layer_psnr_db"),
                _aggregate(deepest, "rgb_psnr_db"),
                "aggregates.deepest_layer_psnr_db",
                errors,
            )
            _verify_aggregate(
                aggregates.get("deepest_layer_rgb_max_absolute_error"),
                _aggregate(deepest, "rgb_max_absolute_error"),
                "aggregates.deepest_layer_rgb_max_absolute_error",
                errors,
            )
            _verify_aggregate(
                aggregates.get("adaptive_fallback_fraction"),
                _aggregate(adaptive_rows, "fallback_fraction"),
                "aggregates.adaptive_fallback_fraction",
                errors,
            )
            for key, expected in (
                ("crossing_worldfoam_rgb_mse_mean", worldfoam_mse),
                ("crossing_sorted_rgb_mse_mean", sorted_mse),
                ("crossing_depth_marginal_rgb_mse_mean", marginal_mse),
            ):
                if not _finite_number(aggregates.get(key)) or not _close(
                    float(aggregates[key]), expected
                ):
                    errors.append(f"aggregates.{key} mismatch")

    scope = report.get("claim_scope")
    if not isinstance(scope, dict) or not EXPECTED_EXCLUSIONS.issubset(
        set(scope.get("does_not_support", ()))
    ):
        errors.append("claim_scope must retain all paper-evidence exclusions")
    if report.get("timing_is_paper_evidence") is not False:
        errors.append("diagnostic CPU timing must not be marked as paper evidence")
    if require_accepted and report.get("accepted") is not True:
        errors.append("report is structurally valid but has failed acceptance gates")
    _verify_figures(report, report_path, errors)
    return errors


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--allow-failed-gates", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    errors = verify_report(
        report,
        report_path=args.report,
        require_accepted=not args.allow_failed_gates,
    )
    if errors:
        raise SystemExit("\n".join(f"- {error}" for error in errors))
    print(
        _canonical_json(
            {
                "accepted": report["accepted"],
                "report": str(args.report),
                "verified": True,
            }
        )
    )


if __name__ == "__main__":
    main()
