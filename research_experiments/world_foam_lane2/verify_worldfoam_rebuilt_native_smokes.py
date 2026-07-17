#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "research_experiments" / "world_foam_lane2" / "results"

REQUIRED_ARTIFACTS: tuple[dict[str, Any], ...] = (
    {
        "label": "direct_power_boundary",
        "path": RESULTS_DIR / "2026-05-21_worldfoam_rebuilt_direct_power_boundary_mps_smoke.json",
        "benchmark": "world_foam_lane2_mps_power_boundary_smoke",
        "min_rows": 2,
        "row_true_keys": ("matches_cpu_fixture",),
    },
    {
        "label": "csr_power_boundary",
        "path": RESULTS_DIR / "2026-05-21_worldfoam_rebuilt_csr_power_boundary_mps_smoke.json",
        "benchmark": "world_foam_lane2_mps_power_boundary_smoke",
        "min_rows": 2,
        "row_true_keys": ("matches_cpu_fixture",),
    },
    {
        "label": "slab_power_boundary",
        "path": RESULTS_DIR / "2026-05-21_worldfoam_rebuilt_slab_power_boundary_mps_smoke.json",
        "benchmark": "world_foam_lane2_mps_power_boundary_smoke",
        "min_rows": 2,
        "row_true_keys": ("matches_cpu_fixture",),
    },
    {
        "label": "direct_shared_realray",
        "path": RESULTS_DIR / "2026-05-21_worldfoam_rebuilt_direct_shared_realray_replay_mps_smoke.json",
        "benchmark": "world_foam_lane2_gate2b_mps_shared_realray_forward_smoke",
        "acceptance_true": ("shared_outputs_match_direct_cpu", "shared_scan_ratio_sublinear"),
    },
    {
        "label": "csr_affine_realray",
        "path": RESULTS_DIR / "2026-05-21_worldfoam_rebuilt_csr_affine_realray_mps_smoke.json",
        "benchmark": "world_foam_lane2_fused_csr_moving_ray_mps_smoke",
        "acceptance_true": ("matches_cpu_reference", "matches_direct_mps", "moving_ray_tracks_present"),
    },
    {
        "label": "slab_affine_vjp_no_ownerupdate",
        "path": RESULTS_DIR / "2026-05-21_worldfoam_rebuilt_slab_affine_realray_vjp_no_ownerupdate_mps_smoke.json",
        "benchmark": "world_foam_lane2_fused_slab_affine_realray_mps_smoke",
        "layout": "tiled",
        "include_ownerupdate": False,
        "include_vjp": True,
        "ownerupdate_checked": False,
    },
    {
        "label": "slab_affine_vjp_ownerupdate_pertrack",
        "path": RESULTS_DIR / "2026-05-21_worldfoam_rebuilt_slab_affine_realray_vjp_ownerupdate_pertrack_mps_smoke.json",
        "benchmark": "world_foam_lane2_fused_slab_affine_realray_mps_smoke",
        "layout": "per-track",
        "include_ownerupdate": True,
        "include_vjp": True,
        "ownerupdate_checked": True,
    },
)

KNOWN_INVALID_ARTIFACT = (
    RESULTS_DIR / "2026-05-21_worldfoam_rebuilt_slab_affine_realray_vjp_mps_smoke.json"
)


def _load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, str(exc)
    if not isinstance(payload, dict):
        return None, "JSON root is not an object"
    return payload, None


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _check_required_artifact(spec: dict[str, Any]) -> dict[str, Any]:
    path = Path(spec["path"])
    failures: list[str] = []
    payload, load_error = _load_json(path)
    if payload is None:
        return {
            "label": spec["label"],
            "path": str(path),
            "status": "failed",
            "failures": [f"could not load artifact: {load_error}"],
        }

    if payload.get("status") != "ok":
        failures.append(f"status is {payload.get('status')!r}, expected 'ok'")
    if payload.get("benchmark") != spec["benchmark"]:
        failures.append(f"benchmark is {payload.get('benchmark')!r}, expected {spec['benchmark']!r}")
    for claim_key in ("quality_claim", "training_claim"):
        if claim_key in payload and payload.get(claim_key) is not False:
            failures.append(f"{claim_key} must be false")

    rows = payload.get("rows")
    if "min_rows" in spec:
        if not isinstance(rows, list) or len(rows) < int(spec["min_rows"]):
            failures.append(f"rows must contain at least {spec['min_rows']} rows")
        elif "row_true_keys" in spec:
            for index, row in enumerate(rows):
                if not isinstance(row, dict):
                    failures.append(f"row {index} is not an object")
                    continue
                for key in spec["row_true_keys"]:
                    if row.get(key) is not True:
                        failures.append(f"row {index}: {key} must be true")

    acceptance = payload.get("acceptance")
    if isinstance(acceptance, dict):
        false_keys = [key for key, value in sorted(acceptance.items()) if value is not True]
        if false_keys:
            failures.append(f"acceptance keys are not true: {false_keys}")
        for key in spec.get("acceptance_true", ()):
            if acceptance.get(key) is not True:
                failures.append(f"acceptance {key} must be true")
    elif "acceptance_true" in spec:
        failures.append("missing acceptance map")

    for key in ("layout", "include_ownerupdate", "include_vjp"):
        if key in spec and payload.get(key) != spec[key]:
            failures.append(f"{key} is {payload.get(key)!r}, expected {spec[key]!r}")

    if "ownerupdate_checked" in spec:
        owner_diag = payload.get("ownerupdate_diagnostics")
        checked = owner_diag.get("checked") if isinstance(owner_diag, dict) else None
        if checked is not spec["ownerupdate_checked"]:
            failures.append(f"ownerupdate_diagnostics.checked is {checked!r}, expected {spec['ownerupdate_checked']!r}")
        if spec["ownerupdate_checked"]:
            if not isinstance(owner_diag, dict) or owner_diag.get("within_strict_tolerance") is not True:
                failures.append("ownerupdate_diagnostics.within_strict_tolerance must be true")
            if not isinstance(owner_diag, dict) or not _finite_number(owner_diag.get("max_error")):
                failures.append("ownerupdate_diagnostics.max_error must be finite")
            owner_vjp = payload.get("mixed_vjp_direct_grad_only_ownerupdate_diagnostics")
            if not isinstance(owner_vjp, dict) or owner_vjp.get("checked") is not True:
                failures.append("ownerupdate VJP diagnostics must be checked")
            elif owner_vjp.get("within_grad_tolerance") is not True:
                failures.append("ownerupdate VJP within_grad_tolerance must be true")

    return {
        "label": spec["label"],
        "path": str(path),
        "status": "ok" if not failures else "failed",
        "failures": failures,
        "benchmark": payload.get("benchmark"),
        "artifact_status": payload.get("status"),
    }


def _classify_known_invalid(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "present": False,
            "status": "ok",
            "classification": "absent",
            "failures": [],
        }
    payload, load_error = _load_json(path)
    if payload is None:
        return {
            "path": str(path),
            "present": True,
            "status": "failed",
            "classification": "unreadable",
            "failures": [f"could not load artifact: {load_error}"],
        }
    acceptance = payload.get("acceptance")
    owner_diag = payload.get("ownerupdate_diagnostics")
    is_expected_invalid = (
        payload.get("status") == "failed"
        and payload.get("benchmark") == "world_foam_lane2_fused_slab_affine_realray_mps_smoke"
        and payload.get("layout") == "tiled"
        and payload.get("include_ownerupdate") is True
        and isinstance(owner_diag, dict)
        and owner_diag.get("max_error") is None
        and isinstance(acceptance, dict)
        and acceptance.get("ownerupdate_matches_explicit_realray") is False
    )
    return {
        "path": str(path),
        "present": True,
        "status": "ok" if is_expected_invalid else "failed",
        "classification": "expected_invalid_tiled_ownerupdate" if is_expected_invalid else "unexpected_shape",
        "failures": [] if is_expected_invalid else ["known invalid artifact no longer has the expected failed shape"],
    }


def verify(
    specs: tuple[dict[str, Any], ...] = REQUIRED_ARTIFACTS,
    known_invalid_artifact: Path = KNOWN_INVALID_ARTIFACT,
) -> dict[str, Any]:
    rows = [_check_required_artifact(spec) for spec in specs]
    invalid = _classify_known_invalid(known_invalid_artifact)
    failures = [
        f"{row['label']}: {failure}"
        for row in rows
        for failure in row.get("failures", [])
    ]
    failures.extend(f"known_invalid: {failure}" for failure in invalid.get("failures", []))
    return {
        "status": "ok" if not failures else "failed",
        "required_count": len(rows),
        "required": rows,
        "known_invalid_tiled_ownerupdate": invalid,
        "failures": failures,
        "quality_claim": False,
        "speed_claim": False,
        "scope": "rebuilt_native_variant_smoke_artifacts_only",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify the rebuilt WorldFoam native smoke artifact bundle.")
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = verify()
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
