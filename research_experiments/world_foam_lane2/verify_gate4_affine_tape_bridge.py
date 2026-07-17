#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "research_experiments" / "world_foam_lane2" / "results"
DEFAULT_ARTIFACT = (
    RESULTS_DIR
    / "2026-05-18_gate4_affine_tape_object_mps_vjp_render32_site12_2_4_8_16_scopefix.json"
)
OWNERUPDATE_ACCEPTANCE_KEYS = (
    "ownerupdate_matches_explicit_realray",
    "mixed_vjp_direct_grad_only_ownerupdate_matches_reduce_grad",
    "mixed_vjp_direct_grad_only_ownerupdate_gradients_finite",
)


def _parse_int_list(value: str) -> tuple[int, ...]:
    out = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not out:
        raise ValueError("expected at least one frame count")
    return out


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _finite(value: Any) -> bool:
    return isinstance(value, (float, int)) and math.isfinite(float(value))


def _positive_finite(value: Any) -> bool:
    return _finite(value) and float(value) > 0.0


def _nested_bool(payload: dict[str, Any], *keys: str) -> bool | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current if isinstance(current, bool) else None


def _nested_number(payload: dict[str, Any], *keys: str) -> float | None:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return float(current) if _finite(current) else None


def _row_number(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    return float(value) if _finite(value) else None


def _validate_ownerupdate_scope(payload: dict[str, Any], failures: list[str]) -> dict[str, Any]:
    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict):
        failures.append("missing acceptance map")
        return {"include_ownerupdate": payload.get("include_ownerupdate")}

    include_ownerupdate = payload.get("include_ownerupdate") is True
    include_vjp = payload.get("include_vjp") is True
    owner_diag = payload.get("ownerupdate_diagnostics")
    owner_vjp_diag = payload.get("mixed_vjp_direct_grad_only_ownerupdate_diagnostics")
    owner_checked = owner_diag.get("checked") if isinstance(owner_diag, dict) else None
    owner_vjp_checked = owner_vjp_diag.get("checked") if isinstance(owner_vjp_diag, dict) else None

    if include_ownerupdate:
        if owner_checked is not True:
            failures.append("ownerupdate_diagnostics.checked must be true when include_ownerupdate is true")
        if acceptance.get("ownerupdate_matches_explicit_realray") is not True:
            failures.append("ownerupdate forward acceptance must be explicit when include_ownerupdate is true")
        if include_vjp:
            if owner_vjp_checked is not True:
                failures.append("ownerupdate VJP diagnostics must be checked when include_ownerupdate and include_vjp are true")
            for key in OWNERUPDATE_ACCEPTANCE_KEYS[1:]:
                if acceptance.get(key) is not True:
                    failures.append(f"missing true ownerupdate VJP acceptance key {key}")
    else:
        for key in OWNERUPDATE_ACCEPTANCE_KEYS:
            if key in acceptance:
                failures.append(f"{key} must not appear when include_ownerupdate is false")
        if owner_checked is not False:
            failures.append("ownerupdate_diagnostics.checked must be false when include_ownerupdate is false")
        if owner_vjp_checked is not False:
            failures.append("ownerupdate VJP diagnostics checked must be false when include_ownerupdate is false")
        if isinstance(owner_diag, dict) and owner_diag.get("within_strict_tolerance") is not None:
            failures.append("ownerupdate forward tolerance must be null when ownerupdate is not checked")
        if isinstance(owner_vjp_diag, dict) and owner_vjp_diag.get("within_grad_tolerance") is not None:
            failures.append("ownerupdate VJP tolerance must be null when ownerupdate is not checked")

    return {
        "include_ownerupdate": include_ownerupdate,
        "include_vjp": include_vjp,
        "ownerupdate_checked": owner_checked,
        "ownerupdate_vjp_checked": owner_vjp_checked,
    }


def verify(args: argparse.Namespace) -> dict[str, Any]:
    failures: list[str] = []
    path = Path(args.artifact)
    try:
        payload = _load_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "failed", "artifact": str(path), "failures": [f"could not load artifact: {exc}"]}

    expected_frames = _parse_int_list(str(args.frame_counts))
    if payload.get("benchmark") != "world_foam_lane2_fused_slab_affine_realray_mps_smoke":
        failures.append(f"unexpected benchmark {payload.get('benchmark')!r}")
    if payload.get("status") != "ok":
        failures.append(f"artifact status is {payload.get('status')!r}")
    if payload.get("quality_claim") is not False or payload.get("training_claim") is not False:
        failures.append("Gate4 bridge artifact must keep quality_claim=false and training_claim=false")
    if payload.get("include_vjp") is not True:
        failures.append("Gate4 bridge verifier requires include_vjp=true")
    if args.require_vjp_seed_mode is not None and payload.get("vjp_seed_mode") != args.require_vjp_seed_mode:
        failures.append(
            f"vjp_seed_mode {payload.get('vjp_seed_mode')!r} did not match "
            f"required {args.require_vjp_seed_mode!r}"
        )
    if payload.get("layout") != "per-track":
        failures.append("Gate4 bridge verifier requires per-track layout")
    if payload.get("candidate_order") != "slab-mid-depth":
        failures.append("Gate4 bridge verifier requires slab-mid-depth candidate ordering")
    if tuple(payload.get("frame_counts", ())) != expected_frames:
        failures.append(f"frame_counts {payload.get('frame_counts')} did not match required {list(expected_frames)}")
    if payload.get("render_size") != args.render_size:
        failures.append(f"render_size {payload.get('render_size')} did not match {args.render_size}")

    acceptance = payload.get("acceptance")
    if not isinstance(acceptance, dict):
        failures.append("missing acceptance map")
    else:
        for key, value in sorted(acceptance.items()):
            if value is not True:
                failures.append(f"acceptance {key} is not true")

    ownerupdate_scope = _validate_ownerupdate_scope(payload, failures)
    if args.require_ownerupdate and ownerupdate_scope.get("include_ownerupdate") is not True:
        failures.append("ownerupdate must be included when --require-ownerupdate is set")

    rows = payload.get("rows")
    rows_by_frame: dict[int, dict[str, Any]] = {}
    if not isinstance(rows, list):
        failures.append("rows must be a list")
    else:
        for row in rows:
            if not isinstance(row, dict):
                failures.append("row is not an object")
                continue
            frame = row.get("frames")
            if not isinstance(frame, int):
                failures.append("row missing integer frames")
                continue
            rows_by_frame[frame] = row
    found_frames = tuple(sorted(rows_by_frame))
    if found_frames != expected_frames:
        failures.append(f"row frames {found_frames} did not match required {expected_frames}")

    time_slabs = payload.get("time_slabs")
    if not isinstance(time_slabs, int) or time_slabs <= 0:
        failures.append("time_slabs must be a positive integer")
        time_slabs = 1

    mixed_storage_by_frame: dict[str, float] = {}
    explicit_storage_by_frame: dict[str, float] = {}
    boundary_ratio_by_frame: dict[str, float] = {}
    mixed_storage_bytes: list[float] = []
    explicit_storage_bytes: list[float] = []
    for frame in expected_frames:
        row = rows_by_frame.get(frame)
        if row is None:
            continue
        if row.get("render_size") != args.render_size:
            failures.append(f"frame {frame}: render_size {row.get('render_size')} did not match {args.render_size}")
        if row.get("site_count") != args.site_count:
            failures.append(f"frame {frame}: site_count {row.get('site_count')} did not match {args.site_count}")
        if row.get("missing_sample_events") != 0:
            failures.append(f"frame {frame}: missing_sample_events must be zero")
        max_candidates = row.get("max_candidates_per_row")
        max_realray_boundaries = payload.get("max_realray_boundaries")
        if isinstance(max_candidates, int) and isinstance(max_realray_boundaries, int):
            if max_candidates > max_realray_boundaries:
                failures.append(f"frame {frame}: max candidates exceed Metal cap")
        else:
            failures.append(f"frame {frame}: missing max candidate/cap values")

        linear_fit = row.get("linear_fit")
        if not isinstance(linear_fit, dict):
            failures.append(f"frame {frame}: missing linear_fit")
        else:
            for key in ("max_origin_residual", "max_direction_residual"):
                value = linear_fit.get(key)
                if not _finite(value) or abs(float(value)) > args.max_affine_residual:
                    failures.append(f"frame {frame}: {key} exceeds {args.max_affine_residual}")

        ratio = _row_number(row, "compiled_boundary_test_ratio")
        expected_ratio = float(time_slabs) / float(frame)
        if ratio is None or abs(ratio - expected_ratio) > args.boundary_ratio_tolerance:
            failures.append(
                f"frame {frame}: compiled boundary ratio {ratio} did not match {expected_ratio:.6g}"
            )
        else:
            boundary_ratio_by_frame[str(frame)] = ratio

        mixed_storage = _row_number(row, "total_mixed_fused_storage_bytes")
        explicit_storage = _row_number(row, "explicit_ray_storage_bytes")
        if mixed_storage is None or mixed_storage <= 0.0:
            failures.append(f"frame {frame}: total_mixed_fused_storage_bytes must be positive")
        else:
            mixed_storage_by_frame[str(frame)] = mixed_storage
            mixed_storage_bytes.append(mixed_storage)
        if explicit_storage is None or explicit_storage <= 0.0:
            failures.append(f"frame {frame}: explicit_ray_storage_bytes must be positive")
        else:
            explicit_storage_by_frame[str(frame)] = explicit_storage
            explicit_storage_bytes.append(explicit_storage)

    mixed_storage_scale = (
        mixed_storage_bytes[-1] / mixed_storage_bytes[0]
        if len(mixed_storage_bytes) == len(expected_frames)
        else float("inf")
    )
    explicit_storage_scale = (
        explicit_storage_bytes[-1] / explicit_storage_bytes[0]
        if len(explicit_storage_bytes) == len(expected_frames)
        else float("nan")
    )
    if mixed_storage_scale > args.max_mixed_storage_scale:
        failures.append(
            f"mixed tape storage scale {mixed_storage_scale:.3f} exceeds {args.max_mixed_storage_scale:.3f}"
        )
    expected_explicit_scale = expected_frames[-1] / expected_frames[0]
    if _finite(explicit_storage_scale) and abs(explicit_storage_scale - expected_explicit_scale) > 1.0e-6:
        failures.append(
            f"explicit ray storage scale {explicit_storage_scale:.3f} did not match frame scale {expected_explicit_scale:.3f}"
        )

    mixed_max_error = _nested_number(payload, "mixed_max_error")
    tolerance = _nested_number(payload, "tolerance") or args.max_mixed_error
    if mixed_max_error is None or mixed_max_error > min(float(tolerance), args.max_mixed_error):
        failures.append(f"mixed max error {mixed_max_error} exceeds allowed tolerance")
    for diag_name in (
        "mixed_vjp_direct_diagnostics",
        "mixed_vjp_direct_grad_only_diagnostics",
        "mixed_vjp_direct_track_diagnostics",
    ):
        if _nested_bool(payload, diag_name, "within_grad_tolerance") is not True:
            failures.append(f"{diag_name}.within_grad_tolerance must be true")
    if _nested_bool(payload, "mixed_vjp_direct_rgb_only_diagnostics", "has_expected_seed_behavior") is not True:
        failures.append("RGB-only VJP diagnostic did not show the expected seed behavior")
    if _nested_bool(payload, "autograd_vjp_diagnostics", "general_modes_match_reduce") is not True:
        failures.append("autograd VJP modes do not match reduce")
    if _nested_bool(payload, "autograd_vjp_diagnostics", "rgb_only_has_expected_seed_behavior") is not True:
        failures.append("autograd RGB-only diagnostic did not show the expected seed behavior")

    coeff16 = payload.get("coeff16_diagnostics")
    coeff16_rejected = isinstance(coeff16, dict) and coeff16.get("within_approx_tolerance") is False
    if args.require_coeff16_rejected and not coeff16_rejected:
        failures.append("pure coeff16 path must remain rejected for this gate")

    return {
        "status": "failed" if failures else "ok",
        "artifact": str(path),
        "failures": failures,
        "frame_counts": list(expected_frames),
        "render_size": args.render_size,
        "site_count": args.site_count,
        "gradient_scope": payload.get("gradient_scope"),
        "vjp_seed_mode": payload.get("vjp_seed_mode"),
        "mixed_max_error": mixed_max_error,
        "coeff16_rejected": coeff16_rejected,
        "mixed_storage_scale_first_to_last": mixed_storage_scale,
        "explicit_ray_storage_scale_first_to_last": explicit_storage_scale,
        "boundary_ratio_by_frame": boundary_ratio_by_frame,
        "mixed_storage_by_frame": mixed_storage_by_frame,
        "explicit_storage_by_frame": explicit_storage_by_frame,
        "ownerupdate_scope": ownerupdate_scope,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify Gate4 affine moving-ray slab tape bridge artifacts.")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--frame-counts", default="2,4,8,16")
    parser.add_argument("--render-size", type=int, default=32)
    parser.add_argument("--site-count", type=int, default=12)
    parser.add_argument("--max-mixed-error", type=float, default=5.0e-4)
    parser.add_argument("--max-affine-residual", type=float, default=1.0e-5)
    parser.add_argument("--max-mixed-storage-scale", type=float, default=1.10)
    parser.add_argument("--boundary-ratio-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--require-ownerupdate", action="store_true")
    parser.add_argument("--require-vjp-seed-mode", choices=("rgb", "rgba-depth"), default=None)
    parser.add_argument("--no-require-coeff16-rejected", dest="require_coeff16_rejected", action="store_false")
    parser.set_defaults(require_coeff16_rejected=True)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = verify(args)
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
