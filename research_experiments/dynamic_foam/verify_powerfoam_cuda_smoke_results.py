from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_OFFICIAL_COMMIT = "96392252ebd0059fe6ca98881b62e12295d9242f"
DYNAMIC_PATCHES = {
    "feature": ROOT / "research_experiments/dynamic_foam/cuda_forks/dynamic_feature_foam.patch",
    "geometry": ROOT / "research_experiments/dynamic_foam/cuda_forks/dynamic_geometry_foam.patch",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_summary(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "summary" in data and isinstance(data["summary"], dict):
        return data["summary"]
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object.")
    return data


def dynamic_patch_kind(summary: dict[str, Any]) -> str:
    kind = summary.get("source", {}).get("dynamic_patch_kind")
    if kind is None:
        kind = summary.get("settings", {}).get("dynamic_patch_kind", "feature")
    return str(kind)


def dynamic_lane_name(kind: str) -> str:
    return "dynamic_geometry_foam_cuda" if kind == "geometry" else "dynamic_feature_foam_cuda"


def source_patch_sha(source: dict[str, Any], kind: str) -> str | None:
    if kind == "feature":
        return source.get("dynamic_feature_patch_sha256") or source.get("dynamic_patch_sha256")
    return source.get("dynamic_geometry_patch_sha256")


def check_summary(
    summary: dict[str, Any],
    *,
    allow_planned: bool,
    require_official_fixture: bool = False,
    require_dynamic_geometry: bool = False,
) -> list[dict[str, Any]]:
    runs = summary.get("runs", [])
    by_name = {str(run.get("name")): run for run in runs if isinstance(run, dict)}
    status = summary.get("status")
    patch_kind = dynamic_patch_kind(summary)
    dynamic_name = "dynamic_feature_foam_cuda"
    dynamic = by_name.get(dynamic_name, {})
    dynamic_metrics = dynamic.get("metrics", {}).get("dynamic", {})
    geometry = by_name.get("dynamic_geometry_foam_cuda", {})
    geometry_metrics = geometry.get("metrics", {}).get("dynamic", {})
    source = summary.get("source", {})
    current_feature_sha = sha256_file(DYNAMIC_PATCHES["feature"]) if DYNAMIC_PATCHES["feature"].exists() else None
    current_geometry_sha = sha256_file(DYNAMIC_PATCHES["geometry"]) if DYNAMIC_PATCHES["geometry"].exists() else None
    geometry_required = require_dynamic_geometry
    checks = [
        {
            "name": "schema_version",
            "passed": summary.get("schema_version") == "powerfoam_cuda_smoke_v1",
            "evidence": summary.get("schema_version"),
        },
        {
            "name": "status",
            "passed": status == "ok" or (allow_planned and status == "planned"),
            "evidence": status,
        },
        {
            "name": "same_clip_settings_present",
            "passed": isinstance(summary.get("clip"), dict)
            and isinstance(summary.get("settings"), dict)
            and summary.get("clip", {}).get("frames") is not None
            and summary.get("settings", {}).get("iterations") is not None,
            "evidence": {"clip": summary.get("clip"), "settings": summary.get("settings")},
        },
        {
            "name": "official_source_pinned",
            "passed": bool(source.get("official_repo_url"))
            and source.get("official_commit") == EXPECTED_OFFICIAL_COMMIT,
            "evidence": source,
        },
        {
            "name": "dynamic_feature_patch_matches_current_file",
            "passed": bool(current_feature_sha) and source_patch_sha(source, "feature") == current_feature_sha,
            "evidence": {
                "summary": source_patch_sha(source, "feature"),
                "current": current_feature_sha,
                "path": str(DYNAMIC_PATCHES["feature"]),
            },
        },
        {
            "name": "dynamic_geometry_patch_matches_current_file",
            "passed": (
                not (geometry_required or source_patch_sha(source, "geometry"))
                or (bool(current_geometry_sha) and source_patch_sha(source, "geometry") == current_geometry_sha)
            ),
            "evidence": {
                "summary": source_patch_sha(source, "geometry"),
                "current": current_geometry_sha,
                "path": str(DYNAMIC_PATCHES["geometry"]),
            },
        },
    ]
    if status == "planned":
        checks.append(
            {
                "name": "planned_commands_present",
                "passed": bool(summary.get("planned_commands")),
                "evidence": summary.get("planned_commands"),
            }
        )
        if require_dynamic_geometry:
            checks.append(
                {
                    "name": "dynamic_geometry_requires_executed_summary",
                    "passed": False,
                    "evidence": "planned summaries cannot prove rendered alpha/support motion",
                }
            )
        if require_official_fixture:
            checks.append(
                {
                    "name": "official_fixture_required",
                    "passed": False,
                    "evidence": "planned summaries do not include a generated official fixture",
                }
            )
        return checks
    checks.extend(
        [
            {
                "name": "cuda_host",
                "passed": bool(summary.get("host", {}).get("torch_cuda_available"))
                and bool(summary.get("host", {}).get("cuda_device_name")),
                "evidence": summary.get("host"),
            },
            {
                "name": "official_fixture_ok_or_skipped",
                "passed": summary.get("official_fixture") is None
                or summary.get("official_fixture", {}).get("status") == "ok",
                "evidence": summary.get("official_fixture"),
            },
            {
                "name": "official_fixture_required",
                "passed": not require_official_fixture
                or (
                    isinstance(summary.get("official_fixture"), dict)
                    and summary.get("official_fixture", {}).get("status") == "ok"
                ),
                "evidence": {
                    "required": require_official_fixture,
                    "official_fixture": summary.get("official_fixture"),
                },
            },
            {
                "name": "official_fixture_commit_matches",
                "passed": summary.get("official_fixture") is None
                or summary.get("official_fixture", {}).get("upstream_powerfoam_commit") == EXPECTED_OFFICIAL_COMMIT,
                "evidence": summary.get("official_fixture"),
            },
            {
                "name": "official_static_cuda_ok",
                "passed": by_name.get("official_static_cuda", {}).get("status") == "ok",
                "evidence": by_name.get("official_static_cuda"),
            },
            {
                "name": "dynamic_feature_foam_cuda_ok",
                "passed": by_name.get("dynamic_feature_foam_cuda", {}).get("status") == "ok",
                "evidence": by_name.get("dynamic_feature_foam_cuda"),
            },
            {
                "name": "comparison_available",
                "passed": bool(summary.get("comparisons", {}).get("available")),
                "evidence": summary.get("comparisons"),
            },
            {
                "name": "dynamic_time_conditioning_active",
                "passed": float(dynamic_metrics.get("camera_time_count", 0.0)) >= 2.0
                and float(dynamic_metrics.get("camera_time_max", 0.0))
                > float(dynamic_metrics.get("camera_time_min", 0.0)),
                "evidence": dynamic_metrics,
            },
            {
                "name": "dynamic_feature_coefficients_moved",
                "passed": float(dynamic_metrics.get("dynamic_texel_sv_rgb_coeff_abs_mean", 0.0)) > 0.0
                and float(dynamic_metrics.get("dynamic_texel_sv_rgb_coeff_abs_max", 0.0)) > 0.0,
                "evidence": dynamic_metrics,
            },
            {
                "name": "dynamic_time_changes_rendered_rgb",
                "passed": float(dynamic_metrics.get("dynamic_time_rgb_delta_mean", 0.0)) > 1.0e-8
                and float(dynamic_metrics.get("dynamic_time_rgb_delta_max", 0.0)) > 1.0e-7,
                "evidence": dynamic_metrics,
            },
            {
                "name": "warm_timing_recorded",
                "passed": all(
                    float(by_name.get(name, {}).get("metrics", {}).get("warm_timing_excluding_step0", {}).get("step_total_ms_mean", 0.0))
                    > 0.0
                    for name in ("official_static_cuda", "dynamic_feature_foam_cuda")
                )
                and (
                    not geometry_required
                    or float(
                        by_name.get("dynamic_geometry_foam_cuda", {})
                        .get("metrics", {})
                        .get("warm_timing_excluding_step0", {})
                        .get("step_total_ms_mean", 0.0)
                    )
                    > 0.0
                )
                and float(summary.get("comparisons", {}).get("static_warm_step_total_ms_mean", 0.0)) > 0.0
                and float(summary.get("comparisons", {}).get("dynamic_warm_step_total_ms_mean", 0.0)) > 0.0,
                "evidence": {
                    "static": by_name.get("official_static_cuda", {}).get("metrics", {}).get(
                        "warm_timing_excluding_step0"
                    ),
                    "dynamic": by_name.get("dynamic_feature_foam_cuda", {}).get("metrics", {}).get(
                        "warm_timing_excluding_step0"
                    ),
                    "geometry": by_name.get("dynamic_geometry_foam_cuda", {}).get("metrics", {}).get(
                        "warm_timing_excluding_step0"
                    ),
                    "comparisons": summary.get("comparisons"),
                },
            },
        ]
    )
    if geometry_required:
        checks.extend(
            [
                {
                    "name": "dynamic_geometry_foam_cuda_ok",
                    "passed": by_name.get("dynamic_geometry_foam_cuda", {}).get("status") == "ok",
                    "evidence": by_name.get("dynamic_geometry_foam_cuda"),
                },
                {
                    "name": "dynamic_geometry_coefficients_present",
                    "passed": any(
                        float(geometry_metrics.get(key, 0.0)) > 0.0
                        for key in (
                            "dynamic_center_coeffs_abs_mean",
                            "dynamic_radius_coeffs_abs_mean",
                            "dynamic_quaternion_coeffs_abs_mean",
                            "dynamic_height_coeffs_abs_mean",
                        )
                    ),
                    "evidence": geometry_metrics,
                },
                {
                    "name": "dynamic_time_changes_scene_points",
                    "passed": (
                        float(geometry_metrics.get("dynamic_time_point_delta_mean", 0.0)) > 1.0e-8
                        and float(geometry_metrics.get("dynamic_time_point_delta_max", 0.0)) > 1.0e-7
                    )
                    or (
                        float(geometry_metrics.get("dynamic_center_delta_mean", 0.0)) > 1.0e-8
                        and float(geometry_metrics.get("dynamic_center_delta_max", 0.0)) > 1.0e-7
                    ),
                    "evidence": geometry_metrics,
                },
                {
                    "name": "dynamic_time_changes_rendered_alpha",
                    "passed": float(geometry_metrics.get("dynamic_time_alpha_delta_mean", 0.0)) > 1.0e-8
                    and float(geometry_metrics.get("dynamic_time_alpha_delta_max", 0.0)) > 1.0e-7,
                    "evidence": geometry_metrics,
                },
                {
                    "name": "dynamic_time_changes_alpha_support",
                    "passed": (
                        float(geometry_metrics.get("dynamic_time_alpha_support_delta_fraction", 0.0)) > 0.0
                        or float(geometry_metrics.get("same_camera_support_delta_mean", 0.0)) > 0.0
                    )
                    and float(geometry_metrics.get("dynamic_time_alpha_support_pixels_0", 0.0)) > 0.0
                    and float(geometry_metrics.get("dynamic_time_alpha_support_pixels_1", 0.0)) > 0.0,
                    "evidence": geometry_metrics,
                },
            ]
        )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate saved PowerFoam CUDA smoke summary JSON.")
    parser.add_argument("summary", type=Path)
    parser.add_argument("--allow-planned", action="store_true")
    parser.add_argument(
        "--require-official-fixture",
        action="store_true",
        help="Fail if the CUDA smoke skipped the official CUDA/Warp parity fixture.",
    )
    parser.add_argument(
        "--require-dynamic-geometry",
        action="store_true",
        help="Fail unless the summary proves time-conditioned scene geometry changes rendered alpha/support.",
    )
    args = parser.parse_args()
    summary = load_summary(args.summary)
    checks = check_summary(
        summary,
        allow_planned=bool(args.allow_planned),
        require_official_fixture=bool(args.require_official_fixture),
        require_dynamic_geometry=bool(args.require_dynamic_geometry),
    )
    report = {"ok": all(bool(check["passed"]) for check in checks), "checks": checks}
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
