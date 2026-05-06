from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_summary(path: Path) -> dict[str, Any]:
    summary_path = path / "dynamic_geometry_summary.json" if path.is_dir() else path
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{summary_path} must contain a JSON object.")
    return payload


def metric(summary: dict[str, Any], name: str) -> float:
    motion = summary.get("motion_vs_repaint", {})
    final = summary.get("final_eval", {})
    if name in motion:
        return float(motion.get(name, 0.0))
    return float(final.get(name, 0.0))


def check_summary(
    summary: dict[str, Any],
    *,
    require_geometry_motion: bool,
    require_alpha_support_motion: bool,
    require_appearance_freeze_control: bool,
    min_screen_delta_px: float,
    min_alpha_delta: float,
    min_support_delta: float,
    max_feature_delta: float,
) -> list[dict[str, Any]]:
    config = summary.get("config", {})
    checks = [
        {
            "name": "schema_version",
            "passed": summary.get("schema_version") == "dynamic_powerfoam_geometry_summary_v1",
            "evidence": summary.get("schema_version"),
        },
        {
            "name": "status",
            "passed": summary.get("status") == "ok",
            "evidence": summary.get("status"),
        },
        {
            "name": "geometry_controls_enabled",
            "passed": bool(config.get("dynamic_centers")) or bool(config.get("dynamic_radii")),
            "evidence": config,
        },
        {
            "name": "final_eval_present",
            "passed": isinstance(summary.get("final_eval"), dict) and bool(summary.get("final_eval")),
            "evidence": summary.get("final_eval"),
        },
    ]
    if require_geometry_motion:
        checks.append(
            {
                "name": "temporal_screen_motion",
                "passed": metric(summary, "state_mean_temporal_screen_delta_px") > min_screen_delta_px
                and metric(summary, "state_p95_temporal_screen_delta_px") > min_screen_delta_px,
                "evidence": summary.get("motion_vs_repaint"),
            }
        )
    if require_alpha_support_motion:
        checks.extend(
            [
                {
                    "name": "temporal_alpha_motion",
                    "passed": metric(summary, "eval_mean_temporal_alpha_delta") > min_alpha_delta,
                    "evidence": summary.get("motion_vs_repaint"),
                },
                {
                    "name": "temporal_support_motion",
                    "passed": metric(summary, "eval_mean_temporal_support_delta") > min_support_delta,
                    "evidence": summary.get("motion_vs_repaint"),
                },
            ]
        )
    if require_appearance_freeze_control:
        checks.extend(
            [
                {
                    "name": "dynamic_features_disabled",
                    "passed": not bool(config.get("dynamic_features")),
                    "evidence": config,
                },
                {
                    "name": "temporal_feature_delta_frozen",
                    "passed": abs(metric(summary, "state_mean_temporal_feature_abs_delta")) <= max_feature_delta,
                    "evidence": summary.get("motion_vs_repaint"),
                },
            ]
        )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a dynamic PowerFoam Metal geometry run summary.")
    parser.add_argument("summary_or_run_dir", type=Path)
    parser.add_argument("--require-geometry-motion", action="store_true")
    parser.add_argument("--require-alpha-support-motion", action="store_true")
    parser.add_argument("--require-appearance-freeze-control", action="store_true")
    parser.add_argument("--min-screen-delta-px", type=float, default=1.0e-5)
    parser.add_argument("--min-alpha-delta", type=float, default=1.0e-6)
    parser.add_argument("--min-support-delta", type=float, default=0.0)
    parser.add_argument("--max-feature-delta", type=float, default=1.0e-8)
    args = parser.parse_args()
    checks = check_summary(
        load_summary(args.summary_or_run_dir),
        require_geometry_motion=bool(args.require_geometry_motion),
        require_alpha_support_motion=bool(args.require_alpha_support_motion),
        require_appearance_freeze_control=bool(args.require_appearance_freeze_control),
        min_screen_delta_px=float(args.min_screen_delta_px),
        min_alpha_delta=float(args.min_alpha_delta),
        min_support_delta=float(args.min_support_delta),
        max_feature_delta=float(args.max_feature_delta),
    )
    report = {"ok": all(bool(check["passed"]) for check in checks), "checks": checks}
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
