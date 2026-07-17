from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .report_artifacts import load_report_json, write_report_json
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import load_report_json, write_report_json


def load_summary(path: Path) -> dict[str, Any]:
    summary_path = path / "dynamic_geometry_summary.json" if path.is_dir() else path
    return load_report_json(summary_path)


def load_frame_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    metric_path = summary.get("artifacts", {}).get("final_per_frame_metrics")
    if not metric_path:
        raise KeyError("summary missing artifacts.final_per_frame_metrics")
    return load_report_json(Path(metric_path))


def metric(summary: dict[str, Any], name: str) -> float:
    motion = summary.get("motion_vs_repaint", {})
    final = summary.get("final_eval", {})
    if name in motion:
        return float(motion.get(name, 0.0))
    return float(final.get(name, 0.0))


def lane_report(summary: dict[str, Any]) -> dict[str, Any]:
    frame_metrics = load_frame_metrics(summary)
    final = summary.get("final_eval", {})
    motion = summary.get("motion_vs_repaint", {})
    return {
        "output_dir": summary.get("output_dir"),
        "config": summary.get("config", {}),
        "final_eval": {
            "eval_l1": final.get("eval_l1"),
            "eval_mse": final.get("eval_mse"),
            "eval_frame_psnr_mean": final.get("eval_frame_psnr_mean"),
            "eval_frame_psnr_min": final.get("eval_frame_psnr_min"),
            "eval_frame_snr_mean": final.get("eval_frame_snr_mean"),
            "eval_frame_snr_min": final.get("eval_frame_snr_min"),
        },
        "motion_vs_repaint": motion,
        "per_frame_metrics": frame_metrics,
    }


def check(condition: bool, name: str, evidence: Any) -> dict[str, Any]:
    return {"name": name, "passed": bool(condition), "evidence": evidence}


def compare(
    geometry_summary: dict[str, Any],
    color_only_summary: dict[str, Any],
    *,
    geometry_motion_threshold_px: float,
    frozen_motion_threshold_px: float,
) -> dict[str, Any]:
    geometry = lane_report(geometry_summary)
    color_only = lane_report(color_only_summary)
    geometry_final = geometry["final_eval"]
    color_final = color_only["final_eval"]
    deltas = {
        key: float(geometry_final[key]) - float(color_final[key])
        for key in (
            "eval_l1",
            "eval_mse",
            "eval_frame_psnr_mean",
            "eval_frame_psnr_min",
            "eval_frame_snr_mean",
            "eval_frame_snr_min",
        )
    }
    checks = [
        check(
            metric(geometry_summary, "state_mean_temporal_screen_delta_px") > geometry_motion_threshold_px,
            "geometry_lane_moves_screen_geometry",
            geometry["motion_vs_repaint"],
        ),
        check(
            metric(geometry_summary, "eval_mean_temporal_alpha_delta") > 0.0
            and metric(geometry_summary, "eval_mean_temporal_support_delta") > 0.0,
            "geometry_lane_changes_alpha_support",
            geometry["motion_vs_repaint"],
        ),
        check(
            metric(geometry_summary, "state_mean_temporal_feature_abs_delta") == 0.0,
            "geometry_lane_keeps_dynamic_features_frozen",
            geometry["motion_vs_repaint"],
        ),
        check(
            metric(color_only_summary, "state_mean_temporal_screen_delta_px") <= frozen_motion_threshold_px
            and metric(color_only_summary, "eval_mean_temporal_alpha_delta") == 0.0
            and metric(color_only_summary, "eval_mean_temporal_support_delta") == 0.0,
            "color_only_lane_keeps_geometry_alpha_support_fixed",
            color_only["motion_vs_repaint"],
        ),
        check(
            metric(color_only_summary, "state_mean_temporal_feature_abs_delta") > 0.0,
            "color_only_lane_repaints_features",
            color_only["motion_vs_repaint"],
        ),
    ]
    return {
        "schema_version": "dynamic_powerfoam_motion_vs_repaint_comparison_v1",
        "ok": all(bool(item["passed"]) for item in checks),
        "checks": checks,
        "geometry_only": geometry,
        "color_only_fixed_geometry": color_only,
        "geometry_minus_color_only": deltas,
        "quality_winner_by_mean_snr": "geometry_only"
        if deltas["eval_frame_snr_mean"] > 0.0
        else "color_only_fixed_geometry",
        "quality_winner_by_min_snr": "geometry_only"
        if deltas["eval_frame_snr_min"] > 0.0
        else "color_only_fixed_geometry",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare dynamic PowerFoam geometry motion against fixed-geometry repaint.")
    parser.add_argument("--geometry-summary", type=Path, required=True)
    parser.add_argument("--color-only-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--geometry-motion-threshold-px", type=float, default=1.0e-5)
    parser.add_argument("--frozen-motion-threshold-px", type=float, default=1.0e-5)
    args = parser.parse_args()
    report = compare(
        load_summary(args.geometry_summary),
        load_summary(args.color_only_summary),
        geometry_motion_threshold_px=float(args.geometry_motion_threshold_px),
        frozen_motion_threshold_px=float(args.frozen_motion_threshold_px),
    )
    write_report_json(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
