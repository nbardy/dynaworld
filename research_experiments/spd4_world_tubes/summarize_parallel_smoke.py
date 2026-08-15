#!/usr/bin/env python3
"""Validate and summarize bounded legacy/SPD(4) Coffee Martini smoke runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
STAR_ROOT = (
    ROOT / "third_party" / "fast-mac-gsplat" / "variants" / "star_uvt_v0"
)
DEFAULT_LEGACY_REPORT = (
    STAR_ROOT
    / "artifacts"
    / "spd4_parallel_smoke"
    / "legacy_tube_v2"
    / "comparison_report.json"
)
DEFAULT_FULL_LIFT_REPORT = (
    ROOT
    / "artifacts"
    / "spd4_parallel_smoke"
    / "full_spd4_legacy_lift"
    / "comparison_report.json"
)
DEFAULT_FULL_ISOTROPIC_REPORT = (
    ROOT
    / "artifacts"
    / "spd4_parallel_smoke"
    / "full_spd4_isotropic"
    / "comparison_report.json"
)
DEFAULT_FULL_MATCHED_REPORT = (
    ROOT
    / "artifacts"
    / "spd4_parallel_smoke"
    / "full_spd4_param_matched_199"
    / "comparison_report.json"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "spd4_parallel_smoke" / "summary.json"


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _lane_summary(path: Path) -> dict[str, Any]:
    report = _load(path)
    meta = report["meta"]
    lane = report["star_uvt"]
    metrics = lane["metrics"]
    protocol = lane["paper_protocol"]
    cost = protocol["cost"]
    timing = protocol["timing"]
    metal_rows = lane["metal_stats"]["rows"]
    return {
        "report": str(path),
        "representation": lane["world_representation"],
        "initialization": {
            "precision_xy": lane["init_precision_xy"],
            "spd4_precision_z": lane["spd4_init_precision_z"],
        },
        "seed": meta["seed"],
        "frame_count": meta["frame_count"],
        "image_size": meta["image_size"],
        "train_cameras": meta["train_cameras"],
        "heldout_cameras": meta["heldout_cameras"],
        "tube_count": lane["tube_count"],
        "geometry_dof_per_atom": lane["geometry_dof_per_atom"],
        "total_dof_per_atom": lane["total_dof_per_atom"],
        "parameter_count": cost["parameter_count"],
        "parameter_bytes": cost["parameter_bytes"],
        "optimizer_state_bytes": cost["optimizer_state_bytes"],
        "serialized_checkpoint_bytes": cost["serialized_checkpoint_bytes"],
        "optimizer_steps": cost["optimizer_steps"],
        "rasterized_pixels": cost["rasterized_pixels"],
        "sampled_peak_current_allocated_bytes": cost[
            "sampled_peak_current_allocated_bytes"
        ],
        "sampled_peak_driver_allocated_bytes": cost[
            "sampled_peak_driver_allocated_bytes"
        ],
        "train_wall_s": timing["train_wall_s"],
        "steady_forward_s": timing["steady_forward_s"],
        "backward_s": timing["backward_s"],
        "heldout": {
            "psnr": metrics["heldout_eval_psnr"],
            "ssim": metrics["heldout_eval_ssim"],
            "lpips": metrics["heldout_eval_lpips"],
            "l1": metrics["heldout_eval_l1"],
        },
        "train_eval": {
            "psnr": metrics["eval_psnr"],
            "ssim": metrics["eval_ssim"],
            "l1": metrics["eval_l1"],
        },
        "metal": {
            "overflow_tiles": sum(
                int(row["stats"]["overflow_tile_count"]) for row in metal_rows
            ),
            "max_tile_count": max(
                int(row["stats"]["max_tile_count"]) for row in metal_rows
            ),
            "uvt_tile_tube_pairs": {
                f"{row['split']}:{row['camera']}": int(
                    row["stats"]["uvt_tile_tube_pairs"]
                )
                for row in metal_rows
            },
        },
    }


def _delta(candidate: dict[str, Any], legacy: dict[str, Any]) -> dict[str, float]:
    return {
        "heldout_psnr_db": (
            candidate["heldout"]["psnr"] - legacy["heldout"]["psnr"]
        ),
        "heldout_ssim": (
            candidate["heldout"]["ssim"] - legacy["heldout"]["ssim"]
        ),
        "heldout_lpips": (
            candidate["heldout"]["lpips"] - legacy["heldout"]["lpips"]
        ),
        "heldout_l1": candidate["heldout"]["l1"] - legacy["heldout"]["l1"],
        "train_eval_psnr_db": (
            candidate["train_eval"]["psnr"] - legacy["train_eval"]["psnr"]
        ),
        "train_wall_ratio": (
            candidate["train_wall_s"] / legacy["train_wall_s"]
        ),
        "steady_forward_ratio": (
            candidate["steady_forward_s"] / legacy["steady_forward_s"]
        ),
        "backward_ratio": candidate["backward_s"] / legacy["backward_s"],
        "driver_memory_ratio": (
            candidate["sampled_peak_driver_allocated_bytes"]
            / legacy["sampled_peak_driver_allocated_bytes"]
        ),
        "parameter_ratio": (
            candidate["parameter_count"] / legacy["parameter_count"]
        ),
    }


def summarize(
    *,
    legacy_path: Path,
    full_lift_path: Path,
    full_isotropic_path: Path,
    full_matched_path: Path,
) -> dict[str, Any]:
    rows = {
        "legacy_256": _lane_summary(legacy_path),
        "full_spd4_256_near_planar": _lane_summary(full_lift_path),
        "full_spd4_256_isotropic": _lane_summary(full_isotropic_path),
        "full_spd4_199_param_matched": _lane_summary(full_matched_path),
    }
    legacy = rows["legacy_256"]
    comparable_keys = (
        "seed",
        "frame_count",
        "image_size",
        "train_cameras",
        "heldout_cameras",
        "optimizer_steps",
        "rasterized_pixels",
    )
    protocol_drift = {
        row_name: [
            key for key in comparable_keys if row[key] != legacy[key]
        ]
        for row_name, row in rows.items()
    }
    checks = {
        "representations_are_distinct": (
            legacy["representation"] == "legacy_tube"
            and all(
                rows[name]["representation"] == "full_spd4"
                for name in rows
                if name != "legacy_256"
            )
        ),
        "shared_protocol_budget": not any(protocol_drift.values()),
        "same_atom_rows_use_256": (
            legacy["tube_count"] == 256
            and rows["full_spd4_256_near_planar"]["tube_count"] == 256
            and rows["full_spd4_256_isotropic"]["tube_count"] == 256
        ),
        "matched_parameter_row_is_within_two_scalars": (
            abs(
                rows["full_spd4_199_param_matched"]["parameter_count"]
                - legacy["parameter_count"]
            )
            <= 2
        ),
        "all_runs_have_zero_tile_overflow": all(
            row["metal"]["overflow_tiles"] == 0 for row in rows.values()
        ),
        "all_sampled_driver_memory_below_100mb": all(
            row["sampled_peak_driver_allocated_bytes"] < 100_000_000
            for row in rows.values()
        ),
    }
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass" if all(checks.values()) else "fail",
        "scope": "bounded_two_step_static_multicamera_metal_smoke",
        "claim_limits": {
            "optimizer_steps": 2,
            "quality_conclusion": False,
            "speed_conclusion": False,
            "paper_baseline": False,
            "purpose": (
                "end-to-end dispatch, memory, parameter accounting, "
                "same-atom and matched-parameter plumbing"
            ),
        },
        "rows": rows,
        "protocol_drift": protocol_drift,
        "deltas_vs_legacy": {
            name: _delta(row, legacy)
            for name, row in rows.items()
            if name != "legacy_256"
        },
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-report", type=Path, default=DEFAULT_LEGACY_REPORT)
    parser.add_argument(
        "--full-lift-report", type=Path, default=DEFAULT_FULL_LIFT_REPORT
    )
    parser.add_argument(
        "--full-isotropic-report",
        type=Path,
        default=DEFAULT_FULL_ISOTROPIC_REPORT,
    )
    parser.add_argument(
        "--full-matched-report", type=Path, default=DEFAULT_FULL_MATCHED_REPORT
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = summarize(
        legacy_path=args.legacy_report.resolve(),
        full_lift_path=args.full_lift_report.resolve(),
        full_isotropic_path=args.full_isotropic_report.resolve(),
        full_matched_path=args.full_matched_report.resolve(),
    )
    output = args.out if args.out.is_absolute() else ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote paired SPD(4) smoke summary to {output}")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
