#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DYNAWORLD = Path(__file__).resolve().parents[2]
RESULTS_DIR = DYNAWORLD / "research_experiments" / "world_foam_lane2" / "results"
DEFAULT_ARTIFACTS = (
    RESULTS_DIR / "2026-05-20_gate1_legacy_sparse_reference_render16_site9_2f.json",
    RESULTS_DIR / "2026-05-20_gate1_stratified_grid_reference_render16_site9_2f.json",
    RESULTS_DIR / "2026-05-20_gate1_legacy_pixel_mean_reference_render16_site9_2f.json",
    RESULTS_DIR / "2026-05-20_gate1_legacy_frame_pixel_mean_reference_render16_site9_2f.json",
    RESULTS_DIR / "2026-05-21_gate1_legacy_frame_patch3_mean_reference_render16_site9_2f.json",
    RESULTS_DIR / "2026-05-20_gate1_stratified_pixel_mean_reference_render16_site9_2f.json",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return payload


def _finite_float(value: Any, *, field: str) -> float:
    if not isinstance(value, (float, int)):
        raise ValueError(f"{field} must be finite number, got {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite number, got {value!r}")
    return result


def _split_metrics(payload: dict[str, Any], split: str) -> dict[str, float]:
    split_payload = payload.get(split)
    if not isinstance(split_payload, dict):
        raise ValueError(f"Gate1 artifact missing {split} object")
    return {
        "psnr": _finite_float(split_payload.get("target_psnr"), field=f"{split}.target_psnr"),
        "l1": _finite_float(split_payload.get("target_l1"), field=f"{split}.target_l1"),
        "mse": _finite_float(split_payload.get("target_mse"), field=f"{split}.target_mse"),
    }


def summarize_gate1_artifact(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    initialization = payload.get("site_initialization")
    if not isinstance(initialization, str) or not initialization:
        raise ValueError(f"{path}: missing site_initialization")
    if payload.get("benchmark") != "world_foam_lane2_gate1_realray_per_sample_reference":
        raise ValueError(f"{path}: expected Gate1 real-ray reference artifact")
    if payload.get("status") != "ok":
        raise ValueError(f"{path}: expected status=ok, got {payload.get('status')!r}")
    return {
        "artifact": str(path),
        "status": payload.get("status"),
        "site_initialization": initialization,
        "config_path": payload.get("config_path"),
        "frame_count": payload.get("frame_count"),
        "render_size": payload.get("render_size"),
        "site_count": payload.get("site_count"),
        "boundary_count": payload.get("boundary_count"),
        "train": _split_metrics(payload, "train"),
        "heldout": _split_metrics(payload, "heldout"),
    }


def _fixture_key(summary: dict[str, Any]) -> tuple[Any, ...]:
    return (
        summary.get("config_path"),
        summary.get("frame_count"),
        summary.get("render_size"),
        summary.get("site_count"),
        summary.get("boundary_count"),
    )


def _compare_to_baseline(summary: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    train_psnr_delta = float(summary["train"]["psnr"]) - float(baseline["train"]["psnr"])
    heldout_psnr_delta = float(summary["heldout"]["psnr"]) - float(baseline["heldout"]["psnr"])
    train_l1_delta = float(summary["train"]["l1"]) - float(baseline["train"]["l1"])
    heldout_l1_delta = float(summary["heldout"]["l1"]) - float(baseline["heldout"]["l1"])
    return {
        "train_psnr_delta_vs_baseline": train_psnr_delta,
        "heldout_psnr_delta_vs_baseline": heldout_psnr_delta,
        "train_l1_delta_vs_baseline": train_l1_delta,
        "heldout_l1_delta_vs_baseline": heldout_l1_delta,
        "improves_train_psnr": train_psnr_delta > 0.0,
        "improves_heldout_psnr": heldout_psnr_delta > 0.0,
        "reduces_train_l1": train_l1_delta < 0.0,
        "reduces_heldout_l1": heldout_l1_delta < 0.0,
        "positive_cpu_reference_candidate": (
            train_psnr_delta > 0.0
            and heldout_psnr_delta > 0.0
            and train_l1_delta < 0.0
            and heldout_l1_delta < 0.0
        ),
    }


def build_report(
    artifacts: tuple[Path, ...],
    *,
    baseline_initialization: str,
) -> dict[str, Any]:
    failures: list[str] = []
    summaries: list[dict[str, Any]] = []
    for artifact in artifacts:
        try:
            summaries.append(summarize_gate1_artifact(artifact))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            failures.append(str(exc))
    if failures:
        return {"status": "failed", "failures": failures}
    if not summaries:
        return {"status": "failed", "failures": ["at least one Gate1 artifact is required"]}

    by_initialization = {}
    for summary in summaries:
        initialization = str(summary["site_initialization"])
        if initialization in by_initialization:
            failures.append(f"duplicate site_initialization {initialization}")
        by_initialization[initialization] = summary
    baseline = by_initialization.get(baseline_initialization)
    if baseline is None:
        failures.append(f"missing baseline initialization {baseline_initialization}")
        return {"status": "failed", "failures": failures}

    baseline_key = _fixture_key(baseline)
    for summary in summaries:
        if _fixture_key(summary) != baseline_key:
            failures.append(f"{summary['artifact']}: fixture does not match baseline")
    if failures:
        return {"status": "failed", "failures": failures}

    rows = []
    for summary in sorted(summaries, key=lambda item: str(item["site_initialization"])):
        comparison = _compare_to_baseline(summary, baseline)
        rows.append({**summary, **comparison})

    best_by_heldout = max(rows, key=lambda item: (float(item["heldout"]["psnr"]), float(item["train"]["psnr"])))
    positive_candidates = [
        row for row in rows if row["site_initialization"] != baseline_initialization and row["positive_cpu_reference_candidate"]
    ]
    rejected_candidates = [
        row for row in rows if row["site_initialization"] != baseline_initialization and not row["positive_cpu_reference_candidate"]
    ]
    next_candidate = max(
        positive_candidates,
        key=lambda item: (float(item["heldout_psnr_delta_vs_baseline"]), float(item["train_psnr_delta_vs_baseline"])),
        default=None,
    )
    return {
        "status": "ok",
        "benchmark": "world_foam_site_initialization_quality_bridge",
        "baseline_initialization": baseline_initialization,
        "fixture": {
            "config_path": baseline["config_path"],
            "frame_count": baseline["frame_count"],
            "render_size": baseline["render_size"],
            "site_count": baseline["site_count"],
            "boundary_count": baseline["boundary_count"],
        },
        "candidate_count": len(rows) - 1,
        "positive_candidate_count": len(positive_candidates),
        "rejected_candidate_count": len(rejected_candidates),
        "best_by_heldout_psnr": best_by_heldout["site_initialization"],
        "next_mps_candidate": next_candidate["site_initialization"] if next_candidate is not None else None,
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Gate1 WorldFoam site initialization quality artifacts.")
    parser.add_argument("--artifact", type=Path, action="append", dest="artifacts")
    parser.add_argument("--baseline-initialization", default="legacy_sparse")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_DIR / "2026-05-20_worldfoam_site_initialization_quality_bridge.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = tuple(args.artifacts) if args.artifacts else DEFAULT_ARTIFACTS
    payload = build_report(artifacts, baseline_initialization=str(args.baseline_initialization))
    text = json.dumps(payload, indent=2, sort_keys=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
