#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DYNAWORLD = Path(__file__).resolve().parents[2]
RESULTS_DIR = DYNAWORLD / "research_experiments" / "world_foam_lane2" / "results"
DEFAULT_QUALITY_BRIDGE = RESULTS_DIR / "2026-05-20_worldfoam_site_initialization_quality_bridge.json"
DEFAULT_TOPOLOGY_ARTIFACT = (
    RESULTS_DIR / "2026-05-20_gate4_affine_candidate_csr_capacity_legacy_frame_pixel_mean_render8_site4_2_4f.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: JSON root must be an object")
    return payload


def _find_quality_candidate(quality: dict[str, Any], candidate: str) -> dict[str, Any] | None:
    rows = quality.get("rows")
    if not isinstance(rows, list):
        return None
    for row in rows:
        if isinstance(row, dict) and row.get("site_initialization") == candidate:
            return row
    return None


def _acceptance_failures(topology: dict[str, Any]) -> list[str]:
    acceptance = topology.get("acceptance")
    if not isinstance(acceptance, dict):
        return ["missing_acceptance"]
    failures = [str(key) for key, value in acceptance.items() if value is not True]
    return sorted(failures)


def build_report(
    *,
    quality_bridge_path: Path,
    topology_artifact_path: Path,
) -> dict[str, Any]:
    failures: list[str] = []
    try:
        quality = _load_json(quality_bridge_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "failed", "failures": [f"quality bridge load failed: {exc}"]}
    try:
        topology = _load_json(topology_artifact_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "failed", "failures": [f"topology artifact load failed: {exc}"]}

    if quality.get("status") != "ok":
        failures.append(f"quality bridge status is not ok: {quality.get('status')!r}")
    candidate = quality.get("next_mps_candidate")
    if not isinstance(candidate, str) or not candidate:
        failures.append("quality bridge did not select next_mps_candidate")
        candidate = None

    quality_candidate: dict[str, Any] | None = None
    if candidate is not None:
        quality_candidate = _find_quality_candidate(quality, candidate)
        if quality_candidate is None:
            failures.append(f"quality bridge missing row for next_mps_candidate {candidate!r}")
        elif quality_candidate.get("positive_cpu_reference_candidate") is not True:
            failures.append(f"quality candidate {candidate!r} is not positive_cpu_reference_candidate")

    if topology.get("status") != "ok":
        failures.append(f"topology artifact status is not ok: {topology.get('status')!r}")
    if topology.get("site_initialization") != candidate:
        failures.append(
            "topology artifact site_initialization does not match next_mps_candidate: "
            f"{topology.get('site_initialization')!r} != {candidate!r}"
        )
    topology_acceptance_failures = _acceptance_failures(topology)
    failures.extend(f"topology acceptance failed: {key}" for key in topology_acceptance_failures)

    frame_counts = topology.get("frame_counts")
    if not isinstance(frame_counts, list) or len(frame_counts) < 2:
        failures.append("topology artifact must cover at least two frame counts")

    status = "ok" if not failures else "failed"
    return {
        "status": status,
        "benchmark": "world_foam_next_mps_candidate_readiness",
        "failures": failures,
        "quality_bridge_artifact": str(quality_bridge_path),
        "topology_artifact": str(topology_artifact_path),
        "next_mps_candidate": candidate,
        "quality_candidate_positive": (
            bool(quality_candidate.get("positive_cpu_reference_candidate")) if quality_candidate is not None else False
        ),
        "quality_train_psnr_delta_vs_baseline": (
            quality_candidate.get("train_psnr_delta_vs_baseline") if quality_candidate is not None else None
        ),
        "quality_heldout_psnr_delta_vs_baseline": (
            quality_candidate.get("heldout_psnr_delta_vs_baseline") if quality_candidate is not None else None
        ),
        "topology_frame_counts": frame_counts if isinstance(frame_counts, list) else None,
        "topology_candidate_count_scale_first_to_last": topology.get("candidate_count_scale_first_to_last"),
        "topology_storage_scale_first_to_last": topology.get("storage_scale_first_to_last"),
        "topology_acceptance_failures": topology_acceptance_failures,
        "ready_for_quiet_mps_quality_speed_run": status == "ok",
        "mps_quality_speed_artifact_required": True,
        "quality_claim": False,
        "speed_claim": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gate the next WorldFoam site-initialization MPS candidate from CPU "
            "quality and topology artifacts."
        )
    )
    parser.add_argument("--quality-bridge", type=Path, default=DEFAULT_QUALITY_BRIDGE)
    parser.add_argument("--topology-artifact", type=Path, default=DEFAULT_TOPOLOGY_ARTIFACT)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=RESULTS_DIR / "2026-05-20_worldfoam_next_mps_candidate_readiness.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_report(
        quality_bridge_path=args.quality_bridge,
        topology_artifact_path=args.topology_artifact,
    )
    text = json.dumps(payload, indent=2, sort_keys=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(text + "\n", encoding="utf-8")
    print(text)
    if payload["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
