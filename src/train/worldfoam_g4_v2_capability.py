"""Source-bound capability contract for the selected-ray G4-v2 stack."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from worldfoam_g4_selected_ray_contract import (
    DEFAULT_CONFIG,
    REQUIRED_ROUTES,
    canonical_sha256,
    file_sha256,
    load_selected_ray_contract,
)


ROOT = Path(__file__).resolve().parents[2]
CAPABILITY_PATH = (
    ROOT / "src" / "train" / "worldfoam_g4_v2_source_capability.json"
)


def _sources() -> tuple[Path, ...]:
    train = ROOT / "src" / "train"
    lane2 = ROOT / "research_experiments" / "world_foam_lane2"
    return (
        train / "worldfoam_g4_v2_capability.py",
        train / "worldfoam_g4_selected_ray_contract.py",
        train / "worldfoam_g4_selected_ray_work_plan.py",
        train / "worldfoam_native4d_public_quality_row.py",
        train / "worldfoam_native4d_public_quality_row_v2.py",
        train / "train_worldfoam_native4d_public_quality_row_v2.py",
        train / "worldfoam_public_quality_dataset_provider.py",
        train / "worldfoam_public_quality_inputs.py",
        train / "gaussian_public_quality_executor.py",
        train / "world_tubes_public_quality_executor.py",
        train / "dynamic_3dgs_public_quality_executor.py",
        train / "worldfoam_native4d_public_quality_executor.py",
        train / "worldfoam_native_heldout_prediction.py",
        train / "worldfoam_spatial_major_heldout_evaluator.py",
        lane2 / "run_worldfoam_g4_v2_pilot.py",
        lane2 / "verify_worldfoam_g4_v2_pilot.py",
        lane2 / "run_worldfoam_memory_scaling_acceptance.py",
        lane2 / "run_worldfoam_public_quality_ablation_v2.py",
        lane2 / "verify_worldfoam_public_quality_ablation_v2.py",
    )


def required_source_capability(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    config, _base, base_path = load_selected_ray_contract(config_path)
    sources = _sources()
    missing = tuple(path for path in sources if not path.is_file())
    if missing:
        raise FileNotFoundError(f"G4-v2 source stack is incomplete: {missing}")
    identities = {
        str(path.relative_to(ROOT)): {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sources
    }
    payload = {
        "schema_version": 2,
        "status": "source_complete_runtime_unmeasured",
        "row_kind": "worldfoam-native4d-public-quality-selected-ray-row-v2",
        "supported_routes": list(REQUIRED_ROUTES),
        "selected_ray_training": True,
        "arbitrary_selected_pixel_ids": True,
        "identical_selected_pixel_schedule_all_routes": True,
        "identical_target_and_loss_budget_all_routes": True,
        "training_loss_identifier": config["training_loss"]["identifier"],
        "training_loss_contract_sha256": canonical_sha256(config["training_loss"]),
        "route_specific_rasterized_work_receipt": True,
        "full_pixel_full_temporal_heldout_evaluation": True,
        "spatial_major_cross_time_worldfoam_heldout_required": True,
        "spatial_major_cross_time_worldfoam_heldout_source_complete": True,
        "parent_process_group_rss_watchdog_required": True,
        "pre_matrix_host_resource_guard_required": True,
        "pre_worker_host_resource_guard_evidence_required": True,
        "host_resource_guard_rechecked_before_every_row": True,
        "pre_matrix_minimum_free_disk_bytes": config["execution"][
            "pre_matrix_minimum_free_disk_bytes"
        ],
        "pre_matrix_minimum_available_memory_bytes": config["execution"][
            "pre_matrix_minimum_available_memory_bytes"
        ],
        "pre_matrix_maximum_swap_used_bytes": config["execution"][
            "pre_matrix_maximum_swap_used_bytes"
        ],
        "pre_matrix_maximum_load_average": config["execution"][
            "pre_matrix_maximum_load_average"
        ],
        "hard_mps_working_set_limit_bytes_per_worker": config["execution"][
            "maximum_mps_working_set_bytes_per_worker"
        ],
        "mps_working_set_limit_receipt_required": True,
        "sampled_mps_peaks_must_not_exceed_hard_cap": True,
        "real_native_only": True,
        "proxy_or_smoke_evidence_permitted": False,
        "v1_all_pixel_contract_mutated": False,
        "runtime_or_memory_measured": False,
        "pilot_receipt_required_before_full_matrix": True,
        "v2_config_sha256": file_sha256(config_path),
        "base_g4_v1_sha256": file_sha256(base_path),
        "source_identities": identities,
    }
    return {**payload, "capability_sha256": canonical_sha256(payload)}


def write_source_capability(
    destination: Path = CAPABILITY_PATH,
    *,
    config_path: Path = DEFAULT_CONFIG,
) -> Path:
    destination = Path(destination).resolve()
    try:
        destination.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError("G4-v2 capability destination leaves the repository") from error
    payload = required_source_capability(config_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    temporary.unlink(missing_ok=True)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=CAPABILITY_PATH)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    payload = required_source_capability(args.config)
    if args.write:
        write_source_capability(args.output, config_path=args.config)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = (
    "CAPABILITY_PATH",
    "required_source_capability",
    "write_source_capability",
)
