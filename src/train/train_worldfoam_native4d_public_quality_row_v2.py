"""CLI for one matched selected-ray G4-v2 public-quality ablation row."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
for import_root in (
    ROOT / "src" / "train",
    ROOT / "research_experiments" / "world_foam_lane2",
):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from worldfoam_native4d_public_quality_row import RowRequest  # noqa: E402
from worldfoam_native4d_public_quality_row_v2 import (  # noqa: E402
    MAXIMUM_MPS_WORKING_SET_BYTES,
    build_v2_row_plan,
    execute_v2_row,
)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--g4-v2-config", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--route",
        choices=(
            "worldfoam_native4d",
            "worldfoam_framewise_replay",
            "world_tubes",
            "dynamic_3dgs",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-capability", type=Path)
    parser.add_argument("--allow-local-mps-execution", action="store_true")
    parser.add_argument(
        "--maximum-mps-working-set-bytes",
        type=int,
        default=MAXIMUM_MPS_WORKING_SET_BYTES,
    )
    parser.add_argument("--wandb-mode", choices=("offline", "online"), default="offline")
    args = parser.parse_args()
    request = RowRequest(
        config_path=args.g4_v2_config,
        protocol_path=args.protocol,
        scene=args.scene,
        seed=args.seed,
        route=args.route,
        output_path=args.output,
        allow_local_mps_execution=bool(args.allow_local_mps_execution),
        dataset_capability_path=args.dataset_capability,
    )
    if args.execute:
        if not args.allow_local_mps_execution:
            raise RuntimeError(
                "G4-v2 row execution requires --allow-local-mps-execution"
            )
        result = execute_v2_row(
            request,
            wandb_mode=args.wandb_mode,
            maximum_mps_working_set_bytes=args.maximum_mps_working_set_bytes,
        )
    else:
        result = build_v2_row_plan(request)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
