"""CLI for one frozen G4 WorldFoam/World-Tubes/dynamic-3DGS ablation row."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "src" / "train"
LANE2 = ROOT / "research_experiments" / "world_foam_lane2"
for import_root in (TRAIN, LANE2, ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from worldfoam_native4d_public_quality_row import (  # noqa: E402
    RowRequest,
    build_row_plan,
    execute_row_lifecycle,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--g4-config", type=Path, required=True)
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
        "--wandb-mode",
        choices=("offline", "online"),
        default="offline",
    )
    args = parser.parse_args()
    request = RowRequest(
        config_path=args.g4_config,
        protocol_path=args.protocol,
        scene=args.scene,
        seed=args.seed,
        route=args.route,
        output_path=args.output,
        allow_local_mps_execution=bool(args.allow_local_mps_execution),
        dataset_capability_path=args.dataset_capability,
    )
    result = (
        execute_row_lifecycle(request, wandb_mode=args.wandb_mode)
        if args.execute
        else build_row_plan(request)
    )
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

