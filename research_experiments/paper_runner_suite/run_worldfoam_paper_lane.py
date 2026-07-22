from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = ROOT / "src" / "train"
for path in (ROOT, TRAIN_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config_utils import load_config_file, serialize_config_value
from paper_training_protocol import resolve_paper_training_protocol
from powerfoam_metal_trainer import run_training
from research_experiments.paper_runner_suite.run_unified_paper_ablation import (
    DEFAULT_PROTOCOL,
    DEFAULT_WORLDFOAM_INITIALIZER,
    powerfoam_config,
    require_execution_safety_acknowledgement,
    resolve_root_path,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Private process-isolated WorldFoam lane for the unified paper runner."
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--worldfoam-initializer", default=DEFAULT_WORLDFOAM_INITIALIZER)
    parser.add_argument("--allow-local-mps-execution", action="store_true")
    parser.add_argument("--allow-high-risk-local-mps", action="store_true")
    args = parser.parse_args()

    protocol_path = resolve_root_path(args.protocol)
    raw_protocol = load_config_file(protocol_path)
    protocol = resolve_paper_training_protocol(raw_protocol)
    cfg = powerfoam_config(
        raw_protocol,
        protocol,
        args.seed,
        resolve_root_path(args.out_dir),
        device=args.device,
        wandb_mode=args.wandb_mode,
        worldfoam_initializer=args.worldfoam_initializer,
    )
    if not args.execute:
        print(json.dumps(serialize_config_value(cfg), indent=2, sort_keys=True))
        return
    require_execution_safety_acknowledgement(
        protocol,
        device=args.device,
        allow_local_mps_execution=args.allow_local_mps_execution,
        allow_high_risk_local_mps=args.allow_high_risk_local_mps,
    )
    run_training(cfg)


if __name__ == "__main__":
    main()
