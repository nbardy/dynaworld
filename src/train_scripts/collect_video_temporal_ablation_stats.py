#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


DEFAULT_METRICS = [
    "Eval/Loss",
    "Eval/L1",
    "Eval/SSIM",
    "Eval/PSNR",
    "Eval/TemporalPredAdjacentL1",
    "Eval/TemporalGTAdjacentL1",
    "Eval/TemporalAdjacentL1Ratio",
    "Eval/TemporalPredToFirstL1",
    "Eval/TemporalToFirstL1Ratio",
    "Eval/DecodedXYZAdjacentL2",
    "Eval/DecodedXYZToFirstL2",
    "Eval/DecodedOpacityAdjacentL1",
    "Eval/DecodedRGBAdjacentL1",
    "Camera/EvalAdjacentRotationDeltaDegrees",
    "Camera/EvalAdjacentTranslationDelta",
    "BankRate/dynamic_motion",
    "BankRate/dynamic_rotation",
    "BankRate/dynamic_alpha_time",
    "_runtime",
]


def _yaml_value(payload: dict[str, Any], key: str, default: Any = None) -> Any:
    value = payload.get(key, {})
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return default


def load_run(run_dir: Path) -> dict[str, Any] | None:
    files_dir = run_dir / "files"
    summary_path = files_dir / "wandb-summary.json"
    config_path = files_dir / "config.yaml"
    if not summary_path.exists() or not config_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    config = yaml.safe_load(config_path.read_text())
    logging_cfg = _yaml_value(config, "logging", {}) or {}
    model_cfg = _yaml_value(config, "model", {}) or {}
    features_cfg = _yaml_value(config, "features", {}) or {}
    return {
        "run_dir": str(run_dir),
        "run_id": run_dir.name.removeprefix("run-").split("-")[-1],
        "run_name": logging_cfg.get("wandb_run_name", run_dir.name),
        "tags": logging_cfg.get("wandb_tags", []),
        "variant": model_cfg.get("variant"),
        "backend": model_cfg.get("video_encoder_backend"),
        "feature_extractor": features_cfg.get("extractor"),
        "feature_model": features_cfg.get("model_id"),
        "cross_attn_layers": model_cfg.get("cross_attn_layers"),
        "static_tokens": model_cfg.get("static_tokens"),
        "dynamic_tokens": model_cfg.get("dynamic_tokens"),
        "summary": summary,
    }


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect local W&B temporal-ablation summaries.")
    parser.add_argument("--wandb-root", type=Path, default=Path("wandb"))
    parser.add_argument("--tag", default="temporal-ablation", help="Only include runs with this W&B tag.")
    parser.add_argument(
        "--include-base",
        action="store_true",
        help="Also include the RGB-uniform strong-init base run without the temporal-ablation tag.",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON records instead of a markdown table.")
    args = parser.parse_args()

    rows = []
    for run_dir in sorted(args.wandb_root.glob("run-*")):
        record = load_run(run_dir)
        if record is None:
            continue
        tags = set(record["tags"])
        is_base = record["run_name"] == "ablate-init-rgb-uniform-strong-video-implicit-128-fast-mac-8192splats"
        is_suite_run = str(record["run_name"]).startswith("ablate-time-")
        if not is_suite_run and not (args.include_base and is_base):
            continue
        if args.tag not in tags and not (args.include_base and is_base):
            continue
        rows.append(record)

    rows.sort(key=lambda item: item["run_name"])
    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return

    headers = [
        "run_name",
        "variant",
        "backend",
        "features",
        "feature_model",
        "cross_attn",
        "static",
        "dynamic",
        *DEFAULT_METRICS,
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        summary = row["summary"]
        values = [
            row["run_name"],
            row["variant"],
            row["backend"],
            row["feature_extractor"],
            row["feature_model"],
            row["cross_attn_layers"],
            row["static_tokens"],
            row["dynamic_tokens"],
            *[summary.get(metric) for metric in DEFAULT_METRICS],
        ]
        print("| " + " | ".join(format_value(value) for value in values) + " |")


if __name__ == "__main__":
    main()
