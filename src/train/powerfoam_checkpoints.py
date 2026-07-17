from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn

from checkpoint_utils import atomic_torch_save
from config_utils import serialize_config_value
from train_artifacts import write_json


def select_best_metric(metrics: dict[str, float]) -> tuple[str, float]:
    if "heldout_eval_psnr" in metrics:
        return "heldout_eval_psnr", float(metrics["heldout_eval_psnr"])
    return "eval_psnr", float(metrics["eval_psnr"])


def save_powerfoam_checkpoint(
    path: Path,
    model: nn.Module,
    cfg: dict[str, Any],
    *,
    step: int | None = None,
    metrics: dict[str, float] | None = None,
    best_metric_name: str | None = None,
    best_metric_value: float | None = None,
) -> None:
    payload = {
        "model": model.state_dict(),
        "config": serialize_config_value(cfg),
    }
    if step is not None:
        payload.update(
            {
                "step": int(step),
                "metrics": metrics or {},
                "best_metric_name": best_metric_name,
                "best_metric_value": best_metric_value,
            }
        )
    atomic_torch_save(payload, path)


def maybe_save_best_powerfoam_checkpoint(
    model: nn.Module,
    cfg: dict[str, Any],
    output_dir: Path,
    *,
    step: int,
    metrics: dict[str, float],
    best_metric_value: float | None,
) -> float:
    metric_name, metric_value = select_best_metric(metrics)
    if best_metric_value is not None and metric_value <= best_metric_value:
        return best_metric_value
    save_powerfoam_checkpoint(
        output_dir / "checkpoint_best.pt",
        model,
        cfg,
        step=step,
        metrics=metrics,
        best_metric_name=metric_name,
        best_metric_value=metric_value,
    )
    summary = {
        "step": int(step),
        "best_metric_name": metric_name,
        "best_metric_value": metric_value,
        "metrics": metrics,
    }
    write_json(output_dir / "best_metrics.json", summary)
    return metric_value
