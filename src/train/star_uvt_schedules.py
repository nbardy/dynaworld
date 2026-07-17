from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FeatureTargetWeightStage:
    label: str
    start_step: int
    end_step: int
    loss_weight: float
    rgb_loss_weight: float
    rgb_grid_loss_weight: float
    rgb_probe_loss_weight: float


@dataclass(frozen=True)
class OptimizerLrStage:
    label: str
    start_step: int
    end_step: int
    lr: float


def _feature_target_enabled(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("feature_target", {}).get("enabled", False))


def _rgb_loss_weight(cfg: dict[str, Any]) -> float:
    if not _feature_target_enabled(cfg):
        return 1.0
    return float(cfg["feature_target"]["rgb_loss_weight"])


def _feature_target_weight_schedule(cfg: dict[str, Any]) -> tuple[FeatureTargetWeightStage, ...]:
    steps = int(cfg["train"]["steps"])
    global_step_offset = int(cfg["train"].get("global_step_offset", 0))
    if global_step_offset < 0:
        raise ValueError("train.global_step_offset must be non-negative")
    schedule_end_step = global_step_offset + steps
    if not _feature_target_enabled(cfg):
        return (
            FeatureTargetWeightStage(
                label="rgb",
                start_step=0,
                end_step=schedule_end_step,
                loss_weight=0.0,
                rgb_loss_weight=1.0,
                rgb_grid_loss_weight=0.0,
                rgb_probe_loss_weight=0.0,
            ),
        )

    feature_target = cfg["feature_target"]
    feature_target.setdefault("rgb_grid_loss_weight", 0.0)
    raw_schedule = feature_target.get("weight_schedule")
    if raw_schedule is None:
        loss_weight = float(feature_target["loss_weight"])
        rgb_loss_weight = float(feature_target["rgb_loss_weight"])
        rgb_grid_loss_weight = float(feature_target.get("rgb_grid_loss_weight", 0.0))
        rgb_probe_loss_weight = float(feature_target.get("rgb_probe_loss_weight", 0.0))
        if (
            loss_weight < 0.0
            or rgb_loss_weight < 0.0
            or rgb_grid_loss_weight < 0.0
            or rgb_probe_loss_weight < 0.0
        ):
            raise ValueError(
                "feature_target loss_weight, rgb_loss_weight, rgb_grid_loss_weight, "
                "and rgb_probe_loss_weight must be non-negative"
            )
        if (
            loss_weight <= 0.0
            and rgb_loss_weight <= 0.0
            and rgb_grid_loss_weight <= 0.0
            and rgb_probe_loss_weight <= 0.0
        ):
            raise ValueError(
                "feature_target loss_weight, rgb_loss_weight, rgb_grid_loss_weight, "
                "and rgb_probe_loss_weight cannot all be <= 0"
            )
        return (
            FeatureTargetWeightStage(
                label="constant",
                start_step=0,
                end_step=schedule_end_step,
                loss_weight=loss_weight,
                rgb_loss_weight=rgb_loss_weight,
                rgb_grid_loss_weight=rgb_grid_loss_weight,
                rgb_probe_loss_weight=rgb_probe_loss_weight,
            ),
        )
    if not isinstance(raw_schedule, list) or not raw_schedule:
        raise ValueError("feature_target.weight_schedule must be a non-empty list when provided")

    stages: list[FeatureTargetWeightStage] = []
    start_step = 0
    for index, raw_stage in enumerate(raw_schedule):
        if not isinstance(raw_stage, Mapping):
            raise TypeError("feature_target.weight_schedule entries must be objects")
        missing = [
            key
            for key in ("until_step", "loss_weight", "rgb_loss_weight")
            if key not in raw_stage
        ]
        if missing:
            raise KeyError(
                "feature_target.weight_schedule entries require "
                f"until_step, loss_weight, and rgb_loss_weight; missing {missing}"
            )
        end_step = int(raw_stage["until_step"])
        if end_step <= start_step:
            raise ValueError(
                "feature_target.weight_schedule until_step values must be strictly increasing "
                "and cover step 0..train.global_step_offset + train.steps; "
                f"got until_step={end_step} after {start_step}"
            )
        loss_weight = float(raw_stage["loss_weight"])
        rgb_loss_weight = float(raw_stage["rgb_loss_weight"])
        rgb_grid_loss_weight = float(
            raw_stage.get("rgb_grid_loss_weight", feature_target.get("rgb_grid_loss_weight", 0.0))
        )
        rgb_probe_loss_weight = float(
            raw_stage.get("rgb_probe_loss_weight", feature_target.get("rgb_probe_loss_weight", 0.0))
        )
        if (
            loss_weight < 0.0
            or rgb_loss_weight < 0.0
            or rgb_grid_loss_weight < 0.0
            or rgb_probe_loss_weight < 0.0
        ):
            raise ValueError("feature_target.weight_schedule weights must be non-negative")
        if (
            loss_weight <= 0.0
            and rgb_loss_weight <= 0.0
            and rgb_grid_loss_weight <= 0.0
            and rgb_probe_loss_weight <= 0.0
        ):
            raise ValueError("feature_target.weight_schedule stage weights cannot all be <= 0")
        stages.append(
            FeatureTargetWeightStage(
                label=str(raw_stage.get("label", f"stage_{index}")),
                start_step=start_step,
                end_step=end_step,
                loss_weight=loss_weight,
                rgb_loss_weight=rgb_loss_weight,
                rgb_grid_loss_weight=rgb_grid_loss_weight,
                rgb_probe_loss_weight=rgb_probe_loss_weight,
            )
        )
        start_step = end_step

    if start_step != schedule_end_step:
        raise ValueError(
            "feature_target.weight_schedule must cover exactly "
            "train.global_step_offset + train.steps; "
            f"last until_step={start_step}, global_step_offset={global_step_offset}, train.steps={steps}"
        )
    return tuple(stages)


def _feature_target_weights_for_step(
    schedule: tuple[FeatureTargetWeightStage, ...],
    step: int,
) -> FeatureTargetWeightStage:
    for stage in schedule:
        if stage.start_step <= step < stage.end_step:
            return stage
    raise IndexError(f"step {step} is outside feature target weight schedule")


def _feature_target_weight_schedule_json(
    schedule: tuple[FeatureTargetWeightStage, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "label": stage.label,
            "start_step": stage.start_step,
            "end_step": stage.end_step,
            "loss_weight": stage.loss_weight,
            "rgb_loss_weight": stage.rgb_loss_weight,
            "rgb_grid_loss_weight": stage.rgb_grid_loss_weight,
            "rgb_probe_loss_weight": stage.rgb_probe_loss_weight,
        }
        for stage in schedule
    ]


def _optimizer_lr_schedule(cfg: dict[str, Any]) -> tuple[OptimizerLrStage, ...]:
    steps = int(cfg["train"]["steps"])
    global_step_offset = int(cfg["train"].get("global_step_offset", 0))
    if global_step_offset < 0:
        raise ValueError("train.global_step_offset must be non-negative")
    schedule_end_step = global_step_offset + steps
    raw_schedule = cfg["train"].get("lr_schedule")
    if raw_schedule is None:
        lr = float(cfg["train"]["lr"])
        if lr <= 0.0:
            raise ValueError("train.lr must be positive")
        return (
            OptimizerLrStage(
                label="constant",
                start_step=0,
                end_step=schedule_end_step,
                lr=lr,
            ),
        )
    if not isinstance(raw_schedule, list) or not raw_schedule:
        raise ValueError("train.lr_schedule must be a non-empty list when provided")

    stages: list[OptimizerLrStage] = []
    start_step = 0
    for index, raw_stage in enumerate(raw_schedule):
        if not isinstance(raw_stage, Mapping):
            raise TypeError("train.lr_schedule entries must be objects")
        missing = [key for key in ("until_step", "lr") if key not in raw_stage]
        if missing:
            raise KeyError(f"train.lr_schedule entries require until_step and lr; missing {missing}")
        end_step = int(raw_stage["until_step"])
        if end_step <= start_step:
            raise ValueError(
                "train.lr_schedule until_step values must be strictly increasing "
                "and cover step 0..train.global_step_offset + train.steps; "
                f"got until_step={end_step} after {start_step}"
            )
        lr = float(raw_stage["lr"])
        if lr <= 0.0:
            raise ValueError("train.lr_schedule lr values must be positive")
        stages.append(
            OptimizerLrStage(
                label=str(raw_stage.get("label", f"stage_{index}")),
                start_step=start_step,
                end_step=end_step,
                lr=lr,
            )
        )
        start_step = end_step

    if start_step != schedule_end_step:
        raise ValueError(
            "train.lr_schedule must cover exactly train.global_step_offset + train.steps; "
            f"last until_step={start_step}, global_step_offset={global_step_offset}, train.steps={steps}"
        )
    return tuple(stages)


def _optimizer_lr_for_step(schedule: tuple[OptimizerLrStage, ...], step: int) -> OptimizerLrStage:
    for stage in schedule:
        if stage.start_step <= step < stage.end_step:
            return stage
    raise IndexError(f"step {step} is outside optimizer LR schedule")


def _optimizer_lr_schedule_json(schedule: tuple[OptimizerLrStage, ...]) -> list[dict[str, Any]]:
    return [
        {
            "label": stage.label,
            "start_step": stage.start_step,
            "end_step": stage.end_step,
            "lr": stage.lr,
        }
        for stage in schedule
    ]
