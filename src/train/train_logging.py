from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from typing import Any

import wandb

from config_utils import serialize_config_value
from pipeline.diagnostics import camera_state_payload
from wandb_media import add_existing_wandb_media as _add_existing_wandb_media


def init_wandb_run(cfg: dict[str, Any]) -> Any | None:
    """Initialize a W&B run from the shared `logging` config block."""

    if not bool(cfg["logging"]["wandb_enabled"]):
        return None
    init_kwargs = {
        "project": cfg["logging"]["wandb_project"],
        "name": cfg["logging"]["wandb_run_name"],
        "tags": cfg["logging"]["wandb_tags"],
        "config": serialize_config_value(cfg),
    }
    if cfg["logging"]["wandb_mode"] is not None:
        init_kwargs["mode"] = str(cfg["logging"]["wandb_mode"])
    if cfg["logging"].get("wandb_run_id") is not None:
        init_kwargs["id"] = str(cfg["logging"]["wandb_run_id"])
    return wandb.init(**init_kwargs)


def finish_wandb_run(run: Any | None = None) -> None:
    active_run = run if run is not None else getattr(wandb, "run", None)
    if active_run is not None:
        finish = getattr(active_run, "finish", None)
        if callable(finish):
            finish()
        else:
            wandb.finish()


@contextmanager
def wandb_run_lifecycle(cfg: dict[str, Any]):
    run = init_wandb_run(cfg)
    try:
        yield run
    finally:
        finish_wandb_run(run)


def log_wandb_payload(payload: Mapping[str, Any], *, step: int | None = None) -> None:
    if step is None:
        wandb.log(dict(payload))
    else:
        wandb.log(dict(payload), step=int(step))


def log_wandb_run_payload(run: Any | None, payload: Mapping[str, Any], *, step: int | None = None) -> None:
    if run is None:
        return
    if step is None:
        run.log(dict(payload))
    else:
        run.log(dict(payload), step=int(step))


def log_wandb_run_payload_lazy(
    run: Any | None,
    payload_factory: Callable[[], Mapping[str, Any]],
    *,
    step: int | None = None,
) -> None:
    if run is None:
        return
    log_wandb_run_payload(run, payload_factory(), step=step)


def set_default_wandb_mode(mode: str = "disabled", *, silent: bool | None = None) -> None:
    os.environ.setdefault("WANDB_MODE", str(mode))
    if silent is not None:
        os.environ.setdefault("WANDB_SILENT", "true" if bool(silent) else "false")


def should_log_step(
    step: int,
    every: int,
    *,
    total_steps: int,
    always_log_last_step: bool,
    log_step_zero: bool = True,
) -> bool:
    """Shared log-cadence gate for trainer loops.

    Training scripts historically open-coded the same modulo/last-step check in
    several files. Keep the policy here so log cadence changes do not drift by
    trainer family.
    """

    if int(step) == 0 and not bool(log_step_zero):
        return False
    interval = max(1, int(every))
    return int(step) % interval == 0 or (bool(always_log_last_step) and int(step) == int(total_steps))


def should_log_from_config(cfg: dict[str, Any], step: int, key: str, *, log_step_zero: bool = True) -> bool:
    return should_log_step(
        step,
        int(cfg["logging"][key]),
        total_steps=int(cfg["train"]["steps"]),
        always_log_last_step=bool(cfg["logging"]["always_log_last_step"]),
        log_step_zero=log_step_zero,
    )


def should_log_video(cfg: dict[str, Any], step: int, *, log_step_zero: bool = True) -> bool:
    return should_log_from_config(cfg, step, "video_log_every", log_step_zero=log_step_zero)


def should_log_image(cfg: dict[str, Any], step: int, *, log_step_zero: bool = True) -> bool:
    return should_log_from_config(cfg, step, "image_log_every", log_step_zero=log_step_zero)


def should_log_scalar(cfg: dict[str, Any], step: int) -> bool:
    return should_log_from_config(cfg, step, "log_every")


def scalar_payload(
    cfg: Mapping[str, Any],
    result: Any,
    *,
    train_sequence_count: int,
    eval_sequence_count: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "Loss": result.loss.item(),
        "Loss/Reconstruction": result.recon_loss.item(),
        "Loss/CameraMotion": result.camera_motion_loss.item(),
        "Loss/CameraTemporal": result.camera_temporal_loss.item(),
        "Loss/CameraGlobal": result.camera_global_loss.item(),
        "Loss/BankRate": result.bank_rate_loss.item(),
        "TrainFrameCount": int(cfg["model"]["train_frame_count"]),
        "SequenceFrames": result.sequence_frame_count,
        "TrainSequenceCount": int(train_sequence_count),
        "EvalSequenceCount": int(eval_sequence_count),
        "InputSize": int(cfg["model"]["size"]),
        "RenderSize": int(cfg["render"]["render_size"]),
    }
    if result.camera_state is not None:
        payload.update(camera_state_payload(result.camera_state))
    for key, value in result.bank_rate_terms.items():
        payload[f"BankRate/{key}"] = value.item()
    for key, value in getattr(result, "aux_loss_terms", {}).items():
        payload[f"Loss/{key}"] = value.item()
    return payload


def flatten_scalar_metrics(prefix: str, value: Any, output: dict[str, float | int]) -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        output[prefix] = value
        return
    if isinstance(value, Mapping):
        for key, nested in value.items():
            flatten_scalar_metrics(f"{prefix}/{key}", nested, output)


def flattened_scalar_payload(prefix: str, row: Mapping[str, Any]) -> dict[str, float | int]:
    payload: dict[str, float | int] = {}
    flatten_scalar_metrics(prefix, row, payload)
    return payload


def mapped_metric_payload(
    metrics: Mapping[str, Any],
    key_map: tuple[tuple[str, str], ...],
    *,
    require: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for metric_key, payload_key in key_map:
        if metric_key in metrics:
            payload[payload_key] = metrics[metric_key]
        elif require:
            raise KeyError(metric_key)
    return payload


def log_wandb_row_outputs(
    row: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    metric_prefix: str,
    image_outputs: tuple[tuple[str, str], ...] = (),
    video_outputs: tuple[tuple[str, str], ...] = (),
) -> None:
    payload: dict[str, Any] = flattened_scalar_payload(metric_prefix, row)
    _add_existing_wandb_media(
        payload,
        cfg["output"],
        image_outputs=image_outputs,
        video_outputs=video_outputs,
    )
    if payload:
        log_wandb_payload(payload)


__all__ = [
    "finish_wandb_run",
    "flatten_scalar_metrics",
    "flattened_scalar_payload",
    "init_wandb_run",
    "log_wandb_payload",
    "log_wandb_run_payload_lazy",
    "log_wandb_row_outputs",
    "mapped_metric_payload",
    "scalar_payload",
    "set_default_wandb_mode",
    "should_log_from_config",
    "should_log_image",
    "should_log_scalar",
    "should_log_step",
    "should_log_video",
    "wandb_run_lifecycle",
]
