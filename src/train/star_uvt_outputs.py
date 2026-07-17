from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from config_utils import path_or_none
from train_artifacts import write_json
from train_logging import log_wandb_row_outputs


ContactSheetWriter = Callable[..., None]
SideBySideWriter = Callable[..., None]


def side_by_side_fps(output_cfg: dict[str, Any], data_cfg: dict[str, Any], *, key: str = "side_by_side_fps") -> float:
    configured = output_cfg.get(key)
    if configured is not None:
        return float(configured)
    data_fps = data_cfg.get("fps")
    return float(data_fps if data_fps is not None else 30.0)


def write_prediction_media(
    *,
    target_thwc: torch.Tensor,
    pred_thwc: torch.Tensor,
    output_cfg: dict[str, Any],
    data_cfg: dict[str, Any],
    contact_sheet_key: str = "contact_sheet",
    side_by_side_video_key: str = "side_by_side_video",
    side_by_side_fps_key: str = "side_by_side_fps",
    contact_sheet_writer: ContactSheetWriter | None = None,
    side_by_side_writer: SideBySideWriter | None = None,
) -> tuple[Path | None, Path | None]:
    contact_sheet = path_or_none(output_cfg.get(contact_sheet_key))
    side_by_side_video = path_or_none(output_cfg.get(side_by_side_video_key))
    if contact_sheet is None and side_by_side_video is None:
        return None, None

    if contact_sheet_writer is None or side_by_side_writer is None:
        from research_project.benchmarks.video_fit_comparison import write_contact_sheet, write_side_by_side_video

        contact_sheet_writer = contact_sheet_writer or write_contact_sheet
        side_by_side_writer = side_by_side_writer or write_side_by_side_video

    if contact_sheet is not None:
        contact_sheet_writer(
            contact_sheet,
            target_thwc,
            pred_thwc,
            None,
            max_frames=int(output_cfg["contact_sheet_frames"]),
            mode=str(output_cfg["contact_sheet_mode"]),
        )
    if side_by_side_video is not None:
        side_by_side_writer(
            side_by_side_video,
            target_thwc,
            pred_thwc,
            None,
            fps=side_by_side_fps(output_cfg, data_cfg, key=side_by_side_fps_key),
        )
    return contact_sheet, side_by_side_video


def write_row_json(row: dict[str, Any], out_json: Any) -> Path | None:
    output_path = path_or_none(out_json)
    if output_path is None:
        return None
    return write_json(output_path, row)


def write_row_json_and_print(row: dict[str, Any], out_json: Any) -> Path | None:
    output_path = write_row_json(row, out_json)
    print(json.dumps(row, indent=2, sort_keys=True))
    return output_path


def log_star_uvt_row_outputs(
    row: dict[str, Any],
    cfg: dict[str, Any],
    *,
    metric_prefix: str,
    image_outputs: tuple[tuple[str, str], ...] = (("contact_sheet", "media/contact_sheet"),),
    video_outputs: tuple[tuple[str, str], ...] = (("side_by_side_video", "media/side_by_side_video"),),
) -> None:
    log_wandb_row_outputs(
        row,
        cfg,
        metric_prefix=metric_prefix,
        image_outputs=image_outputs,
        video_outputs=video_outputs,
    )


__all__ = [
    "log_star_uvt_row_outputs",
    "side_by_side_fps",
    "write_prediction_media",
    "write_row_json",
    "write_row_json_and_print",
]
