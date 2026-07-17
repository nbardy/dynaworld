from __future__ import annotations

from typing import Any


def scheduled_resample_target_cells(
    model_cfg: dict[str, Any],
    *,
    initial_cells: int,
    current_cells: int,
    step: int,
    total_steps: int,
) -> int | None:
    if model_cfg["resample_target_cells"] is not None:
        return int(model_cfg["resample_target_cells"])
    if model_cfg["resample_final_cells"] is None:
        return None

    start = int(model_cfg["resample_from_step"])
    stop = int(model_cfg["resample_until_step"] or max(int(total_steps), start + 1))
    if int(step) < start or int(step) >= stop:
        return int(current_cells)
    if stop - start <= 1:
        return int(model_cfg["resample_final_cells"])

    final_cells = int(model_cfg["resample_final_cells"])
    if int(initial_cells) <= 0:
        raise ValueError("initial_cells must be positive")
    growth = (float(final_cells) / float(initial_cells)) ** (1.0 / float(stop - start - 1))
    return max(1, int(float(initial_cells) * (growth ** float(int(step) - start))))


def should_resample_powerfoam_step(cfg: dict[str, Any], step: int) -> bool:
    return (
        int(cfg["model"]["resample_every"]) > 0
        and int(step) < int(cfg["train"]["steps"])
        and int(step) % int(cfg["model"]["resample_every"]) == 0
    )
