from __future__ import annotations

from typing import Any


MULTICAM_PHASE_CAPABILITIES = (
    "sample_multicam_clip",
    "multicam_bundle",
    "camera_rig",
    "rig_regularization_loss",
)


def trainer_has_capabilities(trainer: Any, names: tuple[str, ...]) -> bool:
    return all(hasattr(trainer, name) for name in names)


def trainer_uses_multicam_phase(trainer: Any) -> bool:
    return trainer_has_capabilities(trainer, MULTICAM_PHASE_CAPABILITIES)
