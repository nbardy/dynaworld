"""Import bootstrap for the source-level WorldFoam kinetic compiler modules."""

from __future__ import annotations

from pathlib import Path

from external_paths import PROJECT_ROOT, ensure_sys_path


WORLD_FOAM_LANE2_ROOT = PROJECT_ROOT / "research_experiments" / "world_foam_lane2"


def ensure_worldfoam_lane2_research_path() -> Path:
    """Make the retained research compiler importable from ``PYTHONPATH=src/train``."""

    if not WORLD_FOAM_LANE2_ROOT.is_dir():
        raise FileNotFoundError(
            f"WorldFoam lane-2 research modules are missing: {WORLD_FOAM_LANE2_ROOT}"
        )
    ensure_sys_path(WORLD_FOAM_LANE2_ROOT)
    return WORLD_FOAM_LANE2_ROOT


__all__ = ["WORLD_FOAM_LANE2_ROOT", "ensure_worldfoam_lane2_research_path"]
