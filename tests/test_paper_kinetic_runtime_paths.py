from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENTRY_MODULES = (
    "paper_kinetic_active_track_program_factory",
    "paper_kinetic_fixed_site_material_step",
    "paper_kinetic_fixed_site_material_state",
    "paper_kinetic_lazy_program_bundles",
    "paper_kinetic_ragged_sample_plan",
    "paper_kinetic_replayable_observations",
    "paper_kinetic_runtime_paths",
    "paper_kinetic_sparse_sample_blocks",
    "paper_kinetic_step_target_frame_cache",
    "paper_kinetic_union_local_bar_assembly",
    "paper_kinetic_world_initializer",
)


def test_paper_kinetic_entries_import_with_canonical_train_pythonpath() -> None:
    """Protect the normal trainer launch contract, not the test runner's path."""

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src" / "train")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    imports = "; ".join(f"import {module}" for module in ENTRY_MODULES)
    completed = subprocess.run(
        (sys.executable, "-c", imports),
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
