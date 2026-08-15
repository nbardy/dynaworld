from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    ROOT
    / "research_experiments"
    / "world_foam_lane2"
    / "run_worldfoam_g4_v2_pilot.py"
)
EXECUTOR = ROOT / "src" / "train" / "worldfoam_native4d_public_quality_executor.py"
LANE2 = SCRIPT.parent
TRAIN = ROOT / "src" / "train"
for path in (LANE2, TRAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from verify_worldfoam_g4_v2_pilot import validate_pilot_receipt


def test_default_pilot_plan_is_source_only_and_emits_no_receipt() -> None:
    from worldfoam_g4_selected_ray_contract import DEFAULT_CONFIG, load_selected_ray_contract

    config, _base, _base_path = load_selected_ray_contract(DEFAULT_CONFIG)
    receipt = (ROOT / str(config["execution"]["pilot_receipt"])).resolve()
    before = (
        (receipt.stat().st_size, receipt.stat().st_mtime_ns)
        if receipt.is_file()
        else None
    )
    completed = subprocess.run(
        (sys.executable, "-I", "-S", str(SCRIPT)),
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    plan = json.loads(completed.stdout)
    after = (
        (receipt.stat().st_size, receipt.stat().st_mtime_ns)
        if receipt.is_file()
        else None
    )
    assert after == before
    assert plan["default_plan_imports_torch"] is False
    assert plan["default_plan_starts_subprocess"] is False
    assert plan["default_plan_samples_host_resources"] is False
    assert plan["default_plan_writes_files"] is False
    assert plan["build_or_rebuild_performed"] is False
    assert plan["routes"] == [
        "worldfoam_native4d",
        "worldfoam_framewise_replay",
    ]
    assert plan["optimizer_steps_per_route"] == 1
    assert plan["selected_target_pixels_per_route"] == 4096
    assert plan["heldout_frame_count_per_route"] == 300
    assert plan["heldout_spatial_track_count_per_route"] == 128
    assert plan["public_quality_evidence"] is False
    assert plan["pilot_only"] is True


def test_independent_verifier_rejects_an_unmeasured_summary() -> None:
    failures = validate_pilot_receipt(
        {
            "status": "pass",
            "pilot_only": True,
            "public_quality_evidence": False,
        },
        verify_files=False,
    )
    assert failures == ["pilot_top_level_keys_changed"]


def test_pilot_transition_is_not_training_finalization() -> None:
    tree = ast.parse(EXECUTOR.read_text(encoding="utf-8"), filename=str(EXECUTOR))
    methods = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "prepare_heldout_pilot_from_current_state"
    ]
    assert len(methods) == 1
    source = ast.get_source_segment(EXECUTOR.read_text(encoding="utf-8"), methods[0])
    assert source is not None
    assert "self._training_finalized" in source
    assert "self._optimizer_steps != 1" in source
    assert "self._prepare_heldout_generation()" in source
    assert "self._heldout_pilot_prepared = True" in source

