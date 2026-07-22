from __future__ import annotations

import json
from pathlib import Path

from config_utils import load_config_file
from research_experiments.paper_runner_suite.run_unified_paper_matrix import (
    DEFAULT_MATRIX,
    MatrixRun,
    expand_matrix,
    flatten_summary,
    write_artifacts,
)


def _evidence(offset: float) -> dict:
    return {
        "schema_version": 1,
        "quality": {
            "eval_psnr": 20.0 + offset,
            "eval_ssim": 0.8,
            "eval_l1": 0.1,
            "heldout_eval_psnr": 18.0 + offset,
            "heldout_eval_ssim": 0.7,
            "heldout_eval_l1": 0.15,
            "heldout_eval_lpips": 0.25,
        },
        "cost": {
            "optimizer_steps": 2,
            "target_frames": 4,
            "rasterized_frames": 4,
            "target_pixels": 30_720,
            "rasterized_pixels": 30_720,
            "parameter_count": 100,
            "trainable_parameter_count": 100,
            "parameter_bytes": 400,
            "optimizer_state_bytes": 800,
            "serialized_checkpoint_bytes": 1_024,
            "sampled_peak_current_allocated_bytes": 2_048,
            "sampled_peak_driver_allocated_bytes": 4_096,
            "elapsed_s": 1.0,
        },
        "timing": {
            "cold_compile_forward_s": 0.2,
            "steady_forward_s": 0.3,
            "steady_forward_calls": 1,
            "backward_s": 0.4,
            "backward_calls": 2,
            "optimizer_s": 0.1,
            "optimizer_calls": 2,
            "train_wall_s": 1.0,
        },
        "diagnostics": {"active_count": 100},
    }


def _summary() -> dict:
    return {
        "status": "complete",
        "protocol": {
            "name": "smoke",
            "dataset": {
                "sample_id": "scene_triplet",
                "train_cameras": ["a", "b"],
                "heldout_cameras": ["c"],
            },
        },
        "lanes": {
            "world_tubes": {"evidence": _evidence(1.0)},
            "worldfoam": {"evidence": _evidence(0.0)},
            "dynamic_3dgs": {"evidence": _evidence(-1.0)},
        },
    }


def test_submission_matrix_expands_to_the_frozen_seven_runs() -> None:
    runs = expand_matrix(load_config_file(DEFAULT_MATRIX))

    assert len(runs) == 7
    assert [run.seed for run in runs[:3]] == [17, 29, 43]
    assert sum(run.role == "pixel_matched_control" for run in runs) == 3
    assert sum(run.role == "sampler_control" for run in runs) == 1
    assert all(run.worldfoam_initializer == "base_config" for run in runs)


def test_matrix_can_select_scene_specific_worldfoam_initialization() -> None:
    protocol = DEFAULT_MATRIX.parent / "coffee_martini_protocol_smoke_2step.jsonc"
    runs = expand_matrix(
        {
            "runs": [
                {
                    "role": "breadth",
                    "protocol": str(protocol),
                    "seeds": [17],
                    "world_tubes_backward_policy": "fast_exploration",
                    "worldfoam_initializer": "video",
                }
            ]
        }
    )

    assert runs[0].worldfoam_initializer == "video"


def test_matrix_artifacts_are_generated_from_validated_evidence(tmp_path: Path) -> None:
    run = MatrixRun(
        role="mechanical_smoke",
        protocol_path=DEFAULT_MATRIX,
        seed=17,
        backward_policy="fast_exploration",
    )
    summary = _summary()
    rows = flatten_summary(run, summary)

    assert [row["lane"] for row in rows] == ["world_tubes", "worldfoam", "dynamic_3dgs"]
    artifacts = write_artifacts(
        tmp_path,
        "test_matrix",
        [{"run": run.as_dict(), "summary": summary}],
    )

    assert all((tmp_path / Path(path).name).exists() for path in artifacts.values())
    payload = json.loads((tmp_path / "paper_rows.json").read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 3
    assert len(payload["aggregated"]) == 3
    assert "LPIPS" in (tmp_path / "paper_table.tex").read_text(encoding="utf-8")
