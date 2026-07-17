from __future__ import annotations

import copy
import json
from pathlib import Path

from research_experiments.star_uvt_feature_tubes.preflight_support_birthsplit_gate import (
    evaluate_preflight,
    markdown_report,
    run_report,
)


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"artifact")
    return path


def _valid_config(tmp_path: Path) -> dict[str, object]:
    resume = _touch(tmp_path / "checkpoints" / "resume.pt")
    colorizer = _touch(tmp_path / "checkpoints" / "colorizer.pt")
    video = _touch(tmp_path / "video.mp4")
    cache_dir = tmp_path / "feature_cache"
    _touch(cache_dir / "sample.pt")
    return {
        "arch": "star_uvt_feature_overfit",
        "colorize": {
            "init_checkpoint": str(colorizer),
        },
        "data": {
            "video_path": str(video),
        },
        "feature_target": {
            "enabled": True,
            "image_vjp_mode": "analytic_sparse_grid_forward_batched",
            "materialization": "target_grid",
            "rgb_probe_checkpoint": str(colorizer),
        },
        "feature_uvt": {
            "render_mode": "feature_direct_gradcache_reduce_vec4",
            "tile_capacity": 128,
            "tube_count": 8192,
        },
        "features": {
            "cache_dir": str(cache_dir),
        },
        "support_birth_split": {
            "center_count": 8,
            "center_strategy": "farthest_xy",
            "enabled": True,
            "opacity": 0.4,
            "reallocate_tubes": 32,
            "support_precision_radius_px": 64.0,
            "support_radius_across_px": 64.0,
            "support_radius_along_px": 64.0,
            "support_radius_px": 64.0,
            "support_shape": "isotropic",
            "target_point_source": "uncovered_brightness",
            "temporal_radius_frames": 64.0,
        },
        "train": {
            "global_step_offset": 1500,
            "require_loss_decrease": True,
            "require_no_tile_overflow": True,
            "resume_checkpoint": str(resume),
            "resume_colorizer": False,
            "resume_optimizer": False,
            "steps": 50,
        },
    }


def _write_config(tmp_path: Path, cfg: dict[str, object]) -> Path:
    path = tmp_path / "config.jsonc"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    return path


def _check(payload: dict[str, object], check_id: str) -> dict[str, object]:
    checks = payload["checks"]
    assert isinstance(checks, list)
    for check in checks:
        if isinstance(check, dict) and check.get("id") == check_id:
            return check
    raise AssertionError(f"missing check {check_id}")


def test_preflight_ready_for_selected_birthsplit_gate(tmp_path: Path) -> None:
    payload = evaluate_preflight(_write_config(tmp_path, _valid_config(tmp_path)), expected_steps=50)

    assert payload["status"] == "ready"
    assert payload["blocking_check_ids"] == []
    assert _check(payload, "support_birth_split.support_radius_px")["status"] == "ok"
    assert _check(payload, "features.cache_files")["actual"] == 1


def test_preflight_blocks_when_required_artifacts_are_missing(tmp_path: Path) -> None:
    cfg = copy.deepcopy(_valid_config(tmp_path))
    cfg["train"]["resume_checkpoint"] = str(tmp_path / "missing" / "resume.pt")
    cfg["colorize"]["init_checkpoint"] = str(tmp_path / "missing" / "colorizer.pt")
    cfg["feature_target"]["rgb_probe_checkpoint"] = str(tmp_path / "missing" / "colorizer.pt")

    payload = evaluate_preflight(_write_config(tmp_path, cfg), expected_steps=50)

    assert payload["status"] == "blocked"
    assert set(payload["blocking_check_ids"]) == {
        "train.resume_checkpoint",
        "colorize.init_checkpoint",
        "feature_target.rgb_probe_checkpoint",
    }
    assert _check(payload, "data.video_path")["status"] == "ok"


def test_preflight_blocks_wrong_step_count(tmp_path: Path) -> None:
    cfg = copy.deepcopy(_valid_config(tmp_path))
    cfg["train"]["steps"] = 20

    payload = evaluate_preflight(_write_config(tmp_path, cfg), expected_steps=50)

    assert payload["status"] == "blocked"
    assert "train.steps" in payload["blocking_check_ids"]
    assert _check(payload, "train.steps")["actual"] == 20


def test_run_report_writes_json_and_markdown(tmp_path: Path) -> None:
    output_dir = tmp_path / "report"

    payload = run_report(
        config_path=_write_config(tmp_path, _valid_config(tmp_path)),
        output_dir=output_dir,
        expected_steps=50,
    )

    assert Path(payload["summary_json"]).exists()
    assert Path(payload["summary_md"]).exists()
    assert "# STAR UVT Birth/Split Preflight" in Path(payload["summary_md"]).read_text(encoding="utf-8")
    assert "| support_birth_split.opacity | ok |" in markdown_report(payload)
