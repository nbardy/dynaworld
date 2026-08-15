from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from research_experiments.world_foam_lane2 import (
    generate_worldfoam_training_memory_assets as generator,
)
from research_experiments.world_foam_lane2.test_verify_worldfoam_training_memory_ablation import (  # noqa: E501
    _artifact,
)


def _write_artifact(path: Path) -> None:
    artifact, _config, _contract = _artifact()
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_g6_assets_require_and_summarize_the_exact_21_row_matrix(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "g6.json"
    _write_artifact(artifact)

    first = generator.build_assets(artifact)
    second = generator.build_assets(artifact)
    assert first == second
    assert set(first) == {
        "g6_native_memory_rows.csv",
        "g6_native_memory_table.tex",
        "g6_native_memory_scaling.svg",
        "g6_native_memory_summary.json",
    }
    summary = json.loads(first["g6_native_memory_summary.json"])
    assert summary["verifier_report"]["accepted"] is True
    assert len(summary["mode_frame_summary"]) == 7
    assert {
        (row["mode"], row["requested_frame_count"])
        for row in summary["mode_frame_summary"]
    } == {
        ("staged_sparse", 8),
        ("fused_union_v2", 8),
        ("fused_union_v2", 64),
        ("fused_union_v2", 300),
        ("per_frame_replay_sequential", 8),
        ("per_frame_replay_sequential", 64),
        ("per_frame_replay_sequential", 300),
    }
    assert b"MPS peak (GiB)" in first["g6_native_memory_table.tex"]
    assert b"Core F/B (s)" in first["g6_native_memory_table.tex"]
    assert b"Process E2E (s)" in first["g6_native_memory_table.tex"]
    assert b"Ordered-word work" in first["g6_native_memory_table.tex"]
    assert ET.fromstring(first["g6_native_memory_scaling.svg"]).tag.endswith(
        "svg"
    )

    rows = {
        (row["mode"], row["requested_frame_count"]): row
        for row in summary["mode_frame_summary"]
    }
    fused = rows[("fused_union_v2", 8)]
    replay = rows[("per_frame_replay_sequential", 8)]
    # Route-local transaction timings intentionally differ in scope and must
    # never be collapsed back into one deceptively comparable "step" column.
    assert fused["core_forward_backward_wall_time_seconds_median"] == 0.2
    assert replay["core_forward_backward_wall_time_seconds_median"] == 0.08
    assert fused["fresh_process_end_to_end_wall_time_seconds_median"] == 0.5
    assert replay["fresh_process_end_to_end_wall_time_seconds_median"] == 0.5
    assert fused["route_transaction_wall_time_seconds_median"] == 0.5
    assert replay["route_transaction_wall_time_seconds_median"] == 0.16


def test_g6_assets_reject_a_tampered_measured_row(tmp_path: Path) -> None:
    artifact = tmp_path / "g6.json"
    _write_artifact(artifact)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["rows"][0]["memory"]["sampled_mps_driver_peak_bytes"] = 9 * 2**30
    artifact.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="independently accepted"):
        generator.build_assets(artifact)
