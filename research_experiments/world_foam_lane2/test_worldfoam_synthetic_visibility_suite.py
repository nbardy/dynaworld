from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

from research_experiments.world_foam_lane2.verify_worldfoam_synthetic_visibility_suite import (
    EXPECTED_CAMERAS,
    EXPECTED_SCENES,
    verify_report,
)
from research_experiments.world_foam_lane2.worldfoam_synthetic_visibility_suite import (
    run_suite,
    write_suite,
)


def _fixture(tmp_path: Path) -> tuple[dict[str, object], Path]:
    report = run_suite(frame_count=5, pixel_count=5, oracle_samples=512)
    path = tmp_path / "summary.json"
    write_suite(report, path)
    return report, path


def test_synthetic_visibility_suite_covers_the_full_scene_camera_matrix(tmp_path: Path) -> None:
    report, path = _fixture(tmp_path)
    assert report["accepted"] is True
    assert len(report["layer_rows"]) == len(EXPECTED_SCENES) * len(EXPECTED_CAMERAS) * 4
    assert len(report["baseline_rows"]) == len(EXPECTED_SCENES) * len(EXPECTED_CAMERAS) * 3
    assert len(report["adaptive_rows"]) == len(EXPECTED_SCENES) * len(EXPECTED_CAMERAS)
    assert verify_report(report, report_path=path) == []


def test_synthetic_visibility_suite_establishes_gauge_and_crossing_contracts(tmp_path: Path) -> None:
    report, _ = _fixture(tmp_path)
    gauge = report["gauge_jacobian"]
    assert gauge["without_physical_jacobian_rgb_max_absolute_error"] > 20.0 * gauge[
        "with_physical_jacobian_rgb_max_absolute_error"
    ]
    aggregates = report["aggregates"]
    assert aggregates["crossing_worldfoam_rgb_mse_mean"] < 0.5 * aggregates[
        "crossing_sorted_rgb_mse_mean"
    ]
    assert aggregates["crossing_worldfoam_rgb_mse_mean"] < 0.5 * aggregates[
        "crossing_depth_marginal_rgb_mse_mean"
    ]


def test_verifier_rejects_tampered_metrics_and_acceptance(tmp_path: Path) -> None:
    report, path = _fixture(tmp_path)
    tampered = copy.deepcopy(report)
    deepest_row = next(row for row in tampered["layer_rows"] if row["layer_count"] == 128)
    deepest_row["rgb_mse"] *= 2.0
    deepest_row["rgb_max_absolute_error"] *= 2.0
    tampered["accepted"] = True
    errors = verify_report(tampered, report_path=path)
    assert any("rgb_psnr_db does not match rgb_mse" in error for error in errors)
    assert any("aggregate" in error for error in errors)


def test_verifier_rejects_figure_byte_drift(tmp_path: Path) -> None:
    report, path = _fixture(tmp_path)
    figure = path.parent / "figures" / report["figure_manifest"][0]["name"]
    figure.write_bytes(figure.read_bytes().replace(b"\n", b"\r\n", 1))
    errors = verify_report(report, report_path=path)
    assert any("figure hash mismatch" in error for error in errors)
    assert any("figure byte count mismatch" in error for error in errors)


def test_report_round_trips_strict_json_and_cli_verifier(tmp_path: Path) -> None:
    report, path = _fixture(tmp_path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["protocol_sha256"] == report["protocol_sha256"]
    verifier = Path(__file__).with_name("verify_worldfoam_synthetic_visibility_suite.py")
    completed = subprocess.run(
        [sys.executable, str(verifier), str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert '"verified":true' in completed.stdout
