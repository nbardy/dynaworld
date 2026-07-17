from __future__ import annotations

from research_experiments.paper_runner_suite.coffee_martini_matched_sweep_report import (
    EXPECTED_HELDOUT_CAMERAS,
    EXPECTED_SEEDS,
    EXPECTED_TRAIN_CAMERAS,
    REPRESENTATIONS,
    build_report,
    verify_report,
)


def test_saved_matched_sweep_satisfies_all_scope_gates() -> None:
    report = build_report()

    assert verify_report(report) == []
    assert report["status"] == "ok"
    assert report["paper_table_ready"] is True
    assert report["seeds"] == EXPECTED_SEEDS
    assert report["train_cameras"] == EXPECTED_TRAIN_CAMERAS
    assert report["heldout_cameras"] == EXPECTED_HELDOUT_CAMERAS
    assert all(report["gates"].values())


def test_saved_matched_sweep_has_three_rows_per_representation_and_world_tubes_wins_psnr() -> None:
    report = build_report()

    assert len(report["rows"]) == 9
    for representation in REPRESENTATIONS:
        assert len([row for row in report["rows"] if row["representation"] == representation]) == 3
    assert report["best_mean_heldout_psnr"] == "world_tubes"


def test_verifier_rejects_a_hidden_failed_gate() -> None:
    report = build_report()
    report["gates"]["media_ok"] = False

    assert "gate media_ok must be true" in verify_report(report)
