from __future__ import annotations

from research_experiments.paper_runner_suite.coffee_martini_train2_holdout1_report import (
    EXPECTED_HELDOUT_CAMERAS,
    EXPECTED_TRAIN_CAMERAS,
    build_report,
    verify_report,
)


def test_current_coffee_martini_artifacts_prove_matched_seed17_pilot() -> None:
    report = build_report()

    assert verify_report(report) == []
    assert report["status"] == "ok"
    assert report["paper_rankable"] is False
    assert report["train_cameras"] == EXPECTED_TRAIN_CAMERAS
    assert report["heldout_cameras"] == EXPECTED_HELDOUT_CAMERAS
    assert report["gates"] == {
        "split_ok": True,
        "calibration_ok": True,
        "separate_train_and_heldout_metrics_ok": True,
        "matched_for_ranking": True,
        "three_seed_repeat_ok": False,
        "wandb_backing_ok": False,
        "world_tubes_promotion_policy_ok": False,
    }


def test_current_coffee_martini_report_has_three_explicit_metric_rows() -> None:
    report = build_report()
    rows = report["rows"]

    assert {row["representation"] for row in rows} == {
        "world_tubes_star_uvt",
        "dynamic_3dgs_fast_mac",
        "worldfoam_powerfoam_metal",
    }
    for row in rows:
        assert row["train_cameras"] == EXPECTED_TRAIN_CAMERAS
        assert row["heldout_cameras"] == EXPECTED_HELDOUT_CAMERAS
        assert row["train_psnr"] > 0.0
        assert row["heldout_psnr"] > 0.0


def test_report_rejects_a_false_paper_ranking_claim() -> None:
    report = build_report()
    report["paper_rankable"] = True

    assert "current mismatched protocol must not be paper-rankable" in verify_report(report)
