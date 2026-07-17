from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_real_video_timing_protocol_acceptance_report import (
    DEFAULT_OUT_DIR,
    EVIDENCE_ORDER,
    assert_real_video_timing_protocol_acceptance_report,
    run_report,
    summarize,
    verify_real_video_timing_protocol_acceptance_report,
)


def _artifact(summary: dict[str, object], benchmark: str) -> dict[str, object]:
    return {
        "path": f"{benchmark}.json",
        "benchmark": benchmark,
        "status": "ok",
        "verifier_errors": [],
        "summary": summary,
    }


def _valid_report() -> dict[str, object]:
    report: dict[str, object] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_timing_protocol_acceptance",
        "base_domain": "real-video projective interval timing acceptance",
        "theory_contract": (
            "This report promotes the fresh-process median timing protocol with warmup discard as the accepted "
            "timing contract. Strict warm-state failures remain diagnostic caveats, and this does not prove full "
            "goal completion."
        ),
        "proves_timing_acceptance": True,
        "does_not_prove_completion": True,
        "fresh_process_median_ratio_threshold": 1.0,
        "evidence": {
            "real_video_acceptance_envelope": _artifact(
                {
                    "broad_quality_distinct_youtube_id_count": 10,
                    "broad_media_distinct_youtube_id_count": 10,
                    "broad_frame_count_count": 4,
                    "all_quality_tethers_match": True,
                    "all_media_tethers_match": True,
                    "all_functional_rows_pass": True,
                    "max_support_rebins": 0,
                    "max_stale_refreshes": 0,
                },
                "star_uvt_projective_real_video_acceptance_envelope",
            ),
            "real_video_timing_variance_envelope": _artifact(
                {
                    "fresh_process_timing_acceptance_status": "pass",
                    "fresh_process_post_warmup_pair_count": 4,
                    "fresh_process_median_no_first_ratio": 0.56,
                    "fresh_process_median_projective_total_ratio": 0.84,
                    "fresh_process_median_feature_state_update_ratio": 0.85,
                    "strict_failure_count": 2,
                    "strict_failed_only_expected_timing": True,
                    "all_cache_support_clean": True,
                    "workload_explains_render_forward_miss_count": 0,
                    "strict_timing_win_claimed": False,
                },
                "star_uvt_projective_real_video_timing_variance_envelope",
            ),
            "real_video_frame_count_breadth_diagnostic": _artifact(
                {
                    "frame_count_breadth_accepted": True,
                    "strict_failed_only_expected_timing": True,
                    "no_first_growth_sublinear": True,
                    "no_first_timing_win": False,
                    "source_frame_count_count": 4,
                    "source_frame_growth_factor": 8.0,
                },
                "star_uvt_projective_real_video_frame_count_breadth_diagnostic",
            ),
        },
        "summary": {},
    }
    report["summary"] = summarize(report)
    return report


def test_real_video_timing_protocol_accepts_valid_fixture() -> None:
    report = _valid_report()

    assert verify_real_video_timing_protocol_acceptance_report(report) == []
    assert report["summary"]["final_timing_protocol_accepted"] is True
    assert report["summary"]["timing_acceptance_gap"] == 0
    assert report["summary"]["strict_warm_state_failure_count"] == 2


def test_real_video_timing_protocol_rejects_fresh_process_median_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["real_video_timing_variance_envelope"]["summary"][  # type: ignore[index]
        "fresh_process_median_no_first_ratio"
    ] = 1.01
    report["summary"] = summarize(report)

    errors = verify_real_video_timing_protocol_acceptance_report(report)

    assert any("fresh_process_median_no_first_ratio" in error for error in errors)


def test_real_video_timing_protocol_rejects_lost_broad_frame_count_context() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["real_video_acceptance_envelope"]["summary"]["broad_frame_count_count"] = 3  # type: ignore[index]
    report["summary"] = summarize(report)

    errors = verify_real_video_timing_protocol_acceptance_report(report)

    assert any("at least four frame counts" in error for error in errors)


def test_real_video_timing_protocol_rejects_workload_explained_strict_miss() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["real_video_timing_variance_envelope"]["summary"][  # type: ignore[index]
        "workload_explains_render_forward_miss_count"
    ] = 1
    report["summary"] = summarize(report)

    errors = verify_real_video_timing_protocol_acceptance_report(report)

    assert any("must not be explained by tile workload changes" in error for error in errors)


def test_real_video_timing_protocol_rejects_false_strict_timing_claim() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["real_video_timing_variance_envelope"]["summary"]["strict_timing_win_claimed"] = True  # type: ignore[index]
    report["summary"] = summarize(report)

    errors = verify_real_video_timing_protocol_acceptance_report(report)

    assert any("strict timing win must not be claimed" in error for error in errors)


def test_real_video_timing_protocol_report_reads_current_saved_artifacts() -> None:
    required = (
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_acceptance_envelope/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_timing_variance_envelope/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_frame_count_breadth_diagnostic/summary.json"),
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        pytest.skip(f"missing optional timing protocol inputs: {missing}")

    report = run_report()

    assert_real_video_timing_protocol_acceptance_report(report)
    assert report["summary"]["final_timing_protocol_accepted"] is True


def test_saved_real_video_timing_protocol_report_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(summary_json.read_text(encoding="utf-8"))

    assert_real_video_timing_protocol_acceptance_report(report)


def test_real_video_timing_protocol_evidence_order_is_stable() -> None:
    assert EVIDENCE_ORDER == (
        "real_video_acceptance_envelope",
        "real_video_timing_variance_envelope",
        "real_video_frame_count_breadth_diagnostic",
    )
