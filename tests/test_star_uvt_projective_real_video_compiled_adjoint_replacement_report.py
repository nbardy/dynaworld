from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_real_video_compiled_adjoint_replacement_report import (
    DEFAULT_OUT_DIR,
    EVIDENCE_ORDER,
    assert_real_video_compiled_adjoint_replacement_report,
    run_report,
    summarize,
    verify_real_video_compiled_adjoint_replacement_report,
)


def _artifact(summary: dict[str, object], benchmark: str) -> dict[str, object]:
    return {
        "path": f"{benchmark}.json",
        "benchmark": benchmark,
        "status": "ok",
        "verifier_errors": [],
        "summary": summary,
    }


def _case_row(idx: int, policy: str) -> dict[str, object]:
    measured = policy == "measured"
    return {
        "scene_id": f"scene_{idx:02d}_seg_000",
        "policy": policy,
        "path": f"cases/scene_{idx:02d}_seg_000_{policy}.json",
        "exists": True,
        "pass": True,
        "loss_decreased": True,
        "uses_projective_interval_main_path": True,
        "uses_rgb_direct_loss": True,
        "base_mode_is_practical_direct_atomic": True,
        "all_gradient_flags_present": True,
        "backward_timing_present": True,
        "render_forward_timing_present": True,
        "projective_interval_fallback_render_mode": "mixed",
        "projective_interval_cache_fallback_marks": 0,
        "projective_interval_cache_visibility_stratifications": 0,
        "projective_interval_cache_support_rebins": 0,
        "projective_interval_cache_stale_refreshes": 0,
        "projective_interval_cache_live_updates": 3 if measured else 2,
        "projective_interval_cache_rebuilds": 1 if measured else 2,
    }


def _valid_report() -> dict[str, object]:
    rows = []
    for idx in range(10):
        rows.append(_case_row(idx, "cadence"))
        rows.append(_case_row(idx, "measured"))
    checks = {
        "trainer_selects_projective_interval_route": True,
        "harness_defines_interval_autograd_function": True,
        "harness_forward_calls_interval_metal": True,
        "harness_backward_calls_direct_interval_metal_vjp": True,
        "harness_wrapper_applies_static_compiled_atlas": True,
        "bridge_backward_treats_visibility_and_bins_as_constants": True,
        "bridge_forward_consumes_interval_compressed_atlas": True,
    }
    report: dict[str, object] = {
        "status": "ok",
        "benchmark": "star_uvt_projective_real_video_compiled_adjoint_replacement",
        "base_domain": "source-distinct real-video projective interval trainer replacement",
        "theory_contract": (
            "This report proves the current practical Sensor-Time Trace Atlas trainer replacement through "
            "the interval Metal direct VJP with visibility order and tile membership treated as compiled "
            "constants. It is not deterministic compact static-STAR promotion and does not prove full goal "
            "completion."
        ),
        "proves_compiled_adjoint_replacement": True,
        "does_not_prove_completion": True,
        "evidence": {
            "real_video_broad10_trainer_matrix": _artifact(
                {
                    "distinct_youtube_id_count": 10,
                    "row_count": 20,
                    "all_rows_pass": True,
                    "all_rows_loss_decreased": True,
                    "all_rows_fallback_free": True,
                    "all_rows_visibility_stratification_free": True,
                    "all_measured_loss_matches_cadence": True,
                    "max_measured_support_rebins": 0,
                    "max_measured_stale_refreshes": 0,
                    "max_measured_vs_cadence_rebuild_ratio": 0.5,
                },
                "star_uvt_projective_real_video_multiscene_trainer_matrix",
            ),
            "real_video_broad10_quality_tether": _artifact(
                {
                    "distinct_youtube_id_count": 10,
                    "all_gradient_flags_present": True,
                    "all_measured_psnr_improves": True,
                },
                "star_uvt_projective_real_video_broad10_quality_tether",
            ),
            "real_video_broad10_media_tether": _artifact(
                {
                    "distinct_youtube_id_count": 10,
                    "all_gradient_flags_present": True,
                    "all_measured_psnr_improves": True,
                },
                "star_uvt_projective_real_video_multiscene_media_tether",
            ),
            "real_video_acceptance_envelope": _artifact(
                {"broad_frame_count_count": 4},
                "star_uvt_projective_real_video_acceptance_envelope",
            ),
            "real_video_timing_protocol_acceptance": _artifact(
                {"final_timing_protocol_accepted": True},
                "star_uvt_projective_real_video_timing_protocol_acceptance",
            ),
            "shared_work": _artifact(
                {
                    "orbit_payload_growth_ratio": 0.125,
                    "trained_shared_to_replay_interval_growth_ratio": 0.148,
                    "max_trained_final_backward_ms_ratio": 0.094,
                },
                "star_uvt_projective_shared_work_goal_audit",
            ),
        },
        "source_contract": {
            "source_paths": {
                "trainer": "src/train/star_uvt_feature_overfit_trainer.py",
                "trainer_harness": "tile_metal_autograd.py",
                "projective_trace": "projective_trace.py",
            },
            "checks": checks,
            "all_checks_pass": True,
        },
        "case_payloads": {
            "case_dir": "cases",
            "expected_case_count": len(rows),
            "rows": rows,
            "missing_paths": [],
        },
        "summary": {},
    }
    report["summary"] = summarize(report)
    return report


def test_compiled_adjoint_replacement_accepts_valid_fixture() -> None:
    report = _valid_report()

    assert verify_real_video_compiled_adjoint_replacement_report(report) == []
    assert_real_video_compiled_adjoint_replacement_report(report)
    assert report["summary"]["final_compiled_adjoint_replacement_accepted"] is True  # type: ignore[index]
    assert report["summary"]["compiled_trainer_replacement_gap"] == 0  # type: ignore[index]


def test_compiled_adjoint_replacement_rejects_source_contract_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["source_contract"]["checks"]["harness_backward_calls_direct_interval_metal_vjp"] = False  # type: ignore[index]
    report["source_contract"]["all_checks_pass"] = False  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_compiled_adjoint_replacement_report(report)

    assert any("source contract check harness_backward_calls_direct_interval_metal_vjp" in error for error in errors)


def test_compiled_adjoint_replacement_rejects_case_main_path_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["case_payloads"]["rows"][1]["uses_projective_interval_main_path"] = False  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_compiled_adjoint_replacement_report(report)

    assert any("uses_projective_interval_main_path must be true" in error for error in errors)
    assert any("final compiled-adjoint replacement must be accepted" in error for error in errors)


def test_compiled_adjoint_replacement_rejects_source_count_regression() -> None:
    report = copy.deepcopy(_valid_report())
    report["evidence"]["real_video_broad10_trainer_matrix"]["summary"]["distinct_youtube_id_count"] = 9  # type: ignore[index]
    report["summary"] = summarize(report)  # type: ignore[arg-type]

    errors = verify_real_video_compiled_adjoint_replacement_report(report)

    assert any("at least ten source-distinct videos" in error for error in errors)


def test_compiled_adjoint_replacement_rejects_stale_summary() -> None:
    report = copy.deepcopy(_valid_report())
    report["summary"]["case_payload_count"] = 19  # type: ignore[index]

    errors = verify_real_video_compiled_adjoint_replacement_report(report)

    assert any("summary case_payload_count mismatch" in error for error in errors)


def test_compiled_adjoint_replacement_report_reads_current_saved_artifacts() -> None:
    required = (
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_trainer_matrix_broad10/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_broad10_quality_tether/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_broad10_media_tether/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_acceptance_envelope/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_timing_protocol_acceptance/summary.json"),
        Path("outputs/benchmarks/2026-05-25_star_uvt_projective_shared_work_goal_audit/summary.json"),
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        pytest.skip(f"missing optional compiled-adjoint replacement inputs: {missing}")

    report = run_report()

    assert_real_video_compiled_adjoint_replacement_report(report)
    assert report["summary"]["case_payload_count"] == 20


def test_saved_compiled_adjoint_replacement_artifact_satisfies_contract() -> None:
    summary_json = DEFAULT_OUT_DIR / "summary.json"
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    report = json.loads(summary_json.read_text(encoding="utf-8"))

    assert_real_video_compiled_adjoint_replacement_report(report)


def test_compiled_adjoint_replacement_evidence_order_is_stable() -> None:
    assert EVIDENCE_ORDER == (
        "real_video_broad10_trainer_matrix",
        "real_video_broad10_quality_tether",
        "real_video_broad10_media_tether",
        "real_video_acceptance_envelope",
        "real_video_timing_protocol_acceptance",
        "shared_work",
    )
