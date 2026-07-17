from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_tail_alpha_image_error_verifier import (
    assert_tail_alpha_image_error_report,
    run_verifier,
    verify_tail_alpha_image_error_report,
)


TAIL_ALPHA_ARTIFACTS = (
    Path("outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error/summary.json"),
    Path(
        "outputs/benchmarks/"
        "2026-05-24_star_uvt_projective_tail_alpha_image_error_tail00035_aggregate/summary.json"
    ),
    Path(
        "outputs/benchmarks/"
        "2026-05-24_star_uvt_projective_tail_alpha_image_error_metal_precision_rerun/summary.json"
    ),
)


def _case(payload: dict[str, object], name: str) -> dict[str, object]:
    cases = payload["cases"]
    assert isinstance(cases, list)
    for raw_case in cases:
        assert isinstance(raw_case, dict)
        if raw_case["name"] == name:
            return raw_case
    raise AssertionError(f"missing case {name}")


def test_tail_alpha_image_error_report_verifier_accepts_generated_cases() -> None:
    payload = run_verifier(tail_alpha_epsilon=3.5e-4)

    assert verify_tail_alpha_image_error_report(payload) == []
    assert_tail_alpha_image_error_report(payload)


def test_tail_alpha_image_error_report_verifier_rejects_positive_error_above_bound() -> None:
    payload = run_verifier(tail_alpha_epsilon=3.5e-4)
    case = _case(payload, "axis_r4_sigma1_opacity05")
    case["max_abs_error"] = float(case["support_tail_alpha_bound"]) * 2.0

    errors = verify_tail_alpha_image_error_report(payload)

    assert any("exceeds tail bound" in error for error in errors)


def test_tail_alpha_image_error_report_verifier_rejects_core_reuse() -> None:
    payload = run_verifier(tail_alpha_epsilon=3.5e-4)
    case = _case(payload, "core_loss_rejected")
    case["certified_rebinned"] = False
    case["certified_reused"] = True

    errors = verify_tail_alpha_image_error_report(payload)

    assert any("must reject stale reuse" in error for error in errors)


def test_tail_alpha_image_error_report_verifier_rejects_missing_aggregate_case() -> None:
    payload = run_verifier(tail_alpha_epsilon=3.5e-4)
    payload["cases"] = [
        case
        for case in payload["cases"]
        if isinstance(case, dict) and case["name"] != "overlapping_tail_aggregate_rejected"
    ]

    errors = verify_tail_alpha_image_error_report(payload)

    assert any("missing required cases" in error for error in errors)


@pytest.mark.parametrize("summary_json", TAIL_ALPHA_ARTIFACTS)
def test_saved_tail_alpha_image_error_artifacts_satisfy_contract(summary_json: Path) -> None:
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    payload = json.loads(summary_json.read_text(encoding="utf-8"))

    assert_tail_alpha_image_error_report(payload)
