from __future__ import annotations

import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_anisotropic_tail_bound_verifier import (
    assert_anisotropic_tail_bound_report,
    run_verifier,
    verify_anisotropic_tail_bound_report,
)


ANISOTROPIC_TAIL_ARTIFACTS = (
    Path("outputs/benchmarks/2026-05-24_star_uvt_projective_anisotropic_tail_bound/summary.json"),
    Path(
        "outputs/benchmarks/"
        "2026-05-24_star_uvt_projective_anisotropic_tail_bound_metal_precision_rerun/summary.json"
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


def test_anisotropic_tail_bound_report_verifier_accepts_generated_cases() -> None:
    payload = run_verifier(tail_alpha_epsilon=1.0e-3)

    assert verify_anisotropic_tail_bound_report(payload) == []
    assert_anisotropic_tail_bound_report(payload)


def test_anisotropic_tail_bound_report_verifier_rejects_positive_error_above_bound() -> None:
    payload = run_verifier(tail_alpha_epsilon=1.0e-3)
    case = _case(payload, "rotated_precision_tail")
    case["max_abs_error"] = float(case["omitted_alpha_bound"]) * 2.0

    errors = verify_anisotropic_tail_bound_report(payload)

    assert any("exceeds omitted bound" in error for error in errors)


def test_anisotropic_tail_bound_report_verifier_rejects_unsummed_overlap_bound() -> None:
    payload = run_verifier(tail_alpha_epsilon=1.0e-3)
    diagonal = _case(payload, "diagonal_sigma_u1_v2_tail")
    summed = _case(payload, "two_trace_same_omitted_tile_sum")
    summed["omitted_alpha_bound"] = float(diagonal["omitted_alpha_bound"]) * 0.5

    errors = verify_anisotropic_tail_bound_report(payload)

    assert any("must exceed each single-tail bound" in error for error in errors)


def test_anisotropic_tail_bound_report_verifier_rejects_core_reuse() -> None:
    payload = run_verifier(tail_alpha_epsilon=1.0e-3)
    case = _case(payload, "anisotropic_core_loss_rejected")
    case["certified_reused"] = True

    errors = verify_anisotropic_tail_bound_report(payload)

    assert any("must reject stale reuse" in error for error in errors)


@pytest.mark.parametrize("summary_json", ANISOTROPIC_TAIL_ARTIFACTS)
def test_saved_anisotropic_tail_bound_artifacts_satisfy_contract(summary_json: Path) -> None:
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    payload = json.loads(summary_json.read_text(encoding="utf-8"))

    assert_anisotropic_tail_bound_report(payload)
