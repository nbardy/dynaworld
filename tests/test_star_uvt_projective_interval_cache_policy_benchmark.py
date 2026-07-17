from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research_experiments.star_uvt_feature_tubes.projective_interval_cache_policy_benchmark import (
    assert_projective_interval_cache_policy_report,
    verify_projective_interval_cache_policy_report,
)


INTERVAL_CACHE_POLICY_ARTIFACTS = (
    Path(
        "outputs/benchmarks/"
        "2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail00035_aggregate/summary.json"
    ),
    Path(
        "outputs/benchmarks/"
        "2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail00045_aggregate/summary.json"
    ),
    Path(
        "outputs/benchmarks/"
        "2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail0006_aggregate/summary.json"
    ),
    Path(
        "outputs/benchmarks/"
        "2026-05-24_star_uvt_projective_interval_cache_policy_8step_guard2_slack_budgeted_cap128_tail001_aggregate/summary.json"
    ),
)


def _valid_payload() -> dict[str, object]:
    return {
        "steps": 8,
        "refresh_every": 2,
        "support_guard_padding": 2.0,
        "support_guard_policy": "slack_budgeted",
        "support_guard_bisect_steps": 8,
        "support_stale_overshoot_epsilon": 0.0,
        "support_stale_tail_alpha_epsilon": 0.001,
        "tile_capacity": 128,
        "comparison": {
            "rebuild_delta_measured_minus_cadence": -3,
            "live_update_delta_measured_minus_cadence": 3,
            "support_rebin_delta_measured_minus_cadence": 0,
            "end_loss_delta_measured_minus_cadence": 0.0,
            "no_first_step_ms_delta_measured_minus_cadence": -2500.0,
        },
        "rows": [
            {
                "policy": "cadence",
                "status": "ok",
                "pass": True,
                "steps": 8,
                "start_loss": 0.2,
                "end_loss": 0.08,
                "loss_decreased": True,
                "no_first_step_ms": 4000.0,
                "tile_overflow_sum": 0,
                "max_tile_count": 70,
                "projective_interval_refresh_policy": "cadence",
                "projective_interval_refresh_every": 2,
                "projective_interval_cache_rebuilds": 4,
                "projective_interval_cache_live_updates": 4,
                "projective_interval_cache_staleness_checks": 4,
                "projective_interval_cache_stale_refreshes": 0,
                "projective_interval_cache_support_rebins": 0,
                "projective_interval_cache_last_support_tail_alpha_bound": 2.4e-4,
                "projective_interval_cache_max_support_tail_alpha_bound": 2.7e-4,
                "projective_interval_cache_visibility_stratifications": 0,
                "projective_interval_cache_fallback_marks": 0,
            },
            {
                "policy": "measured",
                "status": "ok",
                "pass": True,
                "steps": 8,
                "start_loss": 0.2,
                "end_loss": 0.08,
                "loss_decreased": True,
                "no_first_step_ms": 1500.0,
                "tile_overflow_sum": 0,
                "max_tile_count": 70,
                "projective_interval_refresh_policy": "measured",
                "projective_interval_refresh_every": 2,
                "projective_interval_cache_rebuilds": 1,
                "projective_interval_cache_live_updates": 7,
                "projective_interval_cache_staleness_checks": 7,
                "projective_interval_cache_stale_refreshes": 0,
                "projective_interval_cache_support_rebins": 0,
                "projective_interval_cache_last_support_tail_alpha_bound": 7.4e-4,
                "projective_interval_cache_max_support_tail_alpha_bound": 7.4e-4,
                "projective_interval_cache_visibility_stratifications": 0,
                "projective_interval_cache_fallback_marks": 0,
            },
        ],
    }


def _row(payload: dict[str, object], policy: str) -> dict[str, object]:
    rows = payload["rows"]
    assert isinstance(rows, list)
    for raw_row in rows:
        assert isinstance(raw_row, dict)
        if raw_row["policy"] == policy:
            return raw_row
    raise AssertionError(f"missing row {policy}")


def test_interval_cache_policy_report_verifier_accepts_valid_payload() -> None:
    payload = _valid_payload()

    assert verify_projective_interval_cache_policy_report(payload) == []
    assert_projective_interval_cache_policy_report(payload)


def test_interval_cache_policy_report_verifier_rejects_loss_drift() -> None:
    payload = _valid_payload()
    _row(payload, "measured")["end_loss"] = 0.081

    errors = verify_projective_interval_cache_policy_report(payload)

    assert any("end_loss must match cadence" in error for error in errors)


def test_interval_cache_policy_report_verifier_rejects_uncertified_tail_reuse() -> None:
    payload = _valid_payload()
    _row(payload, "measured")["projective_interval_cache_last_support_tail_alpha_bound"] = 0.002

    errors = verify_projective_interval_cache_policy_report(payload)

    assert any("last tail-alpha bound must be in" in error for error in errors)


def test_interval_cache_policy_report_verifier_rejects_missing_rebuild_win() -> None:
    payload = _valid_payload()
    _row(payload, "measured")["projective_interval_cache_rebuilds"] = 4
    payload["comparison"]["rebuild_delta_measured_minus_cadence"] = 0

    errors = verify_projective_interval_cache_policy_report(payload)

    assert any("fewer full rebuilds" in error for error in errors)


@pytest.mark.parametrize("summary_json", INTERVAL_CACHE_POLICY_ARTIFACTS)
def test_saved_interval_cache_policy_artifacts_satisfy_contract(summary_json: Path) -> None:
    if not summary_json.exists():
        pytest.skip(f"missing optional saved artifact: {summary_json}")

    payload = json.loads(summary_json.read_text(encoding="utf-8"))

    assert_projective_interval_cache_policy_report(payload)


def test_saved_interval_cache_policy_tail_epsilon_bracket_is_monotone() -> None:
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in INTERVAL_CACHE_POLICY_ARTIFACTS
        if path.exists()
    ]
    if len(reports) < 2:
        pytest.skip("need at least two saved interval cache-policy artifacts")
    reports = sorted(reports, key=lambda payload: float(payload["support_stale_tail_alpha_epsilon"]))

    rebins = []
    max_tail_bounds = []
    for payload in reports:
        assert_projective_interval_cache_policy_report(payload)
        measured = copy.deepcopy(_row(payload, "measured"))
        rebins.append(int(measured["projective_interval_cache_support_rebins"]))
        max_tail_bounds.append(float(measured["projective_interval_cache_max_support_tail_alpha_bound"]))

    assert rebins == sorted(rebins, reverse=True)
    assert max_tail_bounds == sorted(max_tail_bounds)
