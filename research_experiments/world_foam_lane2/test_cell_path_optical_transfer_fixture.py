from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from research_experiments.world_foam_lane2.cell_path_optical_transfer_fixture import (
    ELEMENT_MAX_ABS_ERROR,
    GRAD_MAX_ABS_ERROR,
    RENDER_MAX_ABS_ERROR,
    TransferElement,
    analytic_prefix_suffix_vjp,
    assert_summary,
    commutator_swap_probe,
    compose,
    constant_run_element,
    decode,
    finite_difference_vjp,
    make_three_run_fixture,
    make_two_run_fixture,
    render_word,
    run_all_checks,
    same_representation_replay_fixture,
)


SAVED_SUMMARY = Path("outputs/benchmarks/2026-07-08_worldfoam_cell_path_optical_transfer_summary.json")


def test_visibility_monoid_associative() -> None:
    fixture = make_three_run_fixture()
    elements = [
        constant_run_element(sigma, length, color)
        for sigma, length, color in zip(
            fixture["sigmas"],
            fixture["lengths"],
            fixture["colors"],
            strict=True,
        )
    ]

    left = compose(compose(elements[0], elements[1]), elements[2])
    right = compose(elements[0], compose(elements[1], elements[2]))

    assert float((left.beta - right.beta).abs().item()) <= ELEMENT_MAX_ABS_ERROR
    assert float((left.m - right.m).abs().max().item()) <= ELEMENT_MAX_ABS_ERROR


def test_constant_run_matches_manual_alpha() -> None:
    fixture = make_two_run_fixture()
    sigma = fixture["sigmas"][0]
    length = fixture["lengths"][0]
    color = fixture["colors"][0]
    background = fixture["background"]

    element = constant_run_element(sigma, length, color)
    alpha = 1.0 - torch.exp(-(sigma * length))
    expected = alpha * color + (1.0 - alpha) * background

    torch.testing.assert_close(decode(element, background), expected, atol=RENDER_MAX_ABS_ERROR, rtol=0.0)
    torch.testing.assert_close(element.beta, 1.0 - alpha, atol=ELEMENT_MAX_ABS_ERROR, rtol=0.0)
    torch.testing.assert_close(element.m, alpha * color, atol=ELEMENT_MAX_ABS_ERROR, rtol=0.0)


def test_cell_path_replay_equivalence() -> None:
    result = same_representation_replay_fixture()

    assert result["render_max_abs_error"] <= RENDER_MAX_ABS_ERROR
    assert result["element_beta_max_abs_error"] <= ELEMENT_MAX_ABS_ERROR
    assert result["element_m_max_abs_error"] <= ELEMENT_MAX_ABS_ERROR


def test_cell_path_vjp_matches_finite_difference() -> None:
    fixture = make_three_run_fixture()

    analytic = analytic_prefix_suffix_vjp(
        fixture["sigmas"],
        fixture["lengths"],
        fixture["colors"],
        fixture["background"],
        fixture["target"],
    )
    finite = finite_difference_vjp(
        fixture["sigmas"],
        fixture["lengths"],
        fixture["colors"],
        fixture["background"],
        fixture["target"],
    )

    for key in ("beta", "m", "delta_tau", "sigma", "length", "color_grad"):
        error = float((analytic[key] - finite[key]).abs().max().item())
        assert error <= GRAD_MAX_ABS_ERROR, key


def test_commutator_swap_bound() -> None:
    probe = commutator_swap_probe()

    torch.testing.assert_close(probe["measured"], probe["expected"], atol=RENDER_MAX_ABS_ERROR, rtol=0.0)
    assert probe["max_abs_error"] <= RENDER_MAX_ABS_ERROR
    assert probe["measured_norm"] == pytest.approx(probe["expected_norm"], abs=RENDER_MAX_ABS_ERROR)


def test_fixture_summary_schema() -> None:
    result = run_all_checks()

    assert_summary(result)
    assert result["status"] == "ok"
    assert result["dtype"] == "float64"
    assert result["fixture"] == "constant_density_owner_run_word"
    assert result["checks"] == {
        "alpha_equivalence": "ok",
        "commutator_swap": "ok",
        "monoid_associative": "ok",
        "replay_equivalence": "ok",
        "vjp_finite_difference": "ok",
    }
    assert result["max_errors"]["render"] <= RENDER_MAX_ABS_ERROR
    assert result["max_errors"]["element"] <= ELEMENT_MAX_ABS_ERROR
    assert result["max_errors"]["grad"] <= GRAD_MAX_ABS_ERROR


def test_independent_beta_and_m_gradients_are_checked() -> None:
    element = TransferElement(beta=torch.tensor(0.7, dtype=torch.float64), m=torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64))

    assert element.beta.dtype == torch.float64
    assert element.m.dtype == torch.float64


def test_saved_cell_path_optical_transfer_summary_satisfies_contract() -> None:
    if not SAVED_SUMMARY.exists():
        pytest.skip(f"missing optional saved artifact: {SAVED_SUMMARY}")

    result = json.loads(SAVED_SUMMARY.read_text(encoding="utf-8"))

    assert_summary(result)
