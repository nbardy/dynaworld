from __future__ import annotations

import pytest
import torch

from research_experiments.world_foam_lane2.finite_element_material_fit import (
    DEFAULT_INTERVALS,
    HELDOUT_INTERVALS,
    TARGET_FIELDS,
    _bernstein_p2_value,
    evaluate_material_field,
    independent_target_outputs,
    restrict_material_interval,
    run_material_value_gate,
)
from research_experiments.world_foam_lane2.finite_element_material_transfer import (
    MaterialMode,
)


def _tensor(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


@pytest.mark.parametrize("mode", tuple(MaterialMode))
def test_full_interval_restriction_preserves_material(mode: MaterialMode) -> None:
    controls_by_mode = {
        MaterialMode.M0_P0_CONSTANT: (0.7, 0.0, 0.0),
        MaterialMode.M1_P0_AFFINE_RGB: (0.7, 0.0, 0.0),
        MaterialMode.M2_POSITIVE_BERNSTEIN_P1: (0.2, 1.1, 0.0),
        MaterialMode.M3_POSITIVE_BERNSTEIN_P2: (0.2, 1.4, 0.5),
        MaterialMode.M4_LOG_P1: (0.9, -0.2, 0.0),
        MaterialMode.M5_CONVEX_LOG_P2: (0.8, -0.35, 0.1),
    }
    controls = _tensor(controls_by_mode[mode])
    length = _tensor(1.3)
    color_front = _tensor((0.2, 0.4, 0.8))
    color_back = _tensor((0.8, 0.3, 0.1))
    local = restrict_material_interval(
        mode,
        controls,
        length,
        color_front,
        color_back,
        0.0,
        1.0,
    )

    torch.testing.assert_close(local.density_controls, controls)
    torch.testing.assert_close(local.length, length)
    torch.testing.assert_close(local.color_front, color_front)
    expected_back = (
        color_back
        if mode == MaterialMode.M1_P0_AFFINE_RGB
        else color_front
    )
    torch.testing.assert_close(local.color_back, expected_back)


def test_bernstein_p2_subdivision_matches_global_polynomial() -> None:
    controls = _tensor((0.08, 3.2, 0.18))
    local = restrict_material_interval(
        MaterialMode.M3_POSITIVE_BERNSTEIN_P2,
        controls,
        _tensor(1.7),
        _tensor((0.8, 0.2, 0.1)),
        _tensor((0.8, 0.2, 0.1)),
        0.17,
        0.83,
    )
    for local_xi in torch.linspace(0.0, 1.0, 17, dtype=torch.float64):
        global_xi = 0.17 + 0.66 * float(local_xi)
        expected = _bernstein_p2_value(controls, global_xi)
        actual = _bernstein_p2_value(
            local.density_controls, float(local_xi)
        )
        torch.testing.assert_close(actual, expected, atol=1.0e-12, rtol=1.0e-12)
    assert bool((local.density_controls >= 0.0).all())


@pytest.mark.parametrize(
    "mode",
    (MaterialMode.M4_LOG_P1, MaterialMode.M5_CONVEX_LOG_P2),
)
def test_log_polynomial_substitution_matches_global_density(
    mode: MaterialMode,
) -> None:
    controls = (
        _tensor((0.9, -0.2, 0.0))
        if mode == MaterialMode.M4_LOG_P1
        else _tensor((12.0, -12.0, 3.0))
    )
    local = restrict_material_interval(
        mode,
        controls,
        _tensor(1.7),
        _tensor((0.2, 0.7, 0.3)),
        _tensor((0.2, 0.7, 0.3)),
        0.13,
        0.79,
    )
    for local_xi in torch.linspace(0.0, 1.0, 17, dtype=torch.float64):
        global_xi = 0.13 + 0.66 * local_xi
        if mode == MaterialMode.M4_LOG_P1:
            expected_q = controls[0] * global_xi + controls[1]
            actual_q = (
                local.density_controls[0] * local_xi
                + local.density_controls[1]
            )
        else:
            expected_q = (
                controls[0] * global_xi.square()
                + controls[1] * global_xi
                + controls[2]
            )
            actual_q = (
                local.density_controls[0] * local_xi.square()
                + local.density_controls[1] * local_xi
                + local.density_controls[2]
            )
        torch.testing.assert_close(actual_q, expected_q)


def test_partial_chords_identify_density_shape() -> None:
    target = TARGET_FIELDS[0]
    controls = _tensor(target.density_controls)
    length = _tensor(target.length)
    color = _tensor(target.color_front)
    shaped = evaluate_material_field(
        target.mode, controls, length, color, color
    )
    constant = evaluate_material_field(
        MaterialMode.M0_P0_CONSTANT,
        _tensor((sum(target.density_controls) / 3.0, 0.0, 0.0)),
        length,
        color,
        color,
    )

    full_index = -1
    torch.testing.assert_close(
        shaped[full_index].element.beta,
        constant[full_index].element.beta,
    )
    partial_beta_error = torch.stack(
        [
            (left.element.beta - right.element.beta).abs()
            for left, right in zip(shaped[:-1], constant[:-1], strict=True)
        ]
    )
    assert float(partial_beta_error.max()) > 0.05


def test_train_and_heldout_chords_are_distinct() -> None:
    assert set(DEFAULT_INTERVALS).isdisjoint(HELDOUT_INTERVALS)
    assert all(0.0 <= start < stop <= 1.0 for start, stop in HELDOUT_INTERVALS)


@pytest.mark.parametrize("target", TARGET_FIELDS, ids=lambda target: target.name)
def test_independent_target_oracle_matches_production_evaluator(target) -> None:
    intervals = DEFAULT_INTERVALS + HELDOUT_INTERVALS
    expected_beta, expected_rgb = independent_target_outputs(target, intervals)
    controls = _tensor(target.density_controls)
    color_front = _tensor(target.color_front)
    color_back = _tensor(target.color_back)
    actual = evaluate_material_field(
        target.mode,
        controls,
        _tensor(target.length),
        color_front,
        color_back,
        intervals,
    )
    actual_beta = torch.stack([transfer.element.beta for transfer in actual])
    actual_rgb = torch.stack([transfer.element.m for transfer in actual])
    torch.testing.assert_close(
        actual_beta, expected_beta, atol=2.0e-12, rtol=2.0e-12
    )
    torch.testing.assert_close(
        actual_rgb, expected_rgb, atol=2.0e-12, rtol=2.0e-12
    )


def test_short_material_value_gate_separates_richer_density_from_p0() -> None:
    payload = run_material_value_gate(
        seeds=(17,),
        steps=100,
        learning_rate=0.04,
        refinement_steps=40,
    )

    assert payload["checks"]["all_rows_finite"]
    assert payload["checks"]["positive_p2_m3_beats_m0_100x"]
    assert payload["checks"]["positive_p2_m3_beats_m1_100x"]
    assert payload["checks"]["positive_p2_m3_challenges_m5"]
    assert payload["checks"]["positive_p2_m3_beats_m5_100x"]
    assert payload["checks"]["log_p2_m5_beats_m0_100x"]
    assert payload["checks"]["log_p2_m5_beats_m1_100x"]
    assert payload["checks"]["log_p2_m5_beats_m3_100x"]
    assert payload["checks"]["m3_m5_matched_serialized_bytes"]
    assert payload["promotion"]["winner"] is None
    assert not payload["promotion"]["eligible_for_native_4d_integration"]
    assert all(row["loss"] == row["heldout_loss"] for row in payload["rows"])


@pytest.mark.parametrize(
    ("start", "stop"),
    ((-0.1, 0.5), (0.5, 0.5), (0.8, 0.2), (0.0, 1.1)),
)
def test_invalid_partial_chord_fails_loudly(start: float, stop: float) -> None:
    with pytest.raises(ValueError, match="0 <= xi_start"):
        restrict_material_interval(
            MaterialMode.M0_P0_CONSTANT,
            _tensor((0.7, 0.0, 0.0)),
            _tensor(1.0),
            _tensor((0.2, 0.3, 0.4)),
            _tensor((0.2, 0.3, 0.4)),
            start,
            stop,
        )
