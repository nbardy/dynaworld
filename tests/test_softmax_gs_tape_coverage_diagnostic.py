from __future__ import annotations

import torch
import pytest

from research_experiments.softmax_gs.diagnose_tape_coverage import summarize_tape


def test_summarize_tape_reports_active_residual_ratios() -> None:
    selected_ids = torch.tensor(
        [
            [[0, 1], [1, -1]],
            [[-1, -1], [2, 3]],
        ],
        dtype=torch.int32,
    )
    selected_weights = torch.tensor(
        [
            [[0.3, 0.1], [0.2, 0.0]],
            [[0.0, 0.0], [0.4, 0.1]],
        ],
        dtype=torch.float32,
    )
    residual_weight = torch.tensor(
        [
            [0.1, 0.2],
            [0.0, 0.3],
        ],
        dtype=torch.float32,
    )
    final_alpha = torch.tensor(
        [
            [0.5, 0.4],
            [0.0, 0.8],
        ],
        dtype=torch.float32,
    )

    summary = summarize_tape(
        selected_ids,
        selected_weights,
        residual_weight,
        final_alpha,
        alpha_eps=1.0e-6,
    )

    assert summary["pixel_count"] == 4
    assert summary["active_pixel_count"] == 3
    assert summary["active_pixel_fraction"] == 0.75
    assert summary["selected_count_mean_active"] == pytest.approx(5.0 / 3.0)
    assert summary["active_residual_over_alpha_max"] == pytest.approx(0.5)
    assert summary["active_selected_mass_over_alpha_mean"] == pytest.approx((0.8 + 0.5 + 0.625) / 3.0)
