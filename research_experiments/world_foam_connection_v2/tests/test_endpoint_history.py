from __future__ import annotations

import torch

from world_foam_connection_v2.endpoint_history import (
    derive_constant_endpoint_history,
    derive_endpoint_generators,
    derive_piecewise_constant_endpoint_history,
)
from world_foam_connection_v2.fixtures import flat_translation_fixed_clips


def test_constant_history_is_derived_from_same_flow_and_cut_kinematics() -> None:
    fixture = flat_translation_fixed_clips(0.25)

    history, receipt = derive_constant_endpoint_history(
        fixture.ray,
        fixture.rate,
        fixture.flow,
        duration=fixture.time,
    )
    near, far = derive_endpoint_generators(
        fixture.ray,
        fixture.rate,
        fixture.flow,
    )

    torch.testing.assert_close(history.near_scalar[0], near.scalar)
    torch.testing.assert_close(history.near_source[0], near.source)
    torch.testing.assert_close(history.far_scalar[0], far.scalar)
    torch.testing.assert_close(history.far_source[0], far.source)
    assert receipt.derived_from_connection_inputs
    assert not receipt.uses_requested_frame_table
    assert receipt.retained_bytes > 0


def test_multistep_oracle_history_is_charged_as_requested_state() -> None:
    snapshots = tuple(flat_translation_fixed_clips(time) for time in (0.0, 0.1))

    history, receipt = derive_piecewise_constant_endpoint_history(
        interval_durations=torch.tensor((0.1, 0.1), dtype=torch.float64),
        rays=tuple(item.ray for item in snapshots),
        rates=tuple(item.rate for item in snapshots),
        flows=tuple(item.flow for item in snapshots),
    )

    assert history.step_count == 2
    assert receipt.uses_requested_frame_table
    assert receipt.retained_scalar_count == 18
    assert receipt.retained_bytes == 18 * 8
