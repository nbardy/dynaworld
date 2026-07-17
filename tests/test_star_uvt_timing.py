from __future__ import annotations

from star_uvt_timing import mean_timing_ms, timing_trace_summary_ms


def test_mean_timing_ms_infers_keys_from_first_row() -> None:
    assert mean_timing_ms([{"a_ms": 1.0, "b_ms": 3.0}, {"a_ms": 5.0, "b_ms": 7.0}]) == {
        "a_ms": 3.0,
        "b_ms": 5.0,
    }


def test_mean_timing_ms_preserves_empty_explicit_key_behavior() -> None:
    assert mean_timing_ms([], ("step_ms", "backward_ms")) == {
        "step_ms": 0.0,
        "backward_ms": 0.0,
    }
    assert mean_timing_ms([]) == {}


def test_timing_trace_summary_ms_reports_min_max_first_last() -> None:
    assert timing_trace_summary_ms(
        [{"step_ms": 4.0, "backward_ms": 3.0}, {"step_ms": 2.0, "backward_ms": 5.0}],
        ("step_ms", "backward_ms"),
    ) == {
        "step_ms": {"min": 2.0, "max": 4.0, "first": 4.0, "last": 2.0},
        "backward_ms": {"min": 3.0, "max": 5.0, "first": 3.0, "last": 5.0},
    }


def test_timing_trace_summary_ms_preserves_empty_summary_shape() -> None:
    assert timing_trace_summary_ms([], ("step_ms",)) == {
        "step_ms": {"min": None, "max": None, "first": None, "last": None}
    }
