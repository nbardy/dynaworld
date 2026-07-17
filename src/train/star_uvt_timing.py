from __future__ import annotations

from collections.abc import Sequence


TimingRow = dict[str, float]


def mean_timing_ms(timings: Sequence[TimingRow], keys: Sequence[str] | None = None) -> dict[str, float]:
    if keys is None:
        if not timings:
            return {}
        keys = tuple(timings[0].keys())
    divisor = float(len(timings)) if timings else 1.0
    return {key: sum(float(row[key]) for row in timings) / divisor for key in keys}


def timing_trace_summary_ms(timings: Sequence[TimingRow], keys: Sequence[str]) -> dict[str, dict[str, float | None]]:
    return {
        key: {
            "min": min(float(row[key]) for row in timings) if timings else None,
            "max": max(float(row[key]) for row in timings) if timings else None,
            "first": float(timings[0][key]) if timings else None,
            "last": float(timings[-1][key]) if timings else None,
        }
        for key in keys
    }


__all__ = ["mean_timing_ms", "timing_trace_summary_ms"]
