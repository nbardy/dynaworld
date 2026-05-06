from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SELECTED_BENCHMARK = (
    ROOT
    / "outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_raytrace_cech_aabb_4k_train_normaldistance_median_2026-05-03.json"
)
REGULAR_BENCHMARK = (
    ROOT
    / "outputs/benchmarks/powerfoam_metal_height_sv_texel_surface_raytrace_regular_triangulation_4k_train_normaldistance_median_2026-05-03.json"
)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise TypeError(f"{path} must contain a JSON list.")
    return rows


def by_cells(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["cells"]): row for row in rows}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_height_sv_raytrace_row(
    row: dict[str, Any],
    *,
    adjacency: str,
    max_steps_cap: int,
    max_total_ms: float | None = None,
) -> None:
    cells = int(row["cells"])
    require(row["renderer"] == "powerfoam_metal", f"{cells}: unexpected renderer {row['renderer']!r}")
    require(row["feature_mode"] == "oriented_height_sv_texel_surface", f"{cells}: not full height+SV mode")
    require(row["adjacency"] == adjacency, f"{cells}: benchmark adjacency is not {adjacency}")
    require(bool(row["raytrace"]), f"{cells}: benchmark is not raytrace")
    require(bool(row["backward"]) and bool(row["backward_supported"]), f"{cells}: backward is not supported")
    require(int(row["width"]) == 3840 and int(row["height"]) == 2160, f"{cells}: benchmark is not UHD 3840x2160")
    require(int(row["feature_dim"]) == 3, f"{cells}: unexpected feature_dim {row['feature_dim']}")
    if max_total_ms is not None:
        require(float(row["total_median_ms"]) <= max_total_ms, f"{cells}: total median exceeds {max_total_ms} ms")
    require(float(row["forward_median_ms"]) > 0.0, f"{cells}: forward median must be positive")
    require(float(row["backward_median_ms"]) > 0.0, f"{cells}: backward median must be positive")
    require(float(row["total_median_ms"]) > 0.0, f"{cells}: total median must be positive")
    require(int(row["raytrace_max_steps"]) <= max_steps_cap, f"{cells}: raytrace steps exceed cap {max_steps_cap}")


def verify_benchmarks(
    *,
    selected_path: Path,
    regular_path: Path,
    max_total_ms: float,
    max_steps_cap: int,
) -> list[dict[str, Any]]:
    selected = by_cells(load_rows(selected_path))
    regular = by_cells(load_rows(regular_path))
    require(set(selected) == {1024, 4096}, f"selected benchmark cells mismatch: {sorted(selected)}")
    require(set(regular) == {1024, 4096}, f"regular benchmark cells mismatch: {sorted(regular)}")
    summary = []
    for cells in sorted(selected):
        selected_row = selected[cells]
        regular_row = regular[cells]
        verify_height_sv_raytrace_row(
            selected_row,
            adjacency="cech_aabb",
            max_total_ms=max_total_ms,
            max_steps_cap=max_steps_cap,
        )
        verify_height_sv_raytrace_row(
            regular_row,
            adjacency="regular_triangulation",
            max_steps_cap=max_steps_cap,
        )
        require(
            float(selected_row["total_median_ms"]) < float(regular_row["total_median_ms"]),
            f"{cells}: selected cech_aabb total median is not faster than regular_triangulation",
        )
        summary.append(
            {
                "cells": cells,
                "selected_total_median_ms": float(selected_row["total_median_ms"]),
                "selected_forward_median_ms": float(selected_row["forward_median_ms"]),
                "selected_backward_median_ms": float(selected_row["backward_median_ms"]),
                "regular_total_median_ms": float(regular_row["total_median_ms"]),
                "selected_raytrace_max_steps": int(selected_row["raytrace_max_steps"]),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify saved PowerFoam Metal 4K benchmark artifacts.")
    parser.add_argument("--selected", type=Path, default=SELECTED_BENCHMARK)
    parser.add_argument("--regular", type=Path, default=REGULAR_BENCHMARK)
    parser.add_argument("--max-total-ms", type=float, default=1200.0)
    parser.add_argument("--max-steps-cap", type=int, default=64)
    args = parser.parse_args()
    summary = verify_benchmarks(
        selected_path=args.selected,
        regular_path=args.regular,
        max_total_ms=float(args.max_total_ms),
        max_steps_cap=int(args.max_steps_cap),
    )
    print(json.dumps({"ok": True, "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
