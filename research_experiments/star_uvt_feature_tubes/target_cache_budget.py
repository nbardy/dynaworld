from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .report_artifacts import ROOT, load_report_json, write_report_json, write_report_text
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import ROOT, load_report_json, write_report_json, write_report_text


OBSERVED_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_cachedchunks_lr005_5step.json"
)
TARGET_GRID_RESULT = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_lr005_5step.json"
)


def _bytes(frames: int, feature_dim: int, resolution: int, dtype_bytes: int) -> int:
    return int(frames * feature_dim * resolution * resolution * dtype_bytes)


def _gib(num_bytes: int) -> float:
    return num_bytes / float(1024**3)


def _mib(num_bytes: int) -> float:
    return num_bytes / float(1024**2)


def _row(frames: int, feature_dim: int, resolution: int, dtype: str, dtype_bytes: int) -> dict[str, Any]:
    num_bytes = _bytes(frames, feature_dim, resolution, dtype_bytes)
    return {
        "frames": frames,
        "feature_dim": feature_dim,
        "resolution": resolution,
        "dtype": dtype,
        "target_cache_mib": _mib(num_bytes),
        "target_cache_gib": _gib(num_bytes),
        "relative_to_observed_64f_512px_f32": num_bytes / float(_bytes(64, 32, 512, 4)),
    }


def _build_report() -> dict[str, Any]:
    observed = load_report_json(OBSERVED_RESULT)
    target_grid = load_report_json(TARGET_GRID_RESULT)
    rows: list[dict[str, Any]] = []
    for dtype, dtype_bytes in (("float32", 4), ("float16", 2)):
        for frames in (32, 64, 128):
            for resolution in (256, 512, 768, 1024):
                for feature_dim in (32, 64):
                    rows.append(_row(frames, feature_dim, resolution, dtype, dtype_bytes))
    return {
        "gate": "star_uvt_vjepa_target_cache_budget",
        "report_date": "2026-05-19",
        "observed_cached_chunks_result": {
            "path": str(OBSERVED_RESULT.relative_to(ROOT)),
            "pass": observed.get("pass"),
            "frames": observed.get("frames"),
            "resolution": observed.get("size"),
            "feature_dim": observed.get("feature_dim"),
            "cached_target_mib": observed.get("feature_target", {}).get("cached_target_mib"),
            "feature_target_load_ms": observed.get("feature_target_load_ms"),
            "step_ms": observed.get("mean_timing_ms", {}).get("step_ms"),
            "feature_target_ms": observed.get("mean_timing_ms", {}).get("feature_target_ms"),
        },
        "observed_target_grid_result": {
            "path": str(TARGET_GRID_RESULT.relative_to(ROOT)),
            "pass": target_grid.get("pass"),
            "target_grid_mib": target_grid.get("feature_target", {}).get("target_grid_mib"),
            "feature_target_load_ms": target_grid.get("feature_target_load_ms"),
            "step_ms": target_grid.get("mean_timing_ms", {}).get("step_ms"),
            "feature_target_ms": target_grid.get("mean_timing_ms", {}).get("feature_target_ms"),
        },
        "rows": rows,
        "conclusion": {
            "cached_chunks_is_short_run_speed_path": True,
            "target_grid_is_lower_memory_speed_path": True,
            "observed_cache_mib": observed.get("feature_target", {}).get("cached_target_mib"),
            "f32_64f_512px_f64_equivalent_gib": _gib(_bytes(64, 64, 512, 4)),
            "f32_128f_512px_f32_equivalent_gib": _gib(_bytes(128, 32, 512, 4)),
            "f32_64f_1024px_f32_equivalent_gib": _gib(_bytes(64, 32, 1024, 4)),
            "next_action": (
                "Use target_grid for the current lower-memory V-JEPA target diagnostic. "
                "Keep cached_chunks as the dense render-grid reference, but do not scale it "
                "to 128f, F64, or 1024px without native-VJP loss or lower-precision target storage."
            ),
        },
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    observed = report["observed_cached_chunks_result"]
    target_grid = report["observed_target_grid_result"]
    selected = [
        row
        for row in report["rows"]
        if row["dtype"] == "float32"
        and row["frames"] in {64, 128}
        and row["feature_dim"] in {32, 64}
        and row["resolution"] in {512, 1024}
    ]
    lines = [
        "# STAR UVT V-JEPA Target Cache Budget",
        "",
        f"Date: {report['report_date']}",
        "",
        "## Observed Gate",
        "",
        f"- result: `{observed['path']}`",
        f"- pass: `{observed['pass']}`",
        f"- observed cache: `{observed['cached_target_mib']:.1f} MiB`",
        f"- target load/prep: `{observed['feature_target_load_ms']:.1f} ms`",
        f"- mean step / target: `{observed['step_ms']:.1f} ms` / `{observed['feature_target_ms']:.1f} ms`",
        "",
        "## Target-Grid Alternative",
        "",
        f"- result: `{target_grid['path']}`",
        f"- pass: `{target_grid['pass']}`",
        f"- target grid: `{target_grid['target_grid_mib']:.1f} MiB`",
        f"- target load/prep: `{target_grid['feature_target_load_ms']:.1f} ms`",
        f"- mean step / target: `{target_grid['step_ms']:.1f} ms` / `{target_grid['feature_target_ms']:.1f} ms`",
        "",
        "## Selected Float32 Scale Points",
        "",
        "| frames | F | res | target GiB | relative to observed |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in selected:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row["frames"]),
                    _fmt(row["feature_dim"]),
                    _fmt(row["resolution"]),
                    _fmt(row["target_cache_gib"]),
                    _fmt(row["relative_to_observed_64f_512px_f32"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            report["conclusion"]["next_action"],
            "",
        ]
    )
    write_report_text(path, "\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "outputs/benchmarks/2026-05-19_star_uvt_target_cache_budget.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=ROOT / "outputs/benchmarks/2026-05-19_star_uvt_target_cache_budget.md",
    )
    args = parser.parse_args()

    report = _build_report()
    write_report_json(args.out_json, report)
    _write_markdown(report, args.out_md)
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md)}, sort_keys=True))


if __name__ == "__main__":
    main()
