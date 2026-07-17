from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .report_artifacts import ROOT, load_report_json, read_report_csv, write_report_json, write_report_text
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import ROOT, load_report_json, read_report_csv, write_report_json, write_report_text

DEFAULT_MATRIX_DIR = (
    ROOT
    / "outputs/benchmarks/2026-05-19_star_uvt_logit_handoff_reduce_matrix_256_512_64f_32768t_f32"
)
DEFAULT_OUT_BASE = ROOT / "outputs/benchmarks/2026-05-19_star_uvt_logit_handoff_reduce_report"
MODES = ("logit_handoff", "logit_handoff_reduce", "logit_handoff_reduce_vec4")


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"missing numeric value for {key} in {row.get('case')}")
    return float(value)


def _int(row: dict[str, str], key: str) -> int:
    value = row.get(key, "")
    if value == "":
        raise ValueError(f"missing integer value for {key} in {row.get('case')}")
    return int(float(value))


def _bool(row: dict[str, str], key: str) -> bool:
    return row.get(key) == "True"


def _row_summary(row: dict[str, str]) -> dict[str, Any]:
    return {
        "mode": row["mode"],
        "size": _int(row, "size"),
        "pass": _bool(row, "pass"),
        "status": row["status"],
        "total_ms": _float(row, "total_ms"),
        "forward_ms": _float(row, "forward_ms"),
        "handoff_prep_ms": _float(row, "handoff_prep_ms"),
        "backward_ms": _float(row, "backward_ms"),
        "tile_overflow_sum": _int(row, "tile_overflow_sum"),
        "tile_unstable_sum": _int(row, "tile_unstable_sum"),
        "fixedbin_eligible": _bool(row, "fixedbin_eligible"),
        "json_path": row["json_path"],
        "log_path": row["log_path"],
    }


def _comparison(plain: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    result = {
        "mode": candidate["mode"],
        "size": candidate["size"],
        "total_ms": candidate["total_ms"],
        "backward_ms": candidate["backward_ms"],
        "forward_ms": candidate["forward_ms"],
        "handoff_prep_ms": candidate["handoff_prep_ms"],
        "total_delta_ms": candidate["total_ms"] - plain["total_ms"],
        "backward_delta_ms": candidate["backward_ms"] - plain["backward_ms"],
        "forward_delta_ms": candidate["forward_ms"] - plain["forward_ms"],
        "handoff_prep_delta_ms": candidate["handoff_prep_ms"] - plain["handoff_prep_ms"],
        "total_speedup_vs_plain": plain["total_ms"] / candidate["total_ms"],
        "backward_speedup_vs_plain": plain["backward_ms"] / candidate["backward_ms"],
    }
    result["total_delta_pct"] = 100.0 * result["total_delta_ms"] / plain["total_ms"]
    result["backward_delta_pct"] = 100.0 * result["backward_delta_ms"] / plain["backward_ms"]
    return result


def _tiny_parity(case_json: Path) -> list[dict[str, Any]]:
    payload = load_report_json(case_json)
    return [
        {
            "feature_dim": item["feature_dim"],
            "mode": item["backward_mode"],
            "pass": item["pass"],
            "max_backward_error": max(float(value) for value in item["backward_max_abs_errors"].values()),
            "forward_feature_max_abs_error": item["forward_feature_max_abs_error"],
            "forward_alpha_max_abs_error": item["forward_alpha_max_abs_error"],
            "tile_overflow_sum": item["tile_overflow_sum"],
            "tile_unstable_sum": item["tile_unstable_sum"],
        }
        for item in payload.get("tiny_parity", [])
    ]


def _fmt_ms(value: float) -> str:
    return f"{value:.1f}ms"


def _fmt_pct(value: float) -> str:
    return f"{value:+.1f}%"


def _markdown_table(rows: list[list[str]]) -> list[str]:
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    lines = []
    for row_index, row in enumerate(rows):
        lines.append("| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |")
        if row_index == 0:
            lines.append("| " + " | ".join("-" * widths[index] for index in range(len(row))) + " |")
    return lines


def build_report(matrix_dir: Path) -> dict[str, Any]:
    rows = [_row_summary(row) for row in read_report_csv(matrix_dir / "summary.csv")]
    grouped: dict[int, dict[str, dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["size"], {})[row["mode"]] = row
    missing: list[str] = []
    for size in sorted(grouped):
        for mode in MODES:
            if mode not in grouped[size]:
                missing.append(f"{size}px/{mode}")
    if missing:
        raise ValueError(f"missing matrix rows: {missing}")

    comparisons: list[dict[str, Any]] = []
    for size in sorted(grouped):
        plain = grouped[size]["logit_handoff"]
        comparisons.append(_comparison(plain, grouped[size]["logit_handoff_reduce"]))
        comparisons.append(_comparison(plain, grouped[size]["logit_handoff_reduce_vec4"]))

    parity: list[dict[str, Any]] = []
    for row in rows:
        parity.extend(_tiny_parity(ROOT / row["json_path"]))

    all_rows_ok = all(row["status"] == "ok" and row["pass"] for row in rows)
    zero_overflow = all(row["tile_overflow_sum"] == 0 and row["tile_unstable_sum"] == 0 for row in rows)
    all_tiny_parity_ok = all(item["pass"] for item in parity)
    by_size = {size: modes for size, modes in grouped.items()}
    vec4_256 = by_size[256]["logit_handoff_reduce_vec4"]
    plain_256 = by_size[256]["logit_handoff"]
    vec4_512 = by_size[512]["logit_handoff_reduce_vec4"]
    plain_512 = by_size[512]["logit_handoff"]
    scalar_512 = by_size[512]["logit_handoff_reduce"]
    validation = {
        "all_rows_ok": all_rows_ok,
        "zero_overflow_and_unstable": zero_overflow,
        "all_tiny_parity_ok": all_tiny_parity_ok,
        "vec4_total_improves_256": vec4_256["total_ms"] < plain_256["total_ms"],
        "vec4_backward_improves_256": vec4_256["backward_ms"] < plain_256["backward_ms"],
        "vec4_total_improves_512": vec4_512["total_ms"] < plain_512["total_ms"],
        "vec4_backward_not_worse_512": vec4_512["backward_ms"] <= plain_512["backward_ms"],
        "scalar_backward_regresses_512": scalar_512["backward_ms"] > plain_512["backward_ms"],
    }
    validation["pass"] = all(validation.values())
    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "matrix_dir": str(matrix_dir.relative_to(ROOT)),
        "rows": rows,
        "comparisons": comparisons,
        "tiny_parity": parity,
        "validation": validation,
        "decision": {
            "promote_to_trainer_default": False,
            "promote_scalar_reduce": False,
            "keep_vec4_as_diagnostic_candidate": True,
            "reason": (
                "logit_handoff_reduce_vec4 is correct and improves this direct synthetic matrix, "
                "but this is not first-class trainer proof and scalar reduce regresses 512px backward."
            ),
        },
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    rows = [
        [
            "size",
            "mode",
            "total",
            "forward",
            "prep",
            "backward",
            "total vs plain",
            "backward vs plain",
            "overflow",
        ]
    ]
    plain_by_size = {
        row["size"]: row for row in report["rows"] if row["mode"] == "logit_handoff"
    }
    comparison_by_key = {
        (item["size"], item["mode"]): item for item in report["comparisons"]
    }
    for row in sorted(report["rows"], key=lambda item: (item["size"], item["mode"])):
        comparison = comparison_by_key.get((row["size"], row["mode"]))
        if comparison is None:
            total_delta = "baseline"
            backward_delta = "baseline"
        else:
            total_delta = f"{_fmt_ms(comparison['total_delta_ms'])} ({_fmt_pct(comparison['total_delta_pct'])})"
            backward_delta = f"{_fmt_ms(comparison['backward_delta_ms'])} ({_fmt_pct(comparison['backward_delta_pct'])})"
        rows.append(
            [
                str(row["size"]),
                row["mode"],
                _fmt_ms(row["total_ms"]),
                _fmt_ms(row["forward_ms"]),
                _fmt_ms(row["handoff_prep_ms"]),
                _fmt_ms(row["backward_ms"]),
                total_delta,
                backward_delta,
                str(row["tile_overflow_sum"]),
            ]
        )

    parity_rows = [["mode", "F", "max backward err", "feature err", "alpha err"]]
    for item in report["tiny_parity"]:
        parity_rows.append(
            [
                item["mode"],
                str(item["feature_dim"]),
                f"{item['max_backward_error']:.3e}",
                f"{item['forward_feature_max_abs_error']:.3e}",
                f"{item['forward_alpha_max_abs_error']:.3e}",
            ]
        )

    validation_lines = [
        f"- `{key}`: `{value}`"
        for key, value in report["validation"].items()
    ]
    lines = [
        "# STAR UVT Logit-Handoff Tile-Slot Reducer Gate",
        "",
        f"Generated: {report['generated']}",
        "",
        "## Scope",
        "",
        "This gate combines the image-space-prep `direct_logit_handoff_backward` path with the existing stable-tile feature-gradient reducers.",
        "New benchmark modes are `logit_handoff_reduce` and `logit_handoff_reduce_vec4`; both fall back to direct atomics for unstable tiles.",
        "",
        "This is a direct synthetic renderer matrix, not a first-class trainer row.",
        "Forward and handoff-prep timings moved even though the edit targets backward, so the safest claim is the backward comparison plus parity/overflow checks.",
        "",
        "## Matrix",
        "",
        *(_markdown_table(rows)),
        "",
        "## Tiny Parity",
        "",
        *(_markdown_table(parity_rows)),
        "",
        "## Validation",
        "",
        *validation_lines,
        "",
        "## Decision",
        "",
        "- Keep `logit_handoff_reduce_vec4` as a live diagnostic candidate for a native-VJP/tile-slot bridge.",
        "- Do not promote `logit_handoff_reduce` scalar mode: its 512px backward row regresses.",
        "- Do not wire this path into the trainer by default until a first-class row proves end-to-end speed and quality.",
        "- Next speed gate: either a trainer-compatible native VJP route or a true scalar fixedbin/tile-slot contribution path that avoids duplicate STAR traversal.",
        "",
        "## Artifacts",
        "",
        f"- Matrix: `{report['matrix_dir']}/summary.md`",
        "- JSON: `outputs/benchmarks/2026-05-19_star_uvt_logit_handoff_reduce_report.json`",
    ]
    write_report_text(path, "\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", type=Path, default=DEFAULT_MATRIX_DIR)
    parser.add_argument("--out-base", type=Path, default=DEFAULT_OUT_BASE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    matrix_dir = args.matrix_dir if args.matrix_dir.is_absolute() else ROOT / args.matrix_dir
    out_base = args.out_base if args.out_base.is_absolute() else ROOT / args.out_base
    report = build_report(matrix_dir)
    write_report_json(out_base.with_suffix(".json"), report)
    write_markdown(report, out_base.with_suffix(".md"))
    print(json.dumps({"pass": report["validation"]["pass"], "out_base": str(out_base)}, sort_keys=True))


if __name__ == "__main__":
    main()
