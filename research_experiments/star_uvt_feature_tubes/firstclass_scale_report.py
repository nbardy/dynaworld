from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


try:
    from .report_artifacts import load_report_json, write_report_json, write_report_text
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import load_report_json, write_report_json, write_report_text


def _run_config(path: Path) -> tuple[dict[str, Any], Path | None]:
    from config_utils import load_config_file
    from trainer_registry import run_config_dict

    cfg = load_config_file(path)
    out_json = cfg.get("output", {}).get("out_json")
    return run_config_dict(cfg, path), (None if out_json is None else Path(out_json))


def _variant_label(source_path: Path | None, row: dict[str, Any]) -> str:
    suffixes: list[str] = []
    if row.get("colorize_pre_norm") is False or (source_path is not None and "no_prenorm" in source_path.name):
        suffixes.append("no-prenorm")
    activation = row.get("colorize_activation")
    if activation is not None and str(activation) != "sigmoid":
        suffixes.append(str(activation))
    hidden = row.get("colorize_hidden_dim")
    if hidden is not None:
        suffixes.append(f"hidden{int(hidden)}")
    init_gain = row.get("colorize_weight_init_gain")
    if init_gain is not None and abs(float(init_gain) - 4.0) > 1e-9:
        suffixes.append(f"gain{float(init_gain):g}")
    alpha_threshold = row.get("alpha_threshold")
    if alpha_threshold is not None and float(alpha_threshold) > 0.0:
        denominator = round(1.0 / float(alpha_threshold))
        if denominator > 0 and abs(float(alpha_threshold) - (1.0 / denominator)) < 1e-9:
            base = f"alpha>=1/{denominator}"
        else:
            base = f"alpha>={float(alpha_threshold):.6g}"
        return "+".join([base, *suffixes])
    if source_path is None:
        return "+".join(["default", *suffixes])
    name = source_path.name
    alpha_match = re.search(r"_alpha1_(\d+)_", name)
    if alpha_match is not None:
        return "+".join([f"alpha>=1/{alpha_match.group(1)}", *suffixes])
    return "+".join(["default", *suffixes])


def _fmt(value: Any, *, precision: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def _summary_row(row: dict[str, Any], source_path: Path | None) -> dict[str, Any]:
    mean = row.get("mean_timing_ms", {})
    last = row.get("last_timing_ms", {})
    tile = row.get("tile_stats", {})
    requested_mode = str(row.get("requested_render_mode") or "feature_direct_atomic")
    effective_mode = str(row.get("effective_render_mode") or requested_mode)
    frame_chunk_size = row.get("frame_chunk_size", row.get("frames"))
    return {
        "source_json": None if source_path is None else str(source_path),
        "variant": _variant_label(source_path, row),
        "requested_render_mode": requested_mode,
        "effective_render_mode": effective_mode,
        "mode_fallback_required": bool(row.get("mode_fallback_required", False)),
        "pass": bool(row.get("pass")),
        "frames": int(row["frames"]),
        "size": int(row["size"]),
        "tubes": int(row["tubes"]),
        "feature_dim": int(row["feature_dim"]),
        "colorize_pre_norm": row.get("colorize_pre_norm"),
        "colorize_activation": row.get("colorize_activation"),
        "colorize_hidden_dim": row.get("colorize_hidden_dim"),
        "colorize_weight_init_gain": row.get("colorize_weight_init_gain"),
        "frame_chunk_size": int(frame_chunk_size),
        "tile_capacity": int(row.get("tile_capacity", tile.get("tile_capacity", 0))),
        "steps": int(row["steps"]),
        "start_loss": row.get("start_loss"),
        "end_loss": row.get("end_loss"),
        "start_psnr": row.get("start_psnr"),
        "end_psnr": row.get("end_psnr"),
        "mean_step_ms": mean.get("step_ms"),
        "mean_forward_ms": mean.get("render_forward_ms"),
        "mean_colorize_loss_ms": mean.get("colorize_loss_ms"),
        "mean_backward_ms": mean.get("backward_ms"),
        "mean_optimizer_ms": mean.get("optimizer_ms"),
        "last_step_ms": last.get("step_ms"),
        "tile_overflow_sum": int(row.get("tile_overflow_sum", 0)),
        "tile_unstable_sum": int(row.get("tile_unstable_sum", 0)),
        "fixedbin_eligible": bool(row.get("fixedbin_eligible", tile.get("fixedbin_eligible", False))),
        "max_tile_count": tile.get("max_tile_count"),
        "p95_tile_count": tile.get("p95_tile_count"),
        "p99_tile_count": tile.get("p99_tile_count"),
        "mean_active_tile_count": tile.get("mean_active_tile_count"),
        "overflow_excess_tube_refs": tile.get("overflow_excess_tube_refs"),
        "active_tile_count": tile.get("active_tile_count"),
        "contact_sheet": row.get("contact_sheet"),
        "side_by_side_video": row.get("side_by_side_video"),
    }


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    headers = (
        "pass",
        "frames",
        "size",
        "tubes",
        "variant",
        "mode",
        "effective",
        "cap",
        "F",
        "pre-norm",
        "activation",
        "hidden",
        "gain",
        "chunk",
        "steps",
        "loss",
        "PSNR",
        "step ms",
        "fwd ms",
        "color/loss ms",
        "bwd ms",
        "overflow",
        "max tile",
        "p95 tile",
        "fixedbin",
    )
    lines = [
        "# STAR UVT Feature First-Class Scale Report",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        loss = f"{_fmt(row['start_loss'], precision=5)} -> {_fmt(row['end_loss'], precision=5)}"
        psnr = f"{_fmt(row['start_psnr'])} -> {_fmt(row['end_psnr'])}"
        values = (
            _fmt(row["pass"]),
            _fmt(row["frames"], precision=0),
            _fmt(row["size"], precision=0),
            _fmt(row["tubes"], precision=0),
            _fmt(row["variant"]),
            _fmt(row["requested_render_mode"]),
            _fmt(row["effective_render_mode"]),
            _fmt(row["tile_capacity"], precision=0),
            _fmt(row["feature_dim"], precision=0),
            _fmt(row.get("colorize_pre_norm")),
            _fmt(row.get("colorize_activation")),
            _fmt(row.get("colorize_hidden_dim"), precision=0),
            _fmt(row.get("colorize_weight_init_gain")),
            _fmt(row["frame_chunk_size"], precision=0),
            _fmt(row["steps"], precision=0),
            loss,
            psnr,
            _fmt(row["mean_step_ms"]),
            _fmt(row["mean_forward_ms"]),
            _fmt(row["mean_colorize_loss_ms"]),
            _fmt(row["mean_backward_ms"]),
            _fmt(row["tile_overflow_sum"], precision=0),
            _fmt(row.get("max_tile_count"), precision=0),
            _fmt(row.get("p95_tile_count")),
            _fmt(row.get("fixedbin_eligible")),
        )
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    for row in rows:
        label = (
            f"{row['frames']}f/{row['size']}px/{row['tubes']}t/"
            f"{row['variant']}/{row['requested_render_mode']}/cap{row['tile_capacity']}/"
            f"F{row['feature_dim']}/chunk{row['frame_chunk_size']}"
        )
        lines.append(f"- `{label}`: `{row['source_json']}`")
        if row.get("mode_fallback_required"):
            lines.append(f"  - effective mode: `{row['effective_render_mode']}` after overflow fallback requirement")
        if row.get("contact_sheet") is not None:
            lines.append(f"  - contact sheet: `{row['contact_sheet']}`")
        if row.get("side_by_side_video") is not None:
            lines.append(f"  - side-by-side video: `{row['side_by_side_video']}`")
    write_report_text(path, "\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", type=Path, nargs="*", default=[])
    parser.add_argument("--result-jsons", type=Path, nargs="*", default=[])
    parser.add_argument("--run", action="store_true", help="Run configs before summarizing.")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    if args.run:
        for config_path in args.configs:
            row, source_path = _run_config(config_path)
            rows.append(_summary_row(row, source_path))
    for result_path in args.result_jsons:
        rows.append(_summary_row(load_report_json(result_path), result_path))
    if not rows:
        raise SystemExit("No rows to summarize. Pass --result-jsons or --run --configs.")

    write_report_json(args.out_json, {"rows": rows})
    _write_markdown(args.out_md, rows)
    print(json.dumps({"rows": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
