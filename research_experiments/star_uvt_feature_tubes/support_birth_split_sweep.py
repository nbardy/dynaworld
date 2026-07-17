from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    from .report_artifacts import (
        ROOT,
        STAR_UVT_ROOT,
        TRAIN_ROOT,
        load_optional_report_json,
        mean_timing_without_first,
        run_logged_subprocess,
        run_star_uvt_feature_trainer_subprocess,
        split_csv_floats,
        split_csv_ints,
        split_csv_strings,
        write_report_json,
        write_report_text,
    )
except ImportError:  # pragma: no cover - direct script execution
    from report_artifacts import (
        ROOT,
        STAR_UVT_ROOT,
        TRAIN_ROOT,
        load_optional_report_json,
        mean_timing_without_first,
        run_logged_subprocess,
        run_star_uvt_feature_trainer_subprocess,
        split_csv_floats,
        split_csv_ints,
        split_csv_strings,
        write_report_json,
        write_report_text,
    )
from config_utils import load_config_file


DENSE_DIAGNOSTIC = ROOT / "research_experiments/star_uvt_feature_tubes/dense_alpha_failure_diagnostic.py"
BASE_CONFIG = (
    ROOT
    / "src/train_configs/"
    "star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_uncovered_from1500_lr001_5step_media.jsonc"
)


BASELINE_DENSE_CASES = {
    "start1500": (
        "src/train_configs/"
        "star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_"
        "targetgrid_rgbprobe40_feature1_lr001_resume50_from1450_lr005sparse_media.jsonc"
    ),
    "topbirth32": "src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_from1500_lr001_5step_media.jsonc",
    "uncovered32": (
        "src/train_configs/"
        "star_uvt_feature_testvideo_64f_512_vjepa_target_birthsplit32_uncovered_from1500_lr001_5step_media.jsonc"
    ),
}


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _float_token(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value).replace(".", "p")


def _label(
    *,
    source: str,
    tubes: int,
    radius: float,
    tile_capacity: int,
    opacity: float | None = None,
    support_shape: str = "isotropic",
    radius_along: float | None = None,
    radius_across: float | None = None,
    precision_radius: float | None = None,
    center_strategy: str = "global_line",
    center_count: int = 1,
) -> str:
    radius_text = _float_token(radius)
    opacity_text = "" if opacity is None else f"_o{str(float(opacity)).replace('.', 'p')}"
    shape_text = ""
    has_explicit_shape_radii = radius_along is not None or radius_across is not None or precision_radius is not None
    if str(support_shape) != "isotropic" or has_explicit_shape_radii:
        radius_along = float(radius if radius_along is None else radius_along)
        radius_across = float(radius if radius_across is None else radius_across)
        precision_radius = float(radius if precision_radius is None else precision_radius)
        shape_text = (
            f"_{support_shape}"
            f"_a{_float_token(radius_along)}"
            f"_x{_float_token(radius_across)}"
            f"_p{_float_token(precision_radius)}"
        )
    center_text = "" if str(center_strategy) == "global_line" and int(center_count) == 1 else (
        f"_{center_strategy}_c{int(center_count)}"
    )
    return f"{source}_n{tubes}_r{radius_text}{shape_text}{center_text}{opacity_text}_cap{tile_capacity}"


def _case_config(
    base: dict[str, Any],
    *,
    source: str,
    tubes: int,
    radius: float,
    tile_capacity: int,
    opacity: float | None,
    support_shape: str,
    radius_along: float | None,
    radius_across: float | None,
    precision_radius: float | None,
    center_strategy: str,
    center_count: int,
    out_json: Path,
    checkpoint: Path,
    label: str,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["support_birth_split"]["target_point_source"] = source
    cfg["support_birth_split"]["reallocate_tubes"] = int(tubes)
    cfg["support_birth_split"]["support_radius_px"] = float(radius)
    cfg["support_birth_split"]["support_shape"] = str(support_shape)
    cfg["support_birth_split"]["support_radius_along_px"] = float(radius if radius_along is None else radius_along)
    cfg["support_birth_split"]["support_radius_across_px"] = float(radius if radius_across is None else radius_across)
    cfg["support_birth_split"]["support_precision_radius_px"] = float(radius if precision_radius is None else precision_radius)
    cfg["support_birth_split"]["center_strategy"] = str(center_strategy)
    cfg["support_birth_split"]["center_count"] = int(center_count)
    if opacity is not None:
        cfg["support_birth_split"]["opacity"] = float(opacity)
    cfg["feature_uvt"]["tile_capacity"] = int(tile_capacity)

    cfg["output"]["out_json"] = _display_path(out_json)
    cfg["output"]["checkpoint"] = _display_path(checkpoint)
    cfg["output"]["contact_sheet"] = None
    cfg["output"]["side_by_side_video"] = None
    cfg["output"]["rgb_probe_contact_sheet"] = None
    cfg["output"]["rgb_probe_side_by_side_video"] = None

    cfg["logging"]["wandb_mode"] = "offline"
    cfg["logging"]["wandb_run_name"] = f"star-uvt-birthsplit-sweep-{label}"
    tags = [str(tag) for tag in cfg["logging"].get("wandb_tags", [])]
    tags = [tag for tag in tags if not tag.startswith(("uncovered_brightness", "low_alpha", "top_brightness"))]
    opacity_tags = [] if opacity is None else [f"o{float(opacity):g}"]
    has_explicit_shape_radii = radius_along is not None or radius_across is not None or precision_radius is not None
    shape_tags = [] if str(support_shape) == "isotropic" and not has_explicit_shape_radii else [
        str(support_shape),
        f"a{float(radius if radius_along is None else radius_along):g}",
        f"x{float(radius if radius_across is None else radius_across):g}",
        f"p{float(radius if precision_radius is None else precision_radius):g}",
    ]
    cfg["logging"]["wandb_tags"] = [
        *tags,
        "birthsplit_sweep",
        source,
        f"n{tubes}",
        f"r{radius:g}",
        *shape_tags,
        str(center_strategy),
        f"c{int(center_count)}",
        *opacity_tags,
        f"cap{tile_capacity}",
    ]
    return cfg


def _row_from_trainer_result(
    *,
    label: str,
    source: str,
    tubes: int,
    radius: float,
    tile_capacity: int,
    opacity: float | None,
    support_shape: str,
    radius_along: float | None,
    radius_across: float | None,
    precision_radius: float | None,
    center_strategy: str,
    center_count: int,
    status: str,
    error: str,
    elapsed_sec: float,
    config_path: Path,
    json_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    data = load_optional_report_json(json_path)
    timing = data.get("mean_timing_ms", {}) if isinstance(data, dict) else {}
    last_timing = data.get("last_timing_ms", {}) if isinstance(data, dict) else {}
    tile_stats = data.get("tile_stats", {}) if isinstance(data, dict) else {}
    support = data.get("support_birth_split", {}) if isinstance(data, dict) else {}
    meta = support.get("target_point_meta", {}) if isinstance(support, dict) else {}
    return {
        "label": label,
        "target_point_source": source,
        "reallocate_tubes": int(tubes),
        "support_radius_px": float(radius),
        "support_shape": str(support_shape),
        "support_radius_along_px": float(radius if radius_along is None else radius_along),
        "support_radius_across_px": float(radius if radius_across is None else radius_across),
        "support_precision_radius_px": float(radius if precision_radius is None else precision_radius),
        "center_strategy": str(center_strategy),
        "center_count": int(center_count),
        "opacity": opacity,
        "tile_capacity": int(tile_capacity),
        "status": status,
        "error": error,
        "elapsed_sec": round(float(elapsed_sec), 3),
        "pass": data.get("pass") if isinstance(data, dict) else None,
        "mean_step_ms": timing.get("step_ms"),
        "no_first_step_ms": mean_timing_without_first(data, "step_ms") if isinstance(data, dict) else None,
        "last_step_ms": last_timing.get("step_ms"),
        "mean_backward_ms": timing.get("backward_ms"),
        "mean_render_ms": timing.get("render_forward_ms"),
        "start_loss": data.get("start_loss") if isinstance(data, dict) else None,
        "end_loss": data.get("end_loss") if isinstance(data, dict) else None,
        "start_feature_target_loss": data.get("start_feature_target_loss") if isinstance(data, dict) else None,
        "end_feature_target_loss": data.get("end_feature_target_loss") if isinstance(data, dict) else None,
        "start_rgb_probe_loss": data.get("start_rgb_probe_loss") if isinstance(data, dict) else None,
        "end_rgb_probe_loss": data.get("end_rgb_probe_loss") if isinstance(data, dict) else None,
        "end_rgb_probe_psnr": data.get("end_rgb_probe_psnr") if isinstance(data, dict) else None,
        "final_full_rgb_psnr": data.get("final_full_rgb_psnr") if isinstance(data, dict) else None,
        "tile_overflow_sum": data.get("tile_overflow_sum") if isinstance(data, dict) else None,
        "tile_unstable_sum": data.get("tile_unstable_sum") if isinstance(data, dict) else None,
        "max_tile_count": tile_stats.get("max_tile_count"),
        "p95_tile_count": tile_stats.get("p95_tile_count"),
        "alpha_sample_ms": support.get("alpha_sample_ms") if isinstance(support, dict) else None,
        "target_candidate_count": meta.get("candidate_count") if isinstance(meta, dict) else None,
        "selected_alpha_mean": meta.get("selected_alpha_mean") if isinstance(meta, dict) else None,
        "selected_alpha_max": meta.get("selected_alpha_max") if isinstance(meta, dict) else None,
        "selected_score_mean": meta.get("selected_score_mean") if isinstance(meta, dict) else None,
        "config_path": _display_path(config_path),
        "json_path": _display_path(json_path),
        "log_path": _display_path(log_path),
    }


def _run_trainer_case(
    *,
    base: dict[str, Any],
    source: str,
    tubes: int,
    radius: float,
    tile_capacity: int,
    opacity: float | None,
    support_shape: str,
    radius_along: float | None,
    radius_across: float | None,
    precision_radius: float | None,
    center_strategy: str,
    center_count: int,
    work_dir: Path,
    out_dir: Path,
    checkpoint_dir: Path,
    python: str,
    timeout_sec: int,
) -> dict[str, Any]:
    label = _label(
        source=source,
        tubes=tubes,
        radius=radius,
        tile_capacity=tile_capacity,
        opacity=opacity,
        support_shape=support_shape,
        radius_along=radius_along,
        radius_across=radius_across,
        precision_radius=precision_radius,
        center_strategy=center_strategy,
        center_count=center_count,
    )
    config_path = work_dir / "configs" / f"{label}.jsonc"
    log_path = work_dir / "logs" / f"{label}.log"
    json_path = out_dir / f"{work_dir.name}_{label}.json"
    checkpoint = checkpoint_dir / f"{work_dir.name}_{label}.pt"
    cfg = _case_config(
        base,
        source=source,
        tubes=tubes,
        radius=radius,
        tile_capacity=tile_capacity,
        opacity=opacity,
        support_shape=support_shape,
        radius_along=radius_along,
        radius_across=radius_across,
        precision_radius=precision_radius,
        center_strategy=center_strategy,
        center_count=center_count,
        out_json=json_path,
        checkpoint=checkpoint,
        label=label,
    )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    write_report_json(config_path, cfg)
    if json_path.exists():
        json_path.unlink()

    result = run_star_uvt_feature_trainer_subprocess(
        config_path=config_path,
        log_path=log_path,
        python=python,
        timeout_sec=timeout_sec,
        tmp_dir=work_dir / "tmp",
        env_defaults={"WANDB_MODE": "offline"},
        env_overrides={"STAR_UVT_TILE_CAPACITY": int(tile_capacity)},
    )
    return _row_from_trainer_result(
        label=label,
        source=source,
        tubes=tubes,
        radius=radius,
        tile_capacity=tile_capacity,
        opacity=opacity,
        support_shape=support_shape,
        radius_along=radius_along,
        radius_across=radius_across,
        precision_radius=precision_radius,
        center_strategy=center_strategy,
        center_count=center_count,
        status=result.status,
        error=result.error,
        elapsed_sec=result.elapsed_sec,
        config_path=config_path,
        json_path=json_path,
        log_path=log_path,
    )


def _run_dense_diagnostic(
    *,
    rows: list[dict[str, Any]],
    out_base: Path,
    work_dir: Path,
    python: str,
    timeout_sec: int,
    include_baselines: bool,
) -> dict[str, Any]:
    eligible_rows = [row for row in rows if row.get("status") == "ok" and row.get("pass") is True]
    dense_tile_capacity = int(eligible_rows[0]["tile_capacity"]) if eligible_rows else 128
    out_json = out_base.with_name(out_base.name + "_dense_support.json")
    out_md = out_base.with_name(out_base.name + "_dense_support.md")
    log_path = work_dir / "logs" / "dense_support.log"
    cases: list[str] = []
    include_baselines = bool(include_baselines and dense_tile_capacity == 128)
    if include_baselines:
        cases.extend(f"{label}={path}" for label, path in BASELINE_DENSE_CASES.items())
    for row in eligible_rows:
        cases.append(f"{row['label']}={row['config_path']}")
    command = [
        python,
        str(DENSE_DIAGNOSTIC),
        *sum((["--case", case] for case in cases), []),
        "--out-json",
        _display_path(out_json),
        "--out-md",
        _display_path(out_md),
        "--date",
        "2026-05-20",
    ]
    result = run_logged_subprocess(
        command,
        log_path=log_path,
        cwd=ROOT,
        timeout_sec=timeout_sec,
        pythonpath=(TRAIN_ROOT, STAR_UVT_ROOT),
        tmp_dir=work_dir / "tmp",
        env_overrides={"STAR_UVT_TILE_CAPACITY": dense_tile_capacity},
    )
    data = load_optional_report_json(out_json)
    dense_by_label: dict[str, dict[str, Any]] = {}
    if isinstance(data, dict):
        for case in data.get("cases", []):
            if isinstance(case, dict) and "label" in case:
                dense_by_label[str(case["label"])] = case
    return {
        "status": result.status,
        "error": result.error,
        "elapsed_sec": round(result.elapsed_sec, 3),
        "out_json": _display_path(out_json),
        "out_md": _display_path(out_md),
        "log_path": _display_path(log_path),
        "case_count": len(cases),
        "tile_capacity": dense_tile_capacity,
        "baseline_cases_included": include_baselines,
        "dense_by_label": dense_by_label,
    }


def _merge_dense(rows: list[dict[str, Any]], dense: dict[str, Any] | None) -> None:
    if not dense:
        return
    by_label = dense.get("dense_by_label", {})
    if not isinstance(by_label, dict):
        return
    for row in rows:
        case = by_label.get(str(row["label"]))
        if not isinstance(case, dict):
            continue
        thresholds = case.get("alpha_thresholds", {})
        alpha01 = thresholds.get("0.1", {}) if isinstance(thresholds, dict) else {}
        alpha05 = thresholds.get("0.5", {}) if isinstance(thresholds, dict) else {}
        row["dense_normal_psnr"] = case.get("normal_black_psnr")
        row["dense_forced_alpha_psnr"] = case.get("forced_alpha_1_psnr")
        row["dense_oracle_psnr"] = case.get("target_background_oracle_psnr")
        row["dense_alpha_mean"] = case.get("alpha_mean")
        row["dense_alpha_gt_0p1"] = alpha01.get("pixel_fraction") if isinstance(alpha01, dict) else None
        row["dense_alpha_gt_0p5"] = alpha05.get("pixel_fraction") if isinstance(alpha05, dict) else None


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    rows = report["rows"]
    columns = (
        "label",
        "status",
        "pass",
        "support_shape",
        "support_radius_along_px",
        "support_radius_across_px",
        "support_precision_radius_px",
        "center_strategy",
        "center_count",
        "opacity",
        "mean_step_ms",
        "no_first_step_ms",
        "mean_backward_ms",
        "mean_render_ms",
        "final_full_rgb_psnr",
        "dense_normal_psnr",
        "dense_forced_alpha_psnr",
        "dense_oracle_psnr",
        "dense_alpha_gt_0p1",
        "dense_alpha_gt_0p5",
        "max_tile_count",
        "p95_tile_count",
        "tile_overflow_sum",
        "selected_alpha_mean",
        "json_path",
    )
    lines = [
        "# STAR UVT Support Birth/Split Sweep",
        "",
        f"Generated: {report['generated']}",
        "",
        "Matched 5-step trainer rows from the sparse step-1500 feature-tube checkpoint.",
        "Rows vary target sampling source, reallocated tube count, support radius, and tile capacity.",
        "Dense columns come from the follow-up dense-support diagnostic and are the promotion gate.",
        "",
        "## Trainer And Dense Results",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(key)) for key in columns) + " |")
    dense = report.get("dense_diagnostic")
    if isinstance(dense, dict):
        dense_status = f"`{dense['status']}`"
        if dense.get("error"):
            dense_status = f"{dense_status} {dense['error']}"
        lines.extend(
            [
                "",
                "## Dense Diagnostic",
                "",
                f"- Status: {dense_status}",
                f"- Cases: `{dense['case_count']}`",
                f"- JSON: `{dense['out_json']}`",
                f"- Markdown: `{dense['out_md']}`",
                f"- Log: `{dense['log_path']}`",
            ]
        )
    best = report.get("best_dense_coverage_row")
    if isinstance(best, dict):
        lines.extend(
            [
                "",
                "## Read",
                "",
                "Best dense alpha `>0.1` row in this sweep:",
                "",
                f"- `{best['label']}` with alpha `>0.1` `{_fmt(best.get('dense_alpha_gt_0p1'))}`, "
                f"normal PSNR `{_fmt(best.get('dense_normal_psnr'))}`, forced-alpha PSNR "
                f"`{_fmt(best.get('dense_forced_alpha_psnr'))}`, and max tile "
                f"`{_fmt(best.get('max_tile_count'))}/{_fmt(best.get('tile_capacity'))}`.",
                "",
                "Promotion rule: keep only rows that raise dense alpha `>0.1` above the previous `0.411` without overflow and without collapsing forced-alpha/oracle metrics.",
            ]
        )
    lines.extend(
        [
            "",
            "## Inputs",
            "",
            f"- Base config: `{report['base_config']}`",
            f"- Work dir: `{report['work_dir']}`",
            f"- Support shapes: `{report['support_shapes']}`",
            f"- Center strategies: `{report['center_strategies']}`",
            f"- Center counts: `{report['center_counts']}`",
        ]
    )
    write_report_text(path, "\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=BASE_CONFIG)
    parser.add_argument("--target-sources", default="uncovered_brightness,low_alpha")
    parser.add_argument("--reallocate-tubes", default="32,64,128")
    parser.add_argument("--support-radii", default="32,64,96")
    parser.add_argument("--support-shapes", default="isotropic")
    parser.add_argument("--support-along-radii", default="")
    parser.add_argument("--support-across-radii", default="")
    parser.add_argument("--support-precision-radii", default="")
    parser.add_argument("--center-strategies", default="global_line")
    parser.add_argument("--center-counts", default="1")
    parser.add_argument("--opacities", default="", help="Optional comma-separated support_birth_split.opacity values.")
    parser.add_argument("--tile-capacities", default="128")
    parser.add_argument("--python", default=str(ROOT / ".venv/bin/python"))
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--dense-timeout-sec", type=int, default=600)
    parser.add_argument("--out-base", type=Path, default=ROOT / "outputs/benchmarks/2026-05-20_star_uvt_birthsplit_support_sweep")
    parser.add_argument("--skip-dense-diagnostic", action="store_true")
    parser.add_argument("--skip-baseline-dense-cases", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = load_config_file(args.base_config)
    out_base = args.out_base if args.out_base.is_absolute() else ROOT / args.out_base
    out_dir = out_base.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / f"{out_base.name}_work"
    checkpoint_dir = ROOT / "outputs/checkpoints"
    work_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    opacity_values: tuple[float | None, ...]
    parsed_opacities = split_csv_floats(args.opacities)
    opacity_values = tuple(parsed_opacities) if parsed_opacities else (None,)
    support_shapes = split_csv_strings(args.support_shapes)
    parsed_along_radii = split_csv_floats(args.support_along_radii)
    parsed_across_radii = split_csv_floats(args.support_across_radii)
    parsed_precision_radii = split_csv_floats(args.support_precision_radii)
    center_strategies = split_csv_strings(args.center_strategies)
    center_counts = split_csv_ints(args.center_counts)
    for source in split_csv_strings(args.target_sources):
        for tubes in split_csv_ints(args.reallocate_tubes):
            for radius in split_csv_floats(args.support_radii):
                for support_shape in support_shapes:
                    along_values: tuple[float | None, ...] = tuple(parsed_along_radii) if parsed_along_radii else (None,)
                    across_values: tuple[float | None, ...] = tuple(parsed_across_radii) if parsed_across_radii else (None,)
                    precision_values: tuple[float | None, ...] = (
                        tuple(parsed_precision_radii) if parsed_precision_radii else (None,)
                    )
                    for radius_along in along_values:
                        for radius_across in across_values:
                            for precision_radius in precision_values:
                                for center_strategy in center_strategies:
                                    for center_count in center_counts:
                                        for opacity in opacity_values:
                                            for tile_capacity in split_csv_ints(args.tile_capacities):
                                                label = _label(
                                                    source=source,
                                                    tubes=tubes,
                                                    radius=radius,
                                                    tile_capacity=tile_capacity,
                                                    opacity=opacity,
                                                    support_shape=support_shape,
                                                    radius_along=radius_along,
                                                    radius_across=radius_across,
                                                    precision_radius=precision_radius,
                                                    center_strategy=center_strategy,
                                                    center_count=center_count,
                                                )
                                                print(f"[birth-split-sweep] {label}")
                                                if args.dry_run:
                                                    rows.append(
                                                        {
                                                            "label": label,
                                                            "target_point_source": source,
                                                            "reallocate_tubes": tubes,
                                                            "support_radius_px": radius,
                                                            "support_shape": support_shape,
                                                            "support_radius_along_px": float(
                                                                radius if radius_along is None else radius_along
                                                            ),
                                                            "support_radius_across_px": float(
                                                                radius if radius_across is None else radius_across
                                                            ),
                                                            "support_precision_radius_px": float(
                                                                radius if precision_radius is None else precision_radius
                                                            ),
                                                            "center_strategy": center_strategy,
                                                            "center_count": center_count,
                                                            "opacity": opacity,
                                                            "tile_capacity": tile_capacity,
                                                            "status": "dry_run",
                                                        }
                                                    )
                                                    continue
                                                rows.append(
                                                    _run_trainer_case(
                                                        base=base,
                                                        source=source,
                                                        tubes=tubes,
                                                        radius=radius,
                                                        tile_capacity=tile_capacity,
                                                        opacity=opacity,
                                                        support_shape=support_shape,
                                                        radius_along=radius_along,
                                                        radius_across=radius_across,
                                                        precision_radius=precision_radius,
                                                        center_strategy=center_strategy,
                                                        center_count=center_count,
                                                        work_dir=work_dir,
                                                        out_dir=out_dir,
                                                        checkpoint_dir=checkpoint_dir,
                                                        python=str(args.python),
                                                        timeout_sec=int(args.timeout_sec),
                                                    )
                                                )

    dense_diagnostic: dict[str, Any] | None = None
    if not args.dry_run and not args.skip_dense_diagnostic:
        dense_diagnostic = _run_dense_diagnostic(
            rows=rows,
            out_base=out_base,
            work_dir=work_dir,
            python=str(args.python),
            timeout_sec=int(args.dense_timeout_sec),
            include_baselines=not bool(args.skip_baseline_dense_cases),
        )
        _merge_dense(rows, dense_diagnostic)

    passing_dense_rows = [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("pass") is True and row.get("tile_overflow_sum") == 0
    ]
    best_dense = None
    if passing_dense_rows:
        best_dense = max(
            passing_dense_rows,
            key=lambda row: (
                float(row.get("dense_alpha_gt_0p1") or -1.0),
                float(row.get("dense_normal_psnr") or -1.0),
            ),
        )

    report = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "base_config": str(args.base_config),
        "work_dir": _display_path(work_dir),
        "support_shapes": support_shapes,
        "center_strategies": center_strategies,
        "center_counts": center_counts,
        "rows": rows,
        "dense_diagnostic": dense_diagnostic,
        "best_dense_coverage_row": best_dense,
        "pass": all(row.get("status") in {"ok", "dry_run"} and row.get("pass", True) is not False for row in rows),
    }
    write_report_json(out_base.with_suffix(".json"), report)
    _write_markdown(report, out_base.with_suffix(".md"))
    print(json.dumps({"out_json": str(out_base.with_suffix(".json")), "out_md": str(out_base.with_suffix(".md"))}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
