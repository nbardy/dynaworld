from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any


from common import DYNAWORLD_ROOT, parse_csv_strings, resolve_dynaworld_path, write_json


SUMMARY_COLUMNS = [
    "run",
    "steps",
    "support_mode",
    "incidence_mode",
    "line_candidate_mode",
    "train_cameras",
    "heldout_camera",
    "train_camera_count",
    "wandb_run_name",
    "elements",
    "basis",
    "radius",
    "alpha_logit",
    "support_knn_k",
    "derived_support_scale",
    "derived_support_floor",
    "derived_support_weight_tau",
    "derived_support_normalize_trace",
    "model_derived_support_trace_mean",
    "model_derived_support_trace_p95",
    "wall_clock_sec",
    "wall_clock_min",
    "eval_psnr",
    "eval_l1",
    "heldout_eval_psnr",
    "heldout_eval_l1",
    "heldout_pose_is_calibrated",
    "alpha_coverage_050",
    "alpha_coverage_090",
    "heldout_alpha_coverage_050",
    "heldout_alpha_coverage_090",
    "projection_coverage_budget",
    "projection_radius_px_p50",
    "projection_radius_px_p95",
    "projection_anisotropy_p95",
    "heldout_projection_coverage_budget",
    "support_line_ratio_mean",
    "support_plane_ratio_mean",
    "support_surface_score_mean",
    "support_fiber_score_mean",
    "support_volume_score_mean",
    "witness_confidence_mean",
    "witness_weak_fraction",
    "heldout_witness_confidence_mean",
    "heldout_witness_weak_fraction",
    "motion_delta_mean",
    "xmap_occ",
    "heldout_xmap_occ",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def expand_roots(patterns: list[str]) -> list[Path]:
    roots: list[Path] = []
    for pattern in patterns:
        resolved = resolve_dynaworld_path(pattern)
        matches = glob.glob(str(resolved))
        roots.extend(Path(match) for match in matches)
    return sorted(set(roots))


def find_metric_dirs(roots: list[Path]) -> list[Path]:
    dirs: list[Path] = []
    for root in roots:
        if (root / "metrics.json").exists():
            dirs.append(root)
            continue
        dirs.extend(path.parent for path in root.rglob("metrics.json"))
    return sorted(set(dirs))


def nested_get(payload: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = payload
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def read_run(run_dir: Path) -> dict[str, Any]:
    metrics = load_json(run_dir / "metrics.json")
    config = load_json(run_dir / "config.json") if (run_dir / "config.json").exists() else {}
    logs = load_json(run_dir / "logs.json") if (run_dir / "logs.json").exists() else []
    wall_clock = load_json(run_dir / "wall_clock.json") if (run_dir / "wall_clock.json").exists() else {}
    probes = {}
    probe_path = run_dir / "probes" / "probe_summary.json"
    if probe_path.exists():
        probes = load_json(probe_path).get("probes", {})

    last_log = logs[-1] if isinstance(logs, list) and logs else {}
    row = {
        "run": str(run_dir.relative_to(DYNAWORLD_ROOT)) if run_dir.is_relative_to(DYNAWORLD_ROOT) else str(run_dir),
        "steps": nested_get(config, "train.steps", last_log.get("step")),
        "support_mode": nested_get(config, "model.support_mode", "screen_disk"),
        "incidence_mode": nested_get(config, "render.incidence_mode", "projected_conic"),
        "line_candidate_mode": nested_get(config, "render.line_candidate_mode", "all_pairs"),
        "train_cameras": ",".join(nested_get(config, "data.multicam_train_cameras", []) or []),
        "heldout_camera": nested_get(config, "data.multicam_heldout_camera", nested_get(config, "data.target_camera")),
        "train_camera_count": len(nested_get(config, "data.multicam_train_cameras", []) or []),
        "wandb_run_name": nested_get(config, "logging.wandb_run_name"),
        "wandb_tags": ",".join(nested_get(config, "logging.wandb_tags", []) or []),
        "output_dir": nested_get(config, "logging.output_dir"),
        "elements": nested_get(config, "model.num_elements", nested_get(config, "model.num_splats")),
        "basis": nested_get(config, "model.num_basis"),
        "radius": nested_get(config, "model.init_radius", nested_get(config, "model.init_scale")),
        "alpha_logit": nested_get(config, "model.init_alpha_logit"),
        "support_knn_k": nested_get(config, "model.support_knn_k"),
        "derived_support_scale": nested_get(config, "model.derived_support_scale"),
        "derived_support_floor": nested_get(config, "model.derived_support_floor"),
        "derived_support_weight_tau": nested_get(config, "model.derived_support_weight_tau"),
        "derived_support_normalize_trace": nested_get(config, "model.derived_support_normalize_trace"),
        "wall_clock_sec": wall_clock.get("elapsed_sec"),
        "wall_clock_min": wall_clock.get("elapsed_min"),
        "last_rgb_l1": last_log.get("rgb_l1"),
    }
    row.update(metrics)
    if probes:
        for probe_name, result in probes.items():
            delta = result.get("delta", {})
            row[f"probe_{probe_name}_delta_render_l1"] = delta.get("delta_render_l1")
            row[f"probe_{probe_name}_delta_target_l1"] = delta.get("delta_target_l1")
            row[f"probe_{probe_name}_delta_xmap_occ"] = delta.get("delta_xmap_occ")
    return row


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if abs(value) >= 1000 or (0 < abs(value) < 0.001):
            return f"{value:.3e}"
        return f"{value:.4f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(column)) for column in columns) + " |")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize material-gauge run directories.")
    parser.add_argument("roots", nargs="+", help="Run directories or glob roots containing metrics.json files.")
    parser.add_argument("--sort-by", default="eval_psnr")
    parser.add_argument("--ascending", action="store_true")
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--columns", default=",".join(SUMMARY_COLUMNS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots = expand_roots(args.roots)
    run_dirs = find_metric_dirs(roots)
    rows = [read_run(run_dir) for run_dir in run_dirs]
    rows.sort(
        key=lambda row: float(row.get(args.sort_by) or float("-inf")),
        reverse=not bool(args.ascending),
    )
    columns = parse_csv_strings(args.columns)
    table = markdown_table(rows, columns)
    print(table)

    if args.out_md:
        out_md = resolve_dynaworld_path(args.out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(table)
    if args.out_json:
        out_json = resolve_dynaworld_path(args.out_json)
        write_json(out_json, rows)


if __name__ == "__main__":
    main()
