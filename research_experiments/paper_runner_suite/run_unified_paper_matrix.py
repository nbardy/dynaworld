from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import load_config_file, serialize_config_value
from paper_training_protocol import resolve_paper_training_protocol
from research_experiments.paper_runner_suite import run_unified_paper_ablation as single


DEFAULT_MATRIX = (
    ROOT / "src" / "train_configs" / "paper_protocols" / "world_tubes_submission_matrix_v1.jsonc"
)
DEFAULT_OUT_DIR = ROOT / "outputs" / "benchmarks" / "world_tubes_submission_matrix_v1"
LANE_ORDER = ("world_tubes", "worldfoam", "dynamic_3dgs")


@dataclass(frozen=True)
class MatrixRun:
    role: str
    protocol_path: Path
    seed: int
    backward_policy: str
    worldfoam_initializer: str = single.DEFAULT_WORLDFOAM_INITIALIZER

    @property
    def key(self) -> str:
        return f"{self.protocol_path.stem}/seed_{self.seed}/{self.backward_policy}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "role": self.role,
            "protocol": single.display_path(self.protocol_path),
            "seed": self.seed,
            "world_tubes_backward_policy": self.backward_policy,
            "worldfoam_initializer": self.worldfoam_initializer,
        }


def expand_matrix(raw: Mapping[str, Any]) -> list[MatrixRun]:
    raw_runs = raw.get("runs")
    if not isinstance(raw_runs, list) or not raw_runs:
        raise ValueError("paper matrix runs must be a non-empty list")
    runs: list[MatrixRun] = []
    for index, row in enumerate(raw_runs):
        if not isinstance(row, Mapping):
            raise ValueError(f"paper matrix run {index} must be an object")
        seeds = row.get("seeds")
        if not isinstance(seeds, list) or not seeds:
            raise ValueError(f"paper matrix run {index} must declare seeds")
        protocol_path = single.resolve_root_path(row["protocol"])
        if not protocol_path.exists():
            raise FileNotFoundError(f"paper matrix protocol does not exist: {protocol_path}")
        for seed in seeds:
            runs.append(
                MatrixRun(
                    role=str(row["role"]),
                    protocol_path=protocol_path,
                    seed=int(seed),
                    backward_policy=str(row["world_tubes_backward_policy"]),
                    worldfoam_initializer=str(
                        row.get("worldfoam_initializer", single.DEFAULT_WORLDFOAM_INITIALIZER)
                    ),
                )
            )
    keys = [run.key for run in runs]
    if len(keys) != len(set(keys)):
        raise ValueError("paper matrix contains duplicate protocol/seed/policy rows")
    return runs


def flatten_summary(run: MatrixRun, summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    if summary.get("status") != "complete":
        raise ValueError(f"matrix input {run.key} is not complete")
    rows = []
    for lane_name in LANE_ORDER:
        lane = summary.get("lanes", {}).get(lane_name)
        if not isinstance(lane, Mapping):
            raise ValueError(f"matrix input {run.key} is missing {lane_name}")
        evidence = lane.get("evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError(f"matrix input {run.key}/{lane_name} has no evidence contract")
        single.validate_lane_evidence(lane_name, evidence)
        quality = evidence["quality"]
        cost = evidence["cost"]
        timing = evidence["timing"]
        rows.append(
            {
                "role": run.role,
                "protocol": summary["protocol"]["name"],
                "scene_sample": summary["protocol"]["dataset"]["sample_id"],
                "train_cameras": "+".join(summary["protocol"]["dataset"]["train_cameras"]),
                "heldout_cameras": "+".join(summary["protocol"]["dataset"]["heldout_cameras"]),
                "seed": run.seed,
                "lane": lane_name,
                "backward_policy": run.backward_policy if lane_name == "world_tubes" else "n/a",
                **quality,
                **cost,
                **timing,
                "diagnostics_json": json.dumps(evidence["diagnostics"], sort_keys=True),
                "run_summary": single.display_path(
                    Path(summary.get("run_summary_path", ""))
                ) if summary.get("run_summary_path") else "",
            }
        )
    return rows


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["role"]), str(row["protocol"]), str(row["lane"])), []).append(row)
    aggregated = []
    for (role, protocol, lane), group in sorted(groups.items()):
        metric_keys = (
            "heldout_eval_psnr",
            "heldout_eval_ssim",
            "heldout_eval_lpips",
            "heldout_eval_l1",
            "serialized_checkpoint_bytes",
            "sampled_peak_current_allocated_bytes",
            "train_wall_s",
        )
        result: dict[str, Any] = {
            "role": role,
            "protocol": protocol,
            "lane": lane,
            "seeds": [int(row["seed"]) for row in group],
            "repeat_count": len(group),
        }
        for key in metric_keys:
            values = [float(row[key]) for row in group]
            result[f"{key}_mean"] = statistics.fmean(values)
            result[f"{key}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        aggregated.append(result)
    return aggregated


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty paper CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _metric_cell(row: Mapping[str, Any], key: str, digits: int) -> str:
    return f"{float(row[f'{key}_mean']):.{digits}f} ± {float(row[f'{key}_std']):.{digits}f}"


def write_tables(markdown_path: Path, latex_path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    header = "| Protocol role | Lane | Seeds | PSNR ↑ | SSIM ↑ | LPIPS ↓ | L1 ↓ |\n|---|---|---:|---:|---:|---:|---:|"
    markdown_rows = [header]
    latex_rows = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Protocol role & Lane & Seeds & PSNR $\\uparrow$ & SSIM $\\uparrow$ & LPIPS $\\downarrow$ & L1 $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        seeds = ",".join(str(seed) for seed in row["seeds"])
        cells = (
            _metric_cell(row, "heldout_eval_psnr", 3),
            _metric_cell(row, "heldout_eval_ssim", 4),
            _metric_cell(row, "heldout_eval_lpips", 4),
            _metric_cell(row, "heldout_eval_l1", 4),
        )
        markdown_rows.append(f"| {row['role']} | {row['lane']} | {seeds} | {' | '.join(cells)} |")
        latex_role = str(row["role"]).replace("_", r"\_")
        latex_lane = str(row["lane"]).replace("_", r"\_")
        latex_rows.append(
            f"{latex_role} & {latex_lane} & {seeds} & "
            + " & ".join(cell.replace("±", r"$\pm$") for cell in cells)
            + r" \\"
        )
    latex_rows.extend(("\\bottomrule", "\\end{tabular}"))
    markdown_path.write_text("\n".join(markdown_rows) + "\n", encoding="utf-8")
    latex_path.write_text("\n".join(latex_rows) + "\n", encoding="utf-8")


def write_psnr_svg(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    width, height = 1000, 420
    margin_left, margin_bottom, margin_top = 70, 90, 35
    plot_height = height - margin_bottom - margin_top
    values = [float(row["heldout_eval_psnr_mean"]) for row in rows]
    upper = max(1.0, max(values) * 1.1)
    slot = (width - margin_left - 20) / max(1, len(rows))
    bars = []
    labels = []
    colors = {"world_tubes": "#3b82f6", "worldfoam": "#10b981", "dynamic_3dgs": "#f59e0b"}
    for index, row in enumerate(rows):
        value = float(row["heldout_eval_psnr_mean"])
        bar_height = plot_height * value / upper
        x = margin_left + index * slot + slot * 0.15
        y = margin_top + plot_height - bar_height
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{slot * 0.7:.1f}" height="{bar_height:.1f}" fill="{colors[row["lane"]]}"/>'
        )
        bars.append(f'<text x="{x + slot * 0.35:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-size="12">{value:.2f}</text>')
        label = f"{row['role']} / {row['lane']}"
        labels.append(
            f'<text x="{x + slot * 0.35:.1f}" y="{height - margin_bottom + 18:.1f}" '
            f'transform="rotate(35 {x + slot * 0.35:.1f} {height - margin_bottom + 18:.1f})" font-size="10">{label}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - 20}" y2="{margin_top + plot_height}" stroke="black"/>'
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="black"/>'
        '<text x="18" y="220" transform="rotate(-90 18 220)" font-size="14">Held-out PSNR (dB)</text>'
        + "".join(bars + labels)
        + "</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")


def write_artifacts(out_dir: Path, matrix_name: str, run_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    row_records = []
    for record in run_records:
        run = MatrixRun(
            role=str(record["run"]["role"]),
            protocol_path=single.resolve_root_path(record["run"]["protocol"]),
            seed=int(record["run"]["seed"]),
            backward_policy=str(record["run"]["world_tubes_backward_policy"]),
            worldfoam_initializer=str(
                record["run"].get("worldfoam_initializer", single.DEFAULT_WORLDFOAM_INITIALIZER)
            ),
        )
        row_records.extend(flatten_summary(run, record["summary"]))
    aggregated = aggregate_rows(row_records)
    artifacts = {
        "rows_json": out_dir / "paper_rows.json",
        "rows_csv": out_dir / "paper_rows.csv",
        "table_markdown": out_dir / "paper_table.md",
        "table_latex": out_dir / "paper_table.tex",
        "psnr_plot": out_dir / "heldout_psnr.svg",
    }
    single.write_json(artifacts["rows_json"], {"matrix": matrix_name, "rows": row_records, "aggregated": aggregated})
    write_csv(artifacts["rows_csv"], row_records)
    write_tables(artifacts["table_markdown"], artifacts["table_latex"], aggregated)
    write_psnr_svg(artifacts["psnr_plot"], aggregated)
    return {name: single.display_path(path) for name, path in artifacts.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument("--require-clean-source", action="store_true")
    args = parser.parse_args()

    matrix_path = single.resolve_root_path(args.matrix)
    raw_matrix = load_config_file(matrix_path)
    runs = expand_matrix(raw_matrix)
    out_dir = single.resolve_root_path(args.out_dir)
    if not args.execute:
        print(
            json.dumps(
                serialize_config_value(
                    {
                        "status": "dry_run",
                        "matrix": raw_matrix["name"],
                        "matrix_path": single.display_path(matrix_path),
                        "out_dir": single.display_path(out_dir),
                        "runs": [run.as_dict() for run in runs],
                    }
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return

    records = []
    for run in runs:
        raw_protocol = load_config_file(run.protocol_path)
        protocol = resolve_paper_training_protocol(raw_protocol)
        summary = single.execute(
            run.protocol_path,
            raw_protocol,
            protocol,
            seed=run.seed,
            out_dir=out_dir,
            backward_policy=run.backward_policy,
            device=args.device,
            wandb_mode=args.wandb_mode,
            reuse_existing=args.reuse_existing,
            worldfoam_initializer=run.worldfoam_initializer,
            require_clean_source=args.require_clean_source,
        )
        summary["run_summary_path"] = str(
            out_dir / protocol.name / f"seed_{run.seed}" / "run_summary.json"
        )
        records.append({"run": run.as_dict(), "summary": summary})
        single.write_json(
            out_dir / "matrix_progress.json",
            {"status": "running", "matrix": raw_matrix["name"], "completed": records},
        )
    artifacts = write_artifacts(out_dir, str(raw_matrix["name"]), records)
    result = {
        "status": "complete",
        "matrix": raw_matrix["name"],
        "run_count": len(records),
        "lane_row_count": len(records) * len(LANE_ORDER),
        "runs": records,
        "artifacts": artifacts,
    }
    single.write_json(out_dir / "matrix_summary.json", result)
    print(json.dumps(serialize_config_value(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
