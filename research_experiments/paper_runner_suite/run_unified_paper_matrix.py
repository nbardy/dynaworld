from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import netrc
import os
import shutil
import socket
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config_utils import load_config_file, serialize_config_value
from paper_training_protocol import (
    lpips_alex_asset_status,
    resolve_paper_training_protocol,
)
from research_experiments.paper_runner_suite import run_unified_paper_ablation as single


DEFAULT_MATRIX = (
    ROOT / "src" / "train_configs" / "paper_protocols" / "world_tubes_submission_matrix_v1.jsonc"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "2026-07-28_world_tubes_submission_matrix_schema2"
)
LANE_ORDER = ("world_tubes", "worldfoam", "dynamic_3dgs")
COMPARISON_MEDIA_PREFIX = {
    "world_tubes": "star_uvt",
    "dynamic_3dgs": "free_dynamic_splats",
}
MATRIX_DISK_RESERVE_BYTES = 8 * (1024**3)
MATRIX_PER_RUN_REPORT_LOG_ALLOWANCE_BYTES = 512 * (1024**2)


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            serialize_config_value(value),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def matrix_output_root(raw: Mapping[str, Any]) -> Path:
    value = raw.get("output_root")
    if value is None:
        return DEFAULT_OUT_DIR
    if not isinstance(value, str) or not value.strip():
        raise ValueError("paper matrix output_root must be a nonempty path")
    return single.resolve_root_path(value)


def resolve_matrix_output_dir(
    raw: Mapping[str, Any],
    requested: str | Path | None,
) -> Path:
    canonical = matrix_output_root(raw)
    if requested is None:
        return canonical
    resolved = single.resolve_root_path(requested)
    if raw.get("output_root") is not None and resolved != canonical:
        raise ValueError(
            "explicit --out-dir disagrees with the matrix-declared "
            f"output_root: expected {single.display_path(canonical)}, "
            f"got {single.display_path(resolved)}"
        )
    return resolved


def wandb_local_readiness(
    mode: str,
    *,
    check_connectivity: bool = False,
) -> dict[str, Any]:
    """Check local W&B prerequisites without importing W&B or exposing secrets."""

    package_present = importlib.util.find_spec("wandb") is not None
    try:
        package_version = importlib.metadata.version("wandb")
    except importlib.metadata.PackageNotFoundError:
        package_version = None
    credential_source = None
    credential_error = None
    if str(os.environ.get("WANDB_API_KEY", "")).strip():
        credential_source = "environment"
    else:
        try:
            authentication = netrc.netrc().authenticators("api.wandb.ai")
        except (FileNotFoundError, netrc.NetrcParseError, OSError) as error:
            authentication = None
            credential_error = type(error).__name__
        if authentication is not None and str(authentication[2]).strip():
            credential_source = "netrc"

    connectivity = {
        "requested": bool(check_connectivity),
        "host": "api.wandb.ai",
        "port": 443,
        "reachable": None,
        "error": None,
    }
    if check_connectivity:
        try:
            with socket.create_connection(
                (connectivity["host"], connectivity["port"]),
                timeout=5.0,
            ):
                connectivity["reachable"] = True
        except OSError as error:
            connectivity["reachable"] = False
            connectivity["error"] = type(error).__name__

    online = mode == "online"
    checks = {
        "supported_mode": mode in {"online", "offline"},
        "wandb_package_installed": package_present and package_version is not None,
        "online_credentials_present": not online or credential_source is not None,
        "connectivity": (
            not online
            or not check_connectivity
            or connectivity["reachable"] is True
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "rejected",
        "execution": "none",
        "mode": mode,
        "project": "dynaworld",
        "entity": os.environ.get("WANDB_ENTITY"),
        "package_version": package_version,
        "credential_source": credential_source,
        "credential_error": credential_error,
        "connectivity": connectivity,
        "checks": checks,
        "remote_project_authorization_checked": False,
    }


def matrix_retained_output_budget(
    runs: Sequence["MatrixRun"],
) -> dict[str, Any]:
    """Conservative retained-output allowance; no data or renderer is loaded."""

    run_estimates = []
    for run in runs:
        protocol = resolve_paper_training_protocol(
            load_config_file(run.protocol_path)
        )
        media_frames = min(32, protocol.dataset.frame_count)
        media_bytes = (
            len(LANE_ORDER)
            * 2
            * media_frames
            * protocol.final_stage.image_size.pixels
            * 3
        )
        representation_state_bytes = (
            protocol.final_stage.primitive_count
            * protocol.dataset.frame_count
            * 512
        )
        retained_bytes = (
            2 * media_bytes
            + representation_state_bytes
            + MATRIX_PER_RUN_REPORT_LOG_ALLOWANCE_BYTES
        )
        run_estimates.append(
            {
                "run_key": run.key,
                "uncompressed_media_and_wandb_copy_bytes": 2 * media_bytes,
                "representation_state_allowance_bytes": representation_state_bytes,
                "report_log_allowance_bytes": (
                    MATRIX_PER_RUN_REPORT_LOG_ALLOWANCE_BYTES
                ),
                "retained_bytes": retained_bytes,
            }
        )
    estimated = sum(item["retained_bytes"] for item in run_estimates)
    safety_adjusted = (3 * estimated + 1) // 2
    required_free = max(
        int(single.LIVE_RESOURCE_THRESHOLDS["disk_free_bytes"]),
        safety_adjusted + MATRIX_DISK_RESERVE_BYTES,
    )
    return {
        "definition": (
            "1.5x conservative allowance for retained media, W&B copies, "
            "representation state, reports, and logs, plus an 8 GiB reserve"
        ),
        "run_count": len(runs),
        "estimated_retained_bytes": estimated,
        "safety_adjusted_retained_bytes": safety_adjusted,
        "reserve_bytes": MATRIX_DISK_RESERVE_BYTES,
        "required_free_bytes": required_free,
        "runs": run_estimates,
    }


def _disk_free_bytes(path: Path) -> int:
    candidate = path.resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return int(shutil.disk_usage(candidate).free)


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
    output_rows: dict[tuple[str, int], list[str]] = {}
    for run in runs:
        protocol = resolve_paper_training_protocol(
            load_config_file(run.protocol_path)
        )
        if protocol.name != run.protocol_path.stem:
            raise ValueError(
                "paper matrix protocol filename and declared name must match "
                f"for stable run keys: {run.protocol_path.stem!r} != "
                f"{protocol.name!r}"
            )
        output_rows.setdefault((protocol.name, run.seed), []).append(run.key)
    collisions = {
        f"{protocol_name}/seed_{seed}": keys
        for (protocol_name, seed), keys in output_rows.items()
        if len(keys) > 1
    }
    if collisions:
        detail = "; ".join(
            f"{output_path}: {', '.join(row_keys)}"
            for output_path, row_keys in sorted(collisions.items())
        )
        raise ValueError(
            "paper matrix rows collide on their actual output directory: "
            + detail
        )
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
                "repository_commit": summary["source"]["repository_commit"],
                "star_uvt_commit": summary["source"]["star_uvt_commit"],
                "source_provenance_scope": summary.get(
                    "source_provenance_scope",
                    "summary-clean provenance; child execution SHA unavailable",
                ),
                "wandb_run_id": lane["wandb"]["run_id"],
                "wandb_validation_scope": "summary mode/run-id metadata",
                "dataset_input_sha256": summary[
                    "common_evidence_contract"
                ]["dataset_input_identity"]["sha256"],
                "decoded_dataset_sha256": summary[
                    "common_evidence_contract"
                ]["decoded_dataset_bundle"]["sha256"],
                "evaluator_sha256": summary["common_evidence_contract"][
                    "evaluator"
                ]["sha256"],
                "runtime_sha256": summary["common_evidence_contract"][
                    "runtime"
                ]["sha256"],
                "sample_schedule_sha256": summary[
                    "common_evidence_contract"
                ]["sample_schedule"]["sha256"],
                "paper_backward_policy": summary[
                    "world_tubes_backward_policy"
                ],
                "route_native_sha256": canonical_json_sha256(
                    lane["route_native_extension"]
                ),
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
        seeds = [int(row["seed"]) for row in group]
        if len(seeds) != len(set(seeds)):
            raise ValueError(
                f"cannot aggregate duplicate seeds for {role}/{protocol}/{lane}"
            )
        compatibility_keys = (
            "repository_commit",
            "star_uvt_commit",
            "dataset_input_sha256",
            "decoded_dataset_sha256",
            "evaluator_sha256",
            "runtime_sha256",
            "paper_backward_policy",
            "route_native_sha256",
        )
        drifted = [
            key
            for key in compatibility_keys
            if len({str(row[key]) for row in group}) != 1
        ]
        if drifted:
            raise ValueError(
                f"cannot aggregate incompatible evidence for "
                f"{role}/{protocol}/{lane}: {', '.join(drifted)}"
            )
        metric_keys = (
            "heldout_eval_psnr",
            "heldout_eval_ssim",
            "heldout_eval_lpips",
            "heldout_eval_l1",
            "parameter_count",
            "parameter_bytes",
            "optimizer_state_bytes",
            "serialized_checkpoint_bytes",
            "sampled_peak_current_allocated_bytes",
            "sampled_peak_driver_allocated_bytes",
            "target_frames",
            "rasterized_frames",
            "target_pixels",
            "rasterized_pixels",
            "train_wall_s",
        )
        result: dict[str, Any] = {
            "role": role,
            "protocol": protocol,
            "lane": lane,
            "seeds": seeds,
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
    header = (
        "| Protocol role | Lane | Seeds | PSNR ↑ | SSIM ↑ | LPIPS ↓ | L1 ↓ "
        "| Train wall (s) | Peak driver (GB) | Parameters | Checkpoint (MB) |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    markdown_rows = [header]
    latex_rows = [
        "\\begin{tabular}{llrrrrrrrrr}",
        "\\toprule",
        (
            "Protocol role & Lane & Seeds & PSNR $\\uparrow$ & SSIM $\\uparrow$ "
            "& LPIPS $\\downarrow$ & L1 $\\downarrow$ & Wall (s) "
            "& Driver (GB) & Params & Checkpoint (MB) \\\\"
        ),
        "\\midrule",
    ]
    for row in rows:
        seeds = ",".join(str(seed) for seed in row["seeds"])
        cells = (
            _metric_cell(row, "heldout_eval_psnr", 3),
            _metric_cell(row, "heldout_eval_ssim", 4),
            _metric_cell(row, "heldout_eval_lpips", 4),
            _metric_cell(row, "heldout_eval_l1", 4),
            _metric_cell(row, "train_wall_s", 2),
            f"{float(row['sampled_peak_driver_allocated_bytes_mean']) / 1.0e9:.3f}",
            f"{float(row['parameter_count_mean']):.0f}",
            f"{float(row['serialized_checkpoint_bytes_mean']) / 1.0e6:.3f}",
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
    primary_rows = [row for row in rows if row["role"] == "primary_progressive"]
    plot_rows = primary_rows or list(rows)
    width, height = 760, 460
    margin_left, margin_bottom, margin_top = 80, 90, 55
    plot_height = height - margin_bottom - margin_top
    values = [
        float(row["heldout_eval_psnr_mean"])
        + float(row["heldout_eval_psnr_std"])
        for row in plot_rows
    ]
    upper = max(1.0, max(values) * 1.12)
    slot = (width - margin_left - 35) / max(1, len(plot_rows))
    bars = []
    labels = []
    colors = {"world_tubes": "#3b82f6", "worldfoam": "#10b981", "dynamic_3dgs": "#f59e0b"}
    for index, row in enumerate(plot_rows):
        value = float(row["heldout_eval_psnr_mean"])
        std = float(row["heldout_eval_psnr_std"])
        bar_height = plot_height * value / upper
        x = margin_left + index * slot + slot * 0.15
        y = margin_top + plot_height - bar_height
        center = x + slot * 0.35
        error_height = plot_height * std / upper
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{slot * 0.7:.1f}" height="{bar_height:.1f}" fill="{colors[row["lane"]]}"/>'
        )
        bars.append(
            f'<line x1="{center:.1f}" y1="{y - error_height:.1f}" '
            f'x2="{center:.1f}" y2="{y + error_height:.1f}" stroke="black"/>'
        )
        bars.append(
            f'<line x1="{center - 7:.1f}" y1="{y - error_height:.1f}" '
            f'x2="{center + 7:.1f}" y2="{y - error_height:.1f}" stroke="black"/>'
        )
        bars.append(
            f'<text x="{center:.1f}" y="{y - error_height - 8:.1f}" '
            f'text-anchor="middle" font-size="13">{value:.3f} ± {std:.3f}</text>'
        )
        label = str(row["lane"]).replace("_", " ")
        labels.append(
            f'<text x="{center:.1f}" y="{height - margin_bottom + 28:.1f}" '
            f'text-anchor="middle" font-size="14">{label}</text>'
        )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="white"/>'
        '<text x="380" y="27" text-anchor="middle" font-size="17" font-weight="bold">'
        'Coffee Martini progressive-512 held-out quality</text>'
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


def _require_retained_media(
    lane_name: str,
    *,
    run_dir: Path,
    steps: int,
) -> None:
    if lane_name in COMPARISON_MEDIA_PREFIX:
        prefix = COMPARISON_MEDIA_PREFIX[lane_name]
        comparison_dir = run_dir / "world_tubes_dynamic_3dgs"
        missing = []
        for split in ("train", "heldout"):
            name = f"{prefix}_{split}_view0_side_by_side.mp4"
            candidates = (
                comparison_dir / name,
                comparison_dir / lane_name / name,
            )
            if not any(path.is_file() and path.stat().st_size > 0 for path in candidates):
                missing.append(split)
        if missing:
            raise ValueError(
                f"existing paper row is missing {lane_name} media: "
                + ", ".join(missing)
            )
        return

    if lane_name == "worldfoam":
        worldfoam_dir = run_dir / "worldfoam"
        missing = [
            split
            for split, name in (
                ("train", f"side_by_side_step_{steps:04d}.mp4"),
                ("heldout", f"heldout_side_by_side_step_{steps:04d}.mp4"),
            )
            if not (worldfoam_dir / name).is_file()
            or (worldfoam_dir / name).stat().st_size <= 0
        ]
        if missing:
            raise ValueError(
                "existing paper row is missing worldfoam media: "
                + ", ".join(missing)
            )
        return

    raise ValueError(f"unsupported paper lane: {lane_name}")


def _summary_uvt_options(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "uvt_world_representation": summary.get(
            "uvt_world_representation",
            single.DEFAULT_UVT_WORLD_REPRESENTATION,
        ),
        "uvt_alpha_mode": summary.get(
            "uvt_alpha_mode",
            single.DEFAULT_UVT_ALPHA_MODE,
        ),
        "uvt_render_backend": summary.get(
            "uvt_render_backend",
            single.DEFAULT_UVT_RENDER_BACKEND,
        ),
        "uvt_amplitude_convention": summary.get(
            "uvt_amplitude_convention",
            single.DEFAULT_UVT_AMPLITUDE_CONVENTION,
        ),
        "uvt_retained_depth_samples": int(
            summary.get(
                "uvt_retained_depth_samples",
                single.DEFAULT_UVT_RETAINED_DEPTH_SAMPLES,
            )
        ),
        "uvt_retained_sigma_extent": float(
            summary.get(
                "uvt_retained_sigma_extent",
                single.DEFAULT_UVT_RETAINED_SIGMA_EXTENT,
            )
        ),
        "uvt_order_certificate_sigma": float(
            summary.get(
                "uvt_order_certificate_sigma",
                single.DEFAULT_UVT_ORDER_CERTIFICATE_SIGMA,
            )
        ),
        "uvt_order_certificate_min_gap": float(
            summary.get(
                "uvt_order_certificate_min_gap",
                single.DEFAULT_UVT_ORDER_CERTIFICATE_MIN_GAP,
            )
        ),
        "uvt_spd4_init_precision_z": summary.get(
            "uvt_spd4_init_precision_z"
        ),
        "frozen_world_replay_compiled": bool(
            summary.get("frozen_world_replay_compiled", False)
        ),
        "frozen_world_max_frames": int(
            summary.get("frozen_world_max_frames", 0)
        ),
    }


def _identity_python(
    lane_name: str,
    identity: Mapping[str, Any],
    *,
    run: MatrixRun,
) -> str:
    command = identity.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(value, str) for value in command)
    ):
        raise ValueError(
            f"existing paper row {lane_name} command is invalid: {run.key}"
        )
    python = Path(command[0])
    if not python.is_absolute() or not python.is_file():
        raise ValueError(
            f"existing paper row {lane_name} Python executable is invalid: "
            f"{run.key}"
        )
    return str(python)


def validate_existing_summary(
    run: MatrixRun,
    summary: Mapping[str, Any],
    *,
    protocol,
    summary_path: Path,
) -> None:
    if summary.get("status") != "complete":
        raise ValueError(f"existing paper row is not complete: {run.key}")
    if int(summary.get("seed", -1)) != run.seed:
        raise ValueError(f"existing paper row seed drifted: {run.key}")
    uvt_options = _summary_uvt_options(summary)
    requested_backward_policy = summary.get(
        "world_tubes_requested_backward_policy",
        summary.get("world_tubes_backward_policy"),
    )
    if requested_backward_policy != run.backward_policy:
        raise ValueError(
            f"existing paper row requested backward policy drifted: {run.key}"
        )
    expected_backward_policy = single.effective_uvt_backward_policy(
        str(uvt_options["uvt_render_backend"]),
        run.backward_policy,
    )
    if summary.get("world_tubes_backward_policy") != expected_backward_policy:
        raise ValueError(
            f"existing paper row effective backward policy drifted: {run.key}"
        )
    if summary.get("worldfoam_initializer") != run.worldfoam_initializer:
        raise ValueError(f"existing paper row WorldFoam initializer drifted: {run.key}")
    initializer_identity = summary.get("worldfoam_initializer_identity")
    single.validate_hashed_contract(
        "existing paper row WorldFoam initializer",
        initializer_identity,
        schema_version=1,
    )
    initializer_file = initializer_identity.get("file")
    if initializer_file is not None:
        if not isinstance(initializer_file, Mapping):
            raise ValueError(
                f"existing paper row WorldFoam initializer file identity is "
                f"invalid: {run.key}"
            )
        initializer_path = single.resolve_root_path(
            str(initializer_file.get("path", ""))
        )
        if (
            not initializer_path.is_file()
            or int(initializer_file.get("bytes", -1))
            != initializer_path.stat().st_size
            or initializer_file.get("sha256")
            != single.file_sha256(initializer_path)
        ):
            raise ValueError(
                f"existing paper row WorldFoam initializer bytes drifted: "
                f"{run.key}"
            )
    source = summary.get("source")
    if not isinstance(source, Mapping):
        raise ValueError(f"existing paper row has no source provenance: {run.key}")
    dirty = [
        key
        for key in ("repository_dirty", "star_uvt_dirty")
        if source.get(key) is not False
    ]
    if dirty:
        raise ValueError(
            f"existing paper row has dirty source provenance "
            f"({', '.join(dirty)}): {run.key}"
        )
    for key in ("repository_commit", "star_uvt_commit"):
        commit = str(source.get(key, ""))
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit.lower()):
            raise ValueError(f"existing paper row has invalid {key}: {run.key}")
    if summary.get("source_finish") != source:
        raise ValueError(
            f"existing paper row source_finish does not match source: {run.key}"
        )
    if summary.get("protocol") != protocol.as_dict():
        raise ValueError(f"existing paper row protocol contract drifted: {run.key}")

    manifest = summary.get("manifest_validation")
    if not isinstance(manifest, Mapping):
        raise ValueError(f"existing paper row has no manifest validation: {run.key}")
    if (
        manifest.get("sample_id") != protocol.dataset.sample_id
        or manifest.get("manifest") != protocol.dataset.manifest
    ):
        raise ValueError(f"existing paper row dataset identity drifted: {run.key}")
    checks = manifest.get("checks")
    if not isinstance(checks, Mapping) or not checks or not all(
        value is True for value in checks.values()
    ):
        raise ValueError(f"existing paper row failed manifest validation: {run.key}")
    current_manifest = single.validate_manifest(protocol)
    if manifest.get("input_identity") != current_manifest["input_identity"]:
        raise ValueError(
            f"existing paper row raw dataset bytes drifted: {run.key}"
        )
    if any(
        manifest.get(key) != current_manifest[key]
        for key in ("dataset", "expected_pose_source")
    ):
        raise ValueError(
            f"existing paper row dataset pose-source contract drifted: {run.key}"
        )

    common = summary.get("common_evidence_contract")
    if not isinstance(common, Mapping) or int(common.get("schema_version", -1)) != 1:
        raise ValueError(
            f"existing paper row has no common evidence contract: {run.key}"
        )
    if common.get("dataset_input_identity") != manifest["input_identity"]:
        raise ValueError(
            f"existing paper row common raw dataset identity drifted: {run.key}"
        )
    for name, schema in (
        ("decoded_dataset_bundle", single.PAPER_DATASET_BUNDLE_SCHEMA_VERSION),
        ("evaluator", single.PAPER_EVALUATOR_SCHEMA_VERSION),
        ("runtime", single.PAPER_RUNTIME_SCHEMA_VERSION),
    ):
        single.validate_hashed_contract(
            f"existing paper row {name}",
            common.get(name),
            schema_version=schema,
        )
    if common["evaluator"] != single.paper_evaluator_contract():
        raise ValueError(
            f"existing paper row evaluator is not canonical: {run.key}"
        )
    sample_schedule = common.get("sample_schedule")
    if (
        not isinstance(sample_schedule, Mapping)
        or int(sample_schedule.get("schema_version", -1))
        != single.PAPER_SAMPLE_SCHEDULE_SCHEMA_VERSION
        or sample_schedule.get("algorithm")
        != single.PAPER_SAMPLE_SCHEDULE_ALGORITHM
        or int(sample_schedule.get("sampler_seed", -1))
        != run.seed + protocol.sampler_seed_offset
        or int(sample_schedule.get("record_count", -1)) != protocol.steps
        or not isinstance(sample_schedule.get("sha256"), str)
        or len(sample_schedule["sha256"]) != 64
    ):
        raise ValueError(
            f"existing paper row sample schedule is invalid: {run.key}"
        )

    run_dir = summary_path.parent
    comparison_report_path = (
        run_dir / "world_tubes_dynamic_3dgs" / "comparison_report.json"
    )
    if not comparison_report_path.is_file():
        raise ValueError(
            f"existing paper row is missing merged comparison report: {run.key}"
        )
    comparison_report = single.load_json(comparison_report_path)
    merged_comparison_report_digest = canonical_json_sha256(comparison_report)
    try:
        single.validate_comparison_report(
            comparison_report,
            protocol,
            backward_policy=run.backward_policy,
            manifest_validation=current_manifest,
            **uvt_options,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"existing paper row merged comparison report is invalid: "
            f"{run.key}"
        ) from error
    comparison_meta = comparison_report["meta"]
    comparison_common = {
        "decoded_dataset_bundle": "paper_dataset_bundle",
        "evaluator": "paper_evaluator",
        "runtime": "paper_runtime",
    }
    drifted_common = [
        common_name
        for common_name, meta_name in comparison_common.items()
        if comparison_meta.get(meta_name) != common[common_name]
    ]
    if drifted_common:
        raise ValueError(
            f"existing paper row merged comparison common evidence drifted "
            f"({', '.join(drifted_common)}): {run.key}"
        )
    if int(comparison_meta.get("seed", -1)) != run.seed:
        raise ValueError(
            f"existing paper row merged comparison seed drifted: {run.key}"
        )
    comparison_schedule = comparison_report["star_uvt"][
        "paper_protocol"
    ]["sample_schedule"]
    if comparison_schedule != common["sample_schedule"]:
        raise ValueError(
            f"existing paper row merged comparison schedule drifted: {run.key}"
        )
    for lane_name, report_key in single.LANE_REPORT_KEYS.items():
        retained_lane = comparison_report[report_key]
        summary_lane = summary.get("lanes", {}).get(lane_name)
        if not isinstance(summary_lane, Mapping):
            raise ValueError(
                f"existing paper row is missing {lane_name}: {run.key}"
            )
        if (
            summary_lane.get("metrics") != retained_lane.get("metrics")
            or summary_lane.get("paper_protocol")
            != retained_lane.get("paper_protocol")
            or summary_lane.get("route_native_extension")
            != comparison_meta["route_native_extensions"][lane_name]
        ):
            raise ValueError(
                f"existing paper row {lane_name} retained report drifted: "
                f"{run.key}"
            )
        rebuilt_evidence = single.build_lane_evidence(
            lane_name,
            retained_lane,
            frame_count=protocol.dataset.frame_count,
        )
        if summary_lane.get("evidence") != rebuilt_evidence:
            raise ValueError(
                f"existing paper row {lane_name} evidence does not match "
                f"the retained report: {run.key}"
            )
    for lane_name in LANE_ORDER:
        _require_retained_media(
            lane_name,
            run_dir=run_dir,
            steps=protocol.steps,
        )
    protocol_sha256 = single.file_sha256(run.protocol_path)
    child_identities = {
        "world_tubes": (
            run_dir
            / "world_tubes_dynamic_3dgs"
            / "world_tubes"
            / "execution_identity.json"
        ),
        "dynamic_3dgs": (
            run_dir
            / "world_tubes_dynamic_3dgs"
            / "dynamic_3dgs"
            / "execution_identity.json"
        ),
        "worldfoam": run_dir / "worldfoam" / "execution_identity.json",
    }
    isolated_comparison_reports: dict[str, Mapping[str, Any]] = {}
    for lane_name, identity_path in child_identities.items():
        if not identity_path.is_file():
            raise ValueError(
                f"existing paper row is missing {lane_name} execution identity: "
                f"{run.key}"
            )
        identity = single.load_json(identity_path)
        if (
            int(identity.get("schema_version", -1)) != 1
            or identity.get("lane") != lane_name
            or identity.get("protocol") != protocol.as_dict()
        ):
            raise ValueError(
                f"existing paper row {lane_name} execution contract drifted: "
                f"{run.key}"
            )
        if (
            identity.get("source_start") != source
            or identity.get("source_finish") != source
            or identity.get("dataset_input_identity")
            != common["dataset_input_identity"]
            or identity.get("protocol_sha256") != protocol_sha256
            or (
                lane_name == "worldfoam"
                and identity.get("initializer_identity")
                != initializer_identity
            )
        ):
            raise ValueError(
                f"existing paper row {lane_name} execution identity drifted: "
                f"{run.key}"
            )
        summary_lane_wandb = summary.get("lanes", {}).get(
            lane_name,
            {},
        ).get("wandb")
        if (
            not isinstance(summary_lane_wandb, Mapping)
            or summary_lane_wandb.get("mode") not in {"online", "offline"}
            or not str(summary_lane_wandb.get("run_id", "")).strip()
        ):
            raise ValueError(
                f"existing paper row has invalid {lane_name} W&B provenance: "
                f"{run.key}"
            )
        python = _identity_python(lane_name, identity, run=run)
        device = str(comparison_meta["device"])
        allow_local_mps_execution = device.lower() == "mps"
        if lane_name in COMPARISON_MEDIA_PREFIX:
            expected_command = single.comparison_command(
                run.protocol_path,
                protocol,
                run.seed,
                identity_path.parent,
                backward_policy=run.backward_policy,
                device=device,
                only_lane=lane_name,
                allow_local_mps_execution=allow_local_mps_execution,
                python=python,
                **uvt_options,
            )
        else:
            expected_command = single.worldfoam_lane_command(
                run.protocol_path,
                run.seed,
                identity_path.parent,
                device=device,
                wandb_mode=str(summary_lane_wandb["mode"]),
                worldfoam_initializer=run.worldfoam_initializer,
                allow_local_mps_execution=allow_local_mps_execution,
                allow_high_risk_local_mps=bool(
                    summary.get("execution_safety", {}).get(
                        "high_risk",
                        False,
                    )
                ),
                python=python,
            )
        if identity["command"] != expected_command:
            raise ValueError(
                f"existing paper row {lane_name} command drifted: {run.key}"
            )
        if lane_name in COMPARISON_MEDIA_PREFIX:
            report_path = identity_path.with_name("comparison_report.json")
            if (
                not report_path.is_file()
                or identity.get("comparison_report_sha256")
                != single.file_sha256(report_path)
            ):
                raise ValueError(
                    f"existing paper row {lane_name} report identity drifted: "
                    f"{run.key}"
                )
            isolated_comparison_reports[lane_name] = single.load_json(
                report_path
            )
            wandb_identity_path = identity_path.with_name(
                "wandb_identity.json"
            )
            if (
                not wandb_identity_path.is_file()
                or identity.get("wandb_identity", {}).get("sha256")
                != single.file_sha256(wandb_identity_path)
            ):
                raise ValueError(
                    f"existing paper row {lane_name} W&B sidecar identity "
                    f"drifted: {run.key}"
                )
        else:
            worldfoam_artifacts = {
                "paper_protocol_summary": "paper_protocol_summary.json",
                "best_metrics": "best_metrics.json",
                "eval_metrics_history": "eval_metrics_history.jsonl",
                "resolved_config": "resolved_config.json",
                "checkpoint_final": "checkpoint_final.pt",
                "train_metrics_history": "train_metrics_history.jsonl",
                "final_train_media": f"side_by_side_step_{protocol.steps:04d}.mp4",
                "final_heldout_media": (
                    f"heldout_side_by_side_step_{protocol.steps:04d}.mp4"
                ),
                "wandb_identity": "wandb_identity.json",
            }
            artifacts = identity.get("artifacts")
            if not isinstance(artifacts, Mapping):
                raise ValueError(
                    f"existing paper row WorldFoam artifact identity is "
                    f"missing: {run.key}"
                )
            for artifact_key, artifact_name in worldfoam_artifacts.items():
                artifact_path = identity_path.parent / artifact_name
                if (
                    not artifact_path.is_file()
                    or artifacts.get(artifact_key, {}).get("sha256")
                    != single.file_sha256(artifact_path)
                ):
                    raise ValueError(
                        f"existing paper row WorldFoam {artifact_key} "
                        f"identity drifted: {run.key}"
                    )
    try:
        rebuilt_comparison_report = single.merge_comparison_lane_reports(
            isolated_comparison_reports
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"existing paper row isolated comparison reports are invalid: "
            f"{run.key}"
        ) from error
    if rebuilt_comparison_report != comparison_report:
        raise ValueError(
            f"existing paper row merged comparison report does not match "
            f"its isolated lane reports: {run.key}"
        )
    worldfoam_dir = run_dir / "worldfoam"
    summary_worldfoam = summary.get("lanes", {}).get("worldfoam")
    if not isinstance(summary_worldfoam, Mapping):
        raise ValueError(
            f"existing paper row is missing worldfoam: {run.key}"
        )
    expected_powerfoam_config = single.powerfoam_config(
        load_config_file(run.protocol_path),
        protocol,
        run.seed,
        worldfoam_dir,
        wandb_mode=str(summary_worldfoam["wandb"]["mode"]),
        device=str(comparison_meta["device"]),
        worldfoam_initializer=run.worldfoam_initializer,
    )
    resolved_config_binding = single.worldfoam_resolved_config_binding(
        expected_powerfoam_config,
        worldfoam_dir / "resolved_config.json",
    )
    worldfoam_execution_identity = single.load_json(
        child_identities["worldfoam"]
    )
    if (
        summary_worldfoam.get("resolved_config_binding")
        != resolved_config_binding
        or worldfoam_execution_identity.get("resolved_config_binding")
        != resolved_config_binding
    ):
        raise ValueError(
            f"existing paper row WorldFoam resolved config binding drifted: "
            f"{run.key}"
        )
    powerfoam_summary = single.load_json(
        worldfoam_dir / "paper_protocol_summary.json"
    )
    try:
        single.validate_lane_cost(
            "worldfoam",
            {
                "steps": powerfoam_summary["cost"]["optimizer_steps"],
                "paper_protocol": powerfoam_summary,
            },
            protocol,
            seed=run.seed,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"existing paper row retained WorldFoam protocol is invalid: "
            f"{run.key}"
        ) from error
    worldfoam_common = {
        "paper_dataset_bundle": "decoded_dataset_bundle",
        "paper_evaluator": "evaluator",
        "paper_runtime": "runtime",
    }
    drifted_worldfoam_common = [
        artifact_name
        for artifact_name, common_name in worldfoam_common.items()
        if powerfoam_summary.get(artifact_name) != common[common_name]
    ]
    if drifted_worldfoam_common:
        raise ValueError(
            f"existing paper row retained WorldFoam common evidence drifted "
            f"({', '.join(drifted_worldfoam_common)}): {run.key}"
        )
    if powerfoam_summary.get("sample_schedule") != common["sample_schedule"]:
        raise ValueError(
            f"existing paper row retained WorldFoam schedule drifted: "
            f"{run.key}"
        )
    checkpoint_path = worldfoam_dir / "checkpoint_final.pt"
    if int(
        powerfoam_summary.get("cost", {}).get(
            "serialized_checkpoint_bytes",
            -1,
        )
    ) != checkpoint_path.stat().st_size:
        raise ValueError(
            f"existing paper row retained WorldFoam checkpoint size drifted: "
            f"{run.key}"
        )
    try:
        powerfoam_metrics = single.load_final_powerfoam_metrics(
            worldfoam_dir / "eval_metrics_history.jsonl",
            expected_step=protocol.steps,
        )
        rebuilt_worldfoam_evidence = single.build_lane_evidence(
            "worldfoam",
            {
                "metrics": powerfoam_metrics,
                "paper_protocol": powerfoam_summary,
            },
            frame_count=protocol.dataset.frame_count,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"existing paper row retained WorldFoam evaluation is invalid: "
            f"{run.key}"
        ) from error
    if (
        summary_worldfoam.get("metrics") != powerfoam_metrics
        or summary_worldfoam.get("paper_protocol") != powerfoam_summary
        or summary_worldfoam.get("route_native_extension")
        != powerfoam_summary.get("route_native_extension")
        or summary_worldfoam.get("evidence")
        != rebuilt_worldfoam_evidence
    ):
        raise ValueError(
            f"existing paper row WorldFoam evidence does not match retained "
            f"artifacts: {run.key}"
        )
    for lane_name in LANE_ORDER:
        lane = summary.get("lanes", {}).get(lane_name)
        if not isinstance(lane, Mapping):
            raise ValueError(f"existing paper row is missing {lane_name}: {run.key}")
        single.validate_route_native_extension_identity(
            lane_name,
            lane.get("route_native_extension"),
        )
        evidence = lane.get("evidence")
        if not isinstance(evidence, Mapping):
            raise ValueError(f"existing paper row has no {lane_name} evidence: {run.key}")
        single.validate_lane_evidence(lane_name, evidence)
        cost = evidence["cost"]
        expected_costs = {
            "optimizer_steps": protocol.steps,
            "target_frames": protocol.target_frame_budget,
            "rasterized_frames": protocol.target_frame_budget,
            "target_pixels": protocol.target_pixel_budget,
            "rasterized_pixels": protocol.target_pixel_budget,
        }
        drifted = [
            key
            for key, expected in expected_costs.items()
            if int(cost[key]) != int(expected)
        ]
        if drifted:
            raise ValueError(
                f"existing paper row {lane_name} cost drifted "
                f"({', '.join(drifted)}): {run.key}"
            )
        wandb = lane.get("wandb")
        if (
            not isinstance(wandb, Mapping)
            or wandb.get("mode") not in {"online", "offline"}
            or not str(wandb.get("run_id", "")).strip()
        ):
            raise ValueError(
                f"existing paper row has invalid {lane_name} W&B provenance: "
                f"{run.key}"
            )
        wandb_identity_path = (
            run_dir
            / (
                f"world_tubes_dynamic_3dgs/{lane_name}/wandb_identity.json"
                if lane_name in COMPARISON_MEDIA_PREFIX
                else "worldfoam/wandb_identity.json"
            )
        )
        if (
            not wandb_identity_path.is_file()
            or single.load_json(wandb_identity_path) != wandb
        ):
            raise ValueError(
                f"existing paper row has mismatched {lane_name} W&B sidecar: "
                f"{run.key}"
            )
        run_file = wandb.get("run_file")
        run_file_path = (
            None
            if not isinstance(run_file, Mapping)
            else Path(str(run_file.get("path", "")))
        )
        if (
            run_file_path is None
            or not run_file_path.is_absolute()
            or not run_file_path.is_file()
            or int(run_file.get("bytes", -1)) != run_file_path.stat().st_size
            or run_file.get("sha256") != single.file_sha256(run_file_path)
        ):
            raise ValueError(
                f"existing paper row has invalid {lane_name} W&B run file: "
                f"{run.key}"
            )
        try:
            single._validate_wandb_run_file_identity(
                run_file,
                run_dir=wandb.get("run_dir", ""),
                run_id=str(wandb["run_id"]),
            )
        except (FileNotFoundError, ValueError) as error:
            raise ValueError(
                f"existing paper row has mismatched {lane_name} W&B run file: "
                f"{run.key}"
            ) from error
        source_digest = hashlib.sha256(
            json.dumps(
                serialize_config_value(dict(source)),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        if wandb.get("source_digest") != source_digest:
            raise ValueError(
                f"existing paper row {lane_name} W&B source drifted: {run.key}"
            )
        if lane_name in COMPARISON_MEDIA_PREFIX:
            if (
                wandb.get("comparison_report_sha256")
                != merged_comparison_report_digest
            ):
                raise ValueError(
                    f"existing paper row {lane_name} W&B merged report drifted: "
                    f"{run.key}"
                )
        elif (
            wandb.get("finalized") is not True
            or wandb.get("paper_protocol_summary_sha256")
            != single.file_sha256(
                run_dir / "worldfoam" / "paper_protocol_summary.json"
            )
            or wandb.get("resolved_config_sha256")
            != single.file_sha256(
                run_dir / "worldfoam" / "resolved_config.json"
            )
        ):
            raise ValueError(
                f"existing paper row WorldFoam W&B evidence drifted: {run.key}"
            )


def load_existing_summary(
    run: MatrixRun,
    out_dir: Path,
) -> dict[str, Any] | None:
    raw_protocol = load_config_file(run.protocol_path)
    protocol = resolve_paper_training_protocol(raw_protocol)
    summary_path = out_dir / protocol.name / f"seed_{run.seed}" / "run_summary.json"
    if not summary_path.exists():
        return None
    summary = single.load_json(summary_path)
    validate_existing_summary(
        run,
        summary,
        protocol=protocol,
        summary_path=summary_path,
    )
    summary["run_summary_path"] = str(summary_path)
    flatten_summary(run, summary)
    return summary


def collect_existing_records(
    runs: Sequence[MatrixRun],
    out_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Load only complete fail-closed summaries; never infer completion from lane debris."""
    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for run in runs:
        summary = load_existing_summary(run, out_dir)
        if summary is None:
            missing.append(run.key)
            continue
        records.append({"run": run.as_dict(), "summary": summary})
    return records, missing


def select_matrix_runs(
    runs: Sequence[MatrixRun],
    requested_keys: Sequence[str],
) -> list[MatrixRun]:
    if not requested_keys:
        return list(runs)
    if len(requested_keys) != len(set(requested_keys)):
        raise ValueError("paper matrix run keys must be unique")
    by_key = {run.key: run for run in runs}
    unknown = [key for key in requested_keys if key not in by_key]
    if unknown:
        raise ValueError(
            "paper matrix run keys are not declared by the matrix: "
            + ", ".join(unknown)
        )
    requested = set(requested_keys)
    return [run for run in runs if run.key in requested]


def matrix_progress_payload(
    *,
    matrix_name: str,
    runs: Sequence[MatrixRun],
    accepted_records: Sequence[Mapping[str, Any]],
    status: str,
    selected_runs: Sequence[MatrixRun],
    new_run_count: int,
) -> dict[str, Any]:
    accepted_by_key = {
        str(record["run"]["key"]): record["run"]
        for record in accepted_records
    }
    return {
        "status": status,
        "matrix": matrix_name,
        "expected_run_count": len(runs),
        "selected_run_count": len(selected_runs),
        "new_run_count": int(new_run_count),
        "accepted_run_count": len(accepted_by_key),
        "accepted_lane_row_count": len(accepted_by_key) * len(LANE_ORDER),
        "accepted_runs": [
            accepted_by_key[run.key]
            for run in runs
            if run.key in accepted_by_key
        ],
        "missing_runs": [
            run.key for run in runs if run.key not in accepted_by_key
        ],
    }


def matrix_failure_payload(
    *,
    matrix_name: str,
    runs: Sequence[MatrixRun],
    accepted_records: Sequence[Mapping[str, Any]],
    selected_runs: Sequence[MatrixRun],
    new_run_count: int,
    failed_run: MatrixRun,
    error: Exception,
) -> dict[str, Any]:
    return {
        **matrix_progress_payload(
            matrix_name=matrix_name,
            runs=runs,
            accepted_records=accepted_records,
            status="failed",
            selected_runs=selected_runs,
            new_run_count=new_run_count,
        ),
        "execution": "bounded",
        "failed_run": failed_run.as_dict(),
        "failure": {
            "exception_type": type(error).__name__,
            "message": str(error),
        },
        "resume_instruction": (
            "Resolve the failed row, then rerun the same exact --run-key with "
            "--reuse-existing. Only validated run_summary.json files count as "
            "accepted."
        ),
    }


def matrix_preflight(
    runs: Sequence[MatrixRun],
    *,
    device: str,
    wandb_mode: str = "online",
    check_wandb_connectivity: bool = False,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    source = single.source_provenance()
    source_error = None
    try:
        single.require_clean_provenance(source)
    except RuntimeError as error:
        source_error = str(error)
    protocol_estimates: dict[str, dict[str, Any]] = {}
    for run in runs:
        key = single.display_path(run.protocol_path)
        if key not in protocol_estimates:
            protocol = resolve_paper_training_protocol(
                load_config_file(run.protocol_path)
            )
            protocol_estimates[key] = single.local_mps_safety_estimate(
                protocol
            )
    high_risk_protocols = [
        path
        for path, estimate in protocol_estimates.items()
        if estimate["high_risk"] is True
    ]
    lpips_assets = lpips_alex_asset_status()
    wandb_readiness = wandb_local_readiness(
        wandb_mode,
        check_connectivity=check_wandb_connectivity,
    )
    retained_output_budget = matrix_retained_output_budget(runs)
    try:
        output_disk_free_bytes = _disk_free_bytes(out_dir)
        output_disk_error = None
    except OSError as error:
        output_disk_free_bytes = 0
        output_disk_error = f"{type(error).__name__}: {error}"
    live_resources = None
    live_resource_error = None
    if device.lower() == "mps":
        try:
            live_resources = single.live_resource_snapshot()
            single.require_live_resources(live_resources)
        except (OSError, RuntimeError, ValueError) as error:
            live_resource_error = str(error)
    else:
        live_resource_error = "paper execution currently requires device=mps"
    checks = {
        "clean_superproject_and_star_source": source_error is None,
        "all_protocol_estimates_below_incident_limit": not high_risk_protocols,
        "live_resource_gate": live_resource_error is None,
        "supported_device": device.lower() == "mps",
        "lpips_alex_assets_exact": lpips_assets["status"] == "pass",
        "wandb_local_readiness": wandb_readiness["status"] == "pass",
        "retained_output_disk_budget": (
            output_disk_free_bytes
            >= int(retained_output_budget["required_free_bytes"])
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "rejected",
        "execution": "none",
        "checks": checks,
        "source": source,
        "source_error": source_error,
        "selected_run_count": len(runs),
        "protocol_estimates": protocol_estimates,
        "high_risk_protocols": high_risk_protocols,
        "lpips_alex_assets": lpips_assets,
        "wandb_readiness": wandb_readiness,
        "retained_output_budget": retained_output_budget,
        "output_root": single.display_path(out_dir),
        "output_disk_free_bytes": output_disk_free_bytes,
        "output_disk_error": output_disk_error,
        "live_resources": live_resources,
        "live_resource_thresholds": single.LIVE_RESOURCE_THRESHOLDS,
        "live_resource_error": live_resource_error,
        "instruction": (
            "A pass only establishes host/source readiness; rerun with "
            "--execute --reuse-existing --max-new-runs 1 and one exact "
            "--run-key to launch a bounded row."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument(
        "--out-dir",
        type=Path,
        help=(
            "Matrix output root. Defaults to the canonical output_root "
            "declared by the matrix config."
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help=(
            "Audit clean source plus incident-calibrated and live host "
            "resources without loading data or launching a renderer."
        ),
    )
    parser.add_argument(
        "--aggregate-existing",
        action="store_true",
        help="Package only complete clean-source run_summary.json files; never launch a trainer.",
    )
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--run-key",
        action="append",
        default=[],
        help=(
            "Execute only this exact matrix key; repeat for multiple keys. "
            "Use the dry-run output to copy keys exactly."
        ),
    )
    parser.add_argument(
        "--max-new-runs",
        type=int,
        default=0,
        help=(
            "Stop successfully after this many previously-unaccepted rows. "
            "Zero means no limit. Limited execution requires --reuse-existing."
        ),
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    parser.add_argument(
        "--check-wandb-connectivity",
        action="store_true",
        help=(
            "During --preflight-only, make one credential-free TCP reachability "
            "probe to api.wandb.ai. No network probe occurs by default."
        ),
    )
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument(
        "--require-clean-source",
        action="store_true",
        help="Compatibility flag; paper execution requires clean source by default.",
    )
    source_group.add_argument(
        "--allow-dirty-source",
        action="store_true",
        help=(
            "Allow labelled mechanical smokes from dirty source. Such runs "
            "cannot enter accepted paper evidence."
        ),
    )
    parser.add_argument(
        "--allow-local-mps-execution",
        action="store_true",
        help="Enable only after explicit user approval; local MPS execution is otherwise fail-closed.",
    )
    parser.add_argument(
        "--allow-high-risk-local-mps",
        action="store_true",
        help="Second acknowledgement for runs estimated above 60%% of host physical memory.",
    )
    args = parser.parse_args()
    selected_modes = sum(
        bool(value)
        for value in (
            args.execute,
            args.preflight_only,
            args.aggregate_existing,
        )
    )
    if selected_modes > 1:
        parser.error(
            "--execute, --preflight-only, and --aggregate-existing are "
            "mutually exclusive"
        )
    if args.check_wandb_connectivity and not args.preflight_only:
        parser.error("--check-wandb-connectivity requires --preflight-only")
    if args.max_new_runs < 0:
        parser.error("--max-new-runs must be nonnegative")
    if args.aggregate_existing and (args.run_key or args.max_new_runs):
        parser.error(
            "--aggregate-existing always audits the complete matrix; "
            "--run-key/--max-new-runs do not apply"
        )
    if args.execute and (args.run_key or args.max_new_runs) and not args.reuse_existing:
        parser.error(
            "bounded matrix execution requires --reuse-existing so accepted "
            "rows cannot be relaunched"
        )

    matrix_path = single.resolve_root_path(args.matrix)
    raw_matrix = load_config_file(matrix_path)
    runs = expand_matrix(raw_matrix)
    try:
        selected_runs = select_matrix_runs(runs, args.run_key)
    except ValueError as error:
        parser.error(str(error))
    try:
        out_dir = resolve_matrix_output_dir(raw_matrix, args.out_dir)
    except ValueError as error:
        parser.error(str(error))
    if args.preflight_only:
        result = matrix_preflight(
            selected_runs,
            device=args.device,
            wandb_mode=args.wandb_mode,
            check_wandb_connectivity=args.check_wandb_connectivity,
            out_dir=out_dir,
        )
        print(json.dumps(serialize_config_value(result), indent=2, sort_keys=True))
        if result["status"] != "pass":
            raise SystemExit(2)
        return
    if args.aggregate_existing:
        records, missing = collect_existing_records(runs, out_dir)
        if not records:
            raise ValueError(f"no complete existing paper rows under {out_dir}")
        artifact_dir = out_dir / "accepted_existing_evidence"
        artifacts = write_artifacts(artifact_dir, str(raw_matrix["name"]), records)
        result = {
            "status": "complete_existing_evidence" if not missing else "partial_existing_evidence",
            "execution": "none",
            "matrix": raw_matrix["name"],
            "expected_run_count": len(runs),
            "accepted_run_count": len(records),
            "accepted_lane_row_count": len(records) * len(LANE_ORDER),
            "accepted_runs": [record["run"] for record in records],
            "missing_runs": missing,
            "artifacts": artifacts,
        }
        single.write_json(out_dir / "existing_evidence_summary.json", result)
        single.write_json(
            out_dir / "matrix_progress.json",
            {
                "status": result["status"],
                "matrix": raw_matrix["name"],
                "expected_run_count": len(runs),
                "accepted_runs": result["accepted_runs"],
                "missing_runs": missing,
            },
        )
        print(json.dumps(serialize_config_value(result), indent=2, sort_keys=True))
        return
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
                        "selected_runs": [
                            run.as_dict() for run in selected_runs
                        ],
                        "max_new_runs": args.max_new_runs,
                    }
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return

    accepted_by_key: dict[str, dict[str, Any]] = {}
    if args.reuse_existing:
        existing_records, _initial_missing = collect_existing_records(runs, out_dir)
        accepted_by_key = {
            str(record["run"]["key"]): dict(record)
            for record in existing_records
        }
        single.write_json(
            out_dir / "matrix_progress.json",
            matrix_progress_payload(
                matrix_name=str(raw_matrix["name"]),
                runs=runs,
                accepted_records=existing_records,
                status="resuming",
                selected_runs=selected_runs,
                new_run_count=0,
            ),
        )

    new_run_count = 0
    for run in selected_runs:
        accepted_record = accepted_by_key.get(run.key)
        if accepted_record is not None:
            continue
        if args.max_new_runs and new_run_count >= args.max_new_runs:
            break
        raw_protocol = load_config_file(run.protocol_path)
        protocol = resolve_paper_training_protocol(raw_protocol)
        try:
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
                require_clean_source=not args.allow_dirty_source,
                allow_local_mps_execution=args.allow_local_mps_execution,
                allow_high_risk_local_mps=args.allow_high_risk_local_mps,
            )
            summary["run_summary_path"] = str(
                out_dir
                / protocol.name
                / f"seed_{run.seed}"
                / "run_summary.json"
            )
            validate_existing_summary(
                run,
                summary,
                protocol=protocol,
                summary_path=Path(summary["run_summary_path"]),
            )
        except Exception as error:
            accepted_records = [
                accepted_by_key[candidate.key]
                for candidate in runs
                if candidate.key in accepted_by_key
            ]
            single.write_json(
                out_dir / "matrix_progress.json",
                matrix_failure_payload(
                    matrix_name=str(raw_matrix["name"]),
                    runs=runs,
                    accepted_records=accepted_records,
                    selected_runs=selected_runs,
                    new_run_count=new_run_count,
                    failed_run=run,
                    error=error,
                ),
            )
            raise
        accepted_by_key[run.key] = {
            "run": run.as_dict(),
            "summary": summary,
        }
        new_run_count += 1
        accepted_records = [
            accepted_by_key[candidate.key]
            for candidate in runs
            if candidate.key in accepted_by_key
        ]
        single.write_json(
            out_dir / "matrix_progress.json",
            matrix_progress_payload(
                matrix_name=str(raw_matrix["name"]),
                runs=runs,
                accepted_records=accepted_records,
                status="running",
                selected_runs=selected_runs,
                new_run_count=new_run_count,
            ),
        )
    records = [
        accepted_by_key[run.key]
        for run in runs
        if run.key in accepted_by_key
    ]
    progress = matrix_progress_payload(
        matrix_name=str(raw_matrix["name"]),
        runs=runs,
        accepted_records=records,
        status="complete" if len(records) == len(runs) else "partial",
        selected_runs=selected_runs,
        new_run_count=new_run_count,
    )
    if progress["missing_runs"]:
        single.write_json(out_dir / "matrix_progress.json", progress)
        result = {
            **progress,
            "execution": "bounded",
            "message": (
                "The requested bounded execution finished; rerun with "
                "--reuse-existing for the next missing row."
            ),
        }
        print(json.dumps(serialize_config_value(result), indent=2, sort_keys=True))
        return

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
    single.write_json(out_dir / "matrix_progress.json", progress)
    print(json.dumps(serialize_config_value(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
