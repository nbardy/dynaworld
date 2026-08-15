from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts" / "spd4_bounded_16f_40step"
DEFAULT_OUT = DEFAULT_ARTIFACT_ROOT / "summary.json"


@dataclass(frozen=True)
class RowSpec:
    label: str
    directory: str
    world_representation: str
    alpha_mode: str
    amplitude_convention: str
    atom_count: int
    parameter_count: int


ROW_SPECS = (
    RowSpec(
        label="legacy_peak",
        directory="legacy_256",
        world_representation="legacy_tube",
        alpha_mode="peak_splat",
        amplitude_convention="fiber_integrated",
        atom_count=256,
        parameter_count=3_584,
    ),
    RowSpec(
        label="full_spd4_peak_parameter_matched",
        directory="full_spd4_199_param_matched_optimized",
        world_representation="full_spd4",
        alpha_mode="peak_splat",
        amplitude_convention="fiber_integrated",
        atom_count=199,
        parameter_count=3_582,
    ),
    RowSpec(
        label="full_spd4_beer_fiber_parameter_matched",
        directory="full_spd4_199_beer_fiber_optimized",
        world_representation="full_spd4",
        alpha_mode="beer_lambert",
        amplitude_convention="fiber_integrated",
        atom_count=199,
        parameter_count=3_582,
    ),
)

QUALITY_KEYS = (
    "heldout_eval_psnr",
    "heldout_eval_ssim",
    "heldout_eval_lpips",
    "heldout_eval_l1",
)
MEDIA_NAMES = (
    "star_uvt_train_view0_preview.png",
    "star_uvt_train_view0_side_by_side.mp4",
    "star_uvt_heldout_view0_preview.png",
    "star_uvt_heldout_view0_side_by_side.mp4",
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _finite_float(value: Any, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def summarize_row(artifact_root: Path, spec: RowSpec) -> dict[str, Any]:
    row_dir = artifact_root / spec.directory
    report_path = row_dir / "comparison_report.json"
    report = _load_json(report_path)
    meta = report["meta"]
    lane = report["star_uvt"]
    paper = lane["paper_protocol"]
    cost = paper["cost"]
    timing = paper["timing"]
    metrics = lane["metrics"]
    metal_rows = lane["metal_stats"]["rows"]
    media = [row_dir / name for name in MEDIA_NAMES]
    return {
        "label": spec.label,
        "artifact_directory": _display_path(row_dir),
        "source_report": _display_path(report_path),
        "source_report_sha256": _sha256(report_path),
        "world_representation": lane["world_representation"],
        "alpha_mode": lane["alpha_mode"],
        "amplitude_convention": lane["amplitude_convention"],
        "render_backend": lane["render_backend"],
        "backward_policy": meta["uvt_backward_policy"]["name"],
        "seed": int(meta["seed"]),
        "frame_count": int(meta["frame_count"]),
        "image_size": list(meta["image_size"]),
        "train_cameras": list(meta["train_cameras"]),
        "heldout_cameras": list(meta["heldout_cameras"]),
        "steps": int(lane["steps"]),
        "atom_count": int(lane["tube_count"]),
        "parameter_count": int(cost["parameter_count"]),
        "parameter_bytes": int(cost["parameter_bytes"]),
        "optimizer_state_bytes": int(cost["optimizer_state_bytes"]),
        "serialized_checkpoint_bytes": int(cost["serialized_checkpoint_bytes"]),
        "target_frames": int(cost["target_frames"]),
        "rasterized_frames": int(cost["rasterized_frames"]),
        "target_pixels": int(cost["target_pixels"]),
        "rasterized_pixels": int(cost["rasterized_pixels"]),
        "sampled_peak_current_allocated_bytes": int(
            cost["sampled_peak_current_allocated_bytes"]
        ),
        "sampled_peak_driver_allocated_bytes": int(
            cost["sampled_peak_driver_allocated_bytes"]
        ),
        "train_wall_s": _finite_float(timing["train_wall_s"], name="train_wall_s"),
        "cold_compile_forward_s": _finite_float(
            timing["cold_compile_forward_s"],
            name="cold_compile_forward_s",
        ),
        "steady_forward_s": _finite_float(
            timing["steady_forward_s"],
            name="steady_forward_s",
        ),
        "backward_s": _finite_float(timing["backward_s"], name="backward_s"),
        "optimizer_s": _finite_float(timing["optimizer_s"], name="optimizer_s"),
        **{
            key: _finite_float(metrics[key], name=key)
            for key in QUALITY_KEYS
        },
        "max_overflow_tile_count": max(
            int(row["stats"]["overflow_tile_count"]) for row in metal_rows
        ),
        "media": [_display_path(path) for path in media],
        "media_all_present": all(path.is_file() for path in media),
    }


def summarize(artifact_root: Path = DEFAULT_ARTIFACT_ROOT) -> dict[str, Any]:
    rows = [summarize_row(artifact_root, spec) for spec in ROW_SPECS]
    by_label = {row["label"]: row for row in rows}
    legacy = by_label["legacy_peak"]
    spd4_peak = by_label["full_spd4_peak_parameter_matched"]
    spd4_beer = by_label["full_spd4_beer_fiber_parameter_matched"]
    return {
        "schema_version": 1,
        "benchmark": "world_tubes_spd4_bounded_16f_40step_seed17",
        "scope": (
            "single-scene single-seed bounded convergence and physical-renderer "
            "ablation; not a publication-scale quality claim"
        ),
        "provenance": {
            "execution_source_state": "uncommitted_working_tree",
            "packaging_note": (
                "Report hashes identify the accepted raw artifacts. The execution "
                "predated the durable source commit and therefore is not clean-source "
                "publication evidence."
            ),
        },
        "rows": rows,
        "summary": {
            "parameter_count_delta_spd4_vs_legacy": (
                spd4_peak["parameter_count"] - legacy["parameter_count"]
            ),
            "spd4_peak_heldout_psnr_gain_db": (
                spd4_peak["heldout_eval_psnr"] - legacy["heldout_eval_psnr"]
            ),
            "spd4_beer_heldout_psnr_gain_db": (
                spd4_beer["heldout_eval_psnr"] - legacy["heldout_eval_psnr"]
            ),
            "spd4_peak_train_wall_ratio": (
                spd4_peak["train_wall_s"] / legacy["train_wall_s"]
            ),
            "spd4_beer_train_wall_ratio": (
                spd4_beer["train_wall_s"] / legacy["train_wall_s"]
            ),
            "spd4_peak_driver_memory_ratio": (
                spd4_peak["sampled_peak_driver_allocated_bytes"]
                / legacy["sampled_peak_driver_allocated_bytes"]
            ),
            "spd4_beer_driver_memory_ratio": (
                spd4_beer["sampled_peak_driver_allocated_bytes"]
                / legacy["sampled_peak_driver_allocated_bytes"]
            ),
        },
    }


def verify(report: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("schema_version") != 1:
        errors.append("schema_version must equal 1")
    if report.get("benchmark") != "world_tubes_spd4_bounded_16f_40step_seed17":
        errors.append("benchmark identity drifted")
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != len(ROW_SPECS):
        return [*errors, f"expected exactly {len(ROW_SPECS)} rows"]
    by_label = {
        row.get("label"): row
        for row in rows
        if isinstance(row, Mapping)
    }
    for spec in ROW_SPECS:
        row = by_label.get(spec.label)
        if row is None:
            errors.append(f"missing row {spec.label}")
            continue
        expected = {
            "world_representation": spec.world_representation,
            "alpha_mode": spec.alpha_mode,
            "amplitude_convention": spec.amplitude_convention,
            "render_backend": "metal_tile",
            "backward_policy": "fast_exploration",
            "seed": 17,
            "frame_count": 16,
            "image_size": [96, 128],
            "train_cameras": ["cam04", "cam09"],
            "heldout_cameras": ["cam06"],
            "steps": 40,
            "atom_count": spec.atom_count,
            "parameter_count": spec.parameter_count,
            "target_frames": 160,
            "rasterized_frames": 160,
            "target_pixels": 1_597_440,
            "rasterized_pixels": 1_597_440,
            "max_overflow_tile_count": 0,
            "media_all_present": True,
        }
        for key, value in expected.items():
            if row.get(key) != value:
                errors.append(
                    f"{spec.label}.{key} expected {value!r}, got {row.get(key)!r}"
                )
        digest = row.get("source_report_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            errors.append(f"{spec.label}.source_report_sha256 is invalid")
        for key in (
            *QUALITY_KEYS,
            "train_wall_s",
            "cold_compile_forward_s",
            "steady_forward_s",
            "backward_s",
            "optimizer_s",
        ):
            try:
                value = float(row[key])
            except (KeyError, TypeError, ValueError):
                errors.append(f"{spec.label}.{key} is missing or nonnumeric")
                continue
            if not math.isfinite(value):
                errors.append(f"{spec.label}.{key} must be finite")
        for key in (
            "parameter_bytes",
            "optimizer_state_bytes",
            "serialized_checkpoint_bytes",
            "sampled_peak_current_allocated_bytes",
            "sampled_peak_driver_allocated_bytes",
        ):
            if not isinstance(row.get(key), int) or int(row[key]) <= 0:
                errors.append(f"{spec.label}.{key} must be a positive integer")

    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        errors.append("summary is missing")
        return errors
    if summary.get("parameter_count_delta_spd4_vs_legacy") != -2:
        errors.append("SPD4 parameter-matched row must be within two scalars of legacy")
    for key in (
        "spd4_peak_heldout_psnr_gain_db",
        "spd4_beer_heldout_psnr_gain_db",
    ):
        try:
            gain = float(summary[key])
        except (KeyError, TypeError, ValueError):
            errors.append(f"summary.{key} is missing or nonnumeric")
        else:
            if not math.isfinite(gain) or gain <= 0.0:
                errors.append(f"summary.{key} must be finite and positive")
    return errors


def assert_valid(report: Mapping[str, Any]) -> None:
    errors = verify(report)
    if errors:
        raise ValueError("invalid bounded SPD4 report:\n- " + "\n- ".join(errors))


def write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# Bounded native SPD(4) World Tubes ablation",
        "",
        str(report["scope"]),
        "",
        "| Row | Atoms | Parameters | PSNR | SSIM | LPIPS | L1 | Train wall (s) | Peak driver (MB) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['label']} | {row['atom_count']} | {row['parameter_count']} | "
            f"{row['heldout_eval_psnr']:.4f} | {row['heldout_eval_ssim']:.5f} | "
            f"{row['heldout_eval_lpips']:.5f} | {row['heldout_eval_l1']:.5f} | "
            f"{row['train_wall_s']:.4f} | "
            f"{row['sampled_peak_driver_allocated_bytes'] / 1_000_000.0:.3f} |"
        )
    lines.extend(
        (
            "",
            "## Claim boundary",
            "",
            str(report["provenance"]["packaging_note"]),
            "",
        )
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--verify-report", type=Path)
    args = parser.parse_args()
    if args.verify_report is not None:
        assert_valid(_load_json(args.verify_report))
        return
    report = summarize(args.artifact_root.resolve())
    assert_valid(report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(args.out.with_suffix(".md"), report)


if __name__ == "__main__":
    main()
