#!/usr/bin/env python3
"""Generate deterministic Paper-B G4-v2 assets from accepted evidence.

The v2 protocol matches selected training targets and RGB-MSE across all four
routes, then evaluates the unchanged full 300-frame held-out camera.  Target
pixels are matched; rasterized work is deliberately reported rather than
claimed equal because the Gaussian controls render full training images while
WorldFoam evaluates the selected rays directly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import mean, stdev
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LANE2 = Path(__file__).resolve().parent
for import_root in (ROOT, LANE2, ROOT / "src" / "train"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import verify_worldfoam_public_quality_ablation as g4_v1_verifier  # noqa: E402
import verify_worldfoam_public_quality_ablation_v2 as g4_v2_verifier  # noqa: E402


DEFAULT_CONFIG = g4_v2_verifier.DEFAULT_CONFIG
REQUIRED_ROUTES = g4_v2_verifier.REQUIRED_ROUTES
file_sha256 = g4_v2_verifier.file_sha256


ASSET_SCHEMA_VERSION = 2
GENERATOR = "worldfoam-public-quality-assets-v2-selected-rays"
DEFAULT_ARTIFACT = (
    ROOT
    / "outputs"
    / "benchmarks"
    / "worldfoam_native4d_g4_public_quality_v2_selected_rays"
    / "worldfoam_public_quality_selected_ray_ablation.json"
)
DEFAULT_OUT_DIR = DEFAULT_ARTIFACT.parent / "paper_assets"
ROUTE_LABELS = {
    "worldfoam_native4d": "WorldFoam compiled",
    "worldfoam_framewise_replay": "WorldFoam replay",
    "world_tubes": "World Tubes (selected-time)",
    "dynamic_3dgs": "Dynamic 3DGS",
}
ROUTE_COLORS = {
    "worldfoam_native4d": "#7b2cbf",
    "worldfoam_framewise_replay": "#c77dff",
    "world_tubes": "#168aad",
    "dynamic_3dgs": "#f4a261",
}


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return payload


def _summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for route in REQUIRED_ROUTES:
        selected = [row for row in rows if row.get("route") == route]
        if len(selected) != 9:
            raise ValueError(f"{route} does not contain nine public rows")
        record: dict[str, Any] = {
            "route": route,
            "label": ROUTE_LABELS[route],
            "row_count": len(selected),
        }
        for metric in (
            "heldout_eval_psnr",
            "heldout_eval_ssim",
            "heldout_eval_lpips",
            "heldout_eval_l1",
        ):
            values = [float(row["metrics"][metric]) for row in selected]
            record[f"{metric}_mean"] = mean(values)
            record[f"{metric}_std"] = stdev(values)
        for metric in (
            "target_pixels",
            "rasterized_pixels",
            "parameter_count",
            "training_and_checkpoint_elapsed_s",
            "full_row_through_heldout_evaluation_elapsed_s",
            "sampled_peak_mps_driver_during_training_and_checkpoint_bytes",
            "sampled_peak_mps_driver_through_heldout_evaluation_bytes",
            "process_lifetime_peak_rss_through_checkpoint_bytes",
            "process_lifetime_peak_rss_through_heldout_evaluation_bytes",
            "parameter_bytes",
            "stored_primitive_state_count",
        ):
            values = [float(row["cost"][metric]) for row in selected]
            record[f"{metric}_mean"] = mean(values)
            record[f"{metric}_std"] = stdev(values)
        result.append(record)
    return result


def _scene_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    scenes = ("coffee_martini", "cook_spinach", "cut_roasted_beef")
    result: list[dict[str, Any]] = []
    for scene in scenes:
        for route in REQUIRED_ROUTES:
            selected = [
                row
                for row in rows
                if row.get("scene") == scene and row.get("route") == route
            ]
            if len(selected) != 3:
                raise ValueError(f"{scene}/{route} does not contain three seeds")
            values = [float(row["metrics"]["heldout_eval_psnr"]) for row in selected]
            result.append(
                {
                    "scene": scene,
                    "route": route,
                    "label": ROUTE_LABELS[route],
                    "psnr_mean": mean(values),
                    "psnr_std": stdev(values),
                }
            )
    return result


def _csv(summary: Sequence[Mapping[str, Any]]) -> bytes:
    columns = tuple(summary[0])
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(summary)
    return stream.getvalue().encode("utf-8")


def _tex(summary: Sequence[Mapping[str, Any]]) -> bytes:
    selected_v2 = all(
        int(round(float(row["target_pixels_mean"]))) == 1_228_800
        for row in summary
    )
    lines = [
        (
            "% Generated only from independently accepted 36-row G4-v2 evidence."
            if selected_v2
            else "% Generated only from independently accepted 36-row G4-v1 evidence."
        ),
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Method & PSNR$\uparrow$ & SSIM$\uparrow$ & LPIPS$\downarrow$ & L1$\downarrow$ & Rasterized Mpx$\downarrow$ & Train+ckpt (s)$\downarrow$ & Train+ckpt MPS GiB$\downarrow$ \\",
        r"\midrule",
    ]
    for row in summary:
        mps_gib = float(
            row[
                "sampled_peak_mps_driver_during_training_and_checkpoint_bytes_mean"
            ]
        ) / 2**30
        lines.append(
            "{} & {:.2f}$\\pm${:.2f} & {:.3f}$\\pm${:.3f} & "
            "{:.3f}$\\pm${:.3f} & {:.4f}$\\pm${:.4f} & {:.2f} & {:.1f} & {:.2f} \\\\".format(
                row["label"],
                row["heldout_eval_psnr_mean"],
                row["heldout_eval_psnr_std"],
                row["heldout_eval_ssim_mean"],
                row["heldout_eval_ssim_std"],
                row["heldout_eval_lpips_mean"],
                row["heldout_eval_lpips_std"],
                row["heldout_eval_l1_mean"],
                row["heldout_eval_l1_std"],
                float(row["rasterized_pixels_mean"]) / 1.0e6,
                row["training_and_checkpoint_elapsed_s_mean"],
                mps_gib,
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{G4-v2 final-checkpoint full-temporal held-out quality over three calibrated Neural3D scenes and seeds 17/29/43. Quality cells report mean $\pm$ standard deviation across all nine rows per route; rasterized work, time, and MPS cells report means, with their dispersion retained in the generated CSV/JSON. Every route consumes the same 1,228,800 selected training targets and RGB-MSE contract; rasterized work is reported rather than claimed equal. Time and MPS cover training through final-checkpoint serialization. Compiled and replay WorldFoam use the identical retained-depth representation.}"
                if selected_v2
                else r"\caption{G4-v1 final-checkpoint full-temporal held-out quality over three calibrated Neural3D scenes and seeds 17/29/43. Quality cells report mean $\pm$ standard deviation across all nine rows per route; time and MPS cells report means, with their dispersion retained in the generated CSV/JSON. Time and MPS cover training through final-checkpoint serialization. Compiled and replay WorldFoam use the identical retained-depth representation.}"
            ),
            r"\label{tab:worldfoam-g4-public-quality}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _svg(scene_rows: Sequence[Mapping[str, Any]]) -> bytes:
    width, height = 1200, 620
    left, right, top, bottom = 90.0, 35.0, 75.0, 120.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_value = max(float(row["psnr_mean"] + row["psnr_std"]) for row in scene_rows)
    y_max = max(15.0, math.ceil(max_value / 5.0) * 5.0)
    scenes = ("coffee_martini", "cook_spinach", "cut_roasted_beef")
    body = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        '<text x="600" y="34" text-anchor="middle" font-family="sans-serif" '
        'font-size="24" font-weight="700">G4 public held-out PSNR</text>',
        '<text x="600" y="58" text-anchor="middle" font-family="sans-serif" '
        'font-size="14" fill="#444">mean ± standard deviation over seeds 17/29/43</text>',
    ]
    for tick in range(0, int(y_max) + 1, 5):
        y = top + plot_height * (1.0 - tick / y_max)
        body.extend(
            [
                f'<line x1="{left}" y1="{y:.2f}" x2="{width-right}" y2="{y:.2f}" stroke="#ddd"/>',
                f'<text x="{left-12}" y="{y+5:.2f}" text-anchor="end" font-family="sans-serif" font-size="13">{tick}</text>',
            ]
        )
    group_width = plot_width / len(scenes)
    bar_width = group_width * 0.16
    gap = group_width * 0.025
    for scene_index, scene in enumerate(scenes):
        selected = [row for row in scene_rows if row["scene"] == scene]
        start = left + scene_index * group_width + group_width * 0.12
        for route_index, row in enumerate(selected):
            value = float(row["psnr_mean"])
            error = float(row["psnr_std"])
            x = start + route_index * (bar_width + gap)
            y = top + plot_height * (1.0 - value / y_max)
            bar_height = top + plot_height - y
            center = x + bar_width / 2.0
            upper = top + plot_height * (1.0 - min(y_max, value + error) / y_max)
            lower = top + plot_height * (1.0 - max(0.0, value - error) / y_max)
            color = ROUTE_COLORS[str(row["route"])]
            body.extend(
                [
                    f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}"/>',
                    f'<line x1="{center:.2f}" y1="{upper:.2f}" x2="{center:.2f}" y2="{lower:.2f}" stroke="#222" stroke-width="2"/>',
                    f'<line x1="{center-6:.2f}" y1="{upper:.2f}" x2="{center+6:.2f}" y2="{upper:.2f}" stroke="#222" stroke-width="2"/>',
                    f'<line x1="{center-6:.2f}" y1="{lower:.2f}" x2="{center+6:.2f}" y2="{lower:.2f}" stroke="#222" stroke-width="2"/>',
                ]
            )
        label = scene.replace("_", " ").title()
        body.append(
            f'<text x="{left+(scene_index+0.5)*group_width:.2f}" y="{height-82}" text-anchor="middle" font-family="sans-serif" font-size="15">{html.escape(label)}</text>'
        )
    legend_y = height - 38
    for index, route in enumerate(REQUIRED_ROUTES):
        x = 125 + index * 260
        body.extend(
            [
                f'<rect x="{x}" y="{legend_y-14}" width="18" height="18" fill="{ROUTE_COLORS[route]}"/>',
                f'<text x="{x+26}" y="{legend_y}" font-family="sans-serif" font-size="14">{html.escape(ROUTE_LABELS[route])}</text>',
            ]
        )
    body.append(
        f'<text transform="translate(24 {top+plot_height/2:.2f}) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="16">PSNR (dB)</text>'
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        "<title>WorldFoam G4 public held-out quality</title>"
        "<desc>Verifier-derived held-out PSNR by public scene and route, with seed standard-deviation error bars.</desc>"
        + "".join(body)
        + "</svg>\n"
    ).encode("utf-8")


def _verify_supported_artifact(
    artifact_path: Path,
    config_path: Path,
) -> tuple[dict[str, Any], str]:
    try:
        v2_report = g4_v2_verifier.verify_artifact_file(
            artifact_path, config_path=config_path
        )
    except Exception as error:
        v2_report = {
            "accepted": False,
            "failures": [f"verifier raised {type(error).__name__}: {error}"],
        }
    if v2_report.get("accepted") is True:
        return dict(v2_report), "g4_v2_selected_rays"
    try:
        v1_report = g4_v1_verifier.verify_artifact_file(
            artifact_path, config_path=config_path
        )
    except Exception as error:
        v1_report = {
            "accepted": False,
            "failures": [f"verifier raised {type(error).__name__}: {error}"],
        }
    if v1_report.get("accepted") is True:
        return dict(v1_report), "g4_v1_all_pixels"
    failures = [
        *(f"v2: {value}" for value in v2_report.get("failures", ())),
        *(f"v1: {value}" for value in v1_report.get("failures", ())),
    ]
    return {"accepted": False, "failures": failures}, "rejected"


def build_assets(artifact_path: Path, config_path: Path) -> dict[str, bytes]:
    report, protocol = _verify_supported_artifact(artifact_path, config_path)
    if report.get("accepted") is not True:
        raise ValueError(
            "G4 assets require an independently accepted artifact: "
            + "; ".join(report.get("failures", ()))
        )
    payload = _load_json(artifact_path)
    rows = payload["rows"]
    summary = _summary(rows)
    scene_rows = _scene_rows(rows)
    return {
        "g4_public_quality_rows.csv": _csv(summary),
        "g4_public_quality_table.tex": _tex(summary),
        "g4_public_quality.svg": _svg(scene_rows),
        "g4_public_quality_summary.json": _canonical_bytes(
            {
                "schema_version": 2,
                "protocol": protocol,
                "artifact_sha256": file_sha256(artifact_path),
                "artifact_canonical_sha256": payload["artifact_sha256"],
                "acceptance": payload["acceptance"],
                "route_summary": summary,
                "scene_psnr_summary": scene_rows,
            }
        ),
    }


def write_assets(
    artifact_path: Path, config_path: Path, output_dir: Path
) -> dict[str, Any]:
    files = build_assets(artifact_path, config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = set(files) | {"manifest.json"}
    existing = {path.name for path in output_dir.iterdir() if path.is_file()}
    unexpected = sorted(existing - expected)
    if unexpected:
        raise ValueError("G4 asset directory contains unexpected files: " + ", ".join(unexpected))
    for name, data in files.items():
        (output_dir / name).write_bytes(data)
    manifest = {
        "schema_version": ASSET_SCHEMA_VERSION,
        "generator": GENERATOR,
        "artifact": str(artifact_path.resolve()),
        "artifact_sha256": file_sha256(artifact_path),
        "config": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "files": [
            {
                "path": name,
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
            for name, data in sorted(files.items())
        ],
    }
    (output_dir / "manifest.json").write_bytes(_canonical_bytes(manifest))
    return manifest


def verify_asset_dir(
    output_dir: Path, artifact_path: Path, config_path: Path
) -> list[str]:
    failures: list[str] = []
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return ["G4 asset manifest is missing"]
    try:
        manifest = _load_json(manifest_path)
        expected = build_assets(artifact_path, config_path)
    except Exception as error:
        return [f"G4 assets could not be independently rebuilt: {error}"]
    if manifest.get("schema_version") != ASSET_SCHEMA_VERSION:
        failures.append("G4 asset schema changed")
    if manifest.get("generator") != GENERATOR:
        failures.append("G4 asset generator changed")
    if manifest.get("artifact_sha256") != file_sha256(artifact_path):
        failures.append("G4 asset artifact binding changed")
    if manifest.get("config_sha256") != file_sha256(config_path):
        failures.append("G4 asset config binding changed")
    rows = manifest.get("files")
    if not isinstance(rows, list):
        failures.append("G4 asset manifest files are missing")
        rows = []
    by_name = {
        str(row.get("path")): row for row in rows if isinstance(row, Mapping)
    }
    if set(by_name) != set(expected):
        failures.append("G4 asset manifest file set changed")
    for name, data in expected.items():
        path = output_dir / name
        if not path.is_file() or path.read_bytes() != data:
            failures.append(f"G4 asset is missing or nondeterministic: {name}")
            continue
        if by_name.get(name, {}).get("bytes") != len(data):
            failures.append(f"G4 asset byte count changed: {name}")
        if by_name.get(name, {}).get("sha256") != hashlib.sha256(data).hexdigest():
            failures.append(f"G4 asset digest changed: {name}")
    actual = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual != set(expected) | {"manifest.json"}:
        failures.append("G4 asset directory contains missing or unexpected files")
    return sorted(set(failures))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate verified WorldFoam G4 assets.")
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        failures = verify_asset_dir(args.out_dir, args.artifact, args.config)
        print(json.dumps({"accepted": not failures, "failures": failures}, indent=2, sort_keys=True))
        return 0 if not failures else 1
    manifest = write_assets(args.artifact, args.config, args.out_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "build_assets",
    "verify_asset_dir",
    "write_assets",
]
