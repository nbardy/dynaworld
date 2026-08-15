#!/usr/bin/env python3
"""Generate deterministic Paper-B G6 assets from accepted native evidence.

This is a presentation layer, not an evidence producer.  It refuses dry plans,
partial matrices, logical-only byte estimates, and artifacts rejected by the
independent 21-row G6 verifier.  Every value below is derived from the accepted
fresh-process rows; the generator never substitutes analytic state formulas for
measured RSS or MPS high-water marks.
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
from statistics import median
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
LANE2 = Path(__file__).resolve().parent
for import_root in (ROOT, LANE2):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from run_worldfoam_training_memory_ablation import DEFAULT_OUTPUT  # noqa: E402
from verify_worldfoam_training_memory_ablation import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_CONTRACT,
    file_sha256,
    verify_artifact_file,
)


ASSET_SCHEMA_VERSION = 1
GENERATOR = "worldfoam-training-memory-assets-v1"
DEFAULT_OUT_DIR = DEFAULT_OUTPUT.parent / "paper_assets"
MODE_ORDER = (
    "staged_sparse",
    "fused_union_v2",
    "per_frame_replay_sequential",
)
MODE_LABELS = {
    "staged_sparse": "Staged sparse reverse",
    "fused_union_v2": "Fused shared reverse",
    "per_frame_replay_sequential": "Sequential replay",
}
MODE_COLORS = {
    "staged_sparse": "#6c757d",
    "fused_union_v2": "#7b2cbf",
    "per_frame_replay_sequential": "#168aad",
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


def _finite(value: Any, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"expected finite numeric value, got {value!r}")
    number = float(value)
    if not math.isfinite(number) or number < 0.0 or (positive and number <= 0.0):
        raise ValueError(f"expected {'positive' if positive else 'nonnegative'} value")
    return number


def _row_values(row: Mapping[str, Any]) -> dict[str, float]:
    mode = str(row["mode"])
    memory = row["memory"]
    execution = row["execution"]
    measurement = row["measurement"]
    watchdog = measurement["parent_watchdog"]
    if mode == "per_frame_replay_sequential":
        accounting = row["work"]["accounting"]
        adapter_measurements = execution["adapter_measurements"]
        core_wall = accounting["step_wall_time_seconds"]
        transaction_wall = adapter_measurements[
            "control_transaction_wall_time_seconds"
        ]
        compile_wall = adapter_measurements[
            "continuous_precompile_wall_time_seconds"
        ]
        ordered = accounting["ordered_word_node_interactions"]
    else:
        core_wall = execution["core_forward_backward_wall_time_seconds"]
        transaction_wall = execution["step_wall_time_seconds"]
        compile_wall = execution["cold_cpu_compile_wall_time_seconds"]
        ordered = row["work"]["ordered_word_node_interactions"]
    return {
        # The two common timing scopes are deliberately separate.  Core time
        # is the route's render/reverse interval.  Fresh-process time comes
        # from the same parent-watchdog scope for every route and includes
        # startup, imports, attestation, preparation, the transaction, and
        # teardown.  Route-specific transaction/compile measurements remain
        # available but are never presented as if they had identical scope.
        "core_forward_backward_wall_time_seconds": _finite(
            core_wall, positive=True
        ),
        "fresh_process_end_to_end_wall_time_seconds": _finite(
            watchdog["elapsed_seconds"], positive=True
        ),
        "route_transaction_wall_time_seconds": _finite(
            transaction_wall, positive=True
        ),
        "route_compile_wall_time_seconds": _finite(
            compile_wall, positive=True
        ),
        "sampled_mps_driver_peak_bytes": _finite(
            memory["sampled_mps_driver_peak_bytes"], positive=True
        ),
        "sampled_mps_driver_delta_bytes": _finite(
            memory["sampled_mps_driver_peak_bytes"]
            - memory["sampled_mps_driver_baseline_bytes"]
        ),
        "process_rss_peak_bytes": _finite(
            memory["process_rss_peak_bytes"], positive=True
        ),
        "process_rss_delta_bytes": _finite(
            memory["process_rss_peak_bytes"]
            - memory["process_rss_baseline_bytes"]
        ),
        "process_group_rss_peak_bytes": _finite(
            memory["parent_process_group_rss_sampled_peak_bytes"], positive=True
        ),
        "ordered_word_node_interactions": _finite(ordered, positive=True),
    }


def _summaries(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_rows = [*payload["rows"], *payload["control_rows"]]
    groups: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    for row in raw_rows:
        key = (str(row["mode"]), int(row["requested_frame_count"]))
        groups.setdefault(key, []).append(row)
    expected = {
        ("staged_sparse", 8),
        ("fused_union_v2", 8),
        ("fused_union_v2", 64),
        ("fused_union_v2", 300),
        ("per_frame_replay_sequential", 8),
        ("per_frame_replay_sequential", 64),
        ("per_frame_replay_sequential", 300),
    }
    if set(groups) != expected:
        raise ValueError("G6 mode/frame grid changed after independent verification")
    summaries: list[dict[str, Any]] = []
    for mode in MODE_ORDER:
        for frame_count in (8, 64, 300):
            rows = groups.get((mode, frame_count))
            if rows is None:
                continue
            if len(rows) != 3 or {int(row["repeat_index"]) for row in rows} != {0, 1, 2}:
                raise ValueError(f"{mode}/F={frame_count} lacks three exact repeats")
            extracted = [_row_values(row) for row in rows]
            summary: dict[str, Any] = {
                "mode": mode,
                "label": MODE_LABELS[mode],
                "requested_frame_count": frame_count,
                "repeat_count": 3,
            }
            for key in extracted[0]:
                values = [float(row[key]) for row in extracted]
                summary[f"{key}_median"] = median(values)
                summary[f"{key}_min"] = min(values)
                summary[f"{key}_max"] = max(values)
            summaries.append(summary)
    return summaries


def _csv(summary: Sequence[Mapping[str, Any]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=tuple(summary[0]),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(summary)
    return stream.getvalue().encode("utf-8")


def _tex(summary: Sequence[Mapping[str, Any]]) -> bytes:
    lines = [
        "% Generated only from independently accepted 21-row G6 evidence.",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        (
            r"Route & $F$ & Core F/B (s)$\downarrow$ & Process E2E (s)$\downarrow$ "
            r"& MPS peak (GiB)$\downarrow$ "
            r"& Process-group RSS (GiB)$\downarrow$ & Ordered-word work$\downarrow$ \\"
        ),
        r"\midrule",
    ]
    for row in summary:
        lines.append(
            "{} & {} & {:.3f} [{:.3f}, {:.3f}] & {:.3f} [{:.3f}, {:.3f}] & {:.3f} & {:.3f} & {:.0f} \\\\".format(
                row["label"],
                row["requested_frame_count"],
                row["core_forward_backward_wall_time_seconds_median"],
                row["core_forward_backward_wall_time_seconds_min"],
                row["core_forward_backward_wall_time_seconds_max"],
                row["fresh_process_end_to_end_wall_time_seconds_median"],
                row["fresh_process_end_to_end_wall_time_seconds_min"],
                row["fresh_process_end_to_end_wall_time_seconds_max"],
                float(row["sampled_mps_driver_peak_bytes_median"]) / 2**30,
                float(row["process_group_rss_peak_bytes_median"]) / 2**30,
                row["ordered_word_node_interactions_median"],
            )
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\caption{Native WorldFoam G6 training-memory and temporal-sharing "
                r"ablation. Values are medians over three fresh processes; brackets "
                r"show measured timing ranges. Core F/B is the render/reverse interval "
                r"for each route. Process E2E uses the identical parent-watchdog scope "
                r"for every route and includes process startup, imports, attestation, "
                r"preparation, transaction, and teardown. Route-specific transaction "
                r"and compile timings remain separate in the JSON/CSV and are never "
                r"mixed. MPS and process-group RSS are "
                r"sampled high-water marks under hard 2-GiB and 4-GiB limits. The "
                r"sequential control reuses the identical compiled representation but "
                r"replays its ordered world work once per requested frame.}"
            ),
            r"\label{tab:worldfoam-g6-native-memory}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _polyline(points: Sequence[tuple[float, float]], color: str) -> str:
    encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<polyline points="{encoded}" fill="none" stroke="{color}" '
        'stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>'
    )


def _svg(summary: Sequence[Mapping[str, Any]]) -> bytes:
    width, height = 1240, 600
    plot_top, plot_height = 90.0, 380.0
    panels = ((85.0, 485.0), (690.0, 485.0))
    selected = {
        (str(row["mode"]), int(row["requested_frame_count"])): row
        for row in summary
    }
    frame_x = {8: 0.0, 64: 0.5, 300: 1.0}
    mps_max = max(
        float(row["sampled_mps_driver_peak_bytes_max"]) / 2**30
        for row in summary
    )
    mps_ymax = max(0.25, math.ceil(mps_max * 4.0) / 4.0)
    normalized: dict[tuple[str, int], float] = {}
    for mode in ("fused_union_v2", "per_frame_replay_sequential"):
        base = float(selected[(mode, 8)]["ordered_word_node_interactions_median"])
        for frame_count in (8, 64, 300):
            normalized[(mode, frame_count)] = (
                float(selected[(mode, frame_count)]["ordered_word_node_interactions_median"])
                / base
            )
    work_ymax = max(2.0, math.ceil(max(normalized.values()) / 5.0) * 5.0)
    body = [
        f'<rect width="{width}" height="{height}" fill="white"/>',
        '<text x="620" y="35" text-anchor="middle" font-family="sans-serif" '
        'font-size="24" font-weight="700">G6 native memory and temporal sharing</text>',
        '<text x="620" y="61" text-anchor="middle" font-family="sans-serif" '
        'font-size="14" fill="#444">medians over three fresh processes; shaded-free raw high-water evidence</text>',
    ]
    titles = ("Sampled MPS driver peak", "Ordered world work / F=8")
    y_maxima = (mps_ymax, work_ymax)
    for panel_index, ((left, plot_width), title, ymax) in enumerate(
        zip(panels, titles, y_maxima)
    ):
        body.append(
            f'<text x="{left + plot_width / 2:.2f}" y="82" text-anchor="middle" '
            f'font-family="sans-serif" font-size="17" font-weight="600">{html.escape(title)}</text>'
        )
        for tick_index in range(5):
            value = ymax * tick_index / 4.0
            y = plot_top + plot_height * (1.0 - tick_index / 4.0)
            body.extend(
                [
                    f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#ddd"/>',
                    f'<text x="{left - 10}" y="{y + 5:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.2f}</text>',
                ]
            )
        for frame_count in (8, 64, 300):
            x = left + frame_x[frame_count] * plot_width
            body.append(
                f'<text x="{x:.2f}" y="{plot_top + plot_height + 28:.2f}" text-anchor="middle" font-family="sans-serif" font-size="13">{frame_count}</text>'
            )
        body.append(
            f'<text x="{left + plot_width / 2:.2f}" y="{plot_top + plot_height + 54:.2f}" text-anchor="middle" font-family="sans-serif" font-size="14">requested frames F</text>'
        )
        for mode in ("fused_union_v2", "per_frame_replay_sequential"):
            points: list[tuple[float, float]] = []
            for frame_count in (8, 64, 300):
                x = left + frame_x[frame_count] * plot_width
                if panel_index == 0:
                    selected_row = selected[(mode, frame_count)]
                    value = float(
                        selected_row["sampled_mps_driver_peak_bytes_median"]
                    ) / 2**30
                    lower_value = float(
                        selected_row["sampled_mps_driver_peak_bytes_min"]
                    ) / 2**30
                    upper_value = float(
                        selected_row["sampled_mps_driver_peak_bytes_max"]
                    ) / 2**30
                else:
                    selected_row = selected[(mode, frame_count)]
                    value = normalized[(mode, frame_count)]
                    base = float(
                        selected[(mode, 8)][
                            "ordered_word_node_interactions_median"
                        ]
                    )
                    lower_value = float(
                        selected_row["ordered_word_node_interactions_min"]
                    ) / base
                    upper_value = float(
                        selected_row["ordered_word_node_interactions_max"]
                    ) / base
                y = plot_top + plot_height * (1.0 - value / ymax)
                upper_y = plot_top + plot_height * (
                    1.0 - min(ymax, upper_value) / ymax
                )
                lower_y = plot_top + plot_height * (
                    1.0 - max(0.0, lower_value) / ymax
                )
                points.append((x, y))
                body.extend(
                    [
                        f'<line x1="{x:.2f}" y1="{upper_y:.2f}" x2="{x:.2f}" y2="{lower_y:.2f}" stroke="#222" stroke-width="1.5"/>',
                        f'<line x1="{x - 5:.2f}" y1="{upper_y:.2f}" x2="{x + 5:.2f}" y2="{upper_y:.2f}" stroke="#222" stroke-width="1.5"/>',
                        f'<line x1="{x - 5:.2f}" y1="{lower_y:.2f}" x2="{x + 5:.2f}" y2="{lower_y:.2f}" stroke="#222" stroke-width="1.5"/>',
                        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="5" fill="{MODE_COLORS[mode]}"/>',
                    ]
                )
            body.append(_polyline(points, MODE_COLORS[mode]))
    for index, mode in enumerate(("fused_union_v2", "per_frame_replay_sequential")):
        x = 340 + index * 330
        body.extend(
            [
                f'<line x1="{x}" y1="555" x2="{x + 34}" y2="555" stroke="{MODE_COLORS[mode]}" stroke-width="4"/>',
                f'<text x="{x + 44}" y="560" font-family="sans-serif" font-size="14">{html.escape(MODE_LABELS[mode])}</text>',
            ]
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        "<title>WorldFoam G6 native memory and temporal sharing</title>"
        "<desc>Verifier-derived sampled MPS high water and normalized ordered-world work across requested frame counts.</desc>"
        + "".join(body)
        + "</svg>\n"
    ).encode("utf-8")


def build_assets(
    artifact_path: Path,
    config_path: Path = DEFAULT_CONFIG,
    contract_path: Path = DEFAULT_CONTRACT,
) -> dict[str, bytes]:
    report = verify_artifact_file(
        artifact_path,
        config_path=config_path,
        contract_path=contract_path,
    )
    if report.get("accepted") is not True:
        raise ValueError(
            "G6 assets require an independently accepted artifact: "
            + "; ".join(str(value) for value in report.get("failures", ()))
        )
    payload = _load_json(artifact_path)
    summary = _summaries(payload)
    return {
        "g6_native_memory_rows.csv": _csv(summary),
        "g6_native_memory_table.tex": _tex(summary),
        "g6_native_memory_scaling.svg": _svg(summary),
        "g6_native_memory_summary.json": _canonical_bytes(
            {
                "schema_version": 1,
                "artifact_sha256": file_sha256(artifact_path),
                "verifier_report": report,
                "timing_scopes": {
                    "core_forward_backward_wall_time_seconds": (
                        "route render/reverse interval; fused/staged native "
                        "coordinator interval or summed sequential per-frame "
                        "render/reverse intervals"
                    ),
                    "fresh_process_end_to_end_wall_time_seconds": (
                        "identical parent-watchdog process scope including "
                        "startup, imports, attestation, preparation, transaction, "
                        "and teardown"
                    ),
                    "route_transaction_wall_time_seconds": (
                        "route-local transaction receipt; retained for audit but "
                        "not claimed as a matched cross-route timing scope"
                    ),
                    "route_compile_wall_time_seconds": (
                        "route-local compile receipt; retained for audit but not "
                        "claimed as a matched cross-route timing scope"
                    ),
                },
                "mode_frame_summary": summary,
            }
        ),
    }


def write_assets(
    artifact_path: Path,
    config_path: Path,
    contract_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    files = build_assets(artifact_path, config_path, contract_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = set(files) | {"manifest.json"}
    existing = {path.name for path in output_dir.iterdir() if path.is_file()}
    unexpected = sorted(existing - expected)
    if unexpected:
        raise ValueError(
            "G6 asset directory contains unexpected files: " + ", ".join(unexpected)
        )
    for name, data in files.items():
        (output_dir / name).write_bytes(data)
    manifest = {
        "schema_version": ASSET_SCHEMA_VERSION,
        "generator": GENERATOR,
        "artifact": str(artifact_path.resolve()),
        "artifact_sha256": file_sha256(artifact_path),
        "config": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "contract": str(contract_path.resolve()),
        "contract_sha256": file_sha256(contract_path),
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
    output_dir: Path,
    artifact_path: Path,
    config_path: Path,
    contract_path: Path,
) -> list[str]:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return ["G6 asset manifest is missing"]
    failures: list[str] = []
    try:
        manifest = _load_json(manifest_path)
        expected = build_assets(artifact_path, config_path, contract_path)
    except Exception as error:
        return [f"G6 assets could not be independently rebuilt: {error}"]
    if manifest.get("schema_version") != ASSET_SCHEMA_VERSION:
        failures.append("G6 asset schema changed")
    if manifest.get("generator") != GENERATOR:
        failures.append("G6 asset generator changed")
    for key, path in (
        ("artifact_sha256", artifact_path),
        ("config_sha256", config_path),
        ("contract_sha256", contract_path),
    ):
        if manifest.get(key) != file_sha256(path):
            failures.append(f"G6 asset {key} binding changed")
    rows = manifest.get("files")
    if not isinstance(rows, list):
        failures.append("G6 asset manifest files are missing")
        rows = []
    by_name = {
        str(row.get("path")): row for row in rows if isinstance(row, Mapping)
    }
    if set(by_name) != set(expected):
        failures.append("G6 asset manifest file set changed")
    for name, data in expected.items():
        path = output_dir / name
        if not path.is_file() or path.read_bytes() != data:
            failures.append(f"G6 asset is missing or nondeterministic: {name}")
            continue
        if by_name.get(name, {}).get("bytes") != len(data):
            failures.append(f"G6 asset byte count changed: {name}")
        if by_name.get(name, {}).get("sha256") != hashlib.sha256(data).hexdigest():
            failures.append(f"G6 asset digest changed: {name}")
    actual = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual != set(expected) | {"manifest.json"}:
        failures.append("G6 asset directory contains missing or unexpected files")
    return sorted(set(failures))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        failures = verify_asset_dir(
            args.out_dir,
            args.artifact,
            args.config,
            args.contract,
        )
        print(
            json.dumps(
                {"accepted": not failures, "failures": failures},
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if not failures else 1
    manifest = write_assets(
        args.artifact,
        args.config,
        args.contract,
        args.out_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
