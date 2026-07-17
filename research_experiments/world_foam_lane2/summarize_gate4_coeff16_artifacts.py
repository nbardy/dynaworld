#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


DYNAWORLD = Path(__file__).resolve().parents[2]
RESULTS_DIR = DYNAWORLD / "research_experiments" / "world_foam_lane2" / "results"
DEFAULT_GLOB = "2026-05-19_gate4_affine_candidate_coeff16*_scale_2_4_8_16_render16_site24_warm3*.json"


def _finite_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _step_ms(row: dict[str, Any], phase: str, stat: str) -> float | None:
    summary = row.get("step_summary")
    if not isinstance(summary, dict):
        return None
    phase_summary = summary.get(phase)
    if not isinstance(phase_summary, dict):
        return None
    value = _finite_float(phase_summary.get(stat))
    return None if value is None else 1000.0 * value


def _env_status(payload: dict[str, Any]) -> str:
    environment = payload.get("benchmark_environment")
    if not isinstance(environment, dict):
        return "missing"
    status = environment.get("status")
    return status if isinstance(status, str) else "missing"


def _load_artifact(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _artifact_summary(path: Path) -> dict[str, Any]:
    payload = _load_artifact(path)
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} has no rows")
    by_frame = {int(row["frame_count"]): row for row in rows if isinstance(row, dict) and "frame_count" in row}
    if not by_frame:
        raise ValueError(f"{path} has no frame_count rows")
    first_frame = min(by_frame)
    last_frame = max(by_frame)
    last_row = by_frame[last_frame]
    return {
        "path": path,
        "tape_mode": str(payload.get("tape_mode", "unknown")),
        "status": str(payload.get("status", "unknown")),
        "env": _env_status(payload),
        "first_frame": first_frame,
        "last_frame": last_frame,
        "total_scale": _finite_float(payload.get("total_step_scale_first_to_last")),
        "backward_scale": _finite_float(payload.get("backward_scale_first_to_last")),
        "storage_scale": _finite_float(payload.get("selected_tape_mps_resident_storage_scale_first_to_last")),
        "last_total_ms": _step_ms(last_row, "total", "median_s"),
        "last_backward_ms": _step_ms(last_row, "backward", "median_s"),
        "last_storage_bytes": int(last_row.get("train_selected_tape_mps_resident_storage_bytes", 0)),
        "last_train_psnr": _finite_float(last_row.get("final_train_psnr")),
        "last_heldout_psnr": _finite_float(last_row.get("final_heldout_psnr")),
    }


def _fmt_float(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _print_markdown(summaries: list[dict[str, Any]]) -> None:
    print("| artifact | mode | env | total scale | backward scale | 16f total ms | 16f backward ms | storage bytes | train/heldout PSNR |")
    print("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")
    for item in summaries:
        path = Path(item["path"])
        psnr = f"{_fmt_float(item['last_train_psnr'])}/{_fmt_float(item['last_heldout_psnr'])}"
        print(
            "| "
            + " | ".join(
                [
                    path.name,
                    str(item["tape_mode"]),
                    str(item["env"]),
                    _fmt_float(item["total_scale"]),
                    _fmt_float(item["backward_scale"]),
                    _fmt_float(item["last_total_ms"]),
                    _fmt_float(item["last_backward_ms"]),
                    str(item["last_storage_bytes"]),
                    psnr,
                ]
            )
            + " |"
        )


def _print_json(summaries: list[dict[str, Any]]) -> None:
    serializable = []
    for item in summaries:
        row = dict(item)
        row["path"] = str(row["path"])
        serializable.append(row)
    print(json.dumps(serializable, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU-only summary of Gate4 coeff16 train/eval artifacts."
    )
    parser.add_argument(
        "artifacts",
        nargs="*",
        type=Path,
        help="Artifacts to summarize. Defaults to the 2026-05-19 coeff16 2/4/8/16f ladder glob.",
    )
    parser.add_argument("--glob", default=DEFAULT_GLOB, help="Results-dir glob used when no artifacts are given.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of markdown.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts = args.artifacts or sorted(RESULTS_DIR.glob(args.glob))
    summaries: list[dict[str, Any]] = []
    failures: list[str] = []
    for path in artifacts:
        try:
            summaries.append(_artifact_summary(path))
        except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
            failures.append(f"{path}: {exc}")
    summaries.sort(
        key=lambda item: (
            item["env"] != "ok",
            float("inf") if item["last_backward_ms"] is None else float(item["last_backward_ms"]),
            str(item["path"]),
        )
    )
    if args.json:
        _print_json(summaries)
    else:
        _print_markdown(summaries)
        if failures:
            print()
            print("Skipped:")
            for failure in failures:
                print(f"- {failure}")
    return 1 if not summaries else 0


if __name__ == "__main__":
    raise SystemExit(main())
