from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config_utils import load_config_file


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = "dynaworld_paper_quality_benchmark_table"
BENCHMARK_SCOPE = "capacity_128_local_video_smoke"
DEFAULT_OUT_DIR = ROOT / "outputs" / "benchmarks" / "2026-07-08_paper_quality_benchmark_table"
DEFAULT_VISUAL_COMPARE = (
    ROOT / "outputs" / "visual_comparisons" / "2026-07-07_three_lane_visual_compare_capacity_128_clean_all_lanes.json"
)
REQUIRED_REPRESENTATIONS = ("world_tubes_star_uvt", "worldfoam_powerfoam", "dynamic_3dgs_fast_mac")


@dataclass(frozen=True)
class LaneContract:
    lane_name: str
    representation: str
    paper: str
    primitive_name: str


LANE_CONTRACTS = {
    "worldtubes_star_uvt_metal": LaneContract(
        lane_name="worldtubes_star_uvt_metal",
        representation="world_tubes_star_uvt",
        paper="world_tubes",
        primitive_name="tubes",
    ),
    "worldfoam_dynamic_powerfoam_metal": LaneContract(
        lane_name="worldfoam_dynamic_powerfoam_metal",
        representation="worldfoam_powerfoam",
        paper="worldfoam",
        primitive_name="cells",
    ),
    "dynamic_gsplat_fast_mac_metal": LaneContract(
        lane_name="dynamic_gsplat_fast_mac_metal",
        representation="dynamic_3dgs_fast_mac",
        paper="baseline",
        primitive_name="gaussians",
    ),
}


def _root_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else ROOT / resolved


def _artifact(path: str | Path) -> str:
    resolved = _root_path(path)
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _load_json(path: str | Path) -> dict[str, Any]:
    resolved = _root_path(path)
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {resolved}")
    return payload


def _side_by_side_path(lane: dict[str, Any]) -> Path:
    for artifact in lane.get("declared_artifacts", []):
        if isinstance(artifact, dict) and artifact.get("label") == "side_by_side_video":
            path = artifact.get("path")
            if isinstance(path, str):
                return _root_path(path)
    raise ValueError(f"Lane {lane.get('name')} has no side_by_side_video artifact")


def _config_path(lane: dict[str, Any]) -> Path:
    path = lane.get("config")
    if not isinstance(path, str):
        raise ValueError(f"Lane {lane.get('name')} has no config path")
    return _root_path(path)


def _primitive_count(contract: LaneContract, config: dict[str, Any]) -> int:
    if contract.representation == "world_tubes_star_uvt":
        return int(config["uvt"]["tube_count"])
    if contract.representation == "worldfoam_powerfoam":
        return int(config["model"]["cells"])
    if contract.representation == "dynamic_3dgs_fast_mac":
        model = config["model"]
        return int(model["tokens"]) * int(model["gaussians_per_token"])
    raise ValueError(f"Unknown representation {contract.representation}")


def _safe_float(value: Any) -> float | None:
    if isinstance(value, int | float):
        return float(value)
    return None


def _aggregate(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _media_metrics_from_side_by_side(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Could not open side-by-side video {path}")
    l1_values: list[float] = []
    mse_values: list[float] = []
    psnr_values: list[float] = []
    width = None
    height = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Expected RGB/BGR video frame in {path}, got shape {frame.shape}")
        height, full_width = int(frame.shape[0]), int(frame.shape[1])
        if full_width % 2 != 0:
            raise ValueError(f"Side-by-side video width must be even in {path}, got {full_width}")
        width = full_width // 2
        target = frame[:, :width, :].astype(np.float64) / 255.0
        render = frame[:, width:, :].astype(np.float64) / 255.0
        diff = render - target
        mse = float(np.mean(diff * diff))
        l1 = float(np.mean(np.abs(diff)))
        psnr = 120.0 if mse == 0.0 else float(-10.0 * math.log10(mse))
        mse_values.append(mse)
        l1_values.append(l1)
        psnr_values.append(psnr)
    cap.release()
    if not psnr_values or width is None or height is None:
        raise ValueError(f"No frames decoded from {path}")
    return {
        "frame_count": len(psnr_values),
        "render_size": int(width),
        "height": int(height),
        "media_psnr_mean": _aggregate(psnr_values)["mean"],
        "media_psnr_min": _aggregate(psnr_values)["min"],
        "media_l1_mean": _aggregate(l1_values)["mean"],
        "media_l1_max": _aggregate(l1_values)["max"],
        "media_mse_mean": _aggregate(mse_values)["mean"],
    }


def _worldtubes_reported_metrics(lane: dict[str, Any]) -> dict[str, Any]:
    for artifact in lane.get("declared_artifacts", []):
        if isinstance(artifact, dict) and artifact.get("label") == "out_json" and isinstance(artifact.get("path"), str):
            payload = _load_json(artifact["path"])
            uvt = payload.get("uvt", {}) if isinstance(payload.get("uvt"), dict) else {}
            return {
                "native_metric_source": _artifact(artifact["path"]),
                "native_psnr": _safe_float(uvt.get("final_psnr")),
                "native_l1": _safe_float(uvt.get("final_l1")),
                "native_ssim": _safe_float(uvt.get("final_ssim_mean")),
            }
    return {"native_metric_source": None, "native_psnr": None, "native_l1": None, "native_ssim": None}


def _worldfoam_reported_metrics(side_by_side_path: Path) -> dict[str, Any]:
    metrics_path = side_by_side_path.with_name(side_by_side_path.name.replace("side_by_side", "per_frame_metrics")).with_suffix(".json")
    if not metrics_path.exists():
        return {"native_metric_source": None, "native_psnr": None, "native_l1": None, "native_ssim": None}
    payload = _load_json(metrics_path)
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    return {
        "native_metric_source": _artifact(metrics_path),
        "native_psnr": _safe_float(summary.get("frame_psnr_mean")),
        "native_l1": _safe_float(summary.get("frame_l1_mean")),
        "native_ssim": None,
    }


def _native_metrics(contract: LaneContract, lane: dict[str, Any], side_by_side_path: Path) -> dict[str, Any]:
    if contract.representation == "world_tubes_star_uvt":
        return _worldtubes_reported_metrics(lane)
    if contract.representation == "worldfoam_powerfoam":
        return _worldfoam_reported_metrics(side_by_side_path)
    return {"native_metric_source": None, "native_psnr": None, "native_l1": None, "native_ssim": None}


def _declared_artifacts_ok(lane: dict[str, Any]) -> tuple[bool, int]:
    artifacts = [item for item in lane.get("declared_artifacts", []) if isinstance(item, dict)]
    if not artifacts:
        return False, 0
    ok = True
    for artifact in artifacts:
        path = artifact.get("path")
        if artifact.get("exists") is not True or not isinstance(path, str) or not _root_path(path).exists():
            ok = False
    return ok, len(artifacts)


def _row_from_lane(lane: dict[str, Any]) -> dict[str, Any] | None:
    name = lane.get("name")
    if not isinstance(name, str) or name not in LANE_CONTRACTS:
        return None
    contract = LANE_CONTRACTS[name]
    side_by_side_path = _side_by_side_path(lane)
    config_path = _config_path(lane)
    config = load_config_file(config_path)
    backend = lane.get("backend") if isinstance(lane.get("backend"), dict) else {}
    data = lane.get("data") if isinstance(lane.get("data"), dict) else {}
    artifacts_ok, artifact_count = _declared_artifacts_ok(lane)
    media_metrics = _media_metrics_from_side_by_side(side_by_side_path)
    native_metrics = _native_metrics(contract, lane, side_by_side_path)
    max_frames = int(data.get("max_frames", media_metrics["frame_count"]))
    render_size = int(backend.get("render_size", media_metrics["render_size"]))
    ok = (
        lane.get("status") == "ok"
        and artifacts_ok
        and media_metrics["frame_count"] == max_frames
        and media_metrics["render_size"] == render_size
        and media_metrics["media_psnr_mean"] > 0.0
        and 0.0 <= media_metrics["media_l1_mean"] <= 1.0
    )
    return {
        "representation": contract.representation,
        "paper": contract.paper,
        "status": "ok" if ok else "failed",
        "benchmark_scope": BENCHMARK_SCOPE,
        "dataset_id": Path(str(data.get("video_path", ""))).stem,
        "video_path": data.get("video_path"),
        "config": _artifact(config_path),
        "side_by_side_video": _artifact(side_by_side_path),
        "artifact_count": artifact_count,
        "primitive_name": contract.primitive_name,
        "primitive_count": _primitive_count(contract, config),
        "steps": backend.get("steps"),
        "elapsed_s": lane.get("elapsed_s"),
        **media_metrics,
        **native_metrics,
    }


def build_report(visual_compare_summary: str | Path = DEFAULT_VISUAL_COMPARE) -> dict[str, Any]:
    visual_path = _root_path(visual_compare_summary)
    payload = _load_json(visual_path)
    rows = []
    for lane in payload.get("lanes", []):
        if isinstance(lane, dict):
            row = _row_from_lane(lane)
            if row is not None:
                rows.append(row)
    rows.sort(key=lambda row: REQUIRED_REPRESENTATIONS.index(str(row["representation"])))
    missing = [
        representation
        for representation in REQUIRED_REPRESENTATIONS
        if not any(row.get("representation") == representation and row.get("status") == "ok" for row in rows)
    ]
    report = {
        "benchmark": BENCHMARK,
        "status": "ok" if not missing and len(rows) == len(REQUIRED_REPRESENTATIONS) else "incomplete",
        "inputs": {"visual_compare_summary": _artifact(visual_path)},
        "rows": rows,
        "missing_rows": [
            {
                "missing_id": f"{representation}_paper_quality_row",
                "representation": representation,
                "needed_for": "matched paper-quality benchmark table",
            }
            for representation in missing
        ],
    }
    report["summary"] = summarize(rows, report["missing_rows"])
    return report


def summarize(rows: list[dict[str, Any]], missing_rows: list[dict[str, str]]) -> dict[str, Any]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    psnr_by_rep = {
        str(row.get("representation")): row.get("media_psnr_mean")
        for row in ok_rows
        if isinstance(row.get("media_psnr_mean"), int | float)
    }
    elapsed_by_rep = {
        str(row.get("representation")): row.get("elapsed_s")
        for row in ok_rows
        if isinstance(row.get("elapsed_s"), int | float)
    }
    return {
        "benchmark_scope": BENCHMARK_SCOPE,
        "row_count": len(rows),
        "green_row_count": len(ok_rows),
        "missing_row_count": len(missing_rows),
        "missing_ids": [row["missing_id"] for row in missing_rows],
        "representation_count": len({str(row.get("representation")) for row in rows}),
        "paper_ready": len(missing_rows) == 0 and len(ok_rows) == len(REQUIRED_REPRESENTATIONS),
        "required_representations_present": all(rep in psnr_by_rep for rep in REQUIRED_REPRESENTATIONS),
        "best_media_psnr_representation": max(psnr_by_rep, key=psnr_by_rep.get) if psnr_by_rep else None,
        "fastest_elapsed_representation": min(elapsed_by_rep, key=elapsed_by_rep.get) if elapsed_by_rep else None,
    }


def verify_paper_quality_benchmark_table_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("benchmark") != BENCHMARK:
        errors.append(f"benchmark must be {BENCHMARK}")
    rows = report.get("rows")
    missing_rows = report.get("missing_rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        errors.append("rows must be a list of objects")
        return errors
    if not isinstance(missing_rows, list) or not all(isinstance(row, dict) for row in missing_rows):
        errors.append("missing_rows must be a list of objects")
        return errors
    by_representation = {str(row.get("representation")): row for row in rows}
    for representation in REQUIRED_REPRESENTATIONS:
        row = by_representation.get(representation)
        if row is None:
            errors.append(f"missing representation row {representation}")
            continue
        if row.get("status") != "ok":
            errors.append(f"{representation} status must be ok")
        if row.get("benchmark_scope") != BENCHMARK_SCOPE:
            errors.append(f"{representation} benchmark_scope must be {BENCHMARK_SCOPE}")
        side_by_side = row.get("side_by_side_video")
        if not isinstance(side_by_side, str) or not _root_path(side_by_side).exists():
            errors.append(f"{representation} side_by_side_video must exist")
        for key in ("primitive_count", "steps", "frame_count", "render_size", "artifact_count"):
            value = row.get(key)
            if not isinstance(value, int | float) or float(value) <= 0.0:
                errors.append(f"{representation} {key} must be positive")
        if row.get("frame_count") != 16:
            errors.append(f"{representation} frame_count must be 16")
        if row.get("render_size") != 128:
            errors.append(f"{representation} render_size must be 128")
        psnr = row.get("media_psnr_mean")
        l1 = row.get("media_l1_mean")
        if not isinstance(psnr, int | float) or float(psnr) <= 0.0:
            errors.append(f"{representation} media_psnr_mean must be positive")
        if not isinstance(l1, int | float) or not (0.0 <= float(l1) <= 1.0):
            errors.append(f"{representation} media_l1_mean must be in [0, 1]")
    if missing_rows:
        errors.append("missing_rows must be empty for the final paper-quality table")
    if report.get("status") != "ok":
        errors.append("report status must be ok")
    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be present")
        return errors
    expected = summarize(rows, missing_rows)
    for key, expected_value in expected.items():
        if summary.get(key) != expected_value:
            errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {summary.get(key)!r}")
    if summary.get("paper_ready") is not True:
        errors.append("summary paper_ready must be true")
    if summary.get("required_representations_present") is not True:
        errors.append("summary required_representations_present must be true")
    return errors


def assert_paper_quality_benchmark_table_report(report: dict[str, Any]) -> None:
    errors = verify_paper_quality_benchmark_table_report(report)
    if errors:
        raise AssertionError("paper quality benchmark table report failed:\n- " + "\n- ".join(errors))


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(report: dict[str, Any], path: str | Path) -> Path:
    resolved = _root_path(path)
    lines = [
        "# DynaWorld Paper Quality Benchmark Table",
        "",
        f"Status: `{report['status']}`.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(report["summary"], indent=2, sort_keys=True),
        "```",
        "",
        "## Rows",
        "",
        "| representation | paper | primitives | steps | elapsed_s | media_psnr_mean | media_l1_mean | native_psnr | native_l1 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["rows"]:
        primitives = f"{row.get('primitive_count')} {row.get('primitive_name')}"
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("representation")),
                    _fmt(row.get("paper")),
                    _fmt(primitives),
                    _fmt(row.get("steps")),
                    _fmt(row.get("elapsed_s")),
                    _fmt(row.get("media_psnr_mean")),
                    _fmt(row.get("media_l1_mean")),
                    _fmt(row.get("native_psnr")),
                    _fmt(row.get("native_l1")),
                ]
            )
            + " |"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return resolved


def write_report(report: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    assert_paper_quality_benchmark_table_report(report)
    resolved = _root_path(out_dir)
    resolved.mkdir(parents=True, exist_ok=True)
    json_path = resolved / "summary.json"
    markdown_path = resolved / "summary.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, markdown_path)
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--verify-report", type=Path, help="validate an existing summary JSON without rerunning")
    args = parser.parse_args()

    if args.verify_report is not None:
        report = json.loads(_root_path(args.verify_report).read_text(encoding="utf-8"))
        assert_paper_quality_benchmark_table_report(report)
        print(f"verified {args.verify_report}")
        return

    report = build_report()
    json_path, markdown_path = write_report(report, args.out_dir)
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
