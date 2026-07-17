from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = "worldfoam_owner_run_metal_comparison_report"
DEFAULT_OUT_DIR = ROOT / "outputs" / "benchmarks" / "2026-07-08_worldfoam_owner_run_metal_comparison_report"
DEFAULT_OPTICAL_TRANSFER_SUMMARY = ROOT / "outputs" / "benchmarks" / "2026-07-08_worldfoam_cell_path_optical_transfer_summary.json"
DEFAULT_VISUAL_COMPARE_SUMMARY = (
    ROOT / "outputs" / "visual_comparisons" / "2026-07-07_three_lane_visual_compare_capacity_128_clean_all_lanes.json"
)
WORLDFOAM_LANE_NAME = "worldfoam_dynamic_powerfoam_metal"


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


def _find_worldfoam_lane(payload: dict[str, Any]) -> dict[str, Any] | None:
    lanes = payload.get("lanes")
    if not isinstance(lanes, list):
        return None
    for lane in lanes:
        if isinstance(lane, dict) and lane.get("name") == WORLDFOAM_LANE_NAME:
            return lane
    return None


def _artifacts_ok(lane: dict[str, Any]) -> tuple[bool, int]:
    artifacts = [item for item in lane.get("declared_artifacts", []) if isinstance(item, dict)]
    if not artifacts:
        return False, 0
    ok = True
    for artifact in artifacts:
        if artifact.get("exists") is not True:
            ok = False
        path = artifact.get("path")
        if not isinstance(path, str) or not (_root_path(path)).exists():
            ok = False
    return ok, len(artifacts)


def _optical_contract_row(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    checks = payload.get("checks", {}) if isinstance(payload.get("checks"), dict) else {}
    max_errors = payload.get("max_errors", {}) if isinstance(payload.get("max_errors"), dict) else {}
    ok = payload.get("status") == "ok" and checks and all(value == "ok" for value in checks.values())
    return {
        "row_id": "optical_transfer_contract",
        "kind": "math_contract",
        "status": "ok" if ok else "failed",
        "artifact": _artifact(path),
        "claim": "owner-run optical-transfer replay and VJP contract is green",
        "render_error": max_errors.get("render"),
        "element_error": max_errors.get("element"),
        "grad_error": max_errors.get("grad"),
        "commutator_error": max_errors.get("commutator"),
    }


def _metal_capacity_row(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    lane = _find_worldfoam_lane(payload)
    if lane is None:
        return {
            "row_id": "worldfoam_metal_capacity_lane",
            "kind": "metal_visual_capacity",
            "status": "missing",
            "artifact": _artifact(path),
            "claim": "WorldFoam/PowerFoam Metal capacity lane exists",
        }
    artifacts_ok, artifact_count = _artifacts_ok(lane)
    backend = lane.get("backend") if isinstance(lane.get("backend"), dict) else {}
    ok = lane.get("status") == "ok" and artifacts_ok and backend.get("metal_backend") == "torch_dynamic_powerfoam_metal"
    return {
        "row_id": "worldfoam_metal_capacity_lane",
        "kind": "metal_visual_capacity",
        "status": "ok" if ok else "failed",
        "artifact": _artifact(path),
        "claim": "WorldFoam/PowerFoam Metal capacity lane exists",
        "render_size": backend.get("render_size"),
        "frames": (lane.get("data") or {}).get("max_frames") if isinstance(lane.get("data"), dict) else None,
        "steps": backend.get("steps"),
        "cell_count": backend.get("cell_count"),
        "device": backend.get("device"),
        "metal_backend": backend.get("metal_backend"),
        "elapsed_s": lane.get("elapsed_s"),
        "artifact_count": artifact_count,
    }


def _bridge_row(optical_row: dict[str, Any], metal_row: dict[str, Any]) -> dict[str, Any]:
    ok = optical_row.get("status") == "ok" and metal_row.get("status") == "ok"
    return {
        "row_id": "owner_run_contract_to_metal_bridge",
        "kind": "contract_bridge",
        "status": "ok" if ok else "failed",
        "claim": "WorldFoam paper runner has both optical-transfer contract evidence and Metal lane evidence",
        "bridge_scope": "contract_plus_visual_capacity_smoke",
        "paper_claim_limit": "not a full optical-transfer parity proof inside the Metal shader",
        "optical_contract_status": optical_row.get("status"),
        "metal_capacity_status": metal_row.get("status"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {str(row.get("row_id")): row for row in rows}
    return {
        "row_count": len(rows),
        "green_row_count": sum(1 for row in rows if row.get("status") == "ok"),
        "owner_run_metal_comparison_rows_ok": (
            by_id.get("optical_transfer_contract", {}).get("status") == "ok"
            and by_id.get("worldfoam_metal_capacity_lane", {}).get("status") == "ok"
            and by_id.get("owner_run_contract_to_metal_bridge", {}).get("status") == "ok"
        ),
        "has_optical_contract": by_id.get("optical_transfer_contract", {}).get("status") == "ok",
        "has_metal_capacity_lane": by_id.get("worldfoam_metal_capacity_lane", {}).get("status") == "ok",
        "bridge_scope": by_id.get("owner_run_contract_to_metal_bridge", {}).get("bridge_scope"),
        "paper_ready": False,
    }


def build_report(
    *,
    optical_transfer_summary: str | Path = DEFAULT_OPTICAL_TRANSFER_SUMMARY,
    visual_compare_summary: str | Path = DEFAULT_VISUAL_COMPARE_SUMMARY,
) -> dict[str, Any]:
    optical_path = _root_path(optical_transfer_summary)
    visual_path = _root_path(visual_compare_summary)
    optical_row = _optical_contract_row(_load_json(optical_path), optical_path)
    metal_row = _metal_capacity_row(_load_json(visual_path), visual_path)
    bridge_row = _bridge_row(optical_row, metal_row)
    rows = [optical_row, metal_row, bridge_row]
    return {
        "benchmark": BENCHMARK,
        "status": "ok" if all(row["status"] == "ok" for row in rows) else "failed",
        "inputs": {
            "optical_transfer_summary": _artifact(optical_path),
            "visual_compare_summary": _artifact(visual_path),
        },
        "rows": rows,
        "summary": summarize(rows),
    }


def verify_worldfoam_owner_run_metal_comparison_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("benchmark") != BENCHMARK:
        errors.append(f"benchmark must be {BENCHMARK}")
    rows = report.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        errors.append("rows must be a list of objects")
        return errors
    by_id = {str(row.get("row_id")): row for row in rows}
    for row_id in ("optical_transfer_contract", "worldfoam_metal_capacity_lane", "owner_run_contract_to_metal_bridge"):
        if row_id not in by_id:
            errors.append(f"missing row {row_id}")
        elif by_id[row_id].get("status") != "ok":
            errors.append(f"{row_id} status must be ok")
    optical = by_id.get("optical_transfer_contract", {})
    if float(optical.get("grad_error", 1.0)) > 1.0e-6:
        errors.append("optical_transfer_contract grad_error must be <= 1e-6")
    if float(optical.get("render_error", 1.0)) > 1.0e-12:
        errors.append("optical_transfer_contract render_error must be <= 1e-12")
    metal = by_id.get("worldfoam_metal_capacity_lane", {})
    for key in ("render_size", "frames", "steps", "cell_count", "artifact_count"):
        value = metal.get(key)
        if not isinstance(value, int | float) or float(value) <= 0.0:
            errors.append(f"worldfoam_metal_capacity_lane {key} must be positive")
    if metal.get("metal_backend") != "torch_dynamic_powerfoam_metal":
        errors.append("worldfoam_metal_capacity_lane must use torch_dynamic_powerfoam_metal")
    bridge = by_id.get("owner_run_contract_to_metal_bridge", {})
    if bridge.get("bridge_scope") != "contract_plus_visual_capacity_smoke":
        errors.append("bridge scope must be contract_plus_visual_capacity_smoke")
    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be present")
        return errors
    expected = summarize(rows)
    for key, expected_value in expected.items():
        if summary.get(key) != expected_value:
            errors.append(f"summary {key} mismatch: expected {expected_value!r}, got {summary.get(key)!r}")
    if summary.get("owner_run_metal_comparison_rows_ok") is not True:
        errors.append("summary owner_run_metal_comparison_rows_ok must be true")
    if summary.get("paper_ready") is not False:
        errors.append("summary paper_ready must remain false")
    if report.get("status") != "ok":
        errors.append("report status must be ok")
    return errors


def assert_worldfoam_owner_run_metal_comparison_report(report: dict[str, Any]) -> None:
    errors = verify_worldfoam_owner_run_metal_comparison_report(report)
    if errors:
        raise AssertionError("worldfoam owner-run/Metal comparison report failed:\n- " + "\n- ".join(errors))


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(report: dict[str, Any], path: str | Path) -> Path:
    resolved = _root_path(path)
    columns = ("row_id", "kind", "status", "claim", "bridge_scope", "paper_claim_limit")
    lines = [
        "# WorldFoam Owner-Run / Metal Comparison Report",
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
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in report["rows"]:
        lines.append("| " + " | ".join(_fmt(row.get(column)) for column in columns) + " |")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return resolved


def write_report(report: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    assert_worldfoam_owner_run_metal_comparison_report(report)
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
        assert_worldfoam_owner_run_metal_comparison_report(report)
        print(f"verified {args.verify_report}")
        return

    report = build_report()
    json_path, markdown_path = write_report(report, args.out_dir)
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
