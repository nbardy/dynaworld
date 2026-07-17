from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = "dynaworld_paper_runner_table_report"
DEFAULT_OUT_DIR = ROOT / "outputs" / "benchmarks" / "2026-07-11_paper_runner_table_report"


@dataclass(frozen=True)
class ReportInputs:
    world_tubes_decisive: Path = (
        ROOT / "outputs" / "benchmarks" / "2026-07-08_star_uvt_projective_decisive_demo_fixture" / "summary.json"
    )
    world_tubes_visibility: Path = (
        ROOT / "outputs" / "benchmarks" / "2026-07-08_star_uvt_projective_visibility_stress_suite" / "summary.json"
    )
    worldfoam_optical_transfer: Path = (
        ROOT / "outputs" / "benchmarks" / "2026-07-08_worldfoam_cell_path_optical_transfer_summary.json"
    )
    worldfoam_owner_run_metal_comparison: Path = (
        ROOT / "outputs" / "benchmarks" / "2026-07-08_worldfoam_owner_run_metal_comparison_report" / "summary.json"
    )
    paper_quality_benchmark_table: Path = (
        ROOT / "outputs" / "benchmarks" / "2026-07-08_paper_quality_benchmark_table" / "summary.json"
    )
    coffee_martini_matched_sweep: Path = (
        ROOT
        / "outputs"
        / "benchmarks"
        / "2026-07-11_coffee_martini_matched_sweep"
        / "report"
        / "summary.json"
    )
    visual_compare_capacity: Path = (
        ROOT / "outputs" / "visual_comparisons" / "2026-07-07_three_lane_visual_compare_capacity_128_clean_all_lanes.json"
    )


def _root_path(path: str | Path) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else ROOT / resolved


def _load_json(path: str | Path) -> dict[str, Any] | None:
    resolved = _root_path(path)
    if not resolved.exists():
        return None
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {resolved}")
    return payload


def _artifact(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _status(ok: bool, missing: bool = False) -> str:
    if missing:
        return "missing"
    return "ok" if ok else "failed"


def _world_tubes_decisive_row(payload: dict[str, Any] | None, path: Path) -> dict[str, Any]:
    if payload is None:
        return {
            "evidence_id": "world_tubes_decisive_demo",
            "paper": "world_tubes",
            "kind": "fixture_replay_equivalence",
            "status": "missing",
            "artifact": _artifact(path),
            "claim": "compiled projective interval atlas matches per-frame replay",
        }
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    ok = (
        payload.get("benchmark") == "star_uvt_projective_decisive_demo"
        and summary.get("all_rows_quality_pass") is True
        and summary.get("all_rows_fallback_free") is True
        and summary.get("compiled_route_present") is True
        and summary.get("replay_route_present") is True
    )
    return {
        "evidence_id": "world_tubes_decisive_demo",
        "paper": "world_tubes",
        "kind": "fixture_replay_equivalence",
        "status": _status(ok),
        "artifact": _artifact(path),
        "claim": "compiled projective interval atlas matches per-frame replay",
        "max_image_abs_error": summary.get("max_image_abs_error_vs_reference"),
        "psnr": summary.get("min_psnr_vs_reference"),
        "interval_entry_ratio": summary.get("compiled_to_replay_interval_entry_ratio"),
        "memory_ratio": summary.get("compiled_to_replay_memory_ratio"),
        "has_real_video_media_rows": summary.get("has_real_video_media_rows") is True,
        "real_video_media_rows_ok": summary.get("real_video_media_rows_ok") is True,
        "real_video_min_psnr": summary.get("real_video_min_psnr"),
        "real_video_max_l1": summary.get("real_video_max_l1"),
        "real_video_min_artifact_count": summary.get("real_video_min_artifact_count"),
    }


def _world_tubes_visibility_row(payload: dict[str, Any] | None, path: Path) -> dict[str, Any]:
    if payload is None:
        return {
            "evidence_id": "world_tubes_visibility_stress",
            "paper": "world_tubes",
            "kind": "visibility_stress_fixture",
            "status": "missing",
            "artifact": _artifact(path),
            "claim": "visibility strata expose raw crossing collapse and repaired crossing",
        }
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    ok = (
        payload.get("benchmark") == "star_uvt_projective_visibility_stress_suite"
        and summary.get("has_collapse_boundary") is True
        and summary.get("has_repaired_crossing_case") is True
        and summary.get("required_case_ids_present") is True
    )
    return {
        "evidence_id": "world_tubes_visibility_stress",
        "paper": "world_tubes",
        "kind": "visibility_stress_fixture",
        "status": _status(ok),
        "artifact": _artifact(path),
        "claim": "visibility strata expose raw crossing collapse and repaired crossing",
        "collapsed_case_count": summary.get("collapsed_case_count"),
        "collapsed_case_ids": summary.get("collapsed_case_ids"),
        "max_quality_error": summary.get("max_quality_error"),
        "max_fallback_sample_fraction": summary.get("max_fallback_sample_fraction"),
    }


def _worldfoam_optical_transfer_row(payload: dict[str, Any] | None, path: Path) -> dict[str, Any]:
    if payload is None:
        return {
            "evidence_id": "worldfoam_optical_transfer_fixture",
            "paper": "worldfoam",
            "kind": "math_fixture",
            "status": "missing",
            "artifact": _artifact(path),
            "claim": "optical-transfer monoid replay and VJP finite differences are correct",
        }
    checks = payload.get("checks", {}) if isinstance(payload.get("checks"), dict) else {}
    max_errors = payload.get("max_errors", {}) if isinstance(payload.get("max_errors"), dict) else {}
    ok = payload.get("status") == "ok" and checks and all(value == "ok" for value in checks.values())
    return {
        "evidence_id": "worldfoam_optical_transfer_fixture",
        "paper": "worldfoam",
        "kind": "math_fixture",
        "status": _status(ok),
        "artifact": _artifact(path),
        "claim": "optical-transfer monoid replay and VJP finite differences are correct",
        "render_error": max_errors.get("render"),
        "element_error": max_errors.get("element"),
        "grad_error": max_errors.get("grad"),
        "commutator_error": max_errors.get("commutator"),
    }


def _worldfoam_owner_run_metal_comparison_row(payload: dict[str, Any] | None, path: Path) -> dict[str, Any]:
    if payload is None:
        return {
            "evidence_id": "worldfoam_owner_run_metal_comparison",
            "paper": "worldfoam",
            "kind": "contract_plus_metal_visual_capacity",
            "status": "missing",
            "artifact": _artifact(path),
            "claim": "WorldFoam optical-transfer contract and Metal visual lane are both green",
        }
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    ok = (
        payload.get("benchmark") == "worldfoam_owner_run_metal_comparison_report"
        and payload.get("status") == "ok"
        and summary.get("owner_run_metal_comparison_rows_ok") is True
    )
    return {
        "evidence_id": "worldfoam_owner_run_metal_comparison",
        "paper": "worldfoam",
        "kind": "contract_plus_metal_visual_capacity",
        "status": _status(ok),
        "artifact": _artifact(path),
        "claim": "WorldFoam optical-transfer contract and Metal visual lane are both green",
        "bridge_scope": summary.get("bridge_scope"),
        "paper_ready": summary.get("paper_ready"),
    }


def _paper_quality_benchmark_table_row(payload: dict[str, Any] | None, path: Path) -> dict[str, Any]:
    if payload is None:
        return {
            "evidence_id": "paper_quality_benchmark_table",
            "paper": "comparison",
            "kind": "matched_capacity_quality_table",
            "status": "missing",
            "artifact": _artifact(path),
            "claim": "matched quality/runtime table exists for all paper representations",
        }
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    ok = (
        payload.get("benchmark") == "dynaworld_paper_quality_benchmark_table"
        and payload.get("status") == "ok"
        and summary.get("paper_ready") is True
        and summary.get("required_representations_present") is True
        and summary.get("missing_row_count") == 0
    )
    return {
        "evidence_id": "paper_quality_benchmark_table",
        "paper": "comparison",
        "kind": "matched_capacity_quality_table",
        "status": _status(ok),
        "artifact": _artifact(path),
        "claim": "matched quality/runtime table exists for all paper representations",
        "benchmark_scope": summary.get("benchmark_scope"),
        "row_count": summary.get("row_count"),
        "best_media_psnr_representation": summary.get("best_media_psnr_representation"),
        "fastest_elapsed_representation": summary.get("fastest_elapsed_representation"),
    }


def _coffee_martini_matched_sweep_row(payload: dict[str, Any] | None, path: Path) -> dict[str, Any]:
    if payload is None:
        return {
            "evidence_id": "coffee_martini_matched_3seed",
            "paper": "comparison",
            "kind": "heldout_camera_quality_table",
            "status": "missing",
            "artifact": _artifact(path),
            "claim": "matched three-seed coffee_martini heldout-camera table exists",
        }
    gates = payload.get("gates", {}) if isinstance(payload.get("gates"), dict) else {}
    ok = (
        payload.get("benchmark") == "coffee_martini_train2_holdout1_matched_3seed"
        and payload.get("status") == "ok"
        and payload.get("paper_table_ready") is True
        and gates
        and all(value is True for value in gates.values())
    )
    return {
        "evidence_id": "coffee_martini_matched_3seed",
        "paper": "comparison",
        "kind": "heldout_camera_quality_table",
        "status": _status(ok),
        "artifact": _artifact(path),
        "claim": "matched three-seed coffee_martini heldout-camera table exists",
        "scope": payload.get("scope"),
        "seeds": payload.get("seeds"),
        "train_cameras": payload.get("train_cameras"),
        "heldout_cameras": payload.get("heldout_cameras"),
        "best_mean_heldout_psnr": payload.get("best_mean_heldout_psnr"),
    }


def _visual_compare_rows(payload: dict[str, Any] | None, path: Path) -> list[dict[str, Any]]:
    if payload is None:
        return [
            {
                "evidence_id": "visual_capacity_missing",
                "paper": "comparison",
                "kind": "visual_capacity",
                "status": "missing",
                "artifact": _artifact(path),
                "claim": "three-lane capacity-tier visual comparison exists",
            }
        ]
    rows = []
    for lane in payload.get("lanes", []):
        if not isinstance(lane, dict):
            continue
        artifacts = [item for item in lane.get("declared_artifacts", []) if isinstance(item, dict)]
        artifacts_ok = all(item.get("exists") is True for item in artifacts) and bool(artifacts)
        ok = lane.get("status") == "ok" and artifacts_ok
        rows.append(
            {
                "evidence_id": f"visual_capacity_{lane.get('name')}",
                "paper": "comparison",
                "kind": "visual_capacity",
                "status": _status(ok),
                "artifact": _artifact(path),
                "claim": "capacity-tier local visual artifact exists",
                "representation": lane.get("representation"),
                "render_size": (lane.get("backend") or {}).get("render_size") if isinstance(lane.get("backend"), dict) else None,
                "steps": (lane.get("backend") or {}).get("steps") if isinstance(lane.get("backend"), dict) else None,
                "elapsed_s": lane.get("elapsed_s"),
                "artifact_count": len(artifacts),
            }
        )
    return rows


def _visual_status_by_representation(evidence_rows: list[dict[str, Any]]) -> dict[str, str]:
    status_by_representation: dict[str, str] = {}
    for row in evidence_rows:
        representation = row.get("representation")
        if isinstance(representation, str):
            status_by_representation[representation] = str(row.get("status"))
    return status_by_representation


def _representation_rows(evidence_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(row.get("evidence_id")): row for row in evidence_rows}
    visual = _visual_status_by_representation(evidence_rows)
    world_tubes_real_video_ok = by_id.get("world_tubes_decisive_demo", {}).get("real_video_media_rows_ok") is True
    worldfoam_owner_run_ok = by_id.get("worldfoam_owner_run_metal_comparison", {}).get("status") == "ok"
    paper_quality_ok = by_id.get("paper_quality_benchmark_table", {}).get("status") == "ok"
    coffee_martini_ok = by_id.get("coffee_martini_matched_3seed", {}).get("status") == "ok"
    world_tubes_fixture_ok = (
        by_id.get("world_tubes_decisive_demo", {}).get("status") == "ok"
        and by_id.get("world_tubes_visibility_stress", {}).get("status") == "ok"
    )
    worldfoam_fixture_ok = by_id.get("worldfoam_optical_transfer_fixture", {}).get("status") == "ok"
    world_tubes_visual_ok = visual.get("worldtubes_star_uvt_metal") == "ok"
    worldfoam_visual_ok = visual.get("worldfoam_dynamic_powerfoam_metal") == "ok"
    dynamic_visual_ok = visual.get("dynamic_gsplat_fast_mac_metal") == "ok"
    return [
        {
            "representation": "world_tubes_star_uvt",
            "paper": "world_tubes",
            "fixture_gates": "ok" if world_tubes_fixture_ok else "incomplete",
            "visual_capacity": visual.get("worldtubes_star_uvt_metal", "missing"),
            "real_video_media": "ok" if world_tubes_real_video_ok else "missing",
            "paper_quality_benchmark": "ok" if paper_quality_ok else "missing",
            "coffee_martini_matched_3seed": "ok" if coffee_martini_ok else "missing",
            "paper_ready": world_tubes_fixture_ok and world_tubes_visual_ok and world_tubes_real_video_ok and paper_quality_ok and coffee_martini_ok,
        },
        {
            "representation": "worldfoam_powerfoam",
            "paper": "worldfoam",
            "fixture_gates": "ok" if worldfoam_fixture_ok else "incomplete",
            "visual_capacity": visual.get("worldfoam_dynamic_powerfoam_metal", "missing"),
            "owner_run_metal_comparison": "ok" if worldfoam_owner_run_ok else "missing",
            "paper_quality_benchmark": "ok" if paper_quality_ok else "missing",
            "coffee_martini_matched_3seed": "ok" if coffee_martini_ok else "missing",
            "paper_ready": worldfoam_fixture_ok and worldfoam_visual_ok and worldfoam_owner_run_ok and paper_quality_ok and coffee_martini_ok,
        },
        {
            "representation": "dynamic_3dgs_fast_mac",
            "paper": "baseline",
            "fixture_gates": "not_applicable",
            "visual_capacity": visual.get("dynamic_gsplat_fast_mac_metal", "missing"),
            "paper_quality_benchmark": "ok" if paper_quality_ok else "missing",
            "coffee_martini_matched_3seed": "ok" if coffee_martini_ok else "missing",
            "paper_ready": dynamic_visual_ok and paper_quality_ok and coffee_martini_ok,
        },
    ]


def _missing_rows(representation_rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    missing = []
    world_tubes = next(
        (row for row in representation_rows if row.get("representation") == "world_tubes_star_uvt"),
        None,
    )
    if world_tubes is None or world_tubes.get("real_video_media") != "ok":
        missing.append(
            {
                "missing_id": "world_tubes_real_video_media_rows",
                "representation": "world_tubes_star_uvt",
                "needed_for": "paper table with real-video quality/runtime artifacts",
            }
        )
    worldfoam = next(
        (row for row in representation_rows if row.get("representation") == "worldfoam_powerfoam"),
        None,
    )
    if worldfoam is None or worldfoam.get("owner_run_metal_comparison") != "ok":
        missing.append(
            {
                "missing_id": "worldfoam_owner_run_metal_comparison_rows",
                "representation": "worldfoam_powerfoam",
                "needed_for": "WorldFoam paper runner parity with optical-transfer contract",
            }
        )
    if any(row.get("paper_quality_benchmark") != "ok" for row in representation_rows):
        missing.append(
            {
                "missing_id": "paper_quality_benchmark_table",
                "representation": "all",
                "needed_for": "final paper ablation table across World Tubes, WorldFoam, and dynamic 3DGS",
            }
        )
    if any(row.get("coffee_martini_matched_3seed") != "ok" for row in representation_rows):
        missing.append(
            {
                "missing_id": "coffee_martini_matched_3seed",
                "representation": "all",
                "needed_for": "real multicamera heldout-camera paper table across all representations",
            }
        )
    for row in representation_rows:
        if row.get("visual_capacity") != "ok":
            missing.append(
                {
                    "missing_id": f"{row.get('representation')}_visual_capacity",
                    "representation": str(row.get("representation")),
                    "needed_for": "visual comparison lane must be green",
                }
            )
    return missing


def summarize(
    evidence_rows: list[dict[str, Any]],
    representation_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, str]],
) -> dict[str, Any]:
    return {
        "evidence_row_count": len(evidence_rows),
        "green_evidence_row_count": sum(1 for row in evidence_rows if row.get("status") == "ok"),
        "representation_count": len(representation_rows),
        "missing_row_count": len(missing_rows),
        "missing_ids": [row["missing_id"] for row in missing_rows],
        "paper_ready": len(missing_rows) == 0,
        "has_world_tubes_fixture_gates": any(
            row.get("representation") == "world_tubes_star_uvt" and row.get("fixture_gates") == "ok"
            for row in representation_rows
        ),
        "has_worldfoam_fixture_gates": any(
            row.get("representation") == "worldfoam_powerfoam" and row.get("fixture_gates") == "ok"
            for row in representation_rows
        ),
        "has_dynamic_3dgs_visual_baseline": any(
            row.get("representation") == "dynamic_3dgs_fast_mac" and row.get("visual_capacity") == "ok"
            for row in representation_rows
        ),
        "has_coffee_martini_matched_3seed": any(
            row.get("representation") == "world_tubes_star_uvt" and row.get("coffee_martini_matched_3seed") == "ok"
            for row in representation_rows
        ),
    }


def build_report(inputs: ReportInputs = ReportInputs()) -> dict[str, Any]:
    evidence_rows = [
        _world_tubes_decisive_row(_load_json(inputs.world_tubes_decisive), inputs.world_tubes_decisive),
        _world_tubes_visibility_row(_load_json(inputs.world_tubes_visibility), inputs.world_tubes_visibility),
        _worldfoam_optical_transfer_row(_load_json(inputs.worldfoam_optical_transfer), inputs.worldfoam_optical_transfer),
        _worldfoam_owner_run_metal_comparison_row(
            _load_json(inputs.worldfoam_owner_run_metal_comparison),
            inputs.worldfoam_owner_run_metal_comparison,
        ),
        _paper_quality_benchmark_table_row(
            _load_json(inputs.paper_quality_benchmark_table),
            inputs.paper_quality_benchmark_table,
        ),
        _coffee_martini_matched_sweep_row(
            _load_json(inputs.coffee_martini_matched_sweep),
            inputs.coffee_martini_matched_sweep,
        ),
    ]
    evidence_rows.extend(_visual_compare_rows(_load_json(inputs.visual_compare_capacity), inputs.visual_compare_capacity))
    representation_rows = _representation_rows(evidence_rows)
    missing_rows = _missing_rows(representation_rows)
    return {
        "benchmark": BENCHMARK,
        "status": "incomplete" if missing_rows else "ok",
        "inputs": {
            "world_tubes_decisive": _artifact(inputs.world_tubes_decisive),
            "world_tubes_visibility": _artifact(inputs.world_tubes_visibility),
            "worldfoam_optical_transfer": _artifact(inputs.worldfoam_optical_transfer),
            "worldfoam_owner_run_metal_comparison": _artifact(inputs.worldfoam_owner_run_metal_comparison),
            "paper_quality_benchmark_table": _artifact(inputs.paper_quality_benchmark_table),
            "coffee_martini_matched_sweep": _artifact(inputs.coffee_martini_matched_sweep),
            "visual_compare_capacity": _artifact(inputs.visual_compare_capacity),
        },
        "evidence_rows": evidence_rows,
        "representation_rows": representation_rows,
        "missing_rows": missing_rows,
        "summary": summarize(evidence_rows, representation_rows, missing_rows),
    }


def _assert_summary_value(actual: Any, expected: Any, key: str, errors: list[str]) -> None:
    if actual != expected:
        errors.append(f"summary {key} mismatch: expected {expected!r}, got {actual!r}")


def verify_paper_runner_table_report(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("benchmark") != BENCHMARK:
        errors.append(f"benchmark must be {BENCHMARK}")
    evidence_rows = report.get("evidence_rows")
    representation_rows = report.get("representation_rows")
    missing_rows = report.get("missing_rows")
    if not isinstance(evidence_rows, list) or not all(isinstance(row, dict) for row in evidence_rows):
        errors.append("evidence_rows must be a list of objects")
        return errors
    if not isinstance(representation_rows, list) or not all(isinstance(row, dict) for row in representation_rows):
        errors.append("representation_rows must be a list of objects")
        return errors
    if not isinstance(missing_rows, list) or not all(isinstance(row, dict) for row in missing_rows):
        errors.append("missing_rows must be a list of objects")
        return errors

    by_evidence = {str(row.get("evidence_id")): row for row in evidence_rows}
    for evidence_id in (
        "world_tubes_decisive_demo",
        "world_tubes_visibility_stress",
        "worldfoam_optical_transfer_fixture",
        "worldfoam_owner_run_metal_comparison",
        "paper_quality_benchmark_table",
        "coffee_martini_matched_3seed",
    ):
        row = by_evidence.get(evidence_id)
        if row is None:
            errors.append(f"missing required evidence row {evidence_id}")
        elif row.get("status") != "ok":
            errors.append(f"required evidence row {evidence_id} must be ok")
    by_representation = {str(row.get("representation")): row for row in representation_rows}
    for representation in ("world_tubes_star_uvt", "worldfoam_powerfoam", "dynamic_3dgs_fast_mac"):
        if representation not in by_representation:
            errors.append(f"missing representation row {representation}")
    summary = report.get("summary")
    if not isinstance(summary, dict):
        errors.append("summary must be present")
        return errors
    expected = summarize(evidence_rows, representation_rows, missing_rows)
    for key, value in expected.items():
        _assert_summary_value(summary.get(key), value, key, errors)
    if missing_rows:
        if report.get("status") != "incomplete":
            errors.append("report status must be incomplete while missing rows exist")
        if summary.get("paper_ready") is not False:
            errors.append("summary paper_ready must be false while required rows are missing")
    else:
        if report.get("status") != "ok":
            errors.append("report status must be ok when no missing rows remain")
        if summary.get("paper_ready") is not True:
            errors.append("summary paper_ready must be true when no missing rows remain")
    if summary.get("has_world_tubes_fixture_gates") is not True:
        errors.append("summary must report has_world_tubes_fixture_gates true")
    if summary.get("has_worldfoam_fixture_gates") is not True:
        errors.append("summary must report has_worldfoam_fixture_gates true")
    if summary.get("has_coffee_martini_matched_3seed") is not True:
        errors.append("summary must report has_coffee_martini_matched_3seed true")
    return errors


def assert_paper_runner_table_report(report: dict[str, Any]) -> None:
    errors = verify_paper_runner_table_report(report)
    if errors:
        raise AssertionError("paper runner table report failed:\n- " + "\n- ".join(errors))


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def write_markdown(report: dict[str, Any], path: str | Path) -> Path:
    resolved = _root_path(path)
    lines = [
        "# DynaWorld Paper Runner Table Report",
        "",
        f"Status: `{report['status']}`.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(report["summary"], indent=2, sort_keys=True),
        "```",
        "",
        "## Representation Rows",
        "",
        "| representation | paper | fixture_gates | visual_capacity | paper_quality_benchmark | coffee_martini_matched_3seed | paper_ready |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["representation_rows"]:
        lines.append(
            "| "
            + " | ".join(
                _fmt(row.get(key))
                for key in (
                    "representation",
                    "paper",
                    "fixture_gates",
                    "visual_capacity",
                    "paper_quality_benchmark",
                    "coffee_martini_matched_3seed",
                    "paper_ready",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Evidence Rows",
            "",
            "| evidence_id | paper | kind | status | claim |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for row in report["evidence_rows"]:
        lines.append(
            "| "
            + " | ".join(_fmt(row.get(key)) for key in ("evidence_id", "paper", "kind", "status", "claim"))
            + " |"
        )
    lines.extend(
        [
            "",
            "## Missing Rows",
            "",
            "| missing_id | representation | needed_for |",
            "| --- | --- | --- |",
        ]
    )
    for row in report["missing_rows"]:
        lines.append(
            "| "
            + " | ".join(_fmt(row.get(key)) for key in ("missing_id", "representation", "needed_for"))
            + " |"
        )
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return resolved


def write_report(report: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    assert_paper_runner_table_report(report)
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
        assert_paper_runner_table_report(report)
        print(f"verified {args.verify_report}")
        return

    report = build_report()
    json_path, markdown_path = write_report(report, args.out_dir)
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
